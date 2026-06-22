"""
parallel_train.py  --  Ape-X style parallel training.

Architecture
------------
N worker processes (CPU-only, no gradients):
    - Each runs its own BalatroEnv / RunEnv
    - Uses a local CPU copy of the policy network for epsilon-greedy selection
    - Pushes completed transitions into a shared multiprocessing.Queue
    - Periodically receives fresh weights from the learner via a Pipe

1 learner process (main process, GPU):
    - Drains the transition queue into the PER buffer
    - Runs GPU training steps as fast as transitions arrive
    - Broadcasts updated weights to all workers every WEIGHT_SYNC_INTERVAL steps
    - Handles checkpointing and logging

Why this works
--------------
Single-agent bottleneck: GPU sits idle while one Python env generates the next
transition. With N workers, transitions arrive N times faster, keeping the GPU
saturated. On a 24-core machine with 10-15% GPU utilisation, 8-12 workers
typically push GPU utilisation to 60-80% and cut wall-clock training time by 3-5x.

Ape-X vs A3C
------------
Unlike A3C, workers here do NOT compute gradients. All learning happens in the
single GPU learner, so there are no gradient conflicts, lock contention, or
staleness issues in the optimizer. Workers only do forward passes (cheap, CPU).

Usage
-----
    from parallel_train import parallel_train_phase

    # Drop-in replacement for train_phase() in main.py
    episodes_run = parallel_train_phase(
        phase         = 5,
        agent         = agent,
        logger        = logger,
        max_episodes  = 25_000,
        min_episodes  = 8_000,
        start_episode = 1,
        n_workers     = 8,   # tune to your CPU count
    )
"""

import os
import sys
import time
import random
import logging
import traceback
import multiprocessing as mp
from queue import Empty
from collections import deque, Counter
from itertools import combinations
from typing import Optional

# ── Tuning constants ──────────────────────────────────────────────────────────

# How often the learner broadcasts weights to workers (in learner train steps).
WEIGHT_SYNC_INTERVAL = 200

# How many transitions a worker batches before a single queue.put().
WORKER_BATCH_SIZE = 16

# Maximum items (each = one WORKER_BATCH_SIZE block) in the transition queue.
MAX_QUEUE_SIZE = 2000

# Seconds without transitions before the watchdog logs a warning.
QUEUE_DRAIN_TIMEOUT = 5.0

# ── Learner throughput constants ──────────────────────────────────────────────
# Per main-loop iteration the learner drains at most DRAIN_CAP transitions, then
# immediately does TRAIN_STEPS_PER_ITER gradient steps.  Keeping these values
# small breaks the "drain 9 000 transitions -> 2 250-step burst -> 6 s stall"
# pattern that starves the GPU between bursts.
#
# The loop runs at ~200-400 Hz so the GPU sees a continuous stream of small
# batches rather than infrequent large bursts.
DRAIN_CAP            = 128   # max transitions consumed from queue per iter
TRAIN_STEPS_PER_ITER = 4     # gradient steps per iter (keeps GPU warm)


# =============================================================================
# Worker process
# =============================================================================

def _worker_fn(
    worker_id:    int,
    phase:        int,
    epsilon:      float,
    trans_queue,
    weights_path: str,
    weights_version,
    status_queue,
    stop_event,
    use_run_env:  bool,
    max_antes:    int,
):
    """
    Worker process entry point.

    Runs episodes indefinitely, pushing batches of COMPLETED n-step transitions
    to trans_queue, until stop_event is set by the learner.

    Key design points
    -----------------
    * Each worker maintains its own NStepBuffer so n-step returns are computed
      per-episode without cross-worker contamination in the learner.
    * Completed n-step tuples (not raw transitions) are sent to the queue, so
      the learner stores them directly into PER without any further buffering.
    * When the queue is full the worker sleeps briefly instead of spinning at
      100 % CPU.
    * Weight-load failures skip the update for one episode rather than killing
      the worker, since a transient file-read race during an atomic rename is
      possible on some OS / filesystem combinations.
    """
    # ---- imports inside worker to avoid pickling issues ----
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import config as cfg
    cfg.set_phase(phase)

    from env import BalatroEnv, RunEnv
    from agent import DQNAgent, NStepBuffer, encode_action, encode_actions

    # CPU-only agent for action selection — no optimizer, no replay buffer
    worker_agent            = DQNAgent(state_size=cfg.STATE_SIZE, device_override="cpu")
    worker_agent.policy_net = worker_agent.policy_net.cpu()
    worker_agent.target_net = worker_agent.target_net.cpu()
    worker_agent.policy_net.eval()
    worker_agent.epsilon    = epsilon

    env = RunEnv(max_antes=max_antes) if use_run_env else BalatroEnv()
    status_queue.put(("started", worker_id, os.getpid()))

    # ---- per-worker n-step buffer ----
    # Collects raw transitions and emits completed n-step returns.
    # Using a lightweight list-backed sink so we don't touch the learner's PER.
    class _ListSink:
        def __init__(self):
            self.items = []
        def push(self, *args):
            self.items.append(args)

    sink          = _ListSink()
    worker_nstep  = NStepBuffer(cfg.N_STEP, cfg.GAMMA, sink)

    pending = []          # completed n-step tuples waiting to be queued
    episodes_done      = 0
    put_fail_count     = 0
    local_weights_version = -1
    ready_reported     = False
    first_batch_reported = False

    try:
        while not stop_event.is_set():
            # ---- pull fresh weights (non-blocking) ----
            current_version = int(weights_version.value)
            if current_version != local_weights_version and os.path.exists(weights_path):
                try:
                    with open(weights_path, "rb") as f:
                        data = f.read()
                    worker_agent.set_weights(data)
                    local_weights_version = current_version
                    if not ready_reported:
                        status_queue.put(("ready", worker_id, local_weights_version))
                        ready_reported = True
                except Exception:
                    # Transient race during atomic rename — skip this update,
                    # pick up the next version on the following episode.
                    status_queue.put((
                        "weight_load_skip",
                        worker_id,
                        current_version,
                        traceback.format_exc(),
                    ))

            # ---- run one episode ----
            state      = env.reset()
            ep_score   = 0
            ep_result  = "unknown"
            ep_hands   = []   # list of (hand_type, round_score)

            while not env.done:
                play_actions    = [(idxs, False) for idxs in env.valid_play_actions()]
                discard_actions = (
                    [(idxs, True) for idxs in env.valid_play_actions()]
                    if env.discards_remaining > 0 else []
                )
                valid_actions = play_actions + discard_actions

                if not valid_actions:
                    raise RuntimeError("Worker has no valid actions while env is not done")

                with torch.no_grad():
                    action_idx = worker_agent.select_action(state, valid_actions, env.hand)

                card_indices, is_discard = valid_actions[action_idx]
                chosen_features          = encode_action(card_indices, env.hand, is_discard)
                next_state, reward, done, info = env.step(card_indices, play=not is_discard)

                # Accumulate episode stats
                ep_score += info.get("score", 0)
                if "result" in info:
                    ep_result = info["result"]
                if "hand_type" in info:
                    ep_hands.append((info["hand_type"], info.get("score", 0)))

                # Build next action features for Double DQN bootstrap
                next_play     = [(idxs, False) for idxs in env.valid_play_actions()]
                next_discards = (
                    [(idxs, True) for idxs in env.valid_play_actions()]
                    if env.discards_remaining > 0 else []
                )
                next_valid        = next_play + next_discards
                next_action_feats = encode_actions(next_valid, env.hand)

                # Push through the per-worker n-step buffer.
                # sink.items accumulates completed n-step tuples.
                worker_nstep.push(
                    state, chosen_features, reward,
                    next_state, next_action_feats, done
                )
                pending.extend(sink.items)
                sink.items = []

                state = next_state

                # ---- flush batch to learner ----
                if len(pending) >= WORKER_BATCH_SIZE:
                    _flush_pending(
                        pending, trans_queue, status_queue,
                        worker_id, first_batch_reported,
                    )
                    first_batch_reported = True
                    pending = []

            # ---- flush remainder at episode end ----
            if pending:
                _flush_pending(
                    pending, trans_queue, status_queue,
                    worker_id, first_batch_reported,
                )
                first_batch_reported = True
                pending = []

            # ---- report episode stats to learner for logging ----
            status_queue.put(("episode_result", worker_id, {
                "score":          ep_score,
                "result":         ep_result,
                "hand_types":     ep_hands,
                "joker_names":    list(env.active_joker_names)
                                  if hasattr(env, "active_joker_names") else [],
                "hands_used":     env.max_hand_plays - env.hands_remaining,
                "discards_used":  env.max_discards   - env.discards_remaining,
                "blinds_cleared": getattr(env, "blinds_cleared", 0),
                "antes_cleared":  getattr(env, "antes_cleared",  0),
            }))

            # Decay epsilon locally (workers explore independently)
            worker_agent.epsilon = max(
                cfg.EPSILON_END, worker_agent.epsilon * cfg.EPSILON_DECAY
            )
            episodes_done += 1
            if episodes_done % 25 == 0:
                status_queue.put(("progress", worker_id, episodes_done))

    except Exception as ex:
        status_queue.put((
            "error",
            worker_id,
            str(ex),
            traceback.format_exc(),
        ))


def _flush_pending(pending, trans_queue, status_queue, worker_id, already_reported):
    """
    Send ``pending`` to the learner queue with backoff on full-queue.

    Sleeps briefly instead of spinning when the queue is at capacity, which
    keeps workers from pegging the CPU while the learner drains.
    Drops the batch after MAX_PUT_RETRIES failed attempts rather than
    blocking forever.
    """
    MAX_PUT_RETRIES = 5
    for attempt in range(MAX_PUT_RETRIES):
        try:
            trans_queue.put(pending, timeout=0.5)
            if not already_reported:
                status_queue.put(("first_batch", worker_id, len(pending)))
            return
        except Exception:
            if attempt < MAX_PUT_RETRIES - 1:
                # Queue full — back off so the learner can drain
                time.sleep(0.05 * (attempt + 1))
            else:
                status_queue.put((
                    "queue_put_error", worker_id,
                    attempt + 1, traceback.format_exc(),
                ))


# =============================================================================
# Learner (main process) — parallel training loop
# =============================================================================

def parallel_train_phase(
    phase:         int,
    agent,                     # DQNAgent, lives on GPU in the learner
    logger:        logging.Logger,
    max_episodes:  int,
    min_episodes:  int,
    start_episode: int = 1,
    n_workers:     int = 8,
) -> int:
    """
    Drop-in replacement for train_phase() that uses N parallel env workers.

    The episode count is approximated — workers don't report episode boundaries,
    so we estimate from total transitions received ÷ avg steps per episode.

    Returns approximate number of episodes run (for curriculum progress tracking).
    """
    import config as cfg
    cfg.set_phase(phase)

    use_run_env = cfg.USE_RUN_ENV
    max_antes   = cfg.MAX_ANTES if use_run_env else 0

    logger.info("=" * 70)
    logger.info(
        f"PHASE {phase}  [PARALLEL x{n_workers}]  |  "
        f"Blind: {cfg.BLIND_TARGET}  |  "
        f"Discards: {cfg.MAX_DISCARDS}  |  "
        f"Jokers: {cfg.NUM_JOKERS} (tier<={cfg.MAX_JOKER_TIER})  |  "
        f"Budget: {max_episodes:,} eps"
    )
    logger.info("=" * 70)

    # ── Launch workers ────────────────────────────────────────────────────────
    stop_event  = mp.Event()
    trans_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)
    status_queue = mp.Queue()

    # Weights are synced via an atomic file write + shared version counter.
    # This avoids large payload deadlocks on Pipe.send in Windows spawn mode.
    weights_sync_dir = os.path.join("checkpoints", "_parallel_sync")
    os.makedirs(weights_sync_dir, exist_ok=True)
    weights_path = os.path.join(weights_sync_dir, f"phase_{phase:02d}_latest_weights.bin")
    weights_version = mp.Value("i", 0)

    workers = []
    for i in range(n_workers):
        # Stagger starting epsilon so workers explore at different rates
        eps = max(cfg.EPSILON_END, agent.epsilon - i * 0.02)
        p = mp.Process(
            target=_worker_fn,
            args=(i, phase, eps, trans_queue, weights_path, weights_version,
                  status_queue, stop_event, use_run_env, max_antes),
            daemon=True,
        )
        p.start()
        workers.append(p)

    logger.info(f"  Started {n_workers} worker processes.")

    # ── Learner loop ──────────────────────────────────────────────────────────
    LOG_INTERVAL        = 50
    CHECKPOINT_INTERVAL = 500

    # Rolling windows — mirror main.py deque sizes exactly
    recent_scores    = deque(maxlen=LOG_INTERVAL)
    recent_losses    = deque(maxlen=LOG_INTERVAL)
    recent_hands     = deque(maxlen=LOG_INTERVAL)
    recent_discards  = deque(maxlen=LOG_INTERVAL)
    recent_results   = deque(maxlen=LOG_INTERVAL)
    recent_blinds    = deque(maxlen=LOG_INTERVAL)
    recent_antes     = deque(maxlen=LOG_INTERVAL)

    hand_type_count  = {}   # hand_type -> play count  (cleared each log)
    hand_type_best   = {}   # hand_type -> best score  (cleared each log)
    joker_count      = {}   # joker name -> appearances (cleared each log)

    # Convergence tracking — same logic as train_phase() in main.py
    win_rate_streak  = 0
    plateau_streak   = 0
    prev_avg_loss    = None
    prev_avg_score   = None

    # Counters
    transitions_received = 0
    train_steps          = 0
    approx_episodes      = 0
    episode_cursor       = start_episode
    prev_episode_cursor  = start_episode

    # Align last_log_ep to the nearest boundary at or below episode_cursor so
    # the first log fires at episode_cursor + LOG_INTERVAL, not at every
    # boundary between 0 and episode_cursor (which would spam thousands of
    # stat-less lines when resuming mid-phase or starting a new phase).
    last_log_ep        = (start_episode // LOG_INTERVAL) * LOG_INTERVAL
    last_ckpt_ep       = start_episode
    phase_done         = False
    start_time         = time.time()
    last_data_time     = start_time
    last_watchdog_log  = 0.0
    worker_started     = set()
    worker_progress    = {}

    # Publish initial weights for workers
    initial_weights = agent.get_weights()
    try:
        tmp_path = weights_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(initial_weights)
        os.replace(tmp_path, weights_path)
        weights_version.value = 1
        logger.info("  Published initial weights for workers.")
    except Exception:
        logger.error("  Failed to publish initial weights:")
        logger.error(traceback.format_exc())

    try:
        while True:
            # ── drain worker status messages ─────────────────────────────────
            while True:
                try:
                    msg = status_queue.get_nowait()
                except Exception:
                    break

                kind = msg[0]
                if kind == "started":
                    _, wid, pid = msg
                    worker_started.add(wid)
                    logger.info(f"  Worker {wid} started (pid {pid}).")
                elif kind == "ready":
                    _, wid, ver = msg
                    logger.info(f"  Worker {wid} ready (weights v{ver}).")
                elif kind == "first_batch":
                    _, wid, bsz = msg
                    logger.info(f"  Worker {wid} sent first batch ({bsz} transitions).")
                elif kind == "progress":
                    _, wid, w_eps = msg
                    worker_progress[wid] = w_eps
                elif kind == "episode_result":
                    _, wid, stats = msg
                    recent_scores.append(stats["score"])
                    recent_results.append(stats["result"])
                    recent_hands.append(stats["hands_used"])
                    recent_discards.append(stats["discards_used"])
                    recent_blinds.append(stats["blinds_cleared"])
                    recent_antes.append(stats["antes_cleared"])
                    for ht, sc in stats["hand_types"]:
                        hand_type_count[ht] = hand_type_count.get(ht, 0) + 1
                        if sc > hand_type_best.get(ht, 0):
                            hand_type_best[ht] = sc
                    for jname in stats["joker_names"]:
                        joker_count[jname] = joker_count.get(jname, 0) + 1
                elif kind == "error":
                    _, wid, err, tb = msg
                    logger.error(f"  Worker {wid} crashed: {err}")
                    logger.error(tb)
                elif kind == "queue_put_error":
                    _, wid, fail_count, tb = msg
                    logger.warning(
                        f"  Worker {wid} failed to enqueue batch "
                        f"({fail_count} failures so far)."
                    )
                elif kind == "weight_load_skip":
                    _, wid, ver, tb = msg
                    logger.debug(f"  Worker {wid} skipped weight sync v{ver} (race).")

            # ── drain transition queue (capped per iter) ──────────────────────
            # Hard cap at DRAIN_CAP transitions per iteration.  Without this cap
            # the learner drains thousands of transitions in one go, then runs
            # n_train = thousands // 4 gradient steps back-to-back for several
            # seconds, during which the GPU sits idle and the queue refills
            # entirely — producing a CPU-100% / GPU-idle burst-stall cycle.
            drained = 0
            try:
                first_batch = trans_queue.get(timeout=0.01)
                batches = [first_batch]
            except Empty:
                batches = []
            except Exception:
                batches = []

            while drained < DRAIN_CAP:
                try:
                    batches.append(trans_queue.get_nowait())
                    drained += WORKER_BATCH_SIZE   # approximate; exact count below
                except Empty:
                    break
                except Exception:
                    break

            drained = 0
            for batch in batches:
                for (state, action_feats, reward,
                     next_state, next_action_feats, done) in batch:
                    agent.replay_buffer.push(
                        state, action_feats, reward,
                        next_state, next_action_feats, done
                    )
                    transitions_received += 1
                    drained += 1
                    if done:
                        approx_episodes += 1
                        episode_cursor  += 1

            if drained > 0:
                last_data_time = time.time()

            # ── gradient steps ────────────────────────────────────────────────
            # Only train when new transitions arrived.  Running gradient steps
            # on an empty-drain iteration causes the replay ratio to explode
            # (>10 updates per new transition) — severe overfitting to early
            # bad experiences that prevents learning once epsilon collapses.
            if drained > 0:
                for _ in range(TRAIN_STEPS_PER_ITER):
                    loss = agent.train_step()
                    if loss is not None:
                        recent_losses.append(loss)
                        train_steps += 1
                        if train_steps % cfg.TARGET_UPDATE == 0:
                            agent.sync_target_network()

            # Decay epsilon once per completed episode, not once per loop iter.
            # At ~400 loop iters/episode, calling every iter decays epsilon
            # 400x faster than intended and collapses exploration prematurely.
            new_episodes = episode_cursor - prev_episode_cursor
            for _ in range(new_episodes):
                agent.decay_epsilon()
            prev_episode_cursor = episode_cursor

            # ── broadcast weights ─────────────────────────────────────────────
            if train_steps > 0 and train_steps % WEIGHT_SYNC_INTERVAL == 0:
                try:
                    weights  = agent.get_weights()
                    tmp_path = weights_path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        f.write(weights)
                    os.replace(tmp_path, weights_path)
                    weights_version.value += 1
                except Exception:
                    logger.error("  Failed to publish refreshed weights:")
                    logger.error(traceback.format_exc())

            # ── logging — identical format to main.py train_phase() ──────────
            # Fire once for every 50-episode boundary crossed in this iteration.
            # episode_cursor can jump by >50 in one drain cycle, so we step
            # through each boundary rather than checking once and skipping ahead.
            ep = episode_cursor
            while (ep - last_log_ep >= LOG_INTERVAL
                   and len(recent_scores) == LOG_INTERVAL):
                last_log_ep += LOG_INTERVAL
                ep_log = last_log_ep
                avg_score    = sum(recent_scores)   / len(recent_scores)
                avg_hands    = sum(recent_hands)    / len(recent_hands)
                avg_discards = sum(recent_discards) / len(recent_discards)
                avg_loss     = sum(recent_losses)   / len(recent_losses) if recent_losses else 0.0
                min_score    = min(recent_scores)
                max_score    = max(recent_scores)
                std_score    = (sum((s - avg_score)**2 for s in recent_scores)
                                / len(recent_scores)) ** 0.5

                top_hand  = max(hand_type_count, key=hand_type_count.get) if hand_type_count else "--"
                best_hand = max(hand_type_best,  key=hand_type_best.get)  if hand_type_best  else "--"

                trend = ("~" if prev_avg_loss is None else
                         "v" if avg_loss < prev_avg_loss - 0.001 else
                         "^" if avg_loss > prev_avg_loss + 0.001 else "~")
                prev_avg_loss = avg_loss

                wins     = sum(1 for r in recent_results if r == "win")
                win_rate = wins / len(recent_results)

                early_stop   = False
                win_str      = ""
                discard_str  = ""
                joker_str    = ""
                progress_str = ""

                if use_run_env:
                    avg_blinds   = sum(recent_blinds) / len(recent_blinds)
                    avg_antes    = sum(recent_antes)  / len(recent_antes)
                    progress_str = (
                        f"  |  Blinds {avg_blinds:.1f}/{max_antes * 3}"
                        f"  Antes {avg_antes:.1f}/{max_antes}"
                    )
                    win_str = f"  |  Runs won% {win_rate * 100:.1f}"
                    if approx_episodes >= min_episodes:
                        if win_rate >= cfg.WIN_RATE_THRESHOLD:
                            win_rate_streak += 1
                        else:
                            win_rate_streak = 0
                        if win_rate_streak >= cfg.WIN_RATE_WINDOW:
                            early_stop = True
                elif cfg.BLIND_TARGET is not None:
                    win_str = f"  |  Win% {win_rate * 100:>5.1f}"
                    if approx_episodes >= min_episodes:
                        if win_rate >= cfg.WIN_RATE_THRESHOLD:
                            win_rate_streak += 1
                        else:
                            win_rate_streak = 0
                        if win_rate_streak >= cfg.WIN_RATE_WINDOW:
                            early_stop = True
                else:
                    # Phase 1: score plateau detection
                    if prev_avg_score is not None and approx_episodes >= min_episodes:
                        if (avg_score - prev_avg_score) < cfg.SCORE_PLATEAU_MIN:
                            plateau_streak += 1
                        else:
                            plateau_streak = 0
                        if plateau_streak >= cfg.PLATEAU_WINDOW:
                            early_stop = True
                    prev_avg_score = avg_score

                if cfg.MAX_DISCARDS > 0:
                    discard_str = f"  |  Discards {avg_discards:.2f}/{cfg.MAX_DISCARDS}"
                if cfg.NUM_JOKERS > 0 and joker_count:
                    joker_str = f"  |  Top joker: {max(joker_count, key=joker_count.get)}"

                logger.info(
                    f"Ph{phase} Ep {ep_log:>5}  |  "
                    f"Score {avg_score:>7.1f} "
                    f"(min {min_score:>5}  max {max_score:>5}  std {std_score:>6.1f})  |  "
                    f"Loss {avg_loss:.4f}{trend}  |  "
                    f"Hands {avg_hands:.2f}/{cfg.MAX_HAND_PLAYS}"
                    f"{discard_str}"
                    f"{progress_str}"
                    f"{win_str}"
                    f"{joker_str}  |  "
                    f"Eps {agent.epsilon:.3f}  |  "
                    f"Most played: {top_hand}  |  "
                    f"Best scored: {best_hand}"
                )

                hand_type_count.clear()
                hand_type_best.clear()
                joker_count.clear()

                if early_stop:
                    logger.info(
                        f"  >> Phase {phase} converged at ~ep {ep_log} "
                        f"({approx_episodes:,}/{max_episodes:,} budget used)."
                    )
                    phase_done = True
                    break   # exits the inner while-log loop only

            if phase_done:
                break       # exits the outer while True training loop

            # ── watchdog ──────────────────────────────────────────────────────
            now = time.time()
            if now - last_watchdog_log >= QUEUE_DRAIN_TIMEOUT:
                last_watchdog_log = now
                alive_workers = sum(1 for w in workers if w.is_alive())
                idle_s = now - last_data_time
                try:
                    qsz = trans_queue.qsize()
                except Exception:
                    qsz = -1

                if idle_s >= QUEUE_DRAIN_TIMEOUT:
                    logger.warning(
                        "  Parallel watchdog: no transitions for %.1fs | "
                        "alive workers %d/%d | started %d/%d | queue size %s | "
                        "train steps %d",
                        idle_s, alive_workers, n_workers,
                        len(worker_started), n_workers,
                        "n/a" if qsz < 0 else str(qsz),
                        train_steps,
                    )

                if alive_workers == 0 and idle_s >= 2.0:
                    logger.error("  All workers exited. Aborting phase.")
                    break

            # ── checkpoint ───────────────────────────────────────────────────
            if ep - last_ckpt_ep >= CHECKPOINT_INTERVAL:
                last_ckpt_ep = ep
                ckpt_dir = os.path.join("checkpoints", f"phase_{phase:02d}")
                os.makedirs(ckpt_dir, exist_ok=True)
                agent.save(os.path.join(ckpt_dir, f"episode_{ep:05d}.pt"))

            # ── budget check ─────────────────────────────────────────────────
            if approx_episodes >= max_episodes:
                logger.info(
                    f"  Phase {phase}: episode budget exhausted "
                    f"({approx_episodes:,} eps)."
                )
                break

    finally:
        # ---- clean shutdown ----
        stop_event.set()
        for w in workers:
            w.join(timeout=3.0)
            if w.is_alive():
                w.terminate()

        # Final checkpoint
        ckpt_dir = os.path.join("checkpoints", f"phase_{phase:02d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"episode_{episode_cursor:05d}_final.pt")
        agent.save(ckpt_path)
        logger.info(f"  -> Phase {phase} final checkpoint: {ckpt_path}")
        logger.info(
            f"  -> Phase {phase} done: ~{approx_episodes:,} episodes, "
            f"{transitions_received:,} transitions, {train_steps:,} gradient steps."
        )

    return approx_episodes