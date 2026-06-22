"""
main.py -- Balatro DQN Training Entry Point
==========================================

Single-phase training:
    python main.py                          # train current phase from scratch
    python main.py --resume checkpoints/   # resume from latest checkpoint
    python main.py --episodes 5000         # override episode count

Full automated curriculum (phases 1 -> 9):
    python main.py --curriculum            # run all phases start to finish
    python main.py --curriculum --start-phase 4          # skip ahead to phase 4
    python main.py --curriculum --start-phase 4 --resume checkpoints/phase_03/

How the curriculum works
------------------------
Each phase trains for up to PHASE_EPISODES[phase] episodes (the budget).
Early stopping advances to the next phase if:
  - PHASE_MIN_EPISODES[phase] has been reached, AND
  - Phase 2-9 (blind target): win rate >= WIN_RATE_THRESHOLD for
    WIN_RATE_WINDOW consecutive log intervals.
  - Phase 1 (no blind): avg score improves < SCORE_PLATEAU_MIN for
    PLATEAU_WINDOW consecutive log intervals.

The agent (weights, replay buffer, epsilon) carries forward between phases --
this is intentional transfer learning. Only the BalatroEnv is recreated at
each transition to pick up the new phase settings.

Checkpoints
-----------
  Mid-phase : checkpoints/phase_NN/episode_NNNNN.pt   (every CHECKPOINT_INTERVAL)
  Phase-end : checkpoints/phase_NN/episode_NNNNN_final.pt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import glob
from collections import deque
from typing import Optional

import config
import torch
from env import BalatroEnv, RunEnv
from agent import DQNAgent, encode_action, encode_actions


def make_env():
    """Factory: return the correct env type for the current phase."""
    if config.USE_RUN_ENV:
        return RunEnv(max_antes=config.MAX_ANTES)
    return BalatroEnv()


# =============================================================================
# Constants
# =============================================================================

CHECKPOINT_INTERVAL = 500    # mid-phase checkpoint every N episodes
LOG_INTERVAL        = 50     # log summary line every N episodes


# =============================================================================
# Logging
# =============================================================================

def setup_logger(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("balatro_dqn")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# =============================================================================
# Checkpoint helpers
# =============================================================================

def latest_checkpoint(directory: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(directory, "*.pt")))
    return files[-1] if files else None


def save_checkpoint(agent: DQNAgent, phase: int, episode: int, suffix: str = "") -> str:
    d = os.path.join("checkpoints", f"phase_{phase:02d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"episode_{episode:05d}{suffix}.pt")
    agent.save(path)
    return path


# =============================================================================
# Single episode
# =============================================================================

def run_episode(env, agent: DQNAgent, train: bool = True) -> dict:
    """
    Run one full episode. Works with both BalatroEnv (single blind)
    and RunEnv (full multi-blind run). Returns a unified stats dict.
    """
    state        = env.reset()
    total_reward = 0.0
    total_score  = 0
    steps        = 0
    losses       = []
    hand_types   = []
    result       = "unknown"
    # Run-specific stats (non-zero only for RunEnv)
    blinds_cleared = 0
    antes_cleared  = 0
    run_result     = None
    shop_purchases = []   # list of buy/sell_and_buy entries this episode

    while not env.done:
        play_actions    = [(idxs, False) for idxs in env.valid_play_actions()]
        discard_actions = (
            [(idxs, True) for idxs in env.valid_play_actions()]
            if env.discards_remaining > 0 else []
        )
        valid_actions = play_actions + discard_actions

        action_idx               = agent.select_action(state, valid_actions, env.hand)
        card_indices, is_discard = valid_actions[action_idx]
        chosen_features          = encode_action(card_indices, env.hand, is_discard)
        next_state, reward, done, info = env.step(card_indices, play=not is_discard)

        if train:
            next_play     = [(idxs, False) for idxs in env.valid_play_actions()]
            next_discards = (
                [(idxs, True) for idxs in env.valid_play_actions()]
                if env.discards_remaining > 0 else []
            )
            next_valid        = next_play + next_discards
            next_action_feats = encode_actions(next_valid, env.hand)
            agent.store(state, chosen_features, reward, next_state, next_action_feats, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        state        = next_state
        total_reward += reward
        total_score  += info.get("score", 0)
        steps        += 1

        if "hand_type" in info:
            hand_types.append((info["hand_type"], info.get("score", 0)))
        if "result" in info:
            result = info["result"]
        # Track run-level outcomes from RunEnv
        if "blind_cleared" in info:
            blinds_cleared += 1
        if "run_result" in info:
            run_result = info["run_result"]
        # Track shop purchases for logging
        for entry in info.get("shop_log", []):
            if entry.get("action") in ("buy", "sell_and_buy"):
                shop_purchases.append(entry)

    # For RunEnv, pull run-level counters
    if isinstance(env, RunEnv):
        antes_cleared  = env.antes_cleared
        blinds_cleared = env.blinds_cleared
        # Use run_result as top-level result for early-stopping logic
        if run_result == "complete":
            result = "win"
        elif run_result == "failed":
            result = "loss"

    return {
        "total_score":    total_score,
        "total_reward":   total_reward,
        "steps":          steps,
        "hands_used":     env.max_hand_plays - env.hands_remaining,
        "discards_used":  env.max_discards   - env.discards_remaining,
        "avg_loss":       sum(losses) / len(losses) if losses else 0.0,
        "hand_types":     hand_types,
        "result":         result,
        "active_jokers":  env.active_joker_names,
        "blinds_cleared": blinds_cleared,
        "antes_cleared":  antes_cleared,
        "shop_purchases": shop_purchases,
    }


# =============================================================================
# Single-phase training loop
# =============================================================================

def train_phase(
    phase:         int,
    agent:         DQNAgent,
    logger:        logging.Logger,
    max_episodes:  int,
    min_episodes:  int,
    start_episode: int = 1,
) -> int:
    """
    Train for one phase. Returns the number of episodes actually run.
    Works with both BalatroEnv (phases 1-9) and RunEnv (phases 10+).
    """
    env       = make_env()
    is_run    = config.USE_RUN_ENV
    max_antes = config.MAX_ANTES if is_run else 0

    logger.info("=" * 70)
    if is_run:
        logger.info(
            f"PHASE {phase}  |  RUN MODE  |  "
            f"Antes: {max_antes}  |  "
            f"Blinds/run: {max_antes * 3}  |  "
            f"Jokers: {config.NUM_JOKERS} (tier<={config.MAX_JOKER_TIER})  |  "
            f"Budget: {max_episodes:,} runs  |  "
            f"GAMMA: {config.GAMMA}"
        )
    else:
        logger.info(
            f"PHASE {phase}  |  "
            f"Blind: {config.BLIND_TARGET}  |  "
            f"Discards: {config.MAX_DISCARDS}  |  "
            f"Jokers: {config.NUM_JOKERS} (tier<={config.MAX_JOKER_TIER})  |  "
            f"Budget: {max_episodes:,} eps  |  "
            f"Min: {min_episodes:,} eps"
        )
    logger.info("=" * 70)

    recent_scores   = deque(maxlen=LOG_INTERVAL)
    recent_rewards  = deque(maxlen=LOG_INTERVAL)
    recent_losses   = deque(maxlen=LOG_INTERVAL)
    recent_hands    = deque(maxlen=LOG_INTERVAL)
    recent_discards = deque(maxlen=LOG_INTERVAL)
    recent_results  = deque(maxlen=LOG_INTERVAL)
    recent_blinds   = deque(maxlen=LOG_INTERVAL)
    recent_antes    = deque(maxlen=LOG_INTERVAL)
    recent_shop_buys = deque(maxlen=LOG_INTERVAL)  # purchases per run

    prev_avg_loss  = None
    prev_avg_score = None
    hand_type_count = {}
    hand_type_best  = {}
    joker_count     = {}

    win_rate_streak = 0
    plateau_streak  = 0
    episodes_run    = 0

    for ep_offset in range(max_episodes):
        episode = start_episode + ep_offset
        stats   = run_episode(env, agent)
        episodes_run += 1

        recent_scores.append(stats["total_score"])
        recent_rewards.append(stats["total_reward"])
        recent_hands.append(stats["hands_used"])
        recent_discards.append(stats["discards_used"])
        recent_results.append(stats["result"])
        recent_blinds.append(stats["blinds_cleared"])
        recent_antes.append(stats["antes_cleared"])
        recent_shop_buys.append(len(stats["shop_purchases"]))
        if stats["avg_loss"] > 0:
            recent_losses.append(stats["avg_loss"])

        for ht, score in stats["hand_types"]:
            hand_type_count[ht] = hand_type_count.get(ht, 0) + 1
            if score > hand_type_best.get(ht, 0):
                hand_type_best[ht] = score
        for jname in stats["active_jokers"]:
            joker_count[jname] = joker_count.get(jname, 0) + 1

        agent.decay_epsilon()
        agent.sync_target_network()

        if episode % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(agent, phase, episode)

        if episode % LOG_INTERVAL == 0 and len(recent_scores) == LOG_INTERVAL:
            avg_score    = sum(recent_scores)    / len(recent_scores)
            avg_reward   = sum(recent_rewards)   / len(recent_rewards)
            avg_hands    = sum(recent_hands)     / len(recent_hands)
            avg_discards = sum(recent_discards)  / len(recent_discards)
            avg_loss     = sum(recent_losses)    / len(recent_losses) if recent_losses else 0.0
            min_score    = min(recent_scores)
            max_score    = max(recent_scores)
            std_score    = (sum((s - avg_score)**2 for s in recent_scores)/len(recent_scores))**0.5

            top_hand  = max(hand_type_count, key=hand_type_count.get) if hand_type_count else "--"
            best_hand = max(hand_type_best,  key=hand_type_best.get)  if hand_type_best  else "--"

            trend = ("~" if prev_avg_loss is None else
                     "v" if avg_loss < prev_avg_loss - 0.001 else
                     "^" if avg_loss > prev_avg_loss + 0.001 else "~")
            prev_avg_loss = avg_loss

            early_stop   = False
            win_str      = ""
            discard_str  = ""
            joker_str    = ""
            progress_str = ""

            wins     = sum(1 for r in recent_results if r == "win")
            win_rate = wins / len(recent_results)

            if is_run:
                avg_blinds  = sum(recent_blinds)    / len(recent_blinds)
                avg_antes   = sum(recent_antes)      / len(recent_antes)
                avg_shop    = sum(recent_shop_buys)  / len(recent_shop_buys)
                progress_str = (
                    f"  |  Blinds {avg_blinds:.1f}/{max_antes * 3}"
                    f"  Antes {avg_antes:.1f}/{max_antes}"
                    f"  Shop {avg_shop:.1f}buys"
                )
                win_str = f"  |  Runs won% {win_rate * 100:.1f}"
                if episodes_run >= min_episodes:
                    if win_rate >= config.WIN_RATE_THRESHOLD:
                        win_rate_streak += 1
                    else:
                        win_rate_streak = 0
                    if win_rate_streak >= config.WIN_RATE_WINDOW:
                        early_stop = True
            elif config.BLIND_TARGET is not None:
                win_str = f"  |  Win% {win_rate * 100:>5.1f}"
                if episodes_run >= min_episodes:
                    if win_rate >= config.WIN_RATE_THRESHOLD:
                        win_rate_streak += 1
                    else:
                        win_rate_streak = 0
                    if win_rate_streak >= config.WIN_RATE_WINDOW:
                        early_stop = True
            else:
                if prev_avg_score is not None and episodes_run >= min_episodes:
                    if (avg_score - prev_avg_score) < config.SCORE_PLATEAU_MIN:
                        plateau_streak += 1
                    else:
                        plateau_streak = 0
                    if plateau_streak >= config.PLATEAU_WINDOW:
                        early_stop = True
                prev_avg_score = avg_score

            if env.max_discards > 0:
                discard_str = f"  |  Discards {avg_discards:.2f}/{env.max_discards}"
            if config.NUM_JOKERS > 0 and joker_count:
                joker_str = f"  |  Top joker: {max(joker_count, key=joker_count.get)}"

            logger.info(
                f"Ph{phase} Ep {episode:>5}  |  "
                f"Score {avg_score:>7.1f} "
                f"(min {min_score:>5}  max {max_score:>5}  std {std_score:>6.1f})  |  "
                f"Reward {avg_reward:>6.3f}  |  "
                f"Loss {avg_loss:.4f}{trend}  |  "
                f"Hands {avg_hands:.2f}/{env.max_hand_plays}"
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
                    f"  >> Phase {phase} converged at episode {episode} "
                    f"({episodes_run:,}/{max_episodes:,} budget used)."
                )
                break

    final_ep  = start_episode + episodes_run - 1
    ckpt_path = save_checkpoint(agent, phase, final_ep, suffix="_final")
    logger.info(f"  -> Phase {phase} checkpoint saved: {ckpt_path}")

    return episodes_run


# =============================================================================
# Curriculum runner
# =============================================================================

def run_curriculum(
    start_phase: int,
    resume_dir:  Optional[str],
    logger:      logging.Logger,
    n_workers:   int = 1,
) -> None:
    """Run all phases from start_phase through the last defined phase."""
    phases = sorted(p for p in config.PHASE_EPISODES if p >= start_phase)

    # Set config to starting phase before building the agent
    config.set_phase(start_phase)
    agent = DQNAgent(state_size=config.STATE_SIZE)

    start_episode = 1
    if resume_dir:
        ckpt = latest_checkpoint(resume_dir)
        if ckpt:
            agent.load(ckpt)
            try:
                basename = os.path.splitext(os.path.basename(ckpt))[0]
                # filename is episode_NNNNN or episode_NNNNN_final
                start_episode = int(basename.split("_")[1]) + 1
            except (ValueError, IndexError):
                pass
            logger.info(f"Resumed from {ckpt}  (continuing from episode {start_episode})")
        else:
            logger.warning(f"No checkpoint found in {resume_dir}, starting fresh.")

    _log_device(agent, logger)
    logger.info(f"STATE_SIZE = {config.STATE_SIZE}")
    logger.info(f"Curriculum: phases {phases}")
    logger.info("-" * 70)

    episode_cursor = start_episode
    for phase in phases:
        config.set_phase(phase)

        # Keep n-step buffer in sync with the phase's gamma
        agent.update_gamma(config.GAMMA)

        # Reset epsilon if this phase specifies one
        eps_reset = config.PHASE_EPSILON_RESET.get(phase)
        if eps_reset is not None:
            old_eps = agent.epsilon
            agent.epsilon = eps_reset
            logger.info(
                f"  Epsilon reset: {old_eps:.3f} -> {eps_reset:.3f} "
                f"(phase {phase} exploration restart)"
            )

        if n_workers and n_workers > 1:
            from parallel_train import parallel_train_phase
            eps_run = parallel_train_phase(
                phase         = phase,
                agent         = agent,
                logger        = logger,
                max_episodes  = config.PHASE_EPISODES[phase],
                min_episodes  = config.PHASE_MIN_EPISODES[phase],
                start_episode = episode_cursor,
                n_workers     = n_workers,
            )
        else:
            eps_run = train_phase(
                phase         = phase,
                agent         = agent,
                logger        = logger,
                max_episodes  = config.PHASE_EPISODES[phase],
                min_episodes  = config.PHASE_MIN_EPISODES[phase],
                start_episode = episode_cursor,
            )
        episode_cursor += eps_run

    logger.info("=" * 70)
    logger.info("Curriculum complete.")
    logger.info(f"Total episodes : {episode_cursor - start_episode:,}")
    logger.info(f"Final epsilon  : {agent.epsilon:.4f}")
    logger.info(f"Total steps    : {agent.steps_done:,}")


# =============================================================================
# Single-phase runner (manual / debug use)
# =============================================================================

def run_single(
    episodes:   int,
    resume_dir: Optional[str],
    logger:     logging.Logger,
) -> None:
    """Train only config.TRAINING_PHASE for the given episode budget."""
    agent = DQNAgent(state_size=config.STATE_SIZE)
    start_episode = 1

    if resume_dir:
        ckpt = latest_checkpoint(resume_dir)
        if ckpt:
            agent.load(ckpt)
            try:
                start_episode = int(
                    os.path.splitext(os.path.basename(ckpt))[0].split("_")[1]
                ) + 1
            except (ValueError, IndexError):
                pass
            logger.info(f"Resuming from {ckpt}")
        else:
            logger.warning(f"No checkpoint in {resume_dir}, starting fresh.")

    _log_device(agent, logger)

    train_phase(
        phase         = config.TRAINING_PHASE,
        agent         = agent,
        logger        = logger,
        max_episodes  = episodes,
        min_episodes  = config.PHASE_MIN_EPISODES.get(config.TRAINING_PHASE, 1_000),
        start_episode = start_episode,
    )


# =============================================================================
# Helpers
# =============================================================================

def _log_device(agent: DQNAgent, logger: logging.Logger) -> None:
    if agent.device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"Device: GPU -- {name} ({vram:.1f} GB VRAM)")
    else:
        logger.info("Device: CPU  (no CUDA GPU detected)")


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train Balatro DQN agent")
    p.add_argument("--curriculum",   action="store_true",
                   help="Run full automated curriculum (phases 1 -> 9)")
    p.add_argument("--start-phase",  type=int, default=None,
                   help="Phase to begin curriculum from (default: config.TRAINING_PHASE)")
    p.add_argument("--episodes",     type=int, default=None,
                   help="Episode budget for single-phase mode")
    p.add_argument("--resume",       type=str, default=None,
                   help="Directory to load latest checkpoint from")
    p.add_argument("--parallel",     action="store_true",
                   help="Use parallel env workers for faster training (Ape-X style)")
    p.add_argument("--workers",      type=int, default=None,
                   help="Number of parallel env workers (default: cpu_count - 1, max 16)")
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing
    args   = parse_args()
    logger = setup_logger()

    # Determine worker count
    n_workers = 1
    if args.parallel:
        if args.workers:
            n_workers = args.workers
        else:
            # Leave one core free for the learner/OS
            n_workers = min(max(1, multiprocessing.cpu_count() - 1), 16)
        logger.info(f"Parallel mode: {n_workers} worker processes")
        # Required for safe multiprocessing on Windows / macOS
        multiprocessing.set_start_method("spawn", force=True)

    if args.curriculum:
        start_phase = args.start_phase or config.TRAINING_PHASE
        run_curriculum(
            start_phase = start_phase,
            resume_dir  = args.resume,
            logger      = logger,
            n_workers   = n_workers,
        )
    else:
        episodes = args.episodes or config.PHASE_EPISODES.get(config.TRAINING_PHASE, 20_000)
        run_single(episodes=episodes, resume_dir=args.resume, logger=logger)
