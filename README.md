# Balatro DQN

A Deep Q-Network agent that learns to play a simplified version of the card game **Balatro**. The agent learns to select and discard cards to beat escalating score targets (blinds), exploit joker bonuses, and eventually clear multi-blind runs across multiple antes.

Built as a Bachelor's End Project (BEP) at TU Eindhoven.

---

## Quickstart

### Requirements

```
Python >= 3.10
torch
numpy
```

Install dependencies:

```bash
pip install torch numpy
```

### Run full automated curriculum (recommended)

Trains through all 12 phases automatically, advancing when each phase converges:

```bash
# Single-threaded (original)
python main.py --curriculum

# Parallel training — uses N CPU workers + 1 GPU learner (Ape-X style)
# Typically 4-6x faster; auto-detects worker count from cpu_count
python main.py --curriculum --parallel

# Explicit worker count (tune until GPU hits 70-80%)
python main.py --curriculum --parallel --workers 10
```

### Resume from a checkpoint

```bash
python main.py --curriculum --start-phase 5 --resume checkpoints/phase_04/
```

### Train a single phase manually

Set `TRAINING_PHASE` in `config.py`, then:

```bash
python main.py
python main.py --episodes 10000
```

---

## Project Structure

```
BEP/
├── main.py            Training entry point and curriculum runner
├── parallel_train.py  Ape-X parallel training (N CPU workers + 1 GPU learner)
├── config.py          All hyperparameters, phase settings, curriculum schedule
├── agent.py           DQN agent, networks, prioritized replay, n-step buffer
├── env.py             BalatroEnv (single blind) and RunEnv (full run)
├── scoring.py         Hand detection and score calculation
├── cards.py           Deck creation and card constants
├── jokers.py          Joker engine, shop heuristic, state encoding
├── jokers.json        Joker definitions (add new jokers here)
├── run_structure.py   Blind targets, money system, shop constants
├── baselines.py       Heuristic baselines and evaluation runner
└── checkpoints/       Saved model weights (created at runtime)
    └── phase_NN/
        ├── episode_NNNNN.pt        Mid-phase checkpoints
        └── episode_NNNNN_final.pt  Phase-end checkpoints
```

---

## Training Curriculum

The agent trains through 12 phases of increasing complexity. Phases 1-9 use a single blind per episode. Phases 10-12 use full multi-blind runs.

| Phase | Env | Blind | Discards | Jokers | Budget | What is learned |
|-------|-----|-------|----------|--------|--------|-----------------|
| 1  | Single | None | 0 | 0 | 20k | Hand types, basic scoring |
| 2  | Single | 300  | 0 | 0 | 5k  | Win condition |
| 3  | Single | 450  | 0 | 0 | 5k  | Consistent hand building |
| 4  | Single | 450  | 3 | 0 | 20k | Discard decisions |
| 5  | Single | 600  | 3 | 0 | 25k | Full single-blind game |
| 6  | Single | 700  | 3 | 1 | 15k | Exploit a single joker |
| 7  | Single | 850  | 3 | 2 | 15k | Adapt hand selection to joker set |
| 8  | Single | 1000 | 3 | 3 | 20k | Suit and count strategies (tier-2 jokers) |
| 9  | Single | 1200 | 3 | 4 | 20k | Full joker complexity |
| 10 | Run    | —    | 3 | 2 | 15k | Blind-to-blind continuity (1 ante) |
| 11 | Run    | —    | 3 | 3 | 20k | Ante progression (2 antes) |
| 12 | Run    | —    | 3 | 4 | 30k | Full run (4 antes, 12 blinds) |

Early stopping advances each phase when win rate exceeds 75% for 5 consecutive log intervals. Phase 1 uses score plateau detection instead. Epsilon is reset at phases 5, 6, 8, 10, 11, and 12 to re-enable exploration when the task changes significantly.

---

## Key Design Decisions

**State vector (167 floats):** encodes per-card features, deck composition, urgency scalars, deck-aware draw quality signals, joker-adjusted score estimates, and run context. See `TECHNICAL.md` for the full breakdown.

**Action encoding:** each candidate action (play or discard of 1-5 cards) is encoded as a 22-float feature vector describing the selected or kept cards, not as an index. This lets the network generalise across different card combinations.

**Split heads:** the network has separate play and discard output heads on a shared trunk. The signed card count in the action features routes each action to the correct head.

**Kicker exclusion fix:** `get_score_components()` in `jokers.py` now correctly excludes kicker cards when computing chip totals for joker scoring, matching `scoring.py`. Without this, High Card hands were over-valued relative to made hands in joker phases, causing a High Card regression.

**Joker system:** 21 jokers across 3 rarities defined entirely in `jokers.json`. New jokers require only a JSON entry. Adding a new joker requires no code changes unless it introduces a new trigger type.

**Shop (phases 10-12):** a greedy heuristic runs automatically between blinds. It scores each offered joker by expected per-hand value contribution and buys the best affordable option, or sells the weakest owned joker to fund an upgrade. The main DQN never touches the shop.

---

## Checkpoints

Checkpoints are saved to `checkpoints/phase_NN/` every 500 episodes and at the end of each phase. To resume:

```bash
# Resume curriculum from phase 6 using phase 5's final checkpoint
python main.py --curriculum --start-phase 6 --resume checkpoints/phase_05/
```

---

## Adding Jokers

Open `jokers.json` and add a new entry following the existing schema. The engine supports:

- **Triggers:** `passive`, `on_hand_played`, `on_card_scored`
- **Conditions:** `none`, `hand_contains`, `hand_type_exact`, `card_count_le`, `card_suit`
- **Effects:** `add_mult`, `add_chips`

No Python changes needed unless introducing a completely new trigger or condition type.

---

## Parallel Training

On multi-core machines the GPU sits idle between env steps. `--parallel` fixes this with an Ape-X architecture:

- **N worker processes** run their own env instances on CPU, collecting transitions and pushing them to a shared queue
- **1 learner process** (GPU) drains the queue into the PER buffer and trains continuously
- Every 200 training steps the learner broadcasts fresh weights to all workers via a `multiprocessing.Pipe`

**Tuning workers:** start with `--workers 8`, check GPU utilisation (target 70-80%), increase by 2 until it stops rising. Beyond ~12 workers the queue fills faster than the learner can drain it.

Three constants at the top of `parallel_train.py` control the tradeoff between throughput and staleness:

| Constant | Default | Effect |
|----------|---------|--------|
| `WEIGHT_SYNC_INTERVAL` | 200 | Steps between weight broadcasts (lower = fresher workers) |
| `WORKER_BATCH_SIZE` | 16 | Transitions per queue push (higher = less IPC overhead) |
| `MAX_QUEUE_SIZE` | 2000 | Queue depth cap (prevents workers running too far ahead) |

---

## Logs

Training logs are written to `logs/training.log`. Each line covers 50 episodes:

```
Ph6 Ep  1500  |  Score 823.4 (min 312  max 2140  std 401.2)  |  ...  |  Win% 78.0  |  Top joker: Jolly Joker  |  Most played: Two Pair
```
