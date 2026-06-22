# Technical Documentation

## Table of Contents

1. [Game Model](#1-game-model)
2. [State Representation](#2-state-representation)
3. [Action Space](#3-action-space)
4. [Reward Design](#4-reward-design)
5. [Neural Network Architecture](#5-neural-network-architecture)
6. [Learning Algorithm](#6-learning-algorithm)
7. [Joker System](#7-joker-system)
8. [Run Structure and Shop](#8-run-structure-and-shop)
9. [Training Curriculum](#9-training-curriculum)
10. [File Reference](#10-file-reference)
11. [Hyperparameter Reference](#11-hyperparameter-reference)

---

## 1. Game Model

The agent plays a simplified version of Balatro. A standard 52-card deck is used with no card enhancements in the base curriculum.

**Card tuple format:** `(rank_index, suit, chip_value, extra_chips, extra_mult)`

| Field | Type | Range | Notes |
|-------|------|-------|-------|
| rank_index | int | 2–14 | Ace = 14, used for hand detection |
| suit | str | S/H/D/C | Spades, Hearts, Diamonds, Clubs |
| chip_value | int | 2–11 | Balatro values: J/Q/K=10, A=11 |
| extra_chips | int | 0+ | Card enhancement bonus chips |
| extra_mult | int | 0+ | Card enhancement bonus mult |

**Hand scoring formula:**

```
score = (base_chips + sum(chip_value + extra_chips)) * (base_mult + sum(extra_mult))
```

Only scoring cards contribute chips — kickers are excluded for pairs, trips, quads, and high card.

**Hand type base values:**

| Hand | Base Chips | Base Mult |
|------|-----------|-----------|
| High Card | 5 | 1 |
| One Pair | 10 | 2 |
| Two Pair | 20 | 2 |
| Three of a Kind | 30 | 3 |
| Straight | 30 | 4 |
| Flush | 35 | 4 |
| Full House | 40 | 4 |
| Four of a Kind | 60 | 7 |
| Straight Flush | 100 | 8 |
| Five of a Kind | 120 | 12 |
| Flush House | 140 | 14 |
| Flush Five | 160 | 16 |

**Per episode (single-blind mode):**
- 8 cards dealt from a shuffled 52-card deck
- Agent plays or discards 1-5 cards per step
- Up to 4 hand plays and 3 discards per episode
- Played cards are removed and replaced from the deck
- Episode ends when hands are exhausted or blind is cleared

---

## 2. State Representation

The state vector has **167 floats**, always the same length regardless of phase. Features not yet relevant (joker features in phase 1, run context in phases 1-9) are zeroed.

### 2.1 Per-card features (40 floats = 8 cards × 5)

For each card in the 8-card hand:

| Feature | Normalisation |
|---------|--------------|
| rank_index | / 14.0 |
| suit index (0-3) | / 3.0 |
| chip_value | / 11.0 (max Ace = 11) |
| extra_chips | / 50.0 |
| extra_mult | / 10.0 |

Positions beyond the current hand size are zero-padded (occurs near deck end).

### 2.2 Deck count vector (52 floats)

One float per unique card in the standard deck. 1.0 if the card is still in the draw pile, 0.0 if it has been dealt or played. Ordered by suit then rank. This gives the agent full information about which cards can still be drawn.

### 2.3 Scalars (4 floats)

| Feature | Formula | Notes |
|---------|---------|-------|
| hands_remaining_norm | hands_remaining / max_hand_plays | |
| discards_remaining_norm | discards_remaining / max_discards | 0 if discards disabled |
| score_progress | min(score / blind_target, 1.0) | 0 in phase 1 |
| score_velocity | min(score / expected_score, 2.0) / 2.0 | how efficiently scoring vs pace |

Score velocity: `expected_score = blind_target * (hands_played / max_hand_plays)`. A value of 0.5 means exactly on pace; above 0.5 means ahead of pace.

### 2.4 Draw features (15 floats)

All draw features are computed deck-aware — a completing rank or suit exhausted from the deck counts as 0 outs, so the agent never chases impossible draws. Features [0:13] are zeroed when `discards_remaining == 0` (draw signals are only actionable when discards are available).

| Index | Feature | Notes |
|-------|---------|-------|
| [0:4] | flush_outs_per_suit | per suit: min(cards_needed, cards_in_deck) / 13 |
| [4] | best_window_fill | fraction of best 5-rank window covered (0–1) |
| [5] | straight_outs | live completing ranks / 2. 1.0=open-ended, 0.5=gutshot/one-ended, 0.0=dead |
| [6] | straight_flush_proximity | best_window_fill × suit agreement in window |
| [7] | pair_to_trip_outs | copies of best pair's rank in deck / 3 |
| [8] | pair_to_twopair_outs | singleton ranks with live duplicates in deck / 8 |
| [9] | twopair_to_fh_outs | if two pairs exist: max copies of either pair rank / 2 |
| [10] | deck_depth | len(deck) / 52 (always included) |
| [11] | best_hand_rank | ordinal of best hand in current 8-card hand / 11 |
| [12] | pair_count | distinct pairs (freq >= 2) / 4 |
| [13] | trip_count | distinct trips (freq >= 3) / 2 |
| [14] | quad_count | distinct quads (freq >= 4) / 2 |

**Straight outs distinction:** 3-4-5-6 gets outs=1.0 (needs 2 or 7 — both checked in deck), A-2-3-4 gets outs=0.5 (only 5 completes it), 3-4-6-7 with all 5s dead gets outs=0.0.

### 2.5 Score context (21 floats)

Joker-aware score estimates computed by iterating all C(8,1)+...+C(8,5) = 218 subsets of the current hand per state call.

| Index | Feature | Notes |
|-------|---------|-------|
| [0:12] | hand_type_score_estimates | best joker-adjusted score per hand type / 1000 |
| [12] | score_needed_per_hand | max(0, blind-score) / hands_remaining / 1000 |
| [13] | best_immediate_play_score | best joker-adjusted score playable right now / 1000 |
| [14] | straight_draw_value | straight_outs × straight_potential / 1000 |
| [15] | flush_draw_value | flush_outs × flush_potential / 1000 |
| [16] | upgrade_draw_value | upgrade_outs × upgrade_potential / 1000 |
| [17] | expected_score_after_discard | best score achievable from kept cards after optimal discard / 1000; lower bound on post-discard value; 0 when no discards remain |
| [18] | discard_worth_it | 1.0 if expected_score_after_discard > best_immediate_play_score, else 0.0; pre-computes the play-vs-discard comparison; 0 when no discards remain |
| [19] | optimal_hand_type_target | argmax(hand_type_score_estimates) / 11; direct "aim for this hand type" signal; with jokers this differs from best_hand_rank — e.g. Crazy Joker makes Straight the target even when Trips is in hand |
| [20] | held_card_joker_activation | fraction of hand cards that would fire at least one on_card_scored joker / HAND_SIZE; tells the agent how well the current hand matches its jokers, not just what the jokers do |

**Kicker exclusion.** `get_score_components()` (used throughout score context) excludes kicker cards exactly as `scoring.py` does. Without this, High Card hands with high-value kickers (e.g. A-K-Q-J-10) had their chip total inflated to nearly match One Pair, collapsing the scoring gap from 3.75x to 1.29x and causing a High Card regression in joker phases. The fix restores the correct 2.25x ratio.

**Joker strategy features.** `optimal_hand_type_target` gives the agent a direct "aim for this" signal computed from the joker-adjusted score estimates — the agent no longer needs to find the argmax of 12 estimates implicitly. `held_card_joker_activation` bridges the gap between knowing what a joker does (from the joker feature block) and knowing how well the current hand aligns with it, which varies episode-to-episode as jokers are randomly assigned. Without this, High Card hands with high-value kickers (e.g. A-K-Q-J-10) had their chip total inflated to nearly match One Pair, collapsing the scoring gap from 3.75× to 1.29× and causing the High Card regression in joker phases. The fix restores the correct 2.25× ratio.

The three draw value features pre-multiply draw quality by joker-adjusted payoff for the target hand type. With Crazy Joker (+12 mult on straights), `straight_draw_value` is high even for a gutshot. With Zany Joker (+12 mult on trips), `upgrade_draw_value` is high for a pair with deck copies available. This lets the agent compare competing discard strategies without needing to learn the multiplication itself.

### 2.6 Joker features (30 floats)

Aggregate encoding of active joker effects. All zeros in phases 1-5.

| Index | Feature | Normalisation |
|-------|---------|--------------|
| [0:12] | mult_bonus_per_hand_type | / 100 |
| [12:24] | chip_bonus_per_hand_type | / 500 |
| [24] | passive_mult | / 100 |
| [25] | passive_chips | / 500 |
| [26:30] | per_suit_mult_bonus (S/H/D/C) | / 20 |

Hand type order for indices [0:12] and [12:24]: High Card, One Pair, Two Pair, Three of a Kind, Straight, Flush, Full House, Four of a Kind, Straight Flush, Five of a Kind, Flush House, Flush Five.

### 2.7 Run context (5 floats)

Zeroed in phases 1-9 (single-blind mode).

| Feature | Formula | Notes |
|---------|---------|-------|
| ante_progress | completed_blinds / total_blinds_in_run | |
| blind_in_ante | blind_idx / 2 (0=Small, 0.5=Big, 1.0=Boss) | |
| money_norm | min(current_gold / 20, 1.0) | |
| interest_next | calculate_interest(money) / 5 | Pre-computes the step function; lets agent plan around the $25 threshold where max interest ($5) is reached |
| joker_slot_utilisation | len(active_jokers) / MAX_JOKER_SLOTS | 0.0 = empty rack, 1.0 = full; tells agent whether buying a joker requires selling one first |

---

## 3. Action Space

Actions are 1-5 card selections from the 8-card hand. Each selection can be either a **play** (score the cards) or a **discard** (replace the cards). The combined action space has up to 438 actions: 219 play actions + 219 discard actions.

### 3.1 Action feature encoding (22 floats)

Each candidate action is encoded as a feature vector rather than an index, so the network can generalise across card combinations.

| Feature | Size | Notes |
|---------|------|-------|
| card selection mask | 8 | binary, which cards are selected |
| hand type one-hot | 12 | of selected cards (play) or kept cards (discard) |
| chip total | 1 | sum of selected/kept chip values / 55 |
| signed card count | 1 | +n/8 for plays, -n/8 for discards |

For **discard actions**, features describe the *kept* cards (what the agent is building toward), not the discarded cards. The negative sign on card_count is the signal that routes this action to the discard head.

### 3.2 Action routing

The network's play and discard heads are selected per-action based on `action_features[..., -1]`:
- Positive → play_head
- Negative → discard_head
- Zero → padding slot (masked during max-Q computation)

---

## 4. Reward Design

### 4.1 Play rewards

| Condition | Reward |
|-----------|--------|
| Phase 1 (no blind) | `round_score * 0.05` |
| Phase 2+ mid-blind | `round_score / blind_target` |
| Win (score >= blind) | `+REWARD_WIN_BONUS * (1 + hands_remaining / max_hands)` |
| Loss (hands exhausted) | `-REWARD_LOSS_PENALTY` |
| Illegal play | `-10.0`, episode ends |

Win bonus scales with efficiency — clearing the blind in 2 hands earns more than clearing it in 4.

### 4.2 Discard reward shaping

Previously discards gave `reward=0.0`, relying on delayed signal from the next played hand. To improve credit assignment:

```
rank_improvement = (hand_rank_after - hand_rank_before) / 11
reward = rank_improvement * DISCARD_REWARD_SCALE   # scale = 0.15
```

Hand rank ordinals: 0=High Card, 1=One Pair, ..., 11=Flush Five. A discard that improves the hand from High Card (0) to One Pair (1) gives reward ≈ 0.014. Scale is small enough that win/loss rewards dominate.

---

## 5. Neural Network Architecture

### 5.1 DQNNetwork

```
Input: concat(state [167], action_features [22]) = 189 floats

Trunk:
    Linear(183, 256) -> ReLU
    Linear(256, 256) -> ReLU

play_head:
    Linear(256, 128) -> ReLU -> Linear(128, 1)

discard_head:
    Linear(256, 128) -> ReLU -> Linear(128, 1)

Output: scalar Q-value (routed through play or discard head by action sign)
```

Total parameters: ~206k.

### 5.2 Split heads rationale

Play decisions optimise for immediate hand value; discard decisions optimise for draw potential improvement. A single head must trade off between these very different objectives. Separate heads allow each to specialise without interference, while the shared trunk learns general game representations.

---

## 6. Learning Algorithm

### 6.1 Double DQN

Standard DQN uses the target network for both selecting and evaluating the best next action, causing systematic Q-value overestimation. Double DQN separates these:

```python
# policy_net selects the best next action
best_actions = policy_net(next_states, next_action_feats).argmax(dim=1)

# target_net evaluates it
q_next = target_net(next_states, next_action_feats).gather(1, best_actions)

# N-step aware target
q_target = rewards + gamma^N * q_next * (1 - dones)
```

### 6.2 Prioritized Experience Replay (PER)

Transitions are sampled with probability proportional to `|TD error|^alpha` rather than uniformly. Critical transitions (ones the network got wrong) are replayed more frequently.

**Implementation:** A SumTree gives O(log n) priority-proportional sampling and O(log n) priority updates. Each leaf stores a priority; internal nodes store subtree sums.

**Parameters:**
- `alpha = 0.6` — priority exponent (0=uniform, 1=fully prioritised)
- `beta` — importance sampling correction, anneals from 0.4 to 1.0 over 200k steps

Importance sampling weights correct the bias introduced by non-uniform sampling:
```
w_i = (N * P(i))^(-beta)
```
Weights are normalised so the maximum weight in each batch equals 1.

### 6.3 N-step returns

Instead of 1-step TD, the agent accumulates 3 steps of actual reward before bootstrapping:

```
G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + gamma^3 * V(s_{t+3})
```

This gives discard actions direct credit for the improved hand they enable one or two steps later. The `NStepBuffer` maintains a queue of recent transitions and flushes the oldest entry with its accumulated n-step return into the PER buffer on each step. At episode end, all remaining entries are flushed.

The bootstrap discount is `gamma^N` (not `gamma`) since N steps of actual reward are already included in the target.

### 6.4 Target network

Soft update with `tau = 0.01` every episode:
```
target = 0.01 * policy + 0.99 * target
```

### 6.5 Gradient clipping

Gradients are clipped to max norm 10.0 before each parameter update to prevent exploding gradients.

### 6.6 Discount factors

| Mode | GAMMA | Rationale |
|------|-------|-----------|
| Single-blind (phases 1-9) | 0.90 | ~4 steps per episode, minimal discounting needed |
| Run mode (phases 10-12) | 0.97 | Up to 36 steps per run; early-blind decisions must see late-run reward |

`set_phase()` switches GAMMA automatically. The n-step buffer's gamma updates via `agent.update_gamma()`.

---

## 7. Joker System

### 7.1 JSON schema

Jokers are defined entirely in `jokers.json`. No code changes are needed to add a new joker unless it introduces a new trigger or condition type.

```json
{
  "id": "crazy_joker",
  "name": "Crazy Joker",
  "description": "+12 Mult if played hand contains a Straight",
  "rarity": "common",
  "tier": 1,
  "trigger": "on_hand_played",
  "condition": {"type": "hand_contains", "hand_type": "Straight"},
  "effect": {"type": "add_mult", "value": 12}
}
```

**Triggers:**
- `passive` — fires every hand unconditionally
- `on_hand_played` — fires once per hand if condition is met
- `on_card_scored` — fires once per qualifying scored card

**Conditions:**
- `none` — always true
- `hand_contains X` — true if played hand type contains pattern X (e.g. Full House contains a Pair)
- `hand_type_exact X` — true if hand type is exactly X
- `card_count_le N` — true if <= N cards played (Half Joker)
- `card_suit S` — per-card: true if card suit matches S

**Effects:**
- `add_mult` — add value to multiplier
- `add_chips` — add value to chips

### 7.2 Scoring with jokers

Joker bonuses are applied inside the chip×mult formula, not after:

```
score = (base_chips + card_chips + joker_chips) * (base_mult + card_mult + joker_mult)
```

### 7.3 Current joker pool (16 jokers)

**Tier 1 (11 jokers) — hand-type conditionals:**

| Joker | Effect |
|-------|--------|
| Joker | +4 Mult (passive) |
| Jolly Joker | +8 Mult if hand contains a Pair |
| Sly Joker | +50 Chips if hand contains a Pair |
| Mad Joker | +10 Mult if hand contains Two Pair |
| Clever Joker | +80 Chips if hand contains Two Pair |
| Zany Joker | +12 Mult if hand contains Three of a Kind |
| Wily Joker | +100 Chips if hand contains Three of a Kind |
| Crazy Joker | +12 Mult if hand contains a Straight |
| Devious Joker | +100 Chips if hand contains a Straight |
| Droll Joker | +10 Mult if hand contains a Flush |
| Crafty Joker | +80 Chips if hand contains a Flush |

**Tier 2 (5 jokers) — per-card and count conditionals:**

| Joker | Effect |
|-------|--------|
| Half Joker | +20 Mult if <= 3 cards played |
| Greedy Joker | +3 Mult per Diamond scored |
| Lusty Joker | +3 Mult per Heart scored |
| Wrathful Joker | +3 Mult per Spade scored |
| Gluttonous Joker | +3 Mult per Club scored |

### 7.4 Hand containment

`hand_contains "One Pair"` fires on: One Pair, Two Pair, Three of a Kind, Full House, Four of a Kind, Five of a Kind, Flush House, Flush Five. This matches real Balatro behaviour — Jolly Joker fires on Full Houses because a Full House contains a pair.

### 7.5 Joker state encoding

The 30-float joker feature block encodes aggregate effects, not joker identity. This generalises to new jokers automatically without state size changes.

---

## 8. Run Structure and Shop

### 8.1 Blind targets

Matches real Balatro scaling:

| Ante | Small Blind | Big Blind | Boss Blind |
|------|------------|-----------|-----------|
| 1 | 300 | 450 | 600 |
| 2 | 800 | 1,200 | 1,600 |
| 3 | 2,000 | 3,000 | 4,000 |
| 4 | 5,000 | 7,500 | 10,000 |

### 8.2 Money system

- Starting gold: $4
- Per cleared blind: base pay varies by position, plus $1 per hand remaining:

| Blind | Base Pay | Example (2 hands left) |
|-------|----------|----------------------|
| Small Blind | $3 | $5 |
| Big Blind | $4 | $6 |
| Boss Blind | $5 | $7 |

- **Interest:** earned after clearing each blind, before the shop. $1 per $5 held, capped at $5. Incentivises saving money rather than spending everything each shop.

| Gold held | Interest |
|-----------|---------|
| $0–4 | $0 |
| $5–9 | $1 |
| $10–14 | $2 |
| $15–19 | $3 |
| $20–24 | $4 |
| $25+ | $5 (cap) |

The $25 threshold is meaningful — saving to reach it gives the maximum $5/shop bonus. The agent sees `interest_next` in the run context state so it can plan around this threshold explicitly.

### 8.3 RunEnv architecture

`RunEnv` wraps `BalatroEnv`. The DQN sees a continuous state stream with no special handling — it steps until `done=True` which signals the run is over (failure or all antes complete). When a blind clears:

1. Money is earned
2. Shop runs (greedy heuristic)
3. Joker lineup may update
4. `_start_blind()` resets the inner env with the new blind target and updated jokers
5. A non-terminal state is returned (run continues)

### 8.4 Shop heuristic

The shop runs automatically between every cleared blind. The DQN is not involved.

**Scoring function:** for each offered joker, estimate expected per-hand score contribution:

```
value = sum over hand_types of:
    P(playing that hand) * expected_bonus(joker, hand_type)
```

Where bonus is converted to mult-equivalent units (`add_chips` values are divided by 15 as a chip-to-mult conversion ratio). A 20% synergy bonus applies if an owned joker already targets the same hand type.

**Decision logic:**
1. Score all offered jokers
2. If best offer score >= `SHOP_BUY_THRESHOLD` (0.5) and affordable and slot available: buy
3. Else if selling the weakest owned joker would fund the purchase and the upgrade margin is >= `SHOP_SELL_UPGRADE_MARGIN` (1.5×): sell then buy
4. Otherwise: skip

Only one transaction per shop visit.

---

## 9. Training Curriculum

### 9.1 Epsilon schedule

Epsilon decays multiplicatively: `epsilon = max(0.05, epsilon * 0.9995)`. At key phase transitions epsilon is reset to re-enable exploration:

| Phase | Reset | Reason |
|-------|-------|--------|
| 5 | 0.35 | Blind jumps from 450 to 600 |
| 6 | 0.30 | Jokers introduced for the first time |
| 8 | 0.20 | Tier-2 jokers require new discard strategies |
| 10 | 0.40 | Run structure is a fundamentally different task |
| 11 | 0.25 | Longer run, more ante scaling |
| 12 | 0.20 | Full run complexity |

### 9.2 Early stopping

Evaluated every 50 episodes:

- **Phases 2-12 (with target):** win rate >= 75% for 5 consecutive log intervals
- **Phase 1 (no target):** avg score improvement < 10 points for 5 consecutive intervals (plateau)
- Minimum episode floors prevent premature stopping on lucky streaks

### 9.3 Transfer learning

Agent weights, optimizer state, replay buffer, and epsilon carry forward between phases. The joker-aware state features are all zero in phases 1-5, so the network learns gracefully — the joker head weights start at random and begin receiving gradients only when jokers are introduced in phase 6.

---

## 10. File Reference

### `config.py`
Central configuration. All tunable values live here.

Key functions:
- `get_phase_settings(phase)` — returns dict for given phase
- `set_phase(phase)` — updates all module-level constants at runtime (used by curriculum runner)

### `scoring.py`
Pure functions, no state.

- `get_hand_rank(hand)` — returns hand type string
- `get_scoring_cards(hand, hand_rank)` — returns cards that contribute chips (kickers excluded)
- `calculate_score(hand)` — returns integer score

### `cards.py`
- `make_deck()` — returns shuffled list of 52 card tuples
- `deck_count_vector(deck)` — returns 52-float vector for state encoding

### `env.py`

**`BalatroEnv`** — single-blind environment
- `reset(**overrides)` — shuffle deck, deal hand, reset counters. Accepts blind_target_override, jokers_override, and run context for RunEnv injection
- `step(card_indices, play)` — returns (state, reward, done, info)
- `valid_play_actions()` — returns all C(8,1)+...+C(8,5) action index lists

**`RunEnv`** — full multi-blind run environment wrapping BalatroEnv
- `reset()` — starts new run from Ante 1 Small Blind, seeds initial jokers
- `step(card_indices, play)` — delegates to inner env, advances blind on win, runs shop
- `_advance_blind()` — increments blind/ante counters, runs shop, returns shop log
- `_start_blind()` — resets inner env with current ante/blind target and joker lineup

### `agent.py`

**`SumTree`** — O(log n) priority tree for PER

**`PrioritizedReplayBuffer`** — PER buffer using SumTree
- `push(...)` — stores at max priority
- `sample(batch_size)` — priority-proportional sampling with IS weights
- `update_priorities(indices, td_errors)` — update after gradient step

**`NStepBuffer`** — accumulates n transitions, pushes n-step returns to replay
- `push(...)` — adds transition, flushes oldest when buffer full or episode ends
- `update_gamma(gamma)` — updates discount factor for run phases

**`DQNNetwork`** — shared trunk + play/discard heads

**`DQNAgent`**
- `select_action(state, valid_actions, hand)` — epsilon-greedy
- `store(...)` — pushes through n-step buffer into PER
- `train_step()` — Double DQN update with PER weights, returns loss
- `update_gamma(gamma)` — propagates gamma change to n-step buffer
- `get_weights()` — serialise policy_net to bytes (for parallel worker sync)
- `set_weights(bytes)` — load serialised weights from learner (worker side)
- `save(path)` / `load(path)` — checkpoint persistence

### `jokers.py`

- `load_jokers(path)` — parses jokers.json, returns dict keyed by id
- `sample_episode_jokers(all_jokers, n, max_tier)` — random selection for single-blind phases
- `apply_jokers_on_hand(jokers, played_cards, hand_type)` — returns (bonus_chips, bonus_mult)
- `get_joker_state_features(jokers)` — returns 30-float state encoding
- `score_joker_value(joker, active_jokers)` — shop heuristic scoring
- `sample_shop(all_jokers, active_jokers, max_tier, n_slots)` — draws shop offerings
- `run_shop(...)` — greedy buy/sell decision, returns (updated_jokers, updated_money, log)

### `run_structure.py`

Constants and helpers for run progression:
- `BLIND_TARGETS` — ante→[small,big,boss] score targets
- `blind_target(ante, blind_idx)` — returns score target
- `money_for_win(hands_remaining)` — gold earned per cleared blind
- Shop constants: `MAX_JOKER_SLOTS`, `SHOP_JOKER_SLOTS`, `JOKER_BUY_PRICE`, thresholds

### `main.py`

- `make_env()` — factory: returns RunEnv or BalatroEnv based on config.USE_RUN_ENV
- `run_episode(env, agent)` — runs one full episode, returns stats dict
- `train_phase(phase, agent, logger, max_episodes, min_episodes)` — one phase loop with early stopping
- `run_curriculum(start_phase, resume_dir, logger, n_workers)` — full automated curriculum; passes n_workers to parallel_train_phase when > 1
- `run_single(episodes, resume_dir, logger)` — single phase (manual use)

### `parallel_train.py` *(new)*

Ape-X style parallel training. Decouples env generation (CPU workers) from gradient updates (GPU learner) to eliminate the env-step/GPU idle cycle.

- `parallel_train_phase(phase, agent, logger, max_episodes, min_episodes, n_workers)` — drop-in replacement for `train_phase`. Spawns N worker processes, runs the learner loop, broadcasts weight updates every `WEIGHT_SYNC_INTERVAL` steps.
- `_worker_fn(...)` — worker process entry point. Runs env loop with a local CPU copy of the policy net; pushes transition batches to a shared queue; polls a Pipe for fresh weights.

**Key constants (tunable at top of file):**

| Constant | Default | Effect |
|----------|---------|--------|
| `WEIGHT_SYNC_INTERVAL` | 200 | Learner steps between weight broadcasts |
| `WORKER_BATCH_SIZE` | 16 | Transitions per queue push |
| `MAX_QUEUE_SIZE` | 2000 | Queue depth cap (memory guard) |

---

## 11. Hyperparameter Reference

### Network and optimisation

| Parameter | Value | Notes |
|-----------|-------|-------|
| LEARNING_RATE | 1e-4 | Adam optimiser |
| GAMMA | 0.90 | Single-blind phases |
| GAMMA_RUN | 0.97 | Run phases (10-12), auto-switched by set_phase() |
| EPSILON_START | 1.0 | |
| EPSILON_END | 0.05 | Floor |
| EPSILON_DECAY | 0.9995 | Multiplicative per episode |
| BATCH_SIZE | 64 | |
| REPLAY_CAPACITY | 100,000 | PER buffer size |

### Prioritized replay

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_STEP | 3 | n-step return accumulation |
| PER_ALPHA | 0.6 | Priority exponent |
| PER_BETA_START | 0.4 | IS correction start |
| PER_BETA_END | 1.0 | IS correction end |
| PER_BETA_STEPS | 200,000 | Annealing duration |

### Reward shaping

| Parameter | Value | Notes |
|-----------|-------|-------|
| REWARD_RAW_SCORE_SCALE | 0.05 | Phase 1 score multiplier |
| REWARD_WIN_BONUS | 5.0 | Blind clear bonus base |
| REWARD_LOSS_PENALTY | 2.0 | Blind failure penalty |
| REWARD_ILLEGAL_ACTION | -10.0 | Playing with no hands |
| REWARD_ILLEGAL_DISCARD | -5.0 | Discarding with no discards |
| DISCARD_REWARD_SCALE | 0.15 | Hand rank improvement shaping |
| REWARD_MADE_HAND_BONUS | 0.05 | Bonus for playing One Pair or better when jokers active; counters High Card regression |
| REWARD_HAND_SIZE_BONUS | 0.1 | Per card beyond 1 in a play |

### Early stopping

| Parameter | Value | Notes |
|-----------|-------|-------|
| WIN_RATE_THRESHOLD | 0.75 | Advance when >= this |
| WIN_RATE_WINDOW | 5 | Consecutive log intervals |
| SCORE_PLATEAU_MIN | 10.0 | Phase 1 improvement threshold |
| PLATEAU_WINDOW | 5 | Consecutive plateau intervals |

### Shop (phases 10-12)

| Parameter | Value | Notes |
|-----------|-------|-------|
| MAX_JOKER_SLOTS | 5 | Maximum jokers held |
| SHOP_JOKER_SLOTS | 2 | Jokers offered per visit |
| JOKER_BUY_PRICE | {1: 4, 2: 6} | By tier |
| SHOP_BUY_THRESHOLD | 0.5 | Min value score to buy |
| SHOP_SELL_UPGRADE_MARGIN | 1.5 | How much better to trigger sell+buy |
