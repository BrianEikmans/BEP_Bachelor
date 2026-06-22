# =============================================================================
# Balatro DQN — Central Configuration
# =============================================================================
# Phases let you progressively increase environment complexity during training.
# Set TRAINING_PHASE to control which features are active.
#
#   Phase 1 — Baseline scoring
#       No discards, no blind target. Agent simply learns to maximise score
#       across a fixed number of hand plays.
#
#   Phase 2 — Blind target
#       Introduce a score threshold the agent must beat. Reward now includes
#       a bonus for clearing the blind and a penalty for falling short.
#
#   Phase 3 — Full game
#       Discards enabled. Agent can trade cards for better draws.
# =============================================================================

TRAINING_PHASE = 1


# -----------------------------------------------------------------------------
# Hand & deck
# -----------------------------------------------------------------------------
HAND_SIZE       = 8    # Cards held at once
DECK_SIZE       = 52   # Standard deck (no jokers at this stage)

# -----------------------------------------------------------------------------
# Phase-gated game rules
# (env.py reads these — do not use the raw values directly, use the helpers)
# -----------------------------------------------------------------------------

_PHASE_SETTINGS = {
    # --- Core curriculum (no jokers) ---
    1: {
        "max_hand_plays":    4,
        "max_discards":      0,
        "blind_target":      None,    # no blind — just maximise score
        "num_jokers":        0,
        "max_joker_tier":    0,
    },
    2: {
        "max_hand_plays":    4,
        "max_discards":      0,
        "blind_target":      300,     # small blind — teaches win condition
        "num_jokers":        0,
        "max_joker_tier":    0,
    },
    3: {
        "max_hand_plays":    4,
        "max_discards":      0,
        "blind_target":      450,     # large blind — consistent hand building
        "num_jokers":        0,
        "max_joker_tier":    0,
    },
    4: {
        "max_hand_plays":    4,
        "max_discards":      3,
        "blind_target":      450,     # discards introduced at familiar blind
        "num_jokers":        0,
        "max_joker_tier":    0,
    },
    5: {
        "max_hand_plays":    4,
        "max_discards":      3,
        "blind_target":      600,     # full game, no jokers
        "num_jokers":        0,
        "max_joker_tier":    0,
    },

    # --- Joker curriculum ---
    # Blind targets are calibrated so the agent must genuinely exploit jokers
    # to win consistently — not just coast on pair bonuses.
    # Rule of thumb: blind ~ no-joker baseline + 60% of expected joker gain.
    # This keeps win rate in the 65-80% range where learning is richest.
    6: {
        "max_hand_plays":    4,
        "max_discards":      3,
        "blind_target":      700,     # was 600 — 1 joker adds ~150-200, blind absorbs most
        "num_jokers":        1,
        "max_joker_tier":    1,
    },
    7: {
        "max_hand_plays":    4,
        "max_discards":      3,
        "blind_target":      850,     # was 650 — must start adapting hand selection
        "num_jokers":        2,
        "max_joker_tier":    1,
    },
    8: {
        "max_hand_plays":    4,
        "max_discards":      3,
        "blind_target":      1000,    # was 700 — tier-2 jokers, suit/count strategy needed
        "num_jokers":        3,
        "max_joker_tier":    2,
    },
    9: {
        "max_hand_plays":    4,
        "max_discards":      3,
        "blind_target":      1200,    # was 800 -- genuine joker exploitation required
        "num_jokers":        4,
        "max_joker_tier":    2,
    },

    # --- Run structure curriculum ---
    # Phases 10-12 use RunEnv instead of BalatroEnv.
    # blind_target is None because RunEnv derives targets from run_structure.py.
    # max_antes controls how many antes are in a full run.
    10: {
        "max_hand_plays":    4,
        "max_discards":      3,
        "blind_target":      None,    # RunEnv handles targets per-blind
        "num_jokers":        2,
        "max_joker_tier":    1,
        "max_antes":         1,       # 1 ante = 3 blinds (Small, Big, Boss)
        "use_run_env":       True,
    },
    11: {
        "max_hand_plays":    4,
        "max_discards":      3,
        "blind_target":      None,
        "num_jokers":        3,
        "max_joker_tier":    2,
        "max_antes":         2,       # 2 antes = 6 blinds
        "use_run_env":       True,
    },
    12: {
        "max_hand_plays":    4,
        "max_discards":      3,
        "blind_target":      None,
        "num_jokers":        4,
        "max_joker_tier":    2,
        "max_antes":         4,       # full run = 12 blinds
        "use_run_env":       True,
    },
}


def get_phase_settings(phase: int = None) -> dict:
    """Return the settings dict for the given phase.
    If phase is None, reads the current TRAINING_PHASE at call time.
    (Do NOT use TRAINING_PHASE as a default argument — defaults are
    evaluated once at import time and would not reflect set_phase() calls.)
    """
    if phase is None:
        phase = TRAINING_PHASE   # read current module-level value at call time
    if phase not in _PHASE_SETTINGS:
        raise ValueError(f"Unknown phase {phase}. Choose from {list(_PHASE_SETTINGS)}")
    return _PHASE_SETTINGS[phase]


def set_phase(phase: int) -> None:
    """
    Switch the active training phase at runtime.

    Updates TRAINING_PHASE and all derived module-level constants so that
    a newly constructed env will use the new phase settings.
    Call this between phases in the curriculum runner.

    Note: the DQN agent (weights, replay buffer, epsilon) is intentionally
    carried forward unchanged -- the curriculum relies on transfer learning.
    """
    import sys
    mod = sys.modules[__name__]

    if phase not in _PHASE_SETTINGS:
        raise ValueError(f"Unknown phase {phase}. Choose from {list(_PHASE_SETTINGS)}")

    s = _PHASE_SETTINGS[phase]
    mod.TRAINING_PHASE = phase
    mod.MAX_HAND_PLAYS = s["max_hand_plays"]
    mod.MAX_DISCARDS   = s["max_discards"]
    mod.BLIND_TARGET   = s["blind_target"]
    mod.NUM_JOKERS     = s["num_jokers"]
    mod.MAX_JOKER_TIER = s["max_joker_tier"]
    mod.USE_RUN_ENV    = s.get("use_run_env", False)
    mod.MAX_ANTES      = s.get("max_antes", 0)
    # Switch discount factor: run phases need higher gamma so early-blind
    # decisions can see reward from later in the run
    mod.GAMMA = mod.GAMMA_RUN if mod.USE_RUN_ENV else 0.90


# Convenience accessors (always reflect current TRAINING_PHASE)
MAX_HAND_PLAYS  = get_phase_settings()["max_hand_plays"]
MAX_DISCARDS    = get_phase_settings()["max_discards"]
BLIND_TARGET    = get_phase_settings()["blind_target"]   # None = disabled
NUM_JOKERS      = get_phase_settings()["num_jokers"]     # 0 = no jokers
MAX_JOKER_TIER  = get_phase_settings()["max_joker_tier"] # 0 = no jokers
USE_RUN_ENV     = get_phase_settings().get("use_run_env", False)
MAX_ANTES       = get_phase_settings().get("max_antes", 0)

# -----------------------------------------------------------------------------
# Curriculum schedule
# -----------------------------------------------------------------------------
# Maximum episodes to spend on each phase. The curriculum runner may advance
# early if the convergence criteria are met before the budget runs out.
# Minimum episodes are enforced so the agent always gets a baseline of
# experience before early-stopping kicks in.
#
# Convergence criteria (evaluated every LOG_INTERVAL episodes):
#   - For phases with a blind target: win rate >= WIN_RATE_THRESHOLD over
#     WIN_RATE_WINDOW consecutive log intervals, with low variance.
#   - For phase 1 (no blind): score plateau — improvement < SCORE_PLATEAU_MIN
#     over PLATEAU_WINDOW consecutive log intervals.
# -----------------------------------------------------------------------------

PHASE_EPISODES: dict = {
    1: 20_000,   # from scratch -- needs time to discover hand types
    2:  5_000,   # familiar blind, converges quickly
    3:  5_000,   # harder blind, no discards ceiling
    4: 20_000,   # discards introduced -- biggest behavioural jump
    5: 25_000,   # increased budget -- harder blind needs more exploration
    6: 15_000,   # 1 joker, higher blind -- needs to learn exploitation
    7: 15_000,   # 2 jokers + harder blind
    8: 20_000,   # tier-2 jokers change discard strategy
    9: 20_000,   # full complexity, single blind
    # Run phases -- episode = full run, not a single blind
    10: 15_000,  # 1 ante (3 blinds), 2 jokers -- learn blind-to-blind continuity
    11: 20_000,  # 2 antes (6 blinds), 3 jokers -- ante progression
    12: 30_000,  # 4 antes (12 blinds), 4 jokers -- full run
}

PHASE_MIN_EPISODES: dict = {
    1:  5_000,
    2:  2_000,
    3:  2_000,
    4:  5_000,
    5:  8_000,
    6:  5_000,
    7:  5_000,
    8:  5_000,
    9:  5_000,
    10: 5_000,
    11: 8_000,
    12: 10_000,
}

# Per-phase epsilon resets.
PHASE_EPSILON_RESET: dict = {
    1:  None,
    2:  None,
    3:  None,
    4:  None,
    5:  0.35,   # blind jumps to 600
    6:  0.30,   # jokers introduced
    7:  None,
    8:  0.35,   # tier-2 jokers change strategy; was 0.20, raised to fix High Card regression
    9:  0.20,   # full joker complexity; small reset to re-explore at hardest blind
    10: 0.40,   # run structure is a fundamentally different task
    11: 0.25,   # longer run, more ante scaling
    12: 0.20,   # full run complexity
}

# Early stopping thresholds
WIN_RATE_THRESHOLD  = 0.75   # advance when win rate >= 75% ...
WIN_RATE_WINDOW     = 5      # ... sustained over this many consecutive log intervals
SCORE_PLATEAU_MIN   = 10.0   # phase 1: advance when avg score improves < this per window
PLATEAU_WINDOW      = 5      # consecutive log intervals of plateau before advancing

# -----------------------------------------------------------------------------
# State vector sizes (computed — do not edit manually)
# -----------------------------------------------------------------------------
# Per card      : rank_index, suit, chip_value, extra_chips, extra_mult  →  5
# Deck counts   : one slot per unique card (52)                          → 52
# Scalars       : hands_remaining, discards_remaining,                   →  4
#                 score_progress, score_velocity
# Draw features : flush_outs(4), best_window_fill(1), straight_outs(1), → 15
#                 straight_flush_proximity(1), deck_depth(1),
#                 best_hand_rank(1), pair_to_trip_outs(1),
#                 pair_to_twopair_outs(1), twopair_to_fh_outs(1),
#                 pair_count(1), trip_count(1), quad_count(1)
# Score context : hand_type_score_estimates(12), score_needed_per_hand(1),-> 21
#                 best_immediate_play_score(1), straight_draw_value(1),
#                 flush_draw_value(1), upgrade_draw_value(1),
#                 expected_score_after_discard(1), discard_worth_it(1),
#                 optimal_hand_type_target(1), held_card_joker_activation(1)
# Joker features: mult_per_hand_type(12), chips_per_hand_type(12),       -> 30
#                 passive_mult(1), passive_chips(1), per_suit_mult(4)
# Run context   : ante_progress(1), blind_in_ante(1), money(1),           ->  5
#                 interest_next(1), joker_slot_utilisation(1)
#
# Total: 40 + 52 + 4 + 15 + 21 + 30 + 5 = 167
# -----------------------------------------------------------------------------

_SCALAR_COUNT         = 4
_DRAW_FEATURE_COUNT   = 15
_SCORE_CONTEXT_COUNT  = 21
_JOKER_FEATURE_COUNT  = 30
_RUN_CONTEXT_COUNT    = 5
STATE_SIZE = (HAND_SIZE * 5 + DECK_SIZE + _SCALAR_COUNT + _DRAW_FEATURE_COUNT
              + _SCORE_CONTEXT_COUNT + _JOKER_FEATURE_COUNT + _RUN_CONTEXT_COUNT)

# -----------------------------------------------------------------------------
# Reward shaping
# -----------------------------------------------------------------------------
REWARD_RAW_SCORE_SCALE  = 0.05   # strong enough that score differences dominate
REWARD_WIN_BONUS        = 5.0    # added on clearing blind  (phase 2+)
REWARD_LOSS_PENALTY     = 2.0    # subtracted on failing blind (phase 2+)
REWARD_ILLEGAL_ACTION   = -10.0  # playing with no hands left
REWARD_ILLEGAL_DISCARD  = -5.0   # discarding with no discards left

# -----------------------------------------------------------------------------
# DQN hyper-parameters
# -----------------------------------------------------------------------------
LEARNING_RATE   = 1e-4
# GAMMA: single-blind phases use 0.90 (only ~4 steps per episode).
# Run phases (10-12) need higher discount -- a run has up to 36 steps
# (4 antes x 3 blinds x ~4 hands) and early hand decisions must see reward
# from later blinds. At 0.90, a step discounted 36 times = 0.02 (near zero).
# set_phase() updates GAMMA automatically when entering a run phase.
GAMMA           = 0.90
GAMMA_RUN       = 0.97    # used for phases 10-12
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 0.9995
BATCH_SIZE      = 64
REPLAY_CAPACITY = 100_000  # increased from 50k -- PER needs larger buffer to be effective
TARGET_UPDATE   = 100

# N-step returns
# Accumulate N steps of actual reward before bootstrapping from target network.
# Better credit assignment for discard decisions whose payoff comes 1-2 steps later.
# n=3 balances bias (too high) vs variance (too low).
N_STEP = 3

# Prioritized Experience Replay (PER)
# PER_ALPHA  : how strongly to prioritise by TD error (0=uniform, 1=fully prioritised)
# PER_BETA_* : importance-sampling correction. Starts low (more bias), anneals to 1.0
#              (fully corrected) over training. Low beta early = more aggressive prioritisation.
PER_ALPHA      = 0.6
PER_BETA_START = 0.4
PER_BETA_END   = 1.0
PER_BETA_STEPS = 200_000   # anneal beta over this many training steps

# Discard reward shaping
# Small immediate reward for improving the best achievable hand rank after a discard.
# Gives the agent direct feedback on discard quality instead of waiting for the next play.
# Scale is intentionally small -- win/loss rewards must dominate.
DISCARD_REWARD_SCALE = 0.15

# Made-hand floor bonus (joker phases only, phases 6+)
# Small bonus added to play reward when the agent plays One Pair or better.
# Directly counters the High Card regression: agents with jokers sometimes
# discard aggressively, exhaust their discards, and end up playing High Card
# because it is the only available action. This bonus makes any made hand
# more attractive than High Card without penalising early phases where
# High Card is legitimately the best available hand.
# Applied inside the score reward, not on top of win/loss bonuses.
# Scale: 0.05 is ~5% of a typical round score reward -- noticeable but not dominant.
REWARD_MADE_HAND_BONUS = 0.05

# Reward bonus per card played beyond 1
REWARD_HAND_SIZE_BONUS = 0.1