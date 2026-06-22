"""
run_structure.py -- Balatro run progression tables.

A Balatro run consists of multiple Antes.
Each Ante has three blinds in order: Small Blind, Big Blind, Boss Blind.
The player must clear each blind in sequence. Failing any blind ends the run.

Blind targets are based on real Balatro values, capped at Ante 4 for
initial training (without booster packs the agent cannot reliably reach
higher antes).

Money is earned after each cleared blind:
    base    : $5
    bonus   : $1 per hand remaining when the blind was cleared
Starting money: $4
"""

# ---------------------------------------------------------------------------
# Blind target table
# ---------------------------------------------------------------------------
# Keys are ante numbers (1-indexed). Values are [small, big, boss] targets.
# These match real Balatro scaling.

BLIND_TARGETS: dict[int, list[int]] = {
    1: [300,   450,    600],
    2: [800,   1_200,  1_600],
    3: [2_000, 3_000,  4_000],
    4: [5_000, 7_500, 10_000],
}

# Maximum ante supported in the current implementation
MAX_ANTES: int = max(BLIND_TARGETS.keys())   # 4

# Number of blinds per ante (always 3 in Balatro: Small, Big, Boss)
BLINDS_PER_ANTE: int = 3

BLIND_NAMES: list[str] = ["Small Blind", "Big Blind", "Boss Blind"]

# ---------------------------------------------------------------------------
# Money system
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Money system
# ---------------------------------------------------------------------------

STARTING_MONEY: int = 4

# Base gold earned per cleared blind, indexed by blind position
# Small Blind = $3, Big Blind = $4, Boss Blind = $5 (real Balatro values)
MONEY_WIN_BASE: dict[int, int] = {
    0: 3,   # Small Blind
    1: 4,   # Big Blind
    2: 5,   # Boss Blind
}

# Bonus gold per hand remaining when the blind is cleared
MONEY_WIN_BONUS: int = 1

# Interest: $1 per $5 held, capped at $5
# e.g. $0-4 -> $0, $5-9 -> $1, $10-14 -> $2, ..., $25+ -> $5
INTEREST_PER_5: int = 1
INTEREST_CAP:   int = 5


def money_for_win(hands_remaining: int, blind_idx: int) -> int:
    """Gold earned after clearing a blind (before interest).

    Args:
        hands_remaining : how many hand plays were unused
        blind_idx       : 0=Small, 1=Big, 2=Boss
    """
    base = MONEY_WIN_BASE.get(blind_idx, 3)
    return base + hands_remaining * MONEY_WIN_BONUS


def calculate_interest(money: int) -> int:
    """
    Interest earned on held gold at the start of each shop visit.

    $1 per $5 held, capped at $5.
    e.g. $12 -> $2 interest, $25 -> $5 interest, $40 -> $5 interest (capped).
    Applied after earning blind money, before the shop.
    """
    return min(money // 5, INTEREST_CAP)


def blind_target(ante: int, blind_idx: int) -> int:
    """
    Return the score target for a given ante and blind position.

    Args:
        ante      : 1-indexed ante number
        blind_idx : 0 = Small, 1 = Big, 2 = Boss
    """
    if ante not in BLIND_TARGETS:
        raise ValueError(f"Ante {ante} not in BLIND_TARGETS (max {MAX_ANTES})")
    if not 0 <= blind_idx < BLINDS_PER_ANTE:
        raise ValueError(f"blind_idx must be 0-{BLINDS_PER_ANTE - 1}")
    return BLIND_TARGETS[ante][blind_idx]


def blind_name(ante: int, blind_idx: int) -> str:
    """Return a human-readable label e.g. 'Ante 2 - Big Blind (1200)'."""
    target = blind_target(ante, blind_idx)
    return f"Ante {ante} - {BLIND_NAMES[blind_idx]} ({target:,})"


def total_blinds(max_ante: int) -> int:
    """Total number of blinds in a full run up to max_ante."""
    return max_ante * BLINDS_PER_ANTE


def run_progress(ante: int, blind_idx: int, max_ante: int) -> float:
    """
    Normalised position within the run [0, 1).
    0.0 = start of run, approaching 1.0 = near end.
    """
    completed = (ante - 1) * BLINDS_PER_ANTE + blind_idx
    return completed / total_blinds(max_ante)


# ---------------------------------------------------------------------------
# Shop system
# ---------------------------------------------------------------------------

# Maximum joker slots the player can hold
MAX_JOKER_SLOTS: int = 5

# Number of joker slots offered in the shop each visit
SHOP_JOKER_SLOTS: int = 2

# Joker buy prices by tier (in gold)
JOKER_BUY_PRICE: dict[int, int] = {
    1: 4,   # tier-1: simple hand-type conditionals
    2: 6,   # tier-2: per-card / count conditionals
}

# Sell price = floor(buy_price / 2)
def joker_sell_price(tier: int) -> int:
    return JOKER_BUY_PRICE[tier] // 2


# Heuristic threshold: only buy a joker if its value score exceeds this.
# Prevents buying weak jokers when money might be needed later.
SHOP_BUY_THRESHOLD: float = 0.5

# Sell-to-upgrade threshold: sell an owned joker if a shop joker is this
# much better (in value score units) and we can afford it after selling.
SHOP_SELL_UPGRADE_MARGIN: float = 1.5

# Reroll base cost ($1 in real Balatro, increases per reroll within a visit)
# We only allow one reroll per visit so cost is always this value.
SHOP_REROLL_COST: int = 1

