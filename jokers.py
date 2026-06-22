"""
jokers.py — Joker loading, effect application, and state encoding.

Adding a new joker:
    1. Add an entry to jokers.json following the schema in that file's _comment.
    2. No code changes needed unless you're introducing a brand-new trigger or
       condition type — in that case add a handler in _condition_met() and
       apply_jokers_on_hand(), and a feature encoder in get_joker_state_features().

Supported trigger types (covers all 16 current jokers):
    passive          Always fires every hand.
    on_hand_played   Fires once per hand if the condition is met.
    on_card_scored   Fires once per qualifying card in the played hand.

Supported condition types:
    none             Always true.
    hand_contains    True if the played hand type contains the given pattern.
    hand_type_exact  True if the played hand type is exactly the given type.
    card_count_le    True if cards played <= value.
    card_suit        Per-card: true if the card's suit matches.

Supported effect types:
    add_mult         Add value to the multiplier.
    add_chips        Add value to the chip count.
"""

import json
import random
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Hand containment map
# ---------------------------------------------------------------------------
# "hand_contains X" is true when the played hand type contains pattern X.
# e.g. a Full House contains a Pair AND a Three of a Kind.
# This lets Jolly Joker (+mult if contains Pair) fire on Full Houses too.

HAND_CONTAINS: Dict[str, set] = {
    "One Pair": {
        "One Pair", "Two Pair", "Three of a Kind",
        "Full House", "Four of a Kind", "Five of a Kind",
        "Flush House", "Flush Five",
    },
    "Two Pair": {
        "Two Pair", "Full House", "Flush House",
    },
    "Three of a Kind": {
        "Three of a Kind", "Full House", "Four of a Kind",
        "Five of a Kind", "Flush House", "Flush Five",
    },
    "Straight": {
        "Straight", "Straight Flush",
    },
    "Flush": {
        "Flush", "Flush House", "Flush Five",
    },
    "Full House": {
        "Full House", "Flush House",
    },
    "Four of a Kind": {
        "Four of a Kind", "Five of a Kind", "Flush Five",
    },
    "Straight Flush": {
        "Straight Flush",
    },
    "Five of a Kind": {
        "Five of a Kind", "Flush Five",
    },
    "Flush House": {
        "Flush House", "Flush Five",
    },
    "Flush Five": {
        "Flush Five",
    },
}

# Ordered hand types — index used in the 12-element state feature slots
HAND_TYPE_ORDER = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Five of a Kind", "Flush House", "Flush Five",
]
HAND_TYPE_IDX = {h: i for i, h in enumerate(HAND_TYPE_ORDER)}

# Suit order — must match cards.SUITS
SUITS_ORDER = ["S", "H", "D", "C"]

# ---------------------------------------------------------------------------
# Base scoring tables (mirrors scoring.py internals)
# Used to compute chip/mult components so joker bonuses apply correctly
# inside (chips + j_chips) * (mult + j_mult) rather than to the final product.
# ---------------------------------------------------------------------------
BASE_SCORE_COMPONENTS: Dict[str, Tuple[int, int]] = {
    "High Card":       (  5,  1),
    "One Pair":        ( 10,  2),
    "Two Pair":        ( 20,  2),
    "Three of a Kind": ( 30,  3),
    "Straight":        ( 30,  4),
    "Flush":           ( 35,  4),
    "Full House":      ( 40,  4),
    "Four of a Kind":  ( 60,  7),
    "Straight Flush":  (100,  8),
    "Five of a Kind":  (120, 12),
    "Flush House":     (140, 14),
    "Flush Five":      (160, 16),
}


def get_score_components(played_cards: list, hand_type: str) -> Tuple[int, int]:
    """
    Return (total_chips, total_mult) for a played hand.

    Returns components separately so joker bonuses apply correctly:
        final_score = (chips + j_chips) * (mult + j_mult)

    Kicker cards are excluded exactly as in scoring.py — only scoring cards
    contribute chips. This is critical for correct relative hand valuation:
    without it, high-card kickers inflate chip totals for weaker hands and
    collapse the scoring gap between High Card and made hands.
    """
    from scoring import get_scoring_cards
    base_chips, base_mult = BASE_SCORE_COMPONENTS.get(hand_type, (5, 1))
    scoring_cards         = get_scoring_cards(played_cards, hand_type)
    card_chips            = sum(c[2] + c[3] for c in scoring_cards)  # chip_value + extra_chips
    card_mult             = sum(c[4] for c in scoring_cards)          # extra_mult
    return base_chips + card_chips, base_mult + card_mult


# ---------------------------------------------------------------------------
# Normalisation caps for state feature encoding
# ---------------------------------------------------------------------------
_MULT_CAP  = 100.0   # max expected total mult bonus from jokers
_CHIP_CAP  = 500.0   # max expected total chip bonus from jokers
_SUIT_CAP  =  20.0   # max per-card suit mult bonus (5 cards × 4 = 20 max per suit joker)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_jokers(path: str = "jokers.json") -> Dict[str, dict]:
    """
    Load all joker definitions from the JSON file.
    Returns a dict keyed by joker id.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return {j["id"]: j for j in data["jokers"]}


def sample_episode_jokers(
    all_jokers: Dict[str, dict],
    num_jokers: int,
    max_tier: int,
) -> List[dict]:
    """
    Randomly select `num_jokers` distinct jokers from those with tier <= max_tier.
    Returns an empty list for phases 1-5 where num_jokers == 0.
    """
    if num_jokers == 0:
        return []
    eligible = [j for j in all_jokers.values() if j["tier"] <= max_tier]
    k = min(num_jokers, len(eligible))
    return random.sample(eligible, k)


def apply_jokers_on_hand(
    jokers: List[dict],
    played_cards: list,
    hand_type: str,
) -> Tuple[int, int]:
    """
    Compute the total (bonus_chips, bonus_mult) contributed by all active jokers
    for a single hand play.

    Args:
        jokers       : list of active joker dicts for this episode
        played_cards : card tuples that were played (rank_idx, suit, chip, ec, em)
        hand_type    : string hand rank e.g. "Full House"

    Returns:
        (bonus_chips, bonus_mult) — added inside the scoring formula, not after
    """
    bonus_chips = 0
    bonus_mult  = 0

    for joker in jokers:
        trigger = joker["trigger"]
        effect  = joker["effect"]
        val     = effect["value"]

        if trigger in ("passive", "on_hand_played"):
            if _condition_met(joker, played_cards, hand_type):
                if effect["type"] == "add_mult":
                    bonus_mult  += val
                elif effect["type"] == "add_mult_random":
                    # Random range — sample uniformly each hand (e.g. Misprint)
                    bonus_mult  += random.randint(int(effect["min"]), int(effect["max"]))
                elif effect["type"] == "add_chips":
                    bonus_chips += val

        elif trigger == "on_card_scored":
            cond = joker["condition"]
            if cond["type"] == "card_suit":
                count = sum(1 for c in played_cards if c[1] == cond["suit"])
                if effect["type"] == "add_mult":
                    bonus_mult  += val * count
                elif effect["type"] == "add_chips":
                    bonus_chips += val * count
            elif cond["type"] == "card_rank_in":
                qualifying = set(cond["ranks"])
                count = sum(1 for c in played_cards if c[0] in qualifying)
                if effect["type"] == "add_mult":
                    bonus_mult  += val * count
                elif effect["type"] == "add_chips":
                    bonus_chips += val * count

    return bonus_chips, bonus_mult


def get_joker_state_features(jokers: List[dict]) -> List[float]:
    """
    Encode the active joker set as 30 normalised floats for the state vector.

    This is an aggregate encoding — the agent sees combined joker effects
    rather than individual joker identities. This scales cleanly as more
    jokers are added without growing the state size.

    Layout (30 floats):
        [ 0:12]  mult_bonus_per_hand_type   total mult bonus if that hand is played / _MULT_CAP
        [12:24]  chip_bonus_per_hand_type   total chip bonus if that hand is played / _CHIP_CAP
        [24]     passive_mult               always-on mult from passive jokers       / _MULT_CAP
        [25]     passive_chips              always-on chips from passive jokers      / _CHIP_CAP
        [26:30]  per_suit_mult_bonus        mult bonus per scored card of each suit  / _SUIT_CAP
                                            order: S, H, D, C
    """
    mult_per_type  = [0.0] * 12
    chips_per_type = [0.0] * 12
    passive_mult   = 0.0
    passive_chips  = 0.0
    suit_mult      = [0.0] * 4   # S, H, D, C

    for joker in jokers:
        trigger = joker["trigger"]
        effect  = joker["effect"]
        val     = float(effect["value"])

        if trigger == "passive":
            if effect["type"] == "add_mult":
                passive_mult  += val
            elif effect["type"] == "add_chips":
                passive_chips += val

        elif trigger == "on_hand_played":
            cond = joker["condition"]

            if cond["type"] == "none":
                # Unconditional on_hand_played — same as passive
                if effect["type"] == "add_mult":
                    passive_mult  += val
                elif effect["type"] == "add_chips":
                    passive_chips += val

            elif cond["type"] == "hand_contains":
                # Mark every hand type that contains the required pattern
                required = cond["hand_type"]
                for ht, idx in HAND_TYPE_IDX.items():
                    if ht in HAND_CONTAINS.get(required, {required}):
                        if effect["type"] == "add_mult":
                            mult_per_type[idx]  += val
                        elif effect["type"] == "add_chips":
                            chips_per_type[idx] += val

            elif cond["type"] == "hand_type_exact":
                idx = HAND_TYPE_IDX.get(cond["hand_type"])
                if idx is not None:
                    if effect["type"] == "add_mult":
                        mult_per_type[idx]  += val
                    elif effect["type"] == "add_chips":
                        chips_per_type[idx] += val

            elif cond["type"] == "card_count_le":
                # Can't pre-compute exactly — encode as a partial passive signal
                # so the agent knows something triggers on small hands
                if effect["type"] == "add_mult":
                    passive_mult  += val * 0.5
                elif effect["type"] == "add_chips":
                    passive_chips += val * 0.5

        elif trigger == "on_card_scored":
            cond = joker["condition"]
            if cond["type"] == "card_suit":
                try:
                    suit_idx = SUITS_ORDER.index(cond["suit"])
                except ValueError:
                    continue
                if effect["type"] == "add_mult":
                    suit_mult[suit_idx] += val
            elif cond["type"] == "card_rank_in":
                # card_rank_in jokers fire on multiple ranks — treat as passive
                # signal since they fire broadly (encoded in passive_mult)
                fraction = len(cond["ranks"]) * 4 / 52   # fraction of deck
                if effect["type"] == "add_mult":
                    passive_mult  += val * fraction * 4   # ~avg 4 scored cards
                elif effect["type"] == "add_chips":
                    passive_chips += val * fraction * 4

    # Normalise and assemble
    features  = [v / _MULT_CAP  for v in mult_per_type]   # [0:12]
    features += [v / _CHIP_CAP  for v in chips_per_type]  # [12:24]
    features += [passive_mult  / _MULT_CAP]                # [24]
    features += [passive_chips / _CHIP_CAP]                # [25]
    features += [v / _SUIT_CAP  for v in suit_mult]        # [26:30]

    return features   # length 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _condition_met(joker: dict, played_cards: list, hand_type: str) -> bool:
    """Return True if the joker's trigger condition is satisfied."""
    cond  = joker["condition"]
    ctype = cond["type"]

    if ctype == "none":
        return True

    if ctype == "hand_contains":
        required = cond["hand_type"]
        return hand_type in HAND_CONTAINS.get(required, {required})

    if ctype == "hand_type_exact":
        return hand_type == cond["hand_type"]

    if ctype == "card_count_le":
        return len(played_cards) <= cond["value"]

    if ctype == "card_rank_in":
        # True if any played card has a qualifying rank
        qualifying = set(cond["ranks"])
        return any(c[0] in qualifying for c in played_cards)

    return False


# ---------------------------------------------------------------------------
# Shop engine
# ---------------------------------------------------------------------------

# Rarity appearance weights — match real Balatro shop probabilities.
# Common jokers appear much more frequently than uncommon/rare.
_RARITY_WEIGHTS: Dict[str, float] = {
    "common":   0.70,
    "uncommon": 0.25,
    "rare":     0.05,
}

# Expected probability of playing each hand type, used to weight joker value.
_HAND_TYPE_PLAY_PROB: Dict[str, float] = {
    "High Card":       0.05,
    "One Pair":        0.20,
    "Two Pair":        0.25,
    "Three of a Kind": 0.15,
    "Straight":        0.08,
    "Flush":           0.10,
    "Full House":      0.10,
    "Four of a Kind":  0.04,
    "Straight Flush":  0.02,
    "Five of a Kind":  0.00,
    "Flush House":     0.00,
    "Flush Five":      0.00,
}

# Average number of cards scored per hand play
_AVG_SCORED_CARDS: float = 4.0

# chips are worth roughly 1/15 of a mult point at typical scoring levels
_CHIP_TO_MULT_RATIO: float = 1.0 / 15.0

# Fraction of deck covered by each rank set (for card_rank_in jokers)
_DECK_SIZE = 52
_CARDS_PER_RANK = 4   # one per suit


def _ranks_in_deck_fraction(ranks: list) -> float:
    """Fraction of the deck covered by the given rank list."""
    return len(ranks) * _CARDS_PER_RANK / _DECK_SIZE


def score_joker_value(joker: dict, active_jokers: List[dict] = None) -> float:
    """
    Estimate the per-hand expected score contribution of a joker.

    Returns a comparable float — higher is better. Used by the shop
    heuristic to rank offered jokers and decide buy/sell/reroll.

    Handles all current trigger/condition/effect types including:
        add_mult_random : uses expected value (min+max)/2
        card_rank_in    : value scales with how many ranks qualify
    """
    if active_jokers is None:
        active_jokers = []

    trigger = joker["trigger"]
    effect  = joker["effect"]
    cond    = joker["condition"]

    # Expected effect value
    if effect["type"] == "add_mult":
        eff_val = float(effect["value"])
    elif effect["type"] == "add_mult_random":
        # Use expected value for random-range jokers (e.g. Misprint)
        eff_val = (float(effect["min"]) + float(effect["max"])) / 2.0
    elif effect["type"] == "add_chips":
        eff_val = float(effect["value"]) * _CHIP_TO_MULT_RATIO
    else:
        eff_val = float(effect.get("value", 0))

    total = 0.0

    if trigger == "passive":
        total = eff_val * sum(_HAND_TYPE_PLAY_PROB.values())

    elif trigger == "on_hand_played":
        ctype = cond["type"]

        if ctype == "none":
            total = eff_val * sum(_HAND_TYPE_PLAY_PROB.values())

        elif ctype in ("hand_contains", "hand_type_exact"):
            required = cond["hand_type"]
            qualifying = (HAND_CONTAINS.get(required, {required})
                          if ctype == "hand_contains" else {required})
            total = eff_val * sum(
                _HAND_TYPE_PLAY_PROB.get(ht, 0) for ht in qualifying
            )

        elif ctype == "card_count_le":
            total = eff_val * 0.15

    elif trigger == "on_card_scored":
        ctype = cond["type"]

        if ctype == "card_suit":
            # One suit = 1/4 of scored cards on average
            total = eff_val * (_AVG_SCORED_CARDS / 4.0)

        elif ctype == "card_rank_in":
            # Fraction of deck that qualifies × avg scored cards
            fraction = _ranks_in_deck_fraction(cond["ranks"])
            total    = eff_val * (_AVG_SCORED_CARDS * fraction)

    # Synergy bonus: 20% if an owned joker targets the same hand type
    if active_jokers and cond["type"] in ("hand_contains", "hand_type_exact"):
        required = cond.get("hand_type", "")
        for owned in active_jokers:
            owned_ht = owned["condition"].get("hand_type", "")
            if owned_ht == required and owned["id"] != joker["id"]:
                total *= 1.2
                break

    return total


def _apply_joker_effect_on_scored(joker: dict, played_cards: list) -> Tuple[int, int]:
    """
    Compute (bonus_chips, bonus_mult) for on_card_scored jokers that need
    per-card rank checking (card_rank_in condition).
    """
    effect = joker["effect"]
    cond   = joker["condition"]
    val    = effect.get("value", 0)

    if cond["type"] == "card_rank_in":
        qualifying_ranks = set(cond["ranks"])
        count = sum(1 for c in played_cards if c[0] in qualifying_ranks)
        if effect["type"] == "add_mult":
            return 0, val * count
        elif effect["type"] == "add_chips":
            return val * count, 0

    return 0, 0


def sample_shop(
    all_jokers:    Dict[str, dict],
    active_jokers: List[dict],
    max_tier:      int,
    n_slots:       int,
) -> List[dict]:
    """
    Sample N distinct jokers for the shop using rarity-weighted probabilities.

    Uses two-stage sampling:
      1. Pick a rarity tier by weight (common 70%, uncommon 25%, rare 5%)
      2. Pick uniformly from eligible jokers of that rarity

    This guarantees the intended rarity distribution regardless of how many
    jokers exist at each rarity level. Falls back to uniform sampling if a
    chosen rarity has no eligible jokers.

    Jokers already owned and those above max_tier are excluded.
    """
    owned_ids = {j["id"] for j in active_jokers}
    eligible  = [
        j for j in all_jokers.values()
        if j["tier"] <= max_tier and j["id"] not in owned_ids
    ]
    if not eligible:
        return []

    # Group eligible by rarity
    by_rarity: Dict[str, list] = {}
    for j in eligible:
        by_rarity.setdefault(j["rarity"], []).append(j)

    rarities        = list(_RARITY_WEIGHTS.keys())
    rarity_weights  = [_RARITY_WEIGHTS[r] for r in rarities]

    chosen   = []
    used_ids = set()

    for _ in range(min(n_slots, len(eligible))):
        # Two-stage: pick rarity, then joker within that rarity
        # If the chosen rarity is exhausted, fall back to any remaining joker
        pool = list(eligible)
        pool = [j for j in pool if j["id"] not in used_ids]
        if not pool:
            break

        # Weighted rarity selection
        total_w = sum(rarity_weights[i] for i, r in enumerate(rarities)
                      if by_rarity.get(r))
        if total_w == 0:
            pick_rarity = None
        else:
            r_val = random.random() * total_w
            cumulative = 0.0
            pick_rarity = rarities[-1]
            for i, r in enumerate(rarities):
                if not by_rarity.get(r):
                    continue
                cumulative += rarity_weights[i]
                if r_val <= cumulative:
                    pick_rarity = r
                    break

        rarity_pool = [j for j in (by_rarity.get(pick_rarity) or [])
                       if j["id"] not in used_ids]

        if rarity_pool:
            pick = random.choice(rarity_pool)
        else:
            # Chosen rarity exhausted — fall back to any remaining eligible
            remaining = [j for j in eligible if j["id"] not in used_ids]
            if not remaining:
                break
            pick = random.choice(remaining)

        chosen.append(pick)
        used_ids.add(pick["id"])

    return chosen


def run_shop(
    all_jokers:     Dict[str, dict],
    active_jokers:  List[dict],
    money:          int,
    max_tier:       int,
    max_slots:      int,
    n_shop_slots:   int,
    buy_threshold:  float,
    sell_margin:    float,
    reroll_cost:    int = 1,
) -> tuple[List[dict], int, List[dict]]:
    """
    Run the greedy shop heuristic for one visit, with optional reroll.

    Strategy:
    1. Sample rarity-weighted shop offerings (excluding owned jokers).
    2. Score every offered joker.
    3. If best offer is below buy_threshold AND a reroll is affordable
       AND we have enough money left over after the reroll to still buy:
       reroll once and re-evaluate.
    4. For the best offered joker:
       a. If affordable and slot available: buy.
       b. If selling worst owned joker would fund it and upgrade is worth it:
          sell then buy.
       c. Otherwise: skip.

    Reroll is capped at one per visit. The reroll cost in real Balatro starts
    at $1 and increases by $1 each reroll within the same shop visit. Since
    we only do one reroll the cost is always reroll_cost (default $1).

    Args:
        all_jokers    : full joker pool
        active_jokers : currently owned jokers
        money         : current gold
        max_tier      : highest tier in this shop
        max_slots     : maximum joker slots
        n_shop_slots  : jokers offered per draw
        buy_threshold : min value score to consider buying
        sell_margin   : sell upgrade threshold (new_score >= old_score * margin)
        reroll_cost   : gold cost for one reroll (default 1)

    Returns:
        (updated_jokers, updated_money, shop_log)
    """
    from run_structure import JOKER_BUY_PRICE, joker_sell_price

    shop_log = []
    offered  = sample_shop(all_jokers, active_jokers, max_tier, n_shop_slots)

    if not offered:
        return active_jokers, money, shop_log

    offered_scores = [(j, score_joker_value(j, active_jokers)) for j in offered]
    best_joker, best_score = max(offered_scores, key=lambda x: x[1])

    # --- Reroll decision ---
    # Reroll if: best offer is weak, we can afford the reroll, and we'll
    # still have enough left to potentially buy what comes next.
    cheapest_buy = min(JOKER_BUY_PRICE.values())
    should_reroll = (
        best_score < buy_threshold
        and money >= reroll_cost + cheapest_buy   # can still afford a buy after
        and len(all_jokers) - len(active_jokers) > n_shop_slots  # more jokers to see
    )

    if should_reroll:
        money   -= reroll_cost
        shop_log.append({
            "action": "reroll",
            "reason": "below_threshold",
            "discarded": [j["name"] for j in offered],
            "cost": reroll_cost,
            "money_left": money,
        })
        offered  = sample_shop(all_jokers, active_jokers, max_tier, n_shop_slots)
        if not offered:
            return active_jokers, money, shop_log
        offered_scores = [(j, score_joker_value(j, active_jokers)) for j in offered]
        best_joker, best_score = max(offered_scores, key=lambda x: x[1])

    # --- Buy decision ---
    if best_score < buy_threshold:
        shop_log.append({"action": "skip", "reason": "below_threshold_after_reroll",
                         "best_offer": best_joker["name"], "score": best_score})
        return active_jokers, money, shop_log

    price    = JOKER_BUY_PRICE.get(best_joker["tier"], 5)
    has_slot = len(active_jokers) < max_slots

    # Case A: can afford and have a slot
    if money >= price and has_slot:
        active_jokers = active_jokers + [best_joker]
        money        -= price
        shop_log.append({"action": "buy", "joker": best_joker["name"],
                         "price": price, "money_left": money})
        return active_jokers, money, shop_log

    # Case B: sell worst owned joker to fund upgrade
    if active_jokers:
        owned_scores             = [(j, score_joker_value(j, active_jokers))
                                    for j in active_jokers]
        worst_joker, worst_score = min(owned_scores, key=lambda x: x[1])
        sell_val                 = joker_sell_price(worst_joker["tier"])
        money_after              = money + sell_val

        worth_selling = (
            best_score >= worst_score * sell_margin
            and money_after >= price
        )

        if worth_selling:
            active_jokers = [j for j in active_jokers if j["id"] != worst_joker["id"]]
            money         = money_after - price
            active_jokers = active_jokers + [best_joker]
            shop_log.append({
                "action": "sell_and_buy",
                "sold": worst_joker["name"], "sold_for": sell_val,
                "bought": best_joker["name"], "price": price,
                "money_left": money,
            })
            return active_jokers, money, shop_log

    shop_log.append({"action": "skip", "reason": "cannot_afford_or_no_slot",
                     "best_offer": best_joker["name"]})
    return active_jokers, money, shop_log
