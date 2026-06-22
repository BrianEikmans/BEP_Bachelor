from collections import Counter


HAND_SCORING = {
    "High Card":       (5,   1),
    "One Pair":        (10,  2),
    "Two Pair":        (20,  2),
    "Three of a Kind": (30,  3),
    "Straight":        (30,  4),
    "Flush":           (35,  4),
    "Full House":      (40,  4),
    "Four of a Kind":  (60,  7),
    "Straight Flush":  (100, 8),
    "Five of a Kind":  (120, 12),
    "Flush House":     (140, 14),
    "Flush Five":      (160, 16),
}


def get_hand_rank(hand):
    """
    Returns the Balatro hand name for the given set of cards.
    Each card is a tuple: (rank_index, suit, chip_value, extra_chips, extra_mult)
    Uses rank_index (2-14, unique per rank) for hand detection.
    """
    ranks = [c[0] for c in hand]  # rank_index, unique per rank
    suits = [c[1] for c in hand]
    n     = len(hand)

    rank_counts = Counter(ranks)
    counts      = list(rank_counts.values())
    max_count   = max(counts)
    pair_count  = counts.count(2)

    is_flush = len(set(suits)) == 1 and n == 5

    unique_ranks = sorted(set(ranks))
    is_straight  = (
        n == 5 and
        len(unique_ranks) == n and
        (
            max(unique_ranks) - min(unique_ranks) == n - 1   # normal straight
            or unique_ranks == [2, 3, 4, 5, 14]              # ace-low: A-2-3-4-5
        )
    )

    # Balatro priority order (highest first)
    if n == 5 and max_count == 5 and is_flush:
        return "Flush Five"
    if n == 5 and max_count == 3 and pair_count == 1 and is_flush:
        return "Flush House"
    if max_count == 5:
        return "Five of a Kind"
    if is_straight and is_flush:
        return "Straight Flush"
    if max_count == 4:
        return "Four of a Kind"
    if max_count == 3 and pair_count == 1:
        return "Full House"
    if is_flush:
        return "Flush"
    if is_straight:
        return "Straight"
    if max_count == 3:
        return "Three of a Kind"
    if pair_count == 2:
        return "Two Pair"
    if pair_count == 1:
        return "One Pair"

    return "High Card"


def get_scoring_cards(hand: list, hand_rank: str) -> list:
    """
    Returns only the cards that contribute chips to the given hand rank.
    Kickers are excluded, matching real Balatro scoring rules.
    Each card is a tuple: (rank_index, suit, chip_value, extra_chips, extra_mult)
    """
    ranks = [c[0] for c in hand]  # rank_index
    rank_counts = Counter(ranks)

    if hand_rank in ("Straight", "Flush", "Straight Flush",
                     "Full House", "Flush House", "Flush Five"):
        return hand

    if hand_rank == "Four of a Kind":
        target = [r for r, cnt in rank_counts.items() if cnt == 4]
    elif hand_rank == "Three of a Kind":
        target = [r for r, cnt in rank_counts.items() if cnt == 3]
    elif hand_rank == "Five of a Kind":
        target = [r for r, cnt in rank_counts.items() if cnt == 5]
    elif hand_rank in ("Two Pair", "One Pair"):
        target = [r for r, cnt in rank_counts.items() if cnt == 2]
    else:
        # High Card — only the single highest card scores
        return [max(hand, key=lambda c: c[0])]

    return [c for c in hand if c[0] in target]


def calculate_score(hand: list) -> int:
    """
    Calculates the score for a played hand.
    Each card is a tuple: (rank_index, suit, chip_value, extra_chips, extra_mult)

    Only scoring cards contribute chips:
        chip_total = sum(chip_value + extra_chips)
        mult_total = sum(extra_mult)
    Score = (base_chips + chip_total) * (base_mult + mult_total)
    """
    hand_rank = get_hand_rank(hand)
    if hand_rank not in HAND_SCORING:
        return 0

    base_chips, base_mult = HAND_SCORING[hand_rank]
    scoring_cards         = get_scoring_cards(hand, hand_rank)
    chip_total            = sum(c[2] + c[3] for c in scoring_cards)  # chip_value + extra_chips
    mult_total            = sum(c[4] for c in scoring_cards)          # extra_mult
    return (base_chips + chip_total) * (base_mult + mult_total)