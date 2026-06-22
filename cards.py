from config import DECK_SIZE


# Rank index: unique integer per rank, used for hand detection (straights, pairs etc.)
RANK_INDEX = {
    '2': 2,  '3': 3,  '4': 4,  '5': 5,  '6': 6,
    '7': 7,  '8': 8,  '9': 9,  '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

# Chip value: Balatro scoring value per rank (face cards = 10, Ace = 11)
RANK_CHIP_VALUE = {
    '2': 2,  '3': 3,  '4': 4,  '5': 5,  '6': 6,
    '7': 7,  '8': 8,  '9': 9,  '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11
}

RANKS = list(RANK_INDEX.keys())
SUITS = ['♠', '♥', '♦', '♣']

# Stable index for every unique card (rank_index * 4 + suit_index)
CARD_INDEX = {
    (rank, suit): r * len(SUITS) + s
    for r, rank in enumerate(RANKS)
    for s, suit in enumerate(SUITS)
}


def make_deck():
    """
    Returns a standard 52-card deck.
    Each card is a tuple: (rank_index, suit, chip_value, extra_chips, extra_mult)

    rank_index  : unique integer 2–14, used for hand detection
    chip_value  : Balatro chip value (J/Q/K=10, A=11), used for scoring
    extra_chips : bonus chips from jokers/enhancements (starts at 0)
    extra_mult  : bonus multiplier from jokers/enhancements (starts at 0)
    """
    deck = []
    for rank in RANKS:
        for suit in SUITS:
            deck.append((RANK_INDEX[rank], suit, RANK_CHIP_VALUE[rank], 0, 0))
    return deck


def deck_count_vector(remaining_cards: list) -> list:
    """
    Encodes which cards are still in the deck as a binary vector of length 52.

    Args:
        remaining_cards : list of card tuples still in the deck

    Returns:
        list of 52 floats — 1.0 if the card is still in the deck, 0.0 if not
    """
    vector = [0.0] * DECK_SIZE
    for card in remaining_cards:
        rank_index, suit, chip_value, extra_chips, extra_mult = card
        rank_str = next(r for r, v in RANK_INDEX.items() if v == rank_index)
        idx = CARD_INDEX.get((rank_str, suit))
        if idx is not None:
            vector[idx] = 1.0
    return vector
