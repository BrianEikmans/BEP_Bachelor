from collections import Counter
from itertools import combinations
import random


# Card rank values in Balatro
RANK_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11
}
RANKS = list(RANK_VALUES.keys())
SUITS = ['♠', '♥', '♦', '♣']
HAND_SIZE = 8
MAX_HAND_PLAYS = 4
MAX_DISCARDS = 3


def make_deck():
    """Standard 52-card deck. Each card: (rank, suit, extra_chips, extra_mult)"""
    deck = []
    for rank in RANKS:
        for suit in SUITS:
            deck.append((RANK_VALUES[rank], suit, 0, 0))
    return deck


def calculate_score(hand):
    """
    Calculates the score of the given hand.
    Each card is a tuple: (rank, suit, extra_chips, extra_mult)

    Score = (base_score + sum(rank + extra_chips)) * (multiplier + sum(extra_mult))
    """
    hand_scoring = {
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

    hand_rank = _get_hand_rank(hand)
    if hand_rank in hand_scoring:
        base_score, multiplier = hand_scoring[hand_rank]
        chip_value_total = sum(card[0] + card[2] for card in hand)  # rank + extra chips
        card_multiplier  = sum(card[3] for card in hand)            # extra mult
        return (base_score + chip_value_total) * (multiplier + card_multiplier)
    return 0


def _get_hand_rank(hand):
    """
    Returns the Balatro hand name for the given set of cards.
    Cards are tuples: (rank, suit, extra_chips, extra_mult)
    """
    ranks = [c[0] for c in hand]
    suits = [c[1] for c in hand]
    n = len(hand)

    rank_counts = Counter(ranks)
    counts      = list(rank_counts.values())
    max_count   = max(counts)
    pair_count  = counts.count(2)

    is_flush = len(set(suits)) == 1

    unique_ranks = sorted(set(ranks))
    is_straight  = (
        len(unique_ranks) == n and
        max(unique_ranks) - min(unique_ranks) == n - 1
    )

    # Balatro priority order
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


class BalatroEnv:
    """
    Balatro-like environment for DQN training.

    Each card in hand is a tuple: (rank, suit, extra_chips, extra_mult)

    State (36 floats):
        8 cards × 4 features: [rank_norm, suit_norm, extra_chips_norm, extra_mult_norm]
        4 scalars:            [hands_remaining, discards_remaining, score_progress, blind_norm]

    Actions:
        card_indices : list of 1–5 indices into self.hand
        play=True    : score the selected cards, remove them, draw replacements
        play=False   : discard the selected cards, draw replacements
    """

    def __init__(self, blind_target=300):
        self.blind_target = blind_target
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Shuffle a fresh deck, deal opening hand, reset all counters."""
        self.deck               = make_deck()
        random.shuffle(self.deck)
        self.hand               = []
        self.hands_remaining    = MAX_HAND_PLAYS
        self.discards_remaining = MAX_DISCARDS
        self.score              = 0
        self.done               = False

        self._deal_to_full()
        return self._get_state()

    def step(self, card_indices, play=True):
        """
        Take one action.

        Args:
            card_indices : list of indices (0–7) into self.hand to act on.
            play         : True  → play & score selected cards.
                           False → discard selected cards, draw replacements.

        Returns:
            state  : list[float]  — new observation vector (length 36)
            reward : float
            done   : bool
            info   : dict
        """
        assert not self.done,                    "Episode finished — call reset()."
        assert 1 <= len(card_indices) <= 5,      "Must select 1–5 cards."

        selected = [self.hand[i] for i in sorted(card_indices)]
        info     = {}

        if play:
            if self.hands_remaining <= 0:
                return self._get_state(), -10.0, True, {"error": "No hands left"}

            round_score           = calculate_score(selected)
            self.score           += round_score
            self.hands_remaining -= 1

            # Remove played cards and draw replacements
            for i in sorted(card_indices, reverse=True):
                self.hand.pop(i)
            self._deal_to_full()

            reward           = round_score / self.blind_target
            info["score"]    = round_score
            info["hand_type"] = _get_hand_rank(selected)

            if self.score >= self.blind_target:
                reward      += 5.0
                self.done    = True
                info["result"] = "win"
            elif self.hands_remaining <= 0:
                reward      -= 2.0
                self.done    = True
                info["result"] = "loss"

        else:  # discard
            if self.discards_remaining <= 0:
                return self._get_state(), -5.0, False, {"error": "No discards left"}

            self.discards_remaining -= 1
            for i in sorted(card_indices, reverse=True):
                self.hand.pop(i)
            self._deal_to_full()

            reward              = 0.0
            info["discarded"]   = len(card_indices)

        return self._get_state(), reward, self.done, info

    def valid_play_actions(self):
        """Return every valid 1–5 card subset as a list of index lists."""
        actions = []
        for r in range(1, 6):
            for combo in combinations(range(len(self.hand)), r):
                actions.append(list(combo))
        return actions

    def render(self):
        """Print a simple text representation of the current game state."""
        rank_lookup = {v: k for k, v in RANK_VALUES.items()}
        hand_str = "  ".join(
            f"{rank_lookup.get(c[0], '?')}{c[1]}"
            + (f"(+{c[2]}c)" if c[2] else "")
            + (f"(+{c[3]}m)" if c[3] else "")
            for c in self.hand
        )
        print(f"Hand  : {hand_str}")
        print(f"Score : {self.score}/{self.blind_target}  "
              f"Plays: {self.hands_remaining}  "
              f"Discards: {self.discards_remaining}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_size(self):
        return HAND_SIZE * 4 + 4  # 36

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deal_to_full(self):
        """Draw cards from the deck until the hand reaches HAND_SIZE."""
        while len(self.hand) < HAND_SIZE and self.deck:
            self.hand.append(self.deck.pop())

    def _get_state(self):
        """
        Encode the current game state as a normalised float vector.

        Per-card features (× 8 cards):
            rank_norm        : rank value / 14
            suit_norm        : suit index / 3
            extra_chips_norm : extra_chips / 50
            extra_mult_norm  : extra_mult  / 10

        Global scalars:
            hands_remaining  / MAX_HAND_PLAYS
            discards_remaining / MAX_DISCARDS
            score_progress   : min(score / blind_target, 1.0)
            blind_norm       : blind_target / 1000
        """
        state = []
        for card in self.hand:
            rank, suit, extra_chips, extra_mult = card
            state.append(rank / 14.0)
            state.append(SUITS.index(suit) / 3.0)
            state.append(extra_chips / 50.0)
            state.append(extra_mult  / 10.0)

        # Pad if hand is smaller than HAND_SIZE (near end of deck)
        while len(state) < HAND_SIZE * 4:
            state.extend([0.0, 0.0, 0.0, 0.0])

        state.append(self.hands_remaining    / MAX_HAND_PLAYS)
        state.append(self.discards_remaining / MAX_DISCARDS)
        state.append(min(self.score / self.blind_target, 1.0))
        state.append(self.blind_target       / 1000.0)

        return state  # length == 36
