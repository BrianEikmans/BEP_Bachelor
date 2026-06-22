import random
from collections import Counter
from itertools import combinations

from cards import SUITS, RANK_INDEX, make_deck, deck_count_vector
from scoring import get_hand_rank, calculate_score
from jokers import (
    load_jokers, sample_episode_jokers,
    apply_jokers_on_hand, get_joker_state_features,
    get_score_components, run_shop,
)
from run_structure import (
    BLIND_TARGETS, MAX_ANTES, BLINDS_PER_ANTE, BLIND_NAMES,
    STARTING_MONEY, money_for_win, calculate_interest,
    blind_target, blind_name, run_progress,
    MAX_JOKER_SLOTS, SHOP_JOKER_SLOTS, JOKER_BUY_PRICE,
    SHOP_BUY_THRESHOLD, SHOP_SELL_UPGRADE_MARGIN, SHOP_REROLL_COST,
)
import config

# Ordinal rank for each hand type — used to encode best achievable hand in state
HAND_RANK_ORDER = {
    "High Card": 0, "One Pair": 1, "Two Pair": 2, "Three of a Kind": 3,
    "Straight": 4, "Flush": 5, "Full House": 6, "Four of a Kind": 7,
    "Straight Flush": 8, "Five of a Kind": 9, "Flush House": 10, "Flush Five": 11,
}
_MAX_HAND_RANK = len(HAND_RANK_ORDER) - 1   # 11


class BalatroEnv:
    """
    Balatro-like environment for DQN training.

    Each card in hand is a tuple: (rank_index, suit, chip_value, extra_chips, extra_mult)

    State vector layout (length = config.STATE_SIZE = 134):
        8 cards × 5 features : [rank_norm, suit_norm, chip_norm, extra_chips_norm, extra_mult_norm]
        52 deck counts        : 1.0 if card still in deck, 0.0 if not
        Scalars (3)           : hands_remaining_norm, discards_remaining_norm, score_progress
        Draw features (9)     : suit_counts(4), longest_straight(1),
                                best_hand_rank(1), pair_count(1), trip_count(1), quad_count(1)
        Joker features (30)   : mult_per_hand_type(12), chips_per_hand_type(12),
                                passive_mult(1), passive_chips(1), per_suit_mult(4)
                                All zeros for phases 1-5 (no active jokers).

    Actions:
        card_indices : list of 1-5 indices into self.hand
        play=True    : score the selected cards, remove them, draw replacements
        play=False   : discard the selected cards, draw replacements (phase 3+)
    """

    # Loaded once at class level — shared across all env instances
    _all_jokers = None

    def __init__(self):
        settings                = config.get_phase_settings()
        self.max_hand_plays     = settings["max_hand_plays"]
        self.max_discards       = settings["max_discards"]
        self.blind_target       = settings["blind_target"]
        self.num_jokers         = settings["num_jokers"]
        self.max_joker_tier     = settings["max_joker_tier"]

        # Run context — set by RunEnv before each blind, zeroed in single-blind mode
        self._run_ante          = 0
        self._run_blind_idx     = 0
        self._run_money         = 0.0
        self._run_max_antes     = 0

        # Load joker definitions once
        if BalatroEnv._all_jokers is None and self.num_jokers > 0:
            BalatroEnv._all_jokers = load_jokers("jokers.json")

        self.active_jokers: list = []
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self,
              blind_target_override=None,
              jokers_override=None,
              run_ante: int = 0,
              run_blind_idx: int = 0,
              run_money: float = 0.0,
              run_max_antes: int = 0):
        """
        Shuffle a fresh deck, deal opening hand, reset all counters.

        Override parameters are used by RunEnv to inject per-blind settings
        without touching config. When called without arguments (single-blind
        mode, phases 1-9) the env reads settings from config as before.

        Args:
            blind_target_override : override blind target for this blind
            jokers_override       : override active jokers (run-level jokers)
            run_ante              : current ante (1-indexed), 0 = single-blind mode
            run_blind_idx         : 0=Small, 1=Big, 2=Boss
            run_money             : current gold in the run
            run_max_antes         : max antes in this run (for normalisation)
        """
        self.deck               = make_deck()
        random.shuffle(self.deck)
        self.hand               = []
        self.hands_remaining    = self.max_hand_plays
        self.discards_remaining = self.max_discards
        self.score              = 0
        self.done               = False

        # Apply overrides
        if blind_target_override is not None:
            self.blind_target = blind_target_override
        if run_ante:
            self._run_ante      = run_ante
            self._run_blind_idx = run_blind_idx
            self._run_money     = run_money
            self._run_max_antes = run_max_antes

        # Joker assignment
        if jokers_override is not None:
            self.active_jokers = jokers_override
        elif self.num_jokers > 0 and BalatroEnv._all_jokers is not None:
            self.active_jokers = sample_episode_jokers(
                BalatroEnv._all_jokers, self.num_jokers, self.max_joker_tier
            )
        else:
            self.active_jokers = []

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
            state  : list[float]  — new observation vector
            reward : float
            done   : bool
            info   : dict
        """
        assert not self.done,               "Episode finished — call reset()."
        assert 1 <= len(card_indices) <= 5, "Must select 1–5 cards."

        selected = [self.hand[i] for i in sorted(card_indices)]
        info     = {}

        if play:
            if self.hands_remaining <= 0:
                return self._get_state(), config.REWARD_ILLEGAL_ACTION, True, {"error": "No hands left"}

            # Determine hand type first (needed for joker condition checks)
            hand_type = get_hand_rank(selected)

            # Score: apply joker bonuses inside the chip*mult formula
            if self.active_jokers:
                chips, mult = get_score_components(selected, hand_type)
                j_chips, j_mult = apply_jokers_on_hand(
                    self.active_jokers, selected, hand_type
                )
                round_score = (chips + j_chips) * (mult + j_mult)
            else:
                round_score = calculate_score(selected)

            self.score           += round_score
            self.hands_remaining -= 1

            for i in sorted(card_indices, reverse=True):
                self.hand.pop(i)
            self._deal_to_full()

            info["score"]     = round_score
            info["hand_type"] = hand_type

            # --- Reward ---
            if self.blind_target is None:
                # Phase 1: pure score reward
                reward = round_score * config.REWARD_RAW_SCORE_SCALE
                if self.hands_remaining <= 0:
                    self.done = True
                    info["result"] = "done"
            else:
                # Phase 2+: reward is score fraction, with win/loss bonuses
                reward = round_score / self.blind_target

                # Made-hand floor bonus (joker phases 6+ only).
                # Adds a small fixed bonus when the agent plays One Pair or better.
                # Counters the High Card regression: agents with jokers can discard
                # aggressively, exhaust discards, and be stuck playing High Card.
                # The bonus makes any made hand more attractive without distorting
                # the no-joker curriculum or overriding the score/win signals.
                if self.active_jokers:
                    hand_ord = HAND_RANK_ORDER.get(hand_type, 0)
                    if hand_ord >= HAND_RANK_ORDER["One Pair"]:
                        reward += config.REWARD_MADE_HAND_BONUS

                if self.score >= self.blind_target:
                    # Bonus scales with hands remaining — finishing faster = more reward
                    efficiency     = self.hands_remaining / self.max_hand_plays
                    reward        += config.REWARD_WIN_BONUS * (1.0 + efficiency)
                    self.done      = True
                    info["result"] = "win"
                elif self.hands_remaining <= 0:
                    reward        -= config.REWARD_LOSS_PENALTY
                    self.done      = True
                    info["result"] = "loss"

        else:  # discard
            if self.max_discards == 0:
                return self._get_state(), config.REWARD_ILLEGAL_DISCARD, False, {"error": "Discards disabled"}
            if self.discards_remaining <= 0:
                return self._get_state(), config.REWARD_ILLEGAL_DISCARD, False, {"error": "No discards left"}

            # Record best hand rank BEFORE discard for reward shaping
            before_rank = HAND_RANK_ORDER.get(get_hand_rank(self.hand), 0)

            self.discards_remaining -= 1
            for i in sorted(card_indices, reverse=True):
                self.hand.pop(i)
            self._deal_to_full()

            # Shaped reward: small bonus for improving the best achievable hand rank.
            # Gives immediate feedback on discard quality instead of waiting for next play.
            after_rank  = HAND_RANK_ORDER.get(get_hand_rank(self.hand), 0)
            rank_improvement = (after_rank - before_rank) / _MAX_HAND_RANK
            reward = rank_improvement * config.DISCARD_REWARD_SCALE

            info["discarded"] = len(card_indices)

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
        rank_lookup = {v: k for k, v in RANK_INDEX.items()}
        hand_str = "  ".join(
            f"{rank_lookup.get(c[0], '?')}{c[1]}"
            + (f"(+{c[3]}c)" if c[3] else "")
            + (f"(+{c[4]}m)" if c[4] else "")
            for c in self.hand
        )
        blind_str = f"/{self.blind_target}" if self.blind_target else ""
        joker_str = f"  Jokers: {', '.join(self.active_joker_names)}" if self.active_jokers else ""
        print(f"Hand  : {hand_str}")
        print(f"Score : {self.score}{blind_str}  "
              f"Plays: {self.hands_remaining}  "
              f"Discards: {self.discards_remaining}"
              f"{joker_str}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_size(self):
        return config.STATE_SIZE

    @property
    def active_joker_names(self) -> list:
        """Names of the jokers active in the current episode."""
        return [j["name"] for j in self.active_jokers]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deal_to_full(self):
        """Draw cards from the deck until the hand reaches HAND_SIZE."""
        while len(self.hand) < config.HAND_SIZE and self.deck:
            self.hand.append(self.deck.pop())

    def _get_draw_features(self) -> list:
        """
        Compute 15 deck-aware draw-potential features.

        Discard-gated features (zeroed when discards_remaining == 0):
            [0:4]  flush_outs_per_suit      per suit: min(needed, in_deck) / 13
            [4]    best_window_fill          fraction of best 5-rank window covered
            [5]    straight_outs             live completing ranks / 2
                                              1.0=open-ended, 0.5=gutshot/one-ended, 0.0=dead
            [6]    straight_flush_proximity  best_window_fill * suit_agreement
                                              how close we are to a straight flush
            [7]    pair_to_trip_outs         copies of best pair's rank in deck / 3
            [8]    pair_to_twopair_outs      singleton ranks in hand with >=1 copy in deck / 8
            [9]    twopair_to_fh_outs        if two pairs exist: max copies of either pair / 2

        Always-included features (relevant for play decisions too):
            [10]   deck_depth               len(deck) / DECK_SIZE
            [11]   best_hand_rank           ordinal of best hand in 8-card hand / 11
            [12]   pair_count               distinct pairs / 4
            [13]   trip_count               distinct trips / 2
            [14]   quad_count               distinct quads / 2
        """
        if not self.hand:
            return [0.0] * 15

        can_discard  = self.discards_remaining > 0
        deck_ranks   = Counter(card[0] for card in self.deck)   # rank → copies remaining
        deck_suits   = Counter(card[1] for card in self.deck)   # suit → copies remaining
        deck_rank_set = set(deck_ranks.keys())
        rank_counts  = Counter(card[0] for card in self.hand)

        # -------------------------------------------------------------------
        # [0:4] Flush outs — deck-aware, discard-gated
        # -------------------------------------------------------------------
        if can_discard:
            hand_suit_counts = Counter(card[1] for card in self.hand)
            flush_features = []
            for suit in SUITS:
                in_hand = hand_suit_counts.get(suit, 0)
                in_deck = deck_suits.get(suit, 0)
                needed  = max(0, 5 - in_hand)
                flush_features.append(0.0 if needed == 0 else min(needed, in_deck) / 13.0)
        else:
            flush_features = [0.0] * 4

        # -------------------------------------------------------------------
        # [4:7] Straight and straight-flush features — deck-aware, discard-gated
        # -------------------------------------------------------------------
        hand_ranks = set(card[0] for card in self.hand)
        windows    = [frozenset({1, 2, 3, 4, 5})] + [frozenset(range(r, r + 5)) for r in range(2, 11)]

        def cards_in_window(window):
            if 1 in window:
                adjusted = frozenset(14 if r == 1 else r for r in window)
                return len(hand_ranks & adjusted), adjusted
            return len(hand_ranks & window), frozenset(window)

        if can_discard:
            # Find best fill and collect completing ranks
            best_fill  = max(cards_in_window(w)[0] for w in windows)
            completing_ranks = set()
            best_windows = []
            for window in windows:
                fill, actual_window = cards_in_window(window)
                if fill == best_fill:
                    best_windows.append((window, actual_window))
                    if (5 - fill) == 1:
                        completing_ranks |= actual_window - hand_ranks

            live_outs = sum(1 for r in completing_ranks if r in deck_rank_set)
            straight_features = [best_fill / 5.0, live_outs / 2.0]

            # Straight flush proximity: among the best-fill windows, how well
            # do the cards in that window agree on a single suit?
            # Value = best_fill / 5.0 * (max_suit_count_in_window / window_size)
            # e.g. 3♠4♠5♠6♠ → fill=0.8, all 4 cards same suit → proximity=0.8
            #      3♠4♠5♥6♦ → fill=0.8, max suit count=2/4    → proximity=0.4
            best_sf_prox = 0.0
            for orig_window, actual_window in best_windows:
                window_cards = [c for c in self.hand if c[0] in actual_window]
                if not window_cards:
                    continue
                suit_counts_in_window = Counter(c[1] for c in window_cards)
                max_suit = max(suit_counts_in_window.values())
                sf_prox  = (best_fill / 5.0) * (max_suit / len(window_cards))
                if sf_prox > best_sf_prox:
                    best_sf_prox = sf_prox
            sf_features = [best_sf_prox]
        else:
            straight_features = [0.0, 0.0]
            sf_features       = [0.0]

        # -------------------------------------------------------------------
        # [7:10] Split upgrade outs — deck-aware, discard-gated
        #
        # pair_to_trip_outs:
        #   How many copies of the best pair's rank remain in deck?
        #   Captures "I have a pair of 7s, how likely can I hit trips?"
        #   Normalised /3 (max 3 remaining copies if holding a pair).
        #
        # pair_to_twopair_outs:
        #   How many singleton ranks in hand have at least one copy in deck?
        #   Capturing "how many new pairs can I draw into?"
        #   Normalised /8 (max singletons in an 8-card hand).
        #
        # twopair_to_fh_outs:
        #   If two pairs exist, max copies of either pair rank in deck.
        #   Captures "can I improve my two pair to a full house?"
        #   Normalised /2 (holding two of a rank means max 2 remain in deck).
        # -------------------------------------------------------------------
        if can_discard:
            # pair_to_trip_outs
            pair_ranks = [r for r, cnt in rank_counts.items() if cnt == 2]
            if pair_ranks:
                best_pair_outs = max(deck_ranks.get(r, 0) for r in pair_ranks)
            else:
                best_pair_outs = 0
            pair_to_trip = best_pair_outs / 3.0

            # pair_to_twopair_outs
            singleton_ranks = [r for r, cnt in rank_counts.items() if cnt == 1]
            live_singletons = sum(1 for r in singleton_ranks if deck_ranks.get(r, 0) > 0)
            pair_to_twopair = live_singletons / 8.0

            # twopair_to_fh_outs
            if len(pair_ranks) >= 2:
                twopair_outs = max(deck_ranks.get(r, 0) for r in pair_ranks)
                twopair_to_fh = min(twopair_outs, 2) / 2.0
            else:
                twopair_to_fh = 0.0

            upgrade_features = [pair_to_trip, pair_to_twopair, twopair_to_fh]
        else:
            upgrade_features = [0.0, 0.0, 0.0]

        # -------------------------------------------------------------------
        # [10] Deck depth — always included
        # -------------------------------------------------------------------
        deck_depth = [len(self.deck) / config.DECK_SIZE]

        # -------------------------------------------------------------------
        # [11] Best hand rank achievable from current 8-card hand
        # -------------------------------------------------------------------
        best_rank_idx = HAND_RANK_ORDER.get(get_hand_rank(self.hand), 0)
        hand_rank_feature = [best_rank_idx / _MAX_HAND_RANK]

        # -------------------------------------------------------------------
        # [12:15] Pair / trip / quad counts — always included
        # -------------------------------------------------------------------
        pairs = sum(1 for c in rank_counts.values() if c >= 2)
        trips = sum(1 for c in rank_counts.values() if c >= 3)
        quads = sum(1 for c in rank_counts.values() if c >= 4)

        return (flush_features + straight_features + sf_features +
                upgrade_features + deck_depth + hand_rank_feature +
                [pairs / 4.0, trips / 2.0, quads / 2.0])

    def _get_score_context(self) -> list:
        """
        Compute 19 joker-aware score context features.

        Returns (19 floats):
            [0:12]  hand_type_score_estimates
                    Best joker-adjusted score from current hand per hand type / 1000.

            [12]    score_needed_per_remaining_hand
                    max(0, blind_target - score) / hands_remaining / 1000.

            [13]    best_immediate_play_score
                    Best joker-adjusted score playable right now / 1000.

            [14]    straight_draw_value
            [15]    flush_draw_value
            [16]    upgrade_draw_value
                    Joker-weighted draw quality (outs x potential score).
                    Zeroed when no discards remain.

            [17]    expected_score_after_discard
                    Best joker-adjusted score achievable from the 5 kept cards
                    after the optimal discard (1-3 cards discarded from the 8).
                    This is a guaranteed lower bound on what we can score after
                    discarding -- the actual draw may improve it further.
                    Zeroed when no discards remain. Normalised / 1000.

            [18]    discard_worth_it
                    1.0 if expected_score_after_discard > best_immediate_play_score,
                    0.0 otherwise. Pre-computes the "should I discard or play?"
                    comparison so the agent doesn't need to learn it implicitly.
                    Zeroed when no discards remain.
        """
        _SCORE_NORM = 1000.0

        hand_type_order = [
            "High Card", "One Pair", "Two Pair", "Three of a Kind",
            "Straight", "Flush", "Full House", "Four of a Kind",
            "Straight Flush", "Five of a Kind", "Flush House", "Flush Five",
        ]

        # --- Iterate all valid play subsets once ---
        best_score_per_type = {ht: 0.0 for ht in hand_type_order}
        best_immediate      = 0.0

        for size in range(1, 6):
            for subset in combinations(self.hand, size):
                ht = get_hand_rank(list(subset))
                if self.active_jokers:
                    chips, mult = get_score_components(list(subset), ht)
                    jc, jm      = apply_jokers_on_hand(self.active_jokers, list(subset), ht)
                    score       = (chips + jc) * (mult + jm)
                else:
                    score = calculate_score(list(subset))

                if score > best_score_per_type[ht]:
                    best_score_per_type[ht] = score
                if score > best_immediate:
                    best_immediate = score

        estimates = [min(best_score_per_type[ht] / _SCORE_NORM, 1.0) for ht in hand_type_order]

        # --- Score needed per remaining hand ---
        if self.blind_target is not None and self.hands_remaining > 0:
            needed       = max(0.0, self.blind_target - self.score)
            score_needed = min(needed / self.hands_remaining / _SCORE_NORM, 1.0)
        else:
            score_needed = 0.0

        # --- Joker-weighted draw values (discard-gated) ---
        if self.discards_remaining > 0 and self.hand:
            deck_rank_set    = set(c[0] for c in self.deck)
            deck_suits_count = Counter(c[1] for c in self.deck)
            hand_ranks       = set(c[0] for c in self.hand)

            # Straight outs
            windows = [frozenset({1,2,3,4,5})] + [frozenset(range(r,r+5)) for r in range(2,11)]
            def ciw(window):
                if 1 in window:
                    adj = frozenset(14 if r==1 else r for r in window)
                    return len(hand_ranks & adj), adj
                return len(hand_ranks & window), frozenset(window)
            best_fill = max(ciw(w)[0] for w in windows)
            completing = set(); best_wins = []
            for w in windows:
                fill, aw = ciw(w)
                if fill == best_fill:
                    best_wins.append((w, aw))
                    if (5-fill) == 1: completing |= aw - hand_ranks
            straight_outs = sum(1 for r in completing if r in deck_rank_set) / 2.0

            # Flush outs
            hand_suit_counts = Counter(c[1] for c in self.hand)
            flush_outs = 0.0; best_flush_cards = []
            for suit in SUITS:
                in_hand = hand_suit_counts.get(suit, 0)
                in_deck = deck_suits_count.get(suit, 0)
                needed_f = max(0, 5 - in_hand)
                if needed_f > 0:
                    so = min(needed_f, in_deck) / 13.0
                    if so > flush_outs:
                        flush_outs = so
                        best_flush_cards = [c for c in self.hand if c[1] == suit]

            # Upgrade outs
            deck_ranks_count = Counter(c[0] for c in self.deck)
            rank_counts = Counter(c[0] for c in self.hand)
            pair_ranks = [r for r,cnt in rank_counts.items() if cnt == 2]
            trip_ranks = [r for r,cnt in rank_counts.items() if cnt == 3]

            from scoring import HAND_SCORING
            # Straight potential
            s_base_chips, s_base_mult = HAND_SCORING["Straight"]
            bwc = []
            for _,aw in best_wins:
                wc = [c for c in self.hand if c[0] in aw]
                if wc and len(wc) > len(bwc): bwc = wc
            sc = sum(c[2]+c[3] for c in bwc)
            jm_s = apply_jokers_on_hand(self.active_jokers, bwc, "Straight")[1] if self.active_jokers else 0
            s_pot = (s_base_chips + sc) * (s_base_mult + jm_s)

            # Flush potential
            f_base_chips, f_base_mult = HAND_SCORING["Flush"]
            fc = sum(c[2]+c[3] for c in best_flush_cards)
            jm_f = apply_jokers_on_hand(self.active_jokers, best_flush_cards, "Flush")[1] if (self.active_jokers and best_flush_cards) else 0
            f_pot = (f_base_chips + fc) * (f_base_mult + jm_f)

            # Upgrade potential
            if trip_ranks:
                upgrade_outs = max(deck_ranks_count.get(r,0) for r in trip_ranks) / 3.0
                uc = [c for c in self.hand if c[0] == trip_ranks[0]]
                ut = "Four of a Kind"
            elif pair_ranks:
                upgrade_outs = max(deck_ranks_count.get(r,0) for r in pair_ranks) / 3.0
                uc = [c for c in self.hand if c[0] == pair_ranks[0]]
                ut = "Three of a Kind"
            else:
                upgrade_outs = 0.0; uc = []; ut = "One Pair"
            u_base_chips, u_base_mult = HAND_SCORING[ut]
            uc2 = sum(c[2]+c[3] for c in uc)
            jm_u = apply_jokers_on_hand(self.active_jokers, uc, ut)[1] if (self.active_jokers and uc) else 0
            u_pot = (u_base_chips + uc2) * (u_base_mult + jm_u)

            straight_draw_value = straight_outs * min(s_pot / _SCORE_NORM, 1.0)
            flush_draw_value    = flush_outs    * min(f_pot / _SCORE_NORM, 1.0)
            upgrade_draw_value  = upgrade_outs  * min(u_pot / _SCORE_NORM, 1.0)
        else:
            straight_draw_value = flush_draw_value = upgrade_draw_value = 0.0

        # --- Expected score after optimal discard (discard-gated) ---
        # Iterate all possible discards of 1-3 cards. For each, find the best
        # joker-adjusted score achievable from the 5 remaining kept cards.
        # This gives a guaranteed lower bound on post-discard score — the actual
        # draw from the deck may improve it, but we can already score at least this.
        # The agent uses this to decide: "even if I draw nothing useful, is
        # discarding still better than playing my best hand right now?"
        if self.discards_remaining > 0 and len(self.hand) >= 6:
            best_after_discard = 0.0
            hand_indices = list(range(len(self.hand)))

            for discard_size in range(1, min(4, len(self.hand) - 5 + 1)):
                for discard_idxs in combinations(hand_indices, discard_size):
                    kept = [self.hand[i] for i in hand_indices if i not in discard_idxs]
                    # Score the best 5-card hand from kept cards
                    for size in range(1, min(6, len(kept) + 1)):
                        for subset in combinations(kept, size):
                            ht = get_hand_rank(list(subset))
                            if self.active_jokers:
                                chips, mult = get_score_components(list(subset), ht)
                                jc, jm     = apply_jokers_on_hand(self.active_jokers, list(subset), ht)
                                s          = (chips + jc) * (mult + jm)
                            else:
                                s = calculate_score(list(subset))
                            if s > best_after_discard:
                                best_after_discard = s

            exp_after_discard = min(best_after_discard / _SCORE_NORM, 1.0)
            discard_worth_it  = 1.0 if best_after_discard > best_immediate else 0.0
        else:
            exp_after_discard = 0.0
            discard_worth_it  = 0.0

        # --- Optimal hand type target ---
        # argmax(hand_type_score_estimates) / 11
        # Direct "aim for this hand type" signal. With jokers, this can differ
        # from best_hand_rank -- e.g. Crazy Joker makes Straight the target
        # even when Trips is currently in hand.
        best_ht_idx    = max(range(len(hand_type_order)),
                             key=lambda i: best_score_per_type[hand_type_order[i]])
        optimal_target = best_ht_idx / (len(hand_type_order) - 1)

        # --- Held-card joker activation count ---
        # For on_card_scored jokers, how many cards in hand would fire the joker
        # if played, normalised by HAND_SIZE.
        # Tells the agent how well the current hand matches its joker set --
        # distinct from the abstract joker feature block which only encodes
        # what the joker does, not how well the hand aligns with it.
        if self.active_jokers:
            activating = 0
            for card in self.hand:
                for joker in self.active_jokers:
                    if joker["trigger"] != "on_card_scored":
                        continue
                    cond = joker["condition"]
                    if cond["type"] == "card_suit" and card[1] == cond["suit"]:
                        activating += 1
                        break
                    elif cond["type"] == "card_rank_in" and card[0] in set(cond["ranks"]):
                        activating += 1
                        break
            joker_activation = activating / config.HAND_SIZE
        else:
            joker_activation = 0.0

        return (estimates
                + [score_needed,
                   min(best_immediate / _SCORE_NORM, 1.0),
                   straight_draw_value,
                   flush_draw_value,
                   upgrade_draw_value,
                   exp_after_discard,
                   discard_worth_it,
                   optimal_target,
                   joker_activation])
        """
        Compute 17 joker-aware score context features.

        Returns (17 floats):
            [0:12]  hand_type_score_estimates
                    Best joker-adjusted score achievable from current hand
                    for each of the 12 hand types, normalised by _SCORE_NORM.
                    0.0 if no subset of the hand produces that type.

            [12]    score_needed_per_remaining_hand
                    max(0, blind_target - score) / hands_remaining / _SCORE_NORM
                    How much the agent needs to average per remaining hand.
                    0.0 when no blind or no hands left.

            [13]    best_immediate_play_score
                    Highest joker-adjusted score from any current subset.
                    The "play now" baseline to compare discards against.

            [14]    straight_draw_value
            [15]    flush_draw_value
            [16]    upgrade_draw_value
                    Joker-weighted draw quality — pre-multiplies draw outs
                    by the joker-adjusted score for the target hand type.
                    Connects "how close am I" with "how much is it worth
                    given my jokers" so the agent can directly compare
                    competing discard strategies without learning the
                    multiplication itself.

                    straight_draw_value = straight_outs * straight_score_est
                    flush_draw_value    = best_flush_outs * flush_score_est
                    upgrade_draw_value  = upgrade_outs * trips_score_est

                    All zeroed when no discards remain (not actionable).
        """
        _SCORE_NORM = 1000.0

        hand_type_order = [
            "High Card", "One Pair", "Two Pair", "Three of a Kind",
            "Straight", "Flush", "Full House", "Four of a Kind",
            "Straight Flush", "Five of a Kind", "Flush House", "Flush Five",
        ]

        # --- Iterate all valid play subsets once ---
        best_score_per_type = {ht: 0.0 for ht in hand_type_order}
        best_immediate      = 0.0

        for size in range(1, 6):
            for subset in combinations(self.hand, size):
                ht = get_hand_rank(list(subset))
                if self.active_jokers:
                    chips, mult = get_score_components(list(subset), ht)
                    jc, jm      = apply_jokers_on_hand(self.active_jokers, list(subset), ht)
                    score       = (chips + jc) * (mult + jm)
                else:
                    score = calculate_score(list(subset))

                if score > best_score_per_type[ht]:
                    best_score_per_type[ht] = score
                if score > best_immediate:
                    best_immediate = score

        estimates = [min(best_score_per_type[ht] / _SCORE_NORM, 1.0) for ht in hand_type_order]

        # --- Score needed per remaining hand ---
        if self.blind_target is not None and self.hands_remaining > 0:
            needed       = max(0.0, self.blind_target - self.score)
            score_needed = min(needed / self.hands_remaining / _SCORE_NORM, 1.0)
        else:
            score_needed = 0.0

        # --- Joker-weighted draw values (discard-gated) ---
        # These connect draw quality with joker-adjusted payoff in a single
        # number the network can directly compare across competing strategies.
        #
        # Key design: we use *potential* score for draw targets, not current
        # score. If the agent doesn't hold a complete straight yet,
        # best_score_per_type["Straight"] would be 0, making the feature
        # useless. Instead we estimate what the drawn hand would score using
        # base scoring + joker bonuses applied to the window cards in hand.
        if self.discards_remaining > 0 and self.hand:
            from jokers import HAND_CONTAINS
            deck_rank_set    = set(c[0] for c in self.deck)
            deck_suits_count = Counter(c[1] for c in self.deck)
            hand_ranks       = set(c[0] for c in self.hand)
            rank_counts      = Counter(c[0] for c in self.hand)

            # -- Straight draw --
            windows = ([frozenset({1,2,3,4,5})]
                       + [frozenset(range(r, r+5)) for r in range(2, 11)])
            def ciw(window):
                if 1 in window:
                    adj = frozenset(14 if r==1 else r for r in window)
                    return len(hand_ranks & adj), adj
                return len(hand_ranks & window), frozenset(window)

            best_fill = max(ciw(w)[0] for w in windows)
            completing = set()
            best_window_cards = []
            for w in windows:
                fill, aw = ciw(w)
                if fill == best_fill:
                    if (5 - fill) == 1:
                        completing |= aw - hand_ranks
                    # Cards in hand that are part of this window
                    if not best_window_cards:
                        best_window_cards = [c for c in self.hand if c[0] in aw]
            straight_outs = sum(1 for r in completing if r in deck_rank_set) / 2.0

            # Estimate straight score: base chips + window card chips + joker bonus
            from scoring import HAND_SCORING
            s_base_chips, s_base_mult = HAND_SCORING["Straight"]
            s_card_chips = sum(c[2] + c[3] for c in best_window_cards)
            if self.active_jokers:
                _, jm = apply_jokers_on_hand(
                    self.active_jokers, best_window_cards, "Straight"
                )
            else:
                jm = 0
            straight_potential = (s_base_chips + s_card_chips) * (s_base_mult + jm)

            # -- Flush draw --
            hand_suit_counts = Counter(c[1] for c in self.hand)
            flush_outs = 0.0
            best_flush_cards = []
            for suit in SUITS:
                in_hand  = hand_suit_counts.get(suit, 0)
                in_deck  = deck_suits_count.get(suit, 0)
                needed_f = max(0, 5 - in_hand)
                if needed_f > 0:
                    suit_outs = min(needed_f, in_deck) / 13.0
                    if suit_outs > flush_outs:
                        flush_outs       = suit_outs
                        best_flush_cards = [c for c in self.hand if c[1] == suit]

            f_base_chips, f_base_mult = HAND_SCORING["Flush"]
            f_card_chips = sum(c[2] + c[3] for c in best_flush_cards)
            if self.active_jokers and best_flush_cards:
                _, jm_f = apply_jokers_on_hand(
                    self.active_jokers, best_flush_cards, "Flush"
                )
            else:
                jm_f = 0
            flush_potential = (f_base_chips + f_card_chips) * (f_base_mult + jm_f)

            # -- Upgrade draw (pair -> trips) --
            deck_ranks_count = Counter(c[0] for c in self.deck)
            pair_ranks  = [r for r, cnt in rank_counts.items() if cnt == 2]
            trip_ranks  = [r for r, cnt in rank_counts.items() if cnt == 3]
            if trip_ranks:
                upgrade_outs   = max(deck_ranks_count.get(r, 0) for r in trip_ranks) / 3.0
                upgrade_cards  = [c for c in self.hand if c[0] == trip_ranks[0]]
                upgrade_target = "Four of a Kind"
            elif pair_ranks:
                upgrade_outs   = max(deck_ranks_count.get(r, 0) for r in pair_ranks) / 3.0
                upgrade_cards  = [c for c in self.hand if c[0] == pair_ranks[0]]
                upgrade_target = "Three of a Kind"
            else:
                upgrade_outs  = 0.0
                upgrade_cards = []
                upgrade_target = "One Pair"

            u_base_chips, u_base_mult = HAND_SCORING[upgrade_target]
            u_card_chips = sum(c[2] + c[3] for c in upgrade_cards)
            if self.active_jokers and upgrade_cards:
                _, jm_u = apply_jokers_on_hand(
                    self.active_jokers, upgrade_cards, upgrade_target
                )
            else:
                jm_u = 0
            upgrade_potential = (u_base_chips + u_card_chips) * (u_base_mult + jm_u)

            # Weighted values: outs × normalised potential score
            straight_draw_value = straight_outs  * min(straight_potential / _SCORE_NORM, 1.0)
            flush_draw_value    = flush_outs     * min(flush_potential    / _SCORE_NORM, 1.0)
            upgrade_draw_value  = upgrade_outs   * min(upgrade_potential  / _SCORE_NORM, 1.0)
        else:
            straight_draw_value = flush_draw_value = upgrade_draw_value = 0.0

        return (estimates
                + [score_needed,
                   min(best_immediate / _SCORE_NORM, 1.0),
                   straight_draw_value,
                   flush_draw_value,
                   upgrade_draw_value])

    def _get_state(self):
        """
        Encode the current game state as a normalised float vector (length 155).

        Per-card features (× HAND_SIZE = 40):
            rank_norm, suit_norm, chip_norm, extra_chips_norm, extra_mult_norm
        Deck count vector (52):
            1.0 if card still in deck, 0.0 otherwise
        Scalars (4):
            hands_remaining_norm, discards_remaining_norm,
            score_progress, score_velocity
        Draw features (15) — see _get_draw_features()
        Score context (14) — see _get_score_context()
            hand_type_score_estimates(12), score_needed_per_hand(1),
            best_immediate_play_score(1)
        Joker features (30) — see jokers.get_joker_state_features()
        """
        state = []

        # Hand encoding
        for card in self.hand:
            rank_index, suit, chip_value, extra_chips, extra_mult = card
            state.append(rank_index  / 14.0)
            state.append(SUITS.index(suit) / 3.0)
            state.append(chip_value  / 11.0)
            state.append(extra_chips / 50.0)
            state.append(extra_mult  / 10.0)

        while len(state) < config.HAND_SIZE * 5:
            state.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Deck count vector
        state.extend(deck_count_vector(self.deck))

        # Scalars
        hands_played = self.max_hand_plays - self.hands_remaining
        state.append(self.hands_remaining / self.max_hand_plays)
        state.append(
            self.discards_remaining / self.max_discards
            if self.max_discards > 0 else 0.0
        )
        state.append(
            min(self.score / self.blind_target, 1.0)
            if self.blind_target is not None else 0.0
        )
        # Score velocity: how efficiently we're scoring relative to target pace.
        # 1.0 = exactly on pace, >1.0 = ahead, 0.0 = no hands played yet or no blind.
        if self.blind_target is not None and hands_played > 0:
            expected = self.blind_target * (hands_played / self.max_hand_plays)
            state.append(min(self.score / expected, 2.0) / 2.0)
        else:
            state.append(0.0)

        # Draw potential features
        state.extend(self._get_draw_features())

        # Score context features (joker-aware hand estimates + urgency signals)
        state.extend(self._get_score_context())

        # Joker features (30 zeros when no jokers active)
        state.extend(get_joker_state_features(self.active_jokers))

        # Run context (5 floats) -- zeroed in single-blind mode (phases 1-9)
        # ante_norm              : how far through the run we are (0 = start)
        # blind_in_ante          : position within the ante (0=Small, 0.5=Big, 1.0=Boss)
        # money_norm             : current gold / 20 (capped at 1.0)
        # interest_next          : interest that will be earned next shop / 5
        # joker_slot_utilisation : active jokers / MAX_JOKER_SLOTS
        #   Tells the agent how full the joker rack is -- relevant for shop
        #   decisions (full rack means must sell to buy a better joker).
        if self._run_max_antes > 0:
            state.append(run_progress(
                self._run_ante, self._run_blind_idx, self._run_max_antes
            ))
            state.append(self._run_blind_idx / (BLINDS_PER_ANTE - 1))
            state.append(min(self._run_money / 20.0, 1.0))
            state.append(calculate_interest(int(self._run_money)) / 5.0)
            state.append(len(self.active_jokers) / MAX_JOKER_SLOTS)
        else:
            state.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        assert len(state) == config.STATE_SIZE, \
            f"State size mismatch: got {len(state)}, expected {config.STATE_SIZE}"

        return state

# =============================================================================
# RunEnv — full multi-blind run environment
# =============================================================================

class RunEnv:
    """
    Wraps BalatroEnv to implement a full Balatro run: multiple antes, each
    with three blinds (Small -> Big -> Boss), with money earned between them.

    The DQN interacts with RunEnv exactly as it would with BalatroEnv — same
    step() / reset() API. The difference is that an "episode" is now a full
    run rather than a single blind. The run ends when:
        - The agent fails to clear a blind (score < blind_target at 0 hands), OR
        - All antes in the run are completed.

    Jokers are assigned once at the start of each run and held constant across
    all blinds. This gives the run a coherent strategic identity.

    State vector is identical to BalatroEnv (161 floats), with the three run
    context features (ante progress, blind position, money) populated from
    run-level state rather than zeroed.
    """

    # Share the joker pool with BalatroEnv
    _all_jokers = None

    def __init__(self, max_antes: int = None):
        settings            = config.get_phase_settings()
        self.max_discards   = settings["max_discards"]
        self.max_hand_plays = settings["max_hand_plays"]
        self.num_jokers     = settings["num_jokers"]
        self.max_joker_tier = settings["max_joker_tier"]
        self.max_antes      = max_antes or settings.get("max_antes", MAX_ANTES)

        # Load joker pool once
        if RunEnv._all_jokers is None and self.num_jokers > 0:
            RunEnv._all_jokers = load_jokers("jokers.json")

        # Build a single reusable BalatroEnv (settings injected per blind)
        self._env = BalatroEnv()

        # Run state (populated by reset_run)
        self.ante           = 1
        self.blind_idx      = 0
        self.money          = STARTING_MONEY
        self.run_jokers     = []
        self.run_done       = False
        self.blinds_cleared = 0   # total blinds cleared this run
        self.antes_cleared  = 0

    # ------------------------------------------------------------------
    # Public API (mirrors BalatroEnv)
    # ------------------------------------------------------------------

    def reset(self):
        """Start a new run from Ante 1 Small Blind."""
        self.ante           = 1
        self.blind_idx      = 0
        self.money          = STARTING_MONEY
        self.run_done       = False
        self.blinds_cleared = 0
        self.antes_cleared  = 0
        self.shop_history   = []   # list of shop_log dicts, one per shop visit

        # Assign starting jokers for the run (pre-shop)
        # In run mode, agents start with no jokers and acquire them via shop.
        # For training stability we seed with a small number of random jokers
        # so the run isn't completely joker-free until the first shop visit.
        if self.num_jokers > 0 and RunEnv._all_jokers is not None:
            seed_n = max(1, self.num_jokers // 2)   # start with half, buy rest
            self.run_jokers = sample_episode_jokers(
                RunEnv._all_jokers, seed_n, self.max_joker_tier
            )
        else:
            self.run_jokers = []

        return self._start_blind()

    def step(self, card_indices, play=True):
        """
        Delegate to the inner blind env. When a blind completes, advance
        the run state automatically and start the next blind.

        Returns the same (state, reward, done, info) tuple as BalatroEnv,
        where done=True means the run is over (failure or all antes cleared).
        """
        state, reward, blind_done, info = self._env.step(card_indices, play=play)

        if blind_done:
            result = info.get("result", "")

            if result == "win":
                # Earn money for clearing this blind (amount depends on blind position)
                earned        = money_for_win(self._env.hands_remaining, self.blind_idx)
                self.money   += earned

                # Apply interest on held gold before the shop
                # $1 per $5 held, capped at $5
                interest      = calculate_interest(self.money)
                self.money   += interest

                info["money_earned"]   = earned
                info["interest"]       = interest
                info["total_money"]    = self.money
                info["blind_cleared"]  = blind_name(self.ante, self.blind_idx)

                self.blinds_cleared += 1
                shop_log = self._advance_blind()   # also runs shop
                info["shop_log"] = shop_log

                if self.run_done:
                    info["run_result"] = "complete"
                    return state, reward, True, info
                else:
                    next_state = self._start_blind()
                    info["run_result"] = "ongoing"
                    return next_state, reward, False, info

            else:
                self.run_done        = True
                info["run_result"]   = "failed"
                info["blind_failed"] = blind_name(self.ante, self.blind_idx)
                return state, reward, True, info

        # Mid-blind step — just pass through
        return state, reward, False, info

    def valid_play_actions(self):
        return self._env.valid_play_actions()

    def render(self):
        print(f"Run: {blind_name(self.ante, self.blind_idx)}  |  "
              f"Money: ${self.money}  |  "
              f"Jokers: {', '.join(j['name'] for j in self.run_jokers)}")
        self._env.render()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_size(self):
        return config.STATE_SIZE

    @property
    def done(self):
        return self.run_done

    @property
    def active_joker_names(self):
        return [j["name"] for j in self.run_jokers]

    @property
    def hands_remaining(self):
        return self._env.hands_remaining

    @property
    def discards_remaining(self):
        return self._env.discards_remaining

    @property
    def hand(self):
        return self._env.hand

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_blind(self):
        """Reset the inner env for the current ante/blind_idx."""
        target = blind_target(self.ante, self.blind_idx)
        return self._env.reset(
            blind_target_override = target,
            jokers_override       = self.run_jokers,
            run_ante              = self.ante,
            run_blind_idx         = self.blind_idx,
            run_money             = float(self.money),
            run_max_antes         = self.max_antes,
        )

    def _advance_blind(self) -> list:
        """
        Move to the next blind, advancing ante if needed.
        Runs the shop between blinds and updates run_jokers and money.
        Returns the shop_log for this visit (list of action dicts).
        """
        self.blind_idx += 1
        if self.blind_idx >= BLINDS_PER_ANTE:
            self.blind_idx    = 0
            self.ante        += 1
            self.antes_cleared += 1
            if self.ante > self.max_antes:
                self.run_done = True
                return []

        # Run the shop (no shop on the very last blind of the run)
        if not self.run_done and RunEnv._all_jokers is not None:
            self.run_jokers, self.money, shop_log = run_shop(
                all_jokers    = RunEnv._all_jokers,
                active_jokers = self.run_jokers,
                money         = self.money,
                max_tier      = self.max_joker_tier,
                max_slots     = MAX_JOKER_SLOTS,
                n_shop_slots  = SHOP_JOKER_SLOTS,
                buy_threshold = SHOP_BUY_THRESHOLD,
                sell_margin   = SHOP_SELL_UPGRADE_MARGIN,
                reroll_cost   = SHOP_REROLL_COST,
            )
            self.shop_history.append(shop_log)
            return shop_log

        return []
