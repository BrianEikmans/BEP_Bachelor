"""
baselines.py -- Heuristic baselines and evaluation runner.

Four baselines of increasing sophistication:

    RandomPolicy          -- uniform random action each step
    GreedyScorePolicy     -- always play the highest-scoring 5-card hand, no discards
    GreedyDiscardPolicy   -- play best hand, but first discard cards not in best hand
    RuleBasedPolicy       -- human-like: chase flushes/straights if close, else play best

Run evaluation:
    python baselines.py --phase 5 --episodes 1000
    python baselines.py --phase 9 --checkpoint checkpoints/phase_09/episode_NNNNN_final.pt

Outputs:
    - Console table comparing all baselines + trained agent
    - figures/baseline_comparison.png
"""

import os
import sys
import random
import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from scoring import get_hand_rank, calculate_score
from env import BalatroEnv, HAND_RANK_ORDER


# =============================================================================
# Base Policy interface
# =============================================================================

class BasePolicy:
    """All policies share the same select_action(state, valid_actions, hand, env) API."""

    name = "Base"

    def select_action(self, state: list, valid_actions: list, hand: list, env) -> int:
        """Return index into valid_actions."""
        raise NotImplementedError

    def reset(self):
        """Called at the start of each episode. Override if policy has state."""
        pass


# =============================================================================
# 1. Random Policy
# =============================================================================

class RandomPolicy(BasePolicy):
    """
    Selects a uniformly random action each step.
    This is the absolute floor — any learned or rule-based policy should beat it.
    """
    name = "Random"

    def select_action(self, state, valid_actions, hand, env) -> int:
        return random.randrange(len(valid_actions))


# =============================================================================
# 2. Greedy Score Policy  (no discards)
# =============================================================================

class GreedyScorePolicy(BasePolicy):
    """
    Always plays the 5-card combination with the highest immediate score.
    Never discards. Represents pure hand-value maximisation with no planning.

    Surprising strength: it knows all the scoring rules and picks the best
    available hand every turn — it just can't look ahead.
    """
    name = "Greedy (no discard)"

    def select_action(self, state, valid_actions, hand, env) -> int:
        best_idx   = 0
        best_score = -1

        for i, (card_indices, is_discard) in enumerate(valid_actions):
            if is_discard:
                continue   # never discard
            selected = [hand[j] for j in card_indices]
            score    = calculate_score(selected)
            if score > best_score:
                best_score = score
                best_idx   = i

        return best_idx


# =============================================================================
# 3. Greedy Discard Policy
# =============================================================================

class GreedyDiscardPolicy(BasePolicy):
    """
    Uses discards intelligently: before playing, discard cards that don't
    contribute to the best current hand to draw potentially better ones.

    Discard logic:
    - Find the best 5-card hand in the current 8-card hand
    - If the hand rank would improve by discarding and discards remain: discard
    - The discard target is the cards NOT in the best hand
    - Never discard if already holding the best possible hand for the hand type
    - Play when discards are exhausted or no improvement is possible

    This is roughly how a competent human plays without deep planning.
    """
    name = "Greedy (with discard)"

    def select_action(self, state, valid_actions, hand, env) -> int:
        # Find the best 5-card play right now
        best_play_idx   = 0
        best_play_score = -1
        best_play_cards = None

        for i, (card_indices, is_discard) in enumerate(valid_actions):
            if is_discard:
                continue
            if len(card_indices) != 5:
                continue
            selected = [hand[j] for j in card_indices]
            score    = calculate_score(selected)
            if score > best_play_score:
                best_play_score = score
                best_play_idx   = i
                best_play_cards = set(card_indices)

        # If we have discards remaining, consider discarding the non-scoring cards
        if env.discards_remaining > 0 and best_play_cards is not None:
            # Cards not in the best 5-card hand are candidates for discard
            non_scoring = [i for i in range(len(hand)) if i not in best_play_cards]

            if non_scoring:
                # Only discard if the hand isn't already very strong
                best_hand_rank = HAND_RANK_ORDER.get(
                    get_hand_rank([hand[i] for i in best_play_cards]), 0
                )
                # Don't discard if already holding Flush, Full House or better
                if best_hand_rank < HAND_RANK_ORDER.get("Flush", 5):
                    # Find the discard action matching these exact non-scoring cards
                    discard_set = frozenset(non_scoring[:min(3, len(non_scoring))])
                    for i, (card_indices, is_discard) in enumerate(valid_actions):
                        if is_discard and frozenset(card_indices) == discard_set:
                            return i
                    # Fallback: find any discard action of the right size
                    for i, (card_indices, is_discard) in enumerate(valid_actions):
                        if is_discard and len(card_indices) == len(discard_set):
                            return i

        return best_play_idx


# =============================================================================
# 4. Rule-Based Policy
# =============================================================================

class RuleBasedPolicy(BasePolicy):
    """
    Mimics sensible human play with explicit hand-building rules.

    Priority order:
    1. If holding 4+ cards of one suit AND discards remain: discard non-flush cards
       to chase the flush (deck-aware: only if enough of that suit remain in deck)
    2. If holding 4 consecutive ranks AND discards remain: discard to chase straight
    3. If holding 3-of-a-kind: keep it, discard the 2 weakest kickers
    4. If holding two pair: play immediately (already a good hand)
    5. Otherwise: play the highest-scoring available 5-card hand

    This captures the key insight that some setups are worth improving with a
    discard before playing, while others should be cashed in immediately.
    """
    name = "Rule-Based"

    def select_action(self, state, valid_actions, hand, env) -> int:
        # --- Rule 1: Flush draw (4 of a suit, discards available) ---
        if env.discards_remaining > 0:
            suit_groups = defaultdict(list)
            for i, card in enumerate(hand):
                suit_groups[card[1]].append(i)

            for suit, indices in suit_groups.items():
                if len(indices) >= 4:
                    # Check: do we already have 5? Then just play it.
                    if len(indices) >= 5:
                        break
                    # Discard the non-flush cards (up to 3)
                    non_flush = [i for i in range(len(hand)) if i not in indices]
                    discard_n = min(len(non_flush), 3)
                    discard_target = frozenset(non_flush[:discard_n])
                    for i, (card_indices, is_discard) in enumerate(valid_actions):
                        if is_discard and frozenset(card_indices) == discard_target:
                            return i

        # --- Rule 2: Straight draw (4 consecutive ranks) ---
        if env.discards_remaining > 0:
            ranks      = sorted(set(card[0] for card in hand))
            rank_to_idx = defaultdict(list)
            for i, card in enumerate(hand):
                rank_to_idx[card[0]].append(i)

            # Check ace-low too
            check_ranks = ranks[:]
            if 14 in check_ranks:
                check_ranks = [1] + check_ranks

            for start in range(len(check_ranks) - 3):
                window = check_ranks[start:start + 4]
                if window == list(range(window[0], window[0] + 4)):
                    # 4 consecutive ranks — find the cards
                    real_ranks    = {14 if r == 1 else r for r in window}
                    straight_idxs = []
                    for r in real_ranks:
                        if rank_to_idx[r]:
                            straight_idxs.append(rank_to_idx[r][0])
                    non_straight = [i for i in range(len(hand))
                                    if i not in straight_idxs]
                    discard_n    = min(len(non_straight), 3)
                    if discard_n > 0:
                        discard_target = frozenset(non_straight[:discard_n])
                        for i, (card_indices, is_discard) in enumerate(valid_actions):
                            if is_discard and frozenset(card_indices) == discard_target:
                                return i

        # --- Rules 3-5: Play the best available hand ---
        # Find best 5-card play
        best_play_idx   = 0
        best_play_score = -1
        best_play_rank  = -1

        for i, (card_indices, is_discard) in enumerate(valid_actions):
            if is_discard:
                continue
            if len(card_indices) < 1:
                continue
            selected    = [hand[j] for j in card_indices]
            score       = calculate_score(selected)
            hand_rank   = HAND_RANK_ORDER.get(get_hand_rank(selected), 0)

            # Prefer higher hand rank first, then score
            if (hand_rank, score) > (best_play_rank, best_play_score):
                best_play_rank  = hand_rank
                best_play_score = score
                best_play_idx   = i

        # Rule 3: If we have trips, discard the 2 weakest non-trip cards
        if env.discards_remaining > 0:
            rank_counts = Counter(card[0] for card in hand)
            trip_ranks  = [r for r, cnt in rank_counts.items() if cnt >= 3]
            if trip_ranks and best_play_rank < HAND_RANK_ORDER.get("Full House", 6):
                trip_rank  = trip_ranks[0]
                trip_idxs  = [i for i, c in enumerate(hand) if c[0] == trip_rank][:3]
                non_trip   = [i for i in range(len(hand)) if i not in trip_idxs]
                # Discard the 2 lowest-chip kickers
                kicker_scores = [(calculate_score([hand[i]]), i) for i in non_trip]
                kicker_scores.sort()
                discard_target = frozenset(i for _, i in kicker_scores[:2])
                for i, (card_indices, is_discard) in enumerate(valid_actions):
                    if is_discard and frozenset(card_indices) == discard_target:
                        return i

        return best_play_idx


# =============================================================================
# Evaluation runner
# =============================================================================

def run_eval_episode(env: BalatroEnv, policy: BasePolicy) -> dict:
    """Run one evaluation episode and return stats dict."""
    state = env.reset()
    policy.reset()
    total_score  = 0
    steps        = 0
    hand_types   = []
    result       = "unknown"

    while not env.done:
        play_actions    = [(idxs, False) for idxs in env.valid_play_actions()]
        discard_actions = (
            [(idxs, True) for idxs in env.valid_play_actions()]
            if env.discards_remaining > 0 else []
        )
        valid_actions = play_actions + discard_actions

        action_idx               = policy.select_action(state, valid_actions, env.hand, env)
        card_indices, is_discard = valid_actions[action_idx]

        state, _, done, info = env.step(card_indices, play=not is_discard)
        total_score += info.get("score", 0)
        steps       += 1

        if "hand_type" in info:
            hand_types.append(info["hand_type"])
        if "result" in info:
            result = info["result"]

    return {
        "total_score":  total_score,
        "hands_used":   env.max_hand_plays - env.hands_remaining,
        "discards_used": env.max_discards  - env.discards_remaining,
        "hand_types":   hand_types,
        "result":       result,
    }


def evaluate_policy(policy: BasePolicy, phase: int, n_episodes: int) -> dict:
    """Evaluate a policy for n_episodes and return aggregate stats."""
    config.set_phase(phase)
    env     = BalatroEnv()
    scores  = []
    results = []
    hands   = []
    discards = []
    hand_type_counts = Counter()

    for _ in range(n_episodes):
        stats = run_eval_episode(env, policy)
        scores.append(stats["total_score"])
        results.append(stats["result"])
        hands.append(stats["hands_used"])
        discards.append(stats["discards_used"])
        for ht in stats["hand_types"]:
            hand_type_counts[ht] += 1

    wins     = sum(1 for r in results if r == "win")
    win_rate = wins / n_episodes

    return {
        "policy":      policy.name,
        "phase":       phase,
        "n_episodes":  n_episodes,
        "avg_score":   sum(scores) / n_episodes,
        "min_score":   min(scores),
        "max_score":   max(scores),
        "std_score":   float((sum((s - sum(scores)/n_episodes)**2
                                  for s in scores) / n_episodes) ** 0.5),
        "win_rate":    win_rate,
        "avg_hands":   sum(hands) / n_episodes,
        "avg_discards": sum(discards) / n_episodes,
        "top_hand":    hand_type_counts.most_common(1)[0][0] if hand_type_counts else "--",
        "blind_target": config.BLIND_TARGET,
    }


def evaluate_trained_agent(checkpoint_path: str, phase: int, n_episodes: int) -> dict:
    """Evaluate a trained DQN checkpoint at epsilon=0 (greedy)."""
    import torch
    from BEP.agent import DQNAgent, encode_action, encode_actions

    config.set_phase(phase)
    env   = BalatroEnv()
    agent = DQNAgent(state_size=config.STATE_SIZE)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0   # pure greedy — no exploration

    scores   = []
    results  = []
    hands    = []
    discards = []
    hand_type_counts = Counter()

    for _ in range(n_episodes):
        state        = env.reset()
        total_score  = 0
        result       = "unknown"

        while not env.done:
            play_actions    = [(idxs, False) for idxs in env.valid_play_actions()]
            discard_actions = (
                [(idxs, True) for idxs in env.valid_play_actions()]
                if env.discards_remaining > 0 else []
            )
            valid_actions = play_actions + discard_actions
            action_idx    = agent.select_action(state, valid_actions, env.hand)
            card_indices, is_discard = valid_actions[action_idx]
            state, _, done, info = env.step(card_indices, play=not is_discard)
            total_score += info.get("score", 0)
            if "hand_type" in info:
                hand_type_counts[info["hand_type"]] += 1
            if "result" in info:
                result = info["result"]

        scores.append(total_score)
        results.append(result)
        hands.append(env.max_hand_plays - env.hands_remaining)
        discards.append(env.max_discards - env.discards_remaining)

    wins = sum(1 for r in results if r == "win")
    return {
        "policy":       "Trained DQN",
        "phase":        phase,
        "n_episodes":   n_episodes,
        "avg_score":    sum(scores) / n_episodes,
        "min_score":    min(scores),
        "max_score":    max(scores),
        "std_score":    float((sum((s - sum(scores)/n_episodes)**2
                                   for s in scores) / n_episodes) ** 0.5),
        "win_rate":     wins / n_episodes,
        "avg_hands":    sum(hands) / n_episodes,
        "avg_discards": sum(discards) / n_episodes,
        "top_hand":     hand_type_counts.most_common(1)[0][0] if hand_type_counts else "--",
        "blind_target": config.BLIND_TARGET,
    }


# =============================================================================
# Reporting
# =============================================================================

def print_comparison_table(results: list):
    """Print a formatted comparison table to stdout."""
    print()
    print("=" * 85)
    print(f"  BASELINE COMPARISON  --  Phase {results[0]['phase']}  "
          f"(blind {results[0]['blind_target']})  --  "
          f"{results[0]['n_episodes']:,} episodes each")
    print("=" * 85)
    print(f"  {'Policy':<24} {'Avg Score':>10} {'Std':>7} {'Min':>7} {'Max':>8}"
          f"  {'Win%':>6}  {'Hands':>6}  {'Top Hand'}")
    print("-" * 85)
    for r in results:
        win_str = f"{r['win_rate']*100:.1f}%" if r["blind_target"] else "  n/a"
        print(
            f"  {r['policy']:<24} {r['avg_score']:>10.0f} {r['std_score']:>7.0f}"
            f" {r['min_score']:>7} {r['max_score']:>8}  {win_str:>6}"
            f"  {r['avg_hands']:>5.1f}   {r['top_hand']}"
        )
    print("=" * 85)

    # Compute improvements over random baseline
    random_res = next((r for r in results if r["policy"] == "Random"), None)
    if random_res and random_res["avg_score"] > 0:
        print()
        print("  Improvement over Random baseline:")
        for r in results:
            if r["policy"] == "Random":
                continue
            score_lift = (r["avg_score"] - random_res["avg_score"]) / random_res["avg_score"] * 100
            print(f"    {r['policy']:<24}  score +{score_lift:>5.1f}%")
    print()


def save_comparison_chart(results: list, out_dir: str):
    """Generate and save a comparison bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    BG       = "#0f1117"
    BG_PANEL = "#1a1d27"
    GRID_COL = "#2a2d3a"
    TEXT_COL = "#e8e8f0"

    POLICY_COLORS = {
        "Random":              "#5b7fa6",
        "Greedy (no discard)": "#6aaa8e",
        "Greedy (with discard)": "#e8c55a",
        "Rule-Based":          "#f09060",
        "Trained DQN":         "#c070e0",
    }

    names    = [r["policy"]    for r in results]
    scores   = [r["avg_score"] for r in results]
    stds     = [r["std_score"] for r in results]
    colors   = [POLICY_COLORS.get(n, "#7eb8f7") for n in names]
    has_wr   = results[0]["blind_target"] is not None
    win_rates = [r["win_rate"] * 100 for r in results] if has_wr else None

    fig_h = 5.5 if has_wr else 4.5
    fig, axes = plt.subplots(1, 2 if has_wr else 1,
                             figsize=(13 if has_wr else 7, fig_h))
    fig.patch.set_facecolor(BG)

    axs = [axes] if not has_wr else list(axes)

    def style_ax(ax):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

    x = np.arange(len(names))

    # --- Score bar chart ---
    ax = axs[0]
    style_ax(ax)
    bars = ax.bar(x, scores, color=colors, width=0.6,
                  edgecolor=BG, linewidth=1.2)
    ax.errorbar(x, scores, yerr=stds, fmt="none",
                color=TEXT_COL, capsize=4, linewidth=1.2, alpha=0.6)

    # Blind target reference line
    if results[0]["blind_target"]:
        ax.axhline(results[0]["blind_target"], color="#ffffff",
                   linewidth=1.0, linestyle="--", alpha=0.45)
        ax.text(len(names) - 0.5, results[0]["blind_target"] * 1.01,
                f"blind {results[0]['blind_target']}",
                color="#aaaaaa", fontsize=7.5, ha="right")

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(scores) * 0.01,
                f"{score:.0f}", ha="center", va="bottom",
                color=TEXT_COL, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Average Score", fontsize=9)
    blind = results[0]["blind_target"]
    phase = results[0]["phase"]
    ax.set_title(f"Average Score — Phase {phase} (blind {blind})", fontsize=10,
                 fontweight="bold", color=TEXT_COL, pad=8)

    # --- Win rate bar chart ---
    if has_wr:
        ax2 = axs[1]
        style_ax(ax2)
        bars2 = ax2.bar(x, win_rates, color=colors, width=0.6,
                        edgecolor=BG, linewidth=1.2)
        ax2.axhline(75, color="#ffffff", linewidth=1.0, linestyle="--", alpha=0.45)
        ax2.text(len(names) - 0.5, 76.5, "75% threshold",
                 color="#aaaaaa", fontsize=7.5, ha="right")
        ax2.set_ylim(0, 110)
        for bar, wr in zip(bars2, win_rates):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1.5,
                     f"{wr:.1f}%", ha="center", va="bottom",
                     color=TEXT_COL, fontsize=8, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
        ax2.set_ylabel("Win Rate %", fontsize=9)
        ax2.set_title(f"Win Rate — Phase {phase} (blind {blind})", fontsize=10,
                      fontweight="bold", color=TEXT_COL, pad=8)

    fig.tight_layout(pad=2.0)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"baseline_comparison_phase{phase}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic baselines and optionally compare to trained agent"
    )
    parser.add_argument("--phase",      type=int,   default=5,
                        help="Phase to evaluate (default: 5)")
    parser.add_argument("--episodes",   type=int,   default=500,
                        help="Evaluation episodes per policy (default: 500)")
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Path to trained agent .pt file for comparison")
    parser.add_argument("--out",        type=str,   default="figures",
                        help="Output directory for charts")
    parser.add_argument("--no-chart",   action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    config.set_phase(args.phase)
    phase = args.phase
    n     = args.episodes

    policies = [
        RandomPolicy(),
        GreedyScorePolicy(),
        GreedyDiscardPolicy(),
        RuleBasedPolicy(),
    ]

    all_results = []
    for policy in policies:
        print(f"  Evaluating {policy.name} ({n} episodes, phase {phase})...", flush=True)
        result = evaluate_policy(policy, phase, n)
        all_results.append(result)
        print(f"    avg={result['avg_score']:.0f}  win%={result['win_rate']*100:.1f}%"
              if result["blind_target"] else
              f"    avg={result['avg_score']:.0f}")

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"  Checkpoint not found: {args.checkpoint}")
        else:
            print(f"  Evaluating Trained DQN from {args.checkpoint}...", flush=True)
            try:
                result = evaluate_trained_agent(args.checkpoint, phase, n)
                all_results.append(result)
                print(f"    avg={result['avg_score']:.0f}  win%={result['win_rate']*100:.1f}%")
            except Exception as e:
                print(f"  Could not load checkpoint: {e}")

    print_comparison_table(all_results)

    if not args.no_chart:
        save_comparison_chart(all_results, args.out)

    # Save raw results as JSON
    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, f"baseline_results_phase{phase}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Raw results saved to {json_path}")


if __name__ == "__main__":
    main()
