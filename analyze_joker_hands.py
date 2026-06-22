"""
analyze_joker_hands.py
======================
Loads a trained checkpoint, runs N episodes (epsilon=0), and
produces a heatmap of: for each joker, which hand types did the agent
play most often while that joker was active?

Usage
-----
    # Full run: collect data from checkpoint AND plot
    python analyze_joker_hands.py \\
        --checkpoint checkpoints/phase_08/episode_20000_final.pt \\
        --phase      8 \\
        --episodes   500 \\
        --output     joker_hand_heatmap.png

    # Skip re-running episodes; re-plot from a previously saved JSON
    python analyze_joker_hands.py \\
        --load-data  joker_hand_data.json \\
        --normalise  hand \\
        --output     joker_hand_heatmap_hand_norm.png

Arguments
---------
    --checkpoint   Path to a .pt checkpoint file.
    --phase        Training phase the checkpoint was saved under (default: 8).
    --episodes     Number of episodes to run (default: 500).
    --output       Output image path (default: joker_hand_heatmap.png).
    --normalise    One of: "joker" (per-joker row sum = 1, default),
                           "hand"  (per-hand-type col sum = 1),
                           "total" (global sum = 1),
                           "none"  (raw counts).
    --save-data    Path to write collected counts as JSON (default: joker_hand_data.json).
    --load-data    Skip episode collection and load counts from this JSON file instead.
    --use-run-env  Force RunEnv (phases 10-12). Auto-detected from phase by default.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Allow running from any working directory by adding the project root to path.
# Assumes this script lives in the same folder as env.py / agent.py, OR one
# level above them (e.g. in a tools/ sub-folder).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE, os.path.dirname(_HERE)):
    if os.path.isfile(os.path.join(_candidate, "env.py")):
        sys.path.insert(0, _candidate)
        break


# ---------------------------------------------------------------------------
# Visual style — matches generate_graphs.py exactly
# ---------------------------------------------------------------------------
BG       = "#0f1117"
BG_PANEL = "#1a1d27"
GRID     = "#2a2d3a"
TEXT     = "#e8e8f0"
ACCENT   = "#7eb8f7"

def _style(fig, ax):
    """Apply the shared dark theme to a figure/axes pair."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Ordered axis labels
# ---------------------------------------------------------------------------
HAND_TYPES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Five of a Kind", "Flush House", "Flush Five",
]


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------

def collect_data(agent, env, n_episodes: int, use_run_env: bool):
    """
    Run n_episodes with epsilon=0 and record which hand type was played
    for each set of active jokers.

    Mirrors run_episode() in main.py exactly:
      - valid_actions = play_actions + discard_actions
      - agent selects from the COMBINED action space (so it can discard)
      - hand_type is recorded only when the chosen action is a play

    Returns:
        counts : dict  { joker_name : { hand_type : int } }
    """
    agent.epsilon = 0.0

    # counts[joker_name][hand_type] = number of times that hand was played
    # while that joker was active
    counts = defaultdict(lambda: defaultdict(int))

    for ep in range(n_episodes):
        state = env.reset()

        while not env.done:
            hand = env.hand

            # --- Mirror main.py run_episode exactly ---
            play_actions    = [(idxs, False) for idxs in env.valid_play_actions()]
            discard_actions = (
                [(idxs, True) for idxs in env.valid_play_actions()]
                if env.discards_remaining > 0 else []
            )
            valid_actions = play_actions + discard_actions

            if not valid_actions:
                break

            action_idx               = agent.select_action(state, valid_actions, hand)
            card_indices, is_discard = valid_actions[action_idx]

            # --- Identify active jokers BEFORE the step ---
            # Both BalatroEnv and RunEnv expose active_joker_names as a property.
            active_jokers = env.active_joker_names

            state, reward, done, info = env.step(card_indices, play=not is_discard)

            # Record hand type only when the agent actually played a hand
            if not is_discard:
                hand_type = info.get("hand_type")
                if hand_type and active_jokers:
                    for joker_name in active_jokers:
                        counts[joker_name][hand_type] += 1

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep + 1}/{n_episodes} done")

    # Convert nested defaultdict → plain dict for JSON serialisation
    return {j: dict(h) for j, h in counts.items()}


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def save_counts(counts: dict, path: str, meta=None):
    """Write counts + optional metadata to a JSON file."""
    payload = {"meta": meta or {}, "counts": counts}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Data saved → {path}")


def load_counts(path: str):
    """Load counts (and metadata) from a previously saved JSON file."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    meta   = payload.get("meta", {})
    counts = payload.get("counts", payload)   # backwards-compat: bare dict
    print(f"Data loaded ← {path}  (meta: {meta})")
    return counts, meta


# ---------------------------------------------------------------------------
# Build matrix
# ---------------------------------------------------------------------------

def build_matrix(counts, normalise: str):
    """
    Convert counts dict to a 2-D numpy array (jokers × hand_types).

    normalise : "joker"  — each row sums to 1  (how does THIS joker affect hand choice?)
                "hand"   — each col sums to 1  (which jokers favour THIS hand?)
                "total"  — global sum = 1
                "none"   — raw counts
    """
    joker_names = sorted(counts.keys())
    matrix = np.zeros((len(joker_names), len(HAND_TYPES)), dtype=np.float64)

    for r, jname in enumerate(joker_names):
        for c, htype in enumerate(HAND_TYPES):
            matrix[r, c] = counts[jname].get(htype, 0)

    if normalise == "joker":
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # avoid div-by-zero for unseen jokers
        matrix = matrix / row_sums
        label = "Fraction of plays per joker (row-normalised)"
    elif normalise == "hand":
        col_sums = matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        matrix = matrix / col_sums
        label = "Fraction of plays per hand type (col-normalised)"
    elif normalise == "total":
        total = matrix.sum()
        matrix = matrix / (total if total > 0 else 1)
        label = "Fraction of total plays"
    else:
        label = "Raw play count"

    return matrix, joker_names, label


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_heatmap(matrix, joker_names, value_label,
                 n_episodes, phase, output_path):
    """Render the joker × hand-type heatmap in the shared dark style."""

    # Drop hand-type columns that were never played (all zeros) to declutter
    col_mask  = matrix.sum(axis=0) > 0
    matrix    = matrix[:, col_mask]
    hand_lbls = [h for h, m in zip(HAND_TYPES, col_mask) if m]

    n_jokers = len(joker_names)

    fig_w = max(12, len(hand_lbls) * 1.1)
    fig_h = max(5,  n_jokers * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _style(fig, ax)
    ax.grid(False)   # grid lines look bad on heatmaps; disable after _style

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=matrix.max() if matrix.max() > 0 else 1)

    # Axis ticks
    ax.set_xticks(range(len(hand_lbls)))
    ax.set_xticklabels(hand_lbls, rotation=35, ha="right", fontsize=9, color=TEXT)
    ax.set_yticks(range(n_jokers))
    ax.set_yticklabels(joker_names, fontsize=9, color=TEXT)

    ax.set_xlabel("Hand Type Played", fontsize=11, labelpad=8)
    ax.set_ylabel("Active Joker",     fontsize=11, labelpad=8)
    ax.set_title(
        f"Joker \u00d7 Hand-Type Heatmap\n"
        f"Phase {phase}  \u00b7  {n_episodes} episodes  \u00b7  {value_label}",
        fontsize=12, fontweight="bold", pad=14,
    )

    # Annotate cells — always black text for readability on the YlOrRd colormap
    fmt = ".2f" if matrix.max() <= 1.0 else ".0f"
    for r in range(n_jokers):
        for c in range(len(hand_lbls)):
            val = matrix[r, c]
            if val == 0:
                continue
            ax.text(c, r, format(val, fmt),
                    ha="center", va="center", fontsize=7,
                    color="black", fontweight="bold")

    # Colour bar — styled to match generate_graphs.py
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(value_label, color=TEXT, fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Heatmap saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Joker × hand-type heatmap from a trained Balatro DQN checkpoint."
    )
    # --- data source (mutually exclusive) ---
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--checkpoint",
                     help="Path to .pt checkpoint file (runs episodes)")
    src.add_argument("--load-data",
                     help="Skip episode collection; load counts from this JSON file")

    parser.add_argument("--phase",    type=int, default=8,
                        help="Training phase the checkpoint belongs to (default: 8)")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes to run (default: 500)")
    parser.add_argument("--output",   default="joker_hand_heatmap.png",
                        help="Output PNG path (default: joker_hand_heatmap.png)")
    parser.add_argument("--normalise",
                        choices=["joker", "hand", "total", "none"],
                        default="joker",
                        help="Normalisation mode (default: joker)")
    parser.add_argument("--save-data", default="joker_hand_data.json",
                        help="Path to write collected counts as JSON "
                             "(default: joker_hand_data.json)")
    parser.add_argument("--use-run-env", action="store_true",
                        help="Force RunEnv (auto-detected for phases 10-12)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Branch A: load previously saved data (no env/agent needed)
    # ------------------------------------------------------------------
    if args.load_data:
        print(f"Loading data from {args.load_data} ...")
        counts, meta = load_counts(args.load_data)
        phase      = meta.get("phase",    args.phase)
        n_episodes = meta.get("episodes", args.episodes)

    # ------------------------------------------------------------------
    # Branch B: run episodes from a checkpoint
    # ------------------------------------------------------------------
    else:
        if not args.checkpoint:
            parser.error("Provide either --checkpoint or --load-data.")

        import config
        config.set_phase(args.phase)

        use_run_env = args.use_run_env or config.USE_RUN_ENV
        phase_cfg   = config.get_phase_settings(args.phase)

        if phase_cfg["num_jokers"] == 0:
            print(f"WARNING: Phase {args.phase} has 0 jokers — heatmap will be empty.\n"
                  "         Use a phase with jokers (e.g. --phase 8).")

        print(f"Phase {args.phase} settings: {phase_cfg}")
        print(f"Env type: {'RunEnv' if use_run_env else 'BalatroEnv'}")

        from env   import BalatroEnv, RunEnv
        from agent import DQNAgent

        env   = RunEnv(max_antes=config.MAX_ANTES) if use_run_env else BalatroEnv()
        agent = DQNAgent(state_size=config.STATE_SIZE)

        print(f"\nLoading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
        print("Checkpoint loaded.\n")

        print(f"Running {args.episodes} episodes...")
        counts = collect_data(agent, env, args.episodes, use_run_env)

        if not counts:
            print("No joker data collected — check that the phase has active jokers "
                  "and that the checkpoint matches the phase.")
            return

        print(f"\nJokers observed: {sorted(counts.keys())}")

        # Save raw counts so the graph can be re-drawn later without re-running
        meta = {"phase": args.phase, "episodes": args.episodes,
                "checkpoint": args.checkpoint, "normalise": args.normalise}
        save_counts(counts, args.save_data, meta)

        phase      = args.phase
        n_episodes = args.episodes

    # ------------------------------------------------------------------
    # Build matrix & plot
    # ------------------------------------------------------------------
    matrix, joker_names, value_label = build_matrix(counts, args.normalise)
    plot_heatmap(matrix, joker_names, value_label,
                 n_episodes, phase, args.output)

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    print("\n--- Summary: most-played hand per joker ---")
    for jname in joker_names:
        row   = {h: counts[jname].get(h, 0) for h in HAND_TYPES}
        top   = max(row, key=row.get)
        total = sum(row.values())
        pct   = 100 * row[top] / total if total else 0
        print(f"  {jname:<30}  →  {top} ({pct:.1f}% of {total} plays)")


if __name__ == "__main__":
    main()