"""
Experiment 2 — Plot semantic sensitivity (line plot per semantic category).

Reads results/semantic_sensitivity_<tag>.npy and produces:
  results/semantic_lineplot_<tag>.png          — per-category line plot
  results/vitality_semantic_combined_<tag>.png — vitality bars + line plot side-by-side

Usage
-----
python experiments/plot_semantic_heatmap.py
    [--sensitivity results/semantic_sensitivity_flux1_dev.npy]
    [--tag flux1_dev]
    [--vitality results/vitality_scores.json]
    [--threshold 0.92]
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

CATEGORIES     = ["colour", "style", "material", "texture", "shape", "layout", "object"]
PAPER_MM_VITAL = [0, 1, 17, 18]
PAPER_SIN_VITAL= [9, 34, 35, 37, 6]
N_MM_DEFAULT   = 19
N_SINGLE_DEFAULT = 38

COLORS = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))


def _get_block_counts(matrix: np.ndarray):
    """Infer (n_mm, n_single) from matrix row count."""
    n_rows = matrix.shape[0]
    if n_rows == N_MM_DEFAULT:
        return N_MM_DEFAULT, 0
    elif n_rows == N_MM_DEFAULT + N_SINGLE_DEFAULT:
        return N_MM_DEFAULT, N_SINGLE_DEFAULT
    else:
        return N_MM_DEFAULT, max(0, n_rows - N_MM_DEFAULT)


def plot_lineplot(matrix: np.ndarray, out_path: str, model_label: str = ""):
    """Line plot per semantic category — MM-DiT top, single-stream bottom."""
    N_MM, N_SINGLE = _get_block_counts(matrix)
    has_single = N_SINGLE > 0

    if has_single:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 5))
        ax2 = None

    x_mm = np.arange(N_MM)
    for ci, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
        ax1.plot(x_mm, matrix[:N_MM, ci], color=color, label=cat,
                 lw=1.8, marker="o", ms=4)

    for v in PAPER_MM_VITAL:
        if v < N_MM:
            ax1.axvspan(v - 0.4, v + 0.4, alpha=0.15, color="red")

    ax1.set_xlim(-0.5, N_MM - 0.5)
    ax1.set_xticks(range(N_MM))
    ax1.set_xticklabels([str(i) for i in range(N_MM)], fontsize=7)
    ax1.set_ylabel("Sensitivity\n|sim_full - sim_ablated|", fontsize=9)
    ax1.yaxis.set_tick_params(labelsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.set_title("MM-DiT blocks (0-%d)  --  red shading = vital layers" % (N_MM - 1),
                  fontsize=11)
    ax1.legend(fontsize=8, loc="upper right", ncol=4)

    if has_single and ax2 is not None:
        x_sin = np.arange(N_SINGLE)
        for ci, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
            ax2.plot(x_sin, matrix[N_MM:N_MM + N_SINGLE, ci],
                     color=color, lw=1.5, marker="o", ms=3)

        for v in PAPER_SIN_VITAL:
            if v < N_SINGLE:
                ax2.axvspan(v - 0.4, v + 0.4, alpha=0.15, color="red")

        ax2.set_xlim(-0.5, N_SINGLE - 0.5)
        ax2.set_xticks(range(N_SINGLE))
        ax2.set_xticklabels([str(i) for i in range(N_SINGLE)], fontsize=7)
        ax2.set_ylabel("Sensitivity\n|sim_full - sim_ablated|", fontsize=9)
        ax2.yaxis.set_tick_params(labelsize=8)
        ax2.grid(True, alpha=0.25)
        ax2.set_title("Single-stream blocks (0-%d)" % (N_SINGLE - 1), fontsize=11)
        ax2.set_xlabel("Layer index", fontsize=9)
    else:
        ax1.set_xlabel("Layer index", fontsize=9)

    title = "Semantic Sensitivity per Layer"
    if model_label:
        title += "  [%s]" % model_label
    title += "\n(DINOv2 |sim_full - sim_ablated|  --  red shading = paper vital layers)"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved: %s" % out_path)
    plt.close()


def plot_combined(matrix: np.ndarray, vitality_path: str,
                  out_path: str, threshold: float, model_label: str = ""):
    """Left: vitality bar chart. Right: semantic line plot (MM-DiT blocks)."""
    N_MM, N_SINGLE = _get_block_counts(matrix)

    with open(vitality_path) as f:
        vdata = json.load(f)

    mm_means  = [float(np.mean(vdata["mm"][str(i)]))     for i in range(N_MM)]
    sin_means = ([float(np.mean(vdata["single"][str(i)])) for i in range(N_SINGLE)]
                 if N_SINGLE > 0 else [])

    fig = plt.figure(figsize=(22, 8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2.5], wspace=0.12)

    # Left: vitality horizontal bar chart
    ax_v = fig.add_subplot(gs[0])
    all_means    = mm_means + sin_means
    n_total      = N_MM + N_SINGLE
    vital_global = PAPER_MM_VITAL + [N_MM + i for i in PAPER_SIN_VITAL if i < N_SINGLE]
    bar_colors   = ["#d62728" if i in vital_global else "#1f77b4" for i in range(n_total)]

    ax_v.barh(range(n_total), all_means, color=bar_colors, height=0.7)
    ax_v.axvline(threshold, color="red", linestyle="--", linewidth=1.2)
    if N_SINGLE > 0:
        ax_v.axhline(N_MM - 0.5, color="black", linewidth=2)

    labels = ([f"MM-{i}" for i in range(N_MM)] +
              [f"S-{i}"  for i in range(N_SINGLE)])
    ax_v.set_yticks(range(n_total))
    ax_v.set_yticklabels(labels, fontsize=6)
    ax_v.invert_yaxis()
    ax_v.set_xlabel("DINOv2 similarity (lower = more vital)", fontsize=8)
    ax_v.set_title("Layer Vitality\n(red = paper vital layers)", fontsize=10)
    for i in vital_global:
        if i < len(all_means):
            ax_v.text(all_means[i] + 0.002, i, "*",
                      va="center", fontsize=10, color="#d62728", fontweight="bold")

    # Right: semantic line plot -- MM-DiT blocks
    ax_s = fig.add_subplot(gs[1])
    x_mm = np.arange(N_MM)
    for ci, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
        ax_s.plot(x_mm, matrix[:N_MM, ci], color=color, label=cat,
                  lw=1.8, marker="o", ms=4)

    for v in PAPER_MM_VITAL:
        if v < N_MM:
            ax_s.axvspan(v - 0.4, v + 0.4, alpha=0.15, color="red")

    ax_s.set_xlim(-0.5, N_MM - 0.5)
    ax_s.set_xticks(range(N_MM))
    ax_s.set_xticklabels([str(i) for i in range(N_MM)], fontsize=8)
    ax_s.set_ylabel("Sensitivity  |sim_full - sim_ablated|", fontsize=9)
    ax_s.set_xlabel("MM-DiT layer index", fontsize=9)
    ax_s.grid(True, alpha=0.25)
    ax_s.set_title("Semantic Sensitivity -- MM-DiT blocks\n"
                   "(red shading = vital layers)", fontsize=10)
    ax_s.legend(fontsize=8, loc="upper right", ncol=4)

    title = "FLUX Layer Analysis -- Vitality x Semantic Specialisation"
    if model_label:
        title += "  [%s]" % model_label
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved: %s" % out_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", type=str, default=None,
                        help="Path to .npy file, or use --tag to auto-resolve")
    parser.add_argument("--tag",         type=str, default=None,
                        help="Model tag e.g. 'flux1_dev' -- resolves to "
                             "results/semantic_sensitivity_<tag>.npy")
    parser.add_argument("--vitality",    type=str,
                        default="results/vitality_scores.json")
    parser.add_argument("--threshold",   type=float, default=0.92)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve sensitivity path
    if args.sensitivity is None and args.tag is None:
        args.sensitivity = "results/semantic_sensitivity_flux1_dev.npy"
        print("No --sensitivity or --tag given, defaulting to %s" % args.sensitivity)
    elif args.tag:
        args.sensitivity = f"results/semantic_sensitivity_{args.tag}.npy"

    tag_str     = args.tag or os.path.splitext(os.path.basename(args.sensitivity))[0]
    model_label = tag_str.replace("_", " ").upper()
    matrix      = np.load(args.sensitivity)

    plot_lineplot(matrix,
                  f"results/semantic_lineplot_{tag_str}.png",
                  model_label=model_label)

    if os.path.exists(args.vitality):
        plot_combined(matrix, args.vitality,
                      f"results/vitality_semantic_combined_{tag_str}.png",
                      args.threshold, model_label=model_label)
    else:
        print("No vitality scores at %s -- skipping combined plot." % args.vitality)
