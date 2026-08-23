"""
Experiment 1 — Plot layer vitality bar chart.

Reads results/vitality_scores.json produced by vitality_sweep.py and
generates results/vitality_plot.png, replicating Figure 4 from the paper.

Usage
-----
python experiments/plot_vitality.py
    [--scores  results/vitality_scores.json]
    [--out     results/vitality_plot.png]
    [--threshold 0.92]
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PAPER_MM_VITAL     = [0, 1, 17, 18]
PAPER_SINGLE_VITAL = [9, 34, 35, 37, 6]


def plot(scores_path: str, out_path: str, threshold: float):
    with open(scores_path) as f:
        data = json.load(f)

    mm_means = [float(np.mean(data["mm"][str(i)])) for i in range(19)]
    mm_std   = [float(np.std(data["mm"][str(i)]))  for i in range(19)]

    sin_means = [float(np.mean(data["single"][str(i)])) for i in range(38)]
    sin_std   = [float(np.std(data["single"][str(i)])) for i in range(38)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    def _bar_colors(means, vital_idx):
        return ["#d62728" if s < threshold else "#1f77b4" for s in means]

    def _add_vital_labels(ax, vital_idx, means):
        for i in vital_idx:
            ax.text(i, means[i] - 0.008, "★", ha="center", va="top",
                    fontsize=9, color="white", fontweight="bold")

    # --- MM-DiT blocks ---
    x1 = np.arange(19)
    bars1 = ax1.bar(x1, mm_means, color=_bar_colors(mm_means, PAPER_MM_VITAL),
                    yerr=mm_std, capsize=3, width=0.7, edgecolor="black", linewidth=0.5)
    ax1.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                label=f"Vitality threshold ({threshold})")
    _add_vital_labels(ax1, PAPER_MM_VITAL, mm_means)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([str(i) for i in range(19)], fontsize=8)
    ax1.set_xlabel("Block index (MM-DiT)", fontsize=11)
    ax1.set_ylabel("DINOv2 cosine similarity when bypassed\n(↑ = less vital)", fontsize=10)
    ax1.set_title("Double-stream (MM-DiT) blocks — Layer Vitality", fontsize=13)
    ax1.set_ylim(max(0, min(mm_means) - 0.05), 1.02)
    ax1.legend(fontsize=9)
    vital_patch   = mpatches.Patch(color="#d62728", label="Vital (sim < threshold)")
    trivial_patch = mpatches.Patch(color="#1f77b4", label="Non-vital")
    ax1.legend(handles=[vital_patch, trivial_patch,
                         plt.Line2D([0], [0], color="red", linestyle="--",
                                    label=f"Threshold ({threshold})")],
               loc="lower right", fontsize=9)

    # --- Single-stream blocks ---
    x2 = np.arange(38)
    bars2 = ax2.bar(x2, sin_means, color=_bar_colors(sin_means, PAPER_SINGLE_VITAL),
                    yerr=sin_std, capsize=2, width=0.7, edgecolor="black", linewidth=0.5)
    ax2.axhline(threshold, color="red", linestyle="--", linewidth=1.2)
    _add_vital_labels(ax2, PAPER_SINGLE_VITAL, sin_means)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(i) for i in range(38)], fontsize=7)
    ax2.set_xlabel("Block index (single-stream)", fontsize=11)
    ax2.set_ylabel("DINOv2 cosine similarity when bypassed\n(↑ = less vital)", fontsize=10)
    ax2.set_title("Single-stream blocks — Layer Vitality", fontsize=13)
    ax2.set_ylim(max(0, min(sin_means) - 0.05), 1.02)
    ax2.legend(handles=[vital_patch, trivial_patch,
                         plt.Line2D([0], [0], color="red", linestyle="--",
                                    label=f"Threshold ({threshold})")],
               loc="lower right", fontsize=9)

    # --- Annotate paper's vital layers ---
    for ax, vital_idx, means in [(ax1, PAPER_MM_VITAL, mm_means),
                                  (ax2, PAPER_SINGLE_VITAL, sin_means)]:
        for i in vital_idx:
            ax.annotate(f"L{i}", xy=(i, means[i]),
                        xytext=(i, means[i] + 0.015),
                        ha="center", fontsize=7, color="#d62728",
                        arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))

    plt.suptitle("FLUX Layer Vitality — DINOv2 Perceptual Similarity When Block Bypassed",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # --- Print summary ---
    print("\n--- MM-DiT vitality summary ---")
    for i, m in enumerate(mm_means):
        marker = " ← VITAL (paper)" if i in PAPER_MM_VITAL else ""
        flag   = " [below threshold]" if m < threshold else ""
        print(f"  MM-{i:2d}: {m:.4f}{flag}{marker}")

    print("\n--- Single-stream vitality summary ---")
    for i, m in enumerate(sin_means):
        marker = " ← VITAL (paper)" if i in PAPER_SINGLE_VITAL else ""
        flag   = " [below threshold]" if m < threshold else ""
        print(f"  S-{i:2d}: {m:.4f}{flag}{marker}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores",    type=str, default="results/vitality_scores.json")
    parser.add_argument("--out",       type=str, default="results/vitality_plot.png")
    parser.add_argument("--threshold", type=float, default=0.92)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.scores, args.out, args.threshold)
