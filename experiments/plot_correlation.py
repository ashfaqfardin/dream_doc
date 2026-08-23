"""
Standalone correlation plotter — re-reads saved JSON results and regenerates
the scatter plots from Experiment 4 without re-running inference.

Usage
-----
python experiments/plot_correlation.py
    [--vitality  results/vitality_scores.json]
    [--balance   results/attention_balance.json]
    [--out       results/vitality_attention_correlation.png]
    [--threshold 0.92]
"""

import argparse
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

N_MM               = 19
PAPER_MM_VITAL     = [0, 1, 17, 18]
PAPER_SINGLE_VITAL = [9, 34, 35, 37, 6]


def plot(vitality_path, balance_path, out_path, threshold):
    with open(vitality_path) as f:
        vdata = json.load(f)
    with open(balance_path) as f:
        bdata = json.load(f)

    mm_vitality = [float(np.mean(vdata["mm"][str(i)])) for i in range(N_MM)]
    mm_entropy  = bdata["mm_entropy"]
    mm_balance  = bdata["mm_balance"]

    vital_flags = [i in PAPER_MM_VITAL for i in range(N_MM)]
    colors = ["#d62728" if v else "#1f77b4" for v in vital_flags]
    sizes  = [120 if v else 60 for v in vital_flags]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _scatter_with_corr(ax, x, y, xlabel, ylabel, title):
        for i in range(N_MM):
            if math.isnan(y[i]):
                continue
            ax.scatter(x[i], y[i], c=colors[i], s=sizes[i],
                       edgecolors="black", linewidths=0.6, zorder=3)
            ax.annotate(str(i), (x[i], y[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)

        valid = [(x[i], y[i]) for i in range(N_MM) if not math.isnan(y[i])]
        if len(valid) > 2:
            xs, ys = zip(*valid)
            r = float(np.corrcoef(xs, ys)[0, 1])
            m, b = np.polyfit(xs, ys, 1)
            xr = np.linspace(min(xs), max(xs), 100)
            ax.plot(xr, m * xr + b, color="grey", linestyle="--", linewidth=1)
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=10, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        vital_p   = mpatches.Patch(color="#d62728", label="Vital (paper)")
        trivial_p = mpatches.Patch(color="#1f77b4", label="Non-vital")
        ax.legend(handles=[vital_p, trivial_p], fontsize=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    _scatter_with_corr(
        axes[0], mm_vitality, mm_entropy,
        "Vitality (DINOv2 sim when bypassed ↑=less vital)",
        "Mean attention entropy (nats)",
        "Vitality vs Entropy",
    )

    _scatter_with_corr(
        axes[1], mm_vitality, mm_balance,
        "Vitality (DINOv2 sim when bypassed ↑=less vital)",
        "Mean text-attn fraction per image token",
        "Vitality vs Text-Image Balance",
    )

    # Entropy vs balance coloured by vitality
    _scatter_with_corr(
        axes[2], mm_entropy, mm_balance,
        "Mean attention entropy",
        "Mean text-attn fraction",
        "Entropy vs Balance (coloured by vitality)",
    )

    # Add threshold indicator
    axes[0].axvline(threshold, color="red", linestyle=":", linewidth=1,
                    label=f"Threshold {threshold}")

    plt.suptitle("Experiment 4 — Do vital layers have higher text-image coupling?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vitality",  type=str, default="results/vitality_scores.json")
    parser.add_argument("--balance",   type=str, default="results/attention_balance.json")
    parser.add_argument("--out",       type=str,
                        default="results/vitality_attention_correlation.png")
    parser.add_argument("--threshold", type=float, default=0.92)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.vitality):
        print(f"ERROR: vitality file not found: {args.vitality}")
        print("Run experiments/vitality_sweep.py first.")
    elif not os.path.exists(args.balance):
        print(f"ERROR: balance file not found: {args.balance}")
        print("Run experiments/attention_balance.py first.")
    else:
        plot(args.vitality, args.balance, args.out, args.threshold)
