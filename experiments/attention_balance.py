"""
Experiment 4 — Vital layers vs attention concentration.

Quantifies:
  1. Attention entropy per layer (lower = more concentrated)
  2. Text-vs-image modality balance per layer
  3. Correlation between vitality score and modality balance

Outputs
-------
results/attention_balance.json
results/vitality_attention_correlation.png

Usage
-----
python experiments/attention_balance.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token YOUR_TOKEN \
    --prompts_n 8 \
    [--device cuda] [--n_steps 15] [--cpu_offload]

Requires results/vitality_scores.json (from Experiment 1).
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.layer_bypass import load_pipeline
from experiments.attention_hooks import AttentionCapture

N_MM     = 19
N_SINGLE = 38

PAPER_MM_VITAL     = [0, 1, 17, 18]
PAPER_SINGLE_VITAL = [9, 34, 35, 37, 6]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def attention_entropy(attn_map: torch.Tensor) -> float:
    """
    Mean entropy of the text→image attention distribution.

    attn_map : (n_heads, n_img_tokens, n_text_tokens)
    Returns scalar (nats).
    """
    p = attn_map.mean(0)                             # (n_img, n_txt)
    p = p / (p.sum(-1, keepdim=True) + 1e-8)        # normalise rows
    ent = -(p * (p + 1e-8).log()).sum(-1)            # (n_img,)
    return float(ent.mean().item())


def text_image_balance(attn_map: torch.Tensor) -> float:
    """
    Mean fraction of attention mass that image tokens send to text tokens.

    attn_map : (n_heads, n_img_tokens, n_text_tokens) — text→image quadrant only.
    The complementary (image→image) mass is 1 - sum_of_text_attn per image token.

    NOTE: because we only stored the text columns, we approximate:
      to_text  = attn_map.sum(-1)           per image token, total text attention
      to_image = 1 - to_text                (approximation)
    """
    to_text  = attn_map.sum(-1)                      # (H, n_img) — total txt mass per img tok
    to_text  = to_text.mean(0)                        # (n_img,)
    # Clamp to [0, 1] in case of fp16 drift
    to_text  = to_text.clamp(0.0, 1.0)
    return float(to_text.mean().item())


# ---------------------------------------------------------------------------
# Compute per-layer metrics across prompts
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_layer_metrics(pipe, prompts: list, device: str, n_steps: int):
    """
    Run inference with attention capture for each prompt and accumulate
    per-layer entropy and balance metrics.

    Returns
    -------
    mm_entropy   : list[float] length 19
    mm_balance   : list[float] length 19
    (single-stream blocks are skipped — they don't have joint attention)
    """
    mm_entropy_acc  = [[] for _ in range(N_MM)]
    mm_balance_acc  = [[] for _ in range(N_MM)]

    for p_idx, prompt in enumerate(prompts):
        print(f"  Prompt {p_idx+1}/{len(prompts)}: {prompt[:60]}...")
        generator = torch.Generator(device=device).manual_seed(0)
        latents = torch.randn(
            (1, 4096, 64),
            generator=generator,
            device=device,
            dtype=pipe.transformer.dtype,
        )

        with AttentionCapture(pipe) as cap:
            _ = pipe(
                prompt,
                height=1024, width=1024,
                guidance_scale=3.5,
                output_type="latent",          # skip VAE decode to save time
                num_inference_steps=n_steps,
                max_sequence_length=512,
                latents=latents,
            )
            maps = dict(cap.maps)

        # Aggregate over timesteps
        for layer_idx in range(N_MM):
            step_maps = [v for (bt, li, _), v in maps.items()
                         if bt == "mm" and li == layer_idx]
            if not step_maps:
                continue
            # Average over timesteps
            mean_map = torch.stack(step_maps).mean(0)  # (H, n_img, n_txt)
            mm_entropy_acc[layer_idx].append(attention_entropy(mean_map))
            mm_balance_acc[layer_idx].append(text_image_balance(mean_map))

    mm_entropy = [float(np.mean(v)) if v else float("nan") for v in mm_entropy_acc]
    mm_balance = [float(np.mean(v)) if v else float("nan") for v in mm_balance_acc]
    return mm_entropy, mm_balance


# ---------------------------------------------------------------------------
# Correlation plot
# ---------------------------------------------------------------------------

def plot_correlation(vitality_scores_path: str, mm_entropy: list,
                     mm_balance: list, out_path: str):
    with open(vitality_scores_path) as f:
        vdata = json.load(f)

    mm_vitality = [float(np.mean(vdata["mm"][str(i)])) for i in range(N_MM)]

    vital_flags = [i in PAPER_MM_VITAL for i in range(N_MM)]
    colors = ["#d62728" if v else "#1f77b4" for v in vital_flags]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Vitality vs Entropy ---
    ax = axes[0]
    for i in range(N_MM):
        ax.scatter(mm_vitality[i], mm_entropy[i], c=colors[i], s=80,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(str(i), (mm_vitality[i], mm_entropy[i]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Vitality score (DINOv2 sim when bypassed)", fontsize=11)
    ax.set_ylabel("Mean attention entropy (nats)", fontsize=11)
    ax.set_title("Vitality vs Attention Entropy\n(MM-DiT blocks)", fontsize=12)
    # Correlation coefficient
    valid = [(mm_vitality[i], mm_entropy[i]) for i in range(N_MM)
             if not math.isnan(mm_entropy[i])]
    if len(valid) > 1:
        xs, ys = zip(*valid)
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    # Trend line
    if len(valid) > 2:
        m, b = np.polyfit(xs, ys, 1)
        xr = np.linspace(min(xs), max(xs), 100)
        ax.plot(xr, m * xr + b, color="grey", linestyle="--", linewidth=1)

    # --- Vitality vs Modality Balance ---
    ax = axes[1]
    for i in range(N_MM):
        ax.scatter(mm_vitality[i], mm_balance[i], c=colors[i], s=80,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(str(i), (mm_vitality[i], mm_balance[i]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Vitality score (DINOv2 sim when bypassed)", fontsize=11)
    ax.set_ylabel("Mean text-attention fraction per image token", fontsize=11)
    ax.set_title("Vitality vs Text-vs-Image Balance\n(MM-DiT blocks)", fontsize=12)

    valid2 = [(mm_vitality[i], mm_balance[i]) for i in range(N_MM)
              if not math.isnan(mm_balance[i])]
    if len(valid2) > 1:
        xs2, ys2 = zip(*valid2)
        r2 = float(np.corrcoef(xs2, ys2)[0, 1])
        ax.text(0.05, 0.95, f"r = {r2:.3f}", transform=ax.transAxes,
                fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        if len(valid2) > 2:
            m2, b2 = np.polyfit(xs2, ys2, 1)
            xr2 = np.linspace(min(xs2), max(xs2), 100)
            ax.plot(xr2, m2 * xr2 + b2, color="grey", linestyle="--", linewidth=1)

    vital_patch   = mpatches.Patch(color="#d62728", label="Vital layer (paper)")
    trivial_patch = mpatches.Patch(color="#1f77b4", label="Non-vital layer")
    for ax in axes:
        ax.legend(handles=[vital_patch, trivial_patch], fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 4 — Are vital layers vital because of high text-image coupling?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",    type=str, required=True)
    parser.add_argument("--prompts_n",   type=int, default=2,
                        help="Number of prompts to average over (default 2 for speed)")
    parser.add_argument("--vitality",    type=str, default="results/vitality_scores.json")
    parser.add_argument("--n_steps",     type=int, default=4,
                        help="Denoising steps (default 4 for speed; paper uses 15)")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    with open("prompts/vitality_prompts.json") as f:
        all_prompts = json.load(f)
    prompts = all_prompts[:args.prompts_n]

    pipe = load_pipeline(args.model_path, args.hf_token, args.device,
                         args.cpu_offload)

    print("Computing per-layer attention metrics...")
    mm_entropy, mm_balance = compute_layer_metrics(pipe, prompts, args.device, args.n_steps)

    result = {
        "mm_entropy":        mm_entropy,
        "mm_balance":        mm_balance,
        "paper_vital_mm":    PAPER_MM_VITAL,
    }
    with open("results/attention_balance.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved → results/attention_balance.json")

    print("\nMM-DiT entropy / balance summary:")
    for i in range(N_MM):
        vital = " ← vital" if i in PAPER_MM_VITAL else ""
        print(f"  MM-{i:2d}: entropy={mm_entropy[i]:.4f}  balance={mm_balance[i]:.4f}{vital}")

    if os.path.exists(args.vitality):
        plot_correlation(args.vitality, mm_entropy, mm_balance,
                         "results/vitality_attention_correlation.png")
    else:
        print(f"\nNo vitality file at {args.vitality} — skipping correlation plot.")
        print("Run experiments/vitality_sweep.py first.")


if __name__ == "__main__":
    main()
