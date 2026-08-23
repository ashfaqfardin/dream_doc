"""
Phase 1 — Latent Delta Composition (Intuition Experiment)

Tests whether FLUX's VAE latent space supports post-hoc object composition
via simple arithmetic:

    delta_i = vae_encode(img_after_edit_i) - vae_encode(img_before_edit_i)
    composed = vae_decode(z_base + delta_1 + delta_2 + ...)

Motivation
----------
K/V injection (Phase 5) fails to ADD objects — injecting cached background
K/V pulls the model back to the background, overriding the edit prompt.
This file tests a completely different approach: let Kontext edit freely, then
compose at the VAE latent level instead of interfering with denoising.

Research backing
----------------
- FLUX's continuous AE has an approximately linear latent space (more so than
  SD's KL-VAE), consistent with Concept Algebra / Concept Sliders literature.
- z(S2) - z(S1) isolates the per-pixel latent change caused by the edit, which
  for objects on stable backgrounds approximates an "object latent".
- Multi-delta accumulation (z0 + Σdeltai) mirrors compositional latent editing
  (Composable Diffusion, StructDiffusion).

Experiments
-----------
Exp A — Baseline: pure chainwise Kontext editing (same as phase1_baseline.py)
Exp B — Delta sanity: decode(z0 + Σdeltai), compare with baseline (tests linearity)
Exp C — Cumulative delta: like B but shown step by step with per-step delta norms
Exp D — Transfer: independently generate S1 and S2, transfer S2-S1 delta to S1

Usage
-----
python NewWork/KontextEval/phase1_intuition_unverified.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_intuition
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline, generate


# ============================================================
# Prompts (identical to phase1_baseline.py)
# ============================================================

PROMPTS = [
    {
        "name": "add_wooden_chair",
        "label": "Add Wooden Chair",
        "text": (
            "Add a wooden chair in the center of the room. "
            "Preserve the existing background and composition."
        ),
    },
    {
        "name": "replace_iron_chair",
        "label": "Replace → Iron Chair",
        "text": (
            "Replace the wooden chair with a modern iron chair. "
            "Keep the chair position, size, and background unchanged."
        ),
    },
    {
        "name": "change_color_red",
        "label": "Color → Red",
        "text": (
            "Change the iron chair color to red. "
            "Preserve the chair shape, position, and surrounding scene."
        ),
    },
    {
        "name": "style_oil_painting",
        "label": "Style → Oil Painting",
        "text": (
            "Transform the entire image into an oil painting style. "
            "Preserve the chair identity and scene composition."
        ),
    },
]


# ============================================================
# VAE helpers
# ============================================================

@torch.no_grad()
def vae_encode(pipe, pil_img: Image.Image) -> torch.Tensor:
    """Encode a PIL image to a normalized latent tensor (1, 16, H/8, W/8) on CPU."""
    t = pipe.image_processor.preprocess(pil_img).to(pipe.device, pipe.vae.dtype)
    raw = pipe.vae.encode(t).latent_dist.mean
    z = (raw - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return z.cpu()


@torch.no_grad()
def vae_decode(pipe, z: torch.Tensor) -> Image.Image:
    """Decode a normalized latent tensor (1, 16, H/8, W/8) to a PIL image."""
    z_dev = (z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor)
    z_dev = z_dev.to(pipe.device, pipe.vae.dtype)
    raw = pipe.vae.decode(z_dev).sample
    return pipe.image_processor.postprocess(raw, output_type="pil")[0]


def delta_norm(delta: torch.Tensor) -> float:
    """L2 norm of a delta latent, averaged per element."""
    return delta.float().norm().item() / delta.numel() ** 0.5


# ============================================================
# Side-by-side comparison save helper
# ============================================================

def save_compare(images, titles, save_path: str, ncols: int | None = None):
    n = len(images)
    if ncols is None:
        ncols = n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = [axes] if n == 1 else list(axes.flat)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",  required=True)
    p.add_argument("--cache_dir", default="./models")
    p.add_argument("--out_dir",   default="results/phase1_intuition")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--guidance",  type=float, default=2.5)
    p.add_argument("--height",    type=int, default=1024)
    p.add_argument("--width",     type=int, default=1024)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def save(img: Image.Image, name: str) -> str:
        path = os.path.join(args.out_dir, f"{name}.png")
        img.save(path)
        return path

    def run(img, prompt_text):
        return generate(
            pipe, prompt_text, img,
            seed=args.seed, num_steps=args.num_steps,
            guidance_scale=args.guidance,
            height=args.height, width=args.width,
        )

    print("Loading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    # ----------------------------------------------------------
    # Step 0: base image
    # ----------------------------------------------------------
    base_image = Image.new("RGB", (args.width, args.height), color=(120, 120, 120))
    save(base_image, "step0_base")
    z0 = vae_encode(pipe, base_image)
    print(f"  z0 shape: {tuple(z0.shape)}  dtype: {z0.dtype}")

    # ----------------------------------------------------------
    # Experiment A: baseline chainwise (same as phase1_baseline.py)
    # ----------------------------------------------------------
    print("\n=== Experiment A — Baseline chainwise ===")
    baseline_imgs = [base_image]
    current = base_image
    for i, p in enumerate(PROMPTS, 1):
        print(f"  Step {i}: {p['label']}")
        out = run(current, p["text"])
        save(out, f"step{i}_baseline")
        baseline_imgs.append(out)
        current = out
    print("  Done. Saved step1..4_baseline.png")

    # ----------------------------------------------------------
    # Experiment B: delta reconstruction sanity check
    #   Encode all images, compute deltas, reconstruct from z0 + Σdelta
    # ----------------------------------------------------------
    print("\n=== Experiment B — Delta reconstruction sanity check ===")

    # Encode all baseline images
    z_baseline = [z0] + [vae_encode(pipe, img) for img in baseline_imgs[1:]]
    deltas = [z_baseline[i] - z_baseline[i - 1] for i in range(1, len(z_baseline))]

    delta_log_lines = ["step | name                  | delta_norm"]
    delta_log_lines.append("-" * 50)

    recon_imgs = [base_image]
    z_accum = z0.clone()
    for i, (d, p) in enumerate(zip(deltas, PROMPTS), 1):
        norm = delta_norm(d)
        delta_log_lines.append(f"  {i}  | {p['name']:22s} | {norm:.5f}")
        z_accum = z_accum + d
        recon = vae_decode(pipe, z_accum)
        save(recon, f"step{i}_delta_recon")
        recon_imgs.append(recon)
        print(f"  Step {i} ({p['label']}): delta_norm={norm:.5f}")

    # Save delta norms
    delta_log_path = os.path.join(args.out_dir, "delta_norms.txt")
    with open(delta_log_path, "w") as f:
        f.write("\n".join(delta_log_lines) + "\n")
        f.write("\nInterpretation:\n")
        f.write("  High norm → large latent change (big edit or artifact)\n")
        f.write("  Low norm  → small latent change (minor tweak)\n")
        f.write("  Style transfer (step 4) should produce the highest norm\n")
    print(f"  Delta norms saved → {delta_log_path}")

    # ----------------------------------------------------------
    # Experiment C: step-by-step comparison grids
    # ----------------------------------------------------------
    print("\n=== Experiment C — Step comparison grids ===")
    for i, p in enumerate(PROMPTS, 1):
        baseline = baseline_imgs[i]
        recon = recon_imgs[i]
        save_compare(
            [baseline_imgs[i - 1], baseline, recon],
            [f"Step {i-1} (input)", f"Step {i} baseline\n{p['label']}", f"Step {i} delta recon\ndecode(z0+Σδ)"],
            save_path=os.path.join(args.out_dir, f"step{i}_compare.png"),
        )
        print(f"  Saved step{i}_compare.png")

    # Full 5-column comparison: baseline chain vs delta recon chain
    save_compare(
        baseline_imgs + recon_imgs[1:],
        [f"Base"] + [f"Baseline\nStep {i}" for i in range(1, 5)] +
        [f"DeltaRecon\nStep {i}" for i in range(1, 5)],
        save_path=os.path.join(args.out_dir, "full_comparison.png"),
        ncols=5,
    )
    print("  Saved full_comparison.png (2 rows × 5 cols: baseline vs delta recon)")

    # ----------------------------------------------------------
    # Experiment D: independent generation + delta transfer
    #   Generate S1 and S2 independently (both from base), transfer delta
    # ----------------------------------------------------------
    print("\n=== Experiment D — Independent generation + delta transfer ===")
    print("  Generating S1 (Step 1) independently from base ...")
    s1_ind = run(base_image, PROMPTS[0]["text"])  # add wooden chair
    save(s1_ind, "expD_s1_independent")

    print("  Generating S2 (Step 2) independently from S1 ...")
    s2_ind = run(s1_ind, PROMPTS[1]["text"])      # replace → iron chair
    save(s2_ind, "expD_s2_independent")

    # Delta from independent run
    z_s1_ind = vae_encode(pipe, s1_ind)
    z_s2_ind = vae_encode(pipe, s2_ind)
    delta_ind = z_s2_ind - z_s1_ind

    # Transfer: apply the (S2-S1) delta from independent run onto S1 from baseline chain
    s1_baseline_z = z_baseline[1]
    z_transfer = s1_baseline_z + delta_ind
    transfer_img = vae_decode(pipe, z_transfer)
    save(transfer_img, "expD_transfer_result")

    save_compare(
        [s1_ind, s2_ind, baseline_imgs[1], transfer_img],
        ["S1 independent\n(wooden chair)",
         "S2 independent\n(iron chair)",
         "S1 baseline chain\n(wooden chair)",
         "Transfer result\nS1_baseline + (S2_ind - S1_ind)"],
        save_path=os.path.join(args.out_dir, "expD_transfer_compare.png"),
    )
    print("  Saved expD_transfer_compare.png")
    print("  KEY QUESTION: Does transfer_result show the iron chair change")
    print("  applied to the baseline S1 scene? If yes → delta transfer works!")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Output directory : {args.out_dir}/")
    print()
    print("Exp A (Baseline):")
    for i in range(1, 5):
        print(f"  step{i}_baseline.png")
    print()
    print("Exp B (Delta reconstruction — are the deltas faithful?):")
    for i in range(1, 5):
        print(f"  step{i}_delta_recon.png")
    print("  delta_norms.txt")
    print()
    print("Exp C (Comparison grids — baseline vs delta recon side by side):")
    for i in range(1, 5):
        print(f"  step{i}_compare.png")
    print("  full_comparison.png")
    print()
    print("Exp D (Independent delta transfer):")
    print("  expD_s1_independent.png")
    print("  expD_s2_independent.png")
    print("  expD_transfer_result.png")
    print("  expD_transfer_compare.png")
    print()
    print("WHAT TO CHECK:")
    print("  1. Do step1..4_delta_recon.png look similar to step1..4_baseline.png?")
    print("     → YES: VAE arithmetic is self-consistent. Delta approach is valid.")
    print("     → NO:  VAE is too non-linear. Approach needs rethinking.")
    print()
    print("  2. Do delta recon images have fewer artifacts at steps 3-4?")
    print("     → YES: Delta composition reduces accumulated errors. Useful!")
    print("     → SAME: Arithmetic consistent but no quality improvement.")
    print()
    print("  3. Does style transfer (step 4) delta recon look correct?")
    print("     → Expected: style IS applied (delta captures global change)")
    print("     → Possible: smearing/blending artifacts for global transforms")
    print()
    print("  4. Does expD_transfer_result show iron chair applied to baseline scene?")
    print("     → YES: Delta is transferable across similar contexts.")
    print("     → NO:  Delta is context-specific, doesn't transfer.")


if __name__ == "__main__":
    main()
