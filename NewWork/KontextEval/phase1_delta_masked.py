"""
Phase 1 — Masked Latent Spatial Compositing

Previous finding (phase1_delta_refine.py):
  - 2-stage pipeline (compose + refine) is viable but vase appears transparent.
  - Root cause: additive delta composition dilutes objects.
    z_base + δ_bike + δ_vase
    = background × 3 (residual in each delta) + objects × 1
    → objects appear weak/transparent when refined.

This file fixes that with spatial latent copy instead of delta addition:

  z_composed = z_base                             (start with clean background)
  z_composed[mask_bike] = z_bike[mask_bike]       (copy FULL bicycle region)
  z_composed[mask_vase] = z_vase[mask_vase]       (copy FULL vase region)

Where masks come from the delta magnitude:
  δ_bike = z_bike − z_base
  mask_bike = per-spatial |δ_bike| > threshold    (where bicycle changed)

Objects get their FULL latent quality (no dilution), background stays clean.
Then a single Kontext refinement pass smooths the hard region boundaries.

Experiments
-----------
  A. Spatial copy, mask threshold sweep: 50th / 70th / 90th percentile
  B. Soft blending (smooth mask) vs hard copy
  C. With and without refinement pass (10 steps, guidance 3.5)
  D. Final comparison: chain baseline vs masked-copy + refine

Usage
-----
python NewWork/KontextEval/phase1_delta_masked.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_masked
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline, generate


# ============================================================
# Prompts
# ============================================================

BASE_PROMPT   = "A modern living room with a sofa and a wooden coffee table."
BICYCLE_PROMPT = (
    "Add a yellow bicycle leaning against the wall on the left side. "
    "Keep the rest of the room exactly the same."
)
VASE_PROMPT   = (
    "Add a white ceramic vase with flowers on the coffee table. "
    "Keep the rest of the room exactly the same."
)
REFINE_PROMPT = (
    "A modern living room with a sofa, a wooden coffee table, a yellow bicycle "
    "leaning against the left wall, and a solid white ceramic vase with vivid "
    "flowers placed on the coffee table. Photorealistic, sharp, high quality."
)


# ============================================================
# VAE helpers
# ============================================================

@torch.no_grad()
def vae_encode(pipe, pil_img: Image.Image) -> torch.Tensor:
    t = pipe.image_processor.preprocess(pil_img).to(pipe.device, pipe.vae.dtype)
    raw = pipe.vae.encode(t).latent_dist.mean
    return ((raw - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor).cpu()


@torch.no_grad()
def vae_decode(pipe, z: torch.Tensor) -> Image.Image:
    z_dev = (z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor)
    z_dev = z_dev.to(pipe.device, pipe.vae.dtype)
    raw = pipe.vae.decode(z_dev).sample
    return pipe.image_processor.postprocess(raw, output_type="pil")[0]


# ============================================================
# Mask computation
# ============================================================

def compute_mask(delta: torch.Tensor, percentile: float = 70.0,
                 soft: bool = False, blur_kernel: int = 5) -> torch.Tensor:
    """
    Compute a spatial mask from a latent delta.

    delta   : (1, C, H, W)  — per-channel latent difference
    Returns : (1, 1, H, W)  — float mask in [0, 1]

    Hard mask: binary 0/1 based on percentile threshold.
    Soft mask: smooth Gaussian-blurred version for feathered blending.
    """
    mag = delta.float().abs().mean(dim=1, keepdim=True)  # (1, 1, H, W)
    flat = mag.reshape(-1)
    thresh = torch.quantile(flat, percentile / 100.0).item()
    binary = (mag >= thresh).float()

    if not soft:
        return binary

    # Gaussian blur for soft edges
    k = blur_kernel
    if k > 1 and k % 2 == 0:
        k += 1
    sigma = k / 3.0
    coords = torch.arange(k, dtype=torch.float32) - k // 2
    gauss_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = gauss_1d[:, None] @ gauss_1d[None, :]
    kernel = kernel[None, None]
    blurred = F.conv2d(binary, kernel, padding=k // 2)
    return blurred.clamp(0.0, 1.0)


def compose_spatial(z_base: torch.Tensor,
                    layers: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """
    Spatial latent copy composition.

    layers: list of (z_src, mask) pairs applied in order.
            mask: (1, 1, H, W) float in [0, 1]
            For each layer: z_out = z_out * (1 - mask) + z_src * mask
    """
    z = z_base.clone()
    for z_src, mask in layers:
        z = z * (1.0 - mask) + z_src * mask
    return z


def delta_to_vis(delta: torch.Tensor) -> Image.Image:
    """Visualise the per-spatial magnitude of a delta as a heatmap."""
    mag = delta.float().abs().mean(dim=1).squeeze(0).numpy()  # (H, W)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    cmap = plt.get_cmap("hot")
    arr = (cmap(mag)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ============================================================
# Grid helper
# ============================================================

def save_grid(images, titles, path, ncols=None):
    n = len(images)
    ncols = ncols or n
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
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",          required=True)
    p.add_argument("--cache_dir",         default="./models")
    p.add_argument("--out_dir",           default="results/phase1_masked")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--num_steps",         type=int,   default=28)
    p.add_argument("--guidance",          type=float, default=2.5)
    p.add_argument("--refine_steps",      type=int,   default=10)
    p.add_argument("--refine_guidance",   type=float, default=3.5)
    p.add_argument("--height",            type=int,   default=1024)
    p.add_argument("--width",             type=int,   default=1024)
    p.add_argument("--device",            default="cuda")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def save(img, name):
        path = os.path.join(args.out_dir, f"{name}.png")
        img.save(path)
        return path

    def run(src, prompt, steps=None, guidance=None, seed=None):
        return generate(
            pipe, prompt, src,
            seed=seed if seed is not None else args.seed,
            num_steps=steps if steps is not None else args.num_steps,
            guidance_scale=guidance if guidance is not None else args.guidance,
            height=args.height, width=args.width,
        )

    print("Loading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    # ----------------------------------------------------------
    # Shared: base + clean object pairs + deltas
    # ----------------------------------------------------------
    print("\n=== Setup: base + clean pairs ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    print("  Base scene …")
    base = run(grey, BASE_PROMPT)
    save(base, "base")
    z_base = vae_encode(pipe, base)

    print("  Clean bicycle …")
    img_bike = run(base, BICYCLE_PROMPT)
    save(img_bike, "clean_bicycle")
    z_bike = vae_encode(pipe, img_bike)
    delta_bike = z_bike - z_base

    print("  Clean vase …")
    img_vase = run(base, VASE_PROMPT)
    save(img_vase, "clean_vase")
    z_vase = vae_encode(pipe, img_vase)
    delta_vase = z_vase - z_base

    # Visualise deltas
    save(delta_to_vis(delta_bike), "delta_bike_heatmap")
    save(delta_to_vis(delta_vase), "delta_vase_heatmap")
    print("  Delta heatmaps saved (bright = where the object changed most)")

    # ----------------------------------------------------------
    # Chain baseline (2 Kontext calls) — the current best to beat
    # ----------------------------------------------------------
    print("\n=== Chain baseline ===")
    chain_bike = run(base, BICYCLE_PROMPT)
    save(chain_bike, "chain_step1_bicycle")
    chain_both = run(chain_bike, VASE_PROMPT)
    save(chain_both, "chain_step2_both")

    # ----------------------------------------------------------
    # Experiment A: Mask threshold sweep
    # ----------------------------------------------------------
    print("\n=== Experiment A: Mask threshold sweep (hard mask) ===")
    threshold_percentiles = [50, 60, 70, 80, 90]
    sweep_raw = []   # composed before refine
    sweep_refined = []

    for pct in threshold_percentiles:
        print(f"  Threshold {pct}th percentile …")
        mask_b = compute_mask(delta_bike, percentile=pct, soft=False)
        mask_v = compute_mask(delta_vase, percentile=pct, soft=False)

        z_comp = compose_spatial(z_base, [(z_bike, mask_b), (z_vase, mask_v)])
        img_comp = vae_decode(pipe, z_comp)
        save(img_comp, f"expA_hard_{pct}pct_raw")
        sweep_raw.append(img_comp)

        img_ref = run(img_comp, REFINE_PROMPT,
                      steps=args.refine_steps, guidance=args.refine_guidance)
        save(img_ref, f"expA_hard_{pct}pct_refined")
        sweep_refined.append(img_ref)

    save_grid(
        sweep_raw + sweep_refined,
        [f"RAW\n{p}th pct" for p in threshold_percentiles] +
        [f"REFINED\n{p}th pct" for p in threshold_percentiles],
        os.path.join(args.out_dir, "expA_threshold_sweep.png"),
        ncols=len(threshold_percentiles),
    )
    print("  Saved: expA_threshold_sweep.png")
    print("  KEY: Low threshold (50%) = larger mask, more pixels copied")
    print("       High threshold (90%) = tight mask, only object core copied")
    print("       Find the threshold where objects look solid and background is clean")

    # ----------------------------------------------------------
    # Experiment B: Soft (feathered) mask vs hard mask
    # ----------------------------------------------------------
    print("\n=== Experiment B: Soft mask (feathered edges) ===")
    best_pct = 70   # use 70th percentile as middle ground

    mask_b_hard = compute_mask(delta_bike, percentile=best_pct, soft=False)
    mask_v_hard = compute_mask(delta_vase, percentile=best_pct, soft=False)
    mask_b_soft = compute_mask(delta_bike, percentile=best_pct, soft=True, blur_kernel=11)
    mask_v_soft = compute_mask(delta_vase, percentile=best_pct, soft=True, blur_kernel=11)

    z_hard = compose_spatial(z_base, [(z_bike, mask_b_hard), (z_vase, mask_v_hard)])
    z_soft = compose_spatial(z_base, [(z_bike, mask_b_soft), (z_vase, mask_v_soft)])

    img_hard_raw = vae_decode(pipe, z_hard)
    img_soft_raw = vae_decode(pipe, z_soft)
    save(img_hard_raw, "expB_hard_raw")
    save(img_soft_raw, "expB_soft_raw")

    img_hard_ref = run(img_hard_raw, REFINE_PROMPT,
                       steps=args.refine_steps, guidance=args.refine_guidance)
    img_soft_ref = run(img_soft_raw, REFINE_PROMPT,
                       steps=args.refine_steps, guidance=args.refine_guidance)
    save(img_hard_ref, "expB_hard_refined")
    save(img_soft_ref, "expB_soft_refined")

    save_grid(
        [img_hard_raw, img_soft_raw, img_hard_ref, img_soft_ref],
        ["Hard mask\n(raw)", "Soft mask\n(raw)",
         "Hard mask\n(refined)", "Soft mask\n(refined)"],
        os.path.join(args.out_dir, "expB_hard_vs_soft.png"),
    )
    print("  Saved: expB_hard_vs_soft.png")
    print("  KEY: Soft mask feathers edges so Kontext sees smoother boundaries")
    print("       Does soft mask reduce boundary artefacts after refinement?")

    # ----------------------------------------------------------
    # Best result: pick the best-looking from A + B
    # Use 70th percentile soft mask as default candidate
    # ----------------------------------------------------------
    best_composed = img_soft_raw
    best_refined  = img_soft_ref

    # ----------------------------------------------------------
    # Final comparison: chain vs masked-copy+refine
    # ----------------------------------------------------------
    print("\n=== Final comparison ===")
    save_grid(
        [base, img_bike, img_vase,
         img_hard_raw, img_soft_raw,
         img_hard_ref, img_soft_ref,
         chain_both],
        ["Base", "Clean bicycle\n(single edit)", "Clean vase\n(single edit)",
         "Hard copy\n(raw, no refine)", "Soft copy\n(raw, no refine)",
         "Hard copy\n+ refine", "Soft copy\n+ refine",
         "Chain baseline\n(2 Kontext calls)"],
        os.path.join(args.out_dir, "FINAL_COMPARISON.png"),
        ncols=4,
    )
    print("  Saved: FINAL_COMPARISON.png  ← main result")

    # Single most important 3-panel: chain vs best-raw vs best-refined
    save_grid(
        [chain_both, best_composed, best_refined],
        ["Chain baseline\n(2 calls — the target to beat)",
         "Masked copy RAW\n(70th pct soft, no refine)",
         "Masked copy + REFINE\n(10 steps, g=3.5)"],
        os.path.join(args.out_dir, "KEY_RESULT.png"),
    )
    print("  Saved: KEY_RESULT.png  ← 3-panel: chain vs raw-copy vs refined-copy")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("WHAT TO CHECK")
    print(f"{'='*60}")
    print("""
delta_bike_heatmap.png / delta_vase_heatmap.png
  Bright = where the object changed most in the latent.
  The mask should form a recognizable shape of the object.
  If the bright region is the whole image → the edit changed everything
  → mask will be large and contaminate the background.

expA_threshold_sweep.png
  Top row (raw, no refine): which threshold gives the cleanest paste?
  Bottom row (refined): which threshold gives the best final quality?
  → Look for: object solid, background matches base, no visible boundary

expB_hard_vs_soft.png
  Hard vs soft mask before and after refinement.
  Soft mask should give smoother transitions at object edges.
  KEY: Is the vase solid (not transparent) in the soft-masked versions?

KEY_RESULT.png  ← most important
  Left:   Chain baseline (2 Kontext calls) — current best
  Middle: Raw masked copy (no refinement)
  Right:  Masked copy + 10-step Kontext refinement

  SUCCESS criteria:
  → Right panel: both objects solid, correct colors, no transparency
  → Right panel quality ≥ Left panel (chain baseline)

If the vase is STILL transparent after masked copy + refine:
  The mask is too small (vase delta is weak) — try threshold 50th pct
  Or: the refinement is too conservative — try guidance 4.0-4.5
""")
    print(f"All results → {args.out_dir}/")


if __name__ == "__main__":
    main()
