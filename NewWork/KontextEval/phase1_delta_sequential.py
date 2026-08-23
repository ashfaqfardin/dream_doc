"""
Phase 1 — Sequential Dilated-Mask Spatial Compositing

Diagnosis from phase1_delta_masked.py:
  - Soft mask: full bike coverage but TRANSPARENT
      The blend zone (mask=0.5) produces:
        z_comp[x] = 0.5 * z_base[x] + 0.5 * z_bike[x]
      Decodes as half-intensity / transparent bike.
  - Hard mask 70/80/90%: not transparent but bike is CLIPPED
      Lower pixels of the bike have weaker delta → fall below top-30%
      threshold → mask excludes the bottom of the frame/wheels.
  - Soft refined: Kontext sees a ghosted bike and repositions it freely.

Root cause: we need soft-mask COVERAGE (full object region) but
hard-copy MODE (no fractional blending, no transparency).

Fix — morphological dilation:
  1. Compute hard binary mask at low percentile (covers more object pixels)
  2. Expand it via max_pool2d (dilation radius = R pixels)
  3. Hard copy z_src into z_comp only where dilated mask == 1
  Result: full bicycle covered, zero transparency, seam only at boundary.

Additional fix — sequential composition:
  Instead of pasting both objects onto z_base (which inherits no quality),
  start the composition FROM z_bike (the clean bicycle image):
    z_comp = z_bike.clone()             ← bicycle fully present, full quality
    z_comp[mask_vase] = z_vase[...]     ← paste vase region hard copy
  The bicycle never needs masking at all — it's already in z_bike.

Pipeline
--------
  1. Generate base + clean bicycle + clean vase (3 Kontext calls)
  2. Compute delta_vase = z_vase − z_base  (need vase mask only)
  3. Binary mask at low pct (50%) → dilate by R pixels
  4. z_comp = z_bike (bicycle at full quality)
  5. z_comp[dilated_mask_vase] = z_vase[dilated_mask_vase]  (hard copy)
  6. decode → Kontext refine (10 steps, g=3.5)
  7. Compare to chain baseline

Experiments
-----------
  A. Dilation radius sweep: R = 0, 4, 8, 12 pixels  (at pct=50, pct=60)
  B. Threshold sweep at best R: pct = 40, 50, 60, 70
  C. Hard copy vs blended copy at best R (confirm dilation fixes transparency)
  D. Final comparison: chain baseline vs dilated-masked + refine

Usage
-----
python NewWork/KontextEval/phase1_delta_sequential.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_sequential
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

BASE_PROMPT = "A modern living room with a sofa and a wooden coffee table."
BICYCLE_PROMPT = (
    "Add a yellow bicycle leaning against the wall on the left side. "
    "Keep the rest of the room exactly the same."
)
VASE_PROMPT = (
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
# Mask helpers
# ============================================================

def compute_hard_mask(delta: torch.Tensor, percentile: float = 50.0) -> torch.Tensor:
    """Binary mask: (1, 1, H, W) float 0/1 — 1 where delta is strongest."""
    mag = delta.float().abs().mean(dim=1, keepdim=True)
    thresh = torch.quantile(mag.reshape(-1), percentile / 100.0).item()
    return (mag >= thresh).float()


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Morphological dilation of a binary mask via max_pool2d.
    radius = 0 → no dilation (returns original)
    radius = R → each 1-pixel expands R pixels in all directions
    """
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius).clamp(0, 1)


def mask_to_vis(mask: torch.Tensor) -> Image.Image:
    """Visualise a (1, 1, H, W) binary/float mask as a greyscale PIL image."""
    arr = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def delta_to_vis(delta: torch.Tensor) -> Image.Image:
    """Heatmap of per-spatial delta magnitude."""
    mag = delta.float().abs().mean(dim=1).squeeze(0).numpy()
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    arr = (plt.get_cmap("hot")(mag)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ============================================================
# Composition
# ============================================================

def hard_paste(z_dst: torch.Tensor, z_src: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    Hard copy: where mask==1, replace z_dst with z_src.
    mask: (1, 1, H, W) binary float.
    Returns new tensor (does not modify z_dst in place).
    """
    return z_dst * (1.0 - mask) + z_src * mask


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
    p.add_argument("--out_dir",           default="results/phase1_sequential")
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
        img.save(os.path.join(args.out_dir, f"{name}.png"))

    def run(src, prompt, steps=None, guidance=None):
        return generate(
            pipe, prompt, src,
            seed=args.seed,
            num_steps=steps if steps is not None else args.num_steps,
            guidance_scale=guidance if guidance is not None else args.guidance,
            height=args.height, width=args.width,
        )

    print("Loading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    # ----------------------------------------------------------
    # Setup: base + clean object pairs
    # ----------------------------------------------------------
    print("\n=== Setup: base + clean pairs ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    print("  Base …")
    base = run(grey, BASE_PROMPT)
    save(base, "base")
    z_base = vae_encode(pipe, base)

    print("  Bicycle …")
    img_bike = run(base, BICYCLE_PROMPT)
    save(img_bike, "clean_bicycle")
    z_bike = vae_encode(pipe, img_bike)
    delta_bike = z_bike - z_base

    print("  Vase …")
    img_vase = run(base, VASE_PROMPT)
    save(img_vase, "clean_vase")
    z_vase = vae_encode(pipe, img_vase)
    delta_vase = z_vase - z_base

    save(delta_to_vis(delta_bike), "delta_bike_heatmap")
    save(delta_to_vis(delta_vase), "delta_vase_heatmap")
    print("  Delta heatmaps saved")

    # ----------------------------------------------------------
    # Chain baseline (current best to beat)
    # ----------------------------------------------------------
    print("\n=== Chain baseline ===")
    chain_bike = run(base, BICYCLE_PROMPT)
    save(chain_bike, "chain_step1_bicycle")
    chain_both = run(chain_bike, VASE_PROMPT)
    save(chain_both, "chain_step2_both")
    print("  Done")

    # ----------------------------------------------------------
    # Experiment A: Dilation radius sweep
    #   Fixed pct=50 (large mask), vary dilation radius
    #   Start composition from z_bike (bicycle already present)
    #   Only need to paste the vase mask on top
    # ----------------------------------------------------------
    print("\n=== Experiment A: Dilation radius sweep (pct=50) ===")
    PCT_A = 50
    radii = [0, 4, 8, 12]
    a_raw, a_refined, a_masks = [], [], []

    mask_v_base = compute_hard_mask(delta_vase, percentile=PCT_A)

    for R in radii:
        print(f"  Radius R={R} …")
        mask_v = dilate_mask(mask_v_base, radius=R)
        a_masks.append(mask_to_vis(mask_v))

        # Sequential composition: start from z_bike, paste vase
        z_comp = hard_paste(z_bike, z_vase, mask_v)
        img_comp = vae_decode(pipe, z_comp)
        save(img_comp, f"expA_R{R}_raw")
        a_raw.append(img_comp)

        img_ref = run(img_comp, REFINE_PROMPT,
                      steps=args.refine_steps, guidance=args.refine_guidance)
        save(img_ref, f"expA_R{R}_refined")
        a_refined.append(img_ref)

    save_grid(
        a_masks + a_raw + a_refined,
        [f"Mask\nR={R}" for R in radii] +
        [f"RAW\nR={R}" for R in radii] +
        [f"REFINED\nR={R}" for R in radii],
        os.path.join(args.out_dir, "expA_dilation_sweep.png"),
        ncols=len(radii),
    )
    print("  Saved: expA_dilation_sweep.png")
    print("  KEY: Row 1 = mask shape. Row 2 = raw paste (bike always present).")
    print("       Row 3 = after Kontext refine.")
    print("       Is the vase solid? Is the bicycle complete?")
    print("       Larger R = more of the vase region copied = more coverage but")
    print("       also more background from z_vase (slight color mismatch possible)")

    # ----------------------------------------------------------
    # Experiment B: Threshold sweep at best dilation radius
    #   Try R=8 (middle ground from A) with pct 40/50/60/70
    # ----------------------------------------------------------
    print("\n=== Experiment B: Threshold sweep at R=8 ===")
    R_B = 8
    pcts = [40, 50, 60, 70]
    b_raw, b_refined, b_masks = [], [], []

    for pct in pcts:
        print(f"  pct={pct} …")
        mask_v = dilate_mask(compute_hard_mask(delta_vase, percentile=pct), radius=R_B)
        b_masks.append(mask_to_vis(mask_v))

        z_comp = hard_paste(z_bike, z_vase, mask_v)
        img_comp = vae_decode(pipe, z_comp)
        save(img_comp, f"expB_pct{pct}_raw")
        b_raw.append(img_comp)

        img_ref = run(img_comp, REFINE_PROMPT,
                      steps=args.refine_steps, guidance=args.refine_guidance)
        save(img_ref, f"expB_pct{pct}_refined")
        b_refined.append(img_ref)

    save_grid(
        b_masks + b_raw + b_refined,
        [f"Mask\npct={p}" for p in pcts] +
        [f"RAW\npct={p}" for p in pcts] +
        [f"REFINED\npct={p}" for p in pcts],
        os.path.join(args.out_dir, "expB_threshold_sweep.png"),
        ncols=len(pcts),
    )
    print("  Saved: expB_threshold_sweep.png")
    print("  KEY: Lower pct = larger initial mask (before dilation).")
    print("       Find the threshold where vase is solid and bike fully visible.")

    # ----------------------------------------------------------
    # Experiment C: Confirm dilation fixes transparency
    #   Compare original soft mask vs dilated hard mask side by side
    #   (No refinement — raw decode only, to isolate the transparency fix)
    # ----------------------------------------------------------
    print("\n=== Experiment C: Soft (old) vs Dilated-hard (new) — transparency check ===")

    # Old approach: Gaussian soft mask at 70% (what failed)
    mag_b = delta_bike.float().abs().mean(dim=1, keepdim=True)
    mag_v = delta_vase.float().abs().mean(dim=1, keepdim=True)
    thresh_b = torch.quantile(mag_b.reshape(-1), 0.70).item()
    thresh_v = torch.quantile(mag_v.reshape(-1), 0.70).item()
    binary_b = (mag_b >= thresh_b).float()
    binary_v = (mag_v >= thresh_v).float()

    k = 11
    sigma = k / 3.0
    coords = torch.arange(k, dtype=torch.float32) - k // 2
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kernel = (g1d[:, None] @ g1d[None, :])[None, None]
    soft_b = F.conv2d(binary_b, kernel, padding=k // 2).clamp(0, 1)
    soft_v = F.conv2d(binary_v, kernel, padding=k // 2).clamp(0, 1)

    z_old = z_base.clone()
    z_old = hard_paste(z_old, z_bike, soft_b)
    z_old = hard_paste(z_old, z_vase, soft_v)
    img_old_raw = vae_decode(pipe, z_old)
    save(img_old_raw, "expC_old_soft_raw")

    # New approach: dilated hard mask, sequential from z_bike
    mask_v_new = dilate_mask(compute_hard_mask(delta_vase, percentile=50), radius=8)
    z_new = hard_paste(z_bike, z_vase, mask_v_new)
    img_new_raw = vae_decode(pipe, z_new)
    save(img_new_raw, "expC_new_dilated_raw")

    # Refine new
    img_new_ref = run(img_new_raw, REFINE_PROMPT,
                      steps=args.refine_steps, guidance=args.refine_guidance)
    save(img_new_ref, "expC_new_dilated_refined")

    save_grid(
        [img_old_raw, img_new_raw, img_new_ref, chain_both],
        ["OLD: soft blend\n(transparent, positions confused)",
         "NEW: dilated hard\n(sequential from z_bike, raw)",
         "NEW: dilated hard\n+ Kontext refine",
         "Chain baseline\n(2 calls — target)"],
        os.path.join(args.out_dir, "expC_transparency_fix.png"),
    )
    print("  Saved: expC_transparency_fix.png")
    print("  KEY: Is panel 2 (new raw) bike solid and non-transparent?")
    print("       Is panel 3 (refined) ≥ panel 4 (chain baseline) quality?")

    # ----------------------------------------------------------
    # Experiment D: Additional refinement guidance sweep
    #   (Use best config from A+B; try g=3.0/3.5/4.0/4.5)
    # ----------------------------------------------------------
    print("\n=== Experiment D: Refinement guidance sweep (best config) ===")
    mask_v_best = dilate_mask(compute_hard_mask(delta_vase, percentile=50), radius=8)
    z_best = hard_paste(z_bike, z_vase, mask_v_best)
    img_best_raw = vae_decode(pipe, z_best)
    save(img_best_raw, "expD_best_raw")

    guidances = [2.5, 3.0, 3.5, 4.0, 4.5]
    d_imgs = []
    for g in guidances:
        print(f"  guidance={g} …")
        img_g = run(img_best_raw, REFINE_PROMPT,
                    steps=args.refine_steps, guidance=g)
        save(img_g, f"expD_g{g:.1f}_refined")
        d_imgs.append(img_g)

    save_grid(
        [img_best_raw] + d_imgs + [chain_both],
        ["Raw\n(dilated hard, from z_bike)"] +
        [f"Refined\ng={g:.1f}" for g in guidances] +
        ["Chain baseline\n(2 calls)"],
        os.path.join(args.out_dir, "expD_guidance_sweep.png"),
    )
    print("  Saved: expD_guidance_sweep.png")
    print("  KEY: Higher guidance = stronger adherence to REFINE_PROMPT description.")
    print("       Find the guidance where vase is solid + bike complete + quality ≥ chain.")

    # ----------------------------------------------------------
    # KEY_RESULT: chain vs best sequential refined
    # ----------------------------------------------------------
    print("\n=== KEY_RESULT (best sequential vs chain) ===")
    # Use R=8, pct=50, guidance=3.5 as the primary candidate
    img_final = run(img_best_raw, REFINE_PROMPT,
                    steps=args.refine_steps, guidance=args.refine_guidance)
    save(img_final, "final_sequential_refined")

    save_grid(
        [base, img_bike, img_vase, img_best_raw, img_final, chain_both],
        ["Base",
         "Clean bicycle\n(single edit)",
         "Clean vase\n(single edit)",
         "Sequential dilated\n(raw: z_bike + vase mask)",
         "Sequential dilated\n+ Kontext refine (final)",
         "Chain baseline\n(2 Kontext calls)"],
        os.path.join(args.out_dir, "KEY_RESULT.png"),
    )
    print("  Saved: KEY_RESULT.png")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("WHAT TO CHECK")
    print(f"{'='*60}")
    print("""
Why this experiment fixes the transparency issue:
  Old: z_comp = z_base + soft_mask_bike*z_bike + soft_mask_vase*z_vase
       At mask=0.5 → 50% z_base + 50% z_bike → TRANSPARENT (ghost)
  New: z_comp = z_bike  (bicycle fully present, zero blending)
       z_comp[dilated_mask_vase] = z_vase[...]  (hard copy, zero blending)
       No fractional values → no transparency by construction.

expA_dilation_sweep.png
  Row 1: mask shapes at R=0/4/8/12 (should see vase region expand)
  Row 2: raw paste (bike should always be COMPLETE — it comes from z_bike)
  Row 3: Kontext-refined
  → Find R where vase is fully covered but not too much background from z_vase

expB_threshold_sweep.png
  Same as A but varying the initial threshold at R=8
  Lower pct (40%) = bigger initial mask before dilation
  → Which pct+R combination gives complete vase without background bleed?

expC_transparency_fix.png  ← most important diagnostic
  Panel 1: old soft-blend method (should show transparent bike)
  Panel 2: new dilated-hard from z_bike (bike should be SOLID, complete)
  Panel 3: panel 2 + Kontext refine
  Panel 4: chain baseline
  SUCCESS: Panel 2 bike is solid (not transparent). Panel 3 ≥ Panel 4.

expD_guidance_sweep.png
  Try g=2.5 to 4.5 for the refinement.
  Too low → boundary seam visible between pasted region and base
  Too high → Kontext ignores the raw input and regenerates freely
  → Optimal: seam smoothed, both objects solid, positions preserved

KEY_RESULT.png
  Final 6-panel comparison.
  Success = panel 5 (sequential+refine) matches or beats panel 6 (chain).

If bike or vase STILL has issues after this:
  Bike missing: z_bike was generated differently this run → check clean_bicycle.png
  Vase still transparent: lower the pct further (try 30%) or increase R to 12
  Boundary seam: increase refine_guidance or refine_steps to 15
""")
    print(f"All results → {args.out_dir}/")


if __name__ == "__main__":
    main()
