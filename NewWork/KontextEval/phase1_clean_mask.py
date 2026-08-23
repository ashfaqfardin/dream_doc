"""
Phase 1 — Clean Mask via Relative Threshold + Morphological Opening

Diagnosis from phase1_pixel_mask.py and phase1_delta_sequential.py:
  - Percentile thresholds (50th, 70th, 90th): cover half the image or clip the object.
  - Pixel-space diff threshold=20: still catches global lighting/shadow changes.
  - Root cause: Kontext makes widespread small changes across the whole image
    (lighting, shadows, reflections) when adding even a small object. No fixed
    absolute or percentile threshold can separate "object" from "global noise."

Fix: two-stage mask that is SELF-CALIBRATING to each delta's own peak:

  Stage 1 — Relative threshold:
    thresh = mag.max() * alpha    (e.g. alpha=0.35)
    binary = (mag > thresh)
    → "Keep only pixels within 35% of the peak brightness"
    → The bike frame is ~100% of the peak; coffee table lighting change is ~5–10%
    → Alpha=0.35 cleanly separates object core from background noise

  Stage 2 — Morphological opening (erode then dilate):
    erode(R=2): removes isolated noise pixels (< 2px radius)
    dilate(R=4): restores the object body back to full size, now noise-free
    → Object blob survives; scattered background dots are eliminated

  Final small dilation (R=1): fills any remaining gaps at object edges.

Composition (explicit mutual exclusion):
  mask_bike = clean_mask(delta_bike)
  mask_vase = clean_mask(delta_vase)
  mask_vase = mask_vase * (1 - mask_bike)   ← vase NEVER touches bike area
  # paste vase first, bike last — bike wins any residual overlap
  z_comp = z_base
  z_comp[mask_vase] = z_vase
  z_comp[mask_bike] = z_bike
  → Refine with Kontext

Why the vase was missing in previous experiment:
  bike mask at pct=50 covered coffee table area → pasting z_bike (no vase)
  over the vase mask's result → vase overwritten → only bike visible.
  With relative threshold, bike mask should cover ONLY left-wall bicycle.

Experiments
-----------
  A. Alpha sweep: 0.20 / 0.30 / 0.40 / 0.50
     Show coverage%, mask overlay, raw paste, refined — per threshold.
     Key: which alpha gives bike mask ~10-20% coverage, vase mask ~3-8%?

  B. Morphology sweep at best alpha: (erode=0,dilate=0), (2,4), (2,6), (3,6)

  C. Final: chain baseline vs clean-mask + refine

Usage
-----
python NewWork/KontextEval/phase1_clean_mask.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_clean_mask
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
# Mask: relative threshold + morphological opening
# ============================================================

def morph_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological erosion: pixel stays 1 only if ALL neighbours within radius are 1."""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=radius)


def morph_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological dilation: pixel becomes 1 if ANY neighbour within radius is 1."""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def clean_mask(delta: torch.Tensor,
               alpha: float = 0.35,
               erode_r: int = 2,
               dilate_r: int = 4,
               final_dilate_r: int = 1) -> torch.Tensor:
    """
    Build a clean binary mask from a latent delta.

    delta: (1, C, H, W)  latent difference
    Returns: (1, 1, H, W) float binary mask

    Steps:
      1. mag = |delta|.mean(channel)
      2. thresh = mag.max() * alpha   ← relative, self-calibrated
      3. binary = mag > thresh
      4. erode(erode_r) → kill isolated noise pixels
      5. dilate(dilate_r) → restore object body, noise-free
      6. dilate(final_dilate_r) → fill remaining edge gaps
    """
    mag = delta.float().abs().mean(dim=1, keepdim=True)  # (1, 1, H, W)
    thresh = mag.max().item() * alpha
    binary = (mag >= thresh).float()
    # morphological open (erode then dilate removes noise)
    opened = morph_dilate(morph_erode(binary, erode_r), dilate_r)
    # small final dilation for edge coverage
    return morph_dilate(opened, final_dilate_r).clamp(0, 1)


def coverage_pct(mask: torch.Tensor) -> float:
    return mask.float().mean().item() * 100.0


# ============================================================
# Composition
# ============================================================

def compose(z_base, layers):
    """Sequential hard-copy. layers = [(z_src, mask), ...]. Later layers win."""
    z = z_base.clone()
    for z_src, mask in layers:
        z = z * (1.0 - mask) + z_src * mask
    return z


# ============================================================
# Mask visualisation: colour overlay on base image
# ============================================================

def overlay(img: Image.Image, mask: torch.Tensor,
            color: tuple[int, int, int] = (255, 80, 0),
            alpha: float = 0.45) -> Image.Image:
    """Draw mask as a tinted overlay on img."""
    H, W = img.size[1], img.size[0]
    mask_np = F.interpolate(mask, size=(H, W), mode="nearest").squeeze().numpy()
    base_np = np.array(img).astype(float)
    tint = np.zeros_like(base_np)
    tint[mask_np > 0.5] = color
    out_np = np.clip(base_np * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    # zero overlap stays original
    out_np[mask_np <= 0.5] = np.array(img)[mask_np <= 0.5]
    return Image.fromarray(out_np)


def both_overlays(img_bike, mask_b, img_vase, mask_v, base):
    """Show both masks together on the base image."""
    H, W = base.size[1], base.size[0]
    mb = F.interpolate(mask_b, size=(H, W), mode="nearest").squeeze().numpy()
    mv = F.interpolate(mask_v, size=(H, W), mode="nearest").squeeze().numpy()
    arr = np.array(base).astype(float)
    out = arr.copy()
    # bike = orange
    bike_clr = np.array([255, 120, 0], dtype=float)
    out[mb > 0.5] = arr[mb > 0.5] * 0.55 + bike_clr * 0.45
    # vase = blue
    vase_clr = np.array([0, 100, 255], dtype=float)
    out[mv > 0.5] = arr[mv > 0.5] * 0.55 + vase_clr * 0.45
    # overlap = purple
    both = (mb > 0.5) & (mv > 0.5)
    out[both] = arr[both] * 0.55 + np.array([180, 0, 200], dtype=float) * 0.45
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


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
        ax.set_title(title, fontsize=8)
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
    p.add_argument("--out_dir",           default="results/phase1_clean_mask")
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
    # Setup
    # ----------------------------------------------------------
    print("\n=== Setup ===")
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

    print("\n=== Chain baseline ===")
    chain_bike = run(base, BICYCLE_PROMPT)
    chain_both = run(chain_bike, VASE_PROMPT)
    save(chain_bike, "chain_step1")
    save(chain_both, "chain_step2_both")

    # ----------------------------------------------------------
    # Experiment A: Alpha (relative threshold) sweep
    #   Fixed morph: erode=2, dilate=4, final=1
    # ----------------------------------------------------------
    print("\n=== Experiment A: Relative alpha sweep ===")
    alphas = [0.20, 0.30, 0.40, 0.50]
    ERODE_A, DILATE_A, FINAL_A = 2, 4, 1

    a_combined, a_bike_ov, a_vase_ov = [], [], []
    a_raw, a_refined = [], []
    coverage_log = []

    for alpha in alphas:
        mb = clean_mask(delta_bike, alpha=alpha,
                        erode_r=ERODE_A, dilate_r=DILATE_A, final_dilate_r=FINAL_A)
        mv = clean_mask(delta_vase, alpha=alpha,
                        erode_r=ERODE_A, dilate_r=DILATE_A, final_dilate_r=FINAL_A)

        pct_b = coverage_pct(mb)
        pct_v = coverage_pct(mv)
        overlap = (mb * mv).mean().item() * 100.0
        coverage_log.append(
            f"alpha={alpha:.2f}  bike={pct_b:.1f}%  vase={pct_v:.1f}%  overlap={overlap:.1f}%"
        )
        print(f"  alpha={alpha:.2f}: bike={pct_b:.1f}% | vase={pct_v:.1f}% | overlap={overlap:.1f}%")

        a_bike_ov.append(overlay(img_bike, mb, color=(255, 120, 0)))
        a_vase_ov.append(overlay(img_vase, mv, color=(0, 100, 255)))
        a_combined.append(both_overlays(img_bike, mb, img_vase, mv, base))

        # mutual exclusion: vase NEVER touches bike area
        mv_safe = mv * (1.0 - mb)

        z_comp = compose(z_base, [(z_vase, mv_safe), (z_bike, mb)])
        img_comp = vae_decode(pipe, z_comp)
        save(img_comp, f"expA_alpha{alpha:.2f}_raw")
        a_raw.append(img_comp)

        img_ref = run(img_comp, REFINE_PROMPT,
                      steps=args.refine_steps, guidance=args.refine_guidance)
        save(img_ref, f"expA_alpha{alpha:.2f}_refined")
        a_refined.append(img_ref)

    # Print coverage table to terminal (key diagnostic)
    print("\n  Coverage table (target: bike ~10-20%, vase ~3-8%, overlap ~0%):")
    for line in coverage_log:
        print(f"    {line}")

    save_grid(
        a_bike_ov + a_vase_ov + a_combined + a_raw + a_refined,
        [f"BIKE MASK\nalpha={a:.2f}" for a in alphas] +
        [f"VASE MASK\nalpha={a:.2f}" for a in alphas] +
        [f"BOTH MASKS\nalpha={a:.2f}\n(orange=bike, blue=vase, purple=overlap)" for a in alphas] +
        [f"RAW\nalpha={a:.2f}" for a in alphas] +
        [f"REFINED\nalpha={a:.2f}" for a in alphas],
        os.path.join(args.out_dir, "expA_alpha_sweep.png"),
        ncols=len(alphas),
    )
    print("  Saved: expA_alpha_sweep.png (5 rows × 4 cols)")
    print("  Rows: bike-mask | vase-mask | both-on-base | raw-paste | refined")
    print("  FIND: alpha where orange = bike only, blue = vase only, no purple (no overlap)")

    # ----------------------------------------------------------
    # Experiment B: Morphology sweep at best alpha
    # ----------------------------------------------------------
    print("\n=== Experiment B: Morphology sweep at alpha=0.35 ===")
    ALPHA_B = 0.35
    morph_configs = [
        (0, 0, 0, "no open"),
        (2, 4, 1, "erode2+dilate4+1"),
        (2, 6, 1, "erode2+dilate6+1"),
        (3, 6, 2, "erode3+dilate6+2"),
    ]
    b_bike_ov, b_vase_ov, b_combined, b_raw, b_refined = [], [], [], [], []

    for erode_r, dilate_r, final_r, label in morph_configs:
        mb = clean_mask(delta_bike, alpha=ALPHA_B,
                        erode_r=erode_r, dilate_r=dilate_r, final_dilate_r=final_r)
        mv = clean_mask(delta_vase, alpha=ALPHA_B,
                        erode_r=erode_r, dilate_r=dilate_r, final_dilate_r=final_r)
        pct_b = coverage_pct(mb)
        pct_v = coverage_pct(mv)
        print(f"  {label}: bike={pct_b:.1f}% | vase={pct_v:.1f}%")

        b_bike_ov.append(overlay(img_bike, mb, color=(255, 120, 0)))
        b_vase_ov.append(overlay(img_vase, mv, color=(0, 100, 255)))
        b_combined.append(both_overlays(img_bike, mb, img_vase, mv, base))

        mv_safe = mv * (1.0 - mb)
        z_comp = compose(z_base, [(z_vase, mv_safe), (z_bike, mb)])
        img_comp = vae_decode(pipe, z_comp)
        save(img_comp, f"expB_{label}_raw")
        b_raw.append(img_comp)

        img_ref = run(img_comp, REFINE_PROMPT,
                      steps=args.refine_steps, guidance=args.refine_guidance)
        save(img_ref, f"expB_{label}_refined")
        b_refined.append(img_ref)

    save_grid(
        b_bike_ov + b_vase_ov + b_combined + b_raw + b_refined,
        [f"BIKE\n{l}" for _, _, _, l in morph_configs] +
        [f"VASE\n{l}" for _, _, _, l in morph_configs] +
        [f"BOTH\n{l}" for _, _, _, l in morph_configs] +
        [f"RAW\n{l}" for _, _, _, l in morph_configs] +
        [f"REFINED\n{l}" for _, _, _, l in morph_configs],
        os.path.join(args.out_dir, "expB_morph_sweep.png"),
        ncols=len(morph_configs),
    )
    print("  Saved: expB_morph_sweep.png")
    print("  KEY: Larger erode removes more noise but shrinks object edges.")
    print("       Larger dilate restores edges. Best: minimal coverage, clean shape.")

    # ----------------------------------------------------------
    # KEY RESULT
    # ----------------------------------------------------------
    print("\n=== KEY RESULT: best config vs chain ===")
    # Use best alpha from coverage table (expect ~0.35) + erode=2, dilate=4, final=1
    ALPHA_BEST = 0.35
    mb_best = clean_mask(delta_bike, alpha=ALPHA_BEST, erode_r=2, dilate_r=4, final_dilate_r=1)
    mv_best = clean_mask(delta_vase, alpha=ALPHA_BEST, erode_r=2, dilate_r=4, final_dilate_r=1)
    mv_best_safe = mv_best * (1.0 - mb_best)

    print(f"  BEST CONFIG: alpha={ALPHA_BEST}")
    print(f"    bike={coverage_pct(mb_best):.1f}%  vase={coverage_pct(mv_best_safe):.1f}%  "
          f"overlap={(mb_best * mv_best).mean().item()*100:.1f}%")

    z_best = compose(z_base, [(z_vase, mv_best_safe), (z_bike, mb_best)])
    img_best_raw = vae_decode(pipe, z_best)
    save(img_best_raw, "key_raw")

    img_best_ref = run(img_best_raw, REFINE_PROMPT,
                       steps=args.refine_steps, guidance=args.refine_guidance)
    save(img_best_ref, "key_refined")

    combined_best = both_overlays(img_bike, mb_best, img_vase, mv_best_safe, base)
    save(combined_best, "key_masks_on_base")

    save_grid(
        [base, combined_best, img_best_raw, img_best_ref, chain_both],
        ["Base scene",
         f"Masks on base\n(orange=bike, blue=vase)\nalpha={ALPHA_BEST}",
         "Raw paste\n(z_base + bike + vase)",
         f"Refined (CANDIDATE)\n({args.refine_steps} steps, g={args.refine_guidance})",
         "Chain baseline (TARGET)\n(2 Kontext calls)"],
        os.path.join(args.out_dir, "KEY_RESULT.png"),
    )
    print("  Saved: KEY_RESULT.png")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("WHAT TO CHECK")
    print(f"{'='*60}")
    print(f"""
Coverage table (printed above) — read this FIRST:
  Target: bike ~10-20% of latent area (bicycle is a large object)
           vase ~3-8%  of latent area (vase is a small object on a table)
           overlap ~ 0% (they're in different parts of the room)
  If bike coverage >> 20%: alpha is too low (catching background)
  If bike coverage << 5%: alpha is too high (missing edges)

expA_alpha_sweep.png — row by row:
  Row 1 (BIKE MASK orange): Does the orange region cover ONLY the bicycle?
    Good: orange fills the bicycle shape on the left wall, nothing else
    Bad:  orange also covers sofa, coffee table, floor
  Row 2 (VASE MASK blue): Does the blue region cover ONLY the vase+flowers?
    Good: blue covers vase+flowers on coffee table, nothing else
    Bad:  blue also covers wall, sofa, floor
  Row 3 (BOTH on base): Are orange and blue SEPARATE (no purple)?
    Good: no overlap, they're in different spatial regions
    Bad:  purple regions = conflict zones
  Row 4 (RAW paste): BOTH objects visible? Background clean?
    If yes: masks are working, Kontext just needs to smooth edges
    If only bike: bike mask is leaking over coffee table, overwriting vase
    If only vase: vase mask is leaking over left wall, overwriting bike
  Row 5 (REFINED): Quality? Seam removed? Both objects solid?

key_masks_on_base.png
  Single combined view of the final masks on the base scene.
  This is the ground truth for what the composition will look like.

KEY_RESULT.png
  Panel 4 (refined) vs Panel 5 (chain baseline):
  SUCCESS = panel 4 has both objects solid, quality ≥ chain.
  FAILURE = object missing or transparency → adjust alpha or morph params.
""")
    print(f"All results → {args.out_dir}/")


if __name__ == "__main__":
    main()
