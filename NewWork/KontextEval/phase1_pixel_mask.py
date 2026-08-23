"""
Phase 1 — Pixel-Space Mask Composition

Diagnosis from phase1_delta_sequential.py:
  - Delta-based mask at pct=50 covers HALF the image (50th percentile = above-median
    pixels, which is always ~50% of the area). Even tiny dilation fills everything.
  - Root cause: FLUX Kontext makes widespread subtle changes across the whole image
    when adding a small object (recomputes lighting, shadows, reflections globally).
    The latent delta is NOT a localized object mask — it's a global edit signal.

Fix — Pixel-space mask:
  |img_vase - img_base| in RGB space tells you exactly which pixels visually changed.
  A vase on a coffee table changes those pixels strongly (delta > 30) and changes
  the rest of the room only slightly (delta < 10). Threshold cleanly separates them.

  This is a semantic mask without any segmentation model.

Composition strategy:
  z_comp = z_base                          (clean background, no noise)
  z_comp[mask_vase] = z_vase[mask_vase]   (hard copy — vase at full quality)
  z_comp[mask_bike] = z_bike[mask_bike]   (hard copy — bike LAST, wins in overlap)
  → refine with Kontext

  Bike is pasted LAST so it wins in any accidental overlap region.
  Background stays z_base, so no color drift from either single-object image.

Experiments
-----------
  A. Pixel diff threshold sweep: 15 / 20 / 30 / 40 (which cleanly isolates each object)
  B. Small dilation in latent space: R = 0, 1, 2, 3 latent pixels (at best threshold)
  C. Composition ordering — bike last (our order) vs vase last
  D. Refinement guidance sweep: 2.5 / 3.0 / 3.5 / 4.0
  E. Final comparison: chain baseline vs pixel-mask + refine

Usage
-----
python NewWork/KontextEval/phase1_pixel_mask.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_pixel_mask
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
from PIL import Image, ImageDraw

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
# Pixel-space mask
# ============================================================

def pixel_diff_mask(img_a: Image.Image, img_b: Image.Image,
                    threshold: float = 20.0,
                    latent_h: int = 128, latent_w: int = 128,
                    dilation_radius: int = 0) -> torch.Tensor:
    """
    Compute a binary mask (1, 1, latent_h, latent_w) from pixel-space difference.

    Steps:
      1. |img_b - img_a| per channel (0–255 scale)
      2. Mean over RGB channels → (H, W) diff map
      3. Downsample to latent spatial size via avg_pool
      4. Threshold: diff > threshold → 1 else 0
      5. Optional morphological dilation via max_pool2d

    Why pixel space (not latent delta):
      Kontext makes widespread subtle latent changes when adding any object
      (recomputes global lighting/shadows). A 50th-percentile latent threshold
      covers half the image. Pixel diff only lights up where colors visibly changed.
    """
    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)
    diff = np.abs(b - a).mean(axis=2)         # (H, W), values 0–255
    diff_t = torch.from_numpy(diff)[None, None]  # (1, 1, H, W)

    # Downsample to latent resolution (average within each 8×8 block)
    diff_lat = F.avg_pool2d(diff_t, kernel_size=8, stride=8)   # (1, 1, lat_h, lat_w)

    # Binary threshold
    binary = (diff_lat >= threshold).float()

    # Optional dilation (small — 1-3 latent pixels to fill gaps in object silhouette)
    if dilation_radius > 0:
        k = 2 * dilation_radius + 1
        binary = F.max_pool2d(binary, kernel_size=k, stride=1, padding=dilation_radius)

    return binary.clamp(0, 1)


def mask_coverage_pct(mask: torch.Tensor) -> float:
    """Fraction of latent pixels that are 1."""
    return mask.float().mean().item() * 100.0


# ============================================================
# Composition
# ============================================================

def compose(z_base: torch.Tensor,
            layers: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """
    Sequential hard-copy composition.
    layers: list of (z_src, mask) applied in order — LATER layers win in overlaps.
    mask: (1, 1, H, W) binary float.
    """
    z = z_base.clone()
    for z_src, mask in layers:
        z = z * (1.0 - mask) + z_src * mask
    return z


# ============================================================
# Visualisation helpers
# ============================================================

def mask_overlay(img: Image.Image, mask: torch.Tensor,
                 color=(255, 0, 0, 80)) -> Image.Image:
    """Overlay a binary latent mask on a PIL image as a semi-transparent tint."""
    H, W = img.size[1], img.size[0]
    # Upsample mask to image resolution
    mask_up = F.interpolate(mask, size=(H, W), mode="nearest").squeeze().numpy()
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(H):
        for x in range(W):
            if mask_up[y, x] > 0.5:
                draw.point((x, y), fill=color)
    base_rgba = img.convert("RGBA")
    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


def mask_to_pil(mask: torch.Tensor, H: int = 256, W: int = 256) -> Image.Image:
    """Resize and return mask as a greyscale-tinted PIL image."""
    arr = (mask.squeeze().float().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize((W, H), Image.NEAREST).convert("RGB")


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
    p.add_argument("--out_dir",           default="results/phase1_pixel_mask")
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
    lh = args.height // 8   # latent height
    lw = args.width  // 8   # latent width
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

    print("  Vase …")
    img_vase = run(base, VASE_PROMPT)
    save(img_vase, "clean_vase")
    z_vase = vae_encode(pipe, img_vase)

    # Chain baseline
    print("\n=== Chain baseline ===")
    chain_bike = run(base, BICYCLE_PROMPT)
    save(chain_bike, "chain_step1_bicycle")
    chain_both = run(chain_bike, VASE_PROMPT)
    save(chain_both, "chain_step2_both")

    # ----------------------------------------------------------
    # Experiment A: Pixel diff threshold sweep
    #   R=2 latent pixels fixed (=16 image pixels of dilation)
    # ----------------------------------------------------------
    print("\n=== Experiment A: Pixel diff threshold sweep ===")
    thresholds = [10, 15, 20, 30]
    R_A = 2

    a_mask_imgs, a_overlay_bike, a_overlay_vase = [], [], []
    a_raw, a_refined = [], []

    for thr in thresholds:
        print(f"  Threshold={thr} …")
        mb = pixel_diff_mask(base, img_bike, threshold=thr,
                             latent_h=lh, latent_w=lw, dilation_radius=R_A)
        mv = pixel_diff_mask(base, img_vase, threshold=thr,
                             latent_h=lh, latent_w=lw, dilation_radius=R_A)
        pct_b = mask_coverage_pct(mb)
        pct_v = mask_coverage_pct(mv)
        print(f"    bike mask covers {pct_b:.1f}% | vase mask covers {pct_v:.1f}%")

        # Visualise masks
        a_overlay_bike.append(mask_overlay(img_bike, mb, color=(255, 100, 0, 100)))
        a_overlay_vase.append(mask_overlay(img_vase, mv, color=(0, 100, 255, 100)))

        # Compose: z_base → paste vase → paste bike (bike wins in overlap)
        z_comp = compose(z_base, [(z_vase, mv), (z_bike, mb)])
        img_comp = vae_decode(pipe, z_comp)
        save(img_comp, f"expA_thr{thr}_raw")
        a_raw.append(img_comp)

        img_ref = run(img_comp, REFINE_PROMPT,
                      steps=args.refine_steps, guidance=args.refine_guidance)
        save(img_ref, f"expA_thr{thr}_refined")
        a_refined.append(img_ref)

    save_grid(
        a_overlay_bike + a_overlay_vase + a_raw + a_refined,
        [f"BIKE MASK thr={t}\n(orange = masked)" for t in thresholds] +
        [f"VASE MASK thr={t}\n(blue = masked)" for t in thresholds] +
        [f"RAW thr={t}" for t in thresholds] +
        [f"REFINED thr={t}" for t in thresholds],
        os.path.join(args.out_dir, "expA_threshold_sweep.png"),
        ncols=len(thresholds),
    )
    print("  Saved: expA_threshold_sweep.png")
    print("  Rows: 1=bike mask, 2=vase mask, 3=raw paste, 4=refined")
    print("  KEY: Which threshold makes bike mask = just the bike, vase mask = just vase?")
    print("       Then: does the raw paste show BOTH objects without blending artifacts?")

    # ----------------------------------------------------------
    # Experiment B: Small dilation sweep at best threshold
    # ----------------------------------------------------------
    print("\n=== Experiment B: Dilation sweep at thr=20 ===")
    THR_B = 20
    radii = [0, 1, 2, 3]
    b_masks_b, b_masks_v, b_raw, b_refined = [], [], [], []

    for R in radii:
        print(f"  R={R} …")
        mb = pixel_diff_mask(base, img_bike, threshold=THR_B,
                             latent_h=lh, latent_w=lw, dilation_radius=R)
        mv = pixel_diff_mask(base, img_vase, threshold=THR_B,
                             latent_h=lh, latent_w=lw, dilation_radius=R)
        b_masks_b.append(mask_overlay(img_bike, mb, color=(255, 100, 0, 100)))
        b_masks_v.append(mask_overlay(img_vase, mv, color=(0, 100, 255, 100)))

        z_comp = compose(z_base, [(z_vase, mv), (z_bike, mb)])
        img_comp = vae_decode(pipe, z_comp)
        save(img_comp, f"expB_R{R}_raw")
        b_raw.append(img_comp)

        img_ref = run(img_comp, REFINE_PROMPT,
                      steps=args.refine_steps, guidance=args.refine_guidance)
        save(img_ref, f"expB_R{R}_refined")
        b_refined.append(img_ref)

    save_grid(
        b_masks_b + b_masks_v + b_raw + b_refined,
        [f"BIKE MASK R={R}" for R in radii] +
        [f"VASE MASK R={R}" for R in radii] +
        [f"RAW R={R}" for R in radii] +
        [f"REFINED R={R}" for R in radii],
        os.path.join(args.out_dir, "expB_dilation_sweep.png"),
        ncols=len(radii),
    )
    print("  Saved: expB_dilation_sweep.png")
    print("  KEY: R=0 may leave gaps at object edges; R=1-2 fills them without overflow")

    # ----------------------------------------------------------
    # Experiment C: Composition ordering
    #   Bike-last (ours) vs Vase-last — does it matter?
    # ----------------------------------------------------------
    print("\n=== Experiment C: Bike-last vs Vase-last ordering ===")
    THR_C = 20
    R_C   = 2
    mb = pixel_diff_mask(base, img_bike, threshold=THR_C,
                         latent_h=lh, latent_w=lw, dilation_radius=R_C)
    mv = pixel_diff_mask(base, img_vase, threshold=THR_C,
                         latent_h=lh, latent_w=lw, dilation_radius=R_C)
    overlap_pct = (mb * mv).float().mean().item() * 100.0
    print(f"  Mask overlap: {overlap_pct:.1f}% of latent pixels")

    # Bike last (bike wins in overlap)
    z_bike_last = compose(z_base, [(z_vase, mv), (z_bike, mb)])
    img_bl_raw = vae_decode(pipe, z_bike_last)
    img_bl_ref = run(img_bl_raw, REFINE_PROMPT,
                     steps=args.refine_steps, guidance=args.refine_guidance)
    save(img_bl_raw, "expC_bike_last_raw")
    save(img_bl_ref, "expC_bike_last_refined")

    # Vase last (vase wins in overlap)
    z_vase_last = compose(z_base, [(z_bike, mb), (z_vase, mv)])
    img_vl_raw = vae_decode(pipe, z_vase_last)
    img_vl_ref = run(img_vl_raw, REFINE_PROMPT,
                     steps=args.refine_steps, guidance=args.refine_guidance)
    save(img_vl_raw, "expC_vase_last_raw")
    save(img_vl_ref, "expC_vase_last_refined")

    save_grid(
        [img_bl_raw, img_vl_raw, img_bl_ref, img_vl_ref, chain_both],
        ["Bike-last RAW\n(bike wins overlap)",
         "Vase-last RAW\n(vase wins overlap)",
         "Bike-last REFINED",
         "Vase-last REFINED",
         "Chain baseline\n(2 calls)"],
        os.path.join(args.out_dir, "expC_ordering.png"),
    )
    print(f"  Saved: expC_ordering.png  (mask overlap = {overlap_pct:.1f}%)")
    print("  If overlap is <1%: ordering doesn't matter (no conflict)")
    print("  If overlap is significant: bike-last preserves bicycle better")

    # ----------------------------------------------------------
    # Experiment D: Refinement guidance sweep
    # ----------------------------------------------------------
    print("\n=== Experiment D: Refinement guidance sweep ===")
    # Use bike-last ordering, thr=20, R=2
    img_d_raw = img_bl_raw
    guidances = [2.0, 2.5, 3.0, 3.5, 4.0]
    d_imgs = []
    for g in guidances:
        print(f"  guidance={g} …")
        img_g = run(img_d_raw, REFINE_PROMPT,
                    steps=args.refine_steps, guidance=g)
        save(img_g, f"expD_g{g:.1f}")
        d_imgs.append(img_g)

    save_grid(
        [img_d_raw] + d_imgs + [chain_both],
        ["RAW\n(before refine)"] +
        [f"g={g:.1f}" for g in guidances] +
        ["Chain baseline"],
        os.path.join(args.out_dir, "expD_guidance_sweep.png"),
    )
    print("  Saved: expD_guidance_sweep.png")
    print("  KEY: Find guidance where boundary seam disappears AND bike/vase stay solid")

    # ----------------------------------------------------------
    # KEY RESULT
    # ----------------------------------------------------------
    print("\n=== KEY RESULT ===")
    # Best candidate: bike-last, thr=20, R=2, g=3.5
    save_grid(
        [base, img_bike, img_vase,
         img_bl_raw, img_bl_ref,
         chain_both],
        ["Base",
         "Clean bicycle\n(single Kontext call)",
         "Clean vase\n(single Kontext call)",
         "Pixel-mask compose\n(raw — z_base + bike + vase)",
         "Pixel-mask + REFINE\n(10 steps, g=3.5)  ← CANDIDATE",
         "Chain baseline\n(2 Kontext calls)  ← TARGET"],
        os.path.join(args.out_dir, "KEY_RESULT.png"),
    )
    print("  Saved: KEY_RESULT.png  ← main comparison")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("WHAT TO CHECK")
    print(f"{'='*60}")
    print(f"""
Why pixel-space mask is different from latent-delta mask:
  Latent delta (previous attempts): |z_vase - z_base|.mean(channel)
    → Kontext recomputes lighting/shadows globally when adding any object
    → 50th percentile of delta = HALF the image. Useless for localization.
  Pixel diff (this file): |img_vase - img_base|.mean(channel)
    → Pixels only change where the vase actually appears in the image
    → Threshold=20 (out of 255) catches only genuine color changes
    → Coverage should be ~5-15% of the image, not 50%

expA_threshold_sweep.png  (most important diagnostic)
  Row 1 (orange overlay on bicycle image): WHERE is the bike mask?
    Should cover: the bicycle shape on the left wall
    Should NOT cover: sofa, coffee table, right wall, floor
  Row 2 (blue overlay on vase image): WHERE is the vase mask?
    Should cover: vase + flowers on the coffee table
    Should NOT cover: sofa, bicycle-wall area, floor
  Row 3 (raw paste):
    Should show: BOTH objects present, background clean, no transparency
  Row 4 (refined):
    Should show: edges smooth, quality matches chain baseline

  If masks look correct but raw paste still looks wrong:
    → Seam between paste regions decodes badly from VAE
    → The Kontext refine step should fix this

expC_ordering.png
  Check if mask overlap percentage is small (< 2%):
    → bicycle is against left wall, vase is on coffee table
    → These shouldn't spatially overlap at all
  If overlap IS large: bike-last ordering preserves the bicycle

KEY_RESULT.png
  Panel 5 (pixel-mask + refine) vs Panel 6 (chain baseline):
  SUCCESS = bicycle solid + complete, vase solid (not transparent),
            quality comparable to chain, background matches base.
""")
    print(f"All results → {args.out_dir}/")


if __name__ == "__main__":
    main()
