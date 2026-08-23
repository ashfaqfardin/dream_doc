"""
Phase 1 — K/V Chain Injection: Bicycle Identity Preservation via Reference Attention

Chain structure (stepwise editing, same ordering as the 2-call chain baseline):

  Step 1: base ─[Kontext 28 steps]──► img_bike   (standard, no injection)
  Step 2: img_bike ─[Kontext + K/V inject]──► result   (vase added, bicycle locked)

Why K/V injection works here where spatial masking failed
---------------------------------------------------------
Spatial masking failed because:
  - The bicycle and vase OVERLAP in 2D projection (bike is partially behind the table)
  - Pasting z_vase (no bike) into the overlap region erases the bike
  - Latent "fault lines" between two source tensors decode as distorted pixels

K/V injection avoids all of this:
  - No latent compositing — a single coherent denoising trajectory
  - Injection works in ATTENTION SPACE, not in pixel or latent space
  - Objects do not need to be spatially disjoint

How the injection works in Step 2
----------------------------------
Kontext concatenates the reference image (img_bike) as a SEPARATE TOKEN SLICE
alongside the noisy generation tokens in every attention call:

  sequence layout: [ txt_tokens (512) | gen_tokens (4096) | ref_tokens (4096) ]

At TIER_A layers (content-similarity, low RoPE frequency), in the bicycle region
of the gen token slice, the K/V are blended with the corresponding ref K/V:

  K_gen[bike_region] = (1-s) * K_gen[bike_region] + s * K_ref[bike_region]
  V_gen[bike_region] = (1-s) * V_gen[bike_region] + s * V_ref[bike_region]

Because Step 2's reference IS img_bike (the clean bicycle image), K_ref encodes
the bicycle's appearance features. Injecting them into the bicycle gen tokens
forces the denoising to reproduce the bicycle faithfully.

The vase region gen tokens are NEVER injected — the prompt drives vase generation
freely there. No spatial mask in the latent is needed.

TIER_A layers (from FreeFlux classification, validated in IncrementalEdit):
  [0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56]
  Content-similarity layers — respond to "what is here" not "where is this."
  Memory note: TIER_ALL (57 layers) collapses to pixel-identical output;
  TIER_A (13 layers) preserves appearance while letting the edit happen.

Bicycle region mask
-------------------
Computed from pixel diff |img_bike - base| > threshold, then downsampled
to the 64x64 latent token grid (one token = 16x16 image pixels in Kontext).
This is the same mask signal, but used only to INDEX into attention, not to
hard-copy latent tensors — so spatial overlap with the vase is not a problem.

Experiments
-----------
  A. Strength sweep: s = 0.0 / 0.3 / 0.5 / 0.7 / 1.0
     s=0.0: standard chain (no injection, baseline)
     s=0.5: 50/50 blend
     s=1.0: full lock (ref K/V completely replaces gen K/V in bicycle region)

  B. Cutoff sweep: inject during first X% of denoising steps
     (0.0, 0.4) / (0.0, 0.6) / (0.0, 0.8)
     Structure is set early; fine details late.

  C. Layer set: TIER_A (13 layers) vs TIER_ALL (57 layers)
     Expected: TIER_ALL collapses to near-identical copy, TIER_A preserves
     bicycle while allowing vase to appear.

Metric: bicycle_diff
  Mean absolute pixel change in the bicycle region between img_bike and result.
  Lower = bicycle identity better preserved.
  Combined with vase_appears (vase diff > base diff) to confirm vase was added.

Usage
-----
python NewWork/KontextEval/phase1_kv_chain.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_kv_chain
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline, generate

# Import IncrementalEdit's injection infrastructure
_ie_path = str(Path(__file__).parent.parent / "IncrementalEdit")
sys.path.insert(0, _ie_path)
from kontext_injection import (
    TIER_A, TIER_ALL, N_LAYERS, N_DOUBLE,
    ZoneMasks, InjectionState,
    KontextInjectionProcessor, install_processor,
    set_determinism,
)

# ============================================================
# Layer sets
# ============================================================

TIER_A_LAYERS: List[int] = list(TIER_A)    # [0,7,8,9,10,18,25,28,37,42,45,50,56]
ALL_LAYERS:    List[int] = list(TIER_ALL)   # 0..56


# ============================================================
# Prompts
# ============================================================

BASE_PROMPT    = "A modern living room with a sofa and a wooden coffee table."
BICYCLE_PROMPT = (
    "Add a yellow bicycle leaning against the wall on the left side. "
    "Keep the rest of the room exactly the same."
)
VASE_PROMPT    = (
    "Add a white ceramic vase with flowers on the coffee table. "
    "Keep the rest of the room exactly the same."
)


# ============================================================
# Mask: pixel diff → token-grid binary mask
# ============================================================

def pixel_to_token_mask(img_a: Image.Image, img_b: Image.Image,
                        h_lat: int, w_lat: int,
                        threshold: float = 40.0) -> np.ndarray:
    """
    Flat bool array of length h_lat*w_lat — True where the images differ.
    Pixel diff is computed in image space (0-255) then downsampled to the
    latent token grid (one token = 16×16 image pixels for FLUX Kontext).
    """
    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)
    diff = np.abs(b - a).mean(axis=2)                             # (H, W) 0-255
    diff_img = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
    diff_down = diff_img.resize((w_lat, h_lat), Image.BILINEAR)   # 64×64
    diff_arr  = np.array(diff_down).astype(np.float32)
    mask = diff_arr >= threshold
    return mask.reshape(-1)                                        # (n_gen,)


def mask_overlay_pil(img: Image.Image, flat_mask: np.ndarray,
                     h_lat: int, w_lat: int,
                     color=(255, 120, 0), alpha=0.4) -> Image.Image:
    """Orange tint where mask==True (upsampled to image size)."""
    H, W = img.size[1], img.size[0]
    token_2d = flat_mask.reshape(h_lat, w_lat).astype(np.uint8) * 255
    token_img = Image.fromarray(token_2d, mode="L").resize((W, H), Image.NEAREST)
    mask_np = np.array(token_img) > 127
    arr = np.array(img).astype(float)
    out = arr.copy()
    out[mask_np] = arr[mask_np] * (1 - alpha) + np.array(color) * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# ============================================================
# Measurement
# ============================================================

def region_diff(img_a: Image.Image, img_b: Image.Image,
                flat_mask: np.ndarray, h_lat: int, w_lat: int) -> float:
    """Mean absolute pixel diff (0-255) inside the masked region."""
    H, W = img_a.size[1], img_a.size[0]
    token_2d = flat_mask.reshape(h_lat, w_lat).astype(np.uint8) * 255
    pixel_mask = np.array(
        Image.fromarray(token_2d, mode="L").resize((W, H), Image.NEAREST)
    ) > 127
    a = np.array(img_a).astype(float)
    b = np.array(img_b).astype(float)
    diff = np.abs(b - a).mean(axis=2)
    return float(diff[pixel_mask].mean()) if pixel_mask.any() else 0.0


# ============================================================
# Injected step-2 generation
# ============================================================

@torch.no_grad()
def run_step2_injected(
    pipe,
    canvas: Image.Image,
    prompt: str,
    bike_mask: np.ndarray,      # flat bool (n_gen,) — bicycle region
    vital_layers: List[int],
    strength: float = 0.7,
    cutoff_frac: Tuple[float, float] = (0.0, 0.6),
    seed: int = 42,
    num_steps: int = 28,
    guidance_scale: float = 2.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
) -> Image.Image:
    """
    Run a single Kontext denoising pass with TIER_A K/V injection.

    In the bicycle region (bike_mask==True), at every TIER_A layer during
    the first `cutoff_frac` fraction of steps:
      K_gen[bike] = (1-strength)*K_gen[bike] + strength*K_ref[bike]
      V_gen[bike] = (1-strength)*V_gen[bike] + strength*V_ref[bike]

    K_ref comes from the reference IMAGE SLICE already present in Kontext's
    joint sequence — no separate caching pass needed.
    strength=0.0 reduces to the unmodified chain.
    """
    h_lat = height // 16
    w_lat = width  // 16
    n_gen = h_lat * w_lat

    # Zone masks: background=bicycle (inject), shell=empty, target=rest (no inject)
    no_shell = np.zeros(n_gen, dtype=bool)
    state = InjectionState(
        mode="edit",
        vital_layers=set(vital_layers),
        n_gen=n_gen,
        n_ref=n_gen,
        cutoff_frac=cutoff_frac,
        strength=strength,
        n_steps=num_steps,
    )
    state.zones = ZoneMasks(
        background=bike_mask.astype(bool),
        shell=no_shell,
        target=np.logical_not(bike_mask),   # everything else: no injection
    ).to_device(device)

    install_processor(pipe, state, max_sequence_length=max_sequence_length)
    generator = set_determinism(seed)
    result = pipe(
        image=canvas,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        generator=generator,
        output_type="pil",
    )
    return result.images[0]


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
    p.add_argument("--hf_token",            required=True)
    p.add_argument("--cache_dir",           default="./models")
    p.add_argument("--out_dir",             default="results/phase1_kv_chain")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--num_steps",           type=int,   default=28)
    p.add_argument("--guidance",            type=float, default=2.5)
    p.add_argument("--bike_threshold",      type=float, default=40.0,
                   help="Pixel diff threshold (0-255) for bicycle token mask")
    p.add_argument("--height",              type=int,   default=1024)
    p.add_argument("--width",               type=int,   default=1024)
    p.add_argument("--device",              default="cuda")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    h_lat = args.height // 16
    w_lat = args.width  // 16

    def save(img, name):
        img.save(os.path.join(args.out_dir, f"{name}.png"))

    def run_standard(src, prompt, steps=None, guidance=None):
        return generate(
            pipe, prompt, src,
            seed=args.seed,
            num_steps=steps or args.num_steps,
            guidance_scale=guidance or args.guidance,
            height=args.height, width=args.width,
        )

    print("Loading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    # ----------------------------------------------------------
    # Chain Step 1: base → bicycle (standard, no injection)
    # ----------------------------------------------------------
    print("\n=== Step 1: Base → Bicycle ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(grey, BASE_PROMPT)
    save(base, "step0_base")

    img_bike = run_standard(base, BICYCLE_PROMPT)
    save(img_bike, "step1_bicycle")

    # Bicycle region mask (token grid)
    bike_mask = pixel_to_token_mask(
        base, img_bike, h_lat, w_lat, threshold=args.bike_threshold
    )
    pct = bike_mask.mean() * 100
    print(f"  Bicycle token mask: {pct:.1f}% of tokens  "
          f"(threshold={args.bike_threshold}, h_lat={h_lat}, w_lat={w_lat})")
    save(mask_overlay_pil(img_bike, bike_mask, h_lat, w_lat), "bike_mask_overlay")

    # ----------------------------------------------------------
    # Chain Step 2 baseline: img_bike → "add vase" (no injection)
    # ----------------------------------------------------------
    print("\n=== Step 2 baseline (no injection, s=0.0) ===")
    chain_baseline = run_standard(img_bike, VASE_PROMPT)
    save(chain_baseline, "step2_chain_baseline")
    bike_diff_base = region_diff(img_bike, chain_baseline, bike_mask, h_lat, w_lat)
    print(f"  Baseline bike_diff = {bike_diff_base:.2f}  (target to beat)")

    scores = {"s=0.0 (baseline)": bike_diff_base}

    # ----------------------------------------------------------
    # Experiment A: Strength sweep  (TIER_A, cutoff=0.6)
    # ----------------------------------------------------------
    print("\n=== Experiment A: Strength sweep (TIER_A layers, cutoff 0→60%) ===")
    strengths = [0.3, 0.5, 0.7, 1.0]
    a_imgs, a_labels = [], []

    for s in strengths:
        print(f"  Strength s={s} …")
        img = run_step2_injected(
            pipe, img_bike, VASE_PROMPT,
            bike_mask=bike_mask,
            vital_layers=TIER_A_LAYERS,
            strength=s,
            cutoff_frac=(0.0, 0.6),
            seed=args.seed,
            num_steps=args.num_steps,
            guidance_scale=args.guidance,
            height=args.height, width=args.width,
            device=args.device,
        )
        diff = region_diff(img_bike, img, bike_mask, h_lat, w_lat)
        scores[f"s={s} TIER_A cut=0.6"] = diff
        save(img, f"expA_s{s:.1f}")
        a_imgs.append(img)
        a_labels.append(f"s={s}  TIER_A\nbike_diff={diff:.2f}")
        print(f"    bike_diff={diff:.2f}  (baseline={bike_diff_base:.2f})")

    save_grid(
        [chain_baseline] + a_imgs,
        [f"Baseline s=0.0\nbike_diff={bike_diff_base:.2f}"] + a_labels,
        os.path.join(args.out_dir, "expA_strength_sweep.png"),
    )
    print("  Saved: expA_strength_sweep.png")
    print("  KEY: Does higher strength → lower bike_diff AND vase still present?")
    print("       s=1.0: fully locked bike (but vase may be suppressed)")
    print("       s=0.5-0.7: best tradeoff expected")

    # ----------------------------------------------------------
    # Experiment B: Cutoff sweep  (best strength from A, TIER_A)
    # ----------------------------------------------------------
    print("\n=== Experiment B: Cutoff sweep (TIER_A, s=0.7) ===")
    cutoffs = [(0.0, 0.4), (0.0, 0.6), (0.0, 0.8)]
    b_imgs, b_labels = [], []

    for cut in cutoffs:
        label = f"cut={cut[0]:.1f}-{cut[1]:.1f}"
        print(f"  Cutoff {cut} …")
        img = run_step2_injected(
            pipe, img_bike, VASE_PROMPT,
            bike_mask=bike_mask,
            vital_layers=TIER_A_LAYERS,
            strength=0.7,
            cutoff_frac=cut,
            seed=args.seed,
            num_steps=args.num_steps,
            guidance_scale=args.guidance,
            height=args.height, width=args.width,
            device=args.device,
        )
        diff = region_diff(img_bike, img, bike_mask, h_lat, w_lat)
        scores[f"s=0.7 TIER_A {label}"] = diff
        save(img, f"expB_{label.replace('.', '')}")
        b_imgs.append(img)
        b_labels.append(f"TIER_A s=0.7\n{label}\nbike_diff={diff:.2f}")
        print(f"    bike_diff={diff:.2f}")

    save_grid(
        [chain_baseline] + b_imgs,
        [f"Baseline\nbike_diff={bike_diff_base:.2f}"] + b_labels,
        os.path.join(args.out_dir, "expB_cutoff_sweep.png"),
    )
    print("  Saved: expB_cutoff_sweep.png")
    print("  KEY: Does earlier cutoff (0→40%) preserve bike as well as 0→60%?")
    print("       Injection only needed during structure-forming early steps.")

    # ----------------------------------------------------------
    # Experiment C: TIER_A vs ALL layers  (s=0.7, best cutoff)
    # ----------------------------------------------------------
    print("\n=== Experiment C: TIER_A vs ALL 57 layers (s=0.7, cut=0→60%) ===")
    results_c = {}
    for lname, layers in [("TIER_A", TIER_A_LAYERS), ("ALL_57", ALL_LAYERS)]:
        print(f"  Layers: {lname} …")
        img = run_step2_injected(
            pipe, img_bike, VASE_PROMPT,
            bike_mask=bike_mask,
            vital_layers=layers,
            strength=0.7,
            cutoff_frac=(0.0, 0.6),
            seed=args.seed,
            num_steps=args.num_steps,
            guidance_scale=args.guidance,
            height=args.height, width=args.width,
            device=args.device,
        )
        diff = region_diff(img_bike, img, bike_mask, h_lat, w_lat)
        scores[f"s=0.7 {lname} cut=0.6"] = diff
        results_c[lname] = (img, diff)
        save(img, f"expC_{lname}")
        print(f"    {lname}: bike_diff={diff:.2f}")

    img_tier_a, diff_tier_a = results_c["TIER_A"]
    img_all,    diff_all    = results_c["ALL_57"]
    save_grid(
        [chain_baseline, img_tier_a, img_all],
        [f"Baseline (no inject)\nbike_diff={bike_diff_base:.2f}",
         f"TIER_A (13 layers)\nbike_diff={diff_tier_a:.2f}",
         f"ALL 57 layers\nbike_diff={diff_all:.2f}"],
        os.path.join(args.out_dir, "expC_tier_a_vs_all.png"),
    )
    print("  Saved: expC_tier_a_vs_all.png")
    print("  Expected: ALL-57 produces near-pixel-identical copy (bike_diff≈0 but NO vase)")
    print("            TIER_A preserves bike better than baseline AND vase is present")

    # ----------------------------------------------------------
    # KEY RESULT: best config vs chain baseline
    # ----------------------------------------------------------
    print("\n=== KEY RESULT ===")
    # Best candidate: use the result from C TIER_A (s=0.7, cut=0.6)
    img_best = img_tier_a
    diff_best = diff_tier_a

    save_grid(
        [base, img_bike, chain_baseline, img_best],
        ["Base scene",
         "Step 1: bicycle added\n(standard chain)",
         f"Step 2 BASELINE\n(no injection)\nbike_diff={bike_diff_base:.2f}",
         f"Step 2 K/V INJECT\n(TIER_A, s=0.7, cut=0→60%)\nbike_diff={diff_best:.2f}"],
        os.path.join(args.out_dir, "KEY_RESULT.png"),
    )
    print("  Saved: KEY_RESULT.png")

    # ----------------------------------------------------------
    # Numeric summary
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print("BICYCLE IDENTITY PRESERVATION — SCORES TABLE")
    print(f"{'='*65}")
    print(f"  {'Config':<40}  {'bike_diff':>9}")
    print(f"  {'-'*40}  {'-'*9}")
    for k, v in sorted(scores.items(), key=lambda x: x[1]):
        marker = " ← BEST" if v == min(scores.values()) else ""
        print(f"  {k:<40}  {v:>9.2f}{marker}")
    print(f"  {'='*50}")
    print(f"  bike_diff: mean abs pixel diff (0-255) in bicycle region")
    print(f"  LOWER = bicycle better preserved vs img_bike")
    print()

    with open(os.path.join(args.out_dir, "scores.txt"), "w") as f:
        f.write("Bicycle identity preservation scores\n")
        f.write("bike_diff: lower = bicycle changed LESS from step-1 output\n\n")
        for k, v in sorted(scores.items(), key=lambda x: x[1]):
            f.write(f"{k:<45} {v:.2f}\n")
    print("  Saved: scores.txt")

    print(f"\n{'='*65}")
    print("WHAT TO CHECK")
    print(f"{'='*65}")
    print(f"""
bike_mask_overlay.png
  Orange tint = the bicycle region (as seen by the injection).
  Token grid = 64×64 (one token = 16×16 image pixels).
  Should cover the bicycle shape on the left wall — not the sofa or table.
  If too large (covering the whole room): increase --bike_threshold (try 60-80).
  If too small (only the frame center): lower --bike_threshold (try 25-30).

expA_strength_sweep.png  ← most important
  Left column: baseline (s=0.0, no injection).
  Right columns: increasing strength.
  For each result, check TWO things:
    1. Is the bicycle IDENTICAL (or near-identical) to step1_bicycle.png?
       YES → injection is working, bicycle preserved
       NO  → injection is too weak (increase s) or bike_mask is wrong
    2. Is the vase PRESENT on the coffee table?
       YES → the injection isn't suppressing the edit
       NO  → injection is too strong (decrease s), try s=0.5
  Success criterion: find s where BOTH are true.
  bike_diff < {bike_diff_base:.2f} (baseline) = better than standard chain.

expB_cutoff_sweep.png
  Does cutting injection off at step 40% instead of 60% still work?
  If bike_diff is similar: shorter cutoff is better (less interference).

expC_tier_a_vs_all.png  ← sanity check
  ALL_57 expected outcome: bike_diff≈0 (bicycle pixel-identical to step1)
  BUT: vase probably not present (injection so strong it blocks edit)
  TIER_A outcome: bike_diff < baseline but > 0, vase present
  This validates that TIER_A is the right layer set.

KEY_RESULT.png
  panel 3 (baseline) vs panel 4 (K/V injected):
  SUCCESS = panel 4 has the same vase AND the bicycle looks MORE like
            step1_bicycle.png than the baseline does.
  bike_diff lower than baseline = we beat the chain at bicycle preservation.
""")
    print(f"All results → {args.out_dir}/")


if __name__ == "__main__":
    main()
