"""
Phase 1 — Two-Stage Latent Delta + Kontext Refinement

Previous finding (phase1_delta_composition.py):
  - Delta composition correctly places objects (positions are accurate)
  - But raw VAE latent arithmetic produces blurry output (out-of-distribution latent)
  - Object removal works cleanly (subtraction is reversible)

This file adds Stage 2: a single Kontext denoising pass on the blurry composite.

Two-stage pipeline
------------------
  Stage 1 — Delta composition (layout):
    z_composed = z_base + δ_bicycle + δ_vase      (blurry but correct positions)

  Stage 2 — Kontext refinement (quality):
    input:  decode(z_composed)                     (blurry composite)
    prompt: full scene description with all objects
    output: sharp, realistic image with both objects

Hypothesis: Kontext sees the blurry layout and sharpens it without
repositioning objects, because it is trained to edit/improve input images.

Key comparisons
---------------
  A. Chainwise baseline:  base → add_bicycle → add_vase    (2 Kontext calls)
  B. Direct single call:  base → "add bicycle and vase"    (1 Kontext call)
  C. Two-stage:           base → compose → refine          (3 gen calls total)
  D. Refinement step sweep: 5 / 10 / 20 / 28 steps        (less = less change)
  E. Refinement guidance:   1.5 / 2.5 / 3.5               (lower = stay closer to input)

Usage
-----
python NewWork/KontextEval/phase1_delta_refine.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_refine
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

BOTH_PROMPT_DIRECT = (
    "Add a yellow bicycle leaning against the left wall AND a white ceramic "
    "vase with flowers on the coffee table. Keep the rest of the room unchanged."
)

# Refinement prompt: describes the FULL desired scene (not an edit instruction)
REFINE_PROMPT = (
    "A modern living room with a sofa, a wooden coffee table, a yellow bicycle "
    "leaning against the left wall, and a white ceramic vase with flowers on "
    "the coffee table. Photorealistic, sharp, high quality."
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
# Grid save helper
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
    p.add_argument("--hf_token",   required=True)
    p.add_argument("--cache_dir",  default="./models")
    p.add_argument("--out_dir",    default="results/phase1_refine")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--num_steps",  type=int,   default=28)
    p.add_argument("--guidance",   type=float, default=2.5)
    p.add_argument("--refine_steps",  type=int,   default=10,
                   help="Default steps for the refinement pass (fewer = less change)")
    p.add_argument("--refine_guidance", type=float, default=2.0,
                   help="Guidance for the refinement pass (lower = stay closer to blurry input)")
    p.add_argument("--height",     type=int,   default=1024)
    p.add_argument("--width",      type=int,   default=1024)
    p.add_argument("--device",     default="cuda")
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
    # Shared setup: base + clean single-object edits + deltas
    # ----------------------------------------------------------
    print("\n=== Setup: base scene + clean object pairs ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    print("  Generating base scene …")
    base = run(grey, BASE_PROMPT)
    save(base, "base")
    z_base = vae_encode(pipe, base)

    print("  Generating base + bicycle (clean) …")
    img_bike = run(base, BICYCLE_PROMPT)
    save(img_bike, "clean_bicycle")
    delta_bike = vae_encode(pipe, img_bike) - z_base

    print("  Generating base + vase (clean) …")
    img_vase = run(base, VASE_PROMPT)
    save(img_vase, "clean_vase")
    delta_vase = vae_encode(pipe, img_vase) - z_base

    # Stage 1: compose deltas → blurry composite
    z_composed = z_base + delta_bike + delta_vase
    img_blurry = vae_decode(pipe, z_composed)
    save(img_blurry, "stage1_blurry_composite")
    print("  Stage 1 done: blurry composite saved")

    # ----------------------------------------------------------
    # Experiment A: chainwise baseline
    # ----------------------------------------------------------
    print("\n=== Experiment A: Chainwise baseline ===")
    chain_bike = run(base, BICYCLE_PROMPT)
    save(chain_bike, "chainA_step1_bicycle")
    chain_both = run(chain_bike, VASE_PROMPT)
    save(chain_both, "chainA_step2_both")
    print("  Done")

    # ----------------------------------------------------------
    # Experiment B: direct single-call (Kontext, both objects at once)
    # ----------------------------------------------------------
    print("\n=== Experiment B: Direct single call (add both at once) ===")
    direct_both = run(base, BOTH_PROMPT_DIRECT)
    save(direct_both, "directB_both_objects")
    print("  Done")

    # ----------------------------------------------------------
    # Experiment C: Two-stage (compose + refine)
    # ----------------------------------------------------------
    print("\n=== Experiment C: Two-stage — compose + refine ===")
    print(f"  Refinement: {args.refine_steps} steps, guidance {args.refine_guidance}")

    # Stage 2: Kontext sees blurry composite, sharpens it
    img_refined = run(
        img_blurry, REFINE_PROMPT,
        steps=args.refine_steps,
        guidance=args.refine_guidance,
    )
    save(img_refined, "stageC_refined")
    print("  Done")

    # Save the 4-panel comparison (key result)
    save_grid(
        [base, img_blurry, img_refined, chain_both],
        ["Base scene",
         "Stage 1\n(blurry composite: δbike+δvase)",
         f"Stage 2 — REFINED\n({args.refine_steps} steps, g={args.refine_guidance})",
         "Chain baseline\n(base→bike→vase)"],
        os.path.join(args.out_dir, "expC_main_comparison.png"),
    )
    print("  Saved: expC_main_comparison.png")
    print("  KEY: Is Stage 2 output sharper than Stage 1 blurry composite?")
    print("       Does it match or exceed chain baseline quality?")
    print("       Are both bicycle AND vase present and correctly placed?")

    # ----------------------------------------------------------
    # Experiment D: Refinement step count sweep
    # ----------------------------------------------------------
    print("\n=== Experiment D: Refinement step count sweep ===")
    step_counts = [5, 10, 20, 28]
    step_imgs = []
    for n_steps in step_counts:
        print(f"  Refining with {n_steps} steps …")
        img_s = run(img_blurry, REFINE_PROMPT,
                    steps=n_steps, guidance=args.refine_guidance)
        save(img_s, f"expD_refine_{n_steps}steps")
        step_imgs.append(img_s)

    save_grid(
        [img_blurry] + step_imgs,
        ["Blurry input\n(Stage 1)"] +
        [f"{n} steps\ng={args.refine_guidance}" for n in step_counts],
        os.path.join(args.out_dir, "expD_step_sweep.png"),
    )
    print("  Saved: expD_step_sweep.png")
    print("  KEY: Fewer steps = stays closer to blurry layout")
    print("       More steps = sharper but may reposition objects")

    # ----------------------------------------------------------
    # Experiment E: Refinement guidance sweep
    # ----------------------------------------------------------
    print("\n=== Experiment E: Refinement guidance sweep ===")
    guidances = [1.0, 2.0, 3.0, 4.0]
    guidance_imgs = []
    for g in guidances:
        print(f"  Refining with guidance {g} …")
        img_g = run(img_blurry, REFINE_PROMPT,
                    steps=args.refine_steps, guidance=g)
        save(img_g, f"expE_refine_g{g:.1f}")
        guidance_imgs.append(img_g)

    save_grid(
        [img_blurry] + guidance_imgs,
        ["Blurry input\n(Stage 1)"] +
        [f"guidance={g}\n{args.refine_steps} steps" for g in guidances],
        os.path.join(args.out_dir, "expE_guidance_sweep.png"),
    )
    print("  Saved: expE_guidance_sweep.png")
    print("  KEY: Low guidance (1.0–2.0) = closer to input, less prompt adherence")
    print("       High guidance (3.0–4.0) = follows prompt more, may drift from layout")

    # ----------------------------------------------------------
    # Full side-by-side: all methods
    # ----------------------------------------------------------
    save_grid(
        [base, img_blurry, img_refined, direct_both, chain_both],
        ["Base", "Stage 1\n(blurry)", "Stage 2\n(refined)",
         "Direct\n(single call)", "Chain\n(2 calls)"],
        os.path.join(args.out_dir, "FULL_METHOD_COMPARISON.png"),
    )
    print("\n  Saved: FULL_METHOD_COMPARISON.png  ← main result")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("WHAT TO CHECK")
    print(f"{'='*60}")
    print("""
expC_main_comparison.png  ← most important
  Panel 3 (Stage 2 refined) vs Panel 4 (chain baseline):
  - Are both objects present and sharp in the refined output?
  - Does it look comparable to the chain baseline?
  → YES: 2-stage pipeline is viable. Positions from delta, quality from Kontext.
  → NO:  Refinement ignores the blurry layout and regenerates freely.

expD_step_sweep.png
  Which step count best preserves layout while removing blur?
  - 5–10 steps: subtle sharpening, layout almost unchanged
  - 20–28 steps: sharper but Kontext may adjust object positions
  → Optimal is the lowest step count that removes visible blur.

expE_guidance_sweep.png
  Which guidance best sharpens without drifting from the blurry layout?
  - g=1.0: very conservative, may not fully sharpen
  - g=2.0–3.0: good balance (try these first)
  - g=4.0: may override layout with its own composition

FULL_METHOD_COMPARISON.png
  Left to right: base → blurry → refined → direct → chain
  For the dissertation: if refined ≈ chain quality AND both objects present,
  the 2-stage method is a working alternative to K/V injection.
""")
    print(f"All results → {args.out_dir}/")


if __name__ == "__main__":
    main()
