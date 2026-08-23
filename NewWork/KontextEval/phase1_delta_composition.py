"""
Phase 1 — Latent Delta Composition (Clean-Pair Extraction)

The previous experiment (phase1_intuition_unverified.py) showed:
  - Delta arithmetic is circular when deltas come from the chain itself
  - BUT cross-context delta transfer works (Exp D)

This file tests the correct approach: extract each object's delta from a
CLEAN INDEPENDENT PAIR (base → single-object edit), then COMPOSE multiple
deltas together on the base latent.

Core question
-------------
If we generate:
  - base scene B
  - B + bicycle  (independent single-step edit)
  - B + vase     (independent single-step edit)

Can we compose both objects via:
  decode(z_B + delta_bicycle + delta_vase)
...WITHOUT chaining (and without K/V injection)?

This directly addresses the Phase 5 problem: K/V injection was overriding
the "add bicycle" edit. The delta approach avoids denoising-time intervention
entirely — each edit is generated cleanly, composition is post-hoc.

Experiments
-----------
1. Two-object composition:  base + delta_bicycle + delta_vase
2. Three-object composition: + delta_plant (tests delta accumulation limit)
3. Object removal:  composed − delta_vase  (should show bicycle only)
4. Cross-background transfer: apply delta_bicycle to a different base scene
5. Chainwise baseline: compare with base → bike → vase chained Kontext edits

Usage
-----
python NewWork/KontextEval/phase1_delta_composition.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_delta_comp
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

OBJECT_PROMPTS = {
    "bicycle": (
        "Add a yellow bicycle leaning against the wall on the left side. "
        "Keep the rest of the room exactly the same."
    ),
    "vase": (
        "Add a white ceramic vase with flowers on the coffee table. "
        "Keep the rest of the room exactly the same."
    ),
    "plant": (
        "Add a tall green potted plant in the right corner. "
        "Keep the rest of the room exactly the same."
    ),
}

ALT_BASE_PROMPT = "A modern living room with a grey sofa and a dark wooden coffee table."


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


def delta_norm(d: torch.Tensor) -> float:
    return d.float().norm().item() / d.numel() ** 0.5


# ============================================================
# Comparison grid
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
    p.add_argument("--hf_token",  required=True)
    p.add_argument("--cache_dir", default="./models")
    p.add_argument("--out_dir",   default="results/phase1_delta_comp")
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

    def save(img, name):
        path = os.path.join(args.out_dir, f"{name}.png")
        img.save(path)
        return path

    def run(src_img, prompt_text, seed=None):
        return generate(
            pipe, prompt_text, src_img,
            seed=seed if seed is not None else args.seed,
            num_steps=args.num_steps, guidance_scale=args.guidance,
            height=args.height, width=args.width,
        )

    print("Loading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    log = []

    # ----------------------------------------------------------
    # Step 0: Generate clean base scene
    # ----------------------------------------------------------
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run(grey, BASE_PROMPT)
    save(base, "base")
    z_base = vae_encode(pipe, base)
    print(f"  z_base shape: {tuple(z_base.shape)}")
    log.append(f"z_base norm: {delta_norm(z_base):.5f}")

    # ----------------------------------------------------------
    # Step 1: Generate each object independently (clean pairs)
    # ----------------------------------------------------------
    print("\n=== Step 1: Independent clean-pair generation ===")
    clean_imgs = {}
    deltas = {}

    for obj_name, prompt in OBJECT_PROMPTS.items():
        print(f"  Generating: {obj_name}")
        img = run(base, prompt)
        save(img, f"clean_{obj_name}")
        z = vae_encode(pipe, img)
        delta = z - z_base
        deltas[obj_name] = delta
        clean_imgs[obj_name] = img
        n = delta_norm(delta)
        log.append(f"delta_{obj_name} norm: {n:.5f}")
        print(f"    delta_norm = {n:.5f}")

    # ----------------------------------------------------------
    # Experiment 1: Two-object delta composition
    #   base + delta_bicycle + delta_vase
    # ----------------------------------------------------------
    print("\n=== Experiment 1: Two-object composition (bicycle + vase) ===")
    z_2obj = z_base + deltas["bicycle"] + deltas["vase"]
    img_2obj = vae_decode(pipe, z_2obj)
    save(img_2obj, "compose_bicycle_vase")

    save_grid(
        [base, clean_imgs["bicycle"], clean_imgs["vase"], img_2obj],
        ["Base scene", "Base + bicycle\n(clean single edit)",
         "Base + vase\n(clean single edit)", "COMPOSED\nbase + δbicycle + δvase"],
        os.path.join(args.out_dir, "exp1_two_object_composition.png"),
    )
    print("  Saved: exp1_two_object_composition.png")
    print("  KEY: Does compose_bicycle_vase show BOTH objects?")

    # ----------------------------------------------------------
    # Experiment 2: Three-object delta composition
    # ----------------------------------------------------------
    print("\n=== Experiment 2: Three-object composition (bicycle + vase + plant) ===")
    z_3obj = z_base + deltas["bicycle"] + deltas["vase"] + deltas["plant"]
    img_3obj = vae_decode(pipe, z_3obj)
    save(img_3obj, "compose_bicycle_vase_plant")

    save_grid(
        [clean_imgs["bicycle"], clean_imgs["vase"], clean_imgs["plant"], img_3obj],
        ["Base + bicycle", "Base + vase", "Base + plant",
         "COMPOSED\nbase + δbike + δvase + δplant"],
        os.path.join(args.out_dir, "exp2_three_object_composition.png"),
    )
    print("  Saved: exp2_three_object_composition.png")
    print("  KEY: Do all three objects appear? Does quality degrade with 3 deltas?")

    # ----------------------------------------------------------
    # Experiment 3: Object removal
    #   Start from two-object composition, subtract vase delta
    # ----------------------------------------------------------
    print("\n=== Experiment 3: Object removal (composed − delta_vase) ===")
    z_remove_vase = z_2obj - deltas["vase"]
    img_remove_vase = vae_decode(pipe, z_remove_vase)
    save(img_remove_vase, "compose_remove_vase")

    # Also try: composed − delta_bicycle
    z_remove_bike = z_2obj - deltas["bicycle"]
    img_remove_bike = vae_decode(pipe, z_remove_bike)
    save(img_remove_bike, "compose_remove_bicycle")

    save_grid(
        [img_2obj, img_remove_vase, img_remove_bike, base],
        ["Composed\n(bike + vase)", "Vase REMOVED\n(− δvase)",
         "Bicycle REMOVED\n(− δbicycle)", "Original base\n(ground truth)"],
        os.path.join(args.out_dir, "exp3_object_removal.png"),
    )
    print("  Saved: exp3_object_removal.png")
    print("  KEY: Does subtraction cleanly remove the target object?")
    print("       Does it look like the original base?")

    # ----------------------------------------------------------
    # Experiment 4: Cross-background delta transfer
    #   Generate a DIFFERENT base scene, transfer bicycle delta onto it
    # ----------------------------------------------------------
    print("\n=== Experiment 4: Cross-background delta transfer ===")
    print("  Generating alternative base scene …")
    alt_base = run(grey, ALT_BASE_PROMPT, seed=args.seed + 1)
    save(alt_base, "alt_base")
    z_alt = vae_encode(pipe, alt_base)

    z_alt_bike = z_alt + deltas["bicycle"]
    img_alt_bike = vae_decode(pipe, z_alt_bike)
    save(img_alt_bike, "alt_base_with_bicycle_delta")

    # For comparison: directly edit alt_base with Kontext
    img_alt_bike_direct = run(alt_base, OBJECT_PROMPTS["bicycle"])
    save(img_alt_bike_direct, "alt_base_with_bicycle_direct")

    save_grid(
        [base, clean_imgs["bicycle"], alt_base, img_alt_bike, img_alt_bike_direct],
        ["Original base", "Original base\n+ bicycle (clean)",
         "Alternative base\n(different scene)",
         "Alt base\n+ δbicycle (TRANSFER)",
         "Alt base\n+ bicycle (Kontext direct)"],
        os.path.join(args.out_dir, "exp4_cross_background_transfer.png"),
    )
    print("  Saved: exp4_cross_background_transfer.png")
    print("  KEY: Transfer vs direct — does the bicycle appear in the right place?")
    print("       Is transfer quality comparable to direct Kontext edit?")

    # ----------------------------------------------------------
    # Experiment 5: Chainwise baseline (for comparison)
    #   The normal way: base → add bicycle → add vase (Kontext chain)
    # ----------------------------------------------------------
    print("\n=== Experiment 5: Chainwise baseline ===")
    chain_bike = run(base, OBJECT_PROMPTS["bicycle"])
    save(chain_bike, "chain_step1_bicycle")
    chain_bike_vase = run(chain_bike, OBJECT_PROMPTS["vase"])
    save(chain_bike_vase, "chain_step2_bicycle_vase")

    save_grid(
        [base, chain_bike, chain_bike_vase, img_2obj],
        ["Base", "Chain step 1\n(add bicycle)",
         "Chain step 2\n(add vase to step1)",
         "DELTA COMPOSE\n(base + δbike + δvase)"],
        os.path.join(args.out_dir, "exp5_chain_vs_compose.png"),
    )
    print("  Saved: exp5_chain_vs_compose.png")
    print("  KEY: Compare chainwise (step2) vs delta composed (last panel).")
    print("       Does delta composition show both objects more cleanly?")

    # ----------------------------------------------------------
    # Final summary grid
    # ----------------------------------------------------------
    save_grid(
        [base,
         clean_imgs["bicycle"], clean_imgs["vase"], clean_imgs["plant"],
         img_2obj, img_3obj,
         img_remove_vase, img_alt_bike, chain_bike_vase],
        ["Base scene",
         "Clean: + bicycle", "Clean: + vase", "Clean: + plant",
         "Compose: bike+vase", "Compose: bike+vase+plant",
         "Remove vase\nfrom compose", "Transfer\nbike to alt base",
         "Chain baseline\n(bike→vase)"],
        os.path.join(args.out_dir, "FULL_SUMMARY.png"),
        ncols=3,
    )
    print("\n  Saved: FULL_SUMMARY.png (3×3 grid of all experiments)")

    # ----------------------------------------------------------
    # Save log
    # ----------------------------------------------------------
    log_path = os.path.join(args.out_dir, "delta_norms.txt")
    with open(log_path, "w") as f:
        f.write("Delta norms (L2 / sqrt(N), lower = smaller edit)\n")
        f.write("=" * 50 + "\n")
        for line in log:
            f.write(line + "\n")
        f.write("\nNote: comparable norms across objects means the VAE\n")
        f.write("      represents their edits at similar magnitudes.\n")
        f.write("Large norms indicate globally disruptive edits.\n")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("WHAT TO CHECK IN RESULTS")
    print(f"{'='*60}")
    print("""
Exp 1 — exp1_two_object_composition.png:
  Do BOTH bicycle and vase appear in the composed image?
  → YES: Delta composition works for multi-object addition
  → NO:  Objects cancel or blur — VAE deltas interfere

Exp 2 — exp2_three_object_composition.png:
  Do all THREE objects appear? Does quality drop with 3 deltas?
  → Expect: some ghosting/blending if 3 deltas compete for space

Exp 3 — exp3_object_removal.png:
  After removing vase from composition, does only bicycle remain?
  Does removing bicycle leave only vase?
  → Clean removal: delta subtraction is reversible
  → Residuals remain: deltas not fully orthogonal

Exp 4 — exp4_cross_background_transfer.png:
  Does the bicycle appear in the alternative scene via delta transfer?
  Compare transfer result vs direct Kontext edit — which is better?
  → Transfer faster (no denoising) but may misplace the object

Exp 5 — exp5_chain_vs_compose.png:
  Compare the rightmost two panels (chain vs delta compose).
  Chain: both objects present but step 2 may change step 1 slightly.
  Delta: both objects from completely independent generations.
  → Key question: does delta compose preserve object quality better?
""")
    print(f"All results in: {args.out_dir}/")


if __name__ == "__main__":
    main()
