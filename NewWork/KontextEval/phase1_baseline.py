"""
Phase 1 — Environment & Baseline (Chainwise Incremental Editing)

Each editing step feeds its output directly into the next step as the input
image, building an incremental chain:

  Step 0: empty grey scene  (base)
  Step 1: add wooden chair          ← input: step 0
  Step 2: replace chair → iron      ← input: step 1
  Step 3: change color → red        ← input: step 2
  Step 4: style → oil painting      ← input: step 3
  Step 5: deterministic verification (same prompt + seed → same pixels)

Usage:
    python NewWork/KontextEval/phase1_baseline.py \
        --hf_token hf_... \
        --cache_dir ./models \
        --out_dir results/phase1
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — saves to PNG instead of showing
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 — Chainwise baseline")
    p.add_argument("--hf_token",  required=True)
    p.add_argument("--cache_dir", default="./models")
    p.add_argument("--out_dir",   default="results/phase1")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--guidance",  type=float, default=3.5)
    p.add_argument("--size",      type=int, default=768)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


# ============================================================
# Utility Functions
# ============================================================

def show_images(images, titles, save_path: str):
    """Save a side-by-side comparison to `save_path`."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def image_hash(image: Image.Image) -> str:
    """Generate MD5 hash for deterministic comparison."""
    return hashlib.md5(image.convert("RGB").tobytes()).hexdigest()


def run_edit(pipe, image: Image.Image, prompt: str,
             seed: int = 42, num_steps: int = 30,
             guidance: float = 3.5, size: int = 768) -> Image.Image:
    generator = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        image=image,
        prompt=prompt,
        generator=generator,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        height=size,
        width=size,
    ).images[0]
    return result


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    print(f"  Double blocks : {len(pipe.transformer.transformer_blocks)}")
    print(f"  Single blocks : {len(pipe.transformer.single_transformer_blocks)}")

    def save(img, name):
        path = os.path.join(args.out_dir, f"{name}.png")
        img.save(path)
        return path

    # ============================================================
    # Step 0: Initial Scene
    # ============================================================

    print("\n===== Step 0: Initial Scene =====")

    base_image = Image.new("RGB", (args.size, args.size), color=(120, 120, 120))
    save(base_image, "step0_empty_scene")

    show_images(
        [base_image],
        ["Initial Empty Scene"],
        save_path=os.path.join(args.out_dir, "step0_compare.png"),
    )

    # ============================================================
    # Step 1: Add Chair
    # ============================================================

    print("\n===== Step 1: Object Addition (Chair) =====")

    chair_add_prompt = (
        "Add a wooden chair in the center of the room. "
        "Preserve the existing background and composition."
    )

    chair_image = run_edit(
        pipe, base_image, chair_add_prompt,
        seed=args.seed, num_steps=args.num_steps,
        guidance=args.guidance, size=args.size,
    )
    save(chair_image, "step1_add_wooden_chair")

    show_images(
        [base_image, chair_image],
        ["Before Adding Chair", "Wooden Chair Added"],
        save_path=os.path.join(args.out_dir, "step1_compare.png"),
    )
    print(f"  Saved step1_add_wooden_chair.png")

    # ============================================================
    # Step 2: Replace Chair
    # ============================================================

    print("\n===== Step 2: Object Replacement (Chair Modification) =====")

    chair_replace_prompt = (
        "Replace the wooden chair with a modern iron chair. "
        "Keep the chair position, size, and background unchanged."
    )

    iron_chair_image = run_edit(
        pipe, chair_image, chair_replace_prompt,
        seed=args.seed, num_steps=args.num_steps,
        guidance=args.guidance, size=args.size,
    )
    save(iron_chair_image, "step2_replace_iron_chair")

    show_images(
        [chair_image, iron_chair_image],
        ["Wooden Chair", "Iron Chair"],
        save_path=os.path.join(args.out_dir, "step2_compare.png"),
    )
    print(f"  Saved step2_replace_iron_chair.png")

    # ============================================================
    # Step 3: Change Chair Color
    # ============================================================

    print("\n===== Step 3: Object Attribute Editing =====")

    chair_color_prompt = (
        "Change the iron chair color to red. "
        "Preserve the chair shape, position, and surrounding scene."
    )

    red_chair_image = run_edit(
        pipe, iron_chair_image, chair_color_prompt,
        seed=args.seed, num_steps=args.num_steps,
        guidance=args.guidance, size=args.size,
    )
    save(red_chair_image, "step3_change_chair_color")

    show_images(
        [iron_chair_image, red_chair_image],
        ["Iron Chair", "Red Iron Chair"],
        save_path=os.path.join(args.out_dir, "step3_compare.png"),
    )
    print(f"  Saved step3_change_chair_color.png")

    # ============================================================
    # Step 4: Style Change
    # ============================================================

    print("\n===== Step 4: Style Transfer =====")

    style_prompt = (
        "Transform the entire image into an oil painting style. "
        "Preserve the chair identity and scene composition."
    )

    style_image = run_edit(
        pipe, red_chair_image, style_prompt,
        seed=args.seed, num_steps=args.num_steps,
        guidance=args.guidance, size=args.size,
    )
    save(style_image, "step4_style_change")

    show_images(
        [red_chair_image, style_image],
        ["Original Style", "Oil Painting Style"],
        save_path=os.path.join(args.out_dir, "step4_compare.png"),
    )
    print(f"  Saved step4_style_change.png")

    # ============================================================
    # Step 5: Deterministic Verification
    # ============================================================

    print("\n===== Step 5: Deterministic Seed Verification =====")

    test_prompt = (
        "Add a wooden chair in the center of the room. "
        "Preserve the original background and composition."
    )

    run_1 = run_edit(
        pipe, base_image, test_prompt,
        seed=1234, num_steps=args.num_steps,
        guidance=args.guidance, size=args.size,
    )

    run_2 = run_edit(
        pipe, base_image, test_prompt,
        seed=1234, num_steps=args.num_steps,
        guidance=args.guidance, size=args.size,
    )

    hash_1 = image_hash(run_1)
    hash_2 = image_hash(run_2)

    print(f"Output 1 Hash: {hash_1}")
    print(f"Output 2 Hash: {hash_2}")

    if hash_1 == hash_2:
        print("✅ Deterministic: Same seed produces identical output")
    else:
        print("❌ Non-deterministic: Outputs differ")

    save(run_1, "deterministic_run_1")
    save(run_2, "deterministic_run_2")

    show_images(
        [run_1, run_2],
        [f"Seed 1234 Run 1\n{hash_1[:12]}", f"Seed 1234 Run 2\n{hash_2[:12]}"],
        save_path=os.path.join(args.out_dir, "step5_determinism.png"),
    )

    # ============================================================
    # Full chain grid
    # ============================================================

    show_images(
        [base_image, chair_image, iron_chair_image, red_chair_image, style_image],
        ["Step 0\nBase", "Step 1\nAdd Chair", "Step 2\nIron Chair",
         "Step 3\nRed Chair", "Step 4\nOil Painting"],
        save_path=os.path.join(args.out_dir, "full_chain.png"),
    )

    # ============================================================
    # Summary
    # ============================================================

    print(f"\n✅ Phase 1 Incremental Editing Baseline Completed")
    print(f"   Results → {args.out_dir}/")
    print(f"   Chain overview → {args.out_dir}/full_chain.png")
    print(f"   Deterministic: {'YES' if hash_1 == hash_2 else 'NO'}")


if __name__ == "__main__":
    main()
