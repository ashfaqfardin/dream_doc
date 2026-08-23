"""
Phase 5 — Attention Injection Prototype

Tests K-only, V-only, and K+V injection at α ∈ {0.25, 0.50, 0.75, 1.00}
on all double-stream blocks. Saves a grid of output images for visual comparison.

Workflow:
  1. Generate a base scene (Step 0), capture K/V.
  2. Apply an edit prompt (Step 1) WITHOUT injection  → baseline.
  3. Apply the same edit WITH K/V injection at each α → method outputs.
  4. Save all images and print a comparison table.

Usage:
    python NewWork/KontextEval/phase5_injection.py \
        --hf_token hf_... \
        --cache_dir ./models \
        --out_dir results/phase5 \
        --base_prompt "a modern living room with a sofa" \
        --edit_prompt "add a yellow bicycle in the living room"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline, generate
from NewWork.KontextEval.utils.attention_utils import (
    attach_capture_processors, attach_inject_processors, restore_processors,
)
from NewWork.KontextEval.utils.cache_utils import cache_summary


ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00]
EXPERIMENTS = [
    ("K_only",  True,  False),
    ("V_only",  False, True),
    ("K_and_V", True,  True),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",    required=True)
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--out_dir",     default="results/phase5")
    p.add_argument("--base_prompt", default="a modern living room with a sofa and a coffee table")
    p.add_argument("--edit_prompt", default="add a yellow bicycle leaning against the wall")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_steps",   type=int, default=28)
    p.add_argument("--guidance",    type=float, default=2.5)
    p.add_argument("--height",      type=int, default=1024)
    p.add_argument("--width",       type=int, default=1024)
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def make_grid(images: list[Image.Image], labels: list[str],
              cols: int = 4) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (w * cols, h * rows), (20, 20, 20))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * h))
    return grid


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    tr = pipe.transformer
    source = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    # --- Step 0: generate base scene and cache K/V ---
    print(f"\n[0] Base generation — {args.base_prompt!r}")
    store = {}
    originals = attach_capture_processors(tr, store)
    base_img = generate(pipe, args.base_prompt, source,
                        seed=args.seed, num_steps=args.num_steps,
                        guidance_scale=args.guidance,
                        height=args.height, width=args.width)
    restore_processors(tr, originals)
    base_img.save(os.path.join(args.out_dir, "step0_base.png"))
    print(f"  Cached {len(store)} tensors")

    # --- Step 1: baseline edit (no injection, α=0) ---
    print(f"\n[1] Baseline edit (no injection) — {args.edit_prompt!r}")
    baseline = generate(pipe, args.edit_prompt, base_img,
                        seed=args.seed, num_steps=args.num_steps,
                        guidance_scale=args.guidance,
                        height=args.height, width=args.width)
    baseline.save(os.path.join(args.out_dir, "step1_baseline.png"))

    # --- Experiments ---
    all_images = [base_img, baseline]
    all_labels = ["base", "baseline (α=0)"]
    summary_rows = []

    for exp_name, use_k, use_v in EXPERIMENTS:
        exp_dir = os.path.join(args.out_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\n[Exp: {exp_name}]")

        for alpha in ALPHAS:
            if alpha == 0.0:
                continue  # already have baseline

            alpha_k = alpha if use_k else 0.0
            alpha_v = alpha if use_v else 0.0

            # Attach injection processors
            inj_originals = attach_inject_processors(
                tr, store, alpha_k=alpha_k, alpha_v=alpha_v
            )
            out = generate(pipe, args.edit_prompt, base_img,
                           seed=args.seed, num_steps=args.num_steps,
                           guidance_scale=args.guidance,
                           height=args.height, width=args.width)
            restore_processors(tr, inj_originals)

            label = f"{exp_name}_α={alpha:.2f}"
            fpath = os.path.join(exp_dir, f"alpha_{alpha:.2f}.png")
            out.save(fpath)
            all_images.append(out)
            all_labels.append(label)
            print(f"  α={alpha:.2f}  →  {fpath}")
            summary_rows.append({"exp": exp_name, "alpha": alpha, "path": fpath})

    # --- Save comparison grid ---
    grid = make_grid(all_images, all_labels, cols=5)
    grid_path = os.path.join(args.out_dir, "comparison_grid.png")
    grid.save(grid_path)
    print(f"\nComparison grid → {grid_path}")

    # --- Print summary table ---
    print("\n--- Phase 5 Summary ---")
    print(f"  Base prompt : {args.base_prompt!r}")
    print(f"  Edit prompt : {args.edit_prompt!r}")
    print(f"\n  {'Experiment':12s}  {'alpha':5s}  Path")
    print("  " + "-" * 60)
    for r in summary_rows:
        print(f"  {r['exp']:12s}  {r['alpha']:.2f}   {r['path']}")


if __name__ == "__main__":
    main()
