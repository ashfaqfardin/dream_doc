"""
Phase 3 — Attention Hooking

Verifies that the CaptureProcessor:
  1. Extracts Q, K, V for all hooked layers.
  2. Produces pixel-identical output to unhooked generation (same seed).
  3. Reports tensor shapes and any overhead (wall-clock time).

Usage:
    python NewWork/KontextEval/phase3_hooking.py \
        --hf_token hf_... \
        --cache_dir ./models \
        --out_dir results/phase3
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline, generate
from NewWork.KontextEval.utils.attention_utils import (
    attach_capture_processors, restore_processors,
)
from NewWork.KontextEval.utils.cache_utils import cache_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",  required=True)
    p.add_argument("--cache_dir", default="./models")
    p.add_argument("--out_dir",   default="results/phase3")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--guidance",  type=float, default=2.5)
    p.add_argument("--height",    type=int, default=1024)
    p.add_argument("--width",     type=int, default=1024)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def image_hash(img: Image.Image) -> str:
    return hashlib.md5(np.array(img.convert("RGB")).tobytes()).hexdigest()


def run_timed(pipe, prompt, source, seed, num_steps, guidance, height, width):
    t0 = time.time()
    img = generate(pipe, prompt, source, seed=seed, num_steps=num_steps,
                   guidance_scale=guidance, height=height, width=width)
    return img, time.time() - t0


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    tr = pipe.transformer

    prompt = "add a small potted plant on the windowsill"
    source = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    # --- Run WITHOUT hooks (baseline timing + hash) ---
    print("\n[1] Running WITHOUT hooks …")
    img_no_hook, t_baseline = run_timed(
        pipe, prompt, source, args.seed, args.num_steps,
        args.guidance, args.height, args.width,
    )
    h_baseline = image_hash(img_no_hook)
    img_no_hook.save(os.path.join(args.out_dir, "no_hook.png"))
    print(f"  Time  : {t_baseline:.1f}s")
    print(f"  Hash  : {h_baseline[:16]}")

    # --- Run WITH hooks (all double + single blocks) ---
    print("\n[2] Running WITH hooks (capture only, all blocks) …")
    store = {}
    originals = attach_capture_processors(tr, store)

    img_hooked, t_hooked = run_timed(
        pipe, prompt, source, args.seed, args.num_steps,
        args.guidance, args.height, args.width,
    )
    restore_processors(tr, originals)

    h_hooked = image_hash(img_hooked)
    img_hooked.save(os.path.join(args.out_dir, "with_hook.png"))
    print(f"  Time  : {t_hooked:.1f}s")
    print(f"  Hash  : {h_hooked[:16]}")

    pixel_identical = (h_baseline == h_hooked)
    overhead_pct    = (t_hooked - t_baseline) / t_baseline * 100
    print(f"\n  Pixel-identical : {'YES ✓' if pixel_identical else 'NO ✗'}")
    print(f"  Overhead        : {overhead_pct:+.1f}%")

    # --- Tensor shapes ---
    print("\n[3] Captured tensor inventory:")
    rows = cache_summary(store)
    print(f"  {'Key':30s}  {'Shape':30s}  {'MB':6s}")
    print("  " + "-" * 70)
    total_mb = 0.0
    for row in rows:
        shape_str = str(row["shape"])
        print(f"  {row['key']:30s}  {shape_str:30s}  {row['size_mb']:6.1f}")
        total_mb += row["size_mb"]

    print(f"\n  Total captured : {len(rows)} tensors  /  {total_mb:.0f} MB")
    print(f"  Keys captured  : {set(k.split('_')[0] + '_' + k.split('_')[1] for k in store if '_K' in k or '_V' in k)}")

    # --- Sample a few shapes for the report ---
    print("\n[4] Sample shapes (block 0 and last double block):")
    for k in ["double_0_K", "double_0_V", "double_0_Q",
              f"double_{len(tr.transformer_blocks)-1}_K",
              "single_0_K"]:
        if k in store:
            print(f"  {k:30s} : {tuple(store[k].shape)}")

    print(f"\nResults saved → {args.out_dir}")


if __name__ == "__main__":
    main()
