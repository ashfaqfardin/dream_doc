"""
Phase 7 — Incremental Editing Pipeline

Full multi-step editing loop:
  Step 0: Generate base scene, cache K/V.
  Step N: Apply edit prompt, re-run with injection from Step N-1 cache,
           capture new K/V, save image, update cache.

Evaluates whether previous content is preserved across multiple edits.

Usage:
    python NewWork/KontextEval/phase7_pipeline.py \
        --hf_token hf_... \
        --cache_dir ./models \
        --out_dir results/phase7 \
        --alpha_k 0.5 --alpha_v 0.5
"""

from __future__ import annotations

import argparse
import json
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
from NewWork.KontextEval.utils.cache_utils import save_cache
from NewWork.KontextEval.utils.metrics import psnr, ssim, lpips_score, dino_similarity


# Default edit sequence from the Phase 7 spec.
DEFAULT_SEQUENCE = [
    {"step": 0, "prompt": "A modern living room",              "note": "base generation"},
    {"step": 1, "prompt": "Add a bicycle",                     "note": "object addition"},
    {"step": 2, "prompt": "Add a vase on the table",           "note": "object addition 2"},
    {"step": 3, "prompt": "Replace bicycle with a car",        "note": "object replacement"},
    {"step": 4, "prompt": "Change car color to red",           "note": "attribute change"},
    {"step": 5, "prompt": "Remove vase",                       "note": "object removal"},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",  required=True)
    p.add_argument("--cache_dir", default="./models")
    p.add_argument("--out_dir",   default="results/phase7")
    p.add_argument("--alpha_k",   type=float, default=0.5)
    p.add_argument("--alpha_v",   type=float, default=0.5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--guidance",  type=float, default=2.5)
    p.add_argument("--height",    type=int, default=1024)
    p.add_argument("--width",     type=int, default=1024)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def run_sequence(pipe, sequence, source, args, tag: str, use_injection: bool):
    """
    Run the full edit sequence.
    tag          : 'baseline' or 'method'
    use_injection: if True, inject K/V from previous step.
    Returns list of dicts with step info + image.
    """
    tr = pipe.transformer
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    cache = {}
    current_img = source
    step_log = []

    for step_info in sequence:
        step = step_info["step"]
        prompt = step_info["prompt"]

        print(f"  [{tag}] Step {step}: {prompt!r}")

        # If injection enabled and we have a cache from the previous step, inject.
        if use_injection and cache and step > 0:
            inj_orig = attach_inject_processors(
                tr, cache, alpha_k=args.alpha_k, alpha_v=args.alpha_v,
            )
        else:
            inj_orig = None

        # Also attach capture to update the cache after this step.
        new_cache = {}
        cap_orig = attach_capture_processors(tr, new_cache)

        out_img = generate(pipe, prompt, current_img,
                           seed=args.seed, num_steps=args.num_steps,
                           guidance_scale=args.guidance,
                           height=args.height, width=args.width)

        restore_processors(tr, cap_orig)
        if inj_orig:
            restore_processors(tr, inj_orig)

        # Update cache with this step's K/V.
        cache = new_cache
        current_img = out_img

        fpath = os.path.join(out_dir, f"step_{step:02d}.png")
        out_img.save(fpath)
        step_log.append({
            "step": step, "prompt": prompt, "image": out_img, "path": fpath,
        })

    return step_log


def compare_logs(log_base, log_method, args, step0_img):
    """Compute content-preservation metrics at each step."""
    results = []
    for b, m in zip(log_base, log_method):
        step = b["step"]
        m_psnr  = psnr(step0_img, m["image"])
        m_ssim  = ssim(step0_img, m["image"])
        m_lpips = lpips_score(step0_img, m["image"], device=args.device)
        m_dino  = dino_similarity(step0_img, m["image"], device=args.device)

        b_psnr  = psnr(step0_img, b["image"])
        b_dino  = dino_similarity(step0_img, b["image"], device=args.device)

        results.append({
            "step": step,
            "prompt": b["prompt"],
            "method_psnr":  m_psnr,  "baseline_psnr":  b_psnr,
            "method_ssim":  m_ssim,
            "method_lpips": m_lpips,
            "method_dino":  m_dino,  "baseline_dino":  b_dino,
        })
    return results


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    source = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    print(f"\n=== Baseline run (no injection) ===")
    log_baseline = run_sequence(pipe, DEFAULT_SEQUENCE, source, args,
                                tag="baseline", use_injection=False)

    print(f"\n=== Method run (K+V injection α_k={args.alpha_k} α_v={args.alpha_v}) ===")
    log_method = run_sequence(pipe, DEFAULT_SEQUENCE, source, args,
                              tag="method", use_injection=True)

    step0_img = log_baseline[0]["image"]

    print(f"\n=== Computing metrics ===")
    results = compare_logs(log_baseline, log_method, args, step0_img)

    # Print results table
    print("\n--- Phase 7 Sequential Robustness Results ---")
    print(f"{'Step':4s}  {'Prompt':35s}  "
          f"{'Base DINO':9s}  {'Meth DINO':9s}  "
          f"{'Base PSNR':9s}  {'Meth PSNR':9s}")
    print("-" * 90)
    for r in results:
        print(f"{r['step']:4d}  {r['prompt']:35s}  "
              f"{r['baseline_dino']:9.3f}  {r['method_dino']:9.3f}  "
              f"{r['baseline_psnr']:9.2f}  {r['method_psnr']:9.2f}")

    # Save JSON
    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "image"}
                   for r in results], f, indent=2)
    print(f"\nFull results → {json_path}")
    print(f"Images     → {args.out_dir}/baseline/  and  {args.out_dir}/method/")


if __name__ == "__main__":
    main()
