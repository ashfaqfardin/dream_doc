"""
Phase 8 — Adaptive Attention Preservation

Replaces the fixed α with a per-layer adaptive α computed from the
cosine similarity between current K and cached K.

High K-similarity → low α (content unchanged, allow free editing).
Low K-similarity  → high α (content has drifted, preserve more).

Compares fixed-α vs adaptive-α on the Phase 7 sequence.

Usage:
    python NewWork/KontextEval/phase8_adaptive.py \
        --hf_token hf_... \
        --cache_dir ./models \
        --out_dir results/phase8
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
    attach_capture_processors, attach_inject_processors,
    attach_adaptive_processors, restore_processors,
)
from NewWork.KontextEval.utils.metrics import psnr, dino_similarity, lpips_score


EDIT_SEQUENCE = [
    "A modern living room",
    "Add a bicycle",
    "Add a vase on the table",
    "Replace bicycle with a car",
    "Change car color to red",
    "Remove vase",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",   required=True)
    p.add_argument("--cache_dir",  default="./models")
    p.add_argument("--out_dir",    default="results/phase8")
    p.add_argument("--fixed_alpha",    type=float, default=0.5)
    p.add_argument("--adaptive_base",  type=float, default=0.5)
    p.add_argument("--adaptive_method", choices=["cosine", "entropy"], default="cosine")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--num_steps",  type=int, default=28)
    p.add_argument("--guidance",   type=float, default=2.5)
    p.add_argument("--height",     type=int, default=1024)
    p.add_argument("--width",      type=int, default=1024)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def run_adaptive_sequence(pipe, prompts, source, args, use_adaptive: bool,
                          tag: str):
    tr = pipe.transformer
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    cache = {}
    current_img = source
    alpha_log   = []

    for step, prompt in enumerate(prompts):
        print(f"  [{tag}] Step {step}: {prompt!r}")

        # Inject from previous cache.
        if cache and step > 0:
            if use_adaptive:
                inj_orig, processors = attach_adaptive_processors(
                    tr, cache, base_alpha=args.adaptive_base,
                    method=args.adaptive_method,
                )
            else:
                inj_orig = attach_inject_processors(
                    tr, cache, alpha_k=args.fixed_alpha, alpha_v=args.fixed_alpha,
                )
                processors = {}
        else:
            inj_orig = None
            processors = {}

        # Capture new K/V for next step.
        new_cache = {}
        cap_orig = attach_capture_processors(tr, new_cache)

        out_img = generate(pipe, prompt, current_img,
                           seed=args.seed, num_steps=args.num_steps,
                           guidance_scale=args.guidance,
                           height=args.height, width=args.width)

        restore_processors(tr, cap_orig)
        if inj_orig:
            restore_processors(tr, inj_orig)

        # Sample per-layer alpha from adaptive processors (for logging).
        if use_adaptive and processors:
            per_layer_alphas = {k: p.last_alpha_k for k, p in processors.items()}
            mean_alpha = sum(per_layer_alphas.values()) / max(len(per_layer_alphas), 1)
            alpha_log.append({"step": step, "mean_alpha": mean_alpha,
                               "per_layer": per_layer_alphas})
            print(f"    mean adaptive α = {mean_alpha:.3f}")
        else:
            alpha_log.append({"step": step, "mean_alpha": args.fixed_alpha})

        cache = new_cache
        current_img = out_img
        out_img.save(os.path.join(out_dir, f"step_{step:02d}.png"))

    return current_img, alpha_log


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    source = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    # Baseline (no injection)
    print("\n=== Baseline (no injection) ===")
    cache_b = {}
    current_b = source
    steps_b = []
    for step, prompt in enumerate(EDIT_SEQUENCE):
        new_c = {}
        orig = attach_capture_processors(pipe.transformer, new_c)
        out = generate(pipe, prompt, current_b, seed=args.seed,
                       num_steps=args.num_steps, guidance_scale=args.guidance,
                       height=args.height, width=args.width)
        restore_processors(pipe.transformer, orig)
        cache_b = new_c
        current_b = out
        out.save(os.path.join(args.out_dir, f"baseline_step_{step:02d}.png"))
        steps_b.append(out)
    base_final = steps_b[-1]

    # Fixed-α
    print(f"\n=== Fixed α = {args.fixed_alpha} ===")
    fixed_final, fixed_log = run_adaptive_sequence(
        pipe, EDIT_SEQUENCE, source, args, use_adaptive=False, tag="fixed_alpha"
    )

    # Adaptive-α
    print(f"\n=== Adaptive α (base={args.adaptive_base}, method={args.adaptive_method}) ===")
    adap_final, adap_log = run_adaptive_sequence(
        pipe, EDIT_SEQUENCE, source, args, use_adaptive=True, tag="adaptive"
    )

    # Reference: step 0 base image
    step0_img = steps_b[0]

    print("\n--- Phase 8 Comparison ---")
    metrics = {}
    for tag, final in [("baseline", base_final), ("fixed", fixed_final), ("adaptive", adap_final)]:
        m = {
            "psnr":  psnr(step0_img, final),
            "lpips": lpips_score(step0_img, final, device=args.device),
            "dino":  dino_similarity(step0_img, final, device=args.device),
        }
        metrics[tag] = m
        print(f"  {tag:10s}  PSNR={m['psnr']:.2f}  LPIPS={m['lpips']:.3f}  "
              f"DINOv2={m['dino']:.3f}")

    # Adaptive alpha progression
    if adap_log:
        print("\n  Adaptive α per step:")
        for entry in adap_log:
            s = entry["step"]
            ma = entry.get("mean_alpha", args.fixed_alpha)
            print(f"    Step {s}: mean α = {ma:.3f}")

    # Save results
    out = {"metrics": metrics, "adaptive_log": adap_log, "fixed_log": fixed_log}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
