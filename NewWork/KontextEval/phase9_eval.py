"""
Phase 9 — Evaluation & Dissertation Results

Runs the full evaluation suite over the Phase 7 output images:
  Content preservation : LPIPS, DINOv2, PSNR, SSIM
  Edit success         : CLIP direction similarity
  Sequential robustness: per-step drift across 5+ edits

Compares:
  Baseline 1: FLUX.1-Kontext native  (no injection)
  Baseline 2: K-only injection
  Baseline 3: V-only injection
  Baseline 4: K+V injection (fixed α)
  Baseline 5: K+V adaptive α

Usage:
    python NewWork/KontextEval/phase9_eval.py \
        --hf_token hf_... \
        --cache_dir ./models \
        --out_dir results/phase9
"""

from __future__ import annotations

import argparse
import csv
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
from NewWork.KontextEval.utils.metrics import (
    psnr, ssim, lpips_score, dino_similarity,
    clip_direction_similarity, evaluate_edit,
)


EDIT_SEQUENCE = [
    {"step": 0, "prompt": "A modern living room",
     "prev_prompt": ""},
    {"step": 1, "prompt": "Add a bicycle in the living room",
     "prev_prompt": "A modern living room"},
    {"step": 2, "prompt": "Add a vase on the table",
     "prev_prompt": "Add a bicycle in the living room"},
    {"step": 3, "prompt": "Replace the bicycle with a car",
     "prev_prompt": "Add a vase on the table"},
    {"step": 4, "prompt": "Change the car color to red",
     "prev_prompt": "Replace the bicycle with a car"},
    {"step": 5, "prompt": "Remove the vase",
     "prev_prompt": "Change the car color to red"},
]


METHODS = {
    "native":    {"alpha_k": 0.0, "alpha_v": 0.0},
    "K_only":    {"alpha_k": 0.5, "alpha_v": 0.0},
    "V_only":    {"alpha_k": 0.0, "alpha_v": 0.5},
    "K_and_V":   {"alpha_k": 0.5, "alpha_v": 0.5},
    "adaptive":  {"alpha_k": None, "alpha_v": None},   # None → adaptive
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",   required=True)
    p.add_argument("--cache_dir",  default="./models")
    p.add_argument("--out_dir",    default="results/phase9")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--num_steps",  type=int, default=28)
    p.add_argument("--guidance",   type=float, default=2.5)
    p.add_argument("--height",     type=int, default=1024)
    p.add_argument("--width",      type=int, default=1024)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def run_method_sequence(pipe, sequence, source, args, method_name: str,
                        alpha_k: float, alpha_v: float, use_adaptive: bool):
    tr = pipe.transformer
    out_dir = os.path.join(args.out_dir, method_name)
    os.makedirs(out_dir, exist_ok=True)

    cache = {}
    current_img = source
    images = []

    for info in sequence:
        step = info["step"]
        prompt = info["prompt"]

        # Injection from previous step cache.
        if cache and step > 0:
            if use_adaptive:
                inj_orig, _ = attach_adaptive_processors(tr, cache)
            else:
                inj_orig = attach_inject_processors(
                    tr, cache, alpha_k=alpha_k, alpha_v=alpha_v,
                )
        else:
            inj_orig = None

        # Capture new cache.
        new_cache = {}
        cap_orig = attach_capture_processors(tr, new_cache)

        out = generate(pipe, prompt, current_img, seed=args.seed,
                       num_steps=args.num_steps, guidance_scale=args.guidance,
                       height=args.height, width=args.width)

        restore_processors(tr, cap_orig)
        if inj_orig:
            restore_processors(tr, inj_orig)

        cache = new_cache
        current_img = out
        out.save(os.path.join(out_dir, f"step_{step:02d}.png"))
        images.append(out)

    return images


def compute_full_metrics(sequence, images_by_method: dict, step0: Image.Image,
                         device: str) -> list[dict]:
    rows = []
    for info in sequence[1:]:  # skip step 0 (base)
        step = info["step"]
        for method_name, images in images_by_method.items():
            img = images[step]
            row = {
                "step":     step,
                "prompt":   info["prompt"],
                "method":   method_name,
                "psnr":     psnr(step0, img),
                "ssim":     ssim(step0, img),
                "lpips":    lpips_score(step0, img, device),
                "dino":     dino_similarity(step0, img, device),
            }
            # CLIP direction similarity (edit alignment)
            if info["prev_prompt"]:
                row["clip_dir"] = clip_direction_similarity(
                    images_by_method["native"][step - 1],  # image before this edit
                    img,
                    info["prev_prompt"],
                    info["prompt"],
                    device=device,
                )
            else:
                row["clip_dir"] = None
            rows.append(row)
    return rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    source = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    images_by_method = {}
    for method_name, cfg in METHODS.items():
        print(f"\n=== Running method: {method_name} ===")
        is_adaptive = cfg["alpha_k"] is None
        imgs = run_method_sequence(
            pipe, EDIT_SEQUENCE, source, args, method_name,
            alpha_k=0.5 if is_adaptive else cfg["alpha_k"],
            alpha_v=0.5 if is_adaptive else cfg["alpha_v"],
            use_adaptive=is_adaptive,
        )
        images_by_method[method_name] = imgs

    step0 = images_by_method["native"][0]

    print("\n=== Computing metrics ===")
    rows = compute_full_metrics(EDIT_SEQUENCE, images_by_method, step0, args.device)

    # --- Print per-step DINOv2 table ---
    print("\n--- Content Preservation: DINOv2 vs base image ---")
    methods = list(METHODS.keys())
    header = f"{'Step':4s}  {'Prompt':35s}  " + "  ".join(f"{m:8s}" for m in methods)
    print(header)
    print("-" * (len(header) + 10))

    for step_info in EDIT_SEQUENCE[1:]:
        step = step_info["step"]
        step_rows = {r["method"]: r for r in rows if r["step"] == step}
        row_str = f"{step:4d}  {step_info['prompt']:35s}  "
        row_str += "  ".join(
            f"{step_rows[m]['dino']:8.3f}" if m in step_rows else "    —   "
            for m in methods
        )
        print(row_str)

    # --- Print CLIP direction similarity ---
    print("\n--- Edit Success: CLIP Direction Similarity ---")
    for step_info in EDIT_SEQUENCE[1:]:
        step = step_info["step"]
        step_rows = {r["method"]: r for r in rows if r["step"] == step}
        row_str = f"Step {step}  {step_info['prompt'][:40]:40s}  "
        row_str += "  ".join(
            f"{step_rows[m].get('clip_dir') or 0:6.3f}" if m in step_rows else "  —  "
            for m in methods
        )
        print(row_str)

    # --- Save CSV ---
    csv_path = os.path.join(args.out_dir, "full_results.csv")
    fieldnames = ["step", "prompt", "method", "psnr", "ssim", "lpips", "dino", "clip_dir"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.out_dir, "full_results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"\nCSV   → {csv_path}")
    print(f"JSON  → {json_path}")
    print(f"Images → {args.out_dir}/<method>/step_NN.png")


if __name__ == "__main__":
    main()
