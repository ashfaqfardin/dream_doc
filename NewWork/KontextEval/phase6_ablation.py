"""
Phase 6 — Layer Ablation

Tests K+V injection on different layer groups to find which blocks
carry the most content-preserving information.

Groups:
  early  : double blocks 0–6
  middle : double blocks 7–13
  late   : double blocks 14–18  +  single blocks 0–37
  all    : all blocks

Usage:
    python NewWork/KontextEval/phase6_ablation.py \
        --hf_token hf_... \
        --cache_dir ./models \
        --out_dir results/phase6
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
from NewWork.KontextEval.utils.metrics import (
    psnr, ssim, lpips_score, dino_similarity,
)


LAYER_GROUPS = {
    "early":        {"double": list(range(0, 7)),   "single": []},
    "middle":       {"double": list(range(7, 14)),  "single": []},
    "late_double":  {"double": list(range(14, 19)), "single": []},
    "late_single":  {"double": [],                  "single": list(range(0, 38))},
    "all_double":   {"double": list(range(0, 19)),  "single": []},
    "all":          {"double": list(range(0, 19)),  "single": list(range(0, 38))},
}

ALPHA_K = 0.5
ALPHA_V = 0.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",    required=True)
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--out_dir",     default="results/phase6")
    p.add_argument("--base_prompt", default="a modern living room with a sofa and a coffee table")
    p.add_argument("--edit_prompt", default="add a yellow bicycle leaning against the wall")
    p.add_argument("--alpha_k",     type=float, default=ALPHA_K)
    p.add_argument("--alpha_v",     type=float, default=ALPHA_V)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_steps",   type=int, default=28)
    p.add_argument("--guidance",    type=float, default=2.5)
    p.add_argument("--height",      type=int, default=1024)
    p.add_argument("--width",       type=int, default=1024)
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    tr = pipe.transformer
    source = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    # --- Capture K/V from base generation ---
    print(f"\n[0] Base: {args.base_prompt!r}")
    store = {}
    orig = attach_capture_processors(tr, store)
    base_img = generate(pipe, args.base_prompt, source, seed=args.seed,
                        num_steps=args.num_steps, guidance_scale=args.guidance,
                        height=args.height, width=args.width)
    restore_processors(tr, orig)
    base_img.save(os.path.join(args.out_dir, "base.png"))

    # --- Baseline edit (no injection) ---
    print(f"\n[1] Baseline edit: {args.edit_prompt!r}")
    baseline = generate(pipe, args.edit_prompt, base_img, seed=args.seed,
                        num_steps=args.num_steps, guidance_scale=args.guidance,
                        height=args.height, width=args.width)
    baseline.save(os.path.join(args.out_dir, "baseline.png"))

    # --- Ablation experiments ---
    print(f"\n[2] Running layer ablation (α_k={args.alpha_k}, α_v={args.alpha_v})")
    results = []

    for group_name, spec in LAYER_GROUPS.items():
        print(f"\n  Group: {group_name}  "
              f"(double={spec['double'][:3]}{'…' if len(spec['double'])>3 else ''}, "
              f"single={len(spec['single'])} blocks)")

        inj_orig = attach_inject_processors(
            tr, store,
            double_blocks=spec["double"] or None,
            single_blocks=spec["single"] or None,
            alpha_k=args.alpha_k, alpha_v=args.alpha_v,
        )
        out = generate(pipe, args.edit_prompt, base_img, seed=args.seed,
                       num_steps=args.num_steps, guidance_scale=args.guidance,
                       height=args.height, width=args.width)
        restore_processors(tr, inj_orig)

        out.save(os.path.join(args.out_dir, f"{group_name}.png"))

        # Metrics vs original base image (content preservation)
        m_psnr  = psnr(base_img, out)
        m_ssim  = ssim(base_img, out)
        m_lpips = lpips_score(base_img, out, device=args.device)
        m_dino  = dino_similarity(base_img, out, device=args.device)

        results.append({
            "group":  group_name,
            "double": len(spec["double"]),
            "single": len(spec["single"]),
            "psnr":   m_psnr,
            "ssim":   m_ssim,
            "lpips":  m_lpips,
            "dino":   m_dino,
        })
        print(f"    PSNR={m_psnr:.2f}  SSIM={m_ssim:.3f}  "
              f"LPIPS={m_lpips:.3f}  DINOv2={m_dino:.3f}")

    # Baseline metrics
    bp = psnr(base_img, baseline)
    bs = ssim(base_img, baseline)
    bl = lpips_score(base_img, baseline, device=args.device)
    bd = dino_similarity(base_img, baseline, device=args.device)
    print(f"\n  Baseline (no injection): PSNR={bp:.2f}  SSIM={bs:.3f}  "
          f"LPIPS={bl:.3f}  DINOv2={bd:.3f}")

    # --- Summary table ---
    print("\n--- Phase 6 Layer Ablation Results ---")
    print(f"{'Group':15s}  {'#dbl':4s}  {'#sgl':4s}  "
          f"{'PSNR':6s}  {'SSIM':5s}  {'LPIPS':5s}  {'DINOv2':6s}")
    print("-" * 60)
    for r in results:
        print(f"{r['group']:15s}  {r['double']:4d}  {r['single']:4d}  "
              f"{r['psnr']:6.2f}  {r['ssim']:5.3f}  {r['lpips']:5.3f}  {r['dino']:6.3f}")
    print(f"{'baseline':15s}  {'—':4s}  {'—':4s}  "
          f"{bp:6.2f}  {bs:5.3f}  {bl:5.3f}  {bd:6.3f}")

    # Save as CSV for easy copy-paste
    import csv
    csv_path = os.path.join(args.out_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved → {csv_path}")


if __name__ == "__main__":
    main()
