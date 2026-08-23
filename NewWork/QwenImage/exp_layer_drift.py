# -*- coding: utf-8 -*-
"""
QwenImage/exp_layer_drift.py — Experiment E2: Layer-wise drift attribution.

Question: Which transformer blocks are responsible for background drift during
sequential object insertion?

Method
------
1. Generate a base room scene (b_0).
2. Probe run A — "null-op": denoise b_0 with prompt "keep unchanged" → b_null.
3. Probe run B — "insertion": denoise b_0 with prompt "place [object]" → b_edit.
4. For each block k, compute:
       drift(k) = MSE(feature_A[k], feature_B[k])
   over background-region tokens only.
5. Plot drift(k) as a function of block index → reveals which layers handle
   background vs. object placement.

Hypothesis (from FreeFlux, arXiv:2503.16153): early blocks encode global scene
layout and are most susceptible to cross-image contamination. Late blocks refine
local textures and are less critical for drift.

If confirmed: anchor only the critical early/mid blocks → better efficiency
and fewer artifacts than full latent anchoring.

Usage
-----
  python NewWork/QwenImage/exp_layer_drift.py \\
      --base_prompt "empty minimalist living room, hardwood floor, white walls, 4K" \\
      --obj_desc    "yellow mountain bicycle" \\
      --hf_token $HF_TOKEN --cache_dir ./models --lightning \\
      --out_dir results/exp_layer_drift

Output
------
  layer_drift.json     — drift scores per block per denoising step
  layer_drift.png      — drift curve plot
  base_scene.png       — generated base room
  nullop_output.png    — null-op pass output
  edit_output.png      — edit pass output
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
from PIL import Image

from baseline import (
    load_pipeline, run_qwen, encode_latent,
    _make_room_prior_mask,
)
from probe import QwenProbe, probe_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--base_img",    default=None, help="Pre-rendered base room PNG")
    g.add_argument("--base_prompt", default=None, help="Prompt to generate base room")
    p.add_argument("--obj_desc",   default="yellow mountain bicycle",
                   help="Object description for the edit run")
    p.add_argument("--out_dir",     default="results/exp_layer_drift")
    p.add_argument("--hf_token",    default=None)
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--lightning",   action="store_true")
    p.add_argument("--height",      type=int, default=1024)
    p.add_argument("--width",       type=int, default=1024)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_steps",   type=int, default=None)
    p.add_argument("--guidance",    type=float, default=None)
    p.add_argument("--steps",       nargs="+", type=int, default=[0, 3, 7],
                   help="Denoising steps to analyze (0-indexed)")
    return p.parse_args()


def plot_drift_curve(
    drift_scores: Dict[str, Dict[int, float]],
    path:         str,
    title:        str = "Layer-wise Drift Attribution",
) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 4))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for i, (label, scores) in enumerate(drift_scores.items()):
            xs = sorted(scores.keys())
            ys = [scores[x] for x in xs]
            ax.plot(xs, ys, label=label, color=colors[i % len(colors)], lw=1.5)
        ax.set_xlabel("Block index")
        ax.set_ylabel("Feature MSE (background tokens)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] {path}")
    except Exception as e:
        print(f"[Plot] Could not save plot: {e}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    num_steps = args.num_steps or (8   if args.lightning else 50)
    guidance  = args.guidance  or (1.0 if args.lightning else 4.0)

    print(f"\n{'═'*60}")
    print(f"  E2: Layer Drift Attribution")
    print(f"  Object: {args.obj_desc}")
    print(f"{'═'*60}")

    pipe = load_pipeline(
        hf_token=args.hf_token, cache_dir=args.cache_dir,
        lightning=args.lightning,
    )

    # ── Base scene ────────────────────────────────────────────────────────────
    if args.base_img:
        if not os.path.isfile(args.base_img):
            raise FileNotFoundError(
                f"--base_img not found: {args.base_img}\n"
                "  Generate it first:\n"
                "    python NewWork/QwenImage/baseline.py \\\n"
                "        --base_prompt 'empty minimalist living room ...' \\\n"
                "        --lightning --hf_token $HF_TOKEN\n"
                "  Then re-run with --base_img results/baseline/base_scene.png"
            )
        base_img = Image.open(args.base_img).convert("RGB").resize(
            (args.width, args.height), Image.LANCZOS)
    elif args.base_prompt:
        print("[Base] Generating ...")
        base_img = run_qwen(
            pipe, image=Image.new("RGB", (args.width, args.height), 0),
            prompt=args.base_prompt, seed=args.seed,
            num_steps=num_steps, guidance=guidance,
            height=args.height, width=args.width,
            negative_prompt="furniture, objects, people, blurry, low quality",
        )
        base_img.save(os.path.join(args.out_dir, "base_scene.png"))
    else:
        raise ValueError("Provide --base_img or --base_prompt.")

    lat_h = args.height // 8
    lat_w  = args.width  // 8

    # Background mask: ceiling + walls (room prior = background region)
    prior  = _make_room_prior_mask(lat_h, lat_w)  # 1=floor/center, 0=ceiling/walls
    bg_np  = (prior[0, 0].numpy() < 0.5)           # background = inverted

    # ── Run A: null-op (no edit) ───────────────────────────────────────────────
    print("\n[Run A] Null-op pass ...")
    data_null, img_null = probe_run(
        pipe, image=base_img.resize((args.width, args.height), Image.LANCZOS),
        prompt="Do not change this image. Keep it exactly as it is.",
        seed=args.seed, num_steps=num_steps, guidance=guidance,
        height=args.height, width=args.width,
        neg="any change, modification, alteration",
        label="nullop",
    )
    img_null.save(os.path.join(args.out_dir, "nullop_output.png"))

    # ── Run B: edit (object insertion) ────────────────────────────────────────
    print(f"\n[Run B] Edit pass — '{args.obj_desc}' ...")
    edit_prompt = (
        f"Place a {args.obj_desc} on the floor of the room. "
        f"Match the room's perspective and lighting. Add a contact shadow."
    )
    data_edit, img_edit = probe_run(
        pipe, image=base_img.resize((args.width, args.height), Image.LANCZOS),
        prompt=edit_prompt,
        seed=args.seed, num_steps=num_steps, guidance=guidance,
        height=args.height, width=args.width,
        label="edit",
    )
    img_edit.save(os.path.join(args.out_dir, "edit_output.png"))

    # ── Drift analysis ────────────────────────────────────────────────────────
    print("\n[Analysis] Computing per-block feature drift ...")
    probe = QwenProbe(pipe)

    all_results: Dict[str, Dict[int, float]] = {}
    for step_idx in args.steps:
        label    = f"step_{step_idx}"
        # Full-sequence distance
        full_dist = probe.layer_feature_distance(
            data_null, data_edit, attr="attn_out", step=step_idx)
        all_results[label] = full_dist

        # Background-token-only distance
        # (Background tokens are roughly the first lat_h//2 rows = ceiling+wall region)
        # We approximate: top-third tokens = background
        n_tokens  = lat_h * lat_w // 4   # patchified = lat/2 × lat/2
        bg_tokens = int(n_tokens * 0.3)   # top ~30% of tokens = ceiling/wall
        bg_label  = f"step_{step_idx}_bg"
        bg_dist: Dict[int, float] = {}
        for bidx in sorted(set(data_null.keys()) & set(data_edit.keys())):
            fa = data_null[bidx].get(step_idx)
            fb = data_edit[bidx].get(step_idx)
            if fa is None or fb is None or fa.attn_out is None or fb.attn_out is None:
                continue
            # attn_out shape: (B, N, C) — slice background tokens
            ta = fa.attn_out.float()[:, :bg_tokens, :]
            tb = fb.attn_out.float()[:, :bg_tokens, :]
            bg_dist[bidx] = float((ta - tb).pow(2).mean().sqrt())
        all_results[bg_label] = bg_dist

    # Attention entropy comparison
    ent_null = probe.attention_entropy(data_null, step=0)
    ent_edit = probe.attention_entropy(data_edit, step=0)
    all_results["entropy_null_step0"] = ent_null
    all_results["entropy_edit_step0"] = ent_edit

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, "layer_drift.json")
    with open(out_path, "w") as f:
        json.dump({k: {str(bk): bv for bk, bv in v.items()}
                   for k, v in all_results.items()}, f, indent=2)
    print(f"[Results] → {out_path}")

    # Plot
    step_results = {k: v for k, v in all_results.items()
                    if not k.startswith("entropy")}
    plot_drift_curve(
        step_results,
        os.path.join(args.out_dir, "layer_drift.png"),
        title=f"Layer drift: null-op vs. '{args.obj_desc}'",
    )

    # Print top-10 most drifting blocks (at step 0, full sequence)
    if "step_0" in all_results:
        top10 = sorted(all_results["step_0"].items(), key=lambda x: -x[1])[:10]
        print(f"\n[Top drift blocks at step 0]")
        for bidx, dist in top10:
            print(f"  Block {bidx:>3}: MSE={dist:.6f}")

    print(f"\n{'═'*60}")
    print(f"  E2 done → {args.out_dir}")
    print(f"  Key insight: blocks with high background MSE in 'step_X_bg' series")
    print(f"  are the prime targets for attention gating (see exp_attention_gate.py).")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
