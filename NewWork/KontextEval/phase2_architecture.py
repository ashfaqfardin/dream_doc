"""
Phase 2 — Architecture Inspection

Prints the full FLUX.1-Kontext-dev transformer structure:
  - Number and type of blocks
  - Attention module names and dimensions
  - Q/K/V projection shapes
  - Example tensor dimensions at 1024×1024

Usage:
    python NewWork/KontextEval/phase2_architecture.py \
        --hf_token hf_... \
        --cache_dir ./models \
        --out_dir results/phase2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",  required=True)
    p.add_argument("--cache_dir", default="./models")
    p.add_argument("--out_dir",   default="results/phase2")
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def inspect_attn_block(block, idx: int, block_type: str) -> dict:
    attn = block.attn
    d = {
        "idx":       idx,
        "type":      block_type,
        "heads":     attn.heads,
        "inner_dim": attn.inner_dim,
        "q_shape":   tuple(attn.to_q.weight.shape),
        "k_shape":   tuple(attn.to_k.weight.shape),
        "v_shape":   tuple(attn.to_v.weight.shape),
    }
    # Double-stream blocks have to_out on the attention module.
    # Single-stream FluxAttention does not — output is fused at block level.
    if hasattr(attn, "to_out") and attn.to_out:
        d["out_shape"] = tuple(attn.to_out[0].weight.shape)
    # Double-stream blocks have add_* projections for the text stream
    if hasattr(attn, "add_q_proj"):
        d["add_q_shape"] = tuple(attn.add_q_proj.weight.shape)
        d["add_k_shape"] = tuple(attn.add_k_proj.weight.shape)
        d["add_v_shape"] = tuple(attn.add_v_proj.weight.shape)
    return d


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    tr = pipe.transformer

    # Collect block info first so we can derive dims from actual weights
    # (tr.config is a FrozenDict — key names vary across diffusers versions)
    double_infos = []
    for i, block in enumerate(tr.transformer_blocks):
        double_infos.append(inspect_attn_block(block, i, "double"))

    single_infos = []
    for i, block in enumerate(tr.single_transformer_blocks):
        single_infos.append(inspect_attn_block(block, i, "single"))

    hidden_dim = double_infos[0]["inner_dim"]
    n_heads    = double_infos[0]["heads"]
    head_dim   = hidden_dim // n_heads

    cfg = tr.config   # FrozenDict — use .get() for optional keys
    guidance_embeds = cfg.get("guidance_embeds", cfg.get("use_guidance_embeds", "N/A"))
    in_channels     = cfg.get("in_channels", "N/A")

    print("\n" + "=" * 70)
    print("FLUX.1-Kontext-dev Transformer Architecture")
    print("=" * 70)
    print(f"Total parameters : {count_params(tr)/1e9:.2f}B")
    print(f"Double-stream    : {len(tr.transformer_blocks)} blocks")
    print(f"Single-stream    : {len(tr.single_transformer_blocks)} blocks")
    print(f"Hidden dim       : {hidden_dim}  ({n_heads} heads × {head_dim} head_dim)")
    print(f"Guidance embeds  : {guidance_embeds}")
    print(f"In channels      : {in_channels}")

    print("\n--- Double-stream blocks (dual self/cross attention) ---")
    for i, info in enumerate(double_infos):
        if i == 0 or i == len(double_infos) - 1:
            print(f"\n  Block {i:2d}:")
            for k, v in info.items():
                print(f"    {k:15s}: {v}")

    print("\n  (intermediate blocks identical to block 0)")

    print("\n--- Single-stream blocks (merged representation) ---")
    for i, info in enumerate(single_infos):
        if i == 0 or i == len(single_infos) - 1:
            print(f"\n  Block {i:2d}:")
            for k, v in info.items():
                print(f"    {k:15s}: {v}")

    # Compute expected tensor shapes at 1024×1024
    print("\n--- Expected tensor dimensions at 1024×1024 ---")
    vae_sf    = pipe.vae_scale_factor
    h_lat     = 1024 // vae_sf
    w_lat     = 1024 // vae_sf
    n_img     = (h_lat // 2) * (w_lat // 2)
    n_kontext = n_img * 2
    n_txt     = 256
    seq_double = n_kontext + n_txt

    print(f"  VAE scale factor     : {vae_sf}")
    print(f"  Latent spatial dim   : {h_lat}×{w_lat}")
    print(f"  Image tokens / image : {n_img}   (= {h_lat//2}×{w_lat//2} packed)")
    print(f"  Kontext total img    : {n_kontext}   (ref + gen)")
    print(f"  Text tokens (T5)     : ~{n_txt}")
    print(f"  Double-block seq len : ~{seq_double}")
    print(f"  Single-block seq len : ~{seq_double}")

    print(f"\n  K/V shape (double) : [1, {n_heads}, {seq_double}, {head_dim}]")
    print(f"                       = [batch, heads, seq, head_dim]")
    print(f"  Bytes per K/V      : {1 * n_heads * seq_double * head_dim * 2 / 1e6:.1f} MB (bfloat16)")

    # Save JSON summary
    import json
    summary = {
        "n_double": len(tr.transformer_blocks),
        "n_single": len(tr.single_transformer_blocks),
        "hidden_size": hidden_dim,
        "heads": n_heads,
        "head_dim": head_dim,
        "guidance_embeds": str(guidance_embeds),
        "in_channels": str(in_channels),
        "n_img_tokens": n_img,
        "n_kontext_tokens": n_kontext,
        "n_txt_tokens": n_txt,
        "seq_len_double": seq_double,
        "kv_shape_double": [1, n_heads, seq_double, head_dim],
        "double_blocks": double_infos,
        "single_blocks": single_infos,
    }
    out_path = os.path.join(args.out_dir, "architecture_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFull summary saved → {out_path}")


if __name__ == "__main__":
    main()
