"""
Phase 4 — Attention Cache Construction

Runs a generation with CaptureProcessor, saves K/V to disk,
reloads them, and verifies numerical equality.

Usage:
    python NewWork/KontextEval/phase4_cache.py \
        --hf_token hf_... \
        --cache_dir ./models \
        --out_dir results/phase4 \
        --prompt "a modern living room"
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
    attach_capture_processors, restore_processors,
)
from NewWork.KontextEval.utils.cache_utils import (
    save_cache, load_cache, verify_cache, cache_summary,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",  required=True)
    p.add_argument("--cache_dir", default="./models")
    p.add_argument("--out_dir",   default="results/phase4")
    p.add_argument("--prompt",    default="a modern living room with a sofa")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--guidance",  type=float, default=2.5)
    p.add_argument("--height",    type=int, default=1024)
    p.add_argument("--width",     type=int, default=1024)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    kv_dir = os.path.join(args.out_dir, "kv_cache")

    print("Loading model …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    tr = pipe.transformer
    source = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    # --- Step 1: capture K/V ---
    print(f"\n[1] Generating with capture hooks — prompt: {args.prompt!r}")
    store = {}
    originals = attach_capture_processors(tr, store)

    img = generate(pipe, args.prompt, source,
                   seed=args.seed, num_steps=args.num_steps,
                   guidance_scale=args.guidance,
                   height=args.height, width=args.width)

    restore_processors(tr, originals)
    img.save(os.path.join(args.out_dir, "generation.png"))
    print(f"  Captured {len(store)} tensors")

    # --- Step 2: save cache (K and V only — Q is not needed for injection) ---
    print(f"\n[2] Saving K/V cache to {kv_dir}")
    store_kv = {k: v for k, v in store.items() if k.endswith("_K") or k.endswith("_V")}
    saved_keys = save_cache(store_kv, kv_dir)
    print(f"  Saved {len(saved_keys)} files")

    # Print a sample of the cache layout
    rows = cache_summary(store_kv)
    kv_rows = rows   # store_kv already contains only K and V
    total_mb = sum(r["size_mb"] for r in kv_rows)
    print(f"\n  Cache layout (K/V only, first 6 entries):")
    print(f"  {'Key':30s}  {'Shape':30s}  {'MB':6s}")
    print("  " + "-" * 70)
    for r in kv_rows[:6]:
        print(f"  {r['key']:30s}  {str(r['shape']):30s}  {r['size_mb']:6.1f}")
    print(f"  …  (total K/V: {len(kv_rows)} tensors / {total_mb:.0f} MB)")

    # --- Step 3: reload ---
    print(f"\n[3] Reloading K/V from disk …")
    store_reload = load_cache(kv_dir, device="cpu")
    print(f"  Loaded {len(store_reload)} tensors")

    # --- Step 4: verify ---
    print(f"\n[4] Verifying numerical equality …")
    report = verify_cache(store_kv, store_reload)

    n_ok = sum(1 for r in report.values() if r.get("status") == "ok")
    n_fail = len(report) - n_ok
    print(f"  Passed : {n_ok} / {len(report)}")
    if n_fail > 0:
        print(f"  Failed : {n_fail}")
        for k, v in report.items():
            if v.get("status") != "ok":
                print(f"    {k}: {v}")

    print("\n--- Cache verification complete ---")
    print(f"  Shape example (double_0_K): {tuple(store['double_0_K'].shape)}")
    print(f"  dtype example             : {store['double_0_K'].dtype}")
    print(f"  Total cache size          : {total_mb:.0f} MB")
    print(f"\nAll files → {args.out_dir}")


if __name__ == "__main__":
    main()
