"""
E6: Incremental Edit Drift Measurement
Runs the full 7-object edit chain and measures scene drift at every step:
  - bg_ssim   : SSIM of background vs. original base (stability)
  - bg_lpips  : LPIPS of full scene vs. original base (perceptual drift)
  - obj_dino  : DINOv2 cosine similarity of placed object vs. reference (identity)
  - obj_clip  : CLIP image-image similarity (semantic identity)

Runtime: ~20-25 min  (7 generations + per-step metrics)
"""
import os, sys, argparse, json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_pipe, compute_ssim, compute_lpips, compute_dino, compute_clip_i


EDITS = [
    {"name": "bicycle", "prompt": "Add a yellow mountain bicycle"},
    {"name": "vase",    "prompt": "Add a black ceramic vase with flowers"},
    {"name": "ball",    "prompt": "Add a yellow rubber ball"},
    {"name": "chair",   "prompt": "Add a wooden chair"},
    {"name": "lamp",    "prompt": "Add a modern floor lamp"},
    {"name": "plant",   "prompt": "Add a potted green plant"},
    {"name": "backpack","prompt": "Add a blue backpack"},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",    required=True,  help="Base scene image (step_00_base.png)")
    p.add_argument("--obj_dir",  required=True,  help="Directory with obj_*.png from Step 1")
    p.add_argument("--out_dir",  default="results/e6_drift")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",    type=int, default=28)
    p.add_argument("--device",   default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe  = load_pipe(args.model_id, args.device)
    base  = Image.open(args.scene).convert("RGB")
    scene = base.copy()
    base.save(os.path.join(args.out_dir, "step_00_base.png"))

    log = []

    for i, edit in enumerate(EDITS, start=1):
        name     = edit["name"]
        obj_path = os.path.join(args.obj_dir, f"obj_{name}.png")
        obj_img  = Image.open(obj_path).convert("RGB") if os.path.isfile(obj_path) else None

        print(f"[{i}/{len(EDITS)}] Adding {name} ...")
        result = pipe(
            image=scene,
            prompt=edit["prompt"],
            num_inference_steps=args.steps,
            guidance_scale=2.5,
            generator=torch.Generator(args.device).manual_seed(42),
        ).images[0]
        result.save(os.path.join(args.out_dir, f"step_{i:02d}_{name}.png"))

        m = {
            "step":       i,
            "name":       name,
            "bg_ssim":    compute_ssim(base, result),
            "bg_lpips":   compute_lpips(base, result, args.device),
            "scene_ssim": compute_ssim(scene, result),   # change vs. previous step
        }
        if obj_img is not None:
            m["obj_dino"] = compute_dino(obj_img, result, args.device)
            m["obj_clip"] = compute_clip_i(obj_img, result, args.device)

        log.append(m)
        print(f"       bg_ssim={m['bg_ssim']:.3f}  bg_lpips={m['bg_lpips']:.3f}"
              + (f"  obj_dino={m['obj_dino']:.3f}" if "obj_dino" in m else ""))

        scene = result

    with open(os.path.join(args.out_dir, "drift_metrics.json"), "w") as f:
        json.dump(log, f, indent=2)

    # ── Plot drift curves ─────────────────────────────────────────────────────
    steps    = [m["step"]    for m in log]
    names_x  = [m["name"]   for m in log]
    bg_ssim  = [m["bg_ssim"]  for m in log]
    bg_lpips = [m["bg_lpips"] for m in log]
    obj_dino = [m.get("obj_dino") for m in log]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(steps, bg_ssim, marker='o', color='steelblue', linewidth=2)
    axes[0].set_ylabel("SSIM"); axes[0].set_title("Background SSIM vs. original base (higher = stable)")
    axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, bg_lpips, marker='o', color='coral', linewidth=2)
    axes[1].set_ylabel("LPIPS"); axes[1].set_title("Background LPIPS vs. original base (lower = stable)")
    axes[1].grid(True, alpha=0.3)

    if any(v is not None for v in obj_dino):
        d = [v if v is not None else 0.0 for v in obj_dino]
        axes[2].plot(steps, d, marker='o', color='seagreen', linewidth=2)
        axes[2].set_ylabel("DINOv2 cosine")
        axes[2].set_title("Object identity in scene vs. reference (higher = preserved)")
        axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)

    plt.xticks(steps, names_x)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "drift_curves.png"), dpi=150)
    plt.close()

    print(f"\nFinal background SSIM : {bg_ssim[-1]:.3f}  (step 1: {bg_ssim[0]:.3f})")
    print(f"Final background LPIPS: {bg_lpips[-1]:.3f}  (step 1: {bg_lpips[0]:.3f})")
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
