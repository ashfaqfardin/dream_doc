"""
E7: Stitched Multi-Object Context

Instead of chaining edits (E6) or multi-context images (E5), this experiment
collapses all reference objects into a single grid image and passes it as one
context alongside the base scene: image=[base_scene, stitch_grid].

Conditions (cumulative — always generated from base, no chaining):
  stitch_01_bicycle      — grid contains only o1
  stitch_02_+vase        — grid contains o1 + o2
  stitch_03_+ball        — grid contains o1 + o2 + o3
  ...
  stitch_07_+backpack    — grid contains all 7 objects

Metrics per condition:
  bg_ssim / bg_lpips     — background stability vs. original base
  dino_{obj}             — DINOv2 cosine: result vs. each reference object
  clip_{obj}             — CLIP-I: result vs. each reference object

Research question: Does stitching all reference objects into one grid image
help the model preserve individual visual identities compared to:
  - E6 (naive chaining, identity collapses at step 2)
  - E5 (multi-context, objects get 65% of scene attention)

Runtime: ~7 × 1 min = ~8 min  (one generation per stitch size)
"""
import os, sys, argparse, json, math
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_pipe, enable_multi_context, compute_ssim, compute_lpips, compute_dino, compute_clip_i


# Fixed object order matching E6 for direct comparison
OBJ_ORDER = ["bicycle", "vase", "ball", "chair", "lamp", "plant", "backpack"]


def make_stitch_grid(obj_imgs: list[Image.Image], grid_size: int = 1024,
                     bg_color: tuple = (180, 180, 180)) -> Image.Image:
    """Arrange obj_imgs in a square grid, always at grid_size × grid_size."""
    n = len(obj_imgs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = grid_size // cols
    cell_h = grid_size // rows
    grid = Image.new("RGB", (grid_size, grid_size), bg_color)
    for i, img in enumerate(obj_imgs):
        r, c = divmod(i, cols)
        thumb = img.resize((cell_w, cell_h), Image.LANCZOS)
        grid.paste(thumb, (c * cell_w, r * cell_h))
    return grid


def build_prompt(names: list[str]) -> str:
    if len(names) == 1:
        return (
            f"Place the {names[0]} from the reference image into the room scene. "
            f"Keep the exact appearance, color, and design of the {names[0]} "
            f"exactly as shown in the reference image."
        )
    body = ", ".join(names[:-1]) + f" and {names[-1]}"
    return (
        f"Place the {body} from the reference image into the room scene. "
        f"Keep the exact appearance, color, and design of each object "
        f"exactly as shown in the reference image."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",     required=True,  help="Base room image")
    p.add_argument("--obj_dir",   required=True,  help="Directory with obj_<name>.png files")
    p.add_argument("--out_dir",   default="results/e7_stitched_context")
    p.add_argument("--model_id",  default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",     type=int, default=28)
    p.add_argument("--grid_size", type=int, default=1024,
                   help="Pixel width/height of the stitched grid image")
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = load_pipe(args.model_id, args.device)
    enable_multi_context(pipe)          # needed for image=[base, stitch] list input

    base = Image.open(args.scene).convert("RGB")

    # Load objects in canonical order
    obj_imgs = {}
    for name in OBJ_ORDER:
        path = os.path.join(args.obj_dir, f"obj_{name}.png")
        if os.path.isfile(path):
            obj_imgs[name] = Image.open(path).convert("RGB")
    available = [n for n in OBJ_ORDER if n in obj_imgs]
    print(f"Found {len(available)} objects: {available}")

    metrics = []

    for k in range(1, len(available) + 1):
        names  = available[:k]
        label  = f"stitch_{k:02d}_{'_'.join(names)}"
        print(f"\n[{k}/{len(available)}] {label}")

        # Build and save the stitch grid
        stitch = make_stitch_grid([obj_imgs[n] for n in names], args.grid_size)
        stitch.save(os.path.join(args.out_dir, f"{label}_grid.png"))

        # Generate scene with stitch grid as the second context image
        prompt = build_prompt(names)
        result = pipe(
            image=[base, stitch],
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=2.5,
            generator=torch.Generator(args.device).manual_seed(42),
        ).images[0]
        result.save(os.path.join(args.out_dir, f"{label}_result.png"))

        # Background stability vs. original base
        m = {
            "k":        k,
            "objects":  names,
            "prompt":   prompt,
            "bg_ssim":  compute_ssim(base, result),
            "bg_lpips": compute_lpips(base, result, args.device),
        }

        # Per-object identity: compare result to each reference object that is
        # supposed to appear in this condition
        for name in names:
            ref = obj_imgs[name]
            m[f"dino_{name}"] = compute_dino(ref, result, args.device)
            m[f"clip_{name}"] = compute_clip_i(ref, result, args.device)

        metrics.append(m)

        print(f"  bg_ssim={m['bg_ssim']:.3f}  bg_lpips={m['bg_lpips']:.3f}")
        for name in names:
            print(f"  {name}: DINO={m[f'dino_{name}']:.3f}  CLIP={m[f'clip_{name}']:.3f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _plot(metrics, available, args.out_dir)
    print(f"\nDone. Results in {args.out_dir}")


def _plot(metrics: list, available: list[str], out_dir: str):
    import numpy as np

    ks     = [m["k"] for m in metrics]
    labels = [m["objects"][-1] for m in metrics]   # last-added object as x-label

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ── Background stability ──────────────────────────────────────────────────
    axes[0, 0].plot(ks, [m["bg_ssim"] for m in metrics],
                    marker='o', color='steelblue', linewidth=2)
    axes[0, 0].set_title("Background SSIM vs original\n(higher = stable)")
    axes[0, 0].set_ylim(0, 1); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ks, [m["bg_lpips"] for m in metrics],
                    marker='o', color='coral', linewidth=2)
    axes[0, 1].set_title("Background LPIPS vs original\n(lower = stable)")
    axes[0, 1].grid(True, alpha=0.3)

    # ── Per-object DINO (identity) ────────────────────────────────────────────
    for name in available:
        dinos  = [m[f"dino_{name}"] for m in metrics if f"dino_{name}" in m]
        ks_obj = [m["k"]            for m in metrics if f"dino_{name}" in m]
        axes[1, 0].plot(ks_obj, dinos, marker='o', linewidth=2, label=name)
    axes[1, 0].set_title("DINOv2 per object vs. reference\n(higher = identity preserved)")
    axes[1, 0].set_ylim(-0.2, 1); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)

    # ── Per-object CLIP ────────────────────────────────────────────────────────
    for name in available:
        clips  = [m[f"clip_{name}"] for m in metrics if f"clip_{name}" in m]
        ks_obj = [m["k"]            for m in metrics if f"clip_{name}" in m]
        axes[1, 1].plot(ks_obj, clips, marker='o', linewidth=2, label=name)
    axes[1, 1].set_title("CLIP-I per object vs. reference\n(higher = identity preserved)")
    axes[1, 1].set_ylim(0, 1); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7)

    for ax in axes.flat:
        ax.set_xticks(ks)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_xlabel("Cumulative objects in stitch (last added)")

    plt.suptitle("E7: Stitched Multi-Object Context — background stability & identity vs stitch size",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_chart.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
