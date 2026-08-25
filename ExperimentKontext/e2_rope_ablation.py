"""
E2: 3D RoPE Temporal Index Ablation
Tests whether the temporal RoPE index (i=1 vs i=2) is meaningful by passing
the same image at different temporal positions and measuring output differences.

Conditions:
  A. single_scene       — baseline (i=1 only)
  B. scene_x2           — same image twice (i=1 and i=2)
  C. scene_then_obj     — scene at i=1, object at i=2
  D. obj_then_scene     — object at i=1, scene at i=2  (reversed order)

If B ≈ A  → temporal index is meaningful (model treats i=1 ≠ i=2)
If B ≠ A  → model is sensitive to duplication / positional redundancy

Runtime: ~5 min  (4 generations)
"""
import os, sys, argparse, json
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_pipe, enable_multi_context, compute_ssim, compute_lpips, compute_clip_i, save_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",    required=True)
    p.add_argument("--obj",      required=True, help="Object image (from Step 1 output)")
    p.add_argument("--prompt",   default="Add the object to the room scene naturally")
    p.add_argument("--out_dir",  default="results/e2_rope_ablation")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",    type=int, default=28)
    p.add_argument("--device",   default="cuda")
    return p.parse_args()


def run(pipe, images, prompt, args):
    return pipe(
        image=images,
        prompt=prompt,
        num_inference_steps=args.steps,
        guidance_scale=2.5,
        generator=torch.Generator(args.device).manual_seed(42),
    ).images[0]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe  = load_pipe(args.model_id, args.device)
    enable_multi_context(pipe)      # supports image=[...] list input
    scene = Image.open(args.scene).convert("RGB")
    obj   = Image.open(args.obj).convert("RGB")

    conditions = {
        "A_single_scene":   scene,
        "B_scene_x2":       [scene, scene],
        "C_scene_then_obj": [scene, obj],
        "D_obj_then_scene": [obj, scene],
    }

    outputs = {}
    for name, imgs in conditions.items():
        print(f"Running {name} ...")
        out = run(pipe, imgs, args.prompt, args)
        out.save(os.path.join(args.out_dir, f"{name}.png"))
        outputs[name] = out

    # ── Metrics vs. baseline A ────────────────────────────────────────────────
    baseline = outputs["A_single_scene"]
    metrics  = {}
    for name, out in outputs.items():
        if name == "A_single_scene":
            continue
        metrics[name] = {
            "SSIM":   compute_ssim(baseline, out),
            "LPIPS":  compute_lpips(baseline, out, args.device),
            "CLIP-I": compute_clip_i(baseline, out, args.device),
        }
        m = metrics[name]
        print(f"  {name}: SSIM={m['SSIM']:.3f}  LPIPS={m['LPIPS']:.3f}  CLIP-I={m['CLIP-I']:.3f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_grid(list(outputs.values()), list(outputs.keys()),
              os.path.join(args.out_dir, "comparison_grid.png"), cols=2)

    print(f"\nKey question — B vs A:")
    if "B_scene_x2" in metrics:
        b = metrics["B_scene_x2"]
        print(f"  SSIM={b['SSIM']:.3f}  LPIPS={b['LPIPS']:.3f}")
        print("  → If SSIM≈1 and LPIPS≈0: duplication has no effect (temporal index is non-redundant)")
        print("  → If SSIM<0.95 or LPIPS>0.1: duplication changes output (model is sensitive to count)")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
