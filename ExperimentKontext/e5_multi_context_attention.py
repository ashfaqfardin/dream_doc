"""
E5: Multi-Context Attention Segregation
Passes [scene, obj1, obj2] through the multi-context pipeline and measures
how much attention each context image receives at every transformer block.

Metrics per block (at the captured step):
  - attention mass to scene (ctx0 / i=1)
  - attention mass to obj1  (ctx1 / i=2)

Uses MultiContextAttnCapture (chunked log-sum-exp) to avoid OOM — the full
seq×seq matrix (>50 GB for 3 context images) is never materialized.

Runtime: ~4 min  (1 generation)
"""
import os, sys, argparse, json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_pipe, enable_multi_context, MultiContextAttnCapture


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",        required=True)
    p.add_argument("--obj1",         required=True)
    p.add_argument("--obj2",         required=True)
    p.add_argument("--prompt",       default="Add the objects to the room scene naturally")
    p.add_argument("--out_dir",      default="results/e5_multi_context_attn")
    p.add_argument("--model_id",     default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",        type=int, default=28)
    p.add_argument("--capture_step", type=int, default=14)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe  = load_pipe(args.model_id, args.device)
    enable_multi_context(pipe)      # supports image=[...] list input

    scene = Image.open(args.scene).convert("RGB")
    obj1  = Image.open(args.obj1).convert("RGB")
    obj2  = Image.open(args.obj2).convert("RGB")

    N_TARGET  = 4096   # tokens for the generated (target) image
    N_CTX_PER = 4096   # tokens per context image
    N_CTX     = 3      # scene + obj1 + obj2

    cap = MultiContextAttnCapture(
        pipe.transformer,
        n_target   = N_TARGET,
        n_ctx_per  = N_CTX_PER,
        n_ctx      = N_CTX,
        capture_steps = {args.capture_step},
        chunk_size = 4096,
    )

    print("Running multi-context generation ...")
    result = pipe(
        image=[scene, obj1, obj2],
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=2.5,
        generator=torch.Generator(args.device).manual_seed(42),
        callback_on_step_end=cap.step_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
    cap.remove()
    result.save(os.path.join(args.out_dir, "result.png"))

    # ── Per-block stats ───────────────────────────────────────────────────────
    stats = {}
    for blk, records in cap.stats.items():
        for step, masses in records:
            if step == args.capture_step:
                stats[blk] = {
                    "scene_mass": masses.get("ctx0", 0.0),
                    "obj1_mass":  masses.get("ctx1", 0.0),
                    "obj2_mass":  masses.get("ctx2", 0.0),
                }

    with open(os.path.join(args.out_dir, "attention_stats.json"), "w") as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)

    if not stats:
        print("No stats captured — sequence may be shorter than expected.")
        return

    blocks     = sorted(stats.keys())
    scene_mass = [stats[b]["scene_mass"] for b in blocks]
    obj1_mass  = [stats[b]["obj1_mass"]  for b in blocks]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(blocks, scene_mass, label="scene (i=1)", color="steelblue", linewidth=2)
    ax.plot(blocks, obj1_mass,  label="obj1  (i=2)", color="coral",     linewidth=2)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Mean attention mass from target tokens to context")
    ax.set_title(f"Per-block attention mass per context image (step {args.capture_step})")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "attention_by_context.png"), dpi=150)
    plt.close()

    avg_scene = np.mean(scene_mass)
    avg_obj1  = np.mean(obj1_mass)
    print(f"Average attention mass — scene: {avg_scene:.4f}  obj1: {avg_obj1:.4f}")
    if avg_obj1 > 1e-6:
        print(f"Ratio scene/obj1: {avg_scene/(avg_obj1+1e-8):.2f}x")
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
