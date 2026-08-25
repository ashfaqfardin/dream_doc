"""
E5: Multi-Context Attention Segregation
Passes [scene, obj1, obj2] through the patched multi-context pipeline and
measures how much attention each context image receives at every block.

Metrics per block:
  - attention mass to scene (i=1) vs obj1 (i=2) vs obj2 (i=3)
  - attention entropy per context image

Requires: patch_diffusers.py applied first.

Runtime: ~4 min  (1 generation with attention capture)
"""
import os, sys, argparse, json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

# E5 passes image=[scene, obj1, obj2] — requires patched prepare_latents
_patch = os.path.join(os.path.dirname(__file__), '..', 'KontextPipeline', 'patch_diffusers.py')
if os.path.isfile(_patch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_patch", _patch)
    _pm  = importlib.util.module_from_spec(spec); spec.loader.exec_module(_pm)
    _path = _pm.find_pipeline_file()
    if _pm.SENTINEL not in _path.read_text(encoding="utf-8"):
        print("Patch not applied. Applying now ...")
        _pm.apply_patch(_path)

from utils import load_pipe, BlockAttentionCapture, attn_entropy


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
    scene = Image.open(args.scene).convert("RGB")
    obj1  = Image.open(args.obj1).convert("RGB")
    obj2  = Image.open(args.obj2).convert("RGB")

    N_TARGET  = 4096   # tokens for the generated (target) image
    N_CTX_PER = 4096   # tokens per context image
    # hidden_states layout: [target | ctx_scene | ctx_obj1 | ctx_obj2]

    cap = BlockAttentionCapture(pipe.transformer, capture_steps={args.capture_step})

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

    # ── Per-block attention stats ─────────────────────────────────────────────
    stats = {}
    for block_idx, records in cap.captures.items():
        for step, w in records:
            if step != args.capture_step:
                continue
            seq = w.shape[-1]
            if seq < N_TARGET + 2 * N_CTX_PER:
                print(f"  Block {block_idx}: seq={seq} too short for 3 contexts, skipping")
                continue

            lo1 = N_TARGET;              hi1 = N_TARGET + N_CTX_PER
            lo2 = N_TARGET + N_CTX_PER; hi2 = N_TARGET + 2 * N_CTX_PER

            w_scene = w[:, :, :N_TARGET, lo1:hi1]
            w_obj1  = w[:, :, :N_TARGET, lo2:hi2]

            stats[block_idx] = {
                "scene_mass":    float(w_scene.sum(-1).mean().item()),
                "obj1_mass":     float(w_obj1.sum(-1).mean().item()),
                "scene_entropy": attn_entropy(w_scene),
                "obj1_entropy":  attn_entropy(w_obj1),
            }

    with open(os.path.join(args.out_dir, "attention_stats.json"), "w") as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)

    if not stats:
        print("No stats captured — check that patch_diffusers.py was applied.")
        return

    blocks     = sorted(stats.keys())
    scene_mass = [stats[b]["scene_mass"]    for b in blocks]
    obj1_mass  = [stats[b]["obj1_mass"]     for b in blocks]
    scene_ent  = [stats[b]["scene_entropy"] for b in blocks]
    obj1_ent   = [stats[b]["obj1_entropy"]  for b in blocks]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(blocks, scene_mass, label="scene (i=1)", color="steelblue", linewidth=2)
    ax1.plot(blocks, obj1_mass,  label="obj1  (i=2)", color="coral",     linewidth=2)
    ax1.set_ylabel("Mean attention mass to context")
    ax1.set_title("How much does each context image receive attention?")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(blocks, scene_ent, label="scene (i=1)", color="steelblue", linewidth=2)
    ax2.plot(blocks, obj1_ent,  label="obj1  (i=2)", color="coral",     linewidth=2)
    ax2.set_xlabel("Block index"); ax2.set_ylabel("Entropy")
    ax2.set_title("Entropy of attention to each context (lower = more focused)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "attention_by_context.png"), dpi=150)
    plt.close()

    avg_scene = np.mean(scene_mass)
    avg_obj1  = np.mean(obj1_mass)
    print(f"Average attention mass — scene: {avg_scene:.4f}  obj1: {avg_obj1:.4f}")
    print(f"Ratio scene/obj1: {avg_scene/(avg_obj1+1e-8):.2f}x")
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
