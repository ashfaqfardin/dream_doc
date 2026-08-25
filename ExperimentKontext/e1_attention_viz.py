"""
E1: Attention Map Visualization
Captures attention weights at every transformer block for one denoising step.
Computes per-block entropy of target→context attention and saves spatial heatmaps.

Runtime: ~3 min  (1 generation + attention capture overhead)
"""
import os, sys, argparse, json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_pipe, BlockAttentionCapture, attn_entropy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",        required=True)
    p.add_argument("--prompt",       default="Add a wooden chair to the room")
    p.add_argument("--out_dir",      default="results/e1_attention_viz")
    p.add_argument("--model_id",     default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",        type=int, default=28)
    p.add_argument("--capture_step", type=int, default=14, help="Which step to visualize (0-indexed)")
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe  = load_pipe(args.model_id, args.device)
    scene = Image.open(args.scene).convert("RGB")
    n_blocks  = len(pipe.transformer.transformer_blocks)
    n_target  = 4096   # 1024×1024 → (128/2)×(128/2) packed tokens

    cap = BlockAttentionCapture(pipe.transformer, capture_steps={args.capture_step})

    print(f"Running generation (capturing step {args.capture_step}/{args.steps}) ...")
    result = pipe(
        image=scene,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=2.5,
        generator=torch.Generator(args.device).manual_seed(0),
        callback_on_step_end=cap.step_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
    cap.remove()
    result.save(os.path.join(args.out_dir, "generated.png"))

    # ── Per-block entropy ─────────────────────────────────────────────────────
    entropies = {}
    for block_idx, records in cap.captures.items():
        for step, w in records:
            if step != args.capture_step:
                continue
            seq = w.shape[-1]
            n_ctx = seq - n_target
            sub = w[:, :, :n_target, n_target:] if n_ctx > 0 else w
            entropies[block_idx] = attn_entropy(sub)

    blocks = sorted(entropies)
    ents   = [entropies[b] for b in blocks]

    with open(os.path.join(args.out_dir, "entropy.json"), "w") as f:
        json.dump({str(k): v for k, v in entropies.items()}, f, indent=2)

    plt.figure(figsize=(12, 4))
    plt.bar(blocks, ents, color="steelblue")
    plt.xlabel("Block index"); plt.ylabel("Attention entropy")
    plt.title(f"Target→Context attention entropy per block (step {args.capture_step})")
    plt.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "entropy_per_block.png"), dpi=150)
    plt.close()

    # ── Spatial heatmaps for 5 representative blocks ──────────────────────────
    selected = [0, n_blocks // 4, n_blocks // 2, 3 * n_blocks // 4, n_blocks - 1]
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4))
    for ax, bi in zip(axes, selected):
        if bi not in cap.captures:
            ax.axis('off'); continue
        for step, w in cap.captures[bi]:
            if step != args.capture_step:
                continue
            seq = w.shape[-1]
            n_ctx = seq - n_target
            if n_ctx > 0:
                hmap = w[0].mean(0)[:n_target, n_target:].float().mean(-1)
            else:
                hmap = w[0].mean(0)[:n_target, :n_target].float().mean(-1)
            side = int(n_target ** 0.5)
            ax.imshow(hmap.reshape(side, side).numpy(), cmap='hot')
            ax.set_title(f"Block {bi}", fontsize=9)
            ax.axis('off')
    plt.suptitle(f"Target→Context spatial attention (step {args.capture_step})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "attn_heatmaps.png"), dpi=150)
    plt.close()

    print(f"Blocks: {n_blocks}  |  Entropy range: [{min(ents):.3f}, {max(ents):.3f}]")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
