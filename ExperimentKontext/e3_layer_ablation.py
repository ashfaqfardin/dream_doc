"""
E3: Layer Ablation — Which Blocks Carry Context Information?
For each layer subset, zeros out target→context attention at those blocks.
Measures SSIM and LPIPS vs. full-context baseline to identify load-bearing layers.

Subsets:
  baseline     — no ablation (full context)
  first_third  — ablate blocks 0..18
  middle_third — ablate blocks 19..37
  last_third   — ablate blocks 38..56
  every_3rd    — ablate every 3rd block
  all_layers   — ablate all (context fully suppressed)

Runtime: ~10 min  (6 generations)
"""
import os, sys, argparse, json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_pipe, compute_ssim, compute_lpips


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",    required=True)
    p.add_argument("--prompt",   default="Add a wooden chair to the room")
    p.add_argument("--out_dir",  default="results/e3_layer_ablation")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",    type=int, default=28)
    p.add_argument("--device",   default="cuda")
    return p.parse_args()


def run_ablated(pipe, scene, prompt, ablate_set: set, n_target: int, args):
    """Run generation with target→context attention zeroed at specified blocks."""
    orig_sdpa   = F.scaled_dot_product_attention
    cur_block   = [-1]
    pre_hooks, post_hooks = [], []

    for i, block in enumerate(pipe.transformer.transformer_blocks):
        pre_hooks.append(block.register_forward_pre_hook(
            lambda m, a, idx=i: cur_block.__setitem__(0, idx)
        ))
        post_hooks.append(block.register_forward_hook(
            lambda m, a, o: cur_block.__setitem__(0, -1)
        ))

    def ablated_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if cur_block[0] in ablate_set:
            n_ctx = k.shape[2] - n_target
            if n_ctx > 0:
                s      = (q.shape[-1] ** -0.5) if scale is None else scale
                scores = q.float() @ k.float().transpose(-2, -1) * s
                scores[:, :, :n_target, n_target:] = float('-inf')  # kill target→context
                w = torch.softmax(scores, dim=-1)
                return (w.to(v.dtype) @ v)
        return orig_sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale)

    F.scaled_dot_product_attention = ablated_sdpa
    result = pipe(
        image=scene,
        prompt=prompt,
        num_inference_steps=args.steps,
        guidance_scale=2.5,
        generator=torch.Generator(args.device).manual_seed(42),
    ).images[0]
    F.scaled_dot_product_attention = orig_sdpa
    for h in pre_hooks + post_hooks:
        h.remove()
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe     = load_pipe(args.model_id, args.device)
    n_blocks = len(pipe.transformer.transformer_blocks)
    n_target = 4096
    t3       = n_blocks // 3
    all_b    = list(range(n_blocks))

    scene = Image.open(args.scene).convert("RGB")

    subsets = {
        "baseline":     set(),
        "first_third":  set(all_b[:t3]),
        "middle_third": set(all_b[t3:2*t3]),
        "last_third":   set(all_b[2*t3:]),
        "every_3rd":    set(all_b[::3]),
        "all_layers":   set(all_b),
    }

    baseline_img = None
    metrics      = {}

    for name, ablate_set in subsets.items():
        print(f"Running {name} ({len(ablate_set)}/{n_blocks} blocks ablated) ...")
        out = run_ablated(pipe, scene, args.prompt, ablate_set, n_target, args)
        out.save(os.path.join(args.out_dir, f"{name}.png"))
        if name == "baseline":
            baseline_img = out
        else:
            metrics[name] = {
                "SSIM":      compute_ssim(baseline_img, out),
                "LPIPS":     compute_lpips(baseline_img, out, args.device),
                "n_ablated": len(ablate_set),
            }
            m = metrics[name]
            print(f"  SSIM={m['SSIM']:.3f}  LPIPS={m['LPIPS']:.3f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    names = list(metrics.keys())
    ssims = [metrics[k]["SSIM"]  for k in names]
    lpips = [metrics[k]["LPIPS"] for k in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(names, ssims, color="steelblue")
    ax1.set_title("SSIM vs baseline\n(lower = more context was carrying info)")
    ax1.set_xticklabels(names, rotation=25, ha='right'); ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(names, lpips, color="coral")
    ax2.set_title("LPIPS vs baseline\n(higher = more context was carrying info)")
    ax2.set_xticklabels(names, rotation=25, ha='right'); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "metrics_chart.png"), dpi=150)
    plt.close()

    print(f"Done. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
