"""
E4: Timestep Commitment Analysis
Decodes the denoising latent at every step and measures SSIM + LPIPS to the final output.
Plots the "commitment curve" — when does the model lock in its spatial layout vs. appearance.

Runtime: ~5 min  (1 generation + 28 VAE decodes)
"""
import os, sys, argparse, json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_pipe, compute_ssim, compute_lpips


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",    required=True)
    p.add_argument("--prompt",   default="Add a wooden chair to the room")
    p.add_argument("--out_dir",  default="results/e4_timestep_commitment")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",    type=int, default=28)
    p.add_argument("--device",   default="cuda")
    return p.parse_args()


def decode_latents(pipe, latents: torch.Tensor, height=1024, width=1024) -> Image.Image:
    unp = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    unp = (unp / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.no_grad():
        img = pipe.vae.decode(unp).sample
    img = (img.float().clamp(-1, 1) + 1) / 2
    arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    pipe  = load_pipe(args.model_id, args.device)
    scene = Image.open(args.scene).convert("RGB")

    step_latents = {}

    def capture_cb(pipe_obj, step, timestep, cb):
        step_latents[step] = cb["latents"].detach().clone()
        return cb

    print("Running generation (capturing latent at every step) ...")
    result = pipe(
        image=scene,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=2.5,
        generator=torch.Generator(args.device).manual_seed(42),
        callback_on_step_end=capture_cb,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
    result.save(os.path.join(args.out_dir, "final.png"))

    # ── Decode each step and measure ──────────────────────────────────────────
    print("Decoding and measuring per step ...")
    steps_sorted = sorted(step_latents.keys())
    ssims, lpips_vals = [], []

    for s in steps_sorted:
        decoded = decode_latents(pipe, step_latents[s])
        decoded.save(os.path.join(frames_dir, f"step_{s:02d}.png"))
        ssims.append(compute_ssim(decoded, result))
        lpips_vals.append(compute_lpips(decoded, result, args.device))
        print(f"  step {s:02d}: SSIM={ssims[-1]:.3f}  LPIPS={lpips_vals[-1]:.3f}")

    metrics = {
        str(s): {"SSIM": ss, "LPIPS": lp}
        for s, ss, lp in zip(steps_sorted, ssims, lpips_vals)
    }
    with open(os.path.join(args.out_dir, "commitment.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Commitment curve plot ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(steps_sorted, ssims, marker='o', color='steelblue', linewidth=2)
    ax1.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='SSIM=0.9')
    ax1.set_ylabel("SSIM to final"); ax1.set_title("Layout commitment — SSIM (higher = closer to final)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(steps_sorted, lpips_vals, marker='o', color='coral', linewidth=2)
    ax2.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='LPIPS=0.1')
    ax2.set_xlabel("Denoising step"); ax2.set_ylabel("LPIPS to final")
    ax2.set_title("Appearance commitment — LPIPS (lower = closer to final)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "commitment_curve.png"), dpi=150)
    plt.close()

    commit_step = next((s for s, ss in zip(steps_sorted, ssims) if ss > 0.9), steps_sorted[-1])
    print(f"\nLayout committed by step {commit_step}/{args.steps} (first SSIM > 0.9)")
    print(f"Frames saved to {frames_dir}")
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
