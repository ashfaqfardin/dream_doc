"""
Experiment 1 — Layer vitality sweep.

For each of the 57 FLUX blocks (19 MM-DiT + 38 single-stream), bypass the
block during generation and measure how much the output changes via DINOv2
cosine similarity.  Low similarity → high vitality.

Paper's exact methodology
--------------------------
- 64 prompts, each paired with its own unique seed (seed = prompt index)
- Reference image for prompt i  → full model,   seed = i
- Ablated  image for prompt i  → layer bypassed, seed = i   (same latent noise)
- Vitality(ℓ) = 1 − mean( DINOv2_cosine_sim(ref_i, ablated_i) )
- Threshold τ = 0.92 used to classify vital vs non-vital

Speed note
----------
Reference images are pre-generated ONCE before the sweep begins.
Total images: 64 refs + 57 layers × 64 = 3,712
At ~30s/image with cpu_offload ≈ 31h.
At ~3s/image on a full GPU           ≈ 3h.

Quick-run mode (--quick): 8 prompts, 4 steps  (~20 min cpu_offload)

Outputs
-------
results/vitality_scores.json

Usage
-----
# Paper-accurate (full GPU recommended):
python experiments/vitality_sweep.py --hf_token YOUR_TOKEN

# Quick sanity-check (cpu_offload friendly):
python experiments/vitality_sweep.py --hf_token YOUR_TOKEN --quick --cpu_offload
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.layer_bypass import generate_with_bypass, load_pipeline

N_MM     = 19
N_SINGLE = 38

DINO_PREPROCESS = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# DINOv2 helpers
# ---------------------------------------------------------------------------

def load_dino(device: str = "cuda"):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           pretrained=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def dino_similarity(img_a: Image.Image, img_b: Image.Image,
                    dino_model, device: str) -> float:
    ta = DINO_PREPROCESS(img_a).unsqueeze(0).to(device)
    tb = DINO_PREPROCESS(img_b).unsqueeze(0).to(device)
    fa = dino_model(ta)
    fb = dino_model(tb)
    return float(F.cosine_similarity(fa, fb, dim=-1).item())


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args):
    os.makedirs("results", exist_ok=True)
    output_path = "results/vitality_scores.json"

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming from {output_path}")
    else:
        results = {"mm": {}, "single": {}}

    with open("prompts/vitality_prompts.json") as f:
        all_prompts = json.load(f)

    # --- paper: 64 prompts, each with its own seed (seed = prompt index) ---
    n_prompts = args.n_prompts
    prompts   = all_prompts[:n_prompts]
    # Each prompt i uses seed=i — same latent noise for ref and ablated
    prompt_seeds = list(range(n_prompts))

    n_done      = len(results["mm"]) + len(results["single"])
    n_remaining = (N_MM + N_SINGLE) - n_done
    print(f"Paper mode: {n_prompts} prompts, 1 seed per prompt (seed = prompt index)")
    print(f"Steps: {args.n_steps}   Layers remaining: {n_remaining} / {N_MM + N_SINGLE}")
    print(f"Total images to generate: {n_prompts} refs + {n_remaining}×{n_prompts} ablated "
          f"= {n_prompts + n_remaining * n_prompts}\n")

    pipe = load_pipeline(args.model_path, args.hf_token, args.device,
                         args.cpu_offload)
    dino = load_dino(args.device)

    # ------------------------------------------------------------------
    # Pre-generate reference images — full model, seed = prompt index
    # (paper: each prompt paired with its own unique seed)
    # ------------------------------------------------------------------
    print("Pre-generating reference images (full model, seed = prompt index)...")
    ref_images: list[Image.Image] = []
    for i, prompt in enumerate(prompts):
        seed = prompt_seeds[i]
        img  = generate_with_bypass(
            pipe, prompt, seed=seed,
            block_type="mm", bypass_idx=None,
            num_inference_steps=args.n_steps, device=args.device,
        )
        ref_images.append(img)
        print(f"  ref {i+1:2d}/{n_prompts}  seed={seed}")

    # ------------------------------------------------------------------
    # Sweep all 57 blocks
    # ------------------------------------------------------------------
    block_list = (
        [("mm",     i) for i in range(N_MM)]    +
        [("single", i) for i in range(N_SINGLE)]
    )

    for block_type, layer_idx in block_list:
        bucket = "mm" if block_type == "mm" else "single"
        key    = str(layer_idx)
        tag    = f"MM-{layer_idx:2d}" if block_type == "mm" else f"S-{layer_idx:2d}"

        if key in results[bucket]:
            print(f"  {tag}: cached, skipping")
            continue

        print(f"  {tag} ...", end=" ", flush=True)

        scores = []
        for i, prompt in enumerate(prompts):
            seed    = prompt_seeds[i]       # same seed as the reference
            ablated = generate_with_bypass(
                pipe, prompt, seed=seed,
                block_type=block_type, bypass_idx=layer_idx,
                num_inference_steps=args.n_steps, device=args.device,
            )
            sim = dino_similarity(ref_images[i], ablated, dino, args.device)
            scores.append(sim)

        results[bucket][key] = scores
        # vitality as paper defines it: 1 - mean(sim)
        vitality = 1.0 - float(np.mean(scores))
        print(f"vitality = {vitality:.4f}  (mean sim = {1-vitality:.4f})")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX layer vitality sweep")
    parser.add_argument("--model_path",  type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",   type=str, required=True)
    parser.add_argument("--n_prompts",  type=int, default=64,
                        help="Number of prompts (paper uses 64)")
    parser.add_argument("--n_steps",    type=int, default=28,
                        help="Denoising steps (paper default is 28 for FLUX.1-dev)")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--quick",      action="store_true",
                        help="Quick mode: 8 prompts, 4 steps — for cpu_offload testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.n_prompts = 8
        args.n_steps   = 4
        print("Quick mode: n_prompts=8, n_steps=4")
    run_sweep(args)
