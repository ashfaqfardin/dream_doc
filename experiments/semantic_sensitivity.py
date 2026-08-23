"""
Experiment 2 — Semantic specialisation of layers.

Uses the same methodology as the paper's vitality sweep:
  - DINOv2 perceptual similarity (not CLIP)
  - Contrastive pairs share the same seed so the only variable is the prompt
  - Sensitivity = |sim_full_pair − sim_ablated_pair|

Intuition
---------
Paper vitality asks:   "does this layer change the image?"
                        compare  same_prompt_full  vs  same_prompt_ablated

We ask:                "does this layer distinguish semantic category X?"
                        compare  pair_sim_full      vs  pair_sim_ablated

If bypassing layer L makes "red apple" and "green apple" look MORE similar
to DINOv2 (sim rises), then L was responsible for encoding colour.

Speed note
----------
Full-model images pre-generated ONCE before the layer loop (same pattern as
the paper's vitality pre-caching approach).

Quick-run (~15-20 min cpu_offload):  --n_pairs 1 --n_steps 4 --only_mm
Paper-style (all layers, 10 pairs):  --n_pairs 10 --n_steps 28

Outputs
-------
results/semantic_sensitivity.npy   shape (n_layers, 7)
results/semantic_sensitivity.json

Usage
-----
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token YOUR_TOKEN \
    [--n_pairs 1] [--n_steps 4] [--only_mm] \
    [--cpu_offload] [--device cuda]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.layer_bypass import generate_with_bypass, load_pipeline, get_block_counts, detect_model_type

CATEGORIES = ["colour", "style", "material", "texture", "shape", "layout", "object"]

DINO_PREPROCESS = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# DINOv2 helpers  (same as vitality_sweep.py — paper's chosen metric)
# ---------------------------------------------------------------------------

def _patch_dinov2_for_python39(hub_dir: str) -> None:
    """Add `from __future__ import annotations` to cached DINOv2 files that use
    Python 3.10+ union-type syntax (float | None). No-op on Python 3.10+."""
    import glob
    if sys.version_info >= (3, 10):
        return
    dinov2_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    if not os.path.isdir(dinov2_dir):
        return
    future_line = "from __future__ import annotations\n"
    for path in glob.glob(os.path.join(dinov2_dir, "**", "*.py"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        if "from __future__" in src:
            continue
        if " | " not in src:
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(future_line + src)


def load_dino(device: str = "cuda", cache_dir: str = "./models"):
    hub_dir = os.path.join(cache_dir, "torch_hub")
    torch.hub.set_dir(hub_dir)
    _patch_dinov2_for_python39(hub_dir)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           pretrained=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def dino_similarity(img_a: Image.Image, img_b: Image.Image,
                    dino_model, device: str) -> float:
    """DINOv2 cosine similarity between two images. Higher = more similar."""
    ta = DINO_PREPROCESS(img_a).unsqueeze(0).to(device)
    tb = DINO_PREPROCESS(img_b).unsqueeze(0).to(device)
    fa = dino_model(ta)
    fb = dino_model(tb)
    return float(F.cosine_similarity(fa, fb, dim=-1).item())


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def model_tag(model_path: str) -> str:
    """Derive a short filesystem-safe tag from a model path."""
    name = model_path.split("/")[-1]          # e.g. "FLUX.1-dev"
    return name.replace(".", "").replace("-", "_").lower()  # "flux1_dev"


def _img_dir(tag: str, layer_label: str) -> str:
    """Return (and create) the image save directory for a given layer."""
    d = os.path.join("results", f"images_{tag}", layer_label)
    os.makedirs(d, exist_ok=True)
    return d


def _save_img(img: Image.Image, tag: str, layer_label: str,
              cat: str, p_idx: int, side: str) -> None:
    path = os.path.join(_img_dir(tag, layer_label),
                        f"{cat}_pair{p_idx:02d}_{side}.png")
    img.save(path)


def run_sweep(args):
    os.makedirs("results", exist_ok=True)
    tag       = args.out_tag or model_tag(args.model_path)
    npy_path  = f"results/semantic_sensitivity_{tag}.npy"
    json_path = f"results/semantic_sensitivity_{tag}.json"
    print(f"Output tag: {tag}")

    with open("prompts/semantic_prompts.json") as f:
        semantic_prompts = json.load(f)

    pipe = load_pipeline(args.model_path, args.hf_token, args.device, args.cpu_offload, args.cache_dir)
    N_MM, N_SINGLE = get_block_counts(pipe)
    model_type = detect_model_type(pipe)
    print(f"Model type : {model_type}  |  MM blocks: {N_MM}  Single blocks: {N_SINGLE}")

    # SD3 has no single-stream blocks — silently cap
    if model_type == "sd3" and N_SINGLE == 0:
        print("  SD3 has no single-stream blocks — sweeping all MM blocks only.")

    block_list = [("mm", i) for i in range(N_MM)]
    if N_SINGLE > 0:
        block_list += [("single", i) for i in range(N_SINGLE)]
    n_layers = len(block_list)

    # Load or initialise result matrix
    if os.path.exists(npy_path):
        matrix = np.load(npy_path)
        if matrix.shape[0] != n_layers:
            print(f"  Saved matrix has {matrix.shape[0]} rows, expected {n_layers}. Starting fresh.")
            matrix = np.full((n_layers, len(CATEGORIES)), np.nan)
        else:
            print(f"Resuming from {npy_path}")
    else:
        matrix = np.full((n_layers, len(CATEGORIES)), np.nan)

    dino = load_dino(args.device, args.cache_dir)

    # ------------------------------------------------------------------
    # Pre-generate ALL full-model images for every pair — done ONCE
    # Each pair (a, b) uses the same seed so the only difference is the prompt.
    # Seed per pair = pair index (mirrors paper's per-prompt seed strategy).
    # ------------------------------------------------------------------
    print("Pre-generating full-model images for all pairs (seed = pair index)...")
    full_cache: dict = {}   # (cat, p_idx, 'a'/'b') → PIL Image
    for cat in CATEGORIES:
        pairs = semantic_prompts.get(cat, [])[:args.n_pairs]
        for p_idx, (prompt_a, prompt_b) in enumerate(pairs):
            seed = p_idx          # unique seed per pair, same for both sides
            for side, prompt in [("a", prompt_a), ("b", prompt_b)]:
                img = generate_with_bypass(
                    pipe, prompt, seed=seed,
                    block_type="mm", bypass_idx=None,
                    num_inference_steps=args.n_steps, device=args.device,
                )
                full_cache[(cat, p_idx, side)] = img
                if args.save_images:
                    _save_img(img, tag, "full", cat, p_idx, side)
    print(f"  Cached {len(full_cache)} full-model images.\n")

    # Pre-compute full-model DINOv2 similarity for each pair (once)
    sim_full_cache: dict = {}   # (cat, p_idx) → float
    for cat in CATEGORIES:
        pairs = semantic_prompts.get(cat, [])[:args.n_pairs]
        for p_idx in range(len(pairs)):
            sim_full_cache[(cat, p_idx)] = dino_similarity(
                full_cache[(cat, p_idx, "a")],
                full_cache[(cat, p_idx, "b")],
                dino, args.device,
            )

    # ------------------------------------------------------------------
    # Layer sweep — generate ablated images only, compare with DINOv2
    #
    # sensitivity = |sim_full_pair - sim_ablated_pair|
    #
    # High sensitivity: bypassing this layer made the two images look MORE
    # similar → that layer was responsible for the semantic difference.
    # ------------------------------------------------------------------
    for global_idx, (block_type, layer_idx) in enumerate(block_list):
        if not np.any(np.isnan(matrix[global_idx])):
            tag = f"MM-{layer_idx}" if block_type == "mm" else f"S-{layer_idx}"
            print(f"  [{global_idx:2d}] {tag}: cached")
            continue

        tag = f"MM-{layer_idx}" if block_type == "mm" else f"S-{layer_idx}"
        print(f"  [{global_idx:2d}] {tag} ...", flush=True)

        for cat_idx, cat in enumerate(CATEGORIES):
            if not np.isnan(matrix[global_idx, cat_idx]):
                continue

            pairs = semantic_prompts.get(cat, [])[:args.n_pairs]
            deltas = []
            for p_idx, (prompt_a, prompt_b) in enumerate(pairs):
                seed = p_idx      # same seed used when generating full-model refs

                img_a_abl = generate_with_bypass(
                    pipe, prompt_a, seed=seed,
                    block_type=block_type, bypass_idx=layer_idx,
                    num_inference_steps=args.n_steps, device=args.device,
                )
                img_b_abl = generate_with_bypass(
                    pipe, prompt_b, seed=seed,
                    block_type=block_type, bypass_idx=layer_idx,
                    num_inference_steps=args.n_steps, device=args.device,
                )

                if args.save_images:
                    layer_label = f"MM-{layer_idx}" if block_type == "mm" else f"S-{layer_idx}"
                    _save_img(img_a_abl, tag, layer_label, cat, p_idx, "a")
                    _save_img(img_b_abl, tag, layer_label, cat, p_idx, "b")

                sim_abl  = dino_similarity(img_a_abl, img_b_abl, dino, args.device)
                sim_full = sim_full_cache[(cat, p_idx)]

                # Positive delta: images became MORE similar after bypass
                # → this layer was encoding the semantic difference
                deltas.append(abs(sim_full - sim_abl))

            matrix[global_idx, cat_idx] = float(np.mean(deltas)) if deltas else 0.0

        print(f"     {dict(zip(CATEGORIES, [round(float(x), 4) for x in matrix[global_idx]]))}")
        np.save(npy_path, matrix)

    # Save JSON
    layer_labels = (
        [f"MM-{i}" for i in range(N_MM)] +
        [f"S-{i}" for i in range(N_SINGLE)]
    )[:n_layers]
    with open(json_path, "w") as f:
        json.dump({
            "metric":     "DINOv2 cosine similarity (same as paper vitality metric)",
            "categories": CATEGORIES,
            "layers":     layer_labels[:n_layers],
            "matrix":     matrix.tolist(),
        }, f, indent=2)

    saved_msg = f"\nDone. Saved:\n  {npy_path}\n  {json_path}"
    if args.save_images:
        saved_msg += f"\n  results/images_{tag}/"
    print(saved_msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",    type=str, required=True)
    parser.add_argument("--n_pairs",     type=int, default=1,
                        help="Contrastive pairs per category (default 1; max 50)")
    parser.add_argument("--n_steps",     type=int, default=4,
                        help="Denoising steps (default 4 quick; use 28 for paper-style)")
    parser.add_argument("--out_tag",     type=str, default=None,
                        help="Tag for output filenames e.g. 'flux1_dev'. "
                             "Auto-derived from model name if omitted.")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--cache_dir",   type=str, default="./models",
                        help="Directory for model weights and torch hub cache (default: ./models)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save generated images to results/images_{tag}/full/ "
                             "and results/images_{tag}/layer_MM-N/ etc.")
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
