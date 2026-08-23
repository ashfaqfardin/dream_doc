"""
Experiment 3 — Text→image patch attention maps.

Generates attention heatmap grids for a single prompt:
  - Rows   : sampled layers
  - Columns: content words

Also produces a temporal grid (fixed layer, varying timestep).

Outputs
-------
results/attention_maps_per_layer.png
results/attention_temporal.png
results/attention_data.pt  (raw captured maps for further analysis)

Usage
-----
python experiments/attention_maps.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token YOUR_TOKEN \
    --prompt "a red wooden chair on a stone floor" \
    [--tokens "red,wooden,chair,stone,floor"] \
    [--device cuda] [--cpu_offload]
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.attention_hooks import AttentionCapture, attn_to_spatial
from experiments.layer_bypass import load_pipeline

N_MM     = 19
N_SINGLE = 38


# ---------------------------------------------------------------------------
# Token index lookup
# ---------------------------------------------------------------------------

def tokenize_and_find(pipe, prompt: str, words: list[str]):
    """
    Return a dict {word: token_index} for T5 tokeniser.
    Uses the first occurrence of each word's first sub-token.
    """
    tokenizer = pipe.tokenizer_2  # T5-XXL
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids  = enc.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    result = {}
    for word in words:
        word_lower = word.lower()
        for idx, tok in enumerate(tokens):
            tok_clean = tok.replace("▁", "").lower()
            if tok_clean == word_lower or word_lower.startswith(tok_clean):
                result[word] = idx
                break
        if word not in result:
            result[word] = None  # not found
            print(f"  Warning: '{word}' not found in token sequence {tokens}")
    return result


# ---------------------------------------------------------------------------
# Generate and capture attention maps
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_with_capture(pipe, prompt: str, seed: int, device: str,
                                n_steps: int = 15):
    """
    Run inference and return (image, captured_maps).

    captured_maps: dict[(\"mm\", layer_idx, step)] = (n_heads, n_img, n_txt)
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, 4096, 64),
        generator=generator,
        device=device,
        dtype=pipe.transformer.dtype,
    )

    with AttentionCapture(pipe) as cap:
        result = pipe(
            prompt,
            height=1024, width=1024,
            guidance_scale=3.5,
            output_type="pil",
            num_inference_steps=n_steps,
            max_sequence_length=512,
            latents=latents,
        )
        maps = dict(cap.maps)  # copy before context exits

    return result.images[0], maps


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _overlay_attn(ax, source_image: Image.Image, attn_map: torch.Tensor,
                  title: str = ""):
    """Overlay a 64×64 attention heatmap on the source image."""
    spatial = attn_map.float().numpy()
    spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)

    img_arr = np.array(source_image.resize((512, 512)))

    ax.imshow(img_arr, alpha=0.5)
    ax.imshow(spatial, cmap="jet", alpha=0.55,
               interpolation="bilinear", extent=[0, 512, 512, 0],
               vmin=0, vmax=1)
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis("off")


def plot_per_layer(source_image: Image.Image, maps: dict,
                   token_map: dict, out_path: str,
                   sample_every: int = 3,
                   timestep_target: int = None):
    """
    Rows = every `sample_every`-th MM-DiT layer.
    Cols = content words.
    """
    sample_layers = list(range(0, N_MM, sample_every))
    words = [w for w, idx in token_map.items() if idx is not None]
    if not words:
        print("No valid token indices — skipping per-layer plot.")
        return

    # Find the timestep closest to the midpoint
    available_steps = sorted({k[2] for k in maps if k[0] == "mm"})
    if not available_steps:
        print("No MM attention maps captured — was AttentionCapture installed?")
        return
    if timestep_target is None:
        mid = len(available_steps) // 2
        step = available_steps[mid]
    else:
        step = min(available_steps, key=lambda s: abs(s - timestep_target))

    print(f"  Per-layer plot: using step {step} (available: {available_steps})")

    fig, axes = plt.subplots(len(sample_layers), len(words),
                              figsize=(4 * len(words), 3 * len(sample_layers)))
    if len(sample_layers) == 1:
        axes = axes[np.newaxis, :]

    for row, l in enumerate(sample_layers):
        key = ("mm", l, step)
        if key not in maps:
            for col in range(len(words)):
                axes[row, col].set_visible(False)
            continue

        attn = maps[key]  # (n_heads, n_img, n_txt)

        for col, word in enumerate(words):
            tok_idx = token_map[word]
            if tok_idx is None or tok_idx >= attn.shape[-1]:
                axes[row, col].set_visible(False)
                continue

            spatial = attn_to_spatial(attn, tok_idx)
            _overlay_attn(axes[row, col], source_image, spatial)

            if row == 0:
                axes[row, col].set_title(f'"{word}"', fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(f"MM-{l}", fontsize=8)

    plt.suptitle(
        f"Text token → image patch attention across MM-DiT layers\n"
        f"(denoising step {step}, averaged over heads)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def plot_temporal(source_image: Image.Image, maps: dict,
                  token_map: dict, out_path: str,
                  fixed_layer: int = 5):
    """
    Rows = early / mid / late denoising timesteps.
    Cols = content words.
    Fix the layer to `fixed_layer`.
    """
    words = [w for w, idx in token_map.items() if idx is not None]
    if not words:
        return

    available_steps = sorted({k[2] for k in maps if k[0] == "mm" and k[1] == fixed_layer})
    if not available_steps:
        print(f"No maps for MM-{fixed_layer} — skipping temporal plot.")
        return

    n_steps = len(available_steps)
    early   = available_steps[0]
    mid     = available_steps[n_steps // 2]
    late    = available_steps[-1]
    timesteps = [("Early", early), ("Mid", mid), ("Late", late)]

    print(f"  Temporal plot: layer MM-{fixed_layer}, steps {[t[1] for t in timesteps]}")

    fig, axes = plt.subplots(3, len(words),
                              figsize=(4 * len(words), 9))
    if len(words) == 1:
        axes = axes[:, np.newaxis]

    for row, (label, step) in enumerate(timesteps):
        key = ("mm", fixed_layer, step)
        if key not in maps:
            for col in range(len(words)):
                axes[row, col].set_visible(False)
            continue

        attn = maps[key]

        for col, word in enumerate(words):
            tok_idx = token_map[word]
            if tok_idx is None or tok_idx >= attn.shape[-1]:
                axes[row, col].set_visible(False)
                continue

            spatial = attn_to_spatial(attn, tok_idx)
            _overlay_attn(axes[row, col], source_image, spatial)

            if row == 0:
                axes[row, col].set_title(f'"{word}"', fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(f"{label}\n(step {step})", fontsize=8)

    plt.suptitle(
        f"Temporal evolution of text→image attention (MM-DiT layer {fixed_layer})",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",   type=str, required=True)
    parser.add_argument("--prompt",     type=str,
                        default="a red wooden chair on a stone floor")
    parser.add_argument("--tokens",     type=str,
                        default="red,wooden,chair,stone,floor",
                        help="Comma-separated list of content words to visualise")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_steps",    type=int, default=4,
                        help="Denoising steps (default 4 for speed; paper uses 15)")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--fixed_layer", type=int, default=5,
                        help="MM-DiT layer to fix for temporal analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    pipe = load_pipeline(args.model_path, args.hf_token, args.device,
                         args.cpu_offload)

    print(f"Running inference with attention capture...")
    print(f"  Prompt: {args.prompt!r}")
    image, maps = run_inference_with_capture(
        pipe, args.prompt, args.seed, args.device, args.n_steps
    )
    image.save("results/attention_source_image.png")
    print(f"  Source image saved → results/attention_source_image.png")
    print(f"  Captured {len(maps)} attention map entries")

    # Save raw maps for Experiment 4
    torch.save(maps, "results/attention_data.pt")
    print("  Raw maps saved → results/attention_data.pt")

    words = [w.strip() for w in args.tokens.split(",") if w.strip()]
    token_map = tokenize_and_find(pipe, args.prompt, words)
    print(f"  Token indices: {token_map}")

    plot_per_layer(
        image, maps, token_map,
        out_path="results/attention_maps_per_layer.png",
    )

    plot_temporal(
        image, maps, token_map,
        out_path="results/attention_temporal.png",
        fixed_layer=args.fixed_layer,
    )


if __name__ == "__main__":
    main()
