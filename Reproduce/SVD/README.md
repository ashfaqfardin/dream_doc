# SVD Style Personalization — Training-Free Style Transfer on Infinity-2B

> Based on "A Training-Free Style-Personalization via SVD-Based Feature Decomposition" (CVPR 2025)
> Implemented from scratch — no official code.

Given a **style reference image** and a **text prompt**, generates a 1024×1024 image that follows the prompt while adopting the visual style of the reference.

All commands are run from the **repo root** (`Cherry_on_top/`).

---

## How it works

The method runs Infinity-2B in a **dual-stream** (batch size 2):

- **Stream 0 — content path**: unmodified autoregressive generation.
- **Stream 1 — generation path**: modified at scales s=3..12.

Two mechanisms are applied to the generation path:

| Mechanism | When | What it does |
|---|---|---|
| **PFB** (Principal Feature Blending) | Scale s=3 only | Injects the dominant SVD component of the style image's features into the generation path's accumulated codes |
| **SAC** (Structural Attention Correction) | Scales s=3..12 | Replaces generation-path Q and K in every self-attention with the content-path Q and K, preserving spatial structure |

---

## Setup

### Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- PyTorch 2.1+ with CUDA

> **Flash Attention must be installed** even though this implementation uses Infinity's `slow_attn` mode. Infinity's `basic.py` imports `flash_attn` unconditionally at module load time, so the package must be present regardless.

Install Python dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers huggingface_hub Pillow numpy sentencepiece
pip install flash-attn --no-build-isolation   # required by Infinity (~5–10 min to compile)
```

### 1. Clone the Infinity repo

Place it at `<repo_root>/Infinity/` (auto-detected) or anywhere and pass `--infinity_repo`:

```bash
# Option A — auto-detected location (recommended)
git clone https://github.com/FoundationVision/Infinity.git Infinity

# Option B — custom location
git clone https://github.com/FoundationVision/Infinity.git /path/to/Infinity
# then pass --infinity_repo /path/to/Infinity when running
```

Install Infinity's dependencies:

```bash
pip install -r Infinity/requirements.txt
```

### 2. Model checkpoints

Checkpoints are **auto-downloaded** from `FoundationVision/infinity` on first run into `./models/`:

- `infinity_2b_reg.pth` (~7 GB) — Infinity-2B transformer
- `infinity_vae_d32.pth` (~1 GB) — BSQ-VAE d32

Set your HuggingFace token to avoid rate limits:

```bash
export HF_TOKEN=hf_your_token_here
```

Or pass `--hf_token YOUR_TOKEN` directly.

### 3. Colab setup

```python
# 1. Clone this repo
!git clone https://github.com/ashfaqfardin/cherry_on_top_exp /content/cherry_on_top_exp
%cd /content/cherry_on_top_exp

# 2. Clone Infinity
!git clone https://github.com/FoundationVision/Infinity.git

# 3. Install flash-attn (required by Infinity even in slow_attn mode)
!pip install ninja packaging whl

!pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# 4. Set your HF token
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"
```

Then run from `/content/cherry_on_top_exp/`:

```bash
!python Reproduce/SVD/run_svd_style.py \
    --style_image  inputs/watercolor_ref.jpg \
    --prompt       "a cat in watercolor painting style" \
    --seed 0 --device cuda --save_images
```

---

## Single run

```bash
python Reproduce/SVD/run_svd_style.py \
    --style_image  inputs/watercolor_ref.jpg \
    --prompt       "a cat sitting on a windowsill in watercolor painting style" \
    --hf_token     YOUR_HF_TOKEN \
    --seed 0 \
    --device cuda \
    --save_images
```

Output: `results/svd_style/output/generated.png`

If checkpoints are already downloaded, you can point to them directly:

```bash
python Reproduce/SVD/run_svd_style.py \
    --infinity_path  models/infinity_2b_reg.pth \
    --vae_path       models/infinity_vae_d32.pth \
    --style_image    inputs/watercolor_ref.jpg \
    --prompt         "a cat sitting on a windowsill in watercolor painting style" \
    --seed 0 --device cuda --save_images
```

---

## Multiple runs (config file)

The config runner loads models once and processes all runs sequentially — much faster than launching the script once per image.

```bash
python Reproduce/SVD/run_svd_style.py \
    --config   prompts/reproduce_svd_style.json \
    --hf_token YOUR_HF_TOKEN \
    --device   cuda \
    --save_images
```

Output per run: `results/svd_style/{name}/generated.png`

In Colab:

```python
!python Reproduce/SVD/run_svd_style.py \
    --config   prompts/reproduce_svd_style.json \
    --device   cuda \
    --save_images
# HF_TOKEN already set in the environment — no need to pass it again
```

### Config format

All keys in `global` apply to every run unless overridden at the run level.
Every key that appears in the [All flags](#all-flags) table can be used in `global` or per-run.

```json
{
  "global": {
    "pfb_alpha": 1.0,
    "generation_steps": 12,
    "pfb_step": 3,
    "sac_steps": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "cfg": 3.0,
    "tau": 1.0,
    "top_k": 900,
    "top_p": 0.97,
    "cfg_insertion_layer": -5,
    "height": 1024,
    "width": 1024,
    "seed": 0
  },
  "runs": [
    {
      "name": "cat_watercolor",
      "style_image": "inputs/watercolor_ref.jpg",
      "prompt": "a cat sitting on a windowsill in watercolor painting style"
    },
    {
      "name": "castle_oil",
      "style_image": "inputs/oilpainting_ref.jpg",
      "prompt": "a medieval castle on a hill in oil painting style",
      "pfb_alpha": 0.8,
      "seed": 42
    }
  ]
}
```

**Resolution precedence**: run key → `global` key → CLI flag → built-in default.
`style_image` paths are relative to the repo root.

A ready-to-run demo config with 4 style presets is at [prompts/reproduce_svd_style.json](../../prompts/reproduce_svd_style.json).
Place your own style images in `inputs/` before running it.

---

## All flags

| Flag | Default | Description |
|---|---|---|
| `--hf_token` | env `HF_TOKEN` | HuggingFace access token for downloads |
| `--infinity_repo` | auto-detected | Path to the cloned Infinity repo (the directory containing `tools/run_infinity.py`) |
| `--infinity_path` | `models/infinity_2b_reg.pth` | Infinity-2B transformer checkpoint (auto-downloaded) |
| `--vae_path` | `models/infinity_vae_d32.pth` | BSQ-VAE d32 checkpoint (auto-downloaded) |
| `--t5_path` | `google/flan-t5-xl` | HuggingFace ID or local path for T5 text encoder |
| `--cache_dir` | `./models` | Directory for downloaded model weights |
| `--style_image` | — | Path to reference style image |
| `--prompt` | — | Text prompt for the generated image |
| `--name` | `output` | Output subfolder name |
| `--pfb_alpha` | `1.0` | SVD reweighting factor α (paper default) |
| `--cfg` | `3.0` | Classifier-free guidance scale |
| `--tau` | `1.0` | Sampling temperature |
| `--top_k` | `900` | Top-k for token sampling |
| `--top_p` | `0.97` | Nucleus sampling threshold |
| `--cfg_insertion_layer` | `-5` | Layer index for CFG insertion (negative = from end) |
| `--seed` | `0` | Random seed |
| `--height` / `--width` | `1024` | Output resolution |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--out_dir` | `results/svd_style` | Root output directory |
| `--save_images` | off | Write output images to disk |
| `--config` | — | Path to JSON config file |
| `--compare` | off | Generate baseline (no style) and styled image in one run; saves `baseline.png`, `styled.png`, and a side-by-side `comparison.png` |
| `--no_pfb` | off | Disable Principal Feature Blending (ablation) |
| `--no_sac` | off | Disable Structural Attention Correction (ablation) |

---

## Validation

### Step 1 — Verify vanilla Infinity first

Before running the style method, confirm Infinity itself works with its own script:

```bash
cd /content/Infinity
python tools/predict.py \
    --model_path ../models/infinity_2b_reg.pth \
    --vae_path   ../models/infinity_vae_d32.pth \
    --text_encoder_path google/flan-t5-xl \
    --prompt "a cat sitting on a windowsill" \
    --device cuda
```

If this generates a clean image, the model and environment are correct.

### Step 2 — Ablation (matches paper Table 2)

```bash
# (a) Baseline — pure Infinity, no style
python Reproduce/SVD/run_svd_style.py \
    --style_image inputs/watercolor_ref.jpg \
    --prompt "A cat, animals, in watercolor painting style" \
    --name ablation_baseline --no_pfb --no_sac --seed 0 --device cuda --save_images

# (b) PFB only
python Reproduce/SVD/run_svd_style.py \
    --style_image inputs/watercolor_ref.jpg \
    --prompt "A cat, animals, in watercolor painting style" \
    --name ablation_pfb_only --no_sac --seed 0 --device cuda --save_images

# (c) SAC only
python Reproduce/SVD/run_svd_style.py \
    --style_image inputs/watercolor_ref.jpg \
    --prompt "A cat, animals, in watercolor painting style" \
    --name ablation_sac_only --no_pfb --seed 0 --device cuda --save_images

# (d) Full method — PFB + SAC (paper default)
python Reproduce/SVD/run_svd_style.py \
    --style_image inputs/watercolor_ref.jpg \
    --prompt "A cat, animals, in watercolor painting style" \
    --name ablation_full --seed 0 --device cuda --save_images
```

Expected trend: (a) no style, (b) style but possible structure distortion, (d) style + stable structure.

### Step 3 — Alpha sweep (matches paper Table 3)

```bash
for alpha in 0.2 0.6 1.0 2.0 5.0; do
  python Reproduce/SVD/run_svd_style.py \
      --style_image inputs/watercolor_ref.jpg \
      --prompt "A cat, animals, in watercolor painting style" \
      --pfb_alpha $alpha \
      --name alpha_${alpha} --seed 0 --device cuda --save_images
done
```

Expected: lower α → more style (risk of content leakage), higher α → less style (closer to baseline).

---

## Notes

- **`lm_head.weight UNEXPECTED` warning**: harmless. This appears when loading `google/flan-t5-xl` as an encoder-only model — the decoder head is unused and safely ignored.
- **VRAM**: ~18 GB for 1024×1024 at bfloat16. A40/A100/H100 recommended. On 16 GB GPUs, reduce resolution or use CPU offloading (not yet implemented).
- **Prompt format**: the prompt describes **content only** — `"A <subject>, <category>"` (e.g. `"A cat, animals"`). Style comes entirely from the reference image via SVD feature extraction; adding style words to the prompt competes with the reference and weakens the transfer.
- **Model**: Infinity-8B is the default (`--model_size 8b`). Pass `--model_size 2b` to use the lighter 2B checkpoint.
