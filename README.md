# Semantic Sensitivity Experiment

## Environment Setup

System libraries on HPC clusters can conflict with the pinned versions this project needs. Use an isolated virtual environment to avoid this.

**Python venv**

```bash
python -m venv cherry_env
source cherry_env/bin/activate      # run this every new session
```

> After activating, all `pip install` and `python` commands below will use the isolated environment and will not interfere with system packages.

---

## Setup

```bash
pip install transformers==4.44.2
```

```bash
cd cherry_on_top_exp/
pip install -e .
```

## HF Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

## Run Experiments

> `--cpu_offload` offloads weights to CPU between inference steps — required on GPUs with limited VRAM (A100 40 GB or less). H100 can run without `--cpu_offload` for faster inference.
> Check disk space before running large models; clean the cache dir when full.

**`--n_pairs`** contrastive prompt pairs per semantic category (max 50).

**`--n_steps`** denoising steps per image.

**`--cache_dir`** directory for model weights and DINOv2 hub cache (default: `./models`).

**`--save_images`** save generated images to `results/images_{tag}/full/` (full-model) and `results/images_{tag}/MM-N/` (bypassed).

### FLUX.1-dev

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 28 \
    --device cuda --cpu_offload \
    --save_images \
    --cache_dir ./models
```

### FLUX.1-schnell

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.1-schnell \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 4 \
    --device cuda --cpu_offload \
    --save_images \
    --cache_dir ./models
```

### FLUX.2-dev

> Requires `transformers>=4.52` and `diffusers` (latest). Restart runtime after upgrading.
```
!pip install -U diffusers
!pip install -U transformers
```

```bash
python experiments/semantic_sensitivity.py \
    --model_path black-forest-labs/FLUX.2-dev \
    --hf_token "$HF_TOKEN" \
    --n_pairs 10 --n_steps 28 \
    --device cuda --cpu_offload \
    --save_images \
    --cache_dir ./models
```

## FluxSpace — Semantic Image Editing

> Based on [FluxSpace](https://github.com/gemlab-vt/FluxSpace) (Dalva et al., CVPR 2025).  
> Requires upstream diffusers — same as FLUX.2-dev:
> ```
> pip install -U diffusers
> ```

Edits a generated image by injecting an attribute into FLUX.1-dev's attention layers while preserving unrelated content.

**Parameter mapping (paper → CLI):**

| Paper symbol | CLI flag | Description |
|---|---|---|
| λ_coarse | `--edit_global_scale` | Global embedding shift (0–1) |
| λ_fine | `--edit_content_scale` | Per-block attention edit strength |
| τ_m | `--attention_threshold` | Cross-attention mask threshold (0–1) |
| start iter i | `--edit_start_iter` | First denoising step to apply edit |

**Global paper defaults:** `--n_steps 30 --seed 0 --attention_threshold 0.5`

### Reproduce all paper runs (config file)

```bash
python Reproduce/FluxSpace/run_fluxspace.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_fluxspace.json \
    --device cuda --cpu_offload \
    --cache_dir ./models \
    --save_images
```

Saves one `original.png` + `edited.png` per run under `results/fluxspace/{name}/`.

---

### Eyeglasses (paper quantitative settings)

λ_coarse=0.8, λ_fine=5, start_iter=3

```bash
python Reproduce/FluxSpace/run_fluxspace.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --prompt "portrait photo of a man" \
    --edit_prompt "eyeglasses" \
    --edit_global_scale 0.8 \
    --edit_content_scale 5 \
    --edit_start_iter 3 \
    --attention_threshold 0.5 \
    --n_steps 30 --seed 0 \
    --device cuda --cpu_offload \
    --cache_dir ./models \
    --save_images
```

### Smile (paper quantitative settings)

λ_coarse=0.5, λ_fine=8, start_iter=5

```bash
python Reproduce/FluxSpace/run_fluxspace.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --prompt "portrait photo of a man" \
    --edit_prompt "smiling" \
    --edit_global_scale 0.5 \
    --edit_content_scale 8 \
    --edit_start_iter 5 \
    --attention_threshold 0.5 \
    --n_steps 30 --seed 0 \
    --device cuda --cpu_offload \
    --cache_dir ./models \
    --save_images
```

### Other paper edit prompts

Fine-grained: `sunglasses`, `beard`, `surprised`, `age`, `gender`, `overweight`, `clown makeup`  
Style: `comics style`, `3D cartoon style`, `anime style`, `cinematic lighting`  
Scene: `fall`, `snow`, `sunny`, `cherry blossom`, `raining`

For style edits the paper uses λ_coarse only (λ_fine=0, τ_m=0):
```bash
python Reproduce/FluxSpace/run_fluxspace.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --hf_token "$HF_TOKEN" \
    --prompt "portrait photo of a man" \
    --edit_prompt "comics style" \
    --edit_global_scale 0.5 \
    --edit_content_scale 0 \
    --attention_threshold 0 \
    --n_steps 30 --seed 0 \
    --device cuda --cpu_offload \
    --cache_dir ./models \
    --save_images
```

Outputs saved to `results/fluxspace/seed{N}/original.png` and `edited.png`.

---

## FreeFlux — Non-Rigid Image Editing

> Based on [FreeFlux](https://github.com/wtybest/FreeFlux) (ICCV 2025).  
> Edits a real image with structural changes (pose, deformation) by inverting the source image and using mutual self-attention control during denoising.

**Dependencies:**
```bash
pip install opencv-python scipy
```

For `add_object` and `bg_replace` tasks, SAM2 is also required:
```bash
pip install git+https://github.com/facebookresearch/sam2
```

**How it works:**
1. Source image is VAE-encoded and inverted via forward DDIM steps
2. A batch of [source, target] is denoised together: image-token keys/values from source are injected into target at layers `start_layer`+ and steps `start_step`+, preserving structure
3. `result.images[1]` is the edited image; `result.images[0]` is the source reconstruction

**Key parameters:**

| Flag | Default | Description |
|---|---|---|
| `--start_step` | 4 | First denoising step to apply attention sharing |
| `--start_layer` | 0 | First transformer layer to apply attention sharing |
| `--n_steps` | 28 | Total denoising steps |
| `--guidance_scale` | 3.5 | CFG scale |

### Non-Rigid Editing (CLI)

```bash
python Reproduce/FreeFlux/non_rigid/run_non_rigid.py \
    --hf_token "$HF_TOKEN" \
    --source_image path/to/image.jpg \
    --source_prompt "a cat sitting on a chair" \
    --target_prompt "a cat jumping over a chair" \
    --n_steps 28 --start_step 4 \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

Saves `source.png`, `source_recon.png`, and `edited.png` under `results/freeflux/non_rigid/output/`.

### Reproduce demo runs (config file)

Replace the `source_image` paths in `prompts/reproduce_freeflux.json` with your own images, then:

```bash
python Reproduce/FreeFlux/non_rigid/run_non_rigid.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_freeflux.json \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

### Add Object (CLI)

Adds a new object to a generated or real image using a 3-pass (generated) or 4-pass (real image) pipeline. The object placement is derived automatically from T5 cross-attention maps — no SAM2 required.

**Key parameters:**

| Flag | Default | Description |
|---|---|---|
| `--source_prompt` | — | Description of the source scene |
| `--target_prompt` | — | Source scene plus the added object |
| `--added_word` | — | The object word(s) to locate in the T5 tokens |
| `--source_image` | — | Path to real image (omit for generated-image mode) |
| `--n_steps` | 50 | Denoising steps (paper default) |
| `--derive_step` | 7 | Step at which spatial mask is derived |

**Generated image (no source image):**

```bash
python Reproduce/FreeFlux/add_object/run_add_object.py \
    --hf_token "$HF_TOKEN" \
    --source_prompt "a dog sitting on grass" \
    --target_prompt "a dog sitting on grass with a ball" \
    --added_word "ball" \
    --n_steps 50 --seed 0 \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

**Real image editing:**

```bash
python Reproduce/FreeFlux/add_object/run_add_object.py \
    --hf_token "$HF_TOKEN" \
    --source_image path/to/image.jpg \
    --source_prompt "a dog sitting on grass" \
    --target_prompt "a dog sitting on grass with a ball" \
    --added_word "ball" \
    --n_steps 50 \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

Saves `source.png` and `edited.png` under `results/freeflux/add_object/output/`.

**Config file mode:**

```bash
python Reproduce/FreeFlux/add_object/run_add_object.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_freeflux_add_object.json \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

### Background Replace (CLI)

Replaces the background of a generated image while keeping the foreground object. Two mask modes:
- `auto` — derives foreground mask from cross-attention maps, then refines with SAM2 auto-sampled points (fully non-interactive)
- `manual` — uses your own SAM2 click coordinates

Requires SAM2 for both modes:
```bash
pip install git+https://github.com/facebookresearch/sam2
```

**Key parameters:**

| Flag | Default | Description |
|---|---|---|
| `--source_prompt` | — | Description of the source scene |
| `--target_prompt` | — | Description of the new background |
| `--foreground_word` | — | Object word(s) to keep in the foreground |
| `--fg_mask_mode` | `auto` | `auto` or `manual` |
| `--point_list` | — | JSON click points for manual mode, e.g. `'[[500,500],[700,550]]'` |
| `--label_list` | — | SAM2 labels for manual mode, e.g. `'[1,0]'` (1=fg, 0=bg) |
| `--shift` | `0 0` | `DX DY` pixels for object-moving mode |
| `--n_steps` | 50 | Denoising steps (paper default) |

**Auto fg mask (no click points needed):**

```bash
python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \
    --hf_token "$HF_TOKEN" \
    --source_prompt "A sports car on the road" \
    --target_prompt "A snowing day" \
    --foreground_word "car" \
    --fg_mask_mode auto \
    --n_steps 50 --seed 2 \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

**Manual click points:**

```bash
python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \
    --hf_token "$HF_TOKEN" \
    --source_prompt "A sports car on the road" \
    --target_prompt "A snowing day" \
    --foreground_word "car" \
    --fg_mask_mode manual \
    --point_list "[[500,500],[700,550],[350,280]]" \
    --label_list "[1,1,0]" \
    --n_steps 50 --seed 2 \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

**Object moving (same background, shifted position):**

```bash
python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \
    --hf_token "$HF_TOKEN" \
    --source_prompt "A sports car on the road" \
    --target_prompt "A sports car on the road" \
    --foreground_word "car" \
    --shift 10 10 \
    --n_steps 50 --seed 2 \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

**Config file mode:**

```bash
python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_freeflux_bg_replace.json \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

Saves `source.png` and `edited.png` under `results/freeflux/bg_replace/output/`.

---

## Plot Results

```bash
python experiments/plot_semantic_heatmap.py --tag flux1_dev --threshold 0.92
python experiments/plot_semantic_heatmap.py --tag flux1_schnell --threshold 0.92
python experiments/plot_semantic_heatmap.py --tag flux2_dev --threshold 0.92
```

## Results

Zip the results folder (includes `.npy`, `.json`, plots, and saved images):

```bash
zip -r results.zip results/
```
