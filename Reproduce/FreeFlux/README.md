# FreeFlux — Training-Free FLUX.1-dev Image Editing

> Based on [FreeFlux](https://github.com/wtybest/FreeFlux) (ICCV 2025, wtybest/FreeFlux).

Three editing tasks, all using FLUX.1-dev with mutual self-attention control:

| Task | Input | SAM2 needed |
|---|---|---|
| `non_rigid` | Text prompts only — shared latents (no source image) | No |
| `add_object` | Text prompts; optionally a real source image | No |
| `bg_replace` | Text prompts — source image generated on the fly | Yes |

All runners are called from the **repo root** (`e:/Cherry_on_top/`).

---

## Source images

Images in `inputs/` are used by the `add_object` task (real-image mode). The `non_rigid` and `bg_replace` tasks generate their source images from prompts — no image file required.

| File | Subject | Source | Used by |
|---|---|---|---|
| `inputs/bottle.jpg` | Glass bottle | Provided | `add_object` |
| `inputs/cat.jpg` | Cat | Wikimedia Commons (`A-Cat.jpg`, CC BY-SA) | `add_object` |
| `inputs/dog.jpg` | Yellow Labrador | Wikimedia Commons (`YellowLabradorLooking_new.jpg`, CC BY-SA) | `add_object` |
| `inputs/car.jpg` | Red Abarth sports car | Wikimedia Commons (`2020_Abarth_595_Competizione_front.jpg`, CC BY-SA) | `add_object` |

---

## Installation

```bash
pip install opencv-python scipy
```

For `bg_replace` only:
```bash
pip install git+https://github.com/facebookresearch/sam2
```

---

## Non-Rigid Editing

Performs non-rigid edits (pose changes, deformations) on **generated** images via shared latents + mutual self-attention control. **No real source image or DDIM inversion needed.**

Both `source_prompt` and `target_prompt` start from the **same seeded random noise**. During denoising, image-token keys/values at selected transformer layers are copied from the source branch into the target branch, preserving spatial structure while the target follows its own prompt.

### Config file

`prompts/reproduce_freeflux.json`:

```json
{
  "global": {
    "model_path": "black-forest-labs/FLUX.1-dev",
    "n_steps": 50,
    "guidance_scale": 3.5,
    "height": 1024,
    "width": 1024,
    "seed": 2
  },
  "runs": [
    {
      "name": "bird_fly",
      "source_prompt": "a bird perched on a branch",
      "target_prompt": "a bird flying from the branch"
    }
  ]
}
```

### Run

```bash
python Reproduce/FreeFlux/non_rigid/run_non_rigid.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_freeflux.json \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

Output: `results/freeflux/non_rigid/{name}/source.png`, `edited.png`

### Single run (no config file)

```bash
python Reproduce/FreeFlux/non_rigid/run_non_rigid.py \
    --hf_token "$HF_TOKEN" \
    --source_prompt "a bird perched on a branch" \
    --target_prompt "a bird flying from the branch" \
    --n_steps 50 --seed 2 \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

### Config fields

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | — | Output subfolder name |
| `source_prompt` | Yes | — | Text description of the source scene |
| `target_prompt` | Yes | — | Text description of the desired edit |
| `seed` | No | 2 | Random seed for shared latents |
| `n_steps` | No | 50 | Denoising steps (paper default: 50) |
| `guidance_scale` | No | 3.5 | CFG scale |
| `height` / `width` | No | 1024 | Output resolution |

---

## Add Object

Inserts a new object into a **generated** or **real** image using cross-attention-derived spatial masks.

### Config file

`prompts/reproduce_freeflux_add_object.json`:

```json
{
  "global": {
    "model_path": "black-forest-labs/FLUX.1-dev",
    "n_steps": 50,
    "guidance_scale": 3.5,
    "height": 1024,
    "width": 1024,
    "derive_step": 7,
    "seed": 0
  },
  "runs": [
    {
      "name": "dog_ball_generated",
      "source_prompt": "a dog sitting on grass",
      "target_prompt": "a dog sitting on grass with a ball",
      "added_word": "ball"
    },
    {
      "name": "dog_ball_real",
      "source_image": "assets/dog.jpg",
      "source_prompt": "a dog sitting on grass",
      "target_prompt": "a dog sitting on grass with a ball",
      "added_word": "ball"
    }
  ]
}
```

Omit `source_image` for generated-image mode; include it for real-image mode.

### Run

```bash
python Reproduce/FreeFlux/add_object/run_add_object.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_freeflux_add_object.json \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

Output: `results/freeflux/add_object/{name}/source.png`, `edited.png`

### Single run — generated image

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

### Single run — real image

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

### Config fields

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | — | Output subfolder name |
| `source_prompt` | Yes | — | Description of the source scene |
| `target_prompt` | Yes | — | Source scene plus the added object |
| `added_word` | Yes | — | The object word(s) in `target_prompt` (used to find T5 token indices) |
| `source_image` | No | — | Path to real source image; omit for generated-image mode |
| `n_steps` | No | 50 | Denoising steps (paper default) |
| `guidance_scale` | No | 3.5 | CFG scale |
| `derive_step` | No | 7 | Step at which spatial mask is derived |
| `seed` | No | 0 | Random seed (generated-image mode only) |
| `height` / `width` | No | 1024 | Output resolution |

---

## Background Replace

Replaces the background of a **generated** image while preserving the foreground object.

Requires SAM2 for mask generation. Two modes:

- `auto` — foreground mask is derived automatically from cross-attention; SAM2 refines it with auto-sampled click points
- `manual` — you provide SAM2 click coordinates

### Config file

`prompts/reproduce_freeflux_bg_replace.json`:

```json
{
  "global": {
    "model_path": "black-forest-labs/FLUX.1-dev",
    "n_steps": 50,
    "guidance_scale": 3.5,
    "height": 1024,
    "width": 1024,
    "seed": 2,
    "fg_mask_mode": "auto",
    "shift": [0, 0],
    "sam2_model": "facebook/sam2-hiera-large"
  },
  "runs": [
    {
      "name": "car_snowing",
      "source_prompt": "A sports car on the road",
      "target_prompt": "A snowing day",
      "foreground_word": "car"
    },
    {
      "name": "car_move",
      "source_prompt": "A sports car on the road",
      "target_prompt": "A sports car on the road",
      "foreground_word": "car",
      "shift": [10, 10]
    }
  ]
}
```

For `manual` mode, add `"fg_mask_mode": "manual"` and `"point_list"` / `"label_list"` per run:

```json
{
  "name": "car_manual",
  "source_prompt": "A sports car on the road",
  "target_prompt": "A snowing day",
  "foreground_word": "car",
  "fg_mask_mode": "manual",
  "point_list": [[500, 500], [700, 550], [350, 280]],
  "label_list": [1, 1, 0]
}
```

### Run

```bash
python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_freeflux_bg_replace.json \
    --device cuda --cpu_offload \
    --cache_dir ./models --save_images
```

Output: `results/freeflux/bg_replace/{name}/source.png`, `edited.png`

### Single run — auto mode

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

### Single run — manual click points

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

### Single run — object moving

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

### Config fields

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | — | Output subfolder name |
| `source_prompt` | Yes | — | Description of the source scene |
| `target_prompt` | Yes | — | Description of the new background |
| `foreground_word` | Yes | — | Word describing the foreground object to keep |
| `fg_mask_mode` | No | `auto` | `auto` or `manual` |
| `point_list` | No* | — | `[[x,y], ...]` click coords for manual mode |
| `label_list` | No* | — | `[1,0,...]` SAM2 labels for manual mode (1=fg, 0=bg) |
| `shift` | No | `[0, 0]` | `[dx, dy]` spatial shift for object-moving |
| `n_steps` | No | 50 | Denoising steps (paper default) |
| `guidance_scale` | No | 3.5 | CFG scale |
| `seed` | No | 2 | Random seed |
| `height` / `width` | No | 1024 | Output resolution |
| `sam2_model` | No | `facebook/sam2-hiera-large` | SAM2 checkpoint |

*Required when `fg_mask_mode` is `manual`.

---

## Common flags

| Flag | Description |
|---|---|
| `--hf_token` | HuggingFace token (required for FLUX.1-dev) |
| `--model_path` | HF model ID or local path (default: `black-forest-labs/FLUX.1-dev`) |
| `--device` | `cuda` or `cpu` (default: `cuda`) |
| `--cpu_offload` | Offload weights to CPU between steps — use on GPUs with <40 GB VRAM |
| `--cache_dir` | Directory to cache model weights (default: `./models`) |
| `--out_dir` | Root output directory (default: `results/freeflux/{task}`) |
| `--save_images` | Write output images to disk |
| `--config` | Path to JSON config file |
