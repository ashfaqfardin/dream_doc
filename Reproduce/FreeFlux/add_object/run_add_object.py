"""
FreeFlux add-object image editing for FLUX.1-dev.

Based on: https://github.com/wtybest/FreeFlux (ICCV 2025)

The method adds a new object to a generated or real image using a 3-pass
(generated) or 4-pass (real image) pipeline:

  Generated image workflow (3 passes):
    1. Standard attention: generate original image with shared random latents
    2. Reasoning pass: derive spatial mask (derive_idx_list) from cross-attention
    3. Add-object pass: regenerate with selective key/value swapping outside the mask

  Real image workflow (4 passes):
    1. Encode + invert the source image via forward DDIM steps
    2. Reasoning pass (guidance=[1, 3.5]): derive spatial mask
    3. Add-object pass (guidance=[1, 3.5]): generate edited image

The added object location is derived automatically from T5 cross-attention maps;
no SAM2 or interactive mask selection is required.

Usage — generated image
-----------------------
python Reproduce/FreeFlux/add_object/run_add_object.py \\
    --hf_token "$HF_TOKEN" \\
    --source_prompt "a dog sitting on grass" \\
    --target_prompt "a dog sitting on grass with a ball" \\
    --added_word "ball" \\
    --n_steps 50 --seed 0 \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images

Usage — real image
------------------
python Reproduce/FreeFlux/add_object/run_add_object.py \\
    --hf_token "$HF_TOKEN" \\
    --source_image path/to/image.jpg \\
    --source_prompt "a dog sitting on grass" \\
    --target_prompt "a dog sitting on grass with a ball" \\
    --added_word "ball" \\
    --n_steps 50 --seed 0 \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images

Usage — config file
-------------------
python Reproduce/FreeFlux/add_object/run_add_object.py \\
    --hf_token "$HF_TOKEN" \\
    --config prompts/reproduce_freeflux_add_object.json \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

# Add repo root to sys.path so 'Reproduce' is importable as a package.
# Script lives at Reproduce/FreeFlux/add_object/run_add_object.py → go up three levels.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffusers.models.attention_processor import FluxAttnProcessor2_0

from Reproduce.FreeFlux.add_object.pipeline_flux_add_object import FluxPipeline
from Reproduce.FreeFlux.add_object.add_object_attn_utils import (
    register_reasoning_attention_control,
    register_add_object_attention_control,
)
from Reproduce.FreeFlux.add_object.transformer_flux_add_object import (
    FluxTransformer2DModel as FluxTransformer2DModel_AddObject,
)
from Reproduce.FreeFlux.utils import get_index_from_subject


# Paper defaults for add_object — must match the trained processor expectations.
_PROCESSOR_ARGS = {
    "start_step": 0,
    "start_layer": 0,
    "layer_idx": [1, 2, 4, 26, 30, 54, 55],
    "step_idx": list(range(0, 50)),
    "total_layers": 57,
    "total_steps": 50,
}


def load_add_object_pipeline(model_path: str, hf_token: str,
                              device: str = "cuda", cpu_offload: bool = False,
                              cache_dir: str = "./models"):
    """Load FluxPipeline and replace the transformer with the add-object custom variant."""
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )

    # Replace standard transformer with the custom one that threads derive_idx_list.
    # Load weights from the same FLUX.1-dev checkpoint.
    custom_transformer = FluxTransformer2DModel_AddObject.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )
    pipe.transformer = custom_transformer

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe


def _register_standard_attention(pipe):
    """Reset to the standard FLUX attention processor."""
    pipe.transformer.set_attn_processor(FluxAttnProcessor2_0())


@torch.no_grad()
def image2latent(pipe, image: Image.Image, height_lat: int, width_lat: int,
                 latent_nudging_scalar: float = 1.15):
    """
    VAE-encode a PIL image and return packed latents.

    height_lat and width_lat are the latent spatial dimensions
    (image height/width divided by the VAE scale factor of 8).
    The nudging scalar is the StableFlow trick for better inversion fidelity.
    """
    device = pipe._execution_device
    img = pipe.image_processor.preprocess(image).to(dtype=pipe.vae.dtype, device=device)
    latents = pipe.vae.encode(img)["latent_dist"].mean
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = latents * latent_nudging_scalar
    latents = pipe._pack_latents(
        latents=latents,
        batch_size=1,
        num_channels_latents=16,
        height=height_lat,
        width=width_lat,
    )
    return latents


@torch.no_grad()
def run_add_object_generated(pipe, source_prompt: str, target_prompt: str,
                              added_word: str, n_steps: int = 50,
                              guidance_scale: float = 3.5,
                              height: int = 1024, width: int = 1024,
                              max_sequence_length: int = 512,
                              derive_step: int = 7,
                              seed: int = 0):
    """
    3-pass add-object workflow for generated images.

    Returns (original_image, edited_image) as PIL Images.
    """
    device = pipe._execution_device
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)

    # Shared random latents for all three passes.
    num_channels_latents = pipe.transformer.config.in_channels // 4
    h_lat = height // pipe.vae_scale_factor
    w_lat = width // pipe.vae_scale_factor
    from diffusers.utils.torch_utils import randn_tensor
    latent_shape = (1, num_channels_latents, h_lat, w_lat)
    raw_latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = pipe._pack_latents(raw_latents, batch_size=1,
                                 num_channels_latents=num_channels_latents,
                                 height=h_lat, width=w_lat)

    common_kwargs = dict(
        height=height, width=width,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
    )

    # ── Pass 1: standard attention → original images ────────────────────────
    print("  [Pass 1/3] Generating original image (standard attention)...")
    _register_standard_attention(pipe)
    result_ori = pipe(
        prompt=[source_prompt, target_prompt],
        latents=latents.repeat(2, 1, 1),
        **common_kwargs,
    )
    image_ori = result_ori.images  # [src_original, tgt_original]

    # ── Pass 2: reasoning → derive spatial mask ─────────────────────────────
    print("  [Pass 2/3] Reasoning pass to derive add-object mask...")
    subject_idx_list = get_index_from_subject(pipe, target_prompt, added_word)
    print(f"            T5 token indices for '{added_word}': {subject_idx_list}")

    register_reasoning_attention_control(pipe, **_PROCESSOR_ARGS)
    derive_idx_list = pipe(
        prompt=[source_prompt, target_prompt],
        latents=latents.repeat(2, 1, 1),
        subject_idx_list=subject_idx_list,
        derive_step=derive_step,
        **common_kwargs,
    )
    print(f"            Derived {len(derive_idx_list)} spatial token indices for object region.")
    if not derive_idx_list:
        print("  WARNING: empty spatial mask — the attention map produced no foreground patches for "
              f"'{added_word}'. The edited image may look identical to the source. "
              "Try a different seed, lower derive_step, or a simpler added_word.")

    # ── Pass 3: add-object attention → edited image ──────────────────────────
    print("  [Pass 3/3] Generating edited image (add-object attention)...")
    register_add_object_attention_control(pipe, **_PROCESSOR_ARGS)
    result_edit = pipe(
        prompt=[source_prompt, target_prompt],
        latents=latents.repeat(2, 1, 1),
        derive_idx_list=derive_idx_list,
        **common_kwargs,
    )

    # images[0] = source-prompt generation (no control), images[1] = edited
    return image_ori[0], result_edit.images[1]


@torch.no_grad()
def run_add_object_real(pipe, source_image_path: str,
                        source_prompt: str, target_prompt: str,
                        added_word: str, n_steps: int = 50,
                        guidance_scale_edit: float = 3.5,
                        height: int = 1024, width: int = 1024,
                        max_sequence_length: int = 512,
                        derive_step: int = 7):
    """
    4-pass add-object workflow for real images.

    Returns (source_image, edited_image) as PIL Images.
    """
    device = pipe._execution_device

    # Load + resize source image
    src_pil = Image.open(source_image_path).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )
    h_lat = height // pipe.vae_scale_factor
    w_lat = width // pipe.vae_scale_factor
    src_latents = image2latent(pipe, src_pil, h_lat, w_lat)

    prompts = [source_prompt, target_prompt]
    guidance = [1.0, guidance_scale_edit]

    common_kwargs = dict(
        height=height, width=width,
        num_inference_steps=n_steps,
        max_sequence_length=max_sequence_length,
    )

    # ── Pass 1: invert source image ──────────────────────────────────────────
    print("  [Pass 1/4] Inverting source image...")
    _register_standard_attention(pipe)
    inverted_latent_list = pipe(
        prompt=source_prompt,
        latents=src_latents.clone(),
        guidance_scale=1.0,
        invert_image=True,
        **common_kwargs,
    )
    # inverted_latent_list is a list of latents, one per forward step.
    # inverted_latent_list[-1] is the most-noisy latent.

    start_latents = inverted_latent_list[-1].repeat(len(prompts), 1, 1)

    # ── Pass 2: reasoning → derive spatial mask ─────────────────────────────
    print("  [Pass 2/4] Reasoning pass to derive add-object mask...")
    subject_idx_list = get_index_from_subject(pipe, target_prompt, added_word)
    print(f"            T5 token indices for '{added_word}': {subject_idx_list}")

    register_reasoning_attention_control(pipe, **_PROCESSOR_ARGS)
    derive_idx_list = pipe(
        prompt=prompts,
        latents=start_latents.clone(),
        guidance_scale=guidance,
        inverted_latent_list=inverted_latent_list,
        subject_idx_list=subject_idx_list,
        derive_step=derive_step,
        **common_kwargs,
    )
    print(f"            Derived {len(derive_idx_list)} spatial token indices for object region.")
    if not derive_idx_list:
        print("  WARNING: empty spatial mask — the attention map produced no foreground patches for "
              f"'{added_word}'. The edited image may look identical to the source. "
              "Try a different seed, lower derive_step, or a simpler added_word.")

    # ── Pass 3: add-object → edited image ───────────────────────────────────
    print("  [Pass 3/4] Generating edited image (add-object attention)...")
    register_add_object_attention_control(pipe, **_PROCESSOR_ARGS)
    result_edit = pipe(
        prompt=prompts,
        latents=start_latents.clone(),
        guidance_scale=guidance,
        inverted_latent_list=inverted_latent_list,
        derive_idx_list=derive_idx_list,
        **common_kwargs,
    )

    # result_edit.images[0] = source reconstruction, [1] = edited
    return src_pil, result_edit.images[1]


def run_single(pipe, cfg: dict, out_dir: str, save_images: bool):
    """Run one experiment and optionally save results."""
    name            = cfg["name"]
    source_prompt   = cfg["source_prompt"]
    target_prompt   = cfg["target_prompt"]
    added_word      = cfg["added_word"]
    source_image    = cfg.get("source_image", None)
    n_steps         = cfg.get("n_steps", 50)
    guidance_scale  = cfg.get("guidance_scale", 3.5)
    height          = cfg.get("height", 1024)
    width           = cfg.get("width", 1024)
    max_seq_len     = cfg.get("max_sequence_length", 512)
    derive_step     = cfg.get("derive_step", 7)
    seed            = cfg.get("seed", 0)

    mode = "real" if source_image else "generated"
    print(f"\n[{name}]  mode={mode}  src='{source_prompt}'  tgt='{target_prompt}'  added='{added_word}'")
    print(f"         steps={n_steps}  guidance={guidance_scale}  size={height}x{width}")

    if source_image:
        src_img, edited_img = run_add_object_real(
            pipe,
            source_image_path=source_image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            added_word=added_word,
            n_steps=n_steps,
            guidance_scale_edit=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_seq_len,
            derive_step=derive_step,
        )
    else:
        src_img, edited_img = run_add_object_generated(
            pipe,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            added_word=added_word,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_seq_len,
            derive_step=derive_step,
            seed=seed,
        )

    if save_images:
        run_dir = os.path.join(out_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        src_img.save(os.path.join(run_dir, "source.png"))
        edited_img.save(os.path.join(run_dir, "edited.png"))
        print(f"  Saved → {run_dir}/")

    return src_img, edited_img


def load_config(config_path: str) -> list:
    """
    Load runs from a JSON config file.

    The JSON must have a "runs" list. An optional "global" dict provides
    default values that each run inherits unless it overrides them.
    """
    with open(config_path) as f:
        data = json.load(f)

    global_defaults = data.get("global", {})
    runs = []
    for run in data["runs"]:
        merged = {**global_defaults, **run}
        if "name" not in merged:
            raise ValueError(f"Each run must have a 'name' field: {run}")
        runs.append(merged)
    return runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="FreeFlux add-object image editing (ICCV 2025)"
    )

    # --- Config-file mode ---
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON run-config file.")

    # --- Single-run mode ---
    parser.add_argument("--source_image",  type=str, default=None,
                        help="Path to real source image. Omit for generated-image mode.")
    parser.add_argument("--source_prompt", type=str, default=None,
                        help="Text description of the source image.")
    parser.add_argument("--target_prompt", type=str, default=None,
                        help="Text description with the added object.")
    parser.add_argument("--added_word",    type=str, default=None,
                        help="The word(s) describing the added object (used to locate T5 token indices).")
    parser.add_argument("--n_steps",             type=int,   default=50,
                        help="Denoising steps (default 50, matches paper)")
    parser.add_argument("--guidance_scale",      type=float, default=3.5)
    parser.add_argument("--height",              type=int,   default=1024)
    parser.add_argument("--width",               type=int,   default=1024)
    parser.add_argument("--max_sequence_length", type=int,   default=512)
    parser.add_argument("--derive_step",         type=int,   default=7,
                        help="Step at which the reasoning pass derives the spatial mask (default 7)")
    parser.add_argument("--seed",                type=int,   default=0,
                        help="Random seed for generated-image mode")

    # --- Infrastructure ---
    parser.add_argument("--model_path",  type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",   type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Offload weights to CPU between steps (saves VRAM)")
    parser.add_argument("--cache_dir",  type=str, default="./models")
    parser.add_argument("--out_dir",    type=str, default="results/freeflux/add_object",
                        help="Output directory for saved images")
    parser.add_argument("--save_images", action="store_true",
                        help="Save source.png and edited.png for each run")

    args = parser.parse_args()

    if args.config is None:
        if args.source_prompt is None or args.target_prompt is None or args.added_word is None:
            parser.error(
                "--source_prompt, --target_prompt, and --added_word are required unless --config is provided."
            )

    return args


def main():
    args = parse_args()

    print(f"[FreeFlux add-object] Loading {args.model_path} + custom transformer...")
    pipe = load_add_object_pipeline(
        args.model_path, args.hf_token,
        device=args.device, cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
    )
    print("[FreeFlux add-object] Model loaded.")

    if args.config:
        runs = load_config(args.config)
        print(f"\n[FreeFlux add-object] Running {len(runs)} experiment(s) from {args.config}")
        for cfg in runs:
            run_single(pipe, cfg, out_dir=args.out_dir, save_images=args.save_images)
        print(f"\n[FreeFlux add-object] All runs complete.")
        if args.save_images:
            print(f"  Results saved to {args.out_dir}/")
    else:
        cfg = {
            "name":               "output",
            "source_image":       args.source_image,
            "source_prompt":      args.source_prompt,
            "target_prompt":      args.target_prompt,
            "added_word":         args.added_word,
            "n_steps":            args.n_steps,
            "guidance_scale":     args.guidance_scale,
            "height":             args.height,
            "width":              args.width,
            "max_sequence_length": args.max_sequence_length,
            "derive_step":        args.derive_step,
            "seed":               args.seed,
        }
        run_single(pipe, cfg, out_dir=args.out_dir, save_images=args.save_images)


if __name__ == "__main__":
    main()
