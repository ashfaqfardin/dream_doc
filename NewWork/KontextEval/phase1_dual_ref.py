# -*- coding: utf-8 -*-
"""
Dual-Reference FLUX.1-Kontext-dev — experimental.

Hypothesis: if we give Kontext BOTH the scene (what to edit) and the
object image (what to insert), the model can copy the object's exact
appearance into the scene, because it has full cross-attention access
to both reference token sequences.

Two modes
---------
stack     Stack scene + obj into one 2048-tall PIL image and pass it as
          the single reference (8192 tokens).  Zero pipeline changes.

dual_ref  Custom denoising loop that concatenates three token sequences:
          [gen_noisy(4096) | ref_scene(4096) | ref_obj(4096)] = 12288 tokens
          Both references use the same 2-D grid positions [0, y, x]
          (same as standard Kontext training).  The model treats them
          as two separate "context" images due to sequence order.

Usage
-----
python NewWork/KontextEval/phase1_dual_ref.py \\
    --scene  results/base_room.png \\
    --obj    results/bike.png \\
    --prompt "Add the bicycle to the room scene naturally." \\
    --mode   dual_ref \\
    --out_dir results/dual_ref \\
    --hf_token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline

LORA_ID      = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
LORA_TRIGGER = "Convert this sketch into real life version, follow exact structure."


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _vae_encode(pipe, image_pil: Image.Image, height: int, width: int,
                device, dtype) -> torch.Tensor:
    """Preprocess PIL image and encode through VAE.  Returns latents (B,C,H,W).
    Uses .mode() (deterministic) so reference encoding is stable across runs."""
    img_t = pipe.image_processor.preprocess(image_pil, height=height, width=width)
    img_t = img_t.to(device=device, dtype=dtype)
    with torch.no_grad():
        dist = pipe.vae.encode(img_t).latent_dist
        latents = dist.mode()       # deterministic mean, not stochastic .sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latents


def _neutralize_white_bg(image_pil: Image.Image,
                          threshold: int = 240,
                          fill: tuple = (128, 128, 128)) -> Image.Image:
    """Replace near-white pixels with neutral grey.
    LoRA generates objects on white background; that white occupies ~90% of ref2
    tokens and drowns out the object signal in cross-attention.  Grey is neutral
    and much less likely to bleed into the generated scene."""
    arr = np.array(image_pil.convert("RGB"), dtype=np.uint8)
    is_white = (arr[:, :, 0] >= threshold) & \
               (arr[:, :, 1] >= threshold) & \
               (arr[:, :, 2] >= threshold)
    arr[is_white] = fill
    return Image.fromarray(arr)


def _pack(pipe, latents: torch.Tensor) -> torch.Tensor:
    """Pack (B, C, H, W) VAE latents → (B, H//2 * W//2, C*4) tokens.
    _pack_latents halves H/W internally, so pass the full latent dims."""
    B, C, H, W = latents.shape
    return pipe._pack_latents(latents, B, C, H, W)


def _img_ids(pipe, latents: torch.Tensor, device, dtype) -> torch.Tensor:
    """Build 2-D RoPE position grid (n_tokens, 3) for a latent tensor.
    The transformer expects ids WITHOUT a batch dimension; squeeze if present."""
    B, C, H, W = latents.shape
    ids = pipe._prepare_latent_image_ids(B, H // 2, W // 2, device, dtype)
    return ids.squeeze(0) if ids.ndim == 3 else ids   # → (n_tokens, 3)


def _noise_latents(pipe, height: int, width: int,
                   device, dtype, generator) -> torch.Tensor:
    """Create initial pure-noise generation latents (B=1, C=16, H//8, W//8)."""
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_ch   = getattr(pipe.vae.config, "latent_channels", 16)
    shape  = (1, n_ch, height // vae_sf, width // vae_sf)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def _decode_latents(pipe, latents_packed: torch.Tensor,
                    height: int, width: int) -> Image.Image:
    """Unpack, shift, decode packed generation latents → PIL Image."""
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    latents = pipe._unpack_latents(latents_packed, height, width, vae_sf)
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.no_grad():
        decoded = pipe.vae.decode(latents)
    raw = decoded.sample if hasattr(decoded, "sample") else decoded[0]
    return pipe.image_processor.postprocess(raw, output_type="pil")[0]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1 — stack (no pipeline modification)
# ──────────────────────────────────────────────────────────────────────────────

def run_stack(pipe, scene: Image.Image, obj_img: Image.Image, prompt: str,
              seed: int = 42, num_steps: int = 28, guidance_scale: float = 2.5,
              height: int = 1024, width: int = 1024) -> Image.Image:
    """
    Vertical stack: paste scene on top, obj on bottom → 1024×2048 reference.
    Pass as standard reference to pipe().  The pipeline encodes it as 8192 tokens
    (two 64×64 grids stacked along the y-axis: y = 0..127 for ref, 0..63 for gen).

    Note: this depends on FluxKontextPipeline NOT resizing the reference image to
    the generation output dimensions.  If it does, results will be squashed.
    """
    ref = Image.new("RGB", (width, height * 2))
    ref.paste(scene.resize((width, height), Image.LANCZOS), (0, 0))
    ref.paste(obj_img.resize((width, height), Image.LANCZOS), (0, height))

    device_str = str(next(pipe.transformer.parameters()).device)
    generator = torch.Generator(device=device_str).manual_seed(seed)
    result = pipe(
        image=ref,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=512,
        generator=generator,
    ).images[0]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2 — dual_ref (custom denoising loop, true dual reference)
# ──────────────────────────────────────────────────────────────────────────────

def run_dual_ref(pipe, scene: Image.Image, obj_img: Image.Image, prompt: str,
                 seed: int = 42, num_steps: int = 28, guidance_scale: float = 2.5,
                 height: int = 1024, width: int = 1024) -> Image.Image:
    """
    True dual-reference via custom denoising loop.

    Token sequence fed to FluxTransformer:
      [gen_noisy (4096) | ref_scene (4096) | ref_obj (4096)] = 12288 tokens

    Both ref_scene and ref_obj use the standard 2-D position grid [0, y, x]
    with y, x ∈ [0, 63], identical to how Kontext was trained for a single ref.
    The model distinguishes them through sequence order (ref_scene precedes
    ref_obj which precedes… nothing — gen comes first).

    After the transformer, we extract only noise_pred[:, :4096, :] (the gen
    portion) and run the standard flow-matching scheduler step.

    ⚠ This is out-of-distribution for the model (trained on 8192-token sequences).
      Expect unpredictable results.  Low guidance_scale (≤ 2.5) is safer.
    """
    device = next(pipe.transformer.parameters()).device
    dtype  = next(pipe.transformer.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    # ── 1. Text embeddings ────────────────────────────────────────────────────
    prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    # ── 2. Reference 1 — scene ────────────────────────────────────────────────
    ref1_latents = _vae_encode(pipe, scene, height, width, device, dtype)
    ref1_packed  = _pack(pipe, ref1_latents)                        # (1, 4096, 64)
    ref1_ids     = _img_ids(pipe, ref1_latents, device, dtype)      # (1, 4096, 3)

    # ── 3. Reference 2 — object ───────────────────────────────────────────────
    obj_clean    = _neutralize_white_bg(obj_img)          # grey bg instead of white
    ref2_latents = _vae_encode(pipe, obj_clean, height, width, device, dtype)
    ref2_packed  = _pack(pipe, ref2_latents)                        # (1, 4096, 64)
    ref2_ids     = _img_ids(pipe, ref2_latents, device, dtype)      # (n_tokens, 3)
    # Give ref2 a distinct frame id (dim-0 = 1) so RoPE can spatially distinguish
    # it from ref1 and gen which both use frame id = 0.
    ref2_ids     = ref2_ids.clone()
    ref2_ids[:, 0] = 1

    # ── 4. Noisy generation latents ───────────────────────────────────────────
    gen_latents = _noise_latents(pipe, height, width, device, dtype, generator)
    gen_packed  = _pack(pipe, gen_latents)                          # (1, 4096, 64)
    gen_ids     = _img_ids(pipe, gen_latents, device, dtype)        # (1, 4096, 3)
    n_gen       = gen_packed.shape[1]                               # 4096

    # ── 5. Scheduler (dynamic shift requires mu) ──────────────────────────────
    try:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    except ImportError:
        from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift

    vae_sf     = getattr(pipe, "vae_scale_factor", 8)
    patch_size = getattr(pipe.transformer.config, "patch_size", 2)
    image_seq_len = (height // vae_sf // patch_size) * (width // vae_sf // patch_size)

    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift",         0.5),
        pipe.scheduler.config.get("max_shift",          1.16),
    )
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps

    # ── 6. Guidance embedding (FLUX.1-dev has guidance_embeds=True) ───────────
    guidance = None
    if getattr(pipe.transformer.config, "guidance_embeds", False):
        guidance = torch.full([1], guidance_scale, dtype=dtype, device=device)

    # ── 7. Denoising loop ─────────────────────────────────────────────────────
    latents = gen_packed   # (1, 4096, 64) — pure noise at t=T

    for i, t in enumerate(timesteps):
        # Build 12288-token sequence: [gen | ref_scene | ref_obj]
        model_input = torch.cat([latents, ref1_packed, ref2_packed], dim=1)  # (1, 12288, 64)
        img_ids     = torch.cat([gen_ids, ref1_ids,   ref2_ids   ], dim=0)  # (12288, 3) — no batch dim

        t_batch = t.expand(1)

        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states        = model_input,
                timestep             = t_batch / 1000,
                guidance             = guidance,
                pooled_projections   = pooled_embeds,
                encoder_hidden_states= prompt_embeds,
                txt_ids              = text_ids,
                img_ids              = img_ids,
                return_dict          = False,
            )[0]                                        # (1, 12288, 64)

        # Extract noise prediction for gen tokens only
        noise_pred = noise_pred[:, :n_gen, :]           # (1, 4096, 64)

        # Flow-matching scheduler step
        latents = pipe.scheduler.step(
            noise_pred, t, latents, return_dict=False
        )[0]                                            # (1, 4096, 64)

        if i % 5 == 0 or i == len(timesteps) - 1:
            print(f"  step {i+1:3d}/{len(timesteps)}", flush=True)

    # ── 8. Decode ─────────────────────────────────────────────────────────────
    return _decode_latents(pipe, latents, height, width)


# ──────────────────────────────────────────────────────────────────────────────
# Base scene generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_base(pipe, prompt: str,
                  seed: int = 42, num_steps: int = 28,
                  guidance_scale: float = 2.5,
                  height: int = 1024, width: int = 1024) -> Image.Image:
    """Generate base scene from a grey canvas using standard Kontext."""
    grey = Image.new("RGB", (width, height), (128, 128, 128))
    device_str = str(next(pipe.transformer.parameters()).device)
    generator = torch.Generator(device=device_str).manual_seed(seed)
    return pipe(
        image=grey,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=512,
        generator=generator,
    ).images[0]


# ──────────────────────────────────────────────────────────────────────────────
# Sketch → object generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_from_sketch(pipe, sketch_path: str, description: str,
                          seed: int = 42, num_steps: int = 28,
                          lora_guidance: float = 4.0,
                          height: int = 1024, width: int = 1024,
                          lora_id: str = LORA_ID) -> Image.Image:
    """Sketch → photorealistic object on white background via Kontext + LoRA."""
    sketch = Image.open(sketch_path).convert("RGB").resize((width, height), Image.LANCZOS)
    device_str = str(next(pipe.transformer.parameters()).device)
    generator = torch.Generator(device=device_str).manual_seed(seed)
    prompt = (f"{LORA_TRIGGER} {description} on a plain white background, "
              "photorealistic, no shadows, studio lighting, high quality.")
    pipe.load_lora_weights(lora_id)
    obj_img = pipe(
        image=sketch,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=lora_guidance,
        height=height,
        width=width,
        max_sequence_length=512,
        generator=generator,
    ).images[0]
    pipe.unload_lora_weights()
    return obj_img


# ──────────────────────────────────────────────────────────────────────────────
# Multi-step loop
# ──────────────────────────────────────────────────────────────────────────────

def run_multi_step(pipe, base_scene: Image.Image, edits: list[dict],
                   mode: str = "dual_ref",
                   seed: int = 42, num_steps: int = 28,
                   guidance_scale: float = 2.5,
                   lora_id: str = LORA_ID, lora_guidance: float = 4.0,
                   height: int = 1024, width: int = 1024,
                   out_dir: str = "results/dual_ref") -> list[Image.Image]:
    """
    Incremental edit loop: base → +obj1 → +obj2 → +obj3 …

    Each edit entry must have:
      name        str   label for filenames
      prompt      str   edit instruction for this step
      sketch      str   path to sketch image  (generates object via LoRA)
      description str   object description for LoRA prompt  (required with sketch)
      obj         str   path to pre-made object image  (alternative to sketch)

    Returns list of images: [base, step1, step2, …]
    """
    os.makedirs(out_dir, exist_ok=True)
    run_fn = run_stack if mode == "stack" else run_dual_ref

    history = [base_scene]
    base_scene.save(os.path.join(out_dir, "step0_base.png"))

    current = base_scene
    for i, edit in enumerate(edits, start=1):
        name   = edit["name"]
        prompt = edit["prompt"]

        print(f"\n{'='*60}")
        print(f"[Step {i}] {name}")
        print(f"{'='*60}")

        # -- Stage A: get object image ----------------------------------------
        if "sketch" in edit:
            print(f"  Generating {name} from sketch …")
            obj_img = generate_from_sketch(
                pipe, edit["sketch"], edit["description"],
                seed=seed, num_steps=num_steps,
                lora_guidance=lora_guidance,
                height=height, width=width, lora_id=lora_id,
            )
            obj_img.save(os.path.join(out_dir, f"obj_{name}.png"))
            print(f"  Saved → obj_{name}.png")
        else:
            obj_img = Image.open(edit["obj"]).convert("RGB")

        print(f"  Prompt: {prompt}")

        result = run_fn(
            pipe, current, obj_img, prompt,
            seed=seed, num_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height, width=width,
        )
        result.save(os.path.join(out_dir, f"step{i}_{name}.png"))
        print(f"Saved → step{i}_{name}.png")

        history.append(result)
        current = result

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final comparison grid: base + each step side by side
    n = len(history)
    grid = Image.new("RGB", (width * n, height))
    for j, img in enumerate(history):
        grid.paste(img.resize((width, height), Image.LANCZOS), (j * width, 0))
    grid.save(os.path.join(out_dir, "KEY_grid.png"))
    print(f"\nSaved → KEY_grid.png  ({n} panels)")

    return history


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dual-reference FLUX.1-Kontext experiment")

    # Base scene — provide one of these two
    p.add_argument("--scene",       default=None,
                   help="Path to existing base scene image")
    p.add_argument("--base_prompt", default=None,
                   help="Text prompt to generate the base scene (grey canvas → Kontext)")

    # Single-step mode
    p.add_argument("--obj",    default=None, help="Object image path (single-step)")
    p.add_argument("--prompt", default=None, help="Edit prompt (single-step)")

    # Multi-step mode via JSON config
    p.add_argument("--config", default=None,
                   help=("JSON file with edit list.  Format: "
                         '[{"name":"bike","obj":"bike.png","prompt":"Add the bike..."},'
                         '{"name":"vase",...}]'))

    p.add_argument("--mode",         default="dual_ref", choices=["stack", "dual_ref"])
    p.add_argument("--out_dir",      default="results/dual_ref")
    p.add_argument("--hf_token",     default=None)
    p.add_argument("--cache_dir",    default="./models")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_steps",    type=int,   default=28)
    p.add_argument("--guidance",     type=float, default=2.5,
                   help="Guidance scale for scene editing steps")
    p.add_argument("--lora_id",      default=LORA_ID,
                   help="HuggingFace repo for sketch-to-image LoRA")
    p.add_argument("--lora_guidance",type=float, default=4.0,
                   help="Guidance scale for LoRA sketch generation steps")
    p.add_argument("--height",       type=int,   default=1024)
    p.add_argument("--width",        type=int,   default=1024)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def main():
    import json

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.scene is None and args.base_prompt is None:
        raise ValueError("Provide --scene (existing image) or --base_prompt (generate it)")

    print("Loading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    print(f"Mode: {args.mode}  Seed={args.seed}  Steps={args.num_steps}  Guidance={args.guidance}")

    if args.scene is not None:
        base_scene = Image.open(args.scene).convert("RGB")
    else:
        print(f"\nGenerating base scene: {args.base_prompt}")
        base_scene = generate_base(
            pipe, args.base_prompt,
            seed=args.seed, num_steps=args.num_steps,
            guidance_scale=args.guidance,
            height=args.height, width=args.width,
        )
        base_scene.save(os.path.join(args.out_dir, "step0_base.png"))
        print("Saved → step0_base.png")

    if args.config is not None:
        # ── Multi-step ──────────────────────────────────────────────────────
        config_dir = os.path.dirname(os.path.abspath(args.config))
        with open(args.config) as f:
            edits = json.load(f)

        # Resolve sketch/obj paths relative to the config file's directory
        for edit in edits:
            for key in ("sketch", "obj"):
                if key in edit and not os.path.isabs(edit[key]):
                    edit[key] = os.path.join(config_dir, edit[key])

        # Validate all input files exist before starting
        missing = []
        for edit in edits:
            for key in ("sketch", "obj"):
                if key in edit and not os.path.exists(edit[key]):
                    missing.append(f"  [{edit['name']}] {key}: {edit[key]}")
        if missing:
            raise FileNotFoundError(
                "Missing input files — put these in the inputs/ directory:\n"
                + "\n".join(missing)
            )

        run_multi_step(
            pipe, base_scene, edits,
            mode=args.mode,
            seed=args.seed, num_steps=args.num_steps,
            guidance_scale=args.guidance,
            lora_id=args.lora_id, lora_guidance=args.lora_guidance,
            height=args.height, width=args.width,
            out_dir=args.out_dir,
        )

    elif args.obj is not None and args.prompt is not None:
        # ── Single-step ─────────────────────────────────────────────────────
        obj_img = Image.open(args.obj).convert("RGB")
        run_fn  = run_stack if args.mode == "stack" else run_dual_ref
        result  = run_fn(
            pipe, base_scene, obj_img, args.prompt,
            seed=args.seed, num_steps=args.num_steps,
            guidance_scale=args.guidance,
            height=args.height, width=args.width,
        )
        out_path = os.path.join(args.out_dir, f"result_{args.mode}_seed{args.seed}.png")
        result.save(out_path)
        print(f"\nSaved → {out_path}")

        strip = Image.new("RGB", (args.width * 3, args.height))
        strip.paste(base_scene.resize((args.width, args.height)), (0, 0))
        strip.paste(obj_img.resize((args.width, args.height)), (args.width, 0))
        strip.paste(result, (args.width * 2, 0))
        strip.save(os.path.join(args.out_dir, f"strip_{args.mode}_seed{args.seed}.png"))
        print("Saved → strip (scene | obj | result)")

    else:
        raise ValueError("Provide either --config OR both --obj and --prompt")


if __name__ == "__main__":
    main()
