# -*- coding: utf-8 -*-
"""
phase1_roi.py — Residual Object Anchoring (ROA) for Identity-Preserving Insertion

Problem with all prior approaches
------------------------------------
  phase1_sketch_vlm : text can't transfer exact visual identity
  phase1_obj_kv     : K/V injection is a soft attention bias — model ignores it
  phase1_sdedit     : noisy-init seeds the ENTIRE gen latent from obj_img (wrong
                      position) and contaminates the scene with grey background

The invented formula: Residual Object Anchoring (ROA)
------------------------------------------------------
  Key insight: subtract the background from the object's VAE latent to isolate
  JUST the object's appearance signal, then add it back as a running correction
  throughout mid-range denoising — enough to anchor identity without overriding
  the model's natural integration of lighting and perspective.

  Step 1 — residual extraction:
      z_obj     = VAE_encode(obj_img)          (bicycle on grey background)
      z_grey    = VAE_encode(grey_canvas)      (pure grey, same size)
      z_residual = z_obj - z_grey              (background subtracted)
      z_res_mean = mean over object-region tokens of z_residual  → (64,) vector
                   (position-agnostic appearance descriptor)

  Step 2 — DAAM probe pass (7 steps, standard pipe):
      Use DAAM attention saliency to find WHERE in the scene the model would
      naturally place the bicycle given the text prompt → target_mask (4096 bool)

  Step 3 — ROA denoising loop (full 28 steps, pure-noise init):
      At each step i with sigma σ_i:
          x_next = scheduler.step(transformer([latents | ref_scene], σ_i, text))

          if σ_lo < σ_i < σ_hi:
              alpha = alpha_max * sin(π * (σ_i - σ_lo) / (σ_hi - σ_lo))
              latents = x_next + alpha * target_mask * z_res_mean
          else:
              latents = x_next

      sigma range [σ_lo=0.10, σ_hi=0.65]:
          - σ > σ_hi : too noisy, z_obj is unrecognizable — skip
          - σ in [σ_lo, σ_hi] : mid-range; inject with sinusoidal weight
          - σ < σ_lo : fine-detail generation — release control

  Why this is different from SDEdit:
    SDEdit seeds ONCE at start from noisy obj_img (wrong position, grey contamination).
    ROA seeds from PURE NOISE and applies REPEATED CORRECTIONS throughout the trajectory.
    The model denoises normally; we just continuously nudge the target zone toward
    "what this object's latent features look like at this noise level."

  Why this is different from K/V injection:
    K/V injection operates in attention space (soft bias on how tokens communicate).
    ROA operates in latent space (direct additive signal to what the model sees as input).

Pipeline
---------
  Stage A  : Sketch → LoRA → obj_img
  Stage VLM: VLM(scene, obj_img) → placement_prompt
  Stage ROA: [DAAM probe 7 steps] → target_mask
             [Custom loop 28 steps] with residual anchoring in target zone
  Accumulated preservation prompts (as in phase1_sdedit.py)

Usage
-----
  python NewWork/KontextEval/phase1_roi.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir results/phase1_roi \\
      --vlm_model Qwen/Qwen2-VL-2B-Instruct

  Sweep alpha_max first:
      python ... --alpha_sweep

Key flags
---------
  --alpha_max     float  Peak residual injection strength. Default 0.15.
                         0.05 = subtle nudge; 0.15 = moderate; 0.30 = strong.
  --sigma_lo      float  Lower sigma bound for injection window. Default 0.10.
  --sigma_hi      float  Upper sigma bound for injection window. Default 0.65.
  --derive_step   int    DAAM target-mask derivation step. Default 7.
  --top_k_frac    float  Fraction of gen tokens as object zone (DAAM). Default 0.10.
  --alpha_sweep          Run alpha=[0.05,0.15,0.30] on first object only.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ── helpers from phase1_dual_ref (vae encode / pack / ids / decode) ───────────
_dr_path = str(Path(__file__).parent / "phase1_dual_ref.py")
_drspec  = importlib.util.spec_from_file_location("phase1_dual_ref", _dr_path)
_drmod   = importlib.util.module_from_spec(_drspec)
_drspec.loader.exec_module(_drmod)

_vae_encode        = _drmod._vae_encode
_pack              = _drmod._pack
_img_ids           = _drmod._img_ids
_decode_latents    = _drmod._decode_latents

# ── helpers from phase1_obj_kv (token mask, DAAM probe, ObjKVInject) ─────────
_kv_path = str(Path(__file__).parent / "phase1_obj_kv.py")
_kvspec  = importlib.util.spec_from_file_location("phase1_obj_kv", _kv_path)
_kvmod   = importlib.util.module_from_spec(_kvspec)
_kvspec.loader.exec_module(_kvmod)

_compute_obj_token_mask = _kvmod._compute_obj_token_mask
_neutralize_white_bg    = _kvmod._neutralize_white_bg
ObjKVInject             = _kvmod.ObjKVInject

# ── helpers from phase1_sdedit (preservation prompt builder) ──────────────────
_sd_path = str(Path(__file__).parent / "phase1_sdedit.py")
_sdspec  = importlib.util.spec_from_file_location("phase1_sdedit", _sd_path)
_sdmod   = importlib.util.module_from_spec(_sdspec)
_sdspec.loader.exec_module(_sdmod)

_build_preserve_prompt = _sdmod._build_preserve_prompt

# ── sibling utilities ─────────────────────────────────────────────────────────
_comp_path = str(Path(__file__).parent / "phase1_composite.py")
_cspec = importlib.util.spec_from_file_location("phase1_composite", _comp_path)
_cmod  = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
save_grid    = _cmod.save_grid
run_standard = _cmod.run_standard

_vlm_path = str(Path(__file__).parent / "phase1_sketch_vlm.py")
_vspec = importlib.util.spec_from_file_location("phase1_sketch_vlm", _vlm_path)
_vmod  = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(_vmod)
generate_from_sketch        = _vmod.generate_from_sketch
vlm_generate_kontext_prompt = _vmod.vlm_generate_kontext_prompt

_sk_path = str(Path(__file__).parent / "phase1_sketch.py")
_sspec = importlib.util.spec_from_file_location("phase1_sketch", _sk_path)
_smod  = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(_smod)
load_vlm = _smod.load_vlm

# ── IncrementalEdit injection framework ───────────────────────────────────────
_ie_path = str(Path(__file__).parent.parent / "IncrementalEdit")
sys.path.insert(0, _ie_path)

from kontext_injection import (  # noqa: E402
    TIER_A, N_DOUBLE, N_LAYERS,
    InjectionState, ZoneMasks,
    get_t5_token_indices, set_determinism,
)
from mask_ops import topk_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle"},
    {"name": "vase",    "description": "white ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball"},
]
BASE_PROMPT  = "A modern living room with a sofa and a wooden coffee table."
LORA_ID      = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
LORA_TRIGGER = "Convert this sketch into real life version, follow exact structure."
_SEP = "═" * 60


# ── ROA bell-curve weight schedule ───────────────────────────────────────────

def _roi_alpha(sigma: float, sigma_lo: float, sigma_hi: float, alpha_max: float) -> float:
    """
    Sinusoidal bell weight that peaks at the midpoint of [sigma_lo, sigma_hi].
    Returns 0 outside the injection window, smoothly rises and falls inside it.

    Why sinusoidal:
      At σ > σ_hi : z_obj is almost pure noise — injecting it adds random signal
      At σ ∈ [σ_lo, σ_hi] : z_obj is recognizably the bicycle — inject here
      At σ < σ_lo : fine-detail generation — release control for natural finish
    """
    if sigma <= sigma_lo or sigma >= sigma_hi:
        return 0.0
    t = (sigma - sigma_lo) / (sigma_hi - sigma_lo)  # 0 → 1 across the window
    return float(alpha_max * np.sin(np.pi * t))


# ── Stage DAAM: probe pass → target mask ─────────────────────────────────────

def _daam_probe(
    pipe,
    scene:      Image.Image,
    prompt:     str,
    obj_noun:   str,
    n_gen:      int,
    vital_layers: List[int],
    derive_step: int,
    top_k_frac: float,
    seed:       int,
    num_steps:  int,
    guidance:   float,
    height:     int,
    width:      int,
    device:     str,
) -> np.ndarray:
    """
    Run a short probe pass in reasoning mode to find WHERE the model would place
    the object.  Returns target_mask (n_gen bool) for the ROA injection loop.

    Uses the existing ObjKVInject processor in reasoning mode with an empty
    obj_kv dict (no K/V injection) — the saliency accumulation still works.
    """
    concept_idx = get_t5_token_indices(pipe, prompt, obj_noun)
    everywhere  = np.ones(n_gen, dtype=bool)
    nothing     = np.zeros(n_gen, dtype=bool)

    if not concept_idx:
        print(f"    [DAAM] '{obj_noun}' not found in T5 tokens — using center fallback.")
        h = int(np.sqrt(n_gen)); w = n_gen // h
        m = np.zeros(n_gen, dtype=bool)
        m[(h // 4) * w : (3 * h // 4) * w] = True
        return m

    probe_steps = min(num_steps, derive_step + 4)
    state = InjectionState(
        mode="reasoning",
        vital_layers=set(vital_layers),
        n_gen=n_gen, n_ref=n_gen,
        cutoff_frac=(0.0, 0.6),
        strength=0.9,
        n_steps=probe_steps,
        derive_step=derive_step,
        concept_token_idx=concept_idx,
    )
    state.concept_layers = (
        {l for l in vital_layers if l < N_DOUBLE} or set(range(N_DOUBLE))
    )
    state.zones = ZoneMasks(
        background=everywhere, shell=nothing, target=nothing,
    ).to_device(device)

    # ObjKVInject with empty kv and obj_strength=0 → only background freeze + DAAM
    proc = ObjKVInject(
        state,
        obj_kv=dict(),
        obj_mask_np=np.ones(n_gen, dtype=bool),
        obj_strength=0.0,
    )
    proc._single_txt_len = 512
    pipe.transformer.set_attn_processor(proc)

    target_mask: Optional[np.ndarray] = None

    def _cb(pipe_ref, step_idx, timestep, cb_kwargs):
        nonlocal target_mask
        if state.mode == "reasoning" and step_idx == state.derive_step:
            if state.saliency_accum is not None and state.saliency_captures > 0:
                sal  = (state.saliency_accum / state.saliency_captures).cpu().numpy()
                tgt  = topk_mask(sal, top_k_frac)
                bg   = np.logical_and(everywhere, ~tgt)
                state.zones = ZoneMasks(
                    background=bg, shell=nothing, target=tgt,
                ).to_device(device)
                target_mask = tgt
                pct = 100.0 * tgt.mean()
                print(f"      [DAAM@{step_idx}] object zone: "
                      f"{tgt.sum()} / {n_gen} tokens ({pct:.1f}%)")
            state.mode = "edit"
        return cb_kwargs

    generator = set_determinism(seed)
    pipe(
        image=scene,
        prompt=prompt,
        num_inference_steps=probe_steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        max_sequence_length=512,
        generator=generator,
        output_type="pil",
        callback_on_step_end=_cb,
        callback_on_step_end_tensor_inputs=[],
    )

    if target_mask is None:
        print("    [DAAM] No saliency captured — using center fallback.")
        target_mask = np.zeros(n_gen, dtype=bool)
        h = int(np.sqrt(n_gen)); w = n_gen // h
        target_mask[(h // 4) * w : (3 * h // 4) * w] = True

    return target_mask


# ── Core ROA denoising loop ───────────────────────────────────────────────────

@torch.no_grad()
def run_roi_kontext(
    pipe,
    scene:      Image.Image,
    obj_img:    Image.Image,
    prompt:     str,
    obj_noun:    str,
    alpha_max:   float = 0.15,
    sigma_lo:    float = 0.10,
    sigma_hi:    float = 0.65,
    inject_mode: str   = "velocity",
    init_sigma:  float = 0.80,
    seed:        int   = 42,
    num_steps:   int   = 28,
    guidance:    float = 2.5,
    height:      int   = 1024,
    width:       int   = 1024,
    vital_layers: List[int] = None,
    derive_step:  int = 7,
    top_k_frac:   float = 0.10,
    device:      str   = "cuda",
) -> Image.Image:
    """
    Residual Object Anchoring (ROA) Kontext insertion.

    Token layout: standard Kontext 8192-token [gen (4096) | ref_scene (4096)]

    Init strategy (init_sigma):
        init_sigma > 0 (default 0.80): seed gen latent from noisy z_obj at that σ level.
            latents = (1 - init_sigma) * z_obj + init_sigma * noise
            This gives FLUX a bicycle prior so it knows WHAT to generate.
            The scene reference + ROA correction then guides WHERE and HOW.
        init_sigma = 0.0 or 1.0: pure noise (FLUX decides everything — often ignores obj).

    Why noisy-obj init is needed:
        Kontext's scene reference is a STRONG attractor — without any object prior in the
        gen latent, FLUX just reproduces the scene. A noisy-obj init (σ≈0.8) gives a loose
        bicycle structure that Kontext refines into a naturally-integrated object.

    Three injection modes (--inject_mode flag):

      'velocity' (default, FreqEdit-inspired):
          Steer the model's velocity prediction toward z_obj BEFORE the scheduler step.
          v_to_obj = mean_{obj_tokens}(z_obj) - mean_{obj_tokens}(latents)
          v_mod = (1-alpha) * v_model + alpha * v_to_obj   [in target zone]
          Natural for flow-matching; doesn't accumulate error across steps.

      'latent' (original ROA):
          Add z_residual mean to latents AFTER the scheduler step.
          z_res_mean = mean_{obj_tokens}(z_obj - z_grey)
          latents += alpha * z_res_mean   [in target zone]
          Simple additive bias; cumulative but bounded by bell schedule.

      'blend' (Add-it inspired):
          Blend target-zone latents toward re-noised z_obj AFTER the scheduler step.
          z_obj_at_t = (1-sigma) * z_obj + sigma * noise   [re-noised to current level]
          latents = (1-alpha) * latents + alpha * z_obj_at_t_mean   [in target zone]
          Matches noise level at each step; strongest identity lock.
    """
    if vital_layers is None:
        vital_layers = list(TIER_A)

    device_t = next(pipe.transformer.parameters()).device
    dtype    = next(pipe.transformer.parameters()).dtype

    vae_sf  = getattr(pipe, "vae_scale_factor", 8)
    h_lat   = height // (vae_sf * 2)
    w_lat   = width  // (vae_sf * 2)
    n_gen   = h_lat * w_lat   # 4096 for 1024×1024

    # ── 1. Text embeddings ────────────────────────────────────────────────────
    prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        device=device_t,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    # ── 2. Scene reference latent ─────────────────────────────────────────────
    ref_latents = _vae_encode(pipe, scene, height, width, device_t, dtype)
    ref_packed  = _pack(pipe, ref_latents)                     # (1, 4096, 64)
    ref_ids     = _img_ids(pipe, ref_latents, device_t, dtype) # (4096, 3)

    # ── 3. Object residual: z_obj - z_grey ───────────────────────────────────
    # z_grey: encode a uniform grey canvas at the same resolution
    grey_canvas = Image.new("RGB", (width, height), (128, 128, 128))
    z_grey_latents = _vae_encode(pipe, grey_canvas, height, width, device_t, dtype)
    z_grey_packed  = _pack(pipe, z_grey_latents)  # (1, 4096, 64)

    obj_neutral    = _neutralize_white_bg(obj_img)
    z_obj_latents  = _vae_encode(pipe, obj_neutral, height, width, device_t, dtype)
    z_obj_packed   = _pack(pipe, z_obj_latents)   # (1, 4096, 64)

    z_residual = z_obj_packed - z_grey_packed      # (1, 4096, 64) pure object signal

    # Compute obj-region token mask (non-grey/non-white pixels in obj_img)
    obj_token_mask = _compute_obj_token_mask(obj_img, h_lat, w_lat)
    obj_idx = np.where(obj_token_mask)[0]
    n_obj   = len(obj_idx)
    print(f"    [ROA] Object-region tokens: {n_obj}/{n_gen} ({100*n_obj/n_gen:.1f}%)")

    if n_obj == 0:
        obj_idx = np.arange(n_gen // 4, 3 * n_gen // 4)

    obj_idx_t = torch.from_numpy(obj_idx).to(device_t)

    # Mean over object-region tokens → (1, 1, 64) appearance anchor
    z_res_mean = z_residual[:, obj_idx_t, :].mean(dim=1, keepdim=True)  # (1, 1, 64)

    print(f"    [ROA] z_res_mean norm: {z_res_mean.norm().item():.4f}")
    print(f"    [ROA] Bell window: σ ∈ [{sigma_lo:.2f}, {sigma_hi:.2f}], "
          f"peak at σ={0.5*(sigma_lo+sigma_hi):.2f}, alpha_max={alpha_max}")

    # ── 4. DAAM probe → target_mask ──────────────────────────────────────────
    print(f"    [ROA] Running DAAM probe ({derive_step} steps) ...")
    target_mask_np = _daam_probe(
        pipe=pipe, scene=scene, prompt=prompt, obj_noun=obj_noun,
        n_gen=n_gen, vital_layers=vital_layers,
        derive_step=derive_step, top_k_frac=top_k_frac,
        seed=seed, num_steps=num_steps,
        guidance=guidance, height=height, width=width, device=device,
    )
    target_idx    = np.where(target_mask_np)[0]
    target_idx_t  = torch.from_numpy(target_idx).to(device_t)

    # ── 5. Reset attn processors to default before the custom loop ────────────
    # _daam_probe leaves ObjKVInject set; clear it before the manual forward calls.
    try:
        from diffusers.models.attention_processor import FluxAttnProcessor2_0
        pipe.transformer.set_attn_processor(FluxAttnProcessor2_0())
    except ImportError:
        # Older diffusers: instantiate from the first existing processor's class
        first_proc = next(iter(pipe.transformer.attn_processors.values()))
        pipe.transformer.set_attn_processor(type(first_proc)())

    # ── 6. Scheduler setup ────────────────────────────────────────────────────
    try:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    except ImportError:
        from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift

    image_seq_len = n_gen  # 4096 for 1024×1024
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len",  4096),
        pipe.scheduler.config.get("base_shift",          0.5),
        pipe.scheduler.config.get("max_shift",           1.16),
    )
    sigmas_np = np.linspace(1.0, 1.0 / num_steps, num_steps)
    pipe.scheduler.set_timesteps(sigmas=sigmas_np, device=device_t, mu=mu)
    timesteps = pipe.scheduler.timesteps  # (T,)

    # ── 7. Gen latent IDs and init ────────────────────────────────────────────
    gen_ids   = _img_ids(pipe, ref_latents, device_t, dtype)   # (4096, 3) position grid
    generator = torch.Generator(device=device_t).manual_seed(seed)
    noise     = torch.randn(1, n_gen, z_obj_packed.shape[-1],
                             device=device_t, dtype=dtype, generator=generator)

    # Noisy-obj init: give FLUX a bicycle prior so the scene reference doesn't
    # completely suppress the object. Pure noise (init_sigma≈1) → FLUX just
    # reproduces the scene; low sigma (≈0.5) → too rigid, wrong position.
    # Default init_sigma=0.80 gives loose bicycle structure + full scene integration.
    _s = float(np.clip(init_sigma, 0.0, 1.0))
    if _s < 0.95:
        latents = (1.0 - _s) * z_obj_packed + _s * noise
        print(f"    [ROA] Init: noisy-obj  init_sigma={_s:.2f}  "
              f"(1-σ)*z_obj + σ*noise")
    else:
        latents = noise
        print(f"    [ROA] Init: pure noise  (init_sigma={_s:.2f})")

    # ── 8. Guidance embedding ─────────────────────────────────────────────────
    guidance_tensor = None
    if getattr(pipe.transformer.config, "guidance_embeds", False):
        guidance_tensor = torch.full([1], guidance, dtype=dtype, device=device_t)

    # ── 9. ROA denoising loop ─────────────────────────────────────────────────
    # For 'blend' mode: fixed noise used to re-noise z_obj at each sigma level.
    # Using a fixed noise tensor keeps the blend deterministic across steps.
    noise_for_blend = torch.randn_like(z_obj_packed,
                                       generator=torch.Generator(device=device_t).manual_seed(seed + 1))

    inject_count = 0
    for i, t in enumerate(timesteps):
        model_input = torch.cat([latents, ref_packed], dim=1)  # (1, 8192, 64)
        img_ids     = torch.cat([gen_ids,  ref_ids],   dim=0)  # (8192, 3)

        noise_pred = pipe.transformer(
            hidden_states         = model_input,
            timestep              = t.expand(1) / 1000,
            guidance              = guidance_tensor,
            pooled_projections    = pooled_embeds,
            encoder_hidden_states = prompt_embeds,
            txt_ids               = text_ids,
            img_ids               = img_ids,
            return_dict           = False,
        )[0][:, :n_gen, :]                                      # (1, 4096, 64)

        sigma_t = float(sigmas_np[i])
        alpha   = _roi_alpha(sigma_t, sigma_lo, sigma_hi, alpha_max)
        n_tgt   = len(target_idx_t)
        active  = alpha > 0.0 and n_tgt > 0

        if inject_mode == "velocity" and active:
            # FreqEdit-inspired: steer velocity toward obj BEFORE scheduler step.
            # v_to_obj = direction from current latent's obj-region mean → z_obj mean.
            # Using means makes this position-agnostic (no spatial mismatch).
            curr_obj_mean = latents[:, obj_idx_t, :].mean(dim=1, keepdim=True)     # (1,1,64)
            z_obj_mean    = z_obj_packed[:, obj_idx_t, :].mean(dim=1, keepdim=True) # (1,1,64)
            v_to_obj      = z_obj_mean - curr_obj_mean                              # (1,1,64)

            noise_pred_mod = noise_pred.clone()
            noise_pred_mod[:, target_idx_t, :] = (
                (1.0 - alpha) * noise_pred[:, target_idx_t, :] +
                alpha * v_to_obj.expand(1, n_tgt, -1)
            )
            latents = pipe.scheduler.step(
                noise_pred_mod, t, latents, return_dict=False
            )[0]
            inject_count += 1

        else:
            # Standard scheduler step (used for 'latent'/'blend' modes, or when alpha=0)
            latents = pipe.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

            if inject_mode == "latent" and active:
                # Original ROA: add z_residual mean to latents in target zone.
                anchor = z_res_mean.expand(1, n_tgt, -1)  # (1, n_tgt, 64)
                latents[:, target_idx_t, :] = (
                    latents[:, target_idx_t, :] + alpha * anchor
                )
                inject_count += 1

            elif inject_mode == "blend" and active:
                # Add-it inspired: blend target zone toward re-noised z_obj at current σ.
                # Uses z_obj mean over obj-region tokens (position-agnostic).
                z_obj_at_t = (1.0 - sigma_t) * z_obj_packed + sigma_t * noise_for_blend
                z_blend_mean = z_obj_at_t[:, obj_idx_t, :].mean(dim=1, keepdim=True)  # (1,1,64)
                latents[:, target_idx_t, :] = (
                    (1.0 - alpha) * latents[:, target_idx_t, :] +
                    alpha * z_blend_mean.expand(1, n_tgt, -1)
                )
                inject_count += 1

        if (i + 1) % 7 == 0 or i == 0 or i + 1 == len(timesteps):
            print(f"      step {i+1:3d}/{len(timesteps)}  σ={sigma_t:.3f}  "
                  f"α={alpha:.4f}  mode={inject_mode}", flush=True)

    print(f"    [ROA] Applied injection at {inject_count}/{len(timesteps)} steps")

    # ── 10. Decode ────────────────────────────────────────────────────────────
    return _decode_latents(pipe, latents, height, width)


# ── Alpha sweep helper ────────────────────────────────────────────────────────

def run_alpha_sweep(
    pipe,
    scene:       Image.Image,
    obj_img:     Image.Image,
    prompt:      str,
    obj_noun:    str,
    name:        str,
    seed:        int,
    num_steps:   int,
    guidance:    float,
    height:      int,
    width:       int,
    out_dir:     str,
    vital_layers: List[int],
    alphas:      Tuple[float, ...] = (0.05, 0.15, 0.30),
    sigma_lo:    float = 0.10,
    sigma_hi:    float = 0.65,
    inject_mode: str   = "velocity",
    init_sigma:  float = 0.80,
    device:      str   = "cuda",
) -> None:
    results = [scene]
    titles  = ["scene"]
    for alpha in alphas:
        print(f"  [sweep] alpha_max={alpha:.2f}  mode={inject_mode}  "
              f"init_sigma={init_sigma:.2f} ...")
        r = run_roi_kontext(
            pipe=pipe, scene=scene, obj_img=obj_img, prompt=prompt,
            obj_noun=obj_noun, alpha_max=alpha,
            sigma_lo=sigma_lo, sigma_hi=sigma_hi,
            inject_mode=inject_mode, init_sigma=init_sigma,
            seed=seed, num_steps=num_steps, guidance=guidance,
            height=height, width=width, vital_layers=vital_layers,
            device=device,
        )
        p = os.path.join(out_dir, f"sweep_{name}_a{int(alpha * 100):03d}.png")
        r.save(p)
        results.append(r)
        titles.append(f"α={alpha:.2f}")

    grid_path = os.path.join(out_dir, f"sweep_{name}_grid.png")
    save_grid(results, titles, grid_path, ncols=len(results))
    print(f"  [sweep] Grid: {grid_path}")
    print(f"          Inspect and choose --alpha_max before the full chain.")


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_roi_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    lora_id:        str,
    vlm_pair:       Tuple,
    vital_layers:   List[int],
    seed:           int,
    num_steps:      int,
    lora_guidance:  float,
    scene_guidance: float,
    alpha_max:      float,
    sigma_lo:       float,
    sigma_hi:       float,
    inject_mode:    str,
    init_sigma:     float,
    derive_step:    int,
    top_k_frac:     float,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
) -> List[Image.Image]:
    """
    Incremental insertion: Base → +Bike → +Vase → +Ball.

    Per object:
      Stage A  : sketch → LoRA → obj_img
      Stage VLM: VLM(scene, obj_img) → placement_prompt
      Stage ROA: DAAM probe → target_mask
                 custom loop → scene with anchored object appearance
    Accumulated preservation prompts appended at each step.
    """
    vlm_model, vlm_proc = vlm_pair
    results   = [base]
    scene     = base
    preserved: List[str] = []

    for i, edit in enumerate(edits):
        name = edit["name"]
        desc = edit["description"]

        sketch_path = os.path.join(sketch_dir, f"{name}.png")
        if not os.path.isfile(sketch_path):
            sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found in {sketch_dir!r}.\n"
                f"Expected '{name}.png' or 'sketch_{name}.png'."
            )

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}")
        if preserved:
            print(f"  Preserving: {preserved}")
        print(f"{'─'*60}")

        # Stage A: sketch → object image
        print(f"  [A] Generating '{desc}' from sketch ...")
        obj_img = generate_from_sketch(
            pipe=pipe, sketch_path=sketch_path, description=desc,
            seed=seed, num_steps=num_steps, guidance=lora_guidance,
            height=height, width=width, lora_id=lora_id, device=device,
        )
        obj_path = os.path.join(out_dir, f"obj_gen_{name}.png")
        obj_img.save(obj_path)
        print(f"      Saved: {obj_path}")

        # Stage VLM: scene + obj_img → placement prompt
        print(f"  [VLM] Generating placement prompt for '{name}' ...")
        raw_prompt   = vlm_generate_kontext_prompt(
            vlm_model=vlm_model, vlm_processor=vlm_proc,
            scene_img=scene, obj_img=obj_img, description=desc,
        )
        final_prompt = _build_preserve_prompt(raw_prompt, preserved)
        obj_noun     = desc.split(" with ")[0].split()[-1].lower()

        print(f"\n  {_SEP}")
        print(f"  [VLM → Prompt]  {name}")
        print(f"  {_SEP}")
        for line in final_prompt.splitlines():
            print(f"  {line}")
        print(f"  {_SEP}\n")
        with open(os.path.join(out_dir, f"vlm_prompt_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(final_prompt)

        # Stage ROA: DAAM probe + residual anchoring loop
        print(f"  [ROA] Inserting '{name}' with Residual Object Anchoring ...")
        print(f"        mode={inject_mode}  alpha_max={alpha_max}  "
              f"sigma=[{sigma_lo},{sigma_hi}]  init_sigma={init_sigma}")
        next_scene = run_roi_kontext(
            pipe=pipe, scene=scene, obj_img=obj_img, prompt=final_prompt,
            obj_noun=obj_noun, alpha_max=alpha_max,
            sigma_lo=sigma_lo, sigma_hi=sigma_hi,
            inject_mode=inject_mode, init_sigma=init_sigma,
            seed=seed, num_steps=num_steps,
            guidance=scene_guidance,
            height=height, width=width,
            vital_layers=vital_layers,
            derive_step=derive_step, top_k_frac=top_k_frac,
            device=device,
        )
        result_path = os.path.join(out_dir, f"result_step{i+1}_{name}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        scene = next_scene
        results.append(scene)
        preserved.append(desc)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return results


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Residual Object Anchoring — novel per-step latent injection for identity preservation."
    )
    p.add_argument("--sketch_dir",    required=True,
                   help="Folder containing {name}.png or sketch_{name}.png files.")
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--cache_dir",     default="./models")
    p.add_argument("--out_dir",       default="results/phase1_roi")
    p.add_argument("--config",        default=None,
                   help="JSON list of {name, description}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",       default=LORA_ID)
    p.add_argument("--lora_guidance", type=float, default=4.0)
    p.add_argument("--guidance",      type=float, default=2.5)
    p.add_argument("--alpha_max",     type=float, default=0.15,
                   help="Peak residual injection strength (bell-curve max). "
                        "0.05=subtle; 0.15=moderate; 0.30=strong. Default 0.15.")
    p.add_argument("--sigma_lo",      type=float, default=0.10,
                   help="Lower sigma bound for injection window. Default 0.10.")
    p.add_argument("--sigma_hi",      type=float, default=0.65,
                   help="Upper sigma bound for injection window. Default 0.65.")
    p.add_argument("--init_sigma",     type=float, default=0.80,
                   help="Noisy-obj init level. 0.80=default (loose bicycle structure). "
                        "0.5=tighter identity (may look pasted). 0.95+=pure noise (no obj).")
    p.add_argument("--inject_mode",   default="velocity",
                   choices=["velocity", "latent", "blend"],
                   help="Injection mode. 'velocity' (default, FreqEdit-inspired): steer "
                        "velocity toward obj before scheduler step. 'latent': add residual "
                        "mean after step. 'blend': blend latents toward re-noised obj.")
    p.add_argument("--alpha_sweep",   action="store_true",
                   help="Run alpha=[0.05,0.15,0.30] on first object only and exit. "
                        "Inspect sweep grid to pick --alpha_max before a full run.")
    p.add_argument("--derive_step",   type=int, default=7,
                   help="Step at which DAAM mask is extracted from probe pass.")
    p.add_argument("--top_k_frac",    type=float, default=0.10,
                   help="Fraction of gen tokens to mark as object zone (DAAM). Default 0.10.")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--num_steps",     type=int, default=28)
    p.add_argument("--height",        type=int, default=1024)
    p.add_argument("--width",         type=int, default=1024)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--vlm_model",     default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--vlm_device",    default="cpu")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)
        print(f"Loaded {len(edits)} edits from {args.config}")

    print(f"\n{_SEP}")
    print(f"  phase1_roi  —  Residual Object Anchoring")
    print(f"{_SEP}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  inject_mode  : {args.inject_mode}")
    print(f"  alpha_max    : {args.alpha_max}  sweep: {args.alpha_sweep}")
    print(f"  sigma window : [{args.sigma_lo}, {args.sigma_hi}]")
    print(f"  VLM          : {args.vlm_model}  [{args.vlm_device}]")
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    print("Loading VLM ...")
    vlm_pair = load_vlm(args.vlm_model, args.cache_dir, args.vlm_device)

    print("\nLoading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    vital_layers = list(TIER_A)

    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base_path = os.path.join(args.out_dir, "base_scene.png")
    base.save(base_path)
    print(f"  Saved: {base_path}")

    if args.alpha_sweep:
        print("\n=== Alpha sweep (first object only) ===")
        edit = edits[0]
        name = edit["name"]
        desc = edit["description"]
        sketch_path = os.path.join(args.sketch_dir, f"{name}.png")
        if not os.path.isfile(sketch_path):
            sketch_path = os.path.join(args.sketch_dir, f"sketch_{name}.png")

        vlm_model, vlm_proc = vlm_pair
        obj_img = generate_from_sketch(
            pipe=pipe, sketch_path=sketch_path, description=desc,
            seed=args.seed, num_steps=args.num_steps, guidance=args.lora_guidance,
            height=args.height, width=args.width, lora_id=args.lora_id, device=args.device,
        )
        obj_img.save(os.path.join(args.out_dir, f"obj_gen_{name}.png"))

        raw_prompt = vlm_generate_kontext_prompt(
            vlm_model=vlm_model, vlm_processor=vlm_proc,
            scene_img=base, obj_img=obj_img, description=desc,
        )
        obj_noun = desc.split(" with ")[0].split()[-1].lower()

        run_alpha_sweep(
            pipe=pipe, scene=base, obj_img=obj_img, prompt=raw_prompt,
            obj_noun=obj_noun, name=name,
            seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
            height=args.height, width=args.width,
            out_dir=args.out_dir, vital_layers=vital_layers,
            sigma_lo=args.sigma_lo, sigma_hi=args.sigma_hi,
            inject_mode=args.inject_mode, init_sigma=args.init_sigma,
            device=args.device,
        )
        print("\nSweep complete — inspect grid then re-run without --alpha_sweep.")
        return

    results = run_roi_chain(
        pipe=pipe, base=base, edits=edits,
        sketch_dir=args.sketch_dir, lora_id=args.lora_id,
        vlm_pair=vlm_pair, vital_layers=vital_layers,
        seed=args.seed, num_steps=args.num_steps,
        lora_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        alpha_max=args.alpha_max,
        sigma_lo=args.sigma_lo, sigma_hi=args.sigma_hi,
        inject_mode=args.inject_mode, init_sigma=args.init_sigma,
        derive_step=args.derive_step, top_k_frac=args.top_k_frac,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
    )

    all_imgs  = results
    all_lbls  = ["base"] + [e["name"] for e in edits]
    grid_path = os.path.join(args.out_dir, "chain_grid.png")
    save_grid(all_imgs, all_lbls, grid_path, ncols=len(all_imgs))
    print(f"\n{_SEP}")
    print(f"  Chain complete.  Grid: {grid_path}")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
