# -*- coding: utf-8 -*-
"""
phase1_collage_kontext.py — AnyDoor's Collage Method inside FLUX Kontext

How AnyDoor works (the idea we're stealing)
--------------------------------------------
AnyDoor builds a "hint collage": it takes the reference object, extracts its
Sobel edge map (structure without colour), then PASTES that into the target scene
at the desired location.  This collage is fed to a ControlNet so the model
SEES the object's shape at the right place and generates natural integration.

How we implement it in Kontext
-------------------------------
FLUX Kontext is conditioned on a reference image — that IS our "ControlNet".
So instead of a raw scene, we pass a COLLAGE SCENE as the Kontext reference:

  collage_scene = scene with obj pasted (soft-masked) at the target location

The VLM tells us WHERE.  FLUX then:
  1. Sees the obj shape/colour at the correct position in the reference
  2. Denoises from pure noise conditioned on that reference
  3. Naturally integrates lighting, shadows, perspective

Key differences from all previous attempts
-------------------------------------------
  phase1_sketch_vlm  : Kontext reference = raw scene, text describes obj → identity lost
  phase1_obj_kv      : K/V injection in attention space → too soft
  phase1_sdedit      : Noisy-obj init → wrong position, grey bg contamination
  phase1_roi         : Additive latent signal → FLUX doesn't respond strongly enough
  THIS FILE          : Kontext reference = scene WITH OBJ ALREADY VISIBLE at target
                       → FLUX has the full visual prior: shape, colour, position

Collage modes (--collage_mode)
-------------------------------
  'full'   (default) : paste actual obj pixels soft-masked into scene
                       Best colour/texture fidelity.
  'sobel'            : paste Sobel edge map of obj instead of pixels
                       More adaptation freedom; matches AnyDoor exactly.
  'blend'            : weighted mix: alpha*full + (1-alpha)*sobel
                       Balance between identity lock and natural adaptation.

Pipeline
---------
  Stage A   : Sketch → LoRA FLUX → obj_img
  Stage BBOX: Placement mask → bbox           (no VLM — mask_{name}.png required)
  Stage POSE: scene_crop = scene[y1:y2, x1:x2] expanded by 20%
              mini_collage = obj pasted into upscaled crop
              Kontext(mini_collage, pose_prompt) → posed result
              diff(posed, scene_crop) → obj_img on white bg
  Stage MASK: Threshold obj_img → ref_mask
  Stage COL : Build collage_scene by pasting obj_img at placement bbox
  Stage K   : FLUX Kontext(reference=collage_scene, prompt=blend_prompt) → result
  Loop      : result → next scene

Usage
-----
  python NewWork/KontextEval/phase1_collage_kontext.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir results/phase1_collage \\
      --vlm_model Qwen/Qwen2.5-VL-7B-Instruct

Key flags
----------
  --collage_mode   full | sobel | blend   Default: full
  --guidance       float  Kontext CFG scale. Default 2.5.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FluxKontextPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from PIL import Image


# ── Pipeline loading ──────────────────────────────────────────────────────────

def load_kontext_pipeline(
    model_path: str = "black-forest-labs/FLUX.1-Kontext-dev",
    hf_token: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = "./models",
    cpu_offload: bool = False,
) -> FluxKontextPipeline:
    pipe = FluxKontextPipeline.from_pretrained(
        model_path, torch_dtype=dtype, token=hf_token, cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── Kontext inference ─────────────────────────────────────────────────────────

def run_standard(
    pipe, canvas: Image.Image, prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    bcg_callback=None,
) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    cb_kwargs: dict = {}
    if bcg_callback is not None:
        cb_kwargs["callback_on_step_end"] = bcg_callback
        cb_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipe(
        prompt=prompt, image=canvas,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width, generator=generator,
        **cb_kwargs,
    ).images[0]


# ── Result grid ───────────────────────────────────────────────────────────────

def save_grid(images, titles, path, ncols=None, figsize_per_cell=(4, 4)):
    n     = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
    )
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Stage A: Sketch → Object image ───────────────────────────────────────────

_LORA_TRIGGER = "Convert this sketch into real life version, follow exact structure."


def generate_from_sketch(
    pipe, sketch_path: str, description: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int, lora_id: str, device: str,
) -> Image.Image:
    sketch_pil = Image.open(sketch_path).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )
    print(f"  Loading LoRA: {lora_id}")
    pipe.load_lora_weights(lora_id)
    prompt = (
        f"{_LORA_TRIGGER} {description} on a plain white background, "
        "photorealistic, no shadows, studio lighting, high quality."
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    obj_img = pipe(
        image=sketch_pil, prompt=prompt,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width,
        max_sequence_length=512,
        generator=generator, output_type="pil",
    ).images[0]
    pipe.unload_lora_weights()
    return obj_img


# ── VLM loading ───────────────────────────────────────────────────────────────

def load_vlm(model_id: str, cache_dir: str, device: str = "cpu"):
    """
    Load a Qwen2-VL or Qwen2.5-VL model for placement bbox prediction.

    device="cpu"  → bfloat16, safe alongside FLUX on any GPU.
    device="cuda" → tries 4-bit NF4 (bitsandbytes) first, then bf16 fallback.
                    pip install -U bitsandbytes>=0.46.1  to enable 4-bit.
    """
    from transformers import AutoProcessor

    print(f"  Loading VLM '{model_id}' on {device} ...")

    def _load_model(load_kwargs: dict, to_device: bool):
        is_qwen25 = "Qwen2.5" in model_id or "Qwen2_5" in model_id
        classes = []
        if is_qwen25:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                classes.append(Qwen2_5_VLForConditionalGeneration)
            except ImportError:
                pass
        try:
            from transformers import Qwen2VLForConditionalGeneration
            classes.append(Qwen2VLForConditionalGeneration)
        except ImportError:
            pass
        if not is_qwen25:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                classes.append(Qwen2_5_VLForConditionalGeneration)
            except ImportError:
                pass
        last_err = None
        for cls in classes:
            try:
                m = cls.from_pretrained(model_id, **load_kwargs)
                if to_device:
                    m = m.to(device)
                return m.eval()
            except Exception as e:
                last_err = e
        try:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
            if to_device:
                m = m.to(device)
            return m.eval()
        except Exception as e:
            last_err = e
        raise RuntimeError(f"Could not load VLM '{model_id}'. Last error: {last_err}")

    model = None
    mode  = ""
    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            model = _load_model(
                dict(quantization_config=quant_cfg, device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            mode = "4-bit NF4 GPU"
        except Exception as e:
            print(f"    [VLM] 4-bit load failed ({e!s:.80})")
            print(f"    [VLM] Falling back to bf16 GPU")
        if model is None:
            model = _load_model(
                dict(torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            mode = "bf16 GPU"
    else:
        model = _load_model(
            dict(torch_dtype=torch.bfloat16, cache_dir=cache_dir),
            to_device=True,
        )
        mode = "bf16 CPU"

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    print(f"  VLM loaded  ({mode})")
    return model, processor

# ── constants ─────────────────────────────────────────────────────────────────
EDITS: List[dict] = [
    # ── Insertion ─────────────────────────────────────────────────────────────
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "insert"},
    {"name": "vase",    "description": "black ceramic vase with flowers",  "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "insert"},
    # ── Removal ───────────────────────────────────────────────────────────────
    {"name": "ball",    "description": "yellow rubber ball",               "action": "remove"},
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "remove"},
]
BASE_PROMPT = "A empty room with a wooden floor, white walls, and a window letting in natural light."
LORA_ID     = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
_SEP        = "═" * 60


# ── Object mask extraction (same logic as phase1_obj_kv) ─────────────────────

def _compute_obj_mask(obj_img: Image.Image,
                       grey: tuple = (128, 128, 128),
                       tolerance: int = 20) -> np.ndarray:
    """
    Binary uint8 mask (H, W): 1 = object pixel, 0 = background.
    Detects both white (LoRA output) and grey (neutralised) backgrounds.
    """
    arr = np.array(obj_img.convert("RGB"), dtype=np.int32)
    is_grey  = ((np.abs(arr[:, :, 0] - grey[0]) <= tolerance) &
                (np.abs(arr[:, :, 1] - grey[1]) <= tolerance) &
                (np.abs(arr[:, :, 2] - grey[2]) <= tolerance))
    is_white = (arr[:, :, 0] >= 230) & (arr[:, :, 1] >= 230) & (arr[:, :, 2] >= 230)
    return (~(is_grey | is_white)).astype(np.uint8)


# ── Placement mask → bbox ────────────────────────────────────────────────────

def _bbox_from_placement_mask(
    mask_path: str,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """
    Extract the bounding box of the white region in a user-drawn placement mask.
    White pixels (>127) mark where the object should be placed.
    Mask resolution is rescaled to (width × height) pixel space.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_np  = np.array(mask_img)
    mh, mw   = mask_np.shape
    ys, xs   = np.where(mask_np > 127)
    if len(ys) == 0:
        return width // 6, height // 2, 5 * width // 6, height
    x1 = max(0,      int(xs.min() * width  / mw))
    y1 = max(0,      int(ys.min() * height / mh))
    x2 = min(width,  int((xs.max() + 1) * width  / mw))
    y2 = min(height, int((ys.max() + 1) * height / mh))
    return x1, y1, x2, y2


# ── K/V injection: capture + inject obj appearance at target zone ─────────────
#
# Optional second pass on top of the collage:
#   1. 1-step capture: run FLUX with obj_img as canvas, record K/V at TIER_A layers
#   2. Full denoising: run FLUX with collage_scene as canvas,
#      inject captured K/V at target-zone gen tokens
#
# Why this helps on top of collage:
#   The collage gives FLUX the visual prior (shape, colour, position).
#   K/V injection reinforces the appearance signal in attention space at the
#   exact target zone so FLUX doesn't drift the colour/texture during denoising.

_TIER_A        = {0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56}
_SINGLE_TXT_LEN = 512   # T5 sequence length used by FluxKontextPipeline


def _neutralize_white_bg(img: Image.Image, threshold: int = 240,
                          fill: tuple = (128, 128, 128)) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    white = (arr[:, :, 0] >= threshold) & (arr[:, :, 1] >= threshold) & (arr[:, :, 2] >= threshold)
    arr[white] = fill
    return Image.fromarray(arr)


def _obj_token_mask(obj_img: Image.Image, h_lat: int, w_lat: int,
                    grey: tuple = (128, 128, 128), tol: int = 10,
                    min_frac: float = 0.05) -> np.ndarray:
    """Bool mask (h_lat*w_lat,): True = object token, False = background."""
    arr = np.array(obj_img.convert("RGB").resize((w_lat, h_lat), Image.LANCZOS))
    is_grey  = (np.abs(arr[:,:,0].astype(int) - grey[0]) <= tol) & \
               (np.abs(arr[:,:,1].astype(int) - grey[1]) <= tol) & \
               (np.abs(arr[:,:,2].astype(int) - grey[2]) <= tol)
    is_white = (arr[:,:,0] >= 230) & (arr[:,:,1] >= 230) & (arr[:,:,2] >= 230)
    mask = ~(is_grey | is_white)
    if mask.mean() < min_frac:
        mask[:] = False
        mask[h_lat//4:3*h_lat//4, w_lat//4:3*w_lat//4] = True
    return mask.reshape(-1)


def _placement_mask_to_token_zone(mask_path: str, height: int, width: int,
                                   pipe) -> np.ndarray:
    """Convert a user-drawn placement mask to a flat bool token mask (n_gen,)."""
    vae_sf    = getattr(pipe, "vae_scale_factor", 8)
    h_lat, w_lat = height // (vae_sf * 2), width // (vae_sf * 2)
    mask_down = Image.open(mask_path).convert("L").resize((w_lat, h_lat), Image.NEAREST)
    zone = (np.array(mask_down) > 127).reshape(-1)
    print(f"    [KV-zone] {zone.sum()} / {h_lat * w_lat} tokens ({100*zone.mean():.1f}%)")
    return zone


class _KVCapture:
    """
    Attention processor: performs standard attention AND records K/V from the
    Kontext ref-token slice at TIER_A layers during a 1-step capture pass.

    In Kontext hidden_states = [gen_tokens | ref_tokens]. Ref starts at
    img_offset + n_gen.  We store (K, V) on CPU for later injection.
    """
    def __init__(self, n_gen: int):
        self.n_gen = n_gen
        self.kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer = 0

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        if self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                self.kv[self._layer] = (
                    k[:, :, r_lo:r_hi, :].detach().cpu(),
                    v[:, :, r_lo:r_hi, :].detach().cpu(),
                )

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[1](attn.to_out[0](out))
            enc_out = attn.to_add_out(enc_out)
            self._layer += 1
            return out, enc_out
        self._layer += 1
        return out


class _KVInject:
    """
    Attention processor: injects captured obj K/V into target-zone gen tokens
    at TIER_A layers during the active denoising window (cutoff_frac).
    """
    def __init__(self, n_gen: int,
                 obj_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                 obj_mask_np: np.ndarray,
                 target_zone: np.ndarray, obj_strength: float,
                 n_steps: int, cutoff_frac: Tuple[float, float]):
        self.n_gen       = n_gen
        self.obj_kv      = obj_kv
        self.obj_mask_np = obj_mask_np
        self.target_zone = target_zone   # bool (n_gen,)
        self.obj_strength = obj_strength
        self.n_steps     = n_steps
        self.cutoff_frac = cutoff_frac
        self._layer = 0
        self._step  = 0

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        lo = int(self.cutoff_frac[0] * self.n_steps)
        hi = int(self.cutoff_frac[1] * self.n_steps)
        if lo <= self._step < hi and self._layer in self.obj_kv:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off, img_off + self.n_gen

            k_gen = k[:, :, g_lo:g_hi, :].clone()
            v_gen = v[:, :, g_lo:g_hi, :].clone()

            k_obj_raw, v_obj_raw = self.obj_kv[self._layer]
            k_obj_raw = k_obj_raw.to(k.device, k.dtype)
            v_obj_raw = v_obj_raw.to(v.device, v.dtype)

            obj_idx = torch.from_numpy(np.where(self.obj_mask_np)[0]).to(k.device)
            if obj_idx.numel() > 0:
                k_obj = k_obj_raw[:, :, obj_idx, :].mean(dim=2, keepdim=True).expand_as(k_gen)
                v_obj = v_obj_raw[:, :, obj_idx, :].mean(dim=2, keepdim=True).expand_as(v_gen)
            else:
                k_obj = k_obj_raw.mean(dim=2, keepdim=True).expand_as(k_gen)
                v_obj = v_obj_raw.mean(dim=2, keepdim=True).expand_as(v_gen)

            tgt   = torch.from_numpy(self.target_zone).to(k.device).view(1, 1, -1, 1)
            s     = self.obj_strength
            k_gen = torch.where(tgt, (1 - s) * k_gen + s * k_obj, k_gen)
            v_gen = torch.where(tgt, (1 - s) * v_gen + s * v_obj, v_gen)

            k = k.clone(); v = v.clone()
            k[:, :, g_lo:g_hi, :] = k_gen
            v[:, :, g_lo:g_hi, :] = v_gen

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[1](attn.to_out[0](out))
            enc_out = attn.to_add_out(enc_out)
            self._layer += 1
            return out, enc_out
        self._layer += 1
        return out


@torch.no_grad()
def run_with_kv_injection(
    pipe,
    canvas:       Image.Image,
    prompt:       str,
    obj_img:      Image.Image,
    obj_mask_np:  np.ndarray,
    target_zone:  np.ndarray,
    seed:         int,
    num_steps:    int,
    guidance:     float,
    height:       int,
    width:        int,
    obj_strength: float = 0.7,
    cutoff_frac:  Tuple[float, float] = (0.0, 0.6),
    bcg_callback=None,
) -> Image.Image:
    """
    FLUX Kontext with K/V injection from obj_img at target_zone tokens.

    Step 1 — 1-step capture pass:
        obj_img (white bg neutralised) as canvas → record K/V at TIER_A layers
        from the ref-token slice.

    Step 2 — Full denoising (num_steps):
        canvas (the collage scene) as reference → at TIER_A layers, blend
        captured K/V into gen tokens that fall inside target_zone.
        Injection active for the first cutoff_frac fraction of steps.

    The captured K/V encode the object's appearance at the feature level.
    The collage already placed the object visually — this reinforces that
    appearance signal in attention space so FLUX doesn't drift it during denoising.
    """
    vae_sf      = getattr(pipe, "vae_scale_factor", 8)
    h_lat       = height // (vae_sf * 2)
    w_lat       = width  // (vae_sf * 2)
    n_gen       = h_lat * w_lat

    orig_procs  = pipe.transformer.attn_processors   # save to restore later

    # ── Stage K-cap: capture K/V from obj_img ────────────────────────────────
    obj_neutral  = _neutralize_white_bg(obj_img)
    capture_proc = _KVCapture(n_gen=n_gen)
    pipe.transformer.set_attn_processor(capture_proc)
    gen0 = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(
        image=obj_neutral,
        prompt="photorealistic object, studio lighting",
        num_inference_steps=1, guidance_scale=1.0,
        height=height, width=width, max_sequence_length=512,
        generator=gen0, output_type="latent",
    )
    obj_kv = dict(capture_proc.kv)
    print(f"    [KV] Captured at {len(obj_kv)}/{len(_TIER_A)} TIER_A layers")

    # ── Stage K-inj: denoising with obj K/V injection ────────────────────────
    inject_proc = _KVInject(
        n_gen=n_gen, obj_kv=obj_kv, obj_mask_np=obj_mask_np,
        target_zone=target_zone, obj_strength=obj_strength,
        n_steps=num_steps, cutoff_frac=cutoff_frac,
    )
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_callback(pipe_ref, step_index, timestep, callback_kwargs):
        inject_proc._step  = step_index + 1
        inject_proc._layer = 0
        if bcg_callback is not None:
            callback_kwargs = bcg_callback(pipe_ref, step_index, timestep, callback_kwargs)
        return callback_kwargs

    tensor_inputs = ["latents"] if bcg_callback is not None else []
    gen1 = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        image=canvas, prompt=prompt,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width, max_sequence_length=512,
        generator=gen1, output_type="pil",
        callback_on_step_end=_step_callback,
        callback_on_step_end_tensor_inputs=tensor_inputs,
    ).images[0]

    pipe.transformer.set_attn_processor(orig_procs)   # restore
    return result


# ── Collage K/V injection ─────────────────────────────────────────────────────
#
# The collage (canvas passed to Kontext) has the object already pasted at the
# target position. During denoising, ref-slice K/V at target_zone positions
# encode the pasted object in scene context — no white bg, no mean-pooling.
# Redirecting gen K/V at those positions from the ref prevents hallucinated
# text/logos while allowing lighting and shadow integration at edges.

class _CollageKVInject:
    """
    At TIER_A layers within cutoff_frac:
        K_gen[zone] = (1-s)*K_gen[zone] + s*K_ref[zone]
        V_gen[zone] = (1-s)*V_gen[zone] + s*V_ref[zone]
    Ref is the collage passed as the Kontext reference image.
    """
    def __init__(self, n_gen: int, target_zone: np.ndarray,
                 obj_strength: float, n_steps: int,
                 cutoff_frac: Tuple[float, float]):
        self.n_gen        = n_gen
        self.target_zone  = target_zone
        self.obj_strength = obj_strength
        self.n_steps      = n_steps
        self.cutoff_frac  = cutoff_frac
        self._layer = 0
        self._step  = 0

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        lo = int(self.cutoff_frac[0] * self.n_steps)
        hi = int(self.cutoff_frac[1] * self.n_steps)
        if lo <= self._step < hi and self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off,              img_off + self.n_gen
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen

            if k.shape[2] >= r_hi:
                k_gen = k[:, :, g_lo:g_hi, :].clone()
                v_gen = v[:, :, g_lo:g_hi, :].clone()
                k_ref = k[:, :, r_lo:r_hi, :]
                v_ref = v[:, :, r_lo:r_hi, :]

                tgt = torch.from_numpy(self.target_zone).to(k.device).view(1, 1, -1, 1)
                s   = self.obj_strength
                k_gen = torch.where(tgt, (1.0 - s) * k_gen + s * k_ref, k_gen)
                v_gen = torch.where(tgt, (1.0 - s) * v_gen + s * v_ref, v_gen)

                k = k.clone(); v = v.clone()
                k[:, :, g_lo:g_hi, :] = k_gen
                v[:, :, g_lo:g_hi, :] = v_gen

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[1](attn.to_out[0](out))
            enc_out = attn.to_add_out(enc_out)
            self._layer += 1
            return out, enc_out
        self._layer += 1
        return out


@torch.no_grad()
def run_with_collage_kv_injection(
    pipe,
    canvas:       Image.Image,
    prompt:       str,
    target_zone:  np.ndarray,
    seed:         int,
    num_steps:    int,
    guidance:     float,
    height:       int,
    width:        int,
    obj_strength: float = 0.5,
    cutoff_frac:  Tuple[float, float] = (0.0, 0.6),
    bcg_callback  = None,
) -> Image.Image:
    """
    Kontext with live collage K/V injection — no pre-capture step needed.
    The reference IS the collage; ref K/V at target positions are used directly.
    """
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_gen  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))

    orig_procs  = pipe.transformer.attn_processors
    inject_proc = _CollageKVInject(
        n_gen=n_gen, target_zone=target_zone,
        obj_strength=obj_strength, n_steps=num_steps,
        cutoff_frac=cutoff_frac,
    )
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_callback(pipe_ref, step_index, timestep, callback_kwargs):
        inject_proc._step  = step_index + 1
        inject_proc._layer = 0
        if bcg_callback is not None:
            callback_kwargs = bcg_callback(pipe_ref, step_index, timestep, callback_kwargs)
        return callback_kwargs

    tensor_inputs = ["latents"] if bcg_callback is not None else []
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        image=canvas, prompt=prompt,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width, max_sequence_length=512,
        generator=gen, output_type="pil",
        callback_on_step_end=_step_callback,
        callback_on_step_end_tensor_inputs=tensor_inputs,
    ).images[0]

    pipe.transformer.set_attn_processor(orig_procs)
    return result


# ── Sobel edge extraction (mirrors AnyDoor's sobel()) ────────────────────────

def _sobel_map(img: np.ndarray, mask: np.ndarray, thresh: int = 30) -> np.ndarray:
    """
    AnyDoor-style high-frequency detail map (datasets/data_utils.py::sobel).
    Sobel-filtered RGB: texture edges where object is, black elsewhere.
    thresh=30 (vs AnyDoor's 50) because LoRA-generated objects have smoother
    gradients than real photos — 50 produces all-black on synthetic output.
    Erosion guard: falls back to un-eroded mask if erosion kills the object.
    """
    H, W = img.shape[:2]
    small   = cv2.resize(img,  (256, 256))
    mask_s  = (cv2.resize(mask.astype(np.uint8), (256, 256)) > 0.5).astype(np.uint8)
    kernel  = np.ones((5, 5), np.uint8)
    eroded  = cv2.erode(mask_s, kernel, iterations=2)
    mask_s  = eroded if eroded.any() else mask_s

    sx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5,
                           cv2.convertScaleAbs(sy), 0.5, 0)
    mag = np.max(mag, axis=-1) * mask_s
    mag[mag < thresh] = 0.0
    mag3 = np.stack([mag, mag, mag], axis=-1)
    edge = (mag3.astype(np.float32) / 255.0 * small.astype(np.float32)).astype(np.uint8)
    return cv2.resize(edge, (W, H))


# ── Core collage builder ──────────────────────────────────────────────────────

def build_collage_scene(
    scene:        Image.Image,
    obj_img:      Image.Image,
    ref_mask:     np.ndarray,
    target_bbox:  Tuple[int, int, int, int],
    collage_mode: str  = "full",
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Build a collage scene with the object hard-pasted at the target location.
    Returns (collage_pil, (x1, y1, x2, y2) paste bbox).

    collage_mode:
      'full'  — paste actual obj pixels, binary mask (hard paste)
      'sobel' — AnyDoor exact: Sobel edge map hard-pasted at full bbox
      'blend' — 60% full + 40% sobel, hard paste
    """
    scene_np = np.array(scene.convert("RGB")).astype(np.float32)
    obj_np   = np.array(obj_img.convert("RGB")).astype(np.float32)
    H, W     = scene_np.shape[:2]

    # Crop to tight bbox around the object in obj_img
    ys, xs   = np.where(ref_mask > 0)
    if len(ys) == 0:
        print("    [COLLAGE] Warning: empty ref_mask, using center crop fallback")
        oh, ow  = obj_np.shape[:2]
        oy1, oy2, ox1, ox2 = oh//4, 3*oh//4, ow//4, 3*ow//4
    else:
        oy1, oy2 = int(ys.min()), int(ys.max()) + 1
        ox1, ox2 = int(xs.min()), int(xs.max()) + 1

    obj_crop   = obj_np[oy1:oy2, ox1:ox2]           # (oh, ow, 3)
    mask_crop  = ref_mask[oy1:oy2, ox1:ox2]         # (oh, ow) uint8

    # Target placement bbox — directly from VLM grounding
    x1, y1, x2, y2 = target_bbox
    zone_h, zone_w  = y2 - y1, x2 - x1
    print(f"    [COLLAGE] Target zone: y=[{y1},{y2}] x=[{x1},{x2}]  "
          f"{zone_w}×{zone_h} px")

    # AnyDoor-exact sobel: resize obj to fill full target zone, hard paste,
    # no alpha blending — Kontext (our "ControlNet") handles the blending.
    if collage_mode == "sobel":
        obj_z      = cv2.resize(obj_crop.astype(np.uint8),  (zone_w, zone_h))
        mask_z     = cv2.resize(mask_crop.astype(np.uint8), (zone_w, zone_h))
        sobel_zone = _sobel_map(obj_z, mask_z)
        collage_np = scene_np.copy().astype(np.uint8)
        collage_np[y1:y2, x1:x2] = sobel_zone
        print(f"    [COLLAGE] AnyDoor hard paste {zone_w}×{zone_h} Sobel at "
              f"scene[{y1}:{y2},{x1}:{x2}]")
        return Image.fromarray(collage_np), (y1, y2, x1, x2)

    # Scale obj to fill ~80% of the target zone (preserve aspect ratio)
    oh, ow  = obj_crop.shape[:2]
    scale   = min((zone_h * 0.80) / oh, (zone_w * 0.80) / ow)
    new_h   = max(4, int(oh * scale))
    new_w   = max(4, int(ow * scale))

    obj_rs  = cv2.resize(obj_crop.astype(np.uint8),  (new_w, new_h)).astype(np.float32)
    mask_rs = cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h))
    mask_rs = (mask_rs > 0.5).astype(np.float32)

    # Choose what to paste
    if collage_mode == "blend":
        sobel_layer = _sobel_map(obj_rs.astype(np.uint8),
                                  mask_rs).astype(np.float32)
        paste_layer = 0.6 * obj_rs + 0.4 * sobel_layer
    else:  # 'full'
        paste_layer = obj_rs

    # Hard binary mask: paste at full opacity, no feathering, no alpha blend.
    # Kontext handles all blending in Stage K.
    mask_3  = np.stack([mask_rs, mask_rs, mask_rs], axis=-1)

    # Center the resized obj in the target zone
    dy = (zone_h - new_h) // 2
    dx = (zone_w - new_w) // 2
    py1, px1 = y1 + dy, x1 + dx
    py2, px2 = py1 + new_h, px1 + new_w

    # Clamp to scene bounds
    py1 = max(0, py1);  py2 = min(H, py2)
    px1 = max(0, px1);  px2 = min(W, px2)
    ch  = py2 - py1;    cw  = px2 - px1
    if ch <= 0 or cw <= 0:
        print("    [COLLAGE] Warning: paste region clipped to zero, using scene as-is")
        return scene, (y1, y2, x1, x2)

    paste_layer = paste_layer[:ch, :cw]
    mask_3      = mask_3[:ch, :cw]

    # Alpha-composite onto scene copy
    collage_np  = scene_np.copy()
    scene_patch = collage_np[py1:py2, px1:px2]
    collage_np[py1:py2, px1:px2] = (
        paste_layer * mask_3 + scene_patch * (1.0 - mask_3)
    )
    collage_np = np.clip(collage_np, 0, 255).astype(np.uint8)

    print(f"    [COLLAGE] Pasted {new_w}×{new_h} obj at scene[{py1}:{py2}, {px1}:{px2}]  "
          f"mode={collage_mode}")

    return Image.fromarray(collage_np), (y1, y2, x1, x2)


# ── Removal collage builder ───────────────────────────────────────────────────

def build_removal_collage(
    scene:     Image.Image,
    mask_path: str,
    height:    int,
    width:     int,
    dilate_px: int = 8,
) -> Image.Image:
    """
    Erase the object region with OpenCV Telea inpaint so Kontext sees a plausible
    background where the object was.  BCG keeps everything outside the mask
    pixel-identical to the original scene.

    dilate_px: expand the inpaint mask so object edges are also erased.
    """
    scene_np = np.array(scene.convert("RGB"))
    mask_img = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
    inpaint_mask = (np.array(mask_img) > 127).astype(np.uint8)
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        inpaint_mask = cv2.dilate(inpaint_mask, np.ones((k, k), np.uint8), iterations=1)
    inpainted = cv2.inpaint(scene_np, inpaint_mask, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
    print(f"    [REMOVAL] Telea inpaint: {int(inpaint_mask.sum())} px erased")
    return Image.fromarray(inpainted)


# ── Object footprint mask (used for BCG tight boundary) ──────────────────────


def _collage_obj_mask(collage: Image.Image, scene: Image.Image,
                       threshold: int = 10) -> np.ndarray:
    """
    Binary uint8 mask (H, W): 1 where collage differs from scene (= pasted obj).
    More reliable than forwarding ref_mask because the paste location shifts
    (80% scale + centering) relative to the original obj_img space.
    """
    diff = np.abs(
        np.array(collage.convert("RGB"), dtype=np.int32) -
        np.array(scene.convert("RGB"),   dtype=np.int32)
    ).max(axis=-1)
    mask = (diff > threshold).astype(np.uint8)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask




# ── BLD latent-space BCG (Blended Latent Diffusion, arxiv:2206.02779) ─────────

def _make_bcg_latent_callback(
    scene:        Image.Image,
    mask_path:    str,
    pipe,
    height:       int,
    width:        int,
    device:       str,
    dtype:        torch.dtype = torch.bfloat16,
    dilation_lat: int = 3,
):
    """
    At every denoising step with noise level sigma_t:
        z_bg_t = (1 - sigma_t) * z0_scene + sigma_t * noise
        latents = latents * mask + z_bg_t * (1 - mask)

    Early steps (sigma≈1): background ≈ noise  → matches noisy interior → no discontinuity
    Late steps  (sigma≈0): background → z0_scene → pixel-identical to original scene

    Mask is the user-drawn placement mask at VAE latent resolution.
    Everything outside the mask converges to the exact original scene.
    """
    vae_h = height // 8
    vae_w = width  // 8

    # Encode the background scene once (z0)
    scene_np = np.array(scene.convert("RGB"), dtype=np.float32) / 255.0
    scene_t  = torch.from_numpy(scene_np).permute(2, 0, 1).unsqueeze(0)
    scene_t  = (scene_t * 2.0 - 1.0).to(dtype)
    with torch.no_grad():
        enc_dev = next(pipe.vae.parameters()).device
        z0 = pipe.vae.encode(scene_t.to(enc_dev)).latent_dist.sample()
        z0 = (z0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z0 = z0.to(device, dtype)

    # Fixed noise — consistent across all steps so background converges coherently
    noise = torch.randn(z0.shape, device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(42))

    # Placement mask at VAE latent resolution
    placement_np = np.array(
        Image.open(mask_path).convert("L").resize((vae_w, vae_h), Image.Resampling.NEAREST)
    )
    mask_lat = (placement_np > 127).astype(np.uint8)
    if dilation_lat > 0:
        k = dilation_lat * 2 + 1
        mask_lat = cv2.dilate(mask_lat, np.ones((k, k), np.uint8), iterations=2)
    mask_soft   = cv2.GaussianBlur(mask_lat.astype(np.float32), (7, 7), 2.0)
    mask_tensor = torch.from_numpy(mask_soft).to(device, dtype)[None, None]  # (1,1,H,W)

    print(f"    [BLD] Latent mask {vae_h}×{vae_w}, edit zone={mask_soft.mean()*100:.1f}%")

    def _callback(_pipeline, _step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if latents.shape[-2:] != (vae_h, vae_w):
            return callback_kwargs  # unexpected shape — skip safely

        # FLUX flow-matching: timestep = sigma × 1000, sigma in [0, 1]
        sigma = max(0.0, min(1.0, float(timestep) / 1000.0))

        z_bg = ((1.0 - sigma) * z0 + sigma * noise).to(latents.dtype).to(latents.device)
        m    = mask_tensor.expand_as(latents).to(latents.dtype).to(latents.device)

        callback_kwargs["latents"] = latents * m + z_bg * (1.0 - m)
        return callback_kwargs

    return _callback


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_collage_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    lora_id:        str,
    seed:           int,
    num_steps:      int,
    lora_guidance:  float,
    scene_guidance: float,
    collage_mode:   str,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
    use_kv:         bool  = False,
    obj_strength:   float = 0.7,
    cutoff_frac:    Tuple[float, float] = (0.0, 0.6),
) -> List[Image.Image]:
    """
    Incremental object insertion via FLUX Kontext with Kontext-based pose adjustment.

    Per object:
      Stage A   : Sketch → LoRA FLUX → obj_img
      Stage BBOX: Placement mask → bbox  (mask_{name}.png required)
      Stage POSE: Kontext(scene_crop+obj, pose_prompt) → obj_img
                  diff extraction → obj on white bg with natural pose
      Stage MASK: Threshold obj_img → ref_mask
      Stage COL : paste obj_img into scene at bbox → collage_scene
      Stage HAR : Harmonize pasted object with background (PCT-Net or Reinhard)
      Stage K   : FLUX Kontext(reference=collage_scene, prompt) → result
    """
    results = [base]
    scene   = base

    for i, edit in enumerate(edits):
        name   = edit["name"]
        desc   = edit["description"]
        action = edit.get("action", "insert")

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}  [{action}]")
        print(f"{'─'*60}")

        # Stage BBOX: placement mask required for both insert and remove
        mask_path = os.path.join(sketch_dir, f"mask_{name}.png")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Placement mask not found: {mask_path}\n"
                f"Draw a white region on a black image to mark where '{name}' should go."
            )
        print(f"  [BBOX] Placement mask: mask_{name}.png")
        bx1, by1, bx2, by2 = _bbox_from_placement_mask(mask_path, width, height)
        print(f"      bbox: x=[{bx1},{bx2}] y=[{by1},{by2}]")
        with open(os.path.join(out_dir, f"bbox_{name}_{action}.txt"), "w") as f:
            f.write(f"bbox: x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

        if action == "insert":
            # Stage A: sketch → object image
            sketch_path = os.path.join(sketch_dir, f"{name}.png")
            if not os.path.isfile(sketch_path):
                sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
            if not os.path.isfile(sketch_path):
                raise FileNotFoundError(
                    f"Sketch not found in {sketch_dir!r}. "
                    f"Expected '{name}.png' or 'sketch_{name}.png'."
                )
            print(f"  [A] Generating '{desc}' from sketch ...")
            obj_img = generate_from_sketch(
                pipe=pipe, sketch_path=sketch_path, description=desc,
                seed=seed, num_steps=num_steps, guidance=lora_guidance,
                height=height, width=width, lora_id=lora_id, device=device,
            )
            obj_img.save(os.path.join(out_dir, f"obj_gen_{name}.png"))

            # Stage MASK: extract object silhouette
            ref_mask = _compute_obj_mask(obj_img)
            n_px = ref_mask.sum()
            print(f"  [MASK] Object pixels: {n_px} ({100*n_px/(height*width):.1f}%)")
            if n_px < 50:
                print("         Fallback: using center 50% crop as object region")
                ref_mask = np.zeros((height, width), dtype=np.uint8)
                ref_mask[height//4:3*height//4, width//4:3*width//4] = 1

            # Stage COL: paste object into scene
            print(f"  [COL] Building collage scene (mode={collage_mode}) ...")
            collage_scene, _ = build_collage_scene(
                scene        = scene,
                obj_img      = obj_img,
                ref_mask     = ref_mask,
                target_bbox  = (bx1, by1, bx2, by2),
                collage_mode = collage_mode,
            )
            collage_scene.save(os.path.join(out_dir, f"collage_{name}.png"))
            print(f"      Saved collage: collage_{name}.png")

            obj_mask_har = _collage_obj_mask(collage_scene, scene)
            n_har = int(obj_mask_har.sum())
            print(f"  [MASK] Object footprint: {n_har} px ({100*n_har/obj_mask_har.size:.1f}%)")

            blend_p = f"A photorealistic room with a {desc} placed naturally."

        else:  # remove
            print(f"  [A] Removal — skipping object generation")

            # Stage COL: inpaint the object away as the Kontext reference
            print(f"  [COL] Building removal collage (Telea inpaint) ...")
            collage_scene = build_removal_collage(
                scene=scene, mask_path=mask_path, height=height, width=width
            )
            collage_scene.save(os.path.join(out_dir, f"collage_remove_{name}.png"))
            print(f"      Saved: collage_remove_{name}.png")

            # For BCG: use the placement mask as the edit boundary
            _pmask_arr = np.array(
                Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
            )
            obj_mask_har = (_pmask_arr > 127).astype(np.uint8)
            n_har = int(obj_mask_har.sum())
            print(f"  [MASK] Removal zone: {n_har} px ({100*n_har/obj_mask_har.size:.1f}%)")

            blend_p = (
                f"{BASE_PROMPT} "
                f"The {desc} has been removed. "
                f"Seamless wooden floor and white walls fill the area. "
                f"No object, clean empty room."
            )

        # Stage K: FLUX Kontext integration pass with latent-space BCG
        step_tag = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"
        print(f"  [K] Kontext integration pass  action={action}  kv={use_kv} ...")
        print(f"      Prompt: {blend_p[:120]}...")
        with open(os.path.join(out_dir, f"prompt_{step_tag}.txt"), "w") as f:
            f.write(blend_p)

        # BLD latent callback: inside mask Kontext edits freely;
        # outside it converges to original scene (pixel-identical at sigma=0).
        pipe_dtype = next(pipe.transformer.parameters()).dtype
        bcg_cb = _make_bcg_latent_callback(
            scene=scene, mask_path=mask_path, pipe=pipe,
            height=height, width=width, device=device, dtype=pipe_dtype,
        )

        if use_kv:
            target_zone = _placement_mask_to_token_zone(mask_path, height, width, pipe)
            print(f"      Collage K/V zone: {target_zone.sum()} tokens ({100*target_zone.mean():.1f}%)")
            next_scene = run_with_collage_kv_injection(
                pipe=pipe, canvas=collage_scene, prompt=blend_p,
                target_zone=target_zone,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width,
                obj_strength=obj_strength, cutoff_frac=cutoff_frac,
                bcg_callback=bcg_cb,
            )
        else:
            next_scene = run_standard(
                pipe=pipe, canvas=collage_scene, prompt=blend_p,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width,
                bcg_callback=bcg_cb,
            )

        result_path = os.path.join(out_dir, f"result_{step_tag}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        # Stage BCG: restore background outside object zone
        print(f"  [BCG] Background restore ...")
        bcg_mask = cv2.dilate(obj_mask_har, np.ones((21, 21), np.uint8), iterations=3)
        bcg_mask = cv2.GaussianBlur(bcg_mask.astype(np.float32), (41, 41), 15)

        result_np = np.array(next_scene.convert("RGB"), dtype=np.float32)
        scene_np  = np.array(scene.convert("RGB"),      dtype=np.float32)
        m3        = bcg_mask[:, :, None]
        bcg_np    = np.clip(result_np * m3 + scene_np * (1.0 - m3), 0, 255).astype(np.uint8)
        next_scene = Image.fromarray(bcg_np)
        bcg_path   = os.path.join(out_dir, f"result_bcg_{step_tag}.png")
        next_scene.save(bcg_path)
        print(f"      Saved: {bcg_path}")

        scene = next_scene
        results.append(scene)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AnyDoor collage method inside FLUX Kontext."
    )
    p.add_argument("--sketch_dir",    required=True)
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--cache_dir",     default="./models")
    p.add_argument("--out_dir",       default="results/phase1_collage")
    p.add_argument("--config",        default=None,
                   help="JSON list of {name, description}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",       default=LORA_ID)
    p.add_argument("--lora_guidance", type=float, default=4.0)
    p.add_argument("--guidance",      type=float, default=2.5,
                   help="Kontext guidance scale for the blending pass. Default 2.5.")
    p.add_argument("--collage_mode",  default="full",
                   choices=["full", "sobel", "blend"],
                   help="What to paste in the collage. "
                        "'full'=actual pixels (default), "
                        "'sobel'=edge map only (AnyDoor-exact), "
                        "'blend'=60%%full+40%%sobel.")
    p.add_argument("--kv_injection",  action="store_true",
                   help="Enable K/V injection from obj_img at target zone on top of collage.")
    p.add_argument("--obj_strength",  type=float, default=0.7,
                   help="K/V injection strength at target zone (0-1). Default 0.7.")
    p.add_argument("--cutoff_frac",   default="0.0,0.6",
                   help="Injection active window as 'start,end' step fractions. Default '0.0,0.6'.")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_steps",     type=int,   default=28)
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)

    print(f"\n{_SEP}")
    print(f"  phase1_collage_kontext  —  AnyDoor Collage + FLUX Kontext")
    print(f"{_SEP}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  Collage mode : {args.collage_mode}")
    print(f"  Guidance     : {args.guidance}")
    print(f"  KV injection : {args.kv_injection}"
          + (f"  strength={args.obj_strength}  cutoff={args.cutoff_frac}"
             if args.kv_injection else ""))
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    print("Loading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token  = args.hf_token,
        device    = args.device,
        cache_dir = args.cache_dir,
    )

    # Base scene
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    print(f"  Saved: base_scene.png")

    _cf = [float(x) for x in args.cutoff_frac.split(",")]
    cutoff_frac: Tuple[float, float] = (_cf[0], _cf[1])

    results = run_collage_chain(
        pipe=pipe, base=base, edits=edits,
        sketch_dir=args.sketch_dir, lora_id=args.lora_id,
        seed=args.seed, num_steps=args.num_steps,
        lora_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        collage_mode=args.collage_mode,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
        use_kv=args.kv_injection,
        obj_strength=args.obj_strength,
        cutoff_frac=cutoff_frac,
    )

    all_imgs = results
    all_lbls = ["base"] + [e["name"] for e in edits]
    save_grid(all_imgs, all_lbls,
              os.path.join(args.out_dir, "chain_grid.png"),
              ncols=len(all_imgs))
    print(f"\n{_SEP}")
    print(f"  Chain complete. Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
