# -*- coding: utf-8 -*-
"""
phase1_collage_qwen.py — AnyDoor's Collage Method with Qwen-Image-Edit-2509

Port of phase1_collage_kontext.py replacing FLUX.1-Kontext-dev with
Qwen/Qwen-Image-Edit-2509  (QwenImageEditPlusPipeline, diffusers).

Key differences from the FLUX version
---------------------------------------
  Model       : Qwen/Qwen-Image-Edit-2509 (20.4B MMDiT, 60 dual-stream blocks)
  VAE         : Wan2.1-based, 8× + 2×2 packing → effective stride 16 px/token
  Conditioning: Qwen is image-to-image — the collage IS the starting image,
                not a separate reference stream (no [gen|ref] token split).
  Stage A     : No FLUX LoRA — Qwen renders the sketch directly via text prompt.
  K/V inject  : Deferred. Qwen has no separate ref-token slice to pull K/V from.
                The collage naturally conditions the edit.
  BCG callback: VAE shift/scale read dynamically from pipe.vae.config.
  All CV/PIL  : COL, BBOX, MASK, pixel BCG — identical to the Kontext version.

Pipeline
---------
  Stage A  : sketch + description → Qwen(image=sketch, prompt) → obj_img
  Stage BBOX: placement mask → bbox  (mask_{name}.png required)
  Stage MASK: threshold obj_img → ref_mask
  Stage COL : paste obj_img into scene → collage_scene
  Stage K  : Qwen(image=collage_scene, prompt=blend_prompt) + BLD latent BCG
  Stage BCG : pixel-space tight-mask composite to restore background

Usage
------
  python NewWork/KontextEval/phase1_collage_qwen.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token   $HF_TOKEN \\
      --cache_dir  ./models \\
      --out_dir    results/phase1_qwen

  Add --cpu_offload if VRAM < 45 GB (FP16) or use a quantised checkpoint.

Key flags
----------
  --guidance        float  CFG scale for Stage K.  Default 3.5
  --lora_guidance   float  CFG scale for Stage A.  Default 3.5
  --collage_mode    full | sobel | blend
  --num_steps       int    Inference steps.         Default 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

# dispatch_attention_fn is available in diffusers >= 0.33 (same version that ships Qwen support)
try:
    from diffusers.models.attention_dispatch import dispatch_attention_fn as _dispatch_attn
    _HAS_DISPATCH = True
except ImportError:
    _HAS_DISPATCH = False


# ── Pipeline loading ──────────────────────────────────────────────────────────

def load_qwen_pipeline(
    model_path: str = "Qwen/Qwen-Image-Edit-2509",
    hf_token: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = "./models",
    cpu_offload: bool = False,
) -> QwenImageEditPlusPipeline:
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path, torch_dtype=dtype, token=hf_token, cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── Qwen inference ────────────────────────────────────────────────────────────

def run_qwen(
    pipe, canvas: Image.Image, prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    bcg_callback=None,
    negative_prompt: str = "blurry, distorted, low quality, watermark, text, artifacts",
) -> Image.Image:
    # true_cfg_scale activates CFG only when negative_prompt is also provided.
    # guidance_scale is present in the signature but silently ignored by this checkpoint.
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    cb_kwargs: dict = {}
    if bcg_callback is not None:
        cb_kwargs["callback_on_step_end"] = bcg_callback
        cb_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canvas,
        num_inference_steps=num_steps, true_cfg_scale=guidance,
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

def generate_from_sketch(
    pipe, sketch_path: str, description: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int, device: str,
) -> Image.Image:
    """
    Qwen does not accept FLUX LoRAs.  We pass the sketch as the input image
    and instruct Qwen to render it photorealistically via text.
    Qwen-Image-Edit LoRAs (trained on Qwen-Image) can be loaded separately
    if available; that is left as a future extension.
    """
    sketch_pil = Image.open(sketch_path).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )
    prompt = (
        f"Render this sketch as a photorealistic {description} "
        "on a plain white background. No shadows. Studio lighting. High quality."
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        image=sketch_pil, prompt=prompt,
        negative_prompt="blurry, distorted, low quality, watermark, text, artifacts",
        num_inference_steps=num_steps, true_cfg_scale=guidance,
        height=height, width=width, generator=generator,
    ).images[0]


# ── Constants ─────────────────────────────────────────────────────────────────

EDITS: List[dict] = [
    # ── Insertion ─────────────────────────────────────────────────────────────
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "insert"},
    {"name": "vase",    "description": "black ceramic vase with flowers",  "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "insert"},
    # ── Removal ───────────────────────────────────────────────────────────────
    {"name": "ball",    "description": "yellow rubber ball",               "action": "remove"},
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "remove"},
]
BASE_PROMPT = (
    "A empty room with a wooden floor, white walls, "
    "and a window letting in natural light."
)
_SEP = "═" * 60


# ── Object mask extraction ────────────────────────────────────────────────────

def _compute_obj_mask(obj_img: Image.Image,
                      grey: tuple = (128, 128, 128),
                      tolerance: int = 20) -> np.ndarray:
    arr = np.array(obj_img.convert("RGB"), dtype=np.int32)
    is_grey  = ((np.abs(arr[:, :, 0] - grey[0]) <= tolerance) &
                (np.abs(arr[:, :, 1] - grey[1]) <= tolerance) &
                (np.abs(arr[:, :, 2] - grey[2]) <= tolerance))
    is_white = (arr[:, :, 0] >= 230) & (arr[:, :, 1] >= 230) & (arr[:, :, 2] >= 230)
    return (~(is_grey | is_white)).astype(np.uint8)


# ── Placement mask → bbox ─────────────────────────────────────────────────────

def _bbox_from_placement_mask(
    mask_path: str, width: int, height: int,
) -> Tuple[int, int, int, int]:
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


# ── Placement mask → flat token zone ─────────────────────────────────────────

def _placement_mask_to_token_zone(mask_path: str, height: int, width: int,
                                   pipe) -> np.ndarray:
    """Convert placement mask to flat bool array (n_img,) at token resolution."""
    vae_sf       = getattr(pipe, "vae_scale_factor", 8)
    h_tok, w_tok = height // (vae_sf * 2), width // (vae_sf * 2)
    mask_down    = Image.open(mask_path).convert("L").resize((w_tok, h_tok), Image.Resampling.NEAREST)
    zone = (np.array(mask_down) > 127).reshape(-1)
    print(f"    [KV-zone] {zone.sum()} / {h_tok * w_tok} tokens ({100*zone.mean():.1f}%)")
    return zone


# ── Sobel edge extraction ─────────────────────────────────────────────────────

def _sobel_map(img: np.ndarray, mask: np.ndarray, thresh: int = 30) -> np.ndarray:
    H, W   = img.shape[:2]
    small  = cv2.resize(img,  (256, 256))
    mask_s = (cv2.resize(mask.astype(np.uint8), (256, 256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask_s, kernel, iterations=2)
    mask_s = eroded if eroded.any() else mask_s
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
    collage_mode: str = "full",
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    scene_np = np.array(scene.convert("RGB")).astype(np.float32)
    obj_np   = np.array(obj_img.convert("RGB")).astype(np.float32)
    H, W     = scene_np.shape[:2]

    ys, xs = np.where(ref_mask > 0)
    if len(ys) == 0:
        print("    [COLLAGE] Warning: empty ref_mask, using center crop fallback")
        oh, ow  = obj_np.shape[:2]
        oy1, oy2, ox1, ox2 = oh // 4, 3 * oh // 4, ow // 4, 3 * ow // 4
    else:
        oy1, oy2 = int(ys.min()), int(ys.max()) + 1
        ox1, ox2 = int(xs.min()), int(xs.max()) + 1

    obj_crop  = obj_np[oy1:oy2, ox1:ox2]
    mask_crop = ref_mask[oy1:oy2, ox1:ox2]

    x1, y1, x2, y2 = target_bbox
    zone_h, zone_w  = y2 - y1, x2 - x1
    print(f"    [COLLAGE] Target zone: y=[{y1},{y2}] x=[{x1},{x2}]  {zone_w}×{zone_h} px")

    if collage_mode == "sobel":
        obj_z      = cv2.resize(obj_crop.astype(np.uint8),  (zone_w, zone_h))
        mask_z     = cv2.resize(mask_crop.astype(np.uint8), (zone_w, zone_h))
        sobel_zone = _sobel_map(obj_z, mask_z)
        collage_np = scene_np.copy().astype(np.uint8)
        collage_np[y1:y2, x1:x2] = sobel_zone
        print(f"    [COLLAGE] AnyDoor hard paste {zone_w}×{zone_h} Sobel at "
              f"scene[{y1}:{y2},{x1}:{x2}]")
        return Image.fromarray(collage_np), (y1, y2, x1, x2)

    oh, ow = obj_crop.shape[:2]
    scale  = min((zone_h * 0.80) / oh, (zone_w * 0.80) / ow)
    new_h  = max(4, int(oh * scale))
    new_w  = max(4, int(ow * scale))

    obj_rs  = cv2.resize(obj_crop.astype(np.uint8),  (new_w, new_h)).astype(np.float32)
    mask_rs = cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h))
    mask_rs = (mask_rs > 0.5).astype(np.float32)

    if collage_mode == "blend":
        sobel_layer = _sobel_map(obj_rs.astype(np.uint8), mask_rs).astype(np.float32)
        paste_layer = 0.6 * obj_rs + 0.4 * sobel_layer
    else:
        paste_layer = obj_rs

    mask_3 = np.stack([mask_rs, mask_rs, mask_rs], axis=-1)
    dy = (zone_h - new_h) // 2
    dx = (zone_w - new_w) // 2
    py1, px1 = y1 + dy, x1 + dx
    py2, px2 = py1 + new_h, px1 + new_w

    py1 = max(0, py1); py2 = min(H, py2)
    px1 = max(0, px1); px2 = min(W, px2)
    ch = py2 - py1; cw = px2 - px1
    if ch <= 0 or cw <= 0:
        print("    [COLLAGE] Warning: paste region clipped to zero, using scene as-is")
        return scene, (y1, y2, x1, x2)

    paste_layer = paste_layer[:ch, :cw]
    mask_3      = mask_3[:ch, :cw]
    collage_np  = scene_np.copy()
    scene_patch = collage_np[py1:py2, px1:px2]
    collage_np[py1:py2, px1:px2] = paste_layer * mask_3 + scene_patch * (1.0 - mask_3)
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
    Erase the object region with OpenCV Telea inpaint so Qwen sees a plausible
    background where the object was.  BCG keeps everything outside the mask
    pixel-identical to the original scene.
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
    diff = np.abs(
        np.array(collage.convert("RGB"), dtype=np.int32) -
        np.array(scene.convert("RGB"),   dtype=np.int32)
    ).max(axis=-1)
    mask = (diff > threshold).astype(np.uint8)
    k    = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask


# ── Latent packing ────────────────────────────────────────────────────────────

def _pack_latents(z: torch.Tensor) -> torch.Tensor:
    """
    Pack spatial latents (B, C, H, W) → token sequence (B, H//2 * W//2, C*4).

    Qwen's denoising loop works in packed token space: the callback receives
    latents shaped (B, n_tok, C_packed) not the VAE's spatial (B, C, H, W).
    Each token covers a 2×2 block at VAE spatial resolution (= 16 image pixels).

    Packing order (row-major, matches diffusers WanPipeline._pack_latents):
      token index = i * (W//2) + j  for spatial block (i, j)
    """
    B, C, H, W = z.shape
    h_tok, w_tok = H // 2, W // 2
    z = z.reshape(B, C, h_tok, 2, w_tok, 2)
    z = z.permute(0, 2, 4, 1, 3, 5)          # (B, h_tok, w_tok, C, 2, 2)
    return z.reshape(B, h_tok * w_tok, C * 4) # (B, n_tok, C*4)


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
    BLD latent BCG in packed token space.

    Qwen's denoising callback receives latents as (B, n_tok, C_packed) where
    n_tok = (H//16) * (W//16) = 4096 for 1024×1024.  The old spatial shape-check
    (lat_h, lat_w) always failed → zero blending happened.  This version:

    1. Encodes the background scene and normalises via Qwen's per-channel
       latents_mean / latents_std vectors (not the FLUX scalar shift_factor).
    2. Packs the encoded latent to (B, n_tok, 64) to match the callback tensor.
    3. Builds the BLD mask at token resolution (64×64) instead of VAE spatial
       resolution (128×128) — each token covers exactly one 16-px image block.
    4. Injects z_bg = (1−σ)·z0 + σ·ε at background tokens every denoising step.
    """
    scene_np = np.array(scene.convert("RGB"), dtype=np.float32) / 255.0
    scene_t  = torch.from_numpy(scene_np).permute(2, 0, 1).unsqueeze(0)
    scene_t  = (scene_t * 2.0 - 1.0).to(dtype).unsqueeze(2)  # Wan2.1: add T=1

    with torch.no_grad():
        enc_dev = next(pipe.vae.parameters()).device
        z0_raw  = pipe.vae.encode(scene_t.to(enc_dev)).latent_dist.mean
        z0_raw  = z0_raw.squeeze(2)  # (1, 16, 128, 128) — remove T=1

    latents_mean = torch.tensor(
        pipe.vae.config.latents_mean, dtype=dtype, device=enc_dev
    ).reshape(1, -1, 1, 1)
    latents_std = torch.tensor(
        pipe.vae.config.latents_std, dtype=dtype, device=enc_dev
    ).reshape(1, -1, 1, 1)
    z0 = ((z0_raw - latents_mean) / latents_std).to(device, dtype)

    # Pack (1, 16, 128, 128) → (1, 4096, 64) — matches callback latent shape
    z0_tok = _pack_latents(z0)
    n_tok  = z0_tok.shape[1]                  # 4096
    h_tok  = z0.shape[-2] // 2               # 64  (VAE 128 / patch 2)
    w_tok  = z0.shape[-1] // 2               # 64

    noise = torch.randn(z0_tok.shape, device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(42))

    # Mask at token resolution (h_tok × w_tok = 64×64), NOT VAE spatial (128×128)
    placement_np = np.array(
        Image.open(mask_path).convert("L").resize((w_tok, h_tok), Image.Resampling.NEAREST)
    )
    mask_lat = (placement_np > 127).astype(np.uint8)
    if dilation_lat > 0:
        k = dilation_lat * 2 + 1
        mask_lat = cv2.dilate(mask_lat, np.ones((k, k), np.uint8), iterations=2)
    mask_soft   = cv2.GaussianBlur(mask_lat.astype(np.float32), (7, 7), 2.0)
    # (1, n_tok, 1) — broadcasts over the C_packed feature dim
    mask_tensor = torch.from_numpy(
        mask_soft.reshape(1, n_tok, 1)
    ).to(device, dtype)

    print(f"    [BLD] token {h_tok}×{w_tok}={n_tok}  "
          f"latents_mean[0]={pipe.vae.config.latents_mean[0]:.4f}  "
          f"edit zone={mask_soft.mean()*100:.1f}%")

    def _callback(_pipeline, _step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if latents.shape != z0_tok.shape:
            return callback_kwargs

        sigma = max(0.0, min(1.0, float(timestep) / 1000.0))
        # Only lock background in late steps (σ < 0.4).
        # Early steps (σ ≥ 0.4) run freely so Qwen can form natural shadows
        # and contact lighting before we start preserving the background.
        if sigma >= 0.4:
            return callback_kwargs

        z_bg  = ((1.0 - sigma) * z0_tok + sigma * noise).to(latents.dtype).to(latents.device)
        m     = mask_tensor.to(latents.dtype).to(latents.device)
        callback_kwargs["latents"] = latents * m + z_bg * (1.0 - m)
        return callback_kwargs

    return _callback


def _make_obj_latent_composite_callback(
    collage:         Image.Image,
    obj_mask_har:    np.ndarray,        # (H, W) uint8 {0,1} — silhouette in scene coords
    pipe,
    height:          int,
    width:           int,
    device:          str,
    dtype:           torch.dtype = torch.bfloat16,
    anchor_strength: float = 0.5,
    inject_frac:     Tuple[float, float] = (0.35, 0.55),
):
    """
    Per-step BLD latent compositing for object identity preservation.

    Works in packed token space (B, n_tok, 64) to match Qwen's callback format.
    """
    collage_np = np.array(collage.convert("RGB"), dtype=np.float32) / 255.0
    collage_t  = torch.from_numpy(collage_np).permute(2, 0, 1).unsqueeze(0)
    collage_t  = (collage_t * 2.0 - 1.0).to(dtype).unsqueeze(2)  # Wan2.1: T=1

    with torch.no_grad():
        enc_dev = next(pipe.vae.parameters()).device
        z_raw   = pipe.vae.encode(collage_t.to(enc_dev)).latent_dist.mean
        z_raw   = z_raw.squeeze(2)  # (1, 16, 128, 128)

    latents_mean = torch.tensor(
        pipe.vae.config.latents_mean, dtype=dtype, device=enc_dev
    ).reshape(1, -1, 1, 1)
    latents_std = torch.tensor(
        pipe.vae.config.latents_std, dtype=dtype, device=enc_dev
    ).reshape(1, -1, 1, 1)
    z_clean = ((z_raw - latents_mean) / latents_std).to(device, dtype)

    # Pack to token space
    z_clean_tok = _pack_latents(z_clean)
    n_tok = z_clean_tok.shape[1]     # 4096
    h_tok = z_clean.shape[-2] // 2  # 64
    w_tok = z_clean.shape[-1] // 2  # 64

    # Bilinear downsample object silhouette to token resolution
    mask_f   = torch.from_numpy(obj_mask_har.astype(np.float32))[None, None]
    mask_tok = F.interpolate(
        mask_f, size=(h_tok, w_tok), mode="bilinear", align_corners=False, antialias=True,
    ).squeeze().numpy()
    mask_tok  = (mask_tok >= 0.5).astype(np.float32)
    mask_soft = cv2.GaussianBlur(mask_tok, (7, 7), 2.0)
    # (1, n_tok, 1) — broadcasts over feature dim
    mask_t    = torch.from_numpy(mask_soft.reshape(1, n_tok, 1)).to(device, dtype)

    lo_s, hi_s = inject_frac
    s = anchor_strength

    print(f"    [OBJ-COMP] token {h_tok}×{w_tok}={n_tok}  "
          f"strength={s}  σ-window=[{lo_s:.2f},{hi_s:.2f}]  "
          f"zone={mask_soft.mean()*100:.1f}%")

    def _callback(_pipeline, _step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if latents.shape != z_clean_tok.shape:
            return callback_kwargs
        sigma = max(0.0, min(1.0, float(timestep) / 1000.0))
        if not (lo_s <= sigma <= hi_s):
            return callback_kwargs
        eps   = torch.randn_like(z_clean_tok)
        z_ref = ((1.0 - sigma) * z_clean_tok + sigma * eps).to(latents.dtype).to(latents.device)
        m     = mask_t.to(latents.dtype).to(latents.device)
        callback_kwargs["latents"] = latents * (1.0 - s * m) + z_ref * (s * m)
        return callback_kwargs

    return _callback


def _compose_callbacks(cb1, cb2):
    """Chain two step-end callbacks; either may be None."""
    if cb1 is None: return cb2
    if cb2 is None: return cb1
    def _cb(pipe, step_idx, ts, kwargs):
        kwargs = cb1(pipe, step_idx, ts, kwargs)
        return cb2(pipe, step_idx, ts, kwargs)
    return _cb


# ── Qwen collage K/V injection ───────────────────────────────────────────────
#
# Architecture:  hidden = [text_tokens | image_tokens]   (joint attention)
#
# Phase 1 — capture (1 forward pass, no CFG, near-clean collage):
#   K, V at target_zone image-token positions are stored per layer.
#   These encode the pasted object's appearance at the correct spatial position
#   with matching RoPE coordinates → no positional collision.
#
# Phase 2 — denoising with injection (cutoff_frac window, default early steps 0–50%):
#   K[zone] = (1-s)*K + s*K_ref
#   V[zone] = (1-s)*V + s*V_ref
#   Late-step injection (sigma↓) avoids the clean/noisy mismatch that causes
#   artifacts when clean reference K/V are injected at high-noise early steps.


def _apply_complex_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Complex-valued RoPE matching apply_rotary_emb_qwen(use_real=False).
    x: (B, S, H, D),  freqs: (S, D//2) complex64."""
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_c * freqs.unsqueeze(1).to(x_c.device)).flatten(3)
    return x_out.type_as(x)


def _qwen_joint_qkv(
    attn,
    hidden_states: torch.Tensor,          # image tokens (B, S_img, D)
    encoder_hidden_states: torch.Tensor,  # text tokens  (B, S_txt, D)
    image_rotary_emb,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Project + norm + RoPE for both streams, concat into joint (B, S_tot, H, D).
    Layout stays (B, S, H, D) — matches QwenDoubleStreamAttnProcessor2_0 exactly.
    Returns (q, k, v, S_txt).
    """
    H = attn.heads

    # Image stream
    img_q = attn.to_q(hidden_states).unflatten(-1, (H, -1))
    img_k = attn.to_k(hidden_states).unflatten(-1, (H, -1))
    img_v = attn.to_v(hidden_states).unflatten(-1, (H, -1))
    if attn.norm_q is not None: img_q = attn.norm_q(img_q)
    if attn.norm_k is not None: img_k = attn.norm_k(img_k)

    # Text stream
    S_txt = encoder_hidden_states.shape[1]
    txt_q = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (H, -1))
    txt_k = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (H, -1))
    txt_v = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (H, -1))
    if attn.norm_added_q is not None: txt_q = attn.norm_added_q(txt_q)
    if attn.norm_added_k is not None: txt_k = attn.norm_added_k(txt_k)

    # Complex RoPE (separate freqs per stream)
    if image_rotary_emb is not None:
        img_freqs, txt_freqs = image_rotary_emb
        img_q = _apply_complex_rope(img_q, img_freqs)
        img_k = _apply_complex_rope(img_k, img_freqs)
        txt_q = _apply_complex_rope(txt_q, txt_freqs)
        txt_k = _apply_complex_rope(txt_k, txt_freqs)

    # Joint [txt, img] — (B, S_tot, H, D), NO transpose
    q = torch.cat([txt_q, img_q], dim=1)
    k = torch.cat([txt_k, img_k], dim=1)
    v = torch.cat([txt_v, img_v], dim=1)
    return q, k, v, S_txt


def _qwen_joint_out(
    attn,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    S_txt: int,
    encoder_hidden_states_mask,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention + output projections matching QwenDoubleStreamAttnProcessor2_0.
    Builds attention mask from encoder padding mask to exclude padding tokens.
    Returns (img_out, txt_out).
    """
    B = q.shape[0]
    S_tot = q.shape[1]

    # Build additive attention mask from text padding mask
    attn_mask = None
    if encoder_hidden_states_mask is not None:
        S_img = S_tot - S_txt
        img_ok = torch.ones((B, S_img), dtype=torch.bool, device=q.device)
        joint_mask = torch.cat([encoder_hidden_states_mask.bool(), img_ok], dim=1)
        attn_mask = joint_mask[:, None, None, :]  # (B, 1, 1, S_tot)

    if _HAS_DISPATCH:
        out = _dispatch_attn(q, k, v, attn_mask=attn_mask)
        out = out.flatten(2, 3).to(q.dtype)  # (B, S_tot, H*D)
    else:
        # Fallback: SDPA with (B, H, S, D) layout
        q2 = q.transpose(1, 2)
        k2 = k.transpose(1, 2)
        v2 = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q2, k2, v2, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).flatten(2, 3).to(q.dtype)  # (B, S_tot, H*D)

    txt_out = out[:, :S_txt].contiguous()
    img_out = out[:, S_txt:].contiguous()
    img_out = attn.to_out[0](img_out)
    if len(attn.to_out) > 1:
        img_out = attn.to_out[1](img_out)
    txt_out = attn.to_add_out(txt_out)
    return img_out, txt_out


class _QwenKVCapture:
    """
    Attention processor: 1-pass capture of K/V at target zone image tokens.
    Run once on the clean collage before denoising begins.
    """
    def __init__(self, n_img: int, target_zone: np.ndarray):
        self.n_img       = n_img
        self.target_zone = target_zone  # bool (n_img,)
        self.kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer      = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, encoder_hidden_states_mask=None,
                 image_rotary_emb=None, **kwargs):
        q, k, v, S_txt = _qwen_joint_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        S_tot = q.shape[1]

        # Input image tokens: positions [S_txt .. S_txt+n_img)
        img_off = S_txt
        if self._layer == 0:
            print(f"[KV-cap] layer0 S_img={hidden_states.shape[1]} S_txt={S_txt} "
                  f"n_img={self.n_img} S_tot={S_tot}  img_off={img_off}  "
                  f"target_off={img_off + self.n_img}")

        zone_np = img_off + np.where(self.target_zone)[0]
        if len(zone_np) > 0 and int(zone_np.max()) < S_tot:
            zone_t = torch.from_numpy(zone_np).to(k.device)
            # k/v are (B, S, H, D) — index along dim 1
            # nan_to_num guards against NaN from early high-sigma scheduler steps
            self.kv[self._layer] = (
                torch.nan_to_num(k[0:1, zone_t, :, :].detach().cpu()),
                torch.nan_to_num(v[0:1, zone_t, :, :].detach().cpu()),
            )

        self._layer += 1
        return _qwen_joint_out(attn, q, k, v, S_txt, encoder_hidden_states_mask)


class _QwenKVInject:
    """
    Attention processor: injects captured collage K/V at target zone image
    tokens during the active denoising window (cutoff_frac).
    """
    def __init__(self, n_img: int, target_zone: np.ndarray,
                 kv_ref: dict, obj_strength: float,
                 n_steps: int, cutoff_frac: Tuple[float, float]):
        self.n_img        = n_img
        self.target_zone  = target_zone
        self.kv_ref       = kv_ref
        self.obj_strength = obj_strength
        self.n_steps      = n_steps
        self.cutoff_frac  = cutoff_frac
        self._layer = 0
        self._step  = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, encoder_hidden_states_mask=None,
                 image_rotary_emb=None, **kwargs):
        q, k, v, S_txt = _qwen_joint_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        S_tot = q.shape[1]
        B = q.shape[0]

        # Target tokens: positions [S_txt+n_img .. S_txt+2*n_img)
        target_off = S_txt + self.n_img
        lo = int(self.cutoff_frac[0] * self.n_steps)
        hi = int(self.cutoff_frac[1] * self.n_steps)

        if lo <= self._step < hi and self._layer in self.kv_ref:
            zone_np = target_off + np.where(self.target_zone)[0]
            if len(zone_np) > 0 and int(zone_np.max()) < S_tot:
                zone_t = torch.from_numpy(zone_np).to(k.device)
                k_ref, v_ref = self.kv_ref[self._layer]
                # k_ref/v_ref are (1, n_zone, H, D); expand to full batch
                k_ref = k_ref.expand(B, -1, -1, -1).to(k.device, k.dtype)
                v_ref = v_ref.expand(B, -1, -1, -1).to(v.device, v.dtype)
                s = self.obj_strength
                k = k.clone()
                v = v.clone()
                # k/v are (B, S, H, D) — index along dim 1
                k[:, zone_t, :, :] = (1 - s) * k[:, zone_t, :, :] + s * k_ref
                v[:, zone_t, :, :] = (1 - s) * v[:, zone_t, :, :] + s * v_ref

        self._layer += 1
        return _qwen_joint_out(attn, q, k, v, S_txt, encoder_hidden_states_mask)


@torch.no_grad()
def run_with_qwen_kv_injection(
    pipe,
    collage:      Image.Image,
    prompt:       str,
    target_zone:  np.ndarray,
    seed:         int,
    num_steps:    int,
    guidance:     float,
    height:       int,
    width:        int,
    obj_strength: float = 0.7,
    cutoff_frac:  Tuple[float, float] = (0.0, 0.5),
    bcg_callback  = None,
) -> Image.Image:
    """
    Phase 1 — 1-step capture pass (no CFG, batch=1):
        collage → Qwen transformer → K,V at target zone image tokens per layer

    Phase 2 — full denoising with injection:
        Within cutoff_frac steps, blend captured K,V into live image tokens
        at target zone positions.  Default window is the EARLY 50% of steps
        (sigma↑, high noise) per Personalize Anything (arxiv:2503.12590) —
        early steps are more receptive to appearance guidance because the
        denoiser is still forming global structure.
    """
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_img  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))

    orig_procs = pipe.transformer.attn_processors

    # ── Phase 1: capture ─────────────────────────────────────────────────────
    capture_proc = _QwenKVCapture(n_img=n_img, target_zone=target_zone)
    pipe.transformer.set_attn_processor(capture_proc)

    # num_inference_steps=1 causes sigma=1.0 → divide-by-zero in the scheduler
    # (stretched_t = 1 - one_minus_z/scale_factor when one_minus_z=0) → NaN K/V.
    # Use 20 steps so sigmas spread across (0,1) and avoid the boundary.
    # A per-step callback resets _layer so keys 0-59 are overwritten each step;
    # the final stored K/V come from the last (most denoised) step.
    _CAP_STEPS = 20

    def _capture_step_cb(pipe_ref, step_idx, timestep, cb_kwargs):
        capture_proc._layer = 0
        return cb_kwargs

    gen0 = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(
        image=collage, prompt=prompt,
        num_inference_steps=_CAP_STEPS, true_cfg_scale=1.0,
        height=height, width=width,
        generator=gen0, output_type="latent",
        callback_on_step_end=_capture_step_cb,
        callback_on_step_end_tensor_inputs=[],
    )
    kv_ref = dict(capture_proc.kv)
    print(f"    [KV] Captured at {len(kv_ref)} / 60 layers  "
          f"zone={target_zone.sum()} tokens ({100*target_zone.mean():.1f}%)")

    # ── Phase 2: denoising with injection ─────────────────────────────────────
    inject_proc = _QwenKVInject(
        n_img=n_img, target_zone=target_zone, kv_ref=kv_ref,
        obj_strength=obj_strength, n_steps=num_steps, cutoff_frac=cutoff_frac,
    )
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pipe_ref, step_idx, timestep, cb_kwargs):
        inject_proc._step  = step_idx + 1
        inject_proc._layer = 0
        if bcg_callback is not None:
            cb_kwargs = bcg_callback(pipe_ref, step_idx, timestep, cb_kwargs)
        return cb_kwargs

    tensor_inputs = ["latents"] if bcg_callback is not None else []
    gen1 = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        image=collage, prompt=prompt,
        negative_prompt="blurry, distorted, low quality, watermark, text, artifacts",
        num_inference_steps=num_steps, true_cfg_scale=guidance,
        height=height, width=width, generator=gen1,
        callback_on_step_end=_step_cb,
        callback_on_step_end_tensor_inputs=tensor_inputs,
    ).images[0]

    pipe.transformer.set_attn_processor(orig_procs)
    return result


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_collage_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    seed:           int,
    num_steps:      int,
    obj_guidance:   float,
    scene_guidance: float,
    collage_mode:   str,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
    use_kv:         bool = False,
    obj_strength:   float = 0.7,
    cutoff_frac:    Tuple[float, float] = (0.0, 0.5),
) -> List[Image.Image]:
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
        step_tag = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"
        with open(os.path.join(out_dir, f"bbox_{step_tag}.txt"), "w") as f:
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
            obj_cache = os.path.join(out_dir, f"obj_gen_{name}.png")
            if os.path.isfile(obj_cache):
                print(f"  [A] Reusing cached obj_gen_{name}.png")
                obj_img = Image.open(obj_cache).convert("RGB")
            else:
                print(f"  [A] Generating '{desc}' from sketch ...")
                obj_img = generate_from_sketch(
                    pipe=pipe, sketch_path=sketch_path, description=desc,
                    seed=seed, num_steps=num_steps, guidance=obj_guidance,
                    height=height, width=width, device=device,
                )
                obj_img.save(obj_cache)
                print(f"      Saved: obj_gen_{name}.png")

            # Stage MASK: extract object silhouette
            ref_mask = _compute_obj_mask(obj_img)
            n_px = ref_mask.sum()
            print(f"  [MASK] Object pixels: {n_px} ({100*n_px/(height*width):.1f}%)")
            if n_px < 50:
                print("         Fallback: using center 50% crop as object region")
                ref_mask = np.zeros((height, width), dtype=np.uint8)
                ref_mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = 1

            # Stage COL: paste object into scene
            print(f"  [COL] Building collage scene (mode={collage_mode}) ...")
            collage_scene, _ = build_collage_scene(
                scene=scene, obj_img=obj_img, ref_mask=ref_mask,
                target_bbox=(bx1, by1, bx2, by2), collage_mode=collage_mode,
            )
            collage_scene.save(os.path.join(out_dir, f"collage_{name}.png"))
            print(f"      Saved: collage_{name}.png")

            obj_mask_har = _collage_obj_mask(collage_scene, scene)
            n_har = int(obj_mask_har.sum())
            print(f"  [MASK] Object footprint: {n_har} px ({100*n_har/obj_mask_har.size:.1f}%)")

            # Qwen instruction-following: declarative sentence works well for insertion
            blend_p = f"A photorealistic room with a {desc} placed naturally on the floor."
            neg_p   = "blurry, distorted, low quality, watermark, text, artifacts"

        else:  # remove
            print(f"  [A] Removal — skipping object generation")

            # Stage COL: inpaint the object away as the Qwen reference image
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

            # Qwen is instruction-following — imperative style outperforms declarative for removal
            blend_p = (
                f"Remove the {desc} from the room completely. "
                f"Fill the area with seamless wooden floor and white wall that match the surroundings. "
                f"Keep all other parts of the room exactly the same."
            )
            # Adding the object to the negative prompt reinforces what to erase
            neg_p = (
                f"blurry, distorted, low quality, watermark, text, artifacts, {desc}"
            )

        # Stage K: Qwen integration pass with BLD latent BCG
        print(f"  [K] Qwen integration pass  action={action}  kv={use_kv} ...")
        print(f"      Prompt: {blend_p[:120]}...")
        with open(os.path.join(out_dir, f"prompt_{step_tag}.txt"), "w") as f:
            f.write(blend_p)

        pipe_dtype = next(pipe.transformer.parameters()).dtype
        bcg_cb = _make_bcg_latent_callback(
            scene=scene, mask_path=mask_path, pipe=pipe,
            height=height, width=width, device=device, dtype=pipe_dtype,
        )

        if use_kv:
            target_zone = _placement_mask_to_token_zone(mask_path, height, width, pipe)
            print(f"      K/V zone: {target_zone.sum()} tokens  "
                  f"strength={obj_strength}  cutoff={cutoff_frac}")
            next_scene = run_with_qwen_kv_injection(
                pipe=pipe, collage=collage_scene, prompt=blend_p,
                target_zone=target_zone,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width,
                obj_strength=obj_strength, cutoff_frac=cutoff_frac,
                bcg_callback=bcg_cb,
            )
        elif action == "insert":
            # Object identity anchor only for insertion (not removal)
            anchor_cb = _make_obj_latent_composite_callback(
                collage=collage_scene, obj_mask_har=obj_mask_har,
                pipe=pipe, height=height, width=width,
                device=device, dtype=pipe_dtype,
                anchor_strength=obj_strength * 0.5,
            )
            combined_cb = _compose_callbacks(bcg_cb, anchor_cb)
            next_scene = run_qwen(
                pipe=pipe, canvas=collage_scene, prompt=blend_p,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width, bcg_callback=combined_cb,
                negative_prompt=neg_p,
            )
        else:  # remove, no kv
            next_scene = run_qwen(
                pipe=pipe, canvas=collage_scene, prompt=blend_p,
                seed=seed, num_steps=num_steps, guidance=scene_guidance,
                height=height, width=width, bcg_callback=bcg_cb,
                negative_prompt=neg_p,
            )

        result_path = os.path.join(out_dir, f"result_{step_tag}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        # Stage BCG: pixel-space tight-mask composite
        print(f"  [BCG] Background restore ...")
        bcg_mask = cv2.dilate(obj_mask_har, np.ones((21, 21), np.uint8), iterations=1)
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
        description="AnyDoor collage method inside Qwen-Image-Edit-2509."
    )
    p.add_argument("--sketch_dir",   required=True,
                   help="Directory containing {name}.png sketches and mask_{name}.png files.")
    p.add_argument("--hf_token",     required=True)
    p.add_argument("--cache_dir",    default="./models")
    p.add_argument("--out_dir",      default="results/phase1_qwen")
    p.add_argument("--model",        default="Qwen/Qwen-Image-Edit-2509",
                   help="HuggingFace model ID. Default: Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--config",       default=None,
                   help="JSON list of {name, description}. Overrides built-in EDITS.")
    p.add_argument("--lora_guidance", type=float, default=3.5,
                   help="CFG scale for Stage A (object generation). Default 3.5.")
    p.add_argument("--guidance",      type=float, default=3.5,
                   help="CFG scale for Stage K (scene integration). Default 3.5.")
    p.add_argument("--collage_mode",  default="full",
                   choices=["full", "sobel", "blend"],
                   help="Collage paste mode. Default: full.")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_steps",    type=int, default=50,
                   help="Inference steps. Qwen typically needs 50. Default 50.")
    p.add_argument("--height",       type=int, default=1024)
    p.add_argument("--width",        type=int, default=1024)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--use_kv",       action="store_true",
                   help="Enable collage K/V injection for object appearance preservation.")
    p.add_argument("--obj_strength", type=float, default=0.7,
                   help="K/V injection strength (0-1). Default 0.7.")
    p.add_argument("--cutoff_frac",  default="0.0,0.5",
                   help="Injection active window as 'start,end' fractions. Default '0.5,1.0'.")
    p.add_argument("--cpu_offload",  action="store_true",
                   help="Enable model CPU offload. Use if VRAM < 45 GB.")
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
    print(f"  phase1_collage_qwen  —  AnyDoor Collage + Qwen-Image-Edit-2509")
    print(f"{_SEP}")
    print(f"  Model        : {args.model}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  Collage mode : {args.collage_mode}")
    print(f"  Guidance     : obj={args.lora_guidance}  scene={args.guidance}")
    print(f"  KV injection : {args.use_kv}"
          + (f"  strength={args.obj_strength}  cutoff={args.cutoff_frac}"
             if args.use_kv else ""))
    print(f"  Steps        : {args.num_steps}")
    print(f"  CPU offload  : {args.cpu_offload}")
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    print(f"Loading {args.model} ...")
    pipe = load_qwen_pipeline(
        model_path  = args.model,
        hf_token    = args.hf_token,
        device      = args.device,
        cache_dir   = args.cache_dir,
        cpu_offload = args.cpu_offload,
    )

    # Base scene: pass a grey canvas, ask Qwen to generate the room
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_qwen(
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
        sketch_dir=args.sketch_dir,
        seed=args.seed, num_steps=args.num_steps,
        obj_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        collage_mode=args.collage_mode,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
        use_kv=args.use_kv,
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
