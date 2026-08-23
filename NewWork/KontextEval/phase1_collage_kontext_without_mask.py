# -*- coding: utf-8 -*-
"""
phase1_collage_kontext_without_mask.py — AnyDoor Collage + FLUX Kontext, mask-free

Differences from phase1_collage_kontext.py
-------------------------------------------
  1. No user-drawn mask_{name}.png files required.
  2. Object placement is decided by monocular depth estimation (Depth Anything V2)
     — no VLM, no annotation.
  3. A rectangle mask is auto-generated from the depth-predicted bbox.

Placement strategy (no VLM)
-----------------------------
  Insert : Depth map of scene → find flat floor region → bbox
  Remove : If object was inserted this session, reuse its tracked bbox.
           Else: depth anomaly detection (objects appear as "bumps" above
           the floor median in the lower image region).

Depth model
-----------
  Primary  : depth-anything/Depth-Anything-V2-Small-hf   (~50 MB, fast)
  Fallback : Intel/dpt-large                             (~800 MB)
  Fallback²: geometric heuristic (lower 40 % of frame)  (no download)

Pipeline
--------
  Stage A    : Sketch → LoRA FLUX → obj_img
  Stage DEPTH: Depth Anything estimates scene depth → floor bbox → mask_np
  Stage K    : Dual K/V injection — clean scene as Kontext reference
                 • Mask zone tokens   ← obj_img K/V  (appearance)
                 • Background tokens  ← scene context K/V  (KV-Edit-style preservation)
               No pixel-space collage. No BLD latent callback.
  Stage BCG  : Pixel-space restore from result-vs-scene diff mask
  Loop       : result → next scene

Usage
-----
  python NewWork/KontextEval/phase1_collage_kontext_without_mask.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir results/phase1_collage_nomask

Key flags
----------
  --depth_model   HF model id for depth estimation.
                  Default: depth-anything/Depth-Anything-V2-Small-hf
  --kv_injection  Enable dual K/V processor (recommended; off by default for baseline).
  --obj_strength  K/V strength for mask zone tokens. Default: 0.5
  --bg_strength   K/V strength for background tokens (KV-Edit). Default: 0.8
  --guidance      float  Kontext CFG scale. Default 2.5.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def load_vlm(model_id: str, cache_dir: str, device: str = "cpu"):
    """
    Load a vision-language model for placement anchor prediction.

    device='cpu'  — bfloat16 on CPU. Safe alongside FLUX on any GPU.
    device='cuda' — tries 4-bit NF4 (bitsandbytes) first, falls back to bf16.
    """
    from transformers import AutoProcessor

    print(f"  Loading VLM '{model_id}' on {device} ...")

    def _try_load(load_kwargs, to_device):
        is_q25 = "Qwen2.5" in model_id or "Qwen2_5" in model_id
        classes = []
        if is_q25:
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
        if not is_q25:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                classes.append(Qwen2_5_VLForConditionalGeneration)
            except ImportError:
                pass
        last_err = None
        for cls in classes:
            try:
                m = cls.from_pretrained(model_id, **load_kwargs)
                return (m.to(device) if to_device else m).eval()
            except Exception as e:
                last_err = e
        try:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
            return (m.to(device) if to_device else m).eval()
        except Exception as e:
            last_err = e
        raise RuntimeError(f"Could not load VLM '{model_id}'. Last error: {last_err}")

    model = None
    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            model = _try_load(
                dict(quantization_config=quant, device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            print("  VLM loaded (4-bit NF4 GPU)")
        except Exception as e:
            print(f"    [VLM] 4-bit load failed ({e!s:.80}) — falling back to bf16")
        if model is None:
            model = _try_load(
                dict(torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            print("  VLM loaded (bf16 GPU)")
    else:
        model = _try_load(
            dict(torch_dtype=torch.bfloat16, cache_dir=cache_dir),
            to_device=True,
        )
        print("  VLM loaded (bf16 CPU)")

    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=True,
    )
    return model, processor


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


# ── Constants ─────────────────────────────────────────────────────────────────

EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "insert"},
    {"name": "vase",    "description": "black ceramic vase with flowers",  "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "remove"},
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "remove"},
]
BASE_PROMPT = "A empty room with a wooden floor, white walls, and a window letting in natural light."
LORA_ID     = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
_SEP        = "═" * 60


# ── Object mask extraction ────────────────────────────────────────────────────

def _compute_obj_mask(obj_img: Image.Image,
                       grey: tuple = (128, 128, 128),
                       tolerance: int = 20) -> np.ndarray:
    """Binary uint8 mask (H, W): 1=object pixel, 0=background."""
    arr = np.array(obj_img.convert("RGB"), dtype=np.int32)
    is_grey  = ((np.abs(arr[:, :, 0] - grey[0]) <= tolerance) &
                (np.abs(arr[:, :, 1] - grey[1]) <= tolerance) &
                (np.abs(arr[:, :, 2] - grey[2]) <= tolerance))
    is_white = (arr[:, :, 0] >= 230) & (arr[:, :, 1] >= 230) & (arr[:, :, 2] >= 230)
    return (~(is_grey | is_white)).astype(np.uint8)


def _neutralize_white_bg(img: Image.Image, threshold: int = 240,
                          fill: tuple = (128, 128, 128)) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    white = (arr[:, :, 0] >= threshold) & (arr[:, :, 1] >= threshold) & (arr[:, :, 2] >= threshold)
    arr[white] = fill
    return Image.fromarray(arr)


def _obj_token_mask(obj_img: Image.Image, h_lat: int, w_lat: int,
                    grey: tuple = (128, 128, 128), tol: int = 10,
                    min_frac: float = 0.05) -> np.ndarray:
    """Bool mask (h_lat*w_lat,): True = object token."""
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


# ── Depth estimation — replaces user masks ────────────────────────────────────

def load_depth_model(
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    cache_dir: str = "./models",
):
    """
    Load a monocular depth estimator.

    Returns (model, processor, kind_str) so the caller can dispatch correctly.
    Falls back through: Depth-Anything-V2 → Intel/dpt-large → None (geometric).
    """
    candidates = [
        (model_id, "depth-anything"),
        # Depth Anything V2 Large (NIPS 2024): same API, ~800 MB, far better than DPT-large
        ("depth-anything/Depth-Anything-V2-Large-hf", "depth-anything"),
    ]
    for mid, kind in candidates:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            proc  = AutoImageProcessor.from_pretrained(mid, cache_dir=cache_dir)
            model = AutoModelForDepthEstimation.from_pretrained(mid, cache_dir=cache_dir)
            print(f"  [Depth] Loaded '{mid}'")
            return model.eval(), proc, kind
        except Exception as e:
            print(f"  [Depth] '{mid}' unavailable: {e!s:.70}")
    print("  [Depth] No depth model loaded — using geometric floor fallback.")
    return None, None, "fallback"


@torch.no_grad()
def estimate_depth(model, processor, scene: Image.Image) -> Optional[np.ndarray]:
    """
    Run depth estimation on scene.
    Returns (H, W) float32 depth map normalised to [0, 1], where larger values
    mean CLOSER to the camera (Depth Anything convention: inverted disparity).
    Returns None if model is None.
    """
    if model is None:
        return None
    inputs = processor(images=scene, return_tensors="pt")
    # Depth models are small — run on CPU to avoid interfering with FLUX on GPU
    out = model(**inputs)
    depth = out.predicted_depth[0].numpy().astype(np.float32)
    lo, hi = depth.min(), depth.max()
    return (depth - lo) / (hi - lo + 1e-8)


def _depth_to_scene_size(depth_np: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinearly resize depth map to (height, width)."""
    return np.array(
        Image.fromarray((depth_np * 255).astype(np.uint8)).resize(
            (width, height), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0


def find_floor_bbox(
    depth_np:   Optional[np.ndarray],
    height:     int,
    width:      int,
    floor_frac: float = 0.42,     # image rows below this are "possible floor"
    pad_frac:   float = 0.06,     # inset padding as fraction of region size
) -> Tuple[int, int, int, int]:
    """
    Select a placement bbox on the floor using depth-guided flat-region detection.

    Algorithm
    ----------
    1. Focus on lower (1 - floor_frac) of the image.
    2. Compute Sobel gradient magnitude of the depth map in that region.
       Low gradient ↔ flat surface ↔ floor.
    3. Threshold at 50th percentile → flat-surface binary mask.
    4. Find the largest connected flat region.
    5. Return its bbox (with inset padding so the object doesn't touch edges).

    Geometric fallback (no depth model): lower-center rectangle.
    """
    floor_y0 = int(height * floor_frac)

    if depth_np is None:
        print(f"    [PLACE] Geometric fallback: center-bottom region")
        return width // 5, floor_y0, 4 * width // 5, int(height * 0.90)

    depth = _depth_to_scene_size(depth_np, height, width)
    floor = depth[floor_y0:]   # (floor_rows, W)

    # Gradient magnitude — low = flat
    sx = cv2.Sobel(floor, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(floor, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sx ** 2 + sy ** 2)

    flat_mask = (grad < np.percentile(grad, 55)).astype(np.uint8)
    k = np.ones((11, 11), np.uint8)
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_CLOSE, k)
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flat_mask)
    if n_labels <= 1:
        # Whole floor is uniformly flat — use centre-bottom
        x1, y1 = width // 5, floor_y0 + (height - floor_y0) // 4
        x2, y2 = 4 * width // 5, int(height * 0.90)
        print(f"    [PLACE] Uniform floor — centre-bottom: x=[{x1},{x2}] y=[{y1},{y2}]")
        return x1, y1, x2, y2

    # Largest component (skip label 0 = background)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp    = (labels == largest)
    ys, xs  = np.where(comp)

    ry1 = int(ys.min()) + floor_y0
    ry2 = int(ys.max()) + floor_y0
    rx1 = int(xs.min())
    rx2 = int(xs.max())

    px = max(12, int((rx2 - rx1) * pad_frac))
    py = max(12, int((ry2 - ry1) * pad_frac))
    x1 = max(0,      rx1 + px)
    y1 = max(0,      ry1 + py)
    x2 = min(width,  rx2 - px)
    y2 = min(height, ry2 - py)

    area_frac = comp.sum() / (flat_mask.shape[0] * flat_mask.shape[1])
    print(f"    [PLACE] Floor bbox: x=[{x1},{x2}] y=[{y1},{y2}]  "
          f"flat_area={area_frac*100:.1f}%")
    return x1, y1, x2, y2


def detect_object_on_floor(
    depth_np:   Optional[np.ndarray],
    height:     int,
    width:      int,
    floor_frac: float = 0.42,
) -> Tuple[int, int, int, int]:
    """
    Locate the most prominent non-floor object in the lower image region.

    Used for removal when the insertion bbox wasn't tracked (e.g. pre-existing
    objects).  Objects appear as "bumps" — depth values closer than the local
    floor median.

    Falls back to centre-bottom if no clear anomaly is found.
    """
    floor_y0 = int(height * floor_frac)

    if depth_np is None:
        print(f"    [DETECT] Geometric fallback: centre-bottom")
        return width // 4, floor_y0, 3 * width // 4, int(height * 0.88)

    depth = _depth_to_scene_size(depth_np, height, width)
    floor = depth[floor_y0:]    # closer values are larger in Depth-Anything

    floor_med   = np.median(floor)
    floor_std   = floor.std()
    # Objects stick OUT → significantly closer (higher depth value) than floor median
    bump_thresh = floor_med + max(0.06, 0.8 * floor_std)
    obj_mask    = (floor > bump_thresh).astype(np.uint8)

    k = np.ones((9, 9), np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, k)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask)
    if n_labels <= 1:
        print(f"    [DETECT] No clear depth anomaly — using centre-bottom fallback")
        return width // 4, floor_y0, 3 * width // 4, int(height * 0.88)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp    = (labels == largest)
    ys, xs  = np.where(comp)

    x1 = max(0,      int(xs.min()))
    y1 = max(0,      int(ys.min()) + floor_y0)
    x2 = min(width,  int(xs.max()))
    y2 = min(height, int(ys.max()) + floor_y0)

    print(f"    [DETECT] Depth-anomaly bbox: x=[{x1},{x2}] y=[{y1},{y2}]")
    return x1, y1, x2, y2


def _rect_mask_from_bbox(
    x1: int, y1: int, x2: int, y2: int,
    height: int, width: int,
) -> np.ndarray:
    """Rectangle uint8 mask (H×W), 255 inside bbox."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _token_zone_from_mask_np(
    mask_np: np.ndarray, height: int, width: int, pipe,
) -> np.ndarray:
    """Convert a uint8 pixel mask (H,W) to a flat bool token mask (n_gen,)."""
    vae_sf   = getattr(pipe, "vae_scale_factor", 8)
    h_lat    = height // (vae_sf * 2)
    w_lat    = width  // (vae_sf * 2)
    mask_pil = Image.fromarray(mask_np).resize((w_lat, h_lat), Image.NEAREST)
    zone     = (np.array(mask_pil) > 127).reshape(-1)
    print(f"    [KV-zone] {zone.sum()} / {h_lat * w_lat} tokens ({100*zone.mean():.1f}%)")
    return zone


# ── DAAM attention heatmap — automatic placement mask ────────────────────────

_HEATMAP_STOP = {"a", "an", "the", "with", "on", "in", "of", "and", "or",
                 "no", "not", "very", "some", "this", "that", "is", "are",
                 "for", "to", "at", "its", "into", "from"}


def _find_desc_token_indices(pipe, prompt: str, description: str) -> List[int]:
    """
    Return T5 token indices for every content word in description.
    Averaging heatmaps across multiple tokens (e.g. 'yellow', 'rubber', 'ball')
    gives a much stronger and more specific placement signal than a single word.
    """
    tokenizer   = pipe.tokenizer_2          # T5 SentencePiece
    ids         = tokenizer.encode(prompt, add_special_tokens=False)
    content     = [w.lower().strip(".,!?") for w in description.split()
                   if w.lower().strip(".,!?") not in _HEATMAP_STOP and len(w) >= 3]

    found: List[int] = []
    for i, tid in enumerate(ids):
        tok = tokenizer.decode([tid]).strip().lower()
        for word in content:
            if (tok == word
                    or (len(tok) >= 3 and tok in word)
                    or (len(word) >= 3 and word in tok)):
                if i not in found:
                    found.append(i)
                break

    if not found:
        print(f"    [HEATMAP] No content tokens for '{description}' — using idx 0")
        return [0]
    print(f"    [HEATMAP] Content-word tokens for '{description}': indices {found[:10]}")
    return found


class _HeatmapCapture:
    """
    DAAM-style attention heatmap capture.

    Computes raw Q_gen · K_text[obj_indices] / √d at each TIER_A double-stream
    layer and accumulates the per-gen-token score.  Averaging over multiple
    description content tokens (e.g. 'yellow', 'rubber', 'ball') is far more
    robust than a single ambiguous word.
    """
    def __init__(self, n_gen: int, obj_token_indices: List[int]):
        self.n_gen             = n_gen
        self.obj_token_indices = obj_token_indices
        self._layer = 0; self._step = 0
        self.heatmap: Optional[np.ndarray] = None
        self._count  = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2)
        k = k.view(B,-1,attn.heads,hd).transpose(1,2)
        v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        # Average raw Q_gen · K_text across all content-word token keys
        if is_double and self._layer in _TIER_A:
            g_lo = txt_len; g_hi = txt_len + self.n_gen
            if q.shape[2] >= g_hi:
                scale = hd ** -0.5
                q_gen = q[:,:,g_lo:g_hi,:]          # (B, H, n_gen, hd)
                acc   = None
                n_hit = 0
                for obj_idx in self.obj_token_indices:
                    if obj_idx < txt_len:
                        score = (q_gen @ k[:,:,obj_idx:obj_idx+1,:].transpose(-1,-2)
                                 ).squeeze(-1) * scale  # (B, H, n_gen)
                        s_np  = score.float().mean(dim=(0,1)).detach().cpu().numpy()
                        acc   = s_np if acc is None else acc + s_np
                        n_hit += 1
                if acc is not None and n_hit > 0:
                    if self.heatmap is None:
                        self.heatmap = np.zeros(self.n_gen, dtype=np.float32)
                    self.heatmap += acc / n_hit
                    self._count  += 1
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out     = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


def _vlm_predict_placement(
    vlm_model, vlm_processor,
    scene: Image.Image, description: str,
) -> Optional[str]:
    """
    Ask the VLM where the object naturally belongs in this scene.
    Returns a short placement phrase like 'on the kitchen counter' or
    'on the grass', ready to be appended to the heatmap prompt.
    Returns None if the VLM output cannot be parsed.
    """
    device = next(vlm_model.parameters()).device

    w, h = scene.size
    if max(w, h) > 768:
        s = 768 / max(w, h)
        scene_q = scene.resize((int(w * s), int(h * s)), Image.LANCZOS).convert("RGB")
    else:
        scene_q = scene.convert("RGB")

    query = (
        f"Where in this scene would a {description} naturally be placed? "
        f"Reply with ONLY a short phrase (2-6 words) starting with 'on', 'against', or 'near'. "
        f"Examples: 'on the floor', 'on the wooden table', 'against the brick wall', "
        f"'on the stone path', 'on the kitchen counter'. "
        f"Give ONLY the phrase — no explanation, no punctuation."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": scene_q},
            {"type": "text",  "text": query},
        ],
    }]

    try:
        text   = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = vlm_processor(
            text=[text], images=[scene_q], padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out_ids = vlm_model.generate(**inputs, max_new_tokens=24, do_sample=False)
        answer = vlm_processor.decode(
            out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip().lower().rstrip(".,!?")

        print(f"    [VLM PLACE] '{answer}'")

        # Accept if it already starts with a valid spatial preposition
        for prefix in ("on ", "against ", "near ", "beside ", "under ", "above "):
            if answer.startswith(prefix):
                # Trim to at most 6 words so we don't append an essay
                return " ".join(answer.split()[:6])

        # VLM gave a full sentence — try to extract the first valid sub-phrase
        for marker in ("on the", "on a", "against the", "near the", "beside the",
                       "under the", "above the"):
            idx = answer.find(marker)
            if idx >= 0:
                fragment = " ".join(answer[idx:].split()[:6]).rstrip(".,!?")
                print(f"    [VLM PLACE] Extracted: '{fragment}'")
                return fragment

        print(f"    [VLM PLACE] Unparseable ('{answer}') — no anchor used")
        return None
    except Exception as e:
        print(f"    [VLM PLACE] Error: {e} — no anchor used")
        return None


@torch.no_grad()
def run_heatmap_pass(
    pipe, scene: Image.Image, prompt: str, description: str,
    seed: int, height: int, width: int,
    n_steps: int = 6,
    occupied_mask: Optional[np.ndarray] = None,
    anchor: Optional[str] = None,
) -> np.ndarray:
    """
    Short n_steps Kontext forward pass → DAAM attention heatmap.

    prompt     : the generation prompt (blend_p).
    description: full object description for content-word token discovery.
    anchor     : optional placement phrase from VLM (e.g. 'on the kitchen
                 counter').  Appended to prompt for the heatmap pass so the
                 T5 contextual encoding of the object tokens is spatially
                 biased toward the predicted surface.  When None the prompt
                 is used as-is — the multi-token description averaging
                 provides the spatial signal.
    Returns (h_lat, w_lat) float32 in [0, 1].
    """
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_gen  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))
    h_lat  = height // (vae_sf * 2)
    w_lat  = width  // (vae_sf * 2)

    if anchor:
        hm_prompt = prompt.rstrip(".! ") + f", {anchor}."
        print(f"    [HEATMAP] Anchor: '{anchor}'")
    else:
        hm_prompt = prompt
        print(f"    [HEATMAP] No anchor — using description tokens only")
    orig_procs        = pipe.transformer.attn_processors
    obj_token_indices = _find_desc_token_indices(pipe, hm_prompt, description)

    cap = _HeatmapCapture(n_gen=n_gen, obj_token_indices=obj_token_indices)
    pipe.transformer.set_attn_processor(cap)

    def _step_cb(pr, si, ts, ck):
        cap._step = si + 1; cap._layer = 0
        return ck

    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(image=scene, prompt=hm_prompt,
         num_inference_steps=n_steps, guidance_scale=2.5,
         height=height, width=width, max_sequence_length=512,
         generator=gen, output_type="latent",
         callback_on_step_end=_step_cb,
         callback_on_step_end_tensor_inputs=[])
    pipe.transformer.set_attn_processor(orig_procs)

    if cap.heatmap is None or cap._count == 0:
        print("    [HEATMAP] No maps captured — returning uniform heatmap")
        return np.ones((h_lat, w_lat), dtype=np.float32)

    hm = (cap.heatmap / cap._count).reshape(h_lat, w_lat)

    # Suppress already-occupied regions so successive objects don't overlap
    if occupied_mask is not None:
        occ = cv2.resize((occupied_mask > 127).astype(np.uint8),
                         (w_lat, h_lat), interpolation=cv2.INTER_NEAREST)
        occ = cv2.dilate(occ, np.ones((5, 5), np.uint8), iterations=2)
        hm  = hm * (1.0 - occ.astype(np.float32))

    lo, hi = hm.min(), hm.max()
    if hi - lo > 1e-6:
        hm = (hm - lo) / (hi - lo)
    peak = np.unravel_index(hm.argmax(), hm.shape)
    print(f"    [HEATMAP] {cap._count} maps, {len(obj_token_indices)} tokens  "
          f"peak=({peak[1]},{peak[0]})")
    return hm.astype(np.float32)


def heatmap_to_mask(
    heatmap:    np.ndarray,
    height:     int,
    width:      int,
    percentile: float = 72,
    blur_ksize: int   = 11,
    blur_sigma: float = 4.0,
    min_frac:   float = 0.04,
    max_frac:   float = 0.45,
) -> np.ndarray:
    """
    (h_lat, w_lat) float heatmap → uint8 pixel mask (H, W).

    1. Gaussian blur.
    2. Threshold at percentile → binary latent-res mask.
    3. Keep largest connected component.
    4. Size clamp (min/max fraction).
    5. Fit minimum-area rotated rectangle (cv2.minAreaRect) → tight
       four-point placement zone that follows floor perspective.
    6. Upsample to (H, W).
    """
    h_lat, w_lat = heatmap.shape
    blurred = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), blur_sigma)
    binary  = (blurred >= np.percentile(blurred, percentile)).astype(np.uint8)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((2, 2), np.uint8))

    # Largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary  = (labels == largest).astype(np.uint8)

    frac = binary.mean()
    if frac < min_frac:
        cy, cx = np.unravel_index(blurred.argmax(), blurred.shape)
        r = max(3, int(min(h_lat, w_lat) * 0.15))
        binary = np.zeros_like(binary)
        binary[max(0,cy-r):min(h_lat,cy+r), max(0,cx-r):min(w_lat,cx+r)] = 1
        print(f"    [HEATMAP] Mask too small ({frac*100:.1f}%) — peak neighbourhood")
    elif frac > max_frac:
        p2     = percentile + (100 - percentile) * 0.5
        binary = (blurred >= np.percentile(blurred, p2)).astype(np.uint8)
        print(f"    [HEATMAP] Mask too large ({frac*100:.1f}%) — re-threshold p={p2:.0f}")

    # Fit minimum-area rotated rectangle → tight four-corner placement zone
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.vstack(contours)
        if len(all_pts) >= 4:
            rect    = cv2.minAreaRect(all_pts)
            box     = cv2.boxPoints(rect).astype(np.int32)
            filled  = np.zeros_like(binary)
            cv2.fillPoly(filled, [box], 1)
            binary  = filled
            cx, cy  = rect[0]
            print(f"    [HEATMAP] Rotated rect: centre=({cx:.0f},{cy:.0f})  "
                  f"size=({rect[1][0]:.0f}×{rect[1][1]:.0f})  angle={rect[2]:.1f}°")

    mask_np = np.array(
        Image.fromarray(binary * 255).resize((width, height), Image.NEAREST)
    ).astype(np.uint8)
    print(f"    [HEATMAP] Final mask: {int((mask_np>127).mean()*100)}% of pixels")
    return mask_np


# ── K/V injection ─────────────────────────────────────────────────────────────

_TIER_A         = {0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56}
_SINGLE_TXT_LEN = 512


def _spatial_align_k_from_obj(
    k_obj: torch.Tensor,      # (1, H, n_gen, d) — K from obj_img capture pass
    obj_mask: np.ndarray,     # (n_gen,) bool — non-background tokens in obj_img
    target_zone: np.ndarray,  # (n_gen,) bool — target zone in scene
    h_lat: int,
    w_lat: int,
) -> torch.Tensor:
    """
    Returns (1, H, n_gen, d): for each scene position, the K from the
    geometrically corresponding position in obj_img (bbox-to-bbox mapping).

    Fixes the mean-pool collapse: instead of every target token seeing the
    same average K, handlebar area gets handlebar K, wheel area gets wheel K.
    """
    obj_2d = obj_mask.reshape(h_lat, w_lat)
    tgt_2d = target_zone.reshape(h_lat, w_lat)

    def _bbox(m):
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return 0, h_lat - 1, 0, w_lat - 1
        return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])

    r0_o, r1_o, c0_o, c1_o = _bbox(obj_2d)
    r0_t, r1_t, c0_t, c1_t = _bbox(tgt_2d)

    RR, CC = np.meshgrid(np.arange(h_lat, dtype=np.float32),
                         np.arange(w_lat, dtype=np.float32), indexing='ij')

    def _map(pos, p0t, p1t, p0o, p1o):
        if p1t == p0t:
            return np.full_like(pos, (p0o + p1o) * 0.5)
        return np.clip(p0o + (pos - p0t) / float(p1t - p0t) * (p1o - p0o), p0o, p1o)

    R_idx = np.clip(np.round(_map(RR, r0_t, r1_t, r0_o, r1_o)).astype(int), 0, h_lat - 1)
    C_idx = np.clip(np.round(_map(CC, c0_t, c1_t, c0_o, c1_o)).astype(int), 0, w_lat - 1)
    flat  = torch.from_numpy((R_idx * w_lat + C_idx).reshape(-1)).to(k_obj.device)
    return k_obj[:, :, flat, :]  # (1, H, n_gen, d)


class _KVCapture:
    def __init__(self, n_gen: int):
        self.n_gen = n_gen
        self.kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer = 0

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq, q], dim=2); k = torch.cat([ek, k], dim=2); v = torch.cat([ev, v], dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb); k = apply_rotary_emb(k, image_rotary_emb)
        if self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                self.kv[self._layer] = (k[:,:,r_lo:r_hi,:].detach().cpu(),
                                        v[:,:,r_lo:r_hi,:].detach().cpu())
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)
        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


class _KVInject:
    def __init__(self, n_gen, obj_kv, obj_mask_np, target_zone, obj_strength, n_steps, cutoff_frac):
        self.n_gen = n_gen; self.obj_kv = obj_kv; self.obj_mask_np = obj_mask_np
        self.target_zone = target_zone; self.obj_strength = obj_strength
        self.n_steps = n_steps; self.cutoff_frac = cutoff_frac
        self._layer = 0; self._step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2); k = k.view(B,-1,attn.heads,hd).transpose(1,2); v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        lo = int(self.cutoff_frac[0]*self.n_steps); hi = int(self.cutoff_frac[1]*self.n_steps)
        if lo <= self._step < hi and self._layer in self.obj_kv:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off, img_off + self.n_gen
            k_gen = k[:,:,g_lo:g_hi,:].clone(); v_gen = v[:,:,g_lo:g_hi,:].clone()
            k_r, v_r = self.obj_kv[self._layer]
            k_r = k_r.to(k.device, k.dtype); v_r = v_r.to(v.device, v.dtype)
            obj_idx = torch.from_numpy(np.where(self.obj_mask_np)[0]).to(k.device)
            if obj_idx.numel() > 0:
                k_obj = k_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(k_gen)
                v_obj = v_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(v_gen)
            else:
                k_obj = k_r.mean(dim=2,keepdim=True).expand_as(k_gen)
                v_obj = v_r.mean(dim=2,keepdim=True).expand_as(v_gen)
            tgt = torch.from_numpy(self.target_zone).to(k.device).view(1,1,-1,1)
            s = self.obj_strength
            k_gen = torch.where(tgt,(1-s)*k_gen+s*k_obj,k_gen)
            v_gen = torch.where(tgt,(1-s)*v_gen+s*v_obj,v_gen)
            k = k.clone(); v = v.clone()
            k[:,:,g_lo:g_hi,:] = k_gen; v[:,:,g_lo:g_hi,:] = v_gen
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


class _DualKVInject:
    """
    Dual K/V injection processor for no-collage object insertion.

      • Background tokens (~target_zone): scene context K/V injected at ALL TIER_A steps.
        Source: reference token slice [r_lo:r_hi] from the live Kontext sequence.
        This is KV-Edit-style background preservation — position-for-position
        replacement keeps the background spatially coherent without BLD latent blending.

      • Object zone tokens (target_zone): obj_img K/V injected during early steps only
        (governed by cutoff_frac). Source: pre-captured via _KVCapture on obj_img.

    No collage, no BLD callback required. Both concerns are handled at attention level.
    """
    def __init__(self, n_gen, obj_kv, obj_mask_np, target_zone,
                 obj_strength, bg_strength, n_steps, cutoff_frac,
                 h_lat: int = 0, w_lat: int = 0):
        self.n_gen = n_gen; self.obj_kv = obj_kv; self.obj_mask_np = obj_mask_np
        self.target_zone = target_zone
        self.obj_strength = obj_strength; self.bg_strength = bg_strength
        self.n_steps = n_steps; self.cutoff_frac = cutoff_frac
        self.h_lat = h_lat; self.w_lat = w_lat
        self._layer = 0; self._step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2)
        k = k.view(B,-1,attn.heads,hd).transpose(1,2)
        v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        if self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off,              img_off + self.n_gen
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                k_gen = k[:,:,g_lo:g_hi,:].clone(); v_gen = v[:,:,g_lo:g_hi,:].clone()
                tgt = torch.from_numpy(self.target_zone).to(k.device)
                # Background: position-for-position scene K/V at every step
                if self.bg_strength > 0:
                    k_ref = k[:,:,r_lo:r_hi,:]; v_ref = v[:,:,r_lo:r_hi,:]
                    bg_t  = (~tgt).view(1,1,-1,1); s_bg = self.bg_strength
                    k_gen = torch.where(bg_t, (1-s_bg)*k_gen + s_bg*k_ref, k_gen)
                    v_gen = torch.where(bg_t, (1-s_bg)*v_gen + s_bg*v_ref, v_gen)
                # Object zone: obj_img K/V during early steps (cutoff_frac gated)
                lo = int(self.cutoff_frac[0]*self.n_steps)
                hi = int(self.cutoff_frac[1]*self.n_steps)
                if lo <= self._step < hi and self._layer in self.obj_kv and self.obj_strength > 0:
                    k_r, v_r = self.obj_kv[self._layer]
                    k_r = k_r.to(k.device, k.dtype); v_r = v_r.to(v.device, v.dtype)
                    # Spatially aligned: each target token gets its geometrically
                    # corresponding obj K/V (bbox-to-bbox), not the mean over all tokens.
                    if self.h_lat > 0 and self.w_lat > 0:
                        k_obj = _spatial_align_k_from_obj(k_r, self.obj_mask_np,
                                                          self.target_zone, self.h_lat, self.w_lat)
                        obj_idx = torch.from_numpy(np.where(self.obj_mask_np)[0]).to(k.device)
                        if obj_idx.numel() > 0:
                            v_obj = _spatial_align_k_from_obj(v_r, self.obj_mask_np,
                                                              self.target_zone, self.h_lat, self.w_lat)
                        else:
                            v_obj = v_r.mean(dim=2, keepdim=True).expand_as(v_gen)
                    else:  # fallback: mean-pool
                        obj_idx = torch.from_numpy(np.where(self.obj_mask_np)[0]).to(k.device)
                        if obj_idx.numel() > 0:
                            k_obj = k_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(k_gen)
                            v_obj = v_r[:,:,obj_idx,:].mean(dim=2,keepdim=True).expand_as(v_gen)
                        else:
                            k_obj = k_r.mean(dim=2,keepdim=True).expand_as(k_gen)
                            v_obj = v_r.mean(dim=2,keepdim=True).expand_as(v_gen)
                    tgt_t = tgt.view(1,1,-1,1); s_obj = self.obj_strength
                    k_gen = torch.where(tgt_t, (1-s_obj)*k_gen + s_obj*k_obj, k_gen)
                    v_gen = torch.where(tgt_t, (1-s_obj)*v_gen + s_obj*v_obj, v_gen)
                k = k.clone(); v = v.clone()
                k[:,:,g_lo:g_hi,:] = k_gen; v[:,:,g_lo:g_hi,:] = v_gen
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


@torch.no_grad()
def run_with_dual_kv_injection(
    pipe, scene, prompt, obj_img, obj_mask_np, target_zone,
    seed, num_steps, guidance, height, width,
    obj_strength=0.5, bg_strength=0.8, cutoff_frac=(0.0, 0.6),
) -> Image.Image:
    """
    No-collage object insertion via dual K/V injection.

    Clean scene is passed as the Kontext reference (unmodified).
    Background preservation and object appearance are both handled
    at the attention level — no pixel-space collage, no BLD latent callback.
    """
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    h_lat  = height // (vae_sf * 2)
    w_lat  = width  // (vae_sf * 2)
    n_gen  = h_lat * w_lat
    orig_procs   = pipe.transformer.attn_processors
    # 1-step pass on obj_img to capture K/V at TIER_A layers
    capture_proc = _KVCapture(n_gen=n_gen)
    pipe.transformer.set_attn_processor(capture_proc)
    gen0 = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(image=_neutralize_white_bg(obj_img),
         prompt="photorealistic object, studio lighting",
         num_inference_steps=1, guidance_scale=1.0,
         height=height, width=width, max_sequence_length=512,
         generator=gen0, output_type="latent")
    obj_kv = dict(capture_proc.kv)
    print(f"    [KV] Captured at {len(obj_kv)}/{len(_TIER_A)} TIER_A layers")
    # Main denoising with dual injection — clean scene as Kontext reference
    inject_proc = _DualKVInject(
        n_gen=n_gen, obj_kv=obj_kv, obj_mask_np=obj_mask_np,
        target_zone=target_zone,
        obj_strength=obj_strength, bg_strength=bg_strength,
        n_steps=num_steps, cutoff_frac=cutoff_frac,
        h_lat=h_lat, w_lat=w_lat,
    )
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pr, si, ts, ck):
        inject_proc._step = si + 1; inject_proc._layer = 0
        return ck

    gen1 = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        image=scene, prompt=prompt,
        num_inference_steps=num_steps, guidance_scale=guidance,
        height=height, width=width, max_sequence_length=512,
        generator=gen1, output_type="pil",
        callback_on_step_end=_step_cb,
        callback_on_step_end_tensor_inputs=[],
    ).images[0]
    pipe.transformer.set_attn_processor(orig_procs)
    return result


@torch.no_grad()
def run_with_kv_injection(
    pipe, canvas, prompt, obj_img, obj_mask_np, target_zone,
    seed, num_steps, guidance, height, width,
    obj_strength=0.7, cutoff_frac=(0.0, 0.6), bcg_callback=None,
) -> Image.Image:
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_gen  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))
    orig_procs   = pipe.transformer.attn_processors
    capture_proc = _KVCapture(n_gen=n_gen)
    pipe.transformer.set_attn_processor(capture_proc)
    gen0 = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe(image=_neutralize_white_bg(obj_img),
         prompt="photorealistic object, studio lighting",
         num_inference_steps=1, guidance_scale=1.0,
         height=height, width=width, max_sequence_length=512,
         generator=gen0, output_type="latent")
    obj_kv = dict(capture_proc.kv)
    print(f"    [KV] Captured at {len(obj_kv)}/{len(_TIER_A)} TIER_A layers")
    inject_proc = _KVInject(n_gen=n_gen, obj_kv=obj_kv, obj_mask_np=obj_mask_np,
                             target_zone=target_zone, obj_strength=obj_strength,
                             n_steps=num_steps, cutoff_frac=cutoff_frac)
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pr, si, ts, ck):
        inject_proc._step = si + 1; inject_proc._layer = 0
        return bcg_callback(pr, si, ts, ck) if bcg_callback else ck

    gen1 = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(image=canvas, prompt=prompt,
                  num_inference_steps=num_steps, guidance_scale=guidance,
                  height=height, width=width, max_sequence_length=512,
                  generator=gen1, output_type="pil",
                  callback_on_step_end=_step_cb,
                  callback_on_step_end_tensor_inputs=["latents"] if bcg_callback else []).images[0]
    pipe.transformer.set_attn_processor(orig_procs)
    return result


class _CollageKVInject:
    def __init__(self, n_gen, target_zone, obj_strength, n_steps, cutoff_frac,
                 target_weights: Optional[np.ndarray] = None):
        self.n_gen = n_gen; self.target_zone = target_zone
        self.obj_strength = obj_strength; self.n_steps = n_steps; self.cutoff_frac = cutoff_frac
        # target_weights: float (n_gen,) in [0,1] from DAAM heatmap — per-token strength.
        # When provided, replaces binary target_zone with a gradient: tokens at the
        # DAAM saliency peak get full obj_strength; tokens at the heatmap periphery
        # get proportionally less, giving a soft spatial transition instead of a hard boundary.
        self.target_weights = target_weights
        self._layer = 0; self._step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        is_double = encoder_hidden_states is not None
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B,-1,attn.heads,hd).transpose(1,2); k = k.view(B,-1,attn.heads,hd).transpose(1,2); v = v.view(B,-1,attn.heads,hd).transpose(1,2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ek = attn.add_k_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            ev = attn.add_v_proj(encoder_hidden_states).view(B,-1,attn.heads,hd).transpose(1,2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]
            q = torch.cat([eq,q],dim=2); k = torch.cat([ek,k],dim=2); v = torch.cat([ev,v],dim=2)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q,image_rotary_emb); k = apply_rotary_emb(k,image_rotary_emb)
        lo = int(self.cutoff_frac[0]*self.n_steps); hi = int(self.cutoff_frac[1]*self.n_steps)
        if lo <= self._step < hi and self._layer in _TIER_A:
            img_off = txt_len if is_double else _SINGLE_TXT_LEN
            g_lo, g_hi = img_off,              img_off + self.n_gen
            r_lo, r_hi = img_off + self.n_gen, img_off + 2 * self.n_gen
            if k.shape[2] >= r_hi:
                k_gen = k[:,:,g_lo:g_hi,:].clone(); v_gen = v[:,:,g_lo:g_hi,:].clone()
                k_ref = k[:,:,r_lo:r_hi,:];         v_ref = v[:,:,r_lo:r_hi,:]
                if self.target_weights is not None:
                    # Soft injection: per-token strength = obj_strength × DAAM saliency weight.
                    # High-saliency tokens (heatmap peak) see full obj_strength;
                    # periphery tokens get proportionally less, no hard edge artifact.
                    w = torch.from_numpy(self.target_weights).to(k.device, k.dtype)
                    s_tok = (self.obj_strength * w).view(1, 1, -1, 1)
                    k_gen = k_gen + s_tok * (k_ref - k_gen)
                    v_gen = v_gen + s_tok * (v_ref - v_gen)
                else:
                    tgt = torch.from_numpy(self.target_zone).to(k.device).view(1,1,-1,1)
                    s = self.obj_strength
                    k_gen = torch.where(tgt,(1.0-s)*k_gen+s*k_ref,k_gen)
                    v_gen = torch.where(tgt,(1.0-s)*v_gen+s*v_ref,v_gen)
                k = k.clone(); v = v.clone()
                k[:,:,g_lo:g_hi,:] = k_gen; v[:,:,g_lo:g_hi,:] = v_gen
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1,2).reshape(B,-1,attn.heads*hd).to(q.dtype)
        if is_double:
            enc_out = out[:,:encoder_hidden_states.shape[1]]
            out = out[:,encoder_hidden_states.shape[1]:]
            out = attn.to_out[1](attn.to_out[0](out)); enc_out = attn.to_add_out(enc_out)
            self._layer += 1; return out, enc_out
        self._layer += 1; return out


@torch.no_grad()
def run_with_collage_kv_injection(
    pipe, canvas, prompt, target_zone,
    seed, num_steps, guidance, height, width,
    obj_strength=0.5, cutoff_frac=(0.0, 0.6), bcg_callback=None,
    target_weights: Optional[np.ndarray] = None,
) -> Image.Image:
    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    n_gen  = (height // (vae_sf * 2)) * (width // (vae_sf * 2))
    orig_procs  = pipe.transformer.attn_processors
    inject_proc = _CollageKVInject(n_gen=n_gen, target_zone=target_zone,
                                   obj_strength=obj_strength, n_steps=num_steps,
                                   cutoff_frac=cutoff_frac, target_weights=target_weights)
    pipe.transformer.set_attn_processor(inject_proc)

    def _step_cb(pr, si, ts, ck):
        inject_proc._step = si + 1; inject_proc._layer = 0
        return bcg_callback(pr, si, ts, ck) if bcg_callback else ck

    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(image=canvas, prompt=prompt,
                  num_inference_steps=num_steps, guidance_scale=guidance,
                  height=height, width=width, max_sequence_length=512,
                  generator=gen, output_type="pil",
                  callback_on_step_end=_step_cb,
                  callback_on_step_end_tensor_inputs=["latents"] if bcg_callback else []).images[0]
    pipe.transformer.set_attn_processor(orig_procs)
    return result


# ── Sobel edge extraction ─────────────────────────────────────────────────────

def _sobel_map(img: np.ndarray, mask: np.ndarray, thresh: int = 30) -> np.ndarray:
    H, W    = img.shape[:2]
    small   = cv2.resize(img, (256, 256))
    mask_s  = (cv2.resize(mask.astype(np.uint8), (256, 256)) > 0.5).astype(np.uint8)
    eroded  = cv2.erode(mask_s, np.ones((5, 5), np.uint8), iterations=2)
    mask_s  = eroded if eroded.any() else mask_s
    sx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5, cv2.convertScaleAbs(sy), 0.5, 0)
    mag = np.max(mag, axis=-1) * mask_s; mag[mag < thresh] = 0.0
    mag3 = np.stack([mag, mag, mag], axis=-1)
    edge = (mag3.astype(np.float32) / 255.0 * small.astype(np.float32)).astype(np.uint8)
    return cv2.resize(edge, (W, H))


# ── Collage builder ───────────────────────────────────────────────────────────

def build_collage_scene(
    scene:        Image.Image,
    obj_img:      Image.Image,
    ref_mask:     np.ndarray,
    target_bbox:  Tuple[int, int, int, int],
    collage_mode: str = "full",
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Paste obj_img into scene at target_bbox.

    This is where the two-image fusion happens at the diffusion level:
    FLUX Kontext receives the collage (obj already placed in scene) as its
    reference image, so it sees both obj_img and scene in a single prior.
    """
    scene_np = np.array(scene.convert("RGB")).astype(np.float32)
    obj_np   = np.array(obj_img.convert("RGB")).astype(np.float32)
    H, W     = scene_np.shape[:2]

    ys, xs = np.where(ref_mask > 0)
    if len(ys) == 0:
        print("    [COLLAGE] Warning: empty ref_mask, using centre crop")
        oh, ow  = obj_np.shape[:2]
        oy1, oy2, ox1, ox2 = oh//4, 3*oh//4, ow//4, 3*ow//4
    else:
        oy1, oy2 = int(ys.min()), int(ys.max()) + 1
        ox1, ox2 = int(xs.min()), int(xs.max()) + 1

    obj_crop  = obj_np[oy1:oy2, ox1:ox2]
    mask_crop = ref_mask[oy1:oy2, ox1:ox2]

    x1, y1, x2, y2 = target_bbox
    zone_h, zone_w  = y2 - y1, x2 - x1
    print(f"    [COLLAGE] Target zone: y=[{y1},{y2}] x=[{x1},{x2}]  {zone_w}×{zone_h} px")

    if collage_mode == "sobel":
        obj_z  = cv2.resize(obj_crop.astype(np.uint8),  (zone_w, zone_h))
        mask_z = cv2.resize(mask_crop.astype(np.uint8), (zone_w, zone_h))
        sobel_zone = _sobel_map(obj_z, mask_z)
        collage_np = scene_np.copy().astype(np.uint8)
        collage_np[y1:y2, x1:x2] = sobel_zone
        return Image.fromarray(collage_np), (y1, y2, x1, x2)

    oh, ow  = obj_crop.shape[:2]
    scale   = min((zone_h * 0.80) / oh, (zone_w * 0.80) / ow)
    new_h   = max(4, int(oh * scale))
    new_w   = max(4, int(ow * scale))
    obj_rs  = cv2.resize(obj_crop.astype(np.uint8),  (new_w, new_h)).astype(np.float32)
    mask_rs = (cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h)) > 0.5).astype(np.float32)

    if collage_mode == "blend":
        sobel_layer = _sobel_map(obj_rs.astype(np.uint8), mask_rs).astype(np.float32)
        paste_layer = 0.6 * obj_rs + 0.4 * sobel_layer
    else:
        paste_layer = obj_rs

    mask_3 = np.stack([mask_rs] * 3, axis=-1)
    dy = (zone_h - new_h) // 2
    dx = (zone_w - new_w) // 2
    py1, px1 = max(0, y1+dy), max(0, x1+dx)
    py2, px2 = min(H, py1+new_h), min(W, px1+new_w)
    ch, cw = py2 - py1, px2 - px1
    if ch <= 0 or cw <= 0:
        print("    [COLLAGE] Warning: paste region zero — returning scene unchanged")
        return scene, (y1, y2, x1, x2)

    paste_layer = paste_layer[:ch, :cw]; mask_3 = mask_3[:ch, :cw]
    collage_np  = scene_np.copy()
    collage_np[py1:py2, px1:px2] = paste_layer * mask_3 + collage_np[py1:py2, px1:px2] * (1.0 - mask_3)
    collage_np  = np.clip(collage_np, 0, 255).astype(np.uint8)
    print(f"    [COLLAGE] Pasted {new_w}×{new_h} at [{py1}:{py2},{px1}:{px2}]  mode={collage_mode}")
    return Image.fromarray(collage_np), (y1, y2, x1, x2)


def build_removal_collage(
    scene:     Image.Image,
    mask_np:   np.ndarray,
    height:    int,
    width:     int,
    dilate_px: int = 8,
) -> Image.Image:
    """Inpaint the object region (supplied as uint8 mask array) using Telea."""
    scene_np     = np.array(scene.convert("RGB").resize((width, height), Image.LANCZOS))
    inpaint_mask = (mask_np > 127).astype(np.uint8)
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        inpaint_mask = cv2.dilate(inpaint_mask, np.ones((k, k), np.uint8), iterations=1)
    inpainted = cv2.inpaint(scene_np, inpaint_mask, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
    print(f"    [REMOVAL] Telea inpaint: {int(inpaint_mask.sum())} px erased")
    return Image.fromarray(inpainted)


def _collage_obj_mask(collage: Image.Image, scene: Image.Image,
                       threshold: int = 10) -> np.ndarray:
    diff = np.abs(
        np.array(collage.convert("RGB"), dtype=np.int32) -
        np.array(scene.convert("RGB"),   dtype=np.int32)
    ).max(axis=-1)
    mask = (diff > threshold).astype(np.uint8)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask


# ── BLD latent BCG (takes mask_np, not mask_path) ────────────────────────────

def _make_bcg_latent_callback(
    scene:        Image.Image,
    mask_np:      np.ndarray,      # uint8 (H,W) — 255 inside edit zone
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
    mask_np is the auto-generated rectangle mask at pixel resolution.
    """
    vae_h = height // 8; vae_w = width // 8

    scene_np = np.array(scene.convert("RGB"), dtype=np.float32) / 255.0
    scene_t  = torch.from_numpy(scene_np).permute(2, 0, 1).unsqueeze(0)
    scene_t  = (scene_t * 2.0 - 1.0).to(dtype)
    with torch.no_grad():
        enc_dev = next(pipe.vae.parameters()).device
        z0 = pipe.vae.encode(scene_t.to(enc_dev)).latent_dist.sample()
        z0 = (z0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z0    = z0.to(device, dtype)
    noise = torch.randn(z0.shape, device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(42))

    mask_lat = np.array(
        Image.fromarray(mask_np).resize((vae_w, vae_h), Image.NEAREST)
    )
    mask_lat = (mask_lat > 127).astype(np.uint8)
    if dilation_lat > 0:
        k = dilation_lat * 2 + 1
        mask_lat = cv2.dilate(mask_lat, np.ones((k, k), np.uint8), iterations=2)
    mask_soft   = cv2.GaussianBlur(mask_lat.astype(np.float32), (7, 7), 2.0)
    mask_tensor = torch.from_numpy(mask_soft).to(device, dtype)[None, None]
    print(f"    [BLD] Latent mask {vae_h}×{vae_w}, edit zone={mask_soft.mean()*100:.1f}%")

    def _callback(_pipeline, _step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if latents.shape[-2:] != (vae_h, vae_w):
            return callback_kwargs
        sigma = max(0.0, min(1.0, float(timestep) / 1000.0))
        z_bg  = ((1.0-sigma)*z0 + sigma*noise).to(latents.dtype).to(latents.device)
        m     = mask_tensor.expand_as(latents).to(latents.dtype).to(latents.device)
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
    depth_model,
    depth_processor,
    seed:           int,
    num_steps:      int,
    lora_guidance:  float,
    scene_guidance: float,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
    use_kv:         bool  = True,
    obj_strength:   float = 0.5,
    bg_strength:    float = 0.8,
    cutoff_frac:    Tuple[float, float] = (0.0, 0.6),
    heatmap_steps:  int   = 6,
    vlm_model                          = None,
    vlm_processor                      = None,
) -> List[Image.Image]:
    """
    Mask-free incremental object insertion / removal.

    Per insert:
      Stage A    : Sketch → LoRA FLUX → obj_img
      Stage DEPTH: estimate_depth(scene) → find_floor_bbox() → mask_np, target_zone
      Stage K    : FLUX Kontext(scene, prompt) with _DualKVInject
                   — no collage, no BLD callback.
      Stage BCG  : pixel-space restore from result-vs-scene diff mask

    Per remove:
      Stage DEPTH: use tracked insertion bbox if available,
                   else detect_object_on_floor(scene depth) → mask_np
      Stage COL  : Telea inpaint → removal collage
      Stage K    : FLUX Kontext(removal_collage, prompt) + BLD BCG

    Insertion bboxes are tracked in _placed dict so removal can reuse them
    without needing a second detection step.
    """
    results = [base]
    scene   = base
    _placed: Dict[str, Tuple[int, int, int, int]] = {}
    per_object_masks: Dict[str, np.ndarray] = {}   # pixel-wise mask per inserted object

    for i, edit in enumerate(edits):
        name   = edit["name"]
        desc   = edit["description"]
        action = edit.get("action", "insert")

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}  [{action}]")
        print(f"{'─'*60}")

        # ── Stage DEPTH: estimate scene depth ────────────────────────────────
        print(f"  [DEPTH] Estimating scene depth ...")
        depth_np = estimate_depth(depth_model, depth_processor, scene)
        if depth_np is not None:
            # Save depth visualisation for inspection
            depth_vis = (depth_np * 255).astype(np.uint8)
            Image.fromarray(depth_vis).save(
                os.path.join(out_dir, f"depth_{i+1}_{name}_{action}.png")
            )

        if action == "insert":
            # Stage A: sketch → object image
            sketch_path = os.path.join(sketch_dir, f"{name}.png")
            if not os.path.isfile(sketch_path):
                sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
            if not os.path.isfile(sketch_path):
                raise FileNotFoundError(
                    f"Sketch not found: expected '{name}.png' or 'sketch_{name}.png' "
                    f"in {sketch_dir!r}"
                )
            print(f"  [A] Generating '{desc}' from sketch ...")
            obj_img = generate_from_sketch(
                pipe=pipe, sketch_path=sketch_path, description=desc,
                seed=seed, num_steps=num_steps, guidance=lora_guidance,
                height=height, width=width, lora_id=lora_id, device=device,
            )
            obj_img.save(os.path.join(out_dir, f"obj_gen_{name}.png"))

            blend_p = f"A photorealistic room with a {desc} placed naturally."

            # VLM predicts scene-aware placement anchor (e.g. 'on the kitchen counter')
            placement_anchor: Optional[str] = None
            if vlm_model is not None:
                print(f"  [VLM] Predicting placement for '{desc}' ...")
                placement_anchor = _vlm_predict_placement(
                    vlm_model, vlm_processor, scene, desc
                )

            # Build pixel-wise occupied map from all previously placed object masks.
            # Each object's exact pixel footprint is suppressed, so the new object's
            # heatmap is free to overlap objects at different depths (e.g. vase in
            # front of bicycle) while still avoiding placing two objects at the exact
            # same location.
            _occ = np.zeros((height, width), dtype=np.uint8)
            for _m in per_object_masks.values():
                _occ = np.clip(
                    _occ.astype(np.int32) + _m.astype(np.int32) * 255, 0, 255
                ).astype(np.uint8)

            # Heatmap-guided placement (DAAM)
            print(f"  [PLACE] Attention heatmap for '{desc}' ...")
            heatmap = run_heatmap_pass(
                pipe=pipe, scene=scene, prompt=blend_p, description=desc,
                seed=seed, height=height, width=width, n_steps=heatmap_steps,
                occupied_mask=_occ, anchor=placement_anchor,
            )
            hm_vis = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            Image.fromarray(cv2.cvtColor(hm_vis, cv2.COLOR_BGR2RGB)).save(
                os.path.join(out_dir, f"heatmap_{name}.png")
            )
            # Soft per-token K/V weights directly from DAAM heatmap (already [0,1]).
            # Bypasses the binary rotated-rectangle step for K/V injection — the
            # saliency gradient is used as-is so periphery tokens blend gently.
            _heatmap_weights = heatmap.reshape(-1).astype(np.float32)
            mask_np = heatmap_to_mask(heatmap, height, width)  # still needed for BLD + bbox
            Image.fromarray(mask_np).save(os.path.join(out_dir, f"mask_pred_{name}.png"))

            # Derive bbox from mask for removal fallback tracking
            ys, xs = np.where(mask_np > 127)
            if len(ys) > 0:
                bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            else:
                bx1, by1 = 0, height // 4
                bx2, by2 = width, 3 * height // 4
            _placed[name] = (bx1, by1, bx2, by2)
            with open(os.path.join(out_dir, f"bbox_{name}_{action}.txt"), "w") as f:
                f.write(f"bbox: x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

            # Stage MASK: obj silhouette to crop the paste region
            ref_mask = _compute_obj_mask(obj_img)
            if ref_mask.sum() < 50:
                print("         Fallback: centre-crop silhouette")
                ref_mask = np.zeros((height, width), dtype=np.uint8)
                ref_mask[height//4:3*height//4, width//4:3*width//4] = 1

            # Stage COL: paste obj at heatmap-guided bbox → Kontext reference
            print(f"  [COL] Building collage (heatmap-guided) ...")
            collage_scene, _ = build_collage_scene(
                scene=scene, obj_img=obj_img, ref_mask=ref_mask,
                target_bbox=(bx1, by1, bx2, by2), collage_mode="full",
            )
            collage_scene.save(os.path.join(out_dir, f"collage_{name}.png"))
            # Collage diff gives obj mask BEFORE generation — more accurate than result diff
            obj_mask_har = _collage_obj_mask(collage_scene, scene)

        else:  # remove
            # Use tracked insertion bbox if available (most reliable)
            if name in _placed:
                bx1, by1, bx2, by2 = _placed[name]
                print(f"  [PLACE] Reusing tracked insertion bbox: "
                      f"x=[{bx1},{bx2}] y=[{by1},{by2}]")
            else:
                # Depth-anomaly detection for pre-existing objects
                print(f"  [DETECT] Depth-anomaly object detection ...")
                bx1, by1, bx2, by2 = detect_object_on_floor(depth_np, height, width)

            # Prefer pixel-wise tracked mask over bbox rectangle
            if name in per_object_masks:
                mask_np = (per_object_masks[name] * 255).astype(np.uint8)
                print(f"  [PLACE] Using pixel-wise mask for removal of '{name}'")
            else:
                mask_np = _rect_mask_from_bbox(bx1, by1, bx2, by2, height, width)
            Image.fromarray(mask_np).save(
                os.path.join(out_dir, f"mask_pred_remove_{name}.png")
            )
            with open(os.path.join(out_dir, f"bbox_{name}_{action}.txt"), "w") as f:
                f.write(f"bbox: x1={bx1} y1={by1} x2={bx2} y2={by2}\n")

            print(f"  [COL] Building removal collage (Telea inpaint) ...")
            collage_scene = build_removal_collage(
                scene=scene, mask_np=mask_np, height=height, width=width,
            )
            collage_scene.save(os.path.join(out_dir, f"collage_remove_{name}.png"))

            obj_mask_har = (mask_np > 127).astype(np.uint8)
            blend_p = (
                f"{BASE_PROMPT} "
                f"The {desc} has been removed. "
                f"Seamless wooden floor and white walls fill the area. "
                f"No object, clean empty room."
            )
            # Remove from tracking once erased
            _placed.pop(name, None)
            per_object_masks.pop(name, None)

        # ── Stage K: Kontext integration ─────────────────────────────────────
        step_tag = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"
        print(f"  [K] Kontext integration pass  kv={use_kv}  action={action} ...")
        with open(os.path.join(out_dir, f"prompt_{step_tag}.txt"), "w") as f:
            f.write(blend_p)

        if action == "insert":
            # BLD callback: at every denoising step, pixels OUTSIDE the placement
            # mask are overwritten with the original scene latents (noised to the
            # current timestep).  This guarantees pixel-perfect background
            # preservation by construction — no separate pixel-space BCG restore needed.
            pipe_dtype = next(pipe.transformer.parameters()).dtype
            bcg_cb = _make_bcg_latent_callback(
                scene=scene, mask_np=mask_np, pipe=pipe,
                height=height, width=width, device=device, dtype=pipe_dtype,
            )
            if use_kv:
                target_zone = _token_zone_from_mask_np(mask_np, height, width, pipe)
                next_scene  = run_with_collage_kv_injection(
                    pipe=pipe, canvas=collage_scene, prompt=blend_p,
                    target_zone=target_zone,
                    seed=seed, num_steps=num_steps, guidance=scene_guidance,
                    height=height, width=width,
                    obj_strength=obj_strength, cutoff_frac=cutoff_frac,
                    bcg_callback=bcg_cb,
                    target_weights=_heatmap_weights,
                )
            else:
                next_scene = run_standard(
                    pipe=pipe, canvas=collage_scene, prompt=blend_p,
                    seed=seed, num_steps=num_steps, guidance=scene_guidance,
                    height=height, width=width,
                    bcg_callback=bcg_cb,
                )

        else:  # remove — keep Telea collage + BLD
            pipe_dtype = next(pipe.transformer.parameters()).dtype
            bcg_cb = _make_bcg_latent_callback(
                scene=scene, mask_np=mask_np, pipe=pipe,
                height=height, width=width, device=device, dtype=pipe_dtype,
            )
            if use_kv:
                target_zone = _token_zone_from_mask_np(mask_np, height, width, pipe)
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

        # ── Pixel-wise object mask (insertion only) ───────────────────────────
        # BLD kept the background pixel-identical to the pre-insertion scene, so
        # a simple diff cleanly isolates the generated object pixels.
        if action == "insert":
            _r   = np.array(next_scene.convert("RGB"), dtype=np.int32)
            _s   = np.array(scene.convert("RGB"),      dtype=np.int32)
            _diff = np.abs(_r - _s).max(axis=2)
            _px  = (_diff > 15).astype(np.uint8)
            _px  = cv2.morphologyEx(_px, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            _px  = cv2.morphologyEx(_px, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
            per_object_masks[name] = _px
            print(f"    [MASK] '{name}' pixel mask: {int(_px.sum())} px "
                  f"({_px.mean()*100:.1f}% of frame)")

        # Save color-coded instance mask overlay on top of the current result
        _COLORS = [(0,220,0),(0,80,255),(255,60,0),(255,200,0),(0,220,220),(200,0,255)]
        _overlay = np.array(next_scene.convert("RGB"), dtype=np.float32)
        for _ci, (_nm, _m) in enumerate(per_object_masks.items()):
            _c = np.array(_COLORS[_ci % len(_COLORS)], dtype=np.float32)
            _overlay[_m > 0] = 0.45 * _overlay[_m > 0] + 0.55 * _c
        Image.fromarray(np.clip(_overlay, 0, 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"masks_overlay_{step_tag}.png")
        )

        # ── Stage BCG: pixel-space background restoration (removal only) ────────
        # Insertion: BLD callback already preserved the background during denoising
        #            — no pixel-space restore needed.
        # Removal  : BLD runs inside the mask region; pixel-space restore cleans up
        #            any boundary bleed from the Telea inpaint collage.
        if action == "remove":
            print(f"  [BCG] Background restore (removal) ...")
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
        description="AnyDoor collage + FLUX Kontext — mask-free via depth estimation."
    )
    p.add_argument("--sketch_dir",    required=True)
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--cache_dir",     default="./models")
    p.add_argument("--out_dir",       default="results/phase1_collage_nomask")
    p.add_argument("--depth_model",
                   default="depth-anything/Depth-Anything-V2-Small-hf",
                   help="HF model id for monocular depth estimation. "
                        "Default: Depth-Anything-V2-Small (~50 MB). "
                        "Falls back to Intel/dpt-large, then geometric heuristic.")
    p.add_argument("--config",        default=None,
                   help="JSON list of {name, description, action}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",       default=LORA_ID)
    p.add_argument("--lora_guidance", type=float, default=4.0)
    p.add_argument("--guidance",      type=float, default=2.5)
    p.add_argument("--heatmap_steps", type=int, default=6,
                   help="Steps for the DAAM attention heatmap pre-pass. Default: 6")
    p.add_argument("--kv_injection",  action="store_true",
                   help="Use dual K/V injection for insertion (default: off for baseline test).")
    p.add_argument("--obj_strength",  type=float, default=0.5,
                   help="K/V injection strength for object zone tokens. Default: 0.5")
    p.add_argument("--bg_strength",   type=float, default=0.8,
                   help="K/V injection strength for background tokens (KV-Edit). Default: 0.8")
    p.add_argument("--cutoff_frac",   default="0.0,0.6")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_steps",     type=int,   default=28)
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--vlm_model",     default=None,
                   help="HF model id for VLM-based placement prediction. "
                        "Optional — if omitted, description tokens alone guide the heatmap. "
                        "Tested: Qwen/Qwen2-VL-2B-Instruct (CPU, ~4 GB RAM), "
                        "Qwen/Qwen2-VL-7B-Instruct (GPU, better reasoning).")
    p.add_argument("--vlm_device",    default="cpu",
                   help="Device for VLM inference. Default: cpu (runs alongside FLUX on GPU).")
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
    print(f"  phase1_collage_kontext_without_mask")
    print(f"{_SEP}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  Depth model  : {args.depth_model}")
    print(f"  VLM model    : {args.vlm_model or 'none (description tokens only)'}")
    print(f"  KV injection : {args.kv_injection}"
          + (f"  obj={args.obj_strength}  bg={args.bg_strength}  cutoff={args.cutoff_frac}"
             if args.kv_injection else ""))
    print(f"  Guidance     : {args.guidance}")
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    print("Loading depth model ...")
    depth_model, depth_processor, _ = load_depth_model(
        model_id=args.depth_model, cache_dir=args.cache_dir,
    )

    vlm_model = vlm_processor = None
    if args.vlm_model:
        vlm_model, vlm_processor = load_vlm(
            args.vlm_model, cache_dir=args.cache_dir, device=args.vlm_device,
        )

    print("\nLoading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

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
        depth_model=depth_model, depth_processor=depth_processor,
        seed=args.seed, num_steps=args.num_steps,
        lora_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
        use_kv=args.kv_injection,
        obj_strength=args.obj_strength,
        bg_strength=args.bg_strength,
        cutoff_frac=cutoff_frac,
        heatmap_steps=args.heatmap_steps,
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
    )

    all_imgs = results
    all_lbls = ["base"] + [f"{e['name']} ({e.get('action','insert')})" for e in edits]
    save_grid(all_imgs, all_lbls,
              os.path.join(args.out_dir, "chain_grid.png"),
              ncols=len(all_imgs))
    print(f"\n{_SEP}")
    print(f"  Chain complete. Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
