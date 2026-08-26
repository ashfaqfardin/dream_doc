"""
E11: Scene-First Placement Planner -> Pose Proposal -> Identity Finalizer
=======================================================================

Motivation
----------
E9/E10 still let the object reference influence placement too early. If the
model has a centre bias, the proposal-derived mask can reinforce that bias.

E11 changes the causal order:

  PASS 1 — scene-first placement planning (NO object reference image)
      current scene + object NAME/TYPE only
          -> several generic placement proposals
          -> derive placement mask from proposal-vs-scene difference
          -> score alternatives and select the best placement region

  PASS 2 — pose proposal (current scene + NEW object reference)
      current scene + selected placement region + object reference
          -> propose how the real object should orient / scale / view itself
             to fit that region
          -> save a "proposed" image (not yet final)

  PASS 3 — identity-preserving finalisation
      current clean scene + proposed image + all references so far
          -> render the final object cleanly using the proposal as a
             placement/pose guide and the references for identity
          -> keep changes only inside the selected soft mask

Key idea
--------
Placement is decided from the scene BEFORE the reference image is introduced.
That separates:
  - WHERE should the object go?   (scene planner)
  - HOW should it pose there?     (pose proposal)
  - WHAT exact object is it?      (identity finaliser)

Assumptions / project utilities
-------------------------------
This file relies on your project's existing utils.py functions:
    load_pipe, enable_multi_context,
    compute_ssim, compute_lpips, compute_dino, compute_clip_i

The multi-context contract is assumed to be:
    image = [scene_context, extra_context_1, ..., extra_context_N]
with image hidden-state token layout:
    [target/noisy | scene_context | extra_1 | ... | extra_N]

Important note
--------------
Because the planner stage intentionally does NOT use the true object reference,
it cannot use object-specific reference attention for localisation. Instead, E11
uses scene-first multi-hypothesis planning + automatic difference masks.
The finaliser still uses reference-aware K/V amplification to preserve identity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

# -----------------------------------------------------------------------------
# Existing project utilities
# -----------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
from utils import (  # noqa: E402
    load_pipe,
    enable_multi_context,
    compute_ssim,
    compute_lpips,
    compute_dino,
    compute_clip_i,
)

try:
    from diffusers.models.attention_dispatch import dispatch_attention_fn
except Exception:
    dispatch_attention_fn = None

try:
    from diffusers.models.embeddings import apply_rotary_emb
except Exception as exc:
    raise ImportError(
        "Could not import diffusers.models.embeddings.apply_rotary_emb. "
        "Use a Diffusers build compatible with FLUX.1-Kontext."
    ) from exc


OBJ_ORDER = ["bicycle", "vase", "ball", "chair", "lamp", "plant", "backpack"]
DEFAULT_IDENTITY_BLOCKS = "13-18"


# =============================================================================
# Generic helpers
# =============================================================================


def parse_int_ranges(spec: str) -> List[int]:
    out = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)



def parse_seeds(spec: str) -> List[int]:
    values = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values



def packed_token_count(height: int, width: int, vae_scale_factor: int) -> int:
    mult = int(vae_scale_factor) * 2
    if height % mult != 0 or width % mult != 0:
        raise ValueError(
            f"Output dimensions must be divisible by {mult}; got {width}x{height}."
        )
    return (height // mult) * (width // mult)



def infer_grid_from_tokens(n_tokens: int, aspect_ratio: float) -> Tuple[int, int]:
    """Return (grid_h, grid_w) using the factor pair closest to w/h ratio."""
    if n_tokens <= 0:
        raise ValueError(f"Invalid token count {n_tokens}.")
    aspect_ratio = max(float(aspect_ratio), 1e-8)
    best = None
    root = int(math.sqrt(n_tokens))
    for h in range(1, root + 1):
        if n_tokens % h:
            continue
        w = n_tokens // h
        for gh, gw in ((h, w), (w, h)):
            err = abs(math.log(max(gw / gh, 1e-8)) - math.log(aspect_ratio))
            if best is None or err < best[0]:
                best = (err, gh, gw)
    if best is None:
        raise RuntimeError(f"Cannot infer a 2-D token grid for {n_tokens} tokens.")
    return int(best[1]), int(best[2])



def robust_normalize(x: np.ndarray, lo_q: float = 0.05, hi_q: float = 0.95) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    lo = float(np.quantile(vals, lo_q))
    hi = float(np.quantile(vals, hi_q))
    if hi <= lo + 1e-8:
        hi = lo + 1e-8
    y = (x - lo) / (hi - lo)
    y[~finite] = 0.0
    return np.clip(y, 0.0, 1.0)



def save_float_map(x: np.ndarray, path: str, out_size: Optional[Tuple[int, int]] = None):
    y = np.uint8(np.clip(robust_normalize(x), 0, 1) * 255)
    img = Image.fromarray(y, mode="L")
    if out_size is not None:
        img = img.resize(out_size, Image.Resampling.BILINEAR)
    img.save(path)



def union_masks(masks: Sequence[Image.Image], size: Tuple[int, int]) -> Optional[Image.Image]:
    if not masks:
        return None
    arr = np.zeros((size[1], size[0]), dtype=np.float32)
    for m in masks:
        cur = np.asarray(m.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        arr = np.maximum(arr, cur)
    return Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), mode="L")



def neutralize_edit_region(img: Image.Image, mask: Image.Image, value: int = 128) -> Image.Image:
    rgb = img.convert("RGB")
    m = mask.convert("L").resize(rgb.size, Image.Resampling.BILINEAR)
    neutral = Image.new("RGB", rgb.size, (value, value, value))
    return Image.composite(neutral, rgb, m)



def mask_bbox(mask: Image.Image, threshold: int = 16) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(mask.convert("L"))
    ys, xs = np.where(arr > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1



def expand_bbox(box: Tuple[int, int, int, int], size: Tuple[int, int], frac: float = 0.08):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    dx, dy = int(round(bw * frac)), int(round(bh * frac))
    W, H = size
    return max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy)



def position_phrase(mask: Image.Image) -> Optional[str]:
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    total = float(arr.sum())
    if total <= 1e-6:
        return None
    ys, xs = np.indices(arr.shape)
    cx = float((xs * arr).sum() / total) / max(arr.shape[1] - 1, 1)
    cy = float((ys * arr).sum() / total) / max(arr.shape[0] - 1, 1)
    horiz = "left" if cx < 0.34 else ("right" if cx > 0.66 else "central")
    vert = "upper" if cy < 0.34 else ("lower" if cy > 0.66 else "middle")
    if horiz == "central" and vert == "middle":
        return "central area"
    if horiz == "central":
        return f"{vert}-central area"
    if vert == "middle":
        return f"middle-{horiz} area"
    return f"{vert}-{horiz} area"



def protect_outside_soft_mask(previous: Image.Image, generated: Image.Image, soft_mask: Image.Image) -> Image.Image:
    prev = previous.convert("RGB")
    gen = generated.convert("RGB").resize(prev.size, Image.Resampling.LANCZOS)
    mask = soft_mask.convert("L").resize(prev.size, Image.Resampling.BILINEAR)
    return Image.composite(gen, prev, mask)



def make_overlay(scene: Image.Image, mask: Image.Image, strength: float = 0.45) -> Image.Image:
    base = np.asarray(scene.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L").resize(scene.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tint = base.copy()
    tint[..., 0] = 255.0
    alpha = (m * float(strength))[..., None]
    out = base * (1.0 - alpha) + tint * alpha
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)), mode="RGB")



def gaussian_blur_array(x: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(np.uint8(np.clip(robust_normalize(x), 0, 1) * 255), mode="L")
    if radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(img, dtype=np.float32) / 255.0



def morph_close(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary.astype(bool)
    kernel = 2 * int(radius) + 1
    img = Image.fromarray(np.uint8(binary) * 255, mode="L")
    img = img.filter(ImageFilter.MaxFilter(size=kernel))
    img = img.filter(ImageFilter.MinFilter(size=kernel))
    return np.asarray(img) > 127



def dilate_binary(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary.astype(bool)
    kernel = 2 * int(radius) + 1
    img = Image.fromarray(np.uint8(binary) * 255, mode="L")
    img = img.filter(ImageFilter.MaxFilter(size=kernel))
    return np.asarray(img) > 127



def best_connected_component(binary: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Keep 8-connected component with largest summed score."""
    H, W = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    best_pixels = []
    best_score = -float("inf")
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0), (1, 1),
    ]

    for y in range(H):
        for x in range(W):
            if not binary[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            pixels = []
            ssum = 0.0
            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                ssum += float(score[cy, cx])
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if ssum > best_score:
                best_score = ssum
                best_pixels = pixels

    out = np.zeros_like(binary, dtype=bool)
    for y, x in best_pixels:
        out[y, x] = True
    return out


# =============================================================================
# Reference foreground masks
# =============================================================================


def extract_reference_foreground_mask(
    obj_img: Image.Image,
    grey_tol: int = 20,
    white_tol: int = 20,
    alpha_threshold: int = 8,
    min_frac: float = 0.02,
    max_frac: float = 0.95,
    dilate_px: int = 1,
    allow_fallback: bool = False,
) -> Image.Image:
    rgba = obj_img.convert("RGBA")
    arr = np.asarray(rgba)
    alpha = arr[..., 3]

    if alpha.min() < 250 and np.mean(alpha < 250) > 0.001:
        mask = alpha > alpha_threshold
        source = "alpha"
    else:
        rgb = arr[..., :3].astype(np.int16)
        is_grey = (
            (np.abs(rgb[..., 0] - 128) <= grey_tol)
            & (np.abs(rgb[..., 1] - 128) <= grey_tol)
            & (np.abs(rgb[..., 2] - 128) <= grey_tol)
        )
        is_white = rgb.min(axis=2) >= (255 - white_tol)
        mask = ~(is_grey | is_white)
        source = "grey/white heuristic"

    frac = float(mask.mean())
    if frac < min_frac or frac > max_frac:
        msg = (
            f"Suspicious foreground mask ({source}): {frac:.2%}; "
            f"expected [{min_frac:.1%}, {max_frac:.1%}]."
        )
        if not allow_fallback:
            raise ValueError(msg + " Fix object alpha/background or pass --allow_mask_fallback.")
        warnings.warn(msg + " Using centre-50% fallback.")
        mask[:] = False
        H, W = mask.shape
        mask[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = True

    out = Image.fromarray(np.uint8(mask) * 255, mode="L")
    if dilate_px > 0:
        out = out.filter(ImageFilter.MaxFilter(size=2 * int(dilate_px) + 1))
    return out


class ReferenceMaskBank:
    """Cache token masks for object references."""

    def __init__(self, masks: Sequence[Image.Image], aspect_ratios: Sequence[float]):
        if len(masks) != len(aspect_ratios):
            raise ValueError("masks/aspect_ratios length mismatch")
        self.masks = [m.convert("L") for m in masks]
        self.aspect_ratios = [float(x) for x in aspect_ratios]
        self._cpu: Dict[int, List[torch.Tensor]] = {}
        self._device: Dict[Tuple[int, str], List[torch.Tensor]] = {}

    def _build_cpu(self, n_ctx: int) -> List[torch.Tensor]:
        if n_ctx in self._cpu:
            return self._cpu[n_ctx]
        result = []
        grids = []
        for mask, ar in zip(self.masks, self.aspect_ratios):
            gh, gw = infer_grid_from_tokens(n_ctx, ar)
            arr = np.asarray(mask.resize((gw, gh), Image.Resampling.NEAREST)) > 127
            result.append(torch.from_numpy(arr.reshape(-1).copy()).bool())
            grids.append((gh, gw))
        self._cpu[n_ctx] = result
        print(f"  [mask-bank] n_ctx={n_ctx}, inferred reference grids={grids}")
        return result

    def get(self, n_ctx: int, device: torch.device) -> List[torch.Tensor]:
        key = (int(n_ctx), str(device))
        if key not in self._device:
            self._device[key] = [m.to(device) for m in self._build_cpu(n_ctx)]
        return self._device[key]


# =============================================================================
# FLUX attention helpers and final identity controller
# =============================================================================


def _project_qkv(attn, hidden_states, encoder_hidden_states=None):
    if getattr(attn, "fused_projections", False):
        if not hasattr(attn, "to_qkv"):
            raise RuntimeError("fused_projections=True but attention has no to_qkv")
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        eq = ek = ev = None
        if encoder_hidden_states is not None:
            if not hasattr(attn, "to_added_qkv"):
                raise RuntimeError("Fused dual-stream FLUX attention has no to_added_qkv")
            eq, ek, ev = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        eq = ek = ev = None
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
    return query, key, value, eq, ek, ev



def _dispatch_attention(query, key, value, attention_mask, backend=None, parallel_config=None):
    if dispatch_attention_fn is not None:
        return dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=backend,
            parallel_config=parallel_config,
        )
    q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    return out.transpose(1, 2)



def infer_context_layout(image_seq_len: int, n_target_tokens: int, n_context_images: int) -> Tuple[int, List[Tuple[int, int]]]:
    """Return n_ctx and slices for each context image within image hidden states."""
    if image_seq_len <= n_target_tokens:
        raise RuntimeError(
            f"image_seq_len={image_seq_len} <= target={n_target_tokens}; unexpected layout"
        )
    remaining = image_seq_len - n_target_tokens
    if n_context_images <= 0 or remaining % n_context_images != 0:
        raise RuntimeError(
            "Context tokens do not divide evenly. "
            f"image_seq={image_seq_len}, target={n_target_tokens}, remaining={remaining}, "
            f"context_images={n_context_images}. Check enable_multi_context()."
        )
    n_ctx = remaining // n_context_images
    slices = []
    for slot in range(n_context_images):
        s = n_target_tokens + slot * n_ctx
        slices.append((s, s + n_ctx))
    return n_ctx, slices


class IdentityKVProcessor:
    """Scale foreground K/V of REFERENCE object slices in selected FLUX blocks.

    Context layout for E11 finalisation:
        [scene, pose_proposal, ref1, ref2, ..., refN]

    The first `n_nonref_context_images` contexts are not amplified.
    Only the subsequent reference slices are amplified.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        n_target_tokens: int,
        n_context_images: int,
        n_nonref_context_images: int,
        mask_bank: ReferenceMaskBank,
        k_scale: float,
        v_scale: float,
        debug_layout: bool,
        block_index: int,
    ):
        self.n_target_tokens = int(n_target_tokens)
        self.n_context_images = int(n_context_images)
        self.n_nonref_context_images = int(n_nonref_context_images)
        self.mask_bank = mask_bank
        self.k_scale = float(k_scale)
        self.v_scale = float(v_scale)
        self.debug_layout = bool(debug_layout)
        self.block_index = int(block_index)
        self._layout_logged = False

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        q, k, v, eq, ek, ev = _project_qkv(attn, hidden_states, encoder_hidden_states)
        q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
        v = v.unflatten(-1, (attn.heads, -1))

        if self.n_context_images > self.n_nonref_context_images and (self.k_scale != 1.0 or self.v_scale != 1.0):
            n_ctx, slices = infer_context_layout(
                image_seq_len=k.shape[1],
                n_target_tokens=self.n_target_tokens,
                n_context_images=self.n_context_images,
            )
            prefix_slices = slices[: self.n_nonref_context_images]
            ref_slices = slices[self.n_nonref_context_images :]
            masks = self.mask_bank.get(n_ctx, k.device)
            if len(masks) != len(ref_slices):
                raise RuntimeError(
                    f"Reference masks={len(masks)} != reference slices={len(ref_slices)}"
                )

            if self.debug_layout and not self._layout_logged:
                print(
                    f"  [final-layout block {self.block_index}] target=0:{self.n_target_tokens}, "
                    f"nonref_contexts={prefix_slices}, refs={ref_slices}, image_seq={k.shape[1]}, n_ctx={n_ctx}"
                )
                self._layout_logged = True

            if self.k_scale != 1.0:
                k = k.clone()
            if self.v_scale != 1.0:
                v = v.clone()

            for (s, e), m in zip(ref_slices, masks):
                if m.numel() != e - s:
                    raise RuntimeError(f"Mask tokens {m.numel()} != slice length {e-s}")
                if self.k_scale != 1.0:
                    scale = 1.0 + (self.k_scale - 1.0) * m.to(k.dtype)
                    k[:, s:e] *= scale[None, :, None, None]
                if self.v_scale != 1.0:
                    scale = 1.0 + (self.v_scale - 1.0) * m.to(v.dtype)
                    v[:, s:e] *= scale[None, :, None, None]

        if attn.added_kv_proj_dim is not None:
            if encoder_hidden_states is None:
                raise RuntimeError("Dual-stream FLUX block expected encoder_hidden_states.")
            eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
            ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
            ev = ev.unflatten(-1, (attn.heads, -1))
            q = torch.cat([eq, q], dim=1)
            k = torch.cat([ek, k], dim=1)
            v = torch.cat([ev, v], dim=1)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        hidden = _dispatch_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            backend=getattr(self, "_attention_backend", None),
            parallel_config=getattr(self, "_parallel_config", None),
        )
        hidden = hidden.flatten(2, 3).to(q.dtype)

        if encoder_hidden_states is not None:
            enc_out, img_out = hidden.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            img_out = attn.to_out[0](img_out.contiguous())
            img_out = attn.to_out[1](img_out)
            enc_out = attn.to_add_out(enc_out.contiguous())
            return img_out, enc_out
        return hidden


class IdentityKVController:
    def __init__(
        self,
        transformer,
        blocks: Sequence[int],
        n_target_tokens: int,
        n_context_images: int,
        n_nonref_context_images: int,
        mask_bank: ReferenceMaskBank,
        k_scale: float,
        v_scale: float,
        debug_layout: bool,
    ):
        self.transformer = transformer
        self.original = {}
        n_blocks = len(transformer.transformer_blocks)
        invalid = [i for i in blocks if i < 0 or i >= n_blocks]
        if invalid:
            raise ValueError(f"Invalid identity blocks {invalid}; model has {n_blocks} blocks.")

        for i in blocks:
            attn = transformer.transformer_blocks[i].attn
            orig = attn.processor
            if "IPAdapter" in orig.__class__.__name__:
                raise RuntimeError(f"Block {i} uses IP-Adapter processor; E11 does not wrap it.")
            self.original[i] = orig
            proc = IdentityKVProcessor(
                n_target_tokens=n_target_tokens,
                n_context_images=n_context_images,
                n_nonref_context_images=n_nonref_context_images,
                mask_bank=mask_bank,
                k_scale=k_scale,
                v_scale=v_scale,
                debug_layout=debug_layout,
                block_index=i,
            )
            proc._attention_backend = getattr(orig, "_attention_backend", None)
            proc._parallel_config = getattr(orig, "_parallel_config", None)
            attn.set_processor(proc)

    def remove(self):
        for i, proc in self.original.items():
            self.transformer.transformer_blocks[i].attn.set_processor(proc)
        self.original.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False


# =============================================================================
# Self-derived placement mask construction
# =============================================================================


@dataclass
class PlacementMaskResult:
    score_map: np.ndarray
    hard_grid: np.ndarray
    hard_mask: Image.Image
    soft_mask: Image.Image
    threshold: float
    grid_fraction: float



def proposal_difference_map(previous: Image.Image, proposal: Image.Image, grid_size: Tuple[int, int], blur_px: float = 5.0) -> np.ndarray:
    prev = previous.convert("RGB").resize(proposal.size, Image.Resampling.LANCZOS)
    a = np.asarray(prev, dtype=np.float32) / 255.0
    b = np.asarray(proposal.convert("RGB"), dtype=np.float32) / 255.0
    diff = np.mean(np.abs(a - b), axis=2)
    diff_img = Image.fromarray(np.uint8(np.clip(diff, 0, 1) * 255), mode="L")
    if blur_px > 0:
        diff_img = diff_img.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    grid_w, grid_h = grid_size
    small = np.asarray(diff_img.resize((grid_w, grid_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return robust_normalize(small)



def derive_difference_mask(
    difference_map: np.ndarray,
    output_size: Tuple[int, int],
    mask_quantile: float,
    min_mask_frac: float,
    max_mask_frac: float,
    score_blur_tokens: float,
    close_tokens: int,
    dilate_tokens: int,
    feather_px: float,
) -> PlacementMaskResult:
    combined = gaussian_blur_array(difference_map, score_blur_tokens)

    q = float(np.clip(mask_quantile, 0.50, 0.995))
    threshold = 0.0
    binary = np.zeros_like(combined, dtype=bool)
    frac = 0.0
    for _ in range(10):
        threshold = float(np.quantile(combined, q))
        binary = morph_close(combined >= threshold, close_tokens)
        binary = best_connected_component(binary, combined)
        frac = float(binary.mean())

        if frac < min_mask_frac and q > 0.50:
            q = max(0.50, q - 0.05)
            continue
        if frac > max_mask_frac and q < 0.995:
            q = min(0.995, q + 0.03)
            continue
        break

    if not binary.any():
        y, x = np.unravel_index(int(np.argmax(combined)), combined.shape)
        binary[max(0, y - 1): min(binary.shape[0], y + 2), max(0, x - 1): min(binary.shape[1], x + 2)] = True
        frac = float(binary.mean())
        warnings.warn("Placement mask collapsed; using a small region around heatmap maximum.")

    hard_grid = dilate_binary(binary, dilate_tokens)
    hard_small = Image.fromarray(np.uint8(hard_grid) * 255, mode="L")
    hard_mask = hard_small.resize(output_size, Image.Resampling.NEAREST)
    soft_mask = hard_mask
    if feather_px > 0:
        soft_mask = hard_mask.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))

    return PlacementMaskResult(
        score_map=combined,
        hard_grid=hard_grid,
        hard_mask=hard_mask,
        soft_mask=soft_mask,
        threshold=threshold,
        grid_fraction=float(hard_grid.mean()),
    )


# =============================================================================
# Prompting, candidate heuristics, and data loading
# =============================================================================


def object_affordance(name: str) -> str:
    priors = {
        "bicycle": "standing naturally on the floor, preferably near a wall or open side area without blocking the room",
        "vase": "resting upright on a suitable existing table, shelf, cabinet, or other stable horizontal surface",
        "ball": "resting naturally on the floor in a small open area",
        "chair": "standing on the floor in a usable open area and aligned plausibly with the room perspective",
        "lamp": "placed where a lamp would naturally belong, on a suitable surface if it is a table lamp or on the floor if its scale implies a floor lamp",
        "plant": "standing naturally on the floor near a wall/corner or on a suitable surface if the scale implies a small plant",
        "backpack": "resting naturally near furniture, against a wall, or on a suitable support rather than floating",
    }
    return priors.get(name, "placed in a physically supported, semantically appropriate location")



def planner_candidate_hints(name: str) -> List[Tuple[str, str]]:
    afford = object_affordance(name)
    return [
        ("free", f"Choose the most natural location anywhere in the room; the {name} should be {afford}."),
        ("left", f"Find a natural location in the LEFT half of the room; the {name} should be {afford}. Avoid the exact image centre."),
        ("right", f"Find a natural location in the RIGHT half of the room; the {name} should be {afford}. Avoid the exact image centre."),
        ("back_or_side", f"Prefer a plausible BACK or SIDE area of the room with physical support; the {name} should be {afford}. Avoid the exact image centre unless genuinely necessary."),
    ]



def build_scene_first_planner_prompt(new_name: str, previous_names: Sequence[str], placement_hint: str) -> str:
    prompt = (
        f"Add a plausible generic {new_name} exactly once. {placement_hint} "
        f"This pass is only for planning where a {new_name} could naturally fit in the room and roughly how large it should appear. "
        f"Do not focus on the exact identity of the {new_name}; use a generic but realistic {new_name}. "
        f"Make sure the {new_name} is physically supported, respects room depth and perspective, and does not float. "
    )
    if previous_names:
        prompt += f"Keep the existing {', '.join(previous_names)} where they are and do not duplicate or alter them. "
    prompt += "Preserve the room layout, camera, furniture, walls, floor, and lighting apart from the planned new object."
    return prompt



def build_pose_prompt(new_name: str, previous_names: Sequence[str], placement_mask: Image.Image) -> str:
    pos = position_phrase(placement_mask)
    where = f" in the {pos}" if pos else " in the selected room region"
    prompt = (
        f"Add the {new_name} exactly once{where}. "
        f"Use the object reference to preserve the {new_name}'s identity-defining structure, exact colors, materials, texture, and distinctive design. "
        f"Adapt its viewpoint, orientation, apparent scale, foreshortening, and perspective so it fits naturally in the room and camera view. "
        f"Do not copy the reference as a flat front-facing sticker. Render the same {new_name} as if naturally photographed in this scene. "
        f"For rigid objects, do not bend, melt, redesign, or change structural geometry. Keep floor or surface contact physically plausible. "
    )
    if previous_names:
        prompt += f"Keep the existing {', '.join(previous_names)} where they are and do not duplicate or alter them. "
    prompt += "Preserve room geometry, camera, lighting, walls, floor, furniture, and unrelated details."
    return prompt



def build_final_prompt(new_name: str, previous_names: Sequence[str], placement_mask: Image.Image) -> str:
    pos = position_phrase(placement_mask)
    where = f" in the {pos}" if pos else " in the selected room region"
    prompt = (
        f"Image A is the clean current room. Image B is a pose-and-placement proposal. The remaining reference images define the object identities. "
        f"Add the {new_name} exactly once{where}. "
        f"Use Image B mainly for the location, approximate pose, viewpoint, and apparent scale of the new {new_name}. "
        f"Use the reference images to preserve exact colors, materials, texture, distinctive parts, and identity-defining geometry. "
        f"Clean up artefacts from the proposal and render a coherent final scene. Do not turn the object into a flat sticker. "
        f"Respect room depth, contact shadows, occlusion, and realistic support. "
    )
    if previous_names:
        prompt += (
            f"Keep the existing {', '.join(previous_names)} exactly where they are and preserve their appearance. "
            "Do not duplicate them. "
        )
    prompt += "Preserve the camera, walls, floor, lighting, furniture, and unrelated scene details. Change only what is necessary."
    return prompt



def load_objects(obj_dir: str):
    refs, originals = {}, {}
    for name in OBJ_ORDER:
        path = os.path.join(obj_dir, f"obj_{name}.png")
        if os.path.isfile(path):
            img = Image.open(path).copy()
            originals[name] = img
            refs[name] = img.convert("RGB")
    return refs, originals


@dataclass
class PlannerCandidate:
    name: str
    hint: str
    seed: int
    prompt: str
    proposal: Image.Image
    difference_map: np.ndarray
    placement: PlacementMaskResult
    score: float
    components: Dict[str, float]



def _mask_centroid01(mask: Image.Image) -> Tuple[float, float]:
    a = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    total = float(a.sum())
    if total <= 1e-8:
        return 0.5, 0.5
    yy, xx = np.indices(a.shape)
    return (float((xx * a).sum() / total) / max(a.shape[1]-1, 1),
            float((yy * a).sum() / total) / max(a.shape[0]-1, 1))



def _mask_iou(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a.convert("L").resize(b.size, Image.Resampling.NEAREST)) > 32
    bb = np.asarray(b.convert("L")) > 32
    u = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / u) if u else 0.0



def _border_fraction(mask: Image.Image, border_frac: float = 0.03) -> float:
    a = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    H, W = a.shape
    by, bx = max(1, int(H*border_frac)), max(1, int(W*border_frac))
    border = np.zeros_like(a, dtype=bool)
    border[:by] = True; border[-by:] = True; border[:, :bx] = True; border[:, -bx:] = True
    total = float(a.sum())
    return float(a[border].sum() / total) if total > 1e-8 else 1.0



def support_prior_score(new_name: str, mask: Image.Image) -> float:
    box = mask_bbox(mask)
    if box is None:
        return 0.0
    W, H = mask.size
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) * 0.5) / max(W, 1)
    cy = ((y0 + y1) * 0.5) / max(H, 1)
    frac = ((x1 - x0) * (y1 - y0)) / max(W * H, 1)

    def gauss(x, mu, sigma):
        sigma = max(float(sigma), 1e-6)
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    if new_name in {"bicycle", "chair", "ball", "plant", "backpack"}:
        vertical = gauss(cy, 0.72, 0.20)
    elif new_name == "vase":
        vertical = max(gauss(cy, 0.55, 0.18), gauss(cy, 0.70, 0.18) * 0.8)
    elif new_name == "lamp":
        vertical = max(gauss(cy, 0.68, 0.22), gauss(cy, 0.50, 0.18) * 0.9)
    else:
        vertical = gauss(cy, 0.65, 0.25)

    # Mild preference against huge planner masks.
    size_score = gauss(frac, 0.08, 0.10)
    # Mild preference against exact centre collapse.
    center_dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / math.sqrt(0.5 ** 2 + 0.5 ** 2)
    centre_term = min(1.0, center_dist / 0.20)

    return float(0.55 * vertical + 0.25 * size_score + 0.20 * centre_term)



def score_scene_first_candidate(
    scene: Image.Image,
    proposal: Image.Image,
    placement: PlacementMaskResult,
    previous_masks: Sequence[Image.Image],
    new_name: str,
    args,
) -> Tuple[float, Dict[str, float]]:
    mask = placement.soft_mask
    prev = np.asarray(scene.convert("RGB"), dtype=np.float32) / 255.0
    prop = np.asarray(proposal.convert("RGB").resize(scene.size), dtype=np.float32) / 255.0
    d = np.mean(np.abs(prop - prev), axis=2)
    m = np.asarray(mask.convert("L").resize(scene.size), dtype=np.float32) / 255.0
    inside = float((d * m).sum() / (m.sum() + 1e-6))
    outside = float((d * (1 - m)).sum() / ((1 - m).sum() + 1e-6))
    locality = inside / (inside + outside + 1e-6)

    overlap = max([_mask_iou(mask, pm) for pm in previous_masks], default=0.0)
    border = _border_fraction(mask)
    cx, cy = _mask_centroid01(mask)
    dist_center = math.sqrt((cx-0.5)**2 + (cy-0.5)**2) / math.sqrt(0.5**2 + 0.5**2)
    exact_center_penalty = max(0.0, 1.0 - dist_center / max(args.center_deadzone, 1e-4))
    support = support_prior_score(new_name, mask)

    score = (
        args.score_locality_weight * locality
        + args.score_support_weight * support
        - args.score_overlap_weight * overlap
        - args.score_border_weight * border
        - args.score_center_weight * exact_center_penalty
    )
    comp = {
        "locality": locality,
        "support_prior": support,
        "previous_overlap": overlap,
        "border_fraction": border,
        "center_distance": dist_center,
        "exact_center_penalty": exact_center_penalty,
        "centroid_x": cx,
        "centroid_y": cy,
    }
    return float(score), comp


# =============================================================================
# Metrics
# =============================================================================


def safe_metric(fn, *args, key: str, record: dict):
    try:
        record[key] = float(fn(*args))
    except Exception as exc:
        warnings.warn(f"Metric {key} failed: {exc}")
        record[key] = None



def add_metrics(
    record: dict,
    base: Image.Image,
    previous: Image.Image,
    result: Image.Image,
    active_names: Sequence[str],
    obj_refs: Dict[str, Image.Image],
    placement_masks: Dict[str, Image.Image],
    new_name: str,
    device: str,
):
    safe_metric(compute_ssim, base, result, key="scene_ssim_global", record=record)
    safe_metric(compute_lpips, base, result, device, key="scene_lpips_global", record=record)

    masks = [placement_masks[n] for n in active_names if n in placement_masks]
    union = union_masks(masks, result.size)
    if union is not None:
        b = neutralize_edit_region(base.resize(result.size), union)
        r = neutralize_edit_region(result, union)
        safe_metric(compute_ssim, b, r, key="background_ssim_approx", record=record)
        safe_metric(compute_lpips, b, r, device, key="background_lpips_approx", record=record)

    if new_name in placement_masks:
        m = placement_masks[new_name]
        p = neutralize_edit_region(previous.resize(result.size), m)
        r = neutralize_edit_region(result, m)
        safe_metric(compute_ssim, p, r, key="outside_new_edit_ssim_approx", record=record)
        safe_metric(compute_lpips, p, r, device, key="outside_new_edit_lpips_approx", record=record)

    for name in active_names:
        m = placement_masks.get(name)
        if m is None:
            continue
        box = mask_bbox(m)
        if box is None:
            continue
        box = expand_bbox(box, result.size, frac=0.08)
        crop = result.crop(box)
        safe_metric(compute_dino, obj_refs[name], crop, device, key=f"dino_{name}_local", record=record)
        safe_metric(compute_clip_i, obj_refs[name], crop, device, key=f"clip_{name}_local", record=record)
        if name != new_name:
            prev_crop = previous.resize(result.size).crop(box)
            safe_metric(compute_dino, prev_crop, crop, device, key=f"dino_{name}_temporal", record=record)
            safe_metric(compute_lpips, prev_crop, crop, device, key=f"lpips_{name}_temporal", record=record)


# =============================================================================
# Three-pass pipeline
# =============================================================================


def run_scene_first_placement_search(
    pipe,
    scene: Image.Image,
    new_name: str,
    previous_names: Sequence[str],
    previous_masks: Sequence[Image.Image],
    target_grid: Tuple[int, int],
    args,
    base_seed: int,
    step_dir: str,
) -> PlannerCandidate:
    cand_root = os.path.join(step_dir, "placement_candidates")
    os.makedirs(cand_root, exist_ok=True)
    hints = planner_candidate_hints(new_name)[: args.placement_candidates]
    candidates: List[PlannerCandidate] = []

    for idx, (tag, hint) in enumerate(hints, start=1):
        cand_seed = int(base_seed + idx * 97)
        cand_dir = os.path.join(cand_root, f"{idx:02d}_{tag}")
        os.makedirs(cand_dir, exist_ok=True)

        prompt = build_scene_first_planner_prompt(new_name, previous_names, hint)
        proposal = pipe(
            image=[scene],
            prompt=prompt,
            height=args.height,
            width=args.width,
            max_area=args.height * args.width,
            num_inference_steps=args.planner_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(args.device).manual_seed(cand_seed),
        ).images[0].convert("RGB")

        diff_map = proposal_difference_map(
            previous=scene,
            proposal=proposal,
            grid_size=target_grid,
            blur_px=args.difference_blur_px,
        )
        placement = derive_difference_mask(
            difference_map=diff_map,
            output_size=scene.size,
            mask_quantile=args.mask_quantile,
            min_mask_frac=args.min_placement_frac,
            max_mask_frac=args.max_placement_frac,
            score_blur_tokens=args.score_blur_tokens,
            close_tokens=args.close_tokens,
            dilate_tokens=args.dilate_tokens,
            feather_px=args.mask_feather_px,
        )
        score, comp = score_scene_first_candidate(
            scene=scene,
            proposal=proposal,
            placement=placement,
            previous_masks=previous_masks,
            new_name=new_name,
            args=args,
        )

        proposal.save(os.path.join(cand_dir, "proposal.png"))
        save_float_map(diff_map, os.path.join(cand_dir, "difference_heatmap.png"), out_size=scene.size)
        save_float_map(placement.score_map, os.path.join(cand_dir, "placement_score_map.png"), out_size=scene.size)
        placement.hard_mask.save(os.path.join(cand_dir, "placement_mask_hard.png"))
        placement.soft_mask.save(os.path.join(cand_dir, "placement_mask_soft.png"))
        make_overlay(scene, placement.soft_mask).save(os.path.join(cand_dir, "overlay.png"))
        with open(os.path.join(cand_dir, "score.json"), "w", encoding="utf-8") as f:
            json.dump({"score": score, **comp}, f, indent=2)

        candidates.append(PlannerCandidate(
            name=tag,
            hint=hint,
            seed=cand_seed,
            prompt=prompt,
            proposal=proposal,
            difference_map=diff_map,
            placement=placement,
            score=score,
            components=comp,
        ))

    best = max(candidates, key=lambda c: c.score)
    with open(os.path.join(step_dir, "placement_selection.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected": best.name,
                "selected_seed": best.seed,
                "selected_score": best.score,
                "selected_components": best.components,
            },
            f,
            indent=2,
        )
    return best



def run_pose_proposal(
    pipe,
    scene: Image.Image,
    new_name: str,
    previous_names: Sequence[str],
    obj_ref: Image.Image,
    placement_mask: Image.Image,
    args,
    seed: int,
) -> Tuple[Image.Image, Image.Image, str]:
    prompt = build_pose_prompt(new_name, previous_names, placement_mask)
    raw = pipe(
        image=[scene, obj_ref],
        prompt=prompt,
        height=args.height,
        width=args.width,
        max_area=args.height * args.width,
        num_inference_steps=args.pose_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(args.device).manual_seed(seed),
    ).images[0].convert("RGB")

    # Keep the proposal local to the selected region.
    proposed = protect_outside_soft_mask(scene, raw, placement_mask)
    return raw, proposed, prompt



def run_final_pass(
    pipe,
    scene: Image.Image,
    pose_proposal: Image.Image,
    new_name: str,
    active_names: Sequence[str],
    obj_refs: Dict[str, Image.Image],
    ref_masks: Dict[str, Image.Image],
    placement_mask: Image.Image,
    n_target_tokens: int,
    identity_blocks: Sequence[int],
    args,
    seed: int,
) -> Tuple[Image.Image, Image.Image, str]:
    previous_names = list(active_names[:-1])
    if args.reference_policy == "current":
        ref_names = [new_name]
    else:
        ref_names = list(active_names)

    prompt = build_final_prompt(new_name, previous_names, placement_mask)
    context = [scene, pose_proposal] + [obj_refs[n] for n in ref_names]

    controller = None
    if args.k_scale != 1.0 or args.v_scale != 1.0:
        bank = ReferenceMaskBank(
            masks=[ref_masks[n] for n in ref_names],
            aspect_ratios=[obj_refs[n].width / obj_refs[n].height for n in ref_names],
        )
        controller = IdentityKVController(
            transformer=pipe.transformer,
            blocks=identity_blocks,
            n_target_tokens=n_target_tokens,
            n_context_images=len(context),
            n_nonref_context_images=2,
            mask_bank=bank,
            k_scale=args.k_scale,
            v_scale=args.v_scale,
            debug_layout=args.debug_layout,
        )

    try:
        raw = pipe(
            image=context,
            prompt=prompt,
            height=args.height,
            width=args.width,
            max_area=args.height * args.width,
            num_inference_steps=args.final_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(args.device).manual_seed(seed),
        ).images[0].convert("RGB")
    finally:
        if controller is not None:
            controller.remove()

    final = raw
    if args.protect_outside_mask:
        final = protect_outside_soft_mask(scene, raw, placement_mask)
    return raw, final, prompt


# =============================================================================
# CLI / main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scene", required=True, help="Base scene image")
    p.add_argument("--obj_dir", required=True, help="Directory with obj_<name>.png")
    p.add_argument("--out_dir", default="results/e11_scene_first_planner_pose_finalizer")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--device", default="cuda")

    # Generation
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--planner_steps", type=int, default=10)
    p.add_argument("--pose_steps", type=int, default=16)
    p.add_argument("--final_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--seeds", default="42")
    p.add_argument(
        "--reference_policy",
        choices=["current", "all"],
        default="all",
        help="Final pass gets only the new ref or every ref seen so far",
    )

    # Planner candidate search
    p.add_argument("--placement_candidates", type=int, default=4, choices=[1, 2, 3, 4])
    p.add_argument("--score_locality_weight", type=float, default=0.85)
    p.add_argument("--score_support_weight", type=float, default=0.55)
    p.add_argument("--score_overlap_weight", type=float, default=1.00)
    p.add_argument("--score_border_weight", type=float, default=0.45)
    p.add_argument("--score_center_weight", type=float, default=0.30)
    p.add_argument("--center_deadzone", type=float, default=0.18)

    # Placement mask from planner proposals
    p.add_argument("--mask_quantile", type=float, default=0.85)
    p.add_argument("--min_placement_frac", type=float, default=0.01)
    p.add_argument("--max_placement_frac", type=float, default=0.35)
    p.add_argument("--score_blur_tokens", type=float, default=1.2)
    p.add_argument("--close_tokens", type=int, default=1)
    p.add_argument("--dilate_tokens", type=int, default=2)
    p.add_argument("--mask_feather_px", type=float, default=12.0)
    p.add_argument("--difference_blur_px", type=float, default=5.0)
    p.add_argument(
        "--protect_outside_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore previous scene outside selected soft placement mask",
    )

    # Final identity attention
    p.add_argument("--identity_blocks", default=DEFAULT_IDENTITY_BLOCKS)
    p.add_argument("--k_scale", type=float, default=1.5)
    p.add_argument("--v_scale", type=float, default=1.0)

    # Object reference foreground mask
    p.add_argument("--grey_tol", type=int, default=20)
    p.add_argument("--white_tol", type=int, default=20)
    p.add_argument("--min_ref_mask_frac", type=float, default=0.02)
    p.add_argument("--max_ref_mask_frac", type=float, default=0.95)
    p.add_argument("--ref_mask_dilate_px", type=int, default=1)
    p.add_argument("--allow_mask_fallback", action="store_true")

    p.add_argument("--debug_layout", action="store_true")
    return p.parse_args()



def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    identity_blocks = parse_int_ranges(args.identity_blocks)

    os.makedirs(args.out_dir, exist_ok=True)

    pipe = load_pipe(args.model_id, args.device)
    enable_multi_context(pipe)

    n_target_tokens = packed_token_count(args.height, args.width, int(pipe.vae_scale_factor))
    target_gh, target_gw = infer_grid_from_tokens(n_target_tokens, args.width / args.height)

    base = Image.open(args.scene).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    obj_refs, obj_originals = load_objects(args.obj_dir)
    available = [n for n in OBJ_ORDER if n in obj_refs]
    if not available:
        raise FileNotFoundError(f"No obj_<name>.png references found in {args.obj_dir}")

    with open(os.path.join(args.out_dir, "config_e11.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    ref_mask_dir = os.path.join(args.out_dir, "reference_foreground_masks")
    os.makedirs(ref_mask_dir, exist_ok=True)

    ref_masks = {}
    for name in available:
        ref_masks[name] = extract_reference_foreground_mask(
            obj_originals[name],
            grey_tol=args.grey_tol,
            white_tol=args.white_tol,
            min_frac=args.min_ref_mask_frac,
            max_frac=args.max_ref_mask_frac,
            dilate_px=args.ref_mask_dilate_px,
            allow_fallback=args.allow_mask_fallback,
        )
        ref_masks[name].save(os.path.join(ref_mask_dir, f"refmask_{name}.png"))
        frac = float(np.asarray(ref_masks[name], dtype=np.float32).mean() / 255.0)
        print(f"reference foreground {name:>10s}: {frac:6.2%}")

    print("\n=== E11: SCENE-FIRST PLANNER -> POSE PROPOSAL -> FINALIZER ===")
    print(f"objects             : {available}")
    print(f"seeds               : {seeds}")
    print(f"target token grid   : {target_gh}x{target_gw} = {n_target_tokens}")
    print(f"planner/pose/final  : {args.planner_steps}/{args.pose_steps}/{args.final_steps}")
    print(f"identity blocks     : {identity_blocks}")
    print(f"K/V scale           : {args.k_scale}/{args.v_scale}")
    print(f"reference policy    : {args.reference_policy}")
    print(f"protect outside     : {args.protect_outside_mask}")
    print(f"placement candidates: {args.placement_candidates}")

    all_metrics = []

    for seed in seeds:
        seed_root = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_root, exist_ok=True)
        current_scene = base.copy()
        placement_masks: Dict[str, Image.Image] = {}

        for step, new_name in enumerate(available, start=1):
            active_names = available[:step]
            previous_names = active_names[:-1]
            step_seed = int(seed + step * 1000)
            step_dir = os.path.join(seed_root, f"step_{step:02d}_add_{new_name}")
            os.makedirs(step_dir, exist_ok=True)

            print(f"\n[seed={seed}] [{step}/{len(available)}] +{new_name}")
            print("  Pass 1: scene-first placement planning")
            planner = run_scene_first_placement_search(
                pipe=pipe,
                scene=current_scene,
                new_name=new_name,
                previous_names=previous_names,
                previous_masks=[placement_masks[n] for n in previous_names if n in placement_masks],
                target_grid=(target_gw, target_gh),
                args=args,
                base_seed=step_seed,
                step_dir=step_dir,
            )
            planner.proposal.save(os.path.join(step_dir, "planner_selected_proposal.png"))
            save_float_map(planner.difference_map, os.path.join(step_dir, "planner_selected_difference_heatmap.png"), out_size=current_scene.size)
            planner.placement.hard_mask.save(os.path.join(step_dir, "placement_mask_hard.png"))
            planner.placement.soft_mask.save(os.path.join(step_dir, "placement_mask_soft.png"))
            make_overlay(current_scene, planner.placement.soft_mask).save(os.path.join(step_dir, "placement_overlay.png"))
            placement_masks[new_name] = planner.placement.soft_mask
            print(f"    selected candidate : {planner.name}  score={planner.score:.3f}")

            print("  Pass 2: pose proposal using selected mask + true reference")
            pose_seed = planner.seed + 50000
            pose_raw, pose_proposed, pose_prompt = run_pose_proposal(
                pipe=pipe,
                scene=current_scene,
                new_name=new_name,
                previous_names=previous_names,
                obj_ref=obj_refs[new_name],
                placement_mask=planner.placement.soft_mask,
                args=args,
                seed=pose_seed,
            )
            pose_raw.save(os.path.join(step_dir, "pose_raw.png"))
            pose_proposed.save(os.path.join(step_dir, "pose_proposed.png"))
            with open(os.path.join(step_dir, "pose_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(pose_prompt)

            print("  Pass 3: identity-preserving finalisation")
            final_seed = pose_seed
            final_raw, final_img, final_prompt = run_final_pass(
                pipe=pipe,
                scene=current_scene,
                pose_proposal=pose_proposed,
                new_name=new_name,
                active_names=active_names,
                obj_refs=obj_refs,
                ref_masks=ref_masks,
                placement_mask=planner.placement.soft_mask,
                n_target_tokens=n_target_tokens,
                identity_blocks=identity_blocks,
                args=args,
                seed=final_seed,
            )
            final_raw.save(os.path.join(step_dir, "final_raw.png"))
            final_img.save(os.path.join(step_dir, "final.png"))
            with open(os.path.join(step_dir, "final_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(final_prompt)

            record = {
                "seed": seed,
                "step": step,
                "new_object": new_name,
                "active_objects": active_names,
                "planner_candidate": planner.name,
                "planner_score": planner.score,
                "planner_components": planner.components,
                "planner_seed": planner.seed,
                "pose_seed": pose_seed,
                "final_seed": final_seed,
                "k_scale": args.k_scale,
                "v_scale": args.v_scale,
                "reference_policy": args.reference_policy,
            }
            add_metrics(
                record=record,
                base=base,
                previous=current_scene,
                result=final_img,
                active_names=active_names,
                obj_refs=obj_refs,
                placement_masks=placement_masks,
                new_name=new_name,
                device=args.device,
            )
            with open(os.path.join(step_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            all_metrics.append(record)

            current_scene = final_img

        current_scene.save(os.path.join(seed_root, "FINAL.png"))

    with open(os.path.join(args.out_dir, "metrics_e11.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nDone. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
