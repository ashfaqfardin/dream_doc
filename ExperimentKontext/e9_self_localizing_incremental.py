"""
E9: Self-Localizing Incremental Object Editing for FLUX.1-Kontext
=================================================================

Goal
----
Incrementally add reference objects while preserving:
  1) the identity of the newly inserted object,
  2) previously inserted objects,
  3) untouched scene content,
with no manually drawn placement masks.

Core idea
---------
Each incremental edit uses TWO passes:

  PASS A — proposal / self-localisation
      current scene + NEW object reference
          -> short FLUX proposal
          -> capture target-to-object reference affinity inside selected
             FLUX double-stream attention blocks
          -> combine attention relevance with proposal-vs-scene change
          -> automatically derive a soft placement mask

  PASS B — final edit
      CLEAN current scene + all seen object references
          -> identity-preserving K scaling in selected blocks
          -> final generation
          -> keep generated pixels inside the self-derived soft mask and
             restore the previous scene outside it

The proposal image itself is NEVER chained. Only the protected final image is
fed into the next edit.

Why hybrid localisation?
------------------------
Attention alone can highlight semantically related existing regions, while raw
pixel difference can include unrelated proposal artefacts. By default E9 uses:

    hybrid = attention_weight * attention
             + (1-attention_weight) * difference

and keeps the strongest connected spatial region. Both signals come from the
model's own proposal pass; no external segmentation model or manual placement
mask is required.

Important assumptions
---------------------
This file relies on your existing utils.py:
    load_pipe, enable_multi_context,
    compute_ssim, compute_lpips, compute_dino, compute_clip_i

The multi-context contract is assumed to be:
    image = [scene_context, object_ref_1, ..., object_ref_N]

with image hidden-state token layout:
    [target/noisy | scene_context | obj_1 | ... | obj_N]

E9 validates that the context section divides evenly between context images.
If your enable_multi_context() uses a different ordering, adjust
infer_context_layout().

Recommended first run
---------------------
python e9_self_localizing_incremental.py \
  --scene room.png \
  --obj_dir objects \
  --out_dir results/e9_self_localizing \
  --proposal_steps 12 \
  --steps 28 \
  --k_scale 1.5 \
  --debug_layout

Useful ablations
----------------
--mask_source attention
--mask_source difference
--mask_source hybrid
--k_scale 1.0               # removes final identity K amplification
--reference_policy current   # only new ref in final pass
--reference_policy all       # all seen refs in final pass (default)

Outputs per edit
----------------
proposal.png
attention_heatmap.png
difference_heatmap.png
hybrid_heatmap.png
placement_mask_hard.png
placement_mask_soft.png
placement_overlay.png
final.png

A metrics JSON is also written at the experiment root.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
DEFAULT_LOCALIZATION_BLOCKS = "13,15,17"


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
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
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


def protect_outside_soft_mask(
    previous: Image.Image,
    generated: Image.Image,
    soft_mask: Image.Image,
) -> Image.Image:
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


# =============================================================================
# Reference foreground masks for object K selection
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
    """
    Cache token masks for object references.

    A separate aspect ratio is retained for every reference because object
    reference images may not all share the same source aspect ratio.
    """

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
        print(f"  [mask-bank] n_ctx={n_ctx}, inferred object grids={grids}")
        return result

    def get(self, n_ctx: int, device: torch.device) -> List[torch.Tensor]:
        key = (int(n_ctx), str(device))
        if key not in self._device:
            self._device[key] = [m.to(device) for m in self._build_cpu(n_ctx)]
        return self._device[key]


# =============================================================================
# FLUX attention helpers
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


def infer_context_layout(
    image_seq_len: int,
    n_target_tokens: int,
    n_context_images: int,
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Return n_ctx and slices for each context image within IMAGE hidden states.

    Layout:
        target | scene | obj1 | obj2 | ...
    """
    if image_seq_len <= n_target_tokens:
        raise RuntimeError(
            f"image_seq_len={image_seq_len} <= target={n_target_tokens}; unexpected layout"
        )
    remaining = image_seq_len - n_target_tokens
    if n_context_images <= 0 or remaining % n_context_images != 0:
        raise RuntimeError(
            "Context tokens do not divide evenly. "
            f"image_seq={image_seq_len}, target={n_target_tokens}, "
            f"remaining={remaining}, context_images={n_context_images}. "
            "Check enable_multi_context()."
        )
    n_ctx = remaining // n_context_images
    slices = []
    for slot in range(n_context_images):
        s = n_target_tokens + slot * n_ctx
        slices.append((s, s + n_ctx))
    return n_ctx, slices


# =============================================================================
# Proposal-pass target -> object affinity capture
# =============================================================================


class PlacementAccumulator:
    def __init__(self, n_target_tokens: int, target_aspect_ratio: float):
        self.n_target_tokens = int(n_target_tokens)
        self.grid_h, self.grid_w = infer_grid_from_tokens(
            n_target_tokens, target_aspect_ratio
        )
        self.sum_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)
        self.count = 0

    def add(self, scores: torch.Tensor):
        """scores: (n_target_tokens,) on any device."""
        x = scores.detach().float().cpu().numpy().reshape(self.grid_h, self.grid_w)
        x = robust_normalize(x)
        self.sum_map += x.astype(np.float64)
        self.count += 1

    def result(self) -> np.ndarray:
        if self.count == 0:
            raise RuntimeError(
                "No placement affinity maps were captured. Check localization blocks "
                "and capture fractions."
            )
        return np.asarray(self.sum_map / float(self.count), dtype=np.float32)


def build_spatial_key_prototypes(
    key_slice: torch.Tensor,
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    max_prototypes: int,
) -> torch.Tensor:
    """
    Compress object foreground K into spatial masked prototypes.

    key_slice: B, Nctx, H, D
    mask:      Nctx bool
    returns:   B, P, H, D
    """
    B, N, Hh, D = key_slice.shape
    if N != grid_h * grid_w:
        raise RuntimeError(f"Reference grid mismatch: {N} != {grid_h}x{grid_w}")

    key_grid = key_slice.reshape(B, grid_h, grid_w, Hh, D)
    mask_grid = mask.reshape(grid_h, grid_w)

    ar = grid_w / max(grid_h, 1)
    bins_x = max(1, int(round(math.sqrt(max_prototypes * ar))))
    bins_y = max(1, int(math.ceil(max_prototypes / bins_x)))
    bins_x = min(bins_x, grid_w)
    bins_y = min(bins_y, grid_h)

    y_edges = np.linspace(0, grid_h, bins_y + 1, dtype=int)
    x_edges = np.linspace(0, grid_w, bins_x + 1, dtype=int)

    protos = []
    for yi in range(bins_y):
        for xi in range(bins_x):
            y0, y1 = int(y_edges[yi]), int(y_edges[yi + 1])
            x0, x1 = int(x_edges[xi]), int(x_edges[xi + 1])
            m = mask_grid[y0:y1, x0:x1].reshape(-1)
            if not bool(m.any()):
                continue
            ks = key_grid[:, y0:y1, x0:x1, :, :].reshape(B, -1, Hh, D)
            protos.append(ks[:, m, :, :].mean(dim=1))  # B,H,D

    if not protos:
        fg = key_slice[:, mask, :, :]
        if fg.shape[1] == 0:
            raise RuntimeError("Object token mask contains no foreground tokens.")
        protos = [fg.mean(dim=1)]

    return torch.stack(protos[:max_prototypes], dim=1)  # B,P,H,D


def compute_target_object_relevance(
    q_target: torch.Tensor,
    k_object: torch.Tensor,
    object_mask: torch.Tensor,
    object_grid: Tuple[int, int],
    max_prototypes: int = 16,
    temperature: float = 5.0,
    query_chunk: int = 512,
) -> torch.Tensor:
    """
    Efficient relevance without materialising target_tokens x object_tokens.

    We spatially pool foreground object keys to <= max_prototypes prototypes,
    use cosine similarity, aggregate prototypes with logsumexp, and average heads.
    """
    gh, gw = object_grid
    protos = build_spatial_key_prototypes(
        k_object, object_mask, gh, gw, max_prototypes=max_prototypes
    )
    protos = F.normalize(protos.float(), dim=-1)

    chunks = []
    temperature = max(float(temperature), 1e-4)
    for s in range(0, q_target.shape[1], int(query_chunk)):
        q = F.normalize(q_target[:, s:s + query_chunk].float(), dim=-1)
        # B,Q,H,D x B,P,H,D -> B,Q,H,P
        sim = torch.einsum("bqhd,bphd->bqhp", q, protos)
        score = torch.logsumexp(sim * temperature, dim=-1) / temperature
        score = score.mean(dim=-1)  # B,Q
        chunks.append(score)

    score = torch.cat(chunks, dim=1)
    return score.mean(dim=0)  # Q


class PlacementCaptureProcessor:
    """
    Standard FLUX dual-stream attention + lightweight target/object relevance capture.

    Proposal context MUST be exactly:
        [current_scene, new_object_reference]
    so context slot 1 is unambiguously the new object.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        n_target_tokens: int,
        mask_bank: ReferenceMaskBank,
        accumulator: PlacementAccumulator,
        object_aspect_ratio: float,
        expected_steps: int,
        capture_start: float,
        capture_end: float,
        capture_every: int,
        max_prototypes: int,
        affinity_temperature: float,
        query_chunk: int,
        debug_layout: bool,
        block_index: int,
    ):
        self.n_target_tokens = int(n_target_tokens)
        self.mask_bank = mask_bank
        self.accumulator = accumulator
        self.object_aspect_ratio = float(object_aspect_ratio)
        self.expected_steps = max(int(expected_steps), 1)
        self.capture_start = float(capture_start)
        self.capture_end = float(capture_end)
        self.capture_every = max(int(capture_every), 1)
        self.max_prototypes = max(int(max_prototypes), 1)
        self.affinity_temperature = float(affinity_temperature)
        self.query_chunk = max(int(query_chunk), 1)
        self.debug_layout = bool(debug_layout)
        self.block_index = int(block_index)
        self.call_count = 0
        self._layout_logged = False

    def _should_capture(self) -> bool:
        idx = self.call_count
        frac = (idx + 0.5) / float(self.expected_steps)
        return (
            self.capture_start <= frac <= self.capture_end
            and idx % self.capture_every == 0
        )

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

        image_seq_len = k.shape[1]
        n_ctx, context_slices = infer_context_layout(
            image_seq_len=image_seq_len,
            n_target_tokens=self.n_target_tokens,
            n_context_images=2,  # scene + NEW object
        )
        scene_slice, object_slice = context_slices

        if self.debug_layout and not self._layout_logged:
            print(
                f"  [proposal-layout block {self.block_index}] "
                f"target=0:{self.n_target_tokens}, scene={scene_slice}, "
                f"new_object={object_slice}, image_seq={image_seq_len}, n_ctx={n_ctx}"
            )
            self._layout_logged = True

        text_len = 0
        if attn.added_kv_proj_dim is not None:
            if encoder_hidden_states is None:
                raise RuntimeError("Dual-stream FLUX block expected encoder_hidden_states.")
            eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
            ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
            ev = ev.unflatten(-1, (attn.heads, -1))
            text_len = encoder_hidden_states.shape[1]
            q = torch.cat([eq, q], dim=1)
            k = torch.cat([ek, k], dim=1)
            v = torch.cat([ev, v], dim=1)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        if self._should_capture():
            image_q = q[:, text_len:, :, :]
            image_k = k[:, text_len:, :, :]
            obj_s, obj_e = object_slice
            obj_mask = self.mask_bank.get(n_ctx, image_k.device)[0]
            gh, gw = infer_grid_from_tokens(n_ctx, self.object_aspect_ratio)
            score = compute_target_object_relevance(
                q_target=image_q[:, :self.n_target_tokens, :, :],
                k_object=image_k[:, obj_s:obj_e, :, :],
                object_mask=obj_mask,
                object_grid=(gh, gw),
                max_prototypes=self.max_prototypes,
                temperature=self.affinity_temperature,
                query_chunk=self.query_chunk,
            )
            self.accumulator.add(score)

        self.call_count += 1

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


class PlacementCaptureController:
    def __init__(
        self,
        transformer,
        blocks: Sequence[int],
        n_target_tokens: int,
        mask_bank: ReferenceMaskBank,
        accumulator: PlacementAccumulator,
        object_aspect_ratio: float,
        proposal_steps: int,
        capture_start: float,
        capture_end: float,
        capture_every: int,
        max_prototypes: int,
        affinity_temperature: float,
        query_chunk: int,
        debug_layout: bool,
    ):
        self.transformer = transformer
        self.original = {}
        n_blocks = len(transformer.transformer_blocks)
        invalid = [i for i in blocks if i < 0 or i >= n_blocks]
        if invalid:
            raise ValueError(f"Invalid localization blocks {invalid}; model has {n_blocks} blocks.")

        for i in blocks:
            attn = transformer.transformer_blocks[i].attn
            orig = attn.processor
            if "IPAdapter" in orig.__class__.__name__:
                raise RuntimeError(f"Block {i} uses IP-Adapter processor; E9 does not wrap it.")
            self.original[i] = orig
            proc = PlacementCaptureProcessor(
                n_target_tokens=n_target_tokens,
                mask_bank=mask_bank,
                accumulator=accumulator,
                object_aspect_ratio=object_aspect_ratio,
                expected_steps=proposal_steps,
                capture_start=capture_start,
                capture_end=capture_end,
                capture_every=capture_every,
                max_prototypes=max_prototypes,
                affinity_temperature=affinity_temperature,
                query_chunk=query_chunk,
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
# Final-pass object-reference K/V controller
# =============================================================================


class IdentityKVProcessor:
    """Scale foreground K/V of object reference slices in selected FLUX blocks."""

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        n_target_tokens: int,
        n_context_images: int,
        mask_bank: ReferenceMaskBank,
        k_scale: float,
        v_scale: float,
        debug_layout: bool,
        block_index: int,
    ):
        self.n_target_tokens = int(n_target_tokens)
        self.n_context_images = int(n_context_images)
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

        if self.n_context_images > 1 and (self.k_scale != 1.0 or self.v_scale != 1.0):
            n_ctx, slices = infer_context_layout(
                image_seq_len=k.shape[1],
                n_target_tokens=self.n_target_tokens,
                n_context_images=self.n_context_images,
            )
            scene_slice = slices[0]
            object_slices = slices[1:]
            masks = self.mask_bank.get(n_ctx, k.device)
            if len(masks) != len(object_slices):
                raise RuntimeError(
                    f"Object masks={len(masks)} != object slices={len(object_slices)}"
                )

            if self.debug_layout and not self._layout_logged:
                print(
                    f"  [final-layout block {self.block_index}] target=0:{self.n_target_tokens}, "
                    f"scene={scene_slice}, objects={object_slices}, image_seq={k.shape[1]}, n_ctx={n_ctx}"
                )
                self._layout_logged = True

            if self.k_scale != 1.0:
                k = k.clone()
            if self.v_scale != 1.0:
                v = v.clone()

            for (s, e), m in zip(object_slices, masks):
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
                raise RuntimeError(f"Block {i} uses IP-Adapter processor; E9 does not wrap it.")
            self.original[i] = orig
            proc = IdentityKVProcessor(
                n_target_tokens=n_target_tokens,
                n_context_images=n_context_images,
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
# Self-derived placement-mask construction
# =============================================================================


def proposal_difference_map(
    previous: Image.Image,
    proposal: Image.Image,
    grid_size: Tuple[int, int],
    blur_px: float = 5.0,
) -> np.ndarray:
    prev = previous.convert("RGB").resize(proposal.size, Image.Resampling.LANCZOS)
    a = np.asarray(prev, dtype=np.float32) / 255.0
    b = np.asarray(proposal.convert("RGB"), dtype=np.float32) / 255.0
    diff = np.mean(np.abs(a - b), axis=2)
    diff_img = Image.fromarray(np.uint8(np.clip(diff, 0, 1) * 255), mode="L")
    if blur_px > 0:
        diff_img = diff_img.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    gw, gh = grid_size
    small = np.asarray(diff_img.resize((gw, gh), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return robust_normalize(small)


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
    """Keep 8-connected component with largest summed localisation score."""
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


@dataclass
class PlacementMaskResult:
    attention: np.ndarray
    difference: np.ndarray
    combined: np.ndarray
    hard_grid: np.ndarray
    hard_mask: Image.Image
    soft_mask: Image.Image
    threshold: float
    grid_fraction: float


def derive_placement_mask(
    attention_map: np.ndarray,
    difference_map: np.ndarray,
    output_size: Tuple[int, int],
    source: str,
    attention_weight: float,
    mask_quantile: float,
    min_mask_frac: float,
    max_mask_frac: float,
    score_blur_tokens: float,
    close_tokens: int,
    dilate_tokens: int,
    feather_px: float,
) -> PlacementMaskResult:
    attn = robust_normalize(attention_map)
    diff = robust_normalize(difference_map)
    if attn.shape != diff.shape:
        raise ValueError(f"attention shape {attn.shape} != difference shape {diff.shape}")

    if source == "attention":
        combined = attn
    elif source == "difference":
        combined = diff
    elif source == "hybrid":
        w = float(np.clip(attention_weight, 0.0, 1.0))
        combined = w * attn + (1.0 - w) * diff
    else:
        raise ValueError(f"Unknown mask source {source}")

    combined = gaussian_blur_array(combined, score_blur_tokens)

    # Start from the requested quantile, then adapt it if connected-component
    # filtering makes the mask implausibly small or large. Lower q -> larger
    # candidate region; higher q -> smaller candidate region.
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
        # Last-resort local peak region; preferable to silently editing whole image.
        y, x = np.unravel_index(int(np.argmax(combined)), combined.shape)
        binary[max(0, y - 1): min(binary.shape[0], y + 2),
               max(0, x - 1): min(binary.shape[1], x + 2)] = True
        frac = float(binary.mean())
        warnings.warn("Placement mask collapsed; using a small region around heatmap maximum.")

    hard_grid = dilate_binary(binary, dilate_tokens)
    hard_small = Image.fromarray(np.uint8(hard_grid) * 255, mode="L")
    hard_mask = hard_small.resize(output_size, Image.Resampling.NEAREST)
    soft_mask = hard_mask
    if feather_px > 0:
        soft_mask = hard_mask.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))

    return PlacementMaskResult(
        attention=attn,
        difference=diff,
        combined=combined,
        hard_grid=hard_grid,
        hard_mask=hard_mask,
        soft_mask=soft_mask,
        threshold=threshold,
        grid_fraction=float(hard_grid.mean()),
    )


# =============================================================================
# Prompting and data loading
# =============================================================================


def build_proposal_prompt(new_name: str, previous_names: Sequence[str]) -> str:
    prompt = (
        f"Add the {new_name} exactly once in a natural, physically plausible empty location in the room. "
        f"Use the reference image to preserve the {new_name}'s shape, color, material, texture, and distinctive design. "
        "Choose a sensible placement that fits the room geometry and perspective. "
    )
    if previous_names:
        prompt += (
            f"Keep the existing {', '.join(previous_names)} where they are and do not duplicate or alter them. "
        )
    prompt += "Preserve the camera, walls, floor, lighting, furniture, and unrelated scene details."
    return prompt


def build_final_prompt(
    new_name: str,
    previous_names: Sequence[str],
    placement_mask: Image.Image,
) -> str:
    pos = position_phrase(placement_mask)
    where = f" in the {pos}" if pos else " in the room"
    prompt = (
        f"Add the {new_name} exactly once{where}, matching its reference image closely. "
        f"Preserve the {new_name}'s shape, color, material, texture, proportions, and distinctive design. "
    )
    if previous_names:
        prompt += (
            f"Keep the existing {', '.join(previous_names)} exactly where they are and preserve their appearance. "
            "Do not re-add or duplicate them. "
        )
    prompt += (
        "Preserve room geometry, camera, lighting, walls, floor, furniture, and unrelated details. "
        "Change only what is necessary to add the new object."
    )
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
    # Global scene trend proxies.
    safe_metric(compute_ssim, base, result, key="scene_ssim_global", record=record)
    safe_metric(compute_lpips, base, result, device, key="scene_lpips_global", record=record)

    # Approx unchanged background: neutralise union of all intended object regions.
    masks = [placement_masks[n] for n in active_names if n in placement_masks]
    union = union_masks(masks, result.size)
    if union is not None:
        b = neutralize_edit_region(base.resize(result.size), union)
        r = neutralize_edit_region(result, union)
        safe_metric(compute_ssim, b, r, key="background_ssim_approx", record=record)
        safe_metric(compute_lpips, b, r, device, key="background_lpips_approx", record=record)

    # Incremental collateral-change metric outside current edit region.
    if new_name in placement_masks:
        m = placement_masks[new_name]
        p = neutralize_edit_region(previous.resize(result.size), m)
        r = neutralize_edit_region(result, m)
        safe_metric(compute_ssim, p, r, key="outside_new_edit_ssim_approx", record=record)
        safe_metric(compute_lpips, p, r, device, key="outside_new_edit_lpips_approx", record=record)

    # Reference-vs-local generated object identity.
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

        # Temporal stability of already inserted objects across this new edit.
        if name != new_name:
            prev_crop = previous.resize(result.size).crop(box)
            safe_metric(
                compute_dino,
                prev_crop,
                crop,
                device,
                key=f"dino_{name}_temporal",
                record=record,
            )
            safe_metric(
                compute_lpips,
                prev_crop,
                crop,
                device,
                key=f"lpips_{name}_temporal",
                record=record,
            )


# =============================================================================
# Proposal and final passes
# =============================================================================


def run_proposal_pass(
    pipe,
    scene: Image.Image,
    new_name: str,
    previous_names: Sequence[str],
    obj_refs: Dict[str, Image.Image],
    ref_masks: Dict[str, Image.Image],
    n_target_tokens: int,
    localization_blocks: Sequence[int],
    args,
    step_seed: int,
) -> Tuple[Image.Image, np.ndarray, str]:
    prompt = build_proposal_prompt(new_name, previous_names)
    accumulator = PlacementAccumulator(
        n_target_tokens=n_target_tokens,
        target_aspect_ratio=args.width / args.height,
    )
    bank = ReferenceMaskBank(
        masks=[ref_masks[new_name]],
        aspect_ratios=[obj_refs[new_name].width / obj_refs[new_name].height],
    )

    context = [scene, obj_refs[new_name]]
    with PlacementCaptureController(
        transformer=pipe.transformer,
        blocks=localization_blocks,
        n_target_tokens=n_target_tokens,
        mask_bank=bank,
        accumulator=accumulator,
        object_aspect_ratio=obj_refs[new_name].width / obj_refs[new_name].height,
        proposal_steps=args.proposal_steps,
        capture_start=args.capture_start,
        capture_end=args.capture_end,
        capture_every=args.capture_every,
        max_prototypes=args.max_prototypes,
        affinity_temperature=args.affinity_temperature,
        query_chunk=args.query_chunk,
        debug_layout=args.debug_layout,
    ):
        proposal = pipe(
            image=context,
            prompt=prompt,
            height=args.height,
            width=args.width,
            max_area=args.height * args.width,
            num_inference_steps=args.proposal_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(args.device).manual_seed(step_seed),
        ).images[0].convert("RGB")

    return proposal, accumulator.result(), prompt


def run_final_pass(
    pipe,
    scene: Image.Image,
    refs: Sequence[str],
    active_names: Sequence[str],
    new_name: str,
    obj_refs: Dict[str, Image.Image],
    ref_masks: Dict[str, Image.Image],
    placement_mask: Image.Image,
    n_target_tokens: int,
    identity_blocks: Sequence[int],
    args,
    step_seed: int,
) -> Tuple[Image.Image, str]:
    prompt = build_final_prompt(new_name, active_names[:-1], placement_mask)
    context = [scene] + [obj_refs[n] for n in refs]

    controller = None
    if args.k_scale != 1.0 or args.v_scale != 1.0:
        bank = ReferenceMaskBank(
            masks=[ref_masks[n] for n in refs],
            aspect_ratios=[obj_refs[n].width / obj_refs[n].height for n in refs],
        )
        controller = IdentityKVController(
            transformer=pipe.transformer,
            blocks=identity_blocks,
            n_target_tokens=n_target_tokens,
            n_context_images=len(context),
            mask_bank=bank,
            k_scale=args.k_scale,
            v_scale=args.v_scale,
            debug_layout=args.debug_layout,
        )

    try:
        generated = pipe(
            image=context,
            prompt=prompt,
            height=args.height,
            width=args.width,
            max_area=args.height * args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(args.device).manual_seed(step_seed),
        ).images[0].convert("RGB")
    finally:
        if controller is not None:
            controller.remove()

    if args.protect_outside_mask:
        generated = protect_outside_soft_mask(
            previous=scene,
            generated=generated,
            soft_mask=placement_mask,
        )
    return generated, prompt


# =============================================================================
# CLI / main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scene", required=True, help="Base scene image")
    p.add_argument("--obj_dir", required=True, help="Directory with obj_<name>.png")
    p.add_argument("--out_dir", default="results/e9_self_localizing_incremental")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--device", default="cuda")

    # Generation
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--proposal_steps", type=int, default=12)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--seeds", default="42")
    p.add_argument(
        "--reference_policy",
        choices=["current", "all"],
        default="all",
        help="Final pass gets only new ref or every ref seen so far",
    )

    # Proposal affinity capture
    p.add_argument("--localization_blocks", default=DEFAULT_LOCALIZATION_BLOCKS)
    p.add_argument("--capture_start", type=float, default=0.20,
                   help="Fraction of proposal denoising after which affinity capture starts")
    p.add_argument("--capture_end", type=float, default=0.90,
                   help="Fraction of proposal denoising at which affinity capture stops")
    p.add_argument("--capture_every", type=int, default=2)
    p.add_argument("--max_prototypes", type=int, default=16)
    p.add_argument("--affinity_temperature", type=float, default=5.0)
    p.add_argument("--query_chunk", type=int, default=512)

    # Automatic placement mask
    p.add_argument("--mask_source", choices=["attention", "difference", "hybrid"], default="hybrid")
    p.add_argument("--attention_weight", type=float, default=0.70)
    p.add_argument("--mask_quantile", type=float, default=0.85,
                   help="Initial threshold quantile; 0.85 keeps strongest ~15% before component filtering")
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
        help="Restore previous scene outside self-derived soft mask",
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
    localization_blocks = parse_int_ranges(args.localization_blocks)
    identity_blocks = parse_int_ranges(args.identity_blocks)

    if not (0.0 <= args.capture_start < args.capture_end <= 1.0):
        raise ValueError("Require 0 <= capture_start < capture_end <= 1")
    if not (0.0 <= args.attention_weight <= 1.0):
        raise ValueError("attention_weight must be in [0,1]")

    os.makedirs(args.out_dir, exist_ok=True)

    pipe = load_pipe(args.model_id, args.device)
    enable_multi_context(pipe)

    n_target_tokens = packed_token_count(
        args.height, args.width, int(pipe.vae_scale_factor)
    )
    target_gh, target_gw = infer_grid_from_tokens(
        n_target_tokens, args.width / args.height
    )

    base = Image.open(args.scene).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.LANCZOS
    )
    obj_refs, obj_originals = load_objects(args.obj_dir)
    available = [n for n in OBJ_ORDER if n in obj_refs]
    if not available:
        raise FileNotFoundError(f"No obj_<name>.png references found in {args.obj_dir}")

    with open(os.path.join(args.out_dir, "config_e9.json"), "w", encoding="utf-8") as f:
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

    print("\n=== E9: SELF-LOCALIZING INCREMENTAL EDITING ===")
    print(f"objects             : {available}")
    print(f"seeds               : {seeds}")
    print(f"target token grid   : {target_gh}x{target_gw} = {n_target_tokens}")
    print(f"proposal steps      : {args.proposal_steps}")
    print(f"final steps         : {args.steps}")
    print(f"localization blocks : {localization_blocks}")
    print(f"identity blocks     : {identity_blocks}")
    print(f"mask source         : {args.mask_source}")
    print(f"attention weight    : {args.attention_weight}")
    print(f"K/V scale           : {args.k_scale}/{args.v_scale}")
    print(f"reference policy    : {args.reference_policy}")
    print(f"protect outside     : {args.protect_outside_mask}")

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
            print("  Pass A: proposal + self-localisation")

            proposal, attention_map, proposal_prompt = run_proposal_pass(
                pipe=pipe,
                scene=current_scene,
                new_name=new_name,
                previous_names=previous_names,
                obj_refs=obj_refs,
                ref_masks=ref_masks,
                n_target_tokens=n_target_tokens,
                localization_blocks=localization_blocks,
                args=args,
                step_seed=step_seed,
            )
            proposal.save(os.path.join(step_dir, "proposal.png"))

            difference_map = proposal_difference_map(
                previous=current_scene,
                proposal=proposal,
                grid_size=(target_gw, target_gh),
                blur_px=args.difference_blur_px,
            )

            placement = derive_placement_mask(
                attention_map=attention_map,
                difference_map=difference_map,
                output_size=current_scene.size,
                source=args.mask_source,
                attention_weight=args.attention_weight,
                mask_quantile=args.mask_quantile,
                min_mask_frac=args.min_placement_frac,
                max_mask_frac=args.max_placement_frac,
                score_blur_tokens=args.score_blur_tokens,
                close_tokens=args.close_tokens,
                dilate_tokens=args.dilate_tokens,
                feather_px=args.mask_feather_px,
            )

            save_float_map(
                placement.attention,
                os.path.join(step_dir, "attention_heatmap.png"),
                out_size=current_scene.size,
            )
            save_float_map(
                placement.difference,
                os.path.join(step_dir, "difference_heatmap.png"),
                out_size=current_scene.size,
            )
            save_float_map(
                placement.combined,
                os.path.join(step_dir, "hybrid_heatmap.png"),
                out_size=current_scene.size,
            )
            placement.hard_mask.save(os.path.join(step_dir, "placement_mask_hard.png"))
            placement.soft_mask.save(os.path.join(step_dir, "placement_mask_soft.png"))
            make_overlay(current_scene, placement.soft_mask).save(
                os.path.join(step_dir, "placement_overlay.png")
            )

            placement_masks[new_name] = placement.soft_mask
            print(
                f"  placement mask: source={args.mask_source}, "
                f"grid_fraction={placement.grid_fraction:.2%}, threshold={placement.threshold:.3f}, "
                f"location={position_phrase(placement.soft_mask)}"
            )

            print("  Pass B: clean-scene final identity-preserving edit")
            refs = [new_name] if args.reference_policy == "current" else active_names
            final, final_prompt = run_final_pass(
                pipe=pipe,
                scene=current_scene,
                refs=refs,
                active_names=active_names,
                new_name=new_name,
                obj_refs=obj_refs,
                ref_masks=ref_masks,
                placement_mask=placement.soft_mask,
                n_target_tokens=n_target_tokens,
                identity_blocks=identity_blocks,
                args=args,
                step_seed=step_seed,
            )
            final.save(os.path.join(step_dir, "final.png"))

            record = {
                "seed": seed,
                "step": step,
                "step_seed": step_seed,
                "new_object": new_name,
                "active_objects": list(active_names),
                "final_reference_objects": list(refs),
                "proposal_prompt": proposal_prompt,
                "final_prompt": final_prompt,
                "mask_source": args.mask_source,
                "attention_weight": args.attention_weight,
                "mask_quantile": args.mask_quantile,
                "placement_grid_fraction": placement.grid_fraction,
                "placement_threshold": placement.threshold,
                "placement_location": position_phrase(placement.soft_mask),
                "placement_bbox": mask_bbox(placement.hard_mask),
                "proposal_steps": args.proposal_steps,
                "final_steps": args.steps,
                "localization_blocks": list(localization_blocks),
                "identity_blocks": list(identity_blocks),
                "k_scale": args.k_scale,
                "v_scale": args.v_scale,
                "protect_outside_mask": args.protect_outside_mask,
            }

            add_metrics(
                record=record,
                base=base,
                previous=current_scene,
                result=final,
                active_names=active_names,
                obj_refs=obj_refs,
                placement_masks=placement_masks,
                new_name=new_name,
                device=args.device,
            )
            all_metrics.append(record)

            print(
                f"  metrics: bgSSIM≈{record.get('background_ssim_approx')} "
                f"bgLPIPS≈{record.get('background_lpips_approx')} "
                f"newDINO={record.get(f'dino_{new_name}_local')}"
            )

            # Critical: chain ONLY protected final result, never proposal.
            current_scene = final

        current_scene.save(os.path.join(seed_root, "FINAL.png"))

    metrics_path = os.path.join(args.out_dir, "metrics_e9.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nE9 complete.")
    print(f"Metrics: {metrics_path}")
    print(f"Images : {args.out_dir}")


if __name__ == "__main__":
    main()
