"""
E8-V3: Incremental multi-reference identity preservation for FLUX.1-Kontext
=============================================================================

What this changes relative to E8-v2
-----------------------------------
1) No global monkey-patch of torch.nn.functional.scaled_dot_product_attention.
   Selected FLUX double-stream attention blocks receive a custom attention
   processor instead.
2) The processor works on the IMAGE stream before text/image concatenation, so
   object-token indexing is not confused by the text-token prefix used by
   current Diffusers FLUX attention.
3) Token counts are validated at runtime. n_ctx_per is inferred from the actual
   hidden-state sequence instead of being blindly hard-coded to 4096.
4) K/V modification clones once and scales slices in-place (no repeated torch.cat).
5) Reference masks use alpha when available; grey/white thresholding is only a
   fallback. Bad masks fail loudly by default.
6) Supports both:
      --mode cumulative : every condition starts from the original base image
      --mode chained    : step k edits the result from step k-1
7) Optional per-object spatial edit masks (mask_<name>.png) can:
      - steer prompt location coarsely
      - protect pixels outside the new edit area by feathered compositing
      - enable local object-identity and approximate background/leakage metrics
8) Global whole-scene metrics are explicitly named GLOBAL/PROXY so they are not
   mistaken for true background/object-local metrics.

Important
---------
This file still relies on your existing utils.py functions:
    load_pipe, enable_multi_context,
    compute_ssim, compute_lpips, compute_dino, compute_clip_i

The multi-context contract assumed here is:
  image = [scene_context, object_ref_1, ..., object_ref_N]
produces an image-token stream laid out as:
  [target/noisy tokens | scene-context tokens | obj1 tokens | ... | objN tokens]

The code ASSERTS this layout by checking that, after the known target token count
is removed, the remaining image tokens divide evenly across all context images.
If your enable_multi_context() uses a different ordering, adjust
ObjectReferenceKVProcessor._infer_layout().

Recommended first experiments
-----------------------------
A. Cumulative vanilla:
   python e8_v3_incremental_identity.py ... --mode cumulative --method vanilla

B. Cumulative K-scale:
   python e8_v3_incremental_identity.py ... --mode cumulative --method k_scale --k_scale 1.5

C. True incremental vanilla, all previous refs retained:
   python e8_v3_incremental_identity.py ... --mode chained --method vanilla --reference_policy all

D. True incremental + reference K scaling:
   python e8_v3_incremental_identity.py ... --mode chained --method k_scale --k_scale 1.5 --reference_policy all

If spatial masks are available, add:
   --edit_masks_dir path/to/masks --protect_outside_edit

Each edit mask must be named mask_<object>.png, white = region allowed to change,
black = protected region.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

# Existing project utilities --------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from utils import (  # noqa: E402
    load_pipe,
    enable_multi_context,
    compute_ssim,
    compute_lpips,
    compute_dino,
    compute_clip_i,
)


# Current Diffusers FLUX uses these two helpers. We keep a fallback to native
# PyTorch SDPA so the script gives a useful error/fallback on older installs.
try:
    from diffusers.models.attention_dispatch import dispatch_attention_fn
except Exception:
    dispatch_attention_fn = None

try:
    from diffusers.models.embeddings import apply_rotary_emb
except Exception as exc:
    raise ImportError(
        "Could not import diffusers.models.embeddings.apply_rotary_emb. "
        "Use a Diffusers version compatible with FLUX.1-Kontext."
    ) from exc


OBJ_ORDER = ["bicycle", "vase", "ball", "chair", "lamp", "plant", "backpack"]
DEFAULT_BLOCKS = tuple(range(13, 19))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def parse_int_ranges(spec: str) -> List[int]:
    """Parse e.g. '13-18,5,7' -> [5, 7, 13, 14, 15, 16, 17, 18]."""
    out = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            lo, hi = min(a, b), max(a, b)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def parse_seeds(spec: str) -> List[int]:
    seeds = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def packed_token_count(height: int, width: int, vae_scale_factor: int) -> int:
    """
    FLUX packs 2x2 latent patches. For dimensions divisible by 2*vae_sf:
      token_h = height / (2*vae_sf)
      token_w = width  / (2*vae_sf)
    """
    mult = vae_scale_factor * 2
    if height % mult != 0 or width % mult != 0:
        raise ValueError(
            f"height/width must be divisible by {mult}; got {height}x{width}."
        )
    return (height // mult) * (width // mult)


def infer_grid_from_tokens(n_tokens: int, aspect_ratio: float) -> Tuple[int, int]:
    """
    Infer (grid_h, grid_w) from token count using the factor pair whose w/h is
    closest to the supplied aspect ratio.
    """
    if n_tokens <= 0:
        raise ValueError(f"Invalid token count: {n_tokens}")

    best = None
    root = int(math.sqrt(n_tokens))
    for h in range(1, root + 1):
        if n_tokens % h != 0:
            continue
        w = n_tokens // h
        for gh, gw in ((h, w), (w, h)):
            err = abs(math.log(max(gw / gh, 1e-8)) - math.log(max(aspect_ratio, 1e-8)))
            if best is None or err < best[0]:
                best = (err, gh, gw)

    if best is None:
        raise RuntimeError(f"Could not factor token count {n_tokens}.")
    return best[1], best[2]


def resize_rgb(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def union_masks(masks: Sequence[Image.Image], size: Tuple[int, int]) -> Optional[Image.Image]:
    if not masks:
        return None
    union = np.zeros((size[1], size[0]), dtype=np.float32)
    for mask in masks:
        arr = np.asarray(mask.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        union = np.maximum(union, arr)
    return Image.fromarray(np.uint8(np.clip(union, 0, 1) * 255), mode="L")


def neutralize_edit_region(
    img: Image.Image,
    edit_mask: Image.Image,
    neutral_value: int = 128,
) -> Image.Image:
    """
    Replace the editable region with the same neutral fill. Comparing two such
    images gives an APPROXIMATE outside-edit metric with existing global metric
    functions. It is not mathematically identical to a truly masked LPIPS/SSIM.
    """
    rgb = img.convert("RGB")
    mask = edit_mask.convert("L").resize(rgb.size, Image.Resampling.BILINEAR)
    neutral = Image.new("RGB", rgb.size, (neutral_value,) * 3)
    return Image.composite(neutral, rgb, mask)


def mask_bbox(mask: Image.Image, threshold: int = 16) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(mask.convert("L"))
    ys, xs = np.where(arr > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(
    box: Tuple[int, int, int, int],
    size: Tuple[int, int],
    frac: float = 0.05,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    dx = int(round(w * frac))
    dy = int(round(h * frac))
    W, H = size
    return max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy)


def position_phrase(mask: Optional[Image.Image]) -> Optional[str]:
    """Convert edit-mask centroid to a coarse natural-language location."""
    if mask is None:
        return None
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    total = arr.sum()
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


def protect_outside_region(
    previous: Image.Image,
    generated: Image.Image,
    edit_mask: Image.Image,
    feather_px: float,
) -> Image.Image:
    """
    Keep generated pixels only inside edit_mask; preserve previous image outside.
    A small feather avoids a hard seam.
    """
    prev = previous.convert("RGB")
    gen = generated.convert("RGB").resize(prev.size, Image.Resampling.LANCZOS)
    mask = edit_mask.convert("L").resize(prev.size, Image.Resampling.BILINEAR)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
    return Image.composite(gen, prev, mask)


# ---------------------------------------------------------------------------
# Reference foreground masks
# ---------------------------------------------------------------------------

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
    """
    Return a binary L mask, white=object foreground.

    Priority:
      1) meaningful alpha channel
      2) grey/white background heuristic

    Unlike E8-v2, a suspicious mask raises by default instead of silently using
    the centre 50% crop. Set allow_fallback=True only for exploratory runs.
    """
    rgba = obj_img.convert("RGBA")
    rgba_arr = np.asarray(rgba)
    alpha = rgba_arr[..., 3]

    # Use alpha only if it contains actual transparency variation.
    if alpha.min() < 250 and np.mean(alpha < 250) > 0.001:
        mask = alpha > alpha_threshold
        source = "alpha"
    else:
        rgb = rgba_arr[..., :3].astype(np.int16)
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
            f"Suspicious reference foreground mask from {source}: foreground={frac:.3%}, "
            f"expected between {min_frac:.1%} and {max_frac:.1%}."
        )
        if not allow_fallback:
            raise ValueError(msg + " Fix the object alpha/background or pass --allow_mask_fallback.")
        warnings.warn(msg + " Falling back to a centre-50% mask.")
        mask[:] = False
        H, W = mask.shape
        mask[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = True

    mask_img = Image.fromarray(np.uint8(mask) * 255, mode="L")
    if dilate_px > 0:
        # MaxFilter kernel must be odd.
        kernel = 2 * int(dilate_px) + 1
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=kernel))
    return mask_img


class ReferenceMaskBank:
    """Build and cache object token masks for whatever context-token grid appears."""

    def __init__(
        self,
        masks: Sequence[Image.Image],
        context_aspect_ratio: float,
    ):
        self.masks = [m.convert("L") for m in masks]
        self.context_aspect_ratio = float(context_aspect_ratio)
        self._cpu_cache: Dict[int, List[torch.Tensor]] = {}
        self._device_cache: Dict[Tuple[int, str], List[torch.Tensor]] = {}

    def _build_cpu(self, n_ctx_per: int) -> List[torch.Tensor]:
        if n_ctx_per in self._cpu_cache:
            return self._cpu_cache[n_ctx_per]

        gh, gw = infer_grid_from_tokens(n_ctx_per, self.context_aspect_ratio)
        out = []
        for mask in self.masks:
            # NEAREST retains binary identity; reference mask was already dilated.
            arr = np.asarray(mask.resize((gw, gh), Image.Resampling.NEAREST)) > 127
            out.append(torch.from_numpy(arr.reshape(-1).copy()).bool())
        self._cpu_cache[n_ctx_per] = out
        print(f"  [mask-bank] context token grid inferred as {gh}x{gw} = {n_ctx_per}")
        return out

    def get(self, n_ctx_per: int, device: torch.device) -> List[torch.Tensor]:
        key = (n_ctx_per, str(device))
        if key not in self._device_cache:
            self._device_cache[key] = [m.to(device=device) for m in self._build_cpu(n_ctx_per)]
        return self._device_cache[key]


# ---------------------------------------------------------------------------
# FLUX attention processor
# ---------------------------------------------------------------------------

def _project_qkv(attn, hidden_states, encoder_hidden_states=None):
    """Local equivalent of current Diffusers FLUX _get_qkv_projections()."""
    if getattr(attn, "fused_projections", False):
        if not hasattr(attn, "to_qkv"):
            raise RuntimeError("Attention reports fused_projections=True but has no to_qkv.")
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None:
            if not hasattr(attn, "to_added_qkv"):
                raise RuntimeError(
                    "Fused FLUX attention with encoder states requires to_added_qkv; "
                    "cannot safely install custom processor."
                )
            encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _run_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    backend=None,
    parallel_config=None,
):
    """
    query/key/value are (B, L, H, D), matching current Diffusers FLUX.
    """
    if dispatch_attention_fn is not None:
        return dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=backend,
            parallel_config=parallel_config,
        )

    # Compatibility fallback for older Diffusers.
    q = query.transpose(1, 2)  # B,H,L,D
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    return out.transpose(1, 2)  # B,L,H,D


class ObjectReferenceKVProcessor:
    """
    FLUX dual-stream attention processor that scales only object-reference K/V.

    Expected IMAGE hidden-state layout from enable_multi_context():
      [ target | scene_context | obj_ref_1 | ... | obj_ref_N ]

    We modify the image-only K/V BEFORE text K/V are concatenated. This avoids
    needing to guess a text prefix offset and matches the current FLUX processor
    structure in Diffusers.

    Note: K scaling is an affinity-magnitude intervention, not a guaranteed
    positive attention bias. Keep k_scale modest and ablate it.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        n_target_tokens: int,
        n_context_images: int,
        mask_bank: ReferenceMaskBank,
        k_scale: float = 1.5,
        v_scale: float = 1.0,
        debug_layout: bool = False,
    ):
        self.n_target_tokens = int(n_target_tokens)
        self.n_context_images = int(n_context_images)  # scene + N object refs
        self.mask_bank = mask_bank
        self.k_scale = float(k_scale)
        self.v_scale = float(v_scale)
        self.debug_layout = bool(debug_layout)
        self._layout_logged = False

    def _infer_layout(self, image_seq_len: int) -> Tuple[int, List[Tuple[int, int]]]:
        if image_seq_len <= self.n_target_tokens:
            raise RuntimeError(
                f"Image sequence ({image_seq_len}) is not larger than target token count "
                f"({self.n_target_tokens}). Multi-context layout is not what this script expects."
            )

        context_total = image_seq_len - self.n_target_tokens
        if self.n_context_images <= 0 or context_total % self.n_context_images != 0:
            raise RuntimeError(
                "Cannot infer equal context-image token slices. "
                f"image_seq={image_seq_len}, target={self.n_target_tokens}, "
                f"context_total={context_total}, n_context_images={self.n_context_images}. "
                "Check enable_multi_context() ordering/preprocessing."
            )

        n_ctx = context_total // self.n_context_images
        # context slot 0 = scene; object reference j is context slot j+1.
        obj_slices = []
        for obj_idx in range(self.n_context_images - 1):
            s = self.n_target_tokens + (1 + obj_idx) * n_ctx
            e = s + n_ctx
            obj_slices.append((s, e))

        if self.debug_layout and not self._layout_logged:
            print(
                "  [attn-layout] "
                f"target=0:{self.n_target_tokens}, "
                f"scene={self.n_target_tokens}:{self.n_target_tokens+n_ctx}, "
                f"objects={obj_slices}, image_seq={image_seq_len}"
            )
            self._layout_logged = True

        return n_ctx, obj_slices

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        query, key, value, encoder_query, encoder_key, encoder_value = _project_qkv(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Modify image-stream K/V before text is concatenated.
        if self.n_context_images > 1 and (self.k_scale != 1.0 or self.v_scale != 1.0):
            n_ctx, obj_slices = self._infer_layout(key.shape[1])
            obj_masks = self.mask_bank.get(n_ctx, key.device)

            if len(obj_masks) != len(obj_slices):
                raise RuntimeError(
                    f"Mask count ({len(obj_masks)}) != object slice count ({len(obj_slices)})."
                )

            if self.k_scale != 1.0:
                key = key.clone()
            if self.v_scale != 1.0:
                value = value.clone()

            for (s, e), mask in zip(obj_slices, obj_masks):
                if mask.numel() != (e - s):
                    raise RuntimeError(
                        f"Object mask has {mask.numel()} tokens, expected {e-s}."
                    )

                # B,L,H,D. Scale foreground tokens only; leave ref background = 1.
                if self.k_scale != 1.0:
                    scale_k = 1.0 + (self.k_scale - 1.0) * mask.to(key.dtype)
                    key[:, s:e, :, :] *= scale_k[None, :, None, None]

                if self.v_scale != 1.0:
                    scale_v = 1.0 + (self.v_scale - 1.0) * mask.to(value.dtype)
                    value[:, s:e, :, :] *= scale_v[None, :, None, None]

        # Text/context projection handling follows current Diffusers FluxAttnProcessor.
        if attn.added_kv_proj_dim is not None:
            if encoder_hidden_states is None:
                raise RuntimeError("Dual-stream FLUX block expected encoder_hidden_states, got None.")

            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden = _run_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            backend=getattr(self, "_attention_backend", None),
            parallel_config=getattr(self, "_parallel_config", None),
        )
        hidden = hidden.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            text_len = encoder_hidden_states.shape[1]
            enc_out, img_out = hidden.split_with_sizes(
                [text_len, hidden.shape[1] - text_len], dim=1
            )
            img_out = attn.to_out[0](img_out.contiguous())
            img_out = attn.to_out[1](img_out)
            enc_out = attn.to_add_out(enc_out.contiguous())
            return img_out, enc_out

        return hidden


class ObjectKVController:
    """Install custom processors only on selected FLUX double-stream blocks."""

    def __init__(
        self,
        transformer,
        block_indices: Sequence[int],
        n_target_tokens: int,
        n_context_images: int,
        mask_bank: ReferenceMaskBank,
        k_scale: float,
        v_scale: float,
        debug_layout: bool,
    ):
        self.transformer = transformer
        self.block_indices = list(block_indices)
        self.original_processors = {}

        n_blocks = len(transformer.transformer_blocks)
        invalid = [i for i in self.block_indices if i < 0 or i >= n_blocks]
        if invalid:
            raise ValueError(f"Invalid block indices {invalid}; model has {n_blocks} double-stream blocks.")

        for i in self.block_indices:
            attn = transformer.transformer_blocks[i].attn
            orig = attn.processor
            orig_name = orig.__class__.__name__
            if "IPAdapter" in orig_name:
                raise RuntimeError(
                    f"Block {i} uses {orig_name}; this V3 processor does not preserve IP-Adapter branches."
                )
            self.original_processors[i] = orig
            new_proc = ObjectReferenceKVProcessor(
                n_target_tokens=n_target_tokens,
                n_context_images=n_context_images,
                mask_bank=mask_bank,
                k_scale=k_scale,
                v_scale=v_scale,
                debug_layout=debug_layout,
            )
            # Preserve any backend choice already configured on the original processor.
            new_proc._attention_backend = getattr(orig, "_attention_backend", None)
            new_proc._parallel_config = getattr(orig, "_parallel_config", None)
            attn.set_processor(new_proc)

    def remove(self):
        for i, proc in self.original_processors.items():
            self.transformer.transformer_blocks[i].attn.set_processor(proc)
        self.original_processors.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False


# ---------------------------------------------------------------------------
# Prompting and data loading
# ---------------------------------------------------------------------------

def build_cumulative_prompt(names: Sequence[str], edit_masks: Dict[str, Image.Image]) -> str:
    chunks = []
    for name in names:
        pos = position_phrase(edit_masks.get(name))
        if pos:
            chunks.append(f"Place the {name} naturally in the {pos} of the room, exactly once.")
        else:
            chunks.append(f"Place the {name} naturally in the room, exactly once.")
    return (
        " ".join(chunks)
        + " Use each corresponding reference image only for that object's identity."
        + " Preserve each object's shape, color, material, texture, and distinctive design."
        + " Keep the rest of the room unchanged and do not duplicate any object."
    )


def build_incremental_prompt(
    new_name: str,
    previous_names: Sequence[str],
    edit_masks: Dict[str, Image.Image],
) -> str:
    pos = position_phrase(edit_masks.get(new_name))
    where = f" in the {pos} of the room" if pos else " naturally in the room"
    previous = ", ".join(previous_names)

    prompt = (
        f"Add the {new_name}{where}, exactly once, using its reference image for identity. "
        f"Preserve the {new_name}'s exact shape, color, material, texture, and distinctive design. "
    )
    if previous_names:
        prompt += (
            f"Keep the existing {previous} exactly where they are and preserve their appearance. "
            "Do not re-add or duplicate them. "
        )
    prompt += (
        "Preserve the room geometry, camera, lighting, walls, floor, furniture, and all unrelated details. "
        "Change only what is necessary to add the new object."
    )
    return prompt


def load_objects(obj_dir: str) -> Tuple[Dict[str, Image.Image], Dict[str, Image.Image]]:
    refs_for_pipe = {}
    originals = {}
    for name in OBJ_ORDER:
        path = os.path.join(obj_dir, f"obj_{name}.png")
        if os.path.isfile(path):
            original = Image.open(path).copy()
            originals[name] = original
            refs_for_pipe[name] = original.convert("RGB")
    return refs_for_pipe, originals


def load_edit_masks(edit_masks_dir: Optional[str], names: Iterable[str], scene_size: Tuple[int, int]):
    masks = {}
    if not edit_masks_dir:
        return masks
    for name in names:
        path = os.path.join(edit_masks_dir, f"mask_{name}.png")
        if os.path.isfile(path):
            masks[name] = Image.open(path).convert("L").resize(scene_size, Image.Resampling.BILINEAR)
        else:
            warnings.warn(f"No edit mask found for {name}: {path}")
    return masks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    edit_masks: Dict[str, Image.Image],
    new_name: str,
    device: str,
):
    # Whole-image scene metrics are useful trend proxies, but NOT true background metrics.
    safe_metric(compute_ssim, base, result, key="scene_ssim_global", record=record)
    safe_metric(compute_lpips, base, result, device, key="scene_lpips_global", record=record)

    if not edit_masks:
        # Explicitly label whole-image object comparisons as proxies.
        for name in active_names:
            safe_metric(
                compute_dino,
                obj_refs[name],
                result,
                device,
                key=f"dino_{name}_proxy_global",
                record=record,
            )
            safe_metric(
                compute_clip_i,
                obj_refs[name],
                result,
                device,
                key=f"clip_{name}_proxy_global",
                record=record,
            )
        return

    # Approximate unchanged-background metric: neutralize all intended edit areas
    # in both images, then use existing global metrics.
    active_masks = [edit_masks[n] for n in active_names if n in edit_masks]
    union = union_masks(active_masks, result.size)
    if union is not None:
        base_bg = neutralize_edit_region(base.resize(result.size), union)
        result_bg = neutralize_edit_region(result, union)
        safe_metric(
            compute_ssim,
            base_bg,
            result_bg,
            key="background_ssim_approx",
            record=record,
        )
        safe_metric(
            compute_lpips,
            base_bg,
            result_bg,
            device,
            key="background_lpips_approx",
            record=record,
        )

    # Approximate edit leakage: neutralize ONLY the current edit region and
    # compare previous -> result. Higher SSIM/lower LPIPS means less collateral change.
    if new_name in edit_masks:
        cur_mask = edit_masks[new_name]
        prev_outside = neutralize_edit_region(previous.resize(result.size), cur_mask)
        res_outside = neutralize_edit_region(result, cur_mask)
        safe_metric(
            compute_ssim,
            prev_outside,
            res_outside,
            key="outside_new_edit_ssim_approx",
            record=record,
        )
        safe_metric(
            compute_lpips,
            prev_outside,
            res_outside,
            device,
            key="outside_new_edit_lpips_approx",
            record=record,
        )

    # Local identity: crop each object's intended edit region from the generated scene.
    # This is substantially better than ref-vs-whole-room, though still not a perfect
    # segmentation-based identity metric.
    for name in active_names:
        mask = edit_masks.get(name)
        if mask is None:
            continue
        box = mask_bbox(mask)
        if box is None:
            continue
        box = expand_bbox(box, result.size, frac=0.05)
        crop = result.crop(box)
        ref = obj_refs[name]
        safe_metric(
            compute_dino,
            ref,
            crop,
            device,
            key=f"dino_{name}_local",
            record=record,
        )
        safe_metric(
            compute_clip_i,
            ref,
            crop,
            device,
            key=f"clip_{name}_local",
            record=record,
        )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scene", required=True, help="Base room image")
    p.add_argument("--obj_dir", required=True, help="Directory with obj_<name>.png")
    p.add_argument("--out_dir", default="results/e8_v3_incremental_identity")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-Kontext-dev")

    p.add_argument("--mode", choices=["cumulative", "chained"], default="chained")
    p.add_argument("--method", choices=["vanilla", "k_scale"], default="k_scale")
    p.add_argument(
        "--reference_policy",
        choices=["current", "all"],
        default="all",
        help="In chained mode, pass only the new object ref or all refs seen so far",
    )

    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seeds", default="42", help="Comma-separated, e.g. 42,123,456")
    p.add_argument("--blocks", default="13-18", help="Double-stream block indices/ranges")

    p.add_argument("--k_scale", type=float, default=1.5)
    p.add_argument("--v_scale", type=float, default=1.0)

    p.add_argument("--grey_tol", type=int, default=20)
    p.add_argument("--white_tol", type=int, default=20)
    p.add_argument("--min_mask_frac", type=float, default=0.02)
    p.add_argument("--max_mask_frac", type=float, default=0.95)
    p.add_argument("--mask_dilate_px", type=int, default=1)
    p.add_argument("--allow_mask_fallback", action="store_true")

    p.add_argument(
        "--edit_masks_dir",
        default=None,
        help="Optional directory with mask_<name>.png; white=allowed edit region",
    )
    p.add_argument(
        "--protect_outside_edit",
        action="store_true",
        help="In chained mode, composite previous pixels back outside current edit mask",
    )
    p.add_argument("--feather_px", type=float, default=8.0)
    p.add_argument("--debug_layout", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def run_condition(
    pipe,
    scene_context: Image.Image,
    object_names_for_refs: Sequence[str],
    active_names: Sequence[str],
    new_name: str,
    obj_refs: Dict[str, Image.Image],
    ref_fg_masks: Dict[str, Image.Image],
    edit_masks: Dict[str, Image.Image],
    n_target_tokens: int,
    blocks: Sequence[int],
    args,
    seed: int,
    prompt: str,
) -> Image.Image:
    context_imgs = [scene_context] + [obj_refs[n] for n in object_names_for_refs]

    controller = None
    if args.method == "k_scale":
        mask_bank = ReferenceMaskBank(
            masks=[ref_fg_masks[n] for n in object_names_for_refs],
            context_aspect_ratio=scene_context.width / scene_context.height,
        )
        controller = ObjectKVController(
            transformer=pipe.transformer,
            block_indices=blocks,
            n_target_tokens=n_target_tokens,
            n_context_images=len(context_imgs),
            mask_bank=mask_bank,
            k_scale=args.k_scale,
            v_scale=args.v_scale,
            debug_layout=args.debug_layout,
        )

    try:
        result = pipe(
            image=context_imgs,
            prompt=prompt,
            height=args.height,
            width=args.width,
            max_area=args.height * args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(args.device).manual_seed(seed),
        ).images[0]
    finally:
        if controller is not None:
            controller.remove()

    # Strong, simple spatial preservation option for the actual incremental task.
    # This is image-space protection, not latent-space inpainting.
    if (
        args.mode == "chained"
        and args.protect_outside_edit
        and new_name in edit_masks
    ):
        result = protect_outside_region(
            previous=scene_context,
            generated=result,
            edit_mask=edit_masks[new_name],
            feather_px=args.feather_px,
        )

    return result.convert("RGB")


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    blocks = parse_int_ranges(args.blocks)

    if args.method == "vanilla" and (args.k_scale != 1.5 or args.v_scale != 1.0):
        warnings.warn("k_scale/v_scale are ignored for --method vanilla.")

    if args.mode != "chained" and args.protect_outside_edit:
        warnings.warn("--protect_outside_edit only applies to --mode chained.")

    os.makedirs(args.out_dir, exist_ok=True)

    pipe = load_pipe(args.model_id, args.device)
    enable_multi_context(pipe)

    vae_sf = int(pipe.vae_scale_factor)
    n_target_tokens = packed_token_count(args.height, args.width, vae_sf)

    base = Image.open(args.scene).convert("RGB")
    # Metrics/protective compositing are easiest to interpret at output resolution.
    base_metric = base.resize((args.width, args.height), Image.Resampling.LANCZOS)

    obj_refs, obj_originals = load_objects(args.obj_dir)
    available = [n for n in OBJ_ORDER if n in obj_refs]
    if not available:
        raise FileNotFoundError(f"No obj_<name>.png files found in {args.obj_dir}")

    edit_masks = load_edit_masks(args.edit_masks_dir, available, base_metric.size)

    # Reference-foreground masks used specifically for K/V selection.
    ref_fg_masks = {}
    for name in available:
        ref_fg_masks[name] = extract_reference_foreground_mask(
            obj_originals[name],
            grey_tol=args.grey_tol,
            white_tol=args.white_tol,
            min_frac=args.min_mask_frac,
            max_frac=args.max_mask_frac,
            dilate_px=args.mask_dilate_px,
            allow_fallback=args.allow_mask_fallback,
        )
        frac = np.asarray(ref_fg_masks[name], dtype=np.float32).mean() / 255.0
        print(f"reference mask {name:>10s}: {frac:6.2%} foreground")

    print("\n=== E8-V3 configuration ===")
    print(f"mode              : {args.mode}")
    print(f"method            : {args.method}")
    print(f"reference_policy  : {args.reference_policy}")
    print(f"objects           : {available}")
    print(f"seeds             : {seeds}")
    print(f"blocks            : {blocks}")
    print(f"target tokens     : {n_target_tokens}")
    print(f"k_scale / v_scale : {args.k_scale} / {args.v_scale}")
    print(f"edit masks        : {sorted(edit_masks.keys()) if edit_masks else 'none'}")
    print(f"protect outside   : {args.protect_outside_edit}")

    all_metrics = []

    for seed in seeds:
        seed_dir = os.path.join(
            args.out_dir,
            args.mode,
            args.method,
            f"seed_{seed}",
        )
        os.makedirs(seed_dir, exist_ok=True)

        current_scene = base_metric.copy()

        for k, new_name in enumerate(available, start=1):
            active_names = available[:k]

            if args.mode == "cumulative":
                scene_context = base_metric
                refs = active_names
                prompt = build_cumulative_prompt(active_names, edit_masks)
                previous_for_metric = base_metric
            else:
                scene_context = current_scene
                refs = [new_name] if args.reference_policy == "current" else active_names
                prompt = build_incremental_prompt(new_name, active_names[:-1], edit_masks)
                previous_for_metric = current_scene

            print(f"\n[seed={seed}] [{k}/{len(available)}] +{new_name}")
            print(f"  scene source : {'original base' if args.mode == 'cumulative' else 'previous result'}")
            print(f"  refs         : {refs}")
            print(f"  prompt       : {prompt}")

            result = run_condition(
                pipe=pipe,
                scene_context=scene_context,
                object_names_for_refs=refs,
                active_names=active_names,
                new_name=new_name,
                obj_refs=obj_refs,
                ref_fg_masks=ref_fg_masks,
                edit_masks=edit_masks,
                n_target_tokens=n_target_tokens,
                blocks=blocks,
                args=args,
                seed=seed,
                prompt=prompt,
            )

            label = f"step_{k:02d}_add_{new_name}"
            result_path = os.path.join(seed_dir, f"{label}.png")
            result.save(result_path)

            record = {
                "seed": seed,
                "step": k,
                "new_object": new_name,
                "active_objects": list(active_names),
                "reference_objects": list(refs),
                "mode": args.mode,
                "method": args.method,
                "reference_policy": args.reference_policy,
                "k_scale": args.k_scale if args.method == "k_scale" else 1.0,
                "v_scale": args.v_scale if args.method == "k_scale" else 1.0,
                "blocks": list(blocks) if args.method == "k_scale" else [],
                "prompt": prompt,
                "result_path": result_path,
            }

            add_metrics(
                record=record,
                base=base_metric,
                previous=previous_for_metric,
                result=result,
                active_names=active_names,
                obj_refs=obj_refs,
                edit_masks=edit_masks,
                new_name=new_name,
                device=args.device,
            )
            all_metrics.append(record)

            # Only true chained mode feeds the result into the next edit.
            if args.mode == "chained":
                current_scene = result

            # Concise progress metrics.
            print(
                f"  scene_global: SSIM={record.get('scene_ssim_global')} "
                f"LPIPS={record.get('scene_lpips_global')}"
            )
            if "background_ssim_approx" in record:
                print(
                    f"  background≈ : SSIM={record.get('background_ssim_approx')} "
                    f"LPIPS={record.get('background_lpips_approx')}"
                )
            local_key = f"dino_{new_name}_local"
            if local_key in record:
                print(
                    f"  new identity: DINO={record.get(local_key)} "
                    f"CLIP={record.get(f'clip_{new_name}_local')}"
                )

        # Save final chained scene separately for convenience.
        if args.mode == "chained":
            current_scene.save(os.path.join(seed_dir, "FINAL.png"))

    metrics_path = os.path.join(
        args.out_dir,
        f"metrics_{args.mode}_{args.method}.json",
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nDone. Metrics: {metrics_path}")
    print(f"Images: {os.path.join(args.out_dir, args.mode, args.method)}")


if __name__ == "__main__":
    main()