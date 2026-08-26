"""
E12: ControlNet Pose + Kontext Finalizer
========================================

Pipeline
--------
This script implements a 3-stage incremental editing pipeline:

1) Scene-first placement planning
   - Uses FLUX.1 Kontext on the current scene ONLY (no object reference image).
   - Generates several generic placement proposals for the object category.
   - Selects the best proposal and derives a placement mask from proposal-vs-scene difference.

2) ControlNet pose/structure generation
   - Converts the selected planner proposal into a structural control image (Canny by default).
   - Uses FLUX ControlNet Inpaint with the selected placement mask to generate a
     scene-consistent *posed* object in the masked region.
   - This stage focuses on WHERE and HOW the object should appear.

3) Kontext identity finalization
   - Uses FLUX Kontext Inpaint with the true object reference image to refine the masked region
     so that the inserted object matches the reference identity (colors, materials, design,
     structure) while preserving the pose/location suggested by stage 2.
   - This stage focuses on WHICH exact object it is.

Why E12?
--------
E11 still asked Kontext to infer pose adaptation mostly from a mask/prompt. Here we split the roles:

    placement mask  = where
    ControlNet      = how it should be posed/shaped
    Kontext ref     = which exact object it should become

Dependencies
------------
Required runtime packages typically include:
  - diffusers (recent version with FLUX pipelines)
  - torch
  - transformers
  - Pillow
Optional but recommended:
  - cv2 (OpenCV) for true Canny edge detection
  - utils.py in the same directory for metrics (compute_ssim/lpips/dino/clip_i)

This file is syntax-checked, but end-to-end execution depends on your environment,
model access, and installed package versions.
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
from PIL import Image, ImageFilter, ImageOps

import torch

try:
    from diffusers import (
        FluxKontextPipeline,
        FluxKontextInpaintPipeline,
        FluxControlNetInpaintPipeline,
        FluxControlNetModel,
    )
except Exception as exc:
    raise ImportError(
        "Could not import required FLUX diffusers pipelines/classes. "
        "Install a recent diffusers version with FLUX support."
    ) from exc

# Optional metrics from your project utils.py
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from utils import compute_ssim, compute_lpips, compute_dino, compute_clip_i  # type: ignore
    HAS_METRICS = True
except Exception:
    HAS_METRICS = False
    compute_ssim = compute_lpips = compute_dino = compute_clip_i = None

OBJ_ORDER = ["bicycle", "vase", "ball", "chair", "lamp", "plant", "backpack"]
DEFAULT_KONTEXT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_FLUX_BASE = "black-forest-labs/FLUX.1-dev"
DEFAULT_CONTROLNET = "InstantX/FLUX.1-dev-Controlnet-Canny"


# =============================================================================
# Generic utilities
# =============================================================================


def parse_seeds(spec: str) -> List[int]:
    vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one seed is required.")
    return vals



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



def make_overlay(scene: Image.Image, mask: Image.Image, strength: float = 0.45) -> Image.Image:
    base = np.asarray(scene.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L").resize(scene.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tint = base.copy()
    tint[..., 0] = 255.0
    alpha = (m * float(strength))[..., None]
    out = base * (1.0 - alpha) + tint * alpha
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)), mode="RGB")



def blur_mask(mask: Image.Image, radius: float) -> Image.Image:
    out = mask.convert("L")
    if radius > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return out



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



def protect_outside_soft_mask(previous: Image.Image, generated: Image.Image, soft_mask: Image.Image) -> Image.Image:
    prev = previous.convert("RGB")
    gen = generated.convert("RGB").resize(prev.size, Image.Resampling.LANCZOS)
    mask = soft_mask.convert("L").resize(prev.size, Image.Resampling.BILINEAR)
    return Image.composite(gen, prev, mask)



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
# Mask morphology
# =============================================================================



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
            f"Suspicious foreground mask ({source}): {frac:.2%}; expected [{min_frac:.1%}, {max_frac:.1%}]."
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


# =============================================================================
# Planner / proposal masks
# =============================================================================


@dataclass
class PlacementMaskResult:
    score_map: np.ndarray
    hard_grid: np.ndarray
    hard_mask: Image.Image
    soft_mask: Image.Image
    threshold: float
    grid_fraction: float


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
        f"Do not focus on exact identity; use a generic but realistic {new_name}. "
        f"Make sure the {new_name} is physically supported, respects room depth and perspective, and does not float. "
    )
    if previous_names:
        prompt += f"Keep the existing {', '.join(previous_names)} where they are and do not duplicate or alter them. "
    prompt += "Preserve the room layout, camera, furniture, walls, floor, and lighting apart from the planned new object."
    return prompt



def build_controlnet_pose_prompt(new_name: str, previous_names: Sequence[str], placement_mask: Image.Image) -> str:
    pos = position_phrase(placement_mask)
    where = f" in the {pos}" if pos else " in the selected room region"
    prompt = (
        f"Insert one realistic {new_name}{where}. "
        f"Follow the structural control image closely for pose, outline, viewpoint, and perspective. "
        f"The {new_name} should fit naturally into the room, with plausible support/contact and realistic scale. "
        f"Preserve the surrounding room and change only the masked region. "
    )
    if previous_names:
        prompt += f"Keep the existing {', '.join(previous_names)} unmodified. "
    return prompt



def build_kontext_final_prompt(new_name: str, previous_names: Sequence[str], placement_mask: Image.Image) -> str:
    pos = position_phrase(placement_mask)
    where = f" in the {pos}" if pos else " in the selected room region"
    prompt = (
        f"Refine the masked object so that it becomes the exact {new_name} from the reference image while keeping its current pose, viewpoint, location, apparent scale, and perspective{where}. "
        f"Preserve the reference object's exact colors, materials, texture, distinctive design, identity-defining geometry, and characteristic parts. "
        f"Do not render the object as a flat sticker. Clean up artefacts and integrate it naturally with room lighting and contact shadows. "
        f"Change only the masked region and preserve the surrounding scene. "
    )
    if previous_names:
        prompt += f"Keep the existing {', '.join(previous_names)} where they are and preserve their appearance. "
    return prompt



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



def _mask_centroid01(mask: Image.Image) -> Tuple[float, float]:
    a = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    total = float(a.sum())
    if total <= 1e-8:
        return 0.5, 0.5
    yy, xx = np.indices(a.shape)
    return (
        float((xx * a).sum() / total) / max(a.shape[1] - 1, 1),
        float((yy * a).sum() / total) / max(a.shape[0] - 1, 1),
    )



def _mask_iou(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a.convert("L").resize(b.size, Image.Resampling.NEAREST)) > 32
    bb = np.asarray(b.convert("L")) > 32
    u = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / u) if u else 0.0



def _border_fraction(mask: Image.Image, border_frac: float = 0.03) -> float:
    a = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    H, W = a.shape
    by, bx = max(1, int(H * border_frac)), max(1, int(W * border_frac))
    border = np.zeros_like(a, dtype=bool)
    border[:by] = True
    border[-by:] = True
    border[:, :bx] = True
    border[:, -bx:] = True
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

    size_score = gauss(frac, 0.08, 0.10)
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
    dist_center = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / math.sqrt(0.5 ** 2 + 0.5 ** 2)
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
# Edge / structure control image
# =============================================================================



def pil_find_edges(img: Image.Image) -> Image.Image:
    x = img.convert("L")
    x = ImageOps.autocontrast(x)
    x = x.filter(ImageFilter.FIND_EDGES)
    x = ImageOps.autocontrast(x)
    arr = np.asarray(x, dtype=np.float32)
    thr = max(16.0, float(np.quantile(arr, 0.75)))
    arr = (arr >= thr).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")



def canny_like_edges(img: Image.Image, low: int = 80, high: int = 180) -> Image.Image:
    try:
        import cv2  # type: ignore
        arr = np.asarray(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=int(low), threshold2=int(high))
        return Image.fromarray(edges, mode="L")
    except Exception:
        return pil_find_edges(img)



def build_control_image_from_planner(
    planner_proposal: Image.Image,
    current_scene: Image.Image,
    placement_mask: Image.Image,
    mode: str = "canny",
    edge_low: int = 80,
    edge_high: int = 180,
    blur_mask_radius: float = 0.0,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Returns:
        control_image_rgb: image given to ControlNet
        edge_map_raw: raw edge image (L)
        edge_map_masked: masked edge image (L)
    """
    if mode != "canny":
        raise ValueError(f"Unsupported control mode {mode}; currently only 'canny' is implemented.")

    raw_edges = canny_like_edges(planner_proposal, low=edge_low, high=edge_high)
    base_mask = placement_mask.convert("L").resize(planner_proposal.size, Image.Resampling.BILINEAR)
    if blur_mask_radius > 0:
        base_mask = base_mask.filter(ImageFilter.GaussianBlur(radius=float(blur_mask_radius)))

    # Restrict structure mainly to the planned object region.
    raw = np.asarray(raw_edges, dtype=np.float32) / 255.0
    m = np.asarray(base_mask, dtype=np.float32) / 255.0
    masked = np.uint8(np.clip(raw * m, 0.0, 1.0) * 255)
    edge_masked = Image.fromarray(masked, mode="L")
    control_image = edge_masked.convert("RGB")
    return control_image, raw_edges, edge_masked


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
    if not HAS_METRICS:
        return

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
# Pipeline loading
# =============================================================================


class PipelineBundle:
    def __init__(self, planner_pipe, control_pipe, final_pipe):
        self.planner_pipe = planner_pipe
        self.control_pipe = control_pipe
        self.final_pipe = final_pipe



def maybe_enable_offload(pipe, device: str, cpu_offload: bool):
    if cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            return pipe
        except Exception as exc:
            warnings.warn(f"Could not enable CPU offload: {exc}")
    try:
        pipe.to(device)
    except Exception:
        pass
    return pipe



def load_pipelines(args) -> PipelineBundle:
    dtype = getattr(torch, args.torch_dtype)

    planner_pipe = FluxKontextPipeline.from_pretrained(args.kontext_model_id, torch_dtype=dtype)
    planner_pipe = maybe_enable_offload(planner_pipe, args.device, args.cpu_offload)

    controlnet = FluxControlNetModel.from_pretrained(args.controlnet_model_id, torch_dtype=dtype)
    control_pipe = FluxControlNetInpaintPipeline.from_pretrained(
        args.flux_base_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    control_pipe = maybe_enable_offload(control_pipe, args.device, args.cpu_offload)

    final_pipe = FluxKontextInpaintPipeline.from_pretrained(args.kontext_model_id, torch_dtype=dtype)
    final_pipe = maybe_enable_offload(final_pipe, args.device, args.cpu_offload)

    return PipelineBundle(planner_pipe, control_pipe, final_pipe)


# =============================================================================
# Three-stage incremental pipeline
# =============================================================================



def run_scene_first_placement_search(
    planner_pipe,
    scene: Image.Image,
    new_name: str,
    previous_names: Sequence[str],
    previous_masks: Sequence[Image.Image],
    grid_size: Tuple[int, int],
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
        proposal = planner_pipe(
            image=scene,
            prompt=prompt,
            height=args.height,
            width=args.width,
            max_area=args.height * args.width,
            num_inference_steps=args.planner_steps,
            guidance_scale=args.planner_guidance_scale,
            generator=torch.Generator(args.device).manual_seed(cand_seed),
        ).images[0].convert("RGB")

        diff_map = proposal_difference_map(
            previous=scene,
            proposal=proposal,
            grid_size=grid_size,
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



def run_controlnet_pose_stage(
    control_pipe,
    scene: Image.Image,
    planner: PlannerCandidate,
    new_name: str,
    previous_names: Sequence[str],
    args,
    pose_seed: int,
    step_dir: str,
) -> Tuple[Image.Image, Image.Image, Image.Image, str]:
    control_image, raw_edges, masked_edges = build_control_image_from_planner(
        planner_proposal=planner.proposal,
        current_scene=scene,
        placement_mask=planner.placement.soft_mask,
        mode=args.control_mode,
        edge_low=args.canny_low,
        edge_high=args.canny_high,
        blur_mask_radius=args.control_mask_blur_px,
    )
    control_image.save(os.path.join(step_dir, "control_image.png"))
    raw_edges.save(os.path.join(step_dir, "control_edges_raw.png"))
    masked_edges.save(os.path.join(step_dir, "control_edges_masked.png"))

    mask_for_pipe = blur_mask(planner.placement.soft_mask, args.inpaint_mask_blur_px)
    mask_for_pipe.save(os.path.join(step_dir, "inpaint_mask_blurred.png"))

    prompt = build_controlnet_pose_prompt(new_name, previous_names, planner.placement.soft_mask)
    posed_scene = control_pipe(
        prompt=prompt,
        image=scene,
        mask_image=mask_for_pipe,
        control_image=control_image,
        control_guidance_start=args.control_guidance_start,
        control_guidance_end=args.control_guidance_end,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        strength=args.control_strength,
        height=args.height,
        width=args.width,
        num_inference_steps=args.pose_steps,
        guidance_scale=args.pose_guidance_scale,
        generator=torch.Generator(args.device).manual_seed(pose_seed),
    ).images[0].convert("RGB")
    return posed_scene, control_image, mask_for_pipe, prompt



def run_kontext_final_stage(
    final_pipe,
    current_scene: Image.Image,
    posed_scene: Image.Image,
    object_ref: Image.Image,
    new_name: str,
    previous_names: Sequence[str],
    mask_for_pipe: Image.Image,
    args,
    final_seed: int,
) -> Tuple[Image.Image, str]:
    # Build a base image that already contains the pose proposal inside the region.
    proposal_composite = protect_outside_soft_mask(current_scene, posed_scene, mask_for_pipe)
    prompt = build_kontext_final_prompt(new_name, previous_names, mask_for_pipe)

    final_scene = final_pipe(
        prompt=prompt,
        image=proposal_composite,
        mask_image=mask_for_pipe,
        image_reference=object_ref,
        strength=args.final_strength,
        height=args.height,
        width=args.width,
        num_inference_steps=args.final_steps,
        guidance_scale=args.final_guidance_scale,
        generator=torch.Generator(args.device).manual_seed(final_seed),
    ).images[0].convert("RGB")
    return final_scene, prompt


# =============================================================================
# CLI
# =============================================================================



def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O
    p.add_argument("--scene", required=True, help="Base scene image path")
    p.add_argument("--obj_dir", required=True, help="Directory with obj_<name>.png files")
    p.add_argument("--out_dir", default="results/e12_controlnet_pose_kontext_finalizer")

    # Models / device
    p.add_argument("--kontext_model_id", default=DEFAULT_KONTEXT_MODEL)
    p.add_argument("--flux_base_model_id", default=DEFAULT_FLUX_BASE)
    p.add_argument("--controlnet_model_id", default=DEFAULT_CONTROLNET)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cpu_offload", action="store_true")

    # Canvas / seeds
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seeds", default="42")

    # Stage 1: planner
    p.add_argument("--placement_candidates", type=int, default=4, choices=[1, 2, 3, 4])
    p.add_argument("--planner_steps", type=int, default=10)
    p.add_argument("--planner_guidance_scale", type=float, default=2.5)
    p.add_argument("--difference_blur_px", type=float, default=5.0)
    p.add_argument("--mask_quantile", type=float, default=0.85)
    p.add_argument("--min_placement_frac", type=float, default=0.01)
    p.add_argument("--max_placement_frac", type=float, default=0.35)
    p.add_argument("--score_blur_tokens", type=float, default=1.2)
    p.add_argument("--close_tokens", type=int, default=1)
    p.add_argument("--dilate_tokens", type=int, default=2)
    p.add_argument("--mask_feather_px", type=float, default=12.0)
    p.add_argument("--score_locality_weight", type=float, default=0.85)
    p.add_argument("--score_support_weight", type=float, default=0.55)
    p.add_argument("--score_overlap_weight", type=float, default=1.00)
    p.add_argument("--score_border_weight", type=float, default=0.45)
    p.add_argument("--score_center_weight", type=float, default=0.30)
    p.add_argument("--center_deadzone", type=float, default=0.18)

    # Stage 2: controlnet pose
    p.add_argument("--control_mode", default="canny", choices=["canny"])
    p.add_argument("--canny_low", type=int, default=80)
    p.add_argument("--canny_high", type=int, default=180)
    p.add_argument("--control_mask_blur_px", type=float, default=0.0)
    p.add_argument("--control_strength", type=float, default=1.0)
    p.add_argument("--pose_steps", type=int, default=24)
    p.add_argument("--pose_guidance_scale", type=float, default=3.5)
    p.add_argument("--controlnet_conditioning_scale", type=float, default=0.85)
    p.add_argument("--control_guidance_start", type=float, default=0.15)
    p.add_argument("--control_guidance_end", type=float, default=0.85)
    p.add_argument("--inpaint_mask_blur_px", type=float, default=10.0)

    # Stage 3: Kontext finalizer
    p.add_argument("--final_strength", type=float, default=0.75)
    p.add_argument("--final_steps", type=int, default=28)
    p.add_argument("--final_guidance_scale", type=float, default=2.5)

    # Reference mask extraction (for diagnostics / future metrics)
    p.add_argument("--grey_tol", type=int, default=20)
    p.add_argument("--white_tol", type=int, default=20)
    p.add_argument("--min_ref_mask_frac", type=float, default=0.02)
    p.add_argument("--max_ref_mask_frac", type=float, default=0.95)
    p.add_argument("--ref_mask_dilate_px", type=int, default=1)
    p.add_argument("--allow_mask_fallback", action="store_true")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================



def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "config_e12.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    base = Image.open(args.scene).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    obj_refs, obj_originals = load_objects(args.obj_dir)
    available = [n for n in OBJ_ORDER if n in obj_refs]
    if not available:
        raise FileNotFoundError(f"No obj_<name>.png references found in {args.obj_dir}")

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

    print("Loading pipelines ...")
    pipes = load_pipelines(args)

    # Planner mask grid: use a modest fixed grid, independent of latent internals.
    planner_grid = (64, 64)

    print("\n=== E12: CONTROLNET POSE + KONTEXT FINALIZER ===")
    print(f"objects   : {available}")
    print(f"seeds     : {seeds}")
    print(f"size      : {args.width}x{args.height}")
    print(f"planner   : {args.planner_steps} steps")
    print(f"pose      : {args.pose_steps} steps, control={args.control_mode}")
    print(f"finalizer : {args.final_steps} steps")
    print(f"metrics   : {'enabled' if HAS_METRICS else 'disabled'}")

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
            print("  Stage 1: scene-first planner")
            planner = run_scene_first_placement_search(
                planner_pipe=pipes.planner_pipe,
                scene=current_scene,
                new_name=new_name,
                previous_names=previous_names,
                previous_masks=[placement_masks[n] for n in previous_names if n in placement_masks],
                grid_size=planner_grid,
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
            print(f"    selected candidate: {planner.name}  score={planner.score:.3f}")

            print("  Stage 2: ControlNet pose/structure")
            pose_seed = planner.seed + 50000
            posed_scene, control_image, mask_for_pipe, pose_prompt = run_controlnet_pose_stage(
                control_pipe=pipes.control_pipe,
                scene=current_scene,
                planner=planner,
                new_name=new_name,
                previous_names=previous_names,
                args=args,
                pose_seed=pose_seed,
                step_dir=step_dir,
            )
            posed_scene.save(os.path.join(step_dir, "pose_controlnet_scene.png"))
            with open(os.path.join(step_dir, "pose_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(pose_prompt)

            print("  Stage 3: Kontext identity finalization")
            final_seed = pose_seed + 17
            final_scene, final_prompt = run_kontext_final_stage(
                final_pipe=pipes.final_pipe,
                current_scene=current_scene,
                posed_scene=posed_scene,
                object_ref=obj_refs[new_name],
                new_name=new_name,
                previous_names=previous_names,
                mask_for_pipe=mask_for_pipe,
                args=args,
                final_seed=final_seed,
            )
            final_scene.save(os.path.join(step_dir, "final.png"))
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
                "control_mode": args.control_mode,
                "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
                "control_guidance_start": args.control_guidance_start,
                "control_guidance_end": args.control_guidance_end,
                "control_strength": args.control_strength,
                "final_strength": args.final_strength,
            }
            add_metrics(
                record=record,
                base=base,
                previous=current_scene,
                result=final_scene,
                active_names=active_names,
                obj_refs=obj_refs,
                placement_masks=placement_masks,
                new_name=new_name,
                device=args.device,
            )
            with open(os.path.join(step_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            all_metrics.append(record)

            current_scene = final_scene

        current_scene.save(os.path.join(seed_root, "FINAL.png"))

    with open(os.path.join(args.out_dir, "metrics_e12.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nDone. Results saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
