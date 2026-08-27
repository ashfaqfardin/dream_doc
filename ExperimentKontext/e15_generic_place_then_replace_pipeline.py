"""
E15: Generic Place-Then-Replace Pipeline
========================================

Purpose
-------
A reusable, object-agnostic FLUX.1 workflow:

    BASE SCENE
        -> generate a GENERIC instance of the requested object at a natural place
        -> detect/mask the newly inserted generic object
        -> replace that generic object with the user's REFERENCE object
        -> optional identity refinement
        -> repeat for the next object

This is intentionally NOT hard-coded for bicycles, vases, bags, rooms, or any
specific image. The user supplies an arbitrary object name and reference image.

Important design choices
------------------------
1. Placement is decided BEFORE the reference image is introduced.
   This prevents the reference pose/background from deciding where the object goes.

2. Placement is not a random coordinate.
   FLUX.1-Kontext is asked to make several natural-placement proposals. Each proposal
   is evaluated for: object detection confidence, locality of the edit, border clipping,
   overlap with previously inserted objects, and implausibly large/small edits.

3. The generic inserted object becomes the pose/placement anchor.
   The second stage masks that object and asks Kontext Inpaint to replace it with the
   exact reference object while preserving the generic object's location, footprint,
   viewpoint, pose, scale, and perspective as much as the model permits.

4. CLIP 77-token truncation is avoided by keeping `prompt` deliberately SHORT and
   moving detailed instructions into FLUX's `prompt_2` (T5) input.

5. Multiple objects are supported sequentially through a JSON manifest.

6. By default, every placement proposal is generated from the unchanged base
   scene, then only its masked generic anchor is transferred onto the cumulative
   edited scene for reference inpainting. This avoids cumulative planning drift.

Input modes
-----------
A) Existing base image:
   --base_image room.png

B) Generate a base scene first:
   --base_prompt "A modern apartment living room ..."

Object specification
--------------------
Recommended: JSON manifest, for example:

[
  {
    "name": "bicycle",
    "reference": "references/my_bicycle.png",
    "placement_hint": "Place it naturally without blocking the walkway.",
    "pose_instruction": "Orient it so its rear side is toward the window."
  },
  {
    "name": "ceramic vase",
    "reference": "references/my_vase.png"
  }
]

Run with:
  --objects_json objects.json

For one object, convenience arguments are also supported:
  --object_name bicycle --reference_image my_bicycle.png

Masking
-------
Fallback `--mask_backend auto`:
  - try GroundingDINO zero-shot object detection on the GENERIC proposal;
  - derive a tighter mask from proposal-vs-pre-edit difference restricted to the
    detected box;
  - if detection fails, fall back to a difference-component mask.

Default `--mask_backend sam2` uses the GroundingDINO box as a SAM 2 prompt
and selects a precise silhouette using SAM confidence plus changed-pixel agreement.

No object-specific detector classes are hard-coded.

Models
------
Placement / generic insertion:
    black-forest-labs/FLUX.1-Kontext-dev

Reference replacement / refinement:
    black-forest-labs/FLUX.1-Kontext-dev via FluxKontextInpaintPipeline

Optional base generation:
    black-forest-labs/FLUX.1-dev via FluxPipeline

Notes
-----
- "Exact identity with arbitrary large viewpoint change" remains a model capability
  limitation, not something code can mathematically guarantee.
- For research comparisons, keep seeds fixed and inspect the saved candidate images,
  masks, generic proposal, replacement, and final output separately.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps

try:
    from diffusers import FluxKontextInpaintPipeline, FluxKontextPipeline, FluxPipeline
except Exception as exc:
    raise ImportError(
        "E15 needs a recent diffusers version containing FluxPipeline, "
        "FluxKontextPipeline and FluxKontextInpaintPipeline."
    ) from exc


DEFAULT_KONTEXT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_DETECTOR = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM2_MODEL = "facebook/sam2-hiera-small"


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class ObjectJob:
    name: str
    reference: str
    placement_hint: str = ""
    pose_instruction: str = ""
    reference_flip: str = "none"
    pose_source: str = "generic"


@dataclass
class Detection:
    score: float
    box: Tuple[int, int, int, int]
    label: str


@dataclass
class CandidateResult:
    index: int
    seed: int
    proposal: Image.Image
    hard_mask: Image.Image
    soft_mask: Image.Image
    score: float
    detection_score: float
    locality: float
    overlap_previous: float
    border_fraction: float
    area_fraction: float
    box: Optional[Tuple[int, int, int, int]]
    prompt_short: str
    prompt_long: str
    valid: bool


# =============================================================================
# Basic helpers
# =============================================================================


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def torch_dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]



def generator_for(device: str, seed: int) -> torch.Generator:
    # torch.Generator("cuda") is ideal when actually running on CUDA. If a custom
    # device string is unsupported, fall back to CPU generator rather than crash.
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        return torch.Generator().manual_seed(int(seed))



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def save_json(data: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def clamp_box(box: Tuple[int, int, int, int], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in box]
    W, H = size
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))
    return x0, y0, x1, y1



def expand_box(box: Tuple[int, int, int, int], size: Tuple[int, int], frac: float) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    dx = int(round(max(1, bw) * float(frac)))
    dy = int(round(max(1, bh) * float(frac)))
    return clamp_box((x0 - dx, y0 - dy, x1 + dx, y1 + dy), size)



def mask_bbox(mask: Image.Image, threshold: int = 16) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(mask.convert("L"))
    ys, xs = np.where(arr > threshold)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1



def mask_area_fraction(mask: Image.Image, threshold: int = 32) -> float:
    return float((np.asarray(mask.convert("L")) > threshold).mean())



def mask_iou(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a.convert("L").resize(b.size, Image.Resampling.NEAREST)) > 32
    bb = np.asarray(b.convert("L")) > 32
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(aa, bb).sum() / union)



def border_fraction(mask: Image.Image, border_frac: float = 0.03) -> float:
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    H, W = arr.shape
    by = max(1, int(round(H * border_frac)))
    bx = max(1, int(round(W * border_frac)))
    border = np.zeros_like(arr, dtype=bool)
    border[:by, :] = True
    border[-by:, :] = True
    border[:, :bx] = True
    border[:, -bx:] = True
    total = float(arr.sum())
    if total <= 1e-8:
        return 1.0
    return float(arr[border].sum() / total)



def union_masks(masks: Sequence[Image.Image], size: Tuple[int, int]) -> Optional[Image.Image]:
    if not masks:
        return None
    arr = np.zeros((size[1], size[0]), dtype=np.float32)
    for m in masks:
        cur = np.asarray(m.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        arr = np.maximum(arr, cur)
    return Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), mode="L")



def make_overlay(scene: Image.Image, mask: Image.Image, strength: float = 0.45) -> Image.Image:
    base = np.asarray(scene.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L").resize(scene.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tint = base.copy()
    tint[..., 0] = 255.0
    alpha = (m * float(strength))[..., None]
    out = base * (1.0 - alpha) + tint * alpha
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)), mode="RGB")



def protect_outside_mask(previous: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    prev = previous.convert("RGB")
    gen = generated.convert("RGB").resize(prev.size, Image.Resampling.LANCZOS)
    m = mask.convert("L").resize(prev.size, Image.Resampling.BILINEAR)
    return Image.composite(gen, prev, m)


def composite_masked_edit(
    background: Image.Image,
    edited_source: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Transfer only a planned local edit onto the cumulative scene."""
    background = background.convert("RGB")
    edited_source = edited_source.convert("RGB").resize(background.size, Image.Resampling.LANCZOS)
    mask = mask.convert("L").resize(background.size, Image.Resampling.BILINEAR)
    return Image.composite(edited_source, background, mask)



def flip_reference(img: Image.Image, mode: str) -> Image.Image:
    mode = (mode or "none").lower()
    if mode == "none":
        return img
    if mode == "horizontal":
        return ImageOps.mirror(img)
    if mode == "vertical":
        return ImageOps.flip(img)
    if mode == "both":
        return ImageOps.flip(ImageOps.mirror(img))
    raise ValueError(f"Unknown reference_flip={mode}")


# =============================================================================
# Prompt design: short CLIP prompt + detailed T5 prompt_2
# =============================================================================


def insertion_prompts(job: ObjectJob, candidate_index: int, candidate_count: int) -> Tuple[str, str]:
    # Deliberately short so CLIP does not hit its 77-token limit.
    short = f"Add one realistic {job.name} naturally to this scene."

    diversity = ""
    if candidate_count > 1:
        diversity = (
            f"This is placement proposal {candidate_index} of {candidate_count}. "
            "Choose a genuinely plausible placement rather than defaulting to the image center. "
        )

    long = (
        f"Add exactly one generic {job.name}. {diversity}"
        "Examine the scene and choose a visually natural, physically plausible location. "
        "Infer the appropriate supporting surface or floor contact, depth, scale, perspective, "
        "orientation, occlusion and free space from the image itself. Avoid floating, clipping, "
        "blocking important scene elements, or arbitrary placement. Preserve everything unrelated. "
    )
    if job.placement_hint:
        long += f"Placement preference: {job.placement_hint.strip()} "
    if job.pose_instruction:
        # Pose instruction is allowed at GENERIC stage because the generic object should establish pose.
        long += f"Pose/orientation requirement: {job.pose_instruction.strip()} "
    return short, long.strip()



def replacement_prompts(job: ObjectJob) -> Tuple[str, str]:
    short = f"Replace the masked {job.name} with the reference {job.name}."
    if job.pose_source == "reference":
        pose_text = (
            "Adopt the object's pose, orientation and visible-part arrangement from the reference image, "
            "while adapting that pose to the target scene's placement, scale and perspective. "
        )
    else:
        pose_text = (
            "Preserve the generic object's orientation, viewpoint and pose while keeping its current "
            "placement, footprint, approximate size and scene perspective. "
        )
    long = (
        f"Replace only the masked generic {job.name} with the exact object shown in the reference image. "
        + pose_text
        + "Transfer identity from the reference: colors, materials, texture, distinctive design, "
        "proportions and characteristic parts. Preserve the surrounding scene. "
    )
    if job.pose_instruction:
        long += f"Keep this pose/orientation requirement: {job.pose_instruction.strip()} "
    return short, long.strip()



def refinement_prompts(job: ObjectJob) -> Tuple[str, str]:
    short = f"Refine the masked {job.name} to match the reference."
    pose_text = (
        "Keep the reference-derived pose and current scene placement. "
        if job.pose_source == "reference"
        else "Keep its current location, pose, viewpoint, scale and perspective. "
    )
    long = (
        f"Refine only the masked {job.name}. " + pose_text +
        "Improve identity agreement with the reference image, especially colors, materials, texture, design, "
        "proportions and distinctive parts. Remove artifacts and preserve everything outside the mask."
    )
    return short, long



def base_prompts(base_prompt: str) -> Tuple[str, str]:
    # User prompt may itself be long. Keep CLIP input compact; pass full detail to T5.
    short = "Generate a realistic scene matching the description."
    return short, base_prompt.strip()


# =============================================================================
# Difference mask extraction
# =============================================================================


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



def image_difference_map(before: Image.Image, after: Image.Image, blur_px: float) -> np.ndarray:
    a = np.asarray(before.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(after.convert("RGB").resize(before.size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    d = np.mean(np.abs(b - a), axis=2)
    img = Image.fromarray(np.uint8(np.clip(d, 0, 1) * 255), mode="L")
    if blur_px > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    return robust_normalize(np.asarray(img, dtype=np.float32) / 255.0)



def best_connected_component(binary: np.ndarray, weights: np.ndarray) -> np.ndarray:
    H, W = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    best_pixels: List[Tuple[int, int]] = []
    best_score = -1.0
    neigh = [
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
            score = 0.0
            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                score += float(weights[cy, cx])
                for dy, dx in neigh:
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < H and 0 <= xx < W and binary[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
            if score > best_score:
                best_score = score
                best_pixels = pixels
    out = np.zeros_like(binary, dtype=bool)
    for y, x in best_pixels:
        out[y, x] = True
    return out



def morph_close(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary.astype(bool)
    size = 2 * int(radius) + 1
    img = Image.fromarray(np.uint8(binary) * 255, mode="L")
    img = img.filter(ImageFilter.MaxFilter(size=size))
    img = img.filter(ImageFilter.MinFilter(size=size))
    return np.asarray(img) > 127



def dilate_mask_image(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.convert("L")
    size = 2 * int(radius) + 1
    return mask.convert("L").filter(ImageFilter.MaxFilter(size=size))


def box_mask(box: Tuple[int, int, int, int], size: Tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(clamp_box(box, size), fill=255)
    return mask


def replacement_envelope(
    object_mask: Image.Image,
    size: Tuple[int, int],
    expand_frac: float,
    feather_px: float,
) -> Tuple[Image.Image, Image.Image]:
    """Grow the SAM silhouette organically without exposing its whole bounding box."""
    bbox = mask_bbox(object_mask)
    if bbox is None:
        raise RuntimeError("Cannot build a replacement envelope from an empty object mask")
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    radius = max(1, int(round(max(bw, bh) * float(expand_frac))))
    hard = dilate_mask_image(object_mask.resize(size, Image.Resampling.NEAREST), radius)
    soft = hard
    if feather_px > 0:
        soft = hard.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
    return hard, soft


def isolate_reference_object(
    reference: Image.Image,
    object_name: str,
    detector: Optional[GenericDetector],
    segmenter: Optional[SAM2BoxSegmenter],
    args,
) -> Tuple[Image.Image, Image.Image]:
    """Segment and crop the reference so its background cannot condition replacement."""
    reference = reference.convert("RGB")
    try:
        det = detector.detect(reference, object_name, args.detection_threshold) if detector else None
    except Exception as exc:
        warnings.warn(f"Reference detector failed for '{object_name}': {exc}")
        det = None
    if det is None or segmenter is None:
        warnings.warn(
            f"Could not SAM-segment reference '{object_name}'; using the unmodified reference."
        )
        return reference, Image.new("L", reference.size, 255)

    # SAM's difference-aware ranking needs an evidence map. Inside a trusted semantic
    # detection box, a uniform map leaves selection primarily to SAM confidence.
    evidence = np.ones((reference.height, reference.width), dtype=np.float32)
    mask = segmenter.segment(reference, det.box, evidence)
    bbox = mask_bbox(mask)
    area = mask_area_fraction(mask)
    if bbox is None or not args.min_ref_mask_frac <= area <= args.max_ref_mask_frac:
        warnings.warn(
            f"Suspicious reference SAM mask for '{object_name}' ({area:.2%}); "
            "using the unmodified reference."
        )
        return reference, Image.new("L", reference.size, 255)

    bbox = expand_box(bbox, reference.size, args.reference_crop_expand_frac)
    crop = reference.crop(bbox)
    crop_mask = mask.crop(bbox)
    # A neutral background minimizes accidental transfer of the reference scene.
    neutral = Image.new("RGB", crop.size, (127, 127, 127))
    isolated = Image.composite(crop, neutral, crop_mask)
    return isolated, mask



def difference_mask(
    before: Image.Image,
    after: Image.Image,
    args,
    restrict_box: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """Return hard_mask, soft_mask, raw difference map.

    If restrict_box is supplied, the threshold/component search is restricted to that
    detected object region, which prevents unrelated scene changes from becoming the mask.
    """
    diff = image_difference_map(before, after, blur_px=args.diff_blur_px)
    H, W = diff.shape
    allowed = np.ones((H, W), dtype=bool)

    if restrict_box is not None:
        x0, y0, x1, y1 = expand_box(restrict_box, (W, H), args.detect_box_expand_frac)
        allowed[:] = False
        allowed[y0:y1, x0:x1] = True

    vals = diff[allowed]
    if vals.size == 0:
        raise RuntimeError("No pixels available for difference-mask extraction.")

    q = float(np.clip(args.diff_quantile, 0.50, 0.995))
    threshold = float(np.quantile(vals, q))
    binary = (diff >= threshold) & allowed
    binary = morph_close(binary, args.mask_close_px)
    binary = best_connected_component(binary, diff)

    # Adapt threshold if component is pathological.
    for _ in range(8):
        frac = float(binary.mean())
        if args.min_mask_frac <= frac <= args.max_mask_frac:
            break
        if frac < args.min_mask_frac and q > 0.50:
            q = max(0.50, q - 0.05)
        elif frac > args.max_mask_frac and q < 0.995:
            q = min(0.995, q + 0.03)
        threshold = float(np.quantile(vals, q))
        binary = (diff >= threshold) & allowed
        binary = morph_close(binary, args.mask_close_px)
        binary = best_connected_component(binary, diff)

    if not binary.any():
        # Fallback to a box mask if detection exists, otherwise a local peak.
        if restrict_box is not None:
            x0, y0, x1, y1 = expand_box(restrict_box, (W, H), 0.03)
            binary[y0:y1, x0:x1] = True
        else:
            y, x = np.unravel_index(int(np.argmax(diff)), diff.shape)
            r = max(4, int(round(min(W, H) * 0.02)))
            binary[max(0, y-r):min(H, y+r+1), max(0, x-r):min(W, x+r+1)] = True

    hard = Image.fromarray(np.uint8(binary) * 255, mode="L")
    hard = dilate_mask_image(hard, args.mask_dilate_px)
    soft = hard
    if args.mask_feather_px > 0:
        soft = hard.filter(ImageFilter.GaussianBlur(radius=float(args.mask_feather_px)))
    return hard, soft, diff


# =============================================================================
# Generic zero-shot detection (optional)
# =============================================================================


class GenericDetector:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError("transformers is required for GroundingDINO detection") from exc

        # Keep detector on CPU by default so it does not compete with FLUX for VRAM.
        detector_device: Any = -1
        if self.device.startswith("cuda"):
            detector_device = 0
        self._pipe = pipeline(
            task="zero-shot-object-detection",
            model=self.model_id,
            device=detector_device,
        )

    def detect_all(self, image: Image.Image, label: str, threshold: float) -> List[Detection]:
        self._load()
        assert self._pipe is not None
        try:
            preds = self._pipe(image, candidate_labels=[label, f"{label}."])
        except TypeError:
            # Some transformers versions use text_queries.
            preds = self._pipe(image, text_queries=[label])

        detections: List[Detection] = []
        for p in preds:
            score = float(p.get("score", 0.0))
            if score < threshold:
                continue
            b = p.get("box", {})
            if isinstance(b, dict):
                box = (
                    int(round(float(b.get("xmin", b.get("x_min", 0))))),
                    int(round(float(b.get("ymin", b.get("y_min", 0))))),
                    int(round(float(b.get("xmax", b.get("x_max", image.width))))),
                    int(round(float(b.get("ymax", b.get("y_max", image.height))))),
                )
            else:
                box = tuple(int(round(float(x))) for x in b)  # type: ignore
            box = clamp_box(box, image.size)
            detections.append(Detection(score=score, box=box, label=str(p.get("label", label))))
        return sorted(detections, key=lambda detection: detection.score, reverse=True)

    def detect(self, image: Image.Image, label: str, threshold: float) -> Optional[Detection]:
        detections = self.detect_all(image, label, threshold)
        return detections[0] if detections else None


class SAM2BoxSegmenter:
    """Lazily loaded SAM 2 predictor driven by a GroundingDINO box."""

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise RuntimeError(
                "SAM 2 is required for --mask_backend sam2. Install it with "
                "`pip install git+https://github.com/facebookresearch/sam2`."
            ) from exc
        self._predictor = SAM2ImagePredictor.from_pretrained(
            self.model_id, device=self.device
        )

    def segment(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
        difference: np.ndarray,
    ) -> Image.Image:
        self._load()
        assert self._predictor is not None
        self._predictor.set_image(np.asarray(image.convert("RGB")))
        masks, scores, _ = self._predictor.predict(
            box=np.asarray(box, dtype=np.float32), multimask_output=True
        )
        masks = np.asarray(masks, dtype=bool)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.shape[0] == 0:
            raise RuntimeError("SAM 2 returned no masks")

        combined_scores = []
        for mask, sam_score in zip(masks, scores):
            changed = float(difference[mask].mean()) if mask.any() else 0.0
            combined_scores.append(float(sam_score) + 0.35 * changed)
        best = masks[int(np.argmax(combined_scores))]
        return Image.fromarray(np.uint8(best) * 255, mode="L")


# =============================================================================
# Candidate scoring
# =============================================================================



def edit_locality(before: Image.Image, after: Image.Image, mask: Image.Image) -> float:
    a = np.asarray(before.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(after.convert("RGB").resize(before.size), dtype=np.float32) / 255.0
    d = np.mean(np.abs(b - a), axis=2)
    m = np.asarray(mask.convert("L").resize(before.size), dtype=np.float32) / 255.0
    inside = float((d * m).sum() / (m.sum() + 1e-6))
    outside = float((d * (1.0 - m)).sum() / ((1.0 - m).sum() + 1e-6))
    return float(inside / (inside + outside + 1e-6))



def area_plausibility(frac: float, target: float = 0.10, sigma: float = 0.12) -> float:
    # Generic, intentionally weak prior: penalize edits that consume nearly the entire image
    # or collapse to tiny noise. It does NOT assume object-specific physical size.
    return float(math.exp(-0.5 * ((float(frac) - target) / sigma) ** 2))



def candidate_score(
    before: Image.Image,
    proposal: Image.Image,
    hard_mask: Image.Image,
    detection_score: float,
    previous_masks: Sequence[Image.Image],
    args,
) -> Tuple[float, Dict[str, float]]:
    locality = edit_locality(before, proposal, hard_mask)
    overlap = max([mask_iou(hard_mask, m) for m in previous_masks], default=0.0)
    border = border_fraction(hard_mask)
    area = mask_area_fraction(hard_mask)
    area_score = area_plausibility(area)

    score = (
        args.score_detection_weight * float(detection_score)
        + args.score_locality_weight * locality
        + args.score_area_weight * area_score
        - args.score_overlap_weight * overlap
        - args.score_border_weight * border
    )
    return float(score), {
        "detection_score": float(detection_score),
        "locality": locality,
        "overlap_previous": overlap,
        "border_fraction": border,
        "area_fraction": area,
        "area_plausibility": area_score,
    }


# =============================================================================
# Pipeline loading
# =============================================================================



def place_pipe_on_device(pipe, args):
    if args.cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            return pipe
        except Exception as exc:
            warnings.warn(f"enable_model_cpu_offload failed: {exc}; falling back to .to(device)")
    pipe.to(args.device)
    return pipe



def load_planner_pipe(args):
    dtype = torch_dtype_from_name(args.torch_dtype)
    pipe = FluxKontextPipeline.from_pretrained(args.kontext_model_id, torch_dtype=dtype)
    return place_pipe_on_device(pipe, args)



def load_inpaint_pipe(args):
    dtype = torch_dtype_from_name(args.torch_dtype)
    pipe = FluxKontextInpaintPipeline.from_pretrained(args.kontext_model_id, torch_dtype=dtype)
    return place_pipe_on_device(pipe, args)


def load_inpaint_pipe_from_planner(planner_pipe, args):
    if args.share_pipeline_components and hasattr(FluxKontextInpaintPipeline, "from_pipe"):
        try:
            return FluxKontextInpaintPipeline.from_pipe(planner_pipe)
        except Exception as exc:
            warnings.warn(f"Could not share Kontext pipeline components: {exc}")
    return load_inpaint_pipe(args)



def generate_base_scene(args) -> Image.Image:
    if not args.base_prompt:
        raise ValueError("generate_base_scene requires --base_prompt")
    dtype = torch_dtype_from_name(args.torch_dtype)
    pipe = FluxPipeline.from_pretrained(args.base_model_id, torch_dtype=dtype)
    pipe = place_pipe_on_device(pipe, args)
    short, long = base_prompts(args.base_prompt)
    out = pipe(
        prompt=short,
        prompt_2=long,
        width=args.width,
        height=args.height,
        num_inference_steps=args.base_steps,
        guidance_scale=args.base_guidance_scale,
        generator=generator_for(args.device, args.seed),
    ).images[0].convert("RGB")
    del pipe
    cleanup_cuda()
    return out


# =============================================================================
# Object job input
# =============================================================================



def load_jobs(args) -> List[ObjectJob]:
    if args.objects_json:
        with open(args.objects_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list) or not raw:
            raise ValueError("objects_json must contain a non-empty JSON list")
        jobs = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"objects_json item {i} must be an object/dict")
            if "name" not in item or "reference" not in item:
                raise ValueError(f"objects_json item {i} needs 'name' and 'reference'")
            jobs.append(ObjectJob(
                name=str(item["name"]),
                reference=str(item["reference"]),
                placement_hint=str(item.get("placement_hint", "")),
                pose_instruction=str(item.get("pose_instruction", "")),
                reference_flip=str(item.get("reference_flip", "none")),
                pose_source=str(item.get("pose_source", args.pose_source or "generic")).lower(),
            ))
            if jobs[-1].pose_source not in {"generic", "reference"}:
                raise ValueError(f"objects_json item {i} has invalid pose_source")
        return jobs

    if args.object_name and args.reference_image:
        return [ObjectJob(
            name=args.object_name,
            reference=args.reference_image,
            placement_hint=args.placement_hint or "",
            pose_instruction=args.pose_instruction or "",
            reference_flip=args.reference_flip,
            pose_source=args.pose_source or "generic",
        )]

    raise ValueError(
        "Provide --objects_json, or provide both --object_name and --reference_image"
    )


# =============================================================================
# Stage 1: generic placement proposals
# =============================================================================



def create_mask_for_proposal(
    before: Image.Image,
    proposal: Image.Image,
    object_name: str,
    detector: Optional[GenericDetector],
    segmenter: Optional[SAM2BoxSegmenter],
    args,
) -> Tuple[Image.Image, Image.Image, np.ndarray, float, Optional[Tuple[int, int, int, int]]]:
    # Establish where this particular edit happened before consulting a semantic
    # detector. This prevents an already-inserted, visually dominant object from
    # being selected again for every later job.
    changed_hard, changed_soft, diff = difference_mask(before, proposal, args)
    changed_box = mask_bbox(changed_hard)
    det = None
    if args.mask_backend in {"auto", "groundingdino", "sam2"} and detector is not None:
        try:
            detections = detector.detect_all(
                proposal, object_name, threshold=args.detection_threshold
            )
            if detections:
                det = max(
                    detections,
                    key=lambda candidate: candidate.score
                    + args.detector_change_weight
                    * mask_iou(box_mask(candidate.box, proposal.size), changed_hard),
                )
        except Exception as exc:
            warnings.warn(f"Detector failed for '{object_name}': {exc}")
            det = None

    if args.mask_backend == "groundingdino" and det is None:
        raise RuntimeError(
            f"GroundingDINO could not detect '{object_name}' and "
            f"--mask_backend {args.mask_backend} was requested."
        )

    restrict_box = det.box if det is not None else None
    if restrict_box is not None and changed_box is not None:
        detected_region = box_mask(restrict_box, proposal.size)
        changed_overlap = mask_iou(detected_region, changed_hard)
        if changed_overlap < args.detector_change_iou:
            warnings.warn(
                f"Ignoring stale GroundingDINO box for '{object_name}' "
                f"(changed-region IoU={changed_overlap:.3f}); using the new edit region."
            )
            restrict_box = changed_box
            det = None
    elif restrict_box is None:
        restrict_box = changed_box

    if restrict_box is not None:
        hard, soft, diff = difference_mask(before, proposal, args, restrict_box=restrict_box)
    else:
        hard, soft = changed_hard, changed_soft
    if args.mask_backend == "sam2":
        if segmenter is None or restrict_box is None:
            raise RuntimeError("SAM 2 masking requires a segmenter and a detected object box")
        hard = segmenter.segment(proposal, restrict_box, diff)
        area = mask_area_fraction(hard)
        if not args.min_mask_frac <= area <= args.max_mask_frac:
            raise RuntimeError(
                f"SAM 2 produced a pathological mask (area fraction {area:.4f}); "
                "adjust mask limits or use --mask_backend auto."
            )
        hard = dilate_mask_image(hard, args.mask_dilate_px)
        soft = hard
        if args.mask_feather_px > 0:
            soft = hard.filter(ImageFilter.GaussianBlur(radius=float(args.mask_feather_px)))
    detection_score = float(det.score) if det is not None else 0.0
    return hard, soft, diff, detection_score, restrict_box



def generate_generic_candidates(
    planner_pipe,
    current_scene: Image.Image,
    job: ObjectJob,
    previous_masks: Sequence[Image.Image],
    detector: Optional[GenericDetector],
    segmenter: Optional[SAM2BoxSegmenter],
    args,
    step_seed: int,
    step_dir: str,
) -> CandidateResult:
    root = os.path.join(step_dir, "placement_candidates")
    ensure_dir(root)
    candidates: List[CandidateResult] = []

    for i in range(1, args.placement_candidates + 1):
        cand_seed = int(step_seed + i * 101)
        short, long = insertion_prompts(job, i, args.placement_candidates)
        proposal = planner_pipe(
            image=current_scene,
            prompt=short,
            prompt_2=long,
            width=args.width,
            height=args.height,
            max_area=args.width * args.height,
            num_inference_steps=args.placement_steps,
            guidance_scale=args.placement_guidance_scale,
            generator=generator_for(args.device, cand_seed),
        ).images[0].convert("RGB")

        hard, soft, diff, det_score, box = create_mask_for_proposal(
            before=current_scene,
            proposal=proposal,
            object_name=job.name,
            detector=detector,
            segmenter=segmenter,
            args=args,
        )

        score, comp = candidate_score(
            before=current_scene,
            proposal=proposal,
            hard_mask=hard,
            detection_score=det_score,
            previous_masks=previous_masks,
            args=args,
        )
        valid = (
            comp["locality"] >= args.min_candidate_locality
            and args.min_mask_frac <= comp["area_fraction"] <= args.max_mask_frac
            and comp["border_fraction"] <= args.max_candidate_border_fraction
            and (not args.require_detection or comp["detection_score"] >= args.detection_threshold)
        )

        cand_dir = os.path.join(root, f"candidate_{i:02d}")
        ensure_dir(cand_dir)
        proposal.save(os.path.join(cand_dir, "generic_proposal.png"))
        hard.save(os.path.join(cand_dir, "object_mask_hard.png"))
        soft.save(os.path.join(cand_dir, "object_mask_soft.png"))
        make_overlay(current_scene, soft).save(os.path.join(cand_dir, "mask_overlay.png"))
        diff_img = Image.fromarray(np.uint8(np.clip(diff, 0, 1) * 255), mode="L")
        diff_img.save(os.path.join(cand_dir, "difference_map.png"))
        with open(os.path.join(cand_dir, "prompt_short.txt"), "w", encoding="utf-8") as f:
            f.write(short)
        with open(os.path.join(cand_dir, "prompt_2.txt"), "w", encoding="utf-8") as f:
            f.write(long)
        save_json({
            "score": score,
            "seed": cand_seed,
            "box": box,
            "valid": valid,
            **comp,
        }, os.path.join(cand_dir, "score.json"))

        candidates.append(CandidateResult(
            index=i,
            seed=cand_seed,
            proposal=proposal,
            hard_mask=hard,
            soft_mask=soft,
            score=score,
            detection_score=comp["detection_score"],
            locality=comp["locality"],
            overlap_previous=comp["overlap_previous"],
            border_fraction=comp["border_fraction"],
            area_fraction=comp["area_fraction"],
            box=box,
            prompt_short=short,
            prompt_long=long,
            valid=valid,
        ))

    valid_candidates = [candidate for candidate in candidates if candidate.valid]
    if not valid_candidates:
        raise RuntimeError(
            f"No valid placement candidate for '{job.name}'. Inspect {root}; "
            "do not replace an unrelated or failed edit."
        )
    best = max(valid_candidates, key=lambda c: c.score)
    save_json({
        "selected_candidate": best.index,
        "selected_seed": best.seed,
        "selected_score": best.score,
        "detection_score": best.detection_score,
        "locality": best.locality,
        "overlap_previous": best.overlap_previous,
        "border_fraction": best.border_fraction,
        "area_fraction": best.area_fraction,
        "box": best.box,
    }, os.path.join(step_dir, "placement_selection.json"))
    return best


# =============================================================================
# Stage 2: replace generic object with the reference object
# =============================================================================



def replace_with_reference(
    inpaint_pipe,
    generic_scene: Image.Image,
    current_scene: Image.Image,
    job: ObjectJob,
    reference: Image.Image,
    hard_mask: Image.Image,
    soft_mask: Image.Image,
    args,
    seed: int,
) -> Tuple[Image.Image, str, str]:
    short, long = replacement_prompts(job)

    # FluxKontextInpaintPipeline uses white pixels as repaint area.
    pipeline_mask = hard_mask
    if args.inpaint_mask_blur_px > 0:
        pipeline_mask = hard_mask.filter(ImageFilter.GaussianBlur(radius=float(args.inpaint_mask_blur_px)))

    generated = inpaint_pipe(
        image=generic_scene,
        mask_image=pipeline_mask,
        image_reference=reference,
        prompt=short,
        prompt_2=long,
        strength=args.replace_strength,
        width=args.width,
        height=args.height,
        max_area=args.width * args.height,
        num_inference_steps=args.replace_steps,
        guidance_scale=args.replace_guidance_scale,
        generator=generator_for(args.device, seed),
    ).images[0].convert("RGB")

    # Hard guarantee at pixel level that unrelated scene areas come from the pre-edit scene.
    protected = protect_outside_mask(current_scene, generated, soft_mask)
    return protected, short, long


# =============================================================================
# Stage 3: optional identity refinement
# =============================================================================



def refine_reference_identity(
    inpaint_pipe,
    replaced_scene: Image.Image,
    current_scene: Image.Image,
    job: ObjectJob,
    reference: Image.Image,
    hard_mask: Image.Image,
    soft_mask: Image.Image,
    args,
    seed: int,
) -> Tuple[Image.Image, str, str]:
    short, long = refinement_prompts(job)
    pipeline_mask = hard_mask
    if args.inpaint_mask_blur_px > 0:
        pipeline_mask = hard_mask.filter(ImageFilter.GaussianBlur(radius=float(args.inpaint_mask_blur_px)))

    generated = inpaint_pipe(
        image=replaced_scene,
        mask_image=pipeline_mask,
        image_reference=reference,
        prompt=short,
        prompt_2=long,
        strength=args.refine_strength,
        width=args.width,
        height=args.height,
        max_area=args.width * args.height,
        num_inference_steps=args.refine_steps,
        guidance_scale=args.refine_guidance_scale,
        generator=generator_for(args.device, seed),
    ).images[0].convert("RGB")

    protected = protect_outside_mask(current_scene, generated, soft_mask)
    return protected, short, long


def segment_final_object(
    final_scene: Image.Image,
    target_mask: Image.Image,
    segmenter: Optional[SAM2BoxSegmenter],
    args,
) -> Image.Image:
    """Track the final geometry rather than retaining the obsolete generic mask."""
    if segmenter is None:
        return target_mask.convert("L")
    bbox = mask_bbox(target_mask)
    if bbox is None:
        return target_mask.convert("L")
    evidence = np.asarray(target_mask.convert("L"), dtype=np.float32) / 255.0
    try:
        final_mask = segmenter.segment(final_scene, bbox, evidence)
        area = mask_area_fraction(final_mask)
        if args.min_mask_frac <= area <= args.max_mask_frac:
            return final_mask
        warnings.warn(f"Final SAM mask has suspicious area {area:.2%}; keeping target mask.")
    except Exception as exc:
        warnings.warn(f"Final SAM tracking failed: {exc}; keeping target mask.")
    return target_mask.convert("L")


# =============================================================================
# CLI
# =============================================================================



def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base scene: choose exactly one.
    base = p.add_mutually_exclusive_group(required=True)
    base.add_argument("--base_image", help="Existing base image")
    base.add_argument("--base_prompt", help="Generate the base scene from this text prompt")

    # Object jobs.
    p.add_argument("--objects_json", help="JSON list describing one or more objects")
    p.add_argument("--object_name", help="Single-object convenience mode")
    p.add_argument("--reference_image", help="Single-object reference image")
    p.add_argument("--placement_hint", default="")
    p.add_argument("--pose_instruction", default="")
    p.add_argument("--reference_flip", choices=["none", "horizontal", "vertical", "both"], default="none")
    p.add_argument(
        "--pose_source",
        choices=["generic", "reference"],
        default=None,
        help="Global pose source; JSON pose_source overrides it per object (default: generic)",
    )

    # Output / models / runtime.
    p.add_argument("--out_dir", default="results/e15_generic_place_then_replace")
    p.add_argument("--kontext_model_id", default=DEFAULT_KONTEXT_MODEL)
    p.add_argument("--base_model_id", default=DEFAULT_BASE_MODEL)
    p.add_argument("--detector_model_id", default=DEFAULT_DETECTOR)
    p.add_argument("--sam2_model_id", default=DEFAULT_SAM2_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--detector_device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--sam2_device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument(
        "--share_pipeline_components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse planner model components in the inpaint pipeline to reduce memory",
    )
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    # Optional base generation.
    p.add_argument("--base_steps", type=int, default=28)
    p.add_argument("--base_guidance_scale", type=float, default=3.5)

    # Placement candidates.
    p.add_argument("--placement_candidates", type=int, default=4)
    p.add_argument("--placement_steps", type=int, default=16)
    p.add_argument("--placement_guidance_scale", type=float, default=2.5)
    p.add_argument(
        "--planning_scene",
        choices=["base", "current"],
        default="base",
        help="Plan every object from the clean base or from the cumulative edited scene",
    )
    p.add_argument("--min_candidate_locality", type=float, default=0.50)
    p.add_argument("--max_candidate_border_fraction", type=float, default=0.35)
    p.add_argument(
        "--require_detection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject proposals unless GroundingDINO confirms the requested class",
    )

    # Masking.
    p.add_argument(
        "--mask_backend",
        choices=["auto", "groundingdino", "sam2", "difference"],
        default="sam2",
        help="sam2 uses a GroundingDINO box prompt for precise segmentation",
    )
    p.add_argument("--detection_threshold", type=float, default=0.20)
    p.add_argument(
        "--detector_change_iou",
        type=float,
        default=0.05,
        help="Reject a detector box that does not overlap the newly changed region",
    )
    p.add_argument(
        "--detector_change_weight",
        type=float,
        default=2.0,
        help="Prefer detections aligned with the new edit over stale high-confidence objects",
    )
    p.add_argument("--detect_box_expand_frac", type=float, default=0.12)
    p.add_argument("--diff_quantile", type=float, default=0.80)
    p.add_argument("--diff_blur_px", type=float, default=3.0)
    p.add_argument("--mask_close_px", type=int, default=3)
    p.add_argument("--mask_dilate_px", type=int, default=10)
    p.add_argument("--mask_feather_px", type=float, default=8.0)
    p.add_argument("--min_mask_frac", type=float, default=0.003)
    p.add_argument("--max_mask_frac", type=float, default=0.40)
    p.add_argument("--min_ref_mask_frac", type=float, default=0.01)
    p.add_argument("--max_ref_mask_frac", type=float, default=0.95)
    p.add_argument("--reference_crop_expand_frac", type=float, default=0.08)

    # Candidate score.
    p.add_argument("--score_detection_weight", type=float, default=1.00)
    p.add_argument("--score_locality_weight", type=float, default=0.80)
    p.add_argument("--score_area_weight", type=float, default=0.25)
    p.add_argument("--score_overlap_weight", type=float, default=1.10)
    p.add_argument("--score_border_weight", type=float, default=0.45)

    # Reference replacement.
    p.add_argument("--replace_strength", type=float, default=0.95)
    p.add_argument("--replace_steps", type=int, default=28)
    p.add_argument("--replace_guidance_scale", type=float, default=2.5)
    p.add_argument("--inpaint_mask_blur_px", type=float, default=5.0)
    p.add_argument(
        "--replacement_mask_expand_frac",
        "--replacement_box_expand_frac",
        dest="replacement_mask_expand_frac",
        type=float,
        default=0.10,
        help="Organic dilation around the SAM silhouette for forming different geometry",
    )

    # Refinement.
    p.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--refine_strength", type=float, default=0.55)
    p.add_argument("--refine_steps", type=int, default=20)
    p.add_argument("--refine_guidance_scale", type=float, default=2.5)

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================



def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    save_json(vars(args), os.path.join(args.out_dir, "config_e15.json"))

    jobs = load_jobs(args)
    for job in jobs:
        if not os.path.isfile(job.reference):
            raise FileNotFoundError(f"Reference image not found for '{job.name}': {job.reference}")

    # Stage 0: prepare/generate base scene.
    if args.base_image:
        current_scene = Image.open(args.base_image).convert("RGB").resize(
            (args.width, args.height), Image.Resampling.LANCZOS
        )
    else:
        print("[Stage 0] Generating base scene ...")
        current_scene = generate_base_scene(args)

    current_scene.save(os.path.join(args.out_dir, "base_scene.png"))
    base_scene = current_scene.copy()

    # Detector is generic and lazy-loaded. In 'difference' mode it is never loaded.
    detector: Optional[GenericDetector] = None
    if args.mask_backend in {"auto", "groundingdino", "sam2"}:
        detector = GenericDetector(args.detector_model_id, device=args.detector_device)
    segmenter: Optional[SAM2BoxSegmenter] = None
    if args.mask_backend == "sam2":
        segmenter = SAM2BoxSegmenter(args.sam2_model_id, device=args.sam2_device)

    print("Loading FLUX.1 Kontext placement pipeline ...")
    planner_pipe = load_planner_pipe(args)
    print("Loading FLUX.1 Kontext inpaint pipeline ...")
    inpaint_pipe = load_inpaint_pipe_from_planner(planner_pipe, args)

    accepted_masks: List[Image.Image] = []
    summary: List[Dict[str, Any]] = []

    print("\n=== E15: GENERIC PLACE -> MASK -> REFERENCE REPLACE -> REFINE ===")
    print(f"objects: {[j.name for j in jobs]}")
    print(f"mask backend: {args.mask_backend}")
    print(f"placement candidates/object: {args.placement_candidates}")
    print(f"planning scene: {args.planning_scene}")
    print("prompt strategy: short CLIP prompt + detailed T5 prompt_2")

    for step, job in enumerate(jobs, start=1):
        step_dir = os.path.join(args.out_dir, f"step_{step:02d}_{job.name.replace(' ', '_')}")
        ensure_dir(step_dir)
        before = current_scene.copy()
        before.save(os.path.join(step_dir, "00_before.png"))
        planning_scene = base_scene if args.planning_scene == "base" else before
        planning_scene.save(os.path.join(step_dir, "00_planning_scene.png"))

        print(f"\n[{step}/{len(jobs)}] {job.name}")
        print("  Stage 1: generating natural generic placement candidates ...")
        best = generate_generic_candidates(
            planner_pipe=planner_pipe,
            current_scene=planning_scene,
            job=job,
            previous_masks=accepted_masks,
            detector=detector,
            segmenter=segmenter,
            args=args,
            step_seed=args.seed + step * 10000,
            step_dir=step_dir,
        )

        best.proposal.save(os.path.join(step_dir, "01_generic_object.png"))
        best.hard_mask.save(os.path.join(step_dir, "02_generic_object_mask_hard.png"))
        best.soft_mask.save(os.path.join(step_dir, "02_generic_object_mask_soft.png"))
        make_overlay(best.proposal, best.soft_mask).save(os.path.join(step_dir, "02_mask_overlay_on_generic.png"))

        # When planning from the clean base, transfer only the selected generic
        # object into the cumulative scene. This preserves earlier objects while
        # retaining a visible pose/placement anchor for the reference inpaint pass.
        generic_anchor_scene = composite_masked_edit(before, best.proposal, best.soft_mask)
        generic_anchor_scene.save(os.path.join(step_dir, "02_generic_anchor_on_current.png"))

        replace_hard, replace_soft = replacement_envelope(
            best.hard_mask,
            best.proposal.size,
            args.replacement_mask_expand_frac,
            args.mask_feather_px,
        )
        replace_hard.save(os.path.join(step_dir, "02_replacement_mask_hard.png"))
        replace_soft.save(os.path.join(step_dir, "02_replacement_mask_soft.png"))
        make_overlay(best.proposal, replace_soft).save(
            os.path.join(step_dir, "02_replacement_mask_overlay.png")
        )

        print(f"    selected candidate {best.index}: score={best.score:.3f}")
        print("  Stage 2: replacing generic object with reference object ...")
        reference = Image.open(job.reference).convert("RGB")
        reference = flip_reference(reference, job.reference_flip)
        reference.save(os.path.join(step_dir, "reference_original.png"))
        reference, reference_mask = isolate_reference_object(
            reference, job.name, detector, segmenter, args
        )
        reference.save(os.path.join(step_dir, "reference_used.png"))
        reference_mask.save(os.path.join(step_dir, "reference_foreground_mask.png"))

        replaced, replace_short, replace_long = replace_with_reference(
            inpaint_pipe=inpaint_pipe,
            generic_scene=generic_anchor_scene,
            current_scene=before,
            job=job,
            reference=reference,
            hard_mask=replace_hard,
            soft_mask=replace_soft,
            args=args,
            seed=best.seed + 5000,
        )
        replaced.save(os.path.join(step_dir, "03_reference_replaced.png"))
        with open(os.path.join(step_dir, "replace_prompt_short.txt"), "w", encoding="utf-8") as f:
            f.write(replace_short)
        with open(os.path.join(step_dir, "replace_prompt_2.txt"), "w", encoding="utf-8") as f:
            f.write(replace_long)

        final = replaced
        if args.refine:
            print("  Stage 3: refining reference identity ...")
            final, refine_short, refine_long = refine_reference_identity(
                inpaint_pipe=inpaint_pipe,
                replaced_scene=replaced,
                current_scene=before,
                job=job,
                reference=reference,
                hard_mask=replace_hard,
                soft_mask=replace_soft,
                args=args,
                seed=best.seed + 9000,
            )
            with open(os.path.join(step_dir, "refine_prompt_short.txt"), "w", encoding="utf-8") as f:
                f.write(refine_short)
            with open(os.path.join(step_dir, "refine_prompt_2.txt"), "w", encoding="utf-8") as f:
                f.write(refine_long)

        final.save(os.path.join(step_dir, "04_final.png"))
        final_mask = segment_final_object(final, replace_hard, segmenter, args)
        final_soft_mask = final_mask
        if args.mask_feather_px > 0:
            final_soft_mask = final_mask.filter(
                ImageFilter.GaussianBlur(radius=float(args.mask_feather_px))
            )
        final_mask.save(os.path.join(step_dir, "04_final_object_mask_hard.png"))
        final_soft_mask.save(os.path.join(step_dir, "04_final_object_mask_soft.png"))
        make_overlay(final, final_soft_mask).save(
            os.path.join(step_dir, "04_final_object_mask_overlay.png")
        )
        current_scene = final
        accepted_masks.append(final_soft_mask)

        step_record = {
            "step": step,
            "object": asdict(job),
            "selected_candidate": best.index,
            "selected_seed": best.seed,
            "selected_score": best.score,
            "detection_score": best.detection_score,
            "locality": best.locality,
            "overlap_previous": best.overlap_previous,
            "border_fraction": best.border_fraction,
            "area_fraction": best.area_fraction,
            "box": best.box,
            "candidate_valid": best.valid,
            "final_mask_area_fraction": mask_area_fraction(final_mask),
            "planning_scene": args.planning_scene,
        }
        save_json(step_record, os.path.join(step_dir, "step_summary.json"))
        summary.append(step_record)

    current_scene.save(os.path.join(args.out_dir, "FINAL.png"))
    save_json(summary, os.path.join(args.out_dir, "summary_e15.json"))

    print(f"\nDone. Final image: {os.path.join(args.out_dir, 'FINAL.png')}")


if __name__ == "__main__":
    main()
