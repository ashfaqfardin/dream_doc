"""
E16: Reference-Composite Harmonizer
==================================

Preserve a reference object's pixels and geometry instead of asking diffusion to
reconstruct it. A disposable generic insertion is used only to estimate a plausible
target box. The SAM reference cutout is fitted into that box, composited onto the
cumulative scene, and Kontext inpaints only its boundary and support-aware interaction
region. The object core is restored exactly after harmonization.

This is intended for rigid or mostly rigid objects. Large novel-view or articulated
pose changes remain outside the guarantees of a single-view reference pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from e15_generic_place_then_replace_pipeline import (
    DEFAULT_DETECTOR,
    DEFAULT_KONTEXT_MODEL,
    GenericDetector,
    SAM2BoxSegmenter,
    border_fraction,
    clamp_box,
    ensure_dir,
    expand_box,
    generator_for,
    image_difference_map,
    load_inpaint_pipe_from_planner,
    load_planner_pipe,
    make_overlay,
    mask_area_fraction,
    mask_bbox,
    mask_iou,
    protect_outside_mask,
    save_json,
)


@dataclass
class ObjectJob:
    name: str
    reference: str
    placement_hint: str = ""
    reference_flip: str = "none"


@dataclass
class Placement:
    proposal: Image.Image
    object_mask: Image.Image
    box: Tuple[int, int, int, int]
    seed: int
    attempt: int
    detection_score: float
    bbox_fraction: float
    border_fraction: float
    overlap_previous: float


def load_jobs(path: str) -> List[ObjectJob]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError("--objects_json must contain a non-empty list")
    jobs = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "name" not in item or "reference" not in item:
            raise ValueError(f"Invalid object entry {index}")
        jobs.append(ObjectJob(
            name=str(item["name"]),
            reference=str(item["reference"]),
            placement_hint=str(item.get("placement_hint", "")),
            reference_flip=str(item.get("reference_flip", "none")),
        ))
    return jobs


def flip_reference(image: Image.Image, mode: str) -> Image.Image:
    from PIL import ImageOps
    if mode == "horizontal":
        return ImageOps.mirror(image)
    if mode == "vertical":
        return ImageOps.flip(image)
    if mode == "both":
        return ImageOps.flip(ImageOps.mirror(image))
    return image


def binary_union(masks: Sequence[Image.Image], size: Tuple[int, int]) -> Image.Image:
    out = np.zeros((size[1], size[0]), dtype=bool)
    for mask in masks:
        out |= np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST)) > 127
    return Image.fromarray(np.uint8(out) * 255, mode="L")


def erode(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.convert("L")
    return mask.convert("L").filter(ImageFilter.MinFilter(2 * radius + 1))


def dilate(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.convert("L")
    return mask.convert("L").filter(ImageFilter.MaxFilter(2 * radius + 1))


def subtract_mask(a: Image.Image, b: Image.Image) -> Image.Image:
    aa = np.asarray(a.convert("L"), dtype=np.int16)
    bb = np.asarray(b.convert("L"), dtype=np.int16)
    return Image.fromarray(np.uint8(np.clip(aa - bb, 0, 255)), mode="L")


def placement_prompt(job: ObjectJob) -> Tuple[str, str]:
    short = f"Add one realistic {job.name} naturally to this room."
    long = (
        f"Add exactly one generic {job.name} only as a placement proposal. Choose a physically plausible "
        "support surface and realistic scale. Keep the complete object comfortably inside the frame. "
        "Do not cover windows, doors, radiators, or important furniture. Preserve everything else. "
    )
    if job.placement_hint:
        long += f"Placement preference: {job.placement_hint.strip()}"
    return short, long


def changed_box_mask(before: Image.Image, after: Image.Image, args) -> Tuple[np.ndarray, Image.Image]:
    diff = image_difference_map(before, after, args.diff_blur_px)
    threshold = float(np.quantile(diff, args.diff_quantile))
    changed = diff >= threshold
    mask = Image.fromarray(np.uint8(changed) * 255, mode="L")
    mask = mask.filter(ImageFilter.MaxFilter(2 * args.diff_close_px + 1))
    mask = mask.filter(ImageFilter.MinFilter(2 * args.diff_close_px + 1))
    return diff, mask


def bbox_fraction(box: Tuple[int, int, int, int], size: Tuple[int, int]) -> float:
    x0, y0, x1, y1 = box
    return float((x1 - x0) * (y1 - y0) / (size[0] * size[1]))


def box_edge_clearance(box: Tuple[int, int, int, int], size: Tuple[int, int]) -> float:
    x0, y0, x1, y1 = box
    return float(min(x0, y0, size[0] - x1, size[1] - y1) / min(size))


def propose_placement(
    planner,
    base_scene: Image.Image,
    job: ObjectJob,
    occupied_masks: Sequence[Image.Image],
    detector: GenericDetector,
    segmenter: SAM2BoxSegmenter,
    args,
    step_seed: int,
    step_dir: str,
) -> Placement:
    root = os.path.join(step_dir, "placement_attempts")
    ensure_dir(root)
    short, long = placement_prompt(job)
    failures = []

    for attempt in range(1, args.placement_attempts + 1):
        seed = step_seed + attempt * 101
        proposal = planner(
            image=base_scene,
            prompt=short,
            prompt_2=long,
            width=args.width,
            height=args.height,
            max_area=args.width * args.height,
            num_inference_steps=args.placement_steps,
            guidance_scale=args.placement_guidance_scale,
            generator=generator_for(args.device, seed),
        ).images[0].convert("RGB")

        diff, changed = changed_box_mask(base_scene, proposal, args)
        detections = detector.detect_all(proposal, job.name, args.detection_threshold)
        if not detections:
            failures.append(f"attempt {attempt}: detector found no {job.name}")
            continue
        detection = max(
            detections,
            key=lambda item: item.score
            + 2.0 * mask_iou(changed, box_image(item.box, proposal.size)),
        )
        object_mask = segmenter.segment(proposal, detection.box, diff)
        box = mask_bbox(object_mask)
        if box is None:
            failures.append(f"attempt {attempt}: empty SAM mask")
            continue

        box_frac = bbox_fraction(box, proposal.size)
        edge = box_edge_clearance(box, proposal.size)
        border = border_fraction(object_mask)
        occupied = binary_union(occupied_masks, proposal.size) if occupied_masks else None
        overlap = mask_iou(object_mask, occupied) if occupied is not None else 0.0
        valid = (
            args.min_bbox_frac <= box_frac <= args.max_bbox_frac
            and edge >= args.min_edge_clearance_frac
            and border <= args.max_border_fraction
            and overlap <= args.max_overlap
        )

        attempt_dir = os.path.join(root, f"attempt_{attempt:02d}")
        ensure_dir(attempt_dir)
        proposal.save(os.path.join(attempt_dir, "proposal.png"))
        object_mask.save(os.path.join(attempt_dir, "sam_mask.png"))
        make_overlay(proposal, object_mask).save(os.path.join(attempt_dir, "overlay.png"))
        save_json({
            "valid": valid,
            "seed": seed,
            "box": box,
            "detection_score": detection.score,
            "bbox_fraction": box_frac,
            "edge_clearance_fraction": edge,
            "border_fraction": border,
            "overlap_previous": overlap,
        }, os.path.join(attempt_dir, "metrics.json"))

        if valid:
            return Placement(
                proposal=proposal,
                object_mask=object_mask,
                box=box,
                seed=seed,
                attempt=attempt,
                detection_score=detection.score,
                bbox_fraction=box_frac,
                border_fraction=border,
                overlap_previous=overlap,
            )
        failures.append(
            f"attempt {attempt}: bbox={box_frac:.3f}, edge={edge:.3f}, "
            f"border={border:.3f}, overlap={overlap:.3f}"
        )

    raise RuntimeError(
        f"No valid placement for '{job.name}' after {args.placement_attempts} attempts:\n"
        + "\n".join(failures)
    )


def box_image(box: Tuple[int, int, int, int], size: Tuple[int, int]) -> Image.Image:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).rectangle(clamp_box(box, size), fill=255)
    return image


def segment_reference(
    reference: Image.Image,
    job: ObjectJob,
    detector: GenericDetector,
    segmenter: SAM2BoxSegmenter,
    args,
) -> Tuple[Image.Image, Image.Image]:
    detections = detector.detect_all(reference, job.name, args.detection_threshold)
    if not detections:
        raise RuntimeError(f"Could not detect '{job.name}' in reference {job.reference}")
    evidence = np.ones((reference.height, reference.width), dtype=np.float32)
    mask = segmenter.segment(reference, detections[0].box, evidence)
    box = mask_bbox(mask)
    if box is None:
        raise RuntimeError(f"SAM returned an empty reference mask for '{job.name}'")
    area = mask_area_fraction(mask)
    if not args.min_ref_mask_frac <= area <= args.max_ref_mask_frac:
        raise RuntimeError(f"Suspicious reference mask for '{job.name}': {area:.2%}")
    return reference.crop(box), mask.crop(box)


def fit_reference_to_box(
    reference_crop: Image.Image,
    reference_mask: Image.Image,
    target_box: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    scale: float,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = target_box
    target_w = max(1, int(round((x1 - x0) * scale)))
    target_h = max(1, int(round((y1 - y0) * scale)))
    ratio = min(target_w / reference_crop.width, target_h / reference_crop.height)
    new_w = max(1, int(round(reference_crop.width * ratio)))
    new_h = max(1, int(round(reference_crop.height * ratio)))
    resized = reference_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_mask = reference_mask.resize((new_w, new_h), Image.Resampling.LANCZOS)

    cx = (x0 + x1) // 2
    bottom = y1
    px = cx - new_w // 2
    py = bottom - new_h
    placed_box = clamp_box((px, py, px + new_w, py + new_h), canvas_size)
    px, py, qx, qy = placed_box
    crop_w, crop_h = qx - px, qy - py
    resized = resized.crop((0, 0, crop_w, crop_h))
    resized_mask = resized_mask.crop((0, 0, crop_w, crop_h))

    rgb_canvas = Image.new("RGB", canvas_size, (0, 0, 0))
    mask_canvas = Image.new("L", canvas_size, 0)
    rgb_canvas.paste(resized, (px, py))
    mask_canvas.paste(resized_mask, (px, py))
    return rgb_canvas, mask_canvas, placed_box


def composite_cutout(scene: Image.Image, cutout_canvas: Image.Image, alpha: Image.Image) -> Image.Image:
    return Image.composite(cutout_canvas, scene.convert("RGB"), alpha.convert("L"))


def build_harmonization_masks(alpha: Image.Image, args) -> Tuple[Image.Image, Image.Image, Image.Image]:
    core = erode(alpha, args.core_erode_px)
    outer = dilate(alpha, args.boundary_expand_px)
    boundary = subtract_mask(outer, core)

    box = mask_bbox(alpha)
    interaction = Image.new("L", alpha.size, 0)
    if box is not None:
        x0, y0, x1, y1 = box
        width, height = x1 - x0, y1 - y0
        side = int(round(width * args.interaction_side_frac))
        down = int(round(height * args.interaction_down_frac))
        interaction = box_image(
            expand_box((x0, max(y0, y1 - max(2, height // 8)), x1, y1 + down), alpha.size, 0.0),
            alpha.size,
        )
        if side > 0:
            interaction = dilate(interaction, side)
    editable = binary_union([boundary, interaction], alpha.size)
    blend = editable.filter(ImageFilter.GaussianBlur(args.mask_feather_px))
    # Never replace the protected reference core during final compositing.
    blend_arr = np.asarray(blend, dtype=np.uint8).copy()
    blend_arr[np.asarray(core) > 127] = 0
    blend = Image.fromarray(blend_arr, mode="L")
    return core, editable, blend


def harmonize(
    pipe,
    composite: Image.Image,
    isolated_reference: Image.Image,
    editable_mask: Image.Image,
    blend_mask: Image.Image,
    core_mask: Image.Image,
    job: ObjectJob,
    args,
    seed: int,
) -> Image.Image:
    prompt = (
        f"Integrate the pasted {job.name} naturally into this scene. Preserve its exact design, parts, "
        "proportions, colors and pose. Edit only boundaries, occlusions, local illumination, floor or "
        "surface contact, and a realistic contact shadow. Do not redesign or duplicate the object."
    )
    mask_for_pipe = editable_mask.filter(ImageFilter.GaussianBlur(args.inpaint_mask_blur_px))
    generated = pipe(
        image=composite,
        mask_image=mask_for_pipe,
        image_reference=isolated_reference,
        prompt=f"Naturally integrate the pasted {job.name}.",
        prompt_2=prompt,
        strength=args.harmonize_strength,
        width=args.width,
        height=args.height,
        max_area=args.width * args.height,
        num_inference_steps=args.harmonize_steps,
        guidance_scale=args.harmonize_guidance_scale,
        generator=generator_for(args.device, seed),
    ).images[0].convert("RGB")
    harmonized = protect_outside_mask(composite, generated, blend_mask)
    # Exact core preservation is the principal difference from e15.
    return Image.composite(composite, harmonized, core_mask)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_image", required=True)
    parser.add_argument("--objects_json", required=True)
    parser.add_argument("--out_dir", default="results/e16_reference_composite")
    parser.add_argument("--kontext_model_id", default=DEFAULT_KONTEXT_MODEL)
    parser.add_argument("--detector_model_id", default=DEFAULT_DETECTOR)
    parser.add_argument("--sam2_model_id", default="facebook/sam2-hiera-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sam2_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--share_pipeline_components", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--placement_attempts", type=int, default=4)
    parser.add_argument("--placement_steps", type=int, default=16)
    parser.add_argument("--placement_guidance_scale", type=float, default=2.5)
    parser.add_argument("--detection_threshold", type=float, default=0.20)
    parser.add_argument("--diff_quantile", type=float, default=0.80)
    parser.add_argument("--diff_blur_px", type=float, default=3.0)
    parser.add_argument("--diff_close_px", type=int, default=3)
    parser.add_argument("--min_bbox_frac", type=float, default=0.01)
    parser.add_argument("--max_bbox_frac", type=float, default=0.22)
    parser.add_argument("--min_edge_clearance_frac", type=float, default=0.04)
    parser.add_argument("--max_border_fraction", type=float, default=0.05)
    parser.add_argument("--max_overlap", type=float, default=0.08)

    parser.add_argument("--reference_scale", type=float, default=0.92)
    parser.add_argument("--min_ref_mask_frac", type=float, default=0.01)
    parser.add_argument("--max_ref_mask_frac", type=float, default=0.95)
    parser.add_argument("--core_erode_px", type=int, default=3)
    parser.add_argument("--boundary_expand_px", type=int, default=12)
    parser.add_argument("--interaction_side_frac", type=float, default=0.02)
    parser.add_argument("--interaction_down_frac", type=float, default=0.12)
    parser.add_argument("--mask_feather_px", type=float, default=6.0)
    parser.add_argument("--inpaint_mask_blur_px", type=float, default=4.0)
    parser.add_argument("--harmonize_strength", type=float, default=0.75)
    parser.add_argument("--harmonize_steps", type=int, default=24)
    parser.add_argument("--harmonize_guidance_scale", type=float, default=2.5)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    save_json(vars(args), os.path.join(args.out_dir, "config_e16.json"))
    jobs = load_jobs(args.objects_json)
    for job in jobs:
        if not os.path.isfile(job.reference):
            raise FileNotFoundError(job.reference)

    base = Image.open(args.base_image).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.LANCZOS
    )
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    current = base.copy()
    occupied_masks: List[Image.Image] = []
    summary: List[Dict[str, Any]] = []

    detector = GenericDetector(args.detector_model_id, args.detector_device)
    segmenter = SAM2BoxSegmenter(args.sam2_model_id, args.sam2_device)
    print("Loading Kontext placement pipeline ...")
    planner = load_planner_pipe(args)
    print("Loading shared Kontext inpaint pipeline ...")
    inpaint = load_inpaint_pipe_from_planner(planner, args)

    print("\n=== E16: PLACE BOX -> COMPOSITE REFERENCE -> HARMONIZE BOUNDARY ===")
    for index, job in enumerate(jobs, start=1):
        step_dir = os.path.join(args.out_dir, f"step_{index:02d}_{job.name.replace(' ', '_')}")
        ensure_dir(step_dir)
        current.save(os.path.join(step_dir, "00_before.png"))

        placement = propose_placement(
            planner, base, job, occupied_masks, detector, segmenter, args,
            args.seed + index * 10000, step_dir,
        )
        placement.proposal.save(os.path.join(step_dir, "01_disposable_placement_probe.png"))
        placement.object_mask.save(os.path.join(step_dir, "01_probe_mask.png"))

        reference = flip_reference(Image.open(job.reference).convert("RGB"), job.reference_flip)
        ref_crop, ref_mask = segment_reference(reference, job, detector, segmenter, args)
        ref_crop.save(os.path.join(step_dir, "02_reference_crop.png"))
        ref_mask.save(os.path.join(step_dir, "02_reference_mask.png"))

        cutout_canvas, placed_alpha, placed_box = fit_reference_to_box(
            ref_crop, ref_mask, placement.box, current.size, args.reference_scale
        )
        composite = composite_cutout(current, cutout_canvas, placed_alpha)
        placed_alpha.save(os.path.join(step_dir, "03_placed_reference_alpha.png"))
        composite.save(os.path.join(step_dir, "03_reference_composite.png"))

        core, editable, blend = build_harmonization_masks(placed_alpha, args)
        core.save(os.path.join(step_dir, "04_core_preserve_mask.png"))
        editable.save(os.path.join(step_dir, "04_harmonize_edit_mask.png"))
        blend.save(os.path.join(step_dir, "04_harmonize_blend_mask.png"))
        make_overlay(composite, blend).save(os.path.join(step_dir, "04_harmonize_overlay.png"))

        final = harmonize(
            inpaint, composite, ref_crop, editable, blend, core, job, args,
            placement.seed + 5000,
        )
        final.save(os.path.join(step_dir, "05_final.png"))
        current = final
        occupied_masks.append(placed_alpha)

        record = {
            "step": index,
            "object": asdict(job),
            "placement_attempt": placement.attempt,
            "placement_seed": placement.seed,
            "probe_box": placement.box,
            "placed_reference_box": placed_box,
            "detection_score": placement.detection_score,
            "bbox_fraction": placement.bbox_fraction,
            "border_fraction": placement.border_fraction,
            "overlap_previous": placement.overlap_previous,
            "placed_mask_fraction": mask_area_fraction(placed_alpha),
        }
        save_json(record, os.path.join(step_dir, "step_summary.json"))
        summary.append(record)
        print(f"[{index}/{len(jobs)}] {job.name}: attempt={placement.attempt}, box={placed_box}")

    current.save(os.path.join(args.out_dir, "FINAL.png"))
    save_json(summary, os.path.join(args.out_dir, "summary_e16.json"))
    print(f"Done: {os.path.join(args.out_dir, 'FINAL.png')}")


if __name__ == "__main__":
    main()