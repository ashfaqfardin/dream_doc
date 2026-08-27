"""E17: Direct Reference Insertion

Minimal baseline: give FLUX.1-Kontext the current scene and one reference image
in the same call, and ask it to place that exact object naturally. There is no
pre-placement mask, generic anchor, cut-and-paste, or inpainting stage.

Post-generation GroundingDINO + SAM2 is diagnostic only. It does not constrain
generation unless --protect_outside_object is enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from e15_generic_place_then_replace_pipeline import (
    DEFAULT_DETECTOR,
    DEFAULT_KONTEXT_MODEL,
    GenericDetector,
    SAM2BoxSegmenter,
    border_fraction,
    ensure_dir,
    generator_for,
    image_difference_map,
    load_planner_pipe,
    make_overlay,
    mask_area_fraction,
    mask_bbox,
    protect_outside_mask,
    save_json,
)
from utils import enable_multi_context


@dataclass
class ObjectJob:
    name: str
    reference: str
    placement_hint: str = ""
    reference_flip: str = "none"


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
    if mode == "horizontal":
        return ImageOps.mirror(image)
    if mode == "vertical":
        return ImageOps.flip(image)
    if mode == "both":
        return ImageOps.flip(ImageOps.mirror(image))
    return image


def prepare_reference_canvas(reference: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Preserve aspect ratio while satisfying the multi-context batch dimensions."""
    return ImageOps.pad(
        reference.convert("RGB"), size, method=Image.Resampling.LANCZOS,
        color=(255, 255, 255), centering=(0.5, 0.5),
    )


def insertion_prompts(job: ObjectJob) -> Tuple[str, str]:
    short = f"Add the exact reference {job.name} naturally to the scene."
    long = (
        f"The first image is the scene to edit. The second image is a reference showing the exact {job.name} "
        "to insert. Add exactly one instance of that reference object to the first image. Choose a physically "
        "plausible location and realistic scale, perspective, support contact, occlusion and illumination. "
        "Preserve its distinctive colors, materials, design, proportions and parts. Keep the complete object "
        "inside the frame. Do not copy the reference background. Preserve everything unrelated in the scene."
    )
    if job.placement_hint:
        long += f" Placement preference: {job.placement_hint.strip()}"
    return short, long


def diagnose_object(
    before: Image.Image,
    generated: Image.Image,
    job: ObjectJob,
    detector: GenericDetector,
    segmenter: Optional[SAM2BoxSegmenter],
    args,
) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    diff = image_difference_map(before, generated, args.diff_blur_px)
    try:
        detections = detector.detect_all(generated, job.name, args.detection_threshold)
    except Exception as exc:
        warnings.warn(f"Post-generation detector failed for '{job.name}': {exc}")
        detections = []
    if not detections:
        return None, {"detected": False}

    detection = detections[0]
    mask = None
    if segmenter is not None:
        try:
            mask = segmenter.segment(generated, detection.box, diff)
        except Exception as exc:
            warnings.warn(f"Post-generation SAM failed for '{job.name}': {exc}")

    metrics: Dict[str, Any] = {
        "detected": True,
        "detection_score": detection.score,
        "detection_box": detection.box,
    }
    if mask is not None:
        box = mask_bbox(mask)
        metrics.update({
            "sam_box": box,
            "mask_area_fraction": mask_area_fraction(mask),
            "border_fraction": border_fraction(mask),
        })
    return mask, metrics


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_image", required=True)
    parser.add_argument("--objects_json", required=True)
    parser.add_argument("--out_dir", default="results/e17_direct_reference")
    parser.add_argument("--kontext_model_id", default=DEFAULT_KONTEXT_MODEL)
    parser.add_argument("--detector_model_id", default=DEFAULT_DETECTOR)
    parser.add_argument("--sam2_model_id", default="facebook/sam2-hiera-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sam2_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--detection_threshold", type=float, default=0.20)
    parser.add_argument("--diff_blur_px", type=float, default=3.0)
    parser.add_argument("--post_mask_dilate_px", type=int, default=12)
    parser.add_argument("--post_mask_feather_px", type=float, default=8.0)
    parser.add_argument(
        "--protect_outside_object",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore the previous scene outside the post-generated SAM mask",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    save_json(vars(args), os.path.join(args.out_dir, "config_e17.json"))
    jobs = load_jobs(args.objects_json)
    for job in jobs:
        if not os.path.isfile(job.reference):
            raise FileNotFoundError(job.reference)

    current = Image.open(args.base_image).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.LANCZOS
    )
    current.save(os.path.join(args.out_dir, "base_scene.png"))

    print("Loading FLUX.1-Kontext direct multi-context pipeline ...")
    pipe = load_planner_pipe(args)
    enable_multi_context(pipe)
    detector = GenericDetector(args.detector_model_id, args.detector_device)
    segmenter = SAM2BoxSegmenter(args.sam2_model_id, args.sam2_device)
    summary = []

    print("\n=== E17: CURRENT SCENE + REFERENCE -> DIRECT INSERTION ===")
    for index, job in enumerate(jobs, start=1):
        step_dir = os.path.join(args.out_dir, f"step_{index:02d}_{job.name.replace(' ', '_')}")
        ensure_dir(step_dir)
        before = current.copy()
        before.save(os.path.join(step_dir, "00_before.png"))

        reference = flip_reference(Image.open(job.reference).convert("RGB"), job.reference_flip)
        reference_canvas = prepare_reference_canvas(reference, before.size)
        reference.save(os.path.join(step_dir, "reference_original.png"))
        reference_canvas.save(os.path.join(step_dir, "reference_context.png"))
        short, long = insertion_prompts(job)
        seed = args.seed + index * 10000
        raw = pipe(
            image=[before, reference_canvas],
            prompt=short,
            prompt_2=long,
            width=args.width,
            height=args.height,
            max_area=args.width * args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator_for(args.device, seed),
        ).images[0].convert("RGB")
        raw.save(os.path.join(step_dir, "01_direct_raw.png"))

        mask, metrics = diagnose_object(before, raw, job, detector, segmenter, args)
        final = raw
        if mask is not None:
            mask = mask.filter(ImageFilter.MaxFilter(2 * args.post_mask_dilate_px + 1))
            soft = mask.filter(ImageFilter.GaussianBlur(args.post_mask_feather_px))
            mask.save(os.path.join(step_dir, "02_post_sam_mask.png"))
            make_overlay(raw, soft).save(os.path.join(step_dir, "02_post_sam_overlay.png"))
            if args.protect_outside_object:
                final = protect_outside_mask(before, raw, soft)
                final.save(os.path.join(step_dir, "03_protected.png"))

        final.save(os.path.join(step_dir, "04_final.png"))
        current = final
        record = {"step": index, "object": asdict(job), "seed": seed, **metrics}
        save_json(record, os.path.join(step_dir, "step_summary.json"))
        summary.append(record)
        print(f"[{index}/{len(jobs)}] {job.name}: detected={metrics.get('detected', False)}")

    current.save(os.path.join(args.out_dir, "FINAL.png"))
    save_json(summary, os.path.join(args.out_dir, "summary_e17.json"))
    print(f"Done: {os.path.join(args.out_dir, 'FINAL.png')}")


if __name__ == "__main__":
    main()
