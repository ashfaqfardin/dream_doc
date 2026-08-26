"""
E14: Generic FLUX.1-Kontext Reference Replacer
==============================================

Goal
----
Generic object replacement/editing code for cases like:
- an image already contains an object (e.g. a bicycle in a room)
- you want to replace that object with the object from another reference image
- keep the new object in nearly the same place/scale as the original target
- optionally change orientation (for example mirror left-right / opposite direction)
- preserve the surrounding scene as much as possible

This script is NOT hard-coded to bicycles. It can be used for any replaceable object,
provided you give either:
  1) a mask image, or
  2) a bounding box, or
  3) enable auto-detection via GroundingDINO (if installed)

Pipeline
--------
1. Load source image and reference image.
2. Obtain the target mask (mask path, box, or auto-detect).
3. Optionally flip the reference (horizontal/vertical) to encourage opposite direction.
4. Run FLUX.1 Kontext inpainting to replace the masked object with the reference object.
5. Optional second refinement pass on the same masked region for stronger identity matching.

Recommended for your bicycle case
---------------------------------
Use:
  --object_name bicycle
  --flip_reference horizontal
  --orientation_instruction "The bicycle must face the opposite left-right direction from the original bicycle, so that its rear side faces toward the window."

Requirements
------------
pip install -U diffusers transformers accelerate pillow torch

Optional for auto detection:
  pip install -U transformers
and internet/model access for GroundingDINO.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import torch

try:
    from diffusers import FluxKontextInpaintPipeline
except Exception as exc:
    raise ImportError(
        "Could not import FluxKontextInpaintPipeline. Install a recent diffusers version with FLUX support."
    ) from exc


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"


@dataclass
class DetectionResult:
    label: str
    score: float
    box: Tuple[int, int, int, int]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def parse_box(box_str: str) -> Tuple[int, int, int, int]:
    parts = [int(float(x.strip())) for x in box_str.split(",")]
    if len(parts) != 4:
        raise ValueError("--box must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid box coordinates; need x1>x0 and y1>y0")
    return x0, y0, x1, y1



def clamp_box(box: Tuple[int, int, int, int], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    W, H = size
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))
    return x0, y0, x1, y1



def expand_box(box: Tuple[int, int, int, int], size: Tuple[int, int], frac: float) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    dx = int(round(bw * frac))
    dy = int(round(bh * frac))
    W, H = size
    return clamp_box((x0 - dx, y0 - dy, x1 + dx, y1 + dy), (W, H))



def box_to_mask(box: Tuple[int, int, int, int], size: Tuple[int, int], blur_px: float = 10.0) -> Image.Image:
    W, H = size
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, fill=255)
    if blur_px > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    return img



def load_and_resize(img_path: str, width: Optional[int], height: Optional[int]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    if width is not None and height is not None:
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img



def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



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



def blur_mask(mask: Image.Image, radius: float) -> Image.Image:
    out = mask.convert("L")
    if radius > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return out



def flip_reference_if_needed(img: Image.Image, mode: str) -> Image.Image:
    mode = mode.lower()
    if mode == "none":
        return img
    if mode == "horizontal":
        return ImageOps.mirror(img)
    if mode == "vertical":
        return ImageOps.flip(img)
    if mode == "both":
        return ImageOps.flip(ImageOps.mirror(img))
    raise ValueError(f"Unknown flip mode: {mode}")


# -----------------------------------------------------------------------------
# Optional auto-detection
# -----------------------------------------------------------------------------


def auto_detect_box(image: Image.Image, object_name: str, box_threshold: float = 0.20) -> DetectionResult:
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(
            "Auto-detection requires transformers. Install it or provide --mask_image / --box."
        ) from exc

    # GroundingDINO zero-shot detection pipeline
    # Candidate label works best with trailing period in many GroundingDINO examples.
    detector = pipeline(
        task="zero-shot-object-detection",
        model="IDEA-Research/grounding-dino-tiny",
        device=0 if torch.cuda.is_available() else -1,
    )
    labels = [object_name, f"{object_name}."]
    preds = detector(image, candidate_labels=labels)

    best = None
    for p in preds:
        score = float(p.get("score", 0.0))
        label = str(p.get("label", object_name))
        box = p.get("box", {})
        if score < box_threshold:
            continue
        cur = DetectionResult(
            label=label,
            score=score,
            box=(int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])),
        )
        if best is None or cur.score > best.score:
            best = cur

    if best is None:
        raise RuntimeError(
            f"Auto-detection could not find '{object_name}'. Provide --box or --mask_image instead."
        )
    return best


# -----------------------------------------------------------------------------
# Prompting
# -----------------------------------------------------------------------------


def build_stage1_prompt(object_name: str, orientation_instruction: str, preserve_instruction: str) -> str:
    return (
        f"Replace the masked {object_name} with the exact {object_name} from the reference image. "
        f"Keep the replacement in almost exactly the same location, footprint, and approximate size as the original masked object. "
        f"Preserve the reference object's color, materials, texture, structure, proportions, and distinctive design details. "
        f"Integrate it naturally into the scene with realistic perspective, lighting, and contact shadows. "
        f"{orientation_instruction} "
        f"{preserve_instruction} "
        f"Do not add extra objects. Do not change unrelated parts of the image."
    )



def build_stage2_prompt(object_name: str, orientation_instruction: str, preserve_instruction: str) -> str:
    return (
        f"Refine only the masked {object_name} so it matches the reference image even more closely. "
        f"Keep the current placement, pose, location, viewpoint, perspective, and approximate size from the existing edited image. "
        f"Strengthen identity fidelity: preserve the reference object's exact colors, materials, texture, structure, and distinctive design. "
        f"{orientation_instruction} "
        f"{preserve_instruction} "
        f"Do not modify the surrounding scene. Do not create duplicates."
    )


# -----------------------------------------------------------------------------
# Core generation
# -----------------------------------------------------------------------------


def run_inpaint(
    pipe: FluxKontextInpaintPipeline,
    image: Image.Image,
    mask_image: Image.Image,
    image_reference: Image.Image,
    prompt: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    device: str,
) -> Image.Image:
    gen = torch.Generator(device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        image_reference=image_reference,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=gen,
    ).images[0]
    return out.convert("RGB")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    p.add_argument("--input_image", required=True, help="Base/source image containing the object to replace")
    p.add_argument("--reference_image", required=True, help="Reference image of the desired replacement object")
    p.add_argument("--object_name", required=True, help="Target object category, e.g. bicycle, chair, vase")
    p.add_argument("--out_dir", default="results/e14_generic_kontext_reference_replacer")

    # Mask specification
    p.add_argument("--mask_image", default=None, help="Optional mask image (white=edit, black=preserve)")
    p.add_argument("--box", default=None, help="Optional box x0,y0,x1,y1 around object to replace")
    p.add_argument("--auto_detect", action="store_true", help="Auto-detect the object with GroundingDINO if mask/box not provided")
    p.add_argument("--expand_box_frac", type=float, default=0.08, help="Expand detected/manual box before making the mask")
    p.add_argument("--mask_blur_px", type=float, default=12.0, help="Blur radius for the inpaint mask")

    # Orientation / behavior
    p.add_argument("--flip_reference", choices=["none", "horizontal", "vertical", "both"], default="none")
    p.add_argument(
        "--orientation_instruction",
        default="Keep the object naturally oriented in the scene.",
        help="Extra instruction for orientation/direction. For example: 'The bicycle must face the opposite left-right direction from the original bicycle, so that its rear side faces toward the window.'",
    )
    p.add_argument(
        "--preserve_instruction",
        default="Preserve the camera view, background, room layout, floor, walls, lighting, and all unrelated objects exactly as they are.",
        help="Global preservation instruction",
    )

    # Generation params
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--width", type=int, default=None, help="Optional resize width; default = source width")
    p.add_argument("--height", type=int, default=None, help="Optional resize height; default = source height")
    p.add_argument("--seed", type=int, default=42)

    # Two-pass generation
    p.add_argument("--two_pass_refine", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage1_strength", type=float, default=0.95)
    p.add_argument("--stage1_steps", type=int, default=28)
    p.add_argument("--stage1_guidance", type=float, default=2.5)
    p.add_argument("--stage2_strength", type=float, default=0.70)
    p.add_argument("--stage2_steps", type=int, default=24)
    p.add_argument("--stage2_guidance", type=float, default=2.5)

    return p.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    save_json(vars(args), os.path.join(args.out_dir, "config_e14.json"))

    # Load images
    source0 = Image.open(args.input_image).convert("RGB")
    target_w = args.width or source0.width
    target_h = args.height or source0.height
    source = source0.resize((target_w, target_h), Image.Resampling.LANCZOS)
    ref0 = Image.open(args.reference_image).convert("RGB")
    ref = flip_reference_if_needed(ref0, args.flip_reference)

    source.save(os.path.join(args.out_dir, "source.png"))
    ref.save(os.path.join(args.out_dir, "reference_used.png"))

    # Mask creation
    det_info = None
    if args.mask_image:
        mask = Image.open(args.mask_image).convert("L").resize((target_w, target_h), Image.Resampling.BILINEAR)
    else:
        box = None
        if args.box:
            box = parse_box(args.box)
            box = clamp_box(box, (source0.width, source0.height))
        elif args.auto_detect:
            det = auto_detect_box(source0, args.object_name)
            det_info = {"label": det.label, "score": det.score, "box": det.box}
            box = det.box
        else:
            raise ValueError("Provide either --mask_image, --box, or --auto_detect")

        if (target_w, target_h) != (source0.width, source0.height):
            sx = target_w / source0.width
            sy = target_h / source0.height
            x0, y0, x1, y1 = box
            box = (int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy)))
        box = expand_box(box, (target_w, target_h), args.expand_box_frac)
        mask = box_to_mask(box, (target_w, target_h), blur_px=args.mask_blur_px)
        det_info = det_info or {"box": box}

    mask = blur_mask(mask, args.mask_blur_px)
    mask.save(os.path.join(args.out_dir, "mask.png"))
    make_overlay(source, mask).save(os.path.join(args.out_dir, "mask_overlay.png"))
    if det_info is not None:
        save_json(det_info, os.path.join(args.out_dir, "detection_or_box.json"))

    # Load pipe
    dtype = getattr(torch, args.torch_dtype)
    pipe = FluxKontextInpaintPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe = maybe_enable_offload(pipe, args.device, args.cpu_offload)

    # Stage 1
    prompt1 = build_stage1_prompt(args.object_name, args.orientation_instruction, args.preserve_instruction)
    stage1 = run_inpaint(
        pipe=pipe,
        image=source,
        mask_image=mask,
        image_reference=ref,
        prompt=prompt1,
        strength=args.stage1_strength,
        steps=args.stage1_steps,
        guidance_scale=args.stage1_guidance,
        seed=args.seed,
        width=target_w,
        height=target_h,
        device=args.device,
    )
    stage1.save(os.path.join(args.out_dir, "stage1.png"))
    with open(os.path.join(args.out_dir, "stage1_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt1)

    final_img = stage1
    if args.two_pass_refine:
        prompt2 = build_stage2_prompt(args.object_name, args.orientation_instruction, args.preserve_instruction)
        final_img = run_inpaint(
            pipe=pipe,
            image=stage1,
            mask_image=mask,
            image_reference=ref,
            prompt=prompt2,
            strength=args.stage2_strength,
            steps=args.stage2_steps,
            guidance_scale=args.stage2_guidance,
            seed=args.seed + 17,
            width=target_w,
            height=target_h,
            device=args.device,
        )
        with open(os.path.join(args.out_dir, "stage2_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt2)
        final_img.save(os.path.join(args.out_dir, "final.png"))
    else:
        final_img.save(os.path.join(args.out_dir, "final.png"))

    print(f"Done. Results saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
