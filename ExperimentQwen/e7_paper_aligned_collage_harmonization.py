"""E7: paper-aligned collage harmonization with native Qwen inpainting.

The reference foreground is pasted exactly into the current scene. The collage
is then supplied as the *only* edit image, so Qwen2.5-VL and the VAE receive the
same image as prescribed by Qwen-Image's trained dual-encoding path. A narrow
boundary/contact mask lets the native inpainting pipeline harmonize seams and
support shadows while preserving background and object core.

There is one Qwen harmonization pass per object and no custom attention/KV code.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from tqdm.auto import tqdm

from e1_baseline import load_pipe, make_generator, save_json
from e2_sam_collage_repaint import composite, place_cutout, probe_placement
from e3_prompt_suite import generate_references, load_suite, reference_key, select_cases, slug
from e5_spatial_kv_collage import (
    RMBG2Cutout,
    generate_base,
    generate_rmbg_cutouts,
)


HERE = Path(__file__).resolve().parent


def morphology(mask: Image.Image, radius: int, operation: str) -> Image.Image:
    if radius <= 0:
        return mask.copy()
    size = 2 * radius + 1
    if operation == "dilate":
        return mask.filter(ImageFilter.MaxFilter(size))
    if operation == "erode":
        return mask.filter(ImageFilter.MinFilter(size))
    raise ValueError(operation)


def make_harmonization_mask(paste_mask: Image.Image, placed_box, args):
    """Return editable boundary/contact mask plus interpretable components."""
    # Make morphology deterministic and prevent faint alpha across the scene
    # from becoming editable.
    binary = paste_mask.convert("L").point(
        lambda value: 255 if value >= args.foreground_threshold else 0
    )
    outer = morphology(binary, args.outer_boundary_px, "dilate")
    core = morphology(binary, args.inner_core_px, "erode")
    ring = ImageChops.subtract(outer, core)

    x0, y0, x1, y1 = map(int, placed_box)
    width, height = x1 - x0, y1 - y0
    contact = Image.new("L", binary.size)
    draw = ImageDraw.Draw(contact)
    contact_width = max(4, round(width * args.contact_width_fraction))
    contact_height = max(3, round(height * args.contact_height_fraction))
    cx = (x0 + x1) // 2
    top = max(0, y1 - round(contact_height * .35))
    bottom = min(binary.height, top + contact_height)
    draw.ellipse(
        (max(0, cx - contact_width // 2), top,
         min(binary.width, cx + contact_width // 2), bottom),
        fill=255,
    )
    # Do not repaint the protected object core even where the contact ellipse
    # overlaps the lower portion of the foreground.
    contact = ImageChops.subtract(contact, core)
    editable = ImageChops.lighter(ring, contact)
    if args.mask_feather_px > 0:
        editable = editable.filter(ImageFilter.GaussianBlur(args.mask_feather_px))
    return editable, {"binary": binary, "core": core, "ring": ring, "contact": contact}


@torch.inference_mode()
def harmonize(editor, collage, mask, name, args, seed):
    cfg = args.true_cfg_scale > 1.0
    prompt = (
        f"Image 1 is the final scene composition and already contains the exact {name}. "
        f"Perform localized inpainting only in the supplied mask around the {name}'s boundary and contact area. "
        "Remove cutout seams, match local illumination and color temperature, and create a subtle physically "
        "consistent contact shadow. Preserve the object's shape, pose, proportions, texture, markings and colors. "
        "Do not move, resize, replace, duplicate or redesign anything. Preserve all unmasked pixels exactly."
    )
    result = editor(
        image=collage,
        mask_image=mask,
        prompt=prompt,
        negative_prompt=args.negative_prompt if cfg else None,
        true_cfg_scale=args.true_cfg_scale,
        strength=args.inpaint_strength,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        generator=make_generator(args.device, seed),
        padding_mask_crop=args.padding_mask_crop,
    )
    return result.images[0].convert("RGB")


def run_case(planner, editor, case, references, cutouts, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    step_dir = case_dir / "steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(planner, case, args, case_dir / "base.png")
    occupied = Image.new("L", current.size)
    history = []
    objects = case["objects"][:args.max_objects]

    for index, item in enumerate(tqdm(
        objects, desc=f"E7 case {case_id:03d}", unit="object", leave=False
    ), 1):
        name, key = item["name"], reference_key(item)
        final_path = step_dir / f"{index:02d}_{slug(name)}_paper_aligned_final.png"
        paste_mask_path = step_dir / f"{index:02d}_{slug(name)}_paste_mask.png"
        if args.resume and final_path.is_file() and paste_mask_path.is_file():
            current = Image.open(final_path).convert("RGB")
            occupied = ImageChops.lighter(
                occupied, Image.open(paste_mask_path).convert("L")
            )
            history.append({"step": index, "name": name, "status": "resumed", "final": str(final_path)})
            continue
        record = references.get(key, {})
        if record.get("status") != "ready":
            if args.missing_policy == "skip":
                history.append({"step": index, "name": name, "status": "skipped_missing_reference"})
                continue
            raise FileNotFoundError(record)

        before = current
        placement_box, probe = probe_placement(
            planner, before, name, cutouts[key], occupied, args,
            args.seed + case_id * 100000 + index * 1000,
            step_dir / f"{index:02d}_{slug(name)}_placement_heatmap.png",
        )
        object_canvas, paste_mask, placed_box = place_cutout(
            cutouts[key], placement_box, before.size, args.object_scale
        )
        collage = composite(before, object_canvas, paste_mask)
        editable, components = make_harmonization_mask(paste_mask, placed_box, args)

        prefix = step_dir / f"{index:02d}_{slug(name)}"
        before.save(f"{prefix}_before.png")
        collage.save(f"{prefix}_collage.png")
        paste_mask.save(paste_mask_path)
        editable.save(f"{prefix}_editable_mask.png")
        for component_name, component in components.items():
            component.save(f"{prefix}_{component_name}_mask.png")

        seed = args.seed + case_id * 10000 + index * 100
        current = harmonize(editor, collage, editable, name, args, seed)
        current.save(final_path)
        occupied = ImageChops.lighter(occupied, paste_mask)
        editable_fraction = float(np.asarray(editable, dtype=np.float32).mean() / 255.0)
        history.append({
            "step": index,
            "name": name,
            "status": "generated",
            "seed": seed,
            "collage": f"{prefix}_collage.png",
            "editable_mask": f"{prefix}_editable_mask.png",
            "final": str(final_path),
            "placed_box": list(placed_box),
            "editable_fraction": editable_fraction,
            "placement_probe": probe,
            "conditioning": "one collage through Qwen2.5-VL and VAE",
            "attention_modification": None,
            "custom_postprocess": None,
        })
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "status": "complete", "objects": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    p.add_argument("--out_dir", default="results/qwen_e7_paper_aligned_harmonization")
    p.add_argument("--case_ids", type=int, nargs="+")
    p.add_argument("--max_objects", type=int, default=3, choices=(1, 2, 3))
    p.add_argument("--missing_policy", choices=("skip", "error"), default="skip")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--lightning_repo", default="lightx2v/Qwen-Image-Lightning")
    p.add_argument("--lightning_weight", default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors")
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--object_seed", type=int, default=1337)
    p.add_argument("--true_cfg_scale", type=float, default=1.0)
    p.add_argument("--negative_prompt", default=" ")
    p.add_argument("--rmbg_model_id", default="briaai/RMBG-2.0")
    p.add_argument("--rmbg_revision", default="54c725d3b17ca83aba490092de8acf6118b8bb06")
    p.add_argument("--rmbg_device", default="cuda")
    p.add_argument("--rmbg_input_size", type=int, default=1024)
    p.add_argument("--rmbg_crop_threshold", type=int, default=8)
    p.add_argument("--probe_steps", type=int, default=4)
    p.add_argument("--probe_quantile", type=float, default=.88)
    p.add_argument("--probe_blur", type=float, default=1.2)
    p.add_argument("--box_margin", type=int, default=24)
    p.add_argument("--occupancy_margin", type=int, default=24)
    p.add_argument("--default_object_height", type=float, default=.25)
    p.add_argument("--object_height_priors")
    p.add_argument("--object_scale", type=float, default=.92)
    p.add_argument("--foreground_threshold", type=int, default=24)
    p.add_argument("--outer_boundary_px", type=int, default=14)
    p.add_argument("--inner_core_px", type=int, default=8)
    p.add_argument("--contact_width_fraction", type=float, default=.65)
    p.add_argument("--contact_height_fraction", type=float, default=.10)
    p.add_argument("--mask_feather_px", type=float, default=2.0)
    p.add_argument("--inpaint_strength", type=float, default=1.0)
    p.add_argument("--padding_mask_crop", type=int, default=None)
    return p.parse_args()


def validate_args(args):
    if not 0 <= args.foreground_threshold <= 255:
        raise ValueError("foreground_threshold must be in [0,255]")
    if args.outer_boundary_px < 0 or args.inner_core_px < 0:
        raise ValueError("boundary/core radii must be non-negative")
    if not 0 < args.contact_width_fraction <= 1.5:
        raise ValueError("contact_width_fraction must be in (0,1.5]")
    if not 0 < args.contact_height_fraction <= .5:
        raise ValueError("contact_height_fraction must be in (0,.5]")
    if not 0 < args.inpaint_strength <= 1:
        raise ValueError("inpaint_strength must be in (0,1]")


def main():
    args = parse_args()
    validate_args(args)
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E7 targets diffusers 0.40.0; found {diffusers.__version__}")
        import kornia  # noqa: F401
        from diffusers import QwenImageEditInpaintPipeline
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E7") from exc

    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")

    # Planner performs base/reference generation and counterfactual placement.
    # from_pipe reuses those exact loaded components; it does not load another
    # transformer, VAE, or text encoder.
    planner = load_pipe(args)
    references = generate_references(planner, cases, args, out, prompt_file)
    editor = QwenImageEditInpaintPipeline.from_pipe(planner)
    editor.set_progress_bar_config(disable=False)

    segmenter = RMBG2Cutout(
        args.rmbg_model_id, args.rmbg_revision, args.rmbg_device, args.rmbg_input_size
    )
    try:
        cutouts = generate_rmbg_cutouts(segmenter, references, args, out)
    finally:
        segmenter.close()

    summary = []
    for case in tqdm(cases, desc="E7 paper-aligned suite", unit="case"):
        summary.append(run_case(planner, editor, case, references, cutouts, args, out))
        save_json(summary, out / "summary.partial.json")
    save_json({
        "method": "single-collage dual encoding plus native localized inpainting",
        "paper_alignment": {
            "vl_input": "collage",
            "vae_input": "same collage",
            "custom_attention": False,
            "editable_region": "object boundary plus support-contact region",
        },
        "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
