"""E15: native broad-mask collage inpainting with one Qwen pipeline.

The method makes a clean RMBG cutout, automatically proposes placement, forms
an aligned collage C=B+O, and runs Qwen's native edit-inpaint pipeline once per
mask-margin ablation. White mask pixels cover a broad interaction box around
the pasted object; black pixels are preserved by the pipeline. There are no
feature hooks, manual latent blends, second edit passes, or external output
composites. Only the selected margin is propagated to the next object.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from tqdm.auto import tqdm

from e1_baseline import fit, lightning_scheduler, make_generator, prepare_peft_lora_backend, save_json
from e2_sam_collage_repaint import (
    box_from_heatmap, composite, latent_image, place_cutout, save_heatmap, spatial_energy,
)
from e3_prompt_suite import load_suite, reference_key, resolve_input, select_cases, slug
from e5_spatial_kv_collage import RMBG2Cutout, generate_rmbg_cutouts
from e12_spatial_reference_card_insertion import E12Evaluator


HERE = Path(__file__).resolve().parent
REFERENCE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def load_inpaint_pipe(args):
    from diffusers import QwenImageEditInpaintPipeline

    stages = tqdm(total=3, desc="Loading Qwen native inpaint", unit="stage", dynamic_ncols=True)
    pipe = QwenImageEditInpaintPipeline.from_pretrained(
        args.model_id, scheduler=lightning_scheduler(), dtype=torch.bfloat16
    )
    stages.update()
    stages.set_description("Loading 8-step Lightning LoRA")
    prepare_peft_lora_backend()
    pipe.load_lora_weights(
        args.lightning_repo, weight_name=args.lightning_weight, adapter_name="lightning"
    )
    pipe.set_adapters(["lightning"], adapter_weights=[args.lora_scale])
    stages.update()
    stages.set_description("Moving inpaint pipeline to GPU")
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    stages.update()
    stages.close()
    return pipe


@torch.inference_mode()
def inpaint(pipe, image, mask, prompt, args, seed, strength=None, output_type="pil", num_steps=None):
    cfg = args.true_cfg_scale > 1.0
    result = pipe(
        image=fit(image, (args.width, args.height)),
        mask_image=mask.convert("L").resize((args.width, args.height), Image.Resampling.LANCZOS),
        prompt=prompt,
        negative_prompt=args.negative_prompt if cfg else None,
        true_cfg_scale=args.true_cfg_scale,
        strength=args.inpaint_strength if strength is None else strength,
        num_inference_steps=args.steps if num_steps is None else num_steps,
        width=args.width,
        height=args.height,
        padding_mask_crop=None,
        generator=make_generator(args.device, seed),
        output_type=output_type,
    )
    if output_type == "latent":
        return latent_image(result)
    return result.images[0].convert("RGB")


def provided_reference(item):
    directory = HERE / "references"
    key = reference_key(item)
    if not directory.is_dir():
        return None
    candidates = [directory / f"{key}{extension}" for extension in REFERENCE_EXTENSIONS]
    candidates += [directory / f"{slug(item['name'])}{extension}" for extension in REFERENCE_EXTENSIONS]
    return next((path for path in candidates if path.is_file()), None)


def generate_references(pipe, cases, args, out, prompt_file):
    directory = out / "references"
    directory.mkdir(parents=True, exist_ok=True)
    unique = {}
    for case in cases:
        for item in case["objects"][: args.max_objects or None]:
            unique.setdefault(reference_key(item), item)
    records = {}
    full_mask = Image.new("L", (args.width, args.height), 255)
    for key, item in tqdm(unique.items(), desc="E15 reference objects", unit="object"):
        name = item["name"]
        supplied = provided_reference(item)
        if supplied is not None:
            records[key] = {
                "name": name, "status": "ready", "image": str(supplied.resolve()),
                "source": "provided_reference", "seed": None,
            }
            continue
        source = resolve_input(item.get("canny_file"), prompt_file)
        if source is None:
            record = {"name": name, "status": "missing", "canny_file": item.get("canny_file")}
            records[key] = record
            if args.missing_policy == "error":
                raise FileNotFoundError(f"Missing reference input for {name}")
            continue
        target = directory / f"{key}.png"
        if not target.is_file() or not args.resume:
            sketch = fit(Image.open(source), (args.width, args.height))
            prompt = (
                f"The input is a sketch of one {name}. Reconstruct one photorealistic {name}, preserving its "
                "complete geometry, pose, viewpoint, proportions, components, and distinctive details. Center it "
                "on a plain neutral background with no other objects, frame, text, or scenery."
            )
            reference = inpaint(
                pipe, sketch, full_mask, prompt, args, args.object_seed,
                strength=1.0,
            )
            reference.save(target)
        records[key] = {
            "name": name, "status": "ready", "image": str(target),
            "source": "generated_from_canny", "seed": args.object_seed,
        }
    save_json(records, out / "references.json")
    return records


def generate_base(pipe, case, args, path):
    if args.resume and path.is_file():
        return fit(Image.open(path), (args.width, args.height))
    blank = Image.new("RGB", (args.width, args.height), "white")
    full_mask = Image.new("L", blank.size, 255)
    prompt = "Replace the complete blank image with this scene: " + case["base_prompt"] + " Fill the frame."
    base = inpaint(
        pipe, blank, full_mask, prompt, args,
        args.seed + int(case["id"]) * 10000, strength=1.0,
    )
    base.save(path)
    return base


def probe_placement(pipe, scene, name, cutout, occupied, args, seed, path):
    """Matched full-frame inpaint probes; their latent delta proposes location."""
    full_mask = Image.new("L", scene.size, 255)
    add_prompt = (
        f"Add one complete {name} at the most physically plausible unoccupied location. Respect support surfaces, "
        "scale, perspective, and existing objects while preserving the scene."
    )
    keep_prompt = "Reconstruct this scene without adding, removing, moving, or changing any object."
    add = inpaint(
        pipe, scene, full_mask, add_prompt, args, seed, strength=1.0,
        output_type="latent", num_steps=args.probe_steps,
    )
    keep = inpaint(
        pipe, scene, full_mask, keep_prompt, args, seed, strength=1.0,
        output_type="latent", num_steps=args.probe_steps,
    )
    heat = spatial_energy(add - keep, args.width, args.height)
    box, smoothed = box_from_heatmap(heat, name, cutout, scene.size, args, occupied)
    save_heatmap(scene, smoothed, box, path)
    return box, {
        "backend": "matched_native_inpaint_latent_delta", "latent_shape": list(add.shape),
        "heat_min": float(smoothed.min()), "heat_max": float(smoothed.max()), "box": list(box),
    }


def broad_interaction_mask(size, placed_box, margin_fraction, args):
    width, height = size
    x0, y0, x1, y1 = placed_box
    object_w, object_h = x1 - x0, y1 - y0
    mx = round(object_w * margin_fraction)
    my = round(object_h * margin_fraction)
    shadow = round(object_h * args.shadow_extension_fraction)
    box = (
        max(0, x0 - mx), max(0, y0 - my),
        min(width, x1 + mx), min(height, y1 + my + shadow),
    )
    mask = Image.new("L", size)
    draw = ImageDraw.Draw(mask)
    radius = max(2, round(min(object_w, object_h) * args.mask_corner_fraction))
    draw.rounded_rectangle(box, radius=radius, fill=255)
    if args.mask_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(args.mask_blur))
    return mask, box


def prompt_for(name):
    return (
        f"The input is a complete scene with one roughly pasted {name}. Edit only the masked area. Preserve that "
        f"exact {name}'s recognizable structure, proportions, components, colors, materials, textures, and details. "
        "Harmonize its boundary, perspective, illumination, contact, and natural shadow with the surrounding scene. "
        "Keep the complete object inside the frame. Do not replace, remove, duplicate, or redesign it. Preserve all "
        "unmasked content exactly. Return the complete scene, never an isolated object or white background."
    )


def margin_name(value):
    return f"margin_{round(value * 100):02d}"


def run_case(pipe, case, references, cutouts, evaluator, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    occupied = Image.new("L", current.size)
    history = []

    for index, item in enumerate(tqdm(
        case["objects"][: args.max_objects or None],
        desc=f"E15 case {case_id:03d}", unit="object", leave=False,
    ), 1):
        name, key = item["name"], reference_key(item)
        prefix = steps_dir / f"{index:02d}_{slug(name)}"
        selected_name = margin_name(args.selected_margin)
        selected_path = Path(f"{prefix}_{selected_name}.png")
        alpha_path = Path(f"{prefix}_paste_alpha.png")
        if args.resume and selected_path.is_file() and alpha_path.is_file():
            current = Image.open(selected_path).convert("RGB")
            occupied = ImageChops.lighter(occupied, Image.open(alpha_path).convert("L"))
            history.append({"step": index, "name": name, "status": "resumed", "final": str(selected_path)})
            continue
        record = references.get(key, {})
        if record.get("status") != "ready" or key not in cutouts:
            if args.missing_policy == "error":
                raise FileNotFoundError(f"No reference/cutout available for {name}: {record}")
            history.append({"step": index, "name": name, "status": "skipped_missing_reference"})
            continue

        before = fit(current, (args.width, args.height))
        box, probe = probe_placement(
            pipe, before, name, cutouts[key], occupied, args,
            args.seed + case_id * 100000 + index * 1000,
            Path(f"{prefix}_placement_heatmap.png"),
        )
        object_canvas, paste_alpha, placed_box = place_cutout(
            cutouts[key], box, before.size, args.object_scale
        )
        collage = composite(before, object_canvas, paste_alpha)
        before.save(Path(f"{prefix}_before.png"))
        collage.save(Path(f"{prefix}_collage.png"))
        paste_alpha.save(alpha_path)
        seed = args.seed + case_id * 10000 + index * 100
        variants = {}
        for margin in args.mask_margins:
            variant = margin_name(margin)
            mask, interaction_box = broad_interaction_mask(before.size, placed_box, margin, args)
            mask_path = Path(f"{prefix}_{variant}_mask.png")
            output_path = Path(f"{prefix}_{variant}.png")
            mask.save(mask_path)
            output = inpaint(pipe, collage, mask, prompt_for(name), args, seed)
            output.save(output_path)
            metrics = evaluator.evaluate(
                name, before, output, cutouts[key], Path(f"{prefix}_{variant}")
            ) if evaluator else None
            variants[variant] = {
                "margin_fraction": margin, "interaction_box": list(interaction_box),
                "mask": str(mask_path), "image": str(output_path), "metrics": metrics,
            }
        if selected_name not in variants:
            raise ValueError("selected_margin must be included in mask_margins")
        current = Image.open(variants[selected_name]["image"]).convert("RGB")
        occupied = ImageChops.lighter(occupied, paste_alpha)
        history.append({
            "step": index, "name": name, "status": "generated", "seed": seed,
            "placed_box": list(placed_box), "probe": probe, "variants": variants,
            "selected_margin": args.selected_margin, "final": variants[selected_name]["image"],
            "pipeline_background_preservation": "native inpaint mask overlay",
            "external_postprocess": None,
        })
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "status": "complete", "objects": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    p.add_argument("--out_dir", default="results/qwen_e15_native_broad_mask_inpaint")
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
    p.add_argument("--inpaint_strength", type=float, default=.85)
    p.add_argument("--mask_margins", type=float, nargs="+", default=[.10, .20, .30])
    p.add_argument("--selected_margin", type=float, default=.20)
    p.add_argument("--shadow_extension_fraction", type=float, default=.12)
    p.add_argument("--mask_corner_fraction", type=float, default=.10)
    p.add_argument("--mask_blur", type=float, default=4.0)
    p.add_argument("--rmbg_model_id", default="briaai/RMBG-2.0")
    p.add_argument("--rmbg_revision", default="54c725d3b17ca83aba490092de8acf6118b8bb06")
    p.add_argument("--rmbg_device", default="cuda")
    p.add_argument("--rmbg_input_size", type=int, default=1024)
    p.add_argument("--rmbg_crop_threshold", type=int, default=8)
    p.add_argument("--probe_steps", type=int, default=4, help="Compatibility field recorded for placement")
    p.add_argument("--probe_quantile", type=float, default=.88)
    p.add_argument("--probe_blur", type=float, default=1.2)
    p.add_argument("--box_margin", type=int, default=64)
    p.add_argument("--occupancy_margin", type=int, default=32)
    p.add_argument("--default_object_height", type=float, default=.25)
    p.add_argument("--object_height_priors")
    p.add_argument("--object_scale", type=float, default=.88)
    p.add_argument("--evaluation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--metric_model_id", default="facebook/dinov2-base")
    p.add_argument("--metric_device", default="cpu")
    p.add_argument("--metric_localization_grid", type=int, default=128)
    p.add_argument("--metric_change_threshold", type=float, default=12.0)
    p.add_argument("--metric_crop_margin", type=float, default=.12)
    p.add_argument("--qualitative_tile_size", type=int, default=320)
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 < args.inpaint_strength <= 1:
        raise ValueError("inpaint_strength must be in (0,1]")
    if any(value < 0 or value >= 1 for value in args.mask_margins):
        raise ValueError("mask_margins must lie in [0,1)")
    if not any(abs(value - args.selected_margin) < 1e-9 for value in args.mask_margins):
        raise ValueError("selected_margin must be included in mask_margins")
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E15 targets diffusers 0.40.0; found {diffusers.__version__}")
        import kornia  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E15") from exc

    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")
    pipe = load_inpaint_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    segmenter = RMBG2Cutout(args.rmbg_model_id, args.rmbg_revision, args.rmbg_device, args.rmbg_input_size)
    evaluator = None
    try:
        cutouts = generate_rmbg_cutouts(segmenter, references, args, out)
        evaluator = E12Evaluator(segmenter, args) if args.evaluation else None
        summary = []
        for case in tqdm(cases, desc="E15 native-inpaint suite", unit="case"):
            summary.append(run_case(pipe, case, references, cutouts, evaluator, args, out))
            save_json(summary, out / "summary.partial.json")
    finally:
        if evaluator is not None:
            evaluator.close()
        segmenter.close()
    save_json({
        "method": "single-pass native broad-mask collage inpainting",
        "qwen_pipeline_instances": 1, "feature_hooks": False,
        "mask_margins": args.mask_margins, "selected_margin": args.selected_margin,
        "external_postprocess": None, "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
