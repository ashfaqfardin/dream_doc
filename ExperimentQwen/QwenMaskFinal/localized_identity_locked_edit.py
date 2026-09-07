"""Localized, identity-locked multistep Qwen collage harmonization.

Spatial ownership for each step:
  * reference core: exact pixels from the placed object,
  * boundary/contact halo: Qwen native inpainting output,
  * everything else: exact pixels from the previous scene.

The user rectangle controls placement only. The cutout alpha controls editing
and compositing; rectangular seams are never introduced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from tqdm.auto import tqdm

from generate_base_images import (
    lightning_scheduler,
    make_generator,
    prepare_peft_lora_backend,
    save_json,
)
from multistep_mask_collage_edit import (
    DEFAULT_BASES,
    DEFAULT_MASKS,
    DEFAULT_PROMPTS,
    DEFAULT_REFERENCES,
    background_alpha,
    crop_cutout,
    load_cases,
    make_collage,
    place_in_rectangle,
    reference_path,
    slug,
)
from placement_masks import extract_rectangles


HERE = Path(__file__).resolve().parent


def load_inpaint_pipeline(args):
    from diffusers import QwenImageEditInpaintPipeline

    loading = tqdm(total=3, desc="Loading localized Qwen inpaint", unit="stage", dynamic_ncols=True)
    pipe = QwenImageEditInpaintPipeline.from_pretrained(
        args.model_id,
        scheduler=lightning_scheduler(),
        torch_dtype=torch.bfloat16,
    )
    loading.update()
    loading.set_description("Loading 8-step Lightning LoRA")
    prepare_peft_lora_backend()
    pipe.load_lora_weights(
        args.lightning_repo,
        weight_name=args.lightning_weight,
        adapter_name="lightning",
    )
    pipe.set_adapters(["lightning"], adapter_weights=[args.lora_scale])
    loading.update()
    loading.set_description(f"Moving localized pipeline to {args.device}")
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    loading.update()
    loading.close()
    return pipe


def odd_size(radius: int) -> int:
    return max(1, radius * 2 + 1)


def binary(mask: Image.Image, threshold: int) -> Image.Image:
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(array >= threshold, 255, 0).astype(np.uint8))


def build_ownership_masks(alpha: Image.Image, args) -> dict[str, Image.Image]:
    """Create non-rectangular core, boundary, halo and native edit masks."""

    silhouette = binary(alpha, args.alpha_threshold)
    core = silhouette.filter(ImageFilter.MinFilter(odd_size(args.core_erode_px)))
    dilated = silhouette.filter(ImageFilter.MaxFilter(odd_size(args.boundary_dilate_px)))

    shadow = Image.new("L", silhouette.size)
    shadow.paste(silhouette, (0, args.shadow_offset_px))
    shadow = shadow.filter(ImageFilter.MaxFilter(odd_size(args.shadow_dilate_px)))
    interaction = ImageChops.lighter(dilated, shadow)

    # The model edits the annulus and contact region, never the identity core.
    edit = ImageChops.subtract(interaction, core)
    return {
        "silhouette": silhouette,
        "core": core,
        "interaction": interaction,
        "edit": edit,
    }


@torch.inference_mode()
def qwen_local_inpaint(pipe, collage: Image.Image, edit_mask: Image.Image, name: str, args) -> Image.Image:
    prompt = f"Naturally integrate the pasted {name}. Preserve its identity and the unmasked scene."
    cfg_enabled = args.true_cfg_scale > 1.0
    result = pipe(
        image=collage,
        mask_image=edit_mask,
        prompt=prompt,
        negative_prompt=args.negative_prompt if cfg_enabled else None,
        true_cfg_scale=args.true_cfg_scale,
        strength=args.inpaint_strength,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        padding_mask_crop=None,
        generator=make_generator(args.device, args.seed),
    )
    return result.images[0].convert("RGB")


def inside_feather(mask: Image.Image, radius: float) -> Image.Image:
    """Feather inward while keeping every pixel outside the mask exactly zero."""

    hard = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    if radius <= 0:
        return Image.fromarray(np.rint(hard * 255).astype(np.uint8))
    blurred = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    weight = blurred * (hard > 0)
    return Image.fromarray(np.rint(np.clip(weight, 0, 1) * 255).astype(np.uint8))


def compose_owned_output(
    before: Image.Image,
    qwen: Image.Image,
    object_canvas: Image.Image,
    paste_alpha: Image.Image,
    masks: dict[str, Image.Image],
    args,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Enforce background and reference ownership in RGB space."""

    outer_weight = inside_feather(masks["interaction"], args.outer_feather_px)
    localized = Image.composite(qwen, before, outer_weight)

    alpha_array = np.asarray(paste_alpha, dtype=np.float32) / 255.0
    core_blur = np.asarray(
        masks["core"].filter(ImageFilter.GaussianBlur(args.core_feather_px)),
        dtype=np.float32,
    ) / 255.0
    core_weight_array = np.clip(core_blur * alpha_array, 0.0, 1.0)
    core_weight = Image.fromarray(np.rint(core_weight_array * 255).astype(np.uint8))
    final = Image.composite(object_canvas, localized, core_weight)
    return final, outer_weight, core_weight


def preservation_metrics(
    before: Image.Image,
    final: Image.Image,
    object_canvas: Image.Image,
    interaction: Image.Image,
    core: Image.Image,
) -> dict:
    before_array = np.asarray(before, dtype=np.float32)
    final_array = np.asarray(final, dtype=np.float32)
    object_array = np.asarray(object_canvas, dtype=np.float32)
    interaction_array = np.asarray(interaction, dtype=np.uint8) > 0
    core_array = np.asarray(core, dtype=np.uint8) > 0
    background = ~interaction_array

    bg_error = np.abs(final_array - before_array)[background]
    core_error = np.abs(final_array - object_array)[core_array]
    bg_mae = float(bg_error.mean()) if bg_error.size else 0.0
    core_mae = float(core_error.mean()) if core_error.size else 0.0
    return {
        "background_mae_outside_interaction": bg_mae,
        "background_max_error_outside_interaction": float(bg_error.max()) if bg_error.size else 0.0,
        "background_changed_fraction": float(
            (np.max(np.abs(final_array - before_array), axis=2)[background] > 0).mean()
        ) if background.any() else 0.0,
        "reference_core_mae": core_mae,
        "interaction_fraction": float(interaction_array.mean()),
        "core_fraction": float(core_array.mean()),
    }


def diagnostic_panel(
    before: Image.Image,
    collage: Image.Image,
    qwen: Image.Image,
    final: Image.Image,
    edit_mask: Image.Image,
    path: Path,
) -> None:
    images = [before, collage, qwen, final, edit_mask.convert("RGB")]
    names = ["Before", "Collage", "Raw Qwen", "Owned final", "Qwen edit mask"]
    tile = 256
    panel = Image.new("RGB", (tile * len(images), tile + 28), "white")
    draw = ImageDraw.Draw(panel)
    for index, (image, name) in enumerate(zip(images, names)):
        panel.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (index * tile, 28))
        draw.text((index * tile + 5, 7), name, fill="black")
    panel.save(path)


def run_case(pipe, case: dict, args) -> dict:
    case_id = int(case["id"])
    base_path = args.base_dir / f"base_{case_id:03d}.png"
    placement_path = args.mask_dir / f"base_{case_id:03d}.png"
    if not base_path.is_file() or not placement_path.is_file():
        raise FileNotFoundError(f"Missing base/mask pair for case {case_id}")

    current = Image.open(base_path).convert("RGB")
    if current.size != (args.width, args.height):
        raise ValueError(f"{base_path} is {current.size}; expected {(args.width, args.height)}")
    rectangles = extract_rectangles(placement_path)
    objects = case["objects"]
    if len(objects) != len(rectangles):
        raise ValueError(f"Case {case_id}: {len(objects)} objects but {len(rectangles)} rectangles")

    case_dir = args.out_dir / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current.save(case_dir / "base.png")
    Image.open(placement_path).save(case_dir / "placement_labels.png")
    history = []

    for step, (item, rectangle) in enumerate(zip(objects, rectangles), start=1):
        name = item["name"]
        prefix = steps_dir / f"{step:02d}_{slug(name)}"
        final_path = Path(f"{prefix}_final.png")
        if args.resume and final_path.is_file():
            current = Image.open(final_path).convert("RGB")
            history.append({"step": step, "name": name, "status": "resumed", "final": str(final_path)})
            continue

        before = current
        reference_file = reference_path(item, args.reference_dir)
        reference = Image.open(reference_file).convert("RGB")
        reference_alpha = background_alpha(reference, args.alpha_low, args.alpha_high)
        cutout_rgb, cutout_alpha = crop_cutout(reference, reference_alpha)
        object_canvas, paste_alpha, placed_box = place_in_rectangle(
            cutout_rgb,
            cutout_alpha,
            rectangle,
            before.size,
            args.object_scale,
        )
        collage = make_collage(before, object_canvas, paste_alpha)
        masks = build_ownership_masks(paste_alpha, args)

        raw_qwen = qwen_local_inpaint(pipe, collage, masks["edit"], name, args)
        final, outer_weight, core_weight = compose_owned_output(
            before, raw_qwen, object_canvas, paste_alpha, masks, args
        )
        metrics = preservation_metrics(
            before, final, object_canvas, masks["interaction"], masks["core"]
        )
        if metrics["background_max_error_outside_interaction"] != 0:
            raise AssertionError(f"Background ownership failed at case {case_id}, step {step}: {metrics}")

        before.save(Path(f"{prefix}_before.png"))
        collage.save(Path(f"{prefix}_collage.png"))
        paste_alpha.save(Path(f"{prefix}_paste_alpha.png"))
        for mask_name, mask in masks.items():
            mask.save(Path(f"{prefix}_{mask_name}_mask.png"))
        raw_qwen.save(Path(f"{prefix}_raw_qwen.png"))
        outer_weight.save(Path(f"{prefix}_outer_blend.png"))
        core_weight.save(Path(f"{prefix}_core_blend.png"))
        final.save(final_path)
        diagnostic_panel(
            before, collage, raw_qwen, final, masks["edit"], Path(f"{prefix}_panel.png")
        )

        history.append(
            {
                "step": step,
                "name": name,
                "status": "generated",
                "seed": args.seed,
                "reference": str(reference_file),
                "rectangle": list(rectangle.box),
                "placed_box": list(placed_box),
                "prompt": f"Naturally integrate the pasted {name}. Preserve its identity and the unmasked scene.",
                "metrics": metrics,
                "final": str(final_path),
            }
        )
        current = final
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "steps": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--mask_dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--out_dir", type=Path, default=HERE / "localized_identity_outputs")
    parser.add_argument("--case_ids", type=int, nargs="+")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--lightning_repo", default="lightx2v/Qwen-Image-Lightning")
    parser.add_argument(
        "--lightning_weight",
        default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--true_cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--inpaint_strength", type=float, default=0.80)
    parser.add_argument("--object_scale", type=float, default=0.92)
    parser.add_argument("--alpha_low", type=float, default=10.0)
    parser.add_argument("--alpha_high", type=float, default=45.0)
    parser.add_argument("--alpha_threshold", type=int, default=24)
    parser.add_argument("--core_erode_px", type=int, default=8)
    parser.add_argument("--boundary_dilate_px", type=int, default=14)
    parser.add_argument("--shadow_offset_px", type=int, default=14)
    parser.add_argument("--shadow_dilate_px", type=int, default=10)
    parser.add_argument("--outer_feather_px", type=float, default=5.0)
    parser.add_argument("--core_feather_px", type=float, default=3.0)
    return parser.parse_args()


def validate_args(args) -> None:
    if not 0 < args.inpaint_strength <= 1:
        raise ValueError("--inpaint_strength must be in (0, 1]")
    if not 0 < args.object_scale <= 1:
        raise ValueError("--object_scale must be in (0, 1]")
    for name in ("core_erode_px", "boundary_dilate_px", "shadow_offset_px", "shadow_dilate_px"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} cannot be negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    for key in ("prompts", "base_dir", "mask_dir", "reference_dir", "out_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.prompts, args.case_ids)
    save_json(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        args.out_dir / "config.json",
    )

    pipe = load_inpaint_pipeline(args)
    summary = []
    for case in tqdm(cases, desc="Localized identity-locked editing", unit="case"):
        summary.append(run_case(pipe, case, args))
        save_json(summary, args.out_dir / "summary.json")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Completed {len(summary)} case(s): {args.out_dir}")


if __name__ == "__main__":
    main()
