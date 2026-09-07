"""Sequential rectangle-guided collage insertion followed by one Qwen pass.

For each case:
    current image + reference cutout placed in rectangle -> collage
    collage -> Qwen ("Naturally integrate ...") -> next current image

There is no placement prediction, feature injection, SAM pass, latent blending,
or post-generation compositing in this controlled baseline.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from generate_base_images import load_pipeline, make_generator, save_json
from placement_masks import PlacementRectangle, extract_rectangles


HERE = Path(__file__).resolve().parent
EXPERIMENT_QWEN = HERE.parent
DEFAULT_PROMPTS = HERE / "e5_prompts.json"
DEFAULT_BASES = HERE / "BaseImages"
DEFAULT_MASKS = HERE / "PlacementMasks"
DEFAULT_REFERENCES = EXPERIMENT_QWEN / "references"
DEFAULT_OUTPUT = HERE / "multistep_outputs"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_cases(path: Path, requested: list[int] | None) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle).get("prompts")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty 'prompts' list")
    if not requested:
        return cases
    wanted = set(requested)
    selected = [case for case in cases if int(case["id"]) in wanted]
    missing = wanted - {int(case["id"]) for case in selected}
    if missing:
        raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    return selected


def reference_path(item: dict, directory: Path) -> Path:
    canny_stem = Path(item.get("canny_file", "")).stem
    object_slug = slug(item["name"])
    source_slug = slug(canny_stem)
    exact = directory / f"{object_slug}__{source_slug}.png"
    if exact.is_file():
        return exact

    # The descriptive name can change between prompt suites (for example,
    # "house plant" versus "potted plant"), while the Canny filename remains
    # the stable object identifier.
    by_source = sorted(directory.glob(f"*__{source_slug}.*")) if source_slug else []
    if len(by_source) == 1:
        return by_source[0]

    by_name = sorted(directory.glob(f"{object_slug}__*"))
    if len(by_name) == 1:
        return by_name[0]

    # Final conservative fallback for repositories containing simple filenames.
    simple = [
        directory / f"{source_slug}.png",
        directory / f"{object_slug}.png",
    ]
    existing = [path for path in simple if path.is_file()]
    if len(existing) == 1:
        return existing[0]

    matches = sorted({*by_source, *by_name, *existing})
    raise FileNotFoundError(
        f"No unique reference image found for {item['name']!r} "
        f"(Canny stem {canny_stem!r}) in {directory}. Matches: "
        f"{[path.name for path in matches]}"
    )


def background_alpha(image: Image.Image, low: float, high: float) -> Image.Image:
    """Estimate a soft foreground alpha from a plain reference background."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
    alpha = np.clip((distance - low) / max(high - low, 1e-6), 0.0, 1.0)

    # Keep a small amount of soft contact-shadow evidence but suppress tiny
    # compression/color fluctuations in the nominally plain background.
    alpha[alpha < 0.03] = 0.0
    return Image.fromarray(np.rint(alpha * 255.0).astype(np.uint8))


def crop_cutout(reference: Image.Image, alpha: Image.Image) -> tuple[Image.Image, Image.Image]:
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("Reference-background extraction produced an empty cutout")
    return reference.crop(bbox), alpha.crop(bbox)


def place_in_rectangle(
    reference: Image.Image,
    alpha: Image.Image,
    rectangle: PlacementRectangle,
    canvas_size: tuple[int, int],
    scale: float,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Contain the complete cutout in the manually supplied rectangle."""

    x0, y0, x1, y1 = rectangle.box
    available_w = max(1, int((x1 - x0) * scale))
    available_h = max(1, int((y1 - y0) * scale))
    ratio = min(available_w / reference.width, available_h / reference.height)
    target_w = max(1, round(reference.width * ratio))
    target_h = max(1, round(reference.height * ratio))
    resized_rgb = reference.resize((target_w, target_h), Image.Resampling.LANCZOS)
    resized_alpha = alpha.resize((target_w, target_h), Image.Resampling.LANCZOS)

    left = x0 + ((x1 - x0) - target_w) // 2
    # Bottom alignment establishes support contact for both floor and surface
    # objects while the rectangle controls the intended depth and scale.
    top = y1 - target_h
    object_canvas = Image.new("RGB", canvas_size)
    alpha_canvas = Image.new("L", canvas_size)
    object_canvas.paste(resized_rgb, (left, top))
    alpha_canvas.paste(resized_alpha, (left, top))
    return object_canvas, alpha_canvas, (left, top, left + target_w, top + target_h)


def make_collage(scene: Image.Image, object_canvas: Image.Image, alpha: Image.Image) -> Image.Image:
    return Image.composite(object_canvas, scene, alpha)


def qwen_integrate(pipe, collage: Image.Image, object_name: str, args) -> Image.Image:
    prompt = f"Naturally integrate the pasted {object_name}. Keep everything else unchanged."
    cfg_enabled = args.true_cfg_scale > 1.0
    result = pipe(
        image=[collage],
        prompt=prompt,
        negative_prompt=args.negative_prompt if cfg_enabled else None,
        true_cfg_scale=args.true_cfg_scale,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        generator=make_generator(args.device, args.seed),
    )
    return result.images[0].convert("RGB")


def run_case(pipe, case: dict, args) -> dict:
    case_id = int(case["id"])
    base_path = args.base_dir / f"base_{case_id:03d}.png"
    mask_path = args.mask_dir / f"base_{case_id:03d}.png"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)

    current = Image.open(base_path).convert("RGB")
    if current.size != (args.width, args.height):
        raise ValueError(f"{base_path} is {current.size}; expected {(args.width, args.height)}")
    rectangles = extract_rectangles(mask_path)
    objects = case.get("objects", [])
    if len(objects) != len(rectangles):
        raise ValueError(
            f"Case {case_id} has {len(objects)} objects but {len(rectangles)} placement rectangles"
        )

    case_dir = args.out_dir / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    cutout_dir = case_dir / "cutouts"
    steps_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)
    current.save(case_dir / "base.png")
    Image.open(mask_path).save(case_dir / "placement_labels.png")

    history = []
    for step, (item, rectangle) in enumerate(zip(objects, rectangles), start=1):
        name = item["name"]
        stem = f"{step:02d}_{slug(name)}"
        after_path = steps_dir / f"{stem}_after.png"
        if args.resume and after_path.is_file():
            current = Image.open(after_path).convert("RGB")
            history.append({"step": step, "name": name, "status": "resumed", "after": str(after_path)})
            continue

        source_path = reference_path(item, args.reference_dir)
        reference = Image.open(source_path).convert("RGB")
        alpha = background_alpha(reference, args.alpha_low, args.alpha_high)
        cutout_rgb, cutout_alpha = crop_cutout(reference, alpha)
        cutout_rgb.save(cutout_dir / f"{stem}_rgb.png")
        cutout_alpha.save(cutout_dir / f"{stem}_alpha.png")

        before_path = steps_dir / f"{stem}_before.png"
        collage_path = steps_dir / f"{stem}_collage.png"
        current.save(before_path)
        object_canvas, paste_alpha, placed_box = place_in_rectangle(
            cutout_rgb, cutout_alpha, rectangle, current.size, args.object_scale
        )
        collage = make_collage(current, object_canvas, paste_alpha)
        collage.save(collage_path)
        paste_alpha.save(steps_dir / f"{stem}_paste_alpha.png")

        current = qwen_integrate(pipe, collage, name, args)
        current.save(after_path)
        history.append(
            {
                "step": step,
                "name": name,
                "status": "generated",
                "seed": args.seed,
                "prompt": f"Naturally integrate the pasted {name}. Keep everything else unchanged.",
                "reference": str(source_path),
                "rectangle_label": rectangle.label,
                "rectangle": list(rectangle.box),
                "placed_box": list(placed_box),
                "before": str(before_path),
                "collage": str(collage_path),
                "after": str(after_path),
            }
        )
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "final": str(case_dir / "FINAL.png"), "steps": len(history)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--mask_dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case_ids", type=int, nargs="+", help="Run only selected case IDs")
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
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed reused for every edit pass")
    parser.add_argument("--true_cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--object_scale", type=float, default=0.92)
    parser.add_argument("--alpha_low", type=float, default=10.0)
    parser.add_argument("--alpha_high", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key in ("prompts", "base_dir", "mask_dir", "reference_dir", "out_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.prompts, args.case_ids)
    save_json(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        args.out_dir / "config.json",
    )

    pipe = load_pipeline(args)
    summary = []
    for case in tqdm(cases, desc="Mask-guided multistep editing", unit="case"):
        summary.append(run_case(pipe, case, args))
        save_json(summary, args.out_dir / "summary.json")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Completed {len(summary)} case(s): {args.out_dir}")


if __name__ == "__main__":
    main()
