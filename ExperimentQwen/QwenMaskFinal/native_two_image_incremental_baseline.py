"""Native two-image Qwen baseline for incremental reference-object insertion.

At turn i, Qwen receives exactly two images:

    [I_(i-1), O_i] -> Qwen -> I_i

There is no collage, cutout, placement-mask conditioning, inpainting mask,
feature injection, or post-generation restoration. Placement rectangles are
loaded only after inference to evaluate localization and background drift.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm

from final_local_collage_harmonization import MetricEvaluator, write_metric_tables
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
    reference_path,
    slug,
)
from placement_masks import extract_rectangles


HERE = Path(__file__).resolve().parent


def load_pipeline(args):
    from diffusers import QwenImageEditPlusPipeline

    loading = tqdm(total=3, desc="Loading native Qwen baseline", unit="stage", dynamic_ncols=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
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
    loading.set_description(f"Moving Qwen baseline to {args.device}")
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    loading.update()
    loading.close()
    return pipe


def fit_reference(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.pad(
        image.convert("RGB"), size, Image.Resampling.LANCZOS, color="white"
    )


@torch.inference_mode()
def native_edit(pipe, current: Image.Image, reference: Image.Image, name: str, args):
    prompt = (
        f"Image 1 is the current scene. Image 2 is a reference image of a {name}. "
        f"Add the Image 2 {name} naturally to Image 1. Preserve the existing scene and objects."
    )
    cfg_enabled = args.true_cfg_scale > 1.0
    result = pipe(
        image=[current, reference],
        prompt=prompt,
        negative_prompt=args.negative_prompt if cfg_enabled else None,
        true_cfg_scale=args.true_cfg_scale,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        generator=make_generator(args.device, args.seed),
    )
    return result.images[0].convert("RGB"), prompt


def rectangle_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).rectangle(
        (box[0], box[1], box[2] - 1, box[3] - 1), fill=255
    )
    return mask


def change_metrics(
    before: Image.Image,
    after: Image.Image,
    target: Image.Image,
    threshold: int,
) -> dict[str, float]:
    before_array = np.asarray(before.convert("RGB"), dtype=np.int16)
    after_array = np.asarray(after.convert("RGB"), dtype=np.int16)
    difference = np.abs(after_array - before_array)
    scalar = difference.max(axis=2)
    changed = scalar > threshold
    inside = np.asarray(target, dtype=np.uint8) > 0
    outside = ~inside
    outside_difference = difference[outside]
    outside_mse = (
        float(np.square(outside_difference.astype(np.float64)).mean())
        if outside_difference.size
        else 0.0
    )
    changed_count = int(changed.sum())
    return {
        "changed_fraction_of_image": float(changed.mean()),
        "changed_pixels_inside_placement_rectangle": float(
            (changed & inside).sum() / max(changed_count, 1)
        ),
        "outside_placement_mae": float(outside_difference.mean())
        if outside_difference.size
        else 0.0,
        "outside_placement_psnr": float("inf")
        if outside_mse == 0
        else float(20 * math.log10(255.0 / math.sqrt(outside_mse))),
        "outside_placement_max_error": float(outside_difference.max())
        if outside_difference.size
        else 0.0,
        "outside_placement_changed_fraction": float(changed[outside].mean())
        if outside.any()
        else 0.0,
    }


def target_region_fidelity(
    evaluator: MetricEvaluator,
    reference_rgb: Image.Image,
    reference_alpha: Image.Image,
    output: Image.Image,
    target_mask: Image.Image,
) -> dict[str, float | None]:
    """Compare the reference with the requested output region.

    The baseline has no generated-object mask. The rectangle is deliberately
    used as the evaluation support, so incorrect or missing placement is not
    hidden by an output-dependent detector.
    """
    reference_view = evaluator.masked_view(reference_rgb, reference_alpha)
    output_view = evaluator.masked_view(output, target_mask)
    return {
        "target_region_dino_similarity": evaluator.dino_similarity(
            reference_view, output_view
        ),
        "target_region_color_similarity": evaluator.color_histogram_similarity(
            reference_rgb, reference_alpha, output, target_mask
        ),
        "target_region_edge_similarity": evaluator.edge_similarity(
            reference_view, output_view
        ),
    }


def prior_region_stability(
    snapshot: Image.Image,
    current: Image.Image,
    region: Image.Image,
    threshold: int,
) -> dict[str, float]:
    support = np.asarray(region, dtype=np.uint8) > 0
    difference = np.abs(
        np.asarray(current, dtype=np.int16) - np.asarray(snapshot, dtype=np.int16)
    )[support]
    mse = float(np.square(difference.astype(np.float64)).mean())
    return {
        "cross_turn_mae": float(difference.mean()),
        "cross_turn_psnr": float("inf")
        if mse == 0
        else float(20 * math.log10(255.0 / math.sqrt(mse))),
        "cross_turn_changed_fraction": float(
            (difference.max(axis=1) > threshold).mean()
        ),
    }


def diagnostic_panel(
    before: Image.Image,
    reference: Image.Image,
    after: Image.Image,
    target_box: tuple[int, int, int, int],
    path: Path,
) -> None:
    marked = after.copy()
    ImageDraw.Draw(marked).rectangle(
        (target_box[0], target_box[1], target_box[2] - 1, target_box[3] - 1),
        outline="red",
        width=5,
    )
    views = [
        ("Before", before),
        ("Reference", reference),
        ("Native Qwen output", after),
        ("Requested region (evaluation only)", marked),
    ]
    tile = 256
    panel = Image.new("RGB", (tile * len(views), tile + 28), "white")
    draw = ImageDraw.Draw(panel)
    for index, (label, image) in enumerate(views):
        panel.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (index * tile, 28))
        draw.text((index * tile + 5, 7), label, fill="black")
    panel.save(path)


def run_case(pipe, evaluator: MetricEvaluator, case: dict, args) -> dict:
    case_id = int(case["id"])
    base_path = args.base_dir / f"base_{case_id:03d}.png"
    mask_path = args.mask_dir / f"base_{case_id:03d}.png"
    if not base_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Missing base/mask pair for case {case_id}")

    current = Image.open(base_path).convert("RGB")
    rectangles = extract_rectangles(mask_path)
    objects = case["objects"]
    if len(rectangles) != len(objects):
        raise ValueError(
            f"Case {case_id}: {len(objects)} objects but {len(rectangles)} rectangles"
        )

    case_dir = args.out_dir / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current.save(case_dir / "base.png")
    Image.open(mask_path).save(case_dir / "placement_labels_evaluation_only.png")
    history: list[dict] = []
    metric_rows: list[dict] = []
    tracked: list[dict] = []

    for step, (item, rectangle) in enumerate(zip(objects, rectangles), start=1):
        name = item["name"]
        prefix = steps_dir / f"{step:02d}_{slug(name)}"
        final_path = Path(f"{prefix}_final.png")
        metrics_path = Path(f"{prefix}_metrics.json")
        if args.resume and final_path.is_file() and metrics_path.is_file():
            current = Image.open(final_path).convert("RGB")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric_rows.append(
                {key: value for key, value in metrics.items() if key != "prior_object_stability"}
            )
            tracked.append(
                {
                    "step": step,
                    "name": name,
                    "snapshot": current.copy(),
                    "region": rectangle_mask(current.size, rectangle.box),
                }
            )
            history.append({"step": step, "name": name, "status": "resumed"})
            continue

        before = current.copy()
        source_path = reference_path(item, args.reference_dir)
        source_reference = Image.open(source_path).convert("RGB")
        reference_input = fit_reference(source_reference, (args.width, args.height))
        reference_alpha = background_alpha(
            source_reference, args.alpha_low, args.alpha_high
        )
        reference_rgb, reference_alpha = crop_cutout(
            source_reference, reference_alpha
        )

        after, prompt = native_edit(pipe, before, reference_input, name, args)
        target = rectangle_mask(after.size, rectangle.box)
        metrics = {
            "case_id": case_id,
            "step": step,
            "object": name,
            **change_metrics(before, after, target, args.change_threshold),
            **target_region_fidelity(
                evaluator, reference_rgb, reference_alpha, after, target
            ),
        }

        prior_records = []
        for record in tracked:
            stability = prior_region_stability(
                record["snapshot"], after, record["region"], args.change_threshold
            )
            stability.update(
                {
                    "source_step": record["step"],
                    "source_object": record["name"],
                    "evaluated_at_step": step,
                }
            )
            prior_records.append(stability)
        metrics["prior_object_stability"] = prior_records
        metrics["prior_objects_mean_mae"] = (
            float(np.mean([record["cross_turn_mae"] for record in prior_records]))
            if prior_records
            else None
        )
        metrics["prior_objects_mean_changed_fraction"] = (
            float(
                np.mean(
                    [record["cross_turn_changed_fraction"] for record in prior_records]
                )
            )
            if prior_records
            else None
        )

        before.save(Path(f"{prefix}_before.png"))
        reference_input.save(Path(f"{prefix}_reference_input.png"))
        target.save(Path(f"{prefix}_target_region_evaluation_only.png"))
        after.save(final_path)
        diagnostic_panel(before, reference_input, after, rectangle.box, Path(f"{prefix}_panel.png"))
        save_json(metrics, metrics_path)

        history.append(
            {
                "step": step,
                "name": name,
                "status": "generated",
                "seed": args.seed,
                "model_inputs": ["current_scene", "reference_object"],
                "placement_used_for_generation": False,
                "evaluation_rectangle": list(rectangle.box),
                "prompt": prompt,
                "reference": str(source_path),
                "final": str(final_path),
                "metrics": metrics,
            }
        )
        metric_rows.append(
            {key: value for key, value in metrics.items() if key != "prior_object_stability"}
        )
        tracked.append(
            {
                "step": step,
                "name": name,
                "snapshot": after.copy(),
                "region": target,
            }
        )
        current = after
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {
        "id": case_id,
        "steps": len(history),
        "final": str(case_dir / "FINAL.png"),
        "metrics": metric_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--mask_dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--out_dir", type=Path, default=HERE / "results" / "NativeTwoImageBaseline_01")
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
    parser.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric_model_id", default="facebook/dinov2-base")
    parser.add_argument("--metric_device", default="cpu")
    parser.add_argument("--change_threshold", type=int, default=8)
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
    evaluator = MetricEvaluator(args.metric_model_id, args.metric_device, args.metrics)
    summary = []
    try:
        for case in tqdm(cases, desc="Native two-image incremental baseline", unit="case"):
            summary.append(run_case(pipe, evaluator, case, args))
            save_json(summary, args.out_dir / "summary.json")
            write_metric_tables(summary, args.out_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        evaluator.close()
    print(f"Completed {len(summary)} baseline case(s): {args.out_dir}")


if __name__ == "__main__":
    main()
