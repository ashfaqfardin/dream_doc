"""Final training-free baseline: local collage harmonization with hard preservation.

At turn i, the reference cutout is placed deterministically inside the user's
rectangle. Qwen edits one square crop containing the complete object and local
scene context. Only the alpha-derived interaction region is copied back:

    I_i = (1 - M_i) * I_{i-1} + M_i * Qwen(local collage crop)

Consequently, every pixel outside M_i remains exactly equal to I_{i-1}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
EXPERIMENT_QWEN = HERE.parent


class MetricEvaluator:
    """Localized reference-fidelity metrics with one shared DINOv2 encoder."""

    def __init__(self, model_id: str, device: str, enabled: bool):
        self.enabled = enabled
        self.device = torch.device(device)
        self.processor = None
        self.model = None
        if enabled:
            from transformers import AutoImageProcessor, AutoModel

            loading = tqdm(total=2, desc="Loading DINOv2 metrics", unit="stage")
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            loading.update()
            self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
            loading.update()
            loading.close()

    @staticmethod
    def masked_view(image: Image.Image, alpha: Image.Image, size: int = 224) -> Image.Image:
        alpha = alpha.convert("L")
        bbox = alpha.getbbox()
        if bbox is None:
            raise ValueError("Cannot evaluate an empty object mask")
        image_crop = image.convert("RGB").crop(bbox)
        alpha_crop = alpha.crop(bbox)
        neutral = Image.new("RGB", image_crop.size, (127, 127, 127))
        neutral.paste(image_crop, (0, 0), alpha_crop)
        side = max(neutral.size)
        square = Image.new("RGB", (side, side), (127, 127, 127))
        square.paste(neutral, ((side - neutral.width) // 2, (side - neutral.height) // 2))
        return square.resize((size, size), Image.Resampling.LANCZOS)

    @torch.inference_mode()
    def dino_similarity(self, reference: Image.Image, generated: Image.Image) -> float | None:
        if not self.enabled:
            return None
        inputs = self.processor(images=[reference, generated], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs)
        features = getattr(output, "pooler_output", None)
        if features is None:
            features = output.last_hidden_state[:, 0]
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        return float((features[0] * features[1]).sum().cpu())

    @staticmethod
    def color_histogram_similarity(
        reference: Image.Image,
        reference_alpha: Image.Image,
        generated: Image.Image,
        generated_alpha: Image.Image,
        bins: int = 32,
    ) -> float:
        similarities = []
        for image, alpha in (
            (reference, reference_alpha),
            (generated, generated_alpha),
        ):
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            weights = np.asarray(alpha.convert("L"), dtype=np.float32).reshape(-1) / 255.0
            histograms = []
            for channel in range(3):
                histogram, _ = np.histogram(
                    rgb[..., channel].reshape(-1),
                    bins=bins,
                    range=(0, 256),
                    weights=weights,
                )
                histogram = histogram.astype(np.float64)
                histogram /= max(histogram.sum(), 1e-12)
                histograms.append(histogram)
            similarities.append(histograms)
        scores = [
            float(np.sqrt(similarities[0][channel] * similarities[1][channel]).sum())
            for channel in range(3)
        ]
        return float(np.mean(scores))

    @staticmethod
    def edge_similarity(reference_view: Image.Image, generated_view: Image.Image) -> float:
        arrays = []
        for image in (reference_view, generated_view):
            gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            dy, dx = np.gradient(gray)
            edge = np.sqrt(dx * dx + dy * dy).reshape(-1)
            arrays.append(edge)
        denominator = float(np.linalg.norm(arrays[0]) * np.linalg.norm(arrays[1]))
        return float(np.dot(arrays[0], arrays[1]) / max(denominator, 1e-12))

    def object_fidelity(
        self,
        reference_rgb: Image.Image,
        reference_alpha: Image.Image,
        final: Image.Image,
        placed_alpha: Image.Image,
    ) -> dict[str, float | None]:
        reference_view = self.masked_view(reference_rgb, reference_alpha)
        final_view = self.masked_view(final, placed_alpha)
        return {
            "dino_identity_similarity": self.dino_similarity(reference_view, final_view),
            "color_histogram_similarity": self.color_histogram_similarity(
                reference_rgb, reference_alpha, final, placed_alpha
            ),
            "edge_structure_similarity": self.edge_similarity(reference_view, final_view),
        }

    def close(self) -> None:
        if self.model is not None:
            self.model.to("cpu")
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_inpaint_pipeline(args):
    from diffusers import QwenImageEditInpaintPipeline

    loading = tqdm(total=3, desc="Loading final Qwen inpaint", unit="stage", dynamic_ncols=True)
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
    loading.set_description(f"Moving Qwen to {args.device}")
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    loading.update()
    loading.close()
    return pipe


def odd_kernel(radius: int) -> int:
    return max(1, radius * 2 + 1)


def hard_mask(alpha: Image.Image, threshold: int) -> Image.Image:
    array = np.asarray(alpha.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(array >= threshold, 255, 0).astype(np.uint8))


def interaction_mask(alpha: Image.Image, args) -> Image.Image:
    """Object silhouette, boundary allowance, and downward contact-shadow area."""

    silhouette = hard_mask(alpha, args.alpha_threshold)
    expanded = silhouette.filter(
        ImageFilter.MaxFilter(odd_kernel(args.interaction_dilate_px))
    )
    shadow = Image.new("L", silhouette.size)
    shadow.paste(silhouette, (args.shadow_offset_x, args.shadow_offset_y))
    shadow = shadow.filter(ImageFilter.MaxFilter(odd_kernel(args.shadow_dilate_px)))
    return ImageChops.lighter(expanded, shadow)


def inward_feather(mask: Image.Image, radius: float) -> Image.Image:
    """Create a soft interior blend while remaining exactly zero outside mask."""

    hard = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    if radius <= 0:
        return Image.fromarray(np.rint(hard * 255).astype(np.uint8))
    soft = np.asarray(
        mask.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32
    ) / 255.0
    return Image.fromarray(np.rint(np.clip(soft * (hard > 0), 0, 1) * 255).astype(np.uint8))


def square_context_box(
    mask: Image.Image,
    image_size: tuple[int, int],
    context_fraction: float,
    minimum_side: int,
) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Interaction mask is empty")
    width, height = image_size
    x0, y0, x1, y1 = bbox
    object_w, object_h = x1 - x0, y1 - y0
    side = max(
        minimum_side,
        round(max(object_w, object_h) * (1.0 + 2.0 * context_fraction)),
    )
    side = min(side, width, height)
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    left = round(center_x - side / 2)
    top = round(center_y - side / 2)
    left = min(max(0, left), width - side)
    top = min(max(0, top), height - side)
    return left, top, left + side, top + side


@torch.inference_mode()
def harmonize_local_crop(
    pipe,
    collage: Image.Image,
    edit_mask: Image.Image,
    crop_box: tuple[int, int, int, int],
    object_name: str,
    args,
) -> tuple[Image.Image, str]:
    local_image = collage.crop(crop_box)
    local_mask = edit_mask.crop(crop_box)
    source_size = local_image.size
    process_size = (args.process_size, args.process_size)
    local_image = local_image.resize(process_size, Image.Resampling.LANCZOS)
    # Keep the native inpainting ownership mask hard after resampling.
    local_mask = local_mask.resize(process_size, Image.Resampling.NEAREST)
    prompt = (
        f"Integrate the pasted {object_name} naturally with the surrounding scene. "
        "Preserve its design, colour, material, position and overall structure."
    )
    cfg_enabled = args.true_cfg_scale > 1.0
    result = pipe(
        image=local_image,
        mask_image=local_mask,
        prompt=prompt,
        negative_prompt=args.negative_prompt if cfg_enabled else None,
        true_cfg_scale=args.true_cfg_scale,
        strength=args.inpaint_strength,
        num_inference_steps=args.steps,
        width=args.process_size,
        height=args.process_size,
        padding_mask_crop=None,
        generator=make_generator(args.device, args.seed),
    )
    generated = result.images[0].convert("RGB").resize(source_size, Image.Resampling.LANCZOS)
    return generated, prompt


def paste_local_result(
    before: Image.Image,
    generated_crop: Image.Image,
    interaction: Image.Image,
    crop_box: tuple[int, int, int, int],
    feather_px: float,
) -> tuple[Image.Image, Image.Image]:
    generated_full = before.copy()
    generated_full.paste(generated_crop, crop_box[:2])
    blend = inward_feather(interaction, feather_px)
    final = Image.composite(generated_full, before, blend)
    return final, blend


def preservation_metrics(
    before: Image.Image,
    final: Image.Image,
    interaction: Image.Image,
) -> dict[str, float]:
    before_array = np.asarray(before, dtype=np.int16)
    final_array = np.asarray(final, dtype=np.int16)
    allowed = np.asarray(interaction.convert("L"), dtype=np.uint8) > 0
    outside = ~allowed
    difference = np.abs(final_array - before_array)
    outside_difference = difference[outside]
    mse = float(np.square(outside_difference.astype(np.float64)).mean()) if outside_difference.size else 0.0
    return {
        "outside_mae": float(outside_difference.mean()) if outside_difference.size else 0.0,
        "outside_psnr": float("inf") if mse == 0 else float(20 * math.log10(255.0 / math.sqrt(mse))),
        "outside_max_error": float(outside_difference.max()) if outside_difference.size else 0.0,
        "outside_changed_fraction": float(
            (difference.max(axis=2)[outside] > 0).mean()
        ) if outside.any() else 0.0,
        "allowed_change_fraction": float(allowed.mean()),
    }


def spatial_metrics(
    before: Image.Image,
    final: Image.Image,
    placed_alpha: Image.Image,
    interaction: Image.Image,
    rectangle: tuple[int, int, int, int],
    change_threshold: int,
) -> dict[str, float]:
    width, height = before.size
    x0, y0, x1, y1 = rectangle
    rectangle_mask = np.zeros((height, width), dtype=bool)
    rectangle_mask[y0:y1, x0:x1] = True
    object_support = np.asarray(placed_alpha.convert("L"), dtype=np.uint8) > 0
    allowed = np.asarray(interaction.convert("L"), dtype=np.uint8) > 0
    difference = np.max(
        np.abs(
            np.asarray(final, dtype=np.int16)
            - np.asarray(before, dtype=np.int16)
        ),
        axis=2,
    )
    changed = difference > change_threshold
    changed_count = int(changed.sum())
    object_count = int(object_support.sum())
    return {
        "object_support_inside_rectangle": float(
            (object_support & rectangle_mask).sum() / max(object_count, 1)
        ),
        "changed_pixels_inside_interaction": float(
            (changed & allowed).sum() / max(changed_count, 1)
        ),
        "changed_pixels_inside_placement_rectangle": float(
            (changed & rectangle_mask).sum() / max(changed_count, 1)
        ),
        "interaction_inside_rectangle": float(
            (allowed & rectangle_mask).sum() / max(int(allowed.sum()), 1)
        ),
        "changed_fraction_of_image": float(changed.mean()),
    }


def cross_turn_stability(
    reference_turn: Image.Image,
    current: Image.Image,
    visible_mask: Image.Image,
    threshold: int,
) -> dict[str, float]:
    visible = np.asarray(visible_mask.convert("L"), dtype=np.uint8) > 0
    if not visible.any():
        return {
            "visible_fraction": 0.0,
            "cross_turn_mae": 0.0,
            "cross_turn_psnr": float("inf"),
            "cross_turn_changed_fraction": 0.0,
        }
    difference = np.abs(
        np.asarray(current, dtype=np.int16)
        - np.asarray(reference_turn, dtype=np.int16)
    )[visible]
    mse = float(np.square(difference.astype(np.float64)).mean())
    return {
        "visible_fraction": float(visible.mean()),
        "cross_turn_mae": float(difference.mean()),
        "cross_turn_psnr": float("inf") if mse == 0 else float(
            20 * math.log10(255.0 / math.sqrt(mse))
        ),
        "cross_turn_changed_fraction": float(
            (difference.max(axis=1) > threshold).mean()
        ),
    }


def diagnostic_panel(
    before: Image.Image,
    collage: Image.Image,
    local_output: Image.Image,
    final: Image.Image,
    interaction: Image.Image,
    crop_box: tuple[int, int, int, int],
    path: Path,
) -> None:
    crop_preview = before.copy()
    draw = ImageDraw.Draw(crop_preview)
    draw.rectangle(
        (crop_box[0], crop_box[1], crop_box[2] - 1, crop_box[3] - 1),
        outline="red",
        width=5,
    )
    views = [
        ("Before + local ROI", crop_preview),
        ("Collage", collage),
        ("Local Qwen output", local_output),
        ("Final", final),
        ("Allowed change", interaction.convert("RGB")),
    ]
    tile = 256
    panel = Image.new("RGB", (tile * len(views), tile + 28), "white")
    panel_draw = ImageDraw.Draw(panel)
    for index, (label, image) in enumerate(views):
        panel.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (index * tile, 28))
        panel_draw.text((index * tile + 5, 7), label, fill="black")
    panel.save(path)


def cutout_cache_key(item: dict) -> str:
    source = Path(item.get("canny_file", "")).stem
    return f"{slug(item['name'])}__{slug(source)}"


def prepare_cutouts(cases: list[dict], args) -> dict[str, tuple[Image.Image, Image.Image]]:
    """Resolve and cache clean reference cutouts before loading Qwen."""

    directory = args.out_dir / "cutout_cache"
    directory.mkdir(parents=True, exist_ok=True)
    unique: dict[str, dict] = {}
    for case in cases:
        for item in case["objects"]:
            unique.setdefault(cutout_cache_key(item), item)

    cutouts: dict[str, tuple[Image.Image, Image.Image]] = {}
    pending: list[tuple[str, dict, Path, Path]] = []
    for key, item in unique.items():
        rgb_path = directory / f"{key}_rgb.png"
        alpha_path = directory / f"{key}_alpha.png"
        if args.resume and rgb_path.is_file() and alpha_path.is_file():
            cutouts[key] = (
                Image.open(rgb_path).convert("RGB"),
                Image.open(alpha_path).convert("L"),
            )
        else:
            pending.append((key, item, rgb_path, alpha_path))

    segmenter = None
    if pending and args.cutout_backend == "rmbg2":
        if str(EXPERIMENT_QWEN) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_QWEN))
        from e5_spatial_kv_collage import RMBG2Cutout

        segmenter = RMBG2Cutout(
            args.rmbg_model_id,
            args.rmbg_revision,
            args.rmbg_device,
            args.rmbg_input_size,
        )
    try:
        for key, item, rgb_path, alpha_path in tqdm(
            pending, desc="Preparing reference cutouts", unit="object"
        ):
            source = reference_path(item, args.reference_dir)
            reference = Image.open(source).convert("RGB")
            if segmenter is not None:
                cutout = segmenter.cutout(item["name"], reference, args.rmbg_crop_threshold)
                rgb, alpha = cutout.rgb, cutout.alpha
            else:
                alpha = background_alpha(reference, args.alpha_low, args.alpha_high)
                rgb, alpha = crop_cutout(reference, alpha)
            rgb.save(rgb_path)
            alpha.save(alpha_path)
            cutouts[key] = (rgb, alpha)
    finally:
        if segmenter is not None:
            segmenter.close()
    return cutouts


def run_case(pipe, case: dict, cutouts: dict, evaluator: MetricEvaluator, args) -> dict:
    case_id = int(case["id"])
    base_path = args.base_dir / f"base_{case_id:03d}.png"
    mask_path = args.mask_dir / f"base_{case_id:03d}.png"
    if not base_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Missing base/mask pair for case {case_id}")
    current = Image.open(base_path).convert("RGB")
    if current.size != (args.width, args.height):
        raise ValueError(f"{base_path} is {current.size}; expected {(args.width, args.height)}")

    rectangles = extract_rectangles(mask_path)
    objects = case["objects"]
    if len(rectangles) != len(objects):
        raise ValueError(f"Case {case_id}: {len(objects)} objects but {len(rectangles)} rectangles")

    case_dir = args.out_dir / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current.save(case_dir / "base.png")
    Image.open(mask_path).save(case_dir / "placement_labels.png")
    history = []
    metric_rows = []
    tracked_objects: list[dict] = []

    for step, (item, rectangle) in enumerate(zip(objects, rectangles), start=1):
        name = item["name"]
        prefix = steps_dir / f"{step:02d}_{slug(name)}"
        final_path = Path(f"{prefix}_final.png")
        metrics_path = Path(f"{prefix}_metrics.json")
        alpha_path = Path(f"{prefix}_object_alpha.png")
        if args.resume and final_path.is_file() and metrics_path.is_file() and alpha_path.is_file():
            current = Image.open(final_path).convert("RGB")
            placed_alpha = Image.open(alpha_path).convert("L")
            new_support = hard_mask(placed_alpha, args.alpha_threshold)
            for tracked in tracked_objects:
                tracked["visible"] = ImageChops.subtract(tracked["visible"], new_support)
            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric_rows.append({key: value for key, value in saved_metrics.items() if not isinstance(value, list)})
            tracked_objects.append(
                {
                    "step": step,
                    "name": name,
                    "snapshot": current.copy(),
                    "visible": new_support,
                }
            )
            history.append({"step": step, "name": name, "status": "resumed", "final": str(final_path)})
            continue

        before = current
        cutout_rgb, cutout_alpha = cutouts[cutout_cache_key(item)]
        object_canvas, paste_alpha, placed_box = place_in_rectangle(
            cutout_rgb,
            cutout_alpha,
            rectangle,
            current.size,
            args.object_scale,
        )
        collage = make_collage(before, object_canvas, paste_alpha)
        allowed = interaction_mask(paste_alpha, args)
        crop_box = square_context_box(
            allowed, current.size, args.context_fraction, args.minimum_crop_side
        )
        generated_crop, prompt = harmonize_local_crop(
            pipe, collage, allowed, crop_box, name, args
        )
        final, blend = paste_local_result(
            before, generated_crop, allowed, crop_box, args.feather_px
        )
        preservation = preservation_metrics(before, final, allowed)
        spatial = spatial_metrics(
            before,
            final,
            paste_alpha,
            allowed,
            rectangle.box,
            args.change_threshold,
        )
        fidelity = evaluator.object_fidelity(
            cutout_rgb,
            cutout_alpha,
            final,
            paste_alpha,
        )
        if preservation["outside_max_error"] != 0:
            raise AssertionError(f"Non-disturbance constraint failed: {preservation}")

        new_object_support = hard_mask(paste_alpha, args.alpha_threshold)
        prior_stability = []
        for tracked in tracked_objects:
            visible = ImageChops.subtract(tracked["visible"], new_object_support)
            stability = cross_turn_stability(
                tracked["snapshot"],
                final,
                visible,
                args.change_threshold,
            )
            stability.update(
                {
                    "source_step": tracked["step"],
                    "source_object": tracked["name"],
                    "evaluated_at_step": step,
                }
            )
            prior_stability.append(stability)
            tracked["visible"] = visible

        metrics = {
            "case_id": case_id,
            "step": step,
            "object": name,
            **preservation,
            **spatial,
            **fidelity,
            "prior_object_stability": prior_stability,
        }
        if prior_stability:
            metrics["prior_objects_mean_mae"] = float(
                np.mean([record["cross_turn_mae"] for record in prior_stability])
            )
            metrics["prior_objects_mean_changed_fraction"] = float(
                np.mean([record["cross_turn_changed_fraction"] for record in prior_stability])
            )
        else:
            metrics["prior_objects_mean_mae"] = None
            metrics["prior_objects_mean_changed_fraction"] = None

        before.save(Path(f"{prefix}_before.png"))
        collage.save(Path(f"{prefix}_collage.png"))
        paste_alpha.save(alpha_path)
        allowed.save(Path(f"{prefix}_interaction_mask.png"))
        blend.save(Path(f"{prefix}_blend_mask.png"))
        generated_crop.save(Path(f"{prefix}_local_qwen.png"))
        final.save(final_path)
        save_json(metrics, metrics_path)
        diagnostic_panel(
            before,
            collage,
            generated_crop,
            final,
            allowed,
            crop_box,
            Path(f"{prefix}_panel.png"),
        )

        history.append(
            {
                "step": step,
                "name": name,
                "status": "generated",
                "seed": args.seed,
                "rectangle": list(rectangle.box),
                "placed_box": list(placed_box),
                "local_crop": list(crop_box),
                "prompt": prompt,
                "metrics": metrics,
                "final": str(final_path),
            }
        )
        metric_rows.append(
            {key: value for key, value in metrics.items() if key != "prior_object_stability"}
        )
        tracked_objects.append(
            {
                "step": step,
                "name": name,
                "snapshot": final.copy(),
                "visible": new_object_support,
            }
        )
        current = final
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {
        "id": case_id,
        "steps": len(history),
        "final": str(case_dir / "FINAL.png"),
        "metrics": metric_rows,
    }


def write_metric_tables(results: list[dict], out_dir: Path) -> None:
    rows = [row for result in results for row in result.get("metrics", [])]
    save_json(rows, out_dir / "metrics.json")
    if not rows:
        return
    preferred = ["case_id", "step", "object"]
    fields = preferred + sorted({key for row in rows for key in row} - set(preferred))
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for field in fields:
        values = [
            float(row[field])
            for row in rows
            if isinstance(row.get(field), (int, float)) and row.get(field) is not None
        ]
        if not values:
            continue
        finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
        aggregate[field] = {
            "count": len(values),
            "finite_count": int(finite.size),
            "infinite_count": len(values) - int(finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std()) if finite.size else None,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
        }
    save_json(aggregate, out_dir / "metrics_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--mask_dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--out_dir", type=Path, default=HERE / "final_local_outputs")
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
    parser.add_argument("--process_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--true_cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--inpaint_strength", type=float, default=0.82)
    parser.add_argument("--object_scale", type=float, default=0.92)
    parser.add_argument("--context_fraction", type=float, default=0.20)
    parser.add_argument("--minimum_crop_side", type=int, default=320)
    parser.add_argument("--alpha_threshold", type=int, default=24)
    parser.add_argument("--interaction_dilate_px", type=int, default=16)
    parser.add_argument("--shadow_offset_x", type=int, default=0)
    parser.add_argument("--shadow_offset_y", type=int, default=16)
    parser.add_argument("--shadow_dilate_px", type=int, default=10)
    parser.add_argument("--feather_px", type=float, default=5.0)
    parser.add_argument("--cutout_backend", choices=("rmbg2", "simple"), default="rmbg2")
    parser.add_argument("--rmbg_model_id", default="briaai/RMBG-2.0")
    parser.add_argument(
        "--rmbg_revision",
        default="54c725d3b17ca83aba490092de8acf6118b8bb06",
    )
    parser.add_argument("--rmbg_device", default="cuda")
    parser.add_argument("--rmbg_input_size", type=int, default=1024)
    parser.add_argument("--rmbg_crop_threshold", type=int, default=8)
    parser.add_argument("--alpha_low", type=float, default=10.0)
    parser.add_argument("--alpha_high", type=float, default=45.0)
    parser.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric_model_id", default="facebook/dinov2-base")
    parser.add_argument("--metric_device", default="cpu")
    parser.add_argument("--change_threshold", type=int, default=8)
    return parser.parse_args()


def validate_args(args) -> None:
    if not 0 < args.inpaint_strength <= 1:
        raise ValueError("--inpaint_strength must lie in (0, 1]")
    if not 0 < args.object_scale <= 1:
        raise ValueError("--object_scale must lie in (0, 1]")
    if args.process_size < 256 or args.process_size % 16:
        raise ValueError("--process_size must be >=256 and divisible by 16")
    if not 0 <= args.context_fraction <= 1:
        raise ValueError("--context_fraction must lie in [0, 1]")
    if args.minimum_crop_side <= 0:
        raise ValueError("--minimum_crop_side must be positive")
    if not 0 <= args.change_threshold <= 255:
        raise ValueError("--change_threshold must lie in [0, 255]")


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

    cutouts = prepare_cutouts(cases, args)
    pipe = load_inpaint_pipeline(args)
    evaluator = MetricEvaluator(args.metric_model_id, args.metric_device, args.metrics)
    summary = []
    try:
        for case in tqdm(cases, desc="Final local collage harmonization", unit="case"):
            summary.append(run_case(pipe, case, cutouts, evaluator, args))
            save_json(summary, args.out_dir / "summary.json")
            write_metric_tables(summary, args.out_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        evaluator.close()
    print(f"Completed {len(summary)} case(s): {args.out_dir}")


if __name__ == "__main__":
    main()
