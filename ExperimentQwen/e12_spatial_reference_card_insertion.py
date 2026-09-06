"""E12: single-image spatial reference-card insertion.

Each edit supplies Qwen exactly one stitched image: a dominant scene B on the
left and a small RMBG-cleaned reference-object card O on the right. Because the
same stitched image follows Qwen's native VL and VAE paths, their image layouts
remain aligned. The card area is an explicit conditioning-strength parameter.
Qwen generates one clean scene output; no masks, K/V hooks, compositing, or
post-generation refinement are applied to that output.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from tqdm.auto import tqdm

from e1_baseline import fit, infer, load_pipe, save_json
from e3_prompt_suite import generate_references, load_suite, reference_key, select_cases, slug
from e5_spatial_kv_collage import RMBG2Cutout, generate_base, generate_rmbg_cutouts


HERE = Path(__file__).resolve().parent


def remap_alpha(alpha, low, high):
    values = np.asarray(alpha.convert("L"), dtype=np.float32).copy() / 255.0
    values = np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)
    values = values * values * (3.0 - 2.0 * values)
    return Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255))


def make_reference_card(base, cutout, args):
    """Attach a narrow object-reference strip without altering scene pixels."""
    width, height = base.size
    ratio = args.reference_area_ratio
    panel_width = max(args.minimum_card_width, round(width * ratio / (1.0 - ratio)))
    panel_width = min(panel_width, round(width * args.maximum_card_width_fraction))
    separator = args.card_separator_width
    stitched = Image.new("RGB", (width + separator + panel_width, height))
    stitched.paste(base.convert("RGB"), (0, 0))

    if args.card_background == "scene_blur":
        sample_width = min(width, max(panel_width, width // 4))
        background = base.crop((width - sample_width, 0, width, height))
        background = background.resize((panel_width, height), Image.Resampling.LANCZOS)
        background = background.filter(ImageFilter.GaussianBlur(args.card_background_blur))
    else:
        background = Image.new(
            "RGB", (panel_width, height), (args.card_gray,) * 3
        )
    stitched.paste(
        Image.new("RGB", (separator, height), (args.separator_gray,) * 3),
        (width, 0),
    )
    stitched.paste(background, (width + separator, 0))

    available_width = max(1, panel_width - 2 * args.card_padding)
    available_height = max(1, round(height * args.card_object_height_fraction))
    scale = min(
        available_width / cutout.rgb.width,
        available_height / cutout.rgb.height,
    )
    object_size = (
        max(1, round(cutout.rgb.width * scale)),
        max(1, round(cutout.rgb.height * scale)),
    )
    rgb = cutout.rgb.resize(object_size, Image.Resampling.LANCZOS)
    alpha = cutout.alpha.resize(object_size, Image.Resampling.LANCZOS)
    alpha = remap_alpha(alpha, args.object_alpha_low, args.object_alpha_high)
    object_x = width + separator + (panel_width - object_size[0]) // 2
    object_y = (height - object_size[1]) // 2
    stitched.paste(rgb, (object_x, object_y), alpha)

    reference_mask = Image.new("L", stitched.size)
    reference_mask.paste(alpha, (object_x, object_y))
    actual_panel_ratio = panel_width * height / (stitched.width * stitched.height)
    actual_object_ratio = (
        float(np.asarray(reference_mask, dtype=np.float32).sum() / 255.0)
        / (stitched.width * stitched.height)
    )
    metadata = {
        "scene_box": [0, 0, width, height],
        "card_box": [width + separator, 0, stitched.width, height],
        "object_box": [object_x, object_y, object_x + object_size[0], object_y + object_size[1]],
        "requested_reference_area_ratio": ratio,
        "actual_card_area_ratio": actual_panel_ratio,
        "actual_alpha_weighted_object_ratio": actual_object_ratio,
        "card_background": args.card_background,
    }
    return stitched, reference_mask, metadata


def connected_bbox(mask):
    """Bounding box of the largest 4-connected component on a small grid."""
    mask = mask.astype(bool)
    seen = np.zeros_like(mask, dtype=bool)
    best = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, component = [(int(y), int(x))], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    if not best:
        return None
    ys, xs = zip(*best)
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def alpha_composite_gray(rgb, alpha, size=518):
    rgb = rgb.convert("RGB")
    alpha = alpha.convert("L")
    canvas = Image.new("RGB", rgb.size, (127, 127, 127))
    canvas.paste(rgb, (0, 0), alpha)
    return fit(canvas, (size, size))


def weighted_rgb_histogram(rgb, alpha, bins=16):
    pixels = np.asarray(rgb.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    weights = np.asarray(alpha.convert("L"), dtype=np.float32).reshape(-1) / 255.0
    histograms = []
    for channel in range(3):
        histogram, _ = np.histogram(
            pixels[:, channel], bins=bins, range=(0, 256), weights=weights
        )
        histogram = histogram.astype(np.float64)
        histogram /= max(histogram.sum(), 1e-12)
        histograms.append(histogram)
    return np.concatenate(histograms) / 3.0


class E12Evaluator:
    """Identity, color, scene-drift metrics and qualitative visualizations."""

    def __init__(self, segmenter, args):
        from transformers import AutoImageProcessor, AutoModel

        self.segmenter = segmenter
        self.args = args
        self.device = torch.device(args.metric_device)
        self.processor = AutoImageProcessor.from_pretrained(args.metric_model_id)
        self.model = AutoModel.from_pretrained(args.metric_model_id).to(self.device).eval()

    @torch.inference_mode()
    def embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs).last_hidden_state
        # Mean patch pooling is less crop-position-sensitive than the CLS token.
        vector = output[:, 1:].mean(dim=1)
        return F.normalize(vector.float(), dim=-1)

    def localize_change(self, before, after):
        width, height = before.size
        grid = self.args.metric_localization_grid
        a = np.asarray(before.resize((grid, grid), Image.Resampling.BOX), dtype=np.float32)
        b = np.asarray(after.resize((grid, grid), Image.Resampling.BOX), dtype=np.float32)
        difference = np.abs(a - b).mean(axis=2)
        changed = difference >= self.args.metric_change_threshold
        box = connected_bbox(changed)
        if box is None:
            return (0, 0, width, height), difference, changed
        x0, y0, x1, y1 = box
        margin_x = round((x1 - x0) * self.args.metric_crop_margin)
        margin_y = round((y1 - y0) * self.args.metric_crop_margin)
        x0, y0 = max(0, x0 - margin_x), max(0, y0 - margin_y)
        x1, y1 = min(grid, x1 + margin_x), min(grid, y1 + margin_y)
        scaled = (
            round(x0 * width / grid), round(y0 * height / grid),
            round(x1 * width / grid), round(y1 * height / grid),
        )
        return scaled, difference, changed

    def evaluate(self, name, before, after, reference_cutout, prefix):
        box, difference, changed = self.localize_change(before, after)
        output_crop = after.crop(box)
        try:
            output_cutout = self.segmenter.cutout(
                name, output_crop, self.args.rmbg_crop_threshold
            )
        except Exception as exc:
            warnings.warn(f"Output RMBG failed for {name}; using the localized crop: {exc}")
            output_cutout = type(reference_cutout)(
                name, output_crop, Image.new("L", output_crop.size, 255), box
            )

        reference_view = alpha_composite_gray(reference_cutout.rgb, reference_cutout.alpha)
        output_view = alpha_composite_gray(output_cutout.rgb, output_cutout.alpha)
        identity = float((self.embedding(reference_view) * self.embedding(output_view)).sum().cpu())
        reference_hist = weighted_rgb_histogram(reference_cutout.rgb, reference_cutout.alpha)
        output_hist = weighted_rgb_histogram(output_cutout.rgb, output_cutout.alpha)
        color_similarity = float(np.sqrt(reference_hist * output_hist).sum())

        before_array = np.asarray(before.convert("RGB"), dtype=np.float32)
        after_array = np.asarray(after.convert("RGB"), dtype=np.float32)
        x0, y0, x1, y1 = box
        background = np.ones(before_array.shape[:2], dtype=bool)
        background[y0:y1, x0:x1] = False
        background_error = np.abs(before_array - after_array).mean(axis=2)[background]
        background_mae = float(background_error.mean()) if background_error.size else None
        background_mse = float(np.square(before_array - after_array).mean(axis=2)[background].mean()) if background_error.size else None
        background_psnr = (
            float(20 * np.log10(255.0 / np.sqrt(max(background_mse, 1e-12))))
            if background_mse is not None else None
        )

        crop_path = Path(f"{prefix}_localized_output.png")
        alpha_path = Path(f"{prefix}_localized_output_alpha.png")
        heatmap_path = Path(f"{prefix}_change_heatmap.png")
        panel_path = Path(f"{prefix}_qualitative_panel.png")
        output_cutout.rgb.save(crop_path)
        output_cutout.alpha.save(alpha_path)
        heat = np.uint8(np.clip(difference / max(difference.max(), 1e-6), 0, 1) * 255)
        heat_rgb = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=2)
        heat_image = Image.fromarray(heat_rgb).resize(before.size, Image.Resampling.NEAREST)
        heat_image.save(heatmap_path)

        tile_size = self.args.qualitative_tile_size
        tiles = [
            fit(before, (tile_size, tile_size)),
            alpha_composite_gray(reference_cutout.rgb, reference_cutout.alpha, tile_size),
            fit(after, (tile_size, tile_size)),
            fit(output_view, (tile_size, tile_size)),
            fit(heat_image, (tile_size, tile_size)),
        ]
        labels = ["Before", "Reference", "After", "Localized output", "Change heatmap"]
        title_height = 26
        panel = Image.new(
            "RGB", (tile_size * len(tiles), tile_size + title_height), "white"
        )
        draw = ImageDraw.Draw(panel)
        for index, tile in enumerate(tiles):
            x = index * tile_size
            panel.paste(tile, (x, title_height))
            draw.text((x + 8, 6), labels[index], fill="black")
        panel.save(panel_path)

        return {
            "dino_identity_cosine": identity,
            "foreground_color_bhattacharyya": color_similarity,
            "changed_pixel_fraction": float(changed.mean()),
            "background_mae_0_255": background_mae,
            "background_psnr_db": background_psnr,
            "localized_box": list(box),
            "diagnostics": {
                "localized_output": str(crop_path), "localized_alpha": str(alpha_path),
                "change_heatmap": str(heatmap_path), "qualitative_panel": str(panel_path),
                "panel_order": ["before", "reference", "after", "localized output", "change heatmap"],
            },
        }

    def close(self):
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_case(pipe, case, references, cutouts, evaluator, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    history = []

    for index, item in enumerate(tqdm(
        case["objects"][: args.max_objects or None],
        desc=f"E12 case {case_id:03d}", unit="object", leave=False,
    ), 1):
        name, key = item["name"], reference_key(item)
        record = references.get(key, {})
        prefix = steps_dir / f"{index:02d}_{slug(name)}"
        final_path = Path(f"{prefix}_after.png")
        if record.get("status") != "ready" or key not in cutouts:
            if args.missing_policy == "error":
                raise FileNotFoundError(f"No reference/cutout available for {name}: {record}")
            history.append({"step": index, "name": name, "status": "skipped_missing_reference"})
            continue
        if args.resume and final_path.is_file():
            current = Image.open(final_path).convert("RGB")
            history.append({"step": index, "name": name, "status": "resumed", "final": str(final_path)})
            continue

        before = fit(current, (args.width, args.height))
        before_path = Path(f"{prefix}_before.png")
        stitched_path = Path(f"{prefix}_stitched_input.png")
        reference_mask_path = Path(f"{prefix}_reference_alpha.png")
        before.save(before_path)
        stitched, reference_mask, stitch_info = make_reference_card(before, cutouts[key], args)
        stitched.save(stitched_path)
        reference_mask.save(reference_mask_path)

        prompt = (
            f"The input is one reference-board image. The large left section is the complete source scene and must "
            f"be the only output scene. The narrow right strip is a temporary visual reference card showing the exact "
            f"{name} to insert; it is not part of the scene. Generate a clean {args.width} by {args.height} scene image "
            f"containing exactly one complete instance of that reference {name} in a physically plausible unoccupied "
            "location. Preserve its distinctive shape, proportions, components, colors, materials, textures, and "
            "markings while adapting scale, pose, perspective, illumination, support contact, shadow, and occlusion "
            "naturally. Preserve the left scene's camera, background, layout, and every previously inserted object. "
            "Remove the temporary right reference strip completely. Do not output a split image, border, reference "
            "card, collage, grid, duplicate object, white background, or gray panel."
        )
        seed = args.seed + case_id * 10000 + index * 100
        current = infer(pipe, [stitched], prompt, args, seed)
        current.save(final_path)
        metrics = evaluator.evaluate(
            name, before, current, cutouts[key], prefix
        ) if evaluator is not None else None
        history.append({
            "step": index,
            "name": name,
            "status": "generated",
            "seed": seed,
            "before": str(before_path),
            "stitched_input": str(stitched_path),
            "reference_alpha": str(reference_mask_path),
            "reference_source": record["image"],
            "stitch": stitch_info,
            "model_image_inputs": 1,
            "metrics": metrics,
            "final": str(final_path),
            "postprocess": None,
        })
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    metric_records = [entry["metrics"] for entry in history if entry.get("metrics")]
    metric_means = {}
    if metric_records:
        for key in (
            "dino_identity_cosine", "foreground_color_bhattacharyya",
            "changed_pixel_fraction", "background_mae_0_255", "background_psnr_db",
        ):
            values = [record[key] for record in metric_records if record[key] is not None]
            metric_means[key] = float(np.mean(values)) if values else None
    return {
        "id": case_id, "status": "complete", "objects": len(history),
        "metrics_mean": metric_means, "final": str(case_dir / "FINAL.png"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    parser.add_argument("--out_dir", default="results/qwen_e12_spatial_reference_card")
    parser.add_argument("--case_ids", type=int, nargs="+")
    parser.add_argument("--max_objects", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--missing_policy", choices=("skip", "error"), default="skip")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--lightning_repo", default="lightx2v/Qwen-Image-Lightning")
    parser.add_argument("--lightning_weight", default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object_seed", type=int, default=1337)
    parser.add_argument("--true_cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--rmbg_model_id", default="briaai/RMBG-2.0")
    parser.add_argument("--rmbg_revision", default="54c725d3b17ca83aba490092de8acf6118b8bb06")
    parser.add_argument("--rmbg_device", default="cuda")
    parser.add_argument("--rmbg_input_size", type=int, default=1024)
    parser.add_argument("--rmbg_crop_threshold", type=int, default=8)
    parser.add_argument("--object_alpha_low", type=float, default=.15)
    parser.add_argument("--object_alpha_high", type=float, default=.85)
    parser.add_argument("--reference_area_ratio", type=float, default=.12)
    parser.add_argument("--minimum_card_width", type=int, default=96)
    parser.add_argument("--maximum_card_width_fraction", type=float, default=.25)
    parser.add_argument("--card_padding", type=int, default=12)
    parser.add_argument("--card_object_height_fraction", type=float, default=.65)
    parser.add_argument("--card_separator_width", type=int, default=4)
    parser.add_argument("--card_background", choices=("gray", "scene_blur"), default="gray")
    parser.add_argument("--card_gray", type=int, default=127)
    parser.add_argument("--separator_gray", type=int, default=80)
    parser.add_argument("--card_background_blur", type=float, default=24.0)
    parser.add_argument("--evaluation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric_model_id", default="facebook/dinov2-base")
    parser.add_argument("--metric_device", default="cpu")
    parser.add_argument("--metric_localization_grid", type=int, default=128)
    parser.add_argument("--metric_change_threshold", type=float, default=12.0)
    parser.add_argument("--metric_crop_margin", type=float, default=.12)
    parser.add_argument("--qualitative_tile_size", type=int, default=320)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.reference_area_ratio < .5:
        raise ValueError("reference_area_ratio must be in (0, 0.5)")
    if not 0 < args.maximum_card_width_fraction < 1:
        raise ValueError("maximum_card_width_fraction must be in (0,1)")
    if not 0 < args.card_object_height_fraction <= 1:
        raise ValueError("card_object_height_fraction must be in (0,1]")
    if not 0 <= args.object_alpha_low < args.object_alpha_high <= 1:
        raise ValueError("object alpha limits must satisfy 0 <= low < high <= 1")
    if not 0 <= args.card_gray <= 255 or not 0 <= args.separator_gray <= 255:
        raise ValueError("card colors must be in [0,255]")
    if args.metric_localization_grid < 16 or args.metric_change_threshold <= 0:
        raise ValueError("invalid metric localization grid or change threshold")
    if args.metric_crop_margin < 0 or args.qualitative_tile_size < 64:
        raise ValueError("invalid metric crop margin or qualitative tile size")
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E12 targets diffusers 0.40.0; found {diffusers.__version__}")
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E12") from exc

    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")
    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    segmenter = RMBG2Cutout(args.rmbg_model_id, args.rmbg_revision, args.rmbg_device, args.rmbg_input_size)
    evaluator = None
    try:
        cutouts = generate_rmbg_cutouts(segmenter, references, args, out)
        if args.evaluation:
            evaluator = E12Evaluator(segmenter, args)
        summary = []
        for case in tqdm(cases, desc="E12 spatial-reference-card suite", unit="case"):
            summary.append(run_case(pipe, case, references, cutouts, evaluator, args, out))
            save_json(summary, out / "summary.partial.json")
    finally:
        if evaluator is not None:
            evaluator.close()
        segmenter.close()
    save_json({
        "method": "single stitched image with spatially budgeted RMBG reference card",
        "model_image_inputs": 1,
        "reference_area_ratio": args.reference_area_ratio,
        "feature_intervention": None,
        "postprocess": None,
        "evaluation": {
            "enabled": args.evaluation,
            "identity_encoder": args.metric_model_id if args.evaluation else None,
            "metrics": [
                "DINOv2 cosine identity", "foreground color Bhattacharyya similarity",
                "changed-pixel fraction", "background MAE", "background PSNR",
            ],
        },
        "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
