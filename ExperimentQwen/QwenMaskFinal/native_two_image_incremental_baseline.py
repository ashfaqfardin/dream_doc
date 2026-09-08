"""Strict native Qwen baseline: [current scene, reference] -> next scene.

No placement mask, collage, cutout, inpainting mask, feature injection, or
post-generation restoration is used at inference or evaluation time.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm

from generate_base_images import lightning_scheduler, make_generator, prepare_peft_lora_backend, save_json
HERE = Path(__file__).resolve().parent
EXPERIMENT_QWEN = HERE.parent
DEFAULT_PROMPTS = HERE / "e5_prompts.json"
DEFAULT_BASES = HERE / "BaseImages"
DEFAULT_REFERENCES = EXPERIMENT_QWEN / "references"


class MetricEvaluator:
    """Optional mask-free DINO proxy used only for descriptive diagnostics."""

    def __init__(self, model_id: str, device: str, enabled: bool):
        self.enabled = enabled
        self.device = torch.device(device)
        self.processor = self.model = None
        if enabled:
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @torch.inference_mode()
    def dino_similarity(self, first: Image.Image, second: Image.Image):
        if not self.enabled:
            return None
        inputs = self.processor(images=[first, second], return_tensors="pt")
        output = self.model(**{key: value.to(self.device) for key, value in inputs.items()})
        features = getattr(output, "pooler_output", None)
        if features is None:
            features = output.last_hidden_state[:, 0]
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        return float((features[0] * features[1]).sum().cpu())

    @staticmethod
    def edge_similarity(first: Image.Image, second: Image.Image) -> float:
        vectors = []
        for image in (first, second):
            gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            dy, dx = np.gradient(gray)
            vectors.append(np.sqrt(dx * dx + dy * dy).reshape(-1))
        denominator = float(np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
        return float(np.dot(vectors[0], vectors[1]) / max(denominator, 1e-12))

    def close(self):
        if self.model is not None:
            self.model.to("cpu")
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
        finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
        if values:
            aggregate[field] = {
                "count": len(values), "finite_count": int(finite.size),
                "mean": float(finite.mean()) if finite.size else None,
                "std": float(finite.std()) if finite.size else None,
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
            }
    save_json(aggregate, out_dir / "metrics_summary.json")


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
    source = slug(Path(item.get("canny_file", "")).stem)
    name = slug(item["name"])
    exact = directory / f"{name}__{source}.png"
    if exact.is_file():
        return exact
    candidates = sorted(directory.glob(f"*__{source}.*")) if source else []
    candidates += sorted(directory.glob(f"{name}__*"))
    candidates += [directory / f"{source}.png", directory / f"{name}.png"]
    matches = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"No unique reference for {item['name']!r} in {directory}; "
        f"matches: {[path.name for path in matches]}"
    )


def load_pipeline(args):
    from diffusers import QwenImageEditPlusPipeline
    loading = tqdm(total=3, desc="Loading native Qwen baseline", unit="stage")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_id, scheduler=lightning_scheduler(), torch_dtype=torch.bfloat16
    )
    loading.update()
    loading.set_description("Loading 8-step Lightning LoRA")
    prepare_peft_lora_backend()
    pipe.load_lora_weights(
        args.lightning_repo, weight_name=args.lightning_weight, adapter_name="lightning"
    )
    pipe.set_adapters(["lightning"], adapter_weights=[args.lora_scale])
    loading.update()
    loading.set_description(f"Moving Qwen baseline to {args.device}")
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    loading.update()
    loading.close()
    return pipe


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.pad(image.convert("RGB"), size, Image.Resampling.LANCZOS, color="white")


@torch.inference_mode()
def edit(pipe, current: Image.Image, reference: Image.Image, name: str, args):
    prompt = (
        f"Image 1 is the current scene. Image 2 is a reference image of a {name}. "
        f"Add the Image 2 {name} naturally to Image 1. Preserve the existing scene and objects."
    )
    cfg_enabled = args.true_cfg_scale > 1.0
    result = pipe(
        image=[current, reference], prompt=prompt,
        negative_prompt=args.negative_prompt if cfg_enabled else None,
        true_cfg_scale=args.true_cfg_scale, num_inference_steps=args.steps,
        width=args.width, height=args.height,
        generator=make_generator(args.device, args.seed),
    )
    return result.images[0].convert("RGB"), prompt


def full_image_metrics(before: Image.Image, after: Image.Image, threshold: int) -> dict:
    a = np.asarray(before.convert("RGB"), dtype=np.int16)
    b = np.asarray(after.convert("RGB"), dtype=np.int16)
    difference = np.abs(b - a)
    mse = float(np.square(difference.astype(np.float64)).mean())
    return {
        "full_image_mae": float(difference.mean()),
        "full_image_psnr": float("inf") if mse == 0 else float(20 * math.log10(255 / math.sqrt(mse))),
        "full_image_max_error": float(difference.max()),
        "full_image_changed_fraction": float((difference.max(axis=2) > threshold).mean()),
    }


def global_reference_metrics(evaluator, reference: Image.Image, output: Image.Image) -> dict:
    """Mask-free whole-image proxy; not equivalent to localized fidelity."""
    reference_view = fit(reference, (224, 224))
    output_view = output.resize((224, 224), Image.Resampling.LANCZOS)
    return {
        "global_reference_dino_proxy": evaluator.dino_similarity(reference_view, output_view),
        "global_reference_edge_proxy": evaluator.edge_similarity(reference_view, output_view),
    }


def panel(before, reference, after, path: Path) -> None:
    views = [("Before", before), ("Reference", reference), ("Native Qwen output", after)]
    tile = 320
    canvas = Image.new("RGB", (tile * 3, tile + 30), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(views):
        canvas.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (index * tile, 30))
        draw.text((index * tile + 6, 7), label, fill="black")
    canvas.save(path)


def run_case(pipe, evaluator, case: dict, args) -> dict:
    case_id = int(case["id"])
    base_path = args.base_dir / f"base_{case_id:03d}.png"
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base image: {base_path}")
    current = Image.open(base_path).convert("RGB")
    case_dir = args.out_dir / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current.save(case_dir / "base.png")
    history, metric_rows = [], []

    for step, item in enumerate(case["objects"], start=1):
        name = item["name"]
        prefix = steps_dir / f"{step:02d}_{slug(name)}"
        final_path = Path(f"{prefix}_final.png")
        metrics_path = Path(f"{prefix}_metrics.json")
        if args.resume and final_path.is_file() and metrics_path.is_file():
            current = Image.open(final_path).convert("RGB")
            metric_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            history.append({"step": step, "name": name, "status": "resumed"})
            continue

        before = current.copy()
        source_path = reference_path(item, args.reference_dir)
        source_reference = Image.open(source_path).convert("RGB")
        reference_input = fit(source_reference, (args.width, args.height))
        after, prompt = edit(pipe, before, reference_input, name, args)
        metrics = {
            "case_id": case_id, "step": step, "object": name,
            **full_image_metrics(before, after, args.change_threshold),
            **global_reference_metrics(evaluator, source_reference, after),
        }
        before.save(Path(f"{prefix}_before.png"))
        reference_input.save(Path(f"{prefix}_reference_input.png"))
        after.save(final_path)
        panel(before, reference_input, after, Path(f"{prefix}_panel.png"))
        save_json(metrics, metrics_path)
        history.append({
            "step": step, "name": name, "status": "generated", "seed": args.seed,
            "model_inputs": ["current_scene", "reference_object"],
            "prompt": prompt, "reference": str(source_path), "final": str(final_path),
            "metrics": metrics,
        })
        metric_rows.append(metrics)
        current = after
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "steps": len(history), "final": str(case_dir / "FINAL.png"), "metrics": metric_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--out_dir", type=Path, default=HERE / "results" / "NativeTwoImageBaseline_02")
    parser.add_argument("--case_ids", type=int, nargs="+")
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
    parser.add_argument("--true_cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric_model_id", default="facebook/dinov2-base")
    parser.add_argument("--metric_device", default="cpu")
    parser.add_argument("--change_threshold", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key in ("prompts", "base_dir", "reference_dir", "out_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.prompts, args.case_ids)
    save_json({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, args.out_dir / "config.json")
    pipe = load_pipeline(args)
    evaluator = MetricEvaluator(args.metric_model_id, args.metric_device, args.metrics)
    summary = []
    try:
        for case in tqdm(cases, desc="Strict native two-image baseline", unit="case"):
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
