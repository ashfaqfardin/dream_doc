"""E3: JSON-driven multi-scene evaluation of the E2 insertion pipeline.

The prompt suite is the source of truth: each case creates its own base image,
turns the listed Canny/sketch images into reusable reference objects, then uses
E2's counterfactual placement + SAM collage + protected Qwen repaint pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageChops
from tqdm.auto import tqdm

from e1_baseline import fit, infer, load_pipe, save_json
from e2_sam_collage_repaint import (
    composite, load_segmenter, place_cutout, probe_placement, protect_scene,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_suite(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("prompts")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty 'prompts' list")
    seen = set()
    for case in cases:
        if not {"id", "base_prompt", "objects"} <= case.keys():
            raise ValueError(f"Malformed E3 case: {case}")
        if case["id"] in seen:
            raise ValueError(f"Duplicate case id {case['id']}")
        seen.add(case["id"])
    return cases


def resolve_input(raw: str | None, prompt_file: Path) -> Path | None:
    if not raw:
        return None
    supplied = Path(raw)
    candidates = [supplied, ROOT / supplied, prompt_file.parent / supplied]
    if supplied.parts and supplied.parts[0] == HERE.name:
        candidates.append(HERE / Path(*supplied.parts[1:]))
    return next((path.resolve() for path in candidates if path.is_file()), None)


def select_cases(cases: list[dict], requested: list[int] | None) -> list[dict]:
    if not requested:
        return cases
    wanted = set(requested)
    selected = [case for case in cases if int(case["id"]) in wanted]
    missing = wanted - {int(case["id"]) for case in selected}
    if missing:
        raise ValueError(f"Unknown E3 case ids: {sorted(missing)}")
    return selected


def reference_key(item: dict) -> str:
    source = Path(item.get("canny_file") or "missing").stem
    return f"{slug(item['name'])}__{slug(source)}"


def generate_references(pipe, cases, args, out: Path, prompt_file: Path):
    reference_dir = out / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    unique = {}
    for case in cases:
        for item in case["objects"][: args.max_objects or None]:
            unique.setdefault(reference_key(item), item)

    records = {}
    missing = []
    for key, item in tqdm(unique.items(), desc="E3 reference objects", unit="object"):
        target = reference_dir / f"{key}.png"
        source = resolve_input(item.get("canny_file"), prompt_file)
        if source is None:
            record = {"name": item["name"], "status": "missing", "canny_file": item.get("canny_file")}
            records[key] = record
            missing.append(record)
            if args.missing_policy == "error":
                raise FileNotFoundError(f"Missing Canny/sketch input for {item['name']}: {item.get('canny_file')}")
            continue
        if not target.is_file() or not args.resume:
            control = fit(Image.open(source), (args.width, args.height))
            prompt = (
                f"{item['obj_prompt']} Image 1 is the object's edge drawing or sketch. "
                "Preserve its contour, structure, pose, proportions and viewpoint. Render exactly one complete object "
                "with realistic materials and studio lighting, centered on a clean plain white background. "
                "Do not add scenery, floor, labels, borders, text or other objects."
            )
            infer(pipe, [control], prompt, args, args.object_seed).save(target)
        records[key] = {
            "name": item["name"], "status": "ready", "canny_file": str(source),
            "image": str(target), "seed": args.object_seed,
        }
    save_json(records, out / "references.json")
    save_json(missing, out / "missing_inputs.json")
    return records


def generate_cutouts(segmenter, references, args, out: Path):
    cutout_dir = out / "cutouts"
    cutout_dir.mkdir(parents=True, exist_ok=True)
    cutouts = {}
    for key, record in tqdm(references.items(), desc="E3 reference cutouts", unit="object"):
        if record.get("status") != "ready":
            continue
        image = Image.open(record["image"]).convert("RGB").resize((args.width, args.height))
        cutout = segmenter.cutout(record["name"], image, args.background_threshold)
        cutout.rgb.save(cutout_dir / f"{key}_rgb.png")
        cutout.alpha.save(cutout_dir / f"{key}_alpha.png")
        cutouts[key] = cutout
    return cutouts


def run_case(pipe, case: dict, references: dict, cutouts: dict, mask_backend: str, args, out: Path):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    base_path = case_dir / "base.png"
    if base_path.is_file() and args.resume:
        base = Image.open(base_path).convert("RGB")
    else:
        blank = Image.new("RGB", (args.width, args.height), "white")
        prompt = "Replace blank Image 1 with this scene: " + case["base_prompt"] + " Fill the complete frame without borders."
        base = infer(pipe, [blank], prompt, args, args.seed + case_id * 10000)
        base.save(base_path)

    current = base
    occupied = Image.new("L", current.size)
    history = []
    objects = case["objects"][: args.max_objects or None]
    for index, item in enumerate(tqdm(objects, desc=f"Case {case_id:03d}", unit="object", leave=False), 1):
        key = reference_key(item)
        record = references.get(key, {})
        after_path = steps_dir / f"{index:02d}_{slug(item['name'])}_after.png"
        if record.get("status") != "ready":
            history.append({"step": index, "name": item["name"], "status": "skipped_missing_reference"})
            continue
        if after_path.is_file() and args.resume:
            current = Image.open(after_path).convert("RGB")
            prior_mask = steps_dir / f"{index:02d}_{slug(item['name'])}_mask.png"
            if prior_mask.is_file():
                occupied = ImageChops.lighter(occupied, Image.open(prior_mask).convert("L"))
            history.append({"step": index, "name": item["name"], "status": "resumed", "after": str(after_path)})
            continue
        before_path = steps_dir / f"{index:02d}_{slug(item['name'])}_before.png"
        current.save(before_path)
        object_name = item["name"]
        heatmap_path = steps_dir / f"{index:02d}_{slug(object_name)}_placement_heatmap.png"
        box, probe_info = probe_placement(
            pipe, current, object_name, cutouts[key], occupied, args,
            args.seed + case_id * 100000 + index * 1000, heatmap_path,
        )
        object_canvas, mask, placed_box = place_cutout(cutouts[key], box, current.size, args.object_scale)
        collage = composite(current, object_canvas, mask)
        collage_path = steps_dir / f"{index:02d}_{slug(object_name)}_collage.png"
        mask_path = steps_dir / f"{index:02d}_{slug(object_name)}_mask.png"
        collage.save(collage_path)
        mask.save(mask_path)
        prompt = (
            f"Image 1 is the scene containing one newly pasted {object_name} already at its required location. "
            f"Harmonize that pasted {object_name} without moving, removing, duplicating or redesigning it. Preserve its "
            "identity, colors, materials, structure, pose and proportions. Correct only its boundary, local perspective, "
            "lighting, contact shadow and physical integration. Preserve every other pixel and every existing object."
        )
        edit_seed = args.seed + case_id * 10000 + index * 100
        raw = infer(pipe, [collage], prompt, args, edit_seed)
        raw_path = steps_dir / f"{index:02d}_{slug(object_name)}_raw_qwen.png"
        raw.save(raw_path)
        current, zone, halo = protect_scene(current, raw, object_canvas, mask, args)
        zone.save(steps_dir / f"{index:02d}_{slug(object_name)}_object_zone.png")
        halo.save(steps_dir / f"{index:02d}_{slug(object_name)}_interaction_halo.png")
        current.save(after_path)
        occupied = ImageChops.lighter(occupied, mask)
        history.append({
            "step": index, "name": item["name"], "status": "generated", "seed": edit_seed,
            "mask_backend": mask_backend, "placement_backend": "denoise_delta",
            "reference": record["image"], "before": str(before_path), "after": str(after_path),
            "configured_box": list(box), "placed_box": list(placed_box), "probe": probe_info,
        })
        save_json(history, case_dir / "history.json")
    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    save_json({"id": case_id, "base_prompt": case["base_prompt"], "objects": objects}, case_dir / "case.json")
    return {"id": case_id, "status": "complete", "final": str(case_dir / "FINAL.png"), "steps": len(history)}


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", default=str(HERE / "e3_prompts.json"))
    parser.add_argument("--out_dir", default="results/qwen_e3_prompt_suite")
    parser.add_argument("--case_ids", type=int, nargs="+", help="Subset of prompt ids; all cases when omitted")
    parser.add_argument("--max_objects", type=int, default=None, help="Limit objects per case for a smoke test")
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
    parser.add_argument("--mask_backend", choices=("auto", "sam2", "difference"), default="auto")
    parser.add_argument("--sam_model_id", default="facebook/sam2-hiera-small")
    parser.add_argument("--sam_device", default="cpu")
    parser.add_argument("--background_threshold", type=float, default=24)
    parser.add_argument("--probe_steps", type=int, default=4)
    parser.add_argument("--probe_quantile", type=float, default=.88)
    parser.add_argument("--probe_blur", type=float, default=1.2)
    parser.add_argument("--box_margin", type=int, default=24)
    parser.add_argument("--occupancy_margin", type=int, default=24)
    parser.add_argument("--default_object_height", type=float, default=.25)
    parser.add_argument("--object_height_priors", default=None)
    parser.add_argument("--object_scale", type=float, default=.92)
    parser.add_argument("--boundary_px", type=int, default=12)
    parser.add_argument("--interaction_px", type=int, default=28)
    parser.add_argument("--feather_px", type=float, default=5)
    parser.add_argument("--preserve_reference_core", action="store_true")
    parser.add_argument("--core_erode_px", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")
    segmenter, active_mask_backend = load_segmenter(args)
    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    cutouts = generate_cutouts(segmenter, references, args, out)
    summary = []
    for case in tqdm(cases, desc="E3 prompt suite", unit="case"):
        summary.append(run_case(pipe, case, references, cutouts, active_mask_backend, args, out))
        save_json(summary, out / "summary.json")
    print(f"Done: {len(summary)} case(s). Results: {out}")


if __name__ == "__main__":
    main()
