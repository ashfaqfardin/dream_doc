"""E3: JSON-driven multi-scene Qwen object-insertion benchmark.

The prompt suite is the source of truth: each case creates its own base image,
turns the listed Canny/sketch images into reusable reference objects, and adds
those references incrementally. Outputs are resumable at object-step level.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from e1_baseline import fit, infer, load_pipe, save_json


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


def run_case(pipe, case: dict, references: dict, args, out: Path):
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
            history.append({"step": index, "name": item["name"], "status": "resumed", "after": str(after_path)})
            continue
        reference = fit(Image.open(record["image"]), (args.width, args.height))
        before_path = steps_dir / f"{index:02d}_{slug(item['name'])}_before.png"
        current.save(before_path)
        prompt = (
            f"Image 1 is the current scene. Image 2 contains the exact {item['name']} reference. Edit only Image 1: "
            f"add exactly one complete {item['name']} from Image 2 at the most physically plausible unoccupied location. "
            "Preserve the reference identity, geometry, colors, materials and proportions while adapting only scale, "
            "perspective, illumination, support contact, occlusion and shadow. Preserve the scene and every existing object. "
            "Do not replace the scene, copy the white reference background, create a panel, or duplicate an object."
        )
        edit_seed = args.seed + case_id * 10000 + index * 100
        current = infer(pipe, [current, reference], prompt, args, edit_seed)
        current.save(after_path)
        history.append({
            "step": index, "name": item["name"], "status": "generated", "seed": edit_seed,
            "reference": record["image"], "before": str(before_path), "after": str(after_path),
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
    return parser.parse_args()


def main():
    args = parse_args()
    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")
    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    summary = []
    for case in tqdm(cases, desc="E3 prompt suite", unit="case"):
        summary.append(run_case(pipe, case, references, args, out))
        save_json(summary, out / "summary.json")
    print(f"Done: {len(summary)} case(s). Results: {out}")


if __name__ == "__main__":
    main()
