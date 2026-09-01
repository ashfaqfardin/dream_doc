"""Run registered ExperimentQwen experiments from one entry point.

Examples:
    python ExperimentQwen/run_all.py
    python ExperimentQwen/run_all.py --only e1
    python ExperimentQwen/run_all.py --skip e1
    python ExperimentQwen/run_all.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SKETCH_DIR = ROOT / "KontextPipeline" / "sketch"
DEFAULT_E1_DIR = ROOT / "results" / "qwen_e1_baseline"


# Add future experiments here. Arguments should not include --device because the
# runner appends the user-selected device consistently to every experiment.
EXPERIMENTS = [
    {
        "id": "e1",
        "name": "Qwen 8-step Baseline: Sketches -> Objects -> Scene -> Sequential Insertions",
        "script": HERE / "e1_baseline.py",
        "args": [
            "--sketch_dir", str(DEFAULT_SKETCH_DIR),
            "--out_dir", str(ROOT / "results" / "qwen_e1_baseline"),
        ],
        "requires": [DEFAULT_SKETCH_DIR],
    },
    {
        "id": "e2",
        "name": "Counterfactual Placement + SAM Collage-and-Repaint",
        "script": HERE / "e2_sam_collage_repaint.py",
        "args": [
            "--e1_dir", str(DEFAULT_E1_DIR),
            "--placement_backend", "denoise_delta",
            "--out_dir", str(ROOT / "results" / "qwen_e2_sam_collage"),
        ],
        "requires": [
            DEFAULT_SKETCH_DIR,
        ],
    },
    {
        "id": "e3",
        "name": "E2 Pipeline at Scale: 20-Scene JSON Prompt Suite",
        "script": HERE / "e3_prompt_suite_baseline.py",
        "args": [
            "--prompts", str(HERE / "e3_prompts.json"),
            "--out_dir", str(ROOT / "results" / "qwen_e3_prompt_suite"),
        ],
        "requires": [HERE / "e3_prompts.json", HERE / "object_canny"],
    },
    {"id":"e4","name":"Feature/Frequency-Locked Verified Insertion","script":HERE/"e4_feature_frequency_locked_insertion.py","args":["--prompts",str(HERE/"e3_prompts.json"),"--out_dir",str(ROOT/"results"/"qwen_e4_feature_frequency")],"requires":[HERE/"e3_prompts.json",HERE/"object_canny"]},
    {"id":"e5","name":"Qwen Object Feature/KV Transplant","script":HERE/"e5_object_feature_transplant.py","args":["--prompts",str(HERE/"e3_prompts.json"),"--out_dir",str(ROOT/"results"/"qwen_e5_object_feature_transplant")],"requires":[HERE/"e3_prompts.json",HERE/"object_canny"]},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="ID", help="Run only these IDs, e.g. --only e1 e2")
    parser.add_argument("--skip", nargs="+", metavar="ID", default=[], help="Skip these experiment IDs")
    parser.add_argument("--device", default="cuda", help="Device forwarded to each experiment")
    parser.add_argument(
        "--e1_dir", type=Path,
        help="Existing E1 output directory for E2. If omitted, common output locations are detected.",
    )
    parser.add_argument(
        "--continue_on_error", action="store_true",
        help="Continue with later experiments after a failure",
    )
    parser.add_argument("--list", action="store_true", help="List registered experiments and exit")
    return parser.parse_args()


def select_experiments(args):
    known = {experiment["id"] for experiment in EXPERIMENTS}
    requested = set(args.only or known)
    skipped = set(args.skip)
    unknown = (requested | skipped) - known
    if unknown:
        raise SystemExit(f"Unknown experiment ID(s): {sorted(unknown)}; available: {sorted(known)}")
    return [item for item in EXPERIMENTS if item["id"] in requested and item["id"] not in skipped]


def validate(experiment):
    missing = []
    if not Path(experiment["script"]).is_file():
        missing.append(str(experiment["script"]))
    for required in experiment.get("requires", []):
        if not Path(required).exists():
            missing.append(str(required))
    return missing


def resolve_e1_dependency(experiment, requested_dir=None):
    """Resolve E1 outputs from runner and direct-script working directories."""
    if experiment["id"] != "e2":
        return
    candidates = []
    if requested_dir is not None:
        candidates.append(requested_dir.resolve())
    candidates.extend([
        DEFAULT_E1_DIR,
        HERE / "results" / "qwen_e1_baseline",
    ])
    # Preserve order while removing duplicate absolute paths.
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    chosen = next(
        (path for path in candidates if (path / "base.png").is_file() and (path / "objects.json").is_file()),
        candidates[0],
    )
    position = experiment["args"].index("--e1_dir") + 1
    experiment["args"][position] = str(chosen)
    complete = (chosen / "base.png").is_file() and (chosen / "objects.json").is_file()
    experiment["requires"] = [] if complete else [DEFAULT_SKETCH_DIR]
    experiment["e1_candidates"] = candidates


def format_duration(seconds):
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:d}:{sec:02d}"


def main():
    args = parse_args()
    if args.list:
        for experiment in EXPERIMENTS:
            print(f"{experiment['id']:>4}  {experiment['name']}")
        return

    selected = select_experiments(args)
    if not selected:
        print("No experiments selected.")
        return

    print(f"Running {len(selected)} experiment(s): {[item['id'] for item in selected]}")
    failures = []
    total_started = time.perf_counter()
    for experiment in selected:
        resolve_e1_dependency(experiment, args.e1_dir)
        missing = validate(experiment)
        print("\n" + "=" * 72)
        print(f"  {experiment['id'].upper()}: {experiment['name']}")
        print("=" * 72, flush=True)
        if missing:
            print("Missing prerequisites:")
            for path in missing:
                print(f"  - {path}")
            if experiment["id"] == "e2":
                print("\nNo reusable E1 setup was found. E2 can generate its own setup, but its sketch directory is also missing.")
                print("Checked optional E1 caches:")
                for directory in experiment.get("e1_candidates", []):
                    print(f"  - {directory}")
                print("Provide KontextPipeline/sketch, or pass `--e1_dir PATH` to an existing E1 output.")
            failures.append(experiment["id"])
            if not args.continue_on_error:
                break
            continue

        command = [
            sys.executable, str(experiment["script"]),
            *map(str, experiment.get("args", [])),
            "--device", args.device,
        ]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=str(ROOT), env=os.environ.copy())
        elapsed = format_duration(time.perf_counter() - started)
        if result.returncode:
            failures.append(experiment["id"])
            print(f"[FAIL] {experiment['id']} exited with code {result.returncode} after {elapsed}")
            if not args.continue_on_error:
                print("Stopping because the experiment failed.")
                break
        else:
            print(f"[OK] {experiment['id']} completed in {elapsed}")

    print("\n" + "=" * 72)
    print(f"Total time: {format_duration(time.perf_counter() - total_started)}")
    if failures:
        print(f"FAILED: {failures}")
        raise SystemExit(1)
    print("All selected experiments completed successfully.")


if __name__ == "__main__":
    main()
