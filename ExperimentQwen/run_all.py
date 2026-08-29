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
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="ID", help="Run only these IDs, e.g. --only e1 e2")
    parser.add_argument("--skip", nargs="+", metavar="ID", default=[], help="Skip these experiment IDs")
    parser.add_argument("--device", default="cuda", help="Device forwarded to each experiment")
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
        missing = validate(experiment)
        print("\n" + "=" * 72)
        print(f"  {experiment['id'].upper()}: {experiment['name']}")
        print("=" * 72, flush=True)
        if missing:
            print("Missing prerequisites:")
            for path in missing:
                print(f"  - {path}")
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
