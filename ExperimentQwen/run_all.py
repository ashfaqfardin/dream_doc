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
    {"id":"e4","name":"3D Feature Trajectory and Reference-Attention Lab","script":HERE/"e4_3d_feature_trajectory_lab.py","args":["--prompts",str(HERE/"e3_prompts.json"),"--out_dir",str(ROOT/"results"/"qwen_e4_3d_lab")],"requires":[HERE/"e3_prompts.json",HERE/"object_canny"]},
    {"id":"e5","name":"One-Pass Collage-Primary Residual Feature Routing with RMBG-2.0","script":HERE/"e5_spatial_kv_collage.py","args":["--prompts",str(HERE/"e5_prompts.json"),"--out_dir",str(ROOT/"results"/"qwen_e5_spatial_kv_collage")],"requires":[HERE/"e5_prompts.json",HERE/"object_canny"]},
    {"id":"e6","name":"Block-Sparse Source-Aware Reference Attention","script":HERE/"e6_block_sparse_reference_attention.py","args":["--prompts",str(HERE/"e5_prompts.json"),"--out_dir",str(ROOT/"results"/"qwen_e6_block_sparse_attention")],"requires":[HERE/"e5_prompts.json",HERE/"object_canny"]},
    {"id":"e8","name":"Training-Free Masked Object-Attention Insertion","script":HERE/"e8_masked_object_attention_insertion.py","args":["--prompts",str(HERE/"e5_prompts.json"),"--out_dir",str(ROOT/"results"/"qwen_e8_masked_object_attention")],"requires":[HERE/"e5_prompts.json",HERE/"object_canny"]},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="ID", help="Run only these IDs, e.g. --only e1 e2")
    parser.add_argument("--skip", nargs="+", metavar="ID", default=[], help="Skip these experiment IDs")
    parser.add_argument("--device", default="cuda", help="Device forwarded to each experiment")
    parser.add_argument("--true_cfg_scale", type=float, help="Optional traditional CFG scale forwarded to experiments")
    parser.add_argument("--negative_prompt", help="Negative prompt; effective only when --true_cfg_scale is greater than 1")
    parser.add_argument("--e4_case_id", type=int, default=1, help="E3 prompt-suite case analyzed by E4")
    parser.add_argument("--e4_object_index", type=int, default=1, help="One-based object index within the selected E4 case")
    parser.add_argument("--e4_all_prompts", action="store_true", help="Run E4 for every object in every e3_prompts.json case")
    parser.add_argument("--e4_max_cases", type=int, help="Limit E4 all-prompts mode to the first N cases")
    parser.add_argument("--e4_max_objects", type=int, help="Limit E4 all-prompts mode to the first N objects per case")
    parser.add_argument("--e4_tokens_per_snapshot", type=int, default=128, help="Spatial tokens retained per E4 layer/step snapshot")
    parser.add_argument("--e4_skip_depth", action="store_true", help="Use image-y as E4's depth proxy instead of loading Depth Anything")
    parser.add_argument("--e5_case_ids", type=int, nargs="+", help="Subset of E3 prompt-suite cases for E5")
    parser.add_argument("--e5_max_objects", type=int, help="Limit objects per E5 case for a smoke test")
    parser.add_argument("--e5_kv_layers", default="middle", help="E5 residual routing layers: middle, all, or explicit indices")
    parser.add_argument("--e5_base_weight", type=float, default=.15, help="Weak aligned base residual strength")
    parser.add_argument("--e5_identity_weight", type=float, default=.35, help="Foreground identity residual strength")
    parser.add_argument("--e6_case_ids", type=int, nargs="+", help="Subset of prompt-suite cases for E6")
    parser.add_argument("--e6_max_objects", type=int, help="Limit objects per E6 case")
    parser.add_argument("--e6_routing_layers", default="middle", help="E6 routed layers")
    parser.add_argument("--e6_base_prior", type=float, default=.10, help="E6 local-base logit prior")
    parser.add_argument("--e6_object_prior", type=float, default=.30, help="E6 object-foreground logit prior")
    parser.add_argument("--e8_case_ids", type=int, nargs="+", help="Subset of prompt-suite cases for E8")
    parser.add_argument("--e8_max_objects", type=int, help="Limit objects per E8 case")
    parser.add_argument("--e8_injection_layers", default="6-35", help="E8 object-attention layers")
    parser.add_argument("--e8_object_mass", type=float, default=.35, help="E8 desired object-to-main attention mass ratio")
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
        if args.true_cfg_scale is not None:
            command.extend(["--true_cfg_scale", str(args.true_cfg_scale)])
        if args.negative_prompt is not None:
            command.extend(["--negative_prompt", args.negative_prompt])
        if experiment["id"] == "e4":
            command.extend(["--tokens_per_snapshot", str(args.e4_tokens_per_snapshot)])
            if args.e4_all_prompts:
                command.append("--all_prompts")
                if args.e4_max_cases is not None:
                    command.extend(["--max_cases", str(args.e4_max_cases)])
                if args.e4_max_objects is not None:
                    command.extend(["--max_objects", str(args.e4_max_objects)])
            else:
                command.extend(["--case_id", str(args.e4_case_id), "--object_index", str(args.e4_object_index)])
            if args.e4_skip_depth:
                command.append("--skip_depth")
        if experiment["id"] == "e5":
            command.extend([
                "--kv_layers", args.e5_kv_layers,
                "--base_residual_weight", str(args.e5_base_weight),
                "--identity_residual_weight", str(args.e5_identity_weight),
            ])
            if args.e5_case_ids:
                command.extend(["--case_ids", *map(str, args.e5_case_ids)])
            if args.e5_max_objects is not None:
                command.extend(["--max_objects", str(args.e5_max_objects)])
        if experiment["id"] == "e6":
            command.extend([
                "--routing_layers", args.e6_routing_layers,
                "--base_attention_prior", str(args.e6_base_prior),
                "--object_attention_prior", str(args.e6_object_prior),
            ])
            if args.e6_case_ids:
                command.extend(["--case_ids", *map(str, args.e6_case_ids)])
            if args.e6_max_objects is not None:
                command.extend(["--max_objects", str(args.e6_max_objects)])
        if experiment["id"] == "e8":
            command.extend([
                "--injection_layers", args.e8_injection_layers,
                "--object_attention_mass", str(args.e8_object_mass),
            ])
            if args.e8_case_ids:
                command.extend(["--case_ids", *map(str, args.e8_case_ids)])
            if args.e8_max_objects is not None:
                command.extend(["--max_objects", str(args.e8_max_objects)])
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
