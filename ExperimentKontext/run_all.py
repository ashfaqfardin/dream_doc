"""
Run all 6 Kontext experiments in recommended order.

Prerequisites — run once before this script:
    python KontextPipeline/run.py --steps 1,2 --out_dir results/kontext_setup

Usage:
    python ExperimentKontext/run_all.py
    python ExperimentKontext/run_all.py --only e1 e4      # run specific experiments
    python ExperimentKontext/run_all.py --skip e6          # skip slow ones
"""
import os
import sys
import argparse
import subprocess


SCENE   = "results/kontext_setup/step_00_base.png"
OBJ_DIR = "results/kontext_setup/objects"
DEVICE  = "cuda"

BASE = os.path.dirname(__file__)

EXPERIMENTS = [
    {
        "id":   "e4",
        "name": "Timestep Commitment",
        "script": os.path.join(BASE, "e4_timestep_commitment.py"),
        "args": ["--scene", SCENE, "--out_dir", "results/e4_timestep_commitment"],
    },
    {
        "id":   "e1",
        "name": "Attention Visualization",
        "script": os.path.join(BASE, "e1_attention_viz.py"),
        "args": ["--scene", SCENE, "--out_dir", "results/e1_attention_viz"],
    },
    {
        "id":   "e3",
        "name": "Layer Ablation",
        "script": os.path.join(BASE, "e3_layer_ablation.py"),
        "args": ["--scene", SCENE, "--out_dir", "results/e3_layer_ablation"],
    },
    {
        "id":   "e2",
        "name": "RoPE Temporal Index Ablation",
        "script": os.path.join(BASE, "e2_rope_ablation.py"),
        "args": ["--scene", SCENE,
                 "--obj",   os.path.join(OBJ_DIR, "obj_bicycle.png"),
                 "--out_dir", "results/e2_rope_ablation"],
    },
    {
        "id":   "e5",
        "name": "Multi-Context Attention Segregation",
        "script": os.path.join(BASE, "e5_multi_context_attention.py"),
        "args": ["--scene", SCENE,
                 "--obj1",  os.path.join(OBJ_DIR, "obj_bicycle.png"),
                 "--obj2",  os.path.join(OBJ_DIR, "obj_vase.png"),
                 "--out_dir", "results/e5_multi_context_attn"],
    },
    {
        "id":   "e6",
        "name": "Drift Measurement  (~25 min)",
        "script": os.path.join(BASE, "e6_drift_measurement.py"),
        "args": ["--scene", SCENE, "--obj_dir", OBJ_DIR,
                 "--out_dir", "results/e6_drift"],
    },
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only",   nargs="+", metavar="ID",
                   help="Run only these experiment IDs (e.g. e1 e4)")
    p.add_argument("--skip",   nargs="+", metavar="ID",
                   help="Skip these experiment IDs")
    p.add_argument("--device", default=DEVICE)
    return p.parse_args()


def check_prerequisites():
    missing = []
    if not os.path.isfile(SCENE):
        missing.append(SCENE)
    if not os.path.isdir(OBJ_DIR):
        missing.append(OBJ_DIR)
    if missing:
        print("ERROR: Missing prerequisites:")
        for m in missing:
            print(f"  {m}")
        print("\nRun first:")
        print("  python KontextPipeline/run.py --steps 1,2 --out_dir results/kontext_setup")
        sys.exit(1)


def run_experiment(exp: dict, device: str) -> bool:
    cmd = [sys.executable, exp["script"]] + exp["args"] + ["--device", device]
    print(f"\n{'='*60}")
    print(f"  {exp['id'].upper()}: {exp['name']}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[FAIL] {exp['id']} exited with code {result.returncode}")
        return False
    return True


def main():
    args = parse_args()
    check_prerequisites()

    only = {x.lower() for x in args.only} if args.only else None
    skip = {x.lower() for x in args.skip} if args.skip else set()

    to_run = [
        e for e in EXPERIMENTS
        if (only is None or e["id"] in only) and e["id"] not in skip
    ]

    if not to_run:
        print("No experiments selected.")
        sys.exit(0)

    print(f"Running {len(to_run)} experiment(s): {[e['id'] for e in to_run]}")

    failed = []
    for exp in to_run:
        ok = run_experiment(exp, args.device)
        if not ok:
            failed.append(exp["id"])
            print(f"Stopping — {exp['id']} failed.")
            break

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED: {failed}")
    else:
        print(f"All done. Results in results/e*/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
