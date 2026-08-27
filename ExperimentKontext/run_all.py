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
    {
        "id":   "e7",
        "name": "Stitched Multi-Object Context  (~8 min)",
        "script": os.path.join(BASE, "e7_stitched_context.py"),
        "args": ["--scene", SCENE, "--obj_dir", OBJ_DIR,
                 "--out_dir", "results/e7_stitched_context"],
    },
    {
        "id":   "e8",
        "name": "Object K/V Attention Concatenation  (~14 min)",
        "script": os.path.join(BASE, "e8_obj_attn_concat.py"),
        "args": ["--scene", SCENE, "--obj_dir", OBJ_DIR,
                 "--out_dir", "results/e8_obj_attn_concat"],
    },
    {
        "id":   "e9",
        "name": "Self-Localizing Incremental Editing  (~40 min)",
        "script": os.path.join(BASE, "e9_self_localizing_incremental.py"),
        "args": ["--scene", SCENE, "--obj_dir", OBJ_DIR,
                 "--out_dir", "results/e9_self_localizing",
                 "--proposal_steps", "12",
                 "--steps", "28",
                 "--k_scale", "1.5"],
    },
    {
        "id":   "e10",
        "name": "Scene-Aware Adaptive Placement  (~2 hr)",
        "script": os.path.join(BASE, "e10_scene_aware_adaptive_placement.py"),
        "args": ["--scene", SCENE, "--obj_dir", OBJ_DIR,
                 "--out_dir", "results/e10_scene_aware_adaptive",
                 "--proposal_steps", "12",
                 "--steps", "28",
                 "--k_scale", "1.5",
                 "--placement_candidates", "1"],
    },
    {
        "id":   "e11",
        "name": "Scene-First Planner -> Pose -> Finalizer  (~3 hr)",
        "script": os.path.join(BASE, "e11_scene_first_planner_pose_finalizer.py"),
        "args": ["--scene", SCENE, "--obj_dir", OBJ_DIR,
                 "--out_dir", "results/e11_scene_first_planner",
                 "--planner_steps", "10",
                 "--pose_steps", "16",
                 "--final_steps", "28",
                 "--k_scale", "1.5",
                 "--placement_candidates", "4"],
    },
    {
        "id":   "e12",
        "name": "ControlNet Pose + Kontext Finalizer  (~4 hr)",
        "script": os.path.join(BASE, "e12_controlnet_pose_kontext_finalizer.py"),
        "args": ["--scene", SCENE, "--obj_dir", OBJ_DIR,
                 "--out_dir", "results/e12_controlnet_pose_kontext",
                 "--planner_steps", "10",
                 "--pose_steps", "24",
                 "--final_steps", "28",
                 "--placement_candidates", "4",
                 "--cpu_offload"],
    },
    {
        "id":   "e14",
        "name": "Generic Kontext Reference Replacer  (~5 min)",
        "script": os.path.join(BASE, "e14_generic_kontext_reference_replacer.py"),
        "args": ["--input_image",     SCENE,
                 "--reference_image", os.path.join(OBJ_DIR, "obj_bicycle.png"),
                 "--object_name",     "bicycle",
                 "--out_dir",         "results/e14_reference_replacer",
                 "--auto_detect"],
    },
    {
        "id":   "e17",
        "name": "Direct Reference Insertion",
        "script": os.path.join(BASE, "e17_direct_reference_insertion.py"),
        "args": ["--base_image",   SCENE,
                 "--objects_json", os.path.join(BASE, "e15_objects.json"),
                 "--out_dir",      "results/e17_direct_reference",
                 "--cpu_offload"],
    },
    {
        "id":   "e16",
        "name": "Reference Composite Harmonizer  (~2 hr)",
        "script": os.path.join(BASE, "e16_reference_composite_harmonizer.py"),
        "args": ["--base_image",   SCENE,
                 "--objects_json", os.path.join(BASE, "e15_objects.json"),
                 "--out_dir",      "results/e16_reference_composite",
                 "--cpu_offload"],
    },
    {
        "id":   "e18",
        "name": "Kontext Architecture Lab  (~1 hr)",
        "script": os.path.join(BASE, "e18_kontext_architecture_lab.py"),
        "args": ["--scene",       SCENE,
                 "--reference",   os.path.join(OBJ_DIR, "obj_bicycle.png"),
                 "--object_name", "bicycle",
                 "--out_dir",     "results/e18_architecture_lab"],
    },
    {
        "id":   "e15",
        "name": "Generic Place-Then-Replace Pipeline  (~2 hr)",
        "script": os.path.join(BASE, "e15_generic_place_then_replace_pipeline.py"),
        "args": ["--base_image",       SCENE,
                 "--objects_json",     os.path.join(BASE, "e15_objects.json"),
                 "--out_dir",          "results/e15_place_then_replace",
                 "--placement_candidates", "1",
                 "--placement_steps",  "16",
                 "--replace_steps",    "28",
                 "--refine_steps",     "20",
                 "--mask_backend",     "sam2",
                 "--cpu_offload"],
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
