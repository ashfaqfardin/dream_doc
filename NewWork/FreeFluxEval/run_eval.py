"""
FreeFlux downstream evaluation runner.

Tests FreeFlux non-rigid attention injection across 100 prompts spanning
10 task categories that go beyond the paper's three target tasks:
  color_change, texture_change, weather_change, time_of_day, style_transfer,
  object_replacement, attribute_change, action_change, seasonal_change, lighting_change

Supports running FLUX.1-dev and FLUX.1-schnell side by side.
Results go to {out_dir}/{model}/  so comparisons are easy.

PowerShell usage (use backtick for line continuation):
  # Smoke-test 5 prompts on both models
  python NewWork/FreeFluxEval/run_eval.py `
      --hf_token $env:HF_TOKEN --limit 5 --save_images

  # One category, dev only
  python NewWork/FreeFluxEval/run_eval.py `
      --hf_token $env:HF_TOKEN --models dev --category color_change --save_images

  # Full 100 on both models, resumable
  python NewWork/FreeFluxEval/run_eval.py `
      --hf_token $env:HF_TOKEN --models dev schnell --resume --save_images
"""

import argparse
import builtins
import contextlib
import json
import os
import sys
import time


@contextlib.contextmanager
def _silence_prints():
    """Suppress all print() calls — catches prints that bypass sys.stdout."""
    _real = builtins.print
    builtins.print = lambda *a, **kw: None
    try:
        yield
    finally:
        builtins.print = _real

from tqdm import tqdm

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Reproduce.FreeFlux.non_rigid.run_non_rigid import (
    load_freeflux_pipeline,
    run_non_rigid_edit,
)

_ALL_CATEGORIES = [
    "color_change",
    "texture_change",
    "weather_change",
    "time_of_day",
    "style_transfer",
    "object_replacement",
    "attribute_change",
    "action_change",
    "seasonal_change",
    "lighting_change",
]

# Per-model inference defaults. schnell requires 0 guidance and runs in 4 steps.
_MODEL_CONFIGS = {
    "dev": {
        "model_path": "black-forest-labs/FLUX.1-dev",
        "n_steps":    28,
        "guidance":   3.5,
    },
    "schnell": {
        "model_path": "black-forest-labs/FLUX.1-schnell",
        "n_steps":    4,
        "guidance":   0.0,
    },
}


def load_config(config_path: str) -> list:
    with open(config_path) as f:
        data = json.load(f)
    global_defaults = data.get("global", {})
    runs = []
    for run in data["runs"]:
        merged = {**global_defaults, **run}
        if "name" not in merged:
            raise ValueError(f"Each run must have a 'name' field: {run}")
        runs.append(merged)
    return runs


def _run_one_model(pipe, runs, model_key, model_cfg, out_dir, save_images, resume):
    model_out = os.path.join(out_dir, model_key)

    # Resume: drop runs whose images are already on disk
    if resume:
        pending, skipped = [], 0
        for r in runs:
            d = os.path.join(model_out, r["name"])
            if (os.path.exists(os.path.join(d, "source.png")) and
                    os.path.exists(os.path.join(d, "edited.png"))):
                skipped += 1
            else:
                pending.append(r)
        if skipped:
            print(f"  [{model_key}] resume: skipping {skipped} done, {len(pending)} remaining")
        runs = pending

    if not runs:
        print(f"  [{model_key}] nothing to run.")
        return []

    summary = []
    eval_start = time.time()

    bar = tqdm(runs, desc=f"[{model_key}]", unit="item",
               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                          "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    for cfg in bar:
        name          = cfg["name"]
        category      = cfg.get("category", "unknown")
        src_prompt    = cfg["source_prompt"]
        tgt_prompt    = cfg["target_prompt"]
        height        = cfg.get("height", 1024)
        width         = cfg.get("width", 1024)
        max_seq_len   = cfg.get("max_sequence_length", 512)
        seed          = cfg.get("seed", 42)
        # Steps and guidance always come from the model config —
        # the JSON defaults are dev-specific and wrong for schnell.
        n_steps        = model_cfg["n_steps"]
        guidance_scale = model_cfg["guidance"]

        bar.set_postfix(run=name, cat=category, refresh=False)

        run_dir = os.path.join(model_out, name)
        if save_images:
            os.makedirs(run_dir, exist_ok=True)

        t0 = time.time()
        status, error_msg = "ok", ""
        try:
            with _silence_prints():
                src_img, edited_img = run_non_rigid_edit(
                    pipe,
                    source_prompt=src_prompt,
                    target_prompt=tgt_prompt,
                    n_steps=n_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    max_sequence_length=max_seq_len,
                    seed=seed,
                )
            if save_images:
                src_img.save(os.path.join(run_dir, "source.png"))
                edited_img.save(os.path.join(run_dir, "edited.png"))
        except Exception as e:
            status, error_msg = "error", str(e)
            tqdm.write(f"  ERROR [{name}]: {e}")

        summary.append({
            "name":          name,
            "category":      category,
            "source_prompt": src_prompt,
            "target_prompt": tgt_prompt,
            "status":        status,
            "error":         error_msg,
            "duration_s":    round(time.time() - t0, 1),
        })

    total_s   = time.time() - eval_start
    ok_count  = sum(1 for r in summary if r["status"] == "ok")
    err_count = len(summary) - ok_count

    os.makedirs(model_out, exist_ok=True)
    summary_path = os.path.join(model_out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model":           model_cfg["model_path"],
            "total":           len(summary),
            "ok":              ok_count,
            "errors":          err_count,
            "total_duration_s": round(total_s, 1),
            "runs":            summary,
        }, f, indent=2)

    print(f"\n  [{model_key}] done — OK: {ok_count}  Errors: {err_count}  "
          f"Total: {total_s/60:.1f} min  → {summary_path}")

    if err_count:
        for r in summary:
            if r["status"] == "error":
                print(f"    FAILED {r['name']}: {r['error']}")

    return summary


def run_eval(args):
    # Strip lone backslashes that sneak in from bash-style copy-paste
    sys.argv = [a for a in sys.argv if a != "\\"]

    runs = load_config(args.config)

    if args.category:
        runs = [r for r in runs if r.get("category") == args.category]
        print(f"Category filter '{args.category}': {len(runs)} runs")

    if args.limit:
        runs = runs[:args.limit]
        print(f"Limit: first {args.limit} runs")

    if not runs:
        print("No runs matched the filters.")
        return

    for model_key in args.models:
        if model_key not in _MODEL_CONFIGS:
            # Allow passing a full HF path directly
            model_cfg = {"model_path": model_key, "n_steps": 28, "guidance": 3.5}
        else:
            model_cfg = _MODEL_CONFIGS[model_key]

        print(f"\n{'='*60}")
        print(f"Model: {model_cfg['model_path']}  "
              f"steps={model_cfg['n_steps']}  guidance={model_cfg['guidance']}")
        print(f"{'='*60}")
        print(f"Loading model (cache: {args.cache_dir}) ...")

        pipe = load_freeflux_pipeline(
            model_cfg["model_path"], args.hf_token,
            device=args.device, cpu_offload=args.cpu_offload,
            cache_dir=args.cache_dir,
        )
        # Suppress diffusers' per-step bar so the outer run bar stays visible.
        pipe.set_progress_bar_config(disable=True)

        _run_one_model(pipe, runs, model_key, model_cfg,
                       out_dir=args.out_dir,
                       save_images=args.save_images,
                       resume=args.resume)

        # Free VRAM before loading the next model
        del pipe
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def parse_args():
    # Strip lone backslashes before argparse sees them (bash copy-paste on Windows)
    sys.argv = [a for a in sys.argv if a != "\\"]

    parser = argparse.ArgumentParser(
        description="FreeFlux downstream task evaluation (100 prompts, 10 categories)"
    )
    parser.add_argument("--config", type=str,
                        default="prompts/freeflux_downstream_eval.json")
    parser.add_argument("--models", nargs="+", default=["dev", "schnell"],
                        help="Which models to run: dev, schnell, or a full HF path. "
                             "Default: both dev and schnell")
    parser.add_argument("--category", type=str, default=None,
                        choices=_ALL_CATEGORIES,
                        help="Run only this category (default: all 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N prompts (smoke testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose output images already exist")
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--out_dir", type=str,
                        default="results/freeflux/downstream_eval",
                        help="Root output dir. Each model gets its own sub-folder.")
    parser.add_argument("--save_images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
