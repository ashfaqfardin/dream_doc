"""
FluxSpace semantic image editing for FLUX.1-dev.

Based on: https://github.com/gemlab-vt/FluxSpace (Dalva et al., CVPR 2025)
Paper: arXiv:2412.09611

The method injects an edit direction into each MM-DiT block's attention:
  - edit_global_scale (0-1): shifts the pooled text embedding toward the edit attribute
  - edit_content_scale      : adds the attribute's orthogonal attention direction per-pixel

Usage — single run
------------------
python Reproduce/FluxSpace/run_fluxspace.py \\
    --hf_token "$HF_TOKEN" \\
    --prompt "portrait photo of a man" \\
    --edit_prompt "eyeglasses" \\
    --edit_global_scale 0.8 --edit_content_scale 5 --edit_start_iter 3 \\
    --n_steps 30 --seed 0 --device cuda --cpu_offload \\
    --cache_dir ./models --save_images

Usage — reproduce all paper runs from config file
-------------------------------------------------
python Reproduce/FluxSpace/run_fluxspace.py \\
    --hf_token "$HF_TOKEN" \\
    --config prompts/reproduce_fluxspace.json \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images

Requires upstream diffusers (pip install -U diffusers), same as FLUX.2-dev.
"""

import argparse
import json
import os
import sys

import torch

# Add repo root to sys.path so 'Reproduce' is importable as a package.
# Script lives at Reproduce/FluxSpace/run_fluxspace.py, so go up two levels.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Identify the local diffusers fork so it can be excluded when loading upstream diffusers.
# FluxSpace requires upstream diffusers (>=0.31.0) because it wraps block.forward() to
# receive joint_attention_kwargs — a calling convention the local fork does not use.
_LOCAL_DIFFUSERS_SRC = os.path.normpath(os.path.join(_REPO_ROOT, "src"))


def load_fluxspace_pipeline(model_path: str, hf_token: str,
                             device: str = "cuda", cpu_offload: bool = False,
                             cache_dir: str = "./models"):
    """
    Load GenerationPipelineSemantic using upstream diffusers.

    Uses the same sys.path-swap pattern as layer_bypass._load_with_upstream_diffusers:
    temporarily prioritise upstream diffusers so the loaded transformer calls blocks
    with joint_attention_kwargs (required by FluxSpace wrappers).
    On success, upstream diffusers stays in sys.modules for the pipeline lifetime.
    """
    sys_diff_dirs = [
        p for p in sys.path
        if p != _LOCAL_DIFFUSERS_SRC
        and os.path.isfile(os.path.join(p, "diffusers", "__init__.py"))
    ]
    if not sys_diff_dirs:
        raise RuntimeError(
            "Upstream diffusers not found. Install it with:\n"
            "  pip install -U diffusers"
        )

    # Snapshot current state
    saved_path = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items()
                  if k == "diffusers" or k.startswith("diffusers.")}

    # Build new sys.path: upstream first, local fork excluded
    other_paths = [p for p in sys.path
                   if p not in sys_diff_dirs
                   and not os.path.isfile(os.path.join(p, "diffusers", "__init__.py"))]
    sys.path[:] = sys_diff_dirs + other_paths
    for k in list(saved_mods):
        del sys.modules[k]

    try:
        # Import while upstream diffusers is active — module is cached for the lifetime of
        # this process, so the pipeline's lazy imports will always resolve to upstream.
        from Reproduce.FluxSpace.flux_semantic_pipeline import GenerationPipelineSemantic

        pipe = GenerationPipelineSemantic.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            token=hf_token,
            cache_dir=cache_dir,
        )

        # Replace transformer blocks with FluxSpace wrappers
        pipe.register_transformer_blocks()

        # Device placement must happen while upstream modules are still active
        # (enable_model_cpu_offload does lazy imports from diffusers.hooks)
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

    except Exception:
        # On failure: fully restore sys.path and sys.modules
        sys.path[:] = saved_path
        for k in list(sys.modules.keys()):
            if k == "diffusers" or k.startswith("diffusers."):
                del sys.modules[k]
        sys.modules.update(saved_mods)
        raise

    # On success: restore sys.path but keep upstream diffusers in sys.modules
    sys.path[:] = saved_path
    return pipe


@torch.no_grad()
def generate_fluxspace(pipe, prompt: str, edit_prompt: str, seed: int = 0,
                       edit_global_scale: float = 0.5,
                       edit_content_scale: float = 0.5,
                       edit_start_iter: int = 0,
                       edit_stop_iter: int = None,
                       attention_threshold: float = 0.5,
                       num_inference_steps: int = 30,
                       guidance_scale: float = 3.5,
                       height: int = 1024, width: int = 1024,
                       max_sequence_length: int = 256,
                       device: str = "cuda"):
    """Generate one image with FluxSpace semantic editing applied."""
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        edit_prompt=edit_prompt,
        generator=generator,
        edit_global_scale=edit_global_scale,
        edit_content_scale=edit_content_scale,
        edit_start_iter=edit_start_iter,
        edit_stop_iter=edit_stop_iter,
        attention_threshold=attention_threshold,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        output_type="pil",
    )
    return result.images[0]


def run_single(pipe, cfg: dict, out_dir: str, save_images: bool, device: str):
    """Run one experiment defined by cfg dict and optionally save images."""
    name             = cfg["name"]
    prompt           = cfg["prompt"]
    edit_prompt      = cfg["edit_prompt"]
    seed             = cfg.get("seed", 0)
    n_steps          = cfg.get("n_steps", 30)
    guidance_scale   = cfg.get("guidance_scale", 3.5)
    height           = cfg.get("height", 1024)
    width            = cfg.get("width", 1024)
    max_seq_len      = cfg.get("max_sequence_length", 256)
    global_scale     = cfg.get("edit_global_scale", 0.5)
    content_scale    = cfg.get("edit_content_scale", 0.5)
    start_iter       = cfg.get("edit_start_iter", 0)
    stop_iter        = cfg.get("edit_stop_iter", None)
    attn_thresh      = cfg.get("attention_threshold", 0.5)

    print(f"\n[{name}]  prompt='{prompt}'  edit='{edit_prompt}'")
    print(f"         λ_coarse={global_scale}  λ_fine={content_scale}  "
          f"start_iter={start_iter}  τ_m={attn_thresh}  steps={n_steps}  seed={seed}")

    print(f"  Generating original...")
    img_original = generate_fluxspace(
        pipe, prompt, edit_prompt=edit_prompt, seed=seed,
        edit_global_scale=0.0, edit_content_scale=0.0,
        edit_start_iter=n_steps, edit_stop_iter=n_steps,
        num_inference_steps=n_steps, guidance_scale=guidance_scale,
        height=height, width=width, max_sequence_length=max_seq_len, device=device,
    )

    print(f"  Generating edited...")
    img_edited = generate_fluxspace(
        pipe, prompt, edit_prompt=edit_prompt, seed=seed,
        edit_global_scale=global_scale, edit_content_scale=content_scale,
        edit_start_iter=start_iter, edit_stop_iter=stop_iter,
        attention_threshold=attn_thresh,
        num_inference_steps=n_steps, guidance_scale=guidance_scale,
        height=height, width=width, max_sequence_length=max_seq_len, device=device,
    )

    if save_images:
        run_dir = os.path.join(out_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        img_original.save(os.path.join(run_dir, "original.png"))
        img_edited.save(os.path.join(run_dir, "edited.png"))
        print(f"  Saved → {run_dir}/")

    return img_original, img_edited


def load_config(config_path: str) -> list[dict]:
    """
    Load runs from a JSON config file.

    The JSON must have a "runs" list. An optional "global" dict provides
    default values that each run inherits unless it overrides them.
    """
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="FluxSpace semantic image editing (Dalva et al., CVPR 2025)"
    )

    # --- Config-file mode ---
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON run-config file (e.g. prompts/reproduce_fluxspace.json). "
                             "When provided, --prompt and --edit_prompt are not required.")

    # --- Single-run mode (required when --config is not used) ---
    parser.add_argument("--prompt",      type=str, default=None,
                        help="Base generation prompt (required without --config)")
    parser.add_argument("--edit_prompt", type=str, default=None,
                        help="Attribute to add/modify (required without --config)")
    parser.add_argument("--edit_global_scale",   type=float, default=0.5,
                        help="λ_coarse: global embedding shift (0–1, default 0.5)")
    parser.add_argument("--edit_content_scale",  type=float, default=0.5,
                        help="λ_fine: per-block attention edit strength (default 0.5)")
    parser.add_argument("--edit_start_iter",     type=int,   default=0,
                        help="First denoising step to apply edit (default 0)")
    parser.add_argument("--edit_stop_iter",      type=int,   default=None,
                        help="Last denoising step to apply edit (default: all steps)")
    parser.add_argument("--attention_threshold", type=float, default=0.5,
                        help="τ_m: cross-attention mask threshold (default 0.5)")
    parser.add_argument("--n_steps",             type=int,   default=30,
                        help="Denoising steps (default 30, matches paper)")
    parser.add_argument("--guidance_scale",      type=float, default=3.5)
    parser.add_argument("--height",              type=int,   default=1024)
    parser.add_argument("--width",               type=int,   default=1024)
    parser.add_argument("--max_sequence_length", type=int,   default=256,
                        help="T5 max token length (default 256, matches demo)")
    parser.add_argument("--seed",                type=int,   default=0)

    # --- Infrastructure (shared between both modes) ---
    parser.add_argument("--model_path",  type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",   type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Offload weights to CPU between steps (saves VRAM)")
    parser.add_argument("--cache_dir",  type=str, default="./models")
    parser.add_argument("--out_dir",    type=str, default="results/fluxspace",
                        help="Output directory for saved images (default: results/fluxspace)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save original.png and edited.png for each run")

    args = parser.parse_args()

    # Validate: without --config, --prompt and --edit_prompt are required
    if args.config is None:
        if args.prompt is None or args.edit_prompt is None:
            parser.error("--prompt and --edit_prompt are required unless --config is provided.")

    return args


def main():
    args = parse_args()

    print("[FluxSpace] Loading FLUX.1-dev via upstream diffusers...")
    pipe = load_fluxspace_pipeline(
        args.model_path, args.hf_token,
        device=args.device, cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
    )
    print("[FluxSpace] Model loaded. Transformer blocks wrapped.")

    if args.config:
        # --- Config-file mode: run all experiments from the JSON ---
        runs = load_config(args.config)
        print(f"\n[FluxSpace] Running {len(runs)} experiment(s) from {args.config}")
        for cfg in runs:
            # CLI infrastructure flags override config globals for model/device/paths,
            # but per-run generation settings come from the config.
            run_single(pipe, cfg,
                       out_dir=args.out_dir,
                       save_images=args.save_images,
                       device=args.device)
        print(f"\n[FluxSpace] All runs complete.")
        if args.save_images:
            print(f"  Results saved to {args.out_dir}/")
    else:
        # --- Single-run mode ---
        cfg = {
            "name": f"seed{args.seed}",
            "prompt": args.prompt,
            "edit_prompt": args.edit_prompt,
            "seed": args.seed,
            "n_steps": args.n_steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "max_sequence_length": args.max_sequence_length,
            "edit_global_scale": args.edit_global_scale,
            "edit_content_scale": args.edit_content_scale,
            "edit_start_iter": args.edit_start_iter,
            "edit_stop_iter": args.edit_stop_iter,
            "attention_threshold": args.attention_threshold,
        }
        run_single(pipe, cfg,
                   out_dir=args.out_dir,
                   save_images=args.save_images,
                   device=args.device)


if __name__ == "__main__":
    main()
