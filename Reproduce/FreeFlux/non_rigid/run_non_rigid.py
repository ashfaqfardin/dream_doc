"""
FreeFlux non-rigid image editing for FLUX.1-dev.

Based on: https://github.com/wtybest/FreeFlux (ICCV 2025)

Performs training-free non-rigid editing (pose changes, deformations) by sharing
image-token keys/values from the source branch into the target branch during a
SINGLE shared-latent denoising pass. No real source image or DDIM inversion needed.

Both [source_prompt, target_prompt] start from the SAME random latents (seeded),
and mutual self-attention at selected layers/steps copies structure from source to
target while the target follows its own prompt.

Usage — single run
------------------
python Reproduce/FreeFlux/non_rigid/run_non_rigid.py \\
    --hf_token "$HF_TOKEN" \\
    --source_prompt "a bird perched on a branch" \\
    --target_prompt "a bird flying from the branch" \\
    --n_steps 50 --seed 2 \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images

Usage — config file
-------------------
python Reproduce/FreeFlux/non_rigid/run_non_rigid.py \\
    --hf_token "$HF_TOKEN" \\
    --config prompts/reproduce_freeflux.json \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image

# Add repo root to sys.path so 'Reproduce' is importable as a package.
# Script lives at Reproduce/FreeFlux/non_rigid/run_non_rigid.py, so go up three levels.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffusers.utils.torch_utils import randn_tensor

from Reproduce.FreeFlux.non_rigid.pipeline_flux_non_rigid import FluxPipeline
from Reproduce.FreeFlux.non_rigid.non_rigid_attn_utils import register_non_rigid_attention_control


# Paper processor args from the original non_rigid.ipynb notebook.
# layer_idx selects which of the 57 transformer layers apply attention sharing.
_PROCESSOR_ARGS = {
    "start_step":   0,
    "start_layer":  0,
    "layer_idx":    [0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56],
    "step_idx":     list(range(0, 50)),
    "total_layers": 57,
    "total_steps":  50,
}


def load_freeflux_pipeline(model_path: str, hf_token: str,
                            device: str = "cuda", cpu_offload: bool = False,
                            cache_dir: str = "./models"):
    """Load the non-rigid FluxPipeline. Attention is registered per-run in run_non_rigid_edit."""
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe


@torch.no_grad()
def run_non_rigid_edit(pipe, source_prompt: str, target_prompt: str,
                       n_steps: int = 50, guidance_scale: float = 3.5,
                       height: int = 1024, width: int = 1024,
                       max_sequence_length: int = 512, seed: int = 2):
    """Generate source and non-rigidly edited target images from shared latents.

    Both prompts start from the SAME seeded random latents. Mutual self-attention
    control at selected layers copies image-token keys/values from the source branch
    into the target branch, transferring spatial structure while the target follows
    its own prompt.

    Returns (source_image, edited_image) as PIL Images.
    """
    prompts = [source_prompt, target_prompt]

    # Create shared random latents: same noise for both source and target.
    # Shape before packing: (1, 16, H//8, W//8) — matches pipeline's prepare_latents.
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    shape = (1, 16, height // 8, width // 8)
    latents = randn_tensor(shape, generator=generator,
                           device=pipe._execution_device, dtype=torch.bfloat16)
    latents = latents.expand(len(prompts), -1, -1, -1).clone()
    latents = pipe._pack_latents(latents, len(prompts), 16, height // 8, width // 8)

    # Register non-rigid attention control (shared keys/values from src → tgt).
    register_non_rigid_attention_control(pipe, **_PROCESSOR_ARGS)

    result = pipe(
        prompts,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=n_steps,
        max_sequence_length=max_sequence_length,
        latents=latents,
        output_type="pil",
    )

    # images[0] = source (generated), images[1] = edited (target prompt + source structure)
    return result.images[0], result.images[1]


def run_single(pipe, cfg: dict, out_dir: str, save_images: bool):
    """Run one experiment defined by cfg dict and optionally save images."""
    name           = cfg["name"]
    source_prompt  = cfg["source_prompt"]
    target_prompt  = cfg["target_prompt"]
    n_steps        = cfg.get("n_steps", 50)
    guidance_scale = cfg.get("guidance_scale", 3.5)
    height         = cfg.get("height", 1024)
    width          = cfg.get("width", 1024)
    max_seq_len    = cfg.get("max_sequence_length", 512)
    seed           = cfg.get("seed", 2)

    print(f"\n[{name}]  source='{source_prompt}'  target='{target_prompt}'")
    print(f"         steps={n_steps}  guidance={guidance_scale}  seed={seed}  size={height}x{width}")

    img_src, img_edited = run_non_rigid_edit(
        pipe,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_seq_len,
        seed=seed,
    )

    if save_images:
        run_dir = os.path.join(out_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        img_src.save(os.path.join(run_dir, "source.png"))
        img_edited.save(os.path.join(run_dir, "edited.png"))
        print(f"  Saved → {run_dir}/")

    return img_src, img_edited


def load_config(config_path: str) -> list:
    """Load runs from a JSON config file.

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
        description="FreeFlux non-rigid image editing (ICCV 2025)"
    )

    # --- Config-file mode ---
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON run-config file. "
                             "When provided, --source_prompt/--target_prompt are not required.")

    # --- Single-run mode ---
    parser.add_argument("--source_prompt", type=str, default=None,
                        help="Text description of the source scene (required without --config)")
    parser.add_argument("--target_prompt", type=str, default=None,
                        help="Text description of the desired edit (required without --config)")
    parser.add_argument("--n_steps",             type=int,   default=50,
                        help="Denoising steps (default 50, matching paper)")
    parser.add_argument("--guidance_scale",      type=float, default=3.5)
    parser.add_argument("--height",              type=int,   default=1024)
    parser.add_argument("--width",               type=int,   default=1024)
    parser.add_argument("--max_sequence_length", type=int,   default=512,
                        help="T5 max token length (default 512)")
    parser.add_argument("--seed",                type=int,   default=2,
                        help="Random seed for shared latents (default 2)")

    # --- Infrastructure ---
    parser.add_argument("--model_path",  type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",   type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Offload weights to CPU between steps (saves VRAM)")
    parser.add_argument("--cache_dir",  type=str, default="./models")
    parser.add_argument("--out_dir",    type=str, default="results/freeflux/non_rigid",
                        help="Output directory for saved images")
    parser.add_argument("--save_images", action="store_true",
                        help="Write source.png and edited.png for each run")

    args = parser.parse_args()

    if args.config is None:
        if args.source_prompt is None or args.target_prompt is None:
            parser.error("--source_prompt and --target_prompt are required unless --config is provided.")

    return args


def main():
    args = parse_args()

    print(f"[FreeFlux] Loading {args.model_path}...")
    pipe = load_freeflux_pipeline(
        args.model_path, args.hf_token,
        device=args.device, cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
    )
    print("[FreeFlux] Model loaded.")

    if args.config:
        runs = load_config(args.config)
        print(f"\n[FreeFlux] Running {len(runs)} experiment(s) from {args.config}")
        for cfg in runs:
            run_single(pipe, cfg, out_dir=args.out_dir, save_images=args.save_images)
        print(f"\n[FreeFlux] All runs complete.")
        if args.save_images:
            print(f"  Results saved to {args.out_dir}/")
    else:
        cfg = {
            "name":               "output",
            "source_prompt":      args.source_prompt,
            "target_prompt":      args.target_prompt,
            "n_steps":            args.n_steps,
            "guidance_scale":     args.guidance_scale,
            "height":             args.height,
            "width":              args.width,
            "max_sequence_length": args.max_sequence_length,
            "seed":               args.seed,
        }
        run_single(pipe, cfg, out_dir=args.out_dir, save_images=args.save_images)


if __name__ == "__main__":
    main()
