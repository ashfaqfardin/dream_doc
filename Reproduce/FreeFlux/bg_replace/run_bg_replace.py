"""
FreeFlux background-replace image editing for FLUX.1-dev.

Based on: https://github.com/wtybest/FreeFlux (ICCV 2025)

Replaces the background of a generated scene image while keeping the foreground
object intact. The pipeline is two passes:

  Pass 1 — generate original image:
    auto  mode: reasoning attention (stores cross-attention) → auto fg mask via SAM2
    manual mode: standard attention → user-provided SAM2 click points

  Pass 2 — bg_replace:
    bg_replace attention + shared latents + SAM2 mask indices → [source, edited] images

SAM2 is REQUIRED for both modes (auto: auto-samples click points; manual: uses yours).
Install it with:
    pip install git+https://github.com/facebookresearch/sam2

Usage — auto fg mask (fully automatic, no click points needed)
-------------------------------------------------------------
python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \\
    --hf_token "$HF_TOKEN" \\
    --source_prompt "A sports car on the road" \\
    --target_prompt "A snowing day" \\
    --foreground_word "car" \\
    --fg_mask_mode auto \\
    --n_steps 50 --seed 2 \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images

Usage — manual click points
---------------------------
python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \\
    --hf_token "$HF_TOKEN" \\
    --source_prompt "A sports car on the road" \\
    --target_prompt "A snowing day" \\
    --foreground_word "car" \\
    --fg_mask_mode manual \\
    --point_list "[[500,500],[700,550],[350,280]]" \\
    --label_list "[1,1,0]" \\
    --n_steps 50 --seed 2 \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images

Usage — object moving (non-zero shift)
--------------------------------------
python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \\
    --hf_token "$HF_TOKEN" \\
    --source_prompt "A sports car on the road" \\
    --target_prompt "A sports car on the road" \\
    --foreground_word "car" \\
    --shift 10 10 \\
    --n_steps 50 --seed 2 \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images

Usage — config file
-------------------
python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \\
    --hf_token "$HF_TOKEN" \\
    --config prompts/reproduce_freeflux_bg_replace.json \\
    --device cuda --cpu_offload \\
    --cache_dir ./models --save_images
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

# Repo root: go up three levels from Reproduce/FreeFlux/bg_replace/
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffusers.models.attention_processor import FluxAttnProcessor2_0
from diffusers.utils.torch_utils import randn_tensor

from Reproduce.FreeFlux.bg_replace.pipeline_flux_bg_replace import FluxPipeline
from Reproduce.FreeFlux.bg_replace.bg_replace_attn_utils import (
    register_bg_replace_attention_control,
    register_reasoning_attention_control,
)
from Reproduce.FreeFlux.bg_replace.transformer_flux_bg_replace import (
    FluxTransformer2DModel as FluxTransformer2DModel_BgReplace,
)
from Reproduce.FreeFlux.utils import (
    get_index_from_subject,
    get_mask_by_click,
    resize_and_get_coordinates,
    derive_fg_mask_from_attn,
    remove_small_white_regions,
    sample_indices,
    erode_image,
    validate_fg_mask_mode,
)

# Paper defaults for bg_replace — all 57 layers, steps 0-44.
_PROCESSOR_ARGS = {
    "start_step": 0,
    "start_layer": 0,
    "layer_idx": list(range(0, 57)),
    "step_idx": list(range(0, 45)),
    "total_layers": 57,
    "total_steps": 50,
}


def load_bg_replace_pipeline(model_path: str, hf_token: str,
                              device: str = "cuda", cpu_offload: bool = False,
                              cache_dir: str = "./models"):
    """Load FluxPipeline and replace transformer with the bg-replace custom variant."""
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )

    custom_transformer = FluxTransformer2DModel_BgReplace.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )
    pipe.transformer = custom_transformer

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe


def _register_standard_attention(pipe):
    pipe.transformer.set_attn_processor(FluxAttnProcessor2_0())
    print(f"Model {pipe.transformer.__class__.__name__} is registered attention processor: FluxAttnProcessor2_0")


def _make_shared_latents(pipe, n_prompts: int, height: int, width: int, seed: int, device: str):
    """Create random packed latents shared across all prompts in the batch."""
    generator = torch.Generator(device=device).manual_seed(seed)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    h_lat = height // pipe.vae_scale_factor
    w_lat = width // pipe.vae_scale_factor
    shape = (1, num_channels_latents, h_lat, w_lat)
    raw = randn_tensor(shape, generator=generator, device=pipe._execution_device, dtype=torch.bfloat16)
    raw = raw.expand(n_prompts, -1, -1, -1).clone()
    latents = pipe._pack_latents(raw, n_prompts, num_channels_latents, h_lat, w_lat)
    return latents


def _clamp_shift(predicted_mask, target_shape, shift):
    """Clamp shift so no fg pixel is pushed outside the latent grid.

    resize_and_get_coordinates() raises ValueError if any fg coordinate after
    shifting falls outside [0, max_row) x [0, max_col). This function pre-computes
    the tightest valid shift and warns when clamping is applied.
    """
    arr = np.array(predicted_mask, dtype=np.float32)
    resized = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = (resized > 0.5).astype(np.uint8)
    rows, cols = np.where(binary == 1)
    if len(rows) == 0:
        return shift  # empty mask — shift is irrelevant

    max_row_grid, max_col_grid = target_shape
    lo_r, hi_r = -int(rows.min()), (max_row_grid - 1) - int(rows.max())
    lo_c, hi_c = -int(cols.min()), (max_col_grid - 1) - int(cols.max())

    clamped = (max(lo_r, min(int(shift[0]), hi_r)), max(lo_c, min(int(shift[1]), hi_c)))
    if clamped != (int(shift[0]), int(shift[1])):
        print(
            f"  Warning: shift {shift} would push fg mask out of the "
            f"{max_row_grid}x{max_col_grid} latent grid — clamped to {clamped}."
        )
    return clamped


@torch.no_grad()
def run_bg_replace(pipe, source_prompt: str, target_prompt: str, foreground_word: str,
                   fg_mask_mode: str = "auto", point_list=None, label_list=None,
                   shift=(0, 0), n_steps: int = 50, guidance_scale: float = 3.5,
                   height: int = 1024, width: int = 1024,
                   max_sequence_length: int = 512, seed: int = 0,
                   device: str = "cuda", sam2_model: str = "facebook/sam2-hiera-large"):
    """
    Full bg_replace workflow.

    fg_mask_mode:
      'auto'   — derives fg mask from cross-attention maps then refines with SAM2 auto-points
      'manual' — uses standard attention; SAM2 mask from user-provided point_list/label_list

    Returns (source_image, edited_image) as PIL Images.
    """
    auto_gen_fg_mask = validate_fg_mask_mode(fg_mask_mode)

    # Load SAM2 predictor (lazy import so non-SAM2 users get a clear error, not a crash at startup)
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor.from_pretrained(sam2_model)
    except ImportError:
        raise ImportError(
            "SAM2 is required for bg_replace. Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/sam2"
        )

    prompts = [source_prompt, target_prompt]

    # ── Pass 1: generate original image, collect fg mask ────────────────────
    if auto_gen_fg_mask:
        print("  [Pass 1/2] Reasoning pass (auto fg mask mode)...")
        subject_idx_list = get_index_from_subject(pipe, source_prompt, foreground_word)
        print(f"            T5 token indices for '{foreground_word}': {subject_idx_list}")
        register_reasoning_attention_control(pipe, **_PROCESSOR_ARGS)
    else:
        print("  [Pass 1/2] Standard attention (manual fg mask mode)...")
        _register_standard_attention(pipe)

    ori_result = pipe(
        source_prompt,
        height=height, width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=n_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    ori_image = ori_result.images[0]

    # ── Derive fg mask ───────────────────────────────────────────────────────
    image_np = np.array(ori_image.convert("RGB"))
    h16 = height // 16
    w16 = width // 16

    if auto_gen_fg_mask:
        print("  Deriving foreground mask from cross-attention...")
        global_store = pipe.transformer.single_transformer_blocks[1].attn.processor.global_store
        derived_mask = 255 * derive_fg_mask_from_attn(
            global_store, subject_idx_list,
            mask_threshold=0.3, height=h16, width=w16
        ).numpy().astype(np.uint8)

        mask_for_point = remove_small_white_regions(derived_mask)
        bg_index    = sample_indices(mask_for_point, value=0, num_samples=4)
        fg_eroded   = erode_image(mask_for_point)
        fg_index    = sample_indices(fg_eroded, value=1, num_samples=4, farthest=True)

        click_points = fg_index + bg_index
        click_labels = [1] * len(fg_index) + [0] * len(bg_index)
        # Use fg-only points (paper default)
        click_points = fg_index
        click_labels = [1] * len(fg_index)
    else:
        if point_list is None or label_list is None:
            raise ValueError(
                "Manual fg_mask_mode requires --point_list and --label_list. "
                "Example: --point_list '[[500,500],[700,550]]' --label_list '[1,0]'"
            )
        click_points = point_list
        click_labels = label_list

    print(f"  Running SAM2 with {len(click_points)} click point(s)...")
    predicted_mask = get_mask_by_click(
        image_np.astype(np.uint8),
        predictor=predictor,
        point_list=click_points,
        lable_list=click_labels,
        device=device,
    )

    shift = _clamp_shift(predicted_mask, [w16, h16], shift)
    source_idx_list, target_idx_list, inpainted_idx_list, remain_idx_list = (
        resize_and_get_coordinates(predicted_mask, [w16, h16], shift=shift)
    )
    print(f"  Mask: {len(source_idx_list)} fg patches, {len(remain_idx_list)} bg patches.")

    # ── Pass 2: bg_replace attention → edited image ──────────────────────────
    print("  [Pass 2/2] Generating bg-replaced image...")
    latents = _make_shared_latents(pipe, len(prompts), height, width, seed, device)

    register_bg_replace_attention_control(pipe, **_PROCESSOR_ARGS)
    result = pipe(
        prompts,
        height=height, width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=n_steps,
        max_sequence_length=max_sequence_length,
        latents=latents,
        src_tar_inpaint_idx_list=[source_idx_list, target_idx_list, inpainted_idx_list, remain_idx_list],
    )

    # result.images[0] = source with original bg, result.images[1] = target with new bg
    return ori_image, result.images[1]


def run_single(pipe, cfg: dict, out_dir: str, save_images: bool):
    """Run one experiment from a config dict and optionally save results."""
    name            = cfg["name"]
    source_prompt   = cfg["source_prompt"]
    target_prompt   = cfg["target_prompt"]
    foreground_word = cfg["foreground_word"]
    fg_mask_mode    = cfg.get("fg_mask_mode", "auto")
    point_list      = cfg.get("point_list", None)
    label_list      = cfg.get("label_list", None)
    shift           = tuple(cfg.get("shift", [0, 0]))
    n_steps         = cfg.get("n_steps", 50)
    guidance_scale  = cfg.get("guidance_scale", 3.5)
    height          = cfg.get("height", 1024)
    width           = cfg.get("width", 1024)
    max_seq_len     = cfg.get("max_sequence_length", 512)
    seed            = cfg.get("seed", 0)
    sam2_model      = cfg.get("sam2_model", "facebook/sam2-hiera-large")

    print(f"\n[{name}]  src='{source_prompt}'  tgt='{target_prompt}'  fg='{foreground_word}'")
    print(f"         fg_mode={fg_mask_mode}  steps={n_steps}  guidance={guidance_scale}  seed={seed}")

    src_img, edited_img = run_bg_replace(
        pipe,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        foreground_word=foreground_word,
        fg_mask_mode=fg_mask_mode,
        point_list=point_list,
        label_list=label_list,
        shift=shift,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_seq_len,
        seed=seed,
        sam2_model=sam2_model,
    )

    if save_images:
        run_dir = os.path.join(out_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        src_img.save(os.path.join(run_dir, "source.png"))
        edited_img.save(os.path.join(run_dir, "edited.png"))
        print(f"  Saved → {run_dir}/")

    return src_img, edited_img


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="FreeFlux background-replace image editing (ICCV 2025)"
    )

    # --- Config-file mode ---
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON run-config file.")

    # --- Single-run mode ---
    parser.add_argument("--source_prompt",   type=str, default=None,
                        help="Text description of the source scene (required without --config)")
    parser.add_argument("--target_prompt",   type=str, default=None,
                        help="Text description of the new background (required without --config)")
    parser.add_argument("--foreground_word", type=str, default=None,
                        help="Word describing the foreground object to keep (required without --config)")

    parser.add_argument("--fg_mask_mode", type=str, default="auto",
                        choices=["auto", "manual"],
                        help="'auto': derive fg mask from cross-attention + SAM2 auto-points; "
                             "'manual': use standard attention + provided click points")
    parser.add_argument("--point_list", type=str, default=None,
                        help="JSON list of [x,y] click coords for manual mode. "
                             "Example: '[[500,500],[700,550]]'")
    parser.add_argument("--label_list", type=str, default=None,
                        help="JSON list of SAM2 labels (1=fg, 0=bg) for manual mode. "
                             "Example: '[1,0]'")
    parser.add_argument("--shift", type=int, nargs=2, default=[0, 0],
                        metavar=("DX", "DY"),
                        help="Spatial shift for object-moving mode (default: 0 0)")
    parser.add_argument("--n_steps",             type=int,   default=50)
    parser.add_argument("--guidance_scale",      type=float, default=3.5)
    parser.add_argument("--height",              type=int,   default=1024)
    parser.add_argument("--width",               type=int,   default=1024)
    parser.add_argument("--max_sequence_length", type=int,   default=512)
    parser.add_argument("--seed",                type=int,   default=2,
                        help="Random seed (default 2, matches paper example)")
    parser.add_argument("--sam2_model", type=str, default="facebook/sam2-hiera-large",
                        help="SAM2 model checkpoint (HuggingFace ID)")

    # --- Infrastructure ---
    parser.add_argument("--model_path",  type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token",   type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--cache_dir",  type=str, default="./models")
    parser.add_argument("--out_dir",    type=str, default="results/freeflux/bg_replace")
    parser.add_argument("--save_images", action="store_true")

    args = parser.parse_args()

    if args.config is None:
        if args.source_prompt is None or args.target_prompt is None or args.foreground_word is None:
            parser.error(
                "--source_prompt, --target_prompt, and --foreground_word are required unless --config is provided."
            )
        if args.fg_mask_mode == "manual" and (args.point_list is None or args.label_list is None):
            parser.error("--point_list and --label_list are required in manual fg_mask_mode.")

    return args


def main():
    args = parse_args()

    print(f"[FreeFlux bg-replace] Loading {args.model_path} + custom transformer...")
    pipe = load_bg_replace_pipeline(
        args.model_path, args.hf_token,
        device=args.device, cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
    )
    print("[FreeFlux bg-replace] Model loaded.")

    if args.config:
        runs = load_config(args.config)
        print(f"\n[FreeFlux bg-replace] Running {len(runs)} experiment(s) from {args.config}")
        for cfg in runs:
            run_single(pipe, cfg, out_dir=args.out_dir, save_images=args.save_images)
        print(f"\n[FreeFlux bg-replace] All runs complete.")
        if args.save_images:
            print(f"  Results saved to {args.out_dir}/")
    else:
        # Parse JSON click-point args
        point_list = json.loads(args.point_list) if args.point_list else None
        label_list = json.loads(args.label_list) if args.label_list else None

        cfg = {
            "name":               "output",
            "source_prompt":      args.source_prompt,
            "target_prompt":      args.target_prompt,
            "foreground_word":    args.foreground_word,
            "fg_mask_mode":       args.fg_mask_mode,
            "point_list":         point_list,
            "label_list":         label_list,
            "shift":              args.shift,
            "n_steps":            args.n_steps,
            "guidance_scale":     args.guidance_scale,
            "height":             args.height,
            "width":              args.width,
            "max_sequence_length": args.max_sequence_length,
            "seed":               args.seed,
            "sam2_model":         args.sam2_model,
        }
        run_single(pipe, cfg, out_dir=args.out_dir, save_images=args.save_images)


if __name__ == "__main__":
    main()
