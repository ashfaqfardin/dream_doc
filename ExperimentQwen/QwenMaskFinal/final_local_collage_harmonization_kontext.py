"""FLUX.1 Kontext variant of the final local collage harmonization baseline.

This script deliberately reuses the complete deterministic and evaluation
pipeline from ``final_local_collage_harmonization.py``.  Only the local
generative harmonizer is replaced:

    QwenImageEditInpaintPipeline -> FluxKontextInpaintPipeline

Keeping the cutouts, placement, masks, crop policy, hard RGB composition, and
metrics unchanged makes the two runs a controlled backbone comparison.

Example
-------
python final_local_collage_harmonization_kontext.py --case_ids 1 2 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

import final_local_collage_harmonization as baseline


DEFAULT_KONTEXT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "final_local_outputs_kontext"
_BASELINE_PARSE_ARGS = baseline.parse_args


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_kontext_inpaint_pipeline(args):
    """Load one native Kontext inpainting pipeline without Qwen's LoRA."""

    try:
        from diffusers import FluxKontextInpaintPipeline
    except ImportError as error:
        raise ImportError(
            "FluxKontextInpaintPipeline is unavailable. Install a recent "
            "Diffusers build that includes FLUX.1 Kontext inpainting support."
        ) from error

    loading = tqdm(
        total=2,
        desc="Loading FLUX.1 Kontext inpaint",
        unit="stage",
        dynamic_ncols=True,
    )
    pipe = FluxKontextInpaintPipeline.from_pretrained(
        args.model_id,
        torch_dtype=_torch_dtype(args.kontext_dtype),
    )
    loading.update()
    if args.cpu_offload:
        loading.set_description("Enabling Kontext CPU offload")
        pipe.enable_model_cpu_offload()
    else:
        loading.set_description(f"Moving Kontext to {args.device}")
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    loading.update()
    loading.close()
    return pipe


@torch.inference_mode()
def harmonize_local_crop_kontext(
    pipe,
    collage: Image.Image,
    edit_mask: Image.Image,
    crop_box: tuple[int, int, int, int],
    object_name: str,
    args,
) -> tuple[Image.Image, str]:
    """Harmonize exactly the same local collage crop using Kontext."""

    local_image = collage.crop(crop_box)
    local_mask = edit_mask.crop(crop_box)
    source_size = local_image.size
    process_size = (args.process_size, args.process_size)
    local_image = local_image.resize(process_size, Image.Resampling.LANCZOS)
    # White pixels are editable in FluxKontextInpaintPipeline.  Nearest-neighbour
    # resizing keeps mask ownership binary at the model resolution.
    local_mask = local_mask.resize(process_size, Image.Resampling.NEAREST)

    prompt = (
        f"Integrate the pasted {object_name} naturally into the surrounding "
        "scene while keeping its position, design, colour, material, and "
        "overall structure unchanged."
    )
    prompt_2 = (
        f"Photorealistically harmonize only the masked {object_name}. Preserve "
        "the pasted object's distinctive identity and silhouette. Match the "
        "local perspective, illumination, contact, and shadow of the nearby "
        "scene. Do not add another object and do not alter unmasked content."
    )

    call_kwargs = {
        "image": local_image,
        "mask_image": local_mask,
        "prompt": prompt,
        "strength": args.inpaint_strength,
        "width": args.process_size,
        "height": args.process_size,
        "max_area": args.kontext_max_area or args.process_size * args.process_size,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": baseline.make_generator(args.device, args.seed),
    }
    if not args.no_prompt_2:
        call_kwargs["prompt_2"] = prompt_2

    generated = pipe(**call_kwargs).images[0].convert("RGB")
    generated = generated.resize(source_size, Image.Resampling.LANCZOS)
    return generated, prompt if args.no_prompt_2 else f"{prompt} {prompt_2}"


def parse_kontext_args() -> argparse.Namespace:
    """Extend the baseline CLI while retaining all of its existing options."""

    extension = argparse.ArgumentParser(add_help=False)
    extension.add_argument("--guidance_scale", type=float, default=2.5)
    extension.add_argument(
        "--kontext_max_area",
        type=int,
        default=None,
        help="Maximum Kontext processing area; defaults to process_size squared",
    )
    extension.add_argument(
        "--kontext_dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    extension.add_argument(
        "--cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    extension.add_argument(
        "--no_prompt_2",
        action="store_true",
        help="Use only the short CLIP-side prompt",
    )
    kontext_args, baseline_argv = extension.parse_known_args()

    supplied_flags = {token.split("=", 1)[0] for token in baseline_argv if token.startswith("--")}
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *baseline_argv]
        args = _BASELINE_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    # Override Qwen-specific defaults but continue respecting explicit values.
    if "--model_id" not in supplied_flags:
        args.model_id = DEFAULT_KONTEXT_MODEL
    if "--out_dir" not in supplied_flags:
        args.out_dir = DEFAULT_OUTPUT
    if "--steps" not in supplied_flags:
        args.steps = 28

    args.guidance_scale = kontext_args.guidance_scale
    args.kontext_max_area = kontext_args.kontext_max_area
    args.kontext_dtype = kontext_args.kontext_dtype
    args.cpu_offload = kontext_args.cpu_offload
    args.no_prompt_2 = kontext_args.no_prompt_2
    args.backbone = "flux1-kontext-dev"
    return args


def main() -> None:
    # The baseline main function resolves paths, prepares/caches RMBG cutouts,
    # runs all turns, enforces hard preservation, and writes identical metrics.
    # Patching these three module-level hooks changes only the model-facing stage.
    baseline.parse_args = parse_kontext_args
    baseline.load_inpaint_pipeline = load_kontext_inpaint_pipeline
    baseline.harmonize_local_crop = harmonize_local_crop_kontext
    baseline.main()


if __name__ == "__main__":
    main()
