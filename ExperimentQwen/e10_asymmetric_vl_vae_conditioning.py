"""E10: asymmetric VL/VAE conditioning for reference-object insertion.

For every incremental edit, the current scene B is the *only* image encoded by
Qwen's VAE and therefore the only latent image canvas. The multimodal prompt
encoder separately sees [B, O], where O is the reference object. Thus O can
provide semantic identity evidence without becoming a competing denoising
canvas. No collage, mask, SAM, KV intervention, compositing, or post-pass is
used. The native Qwen output is the final image for that step.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

from e1_baseline import fit, infer, load_pipe, make_generator, save_json
from e3_prompt_suite import (
    generate_references,
    load_suite,
    reference_key,
    select_cases,
    slug,
)
from e5_spatial_kv_collage import generate_base


HERE = Path(__file__).resolve().parent


def prepare_vl_images(pipe, images):
    """Resize images exactly as Qwen's multimodal prompt encoder expects."""
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        calculate_dimensions,
    )

    prepared = []
    for image in images:
        width, height = calculate_dimensions(
            CONDITION_IMAGE_SIZE, image.width / image.height
        )
        prepared.append(pipe.image_processor.resize(image, height, width))
    return prepared


@torch.inference_mode()
def infer_asymmetric(pipe, base, reference, prompt, args, seed):
    """VL receives [B,O], while the pipeline VAE receives only [B]."""
    semantic_images = prepare_vl_images(pipe, [base, reference])
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt=prompt,
        image=semantic_images,
        device=pipe._execution_device,
        num_images_per_prompt=1,
    )

    negative_embeds = negative_mask = None
    cfg_enabled = args.true_cfg_scale > 1.0
    if cfg_enabled:
        negative_embeds, negative_mask = pipe.encode_prompt(
            prompt=args.negative_prompt,
            image=semantic_images,
            device=pipe._execution_device,
            num_images_per_prompt=1,
        )

    # Critical asymmetry: `image` contains B only. O exists solely in the
    # precomputed VL embeddings and never enters the VAE latent token sequence.
    result = pipe(
        image=[base],
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        negative_prompt_embeds=negative_embeds,
        negative_prompt_embeds_mask=negative_mask,
        true_cfg_scale=args.true_cfg_scale,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        generator=make_generator(args.device, seed),
    )
    return result.images[0].convert("RGB")


def run_case(pipe, case, references, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    history = []

    objects = case["objects"][: args.max_objects or None]
    for index, item in enumerate(tqdm(
        objects, desc=f"E10 case {case_id:03d}", unit="object", leave=False
    ), 1):
        name = item["name"]
        key = reference_key(item)
        record = references.get(key, {})
        prefix = steps_dir / f"{index:02d}_{slug(name)}"
        final_path = Path(f"{prefix}_after.png")

        if record.get("status") != "ready":
            if args.missing_policy == "error":
                raise FileNotFoundError(f"No reference available for {name}: {record}")
            history.append({
                "step": index, "name": name,
                "status": "skipped_missing_reference",
            })
            continue
        if args.resume and final_path.is_file():
            current = Image.open(final_path).convert("RGB")
            history.append({
                "step": index, "name": name, "status": "resumed",
                "final": str(final_path),
            })
            continue

        before = fit(current, (args.width, args.height))
        reference = fit(Image.open(record["image"]), (args.width, args.height))
        before_path = Path(f"{prefix}_before.png")
        reference_path = Path(f"{prefix}_vl_reference.png")
        before.save(before_path)
        reference.save(reference_path)

        prompt = (
            f"Image 1 is the current scene and is the only output canvas. "
            f"Image 2 is a visual identity reference for one {name}; it is not "
            "a scene or an output canvas. Add exactly one complete instance of "
            f"the Image 2 {name} into a physically plausible unoccupied location "
            "in Image 1. Preserve the reference object's distinctive shape, "
            "proportions, components, colors, materials, texture, and markings, "
            "while adapting only its pose, scale, perspective, illumination, "
            "support contact, shadow, and occlusion to the scene. Preserve the "
            "Image 1 background, camera, layout, and every previously inserted "
            "object. Do not reproduce Image 2's background, framing, or layout. "
            "Return one photorealistic scene image, never a collage or grid."
        )
        edit_seed = args.seed + case_id * 10000 + index * 100
        current = infer_asymmetric(
            pipe, before, reference, prompt, args, edit_seed
        )
        current.save(final_path)
        history.append({
            "step": index,
            "name": name,
            "status": "generated",
            "seed": edit_seed,
            "vl_inputs": [str(before_path), str(reference_path)],
            "vae_inputs": [str(before_path)],
            "reference_source": record["image"],
            "final": str(final_path),
            "postprocess": None,
        })
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {
        "id": case_id,
        "status": "complete",
        "objects": len(history),
        "final": str(case_dir / "FINAL.png"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    parser.add_argument("--out_dir", default="results/qwen_e10_asymmetric_vl_vae")
    parser.add_argument("--case_ids", type=int, nargs="+")
    parser.add_argument("--max_objects", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--missing_policy", choices=("skip", "error"), default="skip")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--lightning_repo", default="lightx2v/Qwen-Image-Lightning")
    parser.add_argument(
        "--lightning_weight",
        default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object_seed", type=int, default=1337)
    parser.add_argument("--true_cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default=" ")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(
                f"E10 targets diffusers 0.40.0; found {diffusers.__version__}"
            )
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E10") from exc

    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")

    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    summary = []
    for case in tqdm(cases, desc="E10 asymmetric VL/VAE suite", unit="case"):
        summary.append(run_case(pipe, case, references, args, out))
        save_json(summary, out / "summary.partial.json")

    save_json({
        "method": "asymmetric multimodal conditioning",
        "vl_images": ["current scene", "reference object"],
        "vae_images": ["current scene"],
        "feature_intervention": None,
        "masking": None,
        "postprocess": None,
        "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
