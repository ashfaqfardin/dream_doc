"""E13: two-image collage insertion with timestep-aligned latent blending.

For each object, B is the current scene and C is an aligned rough collage B+O.
Qwen receives exactly [B, C].  During its single denoising pass, output latents
outside the known pasted-object region are softly anchored to the VAE latent of
B noised to the same flow-matching timestep.  The pasted alpha is localization,
not a post-generation compositing mask; the decoded model output is saved as-is.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter
from tqdm.auto import tqdm

from e1_baseline import fit, load_pipe, make_generator, save_json
from e2_sam_collage_repaint import composite, place_cutout, probe_placement
from e3_prompt_suite import generate_references, load_suite, reference_key, select_cases, slug
from e5_spatial_kv_collage import RMBG2Cutout, generate_base, generate_rmbg_cutouts
from e12_spatial_reference_card_insertion import E12Evaluator


HERE = Path(__file__).resolve().parent


def _odd_kernel(radius: int) -> int:
    return max(3, 2 * max(1, int(radius)) + 1)


def make_latent_gate(mask: Image.Image, args, grid_size: tuple[int, int]):
    """Return a soft output-freedom gate: one edits, zero anchors to B."""
    core = mask.convert("L")
    interaction = core.filter(ImageFilter.MaxFilter(_odd_kernel(args.interaction_dilation)))
    interaction = interaction.filter(ImageFilter.GaussianBlur(args.gate_blur))
    core_a = np.asarray(core.resize(grid_size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    halo_a = np.asarray(interaction.resize(grid_size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    core_a = np.clip(core_a, 0.0, 1.0)
    ring_a = np.clip(halo_a - core_a, 0.0, 1.0)
    return core_a, ring_a


class TimestepAlignedLatentBlender:
    """Callback that anchors background to B at the scheduler's next sigma."""

    def __init__(self, pipe, base: Image.Image, paste_mask: Image.Image, args, seed: int):
        self.args = args
        self.steps = []
        self.device = pipe._execution_device
        self.dtype = pipe.transformer.dtype
        latent_h = 2 * (args.height // (pipe.vae_scale_factor * 2))
        latent_w = 2 * (args.width // (pipe.vae_scale_factor * 2))
        channels = pipe.transformer.config.in_channels // 4
        generator = make_generator(args.device, seed)
        noise_5d = torch.randn(
            (1, 1, channels, latent_h, latent_w),
            generator=generator, device=self.device, dtype=self.dtype,
        )
        self.noise = pipe._pack_latents(noise_5d, 1, channels, latent_h, latent_w)

        base_tensor = pipe.image_processor.preprocess(
            fit(base, (args.width, args.height)), args.height, args.width
        ).unsqueeze(2).to(device=self.device, dtype=pipe.vae.dtype)
        base_5d = pipe._encode_vae_image(base_tensor, generator=generator).to(self.dtype)
        base_h, base_w = base_5d.shape[3:]
        if (base_h, base_w) != (latent_h, latent_w):
            raise RuntimeError(
                f"Base latent grid {(base_h, base_w)} does not match output grid {(latent_h, latent_w)}"
            )
        self.base = pipe._pack_latents(base_5d, 1, channels, base_h, base_w)

        grid = (latent_w // 2, latent_h // 2)
        core, ring = make_latent_gate(paste_mask, args, grid)
        self.core = torch.from_numpy(core.reshape(1, -1, 1).copy()).to(self.device, self.dtype)
        self.ring = torch.from_numpy(ring.reshape(1, -1, 1).copy()).to(self.device, self.dtype)
        if self.core.shape[1] != self.base.shape[1]:
            raise RuntimeError(
                f"Gate has {self.core.shape[1]} tokens but Qwen output has {self.base.shape[1]}"
            )

    def schedule(self, fraction: float):
        if fraction < self.args.early_fraction:
            return self.args.early_background_freedom, self.args.early_ring_freedom, "early"
        if fraction < self.args.late_fraction:
            return self.args.middle_background_freedom, self.args.middle_ring_freedom, "middle"
        return self.args.late_background_freedom, self.args.late_ring_freedom, "late"

    @torch.no_grad()
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        # The callback runs after scheduler.step(), so anchor against sigma i+1.
        sigma_index = min(step_index + 1, len(pipe.scheduler.sigmas) - 1)
        sigma = pipe.scheduler.sigmas[sigma_index].to(latents.device, latents.dtype)
        base_at_sigma = (1.0 - sigma) * self.base + sigma * self.noise
        fraction = (step_index + 1) / max(1, self.args.steps)
        background, ring, phase = self.schedule(fraction)
        freedom = background + self.ring * (ring - background) + self.core * (1.0 - background)
        freedom = freedom.clamp(0.0, 1.0)
        blended = freedom * latents + (1.0 - freedom) * base_at_sigma
        self.steps.append({
            "step": int(step_index), "phase": phase, "sigma": float(sigma),
            "background_freedom": float(background), "ring_freedom": float(ring),
        })
        return {"latents": blended}

    def close(self):
        self.base = self.base.cpu()
        self.noise = self.noise.cpu()
        self.core = self.core.cpu()
        self.ring = self.ring.cpu()


@torch.inference_mode()
def infer_blended(pipe, base, collage, mask, prompt, args, seed):
    blender = TimestepAlignedLatentBlender(pipe, base, mask, args, seed)
    try:
        result = pipe(
            image=[base, collage],
            prompt=prompt,
            negative_prompt=args.negative_prompt if args.true_cfg_scale > 1 else None,
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            generator=make_generator(args.device, seed),
            latents=blender.noise,
            callback_on_step_end=blender,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        return result.images[0].convert("RGB"), blender.steps
    finally:
        blender.close()


def run_case(pipe, case, references, cutouts, evaluator, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    occupied = Image.new("L", current.size)
    history = []

    for index, item in enumerate(tqdm(
        case["objects"][: args.max_objects or None],
        desc=f"E13 case {case_id:03d}", unit="object", leave=False,
    ), 1):
        name, key = item["name"], reference_key(item)
        prefix = steps_dir / f"{index:02d}_{slug(name)}"
        final_path = Path(f"{prefix}_after.png")
        mask_path = Path(f"{prefix}_paste_alpha.png")
        if args.resume and final_path.is_file() and mask_path.is_file():
            current = Image.open(final_path).convert("RGB")
            occupied = ImageChops.lighter(occupied, Image.open(mask_path).convert("L"))
            history.append({"step": index, "name": name, "status": "resumed", "final": str(final_path)})
            continue
        record = references.get(key, {})
        if record.get("status") != "ready" or key not in cutouts:
            if args.missing_policy == "error":
                raise FileNotFoundError(f"No reference/cutout available for {name}: {record}")
            history.append({"step": index, "name": name, "status": "skipped_missing_reference"})
            continue

        before = fit(current, (args.width, args.height))
        before_path = Path(f"{prefix}_base_input.png")
        collage_path = Path(f"{prefix}_collage_input.png")
        heatmap_path = Path(f"{prefix}_placement_heatmap.png")
        before.save(before_path)
        box, probe = probe_placement(
            pipe, before, name, cutouts[key], occupied, args,
            args.seed + case_id * 100000 + index * 1000, heatmap_path,
        )
        object_canvas, paste_mask, placed_box = place_cutout(
            cutouts[key], box, before.size, args.object_scale
        )
        collage = composite(before, object_canvas, paste_mask)
        collage.save(collage_path)
        paste_mask.save(mask_path)

        prompt = (
            f"Image 1 is the original scene. Image 2 is a rough aligned composite containing one pasted {name}. "
            f"Generate one photorealistic version of Image 2 by harmonizing that exact {name} naturally into "
            "Image 1. Keep the object's position, complete structure, proportions, colors, materials, textures, "
            "and distinctive details. Correct only its boundary, perspective, illumination, support contact, and "
            "shadow. Preserve Image 1's camera, geometry, background, and every existing object. Do not output an "
            "isolated object, white background, reference board, collage, grid, split image, or duplicate object."
        )
        seed = args.seed + case_id * 10000 + index * 100
        current, blend_trace = infer_blended(pipe, before, collage, paste_mask, prompt, args, seed)
        current.save(final_path)
        metrics = evaluator.evaluate(name, before, current, cutouts[key], prefix) if evaluator else None
        occupied = ImageChops.lighter(occupied, paste_mask)
        history.append({
            "step": index, "name": name, "status": "generated", "seed": seed,
            "base_input": str(before_path), "collage_input": str(collage_path),
            "paste_alpha": str(mask_path), "placed_box": list(placed_box), "probe": probe,
            "model_image_roles": ["base", "aligned_collage"],
            "latent_blending": blend_trace, "metrics": metrics, "final": str(final_path),
            "postprocess": None,
        })
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "status": "complete", "objects": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    p.add_argument("--out_dir", default="results/qwen_e13_masked_latent_blending")
    p.add_argument("--case_ids", type=int, nargs="+")
    p.add_argument("--max_objects", type=int, default=3, choices=(1, 2, 3))
    p.add_argument("--missing_policy", choices=("skip", "error"), default="skip")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--lightning_repo", default="lightx2v/Qwen-Image-Lightning")
    p.add_argument("--lightning_weight", default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors")
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--object_seed", type=int, default=1337)
    p.add_argument("--true_cfg_scale", type=float, default=1.0)
    p.add_argument("--negative_prompt", default=" ")
    p.add_argument("--rmbg_model_id", default="briaai/RMBG-2.0")
    p.add_argument("--rmbg_revision", default="54c725d3b17ca83aba490092de8acf6118b8bb06")
    p.add_argument("--rmbg_device", default="cuda")
    p.add_argument("--rmbg_input_size", type=int, default=1024)
    p.add_argument("--rmbg_crop_threshold", type=int, default=8)
    p.add_argument("--probe_steps", type=int, default=4)
    p.add_argument("--probe_quantile", type=float, default=.88)
    p.add_argument("--probe_blur", type=float, default=1.2)
    p.add_argument("--box_margin", type=int, default=24)
    p.add_argument("--occupancy_margin", type=int, default=24)
    p.add_argument("--default_object_height", type=float, default=.25)
    p.add_argument("--object_height_priors")
    p.add_argument("--object_scale", type=float, default=.92)
    p.add_argument("--interaction_dilation", type=int, default=32)
    p.add_argument("--gate_blur", type=float, default=8.0)
    p.add_argument("--early_fraction", type=float, default=.375)
    p.add_argument("--late_fraction", type=float, default=.75)
    p.add_argument("--early_background_freedom", type=float, default=.40)
    p.add_argument("--middle_background_freedom", type=float, default=.15)
    p.add_argument("--late_background_freedom", type=float, default=.02)
    p.add_argument("--early_ring_freedom", type=float, default=1.0)
    p.add_argument("--middle_ring_freedom", type=float, default=.70)
    p.add_argument("--late_ring_freedom", type=float, default=.35)
    p.add_argument("--evaluation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--metric_model_id", default="facebook/dinov2-base")
    p.add_argument("--metric_device", default="cpu")
    p.add_argument("--metric_localization_grid", type=int, default=128)
    p.add_argument("--metric_change_threshold", type=float, default=12.0)
    p.add_argument("--metric_crop_margin", type=float, default=.12)
    p.add_argument("--qualitative_tile_size", type=int, default=320)
    return p.parse_args()


def main():
    args = parse_args()
    freedoms = [
        args.early_background_freedom, args.middle_background_freedom,
        args.late_background_freedom, args.early_ring_freedom,
        args.middle_ring_freedom, args.late_ring_freedom,
    ]
    if not all(0 <= value <= 1 for value in freedoms):
        raise ValueError("All latent freedom values must lie in [0,1]")
    if not 0 < args.early_fraction < args.late_fraction < 1:
        raise ValueError("Require 0 < early_fraction < late_fraction < 1")
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E13 targets diffusers 0.40.0; found {diffusers.__version__}")
        import kornia  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E13") from exc

    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")
    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    segmenter = RMBG2Cutout(args.rmbg_model_id, args.rmbg_revision, args.rmbg_device, args.rmbg_input_size)
    evaluator = None
    try:
        cutouts = generate_rmbg_cutouts(segmenter, references, args, out)
        evaluator = E12Evaluator(segmenter, args) if args.evaluation else None
        summary = []
        for case in tqdm(cases, desc="E13 masked-latent suite", unit="case"):
            summary.append(run_case(pipe, case, references, cutouts, evaluator, args, out))
            save_json(summary, out / "summary.partial.json")
    finally:
        if evaluator is not None:
            evaluator.close()
        segmenter.close()
    save_json({
        "method": "[base, aligned collage] with timestep-aligned masked latent blending",
        "model_image_roles": ["base", "aligned_collage"],
        "localization": "known RMBG paste alpha",
        "postprocess": None,
        "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
