"""E5: one-pass collage harmonization with spatial base-image K/V sharing.

For each insertion Qwen receives [base/current image, collage image]. During
that single denoising pass, forward hooks replace the collage condition's key
and value tokens outside the pasted-object gate with the corresponding K/V
tokens from the base/current image. The raw Qwen result is the final step
output; there is no pixel-space protection, masking, compositing, or refinement
after inference.

Target architecture: Qwen-Image-Edit-2509 with diffusers==0.40.0.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter
from tqdm.auto import tqdm

from e1_baseline import fit, infer, load_pipe, save_json
from e2_sam_collage_repaint import composite, dilate, load_segmenter, place_cutout, probe_placement
from e3_prompt_suite import (
    generate_cutouts,
    generate_references,
    load_suite,
    reference_key,
    select_cases,
    slug,
)


HERE = Path(__file__).resolve().parent


def parse_layer_spec(spec: str, count: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(count))
    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = sorted(index for index in selected if index < 0 or index >= count)
    if invalid:
        raise ValueError(f"K/V layer indices outside [0, {count - 1}]: {invalid}")
    return sorted(selected)


class SpatialBaseKVShare:
    """Share base-image K/V into the collage stream outside a spatial gate."""

    def __init__(self, pipe, args):
        self.args = args
        self.blocks = list(pipe.transformer.transformer_blocks)
        self.layers = parse_layer_spec(args.kv_layers, len(self.blocks))
        self.handles = []
        self.gate = None
        self.active = False
        self.replacements = {"key": 0, "value": 0}
        self.layout = None

    def set_gate(self, mask: Image.Image):
        gate = dilate(mask.convert("L"), self.args.kv_edit_dilation)
        if self.args.kv_gate_blur > 0:
            gate = gate.filter(ImageFilter.GaussianBlur(self.args.kv_gate_blur))
        self.gate = np.asarray(gate, dtype=np.float32) / 255.0
        return gate

    def _spatial_gate(self, token_count: int, device, dtype):
        height, width = self.gate.shape
        grid_width = max(1, round(math.sqrt(token_count * width / height)))
        grid_height = token_count // grid_width
        if grid_height * grid_width != token_count:
            side = round(math.sqrt(token_count))
            if side * side != token_count:
                raise RuntimeError(f"Cannot map {token_count} condition tokens to a spatial grid")
            grid_height = grid_width = side
        image = Image.fromarray(np.uint8(np.clip(self.gate, 0, 1) * 255))
        values = np.asarray(image.resize((grid_width, grid_height), Image.Resampling.BILINEAR), dtype=np.float32).copy()
        return torch.from_numpy(values.reshape(-1)).to(device=device, dtype=dtype)[None, :, None]

    def _hook(self, kind: str):
        def hook(_module, _inputs, output):
            if not self.active or self.gate is None or not torch.is_tensor(output) or output.ndim != 3:
                return output
            # Image-token packing for [base, collage]: [noisy output, base, collage].
            output_tokens = (self.args.height // 16) * (self.args.width // 16)
            remainder = output.shape[1] - output_tokens
            if remainder <= 0 or remainder % 2:
                raise RuntimeError(
                    f"Unexpected Qwen image-token layout {tuple(output.shape)}; "
                    "expected output tokens followed by equally sized base and collage streams."
                )
            condition_tokens = remainder // 2
            base_start = output_tokens
            collage_start = base_start + condition_tokens
            gate = self._spatial_gate(condition_tokens, output.device, output.dtype)
            base_kv = output[:, base_start:collage_start]
            collage_kv = output[:, collage_start:collage_start + condition_tokens]
            mixed = collage_kv * gate + base_kv * (1 - gate)
            result = output.clone()
            result[:, collage_start:collage_start + condition_tokens] = mixed
            self.replacements[kind] += 1
            self.layout = {
                "output_tokens": output_tokens,
                "tokens_per_condition_image": condition_tokens,
                "sequence_tokens": int(output.shape[1]),
            }
            return result
        return hook

    def install(self):
        for index in self.layers:
            attention = self.blocks[index].attn
            self.handles.append(attention.to_k.register_forward_hook(self._hook("key")))
            self.handles.append(attention.to_v.register_forward_hook(self._hook("value")))

    def begin(self):
        if self.gate is None:
            raise RuntimeError("Set the spatial K/V gate before beginning inference")
        self.replacements = {"key": 0, "value": 0}
        self.layout = None
        self.active = True

    def end(self):
        self.active = False
        expected = len(self.layers) * self.args.steps
        if self.replacements["key"] != expected or self.replacements["value"] != expected:
            raise RuntimeError(
                f"Incomplete K/V intervention: {self.replacements}; expected {expected} calls per projection."
            )
        return {
            "layers": self.layers,
            "projection_calls": dict(self.replacements),
            "token_layout": self.layout,
        }

    def close(self):
        self.active = False
        for handle in self.handles:
            handle.remove()


def generate_base(pipe, case, args, path: Path):
    if args.resume and path.is_file():
        return fit(Image.open(path), (args.width, args.height))
    blank = Image.new("RGB", (args.width, args.height), "white")
    prompt = "Replace blank Image 1 with this scene: " + case["base_prompt"] + " Fill the complete frame without borders."
    base = infer(pipe, [blank], prompt, args, args.seed + int(case["id"]) * 10000)
    base.save(path)
    return base


def run_case(pipe, kv_share, case, references, cutouts, args, out: Path):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    occupied = Image.new("L", current.size)
    history = []
    objects = case["objects"][: args.max_objects or None]

    for index, item in enumerate(tqdm(objects, desc=f"E5 case {case_id:03d}", unit="object", leave=False), 1):
        name = item["name"]
        key = reference_key(item)
        final_path = steps_dir / f"{index:02d}_{slug(name)}_final.png"
        mask_path = steps_dir / f"{index:02d}_{slug(name)}_paste_mask.png"
        if args.resume and final_path.is_file() and mask_path.is_file():
            current = Image.open(final_path).convert("RGB")
            occupied = ImageChops.lighter(occupied, Image.open(mask_path).convert("L"))
            history.append({"step": index, "name": name, "status": "resumed", "final": str(final_path)})
            continue
        record = references.get(key, {})
        if record.get("status") != "ready":
            if args.missing_policy == "skip":
                history.append({"step": index, "name": name, "status": "skipped_missing_reference"})
                continue
            raise FileNotFoundError(record)

        before = current
        before.save(steps_dir / f"{index:02d}_{slug(name)}_base_input.png")
        box, probe = probe_placement(
            pipe, before, name, cutouts[key], occupied, args,
            args.seed + case_id * 100000 + index * 1000,
            steps_dir / f"{index:02d}_{slug(name)}_placement_heatmap.png",
        )
        object_canvas, paste_mask, placed_box = place_cutout(cutouts[key], box, before.size, args.object_scale)
        collage = composite(before, object_canvas, paste_mask)
        collage.save(steps_dir / f"{index:02d}_{slug(name)}_collage_input.png")
        paste_mask.save(mask_path)
        kv_gate = kv_share.set_gate(paste_mask)
        kv_gate.save(steps_dir / f"{index:02d}_{slug(name)}_kv_edit_gate.png")

        prompt = (
            f"Image 1 is the unchanged base scene. Image 2 is the same scene with one pasted {name}. "
            f"Return one edited scene: preserve Image 1 everywhere except the pasted-object region from Image 2. "
            f"In that region, harmonize the Image 2 {name} naturally while preserving its exact identity, structure, "
            "colors, material, pose, position and scale. Do not output either input image as a reference panel."
        )
        kv_share.begin()
        try:
            # This is the only generative edit pass for the insertion.
            current = infer(pipe, [before, collage], prompt, args, args.seed + case_id * 10000 + index * 100)
        finally:
            kv_share.active = False
        intervention = kv_share.end()
        current.save(final_path)
        occupied = ImageChops.lighter(occupied, paste_mask)
        history.append({
            "step": index,
            "name": name,
            "status": "generated",
            "base_input": str(steps_dir / f"{index:02d}_{slug(name)}_base_input.png"),
            "collage_input": str(steps_dir / f"{index:02d}_{slug(name)}_collage_input.png"),
            "final": str(final_path),
            "placed_box": list(placed_box),
            "probe": probe,
            "kv_intervention": intervention,
            "postprocess": None,
        })
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "status": "complete", "objects": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    parser.add_argument("--out_dir", default="results/qwen_e5_spatial_kv_collage")
    parser.add_argument("--case_ids", type=int, nargs="+")
    parser.add_argument("--max_objects", type=int, default=3, choices=(1, 2, 3), help="E5 is capped at three objects per scene")
    parser.add_argument("--missing_policy", choices=("skip", "error"), default="skip")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--lightning_repo", default="lightx2v/Qwen-Image-Lightning")
    parser.add_argument("--lightning_weight", default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object_seed", type=int, default=1337)
    parser.add_argument("--true_cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--mask_backend", choices=("auto", "sam2", "difference"), default="auto")
    parser.add_argument("--sam_model_id", default="facebook/sam2-hiera-small")
    parser.add_argument("--sam_device", default="cpu")
    parser.add_argument("--background_threshold", type=float, default=24)
    parser.add_argument("--probe_steps", type=int, default=4)
    parser.add_argument("--probe_quantile", type=float, default=.88)
    parser.add_argument("--probe_blur", type=float, default=1.2)
    parser.add_argument("--box_margin", type=int, default=24)
    parser.add_argument("--occupancy_margin", type=int, default=24)
    parser.add_argument("--default_object_height", type=float, default=.25)
    parser.add_argument("--object_height_priors")
    parser.add_argument("--object_scale", type=float, default=.92)
    parser.add_argument("--kv_layers", default="all", help="Comma/range specification, e.g. 20-49, or all")
    parser.add_argument("--kv_edit_dilation", type=int, default=24, help="Pixels around pasted object allowed to use collage K/V")
    parser.add_argument("--kv_gate_blur", type=float, default=3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E5 targets diffusers 0.40.0; found {diffusers.__version__}")
    except ImportError:
        pass
    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")

    segmenter, mask_backend = load_segmenter(args)
    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    cutouts = generate_cutouts(segmenter, references, args, out)
    kv_share = SpatialBaseKVShare(pipe, args)
    kv_share.install()
    summary = []
    try:
        for case in tqdm(cases, desc="E5 feature-level suite", unit="case"):
            summary.append(run_case(pipe, kv_share, case, references, cutouts, args, out))
            save_json(summary, out / "summary.json")
    finally:
        kv_share.close()
    save_json({"mask_backend": mask_backend, "cases": summary}, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
