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
import torch.nn.functional as F
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


class SpatialRoutingProcessor:
    """Qwen attention with output-query routing between B and C memories."""

    def __init__(self, controller, original):
        self.controller = controller
        self.original = original

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        encoder_hidden_states_mask=None, attention_mask=None,
        image_rotary_emb=None, **kwargs,
    ):
        controller = self.controller
        if not controller.active:
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, attention_mask, image_rotary_emb, **kwargs,
            )
        if encoder_hidden_states is None or attention_mask is not None:
            raise RuntimeError("E5 requires Qwen's standard double-stream attention inputs")

        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE

        seq_txt = encoder_hidden_states.shape[1]
        seq_img = hidden_states.shape[1]
        output_tokens = (controller.args.height // 16) * (controller.args.width // 16)
        remainder = seq_img - output_tokens
        if remainder <= 0 or remainder % 2:
            raise RuntimeError(
                f"Unexpected [B,C] image-token layout: {seq_img} tokens; expected output + two equal conditions"
            )
        condition_tokens = remainder // 2
        if condition_tokens != output_tokens:
            raise RuntimeError(
                "E5 currently requires square 1024-area B and C inputs matching the output token grid; "
                f"got output={output_tokens}, condition={condition_tokens}."
            )

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)
        head_dim = attn.inner_dim // attn.heads
        img_query = img_query.unflatten(-1, (-1, head_dim))
        img_key = img_key.unflatten(-1, (-1, head_dim))
        img_value = img_value.unflatten(-1, (-1, head_dim))
        txt_query = txt_query.unflatten(-1, (-1, head_dim))
        txt_key = txt_key.unflatten(-1, (-1, head_dim))
        txt_value = txt_value.unflatten(-1, (-1, head_dim))
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            apply_rope = ROPE_PER_DEVICE.get(img_query.device.type, ROPE_PER_DEVICE["cuda"])
            img_query = apply_rope(img_query, img_freqs)
            img_key = apply_rope(img_key, img_freqs)
            txt_query = apply_rope(txt_query, txt_freqs)
            txt_key = apply_rope(txt_key, txt_freqs)

        backend = getattr(self.original, "_attention_backend", None)
        parallel = getattr(self.original, "_parallel_config", None)
        image_mask = torch.ones((hidden_states.shape[0], seq_img), dtype=torch.bool, device=hidden_states.device)
        full_mask = None
        if encoder_hidden_states_mask is not None:
            full_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)[:, None, None, :]

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        joint = dispatch_attention_fn(
            joint_query, joint_key, joint_value, attn_mask=full_mask,
            dropout_p=0.0, is_causal=False, backend=backend, parallel_config=parallel,
        )

        # Shared context retains text and noisy-output self-attention. The only
        # difference between branches is whether condition memory comes from B or C.
        common_key = torch.cat([txt_key, img_key[:, :output_tokens]], dim=1)
        common_value = torch.cat([txt_value, img_value[:, :output_tokens]], dim=1)
        base_slice = slice(output_tokens, output_tokens + condition_tokens)
        collage_slice = slice(output_tokens + condition_tokens, output_tokens + 2 * condition_tokens)
        base_key = torch.cat([common_key, img_key[:, base_slice]], dim=1)
        base_value = torch.cat([common_value, img_value[:, base_slice]], dim=1)
        collage_key = torch.cat([common_key, img_key[:, collage_slice]], dim=1)
        collage_value = torch.cat([common_value, img_value[:, collage_slice]], dim=1)
        branch_mask = None
        if encoder_hidden_states_mask is not None:
            branch_image_mask = torch.ones(
                (hidden_states.shape[0], output_tokens + condition_tokens),
                dtype=torch.bool, device=hidden_states.device,
            )
            branch_mask = torch.cat([encoder_hidden_states_mask, branch_image_mask], dim=1)[:, None, None, :]
        output_query = img_query[:, :output_tokens]
        base_attention = dispatch_attention_fn(
            output_query, base_key, base_value, attn_mask=branch_mask,
            dropout_p=0.0, is_causal=False, backend=backend, parallel_config=parallel,
        )
        collage_attention = dispatch_attention_fn(
            output_query, collage_key, collage_value, attn_mask=branch_mask,
            dropout_p=0.0, is_causal=False, backend=backend, parallel_config=parallel,
        )
        gate = controller.spatial_gate(output_tokens, joint.device, joint.dtype)
        routed = base_attention * (1 - gate) + collage_attention * gate
        joint[:, seq_txt:seq_txt + output_tokens] = routed

        joint = joint.flatten(2, 3).to(joint_query.dtype)
        txt_output = joint[:, :seq_txt]
        img_output = joint[:, seq_txt:]
        img_output = attn.to_out[0](img_output.contiguous())
        if len(attn.to_out) > 1:
            img_output = attn.to_out[1](img_output)
        txt_output = attn.to_add_out(txt_output.contiguous())
        controller.calls += 1
        controller.layout = {
            "output_tokens": output_tokens,
            "tokens_per_condition_image": condition_tokens,
            "sequence_tokens": seq_img,
        }
        return img_output, txt_output


class SpatialBaseKVShare:
    """Install and control query-dependent B/C attention routing."""

    def __init__(self, pipe, args):
        self.args = args
        self.blocks = list(pipe.transformer.transformer_blocks)
        self.layers = parse_layer_spec(args.kv_layers, len(self.blocks))
        self.originals = {}
        self.gate = None
        self.active = False
        self.calls = 0
        self.layout = None

    def set_gate(self, mask: Image.Image):
        gate = dilate(mask.convert("L"), self.args.kv_edit_dilation)
        if self.args.kv_gate_blur > 0:
            gate = gate.filter(ImageFilter.GaussianBlur(self.args.kv_gate_blur))
        self.gate = np.asarray(gate, dtype=np.float32).copy() / 255.0
        return gate

    def spatial_gate(self, token_count, device, dtype):
        height, width = self.gate.shape
        grid_width = max(1, round(math.sqrt(token_count * width / height)))
        grid_height = token_count // grid_width
        if grid_height * grid_width != token_count:
            raise RuntimeError(f"Cannot map {token_count} output tokens to the edit gate")
        gate = Image.fromarray(np.uint8(np.clip(self.gate, 0, 1) * 255)).resize(
            (grid_width, grid_height), Image.Resampling.BILINEAR
        )
        values = torch.from_numpy(np.asarray(gate, dtype=np.float32).copy().reshape(-1) / 255.0)
        return values.to(device=device, dtype=dtype)[None, :, None, None]

    def install(self):
        for index in self.layers:
            attention = self.blocks[index].attn
            self.originals[index] = attention.processor
            attention.set_processor(SpatialRoutingProcessor(self, attention.processor))

    def begin(self):
        if self.gate is None:
            raise RuntimeError("Set the SAM2-derived spatial gate before inference")
        self.calls = 0
        self.layout = None
        self.active = True

    def end(self):
        self.active = False
        cfg_passes = 2 if self.args.true_cfg_scale > 1 and self.args.negative_prompt is not None else 1
        expected = len(self.layers) * self.args.steps * cfg_passes
        if self.calls != expected:
            raise RuntimeError(f"Spatial routing ran {self.calls} times; expected {expected}")
        return {
            "mechanism": "query-dependent base/collage attention routing",
            "layers": self.layers,
            "processor_calls": self.calls,
            "token_layout": self.layout,
        }

    def close(self):
        self.active = False
        for index, processor in self.originals.items():
            self.blocks[index].attn.set_processor(processor)


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
    parser.add_argument("--mask_backend", choices=("sam2",), default="sam2", help="SAM2 is used only for clean reference cutouts")
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
    parser.add_argument("--kv_layers", default="20-49", help="Layers using spatial B/C attention routing")
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
