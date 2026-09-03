"""E5: one-pass [base, collage, cutout] feature-routed harmonization.

RMBG-2.0 creates a soft-alpha object cutout. Qwen receives [base/current image,
collage image, isolated object cutout]. Output queries use base memory outside
the edit gate and a controlled collage-geometry/object-identity mixture inside.
The raw Qwen result is final: no pixel-space operation follows inference.

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
from e2_sam_collage_repaint import Cutout, composite, dilate, place_cutout, probe_placement
from e3_prompt_suite import (
    generate_references,
    load_suite,
    reference_key,
    select_cases,
    slug,
)


HERE = Path(__file__).resolve().parent


def load_rmbg2_model(model_id: str, revision: str):
    """Load RMBG-2.0 across Transformers 4.x and 5.x.

    RMBG-2.0's remote BiRefNet implementation attaches a small, non-HF
    ``Config`` object to some nested modules.  Transformers 5 inspects every
    such object while building its checkpoint conversion map and assumes that
    ``model_type`` exists.  Add that missing metadata only while this model is
    being loaded; do not pin/downgrade Transformers because Qwen shares it.
    """
    from transformers import AutoModelForImageSegmentation
    import transformers.modeling_utils as modeling_utils

    original = getattr(modeling_utils, "get_model_conversion_mapping", None)
    if original is None:
        return AutoModelForImageSegmentation.from_pretrained(
            model_id,
            revision=revision,
            code_revision=revision,
            trust_remote_code=True,
        )

    def compatible_conversion_mapping(model, *args, **kwargs):
        patched = 0
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is None or hasattr(config, "model_type"):
                continue
            try:
                config.model_type = "birefnet"
            except (AttributeError, TypeError):
                # Handles a possible slotted config in a future remote-code
                # revision.  The class belongs to RMBG's loaded module.
                setattr(type(config), "model_type", "birefnet")
            patched += 1
        if patched:
            print(
                f"RMBG-2.0: supplied model_type metadata to "
                f"{patched} nested config(s) for Transformers compatibility."
            )
        return original(model, *args, **kwargs)

    modeling_utils.get_model_conversion_mapping = compatible_conversion_mapping
    try:
        return AutoModelForImageSegmentation.from_pretrained(
            model_id,
            revision=revision,
            code_revision=revision,
            trust_remote_code=True,
        )
    finally:
        modeling_utils.get_model_conversion_mapping = original


class RMBG2Cutout:
    """Soft-alpha foreground extraction using the gated BRIA RMBG-2.0 model."""

    def __init__(self, model_id: str, revision: str, device: str, input_size: int):
        self.device = torch.device(device)
        self.input_size = input_size
        try:
            self.model = load_rmbg2_model(model_id, revision).to(self.device).eval()
        except OSError as exc:
            raise RuntimeError(
                "RMBG-2.0 is gated. Accept its Hugging Face terms and authenticate "
                "the runtime with `huggingface-cli login`."
            ) from exc

    @torch.inference_mode()
    def cutout(self, name: str, image: Image.Image, threshold: float):
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = F.interpolate(
            tensor, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]
        prediction = self.model((tensor - mean) / std)[-1].sigmoid()[0, 0]
        alpha_array = prediction.float().cpu().numpy()
        alpha = Image.fromarray(np.uint8(np.clip(alpha_array, 0, 1) * 255)).resize(
            rgb.size, Image.Resampling.LANCZOS
        )
        support = alpha.point(lambda value: 255 if value >= threshold else 0)
        box = support.getbbox()
        if box is None:
            raise RuntimeError(f"RMBG-2.0 returned an empty alpha matte for {name}")
        return Cutout(name, rgb.crop(box), alpha.crop(box), tuple(map(int, box)))

    def close(self):
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_rmbg_cutouts(segmenter, references, args, out: Path):
    directory = out / "cutouts"
    directory.mkdir(parents=True, exist_ok=True)
    cutouts = {}
    for key, record in tqdm(references.items(), desc="RMBG-2.0 cutouts", unit="object"):
        if record.get("status") != "ready":
            continue
        image = fit(Image.open(record["image"]), (args.width, args.height))
        cutout = segmenter.cutout(record["name"], image, args.rmbg_crop_threshold)
        cutout.rgb.save(directory / f"{key}_rgb.png")
        cutout.alpha.save(directory / f"{key}_alpha.png")
        rgba = cutout.rgb.copy()
        rgba.putalpha(cutout.alpha)
        rgba.save(directory / f"{key}_rgba.png")
        cutouts[key] = cutout
    return cutouts


def object_condition(cutout: Cutout, args):
    """Create O and its aligned foreground-token mask for the third input."""
    canvas = Image.new("RGB", (args.width, args.height), args.object_condition_background)
    alpha_canvas = Image.new("L", canvas.size)
    available_w = round(args.width * args.object_condition_scale)
    available_h = round(args.height * args.object_condition_scale)
    scale = min(available_w / cutout.rgb.width, available_h / cutout.rgb.height)
    size = (max(1, round(cutout.rgb.width * scale)), max(1, round(cutout.rgb.height * scale)))
    rgb = cutout.rgb.resize(size, Image.Resampling.LANCZOS)
    alpha = cutout.alpha.resize(size, Image.Resampling.LANCZOS)
    position = ((args.width - size[0]) // 2, (args.height - size[1]) // 2)
    canvas.paste(rgb, position, alpha)
    alpha_canvas.paste(alpha, position)
    return canvas, alpha_canvas


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
    """Qwen attention with role-specific B, C, and O memories."""

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
        if remainder <= 0 or remainder % 3:
            raise RuntimeError(
                f"Unexpected [B,C,O] token layout: {seq_img}; expected output + three equal conditions"
            )
        condition_tokens = remainder // 3
        if condition_tokens != output_tokens:
            raise RuntimeError(
                "E5 requires square B, C, and O inputs matching the output token grid; "
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

        # Shared context retains text and noisy-output self-attention. Each
        # branch then receives exactly one semantically assigned image memory.
        common_key = torch.cat([txt_key, img_key[:, :output_tokens]], dim=1)
        common_value = torch.cat([txt_value, img_value[:, :output_tokens]], dim=1)
        base_slice = slice(output_tokens, output_tokens + condition_tokens)
        collage_slice = slice(output_tokens + condition_tokens, output_tokens + 2 * condition_tokens)
        object_slice = slice(output_tokens + 2 * condition_tokens, output_tokens + 3 * condition_tokens)
        base_key = torch.cat([common_key, img_key[:, base_slice]], dim=1)
        base_value = torch.cat([common_value, img_value[:, base_slice]], dim=1)
        collage_key = torch.cat([common_key, img_key[:, collage_slice]], dim=1)
        collage_value = torch.cat([common_value, img_value[:, collage_slice]], dim=1)
        foreground = controller.object_foreground_indices(condition_tokens, img_key.device)
        object_key = torch.cat([common_key, img_key[:, object_slice].index_select(1, foreground)], dim=1)
        object_value = torch.cat([common_value, img_value[:, object_slice].index_select(1, foreground)], dim=1)
        branch_mask = None
        if encoder_hidden_states_mask is not None:
            branch_image_mask = torch.ones(
                (hidden_states.shape[0], output_tokens + condition_tokens),
                dtype=torch.bool, device=hidden_states.device,
            )
            branch_mask = torch.cat([encoder_hidden_states_mask, branch_image_mask], dim=1)[:, None, None, :]
        output_query = img_query[:, :output_tokens]
        base_query = torch.cat([output_query, img_query[:, base_slice]], dim=1)
        collage_query = torch.cat([output_query, img_query[:, collage_slice]], dim=1)
        object_query = torch.cat([output_query, img_query[:, object_slice]], dim=1)
        base_result = dispatch_attention_fn(
            base_query, base_key, base_value, attn_mask=branch_mask,
            dropout_p=0.0, is_causal=False, backend=backend, parallel_config=parallel,
        )
        collage_result = dispatch_attention_fn(
            collage_query, collage_key, collage_value, attn_mask=branch_mask,
            dropout_p=0.0, is_causal=False, backend=backend, parallel_config=parallel,
        )
        object_image_mask = None
        if encoder_hidden_states_mask is not None:
            object_keys = output_tokens + int(foreground.numel())
            object_image_mask = torch.ones(
                (hidden_states.shape[0], object_keys), dtype=torch.bool, device=hidden_states.device
            )
            object_image_mask = torch.cat([encoder_hidden_states_mask, object_image_mask], dim=1)[:, None, None, :]
        object_result = dispatch_attention_fn(
            object_query, object_key, object_value, attn_mask=object_image_mask,
            dropout_p=0.0, is_causal=False, backend=backend, parallel_config=parallel,
        )
        base_attention = base_result[:, :output_tokens]
        collage_attention = collage_result[:, :output_tokens]
        object_attention = object_result[:, :output_tokens]
        gate = controller.spatial_gate(output_tokens, joint.device, joint.dtype)
        inside = (
            controller.args.collage_feature_weight * collage_attention
            + controller.args.identity_feature_weight * object_attention
        )
        routed = base_attention * (1 - gate) + inside * gate
        joint[:, seq_txt:seq_txt + output_tokens] = routed
        # Keep B, C, and O as distinct memories for the following layer. Without
        # this, ordinary joint attention would contaminate B with C/O globally.
        image_offset = seq_txt
        joint[:, image_offset + base_slice.start:image_offset + base_slice.stop] = base_result[:, output_tokens:]
        joint[:, image_offset + collage_slice.start:image_offset + collage_slice.stop] = collage_result[:, output_tokens:]
        joint[:, image_offset + object_slice.start:image_offset + object_slice.stop] = object_result[:, output_tokens:]

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
            "object_foreground_tokens": int(foreground.numel()),
        }
        return img_output, txt_output


class SpatialBaseKVShare:
    """Install and control query-dependent B/C/O attention routing."""

    def __init__(self, pipe, args):
        self.args = args
        self.blocks = list(pipe.transformer.transformer_blocks)
        self.layers = parse_layer_spec(args.kv_layers, len(self.blocks))
        self.originals = {}
        self.gate = None
        self.object_alpha = None
        self.gate_cache = {}
        self.foreground_cache = {}
        self.active = False
        self.calls = 0
        self.layout = None

    def set_gates(self, mask: Image.Image, object_alpha: Image.Image):
        gate = dilate(mask.convert("L"), self.args.kv_edit_dilation)
        if self.args.kv_gate_blur > 0:
            gate = gate.filter(ImageFilter.GaussianBlur(self.args.kv_gate_blur))
        self.gate = np.asarray(gate, dtype=np.float32).copy() / 255.0
        self.object_alpha = np.asarray(object_alpha.convert("L"), dtype=np.float32).copy() / 255.0
        self.gate_cache.clear()
        self.foreground_cache.clear()
        return gate

    def spatial_gate(self, token_count, device, dtype):
        cache_key = (token_count, str(device), dtype)
        if cache_key in self.gate_cache:
            return self.gate_cache[cache_key]
        height, width = self.gate.shape
        grid_width = max(1, round(math.sqrt(token_count * width / height)))
        grid_height = token_count // grid_width
        if grid_height * grid_width != token_count:
            raise RuntimeError(f"Cannot map {token_count} output tokens to the edit gate")
        gate = Image.fromarray(np.uint8(np.clip(self.gate, 0, 1) * 255)).resize(
            (grid_width, grid_height), Image.Resampling.BILINEAR
        )
        values = torch.from_numpy(np.asarray(gate, dtype=np.float32).copy().reshape(-1) / 255.0)
        result = values.to(device=device, dtype=dtype)[None, :, None, None]
        self.gate_cache[cache_key] = result
        return result

    def object_foreground_indices(self, token_count, device):
        cache_key = (token_count, str(device))
        if cache_key in self.foreground_cache:
            return self.foreground_cache[cache_key]
        height, width = self.object_alpha.shape
        grid_width = max(1, round(math.sqrt(token_count * width / height)))
        grid_height = token_count // grid_width
        if grid_height * grid_width != token_count:
            raise RuntimeError(f"Cannot map {token_count} object tokens to the RMBG matte")
        alpha = Image.fromarray(np.uint8(np.clip(self.object_alpha, 0, 1) * 255)).resize(
            (grid_width, grid_height), Image.Resampling.BILINEAR
        )
        values = np.asarray(alpha, dtype=np.float32).copy().reshape(-1) / 255.0
        indices = np.flatnonzero(values >= self.args.object_token_alpha_threshold)
        if not len(indices):
            raise RuntimeError("RMBG matte produced no foreground object tokens")
        result = torch.as_tensor(indices, device=device, dtype=torch.long)
        self.foreground_cache[cache_key] = result
        return result

    def install(self):
        for index in self.layers:
            attention = self.blocks[index].attn
            self.originals[index] = attention.processor
            attention.set_processor(SpatialRoutingProcessor(self, attention.processor))

    def begin(self):
        if self.gate is None or self.object_alpha is None:
            raise RuntimeError("Set the RMBG-derived edit and object gates before inference")
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
            "mechanism": "query-dependent base/collage/object attention routing",
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
        final_path = steps_dir / f"{index:02d}_{slug(name)}_bco_final.png"
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
        object_image, object_alpha = object_condition(cutouts[key], args)
        collage.save(steps_dir / f"{index:02d}_{slug(name)}_collage_input.png")
        object_image.save(steps_dir / f"{index:02d}_{slug(name)}_object_input.png")
        object_alpha.save(steps_dir / f"{index:02d}_{slug(name)}_object_alpha.png")
        paste_mask.save(mask_path)
        kv_gate = kv_share.set_gates(paste_mask, object_alpha)
        kv_gate.save(steps_dir / f"{index:02d}_{slug(name)}_kv_edit_gate.png")

        prompt = (
            f"Image 1 is the unchanged base scene. Image 2 is the same scene with one pasted {name}. "
            f"Image 3 is the isolated exact {name} identity. Return one edited scene: preserve Image 1 outside the "
            f"pasted region, use Image 2 for the {name}'s position and scene geometry, and use Image 3 for its exact "
            "identity, structure, colors and material. Harmonize lighting and contact without moving or redesigning "
            "the object. Do not output an isolated object, reference panel, border, collage, or split image."
        )
        kv_share.begin()
        try:
            # This is the only generative edit pass for the insertion.
            current = infer(pipe, [before, collage, object_image], prompt, args, args.seed + case_id * 10000 + index * 100)
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
            "object_input": str(steps_dir / f"{index:02d}_{slug(name)}_object_input.png"),
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
    parser.add_argument("--rmbg_model_id", default="briaai/RMBG-2.0")
    parser.add_argument("--rmbg_revision", default="54c725d3b17ca83aba490092de8acf6118b8bb06", help="Pinned RMBG weights and remote-code revision")
    parser.add_argument("--rmbg_device", default="cuda")
    parser.add_argument("--rmbg_input_size", type=int, default=1024)
    parser.add_argument("--rmbg_crop_threshold", type=int, default=8, help="Alpha threshold used only to find the tight crop")
    parser.add_argument("--probe_steps", type=int, default=4)
    parser.add_argument("--probe_quantile", type=float, default=.88)
    parser.add_argument("--probe_blur", type=float, default=1.2)
    parser.add_argument("--box_margin", type=int, default=24)
    parser.add_argument("--occupancy_margin", type=int, default=24)
    parser.add_argument("--default_object_height", type=float, default=.25)
    parser.add_argument("--object_height_priors")
    parser.add_argument("--object_scale", type=float, default=.92)
    parser.add_argument("--kv_layers", default="all", help="Layers using spatial B/C/O attention routing; all keeps B uncontaminated")
    parser.add_argument("--kv_edit_dilation", type=int, default=24, help="Pixels around pasted object allowed to use collage K/V")
    parser.add_argument("--kv_gate_blur", type=float, default=3.0)
    parser.add_argument("--collage_feature_weight", type=float, default=.45)
    parser.add_argument("--identity_feature_weight", type=float, default=.55)
    parser.add_argument("--object_token_alpha_threshold", type=float, default=.05)
    parser.add_argument("--object_condition_scale", type=float, default=.82)
    parser.add_argument("--object_condition_background", type=int, default=127)
    return parser.parse_args()


def main():
    args = parse_args()
    if not math.isclose(args.collage_feature_weight + args.identity_feature_weight, 1.0, abs_tol=1e-6):
        raise ValueError("collage_feature_weight + identity_feature_weight must equal 1")
    if not 0 <= args.object_token_alpha_threshold <= 1:
        raise ValueError("object_token_alpha_threshold must be in [0, 1]")
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

    try:
        import kornia  # noqa: F401 -- required by RMBG-2.0 remote model code
    except ImportError as exc:
        raise RuntimeError("E5 requires Kornia for RMBG-2.0. Install it with `pip install kornia`.") from exc
    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    segmenter = RMBG2Cutout(args.rmbg_model_id, args.rmbg_revision, args.rmbg_device, args.rmbg_input_size)
    try:
        cutouts = generate_rmbg_cutouts(segmenter, references, args, out)
    finally:
        segmenter.close()
    kv_share = SpatialBaseKVShare(pipe, args)
    kv_share.install()
    summary = []
    try:
        for case in tqdm(cases, desc="E5 feature-level suite", unit="case"):
            summary.append(run_case(pipe, kv_share, case, references, cutouts, args, out))
            save_json(summary, out / "summary.json")
    finally:
        kv_share.close()
    save_json({"cutout_backend": args.rmbg_model_id, "image_roles": ["base", "collage", "object"], "cases": summary}, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
