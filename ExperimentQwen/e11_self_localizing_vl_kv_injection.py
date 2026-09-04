"""E11: asymmetric Qwen editing with self-localizing VL-reference K/V injection.

The current scene B is the only VAE condition. The VL encoder sees [B, O], and
this script preserves token provenance so only Image-2 (O) tokens form the
private reference K/V memory. Selected Qwen transformer layers interpolate
output-token attention toward that memory through a soft, query-derived,
temporally smoothed gate. There is no collage, external placement mask, pixel
composite, or post-generation pass.

This is an experimental training-free intervention, not a trained adapter.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from e1_baseline import fit, infer, load_pipe, make_generator, save_json
from e3_prompt_suite import generate_references, load_suite, reference_key, select_cases, slug
from e5_spatial_kv_collage import generate_base, parse_layer_spec


HERE = Path(__file__).resolve().parent


def prepare_vl_images(pipe, images):
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        calculate_dimensions,
    )

    prepared = []
    for image in images:
        width, height = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
        prepared.append(pipe.image_processor.resize(image, height, width))
    return prepared


@torch.inference_mode()
def encode_with_image_provenance(pipe, prompt, images):
    """Return Qwen VL embeddings plus masks for each expanded image-token run."""
    image_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    image_prefix = "".join(image_template.format(i + 1) for i in range(len(images)))
    text = [pipe.prompt_template_encode.format(image_prefix + prompt)]
    inputs = pipe.processor(
        text=text, images=images, padding=True, return_tensors="pt"
    ).to(pipe._execution_device)
    outputs = pipe.text_encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        output_hidden_states=True,
    )

    valid = inputs["attention_mask"][0].bool()
    ids = inputs["input_ids"][0][valid]
    hidden = outputs.hidden_states[-1][0][valid]
    image_token = getattr(pipe.processor, "image_token", "<|image_pad|>")
    image_token_id = pipe.tokenizer.convert_tokens_to_ids(image_token)
    positions = torch.nonzero(ids == image_token_id, as_tuple=False).flatten()
    if not positions.numel():
        raise RuntimeError(f"VL tokenizer produced no {image_token!r} tokens")

    breaks = torch.nonzero(positions[1:] != positions[:-1] + 1, as_tuple=False).flatten() + 1
    runs = torch.tensor_split(positions, breaks.detach().cpu().tolist())
    if len(runs) != len(images):
        raise RuntimeError(
            f"Expected {len(images)} expanded image-token runs, found {len(runs)}"
        )

    drop = pipe.prompt_template_encode_start_idx
    hidden = hidden[drop:].to(dtype=pipe.text_encoder.dtype)
    provenance = []
    for run in runs:
        mask = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        mask[run] = True
        provenance.append(mask[drop:])
    attention_mask = torch.ones(
        (1, hidden.shape[0]), dtype=torch.long, device=hidden.device
    )
    return hidden.unsqueeze(0), attention_mask, provenance


class SelfLocalizingVLKVProcessor:
    def __init__(self, controller, original, layer_index):
        self.controller = controller
        self.original = original
        self.layer_index = layer_index

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        encoder_hidden_states_mask=None, attention_mask=None,
        image_rotary_emb=None, **kwargs,
    ):
        ctl = self.controller
        if (
            not ctl.active
            or self.layer_index not in ctl.layers
            or encoder_hidden_states is None
            or encoder_hidden_states.shape[1] != ctl.reference_token_mask.numel()
        ):
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, attention_mask, image_rotary_emb, **kwargs,
            )
        if attention_mask is not None:
            raise RuntimeError("E11 expects Qwen's standard double-stream attention")

        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE

        nt = encoder_hidden_states.shape[1]
        ni = hidden_states.shape[1]
        nx = (ctl.args.height // 16) * (ctl.args.width // 16)
        if ni != 2 * nx:
            raise RuntimeError(f"E11 expects output+B image tokens ({2 * nx}), found {ni}")

        iq, ik, iv = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        tq = attn.add_q_proj(encoder_hidden_states)
        tk = attn.add_k_proj(encoder_hidden_states)
        tv = attn.add_v_proj(encoder_hidden_states)
        head_dim = attn.inner_dim // attn.heads
        iq, ik, iv = [x.unflatten(-1, (-1, head_dim)) for x in (iq, ik, iv)]
        tq, tk, tv = [x.unflatten(-1, (-1, head_dim)) for x in (tq, tk, tv)]
        if attn.norm_q is not None:
            iq = attn.norm_q(iq)
        if attn.norm_k is not None:
            ik = attn.norm_k(ik)
        if attn.norm_added_q is not None:
            tq = attn.norm_added_q(tq)
        if attn.norm_added_k is not None:
            tk = attn.norm_added_k(tk)

        # Content descriptors are retained before RoPE for localization; the
        # actual native and reference attention both retain Qwen's RoPE.
        content_q, content_k = iq[:, :nx], tk
        if image_rotary_emb is not None:
            image_freqs, text_freqs = image_rotary_emb
            rope = ROPE_PER_DEVICE.get(iq.device.type, ROPE_PER_DEVICE["cuda"])
            iq, ik = rope(iq, image_freqs), rope(ik, image_freqs)
            tq, tk = rope(tq, text_freqs), rope(tk, text_freqs)

        backend = getattr(self.original, "_attention_backend", None)
        parallel = getattr(self.original, "_parallel_config", None)

        def attend(query, key, value, mask=None):
            return dispatch_attention_fn(
                query, key, value, attn_mask=mask, dropout_p=0.0,
                is_causal=False, backend=backend, parallel_config=parallel,
            )

        full_q = torch.cat([tq, iq], dim=1)
        full_k = torch.cat([tk, ik], dim=1)
        full_v = torch.cat([tv, iv], dim=1)
        full_mask = None
        if encoder_hidden_states_mask is not None:
            valid_image = torch.ones(
                (hidden_states.shape[0], ni), dtype=torch.bool, device=hidden_states.device
            )
            full_mask = torch.cat(
                [encoder_hidden_states_mask, valid_image], dim=1
            )[:, None, None, :]
        native = attend(full_q, full_k, full_v, full_mask)

        reference_indices = torch.nonzero(
            ctl.reference_token_mask.to(tk.device), as_tuple=False
        ).flatten()
        if not reference_indices.numel():
            raise RuntimeError("E11 reference provenance mask contains no tokens")
        reference_k = tk.index_select(1, reference_indices)
        reference_v = tv.index_select(1, reference_indices)
        reference_state = attend(iq[:, :nx], reference_k, reference_v)

        query_descriptor = F.normalize(content_q.float().mean(dim=2), dim=-1)
        key_descriptor = F.normalize(
            content_k.index_select(1, reference_indices).float().mean(dim=2), dim=-1
        )
        confidence = torch.matmul(
            query_descriptor, key_descriptor.transpose(-1, -2)
        ).amax(dim=-1)
        gate = ctl.update_gate(self.layer_index, confidence)
        strength = ctl.injection_strength(self.layer_index)

        native_image = native[:, nt:]
        output_state = native_image[:, :nx]
        output_state = output_state + (
            strength * gate[:, :, None, None].to(output_state.dtype)
            * (reference_state - output_state)
        )
        native_image = native_image.clone()
        native_image[:, :nx] = output_state
        joint = torch.cat([native[:, :nt], native_image], dim=1).flatten(2, 3)
        joint = joint.to(hidden_states.dtype)
        text_out, image_out = joint[:, :nt], joint[:, nt:]
        image_out = attn.to_out[0](image_out.contiguous())
        if len(attn.to_out) > 1:
            image_out = attn.to_out[1](image_out)
        text_out = attn.to_add_out(text_out.contiguous())
        ctl.record(self.layer_index, gate, strength, reference_indices.numel())
        return image_out, text_out


class SelfLocalizingVLKV:
    def __init__(self, pipe, args):
        self.pipe = pipe
        self.args = args
        self.blocks = list(pipe.transformer.transformer_blocks)
        self.layers = parse_layer_spec(args.injection_layers, len(self.blocks))
        self.originals = {}
        self.reference_token_mask = torch.empty(0, dtype=torch.bool)
        self.ema_gates = {}
        self.layer_calls = {}
        self.measurements = []
        self.active = False

    def install(self):
        for index in self.layers:
            block = self.blocks[index]
            self.originals[index] = block.attn.processor
            block.attn.set_processor(
                SelfLocalizingVLKVProcessor(self, block.attn.processor, index)
            )

    def begin(self, reference_token_mask):
        self.reference_token_mask = reference_token_mask.detach().bool()
        self.ema_gates.clear()
        self.layer_calls.clear()
        self.measurements.clear()
        self.active = True

    def update_gate(self, layer, confidence):
        median = confidence.median(dim=1, keepdim=True).values
        mad = (confidence - median).abs().median(dim=1, keepdim=True).values.clamp_min(1e-5)
        robust = (confidence - median) / (1.4826 * mad)
        gate = torch.sigmoid((robust - self.args.gate_threshold) / self.args.gate_temperature)
        gh, gw = self.grid_shape(gate.shape[1])
        gate = F.avg_pool2d(
            gate.reshape(-1, 1, gh, gw),
            kernel_size=2 * self.args.gate_smoothing + 1,
            stride=1, padding=self.args.gate_smoothing,
        ).reshape_as(gate)
        if gate.mean() > self.args.max_gate_area:
            cutoff = torch.quantile(gate.float(), 1.0 - self.args.max_gate_area, dim=1, keepdim=True)
            gate = gate * (gate >= cutoff).to(gate.dtype)
        old = self.ema_gates.get(layer)
        gate = gate if old is None else self.args.gate_ema * old + (1 - self.args.gate_ema) * gate
        self.ema_gates[layer] = gate.detach()
        return gate

    def grid_shape(self, tokens):
        gw = max(1, round(math.sqrt(tokens * self.args.width / self.args.height)))
        gh = tokens // gw
        if gh * gw != tokens:
            raise RuntimeError(f"Cannot map {tokens} output tokens to a grid")
        return gh, gw

    def injection_strength(self, layer):
        call = self.layer_calls.get(layer, 0)
        self.layer_calls[layer] = call + 1
        step_phase = call / max(self.args.steps - 1, 1)
        if step_phase < self.args.injection_start:
            return 0.0
        temporal = math.sin(
            math.pi * (step_phase - self.args.injection_start)
            / max(1.0 - self.args.injection_start, 1e-6)
        ) ** 2
        position = self.layers.index(layer) / max(len(self.layers) - 1, 1)
        layer_envelope = .35 + .65 * math.sin(math.pi * position) ** 2
        return self.args.kv_strength * temporal * layer_envelope

    def record(self, layer, gate, strength, reference_tokens):
        self.measurements.append({
            "layer": layer,
            "call": self.layer_calls[layer] - 1,
            "strength": strength,
            "gate_mean": float(gate.mean().detach().cpu()),
            "gate_max": float(gate.max().detach().cpu()),
            "reference_tokens": int(reference_tokens),
        })

    def end(self, gate_path):
        self.active = False
        usable = [gate.float().cpu() for gate in self.ema_gates.values()]
        if not usable:
            raise RuntimeError("E11 injection produced no localization gates")
        gate = torch.stack(usable).mean(dim=0)[0]
        gh, gw = self.grid_shape(gate.numel())
        image = Image.fromarray(np.uint8(gate.reshape(gh, gw).numpy().clip(0, 1) * 255))
        image.resize((self.args.width, self.args.height), Image.Resampling.BILINEAR).save(gate_path)
        return {
            "layers": self.layers,
            "kv_strength": self.args.kv_strength,
            "gate": str(gate_path),
            "measurements": self.measurements,
        }

    def close(self):
        self.active = False
        for index, original in self.originals.items():
            self.blocks[index].attn.set_processor(original)


@torch.inference_mode()
def infer_injected(pipe, controller, base, reference, prompt, args, seed, gate_path):
    positive_images = prepare_vl_images(pipe, [base, reference])
    positive, positive_mask, provenance = encode_with_image_provenance(
        pipe, prompt, positive_images
    )
    baseline, baseline_mask = pipe.encode_prompt(
        prompt=prompt,
        image=prepare_vl_images(pipe, [base]),
        device=pipe._execution_device,
        num_images_per_prompt=1,
    )
    controller.begin(provenance[1])
    try:
        result = pipe(
            image=[base],
            prompt_embeds=positive,
            prompt_embeds_mask=positive_mask,
            negative_prompt_embeds=baseline,
            negative_prompt_embeds_mask=baseline_mask,
            true_cfg_scale=args.identity_guidance_scale,
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            generator=make_generator(args.device, seed),
        )
        diagnostics = controller.end(gate_path)
    finally:
        controller.active = False
    return result.images[0].convert("RGB"), diagnostics


def run_case(pipe, controller, case, references, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    history = []
    for index, item in enumerate(tqdm(
        case["objects"][: args.max_objects or None],
        desc=f"E11 case {case_id:03d}", unit="object", leave=False,
    ), 1):
        name, key = item["name"], reference_key(item)
        record = references.get(key, {})
        prefix = steps_dir / f"{index:02d}_{slug(name)}"
        final_path = Path(f"{prefix}_after.png")
        gate_path = Path(f"{prefix}_gate.png")
        if record.get("status") != "ready":
            if args.missing_policy == "error":
                raise FileNotFoundError(f"No reference available for {name}: {record}")
            history.append({"step": index, "name": name, "status": "skipped_missing_reference"})
            continue
        if args.resume and final_path.is_file():
            current = Image.open(final_path).convert("RGB")
            history.append({"step": index, "name": name, "status": "resumed", "final": str(final_path)})
            continue

        before = fit(current, (args.width, args.height))
        reference = fit(Image.open(record["image"]), (args.width, args.height))
        before_path = Path(f"{prefix}_before.png")
        reference_path = Path(f"{prefix}_vl_reference.png")
        before.save(before_path)
        reference.save(reference_path)
        prompt = (
            f"Image 1 is the current scene and the only output canvas. Add exactly one complete {name} "
            "into a physically plausible unoccupied location. When an additional reference image is present, "
            "it defines the inserted object's identity only. Match its distinctive shape, proportions, components, "
            "colors, materials, texture, and markings while adapting pose, scale, perspective, lighting, contact, "
            "shadow, and occlusion naturally. Preserve the background and every existing object. Never reproduce "
            "the reference framing or background, and return one scene image rather than a collage or grid."
        )
        seed = args.seed + case_id * 10000 + index * 100
        current, diagnostics = infer_injected(
            pipe, controller, before, reference, prompt, args, seed, gate_path
        )
        current.save(final_path)
        history.append({
            "step": index, "name": name, "status": "generated", "seed": seed,
            "vl_inputs": [str(before_path), str(reference_path)],
            "vae_inputs": [str(before_path)], "reference_source": record["image"],
            "injection": diagnostics, "final": str(final_path), "postprocess": None,
        })
        save_json(history, case_dir / "history.json")
    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "status": "complete", "objects": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    parser.add_argument("--out_dir", default="results/qwen_e11_self_localizing_vl_kv")
    parser.add_argument("--case_ids", type=int, nargs="+")
    parser.add_argument("--max_objects", type=int, default=3, choices=(1, 2, 3))
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
    parser.add_argument("--identity_guidance_scale", type=float, default=1.35)
    parser.add_argument("--injection_layers", default="middle")
    parser.add_argument("--kv_strength", type=float, default=.22)
    parser.add_argument("--injection_start", type=float, default=.25)
    parser.add_argument("--gate_threshold", type=float, default=1.0)
    parser.add_argument("--gate_temperature", type=float, default=.35)
    parser.add_argument("--gate_smoothing", type=int, default=2)
    parser.add_argument("--gate_ema", type=float, default=.8)
    parser.add_argument("--max_gate_area", type=float, default=.35)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.identity_guidance_scale <= 1 or not 0 <= args.kv_strength <= 1:
        raise ValueError("identity_guidance_scale must exceed 1 and kv_strength must be in [0,1]")
    if not 0 <= args.injection_start < 1 or args.gate_temperature <= 0:
        raise ValueError("invalid injection schedule or gate temperature")
    if not 0 <= args.gate_ema < 1 or not 0 < args.max_gate_area <= 1:
        raise ValueError("invalid gate EMA or maximum area")
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E11 targets diffusers 0.40.0; found {diffusers.__version__}")
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E11") from exc

    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")
    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    controller = SelfLocalizingVLKV(pipe, args)
    controller.install()
    summary = []
    try:
        for case in tqdm(cases, desc="E11 self-localizing VL-KV suite", unit="case"):
            summary.append(run_case(pipe, controller, case, references, args, out))
            save_json(summary, out / "summary.partial.json")
    finally:
        controller.close()
    save_json({
        "method": "asymmetric VL/VAE plus provenance-aware self-localizing VL-KV injection",
        "vl_images": ["current scene", "reference object"], "vae_images": ["current scene"],
        "external_mask": None, "postprocess": None, "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
