"""E8: training-free masked object-attention insertion.

The current scene B is Qwen's only semantic image. The denoiser receives B plus
an RMBG-cleaned private object bank O. In selected transformer layers, queries
inside an automatically estimated placement mask attend with one normalized
softmax over concatenated [text, output, base, object-foreground] K/V. Outside
queries receive -inf on all object logits and are exactly base-only attention.

This is an exploratory inference-time ablation, not a trained reference adapter.
No collage, feature addition, pixel compositing, or post-generation pass exists.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops
from tqdm.auto import tqdm

from e1_baseline import load_pipe, make_generator, save_json
from e2_sam_collage_repaint import place_cutout, probe_placement
from e3_prompt_suite import generate_references, load_suite, reference_key, select_cases, slug
from e5_spatial_kv_collage import (
    RMBG2Cutout,
    generate_base,
    generate_rmbg_cutouts,
    parse_layer_spec,
)


HERE = Path(__file__).resolve().parent


def aligned_object_bank(cutout, box, size, args):
    """Place O on a neutral canvas at its intended output coordinates."""
    raw_canvas, alpha, placed_box = place_cutout(cutout, box, size, args.object_scale)
    bank = Image.new("RGB", size, args.object_condition_background)
    # Alpha compositing removes source-background pixels from the RGB bank;
    # alpha itself is retained separately for foreground-token selection.
    bank.paste(raw_canvas, (0, 0), alpha)
    return bank, alpha, placed_box


@torch.inference_mode()
def infer_base_with_private_object(pipe, base, object_image, prompt, args, seed):
    """Encode B semantically while supplying [B,O] only to the denoiser."""
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        calculate_dimensions,
    )

    width, height = calculate_dimensions(CONDITION_IMAGE_SIZE, base.width / base.height)
    semantic_base = [pipe.image_processor.resize(base, height, width)]
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt=prompt, image=semantic_base, device=pipe._execution_device,
        num_images_per_prompt=1,
    )
    negative_embeds = negative_mask = None
    if args.true_cfg_scale > 1 and args.negative_prompt is not None:
        negative_embeds, negative_mask = pipe.encode_prompt(
            prompt=args.negative_prompt, image=semantic_base,
            device=pipe._execution_device, num_images_per_prompt=1,
        )
    result = pipe(
        image=[base, object_image],
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


class MaskedObjectAttentionProcessor:
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
        if not ctl.active:
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, attention_mask, image_rotary_emb, **kwargs,
            )
        if encoder_hidden_states is None or attention_mask is not None:
            raise RuntimeError("E8 requires standard Qwen double-stream attention")

        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE

        nt = encoder_hidden_states.shape[1]
        ni = hidden_states.shape[1]
        nx = (ctl.args.height // 16) * (ctl.args.width // 16)
        remainder = ni - nx
        if remainder <= 0 or remainder % 2:
            raise RuntimeError(f"Unexpected X/B/O token layout: {ni}")
        nc = remainder // 2
        if nc != nx:
            raise RuntimeError(f"E8 requires equal grids; output={nx}, condition={nc}")
        b = slice(nx, 2 * nx)
        o = slice(2 * nx, 3 * nx)

        iq, ik, iv = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        tq = attn.add_q_proj(encoder_hidden_states)
        tk = attn.add_k_proj(encoder_hidden_states)
        tv = attn.add_v_proj(encoder_hidden_states)
        hd = attn.inner_dim // attn.heads
        iq, ik, iv = [value.unflatten(-1, (-1, hd)) for value in (iq, ik, iv)]
        tq, tk, tv = [value.unflatten(-1, (-1, hd)) for value in (tq, tk, tv)]
        if attn.norm_q is not None:
            iq = attn.norm_q(iq)
        if attn.norm_k is not None:
            ik = attn.norm_k(ik)
        if attn.norm_added_q is not None:
            tq = attn.norm_added_q(tq)
        if attn.norm_added_k is not None:
            tk = attn.norm_added_k(tk)
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

        def valid_mask(count):
            if encoder_hidden_states_mask is None:
                return None
            valid = torch.ones(
                (hidden_states.shape[0], count), dtype=torch.bool,
                device=hidden_states.device,
            )
            return torch.cat([encoder_hidden_states_mask, valid], dim=1)[:, None, None, :]

        # Native-equivalent main stream T/X/B. O is excluded completely.
        main_q = torch.cat([tq, iq[:, :nx], iq[:, b]], dim=1)
        main_k = torch.cat([tk, ik[:, :nx], ik[:, b]], dim=1)
        main_v = torch.cat([tv, iv[:, :nx], iv[:, b]], dim=1)
        main = attend(main_q, main_k, main_v, valid_mask(2 * nx))
        main_image = main[:, nt:]

        foreground = ctl.object_foreground_indices(nx, ik.device)
        # O evolves only from text and its own confident foreground memory.
        object_k = torch.cat([tk, ik[:, o].index_select(1, foreground)], dim=1)
        object_v = torch.cat([tv, iv[:, o].index_select(1, foreground)], dim=1)
        object_state = attend(
            iq[:, o], object_k, object_v, valid_mask(int(foreground.numel()))
        )

        if self.layer_index in ctl.layers:
            # One normalized softmax over concatenated base and object memory.
            routed_k = torch.cat([
                tk, ik[:, :nx], ik[:, b], ik[:, o].index_select(1, foreground)
            ], dim=1)
            routed_v = torch.cat([
                tv, iv[:, :nx], iv[:, b], iv[:, o].index_select(1, foreground)
            ], dim=1)
            bias = ctl.attention_bias(
                nt, nx, int(foreground.numel()), encoder_hidden_states_mask,
                iq.device, iq.dtype,
            )
            output_state = attend(iq[:, :nx], routed_k, routed_v, bias)
        else:
            output_state = main_image[:, :nx]

        image_state = torch.empty_like(iq)
        image_state[:, :nx] = output_state
        image_state[:, b] = main_image[:, nx:]
        image_state[:, o] = object_state
        joint = torch.cat([main[:, :nt], image_state], dim=1).flatten(2, 3).to(hidden_states.dtype)
        text_out, image_out = joint[:, :nt], joint[:, nt:]
        image_out = attn.to_out[0](image_out.contiguous())
        if len(attn.to_out) > 1:
            image_out = attn.to_out[1](image_out)
        text_out = attn.to_add_out(text_out.contiguous())
        ctl.calls += 1
        ctl.layout = {"output_tokens": nx, "object_foreground_tokens": int(foreground.numel())}
        return image_out, text_out


class MaskedObjectAttention:
    def __init__(self, pipe, args):
        self.args = args
        self.blocks = list(pipe.transformer.transformer_blocks)
        self.layers = parse_layer_spec(args.injection_layers, len(self.blocks))
        self.originals = {}
        self.mask = None
        self.object_alpha = None
        self.gate_cache = {}
        self.foreground_cache = {}
        self.bias_cache = {}
        self.active = False
        self.calls = 0
        self.layout = None

    @staticmethod
    def grid(tokens, width, height):
        gw = max(1, round(math.sqrt(tokens * width / height)))
        gh = tokens // gw
        if gh * gw != tokens:
            raise RuntimeError(f"Cannot map {tokens} tokens to a grid")
        return gh, gw

    def set_inputs(self, mask, object_alpha):
        self.mask = np.asarray(mask.convert("L"), dtype=np.float32).copy() / 255.0
        self.object_alpha = np.asarray(object_alpha.convert("L"), dtype=np.float32).copy() / 255.0
        self.gate_cache.clear()
        self.foreground_cache.clear()
        self.bias_cache.clear()

    def gate(self, tokens, device):
        key = (tokens, str(device))
        if key in self.gate_cache:
            return self.gate_cache[key]
        h, w = self.mask.shape
        gh, gw = self.grid(tokens, w, h)
        resized = Image.fromarray(np.uint8(np.clip(self.mask, 0, 1) * 255)).resize(
            (gw, gh), Image.Resampling.BOX
        )
        result = torch.from_numpy(
            np.asarray(resized, dtype=np.uint8).copy().reshape(-1) >= self.args.gate_threshold
        ).to(device)
        self.gate_cache[key] = result
        return result

    def object_foreground_indices(self, tokens, device):
        key = (tokens, str(device))
        if key in self.foreground_cache:
            return self.foreground_cache[key]
        h, w = self.object_alpha.shape
        gh, gw = self.grid(tokens, w, h)
        resized = Image.fromarray(np.uint8(np.clip(self.object_alpha, 0, 1) * 255)).resize(
            (gw, gh), Image.Resampling.BOX
        )
        values = np.asarray(resized, dtype=np.float32).copy().reshape(-1) / 255.0
        indices = np.flatnonzero(values >= self.args.object_token_alpha_threshold)
        if not len(indices):
            raise RuntimeError("No confident object foreground tokens")
        result = torch.as_tensor(indices, dtype=torch.long, device=device)
        self.foreground_cache[key] = result
        return result

    def attention_bias(self, nt, nx, no, text_mask, device, dtype):
        signature = None if text_mask is None else tuple(
            text_mask[0].detach().to(device="cpu", dtype=torch.uint8).tolist()
        )
        key = (nt, nx, no, signature, str(device), dtype)
        if key in self.bias_cache:
            return self.bias_cache[key]
        inside = self.gate(nx, device)
        total = nt + 2 * nx + no
        bias = torch.zeros((1, 1, nx, total), dtype=torch.float32, device=device)
        object_start = nt + 2 * nx
        bias[:, :, ~inside, object_start:] = -torch.inf
        # Convert a desired *source mass ratio* into a per-token logit bias.
        # Without this correction, a small O bank is overwhelmed by 2*nx main
        # image tokens even when every O token has the same individual logit.
        per_token_ratio = self.args.object_attention_mass * (2 * nx) / max(no, 1)
        bias[:, :, inside, object_start:] = math.log(per_token_ratio)
        if text_mask is not None:
            invalid = ~text_mask[0].to(device=device, dtype=torch.bool)
            bias[:, :, :, :nt][..., invalid] = -torch.inf
        result = bias.to(dtype)
        self.bias_cache[key] = result
        return result

    def install(self):
        for index, block in enumerate(self.blocks):
            self.originals[index] = block.attn.processor
            block.attn.set_processor(
                MaskedObjectAttentionProcessor(self, block.attn.processor, index)
            )

    def begin(self):
        if self.mask is None or self.object_alpha is None:
            raise RuntimeError("Set placement and object masks before inference")
        self.calls, self.layout, self.active = 0, None, True

    def end(self):
        self.active = False
        cfg = 2 if self.args.true_cfg_scale > 1 and self.args.negative_prompt is not None else 1
        expected = len(self.blocks) * self.args.steps * cfg
        if self.calls != expected:
            raise RuntimeError(f"E8 attention ran {self.calls} calls; expected {expected}")
        return {
            "mechanism": "masked concatenated base/object K/V with one softmax",
            "injection_layers": self.layers,
            "object_attention_mass": self.args.object_attention_mass,
            "processor_calls": self.calls,
            "layout": self.layout,
        }

    def close(self):
        self.active = False
        for index, original in self.originals.items():
            self.blocks[index].attn.set_processor(original)


def run_case(pipe, controller, case, references, cutouts, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    step_dir = case_dir / "steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    occupied = Image.new("L", current.size)
    history = []
    for index, item in enumerate(tqdm(
        case["objects"][:args.max_objects], desc=f"E8 case {case_id:03d}", unit="object", leave=False
    ), 1):
        name, key = item["name"], reference_key(item)
        final = step_dir / f"{index:02d}_{slug(name)}_aligned_mass_attention_final.png"
        mask_file = step_dir / f"{index:02d}_{slug(name)}_placement_mask.png"
        if args.resume and final.is_file() and mask_file.is_file():
            current = Image.open(final).convert("RGB")
            occupied = ImageChops.lighter(occupied, Image.open(mask_file).convert("L"))
            history.append({"step": index, "name": name, "status": "resumed", "final": str(final)})
            continue
        record = references.get(key, {})
        if record.get("status") != "ready":
            if args.missing_policy == "skip":
                history.append({"step": index, "name": name, "status": "skipped_missing_reference"})
                continue
            raise FileNotFoundError(record)
        box, probe = probe_placement(
            pipe, current, name, cutouts[key], occupied, args,
            args.seed + case_id * 100000 + index * 1000,
            step_dir / f"{index:02d}_{slug(name)}_placement_heatmap.png",
        )
        object_image, object_alpha, placed_box = aligned_object_bank(
            cutouts[key], box, current.size, args
        )
        gate = object_alpha
        current.save(step_dir / f"{index:02d}_{slug(name)}_base_input.png")
        gate.save(mask_file)
        object_image.save(step_dir / f"{index:02d}_{slug(name)}_private_object_bank.png")
        object_alpha.save(step_dir / f"{index:02d}_{slug(name)}_object_alpha.png")
        controller.set_inputs(gate, object_alpha)
        controller.begin()
        seed = args.seed + case_id * 10000 + index * 100
        try:
            output = infer_base_with_private_object(
                pipe, current, object_image,
                f"Add exactly one complete {name} naturally at the indicated insertion region in Image 1. "
                f"Preserve the existing scene and use the supplied visual feature reference for the {name}'s "
                "identity, structure, colors and material. Keep it fully inside the frame.",
                args, seed,
            )
        finally:
            controller.active = False
        intervention = controller.end()
        current = output
        current.save(final)
        occupied = ImageChops.lighter(occupied, gate)
        history.append({
            "step": index, "name": name, "status": "generated", "seed": seed,
            "placement_box": list(placed_box), "placement_probe": probe,
            "attention": intervention, "final": str(final), "postprocess": None,
        })
        save_json(history, case_dir / "history.json")
    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "status": "complete", "objects": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    p.add_argument("--out_dir", default="results/qwen_e8_masked_object_attention")
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
    p.add_argument("--object_alpha_low", type=float, default=.15)
    p.add_argument("--object_alpha_high", type=float, default=.85)
    p.add_argument("--object_token_alpha_threshold", type=float, default=.35)
    p.add_argument("--object_condition_scale", type=float, default=.82)
    p.add_argument("--object_condition_background", type=int, default=127)
    p.add_argument("--probe_steps", type=int, default=4)
    p.add_argument("--probe_quantile", type=float, default=.88)
    p.add_argument("--probe_blur", type=float, default=1.2)
    p.add_argument("--box_margin", type=int, default=24)
    p.add_argument("--occupancy_margin", type=int, default=24)
    p.add_argument("--default_object_height", type=float, default=.25)
    p.add_argument("--object_height_priors")
    p.add_argument("--injection_layers", default="6-35")
    p.add_argument("--object_attention_mass", type=float, default=.35, help="Desired O-to-main attention mass ratio inside the placement region")
    p.add_argument("--gate_threshold", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    if args.object_attention_mass <= 0:
        raise ValueError("object_attention_mass must be positive")
    if not 0 <= args.gate_threshold <= 255:
        raise ValueError("gate_threshold must be in [0,255]")
    if not 0 <= args.object_alpha_low < args.object_alpha_high <= 1:
        raise ValueError("object alpha limits must satisfy 0 <= low < high <= 1")
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E8 targets diffusers 0.40.0; found {diffusers.__version__}")
        import kornia  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E8") from exc
    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")
    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    segmenter = RMBG2Cutout(
        args.rmbg_model_id, args.rmbg_revision, args.rmbg_device, args.rmbg_input_size
    )
    try:
        cutouts = generate_rmbg_cutouts(segmenter, references, args, out)
    finally:
        segmenter.close()
    controller = MaskedObjectAttention(pipe, args)
    controller.install()
    summary = []
    try:
        for case in tqdm(cases, desc="E8 masked-attention suite", unit="case"):
            summary.append(run_case(pipe, controller, case, references, cutouts, args, out))
            save_json(summary, out / "summary.partial.json")
    finally:
        controller.close()
    save_json({
        "method": "training-free masked object-attention insertion",
        "semantic_inputs": ["base"],
        "denoiser_inputs": ["base", "private_object_bank"],
        "feature_addition": False,
        "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
