"""E6: block-sparse, source-aware reference attention.

For each insertion, Qwen's semantic encoder sees only collage C. The denoiser
receives [C, O, B], where O is an RMBG foreground bank and B is a preservation
bank. A single normalized attention selects C everywhere, local B outside the
edit region, and foreground O inside it. No raw feature addition, output blend,
second generation, or pixel postprocess is used.

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

from e1_baseline import fit, load_pipe, save_json
from e2_sam_collage_repaint import composite, place_cutout, probe_placement
from e3_prompt_suite import generate_references, load_suite, reference_key, select_cases, slug
from e5_spatial_kv_collage import (
    RMBG2Cutout,
    generate_base,
    generate_rmbg_cutouts,
    infer_with_latent_base,
    object_condition,
    parse_layer_spec,
)


HERE = Path(__file__).resolve().parent


class BlockSparseProcessor:
    """Partitioned attention with one normalized routed softmax for X queries."""

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
            raise RuntimeError("E6 requires standard Qwen double-stream attention")

        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE

        text_tokens = encoder_hidden_states.shape[1]
        image_tokens = hidden_states.shape[1]
        output_tokens = (ctl.args.height // 16) * (ctl.args.width // 16)
        remainder = image_tokens - output_tokens
        if remainder <= 0 or remainder % 3:
            raise RuntimeError(
                f"Unexpected X/C/O/B layout ({image_tokens} image tokens)"
            )
        condition_tokens = remainder // 3
        if condition_tokens != output_tokens:
            raise RuntimeError(
                f"E6 needs equal square grids: X={output_tokens}, condition={condition_tokens}"
            )

        c = slice(output_tokens, 2 * output_tokens)
        o = slice(2 * output_tokens, 3 * output_tokens)
        b = slice(3 * output_tokens, 4 * output_tokens)
        iq, ik, iv = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        tq = attn.add_q_proj(encoder_hidden_states)
        tk = attn.add_k_proj(encoder_hidden_states)
        tv = attn.add_v_proj(encoder_hidden_states)
        head_dim = attn.inner_dim // attn.heads
        iq, ik, iv = [value.unflatten(-1, (-1, head_dim)) for value in (iq, ik, iv)]
        tq, tk, tv = [value.unflatten(-1, (-1, head_dim)) for value in (tq, tk, tv)]
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

        def padding_mask(valid_image_tokens):
            if encoder_hidden_states_mask is None:
                return None
            valid = torch.ones(
                (hidden_states.shape[0], valid_image_tokens),
                dtype=torch.bool, device=hidden_states.device,
            )
            return torch.cat([encoder_hidden_states_mask, valid], dim=1)[:, None, None, :]

        # Main pretrained-like stream: T, noisy output X, and collage C only.
        main_q = torch.cat([tq, iq[:, :output_tokens], iq[:, c]], dim=1)
        main_k = torch.cat([tk, ik[:, :output_tokens], ik[:, c]], dim=1)
        main_v = torch.cat([tv, iv[:, :output_tokens], iv[:, c]], dim=1)
        main = attend(main_q, main_k, main_v, padding_mask(2 * output_tokens))
        main_image = main[:, text_tokens:]

        foreground = ctl.object_foreground_indices(output_tokens, ik.device)
        # O remains a private foreground-only memory across blocks.
        ok = torch.cat([tk, ik[:, o].index_select(1, foreground)], dim=1)
        ov = torch.cat([tv, iv[:, o].index_select(1, foreground)], dim=1)
        o_state = attend(iq[:, o], ok, ov, padding_mask(int(foreground.numel())))
        # B remains a private preservation memory across blocks.
        bk = torch.cat([tk, ik[:, b]], dim=1)
        bv = torch.cat([tv, iv[:, b]], dim=1)
        b_state = attend(iq[:, b], bk, bv, padding_mask(output_tokens))

        if self.layer_index in ctl.layers:
            # One K/V concatenation and one softmax. The additive bias carries
            # both the hard block-sparse mask and log source priors.
            routed_k = torch.cat([
                tk, ik[:, :output_tokens], ik[:, c],
                ik[:, o].index_select(1, foreground), ik[:, b],
            ], dim=1)
            routed_v = torch.cat([
                tv, iv[:, :output_tokens], iv[:, c],
                iv[:, o].index_select(1, foreground), iv[:, b],
            ], dim=1)
            bias = ctl.routing_bias(
                text_tokens, output_tokens, foreground.numel(),
                encoder_hidden_states_mask, iq.device, iq.dtype,
            )
            x_state = attend(iq[:, :output_tokens], routed_k, routed_v, bias)
        else:
            x_state = main_image[:, :output_tokens]

        image_out = torch.empty_like(iq)
        image_out[:, :output_tokens] = x_state
        image_out[:, c] = main_image[:, output_tokens:]
        image_out[:, o] = o_state
        image_out[:, b] = b_state
        joint = torch.cat([main[:, :text_tokens], image_out], dim=1)
        joint = joint.flatten(2, 3).to(hidden_states.dtype)
        text_out, image_out = joint[:, :text_tokens], joint[:, text_tokens:]
        image_out = attn.to_out[0](image_out.contiguous())
        if len(attn.to_out) > 1:
            image_out = attn.to_out[1](image_out)
        text_out = attn.to_add_out(text_out.contiguous())
        ctl.calls += 1
        ctl.layout = {
            "output_tokens": output_tokens,
            "foreground_tokens": int(foreground.numel()),
            "routed_key_tokens": int(routed_k.shape[1]) if self.layer_index in ctl.layers else None,
        }
        return image_out, text_out


class BlockSparseReferenceAttention:
    def __init__(self, pipe, args):
        self.args = args
        self.blocks = list(pipe.transformer.transformer_blocks)
        self.layers = parse_layer_spec(args.routing_layers, len(self.blocks))
        self.originals = {}
        self.gate = None
        self.object_alpha = None
        self.bias_cache = {}
        self.foreground_cache = {}
        self.active = False
        self.calls = 0
        self.layout = None

    def set_masks(self, paste_mask, object_alpha):
        gate = paste_mask.convert("L")
        if self.args.gate_blur > 0:
            gate = gate.filter(ImageFilter.GaussianBlur(self.args.gate_blur))
        self.gate = np.asarray(gate, dtype=np.float32).copy() / 255.0
        self.object_alpha = np.asarray(object_alpha.convert("L"), dtype=np.float32).copy() / 255.0
        self.bias_cache.clear()
        self.foreground_cache.clear()
        return gate

    @staticmethod
    def grid_shape(tokens, width, height):
        grid_w = max(1, round(math.sqrt(tokens * width / height)))
        grid_h = tokens // grid_w
        if grid_h * grid_w != tokens:
            raise RuntimeError(f"Cannot map {tokens} tokens to a spatial grid")
        return grid_h, grid_w

    def object_foreground_indices(self, tokens, device):
        key = (tokens, str(device))
        if key in self.foreground_cache:
            return self.foreground_cache[key]
        h, w = self.object_alpha.shape
        gh, gw = self.grid_shape(tokens, w, h)
        alpha = Image.fromarray(np.uint8(np.clip(self.object_alpha, 0, 1) * 255)).resize(
            (gw, gh), Image.Resampling.BOX
        )
        values = np.asarray(alpha, dtype=np.float32).copy().reshape(-1) / 255.0
        indices = np.flatnonzero(values >= self.args.object_token_alpha_threshold)
        if not len(indices):
            raise RuntimeError("No confident RMBG foreground tokens for the object bank")
        result = torch.as_tensor(indices, dtype=torch.long, device=device)
        self.foreground_cache[key] = result
        return result

    def routing_bias(self, text_tokens, tokens, foreground_tokens, text_mask, device, dtype):
        mask_signature = None if text_mask is None else tuple(
            text_mask[0].detach().to(device="cpu", dtype=torch.uint8).tolist()
        )
        key = (text_tokens, tokens, int(foreground_tokens), mask_signature, str(device), dtype)
        if key in self.bias_cache:
            return self.bias_cache[key]
        h, w = self.gate.shape
        gh, gw = self.grid_shape(tokens, w, h)
        gate = Image.fromarray(np.uint8(np.clip(self.gate, 0, 1) * 255)).resize(
            (gw, gh), Image.Resampling.BOX
        )
        inside = torch.from_numpy(
            (np.asarray(gate, dtype=np.uint8).copy().reshape(-1) >= self.args.gate_threshold)
        ).to(device)

        total_keys = text_tokens + 3 * tokens + int(foreground_tokens)
        # float32 avoids finite bf16 minima leaking through softmax. The
        # dispatcher may cast internally while retaining exact -inf blocks.
        bias = torch.zeros((1, 1, tokens, total_keys), dtype=torch.float32, device=device)
        object_start = text_tokens + 2 * tokens
        base_start = object_start + int(foreground_tokens)
        bias[:, :, ~inside, object_start:base_start] = -torch.inf
        bias[:, :, inside, object_start:base_start] = math.log(self.args.object_attention_prior)
        bias[:, :, inside, base_start:] = -torch.inf

        # Outside queries may read only spatially corresponding local B keys.
        radius = self.args.base_local_radius
        local = torch.zeros((tokens, tokens), dtype=torch.bool, device=device)
        outside_indices = torch.nonzero(~inside, as_tuple=False).flatten()
        for query_index in outside_indices.tolist():
            y, x = divmod(query_index, gw)
            for yy in range(max(0, y - radius), min(gh, y + radius + 1)):
                start = yy * gw + max(0, x - radius)
                stop = yy * gw + min(gw, x + radius + 1)
                local[query_index, start:stop] = True
        base_bias = bias[0, 0, :, base_start:]
        base_bias[~local] = -torch.inf
        base_bias[local] = math.log(self.args.base_attention_prior)
        if text_mask is not None:
            invalid = ~text_mask[0].to(device=device, dtype=torch.bool)
            text_bias = bias[:, :, :, :text_tokens]
            text_bias[..., invalid] = -torch.inf
        result = bias.to(dtype=dtype)
        self.bias_cache[key] = result
        return result

    def install(self):
        # All blocks isolate O/B; only selected blocks expose them to X queries.
        for index, block in enumerate(self.blocks):
            self.originals[index] = block.attn.processor
            block.attn.set_processor(BlockSparseProcessor(self, block.attn.processor, index))

    def begin(self):
        if self.gate is None or self.object_alpha is None:
            raise RuntimeError("Call set_masks before E6 inference")
        self.calls, self.layout, self.active = 0, None, True

    def end(self):
        self.active = False
        cfg_passes = 2 if self.args.true_cfg_scale > 1 and self.args.negative_prompt is not None else 1
        expected = len(self.blocks) * self.args.steps * cfg_passes
        if self.calls != expected:
            raise RuntimeError(f"Block-sparse attention ran {self.calls} times; expected {expected}")
        return {
            "mechanism": "single-softmax block-sparse concatenated K/V attention",
            "isolation_layers": list(range(len(self.blocks))),
            "routing_layers": self.layers,
            "base_prior": self.args.base_attention_prior,
            "object_prior": self.args.object_attention_prior,
            "base_local_radius": self.args.base_local_radius,
            "processor_calls": self.calls,
            "token_layout": self.layout,
        }

    def close(self):
        self.active = False
        for index, original in self.originals.items():
            self.blocks[index].attn.set_processor(original)


def run_case(pipe, router, case, references, cutouts, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    occupied = Image.new("L", current.size)
    history = []
    for index, item in enumerate(tqdm(
        case["objects"][:args.max_objects], desc=f"E6 case {case_id:03d}", unit="object", leave=False
    ), 1):
        name, key = item["name"], reference_key(item)
        final_path = steps_dir / f"{index:02d}_{slug(name)}_block_sparse_final.png"
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
        box, probe = probe_placement(
            pipe, before, name, cutouts[key], occupied, args,
            args.seed + case_id * 100000 + index * 1000,
            steps_dir / f"{index:02d}_{slug(name)}_placement_heatmap.png",
        )
        object_canvas, paste_mask, placed_box = place_cutout(
            cutouts[key], box, before.size, args.object_scale
        )
        collage = composite(before, object_canvas, paste_mask)
        object_image, object_alpha = object_condition(cutouts[key], args)
        before.save(steps_dir / f"{index:02d}_{slug(name)}_base_bank.png")
        collage.save(steps_dir / f"{index:02d}_{slug(name)}_collage.png")
        object_image.save(steps_dir / f"{index:02d}_{slug(name)}_object_bank.png")
        object_alpha.save(steps_dir / f"{index:02d}_{slug(name)}_object_alpha.png")
        paste_mask.save(mask_path)
        router.set_masks(paste_mask, object_alpha).save(
            steps_dir / f"{index:02d}_{slug(name)}_query_gate.png"
        )
        prompt = (
            f"Image 1 is a scene collage containing one pasted {name}. Return one natural edited version of "
            f"Image 1. Keep its composition and the {name}'s exact position, geometry, colors and material. "
            "Harmonize only its lighting, contact and boundary. Do not move, replace, duplicate, or redesign it. "
            "Do not output a panel, border, collage, isolated object, or split image."
        )
        router.begin()
        try:
            current = infer_with_latent_base(
                pipe, collage, object_image, before, prompt, args,
                args.seed + case_id * 10000 + index * 100,
            )
        finally:
            router.active = False
        intervention = router.end()
        current.save(final_path)
        occupied = ImageChops.lighter(occupied, paste_mask)
        history.append({
            "step": index, "name": name, "status": "generated", "final": str(final_path),
            "placed_box": list(placed_box), "probe": probe, "attention": intervention,
            "postprocess": None,
        })
        save_json(history, case_dir / "history.json")
    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "status": "complete", "objects": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    p.add_argument("--out_dir", default="results/qwen_e6_block_sparse_attention")
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
    p.add_argument("--object_scale", type=float, default=.92)
    p.add_argument("--routing_layers", default="middle")
    p.add_argument("--gate_threshold", type=int, default=96, help="0-255 pasted-region threshold")
    p.add_argument("--gate_blur", type=float, default=.75)
    p.add_argument("--base_local_radius", type=int, default=1, help="B-key radius on the latent grid")
    p.add_argument("--base_attention_prior", type=float, default=.10)
    p.add_argument("--object_attention_prior", type=float, default=.30)
    return p.parse_args()


def main():
    args = parse_args()
    if args.base_attention_prior <= 0 or args.object_attention_prior <= 0:
        raise ValueError("Attention priors must be positive")
    if args.base_local_radius < 0:
        raise ValueError("base_local_radius must be non-negative")
    if not 0 <= args.gate_threshold <= 255:
        raise ValueError("gate_threshold must be in [0,255]")
    if not 0 <= args.object_alpha_low < args.object_alpha_high <= 1:
        raise ValueError("object alpha limits must satisfy 0 <= low < high <= 1")
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E6 targets diffusers 0.40.0; found {diffusers.__version__}")
        import kornia  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E6") from exc

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
    router = BlockSparseReferenceAttention(pipe, args)
    router.install()
    summary = []
    try:
        for case in tqdm(cases, desc="E6 block-sparse suite", unit="case"):
            summary.append(run_case(pipe, router, case, references, cutouts, args, out))
            save_json(summary, out / "summary.partial.json")
    finally:
        router.close()
    save_json({
        "mechanism": "block-sparse source-aware normalized attention",
        "semantic_inputs": ["collage"],
        "latent_banks": ["collage", "object_foreground", "base_local"],
        "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
