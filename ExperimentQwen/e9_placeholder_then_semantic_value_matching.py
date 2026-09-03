"""E9: natural placeholder geometry followed by semantic value matching.

Pass 1 asks native Qwen to add a generic category instance, establishing scene-
consistent placement, pose, perspective, contact and occlusion. Pass 2 conditions
on the placeholder plus the clean reference. In selected MMDiT layers, target
queries and keys remain unchanged; only target-object values are interpolated
with semantically matched reference-foreground values. Reference keys never
enter output attention. The raw second-pass image is final.

Inspired by DreamMatcher's separation of Q/K structure and V appearance. This
is a training-free Qwen ablation, not a reproduction of DreamMatcher.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageFilter
from tqdm.auto import tqdm

from e1_baseline import infer, load_pipe, save_json
from e2_sam_collage_repaint import probe_placement
from e3_prompt_suite import generate_references, load_suite, reference_key, select_cases, slug
from e5_spatial_kv_collage import (
    RMBG2Cutout,
    generate_base,
    generate_rmbg_cutouts,
    object_condition,
    parse_layer_spec,
)


HERE = Path(__file__).resolve().parent


def region_words(box, size):
    cx = (box[0] + box[2]) / (2 * size[0])
    cy = (box[1] + box[3]) / (2 * size[1])
    horizontal = "left" if cx < .4 else "right" if cx > .6 else "center"
    vertical = "upper" if cy < .38 else "lower" if cy > .62 else "middle"
    return f"the {vertical}-{horizontal} region"


def placeholder_mask(before, placeholder, box, args):
    """Localize placeholder changes inside an expanded placement proposal."""
    # Float32 is intentional: squaring an int16 RGB delta overflows above 181.
    a = np.asarray(before.convert("RGB"), dtype=np.float32)
    b = np.asarray(placeholder.convert("RGB"), dtype=np.float32)
    delta = np.sqrt(np.square(a - b).sum(axis=2))
    x0, y0, x1, y1 = map(int, box)
    mx = round((x1 - x0) * args.target_box_margin)
    my = round((y1 - y0) * args.target_box_margin)
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(before.width, x1 + mx), min(before.height, y1 + my)
    support = np.zeros(delta.shape, dtype=np.uint8)
    support[y0:y1, x0:x1] = np.uint8(
        delta[y0:y1, x0:x1] >= args.placeholder_difference_threshold
    ) * 255
    mask = Image.fromarray(support)
    if mask.getbbox() is None:
        support[y0:y1, x0:x1] = 255
        mask = Image.fromarray(support)
    if args.target_mask_dilation > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * args.target_mask_dilation + 1))
    if args.target_mask_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(args.target_mask_blur))
    return mask


class SemanticValueMatchingProcessor:
    """Native target attention with correspondence-matched reference values."""

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
            raise RuntimeError("E9 requires standard Qwen double-stream attention")

        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE

        nt = encoder_hidden_states.shape[1]
        ni = hidden_states.shape[1]
        nx = (ctl.args.height // 16) * (ctl.args.width // 16)
        remainder = ni - nx
        if remainder <= 0 or remainder % 2:
            raise RuntimeError(f"Unexpected X/T/R layout: {ni} tokens")
        nc = remainder // 2
        if nc != nx:
            raise RuntimeError(f"E9 requires equal token grids: X={nx}, condition={nc}")
        target = slice(nx, 2 * nx)
        reference = slice(2 * nx, 3 * nx)

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
        # Match using content projections before positional rotation. RoPE is
        # subsequently retained for the actual target structure attention.
        match_key = ik
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

        def valid_mask(image_count):
            if encoder_hidden_states_mask is None:
                return None
            valid = torch.ones(
                (hidden_states.shape[0], image_count), dtype=torch.bool,
                device=hidden_states.device,
            )
            return torch.cat([encoder_hidden_states_mask, valid], dim=1)[:, None, None, :]

        target_values = iv[:, target].clone()
        target_indices = ctl.target_indices(nx, iv.device)
        reference_indices = ctl.reference_indices(nx, iv.device)
        matched_count = 0
        mean_similarity = 0.0
        if self.layer_index in ctl.layers and target_indices.numel() and reference_indices.numel():
            # Mean across heads yields content descriptors without assuming
            # correspondence between head-specific coordinate subspaces.
            target_features = F.normalize(
                match_key[:, target].index_select(1, target_indices).float().mean(dim=2), dim=-1
            )
            reference_features = F.normalize(
                match_key[:, reference].index_select(1, reference_indices).float().mean(dim=2), dim=-1
            )
            similarities = torch.matmul(target_features, reference_features.transpose(-1, -2))
            k = min(ctl.args.match_topk, reference_indices.numel())
            scores, neighbors = similarities.topk(k, dim=-1)
            weights = torch.softmax(scores / ctl.args.match_temperature, dim=-1)
            reference_values = iv[:, reference].index_select(1, reference_indices)
            matched_batches = []
            for batch in range(hidden_states.shape[0]):
                gathered = reference_values[batch][neighbors[batch]]
                matched_batches.append(
                    (gathered * weights[batch, :, :, None, None].to(gathered.dtype)).sum(dim=1)
                )
            matched = torch.stack(matched_batches)
            original = target_values.index_select(1, target_indices)
            mixed = original.lerp(matched, ctl.layer_strength(self.layer_index))
            target_values[:, target_indices] = mixed
            matched_count = int(target_indices.numel())
            mean_similarity = float(scores[..., 0].mean().detach().cpu())

        # Target Q/K remain native; only selected target-object V entries change.
        main_q = torch.cat([tq, iq[:, :nx], iq[:, target]], dim=1)
        main_k = torch.cat([tk, ik[:, :nx], ik[:, target]], dim=1)
        main_v = torch.cat([tv, iv[:, :nx], target_values], dim=1)
        main = attend(main_q, main_k, main_v, valid_mask(2 * nx))

        # Reference stays private, preventing its keys from changing structure.
        rk = torch.cat([tk, ik[:, reference].index_select(1, reference_indices)], dim=1)
        rv = torch.cat([tv, iv[:, reference].index_select(1, reference_indices)], dim=1)
        reference_state = attend(
            iq[:, reference], rk, rv, valid_mask(int(reference_indices.numel()))
        )

        image_state = torch.empty_like(iq)
        main_image = main[:, nt:]
        image_state[:, :nx] = main_image[:, :nx]
        image_state[:, target] = main_image[:, nx:]
        image_state[:, reference] = reference_state
        joint = torch.cat([main[:, :nt], image_state], dim=1).flatten(2, 3).to(hidden_states.dtype)
        text_out, image_out = joint[:, :nt], joint[:, nt:]
        image_out = attn.to_out[0](image_out.contiguous())
        if len(attn.to_out) > 1:
            image_out = attn.to_out[1](image_out)
        text_out = attn.to_add_out(text_out.contiguous())
        ctl.calls += 1
        if matched_count:
            ctl.match_stats.append({
                "layer": self.layer_index,
                "tokens": matched_count,
                "mean_best_similarity": mean_similarity,
                "strength": ctl.layer_strength(self.layer_index),
            })
        return image_out, text_out


class SemanticValueMatcher:
    def __init__(self, pipe, args):
        self.args = args
        self.blocks = list(pipe.transformer.transformer_blocks)
        self.layers = parse_layer_spec(args.match_layers, len(self.blocks))
        self.originals = {}
        self.target_mask = None
        self.reference_alpha = None
        self.target_cache = {}
        self.reference_cache = {}
        self.active = False
        self.calls = 0
        self.match_stats = []

    @staticmethod
    def grid(tokens, width, height):
        gw = max(1, round(math.sqrt(tokens * width / height)))
        gh = tokens // gw
        if gh * gw != tokens:
            raise RuntimeError(f"Cannot map {tokens} tokens to a grid")
        return gh, gw

    def _indices(self, array, tokens, threshold, cache, device):
        key = (tokens, str(device))
        if key in cache:
            return cache[key]
        h, w = array.shape
        gh, gw = self.grid(tokens, w, h)
        image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).resize(
            (gw, gh), Image.Resampling.BOX
        )
        values = np.asarray(image, dtype=np.float32).copy().reshape(-1) / 255.0
        chosen = np.flatnonzero(values >= threshold)
        result = torch.as_tensor(chosen, dtype=torch.long, device=device)
        cache[key] = result
        return result

    def target_indices(self, tokens, device):
        return self._indices(
            self.target_mask, tokens, self.args.target_token_threshold,
            self.target_cache, device,
        )

    def reference_indices(self, tokens, device):
        result = self._indices(
            self.reference_alpha, tokens, self.args.reference_token_threshold,
            self.reference_cache, device,
        )
        if not result.numel():
            raise RuntimeError("Reference matte produced no foreground tokens")
        return result

    def layer_strength(self, layer):
        if len(self.layers) <= 1:
            return self.args.value_strength
        phase = self.layers.index(layer) / (len(self.layers) - 1)
        # Smooth middle-heavy schedule avoids abrupt layer boundaries.
        envelope = math.sin(math.pi * phase) ** 2
        return self.args.value_strength * (.35 + .65 * envelope)

    def set_masks(self, target_mask, reference_alpha):
        self.target_mask = np.asarray(target_mask.convert("L"), dtype=np.float32).copy() / 255.0
        self.reference_alpha = np.asarray(reference_alpha.convert("L"), dtype=np.float32).copy() / 255.0
        self.target_cache.clear()
        self.reference_cache.clear()

    def install(self):
        # All blocks isolate R; selected blocks additionally perform V matching.
        for index, block in enumerate(self.blocks):
            self.originals[index] = block.attn.processor
            block.attn.set_processor(
                SemanticValueMatchingProcessor(self, block.attn.processor, index)
            )

    def begin(self):
        if self.target_mask is None or self.reference_alpha is None:
            raise RuntimeError("Set target/reference masks before matching")
        self.calls, self.match_stats, self.active = 0, [], True

    def end(self):
        self.active = False
        cfg = 2 if self.args.true_cfg_scale > 1 and self.args.negative_prompt is not None else 1
        expected = len(self.blocks) * self.args.steps * cfg
        if self.calls != expected:
            raise RuntimeError(f"E9 processor ran {self.calls} calls; expected {expected}")
        return {
            "mechanism": "semantic correspondence with target-QK/reference-V matching",
            "layers": self.layers,
            "value_strength": self.args.value_strength,
            "match_topk": self.args.match_topk,
            "match_temperature": self.args.match_temperature,
            "measurements": self.match_stats,
        }

    def close(self):
        self.active = False
        for index, original in self.originals.items():
            self.blocks[index].attn.set_processor(original)


def run_case(pipe, matcher, case, references, cutouts, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    step_dir = case_dir / "steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    occupied = Image.new("L", current.size)
    history = []

    for index, item in enumerate(tqdm(
        case["objects"][:args.max_objects], desc=f"E9 case {case_id:03d}", unit="object", leave=False
    ), 1):
        name, key = item["name"], reference_key(item)
        final = step_dir / f"{index:02d}_{slug(name)}_value_matched_final.png"
        target_mask_file = step_dir / f"{index:02d}_{slug(name)}_placeholder_mask.png"
        if args.resume and final.is_file() and target_mask_file.is_file():
            current = Image.open(final).convert("RGB")
            occupied = ImageChops.lighter(occupied, Image.open(target_mask_file).convert("L"))
            history.append({"step": index, "name": name, "status": "resumed", "final": str(final)})
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
            step_dir / f"{index:02d}_{slug(name)}_placement_heatmap.png",
        )
        location = region_words(box, before.size)
        placeholder_prompt = (
            f"Image 1 is the current scene. Add exactly one complete photorealistic {name} naturally in {location}. "
            "Choose physically correct scale, pose, perspective, support, contact shadow and occlusion. Preserve every "
            "existing object and all unrelated scene content. Keep the new object fully inside the frame."
        )
        placeholder_seed = args.seed + case_id * 10000 + index * 200
        placeholder = infer(pipe, [before], placeholder_prompt, args, placeholder_seed)
        target_mask = placeholder_mask(before, placeholder, box, args)
        reference_image, reference_alpha = object_condition(cutouts[key], args)

        prefix = step_dir / f"{index:02d}_{slug(name)}"
        before.save(f"{prefix}_before.png")
        placeholder.save(f"{prefix}_placeholder.png")
        target_mask.save(target_mask_file)
        reference_image.save(f"{prefix}_reference_bank.png")
        reference_alpha.save(f"{prefix}_reference_alpha.png")

        matcher.set_masks(target_mask, reference_alpha)
        matcher.begin()
        replacement_prompt = (
            f"Image 1 is the scene with a naturally placed {name}. Image 2 shows the exact reference {name}. "
            f"Preserve Image 1's composition and the existing {name}'s placement, pose, perspective, scale, contact "
            "and occlusion. Transfer Image 2's distinctive colors, materials, texture, markings and component details "
            "onto that existing object only. Preserve the background and all other objects. Return one scene image."
        )
        replacement_seed = placeholder_seed + 1
        try:
            output = infer(
                pipe, [placeholder, reference_image], replacement_prompt, args, replacement_seed
            )
        finally:
            matcher.active = False
        matching = matcher.end()
        current = output
        current.save(final)
        occupied = ImageChops.lighter(occupied, target_mask)
        history.append({
            "step": index, "name": name, "status": "generated",
            "placeholder_seed": placeholder_seed, "replacement_seed": replacement_seed,
            "placement_box": list(box), "placement_probe": probe,
            "placeholder": f"{prefix}_placeholder.png", "target_mask": str(target_mask_file),
            "reference": record["image"], "matching": matching,
            "final": str(final), "postprocess": None,
        })
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "status": "complete", "objects": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    p.add_argument("--out_dir", default="results/qwen_e9_semantic_value_matching")
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
    p.add_argument("--match_layers", default="12-35")
    p.add_argument("--value_strength", type=float, default=.65)
    p.add_argument("--match_topk", type=int, default=4)
    p.add_argument("--match_temperature", type=float, default=.07)
    p.add_argument("--target_token_threshold", type=float, default=.20)
    p.add_argument("--reference_token_threshold", type=float, default=.35)
    p.add_argument("--placeholder_difference_threshold", type=float, default=18.0)
    p.add_argument("--target_box_margin", type=float, default=.20)
    p.add_argument("--target_mask_dilation", type=int, default=5)
    p.add_argument("--target_mask_blur", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.value_strength <= 1:
        raise ValueError("value_strength must be in [0,1]")
    if args.match_topk < 1 or args.match_temperature <= 0:
        raise ValueError("match_topk and match_temperature must be positive")
    if not 0 <= args.target_token_threshold <= 1 or not 0 <= args.reference_token_threshold <= 1:
        raise ValueError("token thresholds must be in [0,1]")
    if not 0 <= args.object_alpha_low < args.object_alpha_high <= 1:
        raise ValueError("object alpha limits must satisfy 0 <= low < high <= 1")
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E9 targets diffusers 0.40.0; found {diffusers.__version__}")
        import kornia  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E9") from exc

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
    matcher = SemanticValueMatcher(pipe, args)
    matcher.install()
    summary = []
    try:
        for case in tqdm(cases, desc="E9 semantic-value suite", unit="case"):
            summary.append(run_case(pipe, matcher, case, references, cutouts, args, out))
            save_json(summary, out / "summary.partial.json")
    finally:
        matcher.close()
    save_json({
        "method": "placeholder geometry plus semantic matched-value appearance transfer",
        "passes_per_object": 2,
        "structure": "target queries and keys",
        "appearance": "softly matched reference values",
        "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
