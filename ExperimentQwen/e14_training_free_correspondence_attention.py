"""E14: training-free paired-difference correspondence attention.

For each insertion B is the current scene and C=B+O is an aligned rough
collage. The native control receives C alone. Asymmetric variants expose only C
to Qwen2.5-VL while the denoiser receives VAE banks [C, B]. In selected middle
MMDiT blocks, aligned C/B feature differences select object-evidence tokens from C. Generated tokens
then use Qwen's native projections and values with one of three routing rules:
difference-only, correlation-biased, or Sinkhorn+position-biased attention.

No parameters are trained, no K/V values are transplanted, no latent blending
or hard output mask is used, and no post-generation processing is applied.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops
from tqdm.auto import tqdm

from e1_baseline import fit, infer, load_pipe, make_generator, save_json
from e2_sam_collage_repaint import composite, place_cutout, probe_placement
from e3_prompt_suite import generate_references, load_suite, reference_key, select_cases, slug
from e5_spatial_kv_collage import (
    RMBG2Cutout, generate_base, generate_rmbg_cutouts, parse_layer_spec,
)
from e12_spatial_reference_card_insertion import E12Evaluator


HERE = Path(__file__).resolve().parent
VARIANTS = ("native", "asymmetric", "difference", "correlation", "sinkhorn")


def normalize_prior(values: torch.Tensor) -> torch.Tensor:
    """Robustly standardize each query row and bound extreme logit changes."""
    values = values.float()
    center = values.mean(dim=-1, keepdim=True)
    scale = values.std(dim=-1, keepdim=True).clamp_min(1e-5)
    return ((values - center) / scale).clamp(-3.0, 3.0)


def sinkhorn_log_prior(scores: torch.Tensor, iterations: int, temperature: float) -> torch.Tensor:
    """Rectangular entropy-regularized transport with uniform marginals."""
    log_plan = scores.float() / temperature
    log_plan = log_plan - log_plan.amax(dim=(-2, -1), keepdim=True)
    rows, cols = scores.shape[-2:]
    log_row_target = -math.log(rows)
    log_col_target = -math.log(cols)
    for _ in range(iterations):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True) + log_row_target
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-2, keepdim=True) + log_col_target
    return normalize_prior(log_plan)


class CorrespondenceProcessor:
    """Native Qwen attention with routed X queries in selected blocks."""

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
        if not ctl.active or ctl.variant in ("native", "asymmetric"):
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, attention_mask, image_rotary_emb, **kwargs,
            )
        if encoder_hidden_states is None or attention_mask is not None:
            raise RuntimeError("E14 requires standard Qwen double-stream attention")

        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE

        text_tokens = encoder_hidden_states.shape[1]
        image_tokens = hidden_states.shape[1]
        output_tokens = (ctl.args.height // 16) * (ctl.args.width // 16)
        if image_tokens != 3 * output_tokens:
            raise RuntimeError(
                f"Expected packed [X,C,B] with {3 * output_tokens} tokens, got {image_tokens}. "
                "E14 requires square B and C at the output resolution."
            )
        x_slice = slice(0, output_tokens)
        c_slice = slice(output_tokens, 2 * output_tokens)
        b_slice = slice(2 * output_tokens, 3 * output_tokens)

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

        valid_image = torch.ones(
            (hidden_states.shape[0], image_tokens), dtype=torch.bool, device=hidden_states.device
        )
        native_mask = None
        if encoder_hidden_states_mask is not None:
            native_mask = torch.cat([encoder_hidden_states_mask, valid_image], dim=1)[:, None, None, :]
        joint_q = torch.cat([tq, iq], dim=1)
        joint_k = torch.cat([tk, ik], dim=1)
        joint_v = torch.cat([tv, iv], dim=1)
        native = attend(joint_q, joint_k, joint_v, native_mask)

        selected, delta = ctl.select_collage_tokens(
            hidden_states[:, b_slice], hidden_states[:, c_slice], self.layer_index
        )
        selected_k = ik[:, c_slice].index_select(1, selected)
        # Preserve Qwen's complete native T/X/C/B context. Earlier E14 code
        # removed all non-selected C tokens from X attention; that discarded
        # the primary scene-reconstruction stream and could collapse output to
        # the selected object on its reference background. E14 now changes
        # logits only--never the native set or ordering of K/V tokens.
        routed_k = joint_k
        routed_v = joint_v
        key_count = routed_k.shape[1]
        bias = torch.zeros(
            (hidden_states.shape[0], 1, output_tokens, key_count),
            device=hidden_states.device, dtype=iq.dtype,
        )
        if encoder_hidden_states_mask is not None:
            invalid = ~encoder_hidden_states_mask[:, None, None, :].to(torch.bool)
            bias[:, :, :, :text_tokens].masked_fill_(invalid, -torch.inf)

        strength = ctl.strength()
        selected_columns = text_tokens + output_tokens + selected
        query = iq[:, x_slice]
        similarity = torch.einsum("bqhd,bkhd->bhqk", query.float(), selected_k.float())
        similarity = similarity.mean(dim=1) / math.sqrt(head_dim)
        position, query_gate = ctl.position_prior(output_tokens, selected, query.device)
        if ctl.variant == "difference":
            token_prior = normalize_prior(delta.index_select(0, selected).unsqueeze(0))
            prior = token_prior.expand(output_tokens, -1)
        elif ctl.variant == "correlation":
            prior = normalize_prior(similarity) + ctl.args.position_weight * position
        elif ctl.variant == "sinkhorn":
            transport_scores = similarity + ctl.args.position_weight * position
            prior = sinkhorn_log_prior(
                transport_scores, ctl.args.sinkhorn_iterations, ctl.args.sinkhorn_temperature
            )
        else:
            prior = torch.zeros_like(similarity)
        # Crucially retain absolute target distance. Row-normalizing a
        # positional matrix alone gives every background query a preferred
        # object token; Sinkhorn then spreads object evidence over the whole
        # canvas. This broad Gaussian is soft (not an output mask), but drives
        # the intervention continuously to zero away from the placement.
        prior = prior * query_gate[:, None]
        bias[:, :, :, selected_columns] += (strength * prior).unsqueeze(1).to(bias.dtype)
        routed_x = attend(query, routed_k, routed_v, bias)

        image_native = native[:, text_tokens:]
        image_native[:, x_slice] = routed_x
        joint = torch.cat([native[:, :text_tokens], image_native], dim=1)
        joint = joint.flatten(2, 3).to(hidden_states.dtype)
        text_out, image_out = joint[:, :text_tokens], joint[:, text_tokens:]
        image_out = attn.to_out[0](image_out.contiguous())
        if len(attn.to_out) > 1:
            image_out = attn.to_out[1](image_out)
        text_out = attn.to_add_out(text_out.contiguous())

        ctl.record(self.layer_index, selected, delta, prior, strength, output_tokens)
        ctl.calls += 1
        return image_out, text_out


class TrainingFreeCorrespondenceRouter:
    def __init__(self, pipe, args):
        self.args = args
        self.blocks = list(pipe.transformer.transformer_blocks)
        self.layers = parse_layer_spec(args.routing_layers, len(self.blocks))
        self.originals = {}
        self.active = False
        self.variant = "native"
        self.calls = 0
        self.records = []
        self.heat_sum = None
        self.heat_count = 0

    def install(self):
        for index in self.layers:
            block = self.blocks[index]
            self.originals[index] = block.attn.processor
            block.attn.set_processor(CorrespondenceProcessor(self, block.attn.processor, index))

    def begin(self, variant):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown E14 variant: {variant}")
        self.variant = variant
        self.active = variant in ("difference", "correlation", "sinkhorn")
        self.calls = 0
        self.records = []
        self.heat_sum = None
        self.heat_count = 0

    def strength(self):
        cfg_passes = 2 if self.args.true_cfg_scale > 1 and self.args.negative_prompt is not None else 1
        sweep = self.calls // max(1, len(self.layers))
        step = min(self.args.steps - 1, sweep // cfg_passes)
        fraction = step / max(1, self.args.steps - 1)
        triangular = max(0.0, 1.0 - abs(2.0 * fraction - 1.0))
        return self.args.routing_strength * triangular

    def select_collage_tokens(self, base, collage, layer_index):
        base_n = F.normalize(base.float(), dim=-1)
        collage_n = F.normalize(collage.float(), dim=-1)
        delta = (1.0 - (base_n * collage_n).sum(dim=-1)).mean(dim=0)
        tokens = delta.numel()
        count = int(round(tokens * self.args.reference_token_fraction))
        count = min(tokens, max(self.args.minimum_reference_tokens, count))
        selected = torch.topk(delta, k=count, largest=True, sorted=True).indices
        return selected, delta

    def position_prior(self, tokens, selected, device):
        side = round(math.sqrt(tokens))
        if side * side != tokens:
            raise RuntimeError(f"E14 currently requires a square output grid, got {tokens} tokens")
        y = torch.arange(side, device=device, dtype=torch.float32)
        x = torch.arange(side, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2) / max(1, side - 1)
        q = coords[:, None, :]
        k = coords.index_select(0, selected)[None, :, :]
        squared = (q - k).square().sum(dim=-1)
        prior = -squared / (2.0 * self.args.position_sigma ** 2)
        nearest_squared = squared.amin(dim=-1)
        query_gate = torch.exp(
            -nearest_squared / (2.0 * self.args.query_gate_sigma ** 2)
        )
        return normalize_prior(prior), query_gate

    def record(self, layer, selected, delta, prior, strength, tokens):
        with torch.no_grad():
            entropy = None
            coverage = None
            if prior.numel():
                probability = torch.softmax(prior.float(), dim=-1)
                entropy = float((-(probability * probability.clamp_min(1e-12).log()).sum(-1)).mean().cpu())
                received = probability.sum(dim=0)
                coverage = float((received > received.mean() * .1).float().mean().cpu())
                # Peak correspondence is informative per output query; summing
                # a row-normalized distribution would produce an all-one map.
                heat = probability.amax(dim=-1)
                self.heat_sum = heat if self.heat_sum is None else self.heat_sum + heat
                self.heat_count += 1
            self.records.append({
                "layer": int(layer), "strength": float(strength),
                "selected_tokens": int(selected.numel()),
                "delta_mean": float(delta.mean().cpu()), "delta_max": float(delta.max().cpu()),
                "correspondence_entropy": entropy, "reference_token_coverage": coverage,
            })

    def end(self, heatmap_path: Path | None = None):
        self.active = False
        cfg_passes = 2 if self.args.true_cfg_scale > 1 and self.args.negative_prompt is not None else 1
        expected = len(self.layers) * self.args.steps * cfg_passes
        if self.variant in ("difference", "correlation", "sinkhorn") and self.calls != expected:
            raise RuntimeError(f"E14 router ran {self.calls} times; expected {expected}")
        if heatmap_path is not None and self.heat_sum is not None:
            heat = (self.heat_sum / max(1, self.heat_count)).float().cpu().numpy()
            side = round(math.sqrt(heat.size))
            heat = heat.reshape(side, side)
            heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-8)
            color = np.stack([heat, np.sqrt(heat), 1.0 - heat], axis=-1)
            Image.fromarray(np.uint8(np.clip(color, 0, 1) * 255)).resize(
                (self.args.width, self.args.height), Image.Resampling.NEAREST
            ).save(heatmap_path)
        return {
            "variant": self.variant, "training": False, "routing_layers": self.layers,
            "processor_calls": self.calls, "records": self.records,
            "attention_heatmap": str(heatmap_path) if heatmap_path and heatmap_path.is_file() else None,
        }

    def close(self):
        self.active = False
        for index, original in self.originals.items():
            self.blocks[index].attn.set_processor(original)


def insertion_prompt(name):
    return (
        f"The input image is an aligned rough scene composite containing one pasted {name}. "
        f"Return one photorealistic version of this complete scene by harmonizing that exact {name} naturally. "
        "Keep its intended position, complete structure, proportions, colors, materials, textures, and distinctive "
        "details. Correct only its boundary, perspective, illumination, support contact, and shadow. Preserve the "
        "scene's camera, geometry, background, and existing objects. Return one scene only. Do not output an "
        "isolated object, white background, reference board, collage, grid, split image, or duplicate object."
    )


@torch.inference_mode()
def infer_asymmetric(pipe, collage, base, prompt, args, seed):
    """Expose C to VL semantics while providing aligned [C,B] VAE evidence."""
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE, calculate_dimensions,
    )

    condition_width, condition_height = calculate_dimensions(
        CONDITION_IMAGE_SIZE, collage.width / collage.height
    )
    semantic_images = [pipe.image_processor.resize(collage, condition_height, condition_width)]
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt=prompt, image=semantic_images, device=pipe._execution_device,
        num_images_per_prompt=1,
    )
    negative_embeds = negative_mask = None
    if args.true_cfg_scale > 1.0 and args.negative_prompt is not None:
        negative_embeds, negative_mask = pipe.encode_prompt(
            prompt=args.negative_prompt, image=semantic_images,
            device=pipe._execution_device, num_images_per_prompt=1,
        )
    result = pipe(
        image=[collage, base],
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


def run_variant(pipe, router, variant, base, collage, prompt, args, seed, heatmap_path):
    router.begin(variant)
    try:
        if variant == "native":
            # True control: the successful rough collage is the only semantic
            # and reconstructive condition. B cannot compete as another image.
            image = infer(pipe, [collage], prompt, args, seed)
        else:
            # VL still sees C only. B exists solely as an aligned VAE bank.
            image = infer_asymmetric(pipe, collage, base, prompt, args, seed)
    finally:
        router.active = False
    diagnostics = router.end(
        heatmap_path if variant in ("difference", "correlation", "sinkhorn") else None
    )
    return image, diagnostics


def run_case(pipe, router, case, references, cutouts, evaluator, args, out):
    case_id = int(case["id"])
    case_dir = out / "cases" / f"case_{case_id:03d}"
    steps_dir = case_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    current = generate_base(pipe, case, args, case_dir / "base.png")
    occupied = Image.new("L", current.size)
    history = []

    for index, item in enumerate(tqdm(
        case["objects"][: args.max_objects or None],
        desc=f"E14 case {case_id:03d}", unit="object", leave=False,
    ), 1):
        name, key = item["name"], reference_key(item)
        prefix = steps_dir / f"{index:02d}_{slug(name)}"
        selected_final = Path(f"{prefix}_{args.selected_variant}.png")
        mask_path = Path(f"{prefix}_paste_alpha.png")
        if args.resume and selected_final.is_file() and mask_path.is_file():
            current = Image.open(selected_final).convert("RGB")
            occupied = ImageChops.lighter(occupied, Image.open(mask_path).convert("L"))
            history.append({"step": index, "name": name, "status": "resumed", "final": str(selected_final)})
            continue
        record = references.get(key, {})
        if record.get("status") != "ready" or key not in cutouts:
            if args.missing_policy == "error":
                raise FileNotFoundError(f"No reference/cutout available for {name}: {record}")
            history.append({"step": index, "name": name, "status": "skipped_missing_reference"})
            continue

        before = fit(current, (args.width, args.height))
        box, probe = probe_placement(
            pipe, before, name, cutouts[key], occupied, args,
            args.seed + case_id * 100000 + index * 1000,
            Path(f"{prefix}_placement_heatmap.png"),
        )
        object_canvas, paste_mask, placed_box = place_cutout(
            cutouts[key], box, before.size, args.object_scale
        )
        collage = composite(before, object_canvas, paste_mask)
        before.save(Path(f"{prefix}_base.png"))
        collage.save(Path(f"{prefix}_collage.png"))
        paste_mask.save(mask_path)
        prompt = insertion_prompt(name)
        seed = args.seed + case_id * 10000 + index * 100
        results = {}
        for variant in args.variants:
            target = Path(f"{prefix}_{variant}.png")
            routed_heatmap = Path(f"{prefix}_{variant}_attention.png")
            image, diagnostics = run_variant(
                pipe, router, variant, before, collage, prompt, args, seed, routed_heatmap
            )
            image.save(target)
            metrics = evaluator.evaluate(
                name, before, image, cutouts[key], Path(f"{prefix}_{variant}")
            ) if evaluator else None
            results[variant] = {
                "image": str(target), "metrics": metrics, "attention": diagnostics,
            }
        if args.selected_variant not in results:
            raise ValueError("selected_variant must be included in --variants")
        current = Image.open(results[args.selected_variant]["image"]).convert("RGB")
        occupied = ImageChops.lighter(occupied, paste_mask)
        history.append({
            "step": index, "name": name, "status": "generated", "seed": seed,
            "placed_box": list(placed_box), "probe": probe, "variants": results,
            "selected_variant": args.selected_variant,
            "final": results[args.selected_variant]["image"], "postprocess": None,
        })
        save_json(history, case_dir / "history.json")

    current.save(case_dir / "FINAL.png")
    save_json(history, case_dir / "history.json")
    return {"id": case_id, "status": "complete", "objects": len(history), "final": str(case_dir / "FINAL.png")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompts", default=str(HERE / "e5_prompts.json"))
    p.add_argument("--out_dir", default="results/qwen_e14_training_free_correspondence")
    p.add_argument("--case_ids", type=int, nargs="+")
    p.add_argument("--max_objects", type=int, default=3, choices=(1, 2, 3))
    p.add_argument("--missing_policy", choices=("skip", "error"), default="skip")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--selected_variant", choices=VARIANTS, default="correlation")
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
    p.add_argument("--box_margin", type=int, default=64)
    p.add_argument("--occupancy_margin", type=int, default=32)
    p.add_argument("--default_object_height", type=float, default=.25)
    p.add_argument("--object_height_priors")
    p.add_argument("--object_scale", type=float, default=.88)
    p.add_argument("--routing_layers", default="middle")
    p.add_argument("--reference_token_fraction", type=float, default=.08)
    p.add_argument("--minimum_reference_tokens", type=int, default=32)
    p.add_argument("--routing_strength", type=float, default=.15)
    p.add_argument("--position_weight", type=float, default=.35)
    p.add_argument("--position_sigma", type=float, default=.18)
    p.add_argument("--query_gate_sigma", type=float, default=.22, help="Soft absolute-distance envelope around selected collage evidence")
    p.add_argument("--sinkhorn_iterations", type=int, default=5)
    p.add_argument("--sinkhorn_temperature", type=float, default=.20)
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
    if args.selected_variant not in args.variants:
        raise ValueError("--selected_variant must be present in --variants")
    if not 0 < args.reference_token_fraction <= 1:
        raise ValueError("reference_token_fraction must be in (0,1]")
    if args.routing_strength < 0 or args.position_sigma <= 0 or args.query_gate_sigma <= 0:
        raise ValueError("Invalid routing strength or positional sigma")
    if args.sinkhorn_iterations < 1 or args.sinkhorn_temperature <= 0:
        raise ValueError("Invalid Sinkhorn configuration")
    try:
        import diffusers
        if diffusers.__version__ != "0.40.0":
            warnings.warn(f"E14 targets diffusers 0.40.0; found {diffusers.__version__}")
        import kornia  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install ExperimentQwen/requirements.txt before E14") from exc

    prompt_file = Path(args.prompts).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_suite(prompt_file), args.case_ids)
    save_json(vars(args), out / "config.json")
    pipe = load_pipe(args)
    references = generate_references(pipe, cases, args, out, prompt_file)
    segmenter = RMBG2Cutout(args.rmbg_model_id, args.rmbg_revision, args.rmbg_device, args.rmbg_input_size)
    evaluator = None
    router = TrainingFreeCorrespondenceRouter(pipe, args)
    try:
        cutouts = generate_rmbg_cutouts(segmenter, references, args, out)
        evaluator = E12Evaluator(segmenter, args) if args.evaluation else None
        router.install()
        summary = []
        for case in tqdm(cases, desc="E14 correspondence suite", unit="case"):
            summary.append(run_case(pipe, router, case, references, cutouts, evaluator, args, out))
            save_json(summary, out / "summary.partial.json")
    finally:
        router.close()
        if evaluator is not None:
            evaluator.close()
        segmenter.close()
    save_json({
        "method": "training-free aligned patch-difference and correspondence attention",
        "training": False,
        "conditioning": {
            "native": {"vl": ["collage"], "vae": ["collage"]},
            "asymmetric_and_routed": {"vl": ["collage"], "vae": ["collage", "base"]},
        },
        "variants": args.variants, "selected_variant": args.selected_variant,
        "feature_addition": False, "kv_replacement": False,
        "latent_blending": False, "postprocess": None, "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
