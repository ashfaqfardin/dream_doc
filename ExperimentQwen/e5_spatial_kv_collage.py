"""E5: one-pass collage-primary feature-routed harmonization.

The raw Qwen result is final: no pixel-space operation follows inference.
RMBG-2.0 creates a soft-alpha object cutout. Qwen's vision-language encoder
sees only the collage. The denoiser additionally receives object and base as
private latent banks in [collage, cutout, base] order. Collage is the primary trajectory;
weak base and identity residuals are injected outside/inside the edit gate.
The raw Qwen result is final: no pixel-space operation follows inference.
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

from e1_baseline import fit, infer, load_pipe, make_generator, save_json
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
        # Transformers 5 expects this bookkeeping attribute to have been
        # created by the contemporary PreTrainedModel initialization path.
        # RMBG-2.0's older remote BiRefNet class does not create it and has no
        # tied parameters, so an empty mapping is the correct representation.
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}
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
        # RGB alone necessarily still contains original background pixels; save
        # it explicitly as a diagnostic rather than labelling it a cutout.
        cutout.rgb.save(directory / f"{key}_source_crop.png")
        cutout.alpha.save(directory / f"{key}_alpha.png")
        rgba = cutout.rgb.copy()
        rgba.putalpha(cutout.alpha)
        rgba.save(directory / f"{key}_rgba.png")
        preview = Image.new("RGB", cutout.rgb.size, (127, 127, 127))
        preview.paste(cutout.rgb, (0, 0), cutout.alpha)
        preview.save(directory / f"{key}_cutout_preview.png")
        cutouts[key] = cutout
    return cutouts


def object_condition(cutout: Cutout, args):
    """Create an alpha-premultiplied private O bank with no visible background."""
    canvas = Image.new("RGB", (args.width, args.height), args.object_condition_background)
    alpha_canvas = Image.new("L", canvas.size)
    available_w = round(args.width * args.object_condition_scale)
    available_h = round(args.height * args.object_condition_scale)
    scale = min(available_w / cutout.rgb.width, available_h / cutout.rgb.height)
    size = (max(1, round(cutout.rgb.width * scale)), max(1, round(cutout.rgb.height * scale)))
    rgb = cutout.rgb.resize(size, Image.Resampling.LANCZOS)
    alpha = cutout.alpha.resize(size, Image.Resampling.LANCZOS)
    # Suppress uncertain RMBG background and color spill before VAE patching.
    # A smooth confidence remap retains antialiased boundaries without allowing
    # low-alpha red/background pixels to become identity features.
    values = np.asarray(alpha, dtype=np.float32).copy() / 255.0
    low, high = args.object_alpha_low, args.object_alpha_high
    values = np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)
    values = values * values * (3.0 - 2.0 * values)
    alpha = Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255))
    position = ((args.width - size[0]) // 2, (args.height - size[1]) // 2)
    canvas.paste(rgb, position, alpha)
    alpha_canvas.paste(alpha, position)
    return canvas, alpha_canvas


def parse_layer_spec(spec: str, count: int) -> list[int]:
    if spec.strip().lower() == "middle":
        # The intervention is deliberately absent from early composition and
        # late rendering blocks. Keep only the 30%-65% semantic-editing band.
        return list(range(round(count * .30), max(round(count * .65), 1)))
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
    """Collage-primary attention with isolated O and weak aligned B residuals."""

    def __init__(self, controller, original, layer_index):
        self.controller = controller
        self.original = original
        self.layer_index = layer_index

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
                f"Unexpected [C,O,B] token layout: {seq_img}; expected output + three equal conditions"
            )
        condition_tokens = remainder // 3
        if condition_tokens != output_tokens:
            raise RuntimeError(
                "E5 requires square C, O, and B inputs matching the output token grid; "
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
        # Latent order is output, collage, isolated object, base bank. O and B
        # are absent from main joint attention and therefore cannot leak into
        # text, output, or collage tokens globally.
        collage_slice = slice(output_tokens, output_tokens + condition_tokens)
        object_slice = slice(output_tokens + condition_tokens, output_tokens + 2 * condition_tokens)
        base_slice = slice(output_tokens + 2 * condition_tokens, output_tokens + 3 * condition_tokens)
        main_indices = torch.cat([
            torch.arange(output_tokens, device=img_key.device),
            torch.arange(collage_slice.start, collage_slice.stop, device=img_key.device),
        ])
        foreground = controller.object_foreground_indices(condition_tokens, img_key.device)
        main_img_query = img_query.index_select(1, main_indices)
        main_query = torch.cat([txt_query, main_img_query], dim=1)
        main_key = torch.cat([
            txt_key, img_key[:, :output_tokens], img_key[:, collage_slice],
        ], dim=1)
        main_value = torch.cat([
            txt_value, img_value[:, :output_tokens], img_value[:, collage_slice],
        ], dim=1)
        main_mask = None
        if encoder_hidden_states_mask is not None:
            valid_images = torch.ones(
                (hidden_states.shape[0], 2 * output_tokens),
                dtype=torch.bool, device=hidden_states.device,
            )
            main_mask = torch.cat([encoder_hidden_states_mask, valid_images], dim=1)[:, None, None, :]
        main_result = dispatch_attention_fn(
            main_query, main_key, main_value, attn_mask=main_mask,
            dropout_p=0.0, is_causal=False, backend=backend, parallel_config=parallel,
        )
        main_image = main_result[:, seq_txt:]
        collage_attention = main_image[:, :output_tokens]

        # Identity memory exposes only RMBG foreground tokens. It modifies the
        # collage trajectory rather than replacing it.
        object_key = torch.cat([
            txt_key, img_key[:, :output_tokens],
            img_key[:, object_slice].index_select(1, foreground),
        ], dim=1)
        object_value = torch.cat([
            txt_value, img_value[:, :output_tokens],
            img_value[:, object_slice].index_select(1, foreground),
        ], dim=1)
        output_query = img_query[:, :output_tokens]
        object_image_mask = None
        if encoder_hidden_states_mask is not None:
            object_keys = output_tokens + int(foreground.numel())
            object_image_mask = torch.ones(
                (hidden_states.shape[0], object_keys), dtype=torch.bool, device=hidden_states.device
            )
            object_image_mask = torch.cat([encoder_hidden_states_mask, object_image_mask], dim=1)[:, None, None, :]
        object_result = dispatch_attention_fn(
            output_query, object_key, object_value, attn_mask=object_image_mask,
            dropout_p=0.0, is_causal=False, backend=backend, parallel_config=parallel,
        )

        # Keep O alive as an isolated bank. Its full query grid can update, but
        # it can read only text plus its own confident foreground K/V.
        object_bank_key = torch.cat([
            txt_key, img_key[:, object_slice].index_select(1, foreground),
        ], dim=1)
        object_bank_value = torch.cat([
            txt_value, img_value[:, object_slice].index_select(1, foreground),
        ], dim=1)
        object_bank_mask = None
        if encoder_hidden_states_mask is not None:
            valid_object = torch.ones(
                (hidden_states.shape[0], int(foreground.numel())),
                dtype=torch.bool, device=hidden_states.device,
            )
            object_bank_mask = torch.cat(
                [encoder_hidden_states_mask, valid_object], dim=1
            )[:, None, None, :]
        object_bank_result = dispatch_attention_fn(
            img_query[:, object_slice], object_bank_key, object_bank_value,
            attn_mask=object_bank_mask, dropout_p=0.0, is_causal=False,
            backend=backend, parallel_config=parallel,
        )

        # B is a spatially aligned value bank, not a globally retrievable image.
        # Normalize each delta so the configured strengths remain meaningful.
        def bounded_delta(source, target):
            delta = source - target
            target_norm = target.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            delta_norm = delta.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            return delta * (target_norm / delta_norm).clamp(max=1.0).to(delta.dtype)

        aligned_base_delta = bounded_delta(
            img_value[:, base_slice], img_value[:, collage_slice]
        )
        identity_delta = bounded_delta(object_result, collage_attention)
        gate = controller.spatial_gate(output_tokens, collage_attention.device, collage_attention.dtype)
        intervention_scale = 1.0 if self.layer_index in controller.layers else 0.0
        routed = (
            collage_attention
            + intervention_scale * controller.args.base_residual_weight * (1 - gate) * aligned_base_delta
            + intervention_scale * controller.args.identity_residual_weight * gate * identity_delta
        )

        # Update B independently using only text and its own tokens. Therefore B
        # remains useful as a bank but cannot alter the semantic/text stream.
        base_key = torch.cat([txt_key, img_key[:, base_slice]], dim=1)
        base_value = torch.cat([txt_value, img_value[:, base_slice]], dim=1)
        base_mask = None
        if encoder_hidden_states_mask is not None:
            valid_base = torch.ones(
                (hidden_states.shape[0], condition_tokens),
                dtype=torch.bool, device=hidden_states.device,
            )
            base_mask = torch.cat([encoder_hidden_states_mask, valid_base], dim=1)[:, None, None, :]
        base_result = dispatch_attention_fn(
            img_query[:, base_slice], base_key, base_value, attn_mask=base_mask,
            dropout_p=0.0, is_causal=False, backend=backend, parallel_config=parallel,
        )

        joint_image = torch.empty_like(img_query)
        joint_image[:, :output_tokens] = routed
        joint_image[:, collage_slice] = main_image[:, output_tokens:2 * output_tokens]
        joint_image[:, object_slice] = object_bank_result
        joint_image[:, base_slice] = base_result
        joint = torch.cat([main_result[:, :seq_txt], joint_image], dim=1)

        joint = joint.flatten(2, 3).to(img_query.dtype)
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
        # Isolation runs in every block. Only self.layers receive non-zero
        # residual intervention; otherwise B would leak through ordinary blocks.
        for index in range(len(self.blocks)):
            attention = self.blocks[index].attn
            self.originals[index] = attention.processor
            attention.set_processor(SpatialRoutingProcessor(self, attention.processor, index))

    def begin(self):
        if self.gate is None or self.object_alpha is None:
            raise RuntimeError("Set the RMBG-derived edit and object gates before inference")
        self.calls = 0
        self.layout = None
        self.active = True

    def end(self):
        self.active = False
        cfg_passes = 2 if self.args.true_cfg_scale > 1 and self.args.negative_prompt is not None else 1
        expected = len(self.blocks) * self.args.steps * cfg_passes
        if self.calls != expected:
            raise RuntimeError(f"Spatial routing ran {self.calls} times; expected {expected}")
        return {
            "mechanism": "collage-primary normalized residual routing",
            "isolation_layers": list(range(len(self.blocks))),
            "intervention_layers": self.layers,
            "base_residual_weight": self.args.base_residual_weight,
            "identity_residual_weight": self.args.identity_residual_weight,
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


@torch.inference_mode()
def infer_with_latent_base(pipe, collage, object_image, base, prompt, args, seed):
    """Run one pass while hiding B from the multimodal prompt encoder.

    Only C creates semantic prompt embeddings. The denoiser receives C/O/B
    latents; O and B remain isolated banks controlled by the attention router.
    """
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        calculate_dimensions,
    )

    semantic_images = []
    for image in (collage,):
        width, height = calculate_dimensions(
            CONDITION_IMAGE_SIZE, image.width / image.height
        )
        semantic_images.append(pipe.image_processor.resize(image, height, width))
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt=prompt,
        image=semantic_images,
        device=pipe._execution_device,
        num_images_per_prompt=1,
    )
    negative_embeds = negative_mask = None
    if args.true_cfg_scale > 1.0 and args.negative_prompt is not None:
        negative_embeds, negative_mask = pipe.encode_prompt(
            prompt=args.negative_prompt,
            image=semantic_images,
            device=pipe._execution_device,
            num_images_per_prompt=1,
        )
    result = pipe(
        image=[collage, object_image, base],
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
        # Distinct filename prevents --resume from silently reusing results
        # produced by the former three-image global-routing architecture.
        final_path = steps_dir / f"{index:02d}_{slug(name)}_object_gated_final.png"
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
            f"Image 1 is a scene collage containing one pasted {name}. Return one natural edited version of "
            f"Image 1. Keep its composition and the {name}'s position, structure, colors and material. Harmonize "
            "lighting and contact without moving or redesigning "
            "the object. Do not output an isolated object, reference panel, border, collage, or split image."
        )
        kv_share.begin()
        try:
            # This is the only generative edit pass for the insertion.
            current = infer_with_latent_base(
                pipe, collage, object_image, before, prompt, args,
                args.seed + case_id * 10000 + index * 100,
            )
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
    parser.add_argument("--kv_layers", default="middle", help="Layer indices, 'middle' (30%-65%), or 'all'")
    parser.add_argument("--kv_edit_dilation", type=int, default=4, help="Small pixel expansion around the soft edit gate")
    parser.add_argument("--kv_gate_blur", type=float, default=1.0)
    parser.add_argument("--base_residual_weight", type=float, default=.15)
    parser.add_argument("--identity_residual_weight", type=float, default=.35)
    parser.add_argument("--object_token_alpha_threshold", type=float, default=.35)
    parser.add_argument("--object_alpha_low", type=float, default=.15, help="Discard lower-confidence RMBG alpha")
    parser.add_argument("--object_alpha_high", type=float, default=.85, help="Alpha treated as fully foreground")
    parser.add_argument("--object_condition_scale", type=float, default=.82)
    parser.add_argument("--object_condition_background", type=int, default=127)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.base_residual_weight <= 1:
        raise ValueError("base_residual_weight must be in [0, 1]")
    if not 0 <= args.identity_residual_weight <= 1:
        raise ValueError("identity_residual_weight must be in [0, 1]")
    if not 0 <= args.object_token_alpha_threshold <= 1:
        raise ValueError("object_token_alpha_threshold must be in [0, 1]")
    if not 0 <= args.object_alpha_low < args.object_alpha_high <= 1:
        raise ValueError("object alpha limits must satisfy 0 <= low < high <= 1")
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
    save_json({
        "cutout_backend": args.rmbg_model_id,
        "semantic_image_roles": ["collage"],
        "denoiser_latent_roles": ["collage", "object", "base_bank"],
        "cases": summary,
    }, out / "summary.json")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
