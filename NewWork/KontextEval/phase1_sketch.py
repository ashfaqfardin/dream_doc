# -*- coding: utf-8 -*-
"""
Phase 1 -- Sketch-Conditioned Multi-Step Object Insertion

Pipeline
--------
Two-stage, training-free:

  Stage A  Sketch -> Object Image
    Given a hand-drawn sketch of an object, generate a photorealistic version
    of it on a plain white background.

    Mode 'lora'  (default):
      Uses FluxKontextPipeline + gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA.
      Single model throughout (no swap), trigger phrase forces structure fidelity.

    Mode 'controlnet':
      Uses FluxControlNetPipeline (FLUX.1-dev + lineart ControlNet).
      Stronger geometric control; requires loading a second large model and
      swapping it out before Stage B.

  Stage B  Object -> Scene  (shared by both modes)
    1. PROBE  -- find where in the accumulated scene the prompt would place the object
    2. PLACE  -- paste the Stage-A object at the probe-detected location (rough composite)
    3. REFINE -- run Kontext K/V injection with rough_composite as canvas
                  (Kontext smooths seams and adjusts lighting; K/V injection locks background)
    4. BG-REPLACE -- overwrite pure-background pixels with the original base (zero drift)

Mask extraction
---------------
RMBG-2.0 (briaai/RMBG-2.0, BiRefNet) replaces naive pixel-diff-vs-white.
Works regardless of object colour (handles white vase on white background).

Usage
-----
  # LoRA mode (single model, simplest)
  python NewWork/KontextEval/phase1_sketch.py \\
      --sketch_dir inputs/sketches \\
      --hf_token $HF_TOKEN --cache_dir ./models \\
      --out_dir results/phase1_sketch --mode lora

  # ControlNet mode (stronger shape control)
  python NewWork/KontextEval/phase1_sketch.py \\
      --sketch_dir inputs/sketches \\
      --hf_token $HF_TOKEN --cache_dir ./models \\
      --out_dir results/phase1_sketch --mode controlnet

  # Both modes + text-only bg_replace for comparison
  python NewWork/KontextEval/phase1_sketch.py \\
      --sketch_dir inputs/sketches \\
      --hf_token $HF_TOKEN --cache_dir ./models \\
      --out_dir results/phase1_sketch --mode all

Sketch format
-------------
  inputs/sketches/bicycle.png   -- dark lines on light background (standard paper sketch)
  inputs/sketches/vase.png
  inputs/sketches/ball.png
  (name must match the 'name' field in EDITS or --config)
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

# ── import utilities from phase1_composite (avoid duplication) ────────────────
_comp_path = str(Path(__file__).parent / "phase1_composite.py")
_spec = importlib.util.spec_from_file_location("phase1_composite", _comp_path)
_comp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comp)

pixel_to_token_mask  = _comp.pixel_to_token_mask
_token_to_pixel      = _comp._token_to_pixel
color_overlay        = _comp.color_overlay
region_diff          = _comp.region_diff
run_standard         = _comp.run_standard
save_grid            = _comp.save_grid
feather_mask         = _comp.feather_mask          # has interior=1.0 clamp fix
run_kv_injected      = _comp.run_kv_injected
build_stability_grid = _comp.build_stability_grid
run_bg_replace_chain = _comp.run_bg_replace_chain

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline

_ie_path = str(Path(__file__).parent.parent / "IncrementalEdit")
sys.path.insert(0, _ie_path)
from kontext_injection import TIER_A, TIER_ALL

# ============================================================
# Edit list
# ============================================================

EDITS: List[dict] = [
    {
        "name": "bicycle",
        "prompt": (
            "Add a yellow bicycle leaning against the wall on the left side. "
            "Keep the rest of the room exactly the same."
        ),
        "description": "yellow bicycle",
    },
    {
        "name": "vase",
        "prompt": (
            "Add a white ceramic vase with flowers on the coffee table. "
            "Keep the rest of the room exactly the same."
        ),
        "description": "white ceramic vase with flowers",
    },
    {
        "name": "ball",
        "prompt": (
            "Add a yellow round ball in the right corner of the room next to the sofa. "
            "Keep the rest of the room exactly the same."
        ),
        "description": "yellow round ball",
    },
]

BASE_PROMPT   = "A modern living room with a sofa and a wooden coffee table."
TIER_A_LAYERS = list(TIER_A)
ALL_LAYERS    = list(TIER_ALL)

LORA_TRIGGER  = "Convert this sketch into real life version, follow exact structure."

# ============================================================
# RMBG-2.0 (BiRefNet) -- semantic background removal
# ============================================================

def load_rmbg(cache_dir: str, device: str = "cuda"):
    """Load briaai/RMBG-2.0 for semantic foreground mask extraction."""
    try:
        import kornia  # noqa: F401
    except ImportError:
        raise ImportError(
            "briaai/RMBG-2.0 requires 'kornia'. Install it with:\n"
            "  pip install kornia"
        ) from None

    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0",
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).to(device).eval()
    return model


def rmbg_extract_mask(model, obj_img: Image.Image,
                      threshold: float = 0.5,
                      device: str = "cuda") -> np.ndarray:
    """
    Semantic foreground mask via RMBG-2.0.
    Returns bool array (H, W) -- True where the object is.
    Works on white objects on white backgrounds (unlike pixel-diff).
    """
    from torchvision import transforms as T

    tf = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    inp = tf(obj_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)[-1].sigmoid().cpu().squeeze()
    mask_pil = T.ToPILImage()(pred).resize(obj_img.size, Image.BILINEAR)
    return np.array(mask_pil) > int(threshold * 255)


# ============================================================
# Sketch preprocessing (ControlNet mode only)
# ============================================================

def preprocess_sketch(sketch_path: str, size: int = 1024) -> Image.Image:
    """
    Prepare a hand-drawn sketch for lineart ControlNet input.
    Converts black-on-white to white-on-black (ControlNet expected format).
    """
    img = Image.open(sketch_path).convert("L").resize((size, size), Image.LANCZOS)
    arr = np.array(img)
    if arr.mean() > 128:          # dark lines on light background -> invert
        arr = 255 - arr
    return Image.fromarray(arr).convert("RGB")


# ============================================================
# Stage A helpers
# ============================================================

def stage_a_lora(
    pipe,
    edits: List[dict],
    sketch_dir: str,
    lora_id: str,
    seed: int,
    num_steps: int,
    guidance: float,
    height: int,
    width: int,
    rmbg_model,
    rmbg_threshold: float,
    out_dir: str,
    device: str,
) -> Dict[str, Tuple[Image.Image, np.ndarray]]:
    """
    LoRA mode Stage A: pass sketch as Kontext reference + LoRA trigger phrase.
    Returns {name: (obj_img, obj_mask)} for each edit.
    """
    print("\n=== Stage A: Sketch -> Object (LoRA mode) ===")
    print(f"  Loading LoRA: {lora_id}")
    pipe.load_lora_weights(lora_id)

    results: Dict[str, Tuple[Image.Image, np.ndarray]] = {}
    for edit in edits:
        name   = edit["name"]
        desc   = edit.get("description", name)
        sketch_path = os.path.join(sketch_dir, f"{name}.png")

        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found: {sketch_path}\n"
                f"Create a hand-drawn sketch and save it as '{name}.png' in --sketch_dir."
            )

        sketch_pil = Image.open(sketch_path).convert("RGB").resize(
            (width, height), Image.LANCZOS
        )
        prompt_a = (
            f"{LORA_TRIGGER} {desc} on a plain white background, "
            "no shadows, studio lighting, photorealistic."
        )
        print(f"  [{name}] generating from sketch ({num_steps} steps, g={guidance:.1f}) ...")
        generator = torch.Generator(device=device).manual_seed(seed)
        obj_img = pipe(
            image=sketch_pil,
            prompt=prompt_a,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            height=height, width=width,
            max_sequence_length=512,
            generator=generator,
            output_type="pil",
        ).images[0]
        obj_img.save(os.path.join(out_dir, f"sketch_gen_{name}.png"))

        print(f"  [{name}] extracting mask (RMBG-2.0) ...")
        obj_mask = rmbg_extract_mask(rmbg_model, obj_img, rmbg_threshold, device)
        _save_mask_preview(obj_img, obj_mask,
                           os.path.join(out_dir, f"sketch_mask_{name}.png"))
        results[name] = (obj_img, obj_mask)

    print("  Unloading LoRA ...")
    pipe.unload_lora_weights()
    return results


def stage_a_controlnet(
    edits: List[dict],
    sketch_dir: str,
    controlnet_id: str,
    controlnet_scale: float,
    seed: int,
    num_steps: int,
    guidance: float,
    height: int,
    width: int,
    rmbg_model,
    rmbg_threshold: float,
    out_dir: str,
    hf_token: str,
    cache_dir: str,
    device: str,
) -> Dict[str, Tuple[Image.Image, np.ndarray]]:
    """
    ControlNet mode Stage A: FLUX.1-dev + lineart ControlNet.
    Loads the ControlNet pipeline, generates all objects, then unloads
    before returning so GPU memory is free for the Kontext pipeline.
    """
    print("\n=== Stage A: Sketch -> Object (ControlNet mode) ===")

    # Diffusers version check (FluxControlNetPipeline requires >= 0.31)
    import diffusers as _diff
    ver = tuple(int(x) for x in _diff.__version__.split(".")[:2])
    if ver < (0, 31):
        raise RuntimeError(
            f"FluxControlNetPipeline requires diffusers>=0.31, got {_diff.__version__}.\n"
            "Upgrade with: pip install -U diffusers"
        )

    from diffusers import FluxControlNetPipeline, FluxControlNetModel

    print(f"  Loading ControlNet: {controlnet_id}")
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir,
        token=hf_token,
    )
    pipe_cn = FluxControlNetPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        token=hf_token,
    ).to(device)

    results: Dict[str, Tuple[Image.Image, np.ndarray]] = {}
    for edit in edits:
        name   = edit["name"]
        desc   = edit.get("description", name)
        sketch_path = os.path.join(sketch_dir, f"{name}.png")

        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found: {sketch_path}\n"
                f"Create a hand-drawn sketch and save it as '{name}.png' in --sketch_dir."
            )

        edge_map = preprocess_sketch(sketch_path, size=width)
        edge_map.save(os.path.join(out_dir, f"sketch_edge_{name}.png"))

        prompt_a = (
            f"{desc} on a plain white background, "
            "no shadows, studio lighting, photorealistic, high quality."
        )
        print(f"  [{name}] generating from sketch ({num_steps} steps, "
              f"cn_scale={controlnet_scale}) ...")
        generator = torch.Generator(device=device).manual_seed(seed)
        obj_img = pipe_cn(
            prompt=prompt_a,
            control_image=edge_map,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            height=height, width=width,
            generator=generator,
            output_type="pil",
        ).images[0]
        obj_img.save(os.path.join(out_dir, f"sketch_gen_{name}.png"))

        print(f"  [{name}] extracting mask (RMBG-2.0) ...")
        obj_mask = rmbg_extract_mask(rmbg_model, obj_img, rmbg_threshold, device)
        _save_mask_preview(obj_img, obj_mask,
                           os.path.join(out_dir, f"sketch_mask_{name}.png"))
        results[name] = (obj_img, obj_mask)

    print("  Unloading ControlNet pipeline ...")
    del pipe_cn, controlnet
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _save_mask_preview(obj_img: Image.Image, mask: np.ndarray, path: str):
    """Save a side-by-side preview: generated object + extracted mask."""
    W, H = obj_img.size
    mask_preview = Image.fromarray(mask.astype(np.uint8) * 255, "L").convert("RGB")
    combined = Image.new("RGB", (W * 2, H))
    combined.paste(obj_img, (0, 0))
    combined.paste(mask_preview, (W, 0))
    combined.save(path)


def _save_bbox_overlay(scene: Image.Image, bbox: Tuple[int, int, int, int], path: str):
    """Draw the VLM-determined bounding box on the scene image for visual inspection."""
    y1, y2, x1, x2 = bbox
    img  = scene.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 0), width=4)
    img.save(path)


# ============================================================
# VLM Placement Reasoning (Claude Vision)
# ============================================================

def load_vlm(model_id: str, cache_dir: str, device: str = "cpu"):
    """
    Load a VLM for placement reasoning.

    device="cpu"  → bfloat16 on CPU. Safe alongside FLUX on any GPU.
                    Inference ~30-60 s per object.
    device="cuda" → tries 4-bit NF4 (bitsandbytes) first.
                    Qwen2.5-VL-7B: ~4 GB VRAM in 4-bit vs ~15 GB in fp16.
                    If bitsandbytes is missing/outdated, falls back to
                    bf16 with device_map="auto" automatically.
                    To enable 4-bit: pip install -U bitsandbytes>=0.46.1

    GPU memory (4-bit path):
      Kontext-dev          ~24 GB
      Qwen2.5-VL-7B int4   ~4 GB
      Total                ~28 GB  (fits on 40 GB A100/H100)
    """
    from transformers import AutoProcessor

    print(f"  Loading VLM '{model_id}' on {device} ...")

    def _load_model(load_kwargs: dict, to_device: bool) -> object:
        """Try model classes in order; return first that succeeds."""
        # Pick class order based on model_id: Qwen2-VL (not 2.5) should use
        # Qwen2VLForConditionalGeneration to avoid visual-encoder shape mismatches.
        is_qwen25 = "Qwen2.5" in model_id or "Qwen2_5" in model_id
        classes = []
        if is_qwen25:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                classes.append(Qwen2_5_VLForConditionalGeneration)
            except ImportError:
                pass
        try:
            from transformers import Qwen2VLForConditionalGeneration
            classes.append(Qwen2VLForConditionalGeneration)
        except ImportError:
            pass
        if not is_qwen25:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                classes.append(Qwen2_5_VLForConditionalGeneration)
            except ImportError:
                pass

        last_err = None
        for cls in classes:
            try:
                m = cls.from_pretrained(model_id, **load_kwargs)
                if to_device:
                    m = m.to(device)
                return m.eval()
            except Exception as e:
                last_err = e
                continue

        # Final fallback: AutoModel with trust_remote_code
        try:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, **load_kwargs
            )
            if to_device:
                m = m.to(device)
            return m.eval()
        except Exception as e:
            last_err = e

        raise RuntimeError(
            f"Could not load VLM '{model_id}'. Last error: {last_err}"
        )

    model = None
    mode  = ""

    if device == "cuda":
        # ── Attempt 1: 4-bit NF4 via bitsandbytes ────────────────────────
        try:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = _load_model(
                dict(quantization_config=quant_cfg,
                     device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            mode = "4-bit NF4 GPU"
        except Exception as e:
            print(f"    [VLM] 4-bit load failed ({e!s:.80})")
            print(f"    [VLM] Falling back to bf16 on GPU "
                  f"(run: pip install -U bitsandbytes>=0.46.1 for 4-bit)")

        # ── Attempt 2: bf16 with device_map (no quantization) ────────────
        if model is None:
            model = _load_model(
                dict(torch_dtype=torch.bfloat16,
                     device_map="auto", cache_dir=cache_dir),
                to_device=False,
            )
            mode = "bf16 GPU"
    else:
        model = _load_model(
            dict(torch_dtype=torch.bfloat16, cache_dir=cache_dir),
            to_device=True,
        )
        mode = "bf16 CPU"

    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=True,
    )
    print(f"  VLM loaded  ({mode})")
    return model, processor


def reason_placement_vlm(
    vlm_model,
    vlm_processor,
    scene_img: Image.Image,
    obj_img: Image.Image,
    obj_name: str,
    obj_description: str,
    max_new_tokens: int = 512,
) -> dict:
    """
    Ask the VLM to reason about where and how to place the object in the scene.

    Tested with Qwen2-VL-2B / 7B.  Any HuggingFace multi-image VLM that
    supports apply_chat_template should work with the same interface.

    Returns dict with:
      reasoning -- brief explanation
      center_x  -- horizontal centre [0,1] of scene width
      center_y  -- vertical centre   [0,1] of scene height
      width     -- object width  as fraction of scene width
      height    -- object height as fraction of scene height
    """
    device = next(vlm_model.parameters()).device

    def _resize(img: Image.Image, max_side: int) -> Image.Image:
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return img.convert("RGB")

    scene_small = _resize(scene_img, 768)
    obj_small   = _resize(obj_img,   512)

    user_msg = f"""I want to insert a {obj_description} into this indoor room scene.

Image 1: the current scene.
Image 2: the {obj_description} on a plain white background.

Think step by step:
1. Room layout -- identify walls, floor area, existing furniture.
2. Natural placement -- where would a real {obj_name} go in this room?
3. Perspective scale -- objects further back appear smaller and sit higher.
4. Stability -- the object must rest on a visible floor or surface.

Return ONLY valid JSON, no markdown, no extra text:
{{
  "reasoning": "<2-3 sentences>",
  "center_x": <float 0.0-1.0, horizontal centre fraction of scene width>,
  "center_y": <float 0.0-1.0, vertical centre fraction of scene height>,
  "width":    <float 0.0-1.0, object width  fraction of scene width>,
  "height":   <float 0.0-1.0, object height fraction of scene height>
}}
Origin (0,0) = top-left corner."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": scene_small},
                {"type": "image", "image": obj_small},
                {"type": "text",  "text": user_msg},
            ],
        }
    ]

    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Extract PIL images in message order for the processor
    images_in_order = [
        item["image"]
        for msg in messages
        for item in msg["content"]
        if item["type"] == "image"
    ]
    inputs = vlm_processor(
        text=[text],
        images=images_in_order,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        gen_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # Trim input prompt tokens from output
    out_ids  = gen_ids[:, inputs["input_ids"].shape[1]:]
    response = vlm_processor.batch_decode(
        out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    if "```" in response:
        response = response[response.find("{") : response.rfind("}") + 1]
    result = json.loads(response)

    for key in ("center_x", "center_y", "width", "height"):
        result[key] = float(np.clip(result[key], 0.02, 0.98))

    print(f"    VLM:    cx={result['center_x']:.2f}  cy={result['center_y']:.2f}  "
          f"w={result['width']:.2f}  h={result['height']:.2f}")
    print(f"    Reason: {result['reasoning']}")
    return result


def vlm_placement_to_bbox(
    placement: dict, H: int, W: int
) -> Tuple[int, int, int, int]:
    """Convert fractional VLM placement dict -> pixel bbox (y1, y2, x1, x2)."""
    cx = placement["center_x"] * W
    cy = placement["center_y"] * H
    pw = placement["width"]    * W
    ph = placement["height"]   * H
    x1 = max(0, int(cx - pw / 2))
    x2 = min(W, int(cx + pw / 2))
    y1 = max(0, int(cy - ph / 2))
    y2 = min(H, int(cy + ph / 2))
    return y1, y2, x1, x2


def bbox_to_token_mask(
    bbox: Tuple[int, int, int, int],
    h_lat: int, w_lat: int,
    H: int, W: int,
) -> np.ndarray:
    """Convert pixel bbox (y1,y2,x1,x2) -> flat bool token mask of shape (h_lat*w_lat,)."""
    y1, y2, x1, x2 = bbox
    ty1 = int(y1 / H * h_lat)
    ty2 = int(np.ceil(y2 / H * h_lat))
    tx1 = int(x1 / W * w_lat)
    tx2 = int(np.ceil(x2 / W * w_lat))
    mask_2d = np.zeros((h_lat, w_lat), dtype=bool)
    mask_2d[ty1:ty2, tx1:tx2] = True
    return mask_2d.ravel()


def paste_object_at_bbox(
    scene: Image.Image,
    obj_img: Image.Image,
    obj_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    feather_sigma: float = 8.0,
) -> Image.Image:
    """
    Paste the generated object at a VLM-specified pixel bounding box.
    No token-grid snapping: bbox is pixel-accurate.
    """
    y1, y2, x1, x2 = bbox
    bh, bw = y2 - y1, x2 - x1
    H, W   = scene.size[1], scene.size[0]

    if bh <= 0 or bw <= 0:
        print("    [paste_object_at_bbox] degenerate bbox -- skipping paste.")
        return scene

    r, c = np.where(obj_mask)
    if len(r) == 0:
        print("    [paste_object_at_bbox] empty RMBG mask -- skipping paste.")
        return scene

    oy1, oy2 = int(r.min()), int(r.max()) + 1
    ox1, ox2 = int(c.min()), int(c.max()) + 1
    obj_crop  = obj_img.crop((ox1, oy1, ox2, oy2))
    mask_crop = Image.fromarray(obj_mask[oy1:oy2, ox1:ox2].astype(np.uint8) * 255, "L")

    obj_r  = obj_crop.resize((bw, bh), Image.LANCZOS)
    mask_r = mask_crop.resize((bw, bh), Image.BILINEAR)

    alpha_small = feather_mask(np.array(mask_r) > 127, feather_sigma)
    alpha_full  = np.zeros((H, W), dtype=float)
    alpha_full[y1:y2, x1:x2] = alpha_small
    alpha3 = alpha_full[:, :, np.newaxis]

    scene_f = np.array(scene).astype(float)
    paste_f = scene_f.copy()
    paste_f[y1:y2, x1:x2] = np.array(obj_r).astype(float)
    result  = alpha3 * paste_f + (1.0 - alpha3) * scene_f
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ============================================================
# Placement: probe-based fallback (no VLM key)
# ============================================================

def _probe_mask_to_bbox(
    probe_flat_mask: np.ndarray,
    h_lat: int, w_lat: int,
    H: int, W: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Token-level probe mask -> pixel bbox (y1,y2,x1,x2), or None if empty."""
    px = _token_to_pixel(probe_flat_mask, h_lat, w_lat, H, W)
    rows = np.where(px.any(axis=1))[0]
    cols = np.where(px.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _crop_resize_obj(
    obj_img: Image.Image,
    obj_mask: np.ndarray,
    bw: int, bh: int,
    feather_sigma: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Tight-crop obj using RMBG mask, resize to (bw, bh).
    Returns (obj_arr float (bh,bw,3), alpha_small float (bh,bw)), or (None,None).
    """
    r, c = np.where(obj_mask)
    if len(r) == 0:
        return None, None
    oy1, oy2 = int(r.min()), int(r.max()) + 1
    ox1, ox2 = int(c.min()), int(c.max()) + 1
    obj_crop  = obj_img.crop((ox1, oy1, ox2, oy2))
    mask_crop = Image.fromarray(obj_mask[oy1:oy2, ox1:ox2].astype(np.uint8) * 255, "L")
    obj_r  = obj_crop.resize((bw, bh), Image.LANCZOS)
    mask_r = mask_crop.resize((bw, bh), Image.BILINEAR)
    alpha_small = feather_mask(np.array(mask_r) > 127, feather_sigma)
    return np.array(obj_r).astype(float), alpha_small


def lab_luminance_transfer(
    obj_arr: np.ndarray,
    probe_region: np.ndarray,
) -> np.ndarray:
    """
    Correct the sketch object's brightness to match the scene's local lighting.

    Transfers mean + std of the L (luminance) channel from probe_region → obj_arr.
    A/B channels (actual color) are kept from obj_arr, so a yellow bicycle stays
    yellow but gets warm room shadows instead of studio-white brightness.

    Reinhard et al. (2001), "Color Transfer between Images."
    Only L is transferred — not A/B — to avoid unintended color casts.

    Returns uint8 RGB array with corrected luminance.
    """
    try:
        from skimage.color import rgb2lab, lab2rgb
    except ImportError:
        return obj_arr  # skimage not available; skip silently

    obj_f    = obj_arr.astype(np.float32) / 255.0
    probe_f  = probe_region.astype(np.float32) / 255.0

    obj_lab   = rgb2lab(obj_f)
    probe_lab = rgb2lab(probe_f)

    obj_L   = obj_lab[:, :, 0]      # L in [0, 100]
    probe_L = probe_lab[:, :, 0]

    obj_mean,   obj_std   = float(obj_L.mean()),   float(obj_L.std())
    probe_mean, probe_std = float(probe_L.mean()), float(probe_L.std())

    # Skip degenerate cases (flat-colour or near-flat objects)
    if obj_std > 0.5:
        scale          = (probe_std / obj_std) if obj_std > 0 else 1.0
        corrected_L    = (obj_L - obj_mean) * scale + probe_mean
        obj_lab[:, :, 0] = np.clip(corrected_L, 0, 100)

    return (lab2rgb(obj_lab) * 255.0).clip(0, 255).astype(np.uint8)


def poisson_blend_into_scene(
    scene: Image.Image,
    obj_u8: np.ndarray,
    obj_mask_small: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Gradient-domain seamless cloning of obj_u8 into scene at bbox.

    Uses cv2.seamlessClone (Pérez et al., SIGGRAPH 2003).  The Poisson solver
    copies the *gradient field* of the source, then finds pixel values whose
    gradient matches while satisfying the destination boundary conditions.
    Result: boundary is indistinguishable from the destination even when source
    and destination colours differ slightly.

    obj_u8       : uint8 (bh, bw, 3) RGB -- tight-cropped & resized to bbox
    obj_mask_small: bool (bh, bw) -- True where object pixels are
    """
    try:
        import cv2
    except ImportError:
        # Graceful fallback: simple alpha paste
        y1, y2, x1, x2 = bbox
        H, W = scene.size[1], scene.size[0]
        alpha = obj_mask_small.astype(float)[:, :, np.newaxis]
        sarr  = np.array(scene).astype(float)
        patch = sarr[y1:y2, x1:x2].copy()
        patch = alpha * obj_u8 + (1.0 - alpha) * patch
        sarr[y1:y2, x1:x2] = patch
        return Image.fromarray(sarr.clip(0, 255).astype(np.uint8))

    y1, y2, x1, x2 = bbox
    bh, bw  = y2 - y1, x2 - x1
    H, W    = scene.size[1], scene.size[0]

    scene_bgr = cv2.cvtColor(np.array(scene).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Full-scene source: scene with the (colour-corrected) object pasted in
    src_bgr = scene_bgr.copy()
    src_bgr[y1:y2, x1:x2] = cv2.cvtColor(obj_u8, cv2.COLOR_RGB2BGR)

    # Full-scene mask (white = "copy from source", black = "keep destination")
    mask_small = (obj_mask_small.astype(np.uint8) * 255)
    # Erode 3 px so seamlessClone never tries to solve right at the RMBG boundary
    kernel     = np.ones((5, 5), np.uint8)
    mask_small = cv2.erode(mask_small, kernel, iterations=1)

    mask_full            = np.zeros((H, W), dtype=np.uint8)
    mask_full[y1:y2, x1:x2] = mask_small

    if mask_full.sum() == 0:
        return scene  # nothing to blend

    center = (x1 + bw // 2, y1 + bh // 2)   # (cx, cy) — OpenCV x-first

    try:
        result_bgr = cv2.seamlessClone(
            src_bgr, scene_bgr, mask_full, center, cv2.NORMAL_CLONE
        )
        return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    except cv2.error:
        return scene   # safety fallback


def create_harmonized_composite(
    scene: Image.Image,
    img_probe: Image.Image,
    obj_img: Image.Image,
    obj_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    feather_sigma: float = 8.0,
    harmonize: str = "lab+poisson",
) -> Image.Image:
    """
    Build a scene-harmonized rough composite so Kontext only needs light
    refinement (strength≈0.45) instead of wholesale relighting (strength≈0.8).

    Pipeline:
      1. [lab]    LAB luminance transfer from probe region → sketch object
                  → object brightness/shadows match scene; colours unchanged
                  (Reinhard 2001)
      2. [poisson] Poisson seamless cloning into scene
                  → gradient-domain boundary; no hard edge artefacts
                  (Pérez 2003, implemented as cv2.seamlessClone)
      Fallback:   feathered alpha blend if either lib is unavailable

    harmonize: 'lab+poisson' (default, best) | 'lab' | 'poisson' | 'none'
    """
    y1, y2, x1, x2 = bbox
    bh, bw = y2 - y1, x2 - x1
    H, W   = scene.size[1], scene.size[0]

    if bh <= 0 or bw <= 0:
        return scene

    obj_arr, alpha_small = _crop_resize_obj(obj_img, obj_mask, bw, bh, feather_sigma)
    if obj_arr is None:
        return scene

    obj_u8     = obj_arr.astype(np.uint8)
    mask_bool  = alpha_small > 0.5   # binary mask at bbox size for Poisson

    # ── Step 1: LAB luminance correction ─────────────────────────────────────
    if "lab" in harmonize:
        probe_region = np.array(img_probe)[y1:y2, x1:x2]   # (bh, bw, 3)
        obj_u8 = lab_luminance_transfer(obj_u8, probe_region)

    # ── Step 2: Poisson seamless cloning ─────────────────────────────────────
    if "poisson" in harmonize:
        return poisson_blend_into_scene(scene, obj_u8, mask_bool, bbox)

    # ── Fallback: feathered alpha blend ──────────────────────────────────────
    alpha_full = np.zeros((H, W), dtype=float)
    alpha_full[y1:y2, x1:x2] = alpha_small
    alpha3    = alpha_full[:, :, np.newaxis]
    scene_arr = np.array(scene).astype(float)
    paste_arr = scene_arr.copy()
    paste_arr[y1:y2, x1:x2] = obj_u8.astype(float)
    result = alpha3 * paste_arr + (1.0 - alpha3) * scene_arr
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def create_sketch_probe_blend(
    scene: Image.Image,
    img_probe: Image.Image,
    obj_img: Image.Image,
    obj_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    feather_sigma: float = 8.0,
    sketch_alpha: float = 0.5,
) -> Image.Image:
    """
    Legacy: 50/50 alpha blend of sketch appearance + probe lighting.
    Kept for reference; use create_harmonized_composite instead.
    """
    y1, y2, x1, x2 = bbox
    bh, bw = y2 - y1, x2 - x1
    H, W   = scene.size[1], scene.size[0]

    if bh <= 0 or bw <= 0:
        return scene

    obj_arr, alpha_small = _crop_resize_obj(obj_img, obj_mask, bw, bh, feather_sigma)
    if obj_arr is None:
        return scene

    alpha_full = np.zeros((H, W), dtype=float)
    alpha_full[y1:y2, x1:x2] = alpha_small
    alpha3 = alpha_full[:, :, np.newaxis]

    scene_arr  = np.array(scene).astype(float)
    probe_arr  = np.array(img_probe).astype(float)
    sketch_arr = scene_arr.copy()
    sketch_arr[y1:y2, x1:x2] = obj_arr

    result = (
        alpha3 * (sketch_alpha * sketch_arr + (1.0 - sketch_alpha) * probe_arr)
        + (1.0 - alpha3) * scene_arr
    )
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def paste_object_at_probe(
    scene: Image.Image,
    obj_img: Image.Image,
    obj_mask: np.ndarray,
    probe_flat_mask: np.ndarray,
    h_lat: int, w_lat: int,
    feather_sigma: float = 8.0,
) -> Image.Image:
    """Legacy pure-sketch paste (no lighting blend). Used only as a last-resort fallback."""
    H, W = scene.size[1], scene.size[0]
    bbox = _probe_mask_to_bbox(probe_flat_mask, h_lat, w_lat, H, W)
    if bbox is None:
        print("    [paste_object_at_probe] WARNING: empty probe mask -- skipping paste.")
        return scene
    y1, y2, x1, x2 = bbox
    bh, bw = y2 - y1, x2 - x1
    obj_arr, alpha_small = _crop_resize_obj(obj_img, obj_mask, bw, bh, feather_sigma)
    if obj_arr is None:
        print("    [paste_object_at_probe] WARNING: empty RMBG mask -- skipping paste.")
        return scene
    alpha_full = np.zeros((H, W), dtype=float)
    alpha_full[y1:y2, x1:x2] = alpha_small
    alpha3 = alpha_full[:, :, np.newaxis]
    scene_f = np.array(scene).astype(float)
    paste_f = scene_f.copy()
    paste_f[y1:y2, x1:x2] = obj_arr
    result = alpha3 * paste_f + (1.0 - alpha3) * scene_f
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ============================================================
# Stage B: scene insertion chain
# ============================================================

def run_sketch_chain(
    pipe,
    base: Image.Image,
    edits: List[dict],
    generated: Dict[str, Tuple[Image.Image, np.ndarray]],
    h_lat: int, w_lat: int,
    seed: int, num_steps: int, probe_steps: int,
    guidance: float, height: int, width: int,
    strength: float, cutoff: float,
    vital_layers: List[int],
    threshold: float,
    feather_sigma: float,
    out_dir: str,
    mode_tag: str,
    device: str,
    vlm_pair: Optional[Tuple] = None,
    sketch_alpha: float = 0.5,
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    Stage B: place each generated object into the scene sequentially.

    Two placement modes:

    VLM mode (--vlm_model provided):
      An open-source VLM (default: Qwen2-VL-2B) looks at the current
      accumulated scene + the generated object and reasons about WHERE to place
      it and at what SCALE.  Returns a pixel-accurate bounding box with no
      16-pixel token-grid snapping.
      Flow: VLM reason -> place -> K/V inject -> bg_replace

    Probe fallback (no --vlm_model):
      Runs a lightweight text-only Kontext generation, diffs it against the
      current scene to find the region the model would change, and uses that
      token mask as the bounding area for placement.
      Flow: probe -> place -> K/V inject -> bg_replace

    In both cases each object is inserted one at a time into the accumulated
    scene (img_prev = imgs[-1]), so later objects see earlier ones.
    """
    use_vlm = vlm_pair is not None
    placement_log: List[dict] = []
    imgs, obj_masks = [base], []
    all_obj_mask = np.zeros(h_lat * w_lat, dtype=bool)

    for i, edit in enumerate(edits):
        name   = edit["name"]
        prompt = edit["prompt"]
        desc   = edit.get("description", name)
        img_prev = imgs[-1]
        H, W     = img_prev.size[1], img_prev.size[0]

        if name not in generated:
            print(f"\n  [{mode_tag}] step {i+1}  {name}: no generated object -- SKIPPING")
            imgs.append(img_prev)
            obj_masks.append(np.zeros(h_lat * w_lat, dtype=bool))
            continue

        obj_img, obj_mask = generated[name]
        print(f"\n  [{mode_tag}] step {i+1}/{len(edits)}  {name}")

        # ── PROBE: always run regardless of VLM mode ──────────────────────────
        # The probe serves two purposes:
        #   (a) Placement/mask detection (probe-only mode)
        #   (b) Lighting reference for create_sketch_probe_blend (both modes)
        # Running Kontext on the scene + text prompt gives us the object at the
        # RIGHT LOCATION with CORRECT SCENE LIGHTING.  We borrow that lighting
        # even when VLM handles the placement decision.
        print(f"    probe  ({probe_steps} steps) ...")
        img_probe = run_standard(pipe, img_prev, prompt,
                                 seed, probe_steps, guidance, height, width)
        img_probe.save(os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_probe.png"))

        target_raw  = pixel_to_token_mask(img_prev, img_probe, h_lat, w_lat, threshold)
        base_stable = ~pixel_to_token_mask(base, img_probe, h_lat, w_lat, threshold)
        target_clean = target_raw & ~base_stable
        print(f"    probe target={target_clean.mean()*100:.1f}%")

        if use_vlm:
            # ── VLM PATH ──────────────────────────────────────────────────────
            # VLM decides WHERE and HOW BIG.
            # Probe provides SCENE-CONSISTENT LIGHTING for the blend.
            print(f"    asking VLM where/how to place '{name}' ...")
            _vlm_model, _vlm_proc = vlm_pair
            placement = reason_placement_vlm(
                _vlm_model, _vlm_proc,
                img_prev, obj_img, name, desc,
            )
            placement_log.append({"step": i + 1, "name": name, **placement})

            bbox           = vlm_placement_to_bbox(placement, H, W)
            obj_token_mask = bbox_to_token_mask(bbox, h_lat, w_lat, H, W)
            obj_masks.append(obj_token_mask)

            _save_bbox_overlay(img_prev, bbox,
                os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_vlm_bbox.png"))

        else:
            # ── PROBE PATH ────────────────────────────────────────────────────
            # Probe decides WHERE and HOW BIG (Kontext's natural placement).
            obj_masks.append(target_clean)
            color_overlay(img_prev, target_clean, h_lat, w_lat,
                          color=(0, 180, 255), alpha=0.45).save(
                os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_target.png"))

            bbox = _probe_mask_to_bbox(target_clean, h_lat, w_lat, H, W)
            if bbox is None:
                print(f"    WARNING: empty probe mask for '{name}' -- skipping.")
                imgs.append(img_prev)
                continue
            obj_token_mask = bbox_to_token_mask(bbox, h_lat, w_lat, H, W)

        # ── PLACE: probe lighting + sketch appearance blend ────────────────────
        # The sketch object has studio white-bg lighting.
        # The probe image has scene-consistent room lighting at the right location.
        # Blend both: sketch_alpha * sketch + (1-sketch_alpha) * probe
        # so that Kontext's integration step starts from a better base.
        rough_composite = create_sketch_probe_blend(
            scene=img_prev,
            img_probe=img_probe,
            obj_img=obj_img,
            obj_mask=obj_mask,
            bbox=bbox,
            feather_sigma=feather_sigma,
            sketch_alpha=sketch_alpha,
        )
        bg_mask_for_kv = ~obj_token_mask

        rough_composite.save(
            os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_rough.png"))

        # 3. REFINE: Kontext K/V inject from rough_composite
        #    The pasted object in rough_composite is a visual hint for Kontext
        #    (shape, colour, pose).  K/V injection locks background tokens so
        #    the scene outside the object region is not changed.
        print(f"    refine ({num_steps} steps, s={strength}, cutoff={cutoff}) ...")
        img_inject = run_kv_injected(
            pipe,
            canvas=rough_composite,
            prompt=prompt,
            background_mask=bg_mask_for_kv,
            seed=seed, num_steps=num_steps,
            guidance=guidance, height=height, width=width,
            strength=strength, cutoff=cutoff,
            vital_layers=vital_layers,
            device=device,
        )
        img_inject.save(
            os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_inject.png"))

        # 4. BG-REPLACE: hard pixel-copy original base over pure-background tokens
        all_obj_mask = all_obj_mask | obj_token_mask
        px_pure_bg   = _token_to_pixel(~all_obj_mask, h_lat, w_lat, H, W)
        inject_arr   = np.array(img_inject).astype(float)
        base_arr     = np.array(base).astype(float)
        inject_arr[px_pure_bg] = base_arr[px_pure_bg]
        img_curr = Image.fromarray(inject_arr.clip(0, 255).astype(np.uint8))
        img_curr.save(
            os.path.join(out_dir, f"sk_{mode_tag}_step{i+1}_{name}_result.png"))
        imgs.append(img_curr)

    if placement_log:
        log_path = os.path.join(out_dir, f"sk_{mode_tag}_vlm_placements.json")
        with open(log_path, "w") as f:
            json.dump(placement_log, f, indent=2)
        print(f"\n  VLM placements logged: {log_path}")

    return imgs, obj_masks


# ============================================================
# Visualisation
# ============================================================

def build_comparison_grid(
    imgs_by_mode: Dict[str, List[Image.Image]],
    edits: List[dict],
    out_dir: str,
    strength: float,
    feather_sigma: float,
):
    modes = list(imgs_by_mode.keys())
    ncols = len(edits) + 1
    labels = {
        "bg_replace": f"Text-only\nbg_replace (s={strength})",
        "sketch_lora": f"Sketch LoRA\n(σ={feather_sigma}px)",
        "sketch_cn": f"Sketch ControlNet\n(σ={feather_sigma}px)",
    }
    images, titles = [], []
    for mode in modes:
        lbl  = labels.get(mode, mode)
        imgs = imgs_by_mode[mode]
        images.append(imgs[0])
        titles.append(f"{lbl}\nbase")
        for j, edit in enumerate(edits):
            images.append(imgs[j + 1])
            titles.append(f"{lbl}\n{edit['name']}")

    save_grid(
        images, titles,
        os.path.join(out_dir, "KEY_RESULT_comparison.png"),
        ncols=ncols,
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Sketch-conditioned multi-step object insertion via FLUX Kontext."
    )
    p.add_argument("--sketch_dir",       required=True,
                   help="Folder with {name}.png hand-drawn sketches.")
    p.add_argument("--hf_token",         required=True)
    p.add_argument("--cache_dir",        default="./models")
    p.add_argument("--out_dir",          default="results/phase1_sketch")
    p.add_argument("--config",           default=None,
                   help="JSON list of {name, prompt, description}. Overrides built-in EDITS.")
    p.add_argument("--mode",             default="lora",
                   choices=["lora", "controlnet", "all"],
                   help="'lora': Kontext+LoRA (single model). "
                        "'controlnet': FLUX ControlNet (stronger shape control). "
                        "'all': runs lora + controlnet + text-only bg_replace for comparison.")
    p.add_argument("--lora_id",          default="gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA")
    p.add_argument("--lora_guidance",    type=float, default=4.0,
                   help="guidance_scale for Stage A LoRA generation (4.0-5.0 recommended).")
    p.add_argument("--controlnet_id",    default="promeai/FLUX.1-controlnet-lineart-promeai",
                   help="Lineart ControlNet model (better for hand-drawn than Canny).")
    p.add_argument("--controlnet_scale", type=float, default=0.7,
                   help="ControlNet conditioning scale (0.5-0.8 for hand-drawn sketches).")
    p.add_argument("--rmbg_threshold",   type=float, default=0.5,
                   help="RMBG-2.0 alpha threshold for mask binarisation [0,1].")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--num_steps",        type=int,   default=28)
    p.add_argument("--probe_steps",      type=int,   default=28,
                   help="Denoising steps for the probe pass (default 28 = full quality, "
                        "gives accurate placement and better lighting reference).")
    p.add_argument("--guidance",         type=float, default=2.5,
                   help="guidance_scale for Stage B (scene insertion).")
    p.add_argument("--strength",         type=float, default=0.8,
                   help="K/V injection strength for background locking (default 0.8). "
                        "Higher = background locked tighter, Kontext has more freedom to "
                        "relight and integrate the object. 0.3 was too weak (copy-paste look).")
    p.add_argument("--cutoff",           type=float, default=0.65,
                   help="Inject K/V for first CUTOFF fraction of denoising steps (default 0.65).")
    p.add_argument("--threshold",        type=float, default=40.0,
                   help="Pixel diff threshold for probe mask detection.")
    p.add_argument("--feather_sigma",    type=float, default=8.0,
                   help="Gaussian blur radius (px) for object boundary feathering.")
    p.add_argument("--sketch_alpha",     type=float, default=0.5,
                   help="Blend weight for sketch appearance vs probe lighting in the rough "
                        "composite (default 0.5 = equal blend). "
                        "1.0 = pure sketch (original appearance, wrong lighting). "
                        "0.0 = pure probe (correct lighting, Kontext appearance). "
                        "0.3-0.5 recommended for natural integration.")
    p.add_argument("--all_layers",       action="store_true",
                   help="Use all 57 layers for K/V injection (TIER_A is default).")
    p.add_argument("--height",           type=int,   default=1024)
    p.add_argument("--width",            type=int,   default=1024)
    p.add_argument("--device",           default="cuda")
    # VLM placement (optional but strongly recommended)
    p.add_argument("--vlm_model",        default=None,
                   help="HuggingFace model ID for open-source VLM placement reasoning. "
                        "When set, the VLM looks at each scene + object and reasons "
                        "about WHERE and HOW to place it (pixel-accurate, perspective-aware). "
                        "Recommended: 'Qwen/Qwen2-VL-2B-Instruct' (fits on CPU alongside Kontext) "
                        "or 'Qwen/Qwen2-VL-7B-Instruct' (better reasoning, needs more RAM). "
                        "Without this flag the pipeline falls back to the probe method "
                        "(text-only diff -- less accurate, token-grid snapping).")
    p.add_argument("--vlm_device",       default="cpu",
                   help="Device for VLM inference (default: cpu). "
                        "CPU keeps VLM out of GPU VRAM so Kontext can use the full GPU. "
                        "Set to 'cuda' only if your GPU has 40+ GB free after loading Kontext.")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)
        print(f"Loaded {len(edits)} edits from {args.config}")

    h_lat = args.height // 16
    w_lat = args.width  // 16
    vital_layers = ALL_LAYERS if args.all_layers else TIER_A_LAYERS

    print(f"Edit sequence : {[e['name'] for e in edits]}")
    print(f"Mode          : {args.mode}")
    print(f"Sketch dir    : {args.sketch_dir}")
    print(f"feather_sigma : {args.feather_sigma} px")
    print(f"VLM placement : {args.vlm_model or 'probe fallback (no --vlm_model set)'}")

    # ── 0a. Load VLM for placement reasoning (optional) ──────────────────────
    vlm_pair = None
    if args.vlm_model:
        vlm_pair = load_vlm(args.vlm_model, args.cache_dir, args.vlm_device)

    # ── 0b. Load RMBG-2.0 (used by all modes for mask extraction) ────────────
    print("\nLoading RMBG-2.0 for foreground mask extraction ...")
    rmbg_model = load_rmbg(args.cache_dir, args.device)

    # ── 1. Base scene ─────────────────────────────────────────────────────────
    # Base is generated once; shared across all modes for fair comparison.
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))

    # We need Kontext even for base generation; load it now if starting with lora/all
    # For controlnet mode we may load Kontext after Stage A; still need it for base.
    print("Loading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )
    base = run_standard(pipe, grey, BASE_PROMPT,
                        args.seed, args.num_steps, args.guidance,
                        args.height, args.width)
    base.save(os.path.join(args.out_dir, "step0_base.png"))

    # ── 2. Run requested modes ─────────────────────────────────────────────────
    imgs_by_mode:      Dict[str, List[Image.Image]] = {}
    obj_masks_by_mode: Dict[str, List[np.ndarray]]  = {}

    run_lora       = args.mode in ("lora",  "all")
    run_controlnet = args.mode in ("controlnet", "all")
    run_text_only  = args.mode == "all"   # bg_replace baseline for comparison

    # ── Mode: LoRA ────────────────────────────────────────────────────────────
    if run_lora:
        generated = stage_a_lora(
            pipe=pipe,
            edits=edits,
            sketch_dir=args.sketch_dir,
            lora_id=args.lora_id,
            seed=args.seed,
            num_steps=args.num_steps,
            guidance=args.lora_guidance,
            height=args.height, width=args.width,
            rmbg_model=rmbg_model,
            rmbg_threshold=args.rmbg_threshold,
            out_dir=args.out_dir,
            device=args.device,
        )
        print("\n=== Stage B: Scene Insertion (LoRA) ===")
        imgs, masks = run_sketch_chain(
            pipe=pipe, base=base, edits=edits, generated=generated,
            h_lat=h_lat, w_lat=w_lat,
            seed=args.seed, num_steps=args.num_steps, probe_steps=args.probe_steps,
            guidance=args.guidance, height=args.height, width=args.width,
            strength=args.strength, cutoff=args.cutoff,
            vital_layers=vital_layers, threshold=args.threshold,
            feather_sigma=args.feather_sigma,
            out_dir=args.out_dir, mode_tag="lora", device=args.device,
            vlm_pair=vlm_pair, sketch_alpha=args.sketch_alpha,
        )
        imgs_by_mode["sketch_lora"]      = imgs
        obj_masks_by_mode["sketch_lora"] = masks
        imgs[-1].save(os.path.join(args.out_dir, "KEY_lora_final.png"))

    # ── Mode: ControlNet ──────────────────────────────────────────────────────
    if run_controlnet:
        # Offload Kontext if it is loaded, then reload after Stage A
        # (ControlNet FLUX.1-dev + Kontext both need large GPU memory)
        if pipe is not None:
            print("\nOffloading Kontext to free memory for ControlNet ...")
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            pipe = None

        generated_cn = stage_a_controlnet(
            edits=edits,
            sketch_dir=args.sketch_dir,
            controlnet_id=args.controlnet_id,
            controlnet_scale=args.controlnet_scale,
            seed=args.seed,
            num_steps=args.num_steps,
            guidance=args.guidance,
            height=args.height, width=args.width,
            rmbg_model=rmbg_model,
            rmbg_threshold=args.rmbg_threshold,
            out_dir=args.out_dir,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            device=args.device,
        )
        # Reload Kontext for Stage B
        print("\nReloading FLUX.1-Kontext-dev for Stage B ...")
        pipe = load_kontext_pipeline(
            hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
        )
        print("\n=== Stage B: Scene Insertion (ControlNet) ===")
        imgs_cn, masks_cn = run_sketch_chain(
            pipe=pipe, base=base, edits=edits, generated=generated_cn,
            h_lat=h_lat, w_lat=w_lat,
            seed=args.seed, num_steps=args.num_steps, probe_steps=args.probe_steps,
            guidance=args.guidance, height=args.height, width=args.width,
            strength=args.strength, cutoff=args.cutoff,
            vital_layers=vital_layers, threshold=args.threshold,
            feather_sigma=args.feather_sigma,
            out_dir=args.out_dir, mode_tag="cn", device=args.device,
            vlm_pair=vlm_pair, sketch_alpha=args.sketch_alpha,
        )
        imgs_by_mode["sketch_cn"]      = imgs_cn
        obj_masks_by_mode["sketch_cn"] = masks_cn
        imgs_cn[-1].save(os.path.join(args.out_dir, "KEY_controlnet_final.png"))

    # ── Mode: text-only bg_replace (comparison baseline) ─────────────────────
    if run_text_only:
        print("\n=== Text-only bg_replace (comparison baseline) ===")
        if pipe is None:
            pipe = load_kontext_pipeline(
                hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
            )
        imgs_br, masks_br = run_bg_replace_chain(
            pipe=pipe, base=base, edits=edits,
            h_lat=h_lat, w_lat=w_lat,
            seed=args.seed, num_steps=args.num_steps, probe_steps=args.probe_steps,
            guidance=args.guidance, height=args.height, width=args.width,
            strength=args.strength, cutoff=args.cutoff,
            vital_layers=vital_layers, threshold=args.threshold,
            feather_sigma=args.feather_sigma,
            out_dir=args.out_dir, device=args.device,
        )
        imgs_by_mode["bg_replace"]      = imgs_br
        obj_masks_by_mode["bg_replace"] = masks_br
        imgs_br[-1].save(os.path.join(args.out_dir, "KEY_bg_replace_final.png"))

    # ── 3. Visualisation ──────────────────────────────────────────────────────
    if len(imgs_by_mode) > 1:
        print("\n=== Building comparison grid ===")
        build_comparison_grid(
            imgs_by_mode, edits, args.out_dir,
            strength=args.strength, feather_sigma=args.feather_sigma,
        )
        print("  Saved: KEY_RESULT_comparison.png")

    mask_modes = {m: v for m, v in obj_masks_by_mode.items()
                  if any(m_.any() for m_ in v)}
    if mask_modes and len(imgs_by_mode) > 1:
        print("=== Building stability grid ===")
        build_stability_grid(
            base, imgs_by_mode, edits, mask_modes,
            h_lat, w_lat, args.out_dir,
            strength=args.strength, feather_sigma=args.feather_sigma,
        )
        print("  Saved: KEY_RESULT_stability.png")

    # ── 4. What to check ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("WHAT TO CHECK")
    print(f"{'='*72}")
    print(f"""
sketch_gen_{{name}}.png + sketch_mask_{{name}}.png
  Left: Stage A generated object.  Right: RMBG-2.0 mask.
  Mask should cleanly cover the object (including white objects like the vase).
  Leaks into background -> raise --rmbg_threshold (try 0.65).
  Clips part of object  -> lower --rmbg_threshold (try 0.3).

sk_{{mode}}_step{{i}}_{{name}}_vlm_bbox.png  [VLM mode only]
  Orange rectangle = where the VLM decided to place the object.
  If wrong position:  check sk_{{mode}}_vlm_placements.json for the reasoning.
  Adjust the prompt in EDITS or rerun; the VLM reasoning is stochastic.

sk_{{mode}}_step{{i}}_{{name}}_rough.png
  Object pasted at the VLM-determined location (before Kontext refine).
  Should look like a rough paste -- seams visible, lighting not matched yet.
  If object is wrong size:  adjust VLM output via prompt or --vlm_model 7B.

sk_{{mode}}_step{{i}}_{{name}}_inject.png
  After Kontext K/V injection. Seams should be smoothed; lighting adjusted.
  Hard seams still visible -> increase --feather_sigma (12-16 px).
  Object shape lost        -> raise --strength (0.4-0.5).

sk_{{mode}}_vlm_placements.json  [VLM mode only]
  Full VLM reasoning log: reasoning text + coordinates for each object.

KEY_lora_final.png
  Final scene after all three objects inserted.
  Does each object match the hand-drawn sketch shape/pose?
  Is the background pixel-identical to step0_base.png outside object regions?
""")


if __name__ == "__main__":
    main()
