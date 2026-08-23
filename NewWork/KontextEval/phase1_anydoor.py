# -*- coding: utf-8 -*-
"""
phase1_anydoor.py — AnyDoor-Based Object Insertion Pipeline

How this works
--------------
AnyDoor (ali-vilab/AnyDoor) is a ControlLDM model that takes:
  - ref_image  : RGB array of the source object (bicycle on white/grey bg)
  - ref_mask   : binary mask marking the object pixels in ref_image
  - tar_image  : RGB array of the target scene (living room)
  - tar_mask   : binary mask marking WHERE in the scene to place the object
And generates a new image with the object naturally inserted at that location.

This is COMPLETELY SEPARATE from FLUX/Kontext. AnyDoor uses:
  - Stable Diffusion v2.1 as the base diffusion model
  - ControlNet for spatial conditioning
  - DINOv2 ViT-g/14 for identity-preserving features
  - Sobel edge extraction for detail conditioning

Pipeline per object:
  Stage A  : Sketch → LoRA FLUX → obj_img
  Stage MASK: Threshold obj_img → ref_mask (non-white, non-grey pixels)
  Stage VLM : VLM(scene, obj_img) → placement description → placement bbox
  Stage D   : AnyDoor.inference(ref_image, ref_mask, scene, tar_mask) → next_scene
  Loop     : next_scene becomes the scene for the next object

Setup (ONE-TIME, before running this script)
--------------------------------------------
  # 1. Clone AnyDoor repo
  git clone https://github.com/ali-vilab/AnyDoor.git

  # 2. Install AnyDoor dependencies (separate from FLUX env, or add to existing)
  pip install pytorch_lightning==1.5.0 omegaconf einops open_clip_torch timm

  # 3. Download AnyDoor checkpoint (choose one):
  #    Option A — HuggingFace:
  #      huggingface-cli download xichenhku/AnyDoor epoch=1-step=8687.ckpt
  #    Option B — ModelScope:
  #      pip install modelscope
  #      python -c "from modelscope import snapshot_download; snapshot_download('damo/AnyDoor')"

  # 4. Download DINOv2 ViT-g/14 weights:
  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
  # Then update AnyDoor/configs/anydoor.yaml line 83 with the path to this file.

Usage
-----
  python NewWork/KontextEval/phase1_anydoor.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token $HF_TOKEN \\
      --anydoor_dir ./AnyDoor \\
      --anydoor_ckpt ./AnyDoor/epoch=1-step=8687.ckpt \\
      --dinov2_ckpt ./dinov2_vitg14_pretrain.pth \\
      --cache_dir ./models \\
      --out_dir results/phase1_anydoor \\
      --vlm_model Qwen/Qwen2-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


# ── AnyDoor helper functions (inlined from datasets/data_utils.py) ─────────


def _get_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = mask.shape[:2]
    if mask.sum() < 10:
        return 0, h, 0, w
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(y1), int(y2), int(x1), int(x2)


def _expand_image_mask(image: np.ndarray, mask: np.ndarray,
                        ratio: float = 1.4) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    H, W = int(h * ratio), int(w * ratio)
    h1, w1 = (H - h) // 2, (W - w) // 2
    h2, w2 = H - h - h1, W - w - w1
    image = np.pad(image, ((h1, h2), (w1, w2), (0, 0)),
                   constant_values=255)
    mask = np.pad(mask,  ((h1, h2), (w1, w2)),
                  constant_values=0)
    return image, mask


def _expand_bbox(img_or_mask: np.ndarray, yyxx: Tuple,
                  ratio: float = 1.2, min_crop: int = 0) -> Tuple:
    y1, y2, x1, x2 = yyxx
    H, W = img_or_mask.shape[:2]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = max(ratio * (y2 - y1 + 1), min_crop)
    w = max(ratio * (x2 - x1 + 1), min_crop)
    x1 = max(0, int(xc - w * 0.5))
    x2 = min(W, int(xc + w * 0.5))
    y1 = max(0, int(yc - h * 0.5))
    y2 = min(H, int(yc + h * 0.5))
    return (y1, y2, x1, x2)


def _box2square(image: np.ndarray, box: Tuple) -> Tuple:
    H, W = image.shape[:2]
    y1, y2, x1, x2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = y2 - y1, x2 - x1
    if h >= w:
        x1 = cx - h // 2;  x2 = cx + h // 2
    else:
        y1 = cy - w // 2;  y2 = cy + w // 2
    return (max(0, y1), min(H, y2), max(0, x1), min(W, x2))


def _box_in_box(small_box: Tuple, big_box: Tuple) -> Tuple:
    y1, y2, x1, x2 = small_box
    y1b, _, x1b, _ = big_box
    return (y1 - y1b, y2 - y1b, x1 - x1b, x2 - x1b)


def _pad_to_square(image: np.ndarray, pad_value: int = 255) -> np.ndarray:
    H, W = image.shape[:2]
    if H == W:
        return image
    padd = abs(H - W)
    p1, p2 = padd // 2, padd - padd // 2
    if H > W:
        pad_param = ((0, 0), (p1, p2), (0, 0)) if image.ndim == 3 else ((0, 0), (p1, p2))
    else:
        pad_param = ((p1, p2), (0, 0), (0, 0)) if image.ndim == 3 else ((p1, p2), (0, 0))
    return np.pad(image, pad_param, constant_values=pad_value)


def _sobel(img: np.ndarray, mask: np.ndarray, thresh: int = 50) -> np.ndarray:
    H, W = img.shape[:2]
    img_s  = cv2.resize(img,  (256, 256))
    mask_s = (cv2.resize(mask.astype(np.uint8), (256, 256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_s = cv2.erode(mask_s, kernel, iterations=2)
    sx = cv2.Sobel(img_s, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_s, cv2.CV_64F, 0, 1, ksize=3)
    scharr = cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5,
                              cv2.convertScaleAbs(sy), 0.5, 0)
    scharr = np.max(scharr, axis=-1) * mask_s
    scharr[scharr < thresh] = 0.0
    scharr = np.stack([scharr, scharr, scharr], axis=-1)
    scharr = (scharr.astype(np.float32) / 255 * img_s.astype(np.float32)).astype(np.uint8)
    return cv2.resize(scharr, (W, H))


def _crop_back(pred: np.ndarray, tar_image: np.ndarray,
               extra_sizes: Tuple, tar_box_yyxx_crop: Tuple) -> np.ndarray:
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred.astype(np.uint8), (W2, H2))
    m = 5
    gen = tar_image.copy()
    if W1 == H1:
        gen[y1 + m:y2 - m, x1 + m:x2 - m] = pred[m:-m, m:-m]
        return gen
    if W1 < W2:
        p1 = (W2 - W1) // 2;  p2 = W2 - W1 - p1
        pred = pred[:, p1:-p2]
    else:
        p1 = (H2 - H1) // 2;  p2 = H2 - H1 - p1
        pred = pred[p1:-p2]
    gen[y1 + m:y2 - m, x1 + m:x2 - m] = pred[m:-m, m:-m]
    return gen


def _process_pairs(ref_image: np.ndarray, ref_mask: np.ndarray,
                    tar_image: np.ndarray, tar_mask: np.ndarray) -> dict:
    """
    AnyDoor preprocessing: crop + Sobel ref, crop target region, build hint collage.
    Mirrors AnyDoor's process_pairs() from datasets/data_utils.py.
    """
    # ── Reference preprocessing ────────────────────────────────────────────────
    ref_box = _get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask] * 3, axis=-1)
    masked_ref = ref_image * ref_mask_3 + 255 * (1 - ref_mask_3)
    y1, y2, x1, x2 = ref_box
    masked_ref = masked_ref[y1:y2, x1:x2].astype(np.uint8)
    ref_mask_c = ref_mask[y1:y2, x1:x2]

    masked_ref, ref_mask_c = _expand_image_mask(masked_ref, ref_mask_c, ratio=1.2)
    ref_mask_3c = np.stack([ref_mask_c] * 3, axis=-1)
    masked_ref = _pad_to_square(masked_ref,  pad_value=255)
    ref_mask_3c = _pad_to_square(ref_mask_3c * 255, pad_value=0).astype(np.uint8)
    ref_mask_1c = ref_mask_3c[:, :, 0]

    masked_ref  = cv2.resize(masked_ref,   (224, 224)).astype(np.uint8)
    ref_mask_1c = cv2.resize(ref_mask_1c,  (224, 224)).astype(np.uint8)

    ref_sobel = _sobel(masked_ref, ref_mask_1c / 255.0)

    # ── Target preprocessing ───────────────────────────────────────────────────
    tar_box  = _get_bbox_from_mask(tar_mask)
    tar_box  = _expand_bbox(tar_mask, tar_box,  ratio=1.1)
    tar_crop = _expand_bbox(tar_image, tar_box, ratio=2.0)
    tar_crop = _box2square(tar_image, tar_crop)
    y1c, y2c, x1c, x2c = tar_crop

    crop_img  = tar_image[y1c:y2c, x1c:x2c].copy()
    tar_box_l = _box_in_box(tar_box, tar_crop)
    yl1, yl2, xl1, xl2 = tar_box_l

    ref_sobel_r = cv2.resize(ref_sobel, (xl2 - xl1, yl2 - yl1))
    ref_mask_r  = cv2.resize(ref_mask_1c, (xl2 - xl1, yl2 - yl1))
    ref_mask_r  = (ref_mask_r > 128).astype(np.uint8)

    collage      = crop_img.copy()
    collage[yl1:yl2, xl1:xl2] = ref_sobel_r
    collage_mask = np.zeros((*crop_img.shape[:2], 1), dtype=np.float32)
    collage_mask[yl1:yl2, xl1:xl2] = 1.0

    H1, W1 = crop_img.shape[:2]
    crop_img     = _pad_to_square(crop_img,     pad_value=0).astype(np.uint8)
    collage      = _pad_to_square(collage,      pad_value=0).astype(np.uint8)
    collage_mask_sq = _pad_to_square(
        (collage_mask * 255).astype(np.uint8), pad_value=0
    )
    H2, W2 = crop_img.shape[:2]

    crop_img     = cv2.resize(crop_img,     (512, 512)).astype(np.float32)
    collage      = cv2.resize(collage,      (512, 512)).astype(np.float32)
    collage_mask_r = (cv2.resize(collage_mask_sq, (512, 512)).astype(np.float32) / 255 > 0.5).astype(np.float32)

    # Normalize
    ref_norm      = masked_ref.astype(np.float32) / 255.0
    crop_norm     = crop_img  / 127.5 - 1.0
    collage_norm  = collage   / 127.5 - 1.0

    hint = np.concatenate([collage_norm, collage_mask_r[:, :, None]], axis=-1)  # (512,512,4)

    return dict(
        ref              = ref_norm,
        jpg              = crop_norm,
        hint             = hint,
        extra_sizes      = np.array([H1, W1, H2, W2]),
        tar_box_yyxx_crop= np.array(tar_crop),
    )


# ── Object mask extraction ────────────────────────────────────────────────────

def _compute_ref_mask(obj_img: Image.Image,
                       grey: tuple = (128, 128, 128),
                       tolerance: int = 20) -> np.ndarray:
    """
    Returns binary uint8 mask (H, W) where 1 = object, 0 = background.
    The LoRA generates objects on white or grey backgrounds; we detect both.
    """
    arr = np.array(obj_img.convert("RGB"), dtype=np.int32)
    is_grey  = (np.abs(arr[:, :, 0] - grey[0]) <= tolerance) & \
               (np.abs(arr[:, :, 1] - grey[1]) <= tolerance) & \
               (np.abs(arr[:, :, 2] - grey[2]) <= tolerance)
    is_white = (arr[:, :, 0] >= 230) & (arr[:, :, 1] >= 230) & (arr[:, :, 2] >= 230)
    return (~(is_grey | is_white)).astype(np.uint8)


# ── Placement: VLM description → target mask ─────────────────────────────────

def _placement_description_to_bbox(
    description: str,
    scene_w: int,
    scene_h: int,
) -> Tuple[int, int, int, int]:
    """
    Parse VLM placement text → approximate (y1, y2, x1, x2) pixel bbox.

    Covers common room placements:
      horizontal: left | right | center/middle
      vertical:   floor (bottom half) | table (mid) | shelf/wall (upper)
    """
    d = description.lower()

    # Horizontal thirds
    if any(w in d for w in ("left wall", "left side", "left corner",
                             "leftmost", "on the left", "against the left")):
        x1, x2 = 0, scene_w // 3
    elif any(w in d for w in ("right wall", "right side", "right corner",
                               "rightmost", "on the right", "against the right")):
        x1, x2 = 2 * scene_w // 3, scene_w
    else:
        x1, x2 = scene_w // 6, 5 * scene_w // 6   # wide center

    # Vertical placement
    if any(w in d for w in ("coffee table", "on the table", "on top of the table",
                             "table surface")):
        y1, y2 = scene_h // 3, 2 * scene_h // 3
    elif any(w in d for w in ("shelf", "on the shelf", "bookcase")):
        y1, y2 = scene_h // 6, scene_h // 2
    elif any(w in d for w in ("sofa", "on the sofa", "on the couch", "seat")):
        y1, y2 = scene_h // 3, 2 * scene_h // 3
    else:
        # Default: floor level (lower 40% of scene)
        y1, y2 = int(scene_h * 0.55), scene_h

    # Clamp
    y1 = max(0, y1);  y2 = min(scene_h, y2)
    x1 = max(0, x1);  x2 = min(scene_w,  x2)
    return y1, y2, x1, x2


def _make_tar_mask(scene_np: np.ndarray,
                   placement_description: str) -> np.ndarray:
    """Binary uint8 mask (H, W): 1 inside placement region, 0 outside."""
    H, W = scene_np.shape[:2]
    y1, y2, x1, x2 = _placement_description_to_bbox(placement_description, W, H)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    print(f"    [Placement] bbox: y=[{y1},{y2}] x=[{x1},{x2}]  "
          f"({(x2-x1)}×{(y2-y1)} px)")
    return mask


# ── AnyDoor model loading ─────────────────────────────────────────────────────

def load_anydoor_model(anydoor_dir: str, ckpt_path: str):
    """
    Load AnyDoor model + DDIMSampler.
    Requires AnyDoor repo cloned at anydoor_dir.
    """
    anydoor_dir = str(Path(anydoor_dir).resolve())
    if anydoor_dir not in sys.path:
        sys.path.insert(0, anydoor_dir)

    try:
        from cldm.model import create_model, load_state_dict
        from cldm.ddim_hacked import DDIMSampler
    except ImportError as e:
        raise ImportError(
            f"Cannot import AnyDoor modules from {anydoor_dir!r}.\n"
            f"Make sure AnyDoor is cloned there:\n"
            f"  git clone https://github.com/ali-vilab/AnyDoor.git {anydoor_dir}\n"
            f"Original error: {e}"
        )

    config_path = str(Path(anydoor_dir) / "configs" / "anydoor.yaml")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"AnyDoor config not found: {config_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"AnyDoor checkpoint not found: {ckpt_path}\n"
            "Download from:\n"
            "  huggingface-cli download xichenhku/AnyDoor 'epoch=1-step=8687.ckpt' "
            "--local-dir ./AnyDoor"
        )

    print(f"  Loading AnyDoor from {ckpt_path} ...")
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location="cuda"), strict=False)
    model = model.cuda().eval()
    sampler = DDIMSampler(model)
    print("  AnyDoor model loaded.")
    return model, sampler


# ── Core AnyDoor inference ────────────────────────────────────────────────────

@torch.no_grad()
def anydoor_insert(
    model,
    sampler,
    ref_image:    np.ndarray,   # uint8 RGB (H, W, 3)
    ref_mask:     np.ndarray,   # binary uint8 (H, W)
    tar_image:    np.ndarray,   # uint8 RGB (H, W, 3)
    tar_mask:     np.ndarray,   # binary uint8 (H, W)
    ddim_steps:   int   = 50,
    guidance:     float = 5.0,
    seed:         int   = 42,
    strength:     float = 1.0,
) -> np.ndarray:
    """
    Run AnyDoor insertion. Returns uint8 RGB (H, W, 3) numpy array.
    """
    import einops

    torch.manual_seed(seed)
    np.random.seed(seed)

    item = _process_pairs(ref_image, ref_mask, tar_image, tar_mask)

    ref_np   = item["ref"]
    hint_np  = item["hint"]
    num_s    = 1
    H, W     = 512, 512

    control = torch.from_numpy(hint_np.copy()).float().cuda()
    control = einops.rearrange(control.unsqueeze(0), "b h w c -> b c h w")

    clip_in = torch.from_numpy(ref_np.copy()).float().cuda()
    clip_in = einops.rearrange(clip_in.unsqueeze(0), "b h w c -> b c h w")

    cond = {
        "c_concat":   [control],
        "c_crossattn": [model.get_learned_conditioning(clip_in)],
    }
    null_clip = torch.zeros((1, 3, 224, 224), device="cuda")
    un_cond = {
        "c_concat":   [control],
        "c_crossattn": [model.get_learned_conditioning(null_clip)],
    }

    model.control_scales = [strength] * 13

    samples, _ = sampler.sample(
        S                              = ddim_steps,
        batch_size                     = num_s,
        shape                          = (4, H // 8, W // 8),
        conditioning                   = cond,
        verbose                        = False,
        eta                            = 0.0,
        unconditional_guidance_scale   = guidance,
        unconditional_conditioning     = un_cond,
    )

    decoded = model.decode_first_stage(samples)
    decoded = einops.rearrange(decoded, "b c h w -> b h w c")
    pred    = decoded[0].cpu().numpy()
    pred    = np.clip(pred * 127.5 + 127.5, 0, 255).astype(np.uint8)

    result = _crop_back(
        pred,
        tar_image,
        item["extra_sizes"],
        item["tar_box_yyxx_crop"],
    )
    return result.astype(np.uint8)


# ── Sibling helpers (Stage A LoRA generation + VLM) ──────────────────────────

def _load_siblings():
    base = Path(__file__).parent

    _comp = importlib.util.spec_from_file_location("phase1_composite",
                base / "phase1_composite.py")
    comp = importlib.util.module_from_spec(_comp); _comp.loader.exec_module(comp)

    _vlm = importlib.util.spec_from_file_location("phase1_sketch_vlm",
               base / "phase1_sketch_vlm.py")
    vlm = importlib.util.module_from_spec(_vlm); _vlm.loader.exec_module(vlm)

    _sk = importlib.util.spec_from_file_location("phase1_sketch",
              base / "phase1_sketch.py")
    sk = importlib.util.module_from_spec(_sk); _sk.loader.exec_module(sk)

    return comp, vlm, sk


# ── Main incremental pipeline ─────────────────────────────────────────────────

EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle"},
    {"name": "vase",    "description": "white ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball"},
]
BASE_PROMPT  = "A modern living room with a sofa and a wooden coffee table."
LORA_ID      = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
_SEP = "═" * 60


def run_anydoor_chain(
    flux_pipe,
    anydoor_model,
    anydoor_sampler,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    lora_id:        str,
    vlm_pair:       tuple,
    seed:           int,
    num_steps_lora: int,
    lora_guidance:  float,
    ddim_steps:     int,
    anydoor_guidance: float,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
) -> List[Image.Image]:
    comp, vlm_mod, _ = _load_siblings()
    vlm_model, vlm_proc = vlm_pair
    results = [base]
    scene   = base

    for i, edit in enumerate(edits):
        name = edit["name"]
        desc = edit["description"]

        sketch_path = os.path.join(sketch_dir, f"{name}.png")
        if not os.path.isfile(sketch_path):
            sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found: expected {name}.png or sketch_{name}.png "
                f"in {sketch_dir!r}"
            )

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}")
        print(f"{'─'*60}")

        # Stage A: sketch → LoRA FLUX → obj_img
        print(f"  [A] Generating '{desc}' from sketch ...")
        obj_img = vlm_mod.generate_from_sketch(
            pipe        = flux_pipe,
            sketch_path = sketch_path,
            description = desc,
            seed        = seed,
            num_steps   = num_steps_lora,
            guidance    = lora_guidance,
            height      = height,
            width       = width,
            lora_id     = lora_id,
            device      = device,
        )
        obj_path = os.path.join(out_dir, f"obj_gen_{name}.png")
        obj_img.save(obj_path)
        print(f"      Saved: {obj_path}")

        # Stage VLM: get placement description
        print(f"  [VLM] Getting placement description ...")
        placement_prompt = vlm_mod.vlm_generate_kontext_prompt(
            vlm_model    = vlm_model,
            vlm_processor= vlm_proc,
            scene_img    = scene,
            obj_img      = obj_img,
            description  = desc,
        )
        print(f"  [VLM] Placement: {placement_prompt[:120]}...")
        with open(os.path.join(out_dir, f"vlm_prompt_{name}.txt"), "w") as f:
            f.write(placement_prompt)

        # Stage MASK: extract ref_mask from obj_img
        obj_np      = np.array(obj_img.convert("RGB"))
        scene_np    = np.array(scene.convert("RGB"))
        ref_mask_np = _compute_ref_mask(obj_img)
        n_obj_px    = ref_mask_np.sum()
        print(f"  [MASK] Object pixels: {n_obj_px} / {obj_np.shape[0]*obj_np.shape[1]} "
              f"({100*n_obj_px/(obj_np.shape[0]*obj_np.shape[1]):.1f}%)")

        if n_obj_px < 100:
            # Fallback: use center 50% of image as object region
            print("  [MASK] Warning: very few object pixels detected, using center crop.")
            ref_mask_np = np.zeros(obj_np.shape[:2], dtype=np.uint8)
            h, w = obj_np.shape[:2]
            ref_mask_np[h//4:3*h//4, w//4:3*w//4] = 1

        # Stage TAR_MASK: placement description → target region in scene
        tar_mask_np = _make_tar_mask(scene_np, placement_prompt)

        # Visualize masks for debugging
        ref_mask_img = Image.fromarray(ref_mask_np * 255).convert("RGB")
        tar_mask_img = Image.fromarray(tar_mask_np * 255).convert("RGB")
        ref_mask_img.save(os.path.join(out_dir, f"ref_mask_{name}.png"))
        tar_mask_img.save(os.path.join(out_dir, f"tar_mask_{name}.png"))

        # Stage AnyDoor: insert object into scene
        print(f"  [AnyDoor] Inserting '{name}' into scene "
              f"({ddim_steps} DDIM steps, guidance={anydoor_guidance}) ...")
        result_np = anydoor_insert(
            model         = anydoor_model,
            sampler       = anydoor_sampler,
            ref_image     = obj_np,
            ref_mask      = ref_mask_np,
            tar_image     = scene_np,
            tar_mask      = tar_mask_np,
            ddim_steps    = ddim_steps,
            guidance      = anydoor_guidance,
            seed          = seed,
        )
        next_scene = Image.fromarray(result_np)
        result_path = os.path.join(out_dir, f"result_step{i+1}_{name}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        scene = next_scene
        results.append(scene)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AnyDoor-based identity-preserving object insertion pipeline."
    )
    p.add_argument("--sketch_dir",        required=True)
    p.add_argument("--hf_token",          required=True)
    p.add_argument("--anydoor_dir",       default="./AnyDoor",
                   help="Path to cloned AnyDoor repo (git clone https://github.com/ali-vilab/AnyDoor)")
    p.add_argument("--anydoor_ckpt",      required=True,
                   help="Path to AnyDoor checkpoint (epoch=1-step=8687.ckpt)")
    p.add_argument("--cache_dir",         default="./models")
    p.add_argument("--out_dir",           default="results/phase1_anydoor")
    p.add_argument("--config",            default=None,
                   help="JSON list of {name, description}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",           default=LORA_ID)
    p.add_argument("--lora_guidance",     type=float, default=4.0)
    p.add_argument("--ddim_steps",        type=int,   default=50,
                   help="AnyDoor DDIM sampling steps. Default 50.")
    p.add_argument("--anydoor_guidance",  type=float, default=5.0,
                   help="AnyDoor classifier-free guidance scale. Default 5.0.")
    p.add_argument("--num_steps",         type=int,   default=28,
                   help="FLUX LoRA generation steps (Stage A). Default 28.")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--height",            type=int,   default=1024)
    p.add_argument("--width",             type=int,   default=1024)
    p.add_argument("--device",            default="cuda")
    p.add_argument("--vlm_model",         default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--vlm_device",        default="cpu")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)

    print(f"\n{_SEP}")
    print(f"  phase1_anydoor  —  AnyDoor Object Insertion")
    print(f"{_SEP}")
    print(f"  Objects      : {[e['name'] for e in edits]}")
    print(f"  Sketch dir   : {args.sketch_dir}")
    print(f"  AnyDoor dir  : {args.anydoor_dir}")
    print(f"  AnyDoor ckpt : {args.anydoor_ckpt}")
    print(f"  DDIM steps   : {args.ddim_steps}  guidance={args.anydoor_guidance}")
    print(f"  Output       : {args.out_dir}")
    print(f"{_SEP}\n")

    # Load AnyDoor
    print("Loading AnyDoor ...")
    anydoor_model, anydoor_sampler = load_anydoor_model(
        anydoor_dir = args.anydoor_dir,
        ckpt_path   = args.anydoor_ckpt,
    )

    # Load VLM
    print("\nLoading VLM ...")
    comp, vlm_mod, sk_mod = _load_siblings()
    vlm_pair = sk_mod.load_vlm(args.vlm_model, args.cache_dir, args.vlm_device)

    # Load FLUX (for Stage A LoRA generation)
    print("\nLoading FLUX.1-Kontext-dev (Stage A only) ...")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline
    flux_pipe = load_kontext_pipeline(
        hf_token  = args.hf_token,
        device    = args.device,
        cache_dir = args.cache_dir,
    )

    # Generate base scene
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = comp.run_standard(
        pipe      = flux_pipe,
        canvas    = grey,
        prompt    = BASE_PROMPT,
        seed      = args.seed,
        num_steps = args.num_steps,
        guidance  = 2.5,
        height    = args.height,
        width     = args.width,
    )
    base_path = os.path.join(args.out_dir, "base_scene.png")
    base.save(base_path)
    print(f"  Saved: {base_path}")

    # Run chain
    results = run_anydoor_chain(
        flux_pipe         = flux_pipe,
        anydoor_model     = anydoor_model,
        anydoor_sampler   = anydoor_sampler,
        base              = base,
        edits             = edits,
        sketch_dir        = args.sketch_dir,
        lora_id           = args.lora_id,
        vlm_pair          = vlm_pair,
        seed              = args.seed,
        num_steps_lora    = args.num_steps,
        lora_guidance     = args.lora_guidance,
        ddim_steps        = args.ddim_steps,
        anydoor_guidance  = args.anydoor_guidance,
        height            = args.height,
        width             = args.width,
        out_dir           = args.out_dir,
        device            = args.device,
    )

    # Save grid
    all_imgs  = results
    all_lbls  = ["base"] + [e["name"] for e in edits]
    comp.save_grid(all_imgs, all_lbls,
                   os.path.join(args.out_dir, "chain_grid.png"),
                   ncols=len(all_imgs))
    print(f"\n{_SEP}")
    print(f"  Chain complete.  Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
