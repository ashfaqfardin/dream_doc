# -*- coding: utf-8 -*-
"""
Phase 1 -- Clean-Base Compositing: Object Separation for Background Preservation

Core idea
---------
Background quality degrades across edit steps because each Kontext pass
regenerates the full scene, introducing small errors that compound.
The fix: never regenerate the background at all.

For each new object:
  1. PROBE  — run Kontext from img_prev (accumulated scene) to find WHERE
              the object naturally lands.  Uses the full scene context.
  2. GENERATE — run Kontext from BASE (original clean image) to generate
                the object at high quality.  Background in this output is
                pristine because it is derived from the clean base.
  3. COMPOSITE — pixel-paste ONLY the object region from the base-generation
                 onto img_prev.  Background pixels are NEVER touched; they are
                 an exact copy from the previous step.

  result[i][bg_pixels]  = img_prev[i-1][bg_pixels]   # exact pixel copy
  result[i][obj_pixels] = from_base_result[obj_pixels] # clean generation

Boundary quality
----------------
A hard pixel boundary between the pasted object and the preserved background
can look artificial.  Gaussian feathering (--feather_sigma, default 12) creates
a smooth alpha blend over the mask boundary, combining a few pixels from each
side so lighting and colour transition naturally.

Comparison
----------
Three modes run back-to-back:

  baseline   — plain Kontext chain, no injection, no compositing
  kv         — K/V injection chain (phase1_kv_chain_multi approach)
  composite  — this file's new approach

KEY_RESULT_comparison.png, KEY_RESULT_stability.png, and stability.txt
show all three side-by-side.

Usage
-----
python NewWork/KontextEval/phase1_composite.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_composite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline, generate

_ie_path = str(Path(__file__).parent.parent / "IncrementalEdit")
sys.path.insert(0, _ie_path)
from kontext_injection import (
    TIER_A, TIER_ALL,
    ZoneMasks, InjectionState,
    install_processor, set_determinism,
)

# ============================================================
# Edit list and constants
# ============================================================

EDITS: List[dict] = [
    {
        "name": "bicycle",
        "prompt": (
            "Add a yellow bicycle leaning against the wall on the left side. "
            "Keep the rest of the room exactly the same."
        ),
    },
    {
        "name": "vase",
        "prompt": (
            "Add a white ceramic vase with flowers on the coffee table. "
            "Keep the rest of the room exactly the same."
        ),
    },
    {
        "name": "ball",
        "prompt": (
            "Add a yellow round ball in the right corner of the room next to the sofa. "
            "Keep the rest of the room exactly the same."
        ),
    },
]

BASE_PROMPT   = "A modern living room with a sofa and a wooden coffee table."
TIER_A_LAYERS = list(TIER_A)
ALL_LAYERS    = list(TIER_ALL)


# ============================================================
# Image / mask utilities
# ============================================================

def pixel_to_token_mask(img_a: Image.Image, img_b: Image.Image,
                        h_lat: int, w_lat: int,
                        threshold: float = 40.0) -> np.ndarray:
    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)
    diff = np.abs(b - a).mean(axis=2)
    diff_img = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
    diff_down = diff_img.resize((w_lat, h_lat), Image.BILINEAR)
    return (np.array(diff_down).astype(np.float32) >= threshold).reshape(-1)


def _token_to_pixel(flat_mask: np.ndarray, h_lat: int, w_lat: int,
                    H: int, W: int) -> np.ndarray:
    token_2d = flat_mask.reshape(h_lat, w_lat).astype(np.uint8) * 255
    return np.array(
        Image.fromarray(token_2d, "L").resize((W, H), Image.NEAREST)
    ) > 127


def color_overlay(img: Image.Image, flat_mask: np.ndarray,
                  h_lat: int, w_lat: int,
                  color=(255, 120, 0), alpha=0.4) -> Image.Image:
    H, W = img.size[1], img.size[0]
    px = _token_to_pixel(flat_mask, h_lat, w_lat, H, W)
    arr = np.array(img).astype(float)
    out = arr.copy()
    out[px] = arr[px] * (1 - alpha) + np.array(color) * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def region_diff(img_a: Image.Image, img_b: Image.Image,
                flat_mask: np.ndarray, h_lat: int, w_lat: int) -> float:
    H, W = img_a.size[1], img_a.size[0]
    px_mask = _token_to_pixel(flat_mask, h_lat, w_lat, H, W)
    diff = np.abs(
        np.array(img_a).astype(float) - np.array(img_b).astype(float)
    ).mean(axis=2)
    return float(diff[px_mask].mean()) if px_mask.any() else 0.0


def run_standard(pipe, canvas: Image.Image, prompt: str,
                 seed: int, num_steps: int, guidance: float,
                 height: int, width: int) -> Image.Image:
    return generate(
        pipe, prompt, canvas,
        seed=seed, num_steps=num_steps,
        guidance_scale=guidance,
        height=height, width=width,
    )


def save_grid(images, titles, path, ncols=None, figsize_per_cell=(4, 4)):
    n = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
    )
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(t, fontsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Compositing
# ============================================================

def feather_mask(px_mask: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian-feathered alpha from a binary pixel mask.
    Interior (px_mask=True) is clamped to 1.0 — only the exterior ring falls off.
    sigma=0 -> hard binary (no feathering).
    """
    if sigma <= 0:
        return px_mask.astype(float)
    mask_img = Image.fromarray((px_mask * 255).astype(np.uint8), "L")
    blurred  = mask_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    alpha    = np.array(blurred).astype(float) / 255.0
    alpha[px_mask] = 1.0   # interior stays fully opaque; falloff only on exterior ring
    return alpha


def composite_onto(scene: Image.Image, inject: Image.Image,
                   px_target: np.ndarray, sigma: float = 12.0) -> Image.Image:
    """
    Paste inject onto scene using a feathered version of px_target as alpha.
      alpha = 1.0 (inside object)  -- use inject
      alpha = 0.0 (outside object) -- use scene
      alpha = 0..1 (boundary)      -- smooth blend

    sigma: Gaussian blur radius in pixels for boundary feathering.
           Typical: 8–16 px (half a token width = 8 px at 1024×1024 / 64 grid).
           Set to 0 for hard paste (useful for debugging).
    """
    alpha  = feather_mask(px_target, sigma)                  # (H, W) float [0,1]
    alpha3 = alpha[:, :, np.newaxis]                          # (H, W, 1)
    scene_f  = np.array(scene).astype(float)
    inject_f = np.array(inject).astype(float)
    result   = alpha3 * inject_f + (1.0 - alpha3) * scene_f
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ============================================================
# K/V injection helper (for the comparison baseline)
# ============================================================

@torch.no_grad()
def run_kv_injected(
    pipe,
    canvas: Image.Image,
    prompt: str,
    background_mask: np.ndarray,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    strength: float, cutoff: float,
    vital_layers: List[int],
    max_seq_len: int = 512,
    device: str = "cuda",
) -> Image.Image:
    h_lat = height // 16
    w_lat = width  // 16
    n_gen = h_lat * w_lat

    state = InjectionState(
        mode="edit",
        vital_layers=set(vital_layers),
        n_gen=n_gen, n_ref=n_gen,
        cutoff_frac=(0.0, cutoff),
        strength=strength,
        n_steps=num_steps,
    )
    state.zones = ZoneMasks(
        background=background_mask.astype(bool),
        shell=np.zeros(n_gen, dtype=bool),
        target=np.logical_not(background_mask).astype(bool),
    ).to_device(device)

    install_processor(pipe, state, max_sequence_length=max_seq_len)
    generator = set_determinism(seed)
    result = pipe(
        image=canvas, prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        height=height, width=width,
        max_sequence_length=max_seq_len,
        generator=generator,
        output_type="pil",
    )
    return result.images[0]


# ============================================================
# Chain runners
# ============================================================

def run_composite_chain(
    pipe,
    base: Image.Image,
    edits: List[dict],
    h_lat: int, w_lat: int,
    seed: int, num_steps: int, probe_steps: int,
    guidance: float, height: int, width: int,
    threshold: float, feather_sigma: float,
    out_dir: str,
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    Clean-base compositing chain.

    At every step:
      probe  → find object location using accumulated scene (natural context)
      generate → Kontext from original BASE  (pristine reference, no accumulated noise)
      composite → paste ONLY the object region back onto the accumulated scene

    Background pixels are never regenerated — they are pixel-copied exactly
    from the previous step. No drift, no compounding noise.
    """
    imgs, obj_masks = [base], []

    for i, edit in enumerate(edits):
        name    = edit["name"]
        prompt  = edit["prompt"]
        img_prev = imgs[-1]
        H, W    = img_prev.size[1], img_prev.size[0]

        print(f"\n  [composite] step {i+1}/{len(edits)}  {name}")

        # ── 1. Probe from accumulated scene (detect natural placement) ──────
        print(f"    probe  ({probe_steps} steps, canvas = accumulated scene) …")
        img_probe = run_standard(pipe, img_prev, prompt,
                                 seed, probe_steps, guidance, height, width)
        img_probe.save(os.path.join(out_dir, f"comp_step{i+1}_{name}_probe.png"))

        target_raw   = pixel_to_token_mask(img_prev, img_probe, h_lat, w_lat, threshold)
        base_stable  = ~pixel_to_token_mask(base, img_probe, h_lat, w_lat, threshold)
        target_clean = target_raw & ~base_stable
        obj_masks.append(target_clean)

        pct_tgt = target_clean.mean() * 100
        pct_bg  = (~target_clean).mean() * 100
        print(f"    target={pct_tgt:.1f}%  background={pct_bg:.1f}%")

        color_overlay(img_prev, target_clean, h_lat, w_lat,
                      color=(0, 160, 255), alpha=0.45).save(
            os.path.join(out_dir, f"comp_step{i+1}_{name}_target.png"))

        # ── 2. Generate from clean base (no accumulated noise in reference) ─
        print(f"    generate ({num_steps} steps, canvas = clean base) …")
        img_from_base = run_standard(pipe, base, prompt,
                                     seed, num_steps, guidance, height, width)
        img_from_base.save(os.path.join(out_dir, f"comp_step{i+1}_{name}_from_base.png"))

        # ── 3. Composite: paste object region, preserve background exactly ──
        px_target = _token_to_pixel(target_clean, h_lat, w_lat, H, W)
        img_curr  = composite_onto(img_prev, img_from_base, px_target, feather_sigma)
        img_curr.save(os.path.join(out_dir, f"comp_step{i+1}_{name}_result.png"))

        imgs.append(img_curr)

    return imgs, obj_masks


def run_kv_chain(
    pipe,
    base: Image.Image,
    edits: List[dict],
    h_lat: int, w_lat: int,
    seed: int, num_steps: int, probe_steps: int,
    guidance: float, height: int, width: int,
    strength: float, cutoff: float,
    vital_layers: List[int],
    threshold: float,
    out_dir: str,
    device: str = "cuda",
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """K/V injection chain (phase1_kv_chain_multi approach, for comparison)."""
    imgs, obj_masks = [base], []

    for i, edit in enumerate(edits):
        name    = edit["name"]
        prompt  = edit["prompt"]
        img_prev = imgs[-1]

        print(f"\n  [kv] step {i+1}/{len(edits)}  {name}")

        print(f"    probe ({probe_steps} steps) …")
        img_probe = run_standard(pipe, img_prev, prompt,
                                 seed, probe_steps, guidance, height, width)
        img_probe.save(os.path.join(out_dir, f"kv_step{i+1}_{name}_probe.png"))

        target_raw   = pixel_to_token_mask(img_prev, img_probe, h_lat, w_lat, threshold)
        base_stable  = ~pixel_to_token_mask(base, img_probe, h_lat, w_lat, threshold)
        target_clean = target_raw & ~base_stable
        obj_masks.append(target_clean)

        print(f"    inject ({num_steps} steps, s={strength}, cutoff={cutoff}) …")
        background_mask = ~target_clean
        img_curr = run_kv_injected(
            pipe, img_prev, prompt, background_mask,
            seed=seed, num_steps=num_steps,
            guidance=guidance, height=height, width=width,
            strength=strength, cutoff=cutoff,
            vital_layers=vital_layers, device=device,
        )
        img_curr.save(os.path.join(out_dir, f"kv_step{i+1}_{name}_result.png"))

        imgs.append(img_curr)

    return imgs, obj_masks


def run_bg_replace_chain(
    pipe,
    base: Image.Image,
    edits: List[dict],
    h_lat: int, w_lat: int,
    seed: int, num_steps: int, probe_steps: int,
    guidance: float, height: int, width: int,
    strength: float, cutoff: float,
    vital_layers: List[int],
    threshold: float,
    feather_sigma: float,
    out_dir: str,
    device: str = "cuda",
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    K/V injection + background pixel-replace from the original base.

    At every step:
      1. Probe  from img_prev  (natural placement, full scene context)
      2. Inject from img_prev  (K/V injection — objects are coherent with the scene)
      3. Accumulate all_obj_mask = union of all target_clean masks so far
      4. pure_bg = ~all_obj_mask
      5. inject_result[pure_bg] = base[pure_bg]   (hard pixel copy from original)

    Objects come from K/V injection so they see each other and generate coherently.
    Background is an exact copy of the original base at every step — zero drift.
    The feather_sigma controls blending at the object/background boundary only;
    the interior of every object zone stays 100% from the injection result.
    """
    imgs, obj_masks = [base], []
    all_obj_mask = np.zeros(h_lat * w_lat, dtype=bool)

    for i, edit in enumerate(edits):
        name     = edit["name"]
        prompt   = edit["prompt"]
        img_prev = imgs[-1]
        H, W     = img_prev.size[1], img_prev.size[0]

        print(f"\n  [bg_replace] step {i+1}/{len(edits)}  {name}")

        # 1. Probe from accumulated scene
        print(f"    probe  ({probe_steps} steps) ...")
        img_probe = run_standard(pipe, img_prev, prompt,
                                 seed, probe_steps, guidance, height, width)
        img_probe.save(os.path.join(out_dir, f"bgr_step{i+1}_{name}_probe.png"))

        target_raw   = pixel_to_token_mask(img_prev, img_probe, h_lat, w_lat, threshold)
        base_stable  = ~pixel_to_token_mask(base, img_probe, h_lat, w_lat, threshold)
        target_clean = target_raw & ~base_stable
        obj_masks.append(target_clean)

        print(f"    target={target_clean.mean()*100:.1f}%")
        color_overlay(img_prev, target_clean, h_lat, w_lat,
                      color=(0, 200, 100), alpha=0.45).save(
            os.path.join(out_dir, f"bgr_step{i+1}_{name}_target.png"))

        # 2. K/V inject from accumulated scene (objects are context-aware)
        print(f"    inject ({num_steps} steps, s={strength}, cutoff={cutoff}) ...")
        img_inject = run_kv_injected(
            pipe, img_prev, prompt, ~target_clean,
            seed=seed, num_steps=num_steps,
            guidance=guidance, height=height, width=width,
            strength=strength, cutoff=cutoff,
            vital_layers=vital_layers, device=device,
        )
        img_inject.save(os.path.join(out_dir, f"bgr_step{i+1}_{name}_inject.png"))

        # 3. Accumulate every object token seen so far
        all_obj_mask = all_obj_mask | target_clean

        # 4. Replace pure background (no object ever placed there) with original base
        px_all_obj = _token_to_pixel(all_obj_mask, h_lat, w_lat, H, W)
        inject_arr = np.array(img_inject).astype(float)
        base_arr   = np.array(base).astype(float)

        if feather_sigma > 0:
            # Soft boundary: blend at the object edge; interior stays inject, exterior stays base
            alpha = feather_mask(px_all_obj, feather_sigma)   # interior=1.0, exterior falls to 0
            alpha3 = alpha[:, :, np.newaxis]
            result_arr = alpha3 * inject_arr + (1.0 - alpha3) * base_arr
        else:
            result_arr = inject_arr.copy()
            px_pure_bg = ~px_all_obj
            result_arr[px_pure_bg] = base_arr[px_pure_bg]

        img_curr = Image.fromarray(result_arr.clip(0, 255).astype(np.uint8))
        img_curr.save(os.path.join(out_dir, f"bgr_step{i+1}_{name}_result.png"))
        imgs.append(img_curr)

    return imgs, obj_masks


def run_baseline_chain(
    pipe,
    base: Image.Image,
    edits: List[dict],
    seed: int, num_steps: int,
    guidance: float, height: int, width: int,
    out_dir: str,
) -> List[Image.Image]:
    """Plain Kontext chain — no injection, no compositing."""
    imgs = [base]
    for i, edit in enumerate(edits):
        print(f"\n  [baseline] step {i+1}/{len(edits)}  {edit['name']}")
        img = run_standard(pipe, imgs[-1], edit["prompt"],
                           seed, num_steps, guidance, height, width)
        img.save(os.path.join(out_dir, f"bl_step{i+1}_{edit['name']}_result.png"))
        imgs.append(img)
    return imgs


# ============================================================
# Visualisation
# ============================================================

def build_comparison_grid(
    base: Image.Image,
    imgs_by_mode: Dict[str, List[Image.Image]],
    edits: List[dict],
    out_dir: str,
    strength: float,
    feather_sigma: float,
):
    modes = list(imgs_by_mode.keys())
    n_steps = len(edits)
    ncols = n_steps + 1
    labels = {
        "baseline":   "Baseline\n(no inject)",
        "kv":         f"K/V inject\n(s={strength})",
        "composite":  f"Composite\n(σ={feather_sigma}px)",
        "bg_replace": f"BG-Replace\n(s={strength},σ={feather_sigma})",
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


def build_stability_grid(
    base: Image.Image,
    imgs_by_mode: Dict[str, List[Image.Image]],
    edits: List[dict],
    obj_masks_by_mode: Dict[str, List[np.ndarray]],
    h_lat: int, w_lat: int,
    out_dir: str,
    strength: float,
    feather_sigma: float,
) -> Dict[str, Dict[str, float]]:
    """
    Per-region WITHIN-CHAIN stability: for each mode, diff from
    'when the region was first established in that mode's chain' to
    that mode's final result.

    Region boundaries are defined by the kv chain's masks so they are
    comparable across modes.  Each mode uses its OWN intermediate images
    as the 'when_added' reference — this gives true within-chain drift.
    """
    modes = list(imgs_by_mode.keys())
    ref_key   = "kv" if "kv" in obj_masks_by_mode else list(obj_masks_by_mode.keys())[0]
    ref_masks = obj_masks_by_mode[ref_key]

    all_obj = np.zeros(ref_masks[0].shape, dtype=bool)
    for m in ref_masks:
        all_obj |= m
    bg_mask = ~all_obj

    # (region_name, token_mask, step_index_when_added)
    # step=0 means base — compare base to mode's final for background
    region_specs = [("background", bg_mask, 0)]
    for i, edit in enumerate(edits):
        region_specs.append((edit["name"], ref_masks[i], i + 1))

    stability: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in region_specs}

    for mode in modes:
        mode_imgs = imgs_by_mode[mode]
        final     = mode_imgs[-1]
        for name, mask, step_added in region_specs:
            # Use THIS mode's image at step_added (not the kv chain's)
            when_added = mode_imgs[min(step_added, len(mode_imgs) - 1)]
            stability[name][mode] = region_diff(when_added, final, mask, h_lat, w_lat)

    labels = {
        "baseline":   "Baseline",
        "kv":         f"K/V (s={strength})",
        "composite":  f"Composite (σ={feather_sigma})",
        "bg_replace": f"BG-Replace",
    }

    # Visual grid: for each region, show [base/when_added | mode_final ...] per mode
    ref_imgs = imgs_by_mode[ref_key]
    ncols    = len(modes) + 1
    images, titles = [], []

    for name, mask, step_added in region_specs:
        images.append(ref_imgs[step_added])
        titles.append(f"[{name}]\n(kv) when added")
        for mode in modes:
            final_img = imgs_by_mode[mode][-1]
            dval      = stability[name][mode]
            images.append(final_img)
            titles.append(f"[{name}] {labels.get(mode, mode)}\nΔ={dval:.1f}")

    save_grid(
        images, titles,
        os.path.join(out_dir, "KEY_RESULT_stability.png"),
        ncols=ncols,
    )
    return stability


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Clean-base compositing vs K/V injection for multi-step editing."
    )
    p.add_argument("--hf_token",       required=True)
    p.add_argument("--cache_dir",      default="./models")
    p.add_argument("--out_dir",        default="results/phase1_composite")
    p.add_argument("--config",         default=None,
                   help="JSON list of {name, prompt}. Overrides built-in EDITS.")
    p.add_argument("--mode",           default="all",
                   choices=["all", "bg_replace", "composite", "kv", "baseline"],
                   help="'all' runs baseline + kv + bg_replace. "
                        "'composite' is the older approach (ghosting issues).")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--num_steps",      type=int,   default=28)
    p.add_argument("--probe_steps",    type=int,   default=18)
    p.add_argument("--guidance",       type=float, default=2.5)
    p.add_argument("--strength",       type=float, default=0.3,
                   help="K/V injection strength (kv mode only).")
    p.add_argument("--cutoff",         type=float, default=0.4,
                   help="Inject first CUTOFF fraction of steps (kv mode only).")
    p.add_argument("--threshold",      type=float, default=40.0,
                   help="Pixel diff threshold for target mask from probe.")
    p.add_argument("--feather_sigma",  type=float, default=12.0,
                   help="Gaussian blur radius (px) for mask boundary feathering. "
                        "0 = hard paste (no feathering). Recommended: 8–16.")
    p.add_argument("--all_layers",     action="store_true",
                   help="Use all 57 layers for kv mode (control; TIER_A is default).")
    p.add_argument("--height",         type=int,   default=1024)
    p.add_argument("--width",          type=int,   default=1024)
    p.add_argument("--device",         default="cuda")
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
    modes = (["baseline", "kv", "bg_replace"] if args.mode == "all" else [args.mode])

    print(f"Edit sequence  : {[e['name'] for e in edits]}")
    print(f"Modes          : {modes}")
    print(f"feather_sigma  : {args.feather_sigma} px")
    print(f"kv strength    : {args.strength}  cutoff={args.cutoff}  "
          f"layers={'ALL_57' if args.all_layers else 'TIER_A'}")

    # ── Load model ────────────────────────────────────────────
    print("\nLoading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    # ── Base scene ────────────────────────────────────────────
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(pipe, grey, BASE_PROMPT,
                        args.seed, args.num_steps, args.guidance,
                        args.height, args.width)
    base.save(os.path.join(args.out_dir, "step0_base.png"))

    # ── Run chains ────────────────────────────────────────────
    imgs_by_mode:     Dict[str, List[Image.Image]] = {}
    obj_masks_by_mode: Dict[str, List[np.ndarray]] = {}

    for mode in modes:
        print(f"\n{'='*64}")
        print(f"=== {mode.upper()} chain ===")
        print(f"{'='*64}")

        if mode == "baseline":
            imgs = run_baseline_chain(
                pipe, base, edits,
                seed=args.seed, num_steps=args.num_steps,
                guidance=args.guidance,
                height=args.height, width=args.width,
                out_dir=args.out_dir,
            )
            # Baseline has no probe-derived masks; use a dummy (full background)
            n_gen = h_lat * w_lat
            obj_masks: List[np.ndarray] = [np.zeros(n_gen, dtype=bool)] * len(edits)

        elif mode == "kv":
            imgs, obj_masks = run_kv_chain(
                pipe, base, edits,
                h_lat=h_lat, w_lat=w_lat,
                seed=args.seed, num_steps=args.num_steps,
                probe_steps=args.probe_steps,
                guidance=args.guidance,
                height=args.height, width=args.width,
                strength=args.strength, cutoff=args.cutoff,
                vital_layers=vital_layers,
                threshold=args.threshold,
                out_dir=args.out_dir,
                device=args.device,
            )

        elif mode == "composite":
            imgs, obj_masks = run_composite_chain(
                pipe, base, edits,
                h_lat=h_lat, w_lat=w_lat,
                seed=args.seed, num_steps=args.num_steps,
                probe_steps=args.probe_steps,
                guidance=args.guidance,
                height=args.height, width=args.width,
                threshold=args.threshold,
                feather_sigma=args.feather_sigma,
                out_dir=args.out_dir,
            )

        elif mode == "bg_replace":
            imgs, obj_masks = run_bg_replace_chain(
                pipe, base, edits,
                h_lat=h_lat, w_lat=w_lat,
                seed=args.seed, num_steps=args.num_steps,
                probe_steps=args.probe_steps,
                guidance=args.guidance,
                height=args.height, width=args.width,
                strength=args.strength, cutoff=args.cutoff,
                vital_layers=vital_layers,
                threshold=args.threshold,
                feather_sigma=args.feather_sigma,
                out_dir=args.out_dir,
                device=args.device,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        imgs_by_mode[mode]      = imgs
        obj_masks_by_mode[mode] = obj_masks
        imgs[-1].save(os.path.join(args.out_dir, f"KEY_{mode}_final.png"))

    # ── Visuals ───────────────────────────────────────────────
    print("\n=== Building comparison grid ===")
    build_comparison_grid(
        base, imgs_by_mode, edits, args.out_dir,
        strength=args.strength, feather_sigma=args.feather_sigma,
    )
    print("  Saved: KEY_RESULT_comparison.png")

    # Only build stability grid if we have mask-based modes
    mask_modes = {m: v for m, v in obj_masks_by_mode.items() if m != "baseline"}
    if mask_modes:
        print("=== Building stability grid ===")
        mask_imgs = {m: imgs_by_mode[m] for m in mask_modes}
        stability = build_stability_grid(
            base, imgs_by_mode, edits, mask_modes,
            h_lat, w_lat, args.out_dir,
            strength=args.strength, feather_sigma=args.feather_sigma,
        )
        print("  Saved: KEY_RESULT_stability.png")

        # ── Numeric table ─────────────────────────────────────
        region_names = list(stability.keys())
        col_modes    = list(mask_modes.keys())
        if "baseline" in imgs_by_mode:
            # Add baseline as a reference column by computing its diffs manually
            bl_final = imgs_by_mode["baseline"][-1]
            ref_key  = "kv" if "kv" in mask_modes else list(mask_modes.keys())[0]
            ref_masks_list = obj_masks_by_mode[ref_key]
            all_obj_ = np.zeros(ref_masks_list[0].shape, dtype=bool)
            for m in ref_masks_list:
                all_obj_ |= m
            bg_m = ~all_obj_
            region_specs_bl = [("background", bg_m, 0)]
            for idx, edit in enumerate(edits):
                region_specs_bl.append((edit["name"], ref_masks_list[idx], idx + 1))
            ref_imgs_bl = imgs_by_mode[ref_key]
            for rname, rmask, rstep in region_specs_bl:
                when = ref_imgs_bl[rstep]
                stability[rname]["baseline"] = region_diff(when, bl_final, rmask, h_lat, w_lat)
            col_modes = ["baseline"] + col_modes

        col_w = 13
        print(f"\n{'='*72}")
        print("STABILITY TABLE  (mean abs pixel diff — lower = better preserved)")
        print(f"{'='*72}")
        hdr = f"  {'region':<14}" + "".join(f"  {m:>{col_w}}" for m in col_modes)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        file_lines = [
            "Stability = mean abs pixel diff (when-added → final). Lower = better.\n\n",
            f"{'region':<14}" + "".join(f"  {m:>{col_w}}" for m in col_modes) + "\n",
            "-" * 72 + "\n",
        ]

        for name in region_names:
            row = f"  {name:<14}"
            frow = f"{name:<14}"
            for m in col_modes:
                val = stability[name].get(m, float("nan"))
                row  += f"  {val:>{col_w}.2f}"
                frow += f"  {val:>{col_w}.2f}"
            print(row)
            file_lines.append(frow + "\n")

        if "baseline" in col_modes and len(col_modes) > 1:
            print()
            impr = f"  {'improve %':<14}"
            for m in col_modes:
                if m == "baseline":
                    impr += f"  {'(ref)':>{col_w}}"
                else:
                    pcts = [
                        (stability[n].get("baseline", 0) - stability[n].get(m, 0))
                        / max(stability[n].get("baseline", 1e-6), 1e-6) * 100
                        for n in region_names
                    ]
                    impr += f"  {np.mean(pcts):>+{col_w-1}.1f}%"
            print(impr)
            file_lines.append("\n" + impr.strip() + "\n")

        with open(os.path.join(args.out_dir, "stability.txt"), "w") as f:
            f.writelines(file_lines)
        print("\n  Saved: stability.txt")

    # ── What to check ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print("WHAT TO CHECK")
    print(f"{'='*72}")
    print(f"""
KEY_RESULT_comparison.png
  Rows = modes, columns = base → step1 → step2 → step3.
  Watch the background (walls, floor, sofa) across the composite row:
  it should look identical to step0_base.png at every column.

KEY_RESULT_stability.png  +  stability.txt
  Background Δ for composite should be MUCH lower than kv and baseline.
  Object Δ (bicycle, vase, ball) should be similar or better.

comp_step{{i}}_{{name}}_from_base.png
  The raw Kontext output generated from the clean base at each step.
  The background here should be clean; we only use the object region.

comp_step{{i}}_{{name}}_result.png
  After compositing. Check for seams at the object boundary.
  If seams are visible, increase --feather_sigma (try 16 or 24).
  If the feathering blurs the object edge too much, decrease it (try 6).

KEY_composite_final.png vs KEY_kv_final.png
  Side-by-side final images.  The composite final should have the
  crispest background; kv may have better object-to-scene blending.
""")


if __name__ == "__main__":
    main()
