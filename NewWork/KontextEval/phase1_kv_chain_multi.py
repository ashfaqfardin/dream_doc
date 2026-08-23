"""
Phase 1 — K/V Chain Multi: Generalized N-Step Multi-Object Preservation

The problem with the standard Kontext chain in multi-step editing
-----------------------------------------------------------------
At step K, Kontext freely regenerates the entire scene from img_{K-1}.
Objects added in steps 1..K-1 exist only as pixels in that reference image.
Kontext MAY shift, recolor, or erase prior objects when the "add object K"
prompt draws its attention elsewhere.

The compound invariant that fixes it
-------------------------------------
At step K, the reference IS img_{K-1}, which already contains every object
added so far. Injecting ref K/V into every token EXCEPT where object K needs
to appear anchors all prior content to img_{K-1}.

  background_mask = ~target_mask_K

  reference = img_{K-1}  →  carries full edit history
  injection at background_mask  →  protects everything that existed before K

  step 2 reference = img_1  → protects object 1
  step 3 reference = img_2  → protects objects 1 AND 2
  step K reference = img_{K-1}  → protects objects 1..K-1

No per-object mask management needed. background=~target is sufficient
at every step from step 2 onward.

Two-pass design at every step
------------------------------
  PROBE PASS (no injection):
    Run standard Kontext to find the natural placement of the new object.
    Only needs to show WHERE the object lands, not high quality.
    probe_steps (default 18) used for all steps.
    than a vase or ball to form a usable shape.
    probe_steps (default 15) used for steps 2+.

  INJECT PASS (num_steps, with injection):
    target_mask = pixel_to_token_mask(img_{K-1}, img_probe, threshold)
    background_mask = ~target_mask
    Re-run with K/V injection at background_mask.

Why probe at 16 steps for step 1:
  At 16/28 steps the bicycle's structure (frame, wheels, position) is
  committed in the latent. Pixel diff vs base cleanly isolates the bicycle.
  At 10 steps the shape is too noisy — target_mask is unreliable and
  injection blocks the bicycle rather than protecting around it.
  Injecting at step 1 also benefits the base scene: sofa/walls/floor
  are anchored to the base reference while only the bicycle area is free.

Validated defaults (from phase1_kv_chain.py sweep)
----------------------------------------------------
  baseline bike_diff:              48.31  (no injection)
  TIER_A s=0.3 cut=0.4:            10.56  ← best (78% improvement)
  TIER_A s=0.7 cut=0.6:            10.77  (barely worse)
  ALL_57  s=0.7 cut=0.6:           58.80  ← WORSE than baseline

  strength is not sensitive in [0.3, 0.7] → use s=0.3 (less interference)
  earlier cutoff is slightly better        → use cutoff=0.4
  TIER_A (13 content layers) essential    → ALL_57 destroys composition

Edit list (default: bicycle → vase → ball)
------------------------------------------
Configurable via EDITS list below or --config JSON.

Metrics
-------
  stability[K]: mean abs pixel diff in object K's region between
    - step K output  (when object K was first added)
    - FINAL output   (img_N, after all subsequent edits)
  Lower = object K better preserved.

  background_stability: same diff in the base-scene region
    (complement of all object masks) between base and FINAL.
  Lower = original room better preserved.

  improvement%: (baseline_diff - kv_diff) / baseline_diff * 100
  Reported for background + every object.

Outputs
-------
  step0_base.png                     base scene
  baseline_step{K}_{name}.png        baseline chain intermediate
  kv_step{K}_{name}_probe.png        probe pass (steps 2+)
  kv_step{K}_{name}_target.png       target region overlay (blue)
  kv_step{K}_{name}_background.png   background region overlay (orange)
  kv_step{K}_{name}_result.png       injected result

  KEY_RESULT_chain.png               2 rows (baseline/kv) × N+1 cols (step grid)
  KEY_RESULT_stability.png           per-object diff panels with improvement%
  KEY_RESULT_final.png               baseline final vs kv final
  stability.txt                      numeric table

Usage
-----
python NewWork/KontextEval/phase1_kv_chain_multi.py \\
    --hf_token $HF_TOKEN \\
    --cache_dir ./models \\
    --out_dir results/phase1_kv_chain_multi

Add / remove objects:
    edit EDITS list below, or pass --config my_edits.json
    JSON format: [{"name": "bicycle", "prompt": "..."}, ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

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
# Default edit list — change or extend for your scene
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
            "Add a yellow ball in the right corner of the room next to the sofa. "
            "Keep the rest of the room exactly the same."
        ),
    },
]

BASE_PROMPT   = "A modern living room with a sofa and a wooden coffee table."
TIER_A_LAYERS = list(TIER_A)
ALL_LAYERS    = list(TIER_ALL)


# ============================================================
# Mask / image utilities
# ============================================================

def pixel_to_token_mask(img_a: Image.Image, img_b: Image.Image,
                        h_lat: int, w_lat: int,
                        threshold: float = 40.0) -> np.ndarray:
    """Flat bool (n_gen,) — True where |img_b − img_a| > threshold."""
    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)
    diff = np.abs(b - a).mean(axis=2)
    diff_img = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
    diff_down = diff_img.resize((w_lat, h_lat), Image.BILINEAR)
    return (np.array(diff_down).astype(np.float32) >= threshold).reshape(-1)


def _token_to_pixel(flat_mask: np.ndarray, h_lat: int, w_lat: int,
                    H: int, W: int) -> np.ndarray:
    """Upscale flat token mask to full-image boolean mask."""
    token_2d = flat_mask.reshape(h_lat, w_lat).astype(np.uint8) * 255
    return np.array(
        Image.fromarray(token_2d, "L").resize((W, H), Image.NEAREST)
    ) > 127


def color_overlay(img: Image.Image, flat_mask: np.ndarray,
                  h_lat: int, w_lat: int,
                  color=(255, 120, 0), alpha=0.4) -> Image.Image:
    """Tinted overlay where mask==True."""
    H, W = img.size[1], img.size[0]
    px = _token_to_pixel(flat_mask, h_lat, w_lat, H, W)
    arr = np.array(img).astype(float)
    out = arr.copy()
    out[px] = arr[px] * (1 - alpha) + np.array(color) * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def diff_heatmap(img_a: Image.Image, img_b: Image.Image,
                 flat_mask: np.ndarray, h_lat: int, w_lat: int,
                 amplify: float = 5.0) -> Image.Image:
    """
    Red-channel diff heatmap in the masked region.
    High diff → bright red. Outside the mask → faded grayscale original.
    """
    H, W = img_a.size[1], img_a.size[0]
    px_mask = _token_to_pixel(flat_mask, h_lat, w_lat, H, W)
    a = np.array(img_a).astype(float)
    b = np.array(img_b).astype(float)
    diff = np.abs(b - a).mean(axis=2)
    diff_amp = (diff * amplify).clip(0, 255).astype(np.uint8)
    # Red where inside mask
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :, 0] = diff_amp
    # Faded grayscale where outside mask
    gray = (a.mean(axis=2) * 0.25).clip(0, 255).astype(np.uint8)
    outside = ~px_mask
    rgb[outside, 0] = gray[outside]
    rgb[outside, 1] = gray[outside]
    rgb[outside, 2] = gray[outside]
    return Image.fromarray(rgb)


def region_diff(img_a: Image.Image, img_b: Image.Image,
                flat_mask: np.ndarray, h_lat: int, w_lat: int) -> float:
    """Mean absolute pixel diff (0–255) inside the masked region."""
    H, W = img_a.size[1], img_a.size[0]
    px_mask = _token_to_pixel(flat_mask, h_lat, w_lat, H, W)
    diff = np.abs(
        np.array(img_a).astype(float) - np.array(img_b).astype(float)
    ).mean(axis=2)
    return float(diff[px_mask].mean()) if px_mask.any() else 0.0


def reanchor_background(result: Image.Image, base: Image.Image,
                        base_stable_mask: np.ndarray,
                        h_lat: int, w_lat: int,
                        alpha: float = 0.15) -> Image.Image:
    """
    In the base-stable region, blend the inject-pass result toward the
    original base image to cancel compounding VAE drift.

    At each edit step, Kontext introduces a small amount of background noise
    even with K/V injection (injection is not 100% — s=0.3 means 70% of
    background K/V still comes from the generation path). Over N steps this
    compounds. Re-anchoring with alpha=0.15 pulls room pixels 15% back toward
    the original high-quality base image after every inject pass.

    alpha: fraction of BASE to blend in.  0.0 = no change.  1.0 = pure base.
    Recommended range: 0.10–0.25. Higher = stronger correction but may
    introduce a visible seam at object boundaries.
    """
    if alpha <= 0.0:
        return result
    H, W = result.size[1], result.size[0]
    px_stable = _token_to_pixel(base_stable_mask, h_lat, w_lat, H, W)
    res_arr  = np.array(result).astype(float)
    base_arr = np.array(base).astype(float)
    out = res_arr.copy()
    out[px_stable] = (1.0 - alpha) * res_arr[px_stable] + alpha * base_arr[px_stable]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# ============================================================
# Generation helpers
# ============================================================

def run_standard(pipe, canvas: Image.Image, prompt: str,
                 seed: int, num_steps: int, guidance: float,
                 height: int, width: int) -> Image.Image:
    return generate(
        pipe, prompt, canvas,
        seed=seed, num_steps=num_steps,
        guidance_scale=guidance,
        height=height, width=width,
    )


@torch.no_grad()
def run_injected(pipe, canvas: Image.Image, prompt: str,
                 background_mask: np.ndarray,
                 seed: int, num_steps: int, guidance: float,
                 height: int, width: int,
                 strength: float, cutoff: float,
                 vital_layers: List[int],
                 max_seq_len: int = 512,
                 device: str = "cuda") -> Image.Image:
    """
    Kontext denoising with K/V injection at background_mask tokens.
    background_mask: everything that already exists and must be preserved.
    ~background_mask: where the new object generates freely.
    """
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
# N-step chains
# ============================================================

def run_baseline_chain(pipe, base: Image.Image, edits: List[dict],
                       seed: int, num_steps: int, guidance: float,
                       height: int, width: int) -> List[Image.Image]:
    """Standard Kontext chain — no injection at any step."""
    imgs = [base]
    for edit in edits:
        imgs.append(run_standard(pipe, imgs[-1], edit["prompt"],
                                 seed, num_steps, guidance, height, width))
    return imgs


def run_kv_multi_chain(
    pipe, base: Image.Image, edits: List[dict],
    h_lat: int, w_lat: int,
    seed: int, num_steps: int, probe_steps: int,
    guidance: float, height: int, width: int,
    strength: float, cutoff: float,
    vital_layers: List[int],
    threshold: float, out_dir: str,
    bg_reanchor: float = 0.0,
    device: str = "cuda",
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    N-step chain with background=~target probe→inject at every step.

    Background injection at every step
    ------------------------------------
    The base scene (sofa, walls, floor) must be preserved at high quality
    across every edit. It is injected at every step by ensuring those tokens
    are always in background_mask, even if the probe diff accidentally
    captures some room pixels as target.

    At each step we compute TWO masks from the probe output:
      target_raw      = pixel_to_token_mask(img_prev, probe, threshold)
                        tokens where the NEW OBJECT appeared
      base_stable     = ~pixel_to_token_mask(base, probe, threshold)
                        tokens still matching the ORIGINAL BASE in the probe
                        (never changed by any edit — pure room pixels)

    The final target is their difference:
      target_clean    = target_raw & ~base_stable
    This removes any room pixel that leaked into target_raw due to
    probe threshold noise. Room pixels are forced into background_mask
    and always receive K/V injection from the reference.

      background_mask = ~target_clean  ← base scene + all prior objects

    All steps use probe_steps (default 18).

    Returns
    -------
    imgs      : [base, img_1, ..., img_N]            length = N+1
    obj_masks : [target_clean_1, ..., target_clean_N]  length = N
    """
    imgs, obj_masks = [base], []

    for i, edit in enumerate(edits):
        name      = edit["name"]
        prompt    = edit["prompt"]
        img_prev  = imgs[-1]
        n_probe   = probe_steps

        print(f"\n  [step {i+1}/{len(edits)}] {name}")

        # ── Probe pass ──
        print(f"    probe ({n_probe} steps) …")
        img_probe = run_standard(pipe, img_prev, prompt,
                                 seed, n_probe, guidance, height, width)
        img_probe.save(os.path.join(out_dir, f"kv_step{i+1}_{name}_probe.png"))

        # Raw target: where new object appeared vs previous image
        target_raw   = pixel_to_token_mask(img_prev, img_probe, h_lat, w_lat, threshold)

        # Base-stable: tokens still matching the original base in the probe
        # These are pure room pixels — must always be in background_mask
        base_stable  = ~pixel_to_token_mask(base, img_probe, h_lat, w_lat, threshold)

        # Clean target: new object only, with room pixels forced out
        target_clean = target_raw & ~base_stable

        pct_raw   = target_raw.mean()   * 100
        pct_clean = target_clean.mean() * 100
        pct_bg    = (~target_clean).mean() * 100
        print(f"    target raw={pct_raw:.1f}%  clean={pct_clean:.1f}%  "
              f"background(injected)={pct_bg:.1f}%  (threshold={threshold})")

        # Save overlays on img_prev so user can verify
        color_overlay(img_prev, target_clean, h_lat, w_lat,
                      color=(0, 160, 255), alpha=0.45).save(
            os.path.join(out_dir, f"kv_step{i+1}_{name}_target.png"))
        color_overlay(img_prev, base_stable, h_lat, w_lat,
                      color=(0, 220, 80), alpha=0.30).save(
            os.path.join(out_dir, f"kv_step{i+1}_{name}_bg_stable.png"))
        color_overlay(img_prev, ~target_clean, h_lat, w_lat,
                      color=(255, 140, 0), alpha=0.20).save(
            os.path.join(out_dir, f"kv_step{i+1}_{name}_background.png"))

        obj_masks.append(target_clean)

        # ── Inject pass ──
        background_mask = ~target_clean
        print(f"    inject ({num_steps} steps, s={strength}, cutoff={cutoff}) …")
        img_curr = run_injected(
            pipe, img_prev, prompt, background_mask,
            seed=seed, num_steps=num_steps,
            guidance=guidance, height=height, width=width,
            strength=strength, cutoff=cutoff,
            vital_layers=vital_layers, device=device,
        )

        if bg_reanchor > 0.0:
            img_curr = reanchor_background(
                img_curr, base, base_stable, h_lat, w_lat, alpha=bg_reanchor,
            )
            print(f"    bg_reanchor α={bg_reanchor:.2f} applied "
                  f"({base_stable.mean()*100:.1f}% of tokens re-anchored to base)")

        img_curr.save(os.path.join(out_dir, f"kv_step{i+1}_{name}_result.png"))

        imgs.append(img_curr)

    return imgs, obj_masks


# ============================================================
# Visualisation
# ============================================================

def save_grid(images, titles, path, ncols=None, figsize_per_cell=(5, 5)):
    n = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
    )
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=8)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def build_stability_grid(
    base: Image.Image,
    baseline_imgs: List[Image.Image],
    kv_imgs: List[Image.Image],
    edits: List[dict],
    obj_masks: List[np.ndarray],
    h_lat: int, w_lat: int,
    out_dir: str,
    strength: float, cutoff: float,
):
    """
    Per-object (+ background) stability comparison.

    For each region (background + N objects):
      Row A (baseline): [when-added | final | diff heatmap]
      Row B (kv_multi): [when-added | final | diff heatmap]
      Title shows raw diff values and improvement%.

    Also returns the improvement_pct dict for the summary table.
    """
    n = len(edits)

    # Background mask: complement of union of all object masks
    all_obj_union = np.zeros(obj_masks[0].shape, dtype=bool)
    for m in obj_masks:
        all_obj_union |= m
    bg_mask = ~all_obj_union

    # Build regions list: (name, b_ref, k_ref, b_before, k_before, mask)
    #   b_ref    = image at the step this region was established (for stability Δ)
    #   b_before = image BEFORE this region was established (for heatmap source)
    #
    # Heatmap: diff(b_before, b_final) in mask region
    #   Always shows the object/region brightly, including for the last edit
    #   where diff(b_ref, b_final)=0 because b_ref IS b_final.
    #   For ball (last step): diff(vase_image, ball_image) → ball lights up.
    #   For bicycle (step 1): diff(base, final) → bicycle lights up.
    #   For background: diff(base, final) → any room drift lights up.
    regions = [("background", base, base, base, base, bg_mask)]
    for i, edit in enumerate(edits):
        regions.append((
            edit["name"],
            baseline_imgs[i + 1], kv_imgs[i + 1],   # ref: when added
            baseline_imgs[i],     kv_imgs[i],          # before: just prior step
            obj_masks[i],
        ))

    n_regions = len(regions)
    ncols, nrows = 3, 2 * n_regions
    images, titles = [], []
    improvement = {}

    for name, b_ref, k_ref, b_before, k_before, mask in regions:
        b_final = baseline_imgs[-1]
        k_final = kv_imgs[-1]

        # Stability metric: how much did this region change from when it was added → final
        b_diff = region_diff(b_ref, b_final, mask, h_lat, w_lat)
        k_diff = region_diff(k_ref, k_final, mask, h_lat, w_lat)
        pct    = (b_diff - k_diff) / max(b_diff, 1e-6) * 100
        improvement[name] = {"b_diff": b_diff, "k_diff": k_diff, "pct": pct}

        # Heatmap: diff(before, final) — always bright because the object appeared
        b_heat = diff_heatmap(b_before, b_final, mask, h_lat, w_lat)
        k_heat = diff_heatmap(k_before, k_final, mask, h_lat, w_lat)

        images += [b_ref, b_final, b_heat]
        titles += [
            f"[{name}] baseline\nwhen added",
            f"[{name}] baseline final\nstab Δ={b_diff:.1f}",
            f"Baseline: before→final\n(bright = object present)",
        ]
        images += [k_ref, k_final, k_heat]
        titles += [
            f"[{name}] kv_multi\nwhen added",
            f"[{name}] kv final\nstab Δ={k_diff:.1f}  {pct:+.0f}%",
            f"KV: before→final\n(bright = object present)",
        ]

    save_grid(images, titles,
              os.path.join(out_dir, "KEY_RESULT_stability.png"),
              ncols=ncols)
    return improvement


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",    required=True)
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--out_dir",     default="results/phase1_kv_chain_multi")
    p.add_argument("--config",      default=None,
                   help="JSON file list of {name, prompt}. Overrides built-in EDITS.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_steps",   type=int,   default=28)
    p.add_argument("--probe_steps", type=int, default=18,
                   help="Probe steps for all steps (18/28 ≈ 64%% of denoising).")
    p.add_argument("--guidance",    type=float, default=2.5)
    p.add_argument("--strength",    type=float, default=0.3,
                   help="K/V injection weight. Validated: s=0.3 best (10.56 bike_diff). "
                        "Range [0.3, 0.7] barely matters; s=0.7 gives stronger background "
                        "locking with minimal cost (10.77 bike_diff).")
    p.add_argument("--bg_reanchor", type=float, default=0.15,
                   help="After each inject pass, blend base-stable pixels toward the "
                        "original base image by this fraction (0=off, 0.15=default). "
                        "Prevents VAE-drift compounding across edit steps. Range: 0.10-0.25.")
    p.add_argument("--cutoff",      type=float, default=0.4,
                   help="Inject during first CUTOFF fraction of steps. "
                        "Validated: 0.4 slightly better than 0.6.")
    p.add_argument("--threshold",   type=float, default=40.0,
                   help="Pixel diff threshold (0-255) for target mask from probe.")
    p.add_argument("--all_layers",  action="store_true",
                   help="Use all 57 layers (known-bad control: bike_diff=58.80 > baseline).")
    p.add_argument("--height",      type=int,   default=1024)
    p.add_argument("--width",       type=int,   default=1024)
    p.add_argument("--device",      default="cuda")
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

    print(f"Edit sequence: {[e['name'] for e in edits]}")
    print(f"All steps: probe({args.probe_steps}) → inject  "
          f"s={args.strength}, cutoff={args.cutoff}, "
          f"layers={'ALL_57' if args.all_layers else 'TIER_A'}, "
          f"bg_reanchor={args.bg_reanchor}")

    # ── Load model ──────────────────────────────────────────
    print("\nLoading FLUX.1-Kontext-dev …")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    # ── Step 0: base scene ──────────────────────────────────
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(pipe, grey, BASE_PROMPT,
                        args.seed, args.num_steps, args.guidance,
                        args.height, args.width)
    base.save(os.path.join(args.out_dir, "step0_base.png"))

    # ── Baseline chain ───────────────────────────────────────
    print("\n=== BASELINE chain (no injection) ===")
    baseline_imgs = run_baseline_chain(
        pipe, base, edits,
        args.seed, args.num_steps, args.guidance, args.height, args.width,
    )
    for i, edit in enumerate(edits):
        baseline_imgs[i + 1].save(
            os.path.join(args.out_dir, f"baseline_step{i+1}_{edit['name']}.png"))

    # ── K/V multi chain ─────────────────────────────────────
    print("\n=== K/V MULTI chain (step1=standard, steps2+=probe→inject) ===")
    kv_imgs, obj_masks = run_kv_multi_chain(
        pipe, base, edits,
        h_lat=h_lat, w_lat=w_lat,
        seed=args.seed, num_steps=args.num_steps,
        probe_steps=args.probe_steps,
        guidance=args.guidance, height=args.height, width=args.width,
        strength=args.strength, cutoff=args.cutoff,
        vital_layers=vital_layers,
        threshold=args.threshold,
        bg_reanchor=args.bg_reanchor,
        out_dir=args.out_dir, device=args.device,
    )

    # ── Stability grid + improvement % ──────────────────────
    print("\n=== Building stability visualisation ===")
    improvement = build_stability_grid(
        base, baseline_imgs, kv_imgs, edits, obj_masks,
        h_lat, w_lat, args.out_dir, args.strength, args.cutoff,
    )
    print("  Saved: KEY_RESULT_stability.png")

    # ── Chain comparison grid ────────────────────────────────
    n_steps = len(edits)
    chain_imgs = (
        [base] + baseline_imgs[1:]
        + [base] + kv_imgs[1:]
    )
    chain_titles = (
        ["Base"]
        + [f"Baseline step {i+1}\n{edits[i]['name']}" for i in range(n_steps)]
        + ["Base"]
        + [f"KV-Multi step {i+1}\n{edits[i]['name']}" for i in range(n_steps)]
    )
    save_grid(
        chain_imgs, chain_titles,
        os.path.join(args.out_dir, "KEY_RESULT_chain.png"),
        ncols=n_steps + 1,
    )
    print(f"  Saved: KEY_RESULT_chain.png  (2 rows × {n_steps+1} cols)")

    # Final side-by-side
    save_grid(
        [baseline_imgs[-1], kv_imgs[-1]],
        ["Baseline final  (no injection)",
         f"KV-Multi final  (s={args.strength}, cutoff={args.cutoff})"],
        os.path.join(args.out_dir, "KEY_RESULT_final.png"),
    )
    print("  Saved: KEY_RESULT_final.png")

    # ── Numeric summary ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("STABILITY TABLE")
    print(f"{'='*70}")
    print(f"  {'region':<14}  {'baseline Δ':>11}  {'kv Δ':>8}  "
          f"{'improve %':>10}  {'verdict':>10}")
    print(f"  {'-'*14}  {'-'*11}  {'-'*8}  {'-'*10}  {'-'*10}")

    lines = [
        "region_stability: mean abs pixel diff in region vs FINAL image\n"
        "lower = better preserved; improvement% = (baseline-kv)/baseline*100\n\n"
        f"{'region':<14}  {'baseline_diff':>13}  {'kv_diff':>8}  "
        f"{'improve%':>9}  verdict\n"
        f"{'-'*14}  {'-'*13}  {'-'*8}  {'-'*9}  {'-'*10}\n"
    ]

    for name, d in improvement.items():
        bd, kd, pct = d["b_diff"], d["k_diff"], d["pct"]
        verdict = "IMPROVED" if kd < bd else "WORSE"
        print(f"  {name:<14}  {bd:>11.2f}  {kd:>8.2f}  {pct:>+9.1f}%  {verdict}")
        lines.append(f"{name:<14}  {bd:>13.2f}  {kd:>8.2f}  {pct:>+8.1f}%  {verdict}\n")

    avg_pct = np.mean([d["pct"] for d in improvement.values()])
    print(f"\n  Average improvement across all regions: {avg_pct:+.1f}%")
    print(f"  Δ = mean abs pixel diff (0–255); LOWER = better preserved")
    lines.append(f"\nAverage improvement: {avg_pct:+.1f}%\n")

    with open(os.path.join(args.out_dir, "stability.txt"), "w") as f:
        f.writelines(lines)
    print("  Saved: stability.txt")

    # ── What to check ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("WHAT TO CHECK")
    print(f"{'='*70}")
    print(f"""
KEY_RESULT_chain.png  ← step-by-step overview
  Top row (baseline): each edit applied in sequence without injection.
  Bottom row (kv_multi): same edits with background protection from step 2.
  Compare column-by-column: are prior objects more stable in the bottom row?

KEY_RESULT_stability.png  ← per-object analysis (most detailed)
  For each region (background + each object), two rows:
    Row 1 (baseline): when-added | final | diff heatmap (red=change)
    Row 2 (kv_multi): when-added | final | diff heatmap
  The diff heatmap shows exactly which pixels in that region changed.
  improvement% in the title: positive = kv_multi preserved better.
  Bright red in the baseline heatmap + dark baseline in kv = clear win.

KEY_RESULT_final.png
  Direct comparison of the final images.
  Both should have all 3 objects. Quality/identity of prior objects
  (especially bicycle) should be better in kv_multi.

Probe images (kv_step*_probe.png, only for steps 2+)
  These show where Kontext naturally places the new object.
  Target overlay (blue) should cover the new object region.
  If target is too noisy: increase --threshold (try 50-60).
  If target misses the object: decrease --threshold (try 25) or increase --probe_steps.

Stability table interpretation:
  background: how much the original room (sofa/walls/floor) drifted.
  bicycle:    how much the bicycle changed after the vase and lamp were added.
  vase:       how much the vase changed after the lamp was added.
  lamp:       how much the lamp changed vs when it was first placed (step 3).
              (lamp stability is low for both since it IS the last step)

If bicycle is still not forming well:
  → check kv_step1_bicycle_result.png (step 1 is standard, should be fine)
  → if the bicycle is fine in kv_step1, the issue is at step 2 injection
  → try --strength 0.5 and --threshold 50 to give the bicycle a larger free zone
""")
    print(f"All results → {args.out_dir}/")


if __name__ == "__main__":
    main()
