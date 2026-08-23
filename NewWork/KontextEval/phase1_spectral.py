"""
Phase 1 — Spectral Injection: Mathematical Alternatives to K/V Injection

Three injection strategies compared against a no-injection baseline:

  kv (A)        Standard K/V injection (current approach, phase1_kv_chain_multi.py)
                K_gen[bg] = (1−s)·K_gen[bg] + s·K_ref[bg]
                Uniform linear blend across all 128 feature dimensions.

  lowrank (B)   Low-rank K/V injection  (truncated SVD — pure linear algebra)
                K_ref_lr = rank-k approximation of K_ref[bg]  via truncated SVD
                K_gen[bg] = (1−s)·K_gen[bg] + s·K_ref_lr
                Injects only the top-k dominant singular components of the reference
                background features. The remaining (128−k) dimensions are generated
                freely. Sanity: rank_k=128 → identical to K/V injection at s=1.0.
                Parameter: --rank_k (default 16).

  nullspace (C) Edit-direction null-space injection  (projection theorem)
                Δ_tgt = K_gen[tgt] − K_ref[tgt]       (edit drift in target region)
                P_⊥   = I − U_edit · Uᵀ_edit           (project onto edit complement)
                K_new[bg] = K_ref[bg] + (K_gen[bg] − K_ref[bg]) · P_⊥
                Background drift is corrected ONLY in the directions the edit is
                changing things. In edit-orthogonal directions the background generates
                freely. Sanity: edit_rank=0 → no correction (same as baseline).
                Parameter: --edit_rank (default 8).

Both spectral methods monkey-patch the _inject instance method of
KontextInjectionProcessor (from kontext_injection.py), leaving all attention
computation (SDPA, RoPE, layer norm, Q projection) entirely untouched.

Usage
-----
python NewWork/KontextEval/phase1_spectral.py \\
    --hf_token $HF_TOKEN --cache_dir ./models \\
    --out_dir results/phase1_spectral --mode all \\
    --rank_k 16 --edit_rank 8
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
# Shared image / mask utilities
# ============================================================

def _step_active(step: int, n_steps: int, frac: Tuple[float, float]) -> bool:
    return int(frac[0] * n_steps) <= step < int(frac[1] * n_steps)


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
# Spectral injection kernels
# These are module-level functions that replace KontextInjectionProcessor._inject
# via instance-level monkey-patching. Python's descriptor protocol means
# self._inject(k, v, img_offset) calls the lambda directly (no implicit self).
# ============================================================

def _lowrank_inject(k, v, img_offset: int, state: InjectionState, rank_k: int):
    """
    Low-rank background injection.

    Decomposes K_ref[bg] via truncated SVD; injects only the top rank_k
    singular components into K_gen[bg]. The remaining (d−rank_k) dimensions
    continue to generate freely, preserving fine-detail variation while
    anchoring the dominant structural content of the background.

    Formula:  K_gen[bg] = (1−s)·K_gen[bg] + s·(U_k Σ_k Vhᵀ_k)
              where U_k, Σ_k, Vh_k come from the rank_k truncated SVD of K_ref[bg].
    """
    if not _step_active(state.cur_step, state.n_steps, state.cutoff_frac):
        return k, v

    n_gen = state.n_gen
    gen_lo, gen_hi = img_offset, img_offset + n_gen
    ref_lo, ref_hi = gen_hi, gen_hi + state.n_ref

    s     = state.strength
    bg    = state.zones.background  # (n_gen,) bool tensor on device

    k_gen = k[:, :, gen_lo:gen_hi, :].clone()  # (1, H, n_gen, d)
    v_gen = v[:, :, gen_lo:gen_hi, :].clone()

    # (H, n_bg, d) — squeeze the batch-1 dimension after bool-indexing
    K_bg_gen = k_gen[:, :, bg, :].squeeze(0)
    V_bg_gen = v_gen[:, :, bg, :].squeeze(0)
    K_bg_ref = k[:, :, ref_lo:ref_hi, :][:, :, bg, :].squeeze(0)
    V_bg_ref = v[:, :, ref_lo:ref_hi, :][:, :, bg, :].squeeze(0)

    # Batched SVD over H heads: (H, n_bg, d)
    # full_matrices=False with n_bg > d=128:
    #   U (H, n_bg, 128), S (H, 128), Vh (H, 128, 128)
    U_k, S_k, Vh_k = torch.linalg.svd(K_bg_ref.float(), full_matrices=False)
    r = min(rank_k, S_k.shape[-1])
    # Rank-r reconstruction: (H, n_bg, r) * (H, 1, r)  @  (H, r, 128) → (H, n_bg, 128)
    K_ref_lr = ((U_k[:, :, :r] * S_k[:, :r].unsqueeze(1)) @ Vh_k[:, :r, :]).to(k.dtype)

    U_v, S_v, Vh_v = torch.linalg.svd(V_bg_ref.float(), full_matrices=False)
    V_ref_lr = ((U_v[:, :, :r] * S_v[:, :r].unsqueeze(1)) @ Vh_v[:, :r, :]).to(v.dtype)

    k_gen[:, :, bg, :] = ((1 - s) * K_bg_gen + s * K_ref_lr).unsqueeze(0)
    v_gen[:, :, bg, :] = ((1 - s) * V_bg_gen + s * V_ref_lr).unsqueeze(0)

    k = k.clone()
    v = v.clone()
    k[:, :, gen_lo:gen_hi, :] = k_gen
    v[:, :, gen_lo:gen_hi, :] = v_gen
    return k, v


def _nullspace_inject(k, v, img_offset: int, state: InjectionState, edit_rank: int):
    """
    Edit-direction null-space injection.

    Computes the edit subspace from the drift of TARGET tokens
    (Δ = K_gen[tgt] − K_ref[tgt]) via SVD. Then for BACKGROUND tokens,
    corrects only the component of the background drift that lies IN the edit
    subspace; the orthogonal complement is left free for natural generation.

    Formula:  Δ_tgt = K_gen[tgt] − K_ref[tgt]
              U_edit = top edit_rank right-singular vectors of Δ_tgt  (per head)
              P_⊥    = I − U_edit · Uᵀ_edit
              K_new[bg] = K_ref[bg] + (K_gen[bg] − K_ref[bg]) · P_⊥

    Intuition: the background should not change in directions the edit is
    changing things. In edit-orthogonal directions, the background generates
    completely freely.
    """
    if not _step_active(state.cur_step, state.n_steps, state.cutoff_frac):
        return k, v

    n_gen = state.n_gen
    gen_lo, gen_hi = img_offset, img_offset + n_gen
    ref_lo, ref_hi = gen_hi, gen_hi + state.n_ref

    bg  = state.zones.background  # (n_gen,) bool
    tgt = state.zones.target      # (n_gen,) bool = ~background (no shell zone)

    if int(tgt.sum()) == 0:
        return k, v

    k_gen = k[:, :, gen_lo:gen_hi, :].clone()
    v_gen = v[:, :, gen_lo:gen_hi, :].clone()

    K_ref_img = k[:, :, ref_lo:ref_hi, :]
    V_ref_img = v[:, :, ref_lo:ref_hi, :]

    # Target tokens: (H, n_tgt, d)
    delta_K = (k_gen[:, :, tgt, :] - K_ref_img[:, :, tgt, :]).squeeze(0).float()
    delta_V = (v_gen[:, :, tgt, :] - V_ref_img[:, :, tgt, :]).squeeze(0).float()

    d = delta_K.shape[-1]  # 128

    # SVD of (H, n_tgt, d) — full_matrices=False:
    #   When n_tgt < d: Vh shape (H, n_tgt, d)
    #   When n_tgt >= d: Vh shape (H, d, d)
    # Top-r rows of Vh are the top-r right-singular vectors (basis of edit subspace in R^d)
    _, _, Vh_K = torch.linalg.svd(delta_K, full_matrices=False)
    r_K = min(edit_rank, Vh_K.shape[1])
    # P_edit = Vh[:r].T @ Vh[:r]   (d×d projector onto edit subspace)
    # batched: Vh_K[:, :r, :].transpose(-1, -2) @ Vh_K[:, :r, :]
    # (H, d, r_K) @ (H, r_K, d) = (H, d, d)
    P_edit_K = Vh_K[:, :r_K, :].transpose(-1, -2) @ Vh_K[:, :r_K, :]
    P_perp_K = (
        torch.eye(d, dtype=P_edit_K.dtype, device=P_edit_K.device).unsqueeze(0)
        - P_edit_K
    )  # (H, d, d)

    _, _, Vh_V = torch.linalg.svd(delta_V, full_matrices=False)
    r_V = min(edit_rank, Vh_V.shape[1])
    P_edit_V = Vh_V[:, :r_V, :].transpose(-1, -2) @ Vh_V[:, :r_V, :]
    P_perp_V = (
        torch.eye(d, dtype=P_edit_V.dtype, device=P_edit_V.device).unsqueeze(0)
        - P_edit_V
    )

    # Background tokens: (H, n_bg, d)
    K_bg_gen = k_gen[:, :, bg, :].squeeze(0).float()
    K_bg_ref = K_ref_img[:, :, bg, :].squeeze(0).float()
    V_bg_gen = v_gen[:, :, bg, :].squeeze(0).float()
    V_bg_ref = V_ref_img[:, :, bg, :].squeeze(0).float()

    # Remove edit-subspace component from background drift
    # (H, n_bg, d) @ (H, d, d) → (H, n_bg, d)
    K_new = (K_bg_ref + (K_bg_gen - K_bg_ref) @ P_perp_K).to(k.dtype)
    V_new = (V_bg_ref + (V_bg_gen - V_bg_ref) @ P_perp_V).to(v.dtype)

    k_gen[:, :, bg, :] = K_new.unsqueeze(0)
    v_gen[:, :, bg, :] = V_new.unsqueeze(0)

    k = k.clone()
    v = v.clone()
    k[:, :, gen_lo:gen_hi, :] = k_gen
    v[:, :, gen_lo:gen_hi, :] = v_gen
    return k, v


# ============================================================
# Injected generation
# ============================================================

@torch.no_grad()
def run_spectral_injected(
    pipe,
    canvas: Image.Image,
    prompt: str,
    background_mask: np.ndarray,
    spectral_mode: str,
    seed: int,
    num_steps: int,
    guidance: float,
    height: int,
    width: int,
    strength: float,
    cutoff: float,
    vital_layers: List[int],
    rank_k: int = 16,
    edit_rank: int = 8,
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

    proc = install_processor(pipe, state, max_sequence_length=max_seq_len)

    # Replace _inject at the INSTANCE level; Python does not auto-bind plain
    # functions stored as instance attributes, so self._inject(k, v, io)
    # calls our lambda with exactly those three args.
    if spectral_mode == "lowrank":
        proc._inject = (
            lambda k, v, io, _s=state, _r=rank_k: _lowrank_inject(k, v, io, _s, _r)
        )
    elif spectral_mode == "nullspace":
        proc._inject = (
            lambda k, v, io, _s=state, _e=edit_rank: _nullspace_inject(k, v, io, _s, _e)
        )
    # "kv": original _inject is used unchanged

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
# Per-mode N-step chain
# ============================================================

def run_spectral_chain(
    pipe,
    base: Image.Image,
    edits: List[dict],
    mode: str,
    h_lat: int, w_lat: int,
    seed: int, num_steps: int, probe_steps: int,
    guidance: float, height: int, width: int,
    strength: float, cutoff: float,
    vital_layers: List[int],
    threshold: float,
    out_dir: str,
    rank_k: int = 16,
    edit_rank: int = 8,
    device: str = "cuda",
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    Full probe→inject chain for one spectral mode.
    'baseline' runs probe + full standard generation (no injection).
    All others run probe + spectral_injected.

    Returns (imgs, obj_masks):
      imgs[0]     = base
      imgs[i]     = result after edit i-1
      obj_masks[i]= target_clean for edit i (probe-derived)
    """
    prefix = {"baseline": "bl", "kv": "kv", "lowrank": "lr", "nullspace": "ns"}[mode]
    imgs, obj_masks = [base], []

    for i, edit in enumerate(edits):
        name     = edit["name"]
        prompt   = edit["prompt"]
        img_prev = imgs[-1]

        print(f"\n  [{mode}] step {i+1}/{len(edits)}  {name}")

        # ── Probe ──────────────────────────────────────────────
        print(f"    probe ({probe_steps} steps) …")
        img_probe = run_standard(pipe, img_prev, prompt,
                                 seed, probe_steps, guidance, height, width)
        img_probe.save(os.path.join(out_dir, f"{prefix}_step{i+1}_{name}_probe.png"))

        target_raw   = pixel_to_token_mask(img_prev, img_probe, h_lat, w_lat, threshold)
        base_stable  = ~pixel_to_token_mask(base, img_probe, h_lat, w_lat, threshold)
        target_clean = target_raw & ~base_stable
        obj_masks.append(target_clean)

        pct_tgt = target_clean.mean() * 100
        pct_bg  = (~target_clean).mean() * 100
        print(f"    target={pct_tgt:.1f}%  background(injected)={pct_bg:.1f}%")

        color_overlay(img_prev, target_clean, h_lat, w_lat,
                      color=(0, 160, 255), alpha=0.45).save(
            os.path.join(out_dir, f"{prefix}_step{i+1}_{name}_target.png"))

        background_mask = ~target_clean

        # ── Inject ─────────────────────────────────────────────
        if mode == "baseline":
            print(f"    standard ({num_steps} steps, no injection)")
            img_curr = run_standard(pipe, img_prev, prompt,
                                    seed, num_steps, guidance, height, width)
        else:
            print(f"    {mode} inject ({num_steps} steps, s={strength}, cutoff={cutoff})")
            img_curr = run_spectral_injected(
                pipe, img_prev, prompt, background_mask,
                spectral_mode=mode,
                seed=seed, num_steps=num_steps,
                guidance=guidance, height=height, width=width,
                strength=strength, cutoff=cutoff,
                vital_layers=vital_layers,
                rank_k=rank_k, edit_rank=edit_rank,
                device=device,
            )

        img_curr.save(os.path.join(out_dir, f"{prefix}_step{i+1}_{name}_result.png"))
        imgs.append(img_curr)

    return imgs, obj_masks


# ============================================================
# Visualisation
# ============================================================

def build_comparison_grid(
    base: Image.Image,
    imgs_by_mode: Dict[str, List[Image.Image]],
    edits: List[dict],
    out_dir: str,
    strength: float, rank_k: int, edit_rank: int,
):
    """Chain overview: rows = modes, columns = steps."""
    modes = list(imgs_by_mode.keys())
    n_steps = len(edits)
    ncols = n_steps + 1
    labels = {
        "baseline": "Baseline\n(no inject)",
        "kv":       f"K/V inject\n(s={strength})",
        "lowrank":  f"Low-rank\n(k={rank_k}, s={strength})",
        "nullspace": f"Null-space\n(r={edit_rank})",
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


def build_spectral_stability_grid(
    base: Image.Image,
    imgs_by_mode: Dict[str, List[Image.Image]],
    edits: List[dict],
    obj_masks_by_mode: Dict[str, List[np.ndarray]],
    h_lat: int, w_lat: int,
    out_dir: str,
    strength: float, rank_k: int, edit_rank: int,
) -> Dict[str, Dict[str, float]]:
    """
    Stability grid: for each region (background + each object), one row showing
    the 'when-added' reference image (from K/V chain) followed by the final
    image from each mode with the diff value in the title.

    Returns stability[region_name][mode] = mean-abs-pixel-diff.
    """
    modes = list(imgs_by_mode.keys())

    # Reference masks from the KV chain (or first available)
    ref_key   = "kv" if "kv" in obj_masks_by_mode else list(obj_masks_by_mode.keys())[0]
    ref_masks = obj_masks_by_mode[ref_key]

    all_obj = np.zeros(ref_masks[0].shape, dtype=bool)
    for m in ref_masks:
        all_obj |= m
    bg_mask = ~all_obj

    # region_specs: (name, mask, step_added)
    region_specs = [("background", bg_mask, 0)]
    for i, edit in enumerate(edits):
        region_specs.append((edit["name"], ref_masks[i], i + 1))

    # Compute stability per region per mode
    stability: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in region_specs}
    ref_imgs = imgs_by_mode[ref_key]

    for mode in modes:
        imgs  = imgs_by_mode[mode]
        final = imgs[-1]
        for name, mask, step_added in region_specs:
            when_added = ref_imgs[step_added]
            stability[name][mode] = region_diff(when_added, final, mask, h_lat, w_lat)

    # Visual grid: columns = [when_added (kv-ref)] + [final_mode1, final_mode2, ...]
    labels = {
        "baseline": f"Baseline",
        "kv":       f"K/V (s={strength})",
        "lowrank":  f"Low-rank k={rank_k}",
        "nullspace": f"Null-space r={edit_rank}",
    }
    ncols = len(modes) + 1
    images, titles = [], []

    for name, mask, step_added in region_specs:
        images.append(ref_imgs[step_added])
        titles.append(f"[{name}]\nwhen added")
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
        description="Spectral K/V injection: low-rank and null-space alternatives."
    )
    p.add_argument("--hf_token",    required=True)
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--out_dir",     default="results/phase1_spectral")
    p.add_argument("--config",      default=None,
                   help="JSON list of {name, prompt}. Overrides built-in EDITS.")
    p.add_argument("--mode",        default="all",
                   choices=["all", "kv", "lowrank", "nullspace", "baseline"],
                   help="Which mode(s) to run. 'all' runs baseline+kv+lowrank+nullspace.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_steps",   type=int,   default=28)
    p.add_argument("--probe_steps", type=int,   default=18,
                   help="Probe steps for object-placement detection (18/28 ≈ 64%%).")
    p.add_argument("--guidance",    type=float, default=2.5)
    p.add_argument("--strength",    type=float, default=0.3,
                   help="Injection weight for kv and lowrank modes.")
    p.add_argument("--cutoff",      type=float, default=0.4,
                   help="Inject during first CUTOFF fraction of steps.")
    p.add_argument("--rank_k",      type=int,   default=16,
                   help="(lowrank) SVD rank of injected reference approximation. "
                        "rank_k=128 → identical to K/V injection at s=1.0.")
    p.add_argument("--edit_rank",   type=int,   default=8,
                   help="(nullspace) Rank of the edit subspace projected away from "
                        "background. edit_rank=0 → no correction. edit_rank=128 → ~K/V s=1.0.")
    p.add_argument("--threshold",   type=float, default=40.0,
                   help="Pixel diff threshold for target mask from probe.")
    p.add_argument("--all_layers",  action="store_true",
                   help="Use all 57 FLUX layers (control; TIER_A is the validated default).")
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
    modes = (["baseline", "kv", "lowrank", "nullspace"]
             if args.mode == "all" else [args.mode])

    print(f"Edit sequence : {[e['name'] for e in edits]}")
    print(f"Modes         : {modes}")
    print(f"rank_k={args.rank_k}  edit_rank={args.edit_rank}  "
          f"s={args.strength}  cutoff={args.cutoff}  "
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

    # ── Run all chains ────────────────────────────────────────
    imgs_by_mode:     Dict[str, List[Image.Image]]  = {}
    obj_masks_by_mode: Dict[str, List[np.ndarray]]  = {}

    for mode in modes:
        print(f"\n{'='*64}")
        print(f"=== {mode.upper()} chain ===")
        print(f"{'='*64}")
        imgs, obj_masks = run_spectral_chain(
            pipe, base, edits, mode=mode,
            h_lat=h_lat, w_lat=w_lat,
            seed=args.seed, num_steps=args.num_steps,
            probe_steps=args.probe_steps,
            guidance=args.guidance,
            height=args.height, width=args.width,
            strength=args.strength, cutoff=args.cutoff,
            vital_layers=vital_layers,
            threshold=args.threshold,
            out_dir=args.out_dir,
            rank_k=args.rank_k, edit_rank=args.edit_rank,
            device=args.device,
        )
        imgs_by_mode[mode]      = imgs
        obj_masks_by_mode[mode] = obj_masks
        imgs[-1].save(os.path.join(args.out_dir, f"KEY_{mode}_final.png"))

    # ── Build visuals ─────────────────────────────────────────
    print("\n=== Building comparison grid ===")
    build_comparison_grid(
        base, imgs_by_mode, edits, args.out_dir,
        strength=args.strength, rank_k=args.rank_k, edit_rank=args.edit_rank,
    )
    print("  Saved: KEY_RESULT_comparison.png")

    print("=== Building stability grid ===")
    stability = build_spectral_stability_grid(
        base, imgs_by_mode, edits, obj_masks_by_mode,
        h_lat, w_lat, args.out_dir,
        strength=args.strength, rank_k=args.rank_k, edit_rank=args.edit_rank,
    )
    print("  Saved: KEY_RESULT_stability.png")

    # ── Numeric stability table ───────────────────────────────
    region_names = list(stability.keys())
    header_modes = modes

    print(f"\n{'='*72}")
    print("STABILITY TABLE  (mean abs pixel diff — lower = better preserved)")
    print(f"{'='*72}")
    col_w = 12
    hdr = f"  {'region':<14}" + "".join(f"  {m:>{col_w}}" for m in header_modes)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    file_lines = [
        "Stability = mean abs pixel diff between 'when-added' and final image.\n"
        "Lower = better preserved across subsequent edits.\n\n",
        f"{'region':<14}" + "".join(f"  {m:>{col_w}}" for m in header_modes) + "\n",
        "-" * 70 + "\n",
    ]

    for name in region_names:
        row = f"  {name:<14}"
        file_row = f"{name:<14}"
        for mode in header_modes:
            val = stability[name].get(mode, float("nan"))
            row      += f"  {val:>{col_w}.2f}"
            file_row += f"  {val:>{col_w}.2f}"
        print(row)
        file_lines.append(file_row + "\n")

    # Improvement vs baseline
    if "baseline" in header_modes and len(header_modes) > 1:
        print()
        impr_line = f"  {'improve %':<14}"
        for mode in header_modes:
            if mode == "baseline":
                impr_line += f"  {'(ref)':>{col_w}}"
            else:
                pcts = [
                    (stability[n].get("baseline", 0) - stability[n].get(mode, 0))
                    / max(stability[n].get("baseline", 1e-6), 1e-6) * 100
                    for n in region_names
                ]
                avg = float(np.mean(pcts))
                impr_line += f"  {avg:>+{col_w-1}.1f}%"
        print(impr_line)
        file_lines.append("\n" + impr_line.strip() + "\n")

    with open(os.path.join(args.out_dir, "stability.txt"), "w") as f:
        f.writelines(file_lines)
    print("\n  Saved: stability.txt")

    # ── What to look for ─────────────────────────────────────
    print(f"\n{'='*72}")
    print("WHAT TO CHECK")
    print(f"{'='*72}")
    print("""
KEY_RESULT_comparison.png
  Rows = modes, columns = base → step1 → step2 → step3.
  Prior objects should be more stable in injected rows vs baseline.

KEY_RESULT_stability.png
  For each region: 'when added' + final from each mode.
  Lower Δ in the title = better preservation.

KEY_[mode]_final.png  — final image for each mode.

Sanity checks:
  --rank_k 128           → lowrank should closely match kv (at s=1.0)
  --edit_rank 0          → nullspace should match baseline (no correction)
  --mode lowrank --rank_k 16 vs 64 → intermediate ranks trade detail for preservation
""")


if __name__ == "__main__":
    main()
