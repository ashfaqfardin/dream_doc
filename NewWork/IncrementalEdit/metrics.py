"""
Verification numbers (pipelineInc.md §6). Every stage must emit a number a
human can check — this module is where those numbers are computed.

`latent_mse` is written duck-typed (works on numpy arrays OR torch tensors,
without importing torch) because it runs on real torch latents at inference
time but is unit-tested here on numpy arrays (no torch in this build
environment — see pipelineInc.md §10). Everything else operates on decoded
pixel images and is numpy/PIL-native, since pixel metrics only ever run
after a VAE decode.
"""

import math
from typing import Optional, Tuple

import numpy as np


def _to_float(x):
    if hasattr(x, "float"):
        return x.float()
    return np.asarray(x, dtype=np.float64)


def _scalar(x) -> float:
    return float(x.item()) if hasattr(x, "item") else float(x)


def latent_mse(a, b) -> float:
    """Mean squared error between two same-shape tensors. Works for both
    torch.Tensor (real runs) and numpy.ndarray (unit tests) via duck typing:
    `-`, `*`, `.mean()`, `.item()` all exist on both."""
    diff = _to_float(a) - _to_float(b)
    return _scalar((diff * diff).mean())


def masked_latent_mse(edit_lat, ref_lat, token_mask) -> float:
    """latent_mse restricted to tokens where token_mask is True.

    edit_lat, ref_lat : (..., n_tokens, C) — mask applies along the
                        second-to-last axis (the token axis for FLUX's
                        packed latent layout).
    token_mask        : 1-D boolean, length n_tokens.
    """
    mask = np.asarray(token_mask, dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    e = _to_float(edit_lat)
    r = _to_float(ref_lat)
    # numpy path (unit tests): boolean-index directly.
    if isinstance(e, np.ndarray):
        e_sel = e[..., mask, :]
        r_sel = r[..., mask, :]
        return latent_mse(e_sel, r_sel)
    # torch path (real runs): boolean index via a device-matched tensor.
    import torch  # local import: only reached when `e` is already a torch.Tensor

    mask_t = torch.as_tensor(mask, device=e.device)
    e_sel = e.index_select(-2, mask_t.nonzero(as_tuple=True)[0])
    r_sel = r.index_select(-2, mask_t.nonzero(as_tuple=True)[0])
    return latent_mse(e_sel, r_sel)


def image_psnr(a, b, data_range: float = 255.0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def region_psnr(a, b, pixel_mask: np.ndarray, data_range: float = 255.0) -> float:
    """PSNR over the pixels where pixel_mask is True only. With an
    all-True mask this must equal image_psnr(a, b) exactly — asserted in
    test_cpu.py, since that's the cheapest correctness check for the
    masking arithmetic itself.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.asarray(pixel_mask, dtype=bool)
    if m.ndim == 2 and a.ndim == 3:
        m = m[..., None]
        m = np.broadcast_to(m, a.shape)
    elif m.shape != a.shape:
        m = np.broadcast_to(m.reshape(a.shape[:2] + (1,) * (a.ndim - 2)), a.shape)
    if not m.any():
        return float("nan")
    diff = (a - b)[m]
    mse = float(np.mean(diff ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def preservation_gap(background_psnr: float, whole_image_psnr: float) -> float:
    """The single most important derived number (pipelineInc.md §6).
    Positive = frozen region held while the target changed (PASS).
    <= 0 = mask mapping wrong or injection mistuned for this verb."""
    if math.isinf(background_psnr) and math.isinf(whole_image_psnr):
        return 0.0
    if math.isinf(background_psnr):
        return float("inf")
    if math.isinf(whole_image_psnr):
        return float("-inf")
    return background_psnr - whole_image_psnr


def lpips_dist(a, b) -> Optional[float]:
    """Optional perceptual distance. Returns None (never raises) if `lpips`/
    torch aren't installed, OR if the model call itself fails for any reason
    (e.g. an input too small for AlexNet's pooling stack) — pipelineInc.md
    §9's "never let a missing optional dep crash a run" extends to "never
    let an optional metric crash a run," full stop. Real canvas/edit images
    are always full resolution, so this only matters for tiny inputs."""
    try:
        import torch
        import lpips as _lpips
    except ImportError:
        return None
    try:
        try:
            _model = lpips_dist._model
        except AttributeError:
            _model = _lpips.LPIPS(net="alex")
            lpips_dist._model = _model
        a_t = _to_lpips_tensor(a)
        b_t = _to_lpips_tensor(b)
        with torch.no_grad():
            return float(_model(a_t, b_t).item())
    except Exception as exc:  # noqa: BLE001 — see docstring
        print(f"[lpips] skipped ({exc!r})")
        return None


def _to_lpips_tensor(img):
    import torch

    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def attention_leakage(attn_weights, target_mask: np.ndarray, background_mask: np.ndarray) -> float:
    """Mean attention mass from TARGET (editable) tokens onto BACKGROUND
    (frozen) tokens, post-softmax. Rising leakage across steps is an early
    warning of suppression (the edit is being pulled back toward frozen
    content) — pipelineInc.md §6, Stage 4.

    attn_weights : (heads, n_query, n_key) post-softmax attention for one
                   single-stream block, restricted to the generation-token
                   slice on both axes (drop text/reference columns first).
    Runtime-only (needs real attention weights from the model); included
    here for completeness of the metrics API but not exercised by
    test_cpu.py since there is no torch/model in this build environment.
    """
    t = np.asarray(target_mask, dtype=bool)
    b = np.asarray(background_mask, dtype=bool)
    if t.sum() == 0 or b.sum() == 0:
        return float("nan")
    w = np.asarray(attn_weights)
    sub = w[:, t, :][:, :, b]
    return float(sub.mean())
