"""
Step 1 — Mask <-> packed-token mapping.

The single highest-risk piece. FLUX does not operate on an H x W grid: the VAE
latent (C, lh, lw) is split into 2x2 patches and packed into a 1D token sequence
of length (lh/2 * lw/2), each token carrying C*2*2 channels.

We must map a 2D image-resolution mask onto that exact token ordering, or a
penalty/mask lands on the wrong tokens.

This module mirrors diffusers' FluxPipeline._pack_latents / _unpack_latents
ordering. It is written in numpy so it can be unit-tested with no torch/GPU.
The ordering is identical to the torch version; a thin torch adapter lives in
processor.py for use inside the attention hook.

Reference (diffusers, FluxPipeline._pack_latents):
    latents = latents.view(b, c, lh//2, 2, lw//2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (lh//2)*(lw//2), c*4)

So token index t = row_patch * (lw//2) + col_patch, iterating patch-rows then
patch-cols (row-major over the patch grid). That row-major patch ordering is the
only thing the mask logic needs; the channel packing (c,2,2) is irrelevant to a
spatial mask because every token corresponds to exactly one 2x2 spatial patch.
"""

from __future__ import annotations

import numpy as np


def latent_dims_for_image(height: int, width: int, vae_scale: int = 8) -> tuple[int, int]:
    """Latent grid size (lh, lw) for a given image size. FLUX VAE downsamples by 8."""
    if height % vae_scale or width % vae_scale:
        raise ValueError(f"image {height}x{width} not divisible by vae_scale {vae_scale}")
    return height // vae_scale, width // vae_scale


def patch_grid(lh: int, lw: int, patch: int = 2) -> tuple[int, int]:
    """Packed patch-grid size (ph, pw). token_count = ph * pw."""
    if lh % patch or lw % patch:
        raise ValueError(f"latent {lh}x{lw} not divisible by patch {patch}")
    return lh // patch, lw // patch


def mask_to_patch_mask(mask_hw: np.ndarray, lh: int, lw: int, patch: int = 2,
                       coverage: float = 0.5) -> np.ndarray:
    """
    Downsample a 2D image-resolution mask to a boolean patch-grid mask of shape
    (ph, pw). A patch is 'on' if the fraction of its area covered by the mask is
    >= coverage.

    mask_hw: 2D array (H, W), any nonzero = foreground.
    returns: bool array (ph, pw).
    """
    mask = np.asarray(mask_hw)
    if mask.ndim != 2:
        raise ValueError("mask_hw must be 2D")
    H, W = mask.shape
    ph, pw = patch_grid(lh, lw, patch)

    # First bring the image mask down to the latent grid (lh, lw) by block-mean,
    # then down to the patch grid (ph, pw) by 2x2 block-mean. We fold both into a
    # single block-mean from (H, W) -> (ph, pw), which is what actually matters
    # spatially. H must be divisible by ph and W by pw.
    if H % ph or W % pw:
        raise ValueError(f"image {H}x{W} not divisible by patch grid {ph}x{pw}")
    by, bx = H // ph, W // pw
    binm = (mask != 0).astype(np.float64)
    # block-reduce mean over (by, bx) blocks
    frac = binm.reshape(ph, by, pw, bx).mean(axis=(1, 3))
    return frac >= coverage


def patch_mask_to_token_indices(patch_mask: np.ndarray) -> np.ndarray:
    """
    Flatten a (ph, pw) patch mask to token indices using FLUX's row-major patch
    order: token t = row * pw + col.
    returns: int array of token indices that are 'on'.
    """
    pm = np.asarray(patch_mask, dtype=bool)
    if pm.ndim != 2:
        raise ValueError("patch_mask must be 2D (ph, pw)")
    flat = pm.reshape(-1)              # row-major == row*pw+col  ✓ matches FLUX
    return np.nonzero(flat)[0]


def image_mask_to_token_indices(mask_hw: np.ndarray, lh: int, lw: int,
                                patch: int = 2, coverage: float = 0.5) -> np.ndarray:
    """Convenience: 2D image mask -> 1D token indices (FLUX packed order)."""
    pm = mask_to_patch_mask(mask_hw, lh, lw, patch=patch, coverage=coverage)
    return patch_mask_to_token_indices(pm)


def token_scores_to_patch_map(scores: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """
    Inverse: scatter a per-image-token vector (length ph*pw) back onto the 2D
    patch grid (ph, pw). Uses the same row-major order.
    """
    s = np.asarray(scores)
    if s.ndim != 1 or s.shape[0] != ph * pw:
        raise ValueError(f"scores must be 1D length {ph*pw}, got {s.shape}")
    return s.reshape(ph, pw)           # row-major inverse of the above  ✓


def token_scores_to_image_map(scores: np.ndarray, lh: int, lw: int,
                              out_h: int, out_w: int, patch: int = 2) -> np.ndarray:
    """
    Scatter per-token scores to patch grid, then nearest-upsample to (out_h,
    out_w) for thresholding/visualization at image resolution.
    """
    ph, pw = patch_grid(lh, lw, patch)
    pmap = token_scores_to_patch_map(scores, ph, pw)
    ry, rx = out_h // ph, out_w // pw
    if ry * ph != out_h or rx * pw != out_w:
        raise ValueError("out size not an integer multiple of patch grid")
    return np.kron(pmap, np.ones((ry, rx), dtype=pmap.dtype))
