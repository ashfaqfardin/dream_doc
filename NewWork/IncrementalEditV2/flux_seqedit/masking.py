"""
Step 4 core — Otsu gate with confidence, and rough-mask refinement.

All pure numpy/scipy so it tests without torch. The attention processor produces
a per-token L2 norm vector; packing.token_scores_to_patch_map turns that into a
2D map; this module decides whether the object has localized (Otsu inter-class
variance gate) and, if so, thresholds + refines it into a rough binary mask.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation


def normalize01(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1]. Constant input -> all zeros (avoids div-by-zero)."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def otsu_threshold(x01: np.ndarray, bins: int = 256):
    """
    Otsu's method on a [0,1]-normalized array.

    Returns (threshold, inter_class_variance). The inter-class variance is the
    confidence signal: a well-separated bimodal histogram (localized object)
    gives high variance; a unimodal blob (not localized yet) gives low variance.
    """
    x = np.asarray(x01, dtype=np.float64).ravel()
    hist, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5, 0.0
    p = hist / total
    centers = (edges[:-1] + edges[1:]) / 2.0

    omega = np.cumsum(p)                 # class prob below threshold
    mu = np.cumsum(p * centers)          # class mean below threshold
    mu_t = mu[-1]                        # global mean

    denom = omega * (1.0 - omega)
    # inter-class variance across all candidate splits
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2 = np.nan_to_num(sigma_b2, nan=0.0, posinf=0.0, neginf=0.0)

    k = int(np.argmax(sigma_b2))
    return float(centers[k]), float(sigma_b2[k])


def otsu_gate(score_map_2d: np.ndarray, confidence: float):
    """
    Decide whether to lock a mask this step.

    Returns (fired: bool, mask_bool_or_None, thr, variance).
    fired is True only if the normalized inter-class variance clears `confidence`.
    Normalization happens BEFORE Otsu (variance is scale-dependent), per spec.
    """
    norm = normalize01(score_map_2d)
    thr, var = otsu_threshold(norm)
    if var < confidence:
        return False, None, thr, var
    mask = norm >= thr
    return True, mask, thr, var


def refine_mask(mask_bool: np.ndarray, sigma: float = 0.5,
                soft_hard_beta: float = 0.7, hard_thr: float = 0.5,
                dilate_iter: int = 0) -> np.ndarray:
    """
    Rough-mask cleanup: Gaussian smooth -> soft/hard blend -> optional dilation.

    Defaults are deliberately conservative (sigma=0.5, no dilation): the Otsu
    mask already covers the full object; aggressive expansion eats into background
    and degrades BCG. Callers can pass dilate_iter=1 to restore one ring of dilation
    if the object has hard-to-reach speckle interiors.
    """
    m = np.asarray(mask_bool, dtype=np.float64)
    soft = gaussian_filter(m, sigma=sigma)
    # Clamp rather than normalize: keeps the threshold anchored to 0.5 of the
    # blur output rather than the global [min,max] range. This prevents blurred
    # edge-spillover from the background being pulled above the threshold.
    soft = np.clip(soft, 0.0, 1.0)
    blended = soft_hard_beta * soft + (1.0 - soft_hard_beta) * (soft > hard_thr)
    out = blended > hard_thr
    if dilate_iter > 0:
        out = binary_dilation(out, iterations=dilate_iter)
    return out


def soft_prior_fallback_mask(prior_soft_2d: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    Hard-fallback path (spec Step 4): if Otsu never fires by the cutoff, threshold
    the soft mass from the memory-repulsion prior instead so the schedule doesn't
    stall. `prior_soft_2d` is the (normalized) free-space prior for the new object.
    """
    return normalize01(prior_soft_2d) >= thr
