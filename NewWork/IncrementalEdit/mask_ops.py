"""
Token-mask math (pipelineInc.md §2, §3, §10.2) — pure numpy, no torch.

Convention used everywhere in this package: a mask is a 1-D boolean numpy
array of length h_lat * w_lat (row-major over the latent token grid), where
True = "this token belongs to the target/object region." This is the
opposite convention from the original draft (which used True = frozen) —
flipped here because objects, not backgrounds, are the named entities that
get stored in the manifest; "frozen" is always derivable as the complement
of whichever mask(s) are in play for a given call (see zone_masks below).

Semantic derivation (DAAM/ConceptAttention saliency, SAM2) requires the
actual model and lives in kontext_injection.py — this module only holds the
mask arithmetic, which is what's actually unit-testable without a GPU.
"""

from typing import Tuple

import numpy as np
from PIL import Image


def rect_mask(h_lat: int, w_lat: int, region: str) -> np.ndarray:
    """Coarse rectangular mask, kept as a fallback/override path (e.g. for
    `--mask`-free smoke testing) — NOT the primary mask-derivation mechanism
    for real objects, see pipelineInc.md §2. `region` names the TARGET area.
    """
    mask = np.zeros((h_lat, w_lat), dtype=bool)
    if region == "none":
        pass
    elif region == "all":
        mask[:, :] = True
    elif region == "left":
        mask[:, : w_lat // 2] = True
    elif region == "right":
        mask[:, w_lat // 2 :] = True
    elif region == "top":
        mask[: h_lat // 2, :] = True
    elif region == "bottom":
        mask[h_lat // 2 :, :] = True
    elif region == "center":
        h0, h1 = h_lat // 4, h_lat - h_lat // 4
        w0, w1 = w_lat // 4, w_lat - w_lat // 4
        mask[h0:h1, w0:w1] = True
    else:
        raise ValueError(
            f"Unknown region '{region}' — expected one of "
            f"none/all/left/right/top/bottom/center"
        )
    return mask.reshape(-1)


def topk_mask(saliency: np.ndarray, top_k_frac: float) -> np.ndarray:
    """Threshold a per-token saliency score (e.g. DAAM cross-attention, or
    an object-addition placement heatmap) into a binary token mask covering
    the top `top_k_frac` fraction of tokens by score."""
    saliency = np.asarray(saliency).reshape(-1)
    n = saliency.shape[0]
    k = max(1, int(round(top_k_frac * n)))
    if k >= n:
        return np.ones(n, dtype=bool)
    threshold = np.partition(saliency, n - k)[n - k]
    return saliency >= threshold


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """AND. Used to scope a part mask (e.g. 'tire') inside its parent
    object's mask (e.g. 'car_1'), preventing it from matching a visually
    similar thing elsewhere in the scene — pipelineInc.md §1, §2."""
    return np.logical_and(a, b)


def union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.logical_or(a, b)


def invert(a: np.ndarray) -> np.ndarray:
    return np.logical_not(a)


def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a AND NOT b — e.g. shell = parent_mask - target_mask (pipelineInc.md §2)."""
    return np.logical_and(a, np.logical_not(b))


def zone_masks(
    background_complement: np.ndarray,
    target: np.ndarray,
    parent: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the token grid into the three injection zones from pipelineInc.md §2.

    background_complement : mask of everything NOT already excluded from
                             consideration by a parent scope (usually
                             np.ones(n_tokens, dtype=bool) — the whole canvas
                             — except when nested under a parent object).
    target                : the region actually changing this call.
    parent                : parent object's mask, only for part_edit; None
                             for a top-level edit (attribute/replace/remove/add).

    Returns (background, shell, target) as three disjoint boolean masks that
    together cover background_complement. `shell` is empty unless `parent`
    is given (that's what makes part_edit different from attribute/replace).
    """
    target = np.asarray(target, dtype=bool)
    if parent is None:
        background = np.logical_and(background_complement, np.logical_not(target))
        shell = np.zeros_like(target, dtype=bool)
        return background, shell, target

    parent = np.asarray(parent, dtype=bool)
    if not np.array_equal(np.logical_and(target, parent), target):
        raise ValueError(
            "part_edit target mask must be a subset of the parent object's "
            "mask — derive it via intersect(saliency_mask, parent_mask), not "
            "independently."
        )
    background = np.logical_and(background_complement, np.logical_not(parent))
    shell = subtract(parent, target)
    return background, shell, target


def mask_to_image(mask: np.ndarray, h_lat: int, w_lat: int) -> Image.Image:
    """Token mask -> single-channel PNG (white = target), for saving to
    <project_dir>/masks/<object>.png."""
    arr = np.asarray(mask, dtype=bool).reshape(h_lat, w_lat)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def image_to_mask(img: Image.Image, h_lat: int, w_lat: int, threshold: int = 127) -> np.ndarray:
    """Inverse of mask_to_image. Also the entry point for a user-supplied
    `--mask` PNG at arbitrary resolution — resizes with nearest-neighbor to
    the token grid so the mask stays binary, no soft edges introduced."""
    resized = img.convert("L").resize((w_lat, h_lat), Image.NEAREST)
    arr = np.array(resized)
    return (arr > threshold).reshape(-1)


def upsample_mask(token_mask_2d: np.ndarray, height: int, width: int) -> np.ndarray:
    """Token grid (h_lat, w_lat) -> pixel grid (height, width) via np.kron,
    for region_psnr / visual overlays. height/width must be exact multiples
    of the token grid (true for FLUX's fixed 16x downsample: h_lat = H//16)."""
    h_lat, w_lat = token_mask_2d.shape
    if height % h_lat != 0 or width % w_lat != 0:
        raise ValueError(
            f"pixel size ({height},{width}) is not an integer multiple of "
            f"token grid ({h_lat},{w_lat})"
        )
    sy, sx = height // h_lat, width // w_lat
    return np.kron(token_mask_2d.astype(np.uint8), np.ones((sy, sx), dtype=np.uint8)).astype(bool)


def assert_token_count(mask: np.ndarray, h_lat: int, w_lat: int, what: str) -> None:
    """Fail loud on shape mismatch — pipelineInc.md §9. A silently-wrong
    mask length produces confident wrong metrics, not a crash, which is why
    this is a hard assert rather than a coercion."""
    expected = h_lat * w_lat
    actual = int(np.asarray(mask).reshape(-1).shape[0])
    if actual != expected:
        raise ValueError(
            f"{what}: mask has {actual} tokens, expected {expected} "
            f"(h_lat={h_lat} * w_lat={w_lat}). Mask↔token mapping is wrong "
            f"— do not proceed, this is the #1 cause of nonsense metrics."
        )
