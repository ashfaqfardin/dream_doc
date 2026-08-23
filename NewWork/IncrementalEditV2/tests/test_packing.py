"""
Unit tests gating the packing map (spec Step 1). Both must pass before any
model code is written. T1 and T2 fail on *different* bugs:
  T1 (round-trip identity) catches transpose / rotation.
  T2 (localization) catches packing-order (row-major vs col-major) mismatch,
     which T1's symmetry can hide.
"""

import numpy as np
import pytest

from flux_seqedit.packing import (
    latent_dims_for_image,
    patch_grid,
    mask_to_patch_mask,
    patch_mask_to_token_indices,
    image_mask_to_token_indices,
    token_scores_to_patch_map,
    token_scores_to_image_map,
)


# ---- geometry sanity --------------------------------------------------------

def test_latent_and_patch_dims_1024():
    lh, lw = latent_dims_for_image(1024, 1024)
    assert (lh, lw) == (128, 128)
    ph, pw = patch_grid(lh, lw)
    assert (ph, pw) == (64, 64)
    assert ph * pw == 4096  # FLUX token count for 1024^2


def test_nonsquare_dims():
    lh, lw = latent_dims_for_image(512, 1024)
    assert (lh, lw) == (64, 128)
    ph, pw = patch_grid(lh, lw)
    assert (ph, pw) == (32, 64)


# ---- T1: round-trip identity ------------------------------------------------

@pytest.mark.parametrize("H,W", [(1024, 1024), (512, 1024), (768, 512)])
def test_T1_roundtrip_quadrant(H, W):
    """A filled quadrant maps image->tokens->image and returns the same patch
    region. A transpose bug makes the recovered quadrant land in the wrong
    corner and fails here."""
    lh, lw = latent_dims_for_image(H, W)
    ph, pw = patch_grid(lh, lw)

    # top-left quadrant filled at image resolution
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[: H // 2, : W // 2] = 1

    idx = image_mask_to_token_indices(mask, lh, lw)

    # reconstruct a token vector and unpack to patch grid
    scores = np.zeros(ph * pw)
    scores[idx] = 1.0
    patch = token_scores_to_patch_map(scores, ph, pw)

    # expected: top-left quadrant of the *patch* grid
    expected = np.zeros((ph, pw))
    expected[: ph // 2, : pw // 2] = 1.0
    np.testing.assert_array_equal(patch, expected)


def test_T1_roundtrip_offcenter_blob():
    """Off-center asymmetric blob — asymmetry is what actually catches a
    transpose (a centered/symmetric pattern would pass even when transposed)."""
    H = W = 1024
    lh, lw = latent_dims_for_image(H, W)
    ph, pw = patch_grid(lh, lw)

    mask = np.zeros((H, W), dtype=np.uint8)
    # a blob near the top-right, clearly not symmetric under transpose
    mask[64:256, 640:960] = 1

    scores = np.zeros(ph * pw)
    scores[image_mask_to_token_indices(mask, lh, lw)] = 1.0
    patch = token_scores_to_patch_map(scores, ph, pw)

    # same region expressed on the patch grid (image->patch scale = H/ph = 16)
    sy = H // ph
    sx = W // pw
    expected = np.zeros((ph, pw))
    expected[64 // sy: 256 // sy, 640 // sx: 960 // sx] = 1.0
    np.testing.assert_array_equal(patch, expected)


# ---- T2: localization -------------------------------------------------------

def test_T2_localization_single_patch():
    """Light up ONE known image token; assert it unpacks to the correct 2D
    patch cell. This is the test that catches row-major vs col-major mixups:
    token t must map to (t // pw, t % pw)."""
    H = W = 1024
    lh, lw = latent_dims_for_image(H, W)
    ph, pw = patch_grid(lh, lw)

    # pick patch cell (row=7, col=13); its token index in FLUX order:
    r, c = 7, 13
    t = r * pw + c

    scores = np.zeros(ph * pw)
    scores[t] = 1.0
    patch = token_scores_to_patch_map(scores, ph, pw)

    assert patch[r, c] == 1.0
    assert patch.sum() == 1.0  # nothing else lit


def test_T2_localization_matches_forward_index():
    """Forward direction: a mask over patch cell (r,c) must yield exactly the
    token index r*pw+c. Confirms forward and inverse agree on ordering."""
    H = W = 1024
    lh, lw = latent_dims_for_image(H, W)
    ph, pw = patch_grid(lh, lw)
    sy, sx = H // ph, W // pw

    r, c = 20, 3
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[r * sy:(r + 1) * sy, c * sx:(c + 1) * sx] = 1

    idx = image_mask_to_token_indices(mask, lh, lw)
    assert idx.tolist() == [r * pw + c]


def test_T2_image_upsample_roundtrip():
    """token_scores_to_image_map should place a single lit token back at the
    correct image-resolution block."""
    H = W = 1024
    lh, lw = latent_dims_for_image(H, W)
    ph, pw = patch_grid(lh, lw)
    r, c = 5, 9
    t = r * pw + c
    scores = np.zeros(ph * pw)
    scores[t] = 1.0
    img = token_scores_to_image_map(scores, lh, lw, H, W)
    sy, sx = H // ph, W // pw
    block = img[r * sy:(r + 1) * sy, c * sx:(c + 1) * sx]
    assert block.sum() == sy * sx           # whole block lit
    assert img.sum() == sy * sx             # nothing else lit


# ---- coverage threshold behavior -------------------------------------------

def test_coverage_threshold():
    """A patch half-covered should be on at coverage<=0.5, off above."""
    H = W = 1024
    lh, lw = latent_dims_for_image(H, W)
    ph, pw = patch_grid(lh, lw)
    sy, sx = H // ph, W // pw

    mask = np.zeros((H, W), dtype=np.uint8)
    # cover exactly the left half of patch cell (0,0)
    mask[0:sy, 0:sx // 2] = 1

    on_low = mask_to_patch_mask(mask, lh, lw, coverage=0.5)
    on_high = mask_to_patch_mask(mask, lh, lw, coverage=0.75)
    assert on_low[0, 0] == True
    assert on_high[0, 0] == False
