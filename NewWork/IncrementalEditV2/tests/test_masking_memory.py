import numpy as np
import pytest

from flux_seqedit.masking import (
    normalize01, otsu_threshold, otsu_gate, refine_mask, soft_prior_fallback_mask,
)
from flux_seqedit.memory import LayerMemory


# ---- Otsu gate --------------------------------------------------------------

def test_otsu_fires_on_clear_blob():
    """A clean bimodal map (localized object) should fire the gate."""
    m = np.zeros((64, 64))
    m[20:40, 25:45] = 5.0            # bright blob on dark background
    fired, mask, thr, var = otsu_gate(m, confidence=0.02)
    assert fired
    assert mask is not None
    # recovered mask overlaps the true blob strongly
    truth = np.zeros((64, 64), bool); truth[20:40, 25:45] = True
    inter = (mask & truth).sum() / truth.sum()
    assert inter > 0.9


def test_otsu_gate_blocks_unimodal():
    """A near-uniform noisy map (object not localized) should NOT fire."""
    rng = np.random.default_rng(0)
    m = rng.normal(0.5, 0.01, size=(64, 64))   # tiny variance, unimodal
    fired, mask, thr, var = otsu_gate(m, confidence=0.05)
    assert not fired
    assert mask is None


def test_otsu_variance_higher_for_separated():
    """Confidence signal ordering: separated > diffuse."""
    sep = np.zeros((32, 32)); sep[:16] = 1.0
    _, var_sep = otsu_threshold(normalize01(sep))
    rng = np.random.default_rng(1)
    diff = normalize01(rng.normal(0.5, 0.02, size=(32, 32)))
    _, var_diff = otsu_threshold(diff)
    assert var_sep > var_diff


def test_normalize_constant_is_zeros():
    np.testing.assert_array_equal(normalize01(np.full((4, 4), 3.0)), np.zeros((4, 4)))


def test_refine_closes_and_dilates():
    m = np.zeros((32, 32), bool)
    m[10:20, 10:20] = True
    m[15, 15] = False               # small hole
    out = refine_mask(m, dilate_iter=1)
    assert out[15, 15]              # hole closed / dilated over
    assert out.sum() >= m.sum()


def test_soft_prior_fallback():
    prior = np.zeros((16, 16)); prior[4:8, 4:8] = 1.0
    fb = soft_prior_fallback_mask(prior, thr=0.5)
    assert fb[5, 5] and not fb[0, 0]


# ---- memory + generation-order occlusion -----------------------------------

def make_rect(ph, pw, r0, r1, c0, c1):
    m = np.zeros((ph, pw), bool)
    m[r0:r1, c0:c1] = True
    return m


def test_newest_on_top_two_layers():
    mem = LayerMemory(16, 16)
    mem.set_background("bg")
    a = mem.add_layer("jeep", make_rect(16, 16, 4, 12, 2, 10))
    b = mem.add_layer("dog", make_rect(16, 16, 6, 14, 6, 14))  # overlaps jeep

    vis_jeep = mem.visible_mask(a)
    vis_dog = mem.visible_mask(b)
    # dog (newest) keeps its full extent
    np.testing.assert_array_equal(vis_dog, mem.layers[b].raw_mask)
    # jeep loses the overlap region
    overlap = mem.layers[a].raw_mask & mem.layers[b].raw_mask
    assert not (vis_jeep & overlap).any()
    assert (vis_jeep & ~overlap & mem.layers[a].raw_mask).any()


def test_three_layer_cascade_reocclusion():
    """The correctness case: layer 3 must re-occlude layer 1, not only layer 2.
    A truncated cascade (subtracting only the immediately-next layer) fails."""
    ph = pw = 20
    mem = LayerMemory(ph, pw)
    mem.set_background("bg")
    j = mem.add_layer("jeep", make_rect(ph, pw, 2, 18, 2, 18))    # big
    m = mem.add_layer("man",  make_rect(ph, pw, 4, 10, 4, 10))    # on jeep
    d = mem.add_layer("dog",  make_rect(ph, pw, 3, 8,  3, 16))    # overlaps BOTH

    dog_raw = mem.layers[d].raw_mask
    # jeep must lose every pixel the dog covers, even though 'man' sits between
    vis_jeep = mem.visible_mask(j)
    assert not (vis_jeep & dog_raw).any(), "dog failed to re-occlude jeep"
    # man must also lose the dog overlap
    vis_man = mem.visible_mask(m)
    assert not (vis_man & dog_raw).any(), "dog failed to re-occlude man"
    # dog keeps everything
    np.testing.assert_array_equal(mem.visible_mask(d), dog_raw)


def test_raw_mask_preserved_after_later_edits():
    """Storing raw (not visible) masks: an early layer's RAW extent is intact
    even after later edits occlude it, so future recomputation stays correct."""
    ph = pw = 12
    mem = LayerMemory(ph, pw)
    mem.set_background("bg")
    a = mem.add_layer("a", make_rect(ph, pw, 0, 8, 0, 8))
    raw_before = mem.layers[a].raw_mask.copy()
    mem.add_layer("b", make_rect(ph, pw, 4, 12, 4, 12))   # occludes part of a
    # raw extent of 'a' unchanged
    np.testing.assert_array_equal(mem.layers[a].raw_mask, raw_before)
    # but visible shrank
    assert mem.visible_mask(a).sum() < raw_before.sum()


def test_occupied_union_excludes_background():
    ph = pw = 10
    mem = LayerMemory(ph, pw)
    mem.set_background("bg")
    mem.add_layer("x", make_rect(ph, pw, 0, 5, 0, 5))
    occ = mem.occupied_union()
    assert occ.sum() == 25            # only the object, not the full bg frame
    assert not occ[9, 9]
