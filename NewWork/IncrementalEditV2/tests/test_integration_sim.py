"""
End-to-end orchestration test WITHOUT a GPU.

We can't run FLUX here, but we can prove the control logic that sits around it is
correct by driving the exact masking + memory + occlusion path with synthetic
per-step attention-norm maps. This simulates what the processor would emit.

Scenario: a 3-edit session on a 32x32 patch grid.
  edit 1 "a jeep"   -> localizes early, mask locks
  edit 2 "a dog"    -> localizes, overlaps jeep, must occlude it (newest on top)
  edit 3 "fog"      -> never localizes (diffuse) -> hard fallback to free-space prior
"""

import numpy as np
from flux_seqedit.masking import otsu_gate, refine_mask, soft_prior_fallback_mask, normalize01
from flux_seqedit.memory import LayerMemory
from flux_seqedit.torch_adapters import phase_for_step


PH = PW = 32
NUM_STEPS = 28
EARLY_FRAC, LATE_FRAC = 0.25, 0.70
CONF = 0.02
HARD_CUTOFF_FRAC = 0.40


def blob_map(cx, cy, r, strength, noise=0.01, seed=0):
    """Synthetic attention-norm map: a bright gaussian blob + faint noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:PH, 0:PW]
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))
    return strength * g + rng.normal(0, noise, size=(PH, PW))


def diffuse_map(seed=0):
    """No coherent blob — Otsu should never fire."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.5, 0.02, size=(PH, PW))


def run_edit(memory, norm_map_fn, localizes_by=None):
    """
    Simulate one edit's denoising loop. norm_map_fn(step)->2D map.
    Returns the locked mask.
    """
    occupied = memory.occupied_union()
    free = (~occupied).astype(np.float64)
    if not free.any():
        free = np.ones_like(free)
    prior_soft = normalize01(free)

    locked = None
    for i in range(NUM_STEPS):
        phase = phase_for_step(i, NUM_STEPS, EARLY_FRAC, LATE_FRAC)
        if locked is None and phase in ("early", "mid"):
            pmap = norm_map_fn(i)
            fired, mask, thr, var = otsu_gate(pmap, CONF)
            if fired:
                locked = refine_mask(mask)
        if locked is None and i >= HARD_CUTOFF_FRAC * NUM_STEPS:
            locked = soft_prior_fallback_mask(prior_soft)
    return locked


def test_full_three_edit_session():
    mem = LayerMemory(PH, PW)
    mem.set_background("empty room")

    # edit 1: jeep, localizes clearly in the upper-left
    def jeep_map(step):
        # faint early, sharpens after a few steps
        strength = 0.2 if step < 3 else 5.0
        return blob_map(cx=10, cy=10, r=4, strength=strength, seed=step)
    m_jeep = run_edit(mem, jeep_map)
    assert m_jeep is not None and m_jeep.sum() > 0
    j = mem.add_layer("a jeep", m_jeep)

    # edit 2: dog, localizes overlapping jeep region
    def dog_map(step):
        strength = 0.2 if step < 3 else 5.0
        return blob_map(cx=13, cy=13, r=4, strength=strength, seed=100 + step)
    m_dog = run_edit(mem, dog_map)
    assert m_dog is not None and m_dog.sum() > 0
    d = mem.add_layer("a dog", m_dog)

    # newest-on-top: dog keeps full extent, jeep loses the overlap
    vis_jeep = mem.visible_mask(j)
    vis_dog = mem.visible_mask(d)
    np.testing.assert_array_equal(vis_dog, mem.layers[d].raw_mask)
    overlap = mem.layers[j].raw_mask & mem.layers[d].raw_mask
    assert overlap.any(), "test setup: expected jeep/dog overlap"
    assert not (vis_jeep & overlap).any(), "dog must occlude jeep"

    # edit 3: fog, never localizes -> hard fallback to free-space prior
    m_fog = run_edit(mem, lambda step: diffuse_map(step))
    assert m_fog is not None, "hard fallback must produce a mask"
    # fallback mask should sit in free space (not fully inside occupied region)
    occ = mem.occupied_union()
    free_frac = (m_fog & ~occ).sum() / max(1, m_fog.sum())
    assert free_frac > 0.5, "fallback mask should prefer free space"

    mem.add_layer("fog", m_fog)
    assert len(mem) == 4  # bg + 3 edits


def test_otsu_gate_blocks_pure_noise_confidence():
    """The real invariant: on a pure-noise map (no blob at all) the gate must
    NOT fire; once a coherent blob is present it must. Confidence separates the
    two regimes. (Timing of the exact fire-step is a tunable, not an invariant.)"""
    # pure noise, no object -> must not fire
    for step in range(5):
        fired, *_ = otsu_gate(diffuse_map(step), CONF)
        assert not fired, "gate fired on pure noise (no object present)"
    # coherent blob -> must fire
    fired, mask, thr, var = otsu_gate(blob_map(16, 16, 4, 5.0, seed=0), CONF)
    assert fired and mask is not None


def test_hard_fallback_only_when_never_localizes():
    """If a blob DOES localize, we should get the attention mask, not the
    fallback (i.e. fallback isn't masking real detections)."""
    mem = LayerMemory(PH, PW)
    mem.set_background("bg")
    def good(step):
        return blob_map(20, 8, 4, 5.0, seed=step)
    m = run_edit(mem, good)
    # the locked mask should match the blob location, not a full free-space prior
    assert m.sum() < PH * PW * 0.5, "should be a localized mask, not whole-frame fallback"
    # blob was placed at cx=20 (column), cy=8 (row); nonzero gives (rows, cols)
    ys, xs = np.nonzero(m)
    assert 16 < xs.mean() < 24, f"col mean {xs.mean()} not near cx=20"
    assert 4 < ys.mean() < 12, f"row mean {ys.mean()} not near cy=8"
