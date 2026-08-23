"""
Thin torch adapters. The numpy packing/masking logic is the source of truth and
is unit-tested; these wrappers just move between numpy and torch at the pipeline
boundary. Import torch lazily so the numpy core stays importable without it.
"""
from __future__ import annotations
import numpy as np


def occupied_mask_to_token_tensor(patch_mask_2d, device):
    """bool (ph,pw) patch mask -> LongTensor of image-token indices (FLUX order)."""
    import torch
    from .packing import patch_mask_to_token_indices
    idx = patch_mask_to_token_indices(np.asarray(patch_mask_2d))
    return torch.as_tensor(idx, dtype=torch.long, device=device)


def norm_vector_to_patch_map(norm_vec, ph, pw):
    """torch (n_img,) -> numpy (ph,pw) via row-major unpack."""
    from .packing import token_scores_to_patch_map
    v = norm_vec.detach().float().cpu().numpy()
    return token_scores_to_patch_map(v, ph, pw)


def phase_for_step(step_index, total_steps, early_frac=0.25, late_frac=0.7):
    """
    Map a step to a phase. Rectified-flow schedule: step 0 is highest noise.
    early:  step_index < early_frac*total  -> placement prior active, harvest attn
    mid:    between                          -> attn reliable, lock mask
    late:   step_index >= late_frac*total    -> relax mask (soft floor)
    Boundaries are hyperparameters (tune-not-assume, per spec).
    """
    frac = step_index / max(1, total_steps - 1)
    if frac < early_frac:
        return "early"
    if frac >= late_frac:
        return "late"
    return "mid"
