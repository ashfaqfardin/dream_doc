"""
Attention hooking utilities: capture processors, injection processors,
and helpers to attach/detach them from FLUX.1-Kontext transformer blocks.

All processors mirror FluxAttnProcessor2_0 exactly so output is identical
when alpha_k = alpha_v = 0 (pure capture mode).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared projection / reshape helpers
# ---------------------------------------------------------------------------

def _project_and_shape(attn, hidden_states, encoder_hidden_states):
    """
    Compute Q, K, V (and text eq, ek, ev if encoder_hidden_states is given).
    Returns (Q, K, V, n_txt) where Q/K/V are (b, h, seq, dh) and n_txt is the
    number of text tokens prepended (0 for single-stream blocks).
    """
    b = hidden_states.shape[0]
    h = attn.heads

    q = attn.to_q(hidden_states)
    k = attn.to_k(hidden_states)
    v = attn.to_v(hidden_states)

    def shape(t):
        return t.view(b, -1, h, t.shape[-1] // h).transpose(1, 2)

    q, k, v = shape(q), shape(k), shape(v)

    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)

    if encoder_hidden_states is not None:
        eq = attn.add_q_proj(encoder_hidden_states)
        ek = attn.add_k_proj(encoder_hidden_states)
        ev = attn.add_v_proj(encoder_hidden_states)
        eq, ek, ev = shape(eq), shape(ek), shape(ev)
        if getattr(attn, "norm_added_q", None) is not None:
            eq = attn.norm_added_q(eq)
        if getattr(attn, "norm_added_k", None) is not None:
            ek = attn.norm_added_k(ek)
        Q = torch.cat([eq, q], dim=2)   # (b, h, n_txt+n_img, dh)
        K = torch.cat([ek, k], dim=2)
        V = torch.cat([ev, v], dim=2)
        n_txt = eq.shape[2]
    else:
        Q, K, V = q, k, v
        n_txt = 0

    return Q, K, V, n_txt


def _apply_rope_if_needed(Q, K, image_rotary_emb):
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        Q = apply_rotary_emb(Q, image_rotary_emb)
        K = apply_rotary_emb(K, image_rotary_emb)
    return Q, K


def _out_project(attn, out, n_txt, encoder_hidden_states):
    """Split output, apply output projections, return (img_out, txt_out | None)."""
    # out from SDPA: (b, h, seq, dh)  — unpack before transpose to get correct dims
    b, h, seq, dh = out.shape
    out = out.transpose(1, 2).reshape(b, -1, h * dh)

    if encoder_hidden_states is not None:
        txt_out = out[:, :n_txt]
        img_out = out[:, n_txt:]
        img_out = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)
        txt_out = attn.to_add_out(txt_out)
        return img_out, txt_out
    else:
        # Single-stream FluxAttention has no to_out; the block applies projection itself.
        if hasattr(attn, "to_out") and attn.to_out:
            out = attn.to_out[0](out)
            if len(attn.to_out) > 1:
                out = attn.to_out[1](out)
        return out, None


# ---------------------------------------------------------------------------
# Capture processor  (zero-impact: read-only, output unchanged)
# ---------------------------------------------------------------------------

class CaptureProcessor:
    """
    Drop-in for FluxAttnProcessor2_0 that stores K and V tensors.
    Output is bit-for-bit identical to the standard processor.
    """

    def __init__(self, store: dict, key: str):
        """
        store : dict that will receive entries store[key + '_K'] and store[key + '_V'].
        key   : unique identifier, e.g. 'double_7' or 'single_3'.
        """
        self.store = store
        self.key   = key

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        Q, K, V, n_txt = _project_and_shape(attn, hidden_states, encoder_hidden_states)
        Q, K = _apply_rope_if_needed(Q, K, image_rotary_emb)

        # Capture before attention.
        self.store[self.key + "_K"] = K.detach().cpu()
        self.store[self.key + "_V"] = V.detach().cpu()
        self.store[self.key + "_Q"] = Q.detach().cpu()  # useful for Phase 2

        out = F.scaled_dot_product_attention(Q, K, V)
        img_out, txt_out = _out_project(attn, out, n_txt, encoder_hidden_states)

        if encoder_hidden_states is not None:
            return img_out, txt_out
        return img_out


# ---------------------------------------------------------------------------
# Injection processor  (blends cached K/V with current K/V)
# ---------------------------------------------------------------------------

class InjectProcessor:
    """
    Blends cached K and/or V with the current-step K/V:
        K_blend = (1 - alpha_k) * K_current + alpha_k * K_cached
        V_blend = (1 - alpha_v) * V_current + alpha_v * V_cached

    If alpha_k = alpha_v = 0.0  → identical to CaptureProcessor (no injection).
    If alpha_k = alpha_v = 1.0  → fully replace with cached (maximum injection).
    """

    def __init__(self, store: dict, key: str, alpha_k: float = 0.5, alpha_v: float = 0.0):
        self.store   = store
        self.key     = key
        self.alpha_k = alpha_k
        self.alpha_v = alpha_v

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        Q, K, V, n_txt = _project_and_shape(attn, hidden_states, encoder_hidden_states)
        Q, K = _apply_rope_if_needed(Q, K, image_rotary_emb)

        # Inject cached K.
        k_key = self.key + "_K"
        if self.alpha_k > 0.0 and k_key in self.store:
            cached_K = self.store[k_key].to(K.device, dtype=K.dtype)
            if cached_K.shape == K.shape:
                K = (1.0 - self.alpha_k) * K + self.alpha_k * cached_K

        # Inject cached V.
        v_key = self.key + "_V"
        if self.alpha_v > 0.0 and v_key in self.store:
            cached_V = self.store[v_key].to(V.device, dtype=V.dtype)
            if cached_V.shape == V.shape:
                V = (1.0 - self.alpha_v) * V + self.alpha_v * cached_V

        out = F.scaled_dot_product_attention(Q, K, V)
        img_out, txt_out = _out_project(attn, out, n_txt, encoder_hidden_states)

        if encoder_hidden_states is not None:
            return img_out, txt_out
        return img_out


# ---------------------------------------------------------------------------
# Adaptive injection processor  (Phase 8)
# ---------------------------------------------------------------------------

class AdaptiveInjectProcessor:
    """
    Computes per-step alpha_k / alpha_v from cosine similarity between
    current K/V and cached K/V.  High similarity → low alpha (allow free edit).
    Low similarity → high alpha (preserve content).
    """

    def __init__(self, store: dict, key: str, base_alpha: float = 0.5,
                 method: str = "cosine"):
        self.store      = store
        self.key        = key
        self.base_alpha = base_alpha
        self.method     = method
        self.last_alpha_k: float = 0.0
        self.last_alpha_v: float = 0.0

    @staticmethod
    def _cosine_alpha(current: torch.Tensor, cached: torch.Tensor, base: float) -> float:
        a = current.reshape(1, -1).float()
        b = cached.reshape(1, -1).to(a.device).float()
        cos = F.cosine_similarity(a, b, dim=-1).mean().item()
        # Map [−1, 1] → [0, 2*base]: high cos → low alpha, low cos → high alpha.
        alpha = base * (1.0 - cos)
        return float(max(0.0, min(1.0, alpha)))

    @staticmethod
    def _entropy_alpha(current: torch.Tensor, base: float) -> float:
        """Use entropy of softmax(QK^T) — high entropy = diffuse, low alpha."""
        # Approximate: use variance of K norms across heads as proxy.
        norms = current.float().norm(dim=-1)  # (b, h, seq)
        var = norms.var().item()
        alpha = base * math.exp(-var)
        return float(max(0.0, min(1.0, alpha)))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        Q, K, V, n_txt = _project_and_shape(attn, hidden_states, encoder_hidden_states)
        Q, K = _apply_rope_if_needed(Q, K, image_rotary_emb)

        k_key = self.key + "_K"
        v_key = self.key + "_V"

        if k_key in self.store:
            cached_K = self.store[k_key].to(K.device, dtype=K.dtype)
            if cached_K.shape == K.shape:
                if self.method == "cosine":
                    alpha_k = self._cosine_alpha(K, cached_K, self.base_alpha)
                else:
                    alpha_k = self._entropy_alpha(K, self.base_alpha)
                self.last_alpha_k = alpha_k
                K = (1.0 - alpha_k) * K + alpha_k * cached_K

        if v_key in self.store:
            cached_V = self.store[v_key].to(V.device, dtype=V.dtype)
            if cached_V.shape == V.shape:
                alpha_v = self.last_alpha_k   # mirror K's alpha for V
                self.last_alpha_v = alpha_v
                V = (1.0 - alpha_v) * V + alpha_v * cached_V

        out = F.scaled_dot_product_attention(Q, K, V)
        img_out, txt_out = _out_project(attn, out, n_txt, encoder_hidden_states)

        if encoder_hidden_states is not None:
            return img_out, txt_out
        return img_out


# ---------------------------------------------------------------------------
# Attach / detach helpers
# ---------------------------------------------------------------------------

def attach_capture_processors(transformer, store: dict,
                              double_blocks: Optional[list[int]] = None,
                              single_blocks: Optional[list[int]] = None):
    """
    Attach CaptureProcessor to the specified double/single stream blocks.
    double_blocks=None → all 19. single_blocks=None → all 38.
    Returns the mapping {key: original_processor} for later restore.
    """
    originals = {}

    db_range = double_blocks if double_blocks is not None else range(len(transformer.transformer_blocks))
    for i in db_range:
        block = transformer.transformer_blocks[i]
        key = f"double_{i}"
        originals[key] = block.attn.processor
        block.attn.processor = CaptureProcessor(store, key)

    sb_range = single_blocks if single_blocks is not None else range(len(transformer.single_transformer_blocks))
    for i in sb_range:
        block = transformer.single_transformer_blocks[i]
        key = f"single_{i}"
        originals[key] = block.attn.processor
        block.attn.processor = CaptureProcessor(store, key)

    return originals


def attach_inject_processors(transformer, store: dict,
                             double_blocks: Optional[list[int]] = None,
                             single_blocks: Optional[list[int]] = None,
                             alpha_k: float = 0.5, alpha_v: float = 0.0):
    """Attach InjectProcessor to the specified blocks."""
    originals = {}

    db_range = double_blocks if double_blocks is not None else range(len(transformer.transformer_blocks))
    for i in db_range:
        block = transformer.transformer_blocks[i]
        key = f"double_{i}"
        originals[key] = block.attn.processor
        block.attn.processor = InjectProcessor(store, key, alpha_k=alpha_k, alpha_v=alpha_v)

    sb_range = single_blocks if single_blocks is not None else range(len(transformer.single_transformer_blocks))
    for i in sb_range:
        block = transformer.single_transformer_blocks[i]
        key = f"single_{i}"
        originals[key] = block.attn.processor
        block.attn.processor = InjectProcessor(store, key, alpha_k=alpha_k, alpha_v=alpha_v)

    return originals


def attach_adaptive_processors(transformer, store: dict,
                               double_blocks: Optional[list[int]] = None,
                               single_blocks: Optional[list[int]] = None,
                               base_alpha: float = 0.5, method: str = "cosine"):
    """Attach AdaptiveInjectProcessor to the specified blocks."""
    originals = {}
    processors = {}

    db_range = double_blocks if double_blocks is not None else range(len(transformer.transformer_blocks))
    for i in db_range:
        block = transformer.transformer_blocks[i]
        key = f"double_{i}"
        originals[key] = block.attn.processor
        p = AdaptiveInjectProcessor(store, key, base_alpha=base_alpha, method=method)
        block.attn.processor = p
        processors[key] = p

    sb_range = single_blocks if single_blocks is not None else range(len(transformer.single_transformer_blocks))
    for i in sb_range:
        block = transformer.single_transformer_blocks[i]
        key = f"single_{i}"
        originals[key] = block.attn.processor
        p = AdaptiveInjectProcessor(store, key, base_alpha=base_alpha, method=method)
        block.attn.processor = p
        processors[key] = p

    return originals, processors


def restore_processors(transformer, originals: dict):
    """Restore the original processors saved by an attach_* call."""
    for key, proc in originals.items():
        prefix, idx_str = key.split("_", 1)
        idx = int(idx_str)
        if prefix == "double":
            transformer.transformer_blocks[idx].attn.processor = proc
        else:
            transformer.single_transformer_blocks[idx].attn.processor = proc
