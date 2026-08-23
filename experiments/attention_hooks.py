"""
Experiment 3 — Attention capture infrastructure.

Patches FluxAttnProcessor2_0 to compute attention weights manually
(instead of using flash-attention which doesn't expose them), and
stores the text→image quadrant per block per timestep.

Usage (as a module)
-------------------
from experiments.attention_hooks import AttentionCapture

with AttentionCapture(pipe) as capture:
    images = pipe(prompt, ...).images

# capture.maps  →  dict[(block_type, layer_idx, step)] = tensor (n_heads, n_img, n_txt)
# capture.maps is filled during inference

Implementation note
-------------------
FluxAttnProcessor2_0.__call__ concatenates [text_queries, image_queries] into a
single sequence before the attention call:

    query = cat([txt_q, img_q], dim=2)   # (batch, heads, txt+img, head_dim)
    key   = cat([txt_k, img_k], dim=2)
    ...
    hidden_states = F.scaled_dot_product_attention(query, key, value)
    txt_out, img_out = split(hidden_states, [n_txt, n_img], dim=2)

We replace scaled_dot_product_attention with a manual softmax computation
when capture is enabled, so we can extract the full (txt+img) × (txt+img)
attention weight matrix.
"""

import contextlib
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# Storage for captured maps in the current capture context
_GLOBAL_CAPTURE: Optional[Dict] = None


class _CapturingAttnProcessor:
    """
    Drop-in replacement for FluxAttnProcessor2_0 that stores attention weights
    when a capture context is active.
    """

    def __init__(self, original_processor, block_type: str, layer_idx: int):
        self._orig = original_processor
        self.block_type  = block_type
        self.layer_idx   = layer_idx
        self._step_count = 0  # incremented by the capture context

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        image_rotary_emb=None,
        attention_store=None,
        step_index=None,
        index_block=None,
        mm_copy_blocks=None,
    ):
        global _GLOBAL_CAPTURE

        # If no capture is active, fall through to the original processor
        if _GLOBAL_CAPTURE is None:
            return self._orig(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, image_rotary_emb, attention_store,
                step_index, index_block, mm_copy_blocks,
            )

        # --- Manual attention to expose weights ---
        batch_size   = encoder_hidden_states.shape[0]
        n_txt_tokens = encoder_hidden_states.shape[1]
        n_img_tokens = hidden_states.shape[1]
        head_dim     = attn.to_q.out_features // attn.heads

        def _proj_and_reshape(linear, x):
            out = linear(x)
            return out.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        q_img = _proj_and_reshape(attn.to_q, hidden_states)
        k_img = _proj_and_reshape(attn.to_k, hidden_states)
        v_img = _proj_and_reshape(attn.to_v, hidden_states)

        q_txt = _proj_and_reshape(attn.add_q_proj, encoder_hidden_states)
        k_txt = _proj_and_reshape(attn.add_k_proj, encoder_hidden_states)
        v_txt = _proj_and_reshape(attn.add_v_proj, encoder_hidden_states)

        # QK normalisation (rms_norm)
        if attn.norm_q is not None:
            q_img = attn.norm_q(q_img)
            q_txt = attn.norm_added_q(q_txt)
        if attn.norm_k is not None:
            k_img = attn.norm_k(k_img)
            k_txt = attn.norm_added_k(k_txt)

        # RoPE
        if image_rotary_emb is not None:
            from src.diffusers.models.attention_processor import apply_rope
            q_full = torch.cat([q_txt, q_img], dim=2)
            k_full = torch.cat([k_txt, k_img], dim=2)
            q_full, k_full = apply_rope(q_full, k_full, image_rotary_emb)
            q_txt, q_img = q_full[:, :, :n_txt_tokens], q_full[:, :, n_txt_tokens:]
            k_txt, k_img = k_full[:, :, :n_txt_tokens], k_full[:, :, n_txt_tokens:]

        q = torch.cat([q_txt, q_img], dim=2)   # (B, H, txt+img, d)
        k = torch.cat([k_txt, k_img], dim=2)
        v = torch.cat([v_txt, v_img], dim=2)

        # K,V replacement (StableFlow copy_blocks)
        if mm_copy_blocks is not None and index_block in mm_copy_blocks:
            k[1:] = k[:1]
            v[1:] = v[:1]

        scale = math.sqrt(head_dim)
        attn_scores  = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = torch.softmax(attn_scores, dim=-1)   # (B, H, txt+img, txt+img)

        # Store text→image quadrant: rows=image tokens, cols=text tokens
        # Shape: (B, H, n_img, n_txt)
        img_to_txt = attn_weights[:, :, n_txt_tokens:, :n_txt_tokens]

        key = (self.block_type, self.layer_idx, step_index if step_index is not None
               else self._step_count)
        # Average over batch dimension before storing (saves memory)
        _GLOBAL_CAPTURE[key] = img_to_txt.mean(0).detach().cpu()  # (H, n_img, n_txt)

        # Produce output the standard way
        out = torch.matmul(attn_weights, v)  # (B, H, txt+img, d)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(q.dtype)

        txt_out = out[:, :n_txt_tokens]
        img_out = out[:, n_txt_tokens:]

        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)
        txt_out = attn.to_add_out(txt_out)

        return img_out, txt_out


class AttentionCapture:
    """
    Context manager that temporarily replaces all MM-DiT attention processors
    with capturing versions.

    Example
    -------
    with AttentionCapture(pipe) as cap:
        img = pipe(prompt, ...).images[0]

    maps = cap.maps
    # maps[(\"mm\", 5, 3)]  → tensor (n_heads, 4096, n_text_tokens)
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.maps: Dict = {}
        self._original_processors = {}

    def __enter__(self):
        global _GLOBAL_CAPTURE
        _GLOBAL_CAPTURE = self.maps
        self._install_processors()
        return self

    def __exit__(self, *args):
        global _GLOBAL_CAPTURE
        self._restore_processors()
        _GLOBAL_CAPTURE = None

    def _install_processors(self):
        for i, block in enumerate(self.pipe.transformer.transformer_blocks):
            orig = block.attn.processor
            self._original_processors[("mm", i)] = orig
            block.attn.processor = _CapturingAttnProcessor(orig, "mm", i)

    def _restore_processors(self):
        for i, block in enumerate(self.pipe.transformer.transformer_blocks):
            key = ("mm", i)
            if key in self._original_processors:
                block.attn.processor = self._original_processors[key]
        self._original_processors.clear()


# ---------------------------------------------------------------------------
# Spatial reshaping helper
# ---------------------------------------------------------------------------

def attn_to_spatial(
    attn_tensor: torch.Tensor,
    token_idx: int,
    grid_size: int = 64,
) -> torch.Tensor:
    """
    Extract per-image-patch attention for one text token and reshape to 2D.

    Parameters
    ----------
    attn_tensor : (n_heads, n_img_tokens, n_text_tokens)
    token_idx   : which text token to visualise
    grid_size   : sqrt(n_img_tokens), 64 for 1024×1024 FLUX

    Returns
    -------
    Tensor (grid_size, grid_size) — spatial attention map (averaged over heads)
    """
    mean_over_heads = attn_tensor.mean(dim=0)          # (n_img, n_txt)
    token_map = mean_over_heads[:, token_idx]          # (n_img,)
    return token_map.reshape(grid_size, grid_size)


def modality_balance_map(
    full_attn: torch.Tensor,
    n_img_tokens: int,
    grid_size: int = 64,
) -> torch.Tensor:
    """
    For each image token, compute the fraction of attention mass going to
    text tokens vs image tokens.

    Parameters
    ----------
    full_attn    : (n_heads, total_seq, total_seq) — full (txt+img)×(txt+img) matrix
                   Must include BOTH txt and img rows. Reconstruct by running with
                   capture of the full matrix if needed.
    n_img_tokens : number of image tokens (4096 for 1024×1024)

    Returns
    -------
    Tensor (grid_size, grid_size) — text-attention fraction per image patch
    """
    n_txt = full_attn.shape[-1] - n_img_tokens
    img_rows   = full_attn[:, n_txt:, :]              # (H, n_img, total)
    to_text    = img_rows[:, :, :n_txt].sum(-1)        # (H, n_img)
    to_image   = img_rows[:, :, n_txt:].sum(-1)        # (H, n_img)
    balance    = to_text / (to_text + to_image + 1e-8) # (H, n_img)
    return balance.mean(0).reshape(grid_size, grid_size)
