"""
Step 2 + 3 — Custom FLUX attention processor.

Runs in a GPU/torch environment with diffusers installed. Two jobs:
  (2) Extract the per-image-token attention OUTPUT NORM (SDPA-safe: the output is
      available for free; we never materialize the QK^T weight matrix).
  (3) Optionally apply a pre-softmax memory-repulsion penalty on the NEW object's
      text-token -> occupied-image-token scores. This is the ONLY place we touch
      raw scores, and only for a thin [n_obj_txt, n_img] slice, so the bulk of
      attention still runs on fused SDPA.

State (current step, object token span, packed occupied-token indices, penalty)
is passed per-step via `joint_attention_kwargs` from the pipeline, so no globals.

This mirrors diffusers FluxAttnProcessor2_0 (concatenate text+image -> one joint
attention -> split). Confirm field/attr names against your pinned diffusers
version; the structure below matches the mainline 0.30–0.32 layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math

import torch
import torch.nn.functional as F


@dataclass
class ExtractionState:
    """Per-forward-pass control + collection, shared by ref across all hooked blocks."""
    step_index: int = 0
    total_steps: int = 28
    obj_token_span: tuple[int, int] = (0, 0)     # (start,end) in the TEXT token axis
    collect_blocks: Optional[set[int]] = None    # which block ids to accumulate norms from
    apply_prior: bool = False                    # early-phase repulsion on/off
    occupied_token_idx: Optional[torch.Tensor] = None  # LongTensor of image-token indices
    penalty: float = 6.0                         # pre-softmax subtraction magnitude
    # collected outputs:
    norm_accum: dict = field(default_factory=dict)   # block_id -> summed norm vector
    norm_count: int = 0

    def reset_collection(self):
        self.norm_accum.clear()
        self.norm_count = 0

    def aggregate_norm(self) -> Optional[torch.Tensor]:
        """Mean over collected blocks -> per-image-token norm vector, or None."""
        if not self.norm_accum:
            return None
        stacked = torch.stack(list(self.norm_accum.values()), dim=0)
        return stacked.mean(dim=0)


def _apply_rope(x, freqs_cis):
    # placeholder hook point; in real diffusers the rope is applied via
    # apply_rotary_emb before attention. We keep the interface so a drop-in
    # can call the same helper the pipeline uses.
    return x


class NormExtractionFluxAttnProcessor:
    """
    Drop-in replacement for FluxAttnProcessor2_0 that also extracts output norms
    and can apply the repulsion prior. `block_id` identifies which transformer
    block this instance is attached to; `state` is shared across all blocks.
    """

    def __init__(self, block_id: int, state: ExtractionState):
        self.block_id = block_id
        self.state = state

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,               # image tokens (b, n_img, d)
        encoder_hidden_states: torch.Tensor = None, # text tokens (b, n_txt, d)
        attention_mask: torch.Tensor = None,
        image_rotary_emb=None,
        **kwargs,
    ):
        s = self.state
        b, n_img, _ = hidden_states.shape
        n_txt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0

        # ---- projections (image stream) ----
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        # ---- projections (text stream) ----
        if encoder_hidden_states is not None:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)

        h = attn.heads
        def shape(t):
            return t.view(b, -1, h, t.shape[-1] // h).transpose(1, 2)  # (b,h,n,dh)

        q, k, v = shape(q), shape(k), shape(v)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        if encoder_hidden_states is not None:
            eq, ek, ev = shape(eq), shape(ek), shape(ev)
            if getattr(attn, "norm_added_q", None) is not None:
                eq = attn.norm_added_q(eq)
            if getattr(attn, "norm_added_k", None) is not None:
                ek = attn.norm_added_k(ek)
            # FLUX joint order: [text, image] along sequence
            Q = torch.cat([eq, q], dim=2)
            K = torch.cat([ek, k], dim=2)
            V = torch.cat([ev, v], dim=2)
        else:
            Q, K, V = q, k, v

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb  # type: ignore
            Q = apply_rotary_emb(Q, image_rotary_emb)
            K = apply_rotary_emb(K, image_rotary_emb)

        # ---- attention ----
        # Fast path: fused SDPA for the whole joint sequence (no weights kept).
        # Slow path: only when the repulsion prior is active this step, and only
        # to bias the NEW OBJECT's text rows over occupied image columns.
        use_prior = (
            s.apply_prior
            and encoder_hidden_states is not None
            and s.occupied_token_idx is not None
            and s.obj_token_span[1] > s.obj_token_span[0]
        )

        if not use_prior:
            out = F.scaled_dot_product_attention(Q, K, V)
        else:
            out = self._attention_with_prior(Q, K, V, n_txt, n_img)

        # (b,h,seq,dh) -> (b,seq,h*dh)
        out = out.transpose(1, 2).reshape(b, -1, h * (Q.shape[-1]))

        # split back to text/image
        if encoder_hidden_states is not None:
            txt_out = out[:, :n_txt]
            img_out = out[:, n_txt:]
        else:
            txt_out, img_out = None, out

        # ---- Step 2: collect per-image-token output norm ----
        if s.collect_blocks is None or self.block_id in s.collect_blocks:
            with torch.no_grad():
                # L2 over channels, mean over batch -> (n_img,)
                r = img_out.detach().float().norm(dim=-1).mean(dim=0)
                if self.block_id in s.norm_accum:
                    s.norm_accum[self.block_id] += r
                else:
                    s.norm_accum[self.block_id] = r

        # ---- output projections (as in FluxAttnProcessor2_0) ----
        img_out = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)   # dropout
        if encoder_hidden_states is not None:
            txt_out = attn.to_add_out(txt_out)
            return img_out, txt_out
        return img_out

    def _attention_with_prior(self, Q, K, V, n_txt, n_img):
        """
        Biases only the new object's text-token rows against occupied image-token
        columns, pre-softmax. Runs fused SDPA for the full sequence, then rewrites
        only the handful of object-text rows — avoids materialising a full seq×seq
        score matrix (which would be ~36 GB across all hooked blocks at 1024×1024).
        """
        s = self.state
        dh = Q.shape[-1]
        t0, t1 = s.obj_token_span
        occ_cols = s.occupied_token_idx.to(Q.device) + n_txt   # image col offsets

        # Fast path for all tokens.
        out = F.scaled_dot_product_attention(Q, K, V)   # (b,h,seq,dh)

        # Recompute only the n_obj text rows with the penalty applied.
        scale = 1.0 / math.sqrt(dh)
        Q_obj = Q[:, :, t0:t1, :]                               # (b,h,n_obj,dh)
        scores_obj = torch.matmul(Q_obj, K.transpose(-2, -1)) * scale  # (b,h,n_obj,seq)
        scores_obj[:, :, :, occ_cols] -= s.penalty
        attn_w_obj = torch.softmax(scores_obj, dim=-1)
        out_obj = torch.matmul(attn_w_obj, V)                   # (b,h,n_obj,dh)

        # Overwrite the object text rows; leave everything else from fused SDPA.
        out = out.clone()
        out[:, :, t0:t1, :] = out_obj
        return out


def attach_processors(transformer, state: ExtractionState,
                      collect_blocks: Optional[set[int]] = None):
    """
    Register NormExtractionFluxAttnProcessor on each of the 19 dual-stream
    `transformer_blocks`. Single-stream blocks are left on the default processor
    for the safe floor. Returns the list of block ids hooked.
    """
    state.collect_blocks = collect_blocks
    hooked = []
    for i, block in enumerate(transformer.transformer_blocks):
        block.attn.processor = NormExtractionFluxAttnProcessor(block_id=i, state=state)
        hooked.append(i)
    return hooked
