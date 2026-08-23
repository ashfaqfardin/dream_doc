# -*- coding: utf-8 -*-
"""
QwenImage/probe.py — Model probing infrastructure for Qwen-Image-Edit-2509.

Hooks into every transformer block during a denoising run and captures:
  • Q, K, V tensors (pre-softmax, at the attention head level)
  • Attention weights (post-softmax)
  • Block output hidden states

Usage
-----
  from probe import QwenProbe
  probe = QwenProbe(pipe)
  probe.attach()
  _ = pipe(...)           # run any generation — features are captured
  data = probe.detach()   # dict: {block_idx: {step: FeatureFrame}}
  probe.report()          # print feature statistics

Notes on Wan/Qwen architecture
------------------------------
Wan (the backbone of Qwen-Image-Edit) uses an MMDiT transformer similar to
FLUX.1 but adapted for video. Block access follows diffusers convention:
  pipe.transformer.transformer_blocks      (double-stream or unified blocks)

Each block's attention is a diffusers Attention module with attribute layout:
  attn.to_q, attn.to_k, attn.to_v         (Linear projections)
  attn.processor                           (handles the actual attention op)

We hook at the processor level so we see the actual Q/K/V that enter attention,
after any RoPE / normalization that the block applies.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ── Feature frame ─────────────────────────────────────────────────────────────

@dataclass
class FeatureFrame:
    """Features captured at one (block, denoising_step) pair."""
    block_idx:   int
    step:        int
    q:           Optional[torch.Tensor] = None   # (B, H, N, d)
    k:           Optional[torch.Tensor] = None
    v:           Optional[torch.Tensor] = None
    attn_out:    Optional[torch.Tensor] = None   # (B, N, C)
    hidden_out:  Optional[torch.Tensor] = None   # (B, N, C) — full block output

    def cpu(self) -> "FeatureFrame":
        for attr in ("q", "k", "v", "attn_out", "hidden_out"):
            t = getattr(self, attr)
            if t is not None:
                object.__setattr__(self, attr, t.detach().cpu().float())
        return self

    def attn_weights(self, softmax: bool = True) -> Optional[torch.Tensor]:
        """Compute attention weight matrix from stored Q and K."""
        if self.q is None or self.k is None:
            return None
        B, H, N, d = self.q.shape
        scale = d ** -0.5
        scores = torch.einsum("bhnd,bhmd->bhnm", self.q.float(), self.k.float()) * scale
        if softmax:
            return scores.softmax(dim=-1)
        return scores


# ── Capture processor ──────────────────────────────────────────────────────────

class CaptureAttnProcessor:
    """Replaces an attention processor, captures Q/K/V/output, then delegates."""

    def __init__(self, original_processor, store: Dict, block_idx: int,
                 step_counter: List[int], capture_cpu: bool = True):
        self._orig    = original_processor
        self._store   = store
        self._block   = block_idx
        self._counter = step_counter   # shared mutable list [current_step]
        self._cpu     = capture_cpu

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None,
                 encoder_hidden_states_mask=None, *args, **kwargs):
        step = self._counter[0]

        # Compute Q, K, V for capture (image stream only; text stream uses add_q_proj etc.)
        ks = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        q = attn.to_q(hidden_states)
        k = attn.to_k(ks)
        v = attn.to_v(ks)

        # Reshape to (B, H, N, d) for storage
        def reshape_qkv(x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            return x.view(B, N, attn.heads, C // attn.heads).transpose(1, 2)

        q_h = reshape_qkv(q)
        k_h = reshape_qkv(k)
        v_h = reshape_qkv(v)

        # Delegate to the original processor — pass ALL kwargs so RoPE is applied correctly
        out = self._orig(
            attn, hidden_states, encoder_hidden_states,
            attention_mask, *args,
            image_rotary_emb=image_rotary_emb,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            **kwargs,
        )

        # Dual-stream blocks return (img_out, txt_out) tuple; capture img stream only
        _out_t = out[0] if isinstance(out, tuple) else out

        frame = FeatureFrame(
            block_idx=self._block, step=step,
            q=q_h.detach().cpu().float() if self._cpu else q_h.detach(),
            k=k_h.detach().cpu().float() if self._cpu else k_h.detach(),
            v=v_h.detach().cpu().float() if self._cpu else v_h.detach(),
            attn_out=_out_t.detach().cpu().float() if self._cpu else _out_t.detach(),
        )
        self._store.setdefault(self._block, {})[step] = frame
        return out   # return original (tuple or tensor) unchanged


# ── Step-tracking callback ─────────────────────────────────────────────────────

class _StepCounter:
    def __init__(self, counter: List[int]):
        self._counter = counter

    def __call__(self, pipe, i: int, t, callback_kwargs: dict) -> dict:
        self._counter[0] = i
        return callback_kwargs


# ── Block finder ──────────────────────────────────────────────────────────────

def _find_blocks(transformer) -> List[Any]:
    """Return the list of transformer blocks, trying multiple attribute names."""
    for attr in ("transformer_blocks", "blocks", "layers", "encoder_layers"):
        blocks = getattr(transformer, attr, None)
        if blocks is not None:
            return list(blocks)
    # Wan-specific: single_transformer_blocks + double_transformer_blocks
    double = getattr(transformer, "double_transformer_blocks", [])
    single = getattr(transformer, "single_transformer_blocks", [])
    combined = list(double) + list(single)
    if combined:
        return combined
    raise AttributeError(
        "Could not find transformer blocks. "
        "Inspect pipe.transformer attributes and update _find_blocks()."
    )


def _find_attn(block) -> Optional[Any]:
    """Return the attention module from a block."""
    for attr in ("attn", "attn1", "self_attn", "attention"):
        a = getattr(block, attr, None)
        if a is not None:
            return a
    return None


# ── QwenProbe ─────────────────────────────────────────────────────────────────

class QwenProbe:
    """Attach / detach capture hooks on all Qwen-Image-Edit transformer blocks.

    Example::

        probe = QwenProbe(pipe)
        probe.attach()
        image = run_qwen(pipe, image=..., prompt=..., ...)
        data  = probe.detach()
        # data[block_idx][step] → FeatureFrame
        probe.report(data)
    """

    def __init__(self, pipe, capture_cpu: bool = True):
        self.pipe       = pipe
        self.cpu        = capture_cpu
        self._store:    Dict[int, Dict[int, FeatureFrame]] = {}
        self._originals: Dict[int, Any] = {}
        self._step_counter: List[int] = [0]

    # ── Public API ────────────────────────────────────────────────────────────

    def attach(self) -> "QwenProbe":
        """Install capture processors on all blocks. Returns self for chaining."""
        self._store.clear()
        self._originals.clear()
        self._step_counter[0] = 0

        transformer = self.pipe.transformer
        blocks = _find_blocks(transformer)
        print(f"[Probe] Found {len(blocks)} transformer blocks.")

        for idx, block in enumerate(blocks):
            attn = _find_attn(block)
            if attn is None:
                print(f"  [Probe] Block {idx}: no attention module found, skipping.")
                continue
            orig_proc = getattr(attn, "processor", None)
            if orig_proc is None:
                print(f"  [Probe] Block {idx}: no processor attribute, skipping.")
                continue
            self._originals[idx] = orig_proc
            attn.set_processor(CaptureAttnProcessor(
                orig_proc, self._store, idx, self._step_counter, self.cpu
            ))

        print(f"[Probe] Attached to {len(self._originals)}/{len(blocks)} blocks.")
        return self

    def detach(self) -> Dict[int, Dict[int, FeatureFrame]]:
        """Restore original processors and return captured data."""
        transformer = self.pipe.transformer
        blocks      = _find_blocks(transformer)
        for idx, block in enumerate(blocks):
            if idx not in self._originals:
                continue
            attn = _find_attn(block)
            if attn is not None:
                attn.set_processor(self._originals[idx])
        data = dict(self._store)
        self._store.clear()
        self._originals.clear()
        print(f"[Probe] Detached. Captured {len(data)} blocks.")
        return data

    def step_callback(self):
        """Return a callback_on_step_end that increments the step counter."""
        return _StepCounter(self._step_counter)

    # ── Analysis ──────────────────────────────────────────────────────────────

    @staticmethod
    def report(data: Dict[int, Dict[int, FeatureFrame]]) -> None:
        """Print a summary table of captured feature statistics."""
        print(f"\n{'─'*70}")
        print(f"  {'Block':>5}  {'Steps':>5}  {'Q shape':>20}  {'K var (mean)':>14}")
        print(f"{'─'*70}")
        for bidx in sorted(data.keys()):
            steps_data = data[bidx]
            n_steps = len(steps_data)
            frame0  = next(iter(steps_data.values()))
            q_shape = str(tuple(frame0.q.shape)) if frame0.q is not None else "n/a"
            k_var   = (
                float(torch.stack([
                    f.k.float().var() for f in steps_data.values() if f.k is not None
                ]).mean()) if n_steps > 1 else float("nan")
            )
            print(f"  {bidx:>5}  {n_steps:>5}  {q_shape:>20}  {k_var:>14.6f}")
        print(f"{'─'*70}\n")

    @staticmethod
    def layer_feature_distance(
        data_a: Dict[int, Dict[int, FeatureFrame]],
        data_b: Dict[int, Dict[int, FeatureFrame]],
        attr:   str = "attn_out",
        step:   int = 0,
    ) -> Dict[int, float]:
        """Compute per-block L2 distance between two probe runs.

        Used for drift analysis: pass data from a no-edit run and an edit run,
        the distance curve shows which blocks are most affected by the edit.
        """
        distances: Dict[int, float] = {}
        common_blocks = set(data_a.keys()) & set(data_b.keys())
        for bidx in sorted(common_blocks):
            fa = data_a[bidx].get(step)
            fb = data_b[bidx].get(step)
            if fa is None or fb is None:
                continue
            ta = getattr(fa, attr)
            tb = getattr(fb, attr)
            if ta is None or tb is None:
                continue
            dist = float(F.mse_loss(ta.float(), tb.float()).sqrt())
            distances[bidx] = dist
        return distances

    @staticmethod
    def attention_entropy(data: Dict[int, Dict[int, FeatureFrame]],
                          step: int = 0) -> Dict[int, float]:
        """Per-block attention entropy at a given denoising step.

        High entropy = diffuse attention = weaker style/content lock.
        Low entropy  = concentrated attention = strong content binding.
        """
        entropy: Dict[int, float] = {}
        for bidx, steps_data in data.items():
            frame = steps_data.get(step)
            if frame is None:
                continue
            w = frame.attn_weights()
            if w is None:
                continue
            w_clamped = w.clamp(min=1e-9)
            ent = -(w_clamped * w_clamped.log()).sum(dim=-1).mean()
            entropy[bidx] = float(ent)
        return entropy

    @staticmethod
    def background_to_object_attention(
        data:         Dict[int, Dict[int, FeatureFrame]],
        bg_token_ids: np.ndarray,   # indices of background tokens in the sequence
        obj_token_ids: np.ndarray,  # indices of object reference tokens
        step: int = 0,
    ) -> Dict[int, float]:
        """Mean attention weight from background tokens to object tokens.

        This quantifies the "contamination" channel: how much background
        tokens attend to object reference tokens at each block.
        Blocks with high values are candidates for the attention gate.
        """
        flow: Dict[int, float] = {}
        for bidx, steps_data in data.items():
            frame = steps_data.get(step)
            if frame is None:
                continue
            w = frame.attn_weights()       # (B, H, N, N)
            if w is None:
                continue
            # Average over batch and heads
            w_mean = w.float().mean(dim=(0, 1))   # (N, N)
            if max(bg_token_ids) >= w_mean.shape[0] or max(obj_token_ids) >= w_mean.shape[1]:
                continue
            bg_to_obj = w_mean[np.ix_(bg_token_ids, obj_token_ids)].mean()
            flow[bidx] = float(bg_to_obj)
        return flow


# ── Spatial mask → token IDs ───────────────────────────────────────────────────

def mask_to_token_ids(
    mask_np:  np.ndarray,   # (H, W) bool — background=True
    lat_h:    int,
    lat_w:    int,
    n_text:   int = 0,      # number of text tokens before image tokens
) -> np.ndarray:
    """Convert a pixel-space background mask to attention-sequence token indices.

    Qwen patchifies latents with a 2×2 patch stride (FLUX-style), so the
    latent grid (lat_h, lat_w) becomes (lat_h//2, lat_w//2) patches.
    """
    import cv2
    patch_h = lat_h // 2
    patch_w = lat_w // 2
    # Resize mask to patch resolution
    m_small = np.array(
        Image.fromarray(mask_np.astype(np.uint8) * 255).resize(
            (patch_w, patch_h), Image.NEAREST
        )
    ) > 128
    # Flatten to 1D token indices, offset by n_text
    token_ids = np.where(m_small.flatten())[0] + n_text
    return token_ids


# ── Convenience: probe a single run ──────────────────────────────────────────

def probe_run(
    pipe,
    image,
    prompt:    str,
    seed:      int   = 42,
    num_steps: int   = 8,
    guidance:  float = 1.0,
    height:    int   = 1024,
    width:     int   = 1024,
    neg:       str   = "blurry, distorted, low quality",
    label:     str   = "probe",
) -> Tuple[Dict[int, Dict[int, FeatureFrame]], "Image.Image"]:
    """Run a single generation with full feature capture. Returns (data, image)."""
    from baseline import _fix_cat_dims
    probe = QwenProbe(pipe)
    probe.attach()
    sc    = probe.step_callback()
    gen   = torch.Generator(device=pipe.device).manual_seed(seed)
    with _fix_cat_dims():
        img = pipe(
            prompt=prompt, negative_prompt=neg,
            image=image, num_inference_steps=num_steps,
            true_cfg_scale=guidance, height=height, width=width,
            generator=gen,
            callback_on_step_end=sc,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
    data = probe.detach()
    print(f"[Probe] {label}: {len(data)} blocks × {num_steps} steps captured.")
    return data, img


if __name__ == "__main__":
    print("probe.py — import QwenProbe and probe_run from this module.")
    print("See exp_layer_drift.py and exp_attention_gate.py for usage examples.")
