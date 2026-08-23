# -*- coding: utf-8 -*-
"""
test_on_qwen.py — Diagnostic probe for Qwen-Image-Edit-2509

Runs up to 8 tests to confirm assumptions before implementing fixes.

Tests
-----
  T1  VAE config        confirms latents_mean/latents_std exist; prints values
  T2  Token layout      confirms n_img, S_txt, S_tot at runtime via a live hook
  T3  K/V capture       NaN check, norm per layer, how many layers captured
  T4  K layer stats     spatial-frequency analysis across all 60 layers → TIER_A candidates
  T5  VAE round-trip    encode → normalize → denormalize → decode PSNR (confirms normalization)
  T6  BLD callback      confirms callback fires, correct latent shape, sigma values logged
  T7  Injection delta   pixel diff with/without K/V injection in object zone

Usage
-----
  python NewWork/KontextEval/test_on_qwen.py \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir   results/test_qwen \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --tests T1,T2,T3,T4,T5,T6,T7

  # Skip slow tests (T4 needs a full inference pass, T7 needs two):
  --tests T1,T2,T3,T5,T6
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── pretty-print helpers ─────────────────────────────────────────────────────

W = 64

def _sep(title: str):
    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")

def _ok(msg):   print(f"    ✓  {msg}")
def _warn(msg): print(f"    ⚠  {msg}")
def _fail(msg): print(f"    ✗  {msg}")
def _info(msg): print(f"       {msg}")


# ── synthetic test image ──────────────────────────────────────────────────────

def _synth_scene(w: int = 1024, h: int = 1024) -> Image.Image:
    """Gradient + colored regions so VAE encoding is non-trivial."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # horizontal gradient background (grey tones)
    for x in range(w):
        v = int(180 + 40 * np.sin(x / w * np.pi))
        arr[:, x] = [v, v, v]
    # colored rectangle (simulates an inserted object)
    arr[h//3: 2*h//3, w//4: 3*w//4] = [200, 120, 60]
    # some texture dots
    rng = np.random.default_rng(42)
    pts = rng.integers(0, [w, h], size=(200, 2))
    for px, py in pts:
        arr[py, px] = [60, 60, 200]
    return Image.fromarray(arr)


def _synth_mask(w: int = 1024, h: int = 1024) -> np.ndarray:
    """Binary mask covering the coloured rectangle."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[h//3: 2*h//3, w//4: 3*w//4] = 255
    return m


# ── T1: VAE config ───────────────────────────────────────────────────────────

def test_t1_vae_config(pipe, args):
    _sep("T1 — VAE Config")
    cfg = pipe.vae.config
    _info(f"VAE class : {type(pipe.vae).__name__}")
    _info(f"vae_scale_factor : {getattr(pipe, 'vae_scale_factor', 'NOT FOUND')}")

    # Must-have attributes
    for attr in ("latents_mean", "latents_std"):
        if hasattr(cfg, attr):
            vals = list(getattr(cfg, attr))
            _ok(f"{attr} : len={len(vals)}  first4={[f'{v:.4f}' for v in vals[:4]]}  "
                f"last4={[f'{v:.4f}' for v in vals[-4:]]}")
        else:
            _fail(f"{attr} NOT in vae.config — normalization will silently use identity!")

    # Must NOT be present (FLUX-style)
    for attr in ("shift_factor", "scaling_factor"):
        if hasattr(cfg, attr):
            _warn(f"{attr} = {getattr(cfg, attr)} (unexpected for Wan2.1 VAE — "
                  f"check which normalization path the pipeline is actually taking)")
        else:
            _ok(f"{attr} correctly absent")

    # z_dim — tells us the number of latent channels
    for attr in ("z_dim", "latent_channels", "in_channels"):
        if hasattr(cfg, attr):
            _info(f"{attr} = {getattr(cfg, attr)}")

    # Spatial stride via temperal_downsample
    if hasattr(cfg, "temperal_downsample"):
        td = cfg.temperal_downsample
        n_spatial = sum(1 for x in td if x)
        n_spatial += len(td)  # every stage also does spatial
        _info(f"temperal_downsample = {td}  →  spatial stages with downsampling={sum(1 for x in td if x)}")

    # Effective token resolution for 1024×1024
    vsf = getattr(pipe, "vae_scale_factor", 8)
    h_tok = args.height // (vsf * 2)
    w_tok = args.width  // (vsf * 2)
    _info(f"Token grid for {args.height}×{args.width}: {h_tok}×{w_tok} = {h_tok*w_tok} tokens per image")
    _info(f"Total image tokens in sequence: {h_tok*w_tok*2} (input + target)")


# ── T2: Token layout (live hook) ─────────────────────────────────────────────

class _LayoutProbe:
    """Captures sequence dimensions from the first double-stream block."""
    def __init__(self):
        self.record: dict = {}
        self._done = False

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, encoder_hidden_states_mask=None,
                 image_rotary_emb=None, **kwargs):
        if not self._done and encoder_hidden_states is not None:
            self.record = {
                "S_img"   : hidden_states.shape[1],
                "S_txt"   : encoder_hidden_states.shape[1],
                "img_dtype": str(hidden_states.dtype),
                "txt_dtype": str(encoder_hidden_states.dtype),
                "has_mask": encoder_hidden_states_mask is not None,
                "has_rope": image_rotary_emb is not None,
                "rope_type": (
                    type(image_rotary_emb).__name__
                    if image_rotary_emb is not None else "N/A"
                ),
                "rope_len" : (
                    len(image_rotary_emb)
                    if isinstance(image_rotary_emb, (list, tuple)) else "N/A"
                ),
            }
            if image_rotary_emb is not None and isinstance(image_rotary_emb, (list, tuple)):
                for ri, r in enumerate(image_rotary_emb):
                    if isinstance(r, torch.Tensor):
                        self.record[f"rope[{ri}].shape"] = str(r.shape)
                        self.record[f"rope[{ri}].dtype"] = str(r.dtype)
            self._done = True
        # fall through to original processor
        orig = attn._orig_for_probe
        return orig(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    image_rotary_emb=image_rotary_emb, **kwargs)


def test_t2_token_layout(pipe, args):
    _sep("T2 — Token Layout (live hook)")

    probe = _LayoutProbe()
    orig_procs = pipe.transformer.attn_processors

    # Attach probe to layer 0 only
    new_procs = dict(orig_procs)
    first_key = next(iter(new_procs))
    orig_proc = new_procs[first_key]
    probe._orig_for_probe = orig_proc   # stash original
    # We'll monkey-patch only layer 0 via a wrapper class
    # Actually simpler: set all procs to probe for one step, restore immediately

    class _FirstLayerWrapper:
        def __init__(self, orig, probe):
            self._orig  = orig
            self._probe = probe
        def __call__(self, attn, *a, **kw):
            result = self._probe(attn, *a, **kw)
            if result is None:
                result = self._orig(attn, *a, **kw)
            return result

    # Attach a simple print processor to all blocks (probe only fires once)
    class _PrintOnceProc:
        def __init__(self, orig):
            self._orig  = orig
            self._fired = False
        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, encoder_hidden_states_mask=None,
                     image_rotary_emb=None, **kwargs):
            if not self._fired and encoder_hidden_states is not None:
                probe.record.update({
                    "S_img": hidden_states.shape[1],
                    "S_txt": encoder_hidden_states.shape[1],
                    "img_dtype": str(hidden_states.dtype),
                    "has_mask": encoder_hidden_states_mask is not None,
                    "has_rope": image_rotary_emb is not None,
                })
                if image_rotary_emb is not None and isinstance(image_rotary_emb, (list, tuple)):
                    probe.record["rope_n_components"] = len(image_rotary_emb)
                    for ri, r in enumerate(image_rotary_emb):
                        if isinstance(r, torch.Tensor):
                            probe.record[f"rope[{ri}].shape"] = str(r.shape)
                            probe.record[f"rope[{ri}].dtype"] = str(r.dtype)
                probe._done = True
                self._fired = True
            return self._orig(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                              attention_mask=attention_mask,
                              encoder_hidden_states_mask=encoder_hidden_states_mask,
                              image_rotary_emb=image_rotary_emb, **kwargs)

    wrapped = {k: _PrintOnceProc(v) for k, v in orig_procs.items()}
    pipe.transformer.set_attn_processor(wrapped)

    img = _synth_scene(args.width, args.height)
    gen = torch.Generator(device=pipe.device).manual_seed(0)
    try:
        pipe(
            image=img, prompt="a room",
            negative_prompt="blurry",
            num_inference_steps=1, true_cfg_scale=1.0,
            height=args.height, width=args.width,
            generator=gen, output_type="latent",
        )
    except Exception as e:
        _warn(f"1-step inference raised: {e!s:.120}")

    pipe.transformer.set_attn_processor(orig_procs)

    if probe.record:
        vsf = getattr(pipe, "vae_scale_factor", 8)
        n_img = (args.height // (vsf * 2)) * (args.width // (vsf * 2))
        _ok(f"S_img (hidden_states) = {probe.record.get('S_img', '?')}  "
            f"(expected 2×n_img = {2*n_img})")
        _ok(f"S_txt (encoder_hidden_states) = {probe.record.get('S_txt', '?')}")
        _info(f"S_tot (img+txt) = "
              f"{probe.record.get('S_img', 0) + probe.record.get('S_txt', 0)}")
        _ok(f"dtype img={probe.record.get('img_dtype')}  "
            f"has_mask={probe.record.get('has_mask')}  "
            f"has_rope={probe.record.get('has_rope')}")
        if "rope_n_components" in probe.record:
            _ok(f"image_rotary_emb is a {probe.record['rope_n_components']}-tuple")
            for ri in range(probe.record.get("rope_n_components", 0)):
                k = f"rope[{ri}].shape"
                if k in probe.record:
                    _info(f"  rope[{ri}]: shape={probe.record[k]}  "
                          f"dtype={probe.record.get(f'rope[{ri}].dtype','?')}")
    else:
        _fail("Probe did not fire — no double-stream block called, or model architecture differs")

    # Also report dispatch_attention_fn availability
    try:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        _ok("dispatch_attention_fn available (fast path will be used)")
    except ImportError:
        _warn("dispatch_attention_fn NOT available — falling back to SDPA transpose path")


# ── T3: K/V capture quality ───────────────────────────────────────────────────

class _AllLayerKVCapture:
    """Captures K/V from ALL layers (no zone filter) for quality inspection."""
    def __init__(self):
        self.kv:    Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, encoder_hidden_states_mask=None,
                 image_rotary_emb=None, **kwargs):
        # Minimal QKV projection to get K
        H  = attn.heads
        B  = hidden_states.shape[0]
        k_ = attn.to_k(hidden_states).unflatten(-1, (H, -1))
        if attn.norm_k is not None:
            k_ = attn.norm_k(k_)
        v_ = attn.to_v(hidden_states).unflatten(-1, (H, -1))

        self.kv[self._layer] = (
            k_.detach().cpu().float(),
            v_.detach().cpu().float(),
        )
        self._layer += 1

        # Run standard attention to keep the pass valid
        try:
            from diffusers.models.attention_dispatch import dispatch_attention_fn as _d
            q_ = attn.to_q(hidden_states).unflatten(-1, (H, -1))
            if attn.norm_q is not None: q_ = attn.norm_q(q_)
            S_txt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            if encoder_hidden_states is not None:
                tq = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (H, -1))
                tk = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (H, -1))
                tv = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (H, -1))
                if attn.norm_added_q is not None: tq = attn.norm_added_q(tq)
                if attn.norm_added_k is not None: tk = attn.norm_added_k(tk)
                q_ = torch.cat([tq, q_], dim=1)
                k_ = torch.cat([tk, k_], dim=1)
                v_ = torch.cat([tv, v_], dim=1)
            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                img_f, txt_f = image_rotary_emb
                # apply to image portion
                def _cx(x, f):
                    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
                    return torch.view_as_real(xc * f.unsqueeze(1).to(xc.device)).flatten(3).type_as(x)
                if encoder_hidden_states is not None:
                    q_img = q_[:, S_txt:]; q_txt = q_[:, :S_txt]
                    k_img = k_[:, S_txt:]; k_txt = k_[:, :S_txt]
                    q_img = _cx(q_img, img_f); k_img = _cx(k_img, img_f)
                    q_txt = _cx(q_txt, txt_f); k_txt = _cx(k_txt, txt_f)
                    q_ = torch.cat([q_txt, q_img], dim=1)
                    k_ = torch.cat([k_txt, k_img], dim=1)
                else:
                    q_ = _cx(q_, img_f); k_ = _cx(k_, img_f)
            out = _d(q_, k_, v_)
            out = out.flatten(2, 3).to(q_.dtype)
        except Exception:
            q2 = attn.to_q(hidden_states).view(B, -1, H, hidden_states.shape[-1]//H).transpose(1,2)
            k2 = attn.to_k(hidden_states).view(B, -1, H, hidden_states.shape[-1]//H).transpose(1,2)
            v2 = attn.to_v(hidden_states).view(B, -1, H, hidden_states.shape[-1]//H).transpose(1,2)
            out = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0)
            out = out.transpose(1,2).flatten(2,3).to(q2.dtype)
            S_txt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0

        if encoder_hidden_states is not None:
            enc_out = out[:, :S_txt]
            out     = out[:, S_txt:]
            out     = attn.to_out[1](attn.to_out[0](out))
            enc_out = attn.to_add_out(enc_out)
            return out, enc_out
        return out


def test_t3_kv_capture(pipe, args):
    _sep("T3 — K/V Capture Quality")

    img = _synth_scene(args.width, args.height)
    orig_procs = pipe.transformer.attn_processors

    cap = _AllLayerKVCapture()
    pipe.transformer.set_attn_processor(cap)

    _CAP_STEPS = 20
    step_kv_log: List[dict] = []

    def _cb(pipe_ref, step_idx, timestep, cb_kwargs):
        # snapshot after each step
        if step_idx == _CAP_STEPS - 1:   # final step
            for li, (k, v) in cap.kv.items():
                has_nan_k = k.isnan().any().item()
                has_nan_v = v.isnan().any().item()
                step_kv_log.append({
                    "layer": li,
                    "k_shape": tuple(k.shape),
                    "k_norm": k.norm().item(),
                    "k_nan": has_nan_k,
                    "v_nan": has_nan_v,
                    "k_min": k.min().item(),
                    "k_max": k.max().item(),
                })
        cap._layer = 0
        return cb_kwargs

    gen = torch.Generator(device=pipe.device).manual_seed(0)
    try:
        pipe(
            image=img, prompt="a synthetic test image",
            negative_prompt="blurry",
            num_inference_steps=_CAP_STEPS, true_cfg_scale=1.0,
            height=args.height, width=args.width,
            generator=gen, output_type="latent",
            callback_on_step_end=_cb,
            callback_on_step_end_tensor_inputs=[],
        )
    except Exception as e:
        _fail(f"Capture pass raised: {e!s:.150}")

    pipe.transformer.set_attn_processor(orig_procs)

    if not step_kv_log:
        _fail("No K/V captured — processor may not have been called")
        return

    n_total   = len(step_kv_log)
    n_nan_k   = sum(1 for r in step_kv_log if r["k_nan"])
    n_nan_v   = sum(1 for r in step_kv_log if r["v_nan"])
    norms     = [r["k_norm"] for r in step_kv_log]

    _ok(f"Captured {n_total} / 60 layers")
    if n_nan_k == 0:
        _ok("No NaN in K tensors")
    else:
        _fail(f"{n_nan_k} layers have NaN in K — sigma edge case or scheduler bug")
    if n_nan_v == 0:
        _ok("No NaN in V tensors")
    else:
        _fail(f"{n_nan_v} layers have NaN in V")

    _info(f"K norm  min={min(norms):.2f}  max={max(norms):.2f}  "
          f"mean={np.mean(norms):.2f}  std={np.std(norms):.2f}")
    _info(f"K value range: [{min(r['k_min'] for r in step_kv_log):.4f}, "
          f"{max(r['k_max'] for r in step_kv_log):.4f}]")

    # Print per-layer norms in a compact table
    _info("Layer K-norms (every 5th layer):")
    row = "  "
    for r in step_kv_log:
        if r["layer"] % 5 == 0:
            row += f"L{r['layer']:02d}:{r['k_norm']:6.1f}  "
    _info(row)

    # Save norms plot
    fig, ax = plt.subplots(figsize=(12, 3))
    layers = [r["layer"] for r in step_kv_log]
    norms_arr = [r["k_norm"] for r in step_kv_log]
    ax.bar(layers, norms_arr, color="steelblue")
    ax.set_xlabel("Layer"); ax.set_ylabel("K norm")
    ax.set_title("T3 — K norm per layer (final capture step)")
    plt.tight_layout()
    path = os.path.join(args.out_dir, "t3_kv_norms.png")
    plt.savefig(path, dpi=100); plt.close(fig)
    _ok(f"Saved: {path}")


# ── T4: K layer statistics → TIER_A candidates ───────────────────────────────

class _KStatCapture:
    """Captures image-stream K statistics at every layer during inference."""
    def __init__(self, n_img: int, h_tok: int, w_tok: int):
        self.n_img  = n_img
        self.h_tok  = h_tok
        self.w_tok  = w_tok
        self.stats:  Dict[int, dict] = {}
        self._layer = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, encoder_hidden_states_mask=None,
                 image_rotary_emb=None, **kwargs):
        H = attn.heads
        k_ = attn.to_k(hidden_states).unflatten(-1, (H, -1))  # (B, S_img, H, D)
        if attn.norm_k is not None:
            k_ = attn.norm_k(k_)

        # Use only the first n_img tokens (input image) for spatial analysis
        k_in = k_[0, :self.n_img, :, :].float()   # (n_img, H, D)
        k_grid = k_in.reshape(self.h_tok, self.w_tok, H, -1)  # (h, w, H, D)

        # Metric 1: spatial variance — high means high-frequency (structure) layer
        k_flat = k_grid.reshape(self.h_tok * self.w_tok, -1)  # (n_img, H*D)
        spatial_std = k_flat.std(dim=0).mean().item()

        # Metric 2: neighbor similarity — high means low-frequency (content) layer
        dh = (k_grid[1:, :, :, :] - k_grid[:-1, :, :, :]).abs().mean().item()
        dw = (k_grid[:, 1:, :, :] - k_grid[:, :-1, :, :]).abs().mean().item()
        neighbor_grad = (dh + dw) / 2.0

        # Metric 3: FFT low-freq ratio on mean-over-heads projection
        k_2d = k_grid.mean(dim=(2, 3))  # (h_tok, w_tok)
        fft_mag = torch.fft.rfft2(k_2d).abs()  # (h_tok, w_tok//2+1)
        h4 = max(1, self.h_tok // 4)
        w4 = max(1, self.w_tok // 4)
        low_power   = fft_mag[:h4, :w4].mean().item()
        total_power = fft_mag.mean().item() + 1e-8
        low_ratio   = low_power / total_power

        # Running accumulation: average across calls
        if self._layer not in self.stats:
            self.stats[self._layer] = {
                "spatial_std": 0.0, "neighbor_grad": 0.0,
                "low_ratio": 0.0, "count": 0,
                "k_norm": 0.0,
            }
        s = self.stats[self._layer]
        s["spatial_std"]   += spatial_std
        s["neighbor_grad"] += neighbor_grad
        s["low_ratio"]     += low_ratio
        s["k_norm"]        += k_in.norm().item()
        s["count"]         += 1

        self._layer += 1

        # Passthrough: minimal forward to keep the graph valid
        q_ = attn.to_q(hidden_states).unflatten(-1, (H, -1))
        v_ = attn.to_v(hidden_states).unflatten(-1, (H, -1))
        if attn.norm_q is not None: q_ = attn.norm_q(q_)

        S_txt = 0
        if encoder_hidden_states is not None:
            tq = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (H, -1))
            tk = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (H, -1))
            tv = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (H, -1))
            if attn.norm_added_q is not None: tq = attn.norm_added_q(tq)
            if attn.norm_added_k is not None: tk = attn.norm_added_k(tk)
            S_txt = tq.shape[1]
            q_ = torch.cat([tq, q_], dim=1)
            k_ = torch.cat([tk, k_], dim=1)
            v_ = torch.cat([tv, v_], dim=1)

        q2 = q_.transpose(1, 2); k2 = k_.transpose(1, 2); v2 = v_.transpose(1, 2)
        out = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0)
        out = out.transpose(1, 2).flatten(2, 3).to(q2.dtype)

        if encoder_hidden_states is not None:
            enc_out = out[:, :S_txt]
            out     = out[:, S_txt:]
            out     = attn.to_out[1](attn.to_out[0](out))
            enc_out = attn.to_add_out(enc_out)
            return out, enc_out
        return out


def test_t4_k_layer_stats(pipe, args):
    _sep("T4 — K Layer Statistics → TIER_A Candidates")

    vsf   = getattr(pipe, "vae_scale_factor", 8)
    h_tok = args.height // (vsf * 2)
    w_tok = args.width  // (vsf * 2)
    n_img = h_tok * w_tok

    img        = _synth_scene(args.width, args.height)
    orig_procs = pipe.transformer.attn_processors
    stat_cap   = _KStatCapture(n_img=n_img, h_tok=h_tok, w_tok=w_tok)

    pipe.transformer.set_attn_processor(stat_cap)

    def _reset_cb(pipe_ref, step_idx, ts, cb_kwargs):
        stat_cap._layer = 0
        return cb_kwargs

    gen = torch.Generator(device=pipe.device).manual_seed(0)
    _N_STEPS = 5
    try:
        pipe(
            image=img, prompt="a room with objects",
            negative_prompt="blurry",
            num_inference_steps=_N_STEPS, true_cfg_scale=3.5,
            height=args.height, width=args.width,
            generator=gen, output_type="latent",
            callback_on_step_end=_reset_cb,
            callback_on_step_end_tensor_inputs=[],
        )
    except Exception as e:
        _fail(f"Inference raised: {e!s:.150}")
        pipe.transformer.set_attn_processor(orig_procs)
        return

    pipe.transformer.set_attn_processor(orig_procs)

    if not stat_cap.stats:
        _fail("No stats captured")
        return

    # Average over accumulation count
    rows = []
    for li in sorted(stat_cap.stats.keys()):
        s = stat_cap.stats[li]
        c = max(s["count"], 1)
        rows.append({
            "layer":        li,
            "spatial_std":  s["spatial_std"]   / c,
            "neighbor_grad": s["neighbor_grad"] / c,
            "low_ratio":    s["low_ratio"]      / c,
            "k_norm":       s["k_norm"]         / c,
        })

    # Rank by low_ratio (higher = more low-frequency = TIER_A candidate)
    rows_sorted = sorted(rows, key=lambda r: r["low_ratio"], reverse=True)

    _ok(f"Stats collected across {_N_STEPS} steps × {len(rows)} layers")

    # Top-15 TIER_A candidates
    top15 = [r["layer"] for r in rows_sorted[:15]]
    top15_sorted = sorted(top15)
    _ok(f"Top-15 TIER_A candidates (by low-freq ratio): {top15_sorted}")

    # Also suggest set notation
    _info(f"_TIER_A = {set(top15_sorted)}")

    # Print table header
    _info(f"\n  {'Layer':>5}  {'k_norm':>8}  {'spatial_std':>11}  {'nbr_grad':>8}  {'low_ratio':>9}  rank")
    _info(f"  {'─'*5}  {'─'*8}  {'─'*11}  {'─'*8}  {'─'*9}  {'─'*4}")
    for rank, r in enumerate(rows_sorted[:20], 1):
        _info(f"  {r['layer']:>5}  {r['k_norm']:>8.1f}  {r['spatial_std']:>11.4f}  "
              f"{r['neighbor_grad']:>8.4f}  {r['low_ratio']:>9.4f}  #{rank}")

    # Plot: low_ratio and spatial_std per layer
    layers    = [r["layer"]     for r in rows]
    lf_ratios = [r["low_ratio"] for r in rows]
    sp_stds   = [r["spatial_std"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    colors = ["#e74c3c" if li in top15_sorted else "#3498db" for li in layers]
    ax1.bar(layers, lf_ratios, color=colors)
    ax1.set_ylabel("Low-freq ratio ↑ = content layer")
    ax1.set_title(f"T4 — K Layer Statistics  (red = top-15 TIER_A candidates)")
    ax2.bar(layers, sp_stds, color=colors)
    ax2.set_ylabel("Spatial std ↓ = content layer")
    ax2.set_xlabel("Layer index")
    plt.tight_layout()
    path = os.path.join(args.out_dir, "t4_k_layer_stats.png")
    plt.savefig(path, dpi=100); plt.close(fig)
    _ok(f"Saved: {path}")


# ── T5: VAE round-trip (confirms normalization) ───────────────────────────────

def test_t5_vae_roundtrip(pipe, args):
    _sep("T5 — VAE Round-Trip (normalization correctness)")

    dtype  = next(pipe.vae.parameters()).dtype
    device = next(pipe.vae.parameters()).device
    img    = _synth_scene(args.width, args.height)

    img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    img_t  = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_t  = (img_t * 2.0 - 1.0).to(dtype).unsqueeze(2)  # (1, 3, 1, H, W)

    with torch.no_grad():
        z_raw = pipe.vae.encode(img_t.to(device)).latent_dist.mean
        z_raw = z_raw.squeeze(2)  # (1, C, H_lat, W_lat)

    C = z_raw.shape[1]
    _info(f"Raw encoded shape: {tuple(z_raw.shape)}  dtype={z_raw.dtype}")
    _info(f"Raw latent range : [{z_raw.min():.4f}, {z_raw.max():.4f}]  "
          f"mean={z_raw.mean():.4f}  std={z_raw.std():.4f}")

    # ── Method A: FLUX-style (wrong for Qwen) ──────────────────────────────
    sf_a    = getattr(pipe.vae.config, "shift_factor",   0.0)
    scale_a = getattr(pipe.vae.config, "scaling_factor", 1.0)
    z_flux  = (z_raw - sf_a) * scale_a
    _info(f"\n  [FLUX-style sf={sf_a} scale={scale_a}]")
    _info(f"  Normalized range: [{z_flux.min():.4f}, {z_flux.max():.4f}]  "
          f"std={z_flux.std():.4f}")
    if abs(sf_a) < 1e-6 and abs(scale_a - 1.0) < 1e-6:
        _warn("FLUX-style is identity (sf=0, scale=1) → normalization has NO effect")
    else:
        _ok("FLUX-style normalization applied non-trivially")

    # ── Method B: Qwen-style (correct) ─────────────────────────────────────
    if not hasattr(pipe.vae.config, "latents_mean"):
        _fail("pipe.vae.config.latents_mean not found — cannot test Qwen normalization")
    else:
        lm = torch.tensor(pipe.vae.config.latents_mean, dtype=torch.float32,
                           device=z_raw.device).reshape(1, -1, 1, 1)
        ls = torch.tensor(pipe.vae.config.latents_std,  dtype=torch.float32,
                           device=z_raw.device).reshape(1, -1, 1, 1)
        z_qwen = (z_raw.float() - lm) / ls
        _info(f"\n  [Qwen-style per-channel latents_mean/std]")
        _info(f"  latents_mean : [{lm.min():.4f} … {lm.max():.4f}]")
        _info(f"  latents_std  : [{ls.min():.4f} … {ls.max():.4f}]")
        _info(f"  Normalized range: [{z_qwen.min():.4f}, {z_qwen.max():.4f}]  "
              f"std={z_qwen.std():.4f}")

        # A well-normalized latent should have std close to 1.0
        std_val = z_qwen.std().item()
        if 0.7 < std_val < 1.4:
            _ok(f"Normalized std={std_val:.3f} ≈ 1.0  ✓ normalization is correct")
        else:
            _warn(f"Normalized std={std_val:.3f} — expected ≈1.0; check latents_std values")

        # ── Decode round-trip ───────────────────────────────────────────────
        z_denorm = (z_qwen.to(dtype) * ls.to(dtype) + lm.to(dtype)).unsqueeze(2)
        with torch.no_grad():
            decoded = pipe.vae.decode(z_denorm.to(device)).sample  # (1, 3, 1, H, W)
        decoded_np = ((decoded.squeeze(2).squeeze(0).permute(1,2,0).float().cpu().numpy()
                       + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

        # PSNR
        orig_np = np.array(img.convert("RGB"), dtype=np.float32)
        dec_np  = decoded_np.astype(np.float32)
        mse = np.mean((orig_np - dec_np) ** 2)
        psnr = 10 * np.log10((255.0 ** 2) / (mse + 1e-8))
        if psnr > 28:
            _ok(f"Round-trip PSNR = {psnr:.1f} dB  (>28 dB confirms normalization works)")
        elif psnr > 20:
            _warn(f"Round-trip PSNR = {psnr:.1f} dB  (acceptable but not great)")
        else:
            _fail(f"Round-trip PSNR = {psnr:.1f} dB  (<20 dB — normalization broken)")

        # Save comparison
        comp = np.concatenate([orig_np.astype(np.uint8), decoded_np], axis=1)
        path = os.path.join(args.out_dir, "t5_vae_roundtrip.png")
        Image.fromarray(comp).save(path)
        _ok(f"Saved: {path}  (left=original, right=round-tripped)")


# ── T6: BLD callback firing ───────────────────────────────────────────────────

def _pack_latents_t6(z: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) → (B, H//2 * W//2, C*4) — matches Qwen's 2×2 patch packing."""
    B, C, H, W = z.shape
    h_tok, w_tok = H // 2, W // 2
    z = z.reshape(B, C, h_tok, 2, w_tok, 2)
    z = z.permute(0, 2, 4, 1, 3, 5)
    return z.reshape(B, h_tok * w_tok, C * 4)


def test_t6_bld_callback(pipe, args, mask_path: str | None = None):
    _sep("T6 — BLD Callback Firing")

    dtype  = next(pipe.transformer.parameters()).dtype
    device = str(next(pipe.transformer.parameters()).device)

    if mask_path is None or not os.path.isfile(mask_path):
        _warn("No mask file found — saving synthetic mask and using it")
        m = _synth_mask(args.width, args.height)
        mask_path = os.path.join(args.out_dir, "_synth_mask.png")
        Image.fromarray(m).save(mask_path)

    img = _synth_scene(args.width, args.height)

    img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    img_t  = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_t  = (img_t * 2.0 - 1.0).to(dtype).unsqueeze(2)

    with torch.no_grad():
        enc_dev = next(pipe.vae.parameters()).device
        z0_raw  = pipe.vae.encode(img_t.to(enc_dev)).latent_dist.mean
        z0_raw  = z0_raw.squeeze(2)  # (1, 16, 128, 128)

    if hasattr(pipe.vae.config, "latents_mean"):
        lm = torch.tensor(pipe.vae.config.latents_mean, dtype=dtype,
                           device=enc_dev).reshape(1, -1, 1, 1)
        ls = torch.tensor(pipe.vae.config.latents_std,  dtype=dtype,
                           device=enc_dev).reshape(1, -1, 1, 1)
        z0 = ((z0_raw - lm) / ls).to(device, dtype)
    else:
        _warn("latents_mean not found — using identity normalization for test")
        z0 = z0_raw.to(device, dtype)

    _info(f"VAE encode → z0 spatial shape: {tuple(z0.shape)}")

    # Pack (1, 16, 128, 128) → (1, 4096, 64) — matches callback latent shape
    z0_tok = _pack_latents_t6(z0)
    n_tok  = z0_tok.shape[1]         # 4096
    h_tok  = z0.shape[-2] // 2      # 64
    w_tok  = z0.shape[-1] // 2      # 64
    _info(f"Packed z0_tok shape: {tuple(z0_tok.shape)}  (expected (1, {n_tok}, 64))")

    noise = torch.randn(z0_tok.shape, device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(42))

    # Mask at token resolution (64×64), flattened to (1, n_tok, 1)
    placement = np.array(Image.open(mask_path).convert("L").resize(
        (w_tok, h_tok), Image.NEAREST))
    mask_lat  = (placement > 127).astype(np.float32)
    mask_t    = torch.from_numpy(mask_lat.reshape(1, n_tok, 1)).to(device, dtype)

    firing_log: List[dict] = []

    def _bld_cb(_pipeline, step_idx, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        sigma   = max(0.0, min(1.0, float(timestep) / 1000.0))
        shape_ok = latents.shape == z0_tok.shape
        firing_log.append({
            "step":     step_idx,
            "sigma":    sigma,
            "ts":       float(timestep),
            "latent_shape": tuple(latents.shape),
            "shape_ok": shape_ok,
            "modified": False,
        })
        if shape_ok:
            z_bg  = ((1.0 - sigma) * z0_tok + sigma * noise).to(latents.dtype).to(latents.device)
            m     = mask_t.to(latents.dtype).to(latents.device)
            callback_kwargs["latents"] = latents * m + z_bg * (1.0 - m)
            firing_log[-1]["modified"] = True
        return callback_kwargs

    gen = torch.Generator(device=pipe.device).manual_seed(0)
    N = 8
    try:
        pipe(
            image=img, prompt="a room with a colored box",
            negative_prompt="blurry",
            num_inference_steps=N, true_cfg_scale=3.5,
            height=args.height, width=args.width,
            generator=gen, output_type="latent",
            callback_on_step_end=_bld_cb,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    except Exception as e:
        _fail(f"Inference raised: {e!s:.150}")
        return

    _ok(f"Callback fired {len(firing_log)} times across {N} steps")
    n_modified = sum(1 for r in firing_log if r["modified"])
    n_shape_ok = sum(1 for r in firing_log if r["shape_ok"])
    if n_shape_ok == len(firing_log):
        _ok(f"All {n_shape_ok} calls had correct latent shape {firing_log[0]['latent_shape']}")
    else:
        _warn(f"Shape mismatch on {len(firing_log)-n_shape_ok} calls — "
              f"packed latents? callback receiving wrong tensor")
    _ok(f"Callback MODIFIED latents on {n_modified}/{len(firing_log)} calls")

    _info("\n  step  sigma    timestep  latent_shape          shape_ok  modified")
    _info(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*20}  {'─'*8}  {'─'*8}")
    for r in firing_log:
        _info(f"  {r['step']:>4}  {r['sigma']:.4f}   {r['ts']:>8.1f}  "
              f"{str(r['latent_shape']):>20}  {str(r['shape_ok']):>8}  {str(r['modified']):>8}")


# ── T7: Injection delta ───────────────────────────────────────────────────────

def test_t7_injection_delta(pipe, args, collage: Image.Image,
                             mask_path: str, obj_img: Image.Image | None = None):
    _sep("T7 — K/V Injection Delta (with vs without)")

    # Import the injection machinery from the pipeline
    try:
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "phase1_collage_qwen",
            os.path.join(os.path.dirname(__file__), "phase1_collage_qwen.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        run_qwen           = mod.run_qwen
        run_with_qwen_kv_injection = mod.run_with_qwen_kv_injection
        _placement_mask_to_token_zone = mod._placement_mask_to_token_zone
    except Exception as e:
        _warn(f"Could not import phase1_collage_qwen: {e!s:.100}  — using inline inference")
        run_qwen = None

    dtype  = next(pipe.transformer.parameters()).dtype
    device = str(next(pipe.transformer.parameters()).device)

    prompt   = "A photorealistic room with the object placed naturally."
    neg_p    = "blurry, distorted, low quality, watermark"
    gen_seed = 42

    # ── No injection ─────────────────────────────────────────────────────────
    gen = torch.Generator(device=pipe.device).manual_seed(gen_seed)
    out_no_kv = pipe(
        image=collage, prompt=prompt, negative_prompt=neg_p,
        num_inference_steps=10, true_cfg_scale=3.5,
        height=args.height, width=args.width, generator=gen,
    ).images[0]
    out_no_kv.save(os.path.join(args.out_dir, "t7_no_kv.png"))
    _ok("Saved: t7_no_kv.png")

    # ── With injection (current window 0.5-1.0) ───────────────────────────
    vsf   = getattr(pipe, "vae_scale_factor", 8)
    n_img = (args.height // (vsf*2)) * (args.width // (vsf*2))
    orig_procs = pipe.transformer.attn_processors

    # Build inline injector using the same classes but controlled window
    target_zone = np.array(
        Image.open(mask_path).convert("L").resize(
            (args.width // (vsf*2), args.height // (vsf*2)), Image.NEAREST
        )
    ).reshape(-1) > 127
    _info(f"Target zone: {target_zone.sum()} / {n_img} tokens "
          f"({100*target_zone.mean():.1f}%)")

    results = {}
    for label, cf in [("late_0.5-1.0", (0.5, 1.0)),
                      ("early_0.0-0.5", (0.0, 0.5)),
                      ("mid_0.2-0.7",   (0.2, 0.7))]:
        try:
            if run_with_qwen_kv_injection is not None:
                gen = torch.Generator(device=pipe.device).manual_seed(gen_seed)
                out = run_with_qwen_kv_injection(
                    pipe=pipe, collage=collage, prompt=prompt,
                    target_zone=target_zone,
                    seed=gen_seed, num_steps=10, guidance=3.5,
                    height=args.height, width=args.width,
                    obj_strength=0.7, cutoff_frac=cf,
                )
                results[label] = out
                out.save(os.path.join(args.out_dir, f"t7_kv_{label}.png"))
                _ok(f"Saved: t7_kv_{label}.png")
            else:
                _warn(f"Skipping {label} — injection module not available")
        except Exception as e:
            _fail(f"{label} raised: {e!s:.120}")
        finally:
            pipe.transformer.set_attn_processor(orig_procs)

    # ── Pixel diff analysis ───────────────────────────────────────────────
    ref_np = np.array(out_no_kv.convert("RGB"), dtype=np.float32)

    # Object zone at pixel resolution
    zone_px = np.array(Image.open(mask_path).convert("L").resize(
        (args.width, args.height), Image.NEAREST)) > 127

    _info("\n  Pixel-space L2 diff (no_kv as reference):")
    _info(f"  {'Window':>20}  {'full-image':>12}  {'object-zone':>12}  {'background':>12}")
    _info(f"  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*12}")
    for label, out in results.items():
        out_np = np.array(out.convert("RGB"), dtype=np.float32)
        diff   = np.abs(out_np - ref_np)
        full   = diff.mean()
        zone   = diff[zone_px].mean()  if zone_px.any() else float("nan")
        bgnd   = diff[~zone_px].mean() if (~zone_px).any() else float("nan")
        _info(f"  {label:>20}  {full:>12.2f}  {zone:>12.2f}  {bgnd:>12.2f}")
        if zone > 5.0:
            _ok(f"  {label}: injection IS changing the object zone (diff={zone:.1f})")
        else:
            _warn(f"  {label}: injection has very little effect (diff={zone:.1f}) — window may be wrong")

    # Save comparison grid
    imgs   = [collage, out_no_kv] + list(results.values())
    labels = ["collage", "no_kv"]  + list(results.keys())
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    for ax, im, lb in zip(axes, imgs, labels):
        ax.imshow(np.array(im)); ax.axis("off"); ax.set_title(lb, fontsize=8)
    plt.tight_layout()
    path = os.path.join(args.out_dir, "t7_injection_comparison.png")
    plt.savefig(path, dpi=80); plt.close(fig)
    _ok(f"Saved: {path}")


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Qwen-Image-Edit diagnostic probe")
    p.add_argument("--hf_token",   required=True)
    p.add_argument("--cache_dir",  default="./models")
    p.add_argument("--out_dir",    default="results/test_qwen")
    p.add_argument("--sketch_dir", default=None,
                   help="KontextEval inputs dir.  If given, uses real mask/collage for T6/T7.")
    p.add_argument("--mask_name",  default="bicycle",
                   help="Which mask_{name}.png to use for T6/T7 (default: bicycle)")
    p.add_argument("--model",      default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--height",     type=int, default=1024)
    p.add_argument("--width",      type=int, default=1024)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--tests",      default="T1,T2,T3,T4,T5,T6,T7",
                   help="Comma-separated test IDs. T4 and T7 are slow (need inference).")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    tests  = {t.strip().upper() for t in args.tests.split(",")}
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'═'*W}")
    print(f"  test_on_qwen.py — Qwen-Image-Edit-2509 Diagnostic")
    print(f"{'═'*W}")
    print(f"  Model   : {args.model}")
    print(f"  Tests   : {sorted(tests)}")
    print(f"  Out dir : {args.out_dir}")
    print(f"{'─'*W}")

    # Load pipeline (needed for all tests except pure static ones)
    from diffusers import QwenImageEditPlusPipeline
    import torch
    print("\nLoading pipeline ...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        token=args.hf_token, cache_dir=args.cache_dir,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    print("Pipeline loaded.\n")

    if "T1" in tests:
        test_t1_vae_config(pipe, args)

    if "T2" in tests:
        test_t2_token_layout(pipe, args)

    if "T3" in tests:
        test_t3_kv_capture(pipe, args)

    if "T4" in tests:
        test_t4_k_layer_stats(pipe, args)

    if "T5" in tests:
        test_t5_vae_roundtrip(pipe, args)

    if "T6" in tests:
        mask_path = None
        if args.sketch_dir:
            mp = os.path.join(args.sketch_dir, f"mask_{args.mask_name}.png")
            if os.path.isfile(mp):
                mask_path = mp
        test_t6_bld_callback(pipe, args, mask_path=mask_path)

    if "T7" in tests:
        mask_path = None
        collage   = _synth_scene(args.width, args.height)
        if args.sketch_dir:
            mp = os.path.join(args.sketch_dir, f"mask_{args.mask_name}.png")
            if os.path.isfile(mp):
                mask_path = mp
                # Try to use a real collage if one exists from a prior run
                cp = os.path.join(args.out_dir, f"collage_{args.mask_name}.png")
                if os.path.isfile(cp):
                    collage = Image.open(cp).convert("RGB")
                    _info(f"T7: using existing collage: {cp}")
                else:
                    _info("T7: no existing collage found — using synthetic scene")
        if mask_path is None:
            mask_path = os.path.join(args.out_dir, "_synth_mask.png")
            Image.fromarray(_synth_mask(args.width, args.height)).save(mask_path)
        test_t7_injection_delta(pipe, args, collage=collage, mask_path=mask_path)

    print(f"\n{'═'*W}")
    print(f"  All tests complete.  Results in: {args.out_dir}")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()