# -*- coding: utf-8 -*-
"""
phase1_ref_qwen.py — No-mask pipeline: native multi-image reference conditioning

Uses Qwen-Image-Edit-2509's built-in multi-image support (image concatenation
training) to pass scene + object reference as separate images — no canvas
stitching, no crop, no bleed artifacts.

Pipeline per insertion step
----------------------------
  1. Stage A  : sketch → obj_img  (Qwen renders the sketch)
  2. Stage K  : Qwen(image=[scene, obj_img], prompt=place_prompt) → 1024×1024

Pipeline per removal step
--------------------------
  1. Qwen(image=scene, prompt=remove_prompt) → next scene

Usage
------
  python NewWork/KontextEval/phase1_ref_qwen.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token   $HF_TOKEN \\
      --cache_dir  ./models \\
      --out_dir    results/phase1_ref_qwen
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from huggingface_hub import hf_hub_download
from PIL import Image
from typing import Dict, Tuple

try:
    from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen as _apply_rope
except ImportError:
    _apply_rope = None   # type: ignore[assignment]


# ── Constants ─────────────────────────────────────────────────────────────────

EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "insert"},
    {"name": "vase",    "description": "black ceramic vase with flowers",  "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",               "action": "remove"},
    {"name": "bicycle", "description": "yellow mountain bicycle",          "action": "remove"},
]

BASE_PROMPT = (
    "A empty room with a wooden floor, white walls, "
    "and a window letting in natural light."
)

_SEP = "═" * 60

LIGHTNING_REPO = "lightx2v/Qwen-Image-Lightning"
LIGHTNING_SUBFOLDER = "Qwen-Image-Edit-2509"

# Scheduler config required by the Lightning distilled weights.
# shift=1.0 + use_dynamic_shifting=True replaces the base model's higher shift.
LIGHTNING_SCHEDULER_CFG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


# ── Pipeline loading ───────────────────────────────────────────────────────────

def load_pipeline(
    model: str = "Qwen/Qwen-Image-Edit-2509",
    hf_token: str | None = None,
    cache_dir: str = "./models",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
    lightning: bool = False,
    lightning_steps: int = 8,
) -> QwenImageEditPlusPipeline:
    # Auto-enable CPU offload when GPU VRAM is likely too small.
    # bf16 transformer (20B) + text encoder (7B) needs ~55 GB.
    # Lightning LoRA does NOT reduce the bf16 transformer size — offload still needed.
    if not cpu_offload and device != "cpu" and torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if total_gb < 65:
            print(f"  [AUTO] GPU has {total_gb:.0f} GB — enabling cpu_offload "
                  f"(bf16 model needs ~55 GB)")
            cpu_offload = True

    kwargs: dict = dict(torch_dtype=dtype, token=hf_token, cache_dir=cache_dir)
    if lightning:
        kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(
            LIGHTNING_SCHEDULER_CFG
        )
        print(f"  [Lightning] Scheduler: FlowMatch dynamic-shift (shift=1.0, log3 range)")

    pipe = QwenImageEditPlusPipeline.from_pretrained(model, **kwargs)

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if lightning:
        weight_name = (
            f"Qwen-Image-Edit-2509-Lightning-{lightning_steps}steps-V1.0-bf16.safetensors"
        )
        print(f"  [Lightning] Downloading LoRA: {LIGHTNING_REPO}/{LIGHTNING_SUBFOLDER}/{weight_name}")
        lora_path = hf_hub_download(
            repo_id=LIGHTNING_REPO,
            filename=f"{LIGHTNING_SUBFOLDER}/{weight_name}",
            token=hf_token,
            cache_dir=cache_dir,
        )
        pipe.load_lora_weights(lora_path)
        print(f"  [Lightning] LoRA loaded — {lightning_steps}-step distillation active")

    _patch_vae_for_pipeline(pipe)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── VAE patch (video VAE → 4-D I/O for pipeline compatibility) ────────────────

def _patch_vae_for_pipeline(pipe) -> None:
    """Patch the video VAE so encode/decode accept and return 4-D (B,C,H,W) tensors.

    The pipeline's internal squeeze() on a (1,C,1,H,W) output drops BOTH the
    batch dim AND T dim when batch_size=1, producing a 3-D tensor that cannot be
    cat'd with the 4-D denoising latent at line 811 of the pipeline __call__.

    After this patch the VAE behaves as a standard image VAE.
    _encode_latent() and _decode_latent() are updated to match (no manual T dim).
    """
    try:
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    except ImportError:
        try:
            from diffusers.models.autoencoder_kl import DiagonalGaussianDistribution
        except ImportError:
            DiagonalGaussianDistribution = None

    _orig_encode = pipe.vae.encode
    _orig_decode = pipe.vae.decode

    def _encode_4d(x, *a, **kw):
        if x.dim() == 4:
            x = x.unsqueeze(2)          # (B,C,H,W) → (B,C,1,H,W)
        out = _orig_encode(x, *a, **kw)
        if DiagonalGaussianDistribution is not None and hasattr(out, "latent_dist"):
            p = out.latent_dist.parameters
            if p.dim() == 5:            # (B,C,1,H,W) → (B,C,H,W)
                out.latent_dist = DiagonalGaussianDistribution(p.squeeze(2))
        return out

    def _decode_4d(z, *a, **kw):
        if z.dim() == 4:
            z = z.unsqueeze(2)          # (B,C,H,W) → (B,C,1,H,W)
        out = _orig_decode(z, *a, **kw)
        if hasattr(out, "sample") and out.sample.dim() == 5:
            out.sample = out.sample.squeeze(2)   # (B,3,1,H,W) → (B,3,H,W)
        return out

    pipe.vae.encode = _encode_4d
    pipe.vae.decode = _decode_4d
    print("[patch] VAE encode/decode patched: T-dim managed automatically (4-D I/O)")


# ── Qwen call ─────────────────────────────────────────────────────────────────

def run_qwen(
    pipe,
    image,           # PIL Image or list of PIL Images
    prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
    negative_prompt: str = "blurry, distorted, low quality, watermark, text, artifacts",
) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        image=image, num_inference_steps=num_steps,
        true_cfg_scale=guidance,
        height=height, width=width,
        generator=gen,
    ).images[0]


# ── T1: Latent Background Anchoring (LBA) ────────────────────────────────────

def _encode_latent(
    pipe,
    pil_img: Image.Image,
    height:  int,
    width:   int,
) -> torch.Tensor:
    """Encode PIL image → unscaled VAE latent on CPU float32, shape (1, C, H/8, W/8).

    Qwen's VAE is a video VAE: encode() expects (B, C, T, H, W).
    We add a T=1 frame dim before encoding and squeeze it off afterward so all
    downstream code (DSA, latent composite) works with standard 4D tensors.
    """
    img = pil_img.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0        # → [-1, 1]
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    try:
        vae_dev  = next(pipe.vae.parameters()).device
        vae_dtyp = next(pipe.vae.parameters()).dtype
    except StopIteration:
        vae_dev, vae_dtyp = torch.device("cpu"), torch.float32
    t = t.to(vae_dev, vae_dtyp)                 # (1, 3, H, W) — patch adds T=1 internally
    with torch.no_grad():
        z = pipe.vae.encode(t).latent_dist.mean  # (1, C, H/8, W/8) after patch
    return z.cpu().float()


def _decode_latent(
    pipe,
    z: torch.Tensor,   # unscaled (1, C, H/8, W/8), any device
) -> Image.Image:
    """Decode unscaled VAE latent → PIL RGB image.

    Adds and removes the T=1 frame dim required by Qwen's video VAE.
    """
    try:
        vae_dev  = next(pipe.vae.parameters()).device
        vae_dtyp = next(pipe.vae.parameters()).dtype
    except StopIteration:
        vae_dev, vae_dtyp = torch.device("cpu"), torch.float32
    z = z.to(vae_dev, vae_dtyp)                  # (1, C, H/8, W/8) — patch adds T=1 internally
    with torch.no_grad():
        dec = pipe.vae.decode(z).sample          # (1, 3, H, W) after patch
    dec = (dec.float().clamp(-1, 1) / 2 + 0.5)  # → [0, 1]
    arr = (dec[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _latent_composite(
    z_raw:      torch.Tensor,              # (1, C, H/8, W/8) current step output
    z_prev:     torch.Tensor,              # (1, C, H/8, W/8) previous composite
    accum_mask: "torch.Tensor | None",     # (1, 1, H/8, W/8) running object mask
    tau:        float = 0.3,               # per-channel latent-diff threshold
    temp:       float = 0.05,             # sigmoid edge sharpness
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Composite z_raw and z_prev using a soft latent-difference mask.

    Object regions  (|z_raw - z_prev| > tau) → take from z_raw  (new content)
    Background      (|z_raw - z_prev| < tau) → take from z_prev (preserved)

    The accumulated mask only grows across steps: once a pixel is classified as
    part of an inserted object it stays that way.  Background pixels therefore
    always trace back to the initial empty-room encoding, with zero additional
    VAE encode-decode cycles introducing error.

    Returns (z_composite, updated_accum_mask).
    """
    z_raw  = z_raw.cpu().float()
    z_prev = z_prev.cpu().float()

    diff      = (z_raw - z_prev).abs().mean(dim=1, keepdim=True)   # (1,1,H/8,W/8)
    step_mask = torch.sigmoid((diff - tau) / temp)                  # ∈ (0, 1)

    if accum_mask is None:
        accum_mask = step_mask
    else:
        accum_mask = torch.max(accum_mask.cpu().float(), step_mask)  # monotonically grows

    z_comp = accum_mask * z_raw + (1.0 - accum_mask) * z_prev
    return z_comp, accum_mask


# ── Stage A: sketch → object image ────────────────────────────────────────────

def _tight_crop(img: Image.Image, pad: int = 32) -> Image.Image:
    """Crop to the bounding box of non-white pixels, add padding, keep square."""
    arr  = np.array(img.convert("RGB"))
    mask = np.any(arr < 240, axis=2)          # non-white pixels
    if not mask.any():
        return img
    rows  = np.where(mask.any(axis=1))[0]
    cols  = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    r0, r1 = max(0, r0 - pad), min(arr.shape[0], r1 + pad)
    c0, c1 = max(0, c0 - pad), min(arr.shape[1], c1 + pad)
    side   = max(r1 - r0, c1 - c0)           # square crop
    cr = img.crop((c0, r0, c0 + side, r0 + side))
    return cr.resize((512, 512), Image.Resampling.LANCZOS)


def generate_object(
    pipe, sketch_path: str, description: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
) -> Image.Image:
    sketch = Image.open(sketch_path).convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS
    )
    prompt = (
        f"Render this hand-drawn sketch as a photorealistic {description}. "
        f"Place it centered on a plain white background. "
        f"Studio lighting, no shadows, no background objects, high detail."
    )
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    obj = pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality, watermark, text, artifacts, background, room, floor",
        image=sketch, num_inference_steps=num_steps,
        true_cfg_scale=guidance,
        height=height, width=width,
        generator=gen,
    ).images[0]
    # Tight-crop to the object so Picture 2 tokens are mostly object, not white bg
    return _tight_crop(obj)


# ── Prompts ────────────────────────────────────────────────────────────────────

def insert_prompt(description: str) -> str:
    # Pipeline prepends "Picture 1: <img>Picture 2: <img>" with no trailing space,
    # so start with a newline to avoid token boundary issues.
    # Picture 1 = scene, Picture 2 = obj_ref (our image list order).
    return (
        f"\nPicture 1 shows a room. Picture 2 shows a {description}. "
        f"Place the {description} from Picture 2 into the room in Picture 1. "
        f"Choose a natural position off-center — near a wall or corner, resting on the wooden floor. "
        f"Add a realistic contact shadow and match the room lighting. "
        f"Do not move or change any other part of the room."
    )


def remove_prompt(description: str) -> str:
    return (
        f"\nRemove the {description} from this room. "
        f"Fill the area with seamless wooden floor and white wall matching the surroundings. "
        f"Do not change anything else in the room."
    )


# ── SKA (Scene K/V Anchoring) ─────────────────────────────────────────────────
# Diagnostic verdict: blocks 52-57 give +0.020 mean gain at σ<0.3.
# Inject scene K/V from step N into step N+1 to anchor background appearance.

SKA_BLOCKS     = list(range(51, 58))   # inclusive: 52,53,54,55,56,57
SKA_SIGMA_GATE = 0.3                    # activate only for σ < this value


class SKAStore:
    """Persistent K/V cache shared across editing steps.

    Two parallel stores (captured and compared independently):
      kv     / new_kv     — background tokens from scene_cond frame
      obj_kv / new_obj_kv — object tokens from obj frame

    Each entry is a 3-tuple  (k, v, idx)  where:
      k, v : (B, H, n_sel, d)  — K/V of the selected tokens only
      idx  : (n_sel,)           — token positions within the source frame

    Token selection uses the model's own attention: output tokens attend more
    strongly to background scene_cond tokens (they copy background) and to
    actual object pixels in the obj frame (ignoring white fill). Tokens whose
    importance exceeds the per-frame mean are kept.

    Injection: compare-and-replace at only the stored positions.
    Tokens whose current K drifts from the stored reference (cos-sim < thresh)
    are snapped back. Sequence length is never extended.
    """
    # 3-tuple type alias
    _KVI = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def __init__(
        self,
        replace_thresh: float = 0.85,
        inject_bg:      bool  = False,
        inject_mode:    str   = "v",   # "kv" | "k" | "v"
    ):
        self.kv:         Dict[int, "SKAStore._KVI"] = {}
        self.new_kv:     Dict[int, "SKAStore._KVI"] = {}
        self.obj_kv:     Dict[int, "SKAStore._KVI"] = {}
        self.new_obj_kv: Dict[int, "SKAStore._KVI"] = {}
        self.mode:           str   = "normal"
        self.n_out:          int   = 0
        self.has_kv:         bool  = False
        self.replace_thresh: float = replace_thresh
        self.inject_bg:      bool  = inject_bg    # bg injection corrupts walls; off by default
        self.inject_mode:    str   = inject_mode  # "kv" | "k" | "v" — exp1 toggle

    @property
    def has_obj_kv(self) -> bool:
        return bool(self.obj_kv)

    def promote(self):
        """Commit new_kv/new_obj_kv → kv/obj_kv for next step's injection."""
        if self.new_kv:
            self.kv     = dict(self.new_kv)
            self.new_kv = {}
            self.has_kv = True
        if self.new_obj_kv:
            self.obj_kv     = dict(self.new_obj_kv)
            self.new_obj_kv = {}
        self.mode = "normal"


def _attn_importance(
    q_out: torch.Tensor,   # (B, H, n_out, d)  pre-RoPE, output frame queries
    k_ref: torch.Tensor,   # (B, H, n_ref, d)  pre-RoPE, reference frame keys
    head_dim: int,
    n_sample_heads: int = 4,
) -> torch.Tensor:
    """Return per-token importance for k_ref: how much do output queries attend to
    each reference token (summed over queries, averaged over sampled heads).
    Returns shape (B, n_ref).
    """
    H = min(n_sample_heads, q_out.shape[1])
    scale = head_dim ** -0.5
    # (B, H, n_out, n_ref) → sum over output queries → (B, H, n_ref) → mean heads → (B, n_ref)
    with torch.no_grad():
        logits = (q_out[:, :H] @ k_ref[:, :H].transpose(-1, -2)) * scale  # (B,H,n_out,n_ref)
        return logits.softmax(dim=-1).sum(dim=-2).mean(dim=1)              # (B, n_ref)


def _ska_replace(
    img_k:       torch.Tensor,
    img_v:       torch.Tensor,
    slice_start: int,
    ref_kvi:     Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    thresh:      float,
    inject_mode: str = "kv",   # "kv" | "k" | "v"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """At the stored token positions (given by idx), replace current K and/or V
    with the stored reference wherever cosine similarity (averaged over heads) < thresh.
    Operates pre-RoPE; img_k/img_v shapes are (B, L, H, d).

    inject_mode="k"  — replace K only (V stays as-is); tests attention routing
    inject_mode="v"  — replace V only (K stays as-is); tests content retrieval
    inject_mode="kv" — replace both (original behaviour)
    Similarity gating always uses K for the comparison regardless of mode.
    """
    ref_k, ref_v, idx = ref_kvi
    ref_k = ref_k.to(img_k.device, img_k.dtype)   # (B, H, n_sel, d)
    ref_v = ref_v.to(img_v.device, img_v.dtype)
    idx   = idx.to(img_k.device)                   # (n_sel,)

    abs_idx = slice_start + idx                    # absolute positions in img_k

    cur_k = img_k[:, abs_idx].permute(0, 2, 1, 3)  # (B, H, n_sel, d)
    cur_v = img_v[:, abs_idx].permute(0, 2, 1, 3)

    # Similarity is always computed on K (K encodes routing; more diagnostic)
    sim     = F.cosine_similarity(cur_k.mean(dim=1), ref_k.mean(dim=1), dim=-1)  # (B, n_sel)
    replace = (sim < thresh)[:, None, :, None].expand_as(cur_k)                  # (B, H, n_sel, d)

    if inject_mode in ("kv", "k"):
        img_k = img_k.clone()
        img_k[:, abs_idx] = torch.where(replace, ref_k, cur_k).permute(0, 2, 1, 3)
    if inject_mode in ("kv", "v"):
        img_v = img_v.clone()
        img_v[:, abs_idx] = torch.where(replace, ref_v, cur_v).permute(0, 2, 1, 3)
    return img_k, img_v


class QwenSKAProcessor:
    """
    Drop-in replacement for QwenDoubleStreamAttnProcessor2_0 on SKA_BLOCKS.

    "normal" : fully delegates to the stored original processor — zero deviation.
    "capture": project K/V → capture scene-conditioning slice → delegate to original
               for the attention computation itself.
    "inject" : project K/V → capture scene slice → extend K/V with stored scene K/V
               → custom SDPA.  Falls back to original if RoPE or SDPA raise.
    """

    def __init__(self, block_idx: int, store: SKAStore, original_processor):
        self.block_idx = block_idx
        self.store     = store
        self.original  = original_processor   # used for normal mode and fallback

    def __call__(
        self,
        attn,
        hidden_states:              torch.Tensor,
        encoder_hidden_states:      torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        image_rotary_emb:           tuple        | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = self.store.mode

        # ── normal mode: delegate entirely — no numerical deviation ───────────
        if mode == "normal":
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )

        # ── capture / inject mode ─────────────────────────────────────────────
        B, L_img, inner_dim = hidden_states.shape
        L_txt    = encoder_hidden_states.shape[1]
        n_heads  = attn.heads
        head_dim = inner_dim // n_heads
        n_out    = self.store.n_out   # tokens per output frame

        # Project → (B, L, H, d) → norm
        img_q = attn.to_q(hidden_states).reshape(B, L_img, n_heads, head_dim)
        img_k = attn.to_k(hidden_states).reshape(B, L_img, n_heads, head_dim)
        img_v = attn.to_v(hidden_states).reshape(B, L_img, n_heads, head_dim)

        if getattr(attn, "norm_q", None) is not None: img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None: img_k = attn.norm_k(img_k)

        txt_q = attn.add_q_proj(encoder_hidden_states).reshape(B, L_txt, n_heads, head_dim)
        txt_k = attn.add_k_proj(encoder_hidden_states).reshape(B, L_txt, n_heads, head_dim)
        txt_v = attn.add_v_proj(encoder_hidden_states).reshape(B, L_txt, n_heads, head_dim)

        if getattr(attn, "norm_added_q", None) is not None: txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None: txt_k = attn.norm_added_k(txt_k)

        # ── CAPTURE — attention-mask-based selective token capture ───────────────
        # Frame layout: [output(0:n_out) | scene_cond(n_out:2*n_out) | obj(2*n_out:3*n_out)]
        #
        # Derive spatial masks from the model's own attention:
        #   bg mask  — scene_cond tokens that output queries attend to most
        #              (output attends strongly to bg tokens it is copying)
        #   obj mask — obj frame tokens that output queries attend to most
        #              (output attends strongly to actual object pixels, not white fill)
        #
        # Only above-average-importance tokens are stored, so captured K/V is
        # dense on signal and sparse on noise/fill. Each entry is (k, v, idx).
        # Store on CPU to avoid holding GPU memory across editing steps.
        if n_out > 0 and L_img >= 2 * n_out:
            q_out = img_q[:, 0:n_out].permute(0, 2, 1, 3)           # (B, H, n_out, d)
            k_bg  = img_k[:, n_out:2 * n_out].permute(0, 2, 1, 3)   # (B, H, n_out, d)

            bg_imp = _attn_importance(q_out, k_bg, head_dim)          # (B, n_out)
            bg_mask = bg_imp > bg_imp.mean(dim=-1, keepdim=True)       # (B, n_out) bool
            bg_idx  = bg_mask[0].nonzero(as_tuple=True)[0]            # (n_sel,)

            self.store.new_kv[self.block_idx] = (
                img_k[:, n_out:2 * n_out][:, bg_idx].permute(0, 2, 1, 3).detach().cpu(),
                img_v[:, n_out:2 * n_out][:, bg_idx].permute(0, 2, 1, 3).detach().cpu(),
                bg_idx.detach().cpu(),
            )

        if n_out > 0 and L_img >= 3 * n_out:
            q_out  = img_q[:, 0:n_out].permute(0, 2, 1, 3)            # (B, H, n_out, d)
            k_obj  = img_k[:, 2 * n_out:3 * n_out].permute(0, 2, 1, 3)

            obj_imp  = _attn_importance(q_out, k_obj, head_dim)        # (B, n_out)
            obj_mask = obj_imp > obj_imp.mean(dim=-1, keepdim=True)
            obj_idx  = obj_mask[0].nonzero(as_tuple=True)[0]

            self.store.new_obj_kv[self.block_idx] = (
                img_k[:, 2 * n_out:3 * n_out][:, obj_idx].permute(0, 2, 1, 3).detach().cpu(),
                img_v[:, 2 * n_out:3 * n_out][:, obj_idx].permute(0, 2, 1, 3).detach().cpu(),
                obj_idx.detach().cpu(),
            )

        # In capture-only mode the attention computation is handled by the original
        # processor (no injection needed, and we avoid any custom SDPA risk).
        if mode == "capture":
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )

        # ── INJECT mode: compare-and-replace drifted tokens, then custom SDPA ─
        # Compare current pre-RoPE K/V for each reference frame against the
        # stored reference.  Tokens whose cosine similarity (averaged over heads)
        # falls below replace_thresh are snapped back to the stored K/V.
        # This happens pre-RoPE so the correction is in the projected feature space.
        thresh = self.store.replace_thresh

        mode = self.store.inject_mode

        # Background: snap drifted bg tokens in scene_cond frame back to reference.
        # Disabled by default (inject_bg=False): injecting bg K/V from a prior denoising
        # trajectory into the current scene_cond frame conflicts with the model's internal
        # state and produces neon glitch artifacts on walls/ceiling.
        if (self.store.inject_bg
                and self.store.has_kv
                and self.block_idx in self.store.kv
                and L_img >= 2 * n_out):
            img_k, img_v = _ska_replace(
                img_k, img_v, n_out,
                self.store.kv[self.block_idx], thresh, mode,
            )

        # Object: snap drifted obj tokens in obj frame back to reference
        if self.store.has_obj_kv and self.block_idx in self.store.obj_kv and L_img >= 3 * n_out:
            img_k, img_v = _ska_replace(
                img_k, img_v, 2 * n_out,
                self.store.obj_kv[self.block_idx], thresh, mode,
            )

        try:
            # RoPE on (B, L, H, d) — BEFORE head transpose
            if image_rotary_emb is not None and _apply_rope is not None:
                img_freqs, txt_freqs = image_rotary_emb
                img_q = _apply_rope(img_q, img_freqs, use_real=False)
                img_k = _apply_rope(img_k, img_freqs, use_real=False)
                txt_q = _apply_rope(txt_q, txt_freqs, use_real=False)
                txt_k = _apply_rope(txt_k, txt_freqs, use_real=False)

            # Transpose to (B, H, L, d) for SDPA
            img_q = img_q.transpose(1, 2)
            img_k = img_k.transpose(1, 2)
            img_v = img_v.transpose(1, 2)
            txt_q = txt_q.transpose(1, 2)
            txt_k = txt_k.transpose(1, 2)
            txt_v = txt_v.transpose(1, 2)

            # Joint attention — sequence length unchanged (no extra token appending)
            joint_q = torch.cat([txt_q, img_q], dim=2)
            joint_k = torch.cat([txt_k, img_k], dim=2)
            joint_v = torch.cat([txt_v, img_v], dim=2)

            attn_mask = None
            if encoder_hidden_states_mask is not None:
                img_ones = torch.ones((B, L_img), dtype=torch.bool, device=hidden_states.device)
                kv_bool  = torch.cat([encoder_hidden_states_mask, img_ones], dim=1)
                attn_mask = (1.0 - kv_bool.float()[:, None, None, :]) * torch.finfo(joint_q.dtype).min

            joint_out = F.scaled_dot_product_attention(
                joint_q, joint_k, joint_v,
                attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
            )   # (B, H, L_txt+L_img, d)

            txt_out = joint_out[:, :, :L_txt, :].transpose(1, 2).reshape(B, L_txt, inner_dim)
            img_out = joint_out[:, :, L_txt:, :].transpose(1, 2).reshape(B, L_img, inner_dim)

            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)
            txt_out = attn.to_add_out(txt_out)

            return img_out, txt_out

        except Exception:
            # RoPE/SDPA failed — fall back to original for this call.
            # Capture already populated new_kv/new_obj_kv, so next step can inject.
            return self.original(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )


def _make_room_prior_mask(
    lat_h:        int,
    lat_w:        int,
    top_anchor:   float = 0.25,   # top fraction to fully anchor (ceiling)
    side_anchor:  float = 0.12,   # side fraction to fully anchor (wall edges)
    temp:         float = 0.04,   # sigmoid transition sharpness
) -> torch.Tensor:
    """Object-region prior for interior rooms: floor/center is free, ceiling/walls anchored.

    Returns (1, 1, lat_h, lat_w) ∈ [0, 1].
      1  = object region  → model generates freely
      0  = background     → anchored to reference scene during denoising
    """
    y = torch.linspace(0.0, 1.0, lat_h)   # 0 = ceiling, 1 = floor
    x = torch.linspace(0.0, 1.0, lat_w)   # 0 = left wall, 1 = right wall
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Vertical: anchor ceiling (top top_anchor fraction), free toward floor
    v = torch.sigmoid((yy - top_anchor) / temp)

    # Horizontal: anchor side walls, free in the middle
    x_from_center = (xx - 0.5).abs()            # 0 at center, 0.5 at edges
    h = torch.sigmoid((0.5 - side_anchor - x_from_center) / temp)

    mask = (v * h).clamp(0.0, 1.0)
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, lat_h, lat_w)


class _DSACallback:
    """Denoising-Step Anchoring (DSA).

    At each denoising step i, blends the current latent with a noise-matched
    version of the reference scene in the background region.  This prevents
    the diffusion model from generating hallucinated content (neon artifacts)
    in unchanged background areas during the denoising process itself.

    Mathematical guarantee:
        z_t_bg  =  (1 - α) · z_t_model  +  α · z_t_ref
        where z_t_ref  =  (1 - σ_t) · z_prev_scaled  +  σ_t · ε,  ε ~ N(0,I)

    The mixing coefficient α is applied only in the background region
    (1 - obj_mask), so the object region is completely free for generation.
    """

    def __init__(
        self,
        z_prev_unscaled: torch.Tensor,   # (1, C, lat_h, lat_w) unscaled VAE latent
        obj_mask:        torch.Tensor,   # (1, 1, lat_h, lat_w) ∈ [0,1]; 1=object, 0=BG
        alpha:           float = 0.8,    # anchor strength in background (0=free, 1=full lock)
        start_step:      int   = 0,      # first step to begin anchoring (0 = all steps)
    ):
        self.z_prev   = z_prev_unscaled.cpu().float()
        self.obj_mask = obj_mask.cpu().float()
        self.alpha    = alpha
        self.start_step = start_step
        self._noise   = None   # same noise reused across steps for coherence

    def __call__(self, pipe, i: int, t, callback_kwargs: dict) -> dict:
        if i < self.start_step:
            return callback_kwargs
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        dev, dtyp = latents.device, latents.dtype
        scaling   = getattr(pipe.vae.config, "scaling_factor", 0.18215)

        # Scale reference latent into the denoiser's space
        z_scaled = (self.z_prev * scaling).to(dev, dtyp)  # (1, C, ref_H, ref_W)

        # Reuse the same noise draw across steps for temporal coherence
        if self._noise is None or self._noise.shape != z_scaled.shape:
            self._noise = torch.randn_like(z_scaled)
        noise = self._noise.to(dev, dtyp)

        # Normalise timestep t → σ ∈ [0, 1]
        T = float(getattr(pipe.scheduler.config, "num_train_timesteps", 1000))
        t_val = float(t.mean()) if t.dim() > 0 else float(t)
        sigma = t_val / T

        # Flow-matching forward process: x_t = (1-σ)·x_0 + σ·ε
        z_ref_noised = (1.0 - sigma) * z_scaled + sigma * noise  # (1, C, ref_H, ref_W)

        # Extract the spatial slice to anchor (frame 0 for 5D video latents)
        is_5d = latents.dim() == 5
        frame = latents[:, :, 0] if is_5d else latents  # (B, C, lat_H, lat_W)

        # Qwen's pipeline denoises in the patchified token space (lat = VAE/2), so
        # lat_H/lat_W may differ from ref_H/ref_W.  Resize both reference and mask
        # to match whatever resolution the callback delivers — no assumptions.
        lat_h, lat_w = frame.shape[-2], frame.shape[-1]
        ref_h, ref_w = z_ref_noised.shape[-2], z_ref_noised.shape[-1]
        if (lat_h, lat_w) != (ref_h, ref_w):
            z_ref_noised = F.interpolate(
                z_ref_noised.float(), size=(lat_h, lat_w),
                mode="bilinear", align_corners=False,
            ).to(dtyp)

        obj_mask_resized = F.interpolate(
            self.obj_mask.float().to(dev), size=(lat_h, lat_w),
            mode="bilinear", align_corners=False,
        ).to(dtyp)
        bg_weight = self.alpha * (1.0 - obj_mask_resized)  # (1, 1, lat_H, lat_W)

        anchored = (1.0 - bg_weight) * frame + bg_weight * z_ref_noised

        if is_5d:
            latents_out = latents.clone()
            latents_out[:, :, 0] = anchored
        else:
            latents_out = anchored

        callback_kwargs["latents"] = latents_out
        return callback_kwargs


class _SKACallback:
    """
    callback_on_step_end: fires after step i completes, sets mode for step i+1.
    Switches from "normal" to the active phase (capture/inject) once σ < gate.
    """
    def __init__(self, store: SKAStore, inject_start: int, phase: str):
        self.store        = store
        self.inject_start = inject_start
        self.phase        = phase   # "capture" or "inject"

    def __call__(self, pipe, i: int, t, callback_kwargs: dict) -> dict:
        if i + 1 >= self.inject_start:
            self.store.mode = self.phase
        return callback_kwargs


def _install_ska(transformer, blocks: list, store: SKAStore) -> dict:
    originals = {}
    for idx in blocks:
        blk = transformer.transformer_blocks[idx]
        orig = blk.attn.processor
        originals[idx] = orig
        blk.attn.set_processor(QwenSKAProcessor(idx, store, orig))
    return originals


def _restore_ska(transformer, originals: dict) -> None:
    for idx, proc in originals.items():
        transformer.transformer_blocks[idx].attn.set_processor(proc)


def run_qwen_ska(
    pipe,
    image,
    prompt:          str,
    seed:            int,
    num_steps:       int,
    guidance:        float,
    height:          int,
    width:           int,
    negative_prompt: str,
    ska_store:       SKAStore | None,
    ska_phase:       str      | None,   # "capture" | "inject" | None (skip)
    ska_gate:        float    = SKA_SIGMA_GATE,
    dsa_callback:    "_DSACallback | None" = None,  # T3 denoising-step anchoring
) -> Image.Image:
    """
    run_qwen with optional SKA (K/V) and/or DSA (denoising-step anchoring).

    ska_phase="capture" — capture scene K/V for future injection; no injection.
    ska_phase="inject"  — inject previous scene K/V + capture current scene K/V.
    ska_phase=None      — plain run_qwen (removal steps, or SKA disabled).
    dsa_callback        — if given, anchors background latent at each denoising step.

    After the call, ska_store.kv is updated via promote().
    """
    # Build the composed callback: SKA (sets processor mode) + DSA (modifies latents)
    ska_active = ska_phase is not None and ska_store is not None

    if ska_active:
        h_tok = height // (pipe.vae_scale_factor * 2)
        w_tok = width  // (pipe.vae_scale_factor * 2)
        ska_store.n_out      = h_tok * w_tok
        ska_store.mode       = "normal"
        ska_store.new_kv     = {}
        ska_store.new_obj_kv = {}
        inject_start = max(1, int(num_steps * (1.0 - ska_gate)) - 1)
        ska_cb = _SKACallback(ska_store, inject_start, ska_phase)
    else:
        ska_cb = None

    # Compose: SKA fires first (sets mode), DSA fires second (edits latents)
    if ska_cb is not None and dsa_callback is not None:
        _ska_ref, _dsa_ref = ska_cb, dsa_callback
        class _Composed:
            def __call__(self, pipe, i, t, kw):
                kw = _ska_ref(pipe, i, t, kw)
                kw = _dsa_ref(pipe, i, t, kw)
                return kw
        callback = _Composed()
    elif ska_cb is not None:
        callback = ska_cb
    elif dsa_callback is not None:
        callback = dsa_callback
    else:
        # Nothing to do — plain run_qwen
        return run_qwen(pipe, image, prompt, seed, num_steps, guidance,
                        height, width, negative_prompt)

    originals = _install_ska(pipe.transformer, SKA_BLOCKS, ska_store) if ska_active else {}
    try:
        gen = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            image=image, num_inference_steps=num_steps,
            true_cfg_scale=guidance,
            height=height, width=width,
            generator=gen,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
    finally:
        if ska_active:
            _restore_ska(pipe.transformer, originals)
            ska_store.promote()

    return result


# ── Main chain ────────────────────────────────────────────────────────────────

def run_chain(
    pipe,
    base:         Image.Image,
    edits:        List[dict],
    sketch_dir:   str,
    seed:         int,
    num_steps:    int,
    guidance:     float,
    obj_guidance: float,
    height:       int,
    width:        int,
    out_dir:      str,
    use_ska:      bool = True,
    ska_gate:     float = SKA_SIGMA_GATE,
) -> List[Image.Image]:
    results   = [base]
    scene     = base
    obj_cache: dict = {}
    ska_store = SKAStore() if use_ska else None

    for i, edit in enumerate(edits):
        name   = edit["name"]
        desc   = edit["description"]
        action = edit.get("action", "insert")
        tag    = f"{'remove' if action == 'remove' else 'step'}{i+1}_{name}"

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}  [{action}]")
        print(f"{'─'*60}")

        if action == "insert":
            if name not in obj_cache:
                sketch_path = os.path.join(sketch_dir, f"{name}.png")
                if not os.path.isfile(sketch_path):
                    sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
                if not os.path.isfile(sketch_path):
                    raise FileNotFoundError(
                        f"Sketch not found: expected '{name}.png' or 'sketch_{name}.png' "
                        f"in {sketch_dir!r}"
                    )
                print(f"  [A] Generating '{desc}' from sketch ...")
                obj_img = generate_object(
                    pipe=pipe, sketch_path=sketch_path, description=desc,
                    seed=seed, num_steps=num_steps, guidance=obj_guidance,
                    height=height, width=width,
                )
                obj_img.save(os.path.join(out_dir, f"obj_gen_{name}.png"))
                obj_cache[name] = obj_img
                print(f"      Saved: obj_gen_{name}.png")
            else:
                print(f"  [A] Reusing cached obj_gen_{name}.png")
                obj_img = obj_cache[name]

            qwen_image = [scene.convert("RGB"), obj_img.convert("RGB")]
            prompt = insert_prompt(desc)
            neg_p  = "blurry, distorted, low quality, watermark, text, artifacts"

            # SKA phase: first insertion → capture only; subsequent → inject + capture
            ska_phase: str | None = None
            if use_ska and ska_store is not None:
                ska_phase = "inject" if ska_store.has_kv else "capture"
            print(f"  [K] Multi-image pass: [scene, obj_ref] → {width}×{height}"
                  + (f"  SKA={ska_phase}" if ska_phase else ""))

        else:  # remove
            qwen_image = scene.convert("RGB")
            prompt = remove_prompt(desc)
            neg_p  = f"blurry, distorted, low quality, watermark, text, artifacts, {desc}"

            # No injection on removal (gives full freedom for background inpainting),
            # but still capture so the post-removal scene becomes the new reference.
            ska_phase = "capture" if (use_ska and ska_store is not None) else None
            print(f"  [K] Removal pass → {width}×{height}"
                  + (f"  SKA={ska_phase}" if ska_phase else ""))

        with open(os.path.join(out_dir, f"prompt_{tag}.txt"), "w") as f:
            f.write(prompt)
        print(f"      {prompt[:120]}...")

        next_scene = run_qwen_ska(
            pipe=pipe, image=qwen_image, prompt=prompt,
            seed=seed, num_steps=num_steps, guidance=guidance,
            height=height, width=width, negative_prompt=neg_p,
            ska_store=ska_store, ska_phase=ska_phase,
            ska_gate=ska_gate,
        )
        next_scene.save(os.path.join(out_dir, f"result_{tag}.png"))
        print(f"      Saved: result_{tag}.png")

        scene = next_scene
        results.append(scene)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Grid helper ────────────────────────────────────────────────────────────────

def save_grid(images, titles, path, ncols=None, figsize_per=(4, 4)):
    n     = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="No-mask reference pipeline for Qwen-Image-Edit")
    p.add_argument("--sketch_dir",      required=True)
    p.add_argument("--hf_token",        required=True)
    p.add_argument("--cache_dir",       default="./models")
    p.add_argument("--out_dir",         default="results/phase1_ref_qwen")
    p.add_argument("--model",           default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--guidance",        type=float, default=None,
                   help="CFG scale — default 1.0 for --lightning, 4.0 otherwise")
    p.add_argument("--obj_guidance",    type=float, default=None,
                   help="Object render CFG — default same as --guidance")
    p.add_argument("--num_steps",       type=int,   default=None,
                   help="Denoising steps — default 8 for --lightning, 50 otherwise")
    p.add_argument("--height",          type=int,   default=1024)
    p.add_argument("--width",           type=int,   default=1024)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--cpu_offload",     action="store_true",
                   help="Force CPU offload (auto-enabled when GPU < 65 GB)")
    p.add_argument("--base_scene",      default=None,
                   help="Path to a pre-existing base scene PNG — skips Step 0 generation")
    p.add_argument("--no_ska",          action="store_true",
                   help="Disable Scene K/V Anchoring")
    p.add_argument("--ska_gate",        type=float, default=None,
                   help="SKA sigma gate — default 0.3; consider 0.5 for 8-step Lightning")
    # Lightning distillation
    p.add_argument("--lightning",       action="store_true",
                   help="Use Lightning distilled LoRA (lightx2v/Qwen-Image-Lightning)")
    p.add_argument("--lightning_steps", type=int, default=8, choices=[4, 8],
                   help="Lightning LoRA variant: 4 or 8 steps (default: 8)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve None sentinels — Lightning changes the sensible defaults
    num_steps    = args.num_steps    if args.num_steps    is not None else (8    if args.lightning else 50)
    guidance     = args.guidance     if args.guidance     is not None else (1.0  if args.lightning else 4.0)
    obj_guidance = args.obj_guidance if args.obj_guidance is not None else guidance
    ska_gate     = args.ska_gate     if args.ska_gate     is not None else SKA_SIGMA_GATE

    print(f"\n{_SEP}")
    print(f"  phase1_ref_qwen  —  No-mask multi-image reference pipeline")
    print(f"{_SEP}")
    print(f"  Model      : {args.model}")
    if args.lightning:
        print(f"  Lightning  : ON  ({args.lightning_steps}-step LoRA from {LIGHTNING_REPO})")
    print(f"  Sketch dir : {args.sketch_dir}")
    print(f"  Output     : {args.width}×{args.height}")
    print(f"  Guidance   : scene={guidance}  obj={obj_guidance}")
    print(f"  Steps      : {num_steps}")
    print(f"  SKA        : {'OFF (--no_ska)' if args.no_ska else f'ON  blocks={SKA_BLOCKS}  σ<{ska_gate}'}")
    print(f"  Out dir    : {args.out_dir}")
    print(f"{_SEP}\n")

    print(f"Loading {args.model} ...")
    pipe = load_pipeline(
        model=args.model, hf_token=args.hf_token,
        cache_dir=args.cache_dir, device=args.device,
        cpu_offload=args.cpu_offload,
        lightning=args.lightning,
        lightning_steps=args.lightning_steps,
    )

    print("\n=== Step 0: Base scene ===")
    base_path = os.path.join(args.out_dir, "base_scene.png")
    if args.base_scene:
        base = Image.open(args.base_scene).convert("RGB").resize(
            (args.width, args.height), Image.Resampling.LANCZOS)
        base.save(base_path)
        print(f"  Loaded: {args.base_scene}")
    else:
        grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
        base = run_qwen(
            pipe=pipe, image=grey, prompt=BASE_PROMPT,
            seed=args.seed, num_steps=num_steps, guidance=guidance,
            height=args.height, width=args.width,
        )
        base.save(base_path)
        print(f"  Saved: base_scene.png")

    results = run_chain(
        pipe=pipe, base=base, edits=EDITS,
        sketch_dir=args.sketch_dir,
        seed=args.seed, num_steps=num_steps,
        guidance=guidance, obj_guidance=obj_guidance,
        height=args.height, width=args.width,
        out_dir=args.out_dir,
        use_ska=not args.no_ska,
        ska_gate=ska_gate,
    )

    labels = ["base"] + [
        f"{'−' if e['action']=='remove' else '+'}{e['name']}" for e in EDITS
    ]
    save_grid(results, labels,
              os.path.join(args.out_dir, "chain_grid.png"),
              ncols=len(results))

    print(f"\n{_SEP}")
    print(f"  Done. Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
