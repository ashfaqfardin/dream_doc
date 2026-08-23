# -*- coding: utf-8 -*-
"""
test_ska_diagnostic.py — SKA Core-Assumption Diagnostic

Hypothesis: injecting scene S1's K/V as extra context into a forward pass
on scene S2 makes the transformer output more similar to S1's output,
especially for background tokens.

If true → SKA reduces feature-level drift → worth implementing.
If false → model already uses scene conditioning fully → SKA is redundant.

── Three single transformer forward passes (NOT a denoising loop) ────────────
  Pass A  (reference) : transformer(S1, t=0.1)  → h_A, capture K/V blocks 50-59
  Pass B  (baseline)  : transformer(S2, t=0.1)  → h_B  (no injection)
  Pass C  (SKA)       : transformer(S2, t=0.1, +stored S1 K/V) → h_C

  S2 = S1 passed through VAE encode → decode → re-encode
       (one lossy round-trip, exactly what happens between editing steps)

── Metrics ──────────────────────────────────────────────────────────────────
  Per spatial token (64×64 grid for 1024×1024 image):
    drift_B  = ||h_B - h_A|| / ||h_A||   (baseline normalised drift)
    drift_C  = ||h_C - h_A|| / ||h_A||   (SKA normalised drift)
    gain     = drift_B - drift_C          (positive = SKA reduces drift)

  Verdict:
    mean(gain) > 0.01  → SKA has measurable effect → implement
    mean(gain) ≤ 0.01  → model already uses scene conditioning fully → skip SKA

── Usage ────────────────────────────────────────────────────────────────────
  python NewWork/KontextEval/test_ska_diagnostic.py \\
      --scene  results/phase1_ref_qwen/base_scene.png \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir  results/ska_diagnostic \\
      --target_blocks 50 51 52 53 54 55 56 57 58 59
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

# ── Try to import Qwen-specific RoPE helper ───────────────────────────────────
try:
    from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
    _HAS_ROPE = True
except ImportError:
    _HAS_ROPE = False
    print("[WARN] apply_rotary_emb_qwen not importable — RoPE will be skipped (results approximate)")


# ── SKA Store ─────────────────────────────────────────────────────────────────

class SKAStore:
    """Shared mutable state passed to all diagnostic processors."""
    def __init__(self):
        self.kv:   Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.mode: str  = "normal"   # "capture" | "inject" | "normal"


# ── Custom Processor ──────────────────────────────────────────────────────────

class QwenSKADiagnosticProcessor:
    """
    Drop-in replacement for QwenDoubleStreamAttnProcessor2_0.
    Reconstructed from source analysis.

    Modes:
      "normal"  — identical to original processor
      "capture" — normal + stores post-RoPE img K, V into SKAStore
      "inject"  — normal + extends joint K, V with stored S1 K, V
    """

    def __init__(self, block_idx: int, store: SKAStore):
        self.block_idx = block_idx
        self.store     = store

    # ── main call ─────────────────────────────────────────────────────────────
    #
    # RoPE order (critical): apply_rotary_emb_qwen expects (B, L, H, d).
    # We therefore norm → RoPE → transpose, NOT project → transpose → norm → RoPE.

    def __call__(
        self,
        attn,
        hidden_states:           torch.Tensor,
        encoder_hidden_states:   torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        image_rotary_emb:        tuple | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, L_img, inner_dim = hidden_states.shape
        L_txt    = encoder_hidden_states.shape[1]
        n_heads  = attn.heads
        head_dim = inner_dim // n_heads

        # ── image stream: project → (B, L, H, d) ─────────────────────────────
        img_q = attn.to_q(hidden_states).reshape(B, L_img, n_heads, head_dim)
        img_k = attn.to_k(hidden_states).reshape(B, L_img, n_heads, head_dim)
        img_v = attn.to_v(hidden_states).reshape(B, L_img, n_heads, head_dim)

        # norm on last dim (head_dim) — works for any prefix shape
        if getattr(attn, "norm_q", None) is not None: img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None: img_k = attn.norm_k(img_k)

        # ── text stream: project → (B, L_txt, H, d) ──────────────────────────
        txt_q = attn.add_q_proj(encoder_hidden_states).reshape(B, L_txt, n_heads, head_dim)
        txt_k = attn.add_k_proj(encoder_hidden_states).reshape(B, L_txt, n_heads, head_dim)
        txt_v = attn.add_v_proj(encoder_hidden_states).reshape(B, L_txt, n_heads, head_dim)

        if getattr(attn, "norm_added_q", None) is not None: txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None: txt_k = attn.norm_added_k(txt_k)

        # ── RoPE on (B, L, H, d) — BEFORE head transpose ─────────────────────
        if image_rotary_emb is not None and _HAS_ROPE:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        # ── transpose to (B, H, L, d) for SDPA ───────────────────────────────
        img_q = img_q.transpose(1, 2)
        img_k = img_k.transpose(1, 2)
        img_v = img_v.transpose(1, 2)
        txt_q = txt_q.transpose(1, 2)
        txt_k = txt_k.transpose(1, 2)
        txt_v = txt_v.transpose(1, 2)

        # ── CAPTURE: store post-RoPE, post-transpose K, V ─────────────────────
        if self.store.mode == "capture":
            self.store.kv[self.block_idx] = (
                img_k.detach().clone(),
                img_v.detach().clone(),
            )

        # ── INJECT: extend K, V with stored S1 K, V ──────────────────────────
        n_stored = 0
        if self.store.mode == "inject" and self.block_idx in self.store.kv:
            s1_k, s1_v = self.store.kv[self.block_idx]
            s1_k = s1_k.to(img_k.device, img_k.dtype)
            s1_v = s1_v.to(img_v.device, img_v.dtype)
            img_k = torch.cat([img_k, s1_k], dim=2)   # (B, H, L_img+n_stored, d)
            img_v = torch.cat([img_v, s1_v], dim=2)
            n_stored = s1_k.shape[2]

        # ── joint attention (B, H, L_q, d) ───────────────────────────────────
        joint_q = torch.cat([txt_q, img_q], dim=2)   # queries never extended
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        # additive mask (0 = attend, -inf = masked)
        attn_mask = None
        if encoder_hidden_states_mask is not None:
            img_ones = torch.ones((B, L_img), dtype=torch.bool, device=hidden_states.device)
            kv_bool  = torch.cat([encoder_hidden_states_mask, img_ones], dim=1)
            if n_stored > 0:
                kv_bool = torch.cat([
                    kv_bool,
                    torch.ones((B, n_stored), dtype=torch.bool, device=hidden_states.device),
                ], dim=1)
            kv_float  = kv_bool.float()[:, None, None, :]
            attn_mask = (1.0 - kv_float) * torch.finfo(joint_q.dtype).min

        joint_out = F.scaled_dot_product_attention(
            joint_q, joint_k, joint_v,
            attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )   # (B, H, L_txt+L_img, d)

        # ── split, un-transpose, project ──────────────────────────────────────
        txt_out = joint_out[:, :, :L_txt, :].transpose(1, 2).reshape(B, L_txt, inner_dim)
        img_out = joint_out[:, :, L_txt:L_txt + L_img, :].transpose(1, 2).reshape(B, L_img, inner_dim)

        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)
        txt_out = attn.to_add_out(txt_out)

        return img_out, txt_out


# ── Processor management ──────────────────────────────────────────────────────

def install_processors(
    transformer,
    target_blocks: List[int],
    store: SKAStore,
) -> Dict[int, object]:
    """Replace processors on target blocks, return originals for restoration."""
    originals = {}
    for idx in target_blocks:
        block = transformer.transformer_blocks[idx]
        originals[idx] = block.attn.processor
        block.attn.set_processor(QwenSKADiagnosticProcessor(idx, store))
    return originals


def restore_processors(transformer, originals: Dict[int, object]):
    for idx, proc in originals.items():
        transformer.transformer_blocks[idx].attn.set_processor(proc)


# ── Block output capture ──────────────────────────────────────────────────────

def register_output_hooks(
    transformer,
    target_blocks: List[int],
    outputs: Dict[int, torch.Tensor],
) -> List:
    """
    Capture image-stream hidden states AFTER each target block.
    Block returns (encoder_hidden_states, hidden_states) = (txt, img).
    """
    handles = []
    for idx in target_blocks:
        def _hook(module, inp, out, _idx=idx):
            # out = (txt_hidden, img_hidden)
            outputs[_idx] = out[1].detach().cpu()
        h = transformer.transformer_blocks[idx].register_forward_hook(_hook)
        handles.append(h)
    return handles


def remove_hooks(handles: List):
    for h in handles:
        h.remove()


# ── Image → packed latent ─────────────────────────────────────────────────────

def _preprocess_image(pipe, img: Image.Image, device, dtype) -> torch.Tensor:
    """PIL Image → (1, 3, H, W) float tensor normalised to [-1, 1]."""
    from diffusers.image_processor import VaeImageProcessor
    proc = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)
    return proc.preprocess(img).to(device, dtype)


def _vae_encode(pipe, tensor_4d: torch.Tensor) -> torch.Tensor:
    """
    (1, 3, H, W) → (1, C, 1, H/8, W/8) raw VAE latent.
    Qwen uses a 3D causal VAE that expects (B, C, T, H, W).
    For a single image T=1.
    """
    with torch.no_grad():
        latent = pipe.vae.encode(tensor_4d.unsqueeze(2)).latent_dist.mean
    # Handle VAEs that may return 4D (B, C, H, W) for T=1 input
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    return latent  # (1, C, 1, H/8, W/8)


def _normalise_latent(pipe, latent: torch.Tensor) -> torch.Tensor:
    """Per-channel VAE latent normalisation: (1, C, F, H, W) → normalised."""
    device, dtype = latent.device, latent.dtype
    lat_mean = torch.tensor(pipe.vae.config.latents_mean,
                            device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    lat_std  = torch.tensor(pipe.vae.config.latents_std,
                            device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return (latent - lat_mean) / lat_std


def _pack_latent(latent: torch.Tensor) -> torch.Tensor:
    """
    2×2 spatial packing: (1, C, F, H_lat, W_lat) → (1, F*h_tok*w_tok, C*4).
    Matches Qwen pipeline's packed token format.
    """
    _, C, F, H_lat, W_lat = latent.shape
    h_tok, w_tok = H_lat // 2, W_lat // 2
    # reshape into (1, C, F, h_tok, 2, w_tok, 2)
    x = latent.reshape(1, C, F, h_tok, 2, w_tok, 2)
    # permute to (1, F, h_tok, w_tok, C, 2, 2)  — note: dim indices (B=0,C=1,F=2,h=3,2h=4,w=5,2w=6)
    x = x.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
    return x.reshape(1, F * h_tok * w_tok, C * 4)


def encode_image(pipe, img: Image.Image, device, dtype) -> torch.Tensor:
    """Image → normalised packed latent (1, n_tok, C*4)."""
    tensor  = _preprocess_image(pipe, img, device, dtype)
    latent  = _vae_encode(pipe, tensor)
    latent  = _normalise_latent(pipe, latent)
    return _pack_latent(latent)


def vae_roundtrip(pipe, img: Image.Image, device, dtype) -> Image.Image:
    """S2: one VAE encode → decode round-trip (simulates quality loss between editing steps)."""
    tensor = _preprocess_image(pipe, img, device, dtype)
    latent = _vae_encode(pipe, tensor)   # (1, C, 1, H/8, W/8)  — raw, not normalised

    with torch.no_grad():
        decoded = pipe.vae.decode(latent).sample   # (1, 3, 1, H, W) or (1, 3, H, W)

    if decoded.ndim == 5:
        decoded = decoded.squeeze(2)    # → (1, 3, H, W)

    decoded = decoded.clamp(-1, 1)
    decoded = ((decoded + 1) / 2 * 255).byte()
    decoded = decoded[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(decoded)


# ── Single transformer forward pass ──────────────────────────────────────────

@torch.no_grad()
def transformer_forward(
    pipe,
    packed_latent: torch.Tensor,    # (1, n_tok, 64)  clean latent
    prompt_embeds: torch.Tensor,
    prompt_mask:   torch.Tensor | None,
    height: int, width: int,
    sigma: float = 0.3,             # noise level: 0=clean, 1=pure noise
    noise_seed: int = 42,
) -> torch.Tensor:
    """
    One transformer forward pass at noise level σ.
    Adds the same noise (seeded) to both S1 and S2 so drift comparison is fair.
    Returns output packed tensor (1, n_tok, 64).
    """
    device = packed_latent.device
    dtype  = packed_latent.dtype

    # Add noise: flow matching x_t = (1-σ)*x_0 + σ*ε
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn(packed_latent.shape, generator=gen, device=device, dtype=dtype)
    noisy = (1.0 - sigma) * packed_latent + sigma * noise

    h_tok = height // (pipe.vae_scale_factor * 2)
    w_tok = width  // (pipe.vae_scale_factor * 2)
    img_shapes = [[(1, h_tok, w_tok)]]

    timestep = torch.tensor([sigma], device=device, dtype=dtype)

    kwargs = dict(
        hidden_states              = noisy,
        timestep                   = timestep,
        encoder_hidden_states      = prompt_embeds,
        img_shapes                 = img_shapes,
        return_dict                = False,
    )
    if prompt_mask is not None:
        kwargs["encoder_hidden_states_mask"] = prompt_mask

    out = pipe.transformer(**kwargs)[0]   # (1, n_tok, 64)
    return out


# ── Metric computation ────────────────────────────────────────────────────────

def per_token_drift(h_x: torch.Tensor, h_ref: torch.Tensor) -> np.ndarray:
    """
    Normalised L2 drift per token.
    h_x, h_ref: (n_tok, d)
    Returns: (n_tok,) float32
    """
    diff = (h_x - h_ref).float()
    norm = h_ref.float().norm(dim=-1).clamp(min=1e-6)
    return (diff.norm(dim=-1) / norm).numpy()


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(
    drift_B:  np.ndarray,   # (n_tok,) baseline drift per token (after FINAL block)
    drift_C:  np.ndarray,   # (n_tok,) SKA drift per token
    block_gains: Dict[int, float],   # block_idx → mean gain
    scene_s1: Image.Image,
    scene_s2: Image.Image,
    out_dir: str,
    grid_h: int = 64,
    grid_w: int = 64,
):
    gain = drift_B - drift_C           # positive = SKA helped

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flat

    # 0: S1
    axes[0].imshow(scene_s1)
    axes[0].set_title("S1 (reference)")
    axes[0].axis("off")

    # 1: S2
    axes[1].imshow(scene_s2)
    axes[1].set_title("S2 (VAE round-trip)")
    axes[1].axis("off")

    # 2: baseline drift map
    im2 = axes[2].imshow(
        drift_B.reshape(grid_h, grid_w),
        cmap="hot", vmin=0,
    )
    axes[2].set_title("Baseline drift B vs A\n(higher = worse)")
    plt.colorbar(im2, ax=axes[2])

    # 3: SKA drift map
    im3 = axes[3].imshow(
        drift_C.reshape(grid_h, grid_w),
        cmap="hot", vmin=0,
    )
    axes[3].set_title("SKA drift C vs A\n(lower = better)")
    plt.colorbar(im3, ax=axes[3])

    # 4: gain map  (drift_B - drift_C, blue = SKA helps, red = SKA hurts)
    max_abs = max(abs(gain).max(), 1e-6)
    im4 = axes[4].imshow(
        gain.reshape(grid_h, grid_w),
        cmap="RdBu", vmin=-max_abs, vmax=max_abs,
    )
    axes[4].set_title("Gain = drift_B − drift_C\n(blue = SKA helps, red = SKA hurts)")
    plt.colorbar(im4, ax=axes[4])

    # 5: per-block bar chart
    blocks = sorted(block_gains.keys())
    gains  = [block_gains[b] for b in blocks]
    colors = ["steelblue" if g > 0 else "salmon" for g in gains]
    axes[5].bar([str(b) for b in blocks], gains, color=colors)
    axes[5].axhline(0, color="black", linewidth=0.8)
    axes[5].set_xlabel("Block index")
    axes[5].set_ylabel("Mean gain (drift_B − drift_C)")
    axes[5].set_title("SKA gain per block\n(positive = helps)")

    plt.tight_layout()
    path = os.path.join(out_dir, "ska_diagnostic.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Saved: {path}")

    # ── verdict ───────────────────────────────────────────────────────────────
    mean_gain   = float(gain.mean())
    pct_helped  = float((gain > 0).mean()) * 100

    print(f"\n{'═'*55}")
    print(f"  SKA DIAGNOSTIC VERDICT")
    print(f"{'─'*55}")
    print(f"  Mean baseline drift : {drift_B.mean():.4f}")
    print(f"  Mean SKA drift      : {drift_C.mean():.4f}")
    print(f"  Mean gain           : {mean_gain:+.4f}")
    print(f"  Tokens where SKA helps : {pct_helped:.1f}%")
    print(f"{'─'*55}")
    if mean_gain > 0.01:
        print(f"  VERDICT: SKA has measurable effect → IMPLEMENT")
    elif mean_gain > 0:
        print(f"  VERDICT: SKA has marginal effect → TUNE sigma gate first")
    else:
        print(f"  VERDICT: SKA shows no effect → model already uses scene")
        print(f"           conditioning fully. Consider alternative approaches.")
    print(f"{'═'*55}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",         default=None,
                   help="Path to scene image (S1). If omitted, generates one from --scene_prompt.")
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--cache_dir",     default="./models")
    p.add_argument("--out_dir",       default="results/ska_diagnostic")
    p.add_argument("--model",         default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--prompt",        default="A room with a wooden floor and white walls.")
    p.add_argument("--scene_prompt",  default=(
                       "An empty room with a wooden floor, white walls, "
                       "and a window letting in natural light."),
                   help="Text prompt used to generate S1 when --scene is not provided.")
    p.add_argument("--timestep",      type=float, default=0.1,
                   help="Timestep fed to transformer (0=clean, 1=pure noise). "
                        "Use 0.1 for nearly-clean features.")
    p.add_argument("--target_blocks", type=int, nargs="+",
                   default=list(range(50, 60)),
                   help="Transformer block indices to install SKA on (default: 50-59)")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--cpu_offload",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    dtype  = torch.bfloat16

    print(f"\n{'═'*55}")
    print(f"  SKA Diagnostic")
    print(f"{'─'*55}")
    print(f"  Scene   : {args.scene}")
    print(f"  Blocks  : {args.target_blocks}")
    print(f"  Timestep: {args.timestep}")
    print(f"{'═'*55}\n")

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print("Loading pipeline ...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model, torch_dtype=dtype,
        token=args.hf_token, cache_dir=args.cache_dir,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # ── Prepare S1 ───────────────────────────────────────────────────────────
    if args.scene is not None:
        print(f"Loading S1 from {args.scene} ...")
        scene_s1 = Image.open(args.scene).convert("RGB").resize(
            (args.width, args.height), Image.Resampling.LANCZOS
        )
    else:
        print("No --scene provided. Generating base scene with Qwen ...")
        grey     = Image.new("RGB", (args.width, args.height), (200, 200, 190))
        gen      = torch.Generator(device=device).manual_seed(42)
        scene_s1 = pipe(
            prompt           = args.scene_prompt,
            negative_prompt  = "blurry, distorted, low quality, watermark, text",
            image            = grey,
            num_inference_steps = 30,
            true_cfg_scale   = 4.0,
            height           = args.height,
            width            = args.width,
            generator        = gen,
        ).images[0]
        print("  Base scene generated.")

    print("Preparing S2 (VAE round-trip) ...")
    scene_s2 = vae_roundtrip(pipe, scene_s1, device, dtype)

    scene_s1.save(os.path.join(args.out_dir, "s1_input.png"))
    scene_s2.save(os.path.join(args.out_dir, "s2_roundtrip.png"))
    print(f"  S1 saved: s1_input.png")
    print(f"  S2 saved: s2_roundtrip.png  (one VAE encode→decode round-trip)")

    # ── Encode both images ────────────────────────────────────────────────────
    print("Encoding images to latents ...")
    z1 = encode_image(pipe, scene_s1, device, dtype)   # (1, n_tok, 64)
    z2 = encode_image(pipe, scene_s2, device, dtype)
    print(f"  Latent shape: {z1.shape}  (tokens: {z1.shape[1]})")

    # ── Encode prompt ─────────────────────────────────────────────────────────
    print("Encoding prompt ...")
    import inspect as _inspect
    _ep_sig = _inspect.signature(pipe.encode_prompt).parameters

    _ep_kwargs: dict = {"prompt": args.prompt}
    if "device"                     in _ep_sig: _ep_kwargs["device"]                     = device
    if "num_images_per_prompt"      in _ep_sig: _ep_kwargs["num_images_per_prompt"]      = 1
    if "do_classifier_free_guidance" in _ep_sig: _ep_kwargs["do_classifier_free_guidance"] = False
    if "negative_prompt"            in _ep_sig: _ep_kwargs["negative_prompt"]            = ""

    with torch.no_grad():
        enc_result = pipe.encode_prompt(**_ep_kwargs)

    # Wan/Qwen pipelines return varying shapes; grab embeds + mask defensively
    enc_result = enc_result if isinstance(enc_result, (list, tuple)) else (enc_result,)
    prompt_embeds = enc_result[0]
    prompt_mask   = enc_result[2] if len(enc_result) >= 3 else None
    print(f"  Prompt embed shape: {prompt_embeds.shape}")

    # ── Set up SKA store and install processors ───────────────────────────────
    store      = SKAStore()
    originals  = install_processors(pipe.transformer, args.target_blocks, store)

    sigma      = args.timestep   # reuse --timestep as σ
    noise_seed = 42

    # ── PASS A: reference (S1), capture mode ─────────────────────────────────
    print(f"\n[PASS A] Reference forward on S1 (capture mode, σ={sigma}) ...")
    store.mode = "capture"
    block_outs_A: Dict[int, torch.Tensor] = {}
    hooks_A = register_output_hooks(pipe.transformer, args.target_blocks, block_outs_A)

    h_A = transformer_forward(
        pipe, z1, prompt_embeds, prompt_mask,
        args.height, args.width, sigma=sigma, noise_seed=noise_seed,
    )
    remove_hooks(hooks_A)
    print(f"  Captured K/V at {len(store.kv)} blocks: {sorted(store.kv.keys())}")
    print(f"  Output shape: {h_A.shape}")

    # ── PASS B: baseline (S2), normal mode ───────────────────────────────────
    print(f"\n[PASS B] Baseline forward on S2 (no injection, σ={sigma}) ...")
    store.mode = "normal"
    block_outs_B: Dict[int, torch.Tensor] = {}
    hooks_B = register_output_hooks(pipe.transformer, args.target_blocks, block_outs_B)

    h_B = transformer_forward(
        pipe, z2, prompt_embeds, prompt_mask,
        args.height, args.width, sigma=sigma, noise_seed=noise_seed,
    )
    remove_hooks(hooks_B)
    print(f"  Output shape: {h_B.shape}")

    # ── PASS C: SKA (S2 + stored S1 K/V) ─────────────────────────────────────
    print(f"\n[PASS C] SKA forward on S2 (inject stored S1 K/V, σ={sigma}) ...")
    store.mode = "inject"
    block_outs_C: Dict[int, torch.Tensor] = {}
    hooks_C = register_output_hooks(pipe.transformer, args.target_blocks, block_outs_C)

    h_C = transformer_forward(
        pipe, z2, prompt_embeds, prompt_mask,
        args.height, args.width, sigma=sigma, noise_seed=noise_seed,
    )
    remove_hooks(hooks_C)
    print(f"  Output shape: {h_C.shape}")

    # ── Restore original processors ───────────────────────────────────────────
    restore_processors(pipe.transformer, originals)

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing metrics ...")

    # Final output drift (noise prediction, all tokens)
    drift_B = per_token_drift(h_B[0].cpu(), h_A[0].cpu())
    drift_C = per_token_drift(h_C[0].cpu(), h_A[0].cpu())

    # Per-block intermediate hidden state gain
    block_gains: Dict[int, float] = {}
    for blk in args.target_blocks:
        if blk in block_outs_A and blk in block_outs_B and blk in block_outs_C:
            d_B = per_token_drift(block_outs_B[blk][0], block_outs_A[blk][0])
            d_C = per_token_drift(block_outs_C[blk][0], block_outs_A[blk][0])
            block_gains[blk] = float((d_B - d_C).mean())

    print(f"  Final output — baseline drift: {drift_B.mean():.4f} | SKA drift: {drift_C.mean():.4f}")
    print(f"  Per-block gains: { {k: f'{v:+.4f}' for k, v in sorted(block_gains.items())} }")

    # ── Grid size for spatial reshape ─────────────────────────────────────────
    grid_h = args.height // (pipe.vae_scale_factor * 2)
    grid_w = args.width  // (pipe.vae_scale_factor * 2)

    # ── Save per-block gain maps ──────────────────────────────────────────────
    for blk in sorted(block_gains.keys()):
        if blk in block_outs_A and blk in block_outs_B and blk in block_outs_C:
            d_B = per_token_drift(block_outs_B[blk][0], block_outs_A[blk][0])
            d_C = per_token_drift(block_outs_C[blk][0], block_outs_A[blk][0])
            gain_map = (d_B - d_C).reshape(grid_h, grid_w)
            fig, ax = plt.subplots(figsize=(5, 5))
            mx = max(abs(gain_map).max(), 1e-6)
            ax.imshow(gain_map, cmap="RdBu", vmin=-mx, vmax=mx)
            ax.set_title(f"Block {blk} gain  (mean={block_gains[blk]:+.4f})")
            ax.axis("off")
            plt.colorbar(plt.cm.ScalarMappable(
                norm=plt.Normalize(-mx, mx), cmap="RdBu"), ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"block_{blk}_gain.png"),
                        dpi=100, bbox_inches="tight")
            plt.close()

    # ── Main summary plot ─────────────────────────────────────────────────────
    plot_results(drift_B, drift_C, block_gains, scene_s1, scene_s2,
                 args.out_dir, grid_h, grid_w)


if __name__ == "__main__":
    main()
