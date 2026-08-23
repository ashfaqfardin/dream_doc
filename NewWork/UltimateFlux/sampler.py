"""
Shared dual-branch FLUX.1-dev sampler for UltimateFlux.

All tasks share a B=2 denoising loop:
  batch index 0 = source/content branch  (unmodified)
  batch index 1 = edit/styled branch     (receives injections from policy)

InjectionPolicy controls Q/K/V manipulation and block-level hooks at each (layer, step).

Layer index convention (combined 0-based across all 57 blocks):
  double-stream:  0–18  → pipe.transformer.transformer_blocks[i]
  single-stream: 19–56  → pipe.transformer.single_transformer_blocks[i-19]
"""

import os
import sys
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageDraw
from typing import Optional, List, Tuple, Dict, Callable

# ── Layer tier constants (from FreeFlux ICCV 2025, validated on FLUX.1-dev) ──
# Tier A — Content-similarity-dependent layers: appearance, style, texture, object
# Source: FreeFlux non_rigid_attn_utils.py layer_idx (RoPE frequency analysis)
TIER_A = [0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56]

# HOTSPOT — Position-dependent layers (high RoPE frequency): spatial layout
# Source: FreeFlux RoPE frequency analysis
HOTSPOT = [1, 2, 4, 26, 30, 54, 55]

# Tier B — All layers: used when shape/pose deformation is needed
TIER_B = list(range(57))

N_DOUBLE = 19
N_SINGLE = 38
N_LAYERS = N_DOUBLE + N_SINGLE  # 57


# ─────────────────────────────── Base Policy ─────────────────────────────────

class BasePolicy:
    """
    Abstract injection policy.  Subclasses override the three entry points.
    """

    def inject_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int,
        step: int,
        n_steps: int,
        txt_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Called inside every attention layer.  Modify q/k/v for edit branch."""
        return q, k, v

    def inject_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int,
        step: int,
        n_steps: int,
        txt_len: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Optional custom attention computation.  Called AFTER inject_qkv with the
        (potentially modified) q/k/v.

        Return the attention output (B, H, L, D) to replace SDPA, or None to
        let SDPA run as usual.  Policies that need to manipulate pre-softmax
        attention scores (e.g. ColorCtrl) override this method.
        """
        return None

    def get_block_hooks(self) -> Dict[int, Callable]:
        """
        Return {block_idx: hook_fn} for transformer block forward hooks.
        Hooks are registered before generation and removed after.
        block_idx uses the same combined 0-56 convention as layer tiers.
        """
        return {}

    def pre_generate(self, pipe: FluxPipeline, **kwargs) -> None:
        """Called once before the denoising loop (style extraction, inversion, …)."""
        pass

    def observe_attn_out(
        self,
        out: torch.Tensor,
        layer: int,
        step: int,
        txt_len: int = 0,
    ) -> None:
        """
        Called after SDPA with the raw (B,H,L,D) attention output.
        Used by SynPS (NonRigidPolicy) to accumulate M_t measurements.
        Default is a no-op; override in policies that need post-attention state.
        """
        pass

    def post_step(
        self,
        latents: torch.Tensor,
        step: int,
        n_steps: int,
    ) -> torch.Tensor:
        """
        Called after each denoising step with packed latents (B, S, C_packed).
        B=2: index 0 = source branch, index 1 = edit branch.
        Return the (possibly modified) latents.
        Default is a no-op; override for per-step latent guidance.
        """
        return latents

    def post_process(self, src_img: Image.Image, edit_img: Image.Image) -> Image.Image:
        """Called after generation with the decoded PIL images.  Default: no-op."""
        return edit_img


# ─────────────────────────── Attention Processor ─────────────────────────────

class UltimateFluxAttnProcessor:
    """
    Drop-in replacement for FluxAttnProcessor2_0 that delegates Q/K/V
    manipulation to a pluggable BasePolicy.

    Maintains a (cur_step, cur_att_layer) counter identical to FreeFlux's
    processors so policies can gate on specific (layer, step) pairs.
    """

    def __init__(self, policy: BasePolicy, total_layers: int = N_LAYERS, total_steps: int = 4):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("UltimateFluxAttnProcessor requires PyTorch 2.0+")
        self.policy = policy
        self.total_layers = total_layers
        self.total_steps = total_steps
        self.cur_step = 0
        self.cur_att_layer = 0

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def _tick(self):
        self.cur_att_layer += 1
        if self.cur_att_layer == self.total_layers:
            self.cur_att_layer = 0
            self.cur_step = (self.cur_step + 1) % self.total_steps

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        out = self._forward(attn, hidden_states, encoder_hidden_states,
                            attention_mask, image_rotary_emb)
        self._tick()
        return out

    def _forward(self, attn, hidden_states, encoder_hidden_states,
                 attention_mask, image_rotary_emb):
        layer = self.cur_att_layer
        step  = self.cur_step
        is_double = encoder_hidden_states is not None
        batch_size = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        inner_dim = k.shape[-1]
        head_dim  = inner_dim // attn.heads

        q = q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        txt_len = 0
        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            txt_len = eq.shape[2]          # number of text tokens in the prefix
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        # Save pre-RoPE K for SynPS w-scaled injection (accessed via policy._raw_k).
        k_raw = k
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)
        self.policy._raw_k      = k_raw
        self.policy._rotary_emb = image_rotary_emb

        # ── Policy injection ──────────────────────────────────────────────
        q, k, v = self.policy.inject_qkv(q, k, v, layer, step, self.total_steps, txt_len)

        # Policy may override attention computation entirely (e.g. ColorCtrl needs
        # to manipulate pre-softmax scores, which SDPA does not expose).
        custom_out = self.policy.inject_attention(q, k, v, layer, step, self.total_steps, txt_len)
        if custom_out is not None:
            out = custom_out  # (B, H, L, D) — same shape as SDPA output
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                                 attn_mask=attention_mask)
        # SynPS: accumulate per-block S_img/S_txt for M_t (no-op for other policies)
        self.policy.observe_attn_out(out, layer, step, txt_len)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[0](out)
            out     = attn.to_out[1](out)
            enc_out = attn.to_add_out(enc_out)
            return out, enc_out

        return out


# ────────────────────────── Pipeline helpers ──────────────────────────────────

def load_pipeline(
    model_path: str = "black-forest-labs/FLUX.1-dev",
    hf_token: Optional[str] = None,
    device: str = "cuda",
    cpu_offload: bool = False,
    cache_dir: str = "./models",
) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


@torch.no_grad()
def generate_dual_branch(
    pipe: FluxPipeline,
    policy: BasePolicy,
    source_prompt: str,
    edit_prompt: str,
    seed: int = 0,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
    save_intermediates: bool = False,
    intermediate_out_dir: Optional[str] = None,
    intermediate_every: int = 4,
    save_strips: bool = False,
) -> Tuple[Image.Image, Image.Image]:
    """
    Dual-branch denoising with injections from policy.

    Returns (source_image, edit_image).  source_image is the unmodified
    reconstruction; edit_image is the policy-steered output.
    """
    # 1. Policy pre-hook (style extraction, reasoning pass, inversion, etc.)
    policy.pre_generate(
        pipe,
        device=device,
        height=height,
        width=width,
        num_steps=num_steps,
        seed=seed,
        source_prompt=source_prompt,
        edit_prompt=edit_prompt,
        max_sequence_length=max_sequence_length,
        guidance_scale=guidance_scale,
    )

    # 2. Shared initial latent — both branches MUST start from identical noise.
    #    Without this, pipe(prompt=[A, B]) generates different noise for A and B
    #    (sequential samples from the same generator), so K,V injection between
    #    branches fights completely different initial conditions and produces garbage.
    #    FreeFlux's run_non_rigid.py does exactly this: generate one latent, expand.
    exec_device = getattr(pipe, '_execution_device', device)
    num_ch  = pipe.transformer.config.in_channels // 4   # 16 for FLUX
    lat_h   = height // 8
    lat_w   = width  // 8

    # Policy may provide a content-image starting latent (image-anchored style transfer).
    # If so, use it; otherwise fall back to seeded random noise as before.
    override_latent    = getattr(policy, '_initial_latent',     None)
    override_sigmas    = getattr(policy, '_override_sigmas',    None)
    override_timesteps = getattr(policy, '_override_timesteps', None)
    actual_num_steps   = getattr(policy, '_actual_num_steps',   num_steps)

    # Detect three-branch (StyleAligned) mode: policy has both a source latent
    # (_initial_latent) and a parallel style latent (_style_initial_latent).
    # Layout: branch 0 = source (identity anchor), 1 = style (live K/V donor),
    #         2 = edit (receives K_src + V_style blended by inject_qkv).
    override_style_latent = getattr(policy, '_style_initial_latent', None)
    three_branch = override_latent is not None and override_style_latent is not None

    if override_latent is not None:
        # encode_real_image returns PACKED latents [1, seq, C_packed] (3D).
        # Expand batch dim with exactly 3 dims — using 4 dims on a 3D tensor
        # silently prepends a phantom dim and makes FluxPipeline re-pack wrongly.
        src = override_latent.to(exec_device, dtype=torch.bfloat16)
        if three_branch:
            # [src, sty, edit_start=src_copy] → shape [3, seq, C]
            sty = override_style_latent.to(exec_device, dtype=torch.bfloat16)
            shared_latents = torch.cat(
                [src.expand(1, -1, -1).clone(),
                 sty.expand(1, -1, -1).clone(),
                 src.expand(1, -1, -1).clone()],
                dim=0,
            )
        else:
            shared_latents = src.expand(2, -1, -1).clone()   # [1, seq, C] → [2, seq, C]
    else:
        _g   = torch.Generator(device=exec_device).manual_seed(seed)
        _one = randn_tensor(
            (1, num_ch, lat_h, lat_w),
            generator=_g,
            device=exec_device,
            dtype=torch.bfloat16,
        )
        # Pack: (B, C, lat_h, lat_w) → (B, lat_h//2 * lat_w//2, C*4)
        shared_latents = (
            _one.expand(2, -1, -1, -1).clone()
            .view(2, num_ch, lat_h // 2, 2, lat_w // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(2, (lat_h // 2) * (lat_w // 2), num_ch * 4)
        )

    # 3. Install custom attention processor
    proc = UltimateFluxAttnProcessor(
        policy=policy, total_layers=N_LAYERS, total_steps=actual_num_steps
    )
    proc.reset()
    pipe.transformer.set_attn_processor(proc)

    # 4. Register block-level forward hooks
    handles: List = []
    for block_idx, hook_fn in policy.get_block_hooks().items():
        if block_idx < N_DOUBLE:
            block = pipe.transformer.transformer_blocks[block_idx]
        else:
            block = pipe.transformer.single_transformer_blocks[block_idx - N_DOUBLE]
        handles.append(block.register_forward_hook(hook_fn))

    # The step-end callback serves three purposes:
    #   1. Policy post_step: per-step latent guidance (identity, etc.)
    #   2. SAM2 mask building: decode edit latent → segment → set policy._mask
    #   3. Intermediate capture: store latents for grid visualisation
    use_sam2_mask    = getattr(policy, "use_sam2", False)
    use_post_step    = type(policy).post_step is not BasePolicy.post_step
    needs_callback   = save_intermediates or use_sam2_mask or use_post_step

    captured: List[Tuple[int, int, torch.Tensor]] = []

    def _capture(pipe_ref, step_index, timestep, callback_kwargs):
        lats = callback_kwargs["latents"]

        # Per-step latent guidance (identity frequency blending, etc.)
        lats = policy.post_step(lats, step_index, num_steps)
        callback_kwargs["latents"] = lats

        # SAM2: fires as soon as attention scores are stored (after mask_build_step)
        if (use_sam2_mask
                and not getattr(policy, "_mask_built", True)
                and getattr(policy, "_edit_score_map", None) is not None):
            _build_sam2_mask(pipe, policy, lats, height, width)

        if save_intermediates and (step_index % intermediate_every == 0
                                   or step_index == actual_num_steps - 1):
            captured.append((step_index, actual_num_steps, lats.clone().cpu()))

        return callback_kwargs

    # Build pipe() kwargs — for img2img, pass the precomputed sigma schedule so
    # the denoising steps exactly match the noise level of the starting latent.
    # FluxPipeline accepts `sigmas` (list[float], len=actual_steps+1) but NOT `timesteps`.
    pipe_ts_kwargs: dict = {}
    if override_sigmas is not None:
        pipe_ts_kwargs["sigmas"] = override_sigmas
    elif override_timesteps is not None:
        # fallback: older diffusers that accept timesteps directly
        pipe_ts_kwargs["timesteps"] = override_timesteps.tolist()
    else:
        pipe_ts_kwargs["num_inference_steps"] = num_steps

    prompts = (
        [source_prompt, source_prompt, edit_prompt]
        if three_branch
        else [source_prompt, edit_prompt]
    )

    try:
        result = pipe(
            prompt=prompts,
            latents=shared_latents,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            output_type="pil",
            callback_on_step_end=_capture if needs_callback else None,
            callback_on_step_end_tensor_inputs=["latents"] if needs_callback else [],
            **pipe_ts_kwargs,
        )
    finally:
        for h in handles:
            h.remove()

    src_img  = result.images[0]
    # In three-branch mode images[1] is the style branch output (discard it);
    # the styled edit is images[2].
    edit_raw = result.images[2] if three_branch else result.images[1]
    edit_img = policy.post_process(src_img, edit_raw)

    if save_intermediates and intermediate_out_dir:
        if captured:
            _save_intermediate_grids(pipe, captured, height, width, intermediate_out_dir,
                                     save_strips=save_strips)
        _save_color_mask(policy, intermediate_out_dir, height, width)

    return src_img, edit_img




# ──────────────────── PFB + SAC colour editing ────────────────────────────────
#
# Adapted from "SVD-Style" (DGIST 2025) for FLUX.1-dev:
#
#   PFB (Principal Feature Blending) — applied once at denoising step pfb_step:
#       Φ(F, α) = U · diag(exp(-α·i) · S) · Vᵀ
#       lat_edit ← Φ(lat_src) + (lat_edit − Φ(lat_edit))
#     Replaces edit's dominant (structural) singular directions with source's,
#     while keeping edit's residual (detail / colour) from its own trajectory.
#
#   SAC (Structural Attention Correction) — applied for the first sac_steps:
#       In double-stream HOTSPOT layers [1, 2, 4], replace edit's image-token
#       Q and K with source's Q and K.  This forces the edit branch to attend to
#       the same spatial positions as source (identity, layout) while V and text
#       conditioning drive the colour change.


def svd_extract(F: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Φ(F, α): principal feature extraction with exponential singular-value reweighting."""
    B, N, C = F.shape
    out = []
    for b in range(B):
        U, S, Vh = torch.linalg.svd(F[b].float(), full_matrices=False)
        w = torch.exp(-alpha * torch.arange(S.shape[0], device=F.device, dtype=torch.float32))
        out.append(((U * (S * w).unsqueeze(0)) @ Vh).to(F.dtype))
    return torch.stack(out)


def apply_pfb(lat_edit: torch.Tensor, lat_src: torch.Tensor,
              alpha: float = 1.0) -> torch.Tensor:
    """PFB: replace edit's principal components with source's; keep edit's residual."""
    return svd_extract(lat_src, alpha) + (lat_edit - svd_extract(lat_edit, alpha))


def _setup_sac_hooks(pipe, n_double: int):
    """
    Register to_q + to_k hooks on double-stream HOTSPOT layers [1, 2, 4] for SAC.

    Returns (hooks, mode_ref, storage).
      mode_ref  — mutable ['normal'|'store'|'inject'] toggled each step by the caller.
      storage   — {'q_L': Q_tensor, 'k_L': K_tensor} populated during 'store' pass.

    Hooks fire on block.attn.to_q / to_k (pre-reshape, pre-norm, pre-rotary-emb).
    Injecting at this point is safe: the shared layer-norm and RoPE that follow are
    applied identically regardless of which branch's Q/K we use.
    """
    storage  = {}
    mode_ref = ["normal"]
    hooks    = []

    for l in [x for x in HOTSPOT if x < n_double]:
        block = pipe.transformer.transformer_blocks[l]

        def _make(lid, kind):
            key = f"{kind}_{lid}"
            def _hook(module, args, output):
                if mode_ref[0] == "store":
                    storage[key] = output.detach().clone()
                elif mode_ref[0] == "inject" and key in storage:
                    return storage[key]
                return output
            return _hook

        hooks.append(block.attn.to_q.register_forward_hook(_make(l, "q")))
        hooks.append(block.attn.to_k.register_forward_hook(_make(l, "k")))

    return hooks, mode_ref, storage


def _decode_latents(
    pipe: FluxPipeline,
    packed_latents: torch.Tensor,
    height: int,
    width: int,
) -> Image.Image:
    """Unpack FLUX packed latents and decode via the VAE -> PIL image."""
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    latents   = pipe._unpack_latents(packed_latents, height, width, vae_scale)
    latents   = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    raw       = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(raw, output_type="pil")[0]


def _save_intermediate_grids(
    pipe: FluxPipeline,
    captured: List[Tuple[int, int, torch.Tensor]],  # (step_idx, n_steps, latents B=2)
    height: int,
    width: int,
    out_dir: str,
    thumb_size: int = 256,
    save_strips: bool = False,
) -> None:
    """
    Decode captured latents at intermediate steps and write:

    steps.png                 — 2-row grid: top=source, bottom=edit, columns=steps
    source_intermediate.png   — strip of source branch frames  (only if save_strips)
    edit_intermediate.png     — strip of edit   branch frames  (only if save_strips)
    """
    label_h = 22   # pixels reserved for step label above each thumbnail
    pad     = 4    # gap between the two rows in compare image
    bg      = (30, 30, 30)   # dark background
    fg      = (220, 220, 220)  # label text colour

    src_thumbs:  List[Image.Image] = []
    edit_thumbs: List[Image.Image] = []
    step_labels: List[str]         = []

    for step_idx, n_steps, lats_cpu in captured:
        lats = lats_cpu.to(pipe.device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            src_img  = _decode_latents(pipe, lats[:1], height, width)
            edit_img = _decode_latents(pipe, lats[1:], height, width)
        src_thumbs.append(src_img.resize((thumb_size, thumb_size), Image.LANCZOS))
        edit_thumbs.append(edit_img.resize((thumb_size, thumb_size), Image.LANCZOS))
        step_labels.append(f"step {step_idx:02d}/{n_steps}")

    n       = len(step_labels)
    strip_w = n * thumb_size
    strip_h = thumb_size + label_h

    def make_strip(thumbs: List[Image.Image]) -> Image.Image:
        img  = Image.new("RGB", (strip_w, strip_h), bg)
        draw = ImageDraw.Draw(img)
        for i, (thumb, label) in enumerate(zip(thumbs, step_labels)):
            x = i * thumb_size
            img.paste(thumb, (x, label_h))
            # Centre-align label text
            try:
                tw = draw.textlength(label)
            except AttributeError:
                tw = len(label) * 7  # fallback estimate
            draw.text((x + (thumb_size - tw) // 2, 3), label, fill=fg)
        return img

    src_strip  = make_strip(src_thumbs)
    edit_strip = make_strip(edit_thumbs)

    compare_h = strip_h * 2 + pad
    compare   = Image.new("RGB", (strip_w, compare_h), bg)
    compare.paste(src_strip,  (0, 0))
    compare.paste(edit_strip, (0, strip_h + pad))

    os.makedirs(out_dir, exist_ok=True)
    steps_path = os.path.join(out_dir, "steps.png")
    compare.save(steps_path)
    print(f"  Steps         → {steps_path}")
    if save_strips:
        src_strip.save( os.path.join(out_dir, "source_intermediate.png"))
        edit_strip.save(os.path.join(out_dir, "edit_intermediate.png"))
        print(f"  Intermediates → {out_dir}/{{source,edit}}_intermediate.png")


def _save_color_mask(policy, out_dir: str, height: int, width: int) -> None:
    """
    If the policy has a built editing-region mask, save it as editing_mask.png.

    White pixels = editing region (hair/car body — target V used here).
    Black pixels = preserved region (face/background — source V used here).
    """
    import numpy as np

    mask = getattr(policy, "_mask", None)
    invert = True  # ColorCtrlPolicy convention: 0=edit, 1=preserve → invert for white=edit
    if mask is None:
        # BackgroundReplacePolicy stores _token_mask: (1,1,L_img), 1=foreground(preserve)
        token_mask = getattr(policy, "_token_mask", None)
        if token_mask is None:
            return
        mask = token_mask.squeeze()  # (L_img,)
        # 1=foreground=preserve, 0=background=edit → same convention as ColorCtrlPolicy
        invert = True

    # FLUX packs latents: 1024px → 128px (VAE ÷8) → 64px (÷2 pack) → 64×64 = 4096 tokens
    h_tok = height // 16
    w_tok = width  // 16
    if mask.numel() != h_tok * w_tok:
        print(f"  [mask] unexpected size {mask.numel()}, expected {h_tok * w_tok}; skipping")
        return

    # mask: 0 = editing region, 1 = preserved → invert so editing shows white
    mask_2d   = ((1.0 - mask.float().cpu()) if invert else mask.float().cpu()).reshape(h_tok, w_tok).numpy()
    mask_img  = Image.fromarray((mask_2d * 255).astype(np.uint8), mode="L")
    mask_img  = mask_img.resize((width, height), Image.NEAREST)

    save_path = os.path.join(out_dir, "editing_mask.png")
    mask_img.save(save_path)
    print(f"  Mask → {save_path}")


def _fallback_attn_mask(policy, score_map: torch.Tensor, device) -> None:
    """Set policy._mask from attention topk scores when SAM2 fails or is unavailable."""
    n_img    = score_map.shape[0]
    k_top    = max(1, int(getattr(policy, "top_k_frac", 0.25) * n_img))
    edit_idx = score_map.topk(k_top).indices
    pres     = torch.ones(n_img, dtype=torch.float32)
    pres[edit_idx] = 0.0
    policy._mask       = pres.to(device)
    policy._mask_built = True
    print(f"  [ColorCtrl] fallback attention mask: {k_top}/{n_img} editing tokens")


def _build_sam2_mask(
    pipe: FluxPipeline,
    policy,
    lats: torch.Tensor,
    height: int,
    width: int,
) -> None:
    """
    Decode the edit-branch latent, run SAM2 with the attention-derived peak as a
    foreground point prompt, and write the result into policy._mask / _mask_built.

    Falls back to attention topk on any failure so generation always continues.
    """
    import numpy as np

    score_map = getattr(policy, "_edit_score_map", None)
    if score_map is None:
        return

    h_tok, w_tok = height // 16, width // 16

    # Peak attention token → pixel centre coordinate
    peak_idx  = int(score_map.argmax().item())
    peak_row  = peak_idx // w_tok
    peak_col  = peak_idx % w_tok
    px = int((peak_col + 0.5) * (width  / w_tok))
    py = int((peak_row + 0.5) * (height / h_tok))

    # Decode edit branch (index 1)
    try:
        edit_lats = lats[1:2].to(pipe.device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            edit_img = _decode_latents(pipe, edit_lats, height, width)
    except Exception as exc:
        print(f"  [SAM2] latent decode failed: {exc}")
        _fallback_attn_mask(policy, score_map, lats.device)
        return

    # Point-prompted SAM2 segmentation
    try:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model_id = getattr(policy, "sam2_model_id", "facebook/sam2-hiera-large")
        print(f"  [SAM2] point=({px},{py}), loading {sam2_model_id}…")
        sam2      = build_sam2_hf(sam2_model_id, device=str(pipe.device))
        predictor = SAM2ImagePredictor(sam2)
        predictor.set_image(np.array(edit_img.convert("RGB")))
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),          # 1 = foreground
            multimask_output=True,
        )
        best_mask = masks[scores.argmax()]        # (H, W) bool, H=W=1024

        # Downscale pixel mask → FLUX token grid (64×64 for 1024px)
        mask_pil  = Image.fromarray(best_mask.astype(np.uint8) * 255, mode="L")
        tok_np    = np.array(mask_pil.resize((w_tok, h_tok), Image.NEAREST))
        # SAM2 mask: white (255) = segmented object = editing region → preserve_mask = 0
        pres_mask = torch.from_numpy((tok_np < 128).astype(np.float32)).flatten()

        policy._mask       = pres_mask.to(lats.device)
        policy._mask_built = True
        n_edit = int((pres_mask < 0.5).sum().item())
        print(f"  [SAM2] mask ready: {n_edit}/{pres_mask.numel()} editing tokens")

    except Exception as exc:
        print(f"  [SAM2] segmentation failed: {exc}  — falling back to attention mask")
        _fallback_attn_mask(policy, score_map, lats.device)


# ──────────────────── Masked delta-flow helpers ───────────────────────────────

class _MaskCaptureAttnProc:
    """
    Drop-in attention processor that captures img→text attention scores at the
    first single-stream block (no encoder_hidden_states).  Every block still runs
    standard SDPA, so generation quality is unaffected.

    Use with a B=1 (or B=2) probe forward pass.  Scores are always captured from
    the last batch item, so the edit prompt is item index -1 in both cases.
    """

    def __init__(self, txt_len: int = 512):
        self.txt_len      = txt_len
        self._n_single    = 0
        self.edit_scores: Optional[torch.Tensor] = None  # (n_img, txt_len) CPU

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        is_double = encoder_hidden_states is not None
        B         = hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        inner = k.shape[-1]
        hd    = inner // attn.heads

        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        if is_double:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(B, -1, attn.heads, hd).transpose(1, 2)
            ek = ek.view(B, -1, attn.heads, hd).transpose(1, 2)
            ev = ev.view(B, -1, attn.heads, hd).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # Capture img→text scores at the first single-stream block
        if not is_double and self._n_single == 0:
            tl    = self.txt_len
            scale = hd ** -0.5
            q_img = q[-1:, :, tl:,  :].float()   # last batch item, image tokens
            k_txt = k[-1:, :, :tl,  :].float()   # last batch item, text tokens
            raw   = torch.matmul(q_img, k_txt.transpose(-2, -1)) * scale  # (1,H,n_img,n_txt)
            self.edit_scores = raw.mean(dim=(0, 1)).detach().cpu()         # (n_img, n_txt)
        if not is_double:
            self._n_single += 1

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                             attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)

        if is_double:
            enc_out = out[:, :encoder_hidden_states.shape[1]]
            out     = out[:, encoder_hidden_states.shape[1]:]
            out     = attn.to_out[0](out)
            out     = attn.to_out[1](out)
            enc_out = attn.to_add_out(enc_out)
            return out, enc_out
        return out


def _build_attn_mask_from_capture(
    policy,
    edit_scores: torch.Tensor,    # (n_img, n_txt) CPU
    txt_len: int,
    device,
) -> None:
    """Set policy._mask from captured img→text attention scores (attention topk)."""
    color_tok_ids = getattr(policy, "_color_tok_ids", [])
    n_img = edit_scores.shape[0]

    if color_tok_ids:
        ids     = [i for i in color_tok_ids if i < txt_len]
        score   = edit_scores[:, ids].mean(-1) if ids else edit_scores.mean(-1)
    else:
        score   = edit_scores.mean(-1)

    k_top    = max(1, int(getattr(policy, "top_k_frac", 0.25) * n_img))
    edit_idx = score.topk(k_top).indices
    pres     = torch.ones(n_img, dtype=torch.float32)
    pres[edit_idx] = 0.0
    policy._mask       = pres.to(device)
    policy._mask_built = True
    focus = "colour-word" if color_tok_ids else "all-text"
    print(f"  [MaskBuild] attention topk ({focus}): {k_top}/{n_img} editing tokens")


@torch.no_grad()
def generate_masked_delta_flow(
    pipe: FluxPipeline,
    policy,
    source_prompt: str,
    edit_prompt: str,
    seed: int = 0,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
    delta_scale: float = 2.0,
    delta_start_step: Optional[int] = None,
    save_intermediates: bool = False,
    intermediate_out_dir: Optional[str] = None,
    intermediate_every: int = 4,
    save_strips: bool = False,
) -> Tuple[Image.Image, Image.Image]:
    """
    Three-phase masked delta-flow colour editing.

    Tension: colour lives in HIGH-NOISE early steps; identity lives in those same steps.
    Solution: source-only lock for the first delta_start_step steps, then delta injection
    for the rest.  The mask is built on the clean fully-denoised source image (not a
    noisy intermediate) so SAM2 gets the best possible input.

    Phase 1  (steps 0 → num_steps):
        Full source-only run from z_T.  Produces clean src_img for SAM2.
        At step mask_build_step: one B=1 probe with the edit prompt captures
        img→text attention that localises the editing region.
        At step delta_start_step (default num_steps//3 ≈ 9): checkpoint z is saved.

    Mask:
        SAM2 segments src_img using the attention-peak pixel as a point prompt.
        Fallback: attention topk on captured scores.

    Phase 2  (steps delta_start_step → num_steps-1):
        Branch off from the saved checkpoint.  B=2 per step:
            [z+src, z+edit] → v_src, v_edit
            z += dt * (v_src + delta_scale * mask * (v_edit − v_src))
        Steps 0→delta_start_step-1 followed source exactly → identity locked.
        Steps delta_start_step→num_steps-1 inject colour delta → colour changes.

    Returns (source_image, edit_image).
    """
    exec_device = getattr(pipe, "_execution_device", device)

    policy.pre_generate(
        pipe, device=device, height=height, width=width,
        num_steps=num_steps, seed=seed,
        source_prompt=source_prompt, edit_prompt=edit_prompt,
        max_sequence_length=max_sequence_length, guidance_scale=guidance_scale,
    )

    # ── Encode prompts ────────────────────────────────────────────────────────
    def _encode(prompt: str):
        enc = pipe.encode_prompt(
            prompt, prompt_2=None, device=exec_device,
            num_images_per_prompt=1, max_sequence_length=max_sequence_length,
        )
        return (enc[0], enc[1], enc[2]) if len(enc) == 3 else (enc[0], enc[2], enc[4])

    src_pe,  src_pp,  src_txt  = _encode(source_prompt)
    edit_pe, edit_pp, edit_txt = _encode(edit_prompt)

    # ── Initial latent ────────────────────────────────────────────────────────
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    num_ch = pipe.transformer.config.in_channels // 4
    lat_h  = 2 * (height // (vae_scale * 2))
    lat_w  = 2 * (width  // (vae_scale * 2))

    gen   = torch.Generator(device=exec_device).manual_seed(seed)
    raw_z = randn_tensor((1, num_ch, lat_h, lat_w), generator=gen,
                         device=exec_device, dtype=src_pe.dtype)
    z_T   = pipe._pack_latents(raw_z, 1, num_ch, lat_h, lat_w)
    img_ids = pipe._prepare_latent_image_ids(
        1, lat_h // 2, lat_w // 2, exec_device, src_pe.dtype,
    )
    if img_ids.ndim == 3:
        img_ids = img_ids.squeeze(0)

    # ── Timestep schedule (FLUX.1-dev dynamic shifting) ───────────────────────
    sched = pipe.scheduler.config
    if getattr(sched, "use_dynamic_shifting", False):
        base_seq  = getattr(sched, "base_image_seq_len", 256)
        max_seq   = getattr(sched, "max_image_seq_len",  4096)
        base_shft = getattr(sched, "base_shift",         0.5)
        max_shft  = getattr(sched, "max_shift",          1.15)
        m   = (max_shft - base_shft) / (max_seq - base_seq)
        mu  = (lat_h // 2) * (lat_w // 2) * m + (base_shft - m * base_seq)
        pipe.scheduler.set_timesteps(num_steps, device=exec_device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_steps, device=exec_device)
    timesteps    = pipe.scheduler.timesteps
    sigmas       = pipe.scheduler.sigmas
    has_guidance = getattr(pipe.transformer.config, "guidance_embeds", False)
    mdtype = src_pe.dtype   # bfloat16 — transformer weight dtype

    def _fwd(z: torch.Tensor, i: int, pe, pp) -> torch.Tensor:
        # sigmas are float32; after Euler steps z can drift to float32.
        # Cast back to model dtype before every forward pass.
        z = z.to(mdtype)
        B = z.shape[0]
        t = timesteps[i].expand(B)
        g = torch.full([B], guidance_scale, device=exec_device,
                       dtype=mdtype) if has_guidance else None
        return pipe.transformer(
            hidden_states=z, timestep=t / 1000.0, guidance=g,
            pooled_projections=pp, encoder_hidden_states=pe,
            txt_ids=src_txt, img_ids=img_ids, return_dict=False,
        )[0]

    mask_step  = getattr(policy, "mask_build_step", num_steps // 2)
    # delta_start_step: source-only lock for steps 0→N-1; delta injection from N.
    # FLUX dynamic-shifting concentrates sigma budget in the first few steps.
    # sigma at step 3 ≈ 0.40;  calibrated: delta_scale = 1/sigma_start for 100% edit.
    # Default num_steps//9 ≈ 3 — minimal identity lock, full colour window.
    eff_start = (delta_start_step if delta_start_step is not None
                 else max(0, num_steps // 9))
    eff_start = max(0, min(eff_start, num_steps - 1))

    try:
        from diffusers.models.attention_processor import FluxAttnProcessor
    except ImportError:
        from diffusers.models.attention_processor import FluxAttnProcessor2_0 as FluxAttnProcessor

    # ── Phase 1: full source run ───────────────────────────────────────────────
    # Purposes:
    #   1. Save checkpoint at eff_start for Phase 2 to branch off from.
    #   2. Probe at mask_step with edit prompt → attention for SAM2 point.
    #   3. Continue to completion → clean src_img for SAM2 segmentation.
    print(f"[MaskedDeltaFlow] Phase 1: source run ({num_steps} steps), "
          f"checkpoint@{eff_start}, probe@{mask_step}…")
    z            = z_T.clone()
    z_checkpoint = None
    cap          = None
    z1_cache: dict = {}   # step → z (CPU) for intermediate grid

    for i in range(num_steps):
        dt = (sigmas[i + 1] - sigmas[i]).to(mdtype)

        if i == eff_start:
            z_checkpoint = z.clone()   # save before this step

        # Probe: one B=1 edit-prompt pass at mask_step — does NOT alter z.
        if i == mask_step and cap is None:
            _cap = _MaskCaptureAttnProc(txt_len=max_sequence_length)
            pipe.transformer.set_attn_processor(_cap)
            try:
                _fwd(z, i, edit_pe, edit_pp)
            finally:
                pipe.transformer.set_attn_processor(FluxAttnProcessor())
            cap = _cap

        z = z + dt * _fwd(z, i, src_pe, src_pp)

        if save_intermediates and (i % intermediate_every == 0 or i == num_steps - 1):
            z1_cache[i] = z.detach().cpu()

    src_img = _decode_latents(pipe, z, height, width)

    # ── Mask building on clean source image ───────────────────────────────────
    print(f"[MaskedDeltaFlow] Building mask from src_img (probe@{mask_step})…")
    use_sam2 = getattr(policy, "use_sam2", False)

    if use_sam2 and cap is not None and cap.edit_scores is not None:
        import numpy as np
        color_ids = getattr(policy, "_color_tok_ids", [])
        scores    = cap.edit_scores
        ids       = [k for k in color_ids if k < max_sequence_length]
        s1d       = scores[:, ids].mean(-1) if ids else scores.mean(-1)
        h_tok, w_tok = height // 16, width // 16
        pidx = int(s1d.argmax().item())
        px   = int((pidx % w_tok + 0.5) * width  / w_tok)
        py   = int((pidx // w_tok + 0.5) * height / h_tok)
        try:
            from sam2.build_sam import build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print(f"  [SAM2] point=({px},{py}) on clean src_img, "
                  f"loading {policy.sam2_model_id}…")
            pred = SAM2ImagePredictor(build_sam2_hf(policy.sam2_model_id,
                                                    device=str(exec_device)))
            pred.set_image(np.array(src_img.convert("RGB")))
            m_out, s_out, _ = pred.predict(
                point_coords=np.array([[px, py]]),
                point_labels=np.array([1]), multimask_output=True,
            )
            best = m_out[s_out.argmax()]
            mpil = Image.fromarray(best.astype(np.uint8) * 255, mode="L")
            tnp  = np.array(mpil.resize((w_tok, h_tok), Image.NEAREST))
            pres = torch.from_numpy((tnp < 128).astype(np.float32)).flatten()
            policy._mask = pres.to(exec_device); policy._mask_built = True
            print(f"  [SAM2] {int((pres < 0.5).sum())}/{pres.numel()} editing tokens")
        except Exception as exc:
            print(f"  [SAM2] failed: {exc} — using attention topk")
            if cap is not None and cap.edit_scores is not None:
                _build_attn_mask_from_capture(policy, cap.edit_scores,
                                              max_sequence_length, exec_device)
    elif cap is not None and cap.edit_scores is not None:
        _build_attn_mask_from_capture(policy, cap.edit_scores,
                                      max_sequence_length, exec_device)
    else:
        n_img = (lat_h // 2) * (lat_w // 2)
        k_top = max(1, int(getattr(policy, "top_k_frac", 0.25) * n_img))
        pres  = torch.ones(n_img, dtype=torch.float32, device=exec_device)
        pres[n_img // 2 - k_top // 2: n_img // 2 + k_top // 2] = 0.0
        policy._mask = pres; policy._mask_built = True
        print(f"  [MaskBuild] fallback centre mask: {k_top}/{n_img} tokens")

    # edit_mask: (1, N, 1)  1 = editing region  0 = preserved
    edit_mask_2d = (1.0 - policy._mask).to(device=exec_device, dtype=mdtype).view(1, -1, 1)
    n_delta = num_steps - eff_start

    # ── Phase 2: delta-flow from checkpoint ───────────────────────────────────
    # Branch off from z_checkpoint (= source at step eff_start).
    # Steps 0→eff_start-1 were source-only → identity locked.
    # Steps eff_start→num_steps-1: B=2 masked delta → colour changes.
    print(f"[MaskedDeltaFlow] Phase 2: delta from step {eff_start} "
          f"({n_delta} steps, delta_scale={delta_scale}, "
          f"{int(edit_mask_2d.sum().item())} editing tokens)…")

    z        = (z_checkpoint if z_checkpoint is not None else z_T).clone()
    captured: List[Tuple[int, int, torch.Tensor]] = []

    for i in range(eff_start, num_steps):
        dt = (sigmas[i + 1] - sigmas[i]).to(mdtype)

        # B=2: same z with source + edit prompts.
        # v_edit − v_src = pure colour delta; structural components cancel.
        z_b2  = torch.cat([z, z])
        pe_b2 = torch.cat([src_pe, edit_pe])
        pp_b2 = torch.cat([src_pp, edit_pp])
        v_both = _fwd(z_b2, i, pe_b2, pp_b2)
        v_src, v_edit = v_both.chunk(2)

        z = z + dt * (v_src + delta_scale * edit_mask_2d * (v_edit - v_src))

        if save_intermediates and (i % intermediate_every == 0 or i == num_steps - 1):
            z_src_frame = z1_cache.get(i, z.detach().cpu())
            captured.append((i, num_steps,
                             torch.cat([z_src_frame, z.detach().cpu()])))

    edit_img = _decode_latents(pipe, z, height, width)

    if save_intermediates and intermediate_out_dir:
        if captured:
            _save_intermediate_grids(pipe, captured, height, width, intermediate_out_dir,
                                     save_strips=save_strips)
        _save_color_mask(policy, intermediate_out_dir, height, width)

    return src_img, edit_img


@torch.no_grad()
def generate_p2p(
    pipe: FluxPipeline,
    source_prompt: str,
    edit_prompt: str,
    inject_layers: List[int],           # unused; kept for API compat
    seed: int = 0,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    max_sequence_length: int = 512,
    device: str = "cuda",
    anchor_end_frac: float = 0.0,       # unused; kept for API compat
    freq_sigma: float = 0.0,            # unused; kept for API compat
    img2img_strength: float = 0.6,      # unused; kept for API compat
    source_color: str = "",             # unused; kept for API compat
    target_color: str = "",             # unused; kept for API compat
    edge_strength: float = 0.7,         # unused; kept for API compat
    latent_top_k: int = 0,              # unused; kept for API compat
    latent_alpha: float = 1.0,          # unused; kept for API compat
    color_structure_frac: float = 0.0,  # unused; kept for API compat
    inject_frac: float = 0.5,           # unused; kept for API compat
    pfb_step: int = 3,                  # unused; kept for API compat
    pfb_alpha: float = 1.0,             # unused; kept for API compat
    delta_scale: float = 1.5,
) -> Tuple[Image.Image, Image.Image]:
    """
    Colour editing via PFB + SAC, adapted from SVD-Style (DGIST 2025) to FLUX.

    Both branches start from the same z_T (same seed, free to render their own
    colour from step 0).  Two mechanisms preserve identity:

    SAC — steps 0 .. ceil(inject_frac * num_steps):
        Source and edit run as separate forward passes.  In double-stream HOTSPOT
        layers [1,2,4] the edit branch's image-token Q and K are replaced with
        the source branch's Q and K.  This forces the edit branch to attend to the
        same spatial positions (face, plate, car body) while V and text conditioning
        drive the colour change.

    PFB — exactly at denoising step pfb_step (default 3):
        lat_edit ← Φ(lat_src, α) + (lat_edit − Φ(lat_edit, α))
        where Φ is SVD with exp(-α·i) singular-value reweighting.
        Replaces the dominant structural singular directions of lat_edit with
        source's, locking in layout without overwriting colour channels.

    Tuning:
        inject_frac  0.3–0.7   more = tighter identity, less = more freedom
        pfb_step     2–6       step at which latent is corrected (0-indexed)
        pfb_alpha    1.0       higher = fewer singular vectors blended (more structure-only)

    Colour editing via Delta Flow Guidance.

    Phase 1 — source generation (num_steps, source prompt):
        Standard denoising from z_T → lat_src.

    Phase 2 — guided edit (num_steps, from the same z_T):
        At every step, BOTH velocities are evaluated at the SAME current latent
        (lat_edit), with different text conditioning:

            v_src  = model(lat_edit, t, source_prompt)
            v_edit = model(lat_edit, t, edit_prompt)

        Because they share the same input, their difference is a pure colour
        signal — structural components cancel.  The guided velocity is:

            v_guided = v_src + delta_scale × (v_edit − v_src)

        v_src anchors the trajectory to source structure.
        (v_edit − v_src) is the colour-change direction.
        delta_scale amplifies it:
            1.0  → pure edit prompt direction
            1.5  → amplified colour, good default
            2.0  → stronger colour, may introduce minor artefacts

    Returns (src_img, edit_img).
    """
    exec_device = getattr(pipe, "_execution_device", device)

    # -- Encode both prompts --------------------------------------------------
    def _encode(prompt: str):
        enc = pipe.encode_prompt(
            prompt, prompt_2=None, device=exec_device,
            num_images_per_prompt=1, max_sequence_length=max_sequence_length,
        )
        if len(enc) == 3:
            pe, pp, txt_ids = enc
        else:
            pe, _, pp, _, txt_ids, _ = enc
        return pe, pp, txt_ids

    src_pe,  src_pp,  src_txt  = _encode(source_prompt)
    edit_pe, edit_pp, edit_txt = _encode(edit_prompt)

    # -- Initial packed latent z_T --------------------------------------------
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    num_ch    = pipe.transformer.config.in_channels // 4
    lat_h     = 2 * (height // (vae_scale * 2))
    lat_w     = 2 * (width  // (vae_scale * 2))

    gen   = torch.Generator(device=exec_device).manual_seed(seed)
    raw_z = randn_tensor(
        (1, num_ch, lat_h, lat_w),
        generator=gen, device=exec_device, dtype=src_pe.dtype,
    )
    z_T     = pipe._pack_latents(raw_z, 1, num_ch, lat_h, lat_w)
    img_ids = pipe._prepare_latent_image_ids(
        1, lat_h // 2, lat_w // 2, exec_device, src_pe.dtype,
    )
    if img_ids.ndim == 3:
        img_ids = img_ids.squeeze(0)

    # -- Timesteps ------------------------------------------------------------
    image_seq_len = (lat_h // 2) * (lat_w // 2)
    sched_cfg     = pipe.scheduler.config
    if getattr(sched_cfg, "use_dynamic_shifting", False):
        base_seq  = getattr(sched_cfg, "base_image_seq_len", 256)
        max_seq   = getattr(sched_cfg, "max_image_seq_len",  4096)
        base_shft = getattr(sched_cfg, "base_shift",         0.5)
        max_shft  = getattr(sched_cfg, "max_shift",          1.15)
        m  = (max_shft - base_shft) / (max_seq - base_seq)
        mu = image_seq_len * m + (base_shft - m * base_seq)
        pipe.scheduler.set_timesteps(num_steps, device=exec_device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_steps, device=exec_device)
    timesteps = pipe.scheduler.timesteps
    sigmas    = pipe.scheduler.sigmas

    has_guidance = getattr(pipe.transformer.config, "guidance_embeds", False)

    # -- Phase 1: source generation -------------------------------------------
    print(f"[UltimateFlux P2P] Phase 1: source ({num_steps} steps)...")
    lat_src = z_T.clone()
    for i, t in enumerate(timesteps):
        dt  = sigmas[i + 1] - sigmas[i]
        ts1 = t.expand(1)
        g1  = (torch.full([1], guidance_scale, device=exec_device, dtype=lat_src.dtype)
               if has_guidance else None)
        v_s = pipe.transformer(
            hidden_states=lat_src,
            timestep=ts1 / 1000.0,
            guidance=g1,
            pooled_projections=src_pp,
            encoder_hidden_states=src_pe,
            txt_ids=src_txt,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        lat_src = lat_src + dt * v_s

    # -- Phase 2: delta flow guidance -----------------------------------------
    print(
        f"[UltimateFlux P2P] Phase 2: delta flow guidance "
        f"(delta_scale={delta_scale}, {num_steps} steps)..."
    )
    lat_edit = z_T.clone()
    for i, t in enumerate(timesteps):
        dt  = sigmas[i + 1] - sigmas[i]
        ts2 = t.expand(2)
        g2  = (torch.full([2], guidance_scale, device=exec_device, dtype=lat_edit.dtype)
               if has_guidance else None)

        # Both velocities evaluated at THE SAME lat_edit — their difference is
        # a pure colour signal; structural terms cancel.
        lat_b = torch.cat([lat_edit, lat_edit])
        pe_b  = torch.cat([src_pe,   edit_pe])
        pp_b  = torch.cat([src_pp,   edit_pp])

        noise_b = pipe.transformer(
            hidden_states=lat_b,
            timestep=ts2 / 1000.0,
            guidance=g2,
            pooled_projections=pp_b,
            encoder_hidden_states=pe_b,
            txt_ids=src_txt,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        v_src, v_edit = noise_b.chunk(2)
        v_delta = v_edit - v_src                             # (1, N, 64)

        # Soft per-token mask: tokens where source and edit disagree most
        # (hair, car body) receive the full colour delta; tokens where they
        # agree (face outline, background, plate) receive near-zero correction.
        # The model itself localises the colour region — no CV or user mask needed.
        mag  = v_delta.norm(dim=-1, keepdim=True)            # (1, N, 1)
        mask = mag / (mag.max() + 1e-8)                      # (1, N, 1) ∈ [0, 1]

        v_guided = v_src + delta_scale * mask * v_delta
        lat_edit = lat_edit + dt * v_guided

    src_img  = _decode_latents(pipe, lat_src,  height, width)
    edit_img = _decode_latents(pipe, lat_edit, height, width)
    return src_img, edit_img
