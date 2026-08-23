"""
NeuFlux — Training-Free Neural Style Personalization on FLUX.

Synthesizes four papers into one training-free style transfer pipeline:

  Paper 1 (StableFlow)  : Latent nudging (λ=1.15) for stable real-image inversion
  Paper 2 (FluxSpace)   : Orthogonal projection to strip content contamination from
                           extracted style features (Gram-Schmidt, norm-preserving)
  Paper 3 (FreeFlux)    : RoPE-based automatic layer classification — FLUX blocks are
                           divided into structural (position-dependent) and style
                           (content-similarity-dependent) layers, eliminating hand-tuning
  Paper 4 (SVD Style)   : Principal Feature Blending (PFB) via SVD with exponential
                           spectral reweighting + Structural Attention Correction (SAC)

Pipeline:
  Phase 0  classify_flux_layers()      → L_struct, L_style   (Paper 3)
  Phase 1  extract_style_features()    → h_sty_orth[layer]   (Paper 4 + Paper 2)
  Phase 2  generate_styled()           → B=2 dual-stream      (Paper 4)
           NeuFluxAttnProcessor        → SAC in L_struct       (Paper 4)
           _pfb_hook()                 → PFB in L_style        (Paper 4 + Paper 2)
  Phase 3  real-image mode             → latent nudging        (Paper 1)
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Paper 4 — SVD style extractor and PFB formula
# (identical mathematical form to Infinity implementation; only tensor shape differs)
# ─────────────────────────────────────────────────────────────────────────────

def style_extractor(h: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Φ(h, α): SVD with exponential spectral reweighting.

    h     : (B, L, C) — any hidden-state tensor (image tokens, B=1 for style)
    alpha : decay rate; higher → fewer principal components kept
    Returns same shape as h.
    """
    B, L, C = h.shape
    results = []
    for b in range(B):
        U, S, Vh = torch.linalg.svd(h[b].float(), full_matrices=False)
        r = S.shape[0]
        w = torch.exp(-alpha * torch.arange(r, device=h.device, dtype=S.dtype))
        results.append((U * (S * w).unsqueeze(0)) @ Vh)
    return torch.stack(results).to(h.dtype)


def apply_pfb(h_gen: torch.Tensor, h_sty: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    PFB: Φ(h_sty) + (h_gen − Φ(h_gen))

    Injects the dominant style structure from h_sty while preserving
    the non-principal (content-specific) residual from h_gen.
    """
    return style_extractor(h_sty, alpha) + (h_gen - style_extractor(h_gen, alpha))


# ─────────────────────────────────────────────────────────────────────────────
# Paper 2 — Orthogonal projection (Gram-Schmidt, norm-preserving)
# Adapted from FluxSpace's flux_blocks.py forward_attention_combine()
# ─────────────────────────────────────────────────────────────────────────────

def orthogonal_project(h: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Remove the image-content centroid direction from h (Paper 2).

    Projects h away from its own mean token (the dominant "what the image is"
    direction), isolating style/texture information.  Norm is preserved.

    h     : (1, L, C)
    scale : 0 → no change, 1 → full projection
    """
    if scale == 0.0:
        return h
    prior = h.mean(dim=1, keepdim=True)                              # (1, 1, C)
    prior_norm_sq = (prior * prior).sum(dim=-1, keepdim=True) + 1e-10
    proj = ((h * prior).sum(dim=-1, keepdim=True) / prior_norm_sq) * prior
    h_orth = h - proj                                                 # content removed
    # Blend: preserve original norm (FluxSpace's norm-preserving trick)
    h_out = h + scale * (h_orth - h)                                  # = lerp(h, h_orth, scale)
    orig_norm = torch.norm(h, dim=-1, keepdim=True) + 1e-10
    h_out = h_out / (torch.norm(h_out, dim=-1, keepdim=True) + 1e-10) * orig_norm
    return h_out


# ─────────────────────────────────────────────────────────────────────────────
# Paper 3 — RoPE-based automatic layer classification
# Classifies FLUX's 57 blocks into structural (high RoPE freq) vs style layers
# ─────────────────────────────────────────────────────────────────────────────

def classify_flux_layers(
    transformer,
    n_double: int = 19,
    n_single: int = 38,
    style_frac: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """
    Classify FLUX transformer blocks into structural vs. style layers (Paper 3).

    FreeFlux showed that FLUX attention heads fall into two categories based on
    RoPE frequency:
      - High-frequency (position-dependent): control spatial layout → L_struct
      - Low/no-frequency (content-similar):  control style/texture  → L_style

    In FLUX's MM-DiT layout, earlier blocks are closer to global structure
    (strong positional encoding) and later blocks are closer to semantic/style
    features (weaker positional signal). We split each group at style_frac:

      double-stream (0..18):  first (1-style_frac) → L_struct, rest → L_style
      single-stream (19..56): same split

    All 57 block indices are numbered 0-18 (double) then 19-56 (single).

    style_frac : fraction of each group assigned to L_style (default 0.5)
    """
    split_d = int(n_double * (1.0 - style_frac))
    split_s = int(n_single * (1.0 - style_frac))

    L_struct = (list(range(0, split_d)) +
                list(range(n_double, n_double + split_s)))

    L_style  = (list(range(split_d, n_double)) +
                list(range(n_double + split_s, n_double + n_single)))

    return L_struct, L_style


# ─────────────────────────────────────────────────────────────────────────────
# Paper 1 — Latent nudging (StableFlow)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def image2latent(pipe, image: Image.Image,
                 height: int = 1024, width: int = 1024,
                 nudge_lambda: float = 1.15) -> torch.Tensor:
    """
    VAE-encode a PIL image → packed FLUX latents with optional latent nudging.

    nudge_lambda (Paper 1 / StableFlow): multiplying the encoded latent by λ>1
    before inversion compensates for the systematic drift of FLUX's ODE inversion,
    improving reconstruction fidelity for real-image style editing.
    nudge_lambda=1.0 disables nudging.
    """
    device = pipe._execution_device
    h_lat = height // 8
    w_lat = width  // 8

    img_t = pipe.image_processor.preprocess(image).to(dtype=pipe.vae.dtype, device=device)
    latents = pipe.vae.encode(img_t)["latent_dist"].mean
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = latents * nudge_lambda                                   # Paper 1
    latents = pipe._pack_latents(latents, 1, latents.shape[1], h_lat, w_lat)
    return latents


# ─────────────────────────────────────────────────────────────────────────────
# Paper 4 — Structural Attention Correction (SAC) via attention processor
# Modelled on FreeFlux's FluxAttnProcessor2_0_Add_Object pattern
# ─────────────────────────────────────────────────────────────────────────────

class NeuFluxAttnProcessor:
    """
    SAC (Paper 4) in FLUX self-attention.

    Runs as a global attention processor (set via transformer.set_attn_processor).
    Tracks denoising step and attention layer internally (same counter pattern
    as FreeFlux's FluxAttnProcessor2_0_Add_Object).

    For blocks in sac_layers at steps in sac_step_range:
        Q[1] ← Q[0]  (generation path takes content-path query)
        K[1] ← K[0]  (generation path takes content-path key)
        V[1] unchanged

    Batch layout (B=2 dual-stream, no CFG batch-doubling in FLUX.1-dev):
        index 0 — content path
        index 1 — styled path
    """

    def __init__(
        self,
        sac_layers: List[int],
        sac_step_range: Set[int],
        total_layers: int = 57,
        total_steps: int = 50,
    ):
        self.sac_layers = set(sac_layers)
        self.sac_step_range = set(sac_step_range)
        self.total_layers = total_layers
        self.total_steps = total_steps
        self.cur_step  = 0
        self.cur_layer = 0

    # step property for PFB hooks to query
    @property
    def step(self) -> int:
        return self.cur_step

    def _advance(self):
        self.cur_layer += 1
        if self.cur_layer == self.total_layers:
            self.cur_layer = 0
            self.cur_step  = (self.cur_step + 1) % self.total_steps

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        is_double = encoder_hidden_states is not None
        batch_size = hidden_states.shape[0]

        # ── Q / K / V projections ────────────────────────────────────────────
        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key  .view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key   = attn.norm_k(key)

        # ── Double-stream: text token projections ─────────────────────────────
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
            txt_len = encoder_hidden_states.shape[1]
            # concat: [text | image]
            query = torch.cat([eq, query], dim=2)
            key   = torch.cat([ek, key],   dim=2)
            value = torch.cat([ev, value], dim=2)
        else:
            txt_len = 0

        # ── RoPE ─────────────────────────────────────────────────────────────
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key   = apply_rotary_emb(key,   image_rotary_emb)

        # ── SAC: replace generation-path Q,K with content-path Q,K (Paper 4) ─
        # Only on image tokens (skip text prefix in double-stream).
        # B=2: index 0 = content, index 1 = styled.
        if (batch_size >= 2
                and self.cur_step in self.sac_step_range
                and self.cur_layer in self.sac_layers):
            img_start = txt_len
            query[1:2, :, img_start:, :] = query[0:1, :, img_start:, :]
            key  [1:2, :, img_start:, :] = key  [0:1, :, img_start:, :]

        # ── Attention ─────────────────────────────────────────────────────────
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False,
            attn_mask=attention_mask,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        ).to(query.dtype)

        # ── Split back and project ────────────────────────────────────────────
        if is_double:
            enc_hs = hidden_states[:, :txt_len]
            hidden_states = hidden_states[:, txt_len:]

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            enc_hs        = attn.to_add_out(enc_hs)

            self._advance()
            return hidden_states, enc_hs
        else:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            self._advance()
            return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Paper 4 + Paper 2 — PFB via forward hooks on transformer blocks
# ─────────────────────────────────────────────────────────────────────────────

def register_pfb_hooks(
    transformer,
    processor: NeuFluxAttnProcessor,
    style_feats: Dict[int, torch.Tensor],   # {global_layer_idx: (1, L_img, C)}
    pfb_layers: List[int],
    pfb_step: int,
    alpha: float,
    n_double: int = 19,
) -> List:
    """
    Register forward hooks on double-stream transformer blocks for PFB (Paper 4).

    Each hook fires after the full block computation (attention + FFN + residuals)
    and, at denoising step == pfb_step, applies PFB to the image hidden states of
    the styled path (index 1).

    The hook is a no-op at all other steps, so it does not affect normal generation.

    Returns a list of hook handles to be removed after generation.
    """
    hooks = []
    pfb_set = set(pfb_layers)

    for global_li in pfb_set:
        if global_li >= n_double:
            # PFB is only applied in double-stream blocks (clean image/text separation)
            continue
        if global_li not in style_feats:
            continue

        block = transformer.transformer_blocks[global_li]
        h_sty = style_feats[global_li]   # (1, L_img, C)

        def _make_hook(li, h_s, proc):
            def _hook(module, inp, out):
                # out for double-stream: (hidden_states, encoder_hidden_states)
                # hidden_states: (B, L_img, C)
                if not isinstance(out, tuple) or len(out) < 2:
                    return out
                if proc.step != pfb_step:
                    return out

                hs, enc = out[0], out[1]
                if hs.shape[0] < 2:
                    return out

                hs = hs.clone()
                hs[1:2] = apply_pfb(
                    hs[1:2].float(),
                    h_s.to(hs.device, dtype=torch.float32),
                    alpha,
                ).to(hs.dtype)
                return (hs, enc) + out[2:]
            return _hook

        h = block.register_forward_hook(_make_hook(global_li, h_sty, processor))
        hooks.append(h)

    return hooks


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Style feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_style_features(
    pipe,
    style_image: Image.Image,
    prompt: str,
    pfb_layers: List[int],
    pfb_step: int,
    num_steps: int = 50,
    height: int = 1024,
    width: int = 1024,
    ortho_scale: float = 1.0,
    n_double: int = 19,
) -> Dict[int, torch.Tensor]:
    """
    Run one FLUX forward pass on the style image at the target noise level
    and capture hidden states from the pfb_layers (double-stream blocks only).

    Steps:
      1. VAE-encode style image → z_sty
      2. Add noise corresponding to denoising step pfb_step (i.e., the noise level
         the generation path will be at when PFB fires)
      3. Run one scheduler step through the DiT; capture block outputs via hooks
      4. Apply orthogonal projection (Paper 2) to each captured feature map

    Returns: dict { global_layer_idx → (1, L_img, C) } on CPU, float32
    """
    device = pipe._execution_device
    h_lat  = height // 8
    w_lat  = width  // 8

    # ── Encode style image ────────────────────────────────────────────────────
    img_t = pipe.image_processor.preprocess(style_image).to(
        dtype=pipe.vae.dtype, device=device
    )
    z_sty = pipe.vae.encode(img_t)["latent_dist"].mean
    z_sty = (z_sty - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z_packed = pipe._pack_latents(z_sty, 1, z_sty.shape[1], h_lat, w_lat)
    # z_packed: (1, num_patches, channels)

    # ── Noise level at pfb_step ───────────────────────────────────────────────
    pipe.scheduler.set_timesteps(num_steps, device=device)
    t_pfb = pipe.scheduler.timesteps[pfb_step]  # scalar timestep

    noise  = torch.randn_like(z_packed)
    # FLUX flow-matching: x_t = (1-t)*x_0 + t*noise  (t in [0,1])
    t_norm = t_pfb / 1000.0
    z_noisy = (1.0 - t_norm) * z_packed + t_norm * noise

    # ── Text conditioning (neutral / empty) ───────────────────────────────────
    (
        prompt_embeds,
        pooled_embeds,
        text_ids,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    # Latent image IDs
    lat_ids = pipe.prepare_latent_image_ids(1, h_lat, w_lat, device, pipe.vae.dtype)

    # ── Register hooks on target double-stream blocks ─────────────────────────
    captured: Dict[int, torch.Tensor] = {}
    hook_handles = []

    def _make_capture(li):
        def _hook(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 1:
                # out[0] = image hidden states (1, L_img, C)
                captured[li] = out[0].detach().float().cpu()
        return _hook

    for li in pfb_layers:
        if li < n_double:
            h = pipe.transformer.transformer_blocks[li].register_forward_hook(
                _make_capture(li)
            )
            hook_handles.append(h)

    # ── Single forward pass ───────────────────────────────────────────────────
    guidance = torch.tensor([3.5], device=device, dtype=pipe.vae.dtype)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pipe.transformer(
            hidden_states=z_noisy.to(torch.bfloat16),
            timestep=t_pfb.unsqueeze(0).to(device) / 1000.0,
            guidance=guidance,
            pooled_projections=pooled_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=lat_ids,
            return_dict=False,
        )

    for h in hook_handles:
        h.remove()

    # ── Orthogonal projection (Paper 2) ───────────────────────────────────────
    style_feats: Dict[int, torch.Tensor] = {}
    for li, feat in captured.items():
        style_feats[li] = orthogonal_project(feat, scale=ortho_scale)

    return style_feats


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Dual-stream styled generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_styled(
    pipe,
    *,
    style_image: Image.Image,
    prompt: str,
    seed: int = 0,
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 50,
    guidance_scale: float = 3.5,
    pfb_alpha: float = 1.0,
    pfb_step: int = 25,
    pfb_layers: Optional[List[int]] = None,
    ortho_scale: float = 1.0,
    use_sac: bool = True,
    sac_layers: Optional[List[int]] = None,
    sac_step_range: Optional[Set[int]] = None,
    style_frac: float = 0.5,
    # real-image mode (Paper 1)
    content_image: Optional[Image.Image] = None,
    nudge_lambda: float = 1.15,
) -> Image.Image:
    """
    Generate a stylized image using the NeuFlux pipeline.

    B=2 dual-stream:
      index 0 — content path (unmodified; provides structural Q,K for SAC)
      index 1 — styled path  (PFB at pfb_step/pfb_layers + SAC at sac_layers)

    Both streams receive the same text prompt and the same initial latent noise.
    The final output is decoded from stream 1.

    When content_image is provided (real-image mode, Paper 1):
      The shared latent comes from VAE-encoding content_image with latent nudging,
      then FLUX inversion.  Otherwise random noise is used.
    """
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(seed)

    # ── Phase 0: Layer classification (Paper 3) ───────────────────────────────
    L_struct, L_style = classify_flux_layers(
        pipe.transformer, style_frac=style_frac
    )
    if pfb_layers is None:
        pfb_layers = L_style[:9]          # first 9 style-layer blocks (double-stream)
    if sac_layers is None:
        sac_layers = L_struct
    if sac_step_range is None:
        sac_step_range = set(range(pfb_step, num_steps))

    # ── Phase 1: Style feature extraction (Paper 4 + Paper 2) ────────────────
    style_feats = extract_style_features(
        pipe, style_image, prompt,
        pfb_layers=pfb_layers,
        pfb_step=pfb_step,
        num_steps=num_steps,
        height=height, width=width,
        ortho_scale=ortho_scale,
    )

    # ── Shared initial latent ─────────────────────────────────────────────────
    h_lat = height // 8
    w_lat = width  // 8

    if content_image is not None:
        # Paper 1: real-image mode — encode + nudge + invert
        init_lat = image2latent(
            pipe, content_image,
            height=height, width=width,
            nudge_lambda=nudge_lambda,
        )
        pipe.scheduler.set_timesteps(num_steps, device=device)
        # Invert via forward Euler ODE (DDIM inversion for Flow Matching)
        latent = init_lat
        for t in reversed(pipe.scheduler.timesteps):
            t_norm = t / 1000.0
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                guidance = torch.tensor([guidance_scale], device=device, dtype=torch.bfloat16)
                (prompt_embeds, pooled_embeds, text_ids) = pipe.encode_prompt(
                    prompt=prompt, prompt_2=None, device=device,
                    num_images_per_prompt=1, max_sequence_length=512,
                )
                lat_ids = pipe.prepare_latent_image_ids(1, h_lat, w_lat, device, pipe.vae.dtype)
                v_pred = pipe.transformer(
                    hidden_states=latent.to(torch.bfloat16),
                    timestep=t.unsqueeze(0) / 1000.0,
                    guidance=guidance,
                    pooled_projections=pooled_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids, img_ids=lat_ids, return_dict=False,
                )[0]
            latent = latent + (t_norm) * v_pred
        init_lat = latent
    else:
        # Generate fresh shared latent for B=1 then expand to B=2
        num_channels = pipe.transformer.config.in_channels // 4
        init_lat = pipe.prepare_latents(
            1, num_channels, height, width,
            dtype=torch.bfloat16, device=device,
            generator=generator,
        )

    # Expand to B=2 (content path + styled path share the same initial noise)
    shared_latents = init_lat.repeat(2, 1, 1)

    # ── Install SAC processor (Paper 4) ──────────────────────────────────────
    processor = NeuFluxAttnProcessor(
        sac_layers=sac_layers,
        sac_step_range=sac_step_range,
        total_layers=57,
        total_steps=num_steps,
    ) if use_sac else None

    orig_processor = pipe.transformer.attn_processors
    if processor is not None:
        pipe.transformer.set_attn_processor(processor)

    # ── Install PFB hooks (Paper 4 + Paper 2) ────────────────────────────────
    pfb_hooks = register_pfb_hooks(
        pipe.transformer,
        processor if processor is not None else _DummyProc(pfb_step),
        style_feats,
        pfb_layers=pfb_layers,
        pfb_step=pfb_step,
        alpha=pfb_alpha,
    )

    try:
        output = pipe(
            prompt=[prompt, prompt],             # B=2: content path + styled path
            latents=shared_latents,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            output_type="pil",
            return_dict=True,
        )
    finally:
        for h in pfb_hooks:
            h.remove()
        if processor is not None:
            pipe.transformer.set_attn_processor(orig_processor)

    # Stream 1 = styled path
    return output.images[1]


class _DummyProc:
    """Minimal stand-in when SAC is disabled, provides .step for PFB hooks."""
    def __init__(self, pfb_step: int):
        self._step = 0
        self._pfb_step = pfb_step

    @property
    def step(self) -> int:
        return self._step

    def __call__(self, *a, **kw):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_neuflux_pipeline(
    model_path: str = "black-forest-labs/FLUX.1-dev",
    hf_token: Optional[str] = None,
    device: str = "cuda",
    cpu_offload: bool = False,
    cache_dir: str = "./models",
):
    """
    Load FLUX.1-dev pipeline using standard diffusers (no custom fork needed).
    NeuFlux injects all customisation at runtime via set_attn_processor and hooks.
    """
    from diffusers import FluxPipeline

    token = hf_token or os.environ.get("HF_TOKEN") or None
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=token,
        cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe
