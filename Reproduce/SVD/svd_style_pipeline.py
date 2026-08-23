"""
Training-free style personalization via SVD-based feature decomposition.

Paper: "A Training-Free Style-Personalization via SVD-Based Feature Decomposition"
       DGIST 2025

Implements on top of Infinity-2B (scale-wise autoregressive model).

Two mechanisms:
  PFB (Principal Feature Blending) — applied once at scale s=3 (si=2):
      Φ(F, α) = U · diag(exp(-α·i) · σᵢ) · Vᵀ   [SVD with exponential reweighting]
      F3_gen  ← Φ(F3_sty) + (F3_gen − Φ(F3_gen))

  SAC (Structural Attention Correction) — applied at scales s=3..12 (si=2..11):
      Q_gen = W_Q · F_con,  K_gen = W_K · F_con   [replace Q,K with content path]
      V_gen unchanged

Architecture note:
  - Infinity-2B inference uses slow_attn (customized_flash_attn=False)
  - SelfAttention forward: non-flash layout (B, H, L, c)
  - Dual-stream batch: index 0 = content path, index 1 = generation path
  - With CFG (bs=2*B=4): even indices = content, odd = generation
"""

import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from typing import List, Optional, Tuple

# ─────────────────────────── PFB helpers ────────────────────────────


def style_extractor(F_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Φ(F, α): SVD with exponential spectral reweighting (eq. 1 of paper).

    F_in: (B, C, 1, H, W)  — summed_codes format.
    Returns same shape.
    SVD is done per batch item on the (H*W × C) matrix.
    """
    B, C, T, H, W = F_in.shape
    # (B, C, H*W) → (B, H*W, C)
    F_2d = F_in.squeeze(2).reshape(B, C, H * W).permute(0, 2, 1).float()

    results = []
    for b in range(B):
        U, S, Vh = torch.linalg.svd(F_2d[b], full_matrices=False)  # (HW,r), (r,), (r,C)
        r = S.shape[0]
        w = torch.exp(-alpha * torch.arange(r, dtype=S.dtype, device=S.device))
        reweighted = (U * (S * w).unsqueeze(0)) @ Vh  # (HW, C)
        results.append(reweighted)

    F_out = torch.stack(results).permute(0, 2, 1).reshape(B, C, 1, H, W)
    return F_out.to(F_in.dtype)


def apply_pfb(F_gen: torch.Tensor, F_sty: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """PFB blending: Φ(F_sty) + (F_gen − Φ(F_gen))."""
    return style_extractor(F_sty, alpha) + (F_gen - style_extractor(F_gen, alpha))


# ────────────────────── Style feature extraction ─────────────────────


@torch.no_grad()
def get_style_codes(vae, style_img: torch.Tensor, scale_schedule) -> list:
    """
    Encode a style image with the BSQ-VAE and return its per-scale bit indices.

    style_img: (1, 3, H, W) in [-1, 1], on the same device as vae.
    Returns all_bit_indices — a list of per-scale tensors shaped (1, 1, h, w, d).
    """
    scale_schedule = [(int(t), int(h), int(w)) for t, h, w in scale_schedule]
    _, _, _, all_bit_indices, _, _ = vae.encode(style_img, scale_schedule=scale_schedule)
    return all_bit_indices


@torch.no_grad()
def compute_style_summed_codes(
    vae,
    all_bit_indices: list,
    scale_schedule,
    n_scales: int = 3,
    apply_spatial_patchify: bool = False,
) -> torch.Tensor:
    """
    Simulate the first n_scales steps of Infinity's summed_codes accumulation
    using the style image's per-scale codes.  Returns F3_sty of shape
    (1, d, 1, H64, W64).
    """
    vae_schedule = (
        [(t, 2 * h, 2 * w) for t, h, w in scale_schedule]
        if apply_spatial_patchify
        else list(scale_schedule)
    )
    # final_size is (1, H, W) — trilinear expects 5D input (B, C, D, H, W)
    final_size = vae_schedule[-1]  # e.g. (1, 64, 64) for 1024×1024

    summed = None
    for si in range(n_scales):
        codes = vae.quantizer.lfq.indices_to_codes(all_bit_indices[si], label_type="bit_label")
        if codes.dim() == 4:
            codes = codes.unsqueeze(2)             # (B, C, 1, h, w)
        upsampled = F.interpolate(
            codes, size=final_size, mode=vae.quantizer.z_interplote_up
        )                                          # (B, C, 1, H, W)
        summed = upsampled if summed is None else summed + upsampled

    return summed.float()


# ──────────────── SAC: patch / unpatch SelfAttention ─────────────────


def patch_self_attention_for_sac(infinity_model, sac_state: dict) -> dict:
    """
    Monkey-patch every SelfAttention module in the Infinity model so that,
    when sac_state['active'] is True, the generation-path Q and K (odd batch
    indices) are replaced by the content-path Q and K (even batch indices).

    Assumes slow_attn mode (customized_flash_attn=False), which is the default
    for Infinity-2B inference.  Layout: (B, H, L, c).

    Returns the dict of patched modules so they can be restored later.
    """
    from infinity.models.basic import (
        SelfAttention,
        apply_rotary_emb,
    )
    from torch.nn.functional import scaled_dot_product_attention as _slow_attn

    patched: dict = {}

    for name, module in infinity_model.named_modules():
        if not isinstance(module, SelfAttention):
            continue

        _orig = module.forward

        def _make_forward(m, _sac=sac_state, _slow=_slow_attn, _rope=apply_rotary_emb):
            def _forward(
                x,
                attn_bias_or_two_vector,
                attn_fn=None,
                scale_schedule=None,
                rope2d_freqs_grid=None,
                scale_ind=0,
            ):
                B, L, C = x.shape

                # ── fused QKV ──────────────────────────────────────────────
                qkv = F.linear(
                    input=x,
                    weight=m.mat_qkv.weight,
                    bias=torch.cat((m.q_bias, m.zero_k_bias, m.v_bias)),
                ).view(B, L, 3, m.num_heads, m.head_dim)

                # non-flash layout: (B, H, L, c)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

                if m.cos_attn:
                    scale_mul = m.scale_mul_1H11.clamp_max(m.max_scale_mul).exp()
                    q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()
                    k = F.normalize(k, dim=-1, eps=1e-12).contiguous()
                    v = v.contiguous()
                else:
                    q = q.contiguous()
                    k = k.contiguous()
                    v = v.contiguous()

                if rope2d_freqs_grid is not None:
                    q, k = _rope(
                        q, k, scale_schedule, rope2d_freqs_grid,
                        m.pad_to_multiplier, m.rope2d_normalized_by_hw, scale_ind,
                    )

                # ── KV caching ─────────────────────────────────────────────
                if m.caching:
                    if m.cached_k is None:
                        m.cached_k = k
                        m.cached_v = v
                    else:
                        # L_dim = 2 for non-flash (B, H, L, c)
                        k = m.cached_k = torch.cat((m.cached_k, k), dim=2)
                        v = m.cached_v = torch.cat((m.cached_v, v), dim=2)

                # ── SAC: replace generation Q,K with content Q,K ──────────
                # Even indices (0, 2, …) = content path
                # Odd  indices (1, 3, …) = generation path
                # Modifying k here also updates m.cached_k in-place because
                # k and m.cached_k reference the same tensor storage.
                if _sac.get("active") and B >= 2:
                    q[1::2] = q[0::2]
                    k[1::2] = k[0::2]

                # ── attention ──────────────────────────────────────────────
                if m.use_flex_attn and attn_fn is not None:
                    oup = attn_fn(q, k, v, scale=m.scale).transpose(1, 2).reshape(B, L, C)
                else:
                    oup = _slow(
                        query=q,
                        key=k,
                        value=v,
                        scale=m.scale,
                        attn_mask=attn_bias_or_two_vector,
                        dropout_p=0,
                    ).transpose(1, 2).reshape(B, L, C)

                return m.proj_drop(m.proj(oup))

            return _forward

        module._orig_forward = _orig
        module.forward = _make_forward(module)
        patched[name] = module

    return patched


def unpatch_self_attention(patched: dict) -> None:
    """Restore every SelfAttention module to its original forward."""
    for module in patched.values():
        module.forward = module._orig_forward
        del module._orig_forward


# ──────────────────── Text conditioning helpers ───────────────────────


def tile_text_cond(text_cond_tuple, n: int = 2):
    """
    Repeat a single-prompt text_cond_tuple n times for a batch of n identical
    prompts.  Used to create the dual-stream (B=2) conditioning.
    """
    kv_compact, lens, cu_seqlens_k, Ltext = text_cond_tuple
    kv_n = kv_compact.repeat(n, 1)      # (n*L_text, D)
    lens_n = lens * n                    # Python list [L]*n
    offset = cu_seqlens_k[-1]
    cu_n = cu_seqlens_k.clone()
    for _ in range(n - 1):
        cu_n = torch.cat([cu_n, cu_seqlens_k[1:] + offset])
        offset = offset + cu_seqlens_k[-1]
    return (kv_n, lens_n, cu_n, Ltext)


# ──────────────────── Core styled generation loop ────────────────────


def _styled_infer_cfg(
    inf,            # Infinity model instance
    vae,
    F3_sty: torch.Tensor,   # style features at scale 3, shape (1,d,1,H,W)
    pfb_alpha: float,
    sac_state: dict,
    *,
    scale_schedule,
    label_B_or_BLT,         # tiled for B=2
    g_seed=None,
    cfg_list: list,
    tau_list: list,
    cfg_sc: int = 3,
    top_k: int = 900,
    top_p: float = 0.97,
    cfg_insertion_layer=(-5,),
    vae_type: int = 1,
    gumbel: int = 0,
    sampling_per_bits: int = 1,
    use_pfb: bool = True,
    use_sac: bool = True,
    generation_steps: int = 12,   # how many of the 12 scales to run
    pfb_step: int = 3,            # 1-indexed; paper default s=3
    sac_steps: Optional[set] = None,  # 1-indexed set; paper default {3..12}
):
    """
    Modified version of Infinity.autoregressive_infer_cfg that injects:
      • sac_state['active'] = (s in sac_steps and use_sac) before each scale's block loop
      • PFB on summed_codes[1:2] immediately after s == pfb_step

    Returns a uint8 numpy array (H, W, 3) of the GENERATION path (index 1).

    This function is a targeted adaptation of the ~200-line
    autoregressive_infer_cfg method from infinity/models/infinity.py.
    Only the two injection points and the final decode step differ.
    """
    from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_ as _sample

    B = 2  # dual-stream: 0 = content, 1 = generation

    if g_seed is None:
        rng = None
    else:
        inf.rng.manual_seed(g_seed)
        rng = inf.rng

    assert len(cfg_list) >= len(scale_schedule)
    assert len(tau_list) >= len(scale_schedule)

    vae_scale_schedule = (
        [(t, 2 * h, 2 * w) for t, h, w in scale_schedule]
        if inf.apply_spatial_patchify
        else list(scale_schedule)
    )

    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT

    # ── CFG setup ─────────────────────────────────────────────────────
    if any(np.array(cfg_list) != 1):
        bs = 2 * B
        kv_compact_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_compact_un[total: total + le] = (inf.cfg_uncond)[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
    else:
        bs = B

    kv_compact = inf.text_norm(kv_compact)
    sos = cond_BD = inf.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = inf.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + inf.pos_start.expand(bs, 1, -1)

    with torch.amp.autocast("cuda", enabled=False):
        cond_BD_or_gss = inf.shared_ada_lin(cond_BD.float()).float().contiguous()

    # ── KV cache enable ───────────────────────────────────────────────
    from infinity.models.basic import SelfAttention as _SA
    _sa_modules = [m for _, m in inf.named_modules() if isinstance(m, _SA)]
    for sa in _sa_modules:
        sa.kv_caching(True)

    # ── CFG insertion layers ──────────────────────────────────────────
    abs_cfg_layers = []
    add_cfg_on_logits = False
    leng = len(inf.unregistered_blocks)
    for item in cfg_insertion_layer:
        if item == 0:
            add_cfg_on_logits = True
        elif item < 0:
            assert leng + item > 0
            abs_cfg_layers.append(leng + item)
        # item == 1 (add on probs) not used here

    # Convert paper 1-indexed step notation to sets for O(1) lookup
    _sac_steps = sac_steps if sac_steps is not None else set(range(3, len(scale_schedule) + 1))

    num_stages_minus_1 = len(scale_schedule) - 1
    summed_codes = None
    sac_state["active"] = False

    for si, pn in enumerate(scale_schedule):
        s = si + 1  # 1-indexed step (paper notation)
        cfg = cfg_list[si]
        if s > generation_steps:
            break

        # ── Activate SAC for steps in sac_steps (paper: S_fine = {3..12}) ─
        sac_state["active"] = use_sac and (s in _sac_steps)

        attn_fn = None
        if inf.use_flex_attn:
            attn_fn = inf.attn_fn_compile_dict.get(tuple(scale_schedule[: si + 1]), None)

        layer_idx = 0
        for block_idx, b in enumerate(inf.block_chunks):
            if inf.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = inf.add_lvl_embeding(last_stage, si, scale_schedule)
            if not inf.add_lvl_embeding_only_first_block:
                last_stage = inf.add_lvl_embeding(last_stage, si, scale_schedule)

            for m in b.module:
                last_stage = m(
                    x=last_stage,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_bias_or_two_vector=None,
                    attn_fn=attn_fn,
                    scale_schedule=scale_schedule,
                    rope2d_freqs_grid=inf.rope2d_freqs_grid,
                    scale_ind=si,
                )
                if (cfg != 1) and (layer_idx in abs_cfg_layers):
                    last_stage = cfg * last_stage[:B] + (1 - cfg) * last_stage[B:]
                    last_stage = torch.cat((last_stage, last_stage), 0)
                layer_idx += 1

        # ── logits & sampling (conditional branch only) ───────────────
        if (cfg != 1) and add_cfg_on_logits:
            logits_BlV = inf.get_logits(last_stage, cond_BD).mul(1 / tau_list[si])
            logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
        else:
            logits_BlV = inf.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau_list[si])

        tmp_bs, tmp_seq = logits_BlV.shape[:2]
        logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
        idx_Bld = _sample(
            logits_BlV, rng=rng,
            top_k=top_k or inf.top_k,
            top_p=top_p or inf.top_p,
            num_samples=1,
        )[:, :, 0].reshape(tmp_bs, tmp_seq, -1)

        # ── BSQ-VAE decode & summed_codes accumulation ────────────────
        assert pn[0] == 1
        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
        if inf.apply_spatial_patchify:
            idx_Bld = idx_Bld.permute(0, 3, 1, 2)
            idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)
            idx_Bld = idx_Bld.permute(0, 2, 3, 1)
        idx_Bld = idx_Bld.unsqueeze(1)  # (B, 1, h, w, d)

        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type="bit_label")
        if codes.dim() == 4:                       # (B, C, h, w) → need 5D for trilinear
            codes = codes.unsqueeze(2)             # (B, C, 1, h, w)

        if si != num_stages_minus_1:
            upsampled = F.interpolate(codes, size=vae_scale_schedule[-1],
                                      mode=vae.quantizer.z_interplote_up)
            summed_codes = upsampled if summed_codes is None else summed_codes + upsampled

            # ── PFB injection at pfb_step (paper: s=3) ────────────────
            # Applied to summed_codes AFTER accumulation: F3_sty and summed_codes[1:2]
            # are both accumulated over scales 1..pfb_step, so they share the same
            # representation space and the PFB blend is meaningful.
            # The modified summed_codes then conditions all remaining scales (4..12).
            if use_pfb and s == pfb_step:
                sc_dtype = summed_codes.dtype
                summed_codes = summed_codes.clone()
                summed_codes[1:2] = apply_pfb(
                    summed_codes[1:2].float(),       # accumulated generation path
                    F3_sty.to(summed_codes.device),  # accumulated style features
                    pfb_alpha,
                ).to(sc_dtype)

            next_size = vae_scale_schedule[si + 1]
            last_stage = F.interpolate(summed_codes, size=next_size,
                                       mode=vae.quantizer.z_interplote_up)
            last_stage = last_stage.squeeze(-3)
            if inf.apply_spatial_patchify:
                last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)
            last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
            last_stage = torch.permute(last_stage, [0, 2, 1])
        else:
            summed_codes = (
                codes if summed_codes is None else summed_codes + codes
            )

        if si != num_stages_minus_1:
            last_stage = inf.word_embed(inf.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs // B, 1, 1)

    # ── KV cache disable ──────────────────────────────────────────────
    for sa in _sa_modules:
        sa.kv_caching(False)

    # ── Decode generation path (index 1) ─────────────────────────────
    gen_codes = summed_codes[1:2].squeeze(-3)   # (1, d, H, W)
    img = vae.decode(gen_codes)                  # (1, 3, H, W) in [-1, 1]
    img = (img + 1) / 2
    img = img.clamp(0, 1).permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
    return img[0].cpu().numpy()  # (H, W, 3) RGB uint8 — PIL-compatible


# ──────────────── Public entry point ─────────────────────────────────


def preprocess_image(pil_img: Image.Image, h: int, w: int) -> torch.Tensor:
    """Centre-crop + resize → (1, 3, H, W) in [-1, 1]."""
    arr = np.array(pil_img.convert("RGB"))
    # centre-crop to target aspect ratio
    src_h, src_w = arr.shape[:2]
    tgt_ratio = w / h
    src_ratio = src_w / src_h
    if src_ratio > tgt_ratio:
        crop_w = int(src_h * tgt_ratio)
        x0 = (src_w - crop_w) // 2
        arr = arr[:, x0: x0 + crop_w]
    else:
        crop_h = int(src_w / tgt_ratio)
        y0 = (src_h - crop_h) // 2
        arr = arr[y0: y0 + crop_h]
    pil_cropped = Image.fromarray(arr).resize((w, h), Image.LANCZOS)
    t = to_tensor(pil_cropped)          # (3, H, W) in [0, 1]
    t = t.mul(2).sub(1)                 # → [-1, 1]
    return t.unsqueeze(0)               # (1, 3, H, W)


@torch.no_grad()
def generate_styled(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    *,
    style_image: Image.Image,
    prompt: str,
    scale_schedule,
    pfb_alpha: float = 1.0,
    cfg: float = 3.0,
    tau: float = 1.0,
    top_k: int = 900,
    top_p: float = 0.97,
    cfg_insertion_layer: int = -5,
    seed: int = 0,
    height: int = 1024,
    width: int = 1024,
    device: str = "cuda",
    use_pfb: bool = True,
    use_sac: bool = True,
    generation_steps: int = 12,
    pfb_step: int = 3,
    sac_steps: Optional[set] = None,
) -> Image.Image:
    """
    Generate a stylized image given a style reference and a text prompt.

    Dual-stream inference (B=2) following the paper:
      index 0 — content path (unmodified)
      index 1 — generation path with PFB at s=pfb_step and SAC at s∈sac_steps

    use_pfb / use_sac can be toggled for ablation.
    Returns a PIL Image.
    """
    from tools.run_infinity import encode_prompt

    _patchify = getattr(infinity, "apply_spatial_patchify", False)

    # ── Text encoding (tiled ×2 for dual-stream) ──────────────────────
    text_cond    = encode_prompt(text_tokenizer, text_encoder, prompt)
    text_cond_b2 = tile_text_cond(text_cond, n=2)

    # ── Style feature extraction (first pfb_step scales) ─────────────
    style_tensor    = preprocess_image(style_image, height, width).to(device)
    all_bit_indices = get_style_codes(vae, style_tensor, scale_schedule)
    F3_sty = compute_style_summed_codes(
        vae, all_bit_indices, scale_schedule,
        n_scales=pfb_step, apply_spatial_patchify=_patchify,
    ).to(device)

    # ── SAC: patch self-attention ─────────────────────────────────────
    sac_state: dict = {"active": False}
    patched = patch_self_attention_for_sac(infinity, sac_state)

    cfg_list = [cfg] * len(scale_schedule)
    tau_list = [tau] * len(scale_schedule)

    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
            img_np = _styled_infer_cfg(
                infinity, vae,
                F3_sty=F3_sty,
                pfb_alpha=pfb_alpha,
                sac_state=sac_state,
                scale_schedule=scale_schedule,
                label_B_or_BLT=text_cond_b2,
                g_seed=seed,
                cfg_list=cfg_list,
                tau_list=tau_list,
                cfg_insertion_layer=(cfg_insertion_layer,),
                vae_type=1,
                top_k=top_k,
                top_p=top_p,
                use_pfb=use_pfb,
                use_sac=use_sac,
                generation_steps=generation_steps,
                pfb_step=pfb_step,
                sac_steps=sac_steps,
            )
    finally:
        unpatch_self_attention(patched)
        sac_state["active"] = False

    return Image.fromarray(img_np)


def _hf_download_checkpoint(
    repo_id: str,
    filename: str,
    local_dir: str,
    token: str | None = None,
) -> str:
    """Download a single file from HF Hub into local_dir if not already present."""
    import os
    from huggingface_hub import hf_hub_download

    dest = os.path.join(local_dir, filename)
    if os.path.isfile(dest):
        return dest

    os.makedirs(local_dir, exist_ok=True)
    print(f"  Downloading {repo_id}/{filename} → {local_dir}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )
    return dest


def _hf_download_8b_weights(local_dir: str, token: str | None = None) -> str:
    """
    Download the Infinity-8B safetensors shards from FoundationVision/Infinity.
    Returns the local directory containing the shards and index file.
    """
    _INF_REPO = "FoundationVision/Infinity"
    _SHARDS = [
        "infinity_8b_weights/model-00001-of-00004.safetensors",
        "infinity_8b_weights/model-00002-of-00004.safetensors",
        "infinity_8b_weights/model-00003-of-00004.safetensors",
        "infinity_8b_weights/model-00004-of-00004.safetensors",
        "infinity_8b_weights/model.safetensors.index.json",
    ]
    weight_dir = os.path.join(local_dir, "infinity_8b_weights")
    os.makedirs(weight_dir, exist_ok=True)
    for f in _SHARDS:
        dest = os.path.join(local_dir, f)
        if not os.path.isfile(dest):
            print(f"  Downloading {f} …")
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=_INF_REPO, filename=f,
                            local_dir=local_dir, token=token)
    return weight_dir


def _patch_load_sharded_checkpoint() -> None:
    """
    Newer transformers removed load_sharded_checkpoint from modeling_utils.
    Inject a safetensors-based replacement so Infinity's torch_shard loader works.
    """
    import json
    import transformers.modeling_utils as _mu
    if hasattr(_mu, "load_sharded_checkpoint"):
        return  # already present — nothing to do

    from safetensors.torch import load_file as _load_st

    def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
        index_path = os.path.join(folder, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state_dict: dict = {}
        for shard in shard_files:
            print(f"  [8B] loading shard {shard} …")
            state_dict.update(_load_st(os.path.join(folder, shard), device="cpu"))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"  [8B] {len(unexpected)} unexpected keys (ignored)")
        if missing:
            print(f"  [8B] WARNING: {len(missing)} missing keys")
        return model

    _mu.load_sharded_checkpoint = load_sharded_checkpoint
    print("  [8B] patched transformers.modeling_utils.load_sharded_checkpoint")


def load_model(
    model_path: str,
    vae_path: str,
    t5_path: str,
    device: str = "cuda",
    cache_dir: str = "./models",
    hf_token: str | None = None,
    model_size: str = "8b",
):
    """
    Load Infinity (8B by default), BSQ-VAE-d32, and Flan-T5-XL.

    model_path : local path or directory for the Infinity checkpoint.
                 - 8B: directory containing the 4 safetensors shards
                       (auto-downloaded to cache_dir/infinity_8b_weights/)
                 - 2B: path to infinity_2b_reg.pth
                       (auto-downloaded to cache_dir/infinity_2b_reg.pth)
    vae_path   : local path to infinity_vae_d32.pth; auto-downloaded if missing.
    t5_path    : HuggingFace model ID or local dir for the T5 text encoder.
    device     : 'cuda' or 'cpu'
    cache_dir  : directory for downloaded checkpoints
    hf_token   : HuggingFace token; falls back to HF_TOKEN env variable.
    model_size : '8b' (default) or '2b'

    Returns (infinity, vae, text_tokenizer, text_encoder, scale_schedule).
    scale_schedule is for 1024×1024 (pn='1M', aspect 1:1), 12 scales.
    """
    import argparse

    from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

    # ── resolve HF token ──────────────────────────────────────────────
    token = hf_token or os.environ.get("HF_TOKEN") or None
    if token:
        os.environ["HF_TOKEN"] = token

    _INF_REPO = "FoundationVision/infinity"

    # ── auto-download model checkpoint if missing ─────────────────────
    if model_size == "8b":
        weight_dir = os.path.join(cache_dir, "infinity_8b_weights")
        index_file = os.path.join(weight_dir, "model.safetensors.index.json")
        if not os.path.isfile(index_file):
            weight_dir = _hf_download_8b_weights(cache_dir, token=token)
        model_path       = weight_dir
        _model_type      = "infinity_8b"
        _checkpoint_type = "torch_shard"
        # d56 VAE = codebook_dim 14 × 4 positions (2×2 spatial patchify)
        # → word_embed (3584, 56), head (112, 3584) in checkpoint
        _vae_type        = 14
        _patchify        = 1
        _vae_file        = "infinity_vae_d56_f8_14_patchify.pth"
    else:
        if not os.path.isfile(model_path):
            model_path = _hf_download_checkpoint(
                _INF_REPO, "infinity_2b_reg.pth", cache_dir, token=token
            )
        _model_type      = "infinity_2b"
        _checkpoint_type = "torch"
        _vae_type        = 32
        _patchify        = 0
        _vae_file        = "infinity_vae_d32.pth"

    # ── VAE ───────────────────────────────────────────────────────────
    if not os.path.isfile(vae_path):
        vae_path = _hf_download_checkpoint(
            _INF_REPO, _vae_file, cache_dir, token=token
        )

    # ── scale schedule for 1024×1024 (aspect 1:1) ────────────────────
    pn = "1M"
    h_div_w_key = 1.000
    raw_schedule = dynamic_resolution_h_w[h_div_w_key][pn]["scales"]
    scale_schedule = [(1, int(h), int(w)) for (_, h, w) in raw_schedule]

    # ── T5 text encoder ───────────────────────────────────────────────
    _abs_cache = os.path.abspath(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = _abs_cache
    os.environ["HF_HOME"] = _abs_cache
    text_tokenizer, text_encoder = load_tokenizer(t5_path)
    text_encoder = text_encoder.to(device)

    # ── BSQ-VAE ───────────────────────────────────────────────────────
    vae_args = argparse.Namespace(
        vae_type=_vae_type,
        vae_path=vae_path,
        apply_spatial_patchify=_patchify,
    )
    vae = load_visual_tokenizer(vae_args)
    vae = vae.to(device).eval()

    # ── Infinity transformer ──────────────────────────────────────────
    inf_args = argparse.Namespace(
        model_type=_model_type,
        model_path=model_path,
        checkpoint_type=_checkpoint_type,
        cache_dir=cache_dir,
        enable_model_cache=0,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        pn=pn,
        bf16=1,
        vae_type=_vae_type,
        text_channels=2048,
        apply_spatial_patchify=_patchify,
        use_flex_attn=0,
    )
    if model_size == "8b":
        _patch_load_sharded_checkpoint()
    infinity = load_transformer(vae, inf_args)
    infinity = infinity.to(device).eval()

    return infinity, vae, text_tokenizer, text_encoder, scale_schedule
