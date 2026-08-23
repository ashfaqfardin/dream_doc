"""
RealImageStable — Training-Free Real Image Editing on FLUX.1-dev
             via RF-Solver + RF-Edit  (ICML 2025, arXiv 2411.04746)

Reference codebase: github.com/wangjiangshan0725/RF-Solver-Edit

Algorithm:
    Inversion  (z_image → z_noise, guidance = 1, no text bias):
        For each step (t_curr → t_prev), with t_prev > t_curr:
            pred      = v_θ(z, t_curr)                       [NFE 1, second_order=False]
            z_mid     = z + (t_prev − t_curr)/2 · pred
            pred_mid  = v_θ(z_mid, t_mid)                    [NFE 2, second_order=True]
            δv/δt     = (pred_mid − pred) / ((t_prev−t_curr)/2)
            z         = z + dt · pred + ½ · dt² · δv/δt      [second-order Euler]
        At the last `inject_step` inversion steps, save V from
        single-stream layers (19–56) keyed by (t, second_order, layer_id).

    Editing  (z_noise → z_edit, guidance = guidance_scale):
        Same second-order loop, t_curr → t_prev with t_prev < t_curr.
        At the first `inject_step` editing steps, retrieve and inject
        the saved V features (same key scheme → timesteps match).

Usage:
    python NewWork/RealImageStable/run_real_image_stable.py \\
        --hf_token "$HF_TOKEN" \\
        --input          inputs/light.png \\
        --source_prompt  "Glowing marquee GLOW sign on a brick wall." \\
        --prompt         "Glowing marquee FLUX sign on a brick wall." \\
        --save_images
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.flux.pipeline_flux import calculate_shift

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_N_LAYERS = 57


# ──────────────────────── attention processor ─────────────────────────────────

class RFEditProcessor:
    """
    Implements the RF-Edit feature-sharing mechanism inside the attention call.

    During inversion  (mode='inversion', inject=True):
        Saves V from single-stream layers (≥ SINGLE_START) to self.features
        keyed by (current_t, second_order, layer_id).

    During editing  (mode='editing', inject=True):
        Retrieves and injects saved V into the corresponding single-stream layers
        using the same key — ensuring inversion and editing features match at
        every timestep of the trajectory.

    Handles both double-stream blocks (encoder_hidden_states not None)
    and single-stream blocks (encoder_hidden_states None).
    Single-stream blocks in newer diffusers do not have to_out on FluxAttention,
    so we check before calling it.
    """

    def __init__(self):
        self._layer     = 0
        self._txt_seq   = None   # text sequence length, cached from double-stream
        self.mode       = 'pass' # 'inversion' | 'editing' | 'pass'
        self.inject     = False  # whether injection/capture is active this step
        self.current_t  = 0.0   # sigma value for the current step (feature key)
        self.second_order = False  # True during midpoint NFE (feature key)
        self.features: dict = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        layer       = self._layer
        self._layer = (self._layer + 1) % _N_LAYERS

        B        = hidden_states.shape[0]
        head_dim = attn.inner_dim // attn.heads

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        q = q.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        txt_len = 0
        if encoder_hidden_states is not None:
            # ── double-stream block ───────────────────────────────────────────
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            txt_len       = eq.shape[2]
            self._txt_seq = txt_len
            q = torch.cat([eq, q], dim=2)
            k = torch.cat([ek, k], dim=2)
            v = torch.cat([ev, v], dim=2)
        else:
            # ── single-stream block: hidden_states = [txt | img] ─────────────
            txt_len = self._txt_seq or 0

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # ── RF-Edit V capture / inject (single-stream layers only) ────────────
        is_single = (encoder_hidden_states is None)  # layer >= SINGLE_START
        if self.inject and is_single:
            feat_key = (self.current_t, self.second_order, layer)
            if self.mode == 'inversion':
                # save image-token V to CPU to save GPU memory
                self.features[feat_key] = v[:, :, txt_len:, :].detach().cpu()
            elif self.mode == 'editing' and feat_key in self.features:
                stored = self.features[feat_key].to(v.device, dtype=v.dtype)
                v = v.clone()
                v[:, :, txt_len:, :] = stored

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim).to(q.dtype)

        if encoder_hidden_states is not None:
            txt_out = attn.to_add_out(out[:, :txt_len])
            img_out = attn.to_out[0](out[:, txt_len:])
            img_out = attn.to_out[1](img_out)
            return img_out, txt_out
        else:
            # single-stream: to_out lives on the block in newer diffusers, not on FluxAttention
            if hasattr(attn, "to_out") and attn.to_out is not None:
                out = attn.to_out[0](out)
                out = attn.to_out[1](out)
            return out


# ──────────────────────── helpers ─────────────────────────────────────────────

def _vae_encode(pipe, image: Image.Image, height: int, width: int, device) -> torch.Tensor:
    image  = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr    = np.array(image).astype(np.float32) / 255.0 * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        z = pipe.vae.encode(tensor).latent_dist.sample()
        z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return z


def _pack(z: torch.Tensor) -> torch.Tensor:
    B, C, H, W = z.shape
    return (z.view(B, C, H // 2, 2, W // 2, 2)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B, (H // 2) * (W // 2), C * 4))


def _unpack(z: torch.Tensor, height: int, width: int) -> torch.Tensor:
    H, W     = height // 8, width // 8
    B, _, C4 = z.shape
    C        = C4 // 4
    return (z.reshape(B, H // 2, W // 2, C, 2, 2)
             .permute(0, 3, 1, 4, 2, 5)
             .reshape(B, C, H, W))


def _make_image_ids(h_tokens: int, w_tokens: int, device) -> torch.Tensor:
    ids = torch.zeros(h_tokens, w_tokens, 3)
    ids[..., 1] = torch.arange(h_tokens)[:, None]
    ids[..., 2] = torch.arange(w_tokens)[None, :]
    return ids.reshape(-1, 3).to(device)


def _vae_decode(pipe, z: torch.Tensor) -> Image.Image:
    z_dec = z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.no_grad():
        img = pipe.vae.decode(z_dec).sample
    img = ((img.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _call_transformer(pipe, z, sigma, prompt_embeds, pooled_embeds,
                      text_ids, img_ids, guidance_scale, device):
    """Single transformer forward. sigma ∈ [0, 1]."""
    t         = torch.tensor([sigma], device=device, dtype=z.dtype)
    guidance  = torch.full((z.shape[0],), guidance_scale, device=device, dtype=z.dtype)
    return pipe.transformer(
        hidden_states         = z,
        timestep              = t,
        encoder_hidden_states = prompt_embeds,
        pooled_projections    = pooled_embeds,
        txt_ids               = text_ids,
        img_ids               = img_ids,
        guidance              = guidance,
        return_dict           = False,
    )[0]


# ──────────────────────── RF-Solver denoising loop ────────────────────────────

@torch.no_grad()
def _rf_solver_denoise(pipe, z, sigmas, prompt_embeds, pooled_embeds,
                       text_ids, img_ids, guidance_scale, inverse,
                       inject_step, processor, device):
    """
    Second-order RF-Solver loop (matching RF-Solver-Edit sampling.py: denoise()).

    sigmas : list of sigma values in GENERATION order [σ_max, …, 0].
             For inversion the list is reversed internally.

    inject_step : number of steps (counting from the high-noise end of each pass)
                  that have V capture (inversion) / V injection (editing).
    """
    n = len(sigmas) - 1   # number of denoising steps

    # inject_list in GENERATION order: True for first inject_step steps (high noise)
    inject_list = [True] * inject_step + [False] * (n - inject_step)

    if inverse:
        # inversion: run from low sigma to high sigma
        sigmas     = sigmas[::-1]        # [0, …, σ_max]
        inject_list = inject_list[::-1]  # True at HIGH-NOISE end (last inject_step steps)

    for i in range(n):
        t_curr = sigmas[i]
        t_prev = sigmas[i + 1]
        dt     = t_prev - t_curr        # negative for generation, positive for inversion

        do_inject = inject_list[i]
        # feature key uses the HIGHER sigma (t_prev in inversion, t_curr in editing)
        key_t = t_prev if inverse else t_curr

        # ── first-order NFE ────────────────────────────────────────────────────
        processor.inject       = do_inject
        processor.current_t    = key_t
        processor.second_order = False
        pred = _call_transformer(pipe, z, t_curr,
                                 prompt_embeds, pooled_embeds,
                                 text_ids, img_ids, guidance_scale, device)

        # ── midpoint ──────────────────────────────────────────────────────────
        z_mid = z + (dt / 2.0) * pred
        t_mid = t_curr + dt / 2.0

        processor.inject       = do_inject
        processor.current_t    = key_t
        processor.second_order = True
        pred_mid = _call_transformer(pipe, z_mid, t_mid,
                                     prompt_embeds, pooled_embeds,
                                     text_ids, img_ids, guidance_scale, device)

        # ── second-order update ───────────────────────────────────────────────
        # x_{t+dt} = x_t + dt·v + ½·dt²·(dv/dt)
        dv_dt = (pred_mid - pred) / (dt / 2.0)
        z     = z + dt * pred + 0.5 * (dt ** 2) * dv_dt

    return z


# ──────────────────────── main pipeline ───────────────────────────────────────

def load_pipeline(model_path, device, hf_token=None, cache_dir=None):
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    ).to(device)
    return pipe


@torch.no_grad()
def run(pipe: FluxPipeline, args):
    input_image = Image.open(args.input).convert("RGB")
    device      = pipe.device
    H, W        = args.height, args.width

    print(f"  input         : {args.input}")
    print(f"  source_prompt : {args.source_prompt}")
    print(f"  edit_prompt   : {args.prompt}")
    print(f"  steps={args.num_steps}  inject_step={args.inject_step}  "
          f"guidance={args.guidance_scale}")

    # ── 1. Encode text ────────────────────────────────────────────────────────
    src_embeds,  src_pooled,  text_ids = pipe.encode_prompt(
        prompt=args.source_prompt, prompt_2=None,
        device=device, max_sequence_length=512,
    )
    edit_embeds, edit_pooled, _        = pipe.encode_prompt(
        prompt=args.prompt, prompt_2=None,
        device=device, max_sequence_length=512,
    )

    # ── 2. VAE-encode real image ──────────────────────────────────────────────
    z_0     = _vae_encode(pipe, input_image, H, W, device)
    z       = _pack(z_0)
    img_ids = _make_image_ids(H // 16, W // 16, device)

    # ── 3. Build mu-shifted sigma schedule ───────────────────────────────────
    out_seq_len = (H // 16) * (W // 16)
    mu = calculate_shift(
        out_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len",  4096),
        pipe.scheduler.config.get("base_shift",         0.5),
        pipe.scheduler.config.get("max_shift",          1.16),
    )
    # sigmas in GENERATION order: [σ_max, …, σ_1, 0]  (decreasing)
    timesteps_t = torch.linspace(1, 0, args.num_steps + 1)
    sigmas      = pipe.scheduler.time_shift(mu, 1.0, timesteps_t).tolist()

    # ── 4. Install processor ──────────────────────────────────────────────────
    processor = RFEditProcessor()
    pipe.transformer.set_attn_processor(processor)

    # ── 5. Inversion  (z_image → z_noise, guidance = 1) ─────────────────────
    print("  inverting …")
    processor.mode = 'inversion'
    z = _rf_solver_denoise(
        pipe, z, sigmas,
        src_embeds, src_pooled, text_ids, img_ids,
        guidance_scale = 1.0,          # no guidance bias during inversion
        inverse        = True,
        inject_step    = args.inject_step,
        processor      = processor,
        device         = device,
    )
    print(f"  inversion done — {len(processor.features)} feature tensors saved.")

    # ── 6. Editing  (z_noise → z_edit, guidance = guidance_scale) ────────────
    print("  editing …")
    processor.mode = 'editing'
    z = _rf_solver_denoise(
        pipe, z, sigmas,
        edit_embeds, edit_pooled, text_ids, img_ids,
        guidance_scale = args.guidance_scale,
        inverse        = False,
        inject_step    = args.inject_step,
        processor      = processor,
        device         = device,
    )

    # ── 7. Decode ─────────────────────────────────────────────────────────────
    z_final      = _unpack(z, H, W)
    edited_image = _vae_decode(pipe, z_final)

    if args.save_images:
        os.makedirs(args.out_dir, exist_ok=True)
        in_p  = os.path.join(args.out_dir, "input.png")
        out_p = os.path.join(args.out_dir, "edited.png")
        input_image.resize((W, H), Image.LANCZOS).save(in_p)
        edited_image.save(out_p)
        print(f"  saved → {in_p}")
        print(f"  saved → {out_p}")

    return input_image, edited_image


# ──────────────────────── CLI ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RF-Solver + RF-Edit: training-free real image editing on FLUX.1-dev."
    )
    p.add_argument("--input",          required=True,
                   help="Path to the real input image.")
    p.add_argument("--source_prompt",  required=True,
                   help="Description of the INPUT image (drives inversion).")
    p.add_argument("--prompt",         required=True,
                   help="Target description for the edit (drives generation).")
    p.add_argument("--num_steps",      type=int,   default=25,
                   help="Inversion / denoising steps. Default 25 (matches paper).")
    p.add_argument("--inject_step",    type=int,   default=20,
                   help="Steps (from the high-noise end) that share V features "
                        "between inversion and editing. Default 20 (paper default). "
                        "Lower → more edit freedom, less structure preservation.")
    p.add_argument("--guidance_scale", type=float, default=2.0,
                   help="CFG guidance for the editing pass. Default 2 (paper default). "
                        "Increase to 3–5 for stronger prompt adherence.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--height",         type=int,   default=1024)
    p.add_argument("--width",          type=int,   default=1024)
    p.add_argument("--model_path",     default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--hf_token",       required=True)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--cache_dir",      default="./models")
    p.add_argument("--out_dir",        default="results/realimageStable")
    p.add_argument("--save_images",    action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n[RealImageStable] Loading {args.model_path} …")
    pipe = load_pipeline(args.model_path, args.device, args.hf_token, args.cache_dir)
    print("[RealImageStable] Model loaded.\n")
    run(pipe, args)
    print("[RealImageStable] Done.")
