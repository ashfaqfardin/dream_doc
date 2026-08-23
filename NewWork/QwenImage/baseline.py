# -*- coding: utf-8 -*-
"""
QwenImage/baseline.py — Write-once latent-anchoring baseline for sequential
object insertion via Qwen-Image-Edit-2509 (Wan MMDiT architecture).

Invariant
---------
Every background pixel is generated AT MOST ONCE across the full session.
z_anchor grows monotonically: after step k it contains z_0 for the background
and frozen z_j outputs for all placed objects j ≤ k.

Key fixes (vs. prior write_once_compose.py)
-------------------------------------------
  NOTE: we do NOT patch pipe.vae.encode/decode. The pipeline's _encode_vae_image
  normalises with latents_mean.view(1,C,1,1,1) — a 5-D broadcast — so the VAE
  must return 5-D latents (B,C,T,H,W). The pipeline already handles two
  conditioning images correctly: it packs each one separately and concatenates
  their packed sequences along dim=1, so no special VAE treatment is needed.

  _patch_prepare_latents() — safety net for residual dim-count mismatches;
  should be a no-op with the current diffusers pipeline version.

Usage (Colab)
-------------
  # Diagnostics only (no sketch needed)
  python NewWork/QwenImage/baseline.py \\
      --base_prompt "empty minimalist living room, hardwood floor, white walls, 4K" \\
      --diagnostics --diag_steps 5 \\
      --hf_token $HF_TOKEN --cache_dir ./models --lightning \\
      --out_dir results/baseline_diag

  # Write-once composition from edits.json
  python NewWork/QwenImage/baseline.py \\
      --base_prompt "empty minimalist living room, hardwood floor, white walls, 4K" \\
      --edits_json  NewWork/KontextEval/inputs/edits.json \\
      --sketch_dir  NewWork/KontextEval/inputs \\
      --out_dir     results/baseline_compose \\
      --hf_token $HF_TOKEN --cache_dir ./models \\
      --lightning --band_width 16 --alpha_bg 0.9
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from huggingface_hub import hf_hub_download
    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False

try:
    from skimage.metrics import structural_similarity
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

# ── Constants ──────────────────────────────────────────────────────────────────

LIGHTNING_REPO      = "lightx2v/Qwen-Image-Lightning"
LIGHTNING_SUBFOLDER = "Qwen-Image-Edit-2509"

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


# ── ObjectSlot ────────────────────────────────────────────────────────────────

@dataclass
class ObjectSlot:
    name:        str
    description: str
    mask_np:     np.ndarray            # (H, W) bool — pixel space
    color:       Tuple[int, int, int]
    depth_rank:  int = 0
    sketch_crop: Optional[Image.Image] = None

    @property
    def centroid(self) -> Tuple[float, float]:
        rows, cols = np.where(self.mask_np)
        if len(rows) == 0:
            return (0.5, 0.5)
        H, W = self.mask_np.shape
        return (float(rows.mean()) / H, float(cols.mean()) / W)

    @property
    def mask_area_frac(self) -> float:
        return float(self.mask_np.mean())


# ── Room-prior mask (used when no explicit spatial mask is available) ──────────

def _make_room_prior_mask(
    lat_h: int, lat_w: int,
    top_frac: float = 0.25, side_frac: float = 0.12, temp: float = 0.04,
) -> torch.Tensor:
    """(1,1,lat_h,lat_w) ∈ [0,1]: 1=floor/center (free), 0=ceiling/walls (anchored)."""
    y  = torch.linspace(0.0, 1.0, lat_h)
    x  = torch.linspace(0.0, 1.0, lat_w)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    v  = torch.sigmoid((yy - top_frac) / temp)
    h  = torch.sigmoid((0.5 - side_frac - (xx - 0.5).abs()) / temp)
    return (v * h).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)


def slots_from_sketches(
    sketch_paths: List[str], descriptions: List[str],
    height: int, width: int,
    top_frac: float = 0.25, side_frac: float = 0.12,
) -> List["ObjectSlot"]:
    """Create slots from individual sketch files (white-background PNGs).

    Uses a room-prior mask for the write-once callback since no explicit
    spatial masks are provided. z_anchor update uses the same room prior.
    """
    if len(sketch_paths) != len(descriptions):
        raise ValueError("sketch_paths and descriptions must have the same length.")
    prior_lat = _make_room_prior_mask(height // 8, width // 8, top_frac, side_frac)
    prior_np  = F.interpolate(prior_lat, size=(height, width),
                              mode="bilinear", align_corners=False)
    prior_np  = (prior_np[0, 0].numpy() > 0.5)

    import colorsys
    slots: List[ObjectSlot] = []
    for idx, (sp, desc) in enumerate(zip(sketch_paths, descriptions)):
        hue_step = 360 // max(len(sketch_paths), 1)
        r, g, b  = colorsys.hsv_to_rgb(((idx * hue_step) % 360) / 360.0, 0.7, 0.9)
        slots.append(ObjectSlot(
            name        = f"obj_{idx + 1}",
            description = desc,
            mask_np     = prior_np.copy(),
            color       = (int(r * 255), int(g * 255), int(b * 255)),
            depth_rank  = idx,
            sketch_crop = Image.open(sp).convert("RGB"),
        ))
    print(f"[SketchLoader] {len(slots)} slot(s) from sketches — room-prior mask.")
    return slots


def assign_depth_order(slots: List[ObjectSlot]) -> List[ObjectSlot]:
    ordered = sorted(slots, key=lambda s: (s.centroid[0], s.centroid[1]))
    for rank, slot in enumerate(ordered):
        slot.depth_rank = rank
    return ordered


# ── Pipeline loading ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def _fix_cat_dims():
    """Patch torch.cat to fix two-image shape mismatches during pipe().

    Fix 1 — dim-count mismatch: unsqueeze a missing leading batch dim when one
    tensor has one fewer dimension than the other.

    Fix 2 — channel/seq restack: the installed pipeline's prepare_latents for N
    conditioning images packs latents as (B, C_pack, N*seq) by concatenating
    sequences. The denoising loop cats at dim=1 (channel axis), so image_latents
    must be (B, N*C_pack, seq). Detected when both tensors are 3-D, cat is at
    dim=1, and the second tensor's last dim is a whole multiple of the first's.
    Active ONLY inside the pipe() call — all other torch.cat calls are unaffected.
    """
    _orig_cat = torch.cat

    def _safe_cat(tensors, dim=0, *args, **kwargs):
        if (len(tensors) == 2
                and isinstance(tensors[0], torch.Tensor)
                and isinstance(tensors[1], torch.Tensor)):
            a, b = tensors[0], tensors[1]
            # Fix 1: dim-count mismatch — unsqueeze the smaller tensor
            if abs(a.dim() - b.dim()) == 1:
                if a.dim() < b.dim():
                    a = a.unsqueeze(0)
                else:
                    b = b.unsqueeze(0)
                tensors = [a, b]
            # Fix 2: same rank, exactly one non-cat dim where b is a whole multiple
            # of a.  Folds that factor into the cat dim so the cat succeeds.
            # Handles any tensor format: (B,seq,C), (B,C,seq), (seq,C), etc.
            elif a.dim() == b.dim():
                mismatches = [
                    (d, a.shape[d], b.shape[d])
                    for d in range(a.dim())
                    if d != dim and a.shape[d] != b.shape[d]
                ]
                if len(mismatches) == 1:
                    md, a_sz, b_sz = mismatches[0]
                    if b_sz > a_sz and b_sz % a_sz == 0:
                        n = b_sz // a_sz
                        # Split b's dim md → (n, a_sz), then move n to cat dim
                        new_shape = list(b.shape)
                        new_shape[md:md + 1] = [n, a_sz]
                        b_split = b.reshape(new_shape)
                        # Permute: remove n from position md, insert at dim
                        perm = list(range(b_split.dim()))
                        perm.pop(md)
                        perm.insert(dim, md)
                        b_perm = b_split.permute(perm)
                        # Merge dim and dim+1 to produce n× larger cat axis
                        fs = list(b_perm.shape)
                        fs[dim:dim + 2] = [fs[dim] * fs[dim + 1]]
                        b = b_perm.contiguous().reshape(fs)
                        tensors = [a, b]
        return _orig_cat(tensors, dim, *args, **kwargs)

    torch.cat = _safe_cat
    try:
        yield
    finally:
        torch.cat = _orig_cat


def _patch_prepare_latents(pipe) -> None:
    """Fix prepare_latents for multi-image conditioning.

    The installed pipeline packs each conditioning image into (B, C_pack, seq) and
    concatenates them along seq-dim → (B, C_pack, N*seq). But the denoising loop
    cats [noisy, image_latents] at dim=1 (channel dim), so image_latents must be
    (B, N*C_pack, seq) — N images channel-stacked at the SAME seq length.

    Fix: when image_latents.shape[-1] is an integer multiple of latents.shape[-1],
    reshape (B, C, N*seq) → (B, N*C, seq).

    For pipelines that already return the correct shape (same last-dim for both),
    this is a no-op.  Dim-count mismatch is also fixed as a fallback.
    """
    _orig = pipe.prepare_latents

    def _patched(*args, **kwargs):
        result = _orig(*args, **kwargs)
        if isinstance(result, (tuple, list)) and len(result) == 2:
            latents, image_latents = result
            if isinstance(latents, torch.Tensor) and isinstance(image_latents, torch.Tensor):
                if latents.dim() == 3 and image_latents.dim() == 3:
                    lat_last = latents.shape[-1]
                    img_last = image_latents.shape[-1]
                    if img_last > lat_last and img_last % lat_last == 0:
                        # (B, C_pack, N*seq) → (B, N*C_pack, seq)
                        n = img_last // lat_last
                        B, C, _ = image_latents.shape
                        image_latents = (
                            image_latents
                            .view(B, C, n, lat_last)
                            .permute(0, 2, 1, 3)   # (B, n, C, seq)
                            .reshape(B, n * C, lat_last)
                        )
                elif latents.dim() != image_latents.dim():
                    if latents.dim() > image_latents.dim():
                        image_latents = image_latents.unsqueeze(0)
                    else:
                        latents = latents.unsqueeze(0)
            return latents, image_latents
        return result

    pipe.prepare_latents = _patched
    print("[patch] prepare_latents: multi-image channel-restack active")


def load_pipeline(
    model:           str           = "Qwen/Qwen-Image-Edit-2509",
    hf_token:        Optional[str] = None,
    cache_dir:       str           = "./models",
    lightning:       bool          = True,
    lightning_steps: int           = 8,
    dtype:           torch.dtype   = torch.bfloat16,
) -> "QwenImageEditPlusPipeline":
    if not _HAS_DIFFUSERS:
        raise ImportError("pip install diffusers huggingface_hub")
    kwargs: Dict = dict(torch_dtype=dtype, token=hf_token, cache_dir=cache_dir)
    if lightning:
        kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(
            LIGHTNING_SCHEDULER_CFG
        )
    pipe = QwenImageEditPlusPipeline.from_pretrained(model, **kwargs)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if vram_gb < 65:
            print(f"  [load] GPU {vram_gb:.0f} GB → cpu_offload")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        pipe.to("cpu")
    if lightning:
        wname = (
            f"Qwen-Image-Edit-2509-Lightning-{lightning_steps}steps-V1.0-bf16.safetensors"
        )
        lora_path = hf_hub_download(
            repo_id=LIGHTNING_REPO,
            filename=f"{LIGHTNING_SUBFOLDER}/{wname}",
            token=hf_token, cache_dir=cache_dir,
        )
        pipe.load_lora_weights(lora_path)
        print(f"  [load] Lightning {lightning_steps}-step LoRA")
    _patch_prepare_latents(pipe)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── VAE helpers ───────────────────────────────────────────────────────────────

def encode_latent(pipe, pil_img: Image.Image, height: int, width: int) -> torch.Tensor:
    """PIL → unscaled VAE latent (1, C, H/8, W/8) on CPU float32."""
    img = pil_img.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1,3,1,H,W)
    try:
        dev  = next(pipe.vae.parameters()).device
        dtyp = next(pipe.vae.parameters()).dtype
    except StopIteration:
        dev, dtyp = torch.device("cpu"), torch.float32
    with torch.no_grad():
        z = pipe.vae.encode(t.to(dev, dtyp)).latent_dist.mean   # (1,C,1,H/8,W/8)
    return z[:, :, 0].cpu().float()                              # (1,C,H/8,W/8)


def decode_latent(pipe, z: torch.Tensor) -> Image.Image:
    """Unscaled VAE latent (1, C, H/8, W/8) → PIL RGB."""
    try:
        dev  = next(pipe.vae.parameters()).device
        dtyp = next(pipe.vae.parameters()).dtype
    except StopIteration:
        dev, dtyp = torch.device("cpu"), torch.float32
    with torch.no_grad():
        dec = pipe.vae.decode(z.unsqueeze(2).to(dev, dtyp)).sample  # (1,3,1,H,W)
    dec = dec[:, :, 0].float().clamp(-1, 1) / 2 + 0.5
    arr = (dec[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


# ── Mask utilities ────────────────────────────────────────────────────────────

def mask_to_latent(mask_np: np.ndarray, lat_h: int, lat_w: int) -> torch.Tensor:
    m = Image.fromarray(mask_np.astype(np.uint8) * 255).resize(
        (lat_w, lat_h), Image.LANCZOS
    )
    return (torch.from_numpy(np.array(m)).float() / 255.0).unsqueeze(0).unsqueeze(0)


def make_band_weight(mask_np: np.ndarray, band_width: int,
                     lat_h: int, lat_w: int) -> torch.Tensor:
    """(1,1,lat_h,lat_w) ∈ [0,1]: 0=object zone (free), 1=background (anchored).
    Harmonization band ramps 0→1 over band_width pixels outside the mask.
    """
    if _HAS_SCIPY and band_width > 0:
        dist = distance_transform_edt(~mask_np).astype(np.float32)
        weight_px = np.clip(dist / max(band_width, 1), 0.0, 1.0)
        weight_px[mask_np] = 0.0
    else:
        weight_px = (~mask_np).astype(np.float32)
    wimg  = Image.fromarray((weight_px * 255).clip(0, 255).astype(np.uint8))
    w_lat = np.array(wimg.resize((lat_w, lat_h), Image.BILINEAR)) / 255.0
    return torch.from_numpy(w_lat).float().unsqueeze(0).unsqueeze(0)


# ── Write-once state ──────────────────────────────────────────────────────────

class WriteOnceState:
    """Progressive anchor: z_anchor = z_0 in background, z_j in placed slot j."""
    def __init__(self, z_base: torch.Tensor):
        self.z_anchor = z_base.clone().cpu().float()

    def update(self, z_result: torch.Tensor, mask_lat: torch.Tensor) -> None:
        m = F.interpolate(mask_lat.cpu().float(), size=self.z_anchor.shape[-2:],
                          mode="bilinear", align_corners=False)
        self.z_anchor = (1.0 - m) * self.z_anchor + m * z_result.cpu().float()


# ── Write-once denoising callback ─────────────────────────────────────────────

class WriteOnceCallback:
    """At each denoising step: anchor background to z_anchor, leave object zone free."""

    def __init__(self, state: WriteOnceState, band_weight: torch.Tensor,
                 alpha: float = 1.0, start_step: int = 0):
        self.state       = state
        self.band_weight = band_weight.cpu().float()
        self.alpha       = alpha
        self.start_step  = start_step
        self._noise: Optional[torch.Tensor] = None

    def __call__(self, pipe, i: int, t, callback_kwargs: dict) -> dict:
        if i < self.start_step:
            return callback_kwargs
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        dev, dtyp = latents.device, latents.dtype
        scaling   = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        z_scaled  = (self.state.z_anchor * scaling).to(dev, dtyp)

        if self._noise is None or self._noise.shape != z_scaled.shape:
            self._noise = torch.randn_like(z_scaled)
        noise = self._noise.to(dev, dtyp)

        T     = float(getattr(pipe.scheduler.config, "num_train_timesteps", 1000))
        sigma = (float(t.mean()) if t.dim() > 0 else float(t)) / T
        z_ref_noised = (1.0 - sigma) * z_scaled + sigma * noise

        is_5d = latents.dim() == 5
        frame = latents[:, :, 0] if is_5d else latents
        lat_h, lat_w = frame.shape[-2], frame.shape[-1]

        if z_ref_noised.shape[-2:] != (lat_h, lat_w):
            z_ref_noised = F.interpolate(
                z_ref_noised.float(), size=(lat_h, lat_w),
                mode="bilinear", align_corners=False,
            ).to(dtyp)

        w = F.interpolate(
            self.band_weight.to(dev, torch.float32), size=(lat_h, lat_w),
            mode="bilinear", align_corners=False,
        ).to(dtyp) * self.alpha

        anchored = (1.0 - w) * frame + w * z_ref_noised

        if is_5d:
            out = latents.clone(); out[:, :, 0] = anchored
        else:
            out = anchored
        callback_kwargs["latents"] = out
        return callback_kwargs


# ── Qwen call ─────────────────────────────────────────────────────────────────

def run_qwen(pipe, image, prompt: str, seed: int, num_steps: int, guidance: float,
             height: int, width: int,
             negative_prompt: str = "blurry, distorted, low quality, watermark, artifacts",
             callback=None) -> Image.Image:
    gen   = torch.Generator(device=pipe.device).manual_seed(seed)
    extra: Dict = {}
    if callback is not None:
        extra["callback_on_step_end"]               = callback
        extra["callback_on_step_end_tensor_inputs"] = ["latents"]
    with _fix_cat_dims():
        return pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            image=image, num_inference_steps=num_steps,
            true_cfg_scale=guidance, height=height, width=width,
            generator=gen, **extra,
        ).images[0]


# ── Object synthesis (sketch → photorealistic object) ─────────────────────────

def _tight_crop(img: Image.Image, pad: int = 24) -> Image.Image:
    arr  = np.array(img.convert("RGB"))
    mask = np.any(arr < 240, axis=2)
    if not mask.any():
        return img.resize((512, 512), Image.LANCZOS)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = max(0, rows[0] - pad), min(arr.shape[0], rows[-1] + pad)
    c0, c1 = max(0, cols[0] - pad), min(arr.shape[1], cols[-1] + pad)
    side = max(r1 - r0, c1 - c0)
    return img.crop((c0, r0, c0 + side, r0 + side)).resize((512, 512), Image.LANCZOS)


def synth_object(pipe, slot: ObjectSlot, seed: int, num_steps: int,
                 guidance: float, height: int, width: int) -> Image.Image:
    sketch_in = (slot.sketch_crop or Image.new("RGB", (512, 512), (200, 200, 200)))
    sketch_in = sketch_in.convert("RGB").resize((width, height), Image.LANCZOS)
    cy, _     = slot.centroid
    persp     = ("lower, larger (close)" if cy > 0.60 else
                 "upper, smaller (far)" if cy < 0.40 else "mid-scene scale")
    prompt = (
        f"Render the object in this sketch as a photorealistic {slot.description}. "
        f"Use the sketch for exact shape. Plain white background, studio lighting. "
        f"Object placed {persp}."
    )
    result = run_qwen(
        pipe, image=sketch_in, prompt=prompt, seed=seed,
        num_steps=num_steps, guidance=guidance, height=height, width=width,
        negative_prompt="background, room, floor, shadow, multiple objects, blurry",
    )
    return _tight_crop(result)


def overlay_mask_on_scene(scene: Image.Image, mask_np: np.ndarray,
                           color: Tuple[int, int, int] = (255, 200, 0),
                           alpha: float = 0.35) -> Image.Image:
    arr = np.array(scene.convert("RGB")).astype(float)
    overlay = np.array(color, dtype=float)
    arr[mask_np] = (1.0 - alpha) * arr[mask_np] + alpha * overlay
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# ── Scene composer ────────────────────────────────────────────────────────────

class SceneComposer:
    def __init__(self, pipe, height: int = 1024, width: int = 1024,
                 seed: int = 42, num_steps: int = 8, guidance: float = 1.0,
                 obj_guidance: float = 1.0):
        self.pipe        = pipe
        self.H, self.W   = height, width
        self.seed        = seed
        self.num_steps   = num_steps
        self.guidance    = guidance
        self.obj_guidance = obj_guidance

    def compose(self, base_img: Image.Image, slots: List[ObjectSlot],
                band_width: int = 16, alpha_bg: float = 1.0,
                out_dir: Optional[str] = None) -> List[Image.Image]:
        H, W = self.H, self.W
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        print(f"\n[Compose] Encoding base scene ...")
        z_base        = encode_latent(self.pipe, base_img, H, W)
        lat_h, lat_w  = z_base.shape[-2], z_base.shape[-1]
        print(f"          z_base: {tuple(z_base.shape)}")

        state  = WriteOnceState(z_base)
        scene  = base_img.convert("RGB")
        images = [base_img]
        if out_dir:
            base_img.save(os.path.join(out_dir, "base_scene.png"))

        for i, slot in enumerate(slots):
            step = i + 1
            print(f"\n[Step {step}/{len(slots)}]  {slot.name}  '{slot.description}'")

            obj_img = synth_object(
                self.pipe, slot, seed=self.seed,
                num_steps=self.num_steps, guidance=self.obj_guidance,
                height=H, width=W,
            )
            if out_dir:
                obj_img.save(os.path.join(out_dir, f"obj_{slot.name}.png"))

            scene_guided = overlay_mask_on_scene(scene, slot.mask_np, slot.color)
            band_w   = make_band_weight(slot.mask_np, band_width, lat_h, lat_w)
            callback = WriteOnceCallback(state, band_w, alpha=alpha_bg)

            cy, cx = slot.centroid
            vert   = "lower" if cy > 0.55 else "upper" if cy < 0.45 else "mid"
            horiz  = "left"  if cx < 0.40 else "right" if cx > 0.60 else "center"
            prompt = (
                f"\nPicture 1 shows a room with a highlighted placement region. "
                f"Picture 2 shows a {slot.description}. "
                f"Place the {slot.description} from Picture 2 into the highlighted "
                f"{vert}-{horiz} area of the room in Picture 1. "
                f"Match room perspective and lighting. Add a realistic contact shadow. "
                f"Do not change any other part of the room."
            )

            print(f"  [K] Inserting at centroid ({cy:.2f}, {cx:.2f}) ...")
            result = run_qwen(
                self.pipe,
                image=[scene_guided, obj_img.convert("RGB")],
                prompt=prompt, seed=self.seed,
                num_steps=self.num_steps, guidance=self.guidance,
                height=H, width=W, callback=callback,
            )
            if out_dir:
                result.save(os.path.join(out_dir, f"result_step{step}_{slot.name}.png"))

            z_result = encode_latent(self.pipe, result, H, W)
            mask_lat = mask_to_latent(slot.mask_np, lat_h, lat_w)
            state.update(z_result, mask_lat)

            scene = result
            images.append(result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return images


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _img_metrics(img_a: Image.Image, img_b: Image.Image,
                 height: int, width: int) -> Dict:
    a = np.array(img_a.convert("RGB").resize((width, height), Image.LANCZOS))
    b = np.array(img_b.convert("RGB").resize((width, height), Image.LANCZOS))
    out: Dict = {}
    if _HAS_SKIMAGE:
        out["ssim"] = round(float(structural_similarity(
            a, b, channel_axis=2, win_size=7, data_range=255)), 4)
    mse = float(np.mean((a.astype(float) - b.astype(float)) ** 2))
    out["psnr"] = round((10 * np.log10(255 ** 2 / mse)) if mse > 0 else float("inf"), 2)
    return out


def diag_vae_roundtrip(pipe, img: Image.Image, height: int, width: int) -> Dict:
    z   = encode_latent(pipe, img, height, width)
    rec = decode_latent(pipe, z)
    m   = _img_metrics(img, rec, height, width)
    m["label"] = "vae_roundtrip"
    print(f"[Diag] VAE round-trip  SSIM={m.get('ssim','n/a')}  PSNR={m.get('psnr','n/a'):.1f}")
    return m


def diag_nullop(pipe, img: Image.Image, height: int, width: int,
                seed: int = 42, num_steps: int = 8, guidance: float = 1.0) -> Dict:
    img_rs = img.convert("RGB").resize((width, height), Image.LANCZOS)
    result = run_qwen(pipe, image=img_rs,
                      prompt="Do not change this image. Keep it exactly as it is.",
                      negative_prompt="any change, modification, alteration",
                      seed=seed, num_steps=num_steps, guidance=guidance,
                      height=height, width=width)
    m = _img_metrics(img_rs, result, height, width)
    m["label"] = "nullop_drift"
    print(f"[Diag] Null-op drift   SSIM={m.get('ssim','n/a')}  PSNR={m.get('psnr','n/a'):.1f}")
    return m


def diag_degradation_curve(pipe, base_img: Image.Image, n_steps: int = 5,
                            height: int = 1024, width: int = 1024,
                            seed: int = 42, num_steps: int = 8,
                            guidance: float = 1.0,
                            out_dir: Optional[str] = None) -> List[Dict]:
    print(f"\n[Diag] Degradation curve over {n_steps} sequential null-op passes ...")
    ref   = base_img.convert("RGB").resize((width, height), Image.LANCZOS)
    scene = ref.copy()
    rows: List[Dict] = []
    for k in range(n_steps):
        result = run_qwen(pipe, image=scene,
                          prompt="Do not change this image. Keep it exactly as it is.",
                          negative_prompt="any change, modification, alteration",
                          seed=seed + k, num_steps=num_steps, guidance=guidance,
                          height=height, width=width)
        m = _img_metrics(ref, result, height, width)
        m["step"] = k + 1
        rows.append(m)
        print(f"  Step {k+1}/{n_steps}  SSIM={m.get('ssim','n/a')}  PSNR={m.get('psnr','n/a'):.1f}")
        if out_dir:
            result.save(os.path.join(out_dir, f"diag_nullop_step{k+1}.png"))
        scene = result
    return rows


# ── Grid ──────────────────────────────────────────────────────────────────────

def save_grid(images: List[Image.Image], titles: List[str], path: str) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1: axes = [axes]
        for ax, img, t in zip(axes, images, titles):
            ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[Grid] {path}")
    except Exception as e:
        print(f"[Grid] Could not save grid: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--base_img",    help="Pre-rendered base room PNG")
    g.add_argument("--base_prompt", help="Text prompt to generate the base room")
    sm = p.add_mutually_exclusive_group()
    sm.add_argument("--sketch",     default=None,
                    help="Colored sketch overlay (each unique hue = one object)")
    sm.add_argument("--edits_json", default=None,
                    help="edits.json with name/sketch/description per object")
    p.add_argument("--sketch_dir",  default=None,
                   help="Directory with sketch_*.png files (used with --edits_json)")
    p.add_argument("--descs",       nargs="+", default=[])
    p.add_argument("--out_dir",     default="results/baseline")
    p.add_argument("--hf_token",    default=None)
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--model",       default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--lightning",   action="store_true")
    p.add_argument("--lightning_steps", type=int, default=8)
    p.add_argument("--height",      type=int, default=1024)
    p.add_argument("--width",       type=int, default=1024)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_steps",   type=int, default=None)
    p.add_argument("--guidance",    type=float, default=None)
    p.add_argument("--obj_guidance",type=float, default=None)
    p.add_argument("--band_width",  type=int, default=16)
    p.add_argument("--alpha_bg",    type=float, default=1.0)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--diag_steps",  type=int, default=5)
    p.add_argument("--no_eval",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    num_steps = args.num_steps  or (8   if args.lightning else 50)
    guidance  = args.guidance   or (1.0 if args.lightning else 4.0)
    obj_guide = args.obj_guidance or guidance

    print(f"\n{'═'*60}")
    print(f"  baseline.py  —  write-once latent anchoring")
    print(f"  Model     : {'Lightning ' + str(num_steps) + '-step' if args.lightning else str(num_steps) + '-step'}")
    print(f"  Band      : {args.band_width} px   Alpha BG: {args.alpha_bg}")
    print(f"{'═'*60}")

    pipe = load_pipeline(
        model=args.model, hf_token=args.hf_token, cache_dir=args.cache_dir,
        lightning=args.lightning, lightning_steps=args.lightning_steps,
    )

    # ── Base scene ────────────────────────────────────────────────────────────
    if args.base_img:
        base_img = Image.open(args.base_img).convert("RGB").resize(
            (args.width, args.height), Image.LANCZOS)
        print(f"[Base] Loaded: {args.base_img}")
    elif args.base_prompt:
        print(f"[Base] Generating from prompt ...")
        base_img = run_qwen(
            pipe, image=Image.new("RGB", (args.width, args.height), 0),
            prompt=args.base_prompt, seed=args.seed,
            num_steps=num_steps, guidance=guidance,
            height=args.height, width=args.width,
            negative_prompt=(
                "furniture, objects, people, blurry, distorted, "
                "low quality, watermark, cartoon"
            ),
        )
        base_img.save(os.path.join(args.out_dir, "base_scene.png"))
        print(f"[Base] Generated → {args.out_dir}/base_scene.png")
    else:
        raise ValueError("Provide --base_img or --base_prompt.")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    if args.diagnostics:
        print(f"\n{'─'*60}  DIAGNOSTICS")
        d_rt  = diag_vae_roundtrip(pipe, base_img, args.height, args.width)
        d_no  = diag_nullop(pipe, base_img, args.height, args.width,
                            seed=args.seed, num_steps=num_steps, guidance=guidance)
        d_deg = diag_degradation_curve(
            pipe, base_img, n_steps=args.diag_steps,
            height=args.height, width=args.width,
            seed=args.seed, num_steps=num_steps, guidance=guidance,
            out_dir=args.out_dir,
        )
        with open(os.path.join(args.out_dir, "diagnostics.json"), "w") as f:
            json.dump({"roundtrip": d_rt, "nullop": d_no, "degradation": d_deg}, f, indent=2)
        print(f"[Diag] → {args.out_dir}/diagnostics.json")

    # ── Object input ──────────────────────────────────────────────────────────
    if args.edits_json:
        if args.sketch_dir is None:
            raise ValueError("--sketch_dir required with --edits_json.")
        with open(args.edits_json) as f:
            edits = json.load(f)
        sketch_paths = [os.path.join(args.sketch_dir, e["sketch"]) for e in edits]
        descriptions = [e.get("description", e.get("name", "object")) for e in edits]
        for sp in sketch_paths:
            if not os.path.exists(sp):
                raise FileNotFoundError(f"Sketch not found: {sp}")
        slots = slots_from_sketches(sketch_paths, descriptions, args.height, args.width)
        slots = assign_depth_order(slots)
    elif args.sketch:
        raise NotImplementedError(
            "Colored sketch overlay not needed for current inputs. Use --edits_json.")
    else:
        print("[Done] No object input; stopping after diagnostics/base generation.")
        return

    print(f"\n[Slots] {len(slots)} object(s) — farthest → nearest:")
    for s in slots:
        cy, cx = s.centroid
        print(f"  rank {s.depth_rank}  {s.name}: '{s.description}'  "
              f"centroid=({cy:.2f},{cx:.2f})  area={s.mask_area_frac*100:.1f}%")

    # ── Write-once composition ────────────────────────────────────────────────
    composer = SceneComposer(
        pipe=pipe, height=args.height, width=args.width,
        seed=args.seed, num_steps=num_steps,
        guidance=guidance, obj_guidance=obj_guide,
    )
    images = composer.compose(
        base_img=base_img, slots=slots,
        band_width=args.band_width, alpha_bg=args.alpha_bg,
        out_dir=args.out_dir,
    )

    titles = ["base"] + [f"s{i+1} {s.name}" for i, s in enumerate(slots)]
    save_grid(images, titles, os.path.join(args.out_dir, "composition_grid.png"))
    print(f"\n{'═'*60}")
    print(f"  Done. {len(slots)} object(s) composed → {args.out_dir}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
