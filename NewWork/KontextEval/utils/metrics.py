"""
Evaluation metrics for Phase 9:
  - LPIPS  (perceptual similarity)
  - PSNR   (pixel-level reconstruction)
  - SSIM   (structural similarity)
  - DINOv2 cosine similarity  (semantic identity)
  - CLIP direction similarity  (edit alignment)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------

def pil_to_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Resize, normalise to [0,1], return (1,3,H,W) float32."""
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return t


def _to_same_device(*tensors, device: str = "cuda"):
    return [t.to(device) for t in tensors]


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(img_a: Image.Image, img_b: Image.Image) -> float:
    """Peak signal-to-noise ratio (dB). Higher = more similar."""
    a = np.array(img_a.convert("RGB")).astype(np.float64)
    b = np.array(img_b.convert("RGB")).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


# ---------------------------------------------------------------------------
# SSIM  (simple numpy implementation, no extra deps)
# ---------------------------------------------------------------------------

def ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    """Structural similarity index. Returns value in [−1, 1], higher = better."""
    a = np.array(img_a.convert("RGB")).astype(np.float64) / 255.0
    b = np.array(img_b.convert("RGB")).astype(np.float64) / 255.0

    C1, C2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    mu1, mu2 = a.mean(), b.mean()
    sig1, sig2 = a.std(), b.std()
    sig12 = float(np.mean((a - mu1) * (b - mu2)))
    num   = (2 * mu1 * mu2 + C1) * (2 * sig12 + C2)
    denom = (mu1 ** 2 + mu2 ** 2 + C1) * (sig1 ** 2 + sig2 ** 2 + C2)
    return float(num / denom)


# ---------------------------------------------------------------------------
# LPIPS  (requires `pip install lpips`)
# ---------------------------------------------------------------------------

_lpips_model = None

def _get_lpips(device: str = "cuda"):
    global _lpips_model
    if _lpips_model is None:
        import lpips as lpips_lib
        _lpips_model = lpips_lib.LPIPS(net="vgg").to(device)
        _lpips_model.eval()
    return _lpips_model


def lpips_score(img_a: Image.Image, img_b: Image.Image,
                device: str = "cuda") -> float:
    """LPIPS perceptual distance. Lower = more similar."""
    model = _get_lpips(device)
    size = 256
    a = pil_to_tensor(img_a, size).to(device) * 2 - 1  # [−1,1]
    b = pil_to_tensor(img_b, size).to(device) * 2 - 1
    with torch.no_grad():
        dist = model(a, b)
    return float(dist.item())


# ---------------------------------------------------------------------------
# DINOv2 cosine similarity  (requires transformers)
# ---------------------------------------------------------------------------

_dino_model  = None
_dino_proc   = None

def _get_dino(device: str = "cuda"):
    global _dino_model, _dino_proc
    if _dino_model is None:
        from transformers import AutoImageProcessor, AutoModel
        _dino_proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        _dino_model.eval()
    return _dino_model, _dino_proc


def dino_similarity(img_a: Image.Image, img_b: Image.Image,
                    device: str = "cuda") -> float:
    """
    DINOv2 [CLS] cosine similarity. Range [−1, 1], higher = more semantically similar.
    """
    model, proc = _get_dino(device)
    inputs_a = proc(images=img_a, return_tensors="pt").to(device)
    inputs_b = proc(images=img_b, return_tensors="pt").to(device)
    with torch.no_grad():
        feat_a = model(**inputs_a).last_hidden_state[:, 0]   # CLS token
        feat_b = model(**inputs_b).last_hidden_state[:, 0]
    cos = F.cosine_similarity(feat_a, feat_b, dim=-1)
    return float(cos.item())


# ---------------------------------------------------------------------------
# CLIP direction similarity  (edit alignment)
# ---------------------------------------------------------------------------

_clip_model  = None
_clip_preproc = None
_clip_tokenizer = None

def _get_clip(device: str = "cuda"):
    global _clip_model, _clip_preproc, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        _clip_model, _, _clip_preproc = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        _clip_model = _clip_model.to(device).eval()
    return _clip_model, _clip_preproc, _clip_tokenizer


def clip_direction_similarity(
    img_before: Image.Image, img_after: Image.Image,
    prompt_before: str, prompt_after: str,
    device: str = "cuda",
) -> float:
    """
    CLIP direction similarity: measures whether the image edit direction
    aligns with the text edit direction.

    High score (→1) means the image edit is consistent with the prompt change.
    """
    model, preproc, tokenizer = _get_clip(device)

    def enc_img(img):
        t = preproc(img).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode_image(t)

    def enc_txt(text):
        t = tokenizer([text]).to(device)
        with torch.no_grad():
            return model.encode_text(t)

    i_before = enc_img(img_before)
    i_after  = enc_img(img_after)
    t_before = enc_txt(prompt_before)
    t_after  = enc_txt(prompt_after)

    delta_i = (i_after - i_before)
    delta_t = (t_after - t_before)

    cos = F.cosine_similarity(
        delta_i.float(), delta_t.float(), dim=-1
    )
    return float(cos.item())


# ---------------------------------------------------------------------------
# Full evaluation table for one edit
# ---------------------------------------------------------------------------

def evaluate_edit(
    img_original: Image.Image,
    img_baseline: Image.Image,
    img_method: Image.Image,
    prompt_before: str = "",
    prompt_after: str = "",
    device: str = "cuda",
) -> dict:
    """
    Compute the full suite of metrics comparing method to baseline.

    img_original : the pre-edit image  (used for content-preservation metrics)
    img_baseline : FLUX.1-Kontext without injection  (reference output)
    img_method   : your method's output
    """
    results = {}

    # Content preservation vs original
    results["psnr_vs_original"]  = psnr(img_original, img_method)
    results["ssim_vs_original"]  = ssim(img_original, img_method)
    results["lpips_vs_original"] = lpips_score(img_original, img_method, device)
    results["dino_vs_original"]  = dino_similarity(img_original, img_method, device)

    # Delta vs baseline
    results["psnr_vs_baseline"]  = psnr(img_baseline, img_method)
    results["ssim_vs_baseline"]  = ssim(img_baseline, img_method)
    results["lpips_vs_baseline"] = lpips_score(img_baseline, img_method, device)
    results["dino_vs_baseline"]  = dino_similarity(img_baseline, img_method, device)

    # Edit alignment
    if prompt_before and prompt_after:
        results["clip_direction"] = clip_direction_similarity(
            img_original, img_method, prompt_before, prompt_after, device
        )

    return results
