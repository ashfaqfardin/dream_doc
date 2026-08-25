import os, sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

_local = os.path.join(os.path.dirname(__file__), '..', 'KontextPipeline', 'diffusers', 'src')
if os.path.isdir(_local):
    sys.path.insert(0, _local)

from diffusers import FluxKontextPipeline


def load_pipe(model_id="black-forest-labs/FLUX.1-Kontext-dev", device="cuda"):
    pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to(device)
    return pipe


# ── Image helpers ─────────────────────────────────────────────────────────────

def _to_np(img: Image.Image, size=512):
    return np.array(img.convert("RGB").resize((size, size))).astype(np.float32)

def _to_t(img: Image.Image, device, size=512):
    arr = _to_np(img, size) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    from skimage.metrics import structural_similarity as sk_ssim
    a = _to_np(img1).astype(np.uint8)
    b = _to_np(img2).astype(np.uint8)
    return float(sk_ssim(a, b, channel_axis=2, data_range=255))


def compute_lpips(img1: Image.Image, img2: Image.Image, device="cuda") -> float:
    import lpips
    fn = lpips.LPIPS(net='alex').to(device)
    t1 = _to_t(img1, device) * 2 - 1
    t2 = _to_t(img2, device) * 2 - 1
    with torch.no_grad():
        return float(fn(t1, t2).item())


_dino_cache = {}
def compute_dino(img1: Image.Image, img2: Image.Image, device="cuda") -> float:
    from transformers import AutoImageProcessor, AutoModel
    if "dino" not in _dino_cache:
        _dino_cache["dino"] = (
            AutoImageProcessor.from_pretrained("facebook/dinov2-base"),
            AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval(),
        )
    proc, model = _dino_cache["dino"]
    def feat(img):
        inp = proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            return model(**inp).last_hidden_state[:, 0]
    return float(F.cosine_similarity(feat(img1), feat(img2)).item())


_clip_cache = {}
def compute_clip_i(img1: Image.Image, img2: Image.Image, device="cuda") -> float:
    from transformers import CLIPProcessor, CLIPModel
    if "clip" not in _clip_cache:
        _clip_cache["clip"] = (
            CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval(),
        )
    proc, model = _clip_cache["clip"]
    inp = proc(images=[img1, img2], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return float((feats[0] @ feats[1]).item())


def attn_entropy(weights: torch.Tensor) -> float:
    """Mean entropy of attention rows. weights: (B, H, Sq, Sk)"""
    p = weights.float().mean(dim=(0, 1))                    # (Sq, Sk)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return float((-(p * (p + 1e-8).log()).sum(dim=-1)).mean().item())


# ── Attention capture ─────────────────────────────────────────────────────────

class BlockAttentionCapture:
    """
    Captures Q@K^T softmax attention weights from transformer_blocks.
    Hooks track which block is active; SDPA is monkey-patched to record weights.

    Usage:
        cap = BlockAttentionCapture(pipe.transformer, capture_steps={14})
        pipe(..., callback_on_step_end=cap.step_callback, ...)
        cap.remove()
        # cap.captures: {block_idx: [(step, tensor), ...]}
    """
    def __init__(self, transformer, capture_steps=None):
        self.captures = {}
        self._step = [0]
        self._cur_block = [-1]
        self._capture_steps = capture_steps   # None = all steps
        self._orig_sdpa = F.scaled_dot_product_attention
        self._hooks = []
        self._register(transformer)

    def _register(self, transformer):
        for i, block in enumerate(transformer.transformer_blocks):
            self._hooks.append(block.register_forward_pre_hook(
                lambda m, a, idx=i: self._cur_block.__setitem__(0, idx)
            ))
            self._hooks.append(block.register_forward_hook(
                lambda m, a, o: self._cur_block.__setitem__(0, -1)
            ))

        cap = self
        def patched(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            if cap._cur_block[0] >= 0:
                step = cap._step[0]
                if cap._capture_steps is None or step in cap._capture_steps:
                    s = (q.shape[-1] ** -0.5) if scale is None else scale
                    with torch.no_grad():
                        w = torch.softmax((q.float() @ k.float().transpose(-2, -1)) * s, dim=-1)
                    cap.captures.setdefault(cap._cur_block[0], []).append((step, w.cpu()))
            return cap._orig_sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale)

        F.scaled_dot_product_attention = patched

    def step_callback(self, pipe, step, timestep, cb):
        self._step[0] = step
        return cb

    def remove(self):
        F.scaled_dot_product_attention = self._orig_sdpa
        for h in self._hooks:
            h.remove()


# ── Plot helpers ──────────────────────────────────────────────────────────────

def save_grid(images, labels, path, cols=4):
    W, H = images[0].size
    rows  = (len(images) + cols - 1) // cols
    grid  = Image.new("RGB", (W * cols, (H + 25) * rows), (240, 240, 240))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        grid.paste(img, (c * W, r * (H + 25) + 25))
    grid.save(path)


def plot_curve(x, y, xlabel, ylabel, title, path, color="steelblue"):
    plt.figure(figsize=(9, 4))
    plt.plot(x, y, marker='o', linewidth=2, color=color)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
