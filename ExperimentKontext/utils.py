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


def enable_multi_context(pipe):
    """
    Monkey-patch pipe.prepare_latents to support image=[img1, img2, ...].
    Each context image gets its own 3D RoPE temporal index (i=1, 2, ..., N)
    as described in the Kontext paper. Works regardless of diffusers version.
    """
    import types
    from diffusers.utils.torch_utils import randn_tensor

    def prepare_latents_multi(
        self, image, batch_size, num_channels_latents,
        height, width, dtype, device, generator=None, latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width  = 2 * (int(width)  // (self.vae_scale_factor * 2))
        shape  = (batch_size, num_channels_latents, height, width)

        image_latents = image_ids = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents_raw = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents_raw = image

            # Expand single image to fill batch (original behaviour)
            if batch_size > image_latents_raw.shape[0]:
                if batch_size % image_latents_raw.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate image batch of {image_latents_raw.shape[0]} to {batch_size}."
                    )
                image_latents_raw = torch.cat(
                    [image_latents_raw] * (batch_size // image_latents_raw.shape[0]), dim=0
                )

            # n_ctx > 1 when multiple context images were passed as a list
            n_ctx = image_latents_raw.shape[0] // batch_size

            all_packed, all_ids = [], []
            for ctx_idx in range(n_ctx):
                lo, hi = ctx_idx * batch_size, (ctx_idx + 1) * batch_size
                ctx_lat = image_latents_raw[lo:hi].contiguous()
                h, w = ctx_lat.shape[2:]
                packed = self._pack_latents(ctx_lat, batch_size, num_channels_latents, h, w)
                ids    = self._prepare_latent_image_ids(batch_size, h // 2, w // 2, device, dtype)
                ids[..., 0] = ctx_idx + 1          # temporal index: 1, 2, 3, …
                all_packed.append(packed)
                all_ids.append(ids)

            image_latents = torch.cat(all_packed, dim=1)   # cat along sequence dim
            image_ids     = torch.cat(all_ids,    dim=0)   # cat along sequence dim

        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents, latent_ids, image_ids

    pipe.prepare_latents = types.MethodType(prepare_latents_multi, pipe)
    print("Multi-context prepare_latents patched on pipe instance.")


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
    inp = proc(images=[img1, img2], return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(pixel_values=inp["pixel_values"])
    if not isinstance(feats, torch.Tensor):
        feats = feats.image_embeds if hasattr(feats, "image_embeds") else feats.pooler_output
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
        def patched(*args, **kwargs):
            # Handle both positional (q,k,v) and keyword (query=,key=,value=) calling conventions
            q = args[0] if len(args) > 0 else kwargs.get('query')
            k = args[1] if len(args) > 1 else kwargs.get('key')
            v = args[2] if len(args) > 2 else kwargs.get('value')
            scale = args[6] if len(args) > 6 else kwargs.get('scale')
            if cap._cur_block[0] >= 0 and q is not None and k is not None:
                step = cap._step[0]
                if cap._capture_steps is None or step in cap._capture_steps:
                    s = (q.shape[-1] ** -0.5) if scale is None else scale
                    with torch.no_grad():
                        w = torch.softmax((q.float() @ k.float().transpose(-2, -1)) * s, dim=-1)
                    cap.captures.setdefault(cap._cur_block[0], []).append((step, w.cpu()))
            return cap._orig_sdpa(*args, **kwargs)

        F.scaled_dot_product_attention = patched

    def step_callback(self, pipe, step, timestep, cb):
        self._step[0] = step
        return cb

    def remove(self):
        F.scaled_dot_product_attention = self._orig_sdpa
        for h in self._hooks:
            h.remove()


class MultiContextAttnCapture:
    """
    Memory-efficient per-context-image attention mass capture.

    Unlike BlockAttentionCapture (which stores the full seq×seq weight matrix),
    this class computes attention mass per context region using chunked log-sum-exp
    — peak GPU memory is O(B·H·n_target·chunk) instead of O(B·H·seq²).

    Usage:
        cap = MultiContextAttnCapture(
            pipe.transformer,
            n_target=4096,           # tokens for generated image
            n_ctx_per=4096,          # tokens per context image
            n_ctx=3,                 # number of context images
            capture_steps={14},
        )
        pipe(..., callback_on_step_end=cap.step_callback, ...)
        cap.remove()
        # cap.stats: {block_idx: [(step, {"ctx0": mass, "ctx1": mass, ...}), ...]}
    """
    def __init__(self, transformer, n_target, n_ctx_per, n_ctx, capture_steps=None,
                 chunk_size=4096):
        self.stats = {}
        self._step       = [0]
        self._cur_block  = [-1]
        self._capture_steps = capture_steps
        self._n_target   = n_target
        self._n_ctx_per  = n_ctx_per
        self._n_ctx      = n_ctx
        self._chunk      = chunk_size
        self._orig_sdpa  = F.scaled_dot_product_attention
        self._hooks      = []
        self._register(transformer)

    @staticmethod
    def _chunked_mass(q_t, k_all, scale, regions, chunk_size):
        """
        Compute attention mass from q_t to each key region without full allocation.
        q_t   : (B, H, n_t, d)  — target queries
        k_all : (B, H, seq, d)  — all keys
        regions: list of (lo, hi) index pairs in the key sequence
        Returns: list of float masses (one per region)
        """
        seq = k_all.shape[2]

        # Pass 1: chunked log-sum-exp over all keys → log partition function
        log_Z = None
        for c0 in range(0, seq, chunk_size):
            c1 = min(c0 + chunk_size, seq)
            sc = (q_t @ k_all[:, :, c0:c1, :].transpose(-2, -1)) * scale
            lse = torch.logsumexp(sc, dim=-1)         # (B, H, n_t)
            log_Z = lse if log_Z is None else torch.logaddexp(log_Z, lse)
            del sc

        # Pass 2: mass per region = Σ_j exp(score(t,j) − log_Z(t))
        masses = []
        for lo, hi in regions:
            sc = (q_t @ k_all[:, :, lo:hi, :].transpose(-2, -1)) * scale
            mass = (sc - log_Z.unsqueeze(-1)).exp().sum(-1).mean().item()
            masses.append(mass)
            del sc
        return masses

    def _register(self, transformer):
        for i, block in enumerate(transformer.transformer_blocks):
            self._hooks.append(block.register_forward_pre_hook(
                lambda m, a, idx=i: self._cur_block.__setitem__(0, idx)
            ))
            self._hooks.append(block.register_forward_hook(
                lambda m, a, o: self._cur_block.__setitem__(0, -1)
            ))

        cap = self
        def patched(*args, **kwargs):
            q = args[0] if args else kwargs.get('query')
            k = args[1] if len(args) > 1 else kwargs.get('key')
            scale = args[6] if len(args) > 6 else kwargs.get('scale')

            blk  = cap._cur_block[0]
            step = cap._step[0]
            should_capture = (
                blk >= 0 and q is not None and k is not None
                and (cap._capture_steps is None or step in cap._capture_steps)
            )
            if should_capture:
                seq           = k.shape[2]
                total_needed  = cap._n_target + cap._n_ctx * cap._n_ctx_per
                if seq >= total_needed:
                    s  = (q.shape[-1] ** -0.5) if scale is None else scale
                    n_t, n_c = cap._n_target, cap._n_ctx_per
                    regions = [
                        (n_t + i * n_c, n_t + (i + 1) * n_c)
                        for i in range(cap._n_ctx)
                    ]
                    with torch.no_grad():
                        masses = cap._chunked_mass(
                            q[:, :, :n_t, :].float(),
                            k.float(), s, regions, cap._chunk,
                        )
                    entry = {f"ctx{i}": m for i, m in enumerate(masses)}
                    cap.stats.setdefault(blk, []).append((step, entry))

            return cap._orig_sdpa(*args, **kwargs)

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
