"""
E8: Object Attention Concatenation

Each object image is processed in a separate Kontext capture pass. At TIER_A
blocks (13-18) the K and V tensors for the context-image portion of the
sequence are extracted. A pixel-level background mask (near-grey / near-white)
filters out non-object tokens. The scene is then denoised with those filtered
K/V tensors concatenated onto the existing K and V at every TIER_A attention
call — the scene's Q selects relevant object features dynamically.

This sidesteps the E7 stitch failure: there is no composite pixel image for
Kontext to treat as a poster. Objects are injected at attention level only.

Conditions (cumulative — all generated from base scene, no chaining):
  cond_01_bicycle       — bicycle K/V injected
  cond_02_+vase         — bicycle + vase K/V injected
  ...
  cond_07_+backpack     — all 7 objects K/V injected

Metrics: bg_ssim, bg_lpips, dino_{obj}, clip_{obj}  (same as E6/E7)

Runtime: ~7 capture passes (~7 min) + ~7 scene passes (~7 min) = ~14 min total
"""
import os, sys, argparse, json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_pipe, enable_multi_context,
    compute_ssim, compute_lpips, compute_dino, compute_clip_i,
)


# Deepest double-stream blocks — highest context-reader attention mass (from E1/E3)
TIER_A = set(range(13, 19))

OBJ_ORDER = ["bicycle", "vase", "ball", "chair", "lamp", "plant", "backpack"]


# ── Background masking ────────────────────────────────────────────────────────

def make_pixel_mask(
    obj_img: Image.Image,
    n_lat_h: int,
    n_lat_w: int,
    grey_tol: int = 20,
    white_tol: int = 20,
    min_frac: float = 0.05,
) -> torch.BoolTensor:
    """
    Boolean mask of shape (n_lat_h * n_lat_w,): True = object token.
    Background = near-grey (128,128,128) or near-white (255,255,255).
    Falls back to centre 50% crop if fewer than min_frac tokens survive.
    """
    arr = np.array(
        obj_img.convert("RGB").resize((n_lat_w, n_lat_h), Image.Resampling.LANCZOS)
    ).astype(int)

    is_grey = (
        (np.abs(arr[:, :, 0] - 128) <= grey_tol) &
        (np.abs(arr[:, :, 1] - 128) <= grey_tol) &
        (np.abs(arr[:, :, 2] - 128) <= grey_tol)
    )
    is_white = arr.min(axis=2) >= (255 - white_tol)
    mask = ~(is_grey | is_white)

    if mask.mean() < min_frac:
        mask[:] = False
        h0, h1 = n_lat_h // 4, 3 * n_lat_h // 4
        w0, w1 = n_lat_w // 4, 3 * n_lat_w // 4
        mask[h0:h1, w0:w1] = True

    return torch.from_numpy(mask.reshape(-1))


# ── Capture ───────────────────────────────────────────────────────────────────

class ObjectKVCapture:
    """
    Hooks into TIER_A SDPA calls during a Kontext forward pass on an object
    image and records K/V of the context-token slice at a target step.

    Usage:
        cap = ObjectKVCapture(pipe.transformer, n_target=4096, n_ctx_per=4096,
                              capture_step=14)
        pipe(image=[obj_img], ..., callback_on_step_end=cap.step_callback, ...)
        cap.remove()
        kv = cap.result   # {block_idx: (K_cpu, V_cpu)}
    """
    def __init__(self, transformer, n_target: int, n_ctx_per: int, capture_step: int):
        self._captured     = {}
        self._step         = [0]
        self._cur_blk      = [-1]
        self._n_target     = n_target
        self._n_ctx_per    = n_ctx_per
        self._capture_step = capture_step
        self._orig_sdpa    = F.scaled_dot_product_attention
        self._hooks        = []
        self._register(transformer)

    def _register(self, transformer):
        for i, block in enumerate(transformer.transformer_blocks):
            if i not in TIER_A:
                continue
            self._hooks.append(block.register_forward_pre_hook(
                lambda m, a, idx=i: self._cur_blk.__setitem__(0, idx)
            ))
            self._hooks.append(block.register_forward_hook(
                lambda m, a, o: self._cur_blk.__setitem__(0, -1)
            ))

        cap  = self
        orig = self._orig_sdpa

        def patched(*args, **kwargs):
            k = args[1] if len(args) > 1 else kwargs.get('key')
            v = args[2] if len(args) > 2 else kwargs.get('value')
            blk = cap._cur_blk[0]

            if (blk in TIER_A and k is not None
                    and cap._step[0] == cap._capture_step
                    and blk not in cap._captured):
                n_t = cap._n_target
                n_c = cap._n_ctx_per
                if k.shape[2] >= n_t + n_c:
                    cap._captured[blk] = (
                        k[:, :, n_t : n_t + n_c, :].detach().cpu(),
                        v[:, :, n_t : n_t + n_c, :].detach().cpu(),
                    )

            return orig(*args, **kwargs)

        F.scaled_dot_product_attention = patched

    def step_callback(self, pipe, step, timestep, cb):
        self._step[0] = step
        return cb

    def remove(self):
        F.scaled_dot_product_attention = self._orig_sdpa
        for h in self._hooks:
            h.remove()

    @property
    def result(self) -> dict:
        return dict(self._captured)


# ── Injection ─────────────────────────────────────────────────────────────────

class ObjKVAttnExtend:
    """
    During scene denoising, concatenates pre-captured object K/V tokens onto
    the end of the K and V sequences at TIER_A blocks.

    Usage:
        entries = [(obj_kv_dict, mask_bool_tensor), ...]
        ext = ObjKVAttnExtend(pipe.transformer, entries)
        pipe(image=[base_scene], ...)
        ext.remove()
    """
    def __init__(self, transformer, entries: list):
        """
        entries: list of (obj_kv, mask)
            obj_kv : {block: (K_cpu, V_cpu)}
            mask   : BoolTensor (n_ctx_per,) — True = object token
        """
        self._entries  = entries
        self._cur_blk  = [-1]
        self._orig_sdpa = F.scaled_dot_product_attention
        self._hooks    = []
        self._register(transformer)

    def _register(self, transformer):
        for i, block in enumerate(transformer.transformer_blocks):
            if i not in TIER_A:
                continue
            self._hooks.append(block.register_forward_pre_hook(
                lambda m, a, idx=i: self._cur_blk.__setitem__(0, idx)
            ))
            self._hooks.append(block.register_forward_hook(
                lambda m, a, o: self._cur_blk.__setitem__(0, -1)
            ))

        ext  = self
        orig = self._orig_sdpa

        def patched(*args, **kwargs):
            # Extract q/k/v regardless of positional vs keyword calling convention
            if len(args) >= 3:
                q, k, v    = args[0], args[1], args[2]
                rest_args  = args[3:]
                rest_kw    = kwargs
            else:
                # Pop from kwargs so we can rebuild cleanly
                kw = dict(kwargs)
                q = args[0] if len(args) > 0 else kw.pop('query', None)
                k = args[1] if len(args) > 1 else kw.pop('key',   None)
                v = args[2] if len(args) > 2 else kw.pop('value', None)
                rest_args = args[3:]
                rest_kw   = kw

            blk = ext._cur_blk[0]
            if blk in TIER_A and k is not None:
                extra_ks, extra_vs = [], []
                dev, dt = k.device, k.dtype
                B = k.shape[0]

                for obj_kv, mask in ext._entries:
                    if blk not in obj_kv:
                        continue
                    K_obj, V_obj = obj_kv[blk]
                    K_obj = K_obj.to(device=dev, dtype=dt)
                    V_obj = V_obj.to(device=dev, dtype=dt)

                    if mask is not None:
                        m = mask.to(device=dev)
                        K_obj = K_obj[:, :, m, :]
                        V_obj = V_obj[:, :, m, :]

                    # Handle guided (2×) vs unguided (1×) batch
                    if K_obj.shape[0] < B:
                        K_obj = K_obj.expand(B, -1, -1, -1)
                        V_obj = V_obj.expand(B, -1, -1, -1)

                    extra_ks.append(K_obj)
                    extra_vs.append(V_obj)

                if extra_ks:
                    k = torch.cat([k] + extra_ks, dim=2)
                    v = torch.cat([v] + extra_vs, dim=2)

            return orig(q, k, v, *rest_args, **rest_kw)

        F.scaled_dot_product_attention = patched

    def remove(self):
        F.scaled_dot_product_attention = self._orig_sdpa
        for h in self._hooks:
            h.remove()


# ── Helpers ───────────────────────────────────────────────────────────────────

def capture_object_kv(
    pipe,
    obj_img: Image.Image,
    n_target: int,
    n_ctx_per: int,
    capture_step: int,
    total_steps: int,
    device: str,
) -> dict:
    """
    Run one Kontext pass with obj_img as context image.
    Returns {block: (K_cpu, V_cpu)} captured at capture_step.
    """
    cap = ObjectKVCapture(pipe.transformer, n_target, n_ctx_per, capture_step)
    pipe(
        image=[obj_img],
        prompt="A product photo on a clean background.",
        num_inference_steps=total_steps,
        guidance_scale=2.5,
        generator=torch.Generator(device).manual_seed(0),
        callback_on_step_end=cap.step_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    cap.remove()
    return cap.result


def build_prompt(names: list) -> str:
    parts = [f"Place the {n} naturally in the room." for n in names]
    return (
        " ".join(parts)
        + " Keep the exact appearance, color, and design of each object"
        + " exactly as it appears in the reference."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",        required=True,  help="Base room image")
    p.add_argument("--obj_dir",      required=True,  help="Dir with obj_<name>.png files")
    p.add_argument("--out_dir",      default="results/e8_obj_attn_concat")
    p.add_argument("--model_id",     default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",        type=int, default=28)
    p.add_argument("--capture_step", type=int, default=14,
                   help="Which denoising step to capture object K/V from")
    p.add_argument("--grey_tol",     type=int, default=20,
                   help="Pixel tolerance for grey-background mask")
    p.add_argument("--white_tol",    type=int, default=20,
                   help="Pixel tolerance for white-background mask")
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = load_pipe(args.model_id, args.device)
    enable_multi_context(pipe)

    # Token grid dimensions for 1024×1024 images
    vae_sf    = pipe.vae_scale_factor          # 8 for FLUX
    n_lat_h   = 1024 // (vae_sf * 2)          # 64
    n_lat_w   = 1024 // (vae_sf * 2)          # 64
    n_target  = n_lat_h * n_lat_w             # 4096
    n_ctx_per = n_lat_h * n_lat_w             # 4096

    base = Image.open(args.scene).convert("RGB")

    # Load objects
    obj_imgs = {}
    for name in OBJ_ORDER:
        path = os.path.join(args.obj_dir, f"obj_{name}.png")
        if os.path.isfile(path):
            obj_imgs[name] = Image.open(path).convert("RGB")
    available = [n for n in OBJ_ORDER if n in obj_imgs]
    print(f"Found {len(available)} objects: {available}")

    # Pixel masks
    masks = {}
    for name, img in obj_imgs.items():
        masks[name] = make_pixel_mask(
            img, n_lat_h, n_lat_w, args.grey_tol, args.white_tol
        )
        n_obj = int(masks[name].sum().item())
        print(f"  {name}: {n_obj}/{n_lat_h*n_lat_w} object tokens "
              f"({100*n_obj/(n_lat_h*n_lat_w):.1f}%)")

    # Capture K/V once per object
    print(f"\nCapturing object K/V at step {args.capture_step}/{args.steps} ...")
    obj_kv = {}
    for name, img in obj_imgs.items():
        print(f"  {name} ...")
        obj_kv[name] = capture_object_kv(
            pipe, img, n_target, n_ctx_per,
            args.capture_step, args.steps, args.device,
        )
        n_blks = len(obj_kv[name])
        print(f"    → captured {n_blks} TIER_A blocks "
              f"(expected {len(TIER_A)})")

    metrics = []

    for k in range(1, len(available) + 1):
        names = available[:k]
        label = f"cond_{k:02d}_{'_'.join(names)}"
        print(f"\n[{k}/{len(available)}] {label}")

        prompt = build_prompt(names)
        print(f"  Prompt: {prompt}")

        entries = [(obj_kv[n], masks[n]) for n in names]

        ext = ObjKVAttnExtend(pipe.transformer, entries)
        result = pipe(
            image=[base],
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=2.5,
            generator=torch.Generator(args.device).manual_seed(42),
        ).images[0]
        ext.remove()

        result.save(os.path.join(args.out_dir, f"{label}_result.png"))

        m = {
            "k":        k,
            "objects":  names,
            "prompt":   prompt,
            "bg_ssim":  compute_ssim(base, result),
            "bg_lpips": compute_lpips(base, result, args.device),
        }
        for name in names:
            m[f"dino_{name}"] = compute_dino(obj_imgs[name], result, args.device)
            m[f"clip_{name}"] = compute_clip_i(obj_imgs[name], result, args.device)

        metrics.append(m)
        print(f"  bg_ssim={m['bg_ssim']:.3f}  bg_lpips={m['bg_lpips']:.3f}")
        for name in names:
            print(f"  {name}: DINO={m[f'dino_{name}']:.3f}  "
                  f"CLIP={m[f'clip_{name}']:.3f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _plot(metrics, available, args.out_dir)
    print(f"\nDone. Results in {args.out_dir}")


def _plot(metrics: list, available: list, out_dir: str):
    ks     = [m["k"] for m in metrics]
    labels = [m["objects"][-1] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(ks, [m["bg_ssim"]  for m in metrics],
                    marker='o', color='steelblue', linewidth=2)
    axes[0, 0].set_title("Background SSIM vs original\n(higher = stable)")
    axes[0, 0].set_ylim(0, 1); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ks, [m["bg_lpips"] for m in metrics],
                    marker='o', color='coral', linewidth=2)
    axes[0, 1].set_title("Background LPIPS vs original\n(lower = stable)")
    axes[0, 1].grid(True, alpha=0.3)

    for name in available:
        dinos  = [m[f"dino_{name}"] for m in metrics if f"dino_{name}" in m]
        ks_obj = [m["k"]            for m in metrics if f"dino_{name}" in m]
        axes[1, 0].plot(ks_obj, dinos, marker='o', linewidth=2, label=name)
    axes[1, 0].set_title("DINOv2 per object vs. reference\n(higher = identity preserved)")
    axes[1, 0].set_ylim(-0.2, 1); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)

    for name in available:
        clips  = [m[f"clip_{name}"] for m in metrics if f"clip_{name}" in m]
        ks_obj = [m["k"]            for m in metrics if f"clip_{name}" in m]
        axes[1, 1].plot(ks_obj, clips, marker='o', linewidth=2, label=name)
    axes[1, 1].set_title("CLIP-I per object vs. reference\n(higher = identity preserved)")
    axes[1, 1].set_ylim(0, 1); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7)

    for ax in axes.flat:
        ax.set_xticks(ks)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_xlabel("Cumulative objects injected (last added)")

    plt.suptitle(
        "E8: Object K/V Attention Concatenation — background & identity vs object count",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_chart.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
