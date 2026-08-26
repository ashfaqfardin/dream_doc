"""
E8: Object K/V Attention Amplification (in-context)

All objects are passed as separate context images alongside the base scene:
  image = [base_scene, obj1, obj2, ..., objN]

The multi-context patch assigns each a unique temporal RoPE index (1, 2, 3, ...).
At TIER_A blocks (13-18), a hook amplifies the K (and optionally V) of each
object's context-token slice by k_scale. A pixel-level background mask ensures
only real object tokens (non-grey, non-white) are amplified — background tokens
stay at scale 1.0.

Why this fixes E7/E8-v1:
  - E7 stitch: Kontext treats the composite image as one object (a poster).
  - E8-v1 pre-capture: K/V from a separate pass carry features from a different
    denoising context — the scene Q cannot match them → wrong appearance.
  - E8-v2 (this): K/V are computed in the SAME forward pass at the SAME
    timestep. Amplification just redirects existing attention toward object
    tokens without injecting foreign features.

Conditions (cumulative, all from base scene — no chaining):
  cond_01_bicycle       — base + bicycle context, amplified
  cond_02_+vase         — base + bicycle + vase contexts, amplified
  ...
  cond_07_+backpack     — base + all 7 object contexts, amplified

Metrics: bg_ssim, bg_lpips, dino_{obj}, clip_{obj}

Runtime: ~7 × 1 min = ~7 min  (one scene pass per condition, no capture phase)

NOTE: At k=7 the sequence length is ~36 k image tokens. Needs ≥24 GB VRAM.
      Reduce to --steps 20 or skip high-k conditions if memory is tight.
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


TIER_A = set(range(13, 19))   # deepest double-stream blocks

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
    Bool mask (n_lat_h * n_lat_w,): True = object token, False = background.
    Background = near-grey (128,128,128) OR near-white (>=235 on all channels).
    Falls back to centre-50% crop if fewer than min_frac tokens survive.
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


# ── In-context K/V amplifier ──────────────────────────────────────────────────

class InContextObjKVAmplify:
    """
    During a multi-context Kontext pass (image=[base, obj1, obj2, ...]),
    amplifies K (and optionally V) of each object's context token slice
    at TIER_A blocks.

    Sequence layout inside the SDPA at a double-stream block:
      [ target(n_t) | base_scene(n_c) | obj1(n_c) | obj2(n_c) | ... | text(T) ]
      ^--- image stream ---^                                      ^text^

    base_scene is at slot 0  (indices n_t : n_t+n_c)   — NOT amplified
    obj_i      is at slot i+1 (indices n_t+(i+1)*n_c : n_t+(i+2)*n_c)  — amplified

    Usage:
        amp = InContextObjKVAmplify(pipe.transformer, n_target, n_ctx_per,
                                    obj_masks=[mask1, mask2], k_scale=3.0)
        pipe(image=[base, obj1, obj2], ...)
        amp.remove()
    """

    def __init__(
        self,
        transformer,
        n_target: int,
        n_ctx_per: int,
        obj_masks: list,        # list of BoolTensor (n_ctx_per,) per object
        k_scale: float = 3.0,  # amplify object K tokens
        v_scale: float = 1.0,  # amplify object V tokens (1.0 = unchanged)
    ):
        self._n_target  = n_target
        self._n_ctx_per = n_ctx_per
        self._obj_masks = obj_masks
        self._k_scale   = k_scale
        self._v_scale   = v_scale
        self._cur_blk   = [-1]
        self._orig_sdpa = F.scaled_dot_product_attention
        self._hooks     = []
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

        amp  = self
        orig = self._orig_sdpa

        def patched(*args, **kwargs):
            if len(args) >= 3:
                q, k, v   = args[0], args[1], args[2]
                rest_args = args[3:]
                rest_kw   = kwargs
            else:
                kw = dict(kwargs)
                q = args[0] if len(args) > 0 else kw.pop('query', None)
                k = args[1] if len(args) > 1 else kw.pop('key',   None)
                v = args[2] if len(args) > 2 else kw.pop('value', None)
                rest_args = args[3:]
                rest_kw   = kw

            blk = amp._cur_blk[0]
            if blk in TIER_A and k is not None:
                n_t = amp._n_target
                n_c = amp._n_ctx_per
                seq = k.shape[2]

                for obj_idx, mask in enumerate(amp._obj_masks):
                    # slot 0 = base_scene, slot 1+ = objects
                    s = n_t + (1 + obj_idx) * n_c
                    e = s + n_c
                    if e > seq:
                        continue

                    m = mask.to(k.device)   # (n_c,) bool

                    if amp._k_scale != 1.0:
                        K_sl = k[:, :, s:e, :].clone()
                        sc_k = torch.where(m, k.new_full(m.shape, amp._k_scale),
                                           k.new_ones(m.shape))
                        K_sl = K_sl * sc_k[None, None, :, None]
                        k = torch.cat([k[:, :, :s, :], K_sl, k[:, :, e:, :]], dim=2)

                    if amp._v_scale != 1.0:
                        V_sl = v[:, :, s:e, :].clone()
                        sc_v = torch.where(m, v.new_full(m.shape, amp._v_scale),
                                           v.new_ones(m.shape))
                        V_sl = V_sl * sc_v[None, None, :, None]
                        v = torch.cat([v[:, :, :s, :], V_sl, v[:, :, e:, :]], dim=2)

            return orig(q, k, v, *rest_args, **rest_kw)

        F.scaled_dot_product_attention = patched

    def remove(self):
        F.scaled_dot_product_attention = self._orig_sdpa
        for h in self._hooks:
            h.remove()


# ── Prompt ────────────────────────────────────────────────────────────────────

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
    p.add_argument("--scene",    required=True,  help="Base room image")
    p.add_argument("--obj_dir",  required=True,  help="Dir with obj_<name>.png files")
    p.add_argument("--out_dir",  default="results/e8_obj_attn_concat")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",    type=int,   default=28)
    p.add_argument("--k_scale",  type=float, default=3.0,
                   help="K amplification factor for object tokens at TIER_A")
    p.add_argument("--v_scale",  type=float, default=1.0,
                   help="V amplification factor for object tokens (1.0=off)")
    p.add_argument("--grey_tol", type=int,   default=20,
                   help="Pixel tolerance for grey-background mask")
    p.add_argument("--white_tol",type=int,   default=20,
                   help="Pixel tolerance for white-background mask")
    p.add_argument("--device",   default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = load_pipe(args.model_id, args.device)
    enable_multi_context(pipe)

    vae_sf    = pipe.vae_scale_factor        # 8 for FLUX
    n_lat_h   = 1024 // (vae_sf * 2)        # 64  (= latent_h / 2 for 2×2 packing)
    n_lat_w   = 1024 // (vae_sf * 2)        # 64
    n_target  = n_lat_h * n_lat_w           # 4096
    n_ctx_per = n_lat_h * n_lat_w           # 4096 per context image

    base = Image.open(args.scene).convert("RGB")

    obj_imgs = {}
    for name in OBJ_ORDER:
        path = os.path.join(args.obj_dir, f"obj_{name}.png")
        if os.path.isfile(path):
            obj_imgs[name] = Image.open(path).convert("RGB")
    available = [n for n in OBJ_ORDER if n in obj_imgs]
    print(f"Found {len(available)} objects: {available}")
    print(f"k_scale={args.k_scale}  v_scale={args.v_scale}")

    # Precompute pixel masks once per object
    masks = {}
    for name, img in obj_imgs.items():
        masks[name] = make_pixel_mask(
            img, n_lat_h, n_lat_w, args.grey_tol, args.white_tol,
        )
        n_obj = int(masks[name].sum().item())
        print(f"  {name}: {n_obj}/{n_lat_h*n_lat_w} object tokens "
              f"({100*n_obj/(n_lat_h*n_lat_w):.1f}%)")

    metrics = []

    for k in range(1, len(available) + 1):
        names  = available[:k]
        label  = f"cond_{k:02d}_{'_'.join(names)}"
        prompt = build_prompt(names)
        print(f"\n[{k}/{len(available)}] {label}")
        print(f"  Prompt: {prompt}")

        # context = [base_scene, obj1, ..., objk]
        context_imgs = [base] + [obj_imgs[n] for n in names]
        obj_masks_k  = [masks[n] for n in names]

        seq_img_tokens = n_target + len(context_imgs) * n_ctx_per
        print(f"  Sequence: {seq_img_tokens} image tokens + text")

        amp = InContextObjKVAmplify(
            pipe.transformer, n_target, n_ctx_per, obj_masks_k,
            k_scale=args.k_scale, v_scale=args.v_scale,
        )
        result = pipe(
            image=context_imgs,
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=2.5,
            generator=torch.Generator(args.device).manual_seed(42),
        ).images[0]
        amp.remove()

        result.save(os.path.join(args.out_dir, f"{label}_result.png"))

        m = {
            "k":        k,
            "objects":  names,
            "prompt":   prompt,
            "k_scale":  args.k_scale,
            "v_scale":  args.v_scale,
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
    k_sc   = metrics[0]["k_scale"] if metrics else "?"

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
        ax.set_xlabel("Cumulative objects (last added)")

    plt.suptitle(
        f"E8: In-Context Object K/V Amplification (k_scale={k_sc}) — "
        "background & identity vs object count",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_chart.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
