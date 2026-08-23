"""
CLI runner for SVD-based style personalization on Infinity-2B.

Paper: "A Training-Free Style-Personalization via SVD-Based Feature Decomposition"
       DGIST 2025

Run from the repo root (e:/Cherry_on_top/):

    python Reproduce/SVD/run_svd_style.py \\
        --infinity_path  /path/to/infinity_2b.pth \\
        --vae_path       /path/to/infinity_vae_d32.pth \\
        --t5_path        google/flan-t5-xl \\
        --style_image    inputs/watercolor_ref.jpg \\
        --prompt         "a cat sitting on a windowsill in watercolor style" \\
        --seed 0 --height 1024 --width 1024 \\
        --device cuda --cache_dir ./models --save_images

Or pass a config file:

    python Reproduce/SVD/run_svd_style.py \\
        --config prompts/reproduce_svd_style.json \\
        --device cuda --cache_dir ./models --save_images
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # timm, torch.amp deprecations from Infinity

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── make repo root importable ────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── find the Infinity repo ────────────────────────────────────────────
# Check in order: --infinity_repo CLI arg (handled later), sibling of repo
# root, common Colab/cloud paths, current working directory.
_INF_CANDIDATES = [
    os.path.join(_ROOT, "Infinity"),           # <repo_root>/Infinity/
    "/content/Infinity",                        # Colab default clone location
    os.path.join(os.getcwd(), "Infinity"),      # cwd/Infinity/
    os.path.expanduser("~/Infinity"),           # ~/Infinity/
]

def _add_infinity_to_path(path: str) -> bool:
    if os.path.isdir(os.path.join(path, "tools")) and path not in sys.path:
        sys.path.insert(0, path)
        return True
    return False

for _cand in _INF_CANDIDATES:
    if _add_infinity_to_path(_cand):
        break

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
from PIL import Image

from Reproduce.SVD.svd_style_pipeline import generate_styled, load_model


# ──────────────────────────── helpers ────────────────────────────────


def _parse_sac_steps(value, pfb_step: int = 3, generation_steps: int = 12):
    """Parse sac_steps from CLI string, JSON list, or None → set of ints."""
    if value is None or value == "":
        return None  # pipeline uses default {pfb_step..generation_steps}
    if isinstance(value, (list, set)):
        return set(int(s) for s in value)
    # comma-separated string: "3,4,5,6,7,8,9,10,11,12"
    return set(int(s.strip()) for s in str(value).split(",") if s.strip())


def _save(img: Image.Image, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    img.save(path)
    print(f"  saved → {path}")


def _make_comparison(
    normal_img: Image.Image,
    style_img: Image.Image,
    styled_img: Image.Image,
) -> Image.Image:
    """Three-panel strip: normal generation | style reference | styled output."""
    from PIL import ImageDraw

    W, H = normal_img.size
    label_h = 36

    style_resized = style_img.resize((W, H), Image.LANCZOS)

    canvas = Image.new("RGB", (W * 3, H + label_h), (20, 20, 20))
    canvas.paste(normal_img,    (0,      0))
    canvas.paste(style_resized, (W,      0))
    canvas.paste(styled_img,    (W * 2,  0))

    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(["Normal (no style)", "Style Reference", "Styled (PFB + SAC)"]):
        tw = len(label) * 6
        x = i * W + (W - tw) // 2
        draw.text((x, H + 10), label, fill=(210, 210, 210))

    return canvas


def _run_one(
    infinity, vae, text_tokenizer, text_encoder, scale_schedule,
    *,
    name: str,
    style_image_path: str,
    prompt: str,
    pfb_alpha: float,
    cfg: float,
    tau: float,
    top_k: int,
    top_p: float,
    cfg_insertion_layer: int,
    seed: int,
    height: int,
    width: int,
    device: str,
    out_dir: str,
    save_images: bool,
    use_pfb: bool = True,
    use_sac: bool = True,
    generation_steps: int = 12,
    pfb_step: int = 3,
    sac_steps=None,
    compare: bool = False,
) -> Image.Image:
    mode = ("baseline" if not use_pfb and not use_sac
            else "PFB only" if use_pfb and not use_sac
            else "SAC only" if not use_pfb and use_sac
            else "PFB+SAC")
    print(f"\n[{name}]  ({mode}){' [compare]' if compare else ''}")
    print(f"  style image : {style_image_path}")
    print(f"  prompt      : {prompt}")

    style_img = Image.open(style_image_path).convert("RGB")

    _kwargs = dict(
        style_image=style_img,
        prompt=prompt,
        scale_schedule=scale_schedule,
        pfb_alpha=pfb_alpha,
        cfg=cfg,
        tau=tau,
        top_k=top_k,
        top_p=top_p,
        cfg_insertion_layer=cfg_insertion_layer,
        seed=seed,
        height=height,
        width=width,
        device=device,
        generation_steps=generation_steps,
        pfb_step=pfb_step,
        sac_steps=sac_steps,
    )

    if compare:
        print("  [1/2] normal generation …")
        normal_img = generate_styled(
            infinity, vae, text_tokenizer, text_encoder,
            use_pfb=False, use_sac=False, **_kwargs,
        )
        print("  [2/2] styled (PFB+SAC) …")
        styled_img = generate_styled(
            infinity, vae, text_tokenizer, text_encoder,
            use_pfb=use_pfb, use_sac=use_sac, **_kwargs,
        )
        if save_images:
            run_dir = os.path.join(out_dir, name)
            _save(normal_img, run_dir, "normal")
            _save(styled_img, run_dir, "styled")
            _save(_make_comparison(normal_img, style_img, styled_img),
                  run_dir, "comparison")
        return styled_img
    else:
        img = generate_styled(
            infinity, vae, text_tokenizer, text_encoder,
            use_pfb=use_pfb, use_sac=use_sac, **_kwargs,
        )
        if save_images:
            _save(img, os.path.join(out_dir, name), "generated")
        return img


# ──────────────────────────── CLI ────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVD style personalization (Infinity-2B)")

    # ── model ────────────────────────────────────────────────────────
    p.add_argument("--hf_token", type=str, default="",
                   help="HuggingFace access token for downloading checkpoints. "
                        "Falls back to HF_TOKEN environment variable.")
    p.add_argument("--infinity_repo", type=str, default="",
                   help="Path to the cloned Infinity repo (the directory that "
                        "contains tools/run_infinity.py).  Auto-detected when "
                        "omitted.")
    p.add_argument("--model_size", type=str, default="8b", choices=["8b", "2b"],
                   help="Which Infinity model to use: '8b' (default) or '2b'")
    p.add_argument("--infinity_path", type=str, default="",
                   help="Path to Infinity checkpoint (.pth for 2B, directory for 8B).  "
                        "Auto-detected from cache_dir when omitted.")
    p.add_argument("--vae_path", type=str, default="",
                   help="Path to BSQ-VAE d32 checkpoint (.pth).  "
                        "Falls back to cache_dir/infinity_vae_d32.pth")
    p.add_argument("--t5_path", type=str, default="google/flan-t5-xl",
                   help="HuggingFace model ID or local path for T5 text encoder")
    p.add_argument("--cache_dir", type=str, default="./models",
                   help="Directory for cached model weights")

    # ── single-run arguments (ignored when --config is given) ─────────
    p.add_argument("--style_image", type=str, default="",
                   help="Path to reference style image")
    p.add_argument("--prompt", type=str, default="",
                   help='Text prompt, e.g. "a cat in watercolor style"')
    p.add_argument("--name", type=str, default="output",
                   help="Output sub-folder name")

    # ── method hyper-parameters ───────────────────────────────────────
    p.add_argument("--pfb_alpha", type=float, default=1.0,
                   help="SVD exponential reweighting factor α (paper default 1.0)")
    p.add_argument("--compare", action="store_true",
                   help="Generate normal (no style) AND styled image; save both plus "
                        "a side-by-side comparison strip (comparison.png)")
    p.add_argument("--no_pfb", action="store_true",
                   help="Disable Principal Feature Blending (ablation)")
    p.add_argument("--no_sac", action="store_true",
                   help="Disable Structural Attention Correction (ablation)")
    p.add_argument("--generation_steps", type=int, default=12,
                   help="Number of generation scales to run (paper default: 12)")
    p.add_argument("--pfb_step", type=int, default=3,
                   help="1-indexed scale at which PFB is applied (paper default: 3)")
    p.add_argument("--sac_steps", type=str, default="",
                   help="Comma-separated 1-indexed scales for SAC, e.g. '3,4,5,6,7,8,9,10,11,12'. "
                        "Default: all steps from pfb_step to generation_steps.")
    p.add_argument("--cfg", type=float, default=3.0,
                   help="Classifier-free guidance scale")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Sampling temperature")
    p.add_argument("--top_k", type=int, default=900,
                   help="Top-k for sampling")
    p.add_argument("--top_p", type=float, default=0.97,
                   help="Top-p (nucleus) sampling threshold")
    p.add_argument("--cfg_insertion_layer", type=int, default=-5,
                   help="Layer index at which CFG is inserted (negative = from end)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)

    # ── runtime ──────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="results/svd_style",
                   help="Root directory for output images")
    p.add_argument("--save_images", action="store_true",
                   help="Write generated images to disk")
    p.add_argument("--config", type=str, default="",
                   help="Path to JSON config file (overrides single-run args)")

    return p.parse_args()


def _resolve_path(path: str, fallback_dir: str, fallback_name: str) -> str:
    if path:
        return path
    return os.path.join(fallback_dir, fallback_name)


def main() -> None:
    args = parse_args()

    # ── Apply --infinity_repo (overrides the module-level auto-detection) ──
    if args.infinity_repo:
        if not _add_infinity_to_path(args.infinity_repo):
            raise ValueError(
                f"--infinity_repo '{args.infinity_repo}' does not appear to be the "
                f"Infinity repo root (expected a 'tools/' subdirectory inside it)."
            )

    # For 8B the model_path is a directory; for 2B it's a .pth file.
    if args.model_size == "8b":
        inf_ckpt = args.infinity_path or os.path.join(args.cache_dir, "infinity_8b_weights")
    else:
        inf_ckpt = _resolve_path(args.infinity_path, args.cache_dir, "infinity_2b_reg.pth")
    # VAE filename differs by model size; load_model picks the right one when path is empty
    _vae_default = ("infinity_vae_d56_f8_14_patchify.pth" if args.model_size == "8b"
                    else "infinity_vae_d32.pth")
    vae_ckpt = _resolve_path(args.vae_path, args.cache_dir, _vae_default)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or None

    print(f"[Loading Infinity-{args.model_size.upper()} …]")
    infinity, vae, text_tokenizer, text_encoder, scale_schedule = load_model(
        model_path=inf_ckpt,
        vae_path=vae_ckpt,
        t5_path=args.t5_path,
        device=args.device,
        cache_dir=args.cache_dir,
        hf_token=hf_token,
        model_size=args.model_size,
    )

    # ── build run list ───────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            cfg_json = json.load(f)
        global_cfg = cfg_json.get("global", {})
        runs = cfg_json.get("runs", [])
    else:
        if not args.style_image or not args.prompt:
            raise ValueError("Provide --style_image and --prompt, or --config")
        global_cfg = {}
        runs = [
            {
                "name":         args.name,
                "style_image":  args.style_image,
                "prompt":       args.prompt,
            }
        ]

    def _g(key, default):
        """Resolve: run-level → global → CLI arg → default."""
        return global_cfg.get(key, getattr(args, key, default))

    for run in runs:
        _run_one(
            infinity, vae, text_tokenizer, text_encoder, scale_schedule,
            name=run.get("name", "output"),
            style_image_path=run["style_image"],
            prompt=run["prompt"],
            pfb_alpha=run.get("pfb_alpha",          _g("pfb_alpha",          1.0)),
            cfg=run.get("cfg",                       _g("cfg",                3.0)),
            tau=run.get("tau",                       _g("tau",                1.0)),
            top_k=run.get("top_k",                   _g("top_k",              900)),
            top_p=run.get("top_p",                   _g("top_p",              0.97)),
            cfg_insertion_layer=run.get(
                "cfg_insertion_layer",               _g("cfg_insertion_layer", -5)),
            seed=run.get("seed",                     _g("seed",               0)),
            height=run.get("height",                 _g("height",             1024)),
            width=run.get("width",                   _g("width",              1024)),
            device=args.device,
            out_dir=args.out_dir,
            save_images=args.save_images,
            use_pfb=not run.get("no_pfb", args.no_pfb),
            use_sac=not run.get("no_sac", args.no_sac),
            generation_steps=run.get("generation_steps", _g("generation_steps", 12)),
            pfb_step=run.get("pfb_step",                 _g("pfb_step",         3)),
            sac_steps=_parse_sac_steps(
                run.get("sac_steps", _g("sac_steps", args.sac_steps))
            ),
            compare=run.get("compare", _g("compare", args.compare)),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
