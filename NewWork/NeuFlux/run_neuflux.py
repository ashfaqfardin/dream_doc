"""
NeuFlux — Training-Free Neural Style Personalization on FLUX.1-dev

Synthesizes:
  Paper 1 (StableFlow)  — latent nudging for real-image editing
  Paper 2 (FluxSpace)   — orthogonal projection to strip content from style features
  Paper 3 (FreeFlux)    — automatic RoPE-based layer classification
  Paper 4 (SVD Style)   — SVD-based PFB + SAC dual-stream mechanism

Run from the repo root (e:/Cherry_on_top/):

  # Text-to-stylized (default)
  python NewWork/NeuFlux/run_neuflux.py \\
      --hf_token YOUR_HF_TOKEN \\
      --style_image inputs/watercolor_ref.png \\
      --prompt "A cat, watercolor painting" \\
      --seed 42 --device cuda --save_images

  # With baseline comparison panel
  python NewWork/NeuFlux/run_neuflux.py \\
      --style_image inputs/oilpainting_ref.jpg \\
      --prompt "A castle, oil painting" \\
      --compare --save_images

  # Real-image mode (Paper 1: latent nudging + inversion)
  python NewWork/NeuFlux/run_neuflux.py \\
      --style_image inputs/watercolor_ref.png \\
      --content_image inputs/cat.jpg \\
      --prompt "A cat, watercolor painting" \\
      --compare --save_images

  # Batch via JSON config
  python NewWork/NeuFlux/run_neuflux.py \\
      --config prompts/neuflux_demo.json \\
      --device cuda --save_images
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image, ImageDraw

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from NewWork.NeuFlux.neuflux_pipeline import generate_styled, load_neuflux_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(img: Image.Image, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    img.save(path)
    print(f"  saved → {path}")


def _parse_layer_list(value: str) -> list:
    """Parse '10,11,12' or '10-18' or '10-18,30,31' into a sorted int list."""
    if not value:
        return []
    result = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _make_comparison(
    normal_img: Image.Image,
    style_img: Image.Image,
    styled_img: Image.Image,
) -> Image.Image:
    """Three-panel strip: Normal | Style Reference | Styled (NeuFlux)."""
    W, H = normal_img.size
    label_h = 36
    style_resized = style_img.resize((W, H), Image.LANCZOS)

    canvas = Image.new("RGB", (W * 3, H + label_h), (20, 20, 20))
    canvas.paste(normal_img,    (0,      0))
    canvas.paste(style_resized, (W,      0))
    canvas.paste(styled_img,    (W * 2,  0))

    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(["Normal (no style)", "Style Reference", "Styled (NeuFlux)"]):
        tw = len(label) * 6
        x  = i * W + (W - tw) // 2
        draw.text((x, H + 10), label, fill=(210, 210, 210))

    return canvas


def _run_one(
    pipe,
    *,
    name: str,
    style_image_path: str,
    prompt: str,
    content_image_path: str,
    pfb_alpha: float,
    pfb_step: int,
    pfb_layers_str: str,
    ortho_scale: float,
    use_sac: bool,
    sac_layers_str: str,
    style_frac: float,
    nudge_lambda: float,
    cfg: float,
    num_steps: int,
    seed: int,
    height: int,
    width: int,
    device: str,
    out_dir: str,
    save_images: bool,
    compare: bool,
) -> Image.Image:
    style_img = Image.open(style_image_path).convert("RGB")
    content_img = Image.open(content_image_path).convert("RGB") if content_image_path else None

    pfb_layers = _parse_layer_list(pfb_layers_str) or None
    sac_layers = _parse_layer_list(sac_layers_str) or None

    sac_step_range = set(range(pfb_step, num_steps)) if sac_layers is not None else None

    mode = "PFB+SAC" if use_sac else "PFB only"
    real = " [real-image]" if content_img else ""
    print(f"\n[{name}]  ({mode}){real}")
    print(f"  style   : {style_image_path}")
    if content_img:
        print(f"  content : {content_image_path}")
    print(f"  prompt  : {prompt}")

    _kwargs = dict(
        style_image=style_img,
        prompt=prompt,
        seed=seed,
        height=height,
        width=width,
        num_steps=num_steps,
        guidance_scale=cfg,
        pfb_alpha=pfb_alpha,
        pfb_step=pfb_step,
        pfb_layers=pfb_layers,
        ortho_scale=ortho_scale,
        use_sac=use_sac,
        sac_layers=sac_layers,
        sac_step_range=sac_step_range,
        style_frac=style_frac,
        content_image=content_img,
        nudge_lambda=nudge_lambda,
    )

    if compare:
        print("  [1/2] normal generation …")
        # Baseline: no PFB, no SAC, no style (same seed, same prompt)
        normal_img = generate_styled(
            pipe,
            **{**_kwargs, "use_sac": False, "pfb_layers": [], "sac_layers": []},
        )
        print("  [2/2] styled (NeuFlux) …")
        styled_img = generate_styled(pipe, **_kwargs)

        if save_images:
            run_dir = os.path.join(out_dir, name)
            _save(normal_img, run_dir, "normal")
            _save(styled_img, run_dir, "styled")
            _save(_make_comparison(normal_img, style_img, styled_img),
                  run_dir, "comparison")
        return styled_img
    else:
        img = generate_styled(pipe, **_kwargs)
        if save_images:
            _save(img, os.path.join(out_dir, name), "generated")
        return img


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuFlux: Training-Free Style Personalization on FLUX")

    # Model
    p.add_argument("--hf_token",    type=str, default="",
                   help="HuggingFace token (or set HF_TOKEN env var)")
    p.add_argument("--model_path",  type=str, default="black-forest-labs/FLUX.1-dev",
                   help="FLUX.1-dev model ID or local path")
    p.add_argument("--cache_dir",   type=str, default="./models")
    p.add_argument("--cpu_offload", action="store_true",
                   help="Enable sequential CPU offload to reduce VRAM")

    # Single-run inputs
    p.add_argument("--style_image",   type=str, default="",
                   help="Path to reference style image")
    p.add_argument("--content_image", type=str, default="",
                   help="Path to content image for real-image mode (Paper 1)")
    p.add_argument("--prompt",        type=str, default="",
                   help='"<content>, <style category>"  e.g. "A cat, watercolor painting"')
    p.add_argument("--name",          type=str, default="output")

    # Method hyper-parameters
    p.add_argument("--pfb_alpha",   type=float, default=1.0,
                   help="SVD spectral reweighting α (Paper 4, default 1.0)")
    p.add_argument("--pfb_step",    type=int,   default=25,
                   help="Denoising step index to apply PFB (default 25 of 50)")
    p.add_argument("--pfb_layers",  type=str,   default="",
                   help="FLUX block indices for PFB, e.g. '9,10,11,12,13' "
                        "(default: auto from layer classification)")
    p.add_argument("--ortho_scale", type=float, default=1.0,
                   help="Orthogonal projection strength 0-1 (Paper 2, default 1.0)")
    p.add_argument("--no_sac",      action="store_true",
                   help="Disable Structural Attention Correction (ablation)")
    p.add_argument("--sac_layers",  type=str,   default="",
                   help="FLUX block indices for SAC, e.g. '0-8' "
                        "(default: auto from layer classification)")
    p.add_argument("--style_frac",  type=float, default=0.5,
                   help="Fraction of blocks classified as style layers (Paper 3, default 0.5)")
    p.add_argument("--nudge_lambda", type=float, default=1.15,
                   help="Latent nudging scalar for real-image inversion (Paper 1, default 1.15)")

    # Generation
    p.add_argument("--cfg",        type=float, default=3.5)
    p.add_argument("--num_steps",  type=int,   default=50)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--height",     type=int,   default=1024)
    p.add_argument("--width",      type=int,   default=1024)
    p.add_argument("--compare",    action="store_true",
                   help="Generate normal + styled + comparison strip")

    # Runtime
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--out_dir",     type=str, default="results/neuflux")
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--config",      type=str, default="",
                   help="Path to JSON batch config file")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or None

    print(f"[Loading FLUX.1-dev …]")
    pipe = load_neuflux_pipeline(
        model_path=args.model_path,
        hf_token=hf_token,
        device=args.device,
        cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
    )

    # ── Build run list ────────────────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            cfg_json = json.load(f)
        global_cfg = cfg_json.get("global", {})
        runs = cfg_json.get("runs", [])
    else:
        if not args.style_image or not args.prompt:
            raise ValueError("Provide --style_image and --prompt, or --config")
        global_cfg = {}
        runs = [{"name": args.name, "style_image": args.style_image,
                 "prompt": args.prompt, "content_image": args.content_image}]

    def _g(key, default):
        return global_cfg.get(key, getattr(args, key, default))

    for run in runs:
        _run_one(
            pipe,
            name=run.get("name", "output"),
            style_image_path=run["style_image"],
            prompt=run["prompt"],
            content_image_path=run.get("content_image", _g("content_image", "")),
            pfb_alpha=run.get("pfb_alpha",    _g("pfb_alpha",    1.0)),
            pfb_step=run.get("pfb_step",      _g("pfb_step",     25)),
            pfb_layers_str=run.get("pfb_layers", _g("pfb_layers", "")),
            ortho_scale=run.get("ortho_scale", _g("ortho_scale", 1.0)),
            use_sac=not run.get("no_sac",      _g("no_sac",      args.no_sac)),
            sac_layers_str=run.get("sac_layers", _g("sac_layers", "")),
            style_frac=run.get("style_frac",  _g("style_frac",  0.5)),
            nudge_lambda=run.get("nudge_lambda", _g("nudge_lambda", 1.15)),
            cfg=run.get("cfg",             _g("cfg",         3.5)),
            num_steps=run.get("num_steps", _g("num_steps",   50)),
            seed=run.get("seed",           _g("seed",        42)),
            height=run.get("height",       _g("height",      1024)),
            width=run.get("width",         _g("width",       1024)),
            device=args.device,
            out_dir=args.out_dir,
            save_images=args.save_images,
            compare=run.get("compare",     _g("compare",     args.compare)),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
