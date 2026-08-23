# -*- coding: utf-8 -*-
"""
phase1_qwen_multistep_config.py
Config-driven multi-scene, multi-step object insertion using Qwen-Image-Edit-2509
+ integrated 2025-standard evaluation metrics.

Pipeline (all via Qwen — no FLUX/ControlNet at runtime):
  Step 0  : scene_prompt  + black image → Qwen                → base_scene.png
  Step A  : object sketch               + render prompt → Qwen → obj_{name}.png
  Step 1+ : [scene, obj]  + add_prompt  → Qwen + SKA          → result_step{N}_{name}.png

Evaluation metrics (run per scene after all insertions complete):
  BG-LPIPS   : background perceptual distance vs. base scene  (lower = better)
  BG-SSIM    : background structural similarity vs. base scene (higher = better)
  DINOv2-BG  : background patch cosine similarity (DINOv2)    (higher = better)
  VQA-Score  : P(yes | VQA, add_prompt)                       (higher = better)
  MLLM-Judge : Qwen2-VL-2B binary "is the object present?"   (1 = yes)

Usage
------
  python NewWork/KontextEval/phase1_qwen_multistep_config.py \\
      --config    NewWork/KontextEval/config.json \\
      --hf_token  $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir   results/phase1_multistep \\
      --lightning --lightning_steps 8

  # Standard 50-step (no Lightning)
  python NewWork/KontextEval/phase1_qwen_multistep_config.py \\
      --config    NewWork/KontextEval/config.json \\
      --hf_token  $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir   results/phase1_standard

  # Ablation: disable SKA to get baseline
  python NewWork/KontextEval/phase1_qwen_multistep_config.py \\
      --config    NewWork/KontextEval/config.json \\
      --out_dir   results/phase1_no_ska \\
      --no_ska \\
      --hf_token  $HF_TOKEN --cache_dir ./models --lightning
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

# ── Import SKA machinery from phase1_ref_qwen ─────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase1_ref_qwen import (   # noqa: E402
    SKA_SIGMA_GATE,
    SKAStore,
    _tight_crop,
    _encode_latent,
    _decode_latent,
    _latent_composite,
    _make_room_prior_mask,
    _DSACallback,
    insert_prompt,
    run_qwen,
    run_qwen_ska,
    load_pipeline as load_qwen_pipeline,
)

# ── Constants ──────────────────────────────────────────────────────────────────

BG_THRESH  = 30      # pixel-diff threshold for background mask detection
BG_MIN_FRAC = 0.10   # if <10% of pixels qualify as BG, fall back to center crop


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QWEN CALLS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scene(
    pipe,
    scene_prompt: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
) -> Image.Image:
    """Text-only scene generation: Qwen with a black placeholder image."""
    return run_qwen(
        pipe,
        image=Image.new("RGB", (width, height), 0),
        prompt=scene_prompt,
        seed=seed, num_steps=num_steps, guidance=guidance,
        height=height, width=width,
        negative_prompt=(
            "furniture, objects, people, blurry, distorted, "
            "low quality, watermark, cartoon, painting"
        ),
    )


def generate_object_from_sketch(
    pipe,
    sketch_path: str,
    description: str,
    seed: int, num_steps: int, guidance: float,
    height: int, width: int,
) -> Image.Image:
    """Sketch → Qwen → photorealistic isolated object (tight-cropped)."""
    sketch = Image.open(sketch_path).convert("RGB").resize((width, height), Image.LANCZOS)
    prompt = (
        f"Render the object shown in this sketch as a photorealistic {description}. "
        f"Use the sketch to define the exact shape and proportions. "
        f"Plain white background, studio lighting, sharp details, no cast shadows."
    )
    obj = run_qwen(
        pipe,
        image=sketch,
        prompt=prompt,
        seed=seed, num_steps=num_steps, guidance=guidance,
        height=height, width=width,
        negative_prompt=(
            "blurry, distorted, low quality, background, room, floor, "
            "shadow, multiple objects, text, watermark"
        ),
    )
    return _tight_crop(obj)   # crop to non-white bounding box → 512×512


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INSERTION CHAIN (Qwen + SKA)
# ═══════════════════════════════════════════════════════════════════════════════

def run_scene(
    pipe,
    scene_cfg:      dict,
    sketch_dir:     str,
    out_dir:        str,
    seed:           int,
    num_steps:      int,
    guidance:       float,
    obj_guidance:   float,
    height:         int,
    width:          int,
    use_ska:        bool,
    ska_gate:       float,
    inject_mode:    str   = "v",     # "kv" | "k" | "v"
    latent_anchor:  bool  = True,    # T1: post-hoc latent compositing (legacy)
    lat_tau:        float = 0.3,     # T1 latent diff threshold
    lat_temp:       float = 0.05,    # T1 sigmoid edge sharpness
    dsa:            bool  = True,    # T3: denoising-step anchoring (preferred)
    dsa_alpha:      float = 0.8,     # DSA anchor strength in background (0=free, 1=locked)
    dsa_top:        float = 0.25,    # ceiling anchor fraction
    dsa_side:       float = 0.12,    # wall-edge anchor fraction
) -> tuple[list[Image.Image], list[dict]]:
    """Process one scene.  Returns (images, step_meta) for evaluation."""
    scene_id  = scene_cfg["id"]
    objects   = scene_cfg["objects"]
    scene_out = os.path.join(out_dir, scene_id)
    os.makedirs(scene_out, exist_ok=True)

    _sep = "─" * 60
    print(f"\n{'═' * 60}")
    print(f"  Scene: {scene_id}  —  {scene_cfg.get('description', '')}  ({len(objects)} objects)")
    print(f"{'═' * 60}")

    # ── Step 0: generate base scene ───────────────────────────────────────────
    print(f"\n[Step 0] Generating base scene ...")
    print(f"  Prompt: {scene_cfg['scene_prompt'][:120]}...")
    base = generate_scene(
        pipe, scene_cfg["scene_prompt"],
        seed=seed, num_steps=num_steps, guidance=guidance,
        height=height, width=width,
    )
    base.save(os.path.join(scene_out, "base_scene.png"))
    print(f"  Saved: base_scene.png")

    images    = [base]
    step_meta = []           # [{name, description, add_prompt}] per insertion step
    scene     = base
    ska_store = SKAStore(inject_mode=inject_mode) if use_ska else None

    # ── T1: Latent Background Anchoring state ─────────────────────────────────
    z_prev:     "torch.Tensor | None" = None
    accum_mask: "torch.Tensor | None" = None
    if latent_anchor or dsa:
        print(f"  [LBA/DSA] Encoding base scene latent ...")
        z_prev = _encode_latent(pipe, base.convert("RGB"), height, width)
        print(f"            z shape: {tuple(z_prev.shape)}  dtype: {z_prev.dtype}")

    # ── T3: DSA — build room prior mask once (shared across all steps) ─────────
    dsa_obj_mask: "torch.Tensor | None" = None
    if dsa and z_prev is not None:
        lat_h = z_prev.shape[2]
        lat_w = z_prev.shape[3]
        dsa_obj_mask = _make_room_prior_mask(lat_h, lat_w, dsa_top, dsa_side)
        print(f"  [DSA]  Room prior mask: {lat_h}×{lat_w}  "
              f"free={float(dsa_obj_mask.mean())*100:.1f}%  α={dsa_alpha}")

    for i, obj_cfg in enumerate(objects):
        name = obj_cfg["name"]
        desc = obj_cfg["description"]

        print(f"\n{_sep}")
        print(f"  Object {i + 1}/{len(objects)}  —  {name}")
        print(_sep)

        # ── Step A: sketch → Qwen → object image ──────────────────────────────
        sketch_file = obj_cfg.get("sketch", f"sketch_{name}.png")
        sketch_path = os.path.join(sketch_dir, sketch_file)
        if not os.path.isfile(sketch_path):
            sketch_path = os.path.join(sketch_dir, f"{name}.png")
        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found for '{name}': tried "
                f"'{sketch_file}' and '{name}.png' in {sketch_dir!r}"
            )

        print(f"  [A] Rendering '{desc}' from sketch {os.path.basename(sketch_path)} ...")
        obj_img = generate_object_from_sketch(
            pipe, sketch_path, desc,
            seed=seed, num_steps=num_steps, guidance=obj_guidance,
            height=height, width=width,
        )
        obj_img.save(os.path.join(scene_out, f"obj_{name}.png"))
        print(f"      Saved: obj_{name}.png")

        # ── Step K: insert into scene ──────────────────────────────────────────
        add_prompt = obj_cfg.get("add_prompt") or insert_prompt(desc)
        ska_phase: str | None = None
        if use_ska and ska_store is not None:
            ska_phase = "inject" if ska_store.has_kv else "capture"

        print(f"  [K] Inserting → {width}×{height}"
              + (f"  SKA={ska_phase}" if ska_phase else ""))
        print(f"      Prompt: {add_prompt.strip()[:110]}...")

        with open(os.path.join(scene_out, f"prompt_step{i+1}_{name}.txt"), "w") as f:
            f.write(add_prompt)

        # Build per-step DSA callback (fresh noise draw each insertion)
        step_dsa = None
        if dsa and z_prev is not None and dsa_obj_mask is not None:
            step_dsa = _DSACallback(z_prev, dsa_obj_mask, alpha=dsa_alpha)

        updated_raw = run_qwen_ska(
            pipe=pipe,
            image=[scene.convert("RGB"), obj_img.convert("RGB")],
            prompt=add_prompt,
            seed=seed, num_steps=num_steps, guidance=guidance,
            height=height, width=width,
            negative_prompt="blurry, distorted, low quality, watermark, text, artifacts",
            ska_store=ska_store, ska_phase=ska_phase, ska_gate=ska_gate,
            dsa_callback=step_dsa,
        )

        # ── T1: post-hoc latent composite (legacy, off by default) ───────────
        if latent_anchor and z_prev is not None:
            z_raw = _encode_latent(pipe, updated_raw.convert("RGB"), height, width)
            z_comp, accum_mask = _latent_composite(
                z_raw, z_prev, accum_mask, tau=lat_tau, temp=lat_temp,
            )
            updated = _decode_latent(pipe, z_comp)
            obj_frac = float(accum_mask.mean()) * 100
            print(f"      [LBA] object region: {obj_frac:.1f}% of latent grid")
            # Mask visualisation
            m_arr = (accum_mask[0, 0].cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(m_arr).save(
                os.path.join(scene_out, f"lat_mask_step{i+1}_{name}.png")
            )
        else:
            updated = updated_raw

        # ── Update z_prev for next step (DSA anchors to progressive state) ────
        if (latent_anchor or dsa) and z_prev is not None:
            z_prev = _encode_latent(pipe, updated.convert("RGB"), height, width)

        out_path = os.path.join(scene_out, f"result_step{i+1}_{name}.png")
        updated.save(out_path)
        print(f"      Saved: result_step{i+1}_{name}.png")

        images.append(updated)
        step_meta.append({"name": name, "description": desc, "add_prompt": add_prompt})
        scene = updated

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return images, step_meta


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION METRICS  (2025 standard)
# ═══════════════════════════════════════════════════════════════════════════════

def _bg_mask(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that did NOT change between two steps.
    Falls back to the center 50% crop if fewer than BG_MIN_FRAC pixels qualify.
    """
    diff   = np.abs(img_a.astype(int) - img_b.astype(int)).max(axis=2)
    mask   = diff < BG_THRESH
    if mask.mean() < BG_MIN_FRAC:
        h, w   = mask.shape
        mask[:] = False
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    return mask


def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def compute_bg_lpips(img_a: Image.Image, img_b: Image.Image, mask: np.ndarray, net) -> float:
    """LPIPS on background pixels only (Zhang et al. CVPR 2018)."""
    import torch
    a = _to_np(img_a).copy(); b = _to_np(img_b).copy()
    a[~mask] = 128; b[~mask] = 128
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1).float() / 127.5 - 1
    with torch.no_grad():
        return float(net(t(a).unsqueeze(0), t(b).unsqueeze(0)).item())


def compute_bg_ssim(img_a: Image.Image, img_b: Image.Image, mask: np.ndarray) -> float:
    """SSIM on background region (EditBoard AAAI 2025; DLEBench 2025)."""
    from skimage.metrics import structural_similarity
    a = _to_np(img_a); b = _to_np(img_b)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return 0.0
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    win = min(r1 - r0, c1 - c0, 7)
    win = win if win % 2 == 1 else max(win - 1, 1)
    return float(structural_similarity(
        a[r0:r1, c0:c1], b[r0:r1, c0:c1],
        channel_axis=2, win_size=win, data_range=255,
    ))


def compute_dino_bg(img_a: Image.Image, img_b: Image.Image, mask: np.ndarray, dino) -> float:
    """DINOv2 patch cosine similarity on background pixels (SpotEdit arXiv:2508.18159, 2025)."""
    import torch, torch.nn.functional as F
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        f_a = dino(preprocess(img_a).unsqueeze(0))
        f_b = dino(preprocess(img_b).unsqueeze(0))
    return float(F.cosine_similarity(f_a, f_b, dim=-1).mean().item())


def compute_vqa_score(result_img: Image.Image, add_prompt: str, vqa_model) -> float:
    """VQA-Score: P(yes | model, prompt)  (Lin et al. ECCV 2024, arXiv:2404.01291)."""
    try:
        import t2v_metrics
        score = vqa_model(images=[result_img], texts=[add_prompt.strip()])
        return float(score[0][0])
    except Exception:
        return float("nan")


def compute_mllm_judge(result_img: Image.Image, description: str, processor, model, device) -> int:
    """Binary 0/1 — Qwen2-VL-2B answers 'Is there a [description] in this image?'
    (ImgEdit-Judge NeurIPS 2025; LMM4Edit ACM MM 2025).
    """
    import torch
    question = (
        f"Is there a {description} visible in this image? "
        f"Answer only 'yes' or 'no'."
    )
    try:
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": result_img},
            {"type": "text",  "text": question},
        ]}]
        text  = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=8)
        answer = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True)[0].strip().lower()
        return 1 if answer.startswith("yes") else 0
    except Exception:
        return -1   # -1 = eval failed


def load_eval_models(hf_token: str | None, cache_dir: str, device: str = "cpu") -> dict:
    """Lazily load evaluation models with graceful fallback on missing deps."""
    models: dict = {}

    # BG-LPIPS
    try:
        import lpips
        models["lpips"] = lpips.LPIPS(net="alex").eval()
    except Exception as e:
        print(f"  [EVAL] LPIPS unavailable: {e}. Install: pip install lpips")

    # DINOv2
    try:
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                              trust_repo=True).eval().to(device)
        models["dino"] = dino
    except Exception as e:
        print(f"  [EVAL] DINOv2 unavailable: {e}")

    # VQA-Score
    try:
        import t2v_metrics
        models["vqa"] = t2v_metrics.VQAScore(model="clip-flant5-xxl",
                                             cache_dir=cache_dir)
    except Exception as e:
        print(f"  [EVAL] VQA-Score unavailable: {e}. Install: pip install t2v-metrics")

    # MLLM-Judge  (Qwen2-VL-2B)
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        mllm = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            token=hf_token,
            cache_dir=cache_dir,
        ).eval().to(device)
        proc = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", token=hf_token, cache_dir=cache_dir,
        )
        models["mllm"] = mllm
        models["mllm_proc"] = proc
        models["mllm_device"] = device
    except Exception as e:
        print(f"  [EVAL] MLLM-Judge unavailable: {e}. Install: pip install qwen-vl-utils")

    return models


def evaluate_scene(
    scene_id:  str,
    images:    list[Image.Image],
    step_meta: list[dict],
    eval_models: dict,
) -> list[dict]:
    """Compute metrics for each insertion step of a scene.

    images[0] = base scene
    images[1] = after step 1
    ...
    step_meta[i] corresponds to images[i+1]
    """
    rows = []
    base = images[0]

    for i, meta in enumerate(step_meta):
        prev = images[i]
        curr = images[i + 1]
        mask = _bg_mask(_to_np(prev), _to_np(curr))

        row = {
            "scene_id":    scene_id,
            "step":        i + 1,
            "object":      meta["name"],
            "description": meta["description"],
        }

        # BG-LPIPS
        if "lpips" in eval_models:
            try:
                row["bg_lpips"] = round(
                    compute_bg_lpips(base, curr, mask, eval_models["lpips"]), 4
                )
            except Exception as e:
                row["bg_lpips"] = f"err:{e}"
        else:
            row["bg_lpips"] = "n/a"

        # BG-SSIM
        try:
            from skimage.metrics import structural_similarity  # noqa: F401
            row["bg_ssim"] = round(compute_bg_ssim(base, curr, mask), 4)
        except Exception as e:
            row["bg_ssim"] = f"err:{e}"

        # DINOv2-BG
        if "dino" in eval_models:
            try:
                row["dino_bg"] = round(
                    compute_dino_bg(base, curr, mask, eval_models["dino"]), 4
                )
            except Exception as e:
                row["dino_bg"] = f"err:{e}"
        else:
            row["dino_bg"] = "n/a"

        # VQA-Score
        if "vqa" in eval_models:
            row["vqa_score"] = round(
                compute_vqa_score(curr, meta["add_prompt"], eval_models["vqa"]), 4
            )
        else:
            row["vqa_score"] = "n/a"

        # MLLM-Judge
        if "mllm" in eval_models:
            row["mllm_judge"] = compute_mllm_judge(
                curr, meta["description"],
                eval_models["mllm_proc"],
                eval_models["mllm"],
                eval_models["mllm_device"],
            )
        else:
            row["mllm_judge"] = "n/a"

        rows.append(row)
        print(f"  [EVAL] step {i+1} {meta['name']:12s} | "
              f"BG-LPIPS={row['bg_lpips']}  BG-SSIM={row['bg_ssim']}  "
              f"DINO={row['dino_bg']}  VQA={row['vqa_score']}  MLLM={row['mllm_judge']}")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRID + REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_grid(images: list, titles: list, path: str, ncols: int | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n     = len(images)
    ncols = ncols or min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)
    for ax, img, t in zip(axes_flat, images, titles):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ARGS + MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Config-driven multi-scene Qwen insertion + eval metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       required=True,      help="Path to config.json")
    p.add_argument("--out_dir",      default="results/phase1_multistep")
    p.add_argument("--hf_token",     default=None)
    p.add_argument("--cache_dir",    default="./models")
    # Image
    p.add_argument("--height",       type=int,   default=1024)
    p.add_argument("--width",        type=int,   default=1024)
    p.add_argument("--seed",         type=int,   default=42)
    # Qwen
    p.add_argument("--num_steps",    type=int,   default=None,
                   help="Denoising steps (default 8 with --lightning, 50 otherwise)")
    p.add_argument("--guidance",     type=float, default=None,
                   help="CFG scale (default 1.0 Lightning, 4.0 standard)")
    p.add_argument("--obj_guidance", type=float, default=None,
                   help="Object render CFG (default = --guidance)")
    # SKA
    p.add_argument("--ska_gate",     type=float, default=None,
                   help="SKA sigma gate (default 0.3; try 0.5 for 8-step Lightning)")
    p.add_argument("--no_ska",       action="store_true", help="Disable SKA (ablation baseline)")
    # Lightning
    p.add_argument("--lightning",       action="store_true")
    p.add_argument("--lightning_steps", type=int, default=8, choices=[4, 8])
    # Eval
    p.add_argument("--no_eval",      action="store_true",
                   help="Skip evaluation metrics (faster)")
    p.add_argument("--eval_device",  default="cpu",
                   help="Device for eval models (cpu or cuda)")
    p.add_argument("--out_csv",      default=None,
                   help="Path for evaluation CSV (default: out_dir/metrics.csv)")
    p.add_argument("--inject_mode",  default="v", choices=["kv", "k", "v"],
                   help="Exp-1 toggle: inject K only ('k'), V only ('v'), or both ('kv'). Default: 'v'")
    # ── T1: Latent Background Anchoring ───────────────────────────────────────
    p.add_argument("--no_latent_anchor", action="store_true",
                   help="Disable T1 post-hoc latent compositing")
    p.add_argument("--lat_tau",  type=float, default=0.3,
                   help="T1 latent diff threshold (default 0.3)")
    p.add_argument("--lat_temp", type=float, default=0.05,
                   help="T1 sigmoid edge sharpness (default 0.05)")
    # ── T3: Denoising-Step Anchoring ──────────────────────────────────────────
    p.add_argument("--no_dsa",    action="store_true",
                   help="Disable T3 denoising-step anchoring (ablation)")
    p.add_argument("--dsa_alpha", type=float, default=0.8,
                   help="DSA anchor strength in background region (0=free, 1=locked; default 0.8)")
    p.add_argument("--dsa_top",   type=float, default=0.25,
                   help="DSA ceiling anchor fraction from top (default 0.25)")
    p.add_argument("--dsa_side",  type=float, default=0.12,
                   help="DSA wall-edge anchor fraction from sides (default 0.12)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config     = load_config(args.config)
    sketch_dir = config.get("sketch_dir", "NewWork/KontextEval/inputs")
    scenes     = config["scenes"]

    num_steps    = args.num_steps    if args.num_steps    is not None else (8   if args.lightning else 50)
    guidance     = args.guidance     if args.guidance     is not None else (1.0 if args.lightning else 4.0)
    obj_guidance = args.obj_guidance if args.obj_guidance is not None else guidance
    ska_gate     = args.ska_gate     if args.ska_gate     is not None else SKA_SIGMA_GATE
    csv_path     = args.out_csv      or os.path.join(args.out_dir, "metrics.csv")

    print(f"\n{'═' * 60}")
    print(f"  phase1_qwen_multistep_config")
    print(f"  Scenes     : {len(scenes)}")
    print(f"  Qwen model : {'Lightning ' + str(num_steps) + '-step' if args.lightning else 'Standard ' + str(num_steps) + '-step'}")
    print(f"  SKA        : {'OFF (ablation)' if args.no_ska else f'ON  gate={ska_gate}  inject={args.inject_mode}'}")
    print(f"  LBA (T1)   : {'OFF' if args.no_latent_anchor else f'ON  tau={args.lat_tau}  T={args.lat_temp}'}")
    print(f"  DSA (T3)   : {'OFF (ablation)' if args.no_dsa else f'ON  α={args.dsa_alpha}  top={args.dsa_top}  side={args.dsa_side}'}")
    print(f"  Eval       : {'OFF' if args.no_eval else 'ON'}")
    print(f"{'═' * 60}")

    # ── Load Qwen (once, shared across all scenes) ─────────────────────────────
    pipe = load_qwen_pipeline(
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
        lightning=args.lightning,
        lightning_steps=args.lightning_steps,
    )

    # ── Load eval models (after Qwen loads, on CPU to avoid VRAM clash) ────────
    eval_models: dict = {}
    if not args.no_eval:
        print(f"\n[EVAL] Loading evaluation models on {args.eval_device} ...")
        eval_models = load_eval_models(
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            device=args.eval_device,
        )
        available = [k for k in eval_models if not k.startswith("mllm_")]
        print(f"  [EVAL] Available: {available}")

    # ── Process each scene ─────────────────────────────────────────────────────
    all_metric_rows: list[dict] = []

    for scene_cfg in scenes:
        images, step_meta = run_scene(
            pipe=pipe,
            scene_cfg=scene_cfg,
            sketch_dir=sketch_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            num_steps=num_steps,
            guidance=guidance,
            obj_guidance=obj_guidance,
            height=args.height,
            width=args.width,
            use_ska=not args.no_ska,
            ska_gate=ska_gate,
            inject_mode=args.inject_mode,
            latent_anchor=not args.no_latent_anchor,
            lat_tau=args.lat_tau,
            lat_temp=args.lat_temp,
            dsa=not args.no_dsa,
            dsa_alpha=args.dsa_alpha,
            dsa_top=args.dsa_top,
            dsa_side=args.dsa_side,
        )

        # Save grid for this scene
        titles    = ["base"] + [f"s{i+1} {m['name']}" for i, m in enumerate(step_meta)]
        grid_path = os.path.join(args.out_dir, scene_cfg["id"], "grid_all_steps.png")
        save_grid(images, titles, grid_path, ncols=min(4, len(images)))
        print(f"  Grid: {grid_path}")

        # Evaluate
        if not args.no_eval and eval_models:
            print(f"\n[EVAL] Evaluating {scene_cfg['id']} ...")
            rows = evaluate_scene(
                scene_id=scene_cfg["id"],
                images=images,
                step_meta=step_meta,
                eval_models=eval_models,
            )
            all_metric_rows.extend(rows)

            # Per-scene CSV
            scene_csv = os.path.join(args.out_dir, scene_cfg["id"], "metrics.csv")
            save_csv(rows, scene_csv)

    # ── Save global metrics CSV ────────────────────────────────────────────────
    if all_metric_rows:
        save_csv(all_metric_rows, csv_path)

    print(f"\n{'═' * 60}")
    print(f"  Done.  {len(scenes)} scene(s) processed.")
    if all_metric_rows:
        print(f"  Metrics CSV: {csv_path}")
    print(f"  Results in : {args.out_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
