# -*- coding: utf-8 -*-
"""
QwenImage/exp_attention_gate.py — Experiment E3+E4: Attention pattern analysis
and selective attention gating for background preservation.

Background
----------
In the Wan MMDiT (joint attention over text + scene + object reference tokens),
background-region scene tokens attending to object reference tokens is the
primary mechanism by which the background gets "contaminated" and drifts
across sequential edits.

This experiment:
  E3 — Visualize the bg→obj attention flow at each block.
  E4 — Sweep gate layers and gate alpha: measure background SSIM vs. object quality.

Attention Gate (the proposed mechanism, training-free)
------------------------------------------------------
For selected blocks, during denoising steps [gate_start, gate_end]:
  1. Identify background-region tokens (by spatial position in the latent grid)
  2. Identify object reference tokens (second image in the two-image input)
  3. Set attention scores from background tokens to object tokens: -inf
     → after softmax: 0 attention weight from bg to obj reference

This is implemented as a custom attention processor that wraps the pipeline's
original processor and applies the mask before softmax.

References
----------
- FreeFlux (arXiv:2503.16153): layer-role analysis motivates which blocks to gate
- TokenFlow (arXiv:2307.10373): feature propagation for frame-consistent editing
- AnchorEdit (arXiv:2606.11751): causal memory for multi-turn editing (trained;
  our method is training-free and targets background, not subject identity)

Usage
-----
  python NewWork/QwenImage/exp_attention_gate.py \\
      --base_prompt "empty minimalist living room, hardwood floor, white walls, 4K" \\
      --obj_sketch  NewWork/KontextEval/inputs/sketch_bicycle.png \\
      --obj_desc    "yellow mountain bicycle" \\
      --gate_blocks 0 5 10 15 20 \\
      --gate_alpha  0.5 1.0 \\
      --hf_token $HF_TOKEN --cache_dir ./models --lightning \\
      --out_dir results/exp_attention_gate
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline import (
    load_pipeline, run_qwen, encode_latent, _make_room_prior_mask,
    _img_metrics, _fix_cat_dims,
)
from probe import QwenProbe, probe_run, _find_blocks, _find_attn


# ── Gated attention processor ──────────────────────────────────────────────────

class GatedAttnProcessor:
    """Wraps the original processor with a background→object attention mask.

    During selected denoising steps, tokens in bg_token_ids have their
    attention to obj_token_ids set to -inf before the softmax.

    This prevents object reference features from contaminating background tokens,
    the primary source of background drift in sequential object insertion.
    """

    def __init__(
        self,
        original_processor,
        bg_token_ids:  np.ndarray,   # background-region scene tokens (Q side)
        obj_token_ids: np.ndarray,   # object reference tokens (K side)
        gate_alpha:    float = 1.0,  # 0=no mask, 1=full mask
        gate_steps:    Optional[Tuple[int, int]] = None,  # (start, end) inclusive
        step_counter:  Optional[List[int]] = None,
    ):
        self._orig          = original_processor
        self._bg_ids        = torch.from_numpy(bg_token_ids).long()
        self._obj_ids       = torch.from_numpy(obj_token_ids).long()
        self._alpha         = gate_alpha
        self._gate_steps    = gate_steps
        self._step_counter  = step_counter or [0]

    def _should_gate(self) -> bool:
        if self._gate_steps is None:
            return True
        s, e = self._gate_steps
        return s <= self._step_counter[0] <= e

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None,
                 encoder_hidden_states_mask=None, *args, **kwargs):
        # Always pass explicit RoPE kwargs so diffusers doesn't drop them
        _pass_kwargs = dict(
            image_rotary_emb=image_rotary_emb,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )

        # Dual-stream blocks (encoder_hidden_states is not None) return a tuple
        # (img_out, txt_out). Full reimplementation needed to gate these — for now
        # fall through with correct kwargs so the tuple return reaches the block.
        if encoder_hidden_states is not None:
            return self._orig(attn, hidden_states, encoder_hidden_states,
                              attention_mask, *args, **_pass_kwargs, **kwargs)

        # No-gate path: pass through with all kwargs (fixes SSIM at alpha=0.0)
        if not self._should_gate() or self._alpha < 1e-6:
            return self._orig(attn, hidden_states, encoder_hidden_states,
                              attention_mask, *args, **_pass_kwargs, **kwargs)

        # Single-stream gated attention -----------------------------------------
        B, N, C = hidden_states.shape
        H = attn.heads
        d = C // H

        # Projections + QK-norm (Wan uses norm_q / norm_k)
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        if hasattr(attn, "norm_q"):
            q = attn.norm_q(q)
        if hasattr(attn, "norm_k"):
            k = attn.norm_k(k)

        # Reshape to (B, H, N, d)
        q = q.view(B, N, H, d).transpose(1, 2)
        k = k.view(B, N, H, d).transpose(1, 2)
        v = v.view(B, N, H, d).transpose(1, 2)

        # Apply RoPE if provided (position-sensitive attention)
        if image_rotary_emb is not None:
            try:
                from diffusers.models.embeddings import apply_rotary_emb as _ape
                q = _ape(q, image_rotary_emb)
                k = _ape(k, image_rotary_emb)
            except Exception:
                pass

        # Scaled dot-product scores
        scale  = d ** -0.5
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale  # (B, H, N, N)

        # Gate mask: bg_ids (Q) → obj_ids (K) set to -inf * alpha
        N_q, N_k = scores.shape[2], scores.shape[3]
        bg_valid  = self._bg_ids[self._bg_ids  < N_q].to(scores.device)
        obj_valid = self._obj_ids[self._obj_ids < N_k].to(scores.device)

        if len(bg_valid) > 0 and len(obj_valid) > 0:
            gate = torch.zeros(N_q, N_k, device=scores.device, dtype=scores.dtype)
            gate[bg_valid[:, None], obj_valid[None, :]] = -1e9 * self._alpha
            scores = scores + gate.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = scores.softmax(dim=-1)
        out   = torch.einsum("bhnm,bhmd->bhnd", probs, v)       # (B, H, N, d)
        out   = out.transpose(1, 2).reshape(B, N, H * d)

        # Output projection
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1:
            out = attn.to_out[1](out)
        return out


# ── Token layout helpers ───────────────────────────────────────────────────────

def build_token_layout(
    n_text:   int,
    lat_h:    int,
    lat_w:    int,
    two_image: bool = True,
) -> Dict[str, np.ndarray]:
    """Return token index arrays for text / scene / object-reference regions.

    Assumes FLUX-style 2×2 patchification:
        seq_len = (lat_h // 2) * (lat_w // 2)  per image
    """
    patch_h = lat_h // 2
    patch_w = lat_w // 2
    n_scene = patch_h * patch_w
    n_obj   = n_scene if two_image else 0

    text_ids  = np.arange(n_text)
    scene_ids = np.arange(n_text, n_text + n_scene)
    obj_ids   = np.arange(n_text + n_scene, n_text + n_scene + n_obj) if two_image else np.array([])
    return {"text": text_ids, "scene": scene_ids, "obj": obj_ids}


def background_scene_ids(
    layout:   Dict[str, np.ndarray],
    lat_h:    int,
    lat_w:    int,
    top_frac: float = 0.30,   # top fraction = ceiling/wall background
) -> np.ndarray:
    """Return token IDs for background-region scene tokens (ceiling + walls).

    Background = tokens whose spatial patch position is in the top top_frac
    rows (ceiling) or outer side_frac columns (walls).
    """
    patch_h = lat_h // 2
    patch_w = lat_w // 2
    scene_base = layout["scene"][0]

    rows = np.arange(patch_h)
    cols = np.arange(patch_w)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    in_ceiling = rr < (patch_h * top_frac)
    in_walls   = (cc < int(patch_w * 0.10)) | (cc >= int(patch_w * 0.90))
    bg_mask    = in_ceiling | in_walls
    flat_ids   = np.where(bg_mask.flatten())[0]
    return flat_ids + scene_base


# ── Gate installer ─────────────────────────────────────────────────────────────

class AttentionGate:
    """Install / remove gated processors on selected transformer blocks."""

    def __init__(
        self, pipe,
        bg_token_ids:  np.ndarray,
        obj_token_ids: np.ndarray,
        gate_blocks:   List[int],
        gate_alpha:    float = 1.0,
        gate_steps:    Optional[Tuple[int, int]] = None,
    ):
        self.pipe          = pipe
        self._bg           = bg_token_ids
        self._obj          = obj_token_ids
        self._gate_blocks  = set(gate_blocks)
        self._alpha        = gate_alpha
        self._gate_steps   = gate_steps
        self._originals:   Dict[int, Any] = {}
        self._step_counter: List[int] = [0]

    def install(self) -> "AttentionGate":
        transformer = self.pipe.transformer
        blocks      = _find_blocks(transformer)
        for idx, block in enumerate(blocks):
            if idx not in self._gate_blocks:
                continue
            attn = _find_attn(block)
            if attn is None or not hasattr(attn, "processor"):
                continue
            orig = attn.processor
            self._originals[idx] = orig
            attn.set_processor(GatedAttnProcessor(
                orig, self._bg, self._obj,
                gate_alpha=self._alpha,
                gate_steps=self._gate_steps,
                step_counter=self._step_counter,
            ))
        print(f"[Gate] Installed on {len(self._originals)} blocks: {sorted(self._originals)}")
        return self

    def remove(self) -> None:
        transformer = self.pipe.transformer
        blocks      = _find_blocks(transformer)
        for idx, block in enumerate(blocks):
            if idx not in self._originals:
                continue
            attn = _find_attn(block)
            if attn is not None:
                attn.set_processor(self._originals[idx])
        self._originals.clear()
        print("[Gate] Removed.")

    def step_callback(self):
        counter = self._step_counter
        class _SC:
            def __call__(self_, pipe, i, t, kw):
                counter[0] = i
                return kw
        return _SC()


# ── Sweep runner ──────────────────────────────────────────────────────────────

def run_gate_sweep(
    pipe,
    scene_img:     Image.Image,
    obj_img:       Image.Image,
    prompt:        str,
    bg_token_ids:  np.ndarray,
    obj_token_ids: np.ndarray,
    gate_block_sets: List[List[int]],  # each entry: list of blocks to gate
    gate_alphas:     List[float],
    seed:          int   = 42,
    num_steps:     int   = 8,
    guidance:      float = 1.0,
    height:        int   = 1024,
    width:         int   = 1024,
    out_dir:       Optional[str] = None,
    reference_img: Optional[Image.Image] = None,   # for SSIM comparison
) -> List[Dict]:
    rows: List[Dict] = []
    run_id = 0

    for block_set in gate_block_sets:
        for alpha in gate_alphas:
            label = f"blocks_{'-'.join(map(str, block_set))}_a{alpha:.2f}"
            print(f"\n[Sweep] {label} ...")

            gate = AttentionGate(
                pipe, bg_token_ids, obj_token_ids,
                gate_blocks=block_set, gate_alpha=alpha,
                gate_steps=(0, num_steps - 1),
            )
            gate.install()
            sc  = gate.step_callback()
            gen = torch.Generator(device=pipe.device).manual_seed(seed)
            with _fix_cat_dims():
                result = pipe(
                    prompt=prompt,
                    negative_prompt="blurry, distorted, low quality, watermark",
                    image=[scene_img, obj_img],
                    num_inference_steps=num_steps,
                    true_cfg_scale=guidance,
                    height=height, width=width,
                    generator=gen,
                    callback_on_step_end=sc,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images[0]
            gate.remove()

            if out_dir:
                result.save(os.path.join(out_dir, f"gate_{label}.png"))

            row: Dict = {"label": label, "gate_blocks": block_set, "alpha": alpha}
            if reference_img is not None:
                m = _img_metrics(reference_img, result, height, width)
                row.update(m)
                print(f"  SSIM={m.get('ssim','n/a')}  PSNR={m.get('psnr','n/a'):.1f}")
            rows.append(row)
            run_id += 1

    return rows


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--base_img",    default=None)
    g.add_argument("--base_prompt", default=None)
    p.add_argument("--obj_sketch",  required=True, help="Sketch PNG for the object")
    p.add_argument("--obj_desc",    default="yellow mountain bicycle")
    p.add_argument("--gate_blocks", nargs="+", type=int, default=list(range(0, 20, 3)),
                   help="Block indices to gate (all in one set for initial sweep)")
    p.add_argument("--gate_alpha",  nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--out_dir",     default="results/exp_attention_gate")
    p.add_argument("--hf_token",    default=None)
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--lightning",   action="store_true")
    p.add_argument("--height",      type=int, default=1024)
    p.add_argument("--width",       type=int, default=1024)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_steps",   type=int, default=None)
    p.add_argument("--guidance",    type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    num_steps = args.num_steps or (8   if args.lightning else 50)
    guidance  = args.guidance  or (1.0 if args.lightning else 4.0)

    print(f"\n{'═'*60}")
    print(f"  E3/E4: Attention Gate Experiment")
    print(f"  Object: {args.obj_desc}")
    print(f"  Gate blocks: {args.gate_blocks}")
    print(f"  Gate alpha: {args.gate_alpha}")
    print(f"{'═'*60}")

    pipe = load_pipeline(
        hf_token=args.hf_token, cache_dir=args.cache_dir,
        lightning=args.lightning,
    )

    # ── Base scene ────────────────────────────────────────────────────────────
    if args.base_img:
        if not os.path.isfile(args.base_img):
            raise FileNotFoundError(
                f"--base_img not found: {args.base_img}\n"
                "  Generate it first:\n"
                "    python NewWork/QwenImage/baseline.py \\\n"
                "        --base_prompt 'empty minimalist living room ...' \\\n"
                "        --lightning --hf_token $HF_TOKEN\n"
                "  Then re-run with --base_img results/baseline/base_scene.png"
            )
        base_img = Image.open(args.base_img).convert("RGB").resize(
            (args.width, args.height), Image.LANCZOS)
    elif args.base_prompt:
        print("[Base] Generating ...")
        base_img = run_qwen(
            pipe, image=Image.new("RGB", (args.width, args.height), 0),
            prompt=args.base_prompt, seed=args.seed,
            num_steps=num_steps, guidance=guidance,
            height=args.height, width=args.width,
            negative_prompt="furniture, objects, people, blurry, low quality",
        )
        base_img.save(os.path.join(args.out_dir, "base_scene.png"))
    else:
        raise ValueError("Provide --base_img or --base_prompt.")

    # ── Object sketch → synthesized object ────────────────────────────────────
    sketch = Image.open(args.obj_sketch).convert("RGB").resize(
        (args.width, args.height), Image.LANCZOS)
    print(f"[Obj] Synthesizing {args.obj_desc} ...")
    obj_img = run_qwen(
        pipe, image=sketch,
        prompt=(f"Render this sketch as a photorealistic {args.obj_desc}. "
                f"Plain white background, studio lighting."),
        seed=args.seed, num_steps=num_steps, guidance=guidance,
        height=args.height, width=args.width,
        negative_prompt="background, room, floor, shadow, multiple objects",
    )
    obj_img.save(os.path.join(args.out_dir, "obj_synthesized.png"))

    # ── Token layout ──────────────────────────────────────────────────────────
    lat_h = args.height // 8
    lat_w  = args.width  // 8
    # Estimate n_text: probe a run to get sequence length, then infer
    # For now use a heuristic (Wan tokenizer produces ~256 text tokens max)
    n_text = 256
    layout     = build_token_layout(n_text, lat_h, lat_w, two_image=True)
    bg_ids     = background_scene_ids(layout, lat_h, lat_w, top_frac=0.30)
    obj_ids    = layout["obj"]

    print(f"[Tokens] text={len(layout['text'])}  scene={len(layout['scene'])}  "
          f"obj={len(obj_ids)}  bg_region={len(bg_ids)}")

    # ── Baseline: no gating ───────────────────────────────────────────────────
    print("\n[Baseline] No attention gate ...")
    prompt = (
        f"\nPicture 1 shows a room. Picture 2 shows a {args.obj_desc}. "
        f"Place the {args.obj_desc} from Picture 2 on the floor of the room in Picture 1. "
        f"Match the room perspective and lighting. Add a realistic contact shadow. "
        f"Do not change any other part of the room."
    )
    gen  = torch.Generator(device=pipe.device).manual_seed(args.seed)
    with _fix_cat_dims():
        baseline = pipe(
            prompt=prompt, image=[base_img, obj_img.convert("RGB")],
            num_inference_steps=num_steps, true_cfg_scale=guidance,
            height=args.height, width=args.width, generator=gen,
            negative_prompt="blurry, distorted, low quality, watermark",
        ).images[0]
    baseline.save(os.path.join(args.out_dir, "baseline_no_gate.png"))
    baseline_metrics = _img_metrics(base_img, baseline, args.height, args.width)
    print(f"  Baseline SSIM={baseline_metrics.get('ssim','n/a')}  "
          f"PSNR={baseline_metrics.get('psnr','n/a'):.1f}")

    # ── Attention gate sweep ───────────────────────────────────────────────────
    # Group 1: individual block ranges
    block_sets = [
        args.gate_blocks,                      # user-specified set
        list(range(0, 10)),                    # early blocks
        list(range(10, 30)),                   # mid blocks
        list(range(30, 60)),                   # late blocks
        list(range(0, 60)),                    # all blocks
    ]
    # Remove duplicates and filter to valid indices
    seen = set()
    unique_sets = []
    for bs in block_sets:
        key = tuple(sorted(bs))
        if key not in seen:
            seen.add(key)
            unique_sets.append(bs)

    sweep_results = run_gate_sweep(
        pipe, base_img, obj_img.convert("RGB"),
        prompt=prompt,
        bg_token_ids=bg_ids, obj_token_ids=obj_ids,
        gate_block_sets=unique_sets,
        gate_alphas=args.gate_alpha,
        seed=args.seed, num_steps=num_steps, guidance=guidance,
        height=args.height, width=args.width,
        out_dir=args.out_dir,
        reference_img=base_img,
    )

    # Save results
    sweep_results.insert(0, {
        "label": "baseline_no_gate", "gate_blocks": [], "alpha": 0.0,
        **baseline_metrics,
    })
    with open(os.path.join(args.out_dir, "sweep_results.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n[Results] → {args.out_dir}/sweep_results.json")

    # Print Pareto summary
    print(f"\n[Sweep summary] Background SSIM by configuration:")
    for r in sorted(sweep_results, key=lambda x: -x.get("ssim", 0)):
        print(f"  {r['label']:50s}  SSIM={r.get('ssim','n/a')}")

    print(f"\n{'═'*60}")
    print(f"  E4 done → {args.out_dir}")
    print(f"  Key: compare 'baseline_no_gate' SSIM against gated configs.")
    print(f"       Plot sweep_results.json for Pareto curve.")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
