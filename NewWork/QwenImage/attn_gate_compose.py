# -*- coding: utf-8 -*-
"""
QwenImage/attn_gate_compose.py — Attention-Gated Sequential Composition.

Main contribution: Selective Background Attention Masking (SBAM)
-----------------------------------------------------------------
In Wan/Qwen-Image-Edit, all tokens (text + scene + object reference) attend
to each other in full joint self-attention. When inserting object B into scene A:
  • scene-bg tokens attending to object-ref tokens → background drift ✗
  • scene-target tokens attending to object-ref tokens → appearance transfer ✓
  • object-ref tokens attending to scene tokens → perspective/lighting ✓

SBAM surgically zeroes the first of these three flows at selected blocks:
    A[q∈bg_region, k∈obj_ref] = −∞   (before softmax)

This is training-free, runs in the same 8-step Lightning pass, and can be
combined with the write-once latent anchor for a double-layer defence.

Comparison modes (--mode)
--------------------------
  latent      — write-once latent anchoring only (baseline.py method)
  attn        — SBAM attention gating only (no latent anchor)
  combined    — SBAM + latent anchor (proposed)
  none        — no anchoring (naive sequential baseline)

Architecture notes
------------------
Block gate selection uses the output of exp_layer_drift.py:
  --gate_blocks  auto   → use blocks identified as high-drift in layer_drift.json
  --gate_blocks  0-19   → gate first 20 blocks (early = global layout, high drift)
  --gate_blocks  all    → gate all blocks (strongest, may hurt object appearance)

Usage
-----
  # With edits.json:
  python NewWork/QwenImage/attn_gate_compose.py \\
      --base_prompt "empty minimalist living room, hardwood floor, white walls, 4K" \\
      --edits_json  NewWork/KontextEval/inputs/edits.json \\
      --sketch_dir  NewWork/KontextEval/inputs \\
      --mode combined \\
      --gate_blocks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \\
      --gate_alpha 0.8 \\
      --alpha_bg 0.7 --band_width 16 \\
      --out_dir results/attn_gate_compose \\
      --hf_token $HF_TOKEN --cache_dir ./models --lightning
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline import (
    load_pipeline, run_qwen, encode_latent, decode_latent,
    WriteOnceState, WriteOnceCallback, make_band_weight,
    mask_to_latent, _img_metrics, _make_room_prior_mask,
    slots_from_sketches, assign_depth_order, ObjectSlot,
    save_grid, _tight_crop, _fix_cat_dims,
)
from exp_attention_gate import (
    AttentionGate, GatedAttnProcessor,
    build_token_layout, background_scene_ids,
    _find_blocks, _find_attn,
)


# ── Step-aware composite callback ─────────────────────────────────────────────

class CompositeCallback:
    """Chains a list of denoising-step callbacks and maintains step counter."""

    def __init__(self, callbacks: List, step_counter: List[int]):
        self._cbs      = callbacks
        self._counter  = step_counter

    def __call__(self, pipe, i: int, t, callback_kwargs: dict) -> dict:
        self._counter[0] = i
        for cb in self._cbs:
            callback_kwargs = cb(pipe, i, t, callback_kwargs)
        return callback_kwargs


# ── Gated scene composer ───────────────────────────────────────────────────────

class GatedSceneComposer:
    """Multi-step object insertion with SBAM + optional write-once latent anchoring.

    Parameters
    ----------
    mode : 'latent' | 'attn' | 'combined' | 'none'
    gate_blocks : list of block indices to apply SBAM to
    gate_alpha  : gate strength (0=off, 1=full mask)
    gate_steps  : (start, end) denoising step range; None=all steps
    alpha_bg    : write-once anchor strength (latent / combined mode)
    band_width  : harmonization band in pixels (latent / combined mode)
    n_text      : estimated number of text tokens (used for token layout)
    """

    def __init__(
        self,
        pipe,
        mode:         str         = "combined",
        height:       int         = 1024,
        width:        int         = 1024,
        seed:         int         = 42,
        num_steps:    int         = 8,
        guidance:     float       = 1.0,
        obj_guidance: float       = 1.0,
        gate_blocks:  List[int]   = None,
        gate_alpha:   float       = 0.8,
        gate_steps:   Optional[Tuple[int, int]] = None,
        alpha_bg:     float       = 0.9,
        band_width:   int         = 16,
        n_text:       int         = 256,
    ):
        self.pipe        = pipe
        self.mode        = mode
        self.H, self.W   = height, width
        self.seed        = seed
        self.num_steps   = num_steps
        self.guidance    = guidance
        self.obj_guidance = obj_guidance
        self.gate_blocks  = gate_blocks or list(range(20))
        self.gate_alpha   = gate_alpha
        self.gate_steps   = gate_steps
        self.alpha_bg     = alpha_bg
        self.band_width   = band_width
        self.n_text       = n_text

    def _token_layout(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bg_token_ids, obj_token_ids) for current resolution."""
        lat_h  = self.H // 8
        lat_w  = self.W // 8
        layout = build_token_layout(self.n_text, lat_h, lat_w, two_image=True)
        bg_ids = background_scene_ids(layout, lat_h, lat_w, top_frac=0.30)
        return bg_ids, layout["obj"]

    def _synth_object(self, slot: ObjectSlot) -> Image.Image:
        sketch_in = (slot.sketch_crop or Image.new("RGB", (512, 512), (200, 200, 200)))
        sketch_in = sketch_in.convert("RGB").resize((self.W, self.H), Image.LANCZOS)
        cy, _     = slot.centroid
        persp     = ("lower, larger" if cy > 0.60 else
                     "upper, smaller" if cy < 0.40 else "mid-scene scale")
        prompt = (
            f"Render this sketch as a photorealistic {slot.description}. "
            f"Use the sketch shape exactly. Plain white background, studio lighting. "
            f"Object is {persp}."
        )
        result = run_qwen(
            self.pipe, image=sketch_in, prompt=prompt,
            seed=self.seed, num_steps=self.num_steps, guidance=self.obj_guidance,
            height=self.H, width=self.W,
            negative_prompt="background, room, floor, shadow, multiple objects, blurry",
        )
        return _tight_crop(result)

    def compose(
        self,
        base_img: Image.Image,
        slots:    List[ObjectSlot],
        out_dir:  Optional[str] = None,
    ) -> Tuple[List[Image.Image], List[Dict]]:
        """Run the full composition. Returns (image_list, metrics_list)."""
        H, W = self.H, self.W
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        print(f"\n[Compose] Mode: {self.mode.upper()}")
        if self.mode in ("attn", "combined"):
            print(f"  SBAM: {len(self.gate_blocks)} blocks, alpha={self.gate_alpha}")
        if self.mode in ("latent", "combined"):
            print(f"  WriteOnce: alpha_bg={self.alpha_bg}, band={self.band_width} px")

        # Encode base scene
        z_base       = encode_latent(self.pipe, base_img, H, W)
        lat_h, lat_w = z_base.shape[-2], z_base.shape[-1]
        print(f"  z_base: {tuple(z_base.shape)}")

        state    = WriteOnceState(z_base) if self.mode != "none" else None
        bg_ids, obj_token_ids = self._token_layout()

        scene  = base_img.convert("RGB")
        images = [base_img]
        metrics: List[Dict] = []
        if out_dir:
            base_img.save(os.path.join(out_dir, "base_scene.png"))

        for i, slot in enumerate(slots):
            step_num = i + 1
            print(f"\n[Step {step_num}/{len(slots)}]  {slot.name}  '{slot.description}'")

            # Stage B: sketch → object
            obj_img = self._synth_object(slot)
            if out_dir:
                obj_img.save(os.path.join(out_dir, f"obj_{slot.name}.png"))

            # Overlay mask hint on scene
            arr = np.array(scene)
            overlay = np.array(slot.color, dtype=float)
            arr[slot.mask_np] = (0.65 * arr[slot.mask_np] + 0.35 * overlay).clip(0, 255)
            scene_guided = Image.fromarray(arr.astype(np.uint8))

            # Placement prompt
            cy, cx = slot.centroid
            vert   = "lower" if cy > 0.55 else "upper" if cy < 0.45 else "center"
            horiz  = "left"  if cx < 0.40 else "right" if cx > 0.60 else "center"
            prompt = (
                f"\nPicture 1 shows a room with a highlighted placement region. "
                f"Picture 2 shows a {slot.description}. "
                f"Place the {slot.description} from Picture 2 into the highlighted "
                f"{vert}-{horiz} area in Picture 1. Match perspective and lighting. "
                f"Add a realistic contact shadow. Do not change any other part of the room."
            )

            # Build callbacks
            step_counter = [0]
            cbs = []

            # SBAM (attention gate)
            gate = None
            if self.mode in ("attn", "combined"):
                gate = AttentionGate(
                    self.pipe, bg_ids, obj_token_ids,
                    gate_blocks=self.gate_blocks,
                    gate_alpha=self.gate_alpha,
                    gate_steps=self.gate_steps,
                )
                gate._step_counter = step_counter   # share counter
                gate.install()

            # Write-once latent anchor
            if self.mode in ("latent", "combined") and state is not None:
                band_w    = make_band_weight(slot.mask_np, self.band_width, lat_h, lat_w)
                wo_cb     = WriteOnceCallback(state, band_w, alpha=self.alpha_bg)
                cbs.append(wo_cb)

            composite_cb = CompositeCallback(cbs, step_counter)

            print(f"  [K] Inserting at centroid ({cy:.2f}, {cx:.2f}) ...")
            gen = torch.Generator(device=self.pipe.device).manual_seed(self.seed)
            with _fix_cat_dims():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt="blurry, distorted, low quality, watermark, artifacts",
                    image=[scene_guided, obj_img.convert("RGB")],
                    num_inference_steps=self.num_steps,
                    true_cfg_scale=self.guidance,
                    height=H, width=W,
                    generator=gen,
                    callback_on_step_end=composite_cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images[0]

            if gate is not None:
                gate.remove()

            if out_dir:
                result.save(os.path.join(out_dir, f"result_step{step_num}_{slot.name}.png"))

            # Update write-once anchor
            if state is not None:
                z_result = encode_latent(self.pipe, result, H, W)
                mask_lat = mask_to_latent(slot.mask_np, lat_h, lat_w)
                state.update(z_result, mask_lat)

            # Background metrics (compare to base_img)
            m = _img_metrics(base_img, result, H, W)
            m.update({"step": step_num, "object": slot.name,
                       "mode": self.mode, "gate_alpha": self.gate_alpha,
                       "alpha_bg": self.alpha_bg})
            metrics.append(m)
            print(f"  SSIM={m.get('ssim','n/a')}  PSNR={m.get('psnr','n/a'):.1f}")

            scene = result
            images.append(result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return images, metrics


# ── Multi-mode comparison runner ──────────────────────────────────────────────

def run_comparison(
    pipe,
    base_img:     Image.Image,
    slots:        List[ObjectSlot],
    modes:        List[str]         = None,
    gate_blocks:  List[int]         = None,
    gate_alpha:   float             = 0.8,
    alpha_bg:     float             = 0.9,
    band_width:   int               = 16,
    height:       int               = 1024,
    width:        int               = 1024,
    seed:         int               = 42,
    num_steps:    int               = 8,
    guidance:     float             = 1.0,
    obj_guidance: float             = 1.0,
    out_dir:      Optional[str]     = None,
) -> Dict[str, Dict]:
    """Run all four modes and return results dict for paper tables."""
    modes = modes or ["none", "latent", "attn", "combined"]
    all_results: Dict[str, Dict] = {}

    for mode in modes:
        mode_dir = os.path.join(out_dir, mode) if out_dir else None
        composer = GatedSceneComposer(
            pipe, mode=mode,
            height=height, width=width, seed=seed,
            num_steps=num_steps, guidance=guidance, obj_guidance=obj_guidance,
            gate_blocks=gate_blocks or list(range(20)),
            gate_alpha=gate_alpha,
            alpha_bg=alpha_bg,
            band_width=band_width,
        )
        images, metrics = composer.compose(base_img, slots, out_dir=mode_dir)
        all_results[mode] = {"images": images, "metrics": metrics}
        if out_dir and mode_dir:
            titles = ["base"] + [f"s{i+1} {s.name}" for i, s in enumerate(slots)]
            save_grid(images, titles, os.path.join(mode_dir, "grid.png"))

    # Print comparison table
    print(f"\n{'─'*70}")
    print(f"  {'Mode':12}  {'Step':>5}  {'Object':>15}  {'SSIM':>8}  {'PSNR':>8}")
    print(f"{'─'*70}")
    for mode, res in all_results.items():
        for m in res["metrics"]:
            print(f"  {mode:12}  {m['step']:>5}  {m['object']:>15}  "
                  f"{m.get('ssim','n/a'):>8}  {m.get('psnr','n/a'):>8}")
    print(f"{'─'*70}")

    return all_results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--base_img",    default=None)
    g.add_argument("--base_prompt", default=None)
    sm = p.add_mutually_exclusive_group()
    sm.add_argument("--edits_json", default=None)
    sm.add_argument("--sketch",     default=None)
    p.add_argument("--sketch_dir",  default=None)
    p.add_argument("--descs",       nargs="+", default=[])
    p.add_argument("--mode",        default="combined",
                   choices=["none", "latent", "attn", "combined", "compare"],
                   help="'compare' runs all four modes for ablation")
    p.add_argument("--gate_blocks", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--gate_alpha",  type=float, default=0.8)
    p.add_argument("--gate_start",  type=int,   default=0,
                   help="First denoising step to apply SBAM")
    p.add_argument("--gate_end",    type=int,   default=None,
                   help="Last denoising step to apply SBAM (None=all)")
    p.add_argument("--alpha_bg",    type=float, default=0.9)
    p.add_argument("--band_width",  type=int,   default=16)
    p.add_argument("--out_dir",     default="results/attn_gate_compose")
    p.add_argument("--hf_token",    default=None)
    p.add_argument("--cache_dir",   default="./models")
    p.add_argument("--lightning",   action="store_true")
    p.add_argument("--height",      type=int,   default=1024)
    p.add_argument("--width",       type=int,   default=1024)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_steps",   type=int,   default=None)
    p.add_argument("--guidance",    type=float, default=None)
    p.add_argument("--obj_guidance",type=float, default=None)
    p.add_argument("--n_text",      type=int,   default=256,
                   help="Estimated text token count for token layout computation")
    p.add_argument("--drift_json",  default=None,
                   help="layer_drift.json from exp_layer_drift.py for auto-selecting gate_blocks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    num_steps  = args.num_steps    or (8   if args.lightning else 50)
    guidance   = args.guidance     or (1.0 if args.lightning else 4.0)
    obj_guide  = args.obj_guidance or guidance

    # Auto-select gate blocks from drift analysis if provided
    gate_blocks = args.gate_blocks
    if args.drift_json and os.path.exists(args.drift_json):
        with open(args.drift_json) as f:
            drift = json.load(f)
        # Use step_0_bg curve, pick top-25% blocks by drift
        bg_drift = drift.get("step_0_bg", {})
        if bg_drift:
            vals = sorted(bg_drift.items(), key=lambda x: -float(x[1]))
            top_n = max(1, len(vals) // 4)
            gate_blocks = [int(k) for k, _ in vals[:top_n]]
            print(f"[Auto-gate] Selected {len(gate_blocks)} blocks from drift analysis: "
                  f"{gate_blocks[:10]}{'...' if len(gate_blocks) > 10 else ''}")

    gate_steps = (args.gate_start, args.gate_end or num_steps - 1)

    print(f"\n{'═'*60}")
    print(f"  attn_gate_compose.py — SBAM + Write-Once")
    print(f"  Mode        : {args.mode}")
    print(f"  Gate blocks : {gate_blocks[:5]}{'...' if len(gate_blocks) > 5 else ''} "
          f"({len(gate_blocks)} total)")
    print(f"  Gate alpha  : {args.gate_alpha}   steps: {gate_steps}")
    print(f"  Alpha BG    : {args.alpha_bg}     band: {args.band_width} px")
    print(f"{'═'*60}")

    pipe = load_pipeline(
        hf_token=args.hf_token, cache_dir=args.cache_dir,
        lightning=args.lightning, lightning_steps=8,
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
        print(f"[Base] Loaded: {args.base_img}")
    elif args.base_prompt:
        print("[Base] Generating ...")
        base_img = run_qwen(
            pipe, image=Image.new("RGB", (args.width, args.height), 0),
            prompt=args.base_prompt, seed=args.seed,
            num_steps=num_steps, guidance=guidance,
            height=args.height, width=args.width,
            negative_prompt="furniture, objects, people, blurry, low quality, watermark",
        )
        base_img.save(os.path.join(args.out_dir, "base_scene.png"))
        print(f"[Base] Generated → {args.out_dir}/base_scene.png")
    else:
        raise ValueError("Provide --base_img or --base_prompt.")

    # ── Object slots ──────────────────────────────────────────────────────────
    if args.edits_json:
        if args.sketch_dir is None:
            raise ValueError("--sketch_dir required with --edits_json.")
        with open(args.edits_json) as f:
            edits = json.load(f)
        sketch_paths = [os.path.join(args.sketch_dir, e["sketch"]) for e in edits]
        descriptions = [e.get("description", e.get("name", "object")) for e in edits]
        slots = slots_from_sketches(sketch_paths, descriptions, args.height, args.width)
        slots = assign_depth_order(slots)
    else:
        raise ValueError("Provide --edits_json (--sketch not implemented here).")

    print(f"\n[Slots] {len(slots)} object(s):")
    for s in slots:
        print(f"  rank {s.depth_rank}  {s.name}: '{s.description}'")

    # ── Compose ───────────────────────────────────────────────────────────────
    if args.mode == "compare":
        all_results = run_comparison(
            pipe, base_img, slots,
            modes=["none", "latent", "attn", "combined"],
            gate_blocks=gate_blocks,
            gate_alpha=args.gate_alpha,
            alpha_bg=args.alpha_bg,
            band_width=args.band_width,
            height=args.height, width=args.width,
            seed=args.seed, num_steps=num_steps,
            guidance=guidance, obj_guidance=obj_guide,
            out_dir=args.out_dir,
        )
        all_metrics: Dict = {}
        for mode, res in all_results.items():
            all_metrics[mode] = res["metrics"]
        with open(os.path.join(args.out_dir, "comparison_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"[Results] → {args.out_dir}/comparison_metrics.json")
    else:
        composer = GatedSceneComposer(
            pipe, mode=args.mode,
            height=args.height, width=args.width,
            seed=args.seed, num_steps=num_steps,
            guidance=guidance, obj_guidance=obj_guide,
            gate_blocks=gate_blocks,
            gate_alpha=args.gate_alpha,
            gate_steps=gate_steps,
            alpha_bg=args.alpha_bg,
            band_width=args.band_width,
            n_text=args.n_text,
        )
        images, metrics = composer.compose(base_img, slots, out_dir=args.out_dir)
        titles = ["base"] + [f"s{i+1} {s.name}" for i, s in enumerate(slots)]
        save_grid(images, titles, os.path.join(args.out_dir, "composition_grid.png"))
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Results] → {args.out_dir}/metrics.json")

    print(f"\n{'═'*60}")
    print(f"  Done → {args.out_dir}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
