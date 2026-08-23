# -*- coding: utf-8 -*-
"""
phase1_sdedit.py — SDEdit-Style Object Insertion with FLUX.1-Kontext-dev

Problem with phase1_obj_kv.py
------------------------------
  K/V injection at TIER_A is a soft bias — the model can still generate a
  "generic bicycle" rather than the SPECIFIC one from obj_img.  No leading
  paper uses mean-pooled K/V injection; they all operate in latent space.

Solution (from research synthesis)
------------------------------------
  Seed the gen latent from a NOISY version of obj_img at intermediate sigma t*.
  FLUX's flow-matching denoiser then denoises the obj's latent toward the
  scene context provided by the Kontext scene reference.

  FLUX flow-matching noise formula:
      z_t = (1 - sigma) * z_obj + sigma * eps,   eps ~ N(0, I)

  sigma = strength parameter:
      0.3  →  tight identity (minor lighting adaptation, may look pasted)
      0.5  →  balanced  [recommended starting point]
      0.7  →  more natural integration (less rigid pose)

  Literature basis:
    SDEdit (arxiv:2108.01073)          : noisy-init framework; t₀ ≈ 0.35 for compositing
    InsertDiffusion (arxiv:2407.10592) : per-step noisy obj injection for SD
    Blended Latent Diffusion (arxiv:2206.02779): per-step spatial mask blending
    SHINE (arxiv:2509.21278, ICLR 2026): one-step forward noise + IP-Adapter + MSA on FLUX

Accumulated preservation prompt (approach b)
--------------------------------------------
  At each step, the prompt includes EXPLICIT PRESERVATION CLAUSES for all
  previously inserted objects.  Raw concatenation of add-prompts confuses T5
  (Edit-R2, arxiv:2606.05950 — "long-context dilution").  The correct format:

    "<VLM placement prompt>. The yellow mountain bicycle must remain completely
     unchanged. The white ceramic vase must remain completely unchanged."

  ~20 tokens per preserved object; comfortably within T5's 512-token budget.

Pipeline per object
--------------------
  Stage A  : Sketch → LoRA → obj_img           (same as phase1_sketch_vlm)
  Stage VLM: VLM(scene, obj_img) → prompt       (same as phase1_sketch_vlm)
  Stage SD : z_t = noise(z_obj, sigma=strength) (SDEdit init from obj_img)
  Stage B  : Custom Kontext loop, t* → 0:
               gen latent ← z_t  (noisy obj_img)
               ref latent ← z_scene  (Kontext scene reference)
               prompt ← VLM prompt + preservation clauses

Usage
-----
  python NewWork/KontextEval/phase1_sdedit.py \\
      --sketch_dir NewWork/KontextEval/inputs \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir results/phase1_sdedit \\
      --vlm_model Qwen/Qwen2-VL-2B-Instruct \\
      --strength 0.5

  First run with --strength_sweep to pick the best strength value:
  python NewWork/KontextEval/phase1_sdedit.py ... --strength_sweep

Key flags
---------
  --strength         float  SDEdit sigma. 0.5 = recommended. [0, 1]
  --strength_sweep          Run 0.3/0.5/0.7 on FIRST object only, save grid.
  --guidance         float  CFG scale for Stage B. Default 2.5.
  --lora_guidance    float  CFG scale for Stage A LoRA. Default 4.0.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

# ── helpers from phase1_dual_ref (vae encode / pack / ids / decode) ───────────
_dr_path = str(Path(__file__).parent / "phase1_dual_ref.py")
_drspec  = importlib.util.spec_from_file_location("phase1_dual_ref", _dr_path)
_drmod   = importlib.util.module_from_spec(_drspec)
_drspec.loader.exec_module(_drmod)

_vae_encode        = _drmod._vae_encode
_pack              = _drmod._pack
_img_ids           = _drmod._img_ids
_decode_latents    = _drmod._decode_latents
_neutralize_white_bg = _drmod._neutralize_white_bg

# ── sibling pipeline utilities ────────────────────────────────────────────────
_comp_path = str(Path(__file__).parent / "phase1_composite.py")
_cspec = importlib.util.spec_from_file_location("phase1_composite", _comp_path)
_cmod  = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
save_grid    = _cmod.save_grid
run_standard = _cmod.run_standard

_vlm_path = str(Path(__file__).parent / "phase1_sketch_vlm.py")
_vspec = importlib.util.spec_from_file_location("phase1_sketch_vlm", _vlm_path)
_vmod  = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(_vmod)
generate_from_sketch       = _vmod.generate_from_sketch
vlm_generate_kontext_prompt = _vmod.vlm_generate_kontext_prompt

_sk_path = str(Path(__file__).parent / "phase1_sketch.py")
_sspec = importlib.util.spec_from_file_location("phase1_sketch", _sk_path)
_smod  = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(_smod)
load_vlm = _smod.load_vlm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle"},
    {"name": "vase",    "description": "white ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball"},
]
BASE_PROMPT  = "A modern living room with a sofa and a wooden coffee table."
LORA_ID      = "gokaygokay/Sketch-to-Image-Kontext-Dev-LoRA"
LORA_TRIGGER = "Convert this sketch into real life version, follow exact structure."
_SEP = "═" * 60


# ── Accumulated preservation prompt ──────────────────────────────────────────

def _build_preserve_prompt(placement_prompt: str, preserved: List[str]) -> str:
    """
    Append explicit preservation clauses for previously inserted objects.

    DO NOT concatenate raw add-prompts ("Add X. Add Y.") — that causes T5
    long-context dilution (Edit-R2, arxiv:2606.05950).  Instead:
      "<new placement prompt>. The <desc1> must remain completely unchanged.
       The <desc2> must remain completely unchanged."
    """
    if not preserved:
        return placement_prompt
    clauses = " ".join(
        f"The {desc} must remain completely unchanged."
        for desc in preserved
    )
    return f"{placement_prompt} {clauses}"


# ── Core SDEdit + Kontext denoising loop ──────────────────────────────────────

@torch.no_grad()
def run_sdedit_kontext(
    pipe,
    scene:     Image.Image,
    obj_img:   Image.Image,
    prompt:    str,
    strength:  float = 0.5,
    seed:      int   = 42,
    num_steps: int   = 28,
    guidance:  float = 2.5,
    height:    int   = 1024,
    width:     int   = 1024,
) -> Image.Image:
    """
    SDEdit-style Kontext insertion.

    Token layout fed to the transformer (standard Kontext 8192-token sequence):
        [gen_tokens (4096) | ref_scene_tokens (4096)]

    Difference from standard Kontext:
        gen_tokens is initialized from noisy(z_obj, sigma) instead of
        pure Gaussian noise, so the denoiser starts from the object's latent
        and denoises it toward the scene context given by the reference.

    start_step = int((1 - strength) * num_steps)
        → only the final (strength * num_steps) denoising steps run
        → the object's structure/color from sigma_init is preserved
    """
    device = next(pipe.transformer.parameters()).device
    dtype  = next(pipe.transformer.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    # ── 1. Text embeddings ────────────────────────────────────────────────────
    prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    # ── 2. Scene reference latent (Kontext context) ───────────────────────────
    ref_latents = _vae_encode(pipe, scene, height, width, device, dtype)
    ref_packed  = _pack(pipe, ref_latents)                    # (1, 4096, 64)
    ref_ids     = _img_ids(pipe, ref_latents, device, dtype)  # (4096, 3)

    # ── 3. Object latent for SDEdit noisy init ────────────────────────────────
    obj_neutral = _neutralize_white_bg(obj_img)
    obj_latents = _vae_encode(pipe, obj_neutral, height, width, device, dtype)
    obj_packed  = _pack(pipe, obj_latents)                    # (1, 4096, 64)
    gen_ids     = _img_ids(pipe, obj_latents, device, dtype)  # (4096, 3)
    n_gen       = obj_packed.shape[1]                         # 4096

    # ── 4. Scheduler (dynamic mu shift) ──────────────────────────────────────
    try:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    except ImportError:
        from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift

    vae_sf = getattr(pipe, "vae_scale_factor", 8)
    # Hardcode * 2: FLUX packs 2×2 latent patches into one token.
    # pipe.transformer.config.patch_size returns 1 in this diffusers version (reports
    # the VAE step, not the token-packing step) — do NOT use it for image_seq_len.
    image_seq_len = (height // (vae_sf * 2)) * (width // (vae_sf * 2))  # 4096

    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len",  4096),
        pipe.scheduler.config.get("base_shift",          0.5),
        pipe.scheduler.config.get("max_shift",           1.16),
    )
    sigmas_np = np.linspace(1.0, 1.0 / num_steps, num_steps)
    pipe.scheduler.set_timesteps(sigmas=sigmas_np, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps   # (T,) — values in [0, 1000]

    # ── 5. SDEdit noisy init from obj_img ─────────────────────────────────────
    # start_step = first denoising step; skip everything before it.
    # sigma_init = sigma at start_step; used to build the noisy obj latent.
    start_step = min(int((1.0 - strength) * len(timesteps)), len(timesteps) - 1)
    sigma_init = float(sigmas_np[start_step])

    noise   = torch.randn_like(obj_packed, generator=generator)
    # FLUX flow-matching: x_t = (1 - sigma) * x_0 + sigma * eps
    latents = (1.0 - sigma_init) * obj_packed + sigma_init * noise

    print(f"    [SD] strength={strength:.2f}  sigma_init={sigma_init:.3f}  "
          f"start_step={start_step}  "
          f"denoising {len(timesteps) - start_step}/{len(timesteps)} steps")

    # ── 6. Guidance embedding ─────────────────────────────────────────────────
    guidance_tensor = None
    if getattr(pipe.transformer.config, "guidance_embeds", False):
        guidance_tensor = torch.full([1], guidance, dtype=dtype, device=device)

    # ── 7. Denoising loop ─────────────────────────────────────────────────────
    active_timesteps = timesteps[start_step:]
    for i, t in enumerate(active_timesteps):
        # Standard Kontext 8192-token layout: [gen (4096) | ref_scene (4096)]
        model_input = torch.cat([latents, ref_packed], dim=1)   # (1, 8192, 64)
        img_ids     = torch.cat([gen_ids, ref_ids],    dim=0)   # (8192, 3)

        t_batch = t.expand(1)

        noise_pred = pipe.transformer(
            hidden_states         = model_input,
            timestep              = t_batch / 1000,
            guidance              = guidance_tensor,
            pooled_projections    = pooled_embeds,
            encoder_hidden_states = prompt_embeds,
            txt_ids               = text_ids,
            img_ids               = img_ids,
            return_dict           = False,
        )[0]                                                     # (1, 8192, 64)

        noise_pred = noise_pred[:, :n_gen, :]                   # (1, 4096, 64)

        latents = pipe.scheduler.step(
            noise_pred, t, latents, return_dict=False
        )[0]                                                     # (1, 4096, 64)

        if (i + 1) % 7 == 0 or i == 0 or i + 1 == len(active_timesteps):
            print(f"      step {start_step + i + 1:3d}/{len(timesteps)}", flush=True)

    # ── 8. Decode ─────────────────────────────────────────────────────────────
    return _decode_latents(pipe, latents, height, width)


# ── Strength sweep helper ─────────────────────────────────────────────────────

def run_strength_sweep(
    pipe,
    scene:     Image.Image,
    obj_img:   Image.Image,
    prompt:    str,
    name:      str,
    seed:      int,
    num_steps: int,
    guidance:  float,
    height:    int,
    width:     int,
    out_dir:   str,
    strengths: List[float] = (0.3, 0.5, 0.7),
) -> None:
    """
    Run the same edit at multiple strength values, save a comparison grid.
    Use this BEFORE the full chain to pick the best strength for your scene.
    """
    results = [scene]
    titles  = ["scene"]
    for s in strengths:
        print(f"  [sweep] strength={s:.2f} ...")
        r = run_sdedit_kontext(
            pipe=pipe, scene=scene, obj_img=obj_img, prompt=prompt,
            strength=s, seed=seed, num_steps=num_steps,
            guidance=guidance, height=height, width=width,
        )
        p = os.path.join(out_dir, f"sweep_{name}_s{int(s * 100):03d}.png")
        r.save(p)
        results.append(r)
        titles.append(f"s={s:.2f}")

    grid_path = os.path.join(out_dir, f"sweep_{name}_grid.png")
    save_grid(results, titles, grid_path, ncols=len(results))
    print(f"  [sweep] Grid: {grid_path}")
    print(f"          Inspect it and choose --strength before running the full chain.")


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_sdedit_chain(
    pipe,
    base:           Image.Image,
    edits:          List[dict],
    sketch_dir:     str,
    lora_id:        str,
    vlm_pair:       Tuple,
    strength:       float,
    seed:           int,
    num_steps:      int,
    lora_guidance:  float,
    scene_guidance: float,
    height:         int,
    width:          int,
    out_dir:        str,
    device:         str,
) -> List[Image.Image]:
    """
    Incremental insertion: Base → +Bike → +Vase → +Ball.

    Each step:
      Stage A  : sketch → LoRA → obj_img
      Stage VLM: VLM(scene, obj_img) → placement_prompt
      Stage B  : run_sdedit_kontext with accumulated preservation clauses
    """
    vlm_model, vlm_proc = vlm_pair
    results   = [base]
    scene     = base
    preserved: List[str] = []   # descriptions of already-inserted objects

    for i, edit in enumerate(edits):
        name = edit["name"]
        desc = edit["description"]

        sketch_path = os.path.join(sketch_dir, f"{name}.png")
        if not os.path.isfile(sketch_path):
            sketch_path = os.path.join(sketch_dir, f"sketch_{name}.png")
        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found in {sketch_dir!r}.\n"
                f"Expected '{name}.png' or 'sketch_{name}.png'."
            )

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}")
        if preserved:
            print(f"  Preserving: {preserved}")
        print(f"{'─'*60}")

        # Stage A
        print(f"  [A] Generating '{desc}' from sketch ...")
        obj_img = generate_from_sketch(
            pipe=pipe, sketch_path=sketch_path, description=desc,
            seed=seed, num_steps=num_steps, guidance=lora_guidance,
            height=height, width=width, lora_id=lora_id, device=device,
        )
        obj_path = os.path.join(out_dir, f"obj_gen_{name}.png")
        obj_img.save(obj_path)
        print(f"      Saved: {obj_path}")

        # Stage VLM
        print(f"  [VLM] Generating placement prompt ...")
        raw_prompt   = vlm_generate_kontext_prompt(
            vlm_model=vlm_model, vlm_processor=vlm_proc,
            scene_img=scene, obj_img=obj_img, description=desc,
        )
        final_prompt = _build_preserve_prompt(raw_prompt, preserved)

        print(f"\n  {_SEP}")
        print(f"  [VLM → Prompt]  {name}")
        print(f"  {_SEP}")
        for line in final_prompt.splitlines():
            print(f"  {line}")
        print(f"  {_SEP}\n")
        with open(os.path.join(out_dir, f"vlm_prompt_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(final_prompt)

        # Stage B
        print(f"  [B] SDEdit insertion (strength={strength}) ...")
        next_scene = run_sdedit_kontext(
            pipe=pipe, scene=scene, obj_img=obj_img, prompt=final_prompt,
            strength=strength, seed=seed, num_steps=num_steps,
            guidance=scene_guidance, height=height, width=width,
        )
        result_path = os.path.join(out_dir, f"result_step{i+1}_{name}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        scene = next_scene
        results.append(scene)
        preserved.append(desc)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return results


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SDEdit-style object insertion — seed gen latent from noisy obj_img."
    )
    p.add_argument("--sketch_dir",     required=True,
                   help="Folder containing {name}.png or sketch_{name}.png files.")
    p.add_argument("--hf_token",       required=True)
    p.add_argument("--cache_dir",      default="./models")
    p.add_argument("--out_dir",        default="results/phase1_sdedit")
    p.add_argument("--config",         default=None,
                   help="JSON list of {name, description}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",        default=LORA_ID)
    p.add_argument("--lora_guidance",  type=float, default=4.0,
                   help="CFG scale for Stage A LoRA generation.")
    p.add_argument("--guidance",       type=float, default=2.5,
                   help="CFG scale for Stage B Kontext insertion.")
    p.add_argument("--strength",       type=float, default=0.5,
                   help="SDEdit sigma. 0.5=balanced; 0.3=tight identity; 0.7=natural.")
    p.add_argument("--strength_sweep", action="store_true",
                   help="Run strength=[0.3,0.5,0.7] on first object only and exit. "
                        "Inspect sweep grid to pick --strength before a full run.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--num_steps",      type=int,   default=28)
    p.add_argument("--height",         type=int,   default=1024)
    p.add_argument("--width",          type=int,   default=1024)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--vlm_model",      default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--vlm_device",     default="cpu")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)
        print(f"Loaded {len(edits)} edits from {args.config}")

    print(f"\n{_SEP}")
    print(f"  phase1_sdedit  —  SDEdit Object Insertion")
    print(f"{_SEP}")
    print(f"  Objects       : {[e['name'] for e in edits]}")
    print(f"  Sketch dir    : {args.sketch_dir}")
    print(f"  strength      : {args.strength}  (sweep: {args.strength_sweep})")
    print(f"  VLM           : {args.vlm_model}  [{args.vlm_device}]")
    print(f"  LoRA          : {args.lora_id}")
    print(f"  Output        : {args.out_dir}")
    print(f"{_SEP}\n")

    print("Loading VLM ...")
    vlm_pair = load_vlm(args.vlm_model, args.cache_dir, args.vlm_device)

    print("\nLoading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    # Base scene
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base.save(os.path.join(args.out_dir, "step0_base.png"))
    print(f"  Saved: {os.path.join(args.out_dir, 'step0_base.png')}")

    # Optional: strength sweep on first object
    if args.strength_sweep and edits:
        print("\n=== Strength sweep (first object only) ===")
        e0 = edits[0]
        sp = os.path.join(args.sketch_dir, f"{e0['name']}.png")
        if not os.path.isfile(sp):
            sp = os.path.join(args.sketch_dir, f"sketch_{e0['name']}.png")

        print(f"  [A] Generating '{e0['description']}' from sketch ...")
        obj_img0 = generate_from_sketch(
            pipe=pipe, sketch_path=sp, description=e0["description"],
            seed=args.seed, num_steps=args.num_steps, guidance=args.lora_guidance,
            height=args.height, width=args.width, lora_id=args.lora_id, device=args.device,
        )
        obj_img0.save(os.path.join(args.out_dir, f"sweep_obj_{e0['name']}.png"))

        vlm_model, vlm_proc = vlm_pair
        sweep_prompt = vlm_generate_kontext_prompt(
            vlm_model=vlm_model, vlm_processor=vlm_proc,
            scene_img=base, obj_img=obj_img0, description=e0["description"],
        )

        run_strength_sweep(
            pipe=pipe, scene=base, obj_img=obj_img0,
            prompt=sweep_prompt, name=e0["name"],
            seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
            height=args.height, width=args.width, out_dir=args.out_dir,
        )
        print("\n  Sweep complete. Inspect the grid and rerun with --strength <value>.")
        return

    # Full incremental chain
    print("\n=== Incremental insertion ===")
    results = run_sdedit_chain(
        pipe=pipe, base=base, edits=edits,
        sketch_dir=args.sketch_dir, lora_id=args.lora_id,
        vlm_pair=vlm_pair, strength=args.strength,
        seed=args.seed, num_steps=args.num_steps,
        lora_guidance=args.lora_guidance, scene_guidance=args.guidance,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
    )

    final_path = os.path.join(args.out_dir, "KEY_final.png")
    results[-1].save(final_path)
    print(f"\n  Final: {final_path}")

    grid_path = os.path.join(args.out_dir, "KEY_grid.png")
    save_grid(results, ["Base"] + [e["name"] for e in edits], grid_path, ncols=len(results))
    print(f"  Grid : {grid_path}")

    print(f"\n{_SEP}")
    print("  WHAT TO CHECK")
    print(f"{_SEP}")
    print("""
  KEY COMPARISON (run this pipeline vs phase1_sketch_vlm):
    obj_gen_{name}.png          ← the LoRA bicycle / vase / ball
    result_step{N}_{name}.png   ← the inserted version in the scene

    Does the object's COLOR / TEXTURE match obj_gen?
      YES → SDEdit identity preservation is working.
      NO  → try --strength 0.3 (tighter) or lower --guidance.

    Does the lighting / perspective adapt to the room?
      YES → non-rigid integration is working.
      NO  → try --strength 0.7 (more denoising freedom).

  sweep_{name}_grid.png  (if --strength_sweep was used)
    scene | s=0.30 | s=0.50 | s=0.70
    Pick the strength that best balances identity vs. naturalness.

  Background preservation (are previous objects intact)?
    YES → accumulated preservation prompt is working.
    NO  → the reference image (current scene) should carry them;
           if still drifting, increase --guidance slightly.

  KEY_grid.png
    Base | +Bike | +Vase | +Ball
""")


if __name__ == "__main__":
    main()
