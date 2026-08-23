# -*- coding: utf-8 -*-
"""
phase1_sketch_vlm.py  —  VLM-Guided Sketch-to-Scene Insertion

Pipeline (three objects, incremental)
--------------------------------------
  STEP 0   Generate base room scene (Kontext, text only)

  STEP N   For each object (bicycle → vase → ball):

    Stage A  Sketch → photorealistic object image
               FLUX.1-Kontext-dev + Sketch-to-Image LoRA
               → obj_gen_{name}.png

    VLM      (current_scene + obj_img) → Kontext insertion prompt
               Qwen2-VL-2B-Instruct (default, runs on CPU)
               Prompt printed to terminal and saved to vlm_prompt_{name}.txt

    Stage B  Kontext edits the scene with the VLM-generated prompt
               run_standard(pipe, current_scene, vlm_prompt)
               → result_step{N}_{name}.png

  Output grid: KEY_grid.png  (base | +bike | +vase | +ball)

Usage
-----
  python NewWork/KontextEval/phase1_sketch_vlm.py \\
      --sketch_dir inputs/sketches \\
      --hf_token $HF_TOKEN \\
      --cache_dir ./models \\
      --out_dir results/phase1_sketch_vlm \\
      --vlm_model Qwen/Qwen2-VL-2B-Instruct

Sketch format
-------------
  inputs/sketches/bicycle.png   (dark lines on light background)
  inputs/sketches/vase.png
  inputs/sketches/ball.png
  Names must match the 'name' field in EDITS or --config.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

# ── import utilities from phase1_composite ────────────────────────────────────
_comp_path = str(Path(__file__).parent / "phase1_composite.py")
_spec = importlib.util.spec_from_file_location("phase1_composite", _comp_path)
_comp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comp)

run_standard = _comp.run_standard
save_grid    = _comp.save_grid

# ── import load_vlm from phase1_sketch ───────────────────────────────────────
_sketch_path = str(Path(__file__).parent / "phase1_sketch.py")
_sspec = importlib.util.spec_from_file_location("phase1_sketch", _sketch_path)
_smod  = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(_smod)

load_vlm = _smod.load_vlm

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from NewWork.KontextEval.utils.model_utils import load_kontext_pipeline

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


# ── Stage A: Sketch → Object Image ───────────────────────────────────────────

def generate_from_sketch(
    pipe,
    sketch_path: str,
    description: str,
    seed: int,
    num_steps: int,
    guidance: float,
    height: int,
    width: int,
    lora_id: str,
    device: str,
) -> Image.Image:
    """
    Generate a photorealistic object from a hand-drawn sketch.

    Uses FLUX.1-Kontext-dev with the Sketch-to-Image LoRA.
    The sketch is passed as Kontext's reference image; the LoRA trigger phrase
    instructs the model to follow the sketch's structure faithfully.

    The generated image has a plain white background — it is used as visual
    context for the VLM, not pasted into the scene directly.
    """
    sketch_pil = Image.open(sketch_path).convert("RGB").resize(
        (width, height), Image.LANCZOS
    )

    print(f"  Loading LoRA: {lora_id}")
    pipe.load_lora_weights(lora_id)

    prompt = (
        f"{LORA_TRIGGER} {description} on a plain white background, "
        "photorealistic, no shadows, studio lighting, high quality."
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    obj_img = pipe(
        image=sketch_pil,
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        max_sequence_length=512,
        generator=generator,
        output_type="pil",
    ).images[0]

    pipe.unload_lora_weights()
    return obj_img


# ── VLM: generate Kontext insertion prompt ───────────────────────────────────

def vlm_generate_kontext_prompt(
    vlm_model,
    vlm_processor,
    scene_img: Image.Image,
    obj_img: Image.Image,
    description: str,
    max_new_tokens: int = 512,
) -> str:
    """
    Ask a Vision-Language Model to write a Kontext insertion prompt.

    The VLM receives:
      Image 1 — current accumulated room scene
      Image 2 — photorealistic object on white background

    It outputs a plain-text prompt for Kontext that:
      • Describes the object's appearance accurately (from Image 2)
      • Specifies where in the room to place it (from Image 1)
      • Sets orientation, scale, and surface contact
      • Ends with: 'Do not change any other part of the room.'

    The prompt is then passed directly to Kontext with the scene as reference.
    """
    device = next(vlm_model.parameters()).device

    def _resize(img: Image.Image, max_side: int) -> Image.Image:
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return img.convert("RGB")

    scene_small = _resize(scene_img, 768)
    obj_small   = _resize(obj_img,   512)

    instruction = (
        f"You are an expert image composition assistant.\n\n"
        f"Image 1: A room scene where I want to insert a new object.\n"
        f"Image 2: The {description} I want to insert (on a white background).\n\n"
        f"Write a single, complete prompt for an AI image editor called Kontext "
        f"that will add the {description} from Image 2 into the room in Image 1.\n\n"
        f"Your prompt MUST:\n"
        f"1. Describe the {description}'s exact visual appearance (color, texture, "
        f"style, proportions) as seen in Image 2\n"
        f"2. Specify WHERE in the room to place it (which wall / surface / corner, "
        f"relative to existing furniture)\n"
        f"3. Specify orientation and how it contacts the floor or surface\n"
        f"4. Set a realistic perspective-correct scale given the room's depth\n"
        f"5. End with exactly this sentence: "
        f"'Do not change any other part of the room.'\n\n"
        f"Output ONLY the prompt text. No JSON, no explanation, no markdown."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": scene_small},
                {"type": "image", "image": obj_small},
                {"type": "text",  "text": instruction},
            ],
        }
    ]

    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images_in_order = [
        item["image"]
        for msg in messages
        for item in msg["content"]
        if item["type"] == "image"
    ]
    inputs = vlm_processor(
        text=[text],
        images=images_in_order,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        gen_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    out_ids  = gen_ids[:, inputs["input_ids"].shape[1]:]
    response = vlm_processor.batch_decode(
        out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # Strip any accidental markdown code-fence wrappers
    if response.startswith("```"):
        lines = response.splitlines()
        response = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()

    return response


# ── Main incremental pipeline ─────────────────────────────────────────────────

def run_vlm_guided_chain(
    pipe,
    base: Image.Image,
    edits: List[dict],
    sketch_dir: str,
    lora_id: str,
    vlm_pair: Tuple,
    seed: int,
    num_steps: int,
    lora_guidance: float,
    scene_guidance: float,
    height: int,
    width: int,
    out_dir: str,
    device: str,
) -> List[Image.Image]:
    """
    Incremental VLM-guided insertion: Base → +Bike → +Vase → +Ball.

    For each object:
      1. Generate photorealistic object from sketch  (Stage A)
      2. VLM writes the Kontext insertion prompt     (VLM step)
      3. Kontext edits the current scene             (Stage B)
    """
    vlm_model, vlm_proc = vlm_pair
    results = [base]
    scene   = base

    for i, edit in enumerate(edits):
        name = edit["name"]
        desc = edit["description"]
        sketch_path = os.path.join(sketch_dir, f"{name}.png")

        if not os.path.isfile(sketch_path):
            raise FileNotFoundError(
                f"Sketch not found: {sketch_path}\n"
                f"Save your sketch as '{name}.png' in --sketch_dir."
            )

        print(f"\n{'─'*60}")
        print(f"  Step {i+1}/{len(edits)}  —  {name}")
        print(f"{'─'*60}")

        # ── Stage A: sketch → object image ────────────────────────────────────
        print(f"  [A] Generating '{desc}' from sketch ...")
        obj_img = generate_from_sketch(
            pipe=pipe,
            sketch_path=sketch_path,
            description=desc,
            seed=seed,
            num_steps=num_steps,
            guidance=lora_guidance,
            height=height,
            width=width,
            lora_id=lora_id,
            device=device,
        )
        obj_path = os.path.join(out_dir, f"obj_gen_{name}.png")
        obj_img.save(obj_path)
        print(f"      Saved: {obj_path}")

        # ── VLM: current scene + object → Kontext prompt ──────────────────────
        print(f"  [VLM] Generating Kontext prompt for '{name}' ...")
        kontext_prompt = vlm_generate_kontext_prompt(
            vlm_model=vlm_model,
            vlm_processor=vlm_proc,
            scene_img=scene,
            obj_img=obj_img,
            description=desc,
        )

        # Print prompt to terminal with clear formatting
        print(f"\n  {_SEP}")
        print(f"  [VLM → Kontext prompt]  {name}")
        print(f"  {_SEP}")
        for line in kontext_prompt.splitlines():
            print(f"  {line}")
        print(f"  {_SEP}\n")

        # Save prompt to file for debugging / reproducibility
        prompt_path = os.path.join(out_dir, f"vlm_prompt_{name}.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(kontext_prompt)
        print(f"      Saved: {prompt_path}")

        # ── Stage B: Kontext edits the scene ──────────────────────────────────
        print(f"  [B] Kontext integration ({num_steps} steps, g={scene_guidance}) ...")
        next_scene = run_standard(
            pipe=pipe,
            canvas=scene,
            prompt=kontext_prompt,
            seed=seed,
            num_steps=num_steps,
            guidance=scene_guidance,
            height=height,
            width=width,
        )
        result_path = os.path.join(out_dir, f"result_step{i+1}_{name}.png")
        next_scene.save(result_path)
        print(f"      Saved: {result_path}")

        scene = next_scene
        results.append(scene)

    return results


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="VLM-guided sketch-to-scene incremental object insertion."
    )
    p.add_argument("--sketch_dir",    required=True,
                   help="Folder with {name}.png hand-drawn sketches.")
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--cache_dir",     default="./models")
    p.add_argument("--out_dir",       default="results/phase1_sketch_vlm")
    p.add_argument("--config",        default=None,
                   help="JSON list of {name, description}. Overrides built-in EDITS.")
    p.add_argument("--lora_id",       default=LORA_ID,
                   help="Sketch-to-Image LoRA for Stage A object generation.")
    p.add_argument("--lora_guidance", type=float, default=4.0,
                   help="guidance_scale for Stage A LoRA generation.")
    p.add_argument("--guidance",      type=float, default=2.5,
                   help="guidance_scale for Stage B Kontext scene editing.")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_steps",     type=int,   default=28)
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--vlm_model",     default="Qwen/Qwen2-VL-2B-Instruct",
                   help="HuggingFace model ID for VLM prompt generation. "
                        "Tested: Qwen/Qwen2-VL-2B-Instruct (CPU, ~4 GB RAM), "
                        "Qwen/Qwen2-VL-7B-Instruct (GPU, better reasoning).")
    p.add_argument("--vlm_device",    default="cpu",
                   help="Device for VLM inference. Default 'cpu' keeps GPU free for Kontext.")
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
    print(f"  phase1_sketch_vlm  —  VLM-Guided Sketch-to-Scene Insertion")
    print(f"{_SEP}")
    print(f"  Objects    : {[e['name'] for e in edits]}")
    print(f"  Sketch dir : {args.sketch_dir}")
    print(f"  VLM        : {args.vlm_model}  [{args.vlm_device}]")
    print(f"  LoRA       : {args.lora_id}")
    print(f"  Output     : {args.out_dir}")
    print(f"{_SEP}\n")

    # ── 0. Load VLM ───────────────────────────────────────────────────────────
    print("Loading VLM ...")
    vlm_pair = load_vlm(args.vlm_model, args.cache_dir, args.vlm_device)

    # ── 1. Load Kontext ───────────────────────────────────────────────────────
    print("\nLoading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    # ── 2. Base scene ─────────────────────────────────────────────────────────
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(
        pipe=pipe,
        canvas=grey,
        prompt=BASE_PROMPT,
        seed=args.seed,
        num_steps=args.num_steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
    )
    base_path = os.path.join(args.out_dir, "step0_base.png")
    base.save(base_path)
    print(f"  Saved: {base_path}")

    # ── 3. Incremental insertion ──────────────────────────────────────────────
    print("\n=== Incremental insertion ===")
    results = run_vlm_guided_chain(
        pipe=pipe,
        base=base,
        edits=edits,
        sketch_dir=args.sketch_dir,
        lora_id=args.lora_id,
        vlm_pair=vlm_pair,
        seed=args.seed,
        num_steps=args.num_steps,
        lora_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        height=args.height,
        width=args.width,
        out_dir=args.out_dir,
        device=args.device,
    )

    # ── 4. Save final + grid ──────────────────────────────────────────────────
    final_path = os.path.join(args.out_dir, "KEY_final.png")
    results[-1].save(final_path)
    print(f"\n  Final scene saved: {final_path}")

    titles = ["Base"] + [e["name"] for e in edits]
    grid_path = os.path.join(args.out_dir, "KEY_grid.png")
    save_grid(results, titles, grid_path, ncols=len(results))
    print(f"  Grid saved        : {grid_path}")

    # ── 5. What to check ──────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  WHAT TO CHECK")
    print(f"{_SEP}")
    print("""
  obj_gen_{name}.png
    Stage A output: realistic from sketch.
    Should reflect the sketch's shape and proportions.
    If quality is poor → lower --lora_guidance (try 3.0) or
    check the sketch is clear dark-lines-on-light-background.

  vlm_prompt_{name}.txt  (also printed to terminal)
    The VLM-generated prompt Kontext received.
    Should describe the object's appearance AND placement accurately.
    If VLM output is vague → try --vlm_model Qwen/Qwen2-VL-7B-Instruct.

  result_step{N}_{name}.png
    Kontext's output after each insertion step.
    Object should be placed at the VLM-described location.
    Background should be largely unchanged.
    If background drifts → lower --guidance (try 2.0) or
    add 'Do not change any other part of the room.' explicitly.

  KEY_grid.png
    Side-by-side progression: Base | +Bike | +Vase | +Ball.
""")


if __name__ == "__main__":
    main()
