"""
run_graph.py — Graph-based scene editing with stitched object conditioning.

Architecture:
  State: accumulated = [list of object names currently in scene]

  Every generation call:
    pipe(image=[base, stitch(accumulated_obj_imgs)], prompt=...)
  Always regenerates from base, so:
    Add: accumulated.append(name)  → restitch → regenerate
    Remove: accumulated.remove(name) → restitch → regenerate

  No chaining of scene outputs → no artifact accumulation, removal is free.

Object images:
  Scanned from --obj_dir at runtime. Drop any .png/.jpg into that folder
  and it becomes available. Optionally provide descriptions.json alongside
  the images for richer prompts.

  If --sketch_dir is given, missing object images are generated first
  (sketch → realistic object via the edit pipeline).

Modes:
  batch       (default) Adds all objects in --obj_dir in sorted order.
  --interactive         Reads freeform commands from stdin; LLM parses them.
"""

import gc
import json
import math
import os
import re
import argparse
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_PROMPT = (
    "A photorealistic empty room with a wooden floor, white walls, "
    "and a window letting in natural light. No objects on the floor."
)


# ── Object directory helpers ──────────────────────────────────────────────────

def scan_obj_dir(obj_dir: str) -> dict:
    """Returns {name: path} for every image file in obj_dir."""
    supported = {".png", ".jpg", ".jpeg"}
    result = {}
    for fname in sorted(os.listdir(obj_dir)):
        if os.path.splitext(fname)[1].lower() in supported:
            name = os.path.splitext(fname)[0]
            result[name] = os.path.join(obj_dir, fname)
    return result


def load_descriptions(obj_dir: str, available: list) -> dict:
    """
    Load descriptions from descriptions.json if it exists, otherwise fall
    back to the object name itself.

    descriptions.json format: {"bicycle": "yellow mountain bicycle", ...}
    """
    desc_path = os.path.join(obj_dir, "descriptions.json")
    if os.path.isfile(desc_path):
        with open(desc_path) as f:
            stored = json.load(f)
    else:
        stored = {}
    return {name: stored.get(name, name.replace("_", " ")) for name in available}


def load_object_images(obj_paths: dict, cell: int) -> dict:
    """Returns {name: PIL.Image} resized to (cell, cell)."""
    return {
        name: Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS)
        for name, path in obj_paths.items()
    }


# ── Stitching ─────────────────────────────────────────────────────────────────

def stitch(images: list, canvas_size: int = 1024) -> Image.Image:
    """
    Square grid, always canvas_size × canvas_size.
    Objects fill left-to-right, top-to-bottom. Empty cells are white.

    N≤4 → 2×2 grid (512×512 per cell)
    N≤9 → 3×3 grid (341×341 per cell)
    """
    n = len(images)
    if n == 0:
        return Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    grid = max(2, math.ceil(math.sqrt(n)))
    cell = canvas_size // grid
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    for i, img in enumerate(images):
        row, col = divmod(i, grid)
        canvas.paste(img.resize((cell, cell), Image.LANCZOS), (col * cell, row * cell))
    return canvas


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(accumulated: list, descriptions: dict) -> str:
    if not accumulated:
        return BASE_PROMPT
    obj_list = ", ".join(descriptions.get(n, n) for n in accumulated)
    return (
        "A photorealistic room with a wooden floor, white walls, and a window "
        "letting in natural light. "
        f"The room contains the following objects arranged naturally: {obj_list}. "
        "Objects do not overlap. Each is naturally placed on the floor or an "
        "appropriate surface. Preserve realistic lighting and perspective."
    )


# ── Generation pipeline steps ─────────────────────────────────────────────────

def generate_base(pipe, args) -> Image.Image:
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 200))
    with torch.inference_mode():
        return pipe(
            image=[grey],
            prompt=BASE_PROMPT,
            generator=torch.manual_seed(args.seed),
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt="objects on floor, furniture, cluttered, dark",
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
        ).images[0]


def generate_scene(pipe, base: Image.Image, stitched: Image.Image,
                   prompt: str, args) -> Image.Image:
    with torch.inference_mode():
        return pipe(
            image=[base, stitched],
            prompt=prompt,
            generator=torch.manual_seed(args.seed),
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt=(
                "blurry, distorted, artifacts, duplicate objects, "
                "wrong perspective, missing objects"
            ),
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
        ).images[0]


def sketch_to_object(pipe, sketch: Image.Image, description: str, args) -> Image.Image:
    prompt = (
        f"Convert this sketch into a photorealistic {description} "
        f"on a plain white background. Keep the object centered and well-lit."
    )
    with torch.inference_mode():
        return pipe(
            image=[sketch],
            prompt=prompt,
            generator=torch.manual_seed(args.seed),
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt="blurry, distorted, low quality, extra objects, background clutter",
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
        ).images[0]


# ── LLM command parser ────────────────────────────────────────────────────────

def load_llm(model_id: str, device: str):
    print(f"Loading LLM parser: {model_id} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print("LLM ready.\n")
    return tokenizer, model


def parse_command(tokenizer, model, command: str,
                  available: list, accumulated: list) -> dict:
    """
    Returns {"action": "add"/"remove"/"none", "object": name or None}.
    """
    system = (
        "You are a scene editor assistant.\n"
        f"Available objects: {available}\n"
        f"Currently in scene: {accumulated}\n"
        "Parse the user command. Return ONLY a JSON object with exactly two keys:\n"
        '  "action": one of "add", "remove"\n'
        '  "object": one name from the available list, or null if unclear\n'
        "Return ONLY the JSON. No explanation."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": command},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    response = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    match = re.search(r'\{.*?\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"action": "none", "object": None}


# ── State update helpers ──────────────────────────────────────────────────────

def apply_action(action: str, obj_name: str, accumulated: list,
                 available: list) -> tuple:
    """
    Returns (updated_accumulated, error_message_or_None).
    Does not mutate the list; returns a new one.
    """
    acc = list(accumulated)
    if obj_name not in available:
        return acc, f"'{obj_name}' not in available objects: {available}"
    if action == "add":
        if obj_name in acc:
            return acc, f"'{obj_name}' is already in the scene."
        acc.append(obj_name)
    elif action == "remove":
        if obj_name not in acc:
            return acc, f"'{obj_name}' is not in the scene."
        acc.remove(obj_name)
    else:
        return acc, f"Unknown action '{action}'."
    return acc, None


def save_scene(scene: Image.Image, accumulated: list, out_dir: str,
               step: int, stitched: Image.Image = None):
    label = "_".join(accumulated) if accumulated else "empty"
    scene.save(os.path.join(out_dir, f"step{step:02d}_{label}.png"))
    if stitched is not None:
        stitched.save(os.path.join(out_dir, f"step{step:02d}_{label}_stitched.png"))
    print(f"  Saved step{step:02d}_{label}.png")


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(pipe, base: Image.Image, obj_images: dict,
              descriptions: dict, args):
    """Add all objects in sorted order, then optionally remove them."""
    accumulated = []
    step = 0

    for name in obj_images:
        accumulated.append(name)
        stitched = stitch([obj_images[n] for n in accumulated], args.width)
        prompt = build_prompt(accumulated, descriptions)
        print(f"\n[ADD] {name}  →  scene: {accumulated}")
        scene = generate_scene(pipe, base, stitched, prompt, args)
        save_scene(scene, accumulated, args.out_dir, step, stitched)
        step += 1

    print(f"\nBatch done. {step} scenes saved to {args.out_dir}/")


# ── Interactive mode ──────────────────────────────────────────────────────────

def run_interactive(pipe, base: Image.Image, obj_images: dict,
                    descriptions: dict, tokenizer, llm, args):
    available = list(obj_images.keys())
    accumulated = []
    scene = base
    step = 0

    print(f"Available objects: {available}")
    print("Commands: 'add <object>', 'remove <object>', 'show', 'quit'\n")

    while True:
        try:
            command = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not command:
            continue
        if command.lower() in ("quit", "exit", "q"):
            break
        if command.lower() == "show":
            print(f"  Scene: {accumulated}")
            continue

        parsed = parse_command(tokenizer, llm, command, available, accumulated)
        action   = parsed.get("action", "none")
        obj_name = parsed.get("object")

        if action == "none" or obj_name is None:
            print(f"  Could not parse. Try: 'add bicycle' or 'remove vase'.")
            continue

        new_accumulated, err = apply_action(action, obj_name, accumulated, available)
        if err:
            print(f"  {err}")
            continue

        accumulated = new_accumulated
        print(f"  [{action.upper()}] {obj_name}  →  scene: {accumulated}")

        if not accumulated:
            scene = base
            base.save(os.path.join(args.out_dir, f"step{step:02d}_empty.png"))
            print(f"  Scene is empty.")
        else:
            stitched = stitch([obj_images[n] for n in accumulated], args.width)
            prompt   = build_prompt(accumulated, descriptions)
            scene    = generate_scene(pipe, base, stitched, prompt, args)
            save_scene(scene, accumulated, args.out_dir, step, stitched)

        step += 1

    print(f"\nInteractive session ended. {step} steps saved to {args.out_dir}/")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Graph-based scene editing with stitched object conditioning."
    )
    # Directories
    p.add_argument("--obj_dir",       required=True,
                   help="Folder of object images. Each file's stem becomes the object name.")
    p.add_argument("--sketch_dir",    default=None,
                   help="If set, generate missing object images from sketches first.")
    p.add_argument("--out_dir",       default="results/qwen_graph")

    # Models
    p.add_argument("--model_id",      default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--llm_model_id",  default="Qwen/Qwen2.5-3B-Instruct")

    # Generation
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--num_steps",     type=int,   default=40)
    p.add_argument("--true_cfg_scale",type=float, default=4.0)
    p.add_argument("--guidance_scale",type=float, default=None)
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--cell_size",     type=int,   default=512,
                   help="Width and height of each object slot in the stitched strip.")

    # Mode
    p.add_argument("--interactive",   action="store_true",
                   help="Interactive mode: parse freeform commands via LLM.")
    p.add_argument("--llm_device",    default="cpu",
                   help="Device for LLM parser. cpu keeps the GPU free for generation.")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load generation pipeline ───────────────────────────────────────────
    print(f"Loading generation pipeline: {args.model_id} ...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config, use_dynamic_shifting=False
    )
    pipe.set_progress_bar_config(disable=None)
    print("Pipeline ready.\n")

    # ── Generate missing objects from sketches ─────────────────────────────
    if args.sketch_dir is not None:
        os.makedirs(args.obj_dir, exist_ok=True)
        descriptions_path = os.path.join(args.obj_dir, "descriptions.json")
        stored_desc = {}
        if os.path.isfile(descriptions_path):
            with open(descriptions_path) as f:
                stored_desc = json.load(f)

        print("Generating object images from sketches ...")
        for fname in sorted(os.listdir(args.sketch_dir)):
            if os.path.splitext(fname)[1].lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            name     = os.path.splitext(fname)[0]
            out_path = os.path.join(args.obj_dir, f"{name}.png")
            if os.path.isfile(out_path):
                print(f"  [skip] {name}")
                continue
            sketch  = Image.open(
                os.path.join(args.sketch_dir, fname)
            ).convert("RGB").resize((args.width, args.height))
            desc    = stored_desc.get(name, name.replace("_", " "))
            print(f"  [S→O] {name}: {desc}")
            obj_img = sketch_to_object(pipe, sketch, desc, args)
            obj_img.save(out_path)
            print(f"        Saved {out_path}")
        print()

    # ── Scan object directory ──────────────────────────────────────────────
    obj_paths = scan_obj_dir(args.obj_dir)
    if not obj_paths:
        raise RuntimeError(f"No object images found in {args.obj_dir!r}")

    available    = list(obj_paths.keys())
    descriptions = load_descriptions(args.obj_dir, available)
    obj_images   = load_object_images(obj_paths, args.cell_size)
    print(f"Available objects ({len(available)}): {available}\n")

    # ── Base scene ─────────────────────────────────────────────────────────
    base_path = os.path.join(args.out_dir, "base_scene.png")
    if os.path.isfile(base_path):
        print(f"Loading cached base scene: {base_path}")
        base = Image.open(base_path).convert("RGB")
    else:
        print("Generating base room scene ...")
        base = generate_base(pipe, args)
        base.save(base_path)
        print(f"Saved: {base_path}\n")

    # ── Run ────────────────────────────────────────────────────────────────
    if args.interactive:
        tokenizer, llm = load_llm(args.llm_model_id, args.llm_device)
        run_interactive(pipe, base, obj_images, descriptions, tokenizer, llm, args)
    else:
        run_batch(pipe, base, obj_images, descriptions, args)

    print(f"\nDone. All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
