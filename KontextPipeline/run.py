import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusers", "src"))

from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image


BASE_PROMPT = (
    "A photorealistic empty room with a wooden floor, white walls, "
    "and a window letting in natural light. No objects on the floor."
)

EDITS = [
    {"name": "bicycle", "description": "yellow mountain bicycle",        "scene_prompt": "Add a yellow mountain bicycle"},
    {"name": "vase",    "description": "black ceramic vase with flowers","scene_prompt": "Add a black ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball",             "scene_prompt": "Add a yellow rubber ball"},
    {"name": "chair",   "description": "wooden chair",                   "scene_prompt": "Add a wooden chair"},
    {"name": "lamp",    "description": "modern floor lamp",              "scene_prompt": "Add a modern floor lamp"},
    {"name": "plant",   "description": "potted green plant",             "scene_prompt": "Add a potted green plant"},
    {"name": "backpack","description": "blue backpack",                  "scene_prompt": "Add a blue backpack"},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sketch_dir",     default="KontextPipeline/sketch", help="Folder containing sketch PNGs")
    p.add_argument("--base_image",     default=None,   help="Path or URL to base scene (generated if omitted)")
    p.add_argument("--out_dir",        default="results/kontext_incremental")
    p.add_argument("--model_id",       default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--num_steps",      type=int,   default=28)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--steps",          default="1,2,3", help="Comma-separated steps to run: 1=sketch→obj, 2=base, 3=edits")
    return p.parse_args()


# ── Step 1: Sketch → Object ───────────────────────────────────────────────────

def sketch_to_object(pipe, sketch_path: str, description: str, args) -> Image.Image:
    sketch = Image.open(sketch_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    prompt = (
        f"Render this sketch as a photorealistic {description} "
        f"on a plain white background. Keep the object centered and well-lit."
    )
    return pipe(
        image=sketch,
        prompt=prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        generator=torch.Generator(args.device).manual_seed(args.seed),
    ).images[0]


# ── Step 2: Generate base scene ───────────────────────────────────────────────

def generate_base(pipe, args) -> Image.Image:
    canvas = Image.new("RGB", (1024, 1024), (200, 200, 200))
    return pipe(
        image=canvas,
        prompt=BASE_PROMPT,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        generator=torch.Generator(args.device).manual_seed(args.seed),
    ).images[0]


# ── Step 3: Incremental scene edits ──────────────────────────────────────────

def add_object(pipe, scene: Image.Image, prompt: str, args) -> Image.Image:
    return pipe(
        image=scene,
        prompt=prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        generator=torch.Generator(args.device).manual_seed(args.seed),
    ).images[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    obj_dir = os.path.join(args.out_dir, "objects")
    os.makedirs(obj_dir, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    )
    pipe.to(args.device)
    print("Pipeline ready.\n")

    run_steps = {int(s) for s in args.steps.split(",")}

    # ── Step 1: Sketch → Object ───────────────────────────────────────────────
    objects = {}
    if 1 not in run_steps:
        print("=== Step 1: SKIPPED ===")
    else:
        print("=== Step 1: Sketch → Object ===")
        for edit in EDITS:
            name = edit["name"]
            sketch_path = os.path.join(args.sketch_dir, f"{name}.png")
            if not os.path.isfile(sketch_path):
                print(f"  [SKIP] No sketch found for '{name}' at {sketch_path}")
                continue
            print(f"  [{name}] {sketch_path} → obj_{name}.png")
            obj_img = sketch_to_object(pipe, sketch_path, edit["description"], args)
            out_path = os.path.join(obj_dir, f"obj_{name}.png")
            obj_img.save(out_path)
            objects[name] = obj_img
            print(f"         Saved {out_path}")

    # ── Step 2: Base scene ────────────────────────────────────────────────────
    if 2 not in run_steps:
        print("\n=== Step 2: SKIPPED ===")
        scene = None
    else:
        print("\n=== Step 2: Base Scene ===")
        if args.base_image is not None:
            src = args.base_image
            scene = load_image(src) if src.startswith("http") else Image.open(src).convert("RGB")
            print("  Loaded from:", src)
        else:
            print("  Generating from grey canvas ...")
            scene = generate_base(pipe, args)
            print("  Done.")
        scene.save(os.path.join(args.out_dir, "step_00_base.png"))
        print("  Saved step_00_base.png")

    # ── Step 3: Incremental edits ─────────────────────────────────────────────
    if 3 not in run_steps:
        print("\n=== Step 3: SKIPPED ===")
    else:
        if scene is None:
            print("\n=== Step 3: SKIPPED (no base scene) ===")
        else:
            print("\n=== Step 3: Incremental Edits ===")
            for i, edit in enumerate(EDITS, start=1):
                name = edit["name"]
                if name not in objects:
                    print(f"  [{i}] SKIP '{name}' (no object image)")
                    continue
                print(f"  [{i}/{len(EDITS)}] Adding {name}: {edit['scene_prompt']}")
                scene = add_object(pipe, scene, edit["scene_prompt"], args)
                out_path = os.path.join(args.out_dir, f"step_{i:02d}_{name}.png")
                scene.save(out_path)
                print(f"        Saved {out_path}")

    print(f"\nDone. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()
