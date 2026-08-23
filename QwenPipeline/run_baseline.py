import os
import argparse
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler


# ── Edit list ─────────────────────────────────────────────────────────────────

EDITS = [
    {"name": "bicycle", "description": "yellow mountain bicycle"},
    {"name": "vase",    "description": "black ceramic vase with flowers"},
    {"name": "ball",    "description": "yellow rubber ball"},
    {"name": "chair",   "description": "wooden chair"},
    {"name": "lamp",    "description": "modern lamp"},
    {"name": "plant",   "description": "potted green plant"},
    {"name": "backpack", "description": "blue backpack"}
]

BASE_PROMPT = (
    "A photorealistic empty room with a wooden floor, white walls, "
    "and a window letting in natural light. No objects on the floor."
)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def sketch_to_object(pipe, sketch: Image.Image, description: str, args) -> Image.Image:
    """Step 1: Si → Oi  (sketch → realistic object image)"""
    prompt = (
        f"Convert this sketch into a photorealistic {description} "
        f"on a plain white background. Keep the object centered and well-lit."
    )
    inputs = {
        "image": [sketch],
        "prompt": prompt,
        "generator": torch.manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": "blurry, distorted, low quality, extra objects, background clutter",
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        return pipe(**inputs).images[0]


def generate_base(pipe, args) -> Image.Image:
    """Step 2: Generate base room scene Bi"""
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    inputs = {
        "image": [grey],
        "prompt": BASE_PROMPT,
        "generator": torch.manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": "objects on floor, furniture, cluttered, dark",
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        return pipe(**inputs).images[0]


def add_object(pipe, scene: Image.Image, obj: Image.Image, description: str, args) -> Image.Image:
    """Step 3: Bn + On → B(n+1)  (add object into current scene)"""
    prompt = (
        f"The first image is a room scene. The second image shows a {description}. "
        f"Place the {description} naturally on the floor of the room. "
        f"Preserve the room's lighting, perspective, and all existing contents exactly."
    )
    inputs = {
        "image": [scene, obj],
        "prompt": prompt,
        "generator": torch.manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": "blurry, distorted, duplicate objects, wrong perspective",
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        return pipe(**inputs).images[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_sketch(sketch_dir: str, name: str) -> str:
    for fname in (f"{name}.png", f"sketch_{name}.png", f"{name}.jpg"):
        p = os.path.join(sketch_dir, fname)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Sketch not found for '{name}' in {sketch_dir!r}")


def save_grid(images: list, labels: list, path: str):
    W, H  = images[0].size
    n     = len(images)
    grid  = Image.new("RGB", (W * n, H + 30), (240, 240, 240))
    for i, (img, label) in enumerate(zip(images, labels)):
        grid.paste(img, (i * W, 30))
    grid.save(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sketch_dir",    required=True)
    p.add_argument("--out_dir",       default="results/qwen_pipeline")
    p.add_argument("--model_id",      default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--num_steps",     type=int,   default=40)
    p.add_argument("--true_cfg_scale",type=float, default=4.0)
    p.add_argument("--guidance_scale",type=float, default=None) # To fix guidance scale warning, set to None to use default value from the model
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    )
    pipe.to(args.device)
    _patch_scheduler(pipe)
    pipe.set_progress_bar_config(disable=None)
    print("Pipeline loaded.\n")

    results = []

    # ── Step 1: Sketch → Object for each edit ─────────────────────────────
    objects = {}
    for edit in EDITS:
        name = edit["name"]
        desc = edit["description"]
        sketch_path = find_sketch(args.sketch_dir, name)
        sketch      = Image.open(sketch_path).convert("RGB").resize((args.width, args.height))

        print(f"[S→O] {name}: {sketch_path}")
        obj_img = sketch_to_object(pipe, sketch, desc, args)
        obj_img.save(os.path.join(args.out_dir, f"obj_{name}.png"))
        objects[name] = obj_img
        print(f"      Saved obj_{name}.png")

    # ── Step 2: Generate base scene ────────────────────────────────────────
    print("\n[BASE] Generating base room scene ...")
    base = generate_base(pipe, args)
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    results.append(base)
    print("       Saved base_scene.png")

    # ── Step 3: Incremental insertion B1+O1=B2, B2+O2=B3, ... ─────────────
    scene = base
    for i, edit in enumerate(EDITS):
        name = edit["name"]
        desc = edit["description"]
        print(f"\n[ADD] Step {i+1}/{len(EDITS)}: inserting '{desc}' ...")
        scene = add_object(pipe, scene, objects[name], desc, args)
        scene.save(os.path.join(args.out_dir, f"scene_step{i+1}_{name}.png"))
        results.append(scene)
        print(f"      Saved scene_step{i+1}_{name}.png")

    # ── Save chain grid ────────────────────────────────────────────────────
    labels = ["base"] + [e["name"] for e in EDITS]
    save_grid(results, labels, os.path.join(args.out_dir, "chain_grid.png"))
    print(f"\nDone. Grid saved: {args.out_dir}/chain_grid.png")


if __name__ == "__main__":
    main()
