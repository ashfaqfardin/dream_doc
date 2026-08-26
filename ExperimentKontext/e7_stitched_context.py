"""
E7: Stitched Multi-Object Context

Instead of chaining edits (E6) or multi-context images (E5), this experiment
collapses all reference objects into a single grid image and passes it as one
context alongside the base scene: image=[base_scene, stitch_grid].

Conditions (cumulative — always generated from base, no chaining):
  stitch_01_bicycle      — grid contains only o1
  stitch_02_+vase        — grid contains o1 + o2
  ...
  stitch_07_+backpack    — grid contains all 7 objects

Metrics per condition:
  bg_ssim / bg_lpips     — background stability vs. original base
  dino_{obj}             — DINOv2 cosine: result vs. each reference object
  clip_{obj}             — CLIP-I: result vs. each reference object

Optional VLM placement (--vlm_model):
  A VLM analyses the base scene and suggests specific, room-aware placements
  for each object. The prompt then references both the grid position and the
  suggested location, helping Kontext place objects correctly.

Runtime: ~7 × 1 min = ~8 min  (one generation per stitch size)
"""
import os, sys, argparse, json, math
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_pipe, enable_multi_context, compute_ssim, compute_lpips, compute_dino, compute_clip_i


# Fixed object order matching E6 for direct comparison
OBJ_ORDER = ["bicycle", "vase", "ball", "chair", "lamp", "plant", "backpack"]

# Grid cell position labels (row, col) → human-readable
_ROW = ["top", "middle", "bottom"]
_COL = ["left", "center", "right"]


def grid_position(idx: int, dim: int) -> str:
    """Human-readable label for cell idx in a dim×dim grid."""
    if dim == 1:
        return "in the reference image"
    r, c = divmod(idx, dim)
    return f"{_ROW[min(r, 2)]}-{_COL[min(c, 2)]}"


def make_stitch_grid(obj_imgs: list, grid_size: int = 1024,
                     bg_color: tuple = (180, 180, 180)) -> Image.Image:
    """
    Arrange obj_imgs in a square dim×dim grid (always square).
      1 image  → 1×1
      2–4 imgs → 2×2
      5–9 imgs → 3×3
    Empty cells are filled with bg_color.
    Output is always grid_size × grid_size pixels.
    """
    n = len(obj_imgs)
    dim = math.ceil(math.sqrt(n))
    cell = grid_size // dim
    grid = Image.new("RGB", (grid_size, grid_size), bg_color)
    for i, img in enumerate(obj_imgs):
        r, c = divmod(i, dim)
        thumb = img.resize((cell, cell), Image.Resampling.LANCZOS)
        grid.paste(thumb, (c * cell, r * cell))
    return grid


# ── VLM placement ──────────────────────────────────────────────────────────────

_vlm_cache = {}

def vlm_suggest_placements(scene: Image.Image, names: list, positions: list,
                            vlm_model: str, device: str) -> dict:
    """
    Ask a VLM to suggest room-aware placement for each object.
    Returns {name: placement_string}.
    """
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as _VLMCls
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration as _VLMCls
        except ImportError:
            from transformers import AutoModel as _VLMCls

    if vlm_model not in _vlm_cache:
        print(f"Loading VLM: {vlm_model} ...")
        proc  = AutoProcessor.from_pretrained(vlm_model, trust_remote_code=True)
        model = _VLMCls.from_pretrained(
            vlm_model, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
        _vlm_cache[vlm_model] = (proc, model)
    proc, model = _vlm_cache[vlm_model]

    obj_lines = "\n".join(
        f"- {name} (shown {pos} in the reference grid)"
        for name, pos in zip(names, positions)
    )
    question = (
        "Look at this room carefully.\n"
        f"I want to place the following objects into it:\n{obj_lines}\n\n"
        "For each object, suggest one specific, realistic placement in the room "
        "(e.g. 'against the left wall on the floor', 'on the windowsill', "
        "'in the far corner'). Be brief — one phrase per object.\n"
        "Reply ONLY in this format, one line per object:\n"
        "object_name: placement description"
    )

    # Build multimodal input — works with Qwen2-VL style processors
    try:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": scene},
            {"type": "text",  "text": question},
        ]}]
        text_in = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs  = proc(text=[text_in], images=[scene], return_tensors="pt").to(device)
    except Exception:
        # Fallback for simpler processor APIs
        inputs = proc(images=scene, text=question, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    raw = proc.decode(out[0], skip_special_tokens=True)

    # Parse "name: description" lines
    placements = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            for name in names:
                if name in key:
                    placements[name] = val.strip()
                    break

    # Fallback for any unparsed objects
    for name in names:
        if name not in placements:
            placements[name] = "naturally in the room scene"

    return placements


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_prompt(names: list, positions: list,
                 placements: dict | None = None) -> str:
    """
    Build the Kontext prompt.
    If placements is provided (from VLM), each object gets a specific location.
    Otherwise falls back to a generic prompt.
    """
    lines = []
    for name, pos in zip(names, positions):
        ref_hint = f"(shown {pos} in the reference image)"
        if placements and name in placements:
            loc = placements[name]
            lines.append(
                f"Place the {name} {ref_hint} {loc}."
            )
        else:
            lines.append(
                f"Place the {name} {ref_hint} naturally in the room."
            )

    body = " ".join(lines)
    return (
        body + " Keep the exact appearance, color, and design of each object "
        "exactly as shown in the reference image."
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",     required=True,  help="Base room image")
    p.add_argument("--obj_dir",   required=True,  help="Directory with obj_<name>.png files")
    p.add_argument("--out_dir",   default="results/e7_stitched_context")
    p.add_argument("--model_id",  default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--steps",     type=int, default=28)
    p.add_argument("--grid_size", type=int, default=1024,
                   help="Pixel width/height of the stitched grid image")
    p.add_argument("--vlm_model", default=None,
                   help="VLM for room-aware placement prompts "
                        "(e.g. Qwen/Qwen2-VL-2B-Instruct). "
                        "Omit to use generic prompts.")
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = load_pipe(args.model_id, args.device)
    enable_multi_context(pipe)

    base = Image.open(args.scene).convert("RGB")

    obj_imgs = {}
    for name in OBJ_ORDER:
        path = os.path.join(args.obj_dir, f"obj_{name}.png")
        if os.path.isfile(path):
            obj_imgs[name] = Image.open(path).convert("RGB")
    available = [n for n in OBJ_ORDER if n in obj_imgs]
    print(f"Found {len(available)} objects: {available}")
    if args.vlm_model:
        print(f"VLM placement enabled: {args.vlm_model}")

    metrics = []

    for k in range(1, len(available) + 1):
        names = available[:k]
        dim   = math.ceil(math.sqrt(k))
        positions = [grid_position(i, dim) for i in range(k)]
        label = f"stitch_{k:02d}_{'_'.join(names)}"
        print(f"\n[{k}/{len(available)}] {label}")

        # Build stitch grid
        stitch = make_stitch_grid([obj_imgs[n] for n in names], args.grid_size)
        stitch.save(os.path.join(args.out_dir, f"{label}_grid.png"))

        # Get VLM placements (or None for generic prompt)
        placements = None
        if args.vlm_model:
            placements = vlm_suggest_placements(
                base, names, positions, args.vlm_model, args.device
            )
            print("  VLM placements:")
            for name in names:
                print(f"    {name}: {placements.get(name, '—')}")

        prompt = build_prompt(names, positions, placements)
        print(f"  Prompt: {prompt}")

        result = pipe(
            image=[base, stitch],
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=2.5,
            generator=torch.Generator(args.device).manual_seed(42),
        ).images[0]
        result.save(os.path.join(args.out_dir, f"{label}_result.png"))

        m = {
            "k":        k,
            "objects":  names,
            "prompt":   prompt,
            "bg_ssim":  compute_ssim(base, result),
            "bg_lpips": compute_lpips(base, result, args.device),
        }
        for name in names:
            m[f"dino_{name}"] = compute_dino(obj_imgs[name], result, args.device)
            m[f"clip_{name}"] = compute_clip_i(obj_imgs[name], result, args.device)

        metrics.append(m)
        print(f"  bg_ssim={m['bg_ssim']:.3f}  bg_lpips={m['bg_lpips']:.3f}")
        for name in names:
            print(f"  {name}: DINO={m[f'dino_{name}']:.3f}  CLIP={m[f'clip_{name}']:.3f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _plot(metrics, available, args.out_dir)
    print(f"\nDone. Results in {args.out_dir}")


def _plot(metrics: list, available: list, out_dir: str):
    ks     = [m["k"] for m in metrics]
    labels = [m["objects"][-1] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(ks, [m["bg_ssim"] for m in metrics],
                    marker='o', color='steelblue', linewidth=2)
    axes[0, 0].set_title("Background SSIM vs original\n(higher = stable)")
    axes[0, 0].set_ylim(0, 1); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ks, [m["bg_lpips"] for m in metrics],
                    marker='o', color='coral', linewidth=2)
    axes[0, 1].set_title("Background LPIPS vs original\n(lower = stable)")
    axes[0, 1].grid(True, alpha=0.3)

    for name in available:
        dinos  = [m[f"dino_{name}"] for m in metrics if f"dino_{name}" in m]
        ks_obj = [m["k"]            for m in metrics if f"dino_{name}" in m]
        axes[1, 0].plot(ks_obj, dinos, marker='o', linewidth=2, label=name)
    axes[1, 0].set_title("DINOv2 per object vs. reference\n(higher = identity preserved)")
    axes[1, 0].set_ylim(-0.2, 1); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)

    for name in available:
        clips  = [m[f"clip_{name}"] for m in metrics if f"clip_{name}" in m]
        ks_obj = [m["k"]            for m in metrics if f"clip_{name}" in m]
        axes[1, 1].plot(ks_obj, clips, marker='o', linewidth=2, label=name)
    axes[1, 1].set_title("CLIP-I per object vs. reference\n(higher = identity preserved)")
    axes[1, 1].set_ylim(0, 1); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7)

    for ax in axes.flat:
        ax.set_xticks(ks)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_xlabel("Cumulative objects in stitch (last added)")

    plt.suptitle(
        "E7: Stitched Multi-Object Context — background stability & identity vs stitch size",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_chart.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
