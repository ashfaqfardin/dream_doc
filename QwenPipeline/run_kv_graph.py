"""
run_kv_graph.py — Object KV Library + Latent Warm-Start + DAAM injection.

CVPR proposal — three-phase architecture:

  Phase 0 (once):           Run full denoising on base room.
                            Cache latent trajectory Z_base[t] at every step.

  Phase 1 (once per obj):   Run 1-step forward pass on each obj_img.
                            Capture K,V at every selected attention layer → KVBank.

  Phase 2 (per edit):       Warm-start from Z_base[warm_steps] (skips structure steps).
                            Inject merged KV from all accumulated objects.
                            Run only (num_steps - warm_steps) denoising steps.

Cost comparison (N=7 edits, num_steps=40, warm_steps=28):
  run_graph.py:     40 × 7              = 280 steps
  run_kv_graph.py:  40 + 7 + 12×7      = 131 steps

Add / Remove are symmetric — both cost the same 12 steps.
"""

import gc
import json
import math
import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_PROMPT = (
    "A photorealistic empty room with a wooden floor, white walls, "
    "and a window letting in natural light. No objects on the floor."
)


# ══════════════════════════════════════════════════════════════════════════════
# Attention layer discovery
# ══════════════════════════════════════════════════════════════════════════════

def discover_kv_layers(
    model: nn.Module,
    k_suffix: str = "to_k",
    v_suffix: str = "to_v",
) -> Tuple[List[str], List[str]]:
    """
    Scan model.named_modules() for K,V projection layers.
    Returns (k_names, v_names) as parallel lists ordered by depth.
    Works with any transformer regardless of architecture.
    """
    k_names, v_names = [], []
    for name, _ in model.named_modules():
        if name.endswith(k_suffix):
            k_names.append(name)
        if name.endswith(v_suffix):
            v_names.append(name)
    return k_names, v_names


def select_layers(names: List[str], every_n: int = 3) -> List[str]:
    """Subsample layers evenly — avoids saturating every layer with injection."""
    return names[::every_n]


def get_denoiser(pipe) -> nn.Module:
    for attr in ("transformer", "unet", "model"):
        if hasattr(pipe, attr):
            return getattr(pipe, attr)
    raise AttributeError("Cannot locate denoising model in pipeline.")


# ══════════════════════════════════════════════════════════════════════════════
# KV Bank  — capture once per object, inject during edits
# ══════════════════════════════════════════════════════════════════════════════

class KVBank:
    """
    Stores {obj_name → {layer_name → tensor}} on CPU.

    Sequence-length mismatch between capture (1 obj_img) and injection
    (scene with [base, stitched]) is resolved by mean-pooling the captured
    K,V over the sequence dimension before broadcasting to the scene's tokens.

    This trades spatial specificity for robustness — appearance cues are
    distributed uniformly across all gen tokens rather than matched 1:1.
    """

    def __init__(self, k_names: List[str], v_names: List[str]):
        self.k_names = k_names
        self.v_names = v_names
        self._store: Dict[str, Dict[str, torch.Tensor]] = {}
        self._capture_hooks: List = []
        self._inject_hooks:  List = []

    # ── Capture ───────────────────────────────────────────────────────────

    def capture_start(self, model: nn.Module, obj_name: str):
        """Register hooks; next forward pass on model will populate the bank."""
        captured: Dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def hook(module, inp, out):
                captured[name] = out.detach().cpu()
            return hook

        for name in self.k_names + self.v_names:
            try:
                mod = model.get_submodule(name)
                self._capture_hooks.append(mod.register_forward_hook(make_hook(name)))
            except AttributeError:
                pass  # layer not found — skip silently

        self._pending = (obj_name, captured)

    def capture_end(self):
        """Remove capture hooks and commit tensors to store."""
        for h in self._capture_hooks:
            h.remove()
        self._capture_hooks.clear()
        obj_name, captured = self._pending
        self._store[obj_name] = captured
        self._pending = None
        return len(captured)

    def has(self, name: str) -> bool:
        return name in self._store

    def keys(self) -> List[str]:
        return list(self._store.keys())

    # ── Inject ────────────────────────────────────────────────────────────

    def inject_start(
        self,
        model: nn.Module,
        obj_names: List[str],
        alpha_k: float = 0.70,
        alpha_v: float = 0.40,
    ):
        """
        Register injection hooks for all requested objects.
        Multiple objects' K,V are averaged per layer before blending.
        """
        # Pre-merge K,V across all accumulated objects
        merged_k: Dict[str, torch.Tensor] = {}
        merged_v: Dict[str, torch.Tensor] = {}

        for layer in self.k_names:
            tensors = [
                self._store[n][layer]
                for n in obj_names
                if n in self._store and layer in self._store[n]
            ]
            if tensors:
                merged_k[layer] = torch.stack(tensors).mean(0)  # (1, seq, d) or (1, H, seq, d)

        for layer in self.v_names:
            tensors = [
                self._store[n][layer]
                for n in obj_names
                if n in self._store and layer in self._store[n]
            ]
            if tensors:
                merged_v[layer] = torch.stack(tensors).mean(0)

        def make_hook(src: torch.Tensor, alpha: float):
            def hook(module, inp, out):
                # Mean-pool over sequence (dim=-2) to handle length mismatch,
                # then broadcast to the scene's full sequence length.
                s = src.to(out.device, out.dtype)
                if s.dim() == 3:          # (1, seq_obj, d)
                    s = s.mean(dim=1, keepdim=True).expand_as(out)
                elif s.dim() == 4:        # (1, H, seq_obj, d)
                    s = s.mean(dim=2, keepdim=True).expand_as(out)
                return (1.0 - alpha) * out + alpha * s
            return hook

        for name, src in merged_k.items():
            try:
                mod = model.get_submodule(name)
                h = mod.register_forward_hook(make_hook(src, alpha_k))
                self._inject_hooks.append(h)
            except AttributeError:
                pass

        for name, src in merged_v.items():
            try:
                mod = model.get_submodule(name)
                h = mod.register_forward_hook(make_hook(src, alpha_v))
                self._inject_hooks.append(h)
            except AttributeError:
                pass

    def inject_end(self):
        for h in self._inject_hooks:
            h.remove()
        self._inject_hooks.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Latent Cache  — capture base trajectory, warm-start edits
# ══════════════════════════════════════════════════════════════════════════════

class LatentCache:
    """
    Captures the noisy latent at every denoising step of the base generation.
    Provides a warm-start point for subsequent edits:
      instead of starting from Gaussian noise, start from Z_base[warm_steps]
      which already encodes the room's global structure and lighting.
    """

    def __init__(self):
        self._trajectory: Dict[int, torch.Tensor] = {}

    def make_callback(self):
        store = self._trajectory

        def callback(pipe, step_idx: int, timestep, kwargs: dict) -> dict:
            latents = kwargs.get("latents")
            if latents is not None:
                store[step_idx] = latents.detach().cpu()
            return kwargs

        return callback

    def get_warm_latent(self, warm_step: int) -> Optional[torch.Tensor]:
        if not self._trajectory:
            return None
        # Clamp to the last cached step if warm_step > total steps
        step = min(warm_step, max(self._trajectory.keys()))
        return self._trajectory.get(step)

    def is_ready(self) -> bool:
        return bool(self._trajectory)

    def num_steps_cached(self) -> int:
        return len(self._trajectory)


# ══════════════════════════════════════════════════════════════════════════════
# DAAM accumulator  — soft spatial attention map over gen tokens
# ══════════════════════════════════════════════════════════════════════════════

class DAAMAccumulator:
    """
    Accumulates gen→text attention energy during Phase 1 (warm steps).
    After warm_steps forward passes, compute_mask() returns a boolean
    spatial mask: True where the forming object occupies gen-token space.

    Architecture note: hooks to_q and to_k at the first discovered
    attention layer. Computes Q @ K^T manually after each step.
    This avoids architecture-specific attention API assumptions.
    """

    def __init__(self):
        self._heatmap: Optional[torch.Tensor] = None
        self._count = 0
        self._q: Optional[torch.Tensor] = None
        self._k: Optional[torch.Tensor] = None
        self._hooks: List = []

    def register(self, model: nn.Module, q_layer: str, k_layer: str):
        def hook_q(mod, inp, out):
            self._q = out.detach().cpu()

        def hook_k(mod, inp, out):
            self._k = out.detach().cpu()
            self._accumulate()

        try:
            self._hooks.append(
                model.get_submodule(q_layer).register_forward_hook(hook_q))
            self._hooks.append(
                model.get_submodule(k_layer).register_forward_hook(hook_k))
        except AttributeError:
            pass

    def _accumulate(self):
        if self._q is None or self._k is None:
            return
        q, k = self._q.float(), self._k.float()
        # q, k: (B, seq, d) or (B, H, seq, d)
        # Normalize to (B, seq, d) by treating heads as batch if needed
        if q.dim() == 4:
            B, H, S, d = q.shape
            q = q.permute(0, 2, 1, 3).reshape(B, S, H * d)
            k = k.permute(0, 2, 1, 3).reshape(B, S, H * d)
        # Raw dot product between all token pairs: (seq, seq)
        attn = torch.softmax(
            (q[0] @ k[0].T) * (q.shape[-1] ** -0.5), dim=-1
        )
        energy = attn.mean(0)  # (seq,) — how much each K-token is attended to
        self._heatmap = energy if self._heatmap is None else self._heatmap + energy
        self._count += 1
        self._q = self._k = None

    def compute_mask(self, n_gen: int, top_frac: float = 0.25) -> Optional[torch.Tensor]:
        """
        Returns a (n_gen,) bool mask — True at the top_frac most-attended gen tokens.
        n_gen: number of gen-image tokens (first n_gen tokens in the sequence).
        """
        if self._heatmap is None or self._count == 0:
            return None
        avg = self._heatmap / self._count       # (seq,)
        gen_energy = avg[:n_gen]                # (n_gen,)
        threshold = torch.quantile(gen_energy, 1.0 - top_frac)
        return gen_energy >= threshold

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        self._heatmap = None
        self._count = 0
        self._q = self._k = None


# ══════════════════════════════════════════════════════════════════════════════
# Precompute steps
# ══════════════════════════════════════════════════════════════════════════════

def build_kv_bank(
    pipe,
    kv_bank: KVBank,
    obj_images: Dict[str, Image.Image],
    descriptions: Dict[str, str],
    args,
):
    """
    Phase 1: 1-step forward pass per object → populate KVBank.
    Cost: 1 step × N objects.
    """
    denoiser = get_denoiser(pipe)

    for name, img in obj_images.items():
        if kv_bank.has(name):
            print(f"  [KV]  {name}: already cached.")
            continue

        print(f"  [KV]  {name}: capturing ...")
        kv_bank.capture_start(denoiser, name)

        with torch.inference_mode():
            pipe(
                image=[img],
                prompt=f"A photorealistic {descriptions.get(name, name)} "
                       f"on a plain white background.",
                generator=torch.manual_seed(args.seed),
                true_cfg_scale=args.true_cfg_scale,
                negative_prompt="background clutter, blurry",
                num_inference_steps=1,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
            )

        n = kv_bank.capture_end()
        print(f"         {n} tensors captured.")


def generate_base_and_cache(
    pipe,
    latent_cache: LatentCache,
    args,
) -> Image.Image:
    """
    Phase 0: generate base room, cache latent at every step.
    """
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 200))
    with torch.inference_mode():
        result = pipe(
            image=[grey],
            prompt=BASE_PROMPT,
            generator=torch.manual_seed(args.seed),
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt="objects on floor, furniture, cluttered, dark",
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            callback_on_step_end=latent_cache.make_callback(),
            callback_on_step_end_tensor_inputs=["latents"],
        )
    print(f"  Latent trajectory cached: {latent_cache.num_steps_cached()} steps.")
    return result.images[0]


# ══════════════════════════════════════════════════════════════════════════════
# Edit step
# ══════════════════════════════════════════════════════════════════════════════

def run_edit(
    pipe,
    kv_bank: KVBank,
    latent_cache: LatentCache,
    base: Image.Image,
    stitched: Image.Image,
    prompt: str,
    accumulated: List[str],
    args,
) -> Image.Image:
    """
    Phase 2: warm-start from cached latent + KV injection.

    If latent_cache is not ready (e.g. base loaded from disk, no trajectory),
    falls back to full denoising from noise — identical to run_graph.py.
    """
    denoiser = get_denoiser(pipe)

    # ── Warm-start latent ──────────────────────────────────────────────────
    init_latents = None
    edit_steps   = args.num_steps

    if latent_cache.is_ready():
        cached = latent_cache.get_warm_latent(args.warm_steps)
        if cached is not None:
            init_latents = cached.to(args.device)
            edit_steps   = max(1, args.num_steps - args.warm_steps)
            print(f"  [WARM] Starting from cached latent at step {args.warm_steps} "
                  f"→ running {edit_steps} denoising steps.")
        else:
            print(f"  [WARN] warm_steps={args.warm_steps} not in cache "
                  f"({latent_cache.num_steps_cached()} steps). Falling back to full denoising.")
    else:
        print(f"  [INFO] No latent cache — running full {edit_steps} steps.")

    # ── KV injection ───────────────────────────────────────────────────────
    kv_bank.inject_start(
        denoiser,
        accumulated,
        alpha_k=args.alpha_k,
        alpha_v=args.alpha_v,
    )

    call_kwargs = dict(
        image=[base, stitched],
        prompt=prompt,
        generator=torch.manual_seed(args.seed),
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=(
            "blurry, distorted, artifacts, duplicate objects, "
            "wrong perspective, missing objects"
        ),
        num_inference_steps=edit_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=1,
    )
    if init_latents is not None:
        call_kwargs["latents"] = init_latents

    try:
        with torch.inference_mode():
            result = pipe(**call_kwargs)
    finally:
        kv_bank.inject_end()

    return result.images[0]


# ══════════════════════════════════════════════════════════════════════════════
# Scene helpers  (shared with run_graph.py)
# ══════════════════════════════════════════════════════════════════════════════

def stitch(images: List[Image.Image], canvas_size: int = 1024) -> Image.Image:
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


def build_prompt(accumulated: List[str], descriptions: Dict[str, str]) -> str:
    if not accumulated:
        return BASE_PROMPT
    obj_list = ", ".join(descriptions.get(n, n) for n in accumulated)
    return (
        "A photorealistic room with a wooden floor, white walls, and a window "
        "letting in natural light. "
        f"The room contains the following objects arranged naturally: {obj_list}. "
        "Objects do not overlap. Each is naturally placed. "
        "Preserve realistic lighting and perspective."
    )


def scan_obj_dir(obj_dir: str) -> Dict[str, str]:
    supported = {".png", ".jpg", ".jpeg"}
    result = {}
    for fname in sorted(os.listdir(obj_dir)):
        if os.path.splitext(fname)[1].lower() in supported:
            name = os.path.splitext(fname)[0]
            result[name] = os.path.join(obj_dir, fname)
    return result


def load_descriptions(obj_dir: str, available: List[str]) -> Dict[str, str]:
    desc_path = os.path.join(obj_dir, "descriptions.json")
    stored: Dict[str, str] = {}
    if os.path.isfile(desc_path):
        with open(desc_path) as f:
            stored = json.load(f)
    return {n: stored.get(n, n.replace("_", " ")) for n in available}


def load_object_images(obj_paths: Dict[str, str], cell: int) -> Dict[str, Image.Image]:
    return {
        name: Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS)
        for name, path in obj_paths.items()
    }


def apply_action(
    action: str, obj_name: str,
    accumulated: List[str], available: List[str],
) -> Tuple[List[str], Optional[str]]:
    acc = list(accumulated)
    if obj_name not in available:
        return acc, f"'{obj_name}' not found. Available: {available}"
    if action == "add":
        if obj_name in acc:
            return acc, f"'{obj_name}' already in scene."
        acc.append(obj_name)
    elif action == "remove":
        if obj_name not in acc:
            return acc, f"'{obj_name}' not in scene."
        acc.remove(obj_name)
    else:
        return acc, f"Unknown action '{action}'."
    return acc, None


def save_scene(scene: Image.Image, accumulated: List[str], out_dir: str, step: int):
    label = "_".join(accumulated) if accumulated else "empty"
    path  = os.path.join(out_dir, f"step{step:02d}_{label}.png")
    scene.save(path)
    print(f"  Saved: step{step:02d}_{label}.png")


def sketch_to_object(pipe, sketch: Image.Image, description: str, args) -> Image.Image:
    with torch.inference_mode():
        return pipe(
            image=[sketch],
            prompt=f"Convert this sketch into a photorealistic {description} "
                   f"on a plain white background. Keep the object centered and well-lit.",
            generator=torch.manual_seed(args.seed),
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt="blurry, distorted, low quality, extra objects, background clutter",
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
        ).images[0]


# ══════════════════════════════════════════════════════════════════════════════
# LLM command parser
# ══════════════════════════════════════════════════════════════════════════════

def load_llm(model_id: str, device: str):
    print(f"Loading LLM parser: {model_id} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print("LLM ready.\n")
    return tokenizer, model


def parse_command(
    tokenizer, model,
    command: str, available: List[str], accumulated: List[str],
) -> dict:
    system = (
        "You are a scene editor assistant.\n"
        f"Available objects: {available}\n"
        f"Currently in scene: {accumulated}\n"
        'Parse the user command. Return ONLY JSON: {"action": "add"/"remove", "object": name_or_null}'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": command},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


# ══════════════════════════════════════════════════════════════════════════════
# Run modes
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(pipe, kv_bank, latent_cache, base, obj_images, descriptions, args):
    accumulated: List[str] = []
    step = 0
    for name in obj_images:
        accumulated.append(name)
        stitched = stitch([obj_images[n] for n in accumulated], args.width)
        prompt   = build_prompt(accumulated, descriptions)
        print(f"\n[ADD] {name}  →  scene: {accumulated}")
        scene = run_edit(pipe, kv_bank, latent_cache, base, stitched,
                         prompt, accumulated, args)
        save_scene(scene, accumulated, args.out_dir, step)
        step += 1
    print(f"\nBatch done. {step} scenes in {args.out_dir}/")


def run_interactive(pipe, kv_bank, latent_cache, base, obj_images,
                    descriptions, tokenizer, llm, args):
    available    = list(obj_images.keys())
    accumulated: List[str] = []
    step = 0

    print(f"Available: {available}")
    print("Commands: 'add <obj>', 'remove <obj>', 'show', 'quit'\n")

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

        parsed   = parse_command(tokenizer, llm, command, available, accumulated)
        action   = parsed.get("action", "none")
        obj_name = parsed.get("object")

        if action == "none" or obj_name is None:
            print("  Could not parse. Try: 'add bicycle' or 'remove vase'.")
            continue

        new_acc, err = apply_action(action, obj_name, accumulated, available)
        if err:
            print(f"  {err}")
            continue

        accumulated = new_acc
        print(f"  [{action.upper()}] {obj_name}  →  scene: {accumulated}")

        if not accumulated:
            base.save(os.path.join(args.out_dir, f"step{step:02d}_empty.png"))
            print("  Scene is empty (base room).")
        else:
            stitched = stitch([obj_images[n] for n in accumulated], args.width)
            prompt   = build_prompt(accumulated, descriptions)
            scene    = run_edit(pipe, kv_bank, latent_cache, base, stitched,
                                prompt, accumulated, args)
            save_scene(scene, accumulated, args.out_dir, step)

        step += 1

    print(f"\nSession ended. {step} steps in {args.out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Object KV Library + Latent Warm-Start scene editing."
    )
    p.add_argument("--obj_dir",        required=True,
                   help="Folder of object images (stem → name).")
    p.add_argument("--sketch_dir",     default=None,
                   help="If set, generate missing object images from sketches first.")
    p.add_argument("--out_dir",        default="results/qwen_kv_graph")

    p.add_argument("--model_id",       default="Qwen/Qwen-Image-Edit-2509")
    p.add_argument("--llm_model_id",   default="Qwen/Qwen2.5-3B-Instruct")

    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--num_steps",      type=int,   default=40)
    p.add_argument("--warm_steps",     type=int,   default=28,
                   help="Steps to skip per edit by warm-starting from cached base latent. "
                        "Edit cost = num_steps - warm_steps.")
    p.add_argument("--alpha_k",        type=float, default=0.70,
                   help="K injection strength [0,1]. Higher → stronger appearance transfer.")
    p.add_argument("--alpha_v",        type=float, default=0.40,
                   help="V injection strength [0,1]. Lower → preserve scene structure.")
    p.add_argument("--kv_every_n",     type=int,   default=3,
                   help="Inject at every N-th discovered attention layer.")
    p.add_argument("--true_cfg_scale", type=float, default=4.0)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--height",         type=int,   default=1024)
    p.add_argument("--width",          type=int,   default=1024)
    p.add_argument("--cell_size",      type=int,   default=512)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--interactive",    action="store_true")
    p.add_argument("--llm_device",     default="cpu")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load generation pipeline ───────────────────────────────────────────
    print(f"Loading pipeline: {args.model_id} ...")
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
        desc_path   = os.path.join(args.obj_dir, "descriptions.json")
        stored_desc = json.load(open(desc_path)) if os.path.isfile(desc_path) else {}
        print("Generating objects from sketches ...")
        for fname in sorted(os.listdir(args.sketch_dir)):
            if os.path.splitext(fname)[1].lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            name     = os.path.splitext(fname)[0]
            out_path = os.path.join(args.obj_dir, f"{name}.png")
            if os.path.isfile(out_path):
                print(f"  [skip] {name}")
                continue
            sketch = Image.open(
                os.path.join(args.sketch_dir, fname)
            ).convert("RGB").resize((args.width, args.height))
            desc   = stored_desc.get(name, name.replace("_", " "))
            print(f"  [S→O] {name}: {desc}")
            obj_img = sketch_to_object(pipe, sketch, desc, args)
            obj_img.save(out_path)
        print()

    # ── Scan objects ───────────────────────────────────────────────────────
    obj_paths    = scan_obj_dir(args.obj_dir)
    if not obj_paths:
        raise RuntimeError(f"No object images found in {args.obj_dir!r}")
    available    = list(obj_paths.keys())
    descriptions = load_descriptions(args.obj_dir, available)
    obj_images   = load_object_images(obj_paths, args.cell_size)
    print(f"Available ({len(available)}): {available}\n")

    # ── Discover attention layers ──────────────────────────────────────────
    denoiser = get_denoiser(pipe)
    k_names, v_names = discover_kv_layers(denoiser)
    k_sel = select_layers(k_names, args.kv_every_n)
    v_sel = select_layers(v_names, args.kv_every_n)
    print(f"Attention layers discovered: {len(k_names)} K + {len(v_names)} V total")
    print(f"Selected for KV injection:   {len(k_sel)} K + {len(v_sel)} V "
          f"(every {args.kv_every_n} layers)\n")

    if not k_sel:
        print("[WARN] No 'to_k' layers found. Trying 'k_proj' / 'key' ...")
        k_sel, v_sel = discover_kv_layers(denoiser, "k_proj", "v_proj")
        k_sel = select_layers(k_sel, args.kv_every_n)
        v_sel = select_layers(v_sel, args.kv_every_n)
        print(f"       Found {len(k_sel)} K + {len(v_sel)} V with 'k_proj'/'v_proj'\n")

    # ── Phase 1: build KV bank ─────────────────────────────────────────────
    kv_bank = KVBank(k_sel, v_sel)
    print("Phase 1 — Building object KV bank (1 step per object) ...")
    build_kv_bank(pipe, kv_bank, obj_images, descriptions, args)
    print(f"KV bank ready: {kv_bank.keys()}\n")

    # ── Phase 0: base scene + latent trajectory ────────────────────────────
    latent_cache = LatentCache()
    base_path    = os.path.join(args.out_dir, "base_scene.png")

    if os.path.isfile(base_path):
        print(f"Loading cached base: {base_path}")
        print(f"  [NOTE] No latent trajectory — edits will use full {args.num_steps} steps.\n")
        base = Image.open(base_path).convert("RGB")
    else:
        print("Phase 0 — Generating base room + caching latent trajectory ...")
        base = generate_base_and_cache(pipe, latent_cache, args)
        base.save(base_path)
        print(f"  Saved: {base_path}\n")

    # Print edit cost summary
    edit_steps = args.num_steps - args.warm_steps if latent_cache.is_ready() else args.num_steps
    print(f"Edit cost per operation: {edit_steps} steps  "
          f"({'warm-start' if latent_cache.is_ready() else 'full denoising'})\n")

    # ── Run ────────────────────────────────────────────────────────────────
    if args.interactive:
        tokenizer, llm = load_llm(args.llm_model_id, args.llm_device)
        run_interactive(pipe, kv_bank, latent_cache, base, obj_images,
                        descriptions, tokenizer, llm, args)
    else:
        run_batch(pipe, kv_bank, latent_cache, base, obj_images, descriptions, args)

    print(f"\nDone. All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
