"""Generate only the base scenes for the Qwen mask-guided pipeline.

This stage intentionally performs no object generation, placement, masking, or
editing.  It reads ``e5_prompts.json`` and renders one empty base image for each
selected case with Qwen-Image-Edit-2509 + the 8-step Lightning LoRA.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
DEFAULT_PROMPTS = HERE / "e5_prompts.json"
DEFAULT_OUTPUT = HERE / "outputs"


def save_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("prompts")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty 'prompts' list")

    seen: set[int] = set()
    for case in cases:
        if not isinstance(case, dict) or "id" not in case or "base_prompt" not in case:
            raise ValueError(f"Malformed prompt case: {case!r}")
        case_id = int(case["id"])
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
    return cases


def select_cases(cases: list[dict], requested: list[int] | None) -> list[dict]:
    if not requested:
        return cases
    wanted = set(requested)
    selected = [case for case in cases if int(case["id"]) in wanted]
    missing = wanted - {int(case["id"]) for case in selected}
    if missing:
        raise ValueError(f"Unknown case ids: {sorted(missing)}")
    return selected


def lightning_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_config(
        {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
    )


def prepare_peft_lora_backend() -> None:
    """Ignore an obsolete optional TorchAO backend for this bf16 LoRA.

    PEFT probes its TorchAO LoRA dispatcher whenever torchao is installed. Old
    Colab images can contain torchao 0.10.0, whose availability check raises
    before PEFT reaches its ordinary torch.nn.Linear dispatcher. This pipeline
    is bf16 and not TorchAO-quantized, so disabling only that optional probe is
    the correct fallback.
    """
    try:
        installed = version("torchao")
    except PackageNotFoundError:
        return

    try:
        from packaging.version import Version

        incompatible = Version(installed) <= Version("0.16.0")
    except Exception:
        incompatible = installed.startswith(("0.0", "0.1"))

    if not incompatible:
        return

    try:
        from peft import import_utils as peft_import_utils
        from peft.tuners.lora import torchao as peft_torchao

        # PEFT versions call one or both of these module-local symbols.
        peft_import_utils.is_torchao_available = lambda: False
        peft_torchao.is_torchao_available = lambda: False
        warnings.warn(
            f"torchao {installed} is incompatible with PEFT LoRA loading; "
            "disabled the optional TorchAO dispatcher for this bf16 pipeline.",
            stacklevel=2,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Installed torchao {installed} is incompatible with PEFT. "
            "Run `pip uninstall -y torchao` or install torchao>0.16.0, then retry."
        ) from exc


def load_pipeline(args):
    from diffusers import QwenImageEditPlusPipeline

    loading = tqdm(total=3, desc="Loading Qwen base generator", unit="stage", dynamic_ncols=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_id,
        scheduler=lightning_scheduler(),
        torch_dtype=torch.bfloat16,
    )
    loading.update()
    loading.set_description("Loading 8-step Lightning LoRA")
    prepare_peft_lora_backend()
    pipe.load_lora_weights(
        args.lightning_repo,
        weight_name=args.lightning_weight,
        adapter_name="lightning",
    )
    pipe.set_adapters(["lightning"], adapter_weights=[args.lora_scale])
    loading.update()
    loading.set_description(f"Moving Qwen to {args.device}")
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    loading.update()
    loading.close()
    return pipe


def make_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.startswith("cuda") else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def generate_base(pipe, prompt: str, args, seed: int) -> Image.Image:
    blank = Image.new("RGB", (args.width, args.height), "white")
    instruction = (
        "Replace the complete blank Image 1 with the following scene. "
        f"{prompt} "
        "Fill the entire frame. Do not leave a white border, blank canvas, text, "
        "watermark, split view, collage, or reference panel."
    )
    cfg_enabled = args.true_cfg_scale > 1.0
    result = pipe(
        image=[blank],
        prompt=instruction,
        negative_prompt=args.negative_prompt if cfg_enabled else None,
        true_cfg_scale=args.true_cfg_scale,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        generator=make_generator(args.device, seed),
    )
    return result.images[0].convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case_ids", type=int, nargs="+", help="Generate only these case IDs")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--lightning_repo", default="lightx2v/Qwen-Image-Lightning")
    parser.add_argument(
        "--lightning_weight",
        default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="Base seed; case ID is added deterministically")
    parser.add_argument("--true_cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.prompts = args.prompts.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(load_cases(args.prompts), args.case_ids)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    save_json({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, args.out_dir / "config.json")

    pending = []
    records = []
    for case in cases:
        case_id = int(case["id"])
        target = args.out_dir / f"base_{case_id:03d}.png"
        seed = args.seed + case_id * 10_000
        record = {
            "id": case_id,
            "base_prompt": case["base_prompt"],
            "seed": seed,
            "image": str(target.resolve()),
        }
        records.append(record)
        if not (args.resume and target.is_file()):
            pending.append((case, target, seed))

    if pending:
        pipe = load_pipeline(args)
        for case, target, seed in tqdm(pending, desc="Generating base images", unit="base", dynamic_ncols=True):
            generate_base(pipe, case["base_prompt"], args, seed).save(target)
    else:
        print("All requested base images already exist; nothing to generate.")

    save_json(records, args.out_dir / "bases.json")
    print(f"Ready: {len(records)} base image record(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
