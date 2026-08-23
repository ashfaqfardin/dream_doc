"""
GPU demo — requires torch + diffusers + a FLUX.1-dev checkout.

    pip install -r requirements.txt
    python demo.py --hf_token hf_...

This runs a 3-edit sequential session and saves each turn's image + mask.
On CPU-only / no-diffusers environments this will raise on import — that's
expected; the tested logic core runs without it (see tests/).
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="IncrementalEdit FLUX.1-dev demo")
    parser.add_argument("--hf_token",   type=str, required=True,
                        help="HuggingFace token for FLUX.1-dev (gated model)")
    parser.add_argument("--cache_dir",  type=str, default="./models",
                        help="Local directory to cache downloaded weights")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Offload model components to CPU between uses (saves VRAM)")
    parser.add_argument("--num_steps",  type=int, default=28)
    parser.add_argument("--height",     type=int, default=1024)
    parser.add_argument("--width",      type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from diffusers import FluxPipeline
    from flux_seqedit.pipeline import SequentialEditor, EditConfig

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    editor = SequentialEditor(
        pipe,
        EditConfig(height=args.height, width=args.width, num_steps=args.num_steps),
    )

    session = [
        ("a cozy empty living room", None),
        ("a cozy empty living room with a red armchair", "a red armchair"),
        ("a cozy living room with a red armchair and a sleeping cat", "a sleeping cat"),
    ]

    bg_prompt, _ = session[0]
    editor.memory.set_background(bg_prompt)

    for i, (prompt, obj) in enumerate(session[1:], start=1):
        out = editor.add_object(prompt, obj, seed=42 + i)
        out["image"].save(f"edit_{i}_{obj.replace(' ', '_')}.png")
        print(f"--- edit {i}: added '{obj}' ---")
        print(out["memory_summary"])


if __name__ == "__main__":
    main()
