"""
Quick test: does FluxKontextPipeline accept image=[img1, img2, ...]?
Uses num_inference_steps=1 to fail fast without full diffusion.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusers", "src"))

from diffusers import FluxKontextPipeline
from PIL import Image

DEVICE = "cuda"
MODEL  = "black-forest-labs/FLUX.1-Kontext-dev"

def solid(color, size=512):
    return Image.new("RGB", (size, size), color)

pipe = FluxKontextPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.to(DEVICE)
print("Pipeline loaded.\n")

img_a = solid((200, 100, 100))  # reddish
img_b = solid((100, 200, 100))  # greenish
img_c = solid((100, 100, 200))  # bluish

tests = [
    ("single image",       img_a),
    ("list of 2 images",   [img_a, img_b]),
    ("list of 3 images",   [img_a, img_b, img_c]),
]

for label, images in tests:
    print(f"Testing: {label} ...", end=" ", flush=True)
    try:
        out = pipe(
            image=images,
            prompt="a simple test",
            num_inference_steps=1,
            guidance_scale=2.5,
        ).images[0]
        print(f"OK  — output size: {out.size}")
    except Exception as e:
        print(f"FAIL — {type(e).__name__}: {e}")
