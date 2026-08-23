#!/usr/bin/env bash
# Task 3 — Object replacement
# Swaps an object while keeping the background and scene composition intact.
# Optionally supply --mask_image to restrict injection to the object region.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

echo "=== Task 3: Object replacement ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task object_replace \
    --name apple_to_orange \
    --source_prompt "a red apple on a wooden table" \
    --edit_prompt   "an orange on a wooden table" \
    --seed 10

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task object_replace \
    --name wooden_chair_to_metal \
    --source_prompt "a wooden chair in a bright room" \
    --edit_prompt   "a metal chair in a bright room" \
    --seed 15

echo "=== Object replacement complete. Results in results/ultimateflux/ ==="
