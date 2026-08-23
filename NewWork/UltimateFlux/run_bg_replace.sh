#!/usr/bin/env bash
# Task 4 — Background replacement
# Regenerates the background while preserving the foreground subject.
# SAM2 is used automatically to segment the foreground (pip install sam2).
# Alternatively supply --fg_mask_image for a hand-drawn mask.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

echo "=== Task 4: Background replacement ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task bg_replace \
    --name cat_forest_to_beach \
    --source_prompt "a cat sitting in a forest" \
    --edit_prompt   "a cat sitting on a sunny beach" \
    --use_sam2 \
    --save_intermediates \
    --seed 20

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task bg_replace \
    --name dog_park_to_city \
    --source_prompt "a dog standing in a park" \
    --edit_prompt   "a dog standing on a city street" \
    --use_sam2 \
    --save_intermediates \
    --seed 25

echo "=== Background replacement complete. Results in results/ultimateflux/ ==="
