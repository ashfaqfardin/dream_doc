#!/usr/bin/env bash
# Task 2 — Object addition (FreeFlux position-dependent layout layers)
# Adds a new object to a scene while preserving the existing layout.
# --added_word  : the single word for the new object (used for attention-based
#                 mask derivation via the reasoning pass).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

echo "=== Task 2: Object addition ==="

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task object_add \
    --name table_add_vase \
    --source_prompt "a wooden dining table in a bright room" \
    --edit_prompt   "a wooden dining table with a vase of flowers in a bright room" \
    --added_word    "vase" \
    --seed 50

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task object_add \
    --name park_add_dog \
    --source_prompt "a park bench surrounded by trees" \
    --edit_prompt   "a park bench with a dog sitting next to it surrounded by trees" \
    --added_word    "dog" \
    --seed 55

echo "=== Object addition complete. Results in results/ultimateflux/ ==="
