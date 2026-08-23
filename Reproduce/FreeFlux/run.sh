#!/usr/bin/env bash
# FreeFlux — all three editing variants
# Can be run from anywhere: bash Reproduce/FreeFlux/run.sh

# Resolve repo root from the script's own location (works regardless of CWD)
cd "$(dirname "$0")/../.." || exit 1

# ── Non-rigid editing (no extra dependencies) ────────────────────────────────
python Reproduce/FreeFlux/non_rigid/run_non_rigid.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_freeflux.json \
    --device cuda \
    --cache_dir ./models --save_images

# ── Add-object (no extra dependencies) ──────────────────────────────────────
python Reproduce/FreeFlux/add_object/run_add_object.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_freeflux_add_object.json \
    --device cuda \
    --cache_dir ./models --save_images

# ── Background replace (requires SAM2) ───────────────────────────────────────
pip install git+https://github.com/facebookresearch/sam2

python Reproduce/FreeFlux/bg_replace/run_bg_replace.py \
    --hf_token "$HF_TOKEN" \
    --config prompts/reproduce_freeflux_bg_replace.json \
    --device cuda \
    --cache_dir ./models --save_images
