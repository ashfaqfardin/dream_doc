#!/usr/bin/env bash
# Task 7 — Style transfer (dual-branch from random noise)
#
# Pipeline:
#   1. Extract style K,V from reference image at the denoising noise level.
#   2. Run B=2 dual-branch denoising from random noise (same seed both branches).
#      Branch 0 (source): plain generation, no injection → source.png
#      Branch 1 (edit):   style K,V injected at TIER_A layers  → edit.png
#
# ── Injection formula at ALL 13 TIER_A layers ────────────────────────────────────
#
#   Attn( Q_edit , K_src , V_style )
#
#   Q_edit = edit branch Q (unchanged)                         → coherent with denoising
#   K_src  = source branch K                                   → spatial routing locked
#   V_sty  = style reference V (captured at t=0.3)             → colour/texture/strokes
#
#   Q-AdaIN removed: it distorted Q stats at high noise → blurry outputs.
#
# ── Injection improvements ───────────────────────────────────────────────────────
#
#   Layer-stratified V blend (Scheduled Style Injection):
#     Identity layers {0,7,8,9,10} use  style_strength × 0.6  (preserves structure)
#     Texture  layers {18,25..56}   use  style_strength × 1.0  (full style signal)
#
#   LAB histogram matching (post-process):
#     After generation, AdaIN in CIE-LAB space transfers the style reference's
#     colour palette onto edit.png.  L channel: half strength (no overexposure).
#     A/B channels: full strength (palette replacement).
#
# ── Key parameters ──────────────────────────────────────────────────────────────
#   --style_image              reference image whose style to transfer
#   --style_strength           V_style blend weight (1.0 = full style, default)
#   --content_strength         0.6 two-stage default; 0.85 when --content_image given
#                              lower = stronger identity, higher = more style freedom
#   --color_transfer_strength  LAB post-processing strength (0.6 default; 0.0 = off)
#
# Sources: StyleID arXiv:2312.09008, HAM arXiv:2603.24043, FreeFlux (TIER_A)
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

echo "=== Task 7: Identity-preserving style transfer ==="
# Place your source image in inputs/source.png and style reference in inputs/*_ref.*
# Adjust --content_strength: lower (0.6) = stronger identity, higher (0.9) = stronger style.

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name cat_watercolor \
    --prompt "A cat, watercolor painting" \
    --style_image inputs/watercolor_ref.png \
    --style_description "watercolor painting" \
    --style_strength 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name flower_watercolor \
    --prompt "A flower, watercolor painting" \
    --style_image inputs/watercolor_ref.png \
    --style_description "watercolor painting" \
    --style_strength 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name castle_oilpainting \
    --prompt "A castle, oil painting" \
    --style_image inputs/oilpainting_ref.jpg \
    --style_description "oil painting" \
    --style_strength 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name robot_3d \
    --prompt "A robot, 3D cartoon" \
    --style_image inputs/cartoon3d_ref.png \
    --style_description "3D cartoon rendering" \
    --style_strength 1.0 \
    --seed 42

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task style \
    --name cat_crayon \
    --prompt "A cat, crayon drawing" \
    --style_image inputs/drawing_ref.png \
    --style_description "crayon drawing" \
    --style_strength 1.0 \
    --seed 42

echo "=== Style transfer complete. Results in results/ultimateflux/ ==="
