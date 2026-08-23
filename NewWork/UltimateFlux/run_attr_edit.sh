#!/usr/bin/env bash
# Task 5 — Fine-grained attribute editing
#
# inject_layers controls the editing mode:
#   (default / omit)        → _PRESERVE_LAYERS (TIER_A + HOTSPOT, 20 layers)
#                              Tightest identity lock — for ADDING an attribute
#                              (glasses, hat, beard).
#   --inject_layers double_stream
#                           → Lock all 19 double-stream (joint text-image) blocks
#                              with K+V injection; 38 single-stream blocks are FREE.
#                              Recommended if colour change is weak.
#
#   --inject_layers color   → Kontext-style colour editing (KontextColorPolicy).
#                              SAC in all 19 double-stream blocks: replace edit-branch
#                              image Q and K with source Q and K → locks face geometry
#                              and object shape.  Text Q/K stay from edit branch so the
#                              edit prompt ("blonde", "blue") conditions V freely.
#                              Single-stream blocks (19-56) run with NO injection —
#                              edit text drives colour through all 38 refinement blocks.
#                              Uses generate_dual_branch (standard B=2 loop).
#                              Optional: --svd_alpha 1.0 adds PFB at block 1 for extra
#                              identity anchoring (start without it; add if face drifts).
#
#   --inject_layers tier_a  → TIER_A only (13 layers)
#                              Appearance preserved, position flexible —
#                              for shape-linked edits (breed change).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="black-forest-labs/FLUX.1-dev"
STEPS=28
CFG=3.5
H=1024
W=1024

echo "=== Task 5: Fine-grained attribute editing ==="

# ── Colour editing: Kontext-style SAC (Q+K injection in double-stream) ────────
#
# Algorithm:
#   1. B=2 dual-branch loop: [source_prompt, edit_prompt] share identical z_T.
#   2. In every double-stream block (0-18), edit-branch image-token Q and K are
#      replaced with source-branch values.  This forces the edit branch to attend
#      to exactly the same spatial positions as the source (face, car silhouette).
#      Text tokens keep the edit branch's Q/K so "blonde"/"blue" conditions V.
#   3. Single-stream blocks (19-56): no injection.  Edit text drives colour.
#
# Tuning:
#   --ss_q_steps_frac 0.0 0.5    Single-stream Q injection window (default first 50%).
#                                 Raise end toward 1.0 for tighter identity.
#                                 Lower end toward 0.0 for stronger colour.
#                                 K injection runs for all steps regardless.
#   --qk_steps_frac 0.0 0.7      Reduce double-stream Q+K window if colour is weak.
#   --svd_alpha 1.0              Enable PFB at block 1 for extra structural anchor.

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name woman_hair_color \
    --source_prompt "a woman with red hair" \
    --edit_prompt   "a woman with black hair" \
    --ss_q_steps_frac 0.0 0.5 \
    --inject_layers color \
    --save_intermediates --intermediate_every 4 \
    --seed 35

python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name car_color_change \
    --source_prompt "a green sports car on a road" \
    --edit_prompt   "a black sports car on a road" \
    --ss_q_steps_frac 0.0 0.5 \
    --inject_layers color \
    --save_intermediates --intermediate_every 4 \
    --seed 40

# ── Add an accessory (strong identity preservation) ───────────────────────────
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name man_add_glasses \
    --source_prompt "a portrait photo of a man" \
    --edit_prompt   "a portrait photo of a man wearing eyeglasses" \
    --seed 30

# ── Shape-linked edit (tier_a — preserves appearance, loosens layout) ─────────
python NewWork/UltimateFlux/run_ultimateflux.py \
    --hf_token "$HF_TOKEN" --model_path "$MODEL" \
    --device cuda --cache_dir ./models --save_images \
    --num_steps "$STEPS" --guidance_scale "$CFG" --height "$H" --width "$W" \
    --task attr_edit \
    --name dog_breed_shape \
    --source_prompt "a labrador sitting on grass" \
    --edit_prompt   "a husky sitting on grass" \
    --inject_layers tier_a \
    --seed 45

echo "=== Attribute editing complete. Results in results/ultimateflux/ ==="
