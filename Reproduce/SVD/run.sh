#!/usr/bin/env bash
# SVD Style Personalization on Infinity-8B
# Can be run from anywhere: bash Reproduce/SVD/run.sh

# Resolve repo root from the script's own location
cd "$(dirname "$0")/../.." || exit 1

# ── Clone Infinity repo if not already present ───────────────────────────────
if [ ! -d "Infinity" ]; then
    git clone https://github.com/FoundationVision/Infinity.git Infinity
fi

# 1. Install prerequisites
pip install ninja packaging whl

# 2. Directly grab the fast pre-compiled wheel for your exact Python 3.12 + PyTorch setup
pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"


# ── Run all style personalization configs ────────────────────────────────────
python Reproduce/SVD/run_svd_style.py \
    --hf_token "$HF_TOKEN" \
    --infinity_repo Infinity \
    --model_size 8b \
    --config prompts/reproduce_svd_style.json \
    --device cuda \
    --cache_dir ./models --save_images
