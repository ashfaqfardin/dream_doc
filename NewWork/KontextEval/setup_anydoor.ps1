# setup_anydoor.ps1 — One-time setup for AnyDoor
# Run from e:\Cherry_on_top\
#
# Usage:
#   cd e:\Cherry_on_top
#   .\NewWork\KontextEval\setup_anydoor.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ROOT

Write-Host "=== AnyDoor Setup ===" -ForegroundColor Cyan
Write-Host "Working dir: $ROOT"

# ── 1. Clone AnyDoor repo ──────────────────────────────────────────────────────
if (-not (Test-Path "AnyDoor")) {
    Write-Host "`n[1] Cloning AnyDoor repo ..." -ForegroundColor Yellow
    git clone https://github.com/ali-vilab/AnyDoor.git
} else {
    Write-Host "`n[1] AnyDoor already cloned at $ROOT\AnyDoor" -ForegroundColor Green
}

# ── 2. Install AnyDoor dependencies ───────────────────────────────────────────
Write-Host "`n[2] Installing AnyDoor Python dependencies ..." -ForegroundColor Yellow
# Note: We don't install pytorch_lightning==1.5.0 strictly since our FLUX env
# may have a newer version. AnyDoor's cldm module uses LightningModule but only
# for model definition — inference doesn't call trainer methods.
pip install "pytorch_lightning==1.5.0" omegaconf einops open_clip_torch timm albumentations

# ── 3. Download DINOv2 ViT-g/14 weights ───────────────────────────────────────
Write-Host "`n[3] Downloading DINOv2 ViT-g/14 weights ..." -ForegroundColor Yellow
$dinov2_path = "AnyDoor\dinov2_vitg14_pretrain.pth"
if (-not (Test-Path $dinov2_path)) {
    # Use Python to download (avoids wget on Windows)
    python -c @"
import urllib.request, os, sys
url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth'
out = 'AnyDoor/dinov2_vitg14_pretrain.pth'
print(f'Downloading {url} ...')
print('This is ~3.5 GB, may take a while.')
urllib.request.urlretrieve(url, out, reporthook=lambda b,bs,t: print(f'  {b*bs/1e9:.2f}/{t/1e9:.2f} GB', end='\r'))
print(f'\nSaved: {out}')
"@
} else {
    Write-Host "  DINOv2 already at $dinov2_path" -ForegroundColor Green
}

# ── 4. Update anydoor.yaml with DINOv2 path ───────────────────────────────────
Write-Host "`n[4] Patching AnyDoor/configs/anydoor.yaml with DINOv2 path ..." -ForegroundColor Yellow
$yaml_path = "AnyDoor\configs\anydoor.yaml"
$dinov2_abs = (Resolve-Path $dinov2_path).Path.Replace('\', '/')
$yaml_content = Get-Content $yaml_path -Raw
# Replace the weight path line (line 83 area, looks like: weight: path/dinov2_vitg14_pretrain.pth)
$yaml_content = $yaml_content -replace 'weight:\s+path/dinov2_vitg14_pretrain\.pth', "weight: $dinov2_abs"
$yaml_content = $yaml_content -replace 'weight:\s+\./dinov2_vitg14_pretrain\.pth',   "weight: $dinov2_abs"
Set-Content $yaml_path $yaml_content -Encoding UTF8
Write-Host "  Patched: $yaml_path"

# ── 5. Download AnyDoor checkpoint ────────────────────────────────────────────
Write-Host "`n[5] Downloading AnyDoor checkpoint ..." -ForegroundColor Yellow
$ckpt_path = "AnyDoor\epoch=1-step=8687.ckpt"
if (-not (Test-Path $ckpt_path)) {
    Write-Host "  Trying huggingface-cli download ..."
    # The checkpoint is in the HuggingFace space repo
    python -c @"
from huggingface_hub import hf_hub_download
import shutil, os
print('Downloading AnyDoor checkpoint from HuggingFace...')
print('This is ~5-8 GB, may take a while.')
path = hf_hub_download(
    repo_id   = 'xichenhku/AnyDoor',
    filename  = 'epoch=1-step=8687.ckpt',
    repo_type = 'space',
    local_dir = 'AnyDoor',
)
print(f'Saved: {path}')
"@
} else {
    Write-Host "  Checkpoint already at $ckpt_path" -ForegroundColor Green
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Now run the pipeline:" -ForegroundColor Cyan
Write-Host @"
python NewWork/KontextEval/phase1_anydoor.py ``
    --sketch_dir NewWork/KontextEval/inputs ``
    --hf_token `$env:HF_TOKEN ``
    --anydoor_dir ./AnyDoor ``
    --anydoor_ckpt ./AnyDoor/epoch=1-step=8687.ckpt ``
    --cache_dir ./models ``
    --out_dir results/phase1_anydoor ``
    --vlm_model Qwen/Qwen2-VL-2B-Instruct
"@
