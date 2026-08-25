# ─────────────────────────────────────────────────────────────────────────────
# Run all 6 Kontext experiments in recommended order.
#
# Prerequisites — run once before this script:
#   python KontextPipeline/run.py --steps 1,2 --out_dir results/kontext_setup
#
# That generates:
#   results/kontext_setup/step_00_base.png     ← base scene
#   results/kontext_setup/objects/obj_*.png    ← object images from Step 1
#
# Usage:
#   cd e:\Cherry_on_top
#   .\ExperimentKontext\run_all.ps1
# ─────────────────────────────────────────────────────────────────────────────

$SCENE   = "results/kontext_setup/step_00_base.png"
$OBJ_DIR = "results/kontext_setup/objects"
$OBJ1    = "$OBJ_DIR/obj_bicycle.png"
$OBJ2    = "$OBJ_DIR/obj_vase.png"
$DEVICE  = "cuda"

# ── Check prerequisites ───────────────────────────────────────────────────────
if (-not (Test-Path $SCENE)) {
    Write-Host "ERROR: Base scene not found at $SCENE"
    Write-Host "Run first: python KontextPipeline/run.py --steps 1,2 --out_dir results/kontext_setup"
    exit 1
}

Write-Host "=== E4: Timestep Commitment (run this first — informs all others) ===" -ForegroundColor Cyan
python ExperimentKontext/e4_timestep_commitment.py `
    --scene   $SCENE `
    --out_dir results/e4_timestep_commitment `
    --device  $DEVICE
if (-not $?) { Write-Host "E4 failed" -ForegroundColor Red; exit 1 }

Write-Host "`n=== E1: Attention Visualization ===" -ForegroundColor Cyan
python ExperimentKontext/e1_attention_viz.py `
    --scene   $SCENE `
    --out_dir results/e1_attention_viz `
    --device  $DEVICE
if (-not $?) { Write-Host "E1 failed" -ForegroundColor Red; exit 1 }

Write-Host "`n=== E3: Layer Ablation ===" -ForegroundColor Cyan
python ExperimentKontext/e3_layer_ablation.py `
    --scene   $SCENE `
    --out_dir results/e3_layer_ablation `
    --device  $DEVICE
if (-not $?) { Write-Host "E3 failed" -ForegroundColor Red; exit 1 }

Write-Host "`n=== E2: RoPE Temporal Index Ablation ===" -ForegroundColor Cyan
python ExperimentKontext/e2_rope_ablation.py `
    --scene   $SCENE `
    --obj     $OBJ1 `
    --out_dir results/e2_rope_ablation `
    --device  $DEVICE
if (-not $?) { Write-Host "E2 failed" -ForegroundColor Red; exit 1 }

Write-Host "`n=== E5: Multi-Context Attention Segregation ===" -ForegroundColor Cyan
python ExperimentKontext/e5_multi_context_attention.py `
    --scene   $SCENE `
    --obj1    $OBJ1 `
    --obj2    $OBJ2 `
    --out_dir results/e5_multi_context_attn `
    --device  $DEVICE
if (-not $?) { Write-Host "E5 failed" -ForegroundColor Red; exit 1 }

Write-Host "`n=== E6: Drift Measurement (longest — ~25 min) ===" -ForegroundColor Cyan
python ExperimentKontext/e6_drift_measurement.py `
    --scene   $SCENE `
    --obj_dir $OBJ_DIR `
    --out_dir results/e6_drift `
    --device  $DEVICE
if (-not $?) { Write-Host "E6 failed" -ForegroundColor Red; exit 1 }

Write-Host "`nAll experiments done. Results in results/e*" -ForegroundColor Green
