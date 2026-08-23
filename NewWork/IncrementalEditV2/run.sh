#!/usr/bin/env bash
#
# run.sh — entry point for IncrementEdit on a GPU box.
#
# Usage:
#   ./run.sh                 # setup check -> CPU tests -> GPU demo
#   ./run.sh --tests-only    # just the logic-core tests (no GPU, no FLUX weights)
#   ./run.sh --demo-only     # skip tests, go straight to the FLUX demo
#   ./run.sh --setup         # install deps into the active env, then exit
#
# The default order runs the CPU tests BEFORE loading FLUX on purpose: they catch
# token-order / occlusion-cascade bugs in well under a second, instead of after
# minutes of model loading.

set -euo pipefail

# --- locate project root (dir this script lives in) --------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PY="${PYTHON:-python}"
DO_SETUP=0
DO_TESTS=1
DO_DEMO=1

for arg in "$@"; do
  case "$arg" in
    --setup)       DO_SETUP=1; DO_TESTS=0; DO_DEMO=0 ;;
    --tests-only)  DO_DEMO=0 ;;
    --demo-only)   DO_TESTS=0 ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $arg (see --help)"; exit 2 ;;
  esac
done

echo "=== IncrementEdit ==="
echo "project root: $PROJECT_ROOT"

# --- optional setup ----------------------------------------------------------
if [[ "$DO_SETUP" == "1" ]]; then
  echo "--- installing requirements ---"
  $PY -m pip install -r requirements.txt
  echo "done. FLUX.1-dev is gated: run 'huggingface-cli login' if you haven't."
  exit 0
fi

# --- environment report ------------------------------------------------------
echo "--- environment ---"
$PY - <<'PYEOF'
import importlib, sys
print("python:", sys.version.split()[0])
for m in ("numpy", "scipy", "torch", "diffusers", "transformers"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:12s} MISSING ({type(e).__name__})")
try:
    import torch
    print("  cuda:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
except Exception:
    print("  cuda: torch not importable")
PYEOF

# --- CPU logic-core tests (no GPU / no FLUX weights) -------------------------
if [[ "$DO_TESTS" == "1" ]]; then
  echo "--- running logic-core tests (CPU) ---"
  $PY -m pytest tests/ -q
  echo "tests passed."
fi

# --- GPU demo ----------------------------------------------------------------
if [[ "$DO_DEMO" == "1" ]]; then
  echo "--- running FLUX demo (needs GPU + FLUX.1-dev weights) ---"
  mkdir -p outputs
  # write demo images into outputs/ regardless of where demo.py is run from
  ( cd outputs && $PY "$PROJECT_ROOT/demo.py" )
  echo "demo complete. images + masks in: $PROJECT_ROOT/outputs/"
fi
