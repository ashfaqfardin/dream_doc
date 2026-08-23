#!/usr/bin/env bash
# UltimateFlux — run all tasks end-to-end.
# To run a single task instead, call the individual script directly:
#   bash NewWork/UltimateFlux/run_style.sh
#   bash NewWork/UltimateFlux/run_non_rigid.sh
#   bash NewWork/UltimateFlux/run_object_add.sh
#   bash NewWork/UltimateFlux/run_object_replace.sh
#   bash NewWork/UltimateFlux/run_bg_replace.sh
#   bash NewWork/UltimateFlux/run_attr_edit.sh
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"

bash "$SCRIPT_DIR/run_non_rigid.sh"
bash "$SCRIPT_DIR/run_object_add.sh"
bash "$SCRIPT_DIR/run_object_replace.sh"
bash "$SCRIPT_DIR/run_bg_replace.sh"
bash "$SCRIPT_DIR/run_attr_edit.sh"
bash "$SCRIPT_DIR/run_style.sh"

echo "=== UltimateFlux complete. Results in results/ultimateflux/ ==="
