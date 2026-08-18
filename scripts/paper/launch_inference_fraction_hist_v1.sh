#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/simon/otitenet
cd "$ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/paper_outputs/inference_fraction_hist_v1/launcher_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] Starting historical-plus-inference fraction experiment"
"$ROOT/.conda/bin/python" scripts/paper/run_inference_fraction_experiments.py \
  --scenario-manifest data/otite_ds_224/USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v1_seed42_scenarios.csv \
  --experiment-label inference_fraction_hist_v1 \
  --output-root paper_outputs/inference_fraction_hist_v1 \
  --top-configs 8
LATEST=$(find "$ROOT/paper_outputs/inference_fraction_hist_v1" -maxdepth 1 -mindepth 1 -type d -name "run_*" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d" " -f2-)
"$ROOT/.conda/bin/python" scripts/paper/plot_inference_fraction_performance.py --experiment-root "$LATEST" --metric test_mcc
"$ROOT/.conda/bin/python" scripts/paper/plot_inference_fraction_performance.py --experiment-root "$LATEST" --metric valid_mcc
echo "[$(date -Is)] completed"
