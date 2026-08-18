#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/simon/otitenet
cd "$ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/paper_outputs/inference_fraction_hist_v2/launcher_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] Starting corrected historical-plus-inference fraction experiment"
"$ROOT/.conda/bin/python" scripts/paper/run_inference_fraction_experiments.py \
  --scenario-manifest data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v2_seed42_scenarios.csv \
  --selected-configs paper_outputs/inference_fraction_hist_v1/run_20260805T173121Z/selected_configs.csv \
  --experiment-label inference_fraction_hist_v2 \
  --output-root paper_outputs/inference_fraction_hist_v2 \
  --top-configs 8 \
  --num-workers 4
LATEST=$(find "$ROOT/paper_outputs/inference_fraction_hist_v2" -maxdepth 1 -mindepth 1 -type d -name "run_*" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d" " -f2-)
"$ROOT/.conda/bin/python" scripts/paper/plot_inference_fraction_performance.py --experiment-root "$LATEST" --metric test_mcc
"$ROOT/.conda/bin/python" scripts/paper/plot_inference_fraction_performance.py --experiment-root "$LATEST" --metric valid_mcc
echo "[$(date -Is)] completed"
