#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/simon/otitenet
cd "$ROOT"
# Multiprocess DataLoaders and Optuna/SQLite need more than the distro's small
# default fd allowance during a multi-day study. The loader fix prevents leaks;
# this is additional headroom for transient workers and open artifacts.
ulimit -n 65535
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/paper_outputs/inference_fraction_fresh_optuna/launcher_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] Starting fresh Siamese then CNN/MLP fraction studies"
exec "$ROOT/.conda/bin/python" scripts/paper/run_inference_fraction_optuna_fresh.py \
  --scenario-manifest data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v2_seed42_scenarios.csv \
  --output-root paper_outputs/inference_fraction_fresh_optuna \
  --n-trials 20 \
  --num-workers 8 \
  --siamese-batch-size 64 \
  --cnn-batch-size 128
