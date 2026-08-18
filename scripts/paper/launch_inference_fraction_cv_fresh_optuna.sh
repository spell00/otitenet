#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/simon/otitenet
BASE="$ROOT/data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference_20260803"
OUT_BASE="$ROOT/data/otite_ds_64"
PREFIX=USA_Turquie_Chili_GMFUNL_inference_fraction_cv5
MANIFEST="$OUT_BASE/${PREFIX}_seed42_scenarios.csv"
OUTPUT_ROOT="$ROOT/paper_outputs/inference_fraction_cv_fresh_optuna"

cd "$ROOT"
ulimit -n 65535

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUTPUT_ROOT/launcher_${STAMP}.log"
mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] Preparing/reusing five-fold inference cross-test scenarios"
if [[ ! -f "$MANIFEST" ]]; then
  "$ROOT/.conda/bin/python" scripts/paper/prepare_inference_fraction_cv_scenarios.py \
    --base-dir "$BASE" \
    --out-base "$OUT_BASE" \
    --prefix "$PREFIX" \
    --fractions 0.5,0.25,0.1,0.05,0.02,0 \
    --n-splits 5 \
    --group-column name \
    --seed 42 \
    --image-mode hardlink
else
  echo "Reusing existing scenario manifest: $MANIFEST"
fi

echo "[$(date -Is)] Validating all commands before launching"
"$ROOT/.conda/bin/python" scripts/paper/run_inference_fraction_cv_optuna_fresh.py \
  --scenario-manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT/dry_runs" \
  --n-trials 20 \
  --num-workers 8 \
  --siamese-batch-size 64 \
  --cnn-batch-size 128 \
  --dry-run

echo "[$(date -Is)] Starting 30 Siamese then 30 CNN/MLP fresh Optuna studies"
exec "$ROOT/.conda/bin/python" scripts/paper/run_inference_fraction_cv_optuna_fresh.py \
  --scenario-manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --n-trials 20 \
  --num-workers 8 \
  --siamese-batch-size 64 \
  --cnn-batch-size 128
