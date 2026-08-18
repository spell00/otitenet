#!/usr/bin/env bash
# Queue the inference-only fraction experiment after the active E03 run ends.
set -Eeuo pipefail
ROOT=/home/simon/otitenet
cd "$ROOT"
QUEUE_LOG="$ROOT/paper_outputs/inference_fraction_v1/queue_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$(dirname "$QUEUE_LOG")"
exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "[$(date -Is)] queued inference-fraction V1 experiment"
echo "queue_log=$QUEUE_LOG"
while pgrep -f "otitenet.train.train_triplet_new.*--run_tag PAPER_E03_PREVBEST" >/dev/null; do
  echo "[$(date -Is)] Waiting for active PAPER_E03_PREVBEST run to finish..."
  sleep 60
done
echo "[$(date -Is)] Starting inference-fraction experiment"
"$ROOT/.conda/bin/python" scripts/paper/run_inference_fraction_experiments.py --top-configs 8
LATEST=$(find "$ROOT/paper_outputs/inference_fraction_v1" -maxdepth 1 -mindepth 1 -type d -name "run_*" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d" " -f2-)
"$ROOT/.conda/bin/python" scripts/paper/plot_inference_fraction_performance.py --experiment-root "$LATEST" --metric test_mcc
"$ROOT/.conda/bin/python" scripts/paper/plot_inference_fraction_performance.py --experiment-root "$LATEST" --metric valid_mcc
echo "[$(date -Is)] inference-fraction V1 experiment and figures complete"
