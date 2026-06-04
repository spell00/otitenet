#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-compact}"
DEPLOYMENT_DIR="$ROOT_DIR/data/mobile_deployments/current"
MANIFEST="$DEPLOYMENT_DIR/manifest.json"
PYTORCH_MANIFEST="$DEPLOYMENT_DIR/manifest.pytorch.json"

cd "$ROOT_DIR"

"$ROOT_DIR/scripts/ensure_desktop_python.sh" prepare "$VARIANT"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.conda/bin/python}"

case "$VARIANT" in
  compact)
    "$PYTHON_BIN" scripts/export_offline_onnx_model.py \
      --embedding-output \
      --no-quantize \
      --keep-pytorch \
      --check-tolerance 1e-4
    ;;
  exact|full|torch)
    if [[ ! -f "$PYTORCH_MANIFEST" ]]; then
      echo "Missing exact PyTorch manifest backup: $PYTORCH_MANIFEST" >&2
      echo "Recreate it with scripts/create_mobile_deployment.py --deployment-type torch_prototype ..." >&2
      exit 1
    fi
    cp "$PYTORCH_MANIFEST" "$MANIFEST"
    echo "Restored exact PyTorch prototype deployment manifest: $MANIFEST"
    ;;
  *)
    echo "Unknown desktop variant: $VARIANT" >&2
    echo "Use compact or exact." >&2
    exit 1
    ;;
esac
