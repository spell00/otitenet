#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OTITENET_DEBUGPY_HOST="${OTITENET_DEBUGPY_HOST:-127.0.0.1}"
export OTITENET_DEBUGPY_PORT="${OTITENET_DEBUGPY_PORT:-5679}"

exec ./.conda/bin/python -m debugpy \
  --listen "${OTITENET_DEBUGPY_HOST}:${OTITENET_DEBUGPY_PORT}" \
  --wait-for-client \
  -m streamlit run app.py \
  --server.address 127.0.0.1 \
  --server.port "${STREAMLIT_PORT:-8502}"
