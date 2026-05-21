#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m unittest discover -s tests -v
