#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-sidecar}"
VARIANT="${2:-compact}"
EXPECTED_PYTHON="${OTITENET_DESKTOP_PYTHON:-$ROOT_DIR/.conda/bin/python}"
PYTHON_BIN="${PYTHON:-$EXPECTED_PYTHON}"

if [[ "$PYTHON_BIN" != */* ]]; then
  RESOLVED_PYTHON="$(command -v "$PYTHON_BIN" || true)"
  if [[ -n "$RESOLVED_PYTHON" ]]; then
    PYTHON_BIN="$RESOLVED_PYTHON"
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Desktop build Python is not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON=$EXPECTED_PYTHON or install the project .conda environment." >&2
  exit 1
fi

PYTHON_INFO="$("$PYTHON_BIN" - <<'PY'
import json
import os
import sys

print(json.dumps({
    "executable": os.path.realpath(sys.executable),
    "prefix": os.path.realpath(sys.prefix),
    "version": list(sys.version_info[:3]),
}))
PY
)"

EXPECTED_REAL="$(realpath "$EXPECTED_PYTHON" 2>/dev/null || true)"

"$PYTHON_BIN" - "$PYTHON_INFO" "$EXPECTED_REAL" "$MODE" "$VARIANT" <<'PY'
import importlib
import json
import os
import sys

info = json.loads(sys.argv[1])
expected = sys.argv[2]
mode = sys.argv[3]
variant = sys.argv[4]

errors = []
version = tuple(info["version"])
if version[:2] != (3, 11):
    errors.append(
        f"expected Python 3.11.x, got {version[0]}.{version[1]}.{version[2]}"
    )

allow_foreign = os.environ.get("OTITENET_ALLOW_NONPROJECT_PYTHON") == "1"
if expected and info["executable"] != expected and not allow_foreign:
    errors.append(
        "expected project Python at "
        f"{expected}, got {info['executable']}. "
        "Set PYTHON to the project interpreter, or set "
        "OTITENET_ALLOW_NONPROJECT_PYTHON=1 only for deliberate debugging."
    )

required = {
    "numpy": "numpy",
    "PIL": "Pillow",
    "streamlit": "streamlit",
    "onnxruntime": "onnxruntime",
    "joblib": "joblib",
    "sklearn": "scikit-learn",
}

if mode in {"sidecar", "tauri"}:
    required["PyInstaller"] = "pyinstaller"

if mode in {"prepare", "tauri"} or variant in {"exact", "full", "torch"}:
    required["torch"] = "torch"
    required["torchvision"] = "torchvision"

if mode in {"prepare", "tauri"} and variant == "compact":
    required["onnx"] = "onnx"

missing = []
for module, package in required.items():
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{package} ({module}: {exc})")

if missing:
    errors.append(
        "missing required desktop build packages: "
        + ", ".join(missing)
        + ". Install with this interpreter, for example: "
        + f"{info['executable']} -m pip install -r requirements-desktop.txt "
        + "and, for compact export, "
        + f"{info['executable']} -m pip install -r requirements-export.txt"
    )

if errors:
    print("Desktop build Python preflight failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    sys.exit(1)

print(
    "Desktop build Python OK: "
    f"{info['executable']} "
    f"({version[0]}.{version[1]}.{version[2]}), "
    f"mode={mode}, variant={variant}"
)
PY

export PYTHON="$PYTHON_BIN"
