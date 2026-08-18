#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-compact}"
TARGET="${OTITENET_WINDOWS_TARGET:-x86_64-pc-windows-msvc}"
BUNDLE_KIND="${OTITENET_WINDOWS_BUNDLE:-nsis}"

if [[ "$VARIANT" != "compact" && "$VARIANT" != "exact" && "$VARIANT" != "full" && "$VARIANT" != "torch" ]]; then
  echo "Unknown Windows Tauri build variant: ${VARIANT}" >&2
  echo "Use compact or exact." >&2
  exit 1
fi
if [[ "$VARIANT" == "full" || "$VARIANT" == "torch" ]]; then
  VARIANT="exact"
fi

cd "$ROOT_DIR"

bash -n   "$ROOT_DIR/scripts/ensure_desktop_python.sh"   "$ROOT_DIR/scripts/prepare_desktop_variant.sh"   "$ROOT_DIR/scripts/build_tauri_windows.sh"

for cmd in npm rustup cargo makensis; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command for Windows Tauri cross-build: $cmd" >&2
    echo "Install NSIS/LLVM/cargo-xwin as described in Tauri's Windows cross-compilation docs." >&2
    exit 1
  fi
done
if ! cargo xwin --version >/dev/null 2>&1; then
  echo "Missing cargo-xwin. Install with: cargo install --locked cargo-xwin" >&2
  exit 1
fi

rustup target add "$TARGET"

npm run "desktop:prepare:${VARIANT}"

# The Tauri shell can be cross-compiled on Linux, but the Python/Streamlit
# sidecar must be a Windows executable. Build it on Windows/CI with:
#   set OTITENET_PYINSTALLER_ONEFILE=1
#   python -m PyInstaller packaging/pyinstaller/otitenet_streamlit.spec --clean -y
# Then copy the produced otitenet-streamlit.exe here.
SIDECAR_EXE="${OTITENET_WINDOWS_SIDECAR_EXE:-}"
if [[ -z "$SIDECAR_EXE" ]]; then
  for candidate in     "$ROOT_DIR/dist/otitenet-streamlit-windows-${VARIANT}/otitenet-streamlit.exe"     "$ROOT_DIR/dist/otitenet-streamlit-windows/otitenet-streamlit.exe"     "$ROOT_DIR/dist/otitenet-streamlit/otitenet-streamlit.exe"
  do
    if [[ -f "$candidate" ]]; then
      SIDECAR_EXE="$candidate"
      break
    fi
  done
fi

if [[ -z "$SIDECAR_EXE" || ! -f "$SIDECAR_EXE" ]]; then
  cat >&2 <<EOF
Missing Windows Streamlit sidecar executable.
Expected one of:
  dist/otitenet-streamlit-windows-${VARIANT}/otitenet-streamlit.exe
  dist/otitenet-streamlit-windows/otitenet-streamlit.exe
or set:
  OTITENET_WINDOWS_SIDECAR_EXE=/path/to/otitenet-streamlit.exe

PyInstaller cannot reliably produce a Windows executable from Linux. Build the
sidecar on Windows or CI with OTITENET_PYINSTALLER_ONEFILE=1, then rerun this script.
EOF
  exit 1
fi

mkdir -p "$ROOT_DIR/desktop/src-tauri/binaries"
cp -f "$SIDECAR_EXE" "$ROOT_DIR/desktop/src-tauri/binaries/otitenet-streamlit-${TARGET}.exe"

echo "Using Windows sidecar: $SIDECAR_EXE"
echo "Copied sidecar to desktop/src-tauri/binaries/otitenet-streamlit-${TARGET}.exe"

cd "$ROOT_DIR/desktop"
npm run tauri:build -- --runner cargo-xwin --target "$TARGET" --bundles "$BUNDLE_KIND"

BUNDLE_DIR="$ROOT_DIR/desktop/src-tauri/target/${TARGET}/release/bundle/${BUNDLE_KIND}"
echo "Windows Tauri bundle output: $BUNDLE_DIR"
find "$BUNDLE_DIR" -maxdepth 1 -type f -print 2>/dev/null || true
