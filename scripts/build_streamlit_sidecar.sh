#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${OTITENET_DESKTOP_VARIANT:-${1:-compact}}"
if [[ "$VARIANT" != "compact" && "$VARIANT" != "exact" && "$VARIANT" != "full" && "$VARIANT" != "torch" ]]; then
  echo "Unknown desktop sidecar variant: ${VARIANT}" >&2
  echo "Use compact or exact." >&2
  exit 1
fi

if [[ "$VARIANT" == "full" || "$VARIANT" == "torch" ]]; then
  VARIANT="exact"
fi

export OTITENET_DESKTOP_VARIANT="$VARIANT"

cd "$ROOT_DIR"

"$ROOT_DIR/scripts/ensure_desktop_python.sh" sidecar "$VARIANT"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.conda/bin/python}"

echo "Building OtiteNet Streamlit sidecar variant: ${VARIANT}"
"$PYTHON_BIN" -m PyInstaller packaging/pyinstaller/otitenet_streamlit.spec --clean -y

DIST_DIR="dist/otitenet-streamlit"
SIDECAR_EXE="${DIST_DIR}/otitenet-streamlit"
VARIANT_DIST_DIR="dist/otitenet-streamlit-${VARIANT}"

if [[ ! -x "$SIDECAR_EXE" ]]; then
  echo "Expected PyInstaller output at ${SIDECAR_EXE}" >&2
  exit 1
fi

rm -rf "$VARIANT_DIST_DIR"
cp -R "$DIST_DIR" "$VARIANT_DIST_DIR"
echo "Saved ${VARIANT} sidecar at ${VARIANT_DIST_DIR}/otitenet-streamlit"

if command -v rustc >/dev/null 2>&1; then
  HOST_TRIPLET="$(rustc -Vv | sed -n 's/^host: //p')"
  if [[ -n "$HOST_TRIPLET" ]]; then
    mkdir -p desktop/src-tauri/binaries
    ACTIVE_RUNTIME_DIR="desktop/src-tauri/binaries/otitenet-streamlit-runtime"
    rm -rf "$ACTIVE_RUNTIME_DIR"
    cp -R "$DIST_DIR" "$ACTIVE_RUNTIME_DIR"

    cat > "desktop/src-tauri/binaries/otitenet-streamlit-${HOST_TRIPLET}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for candidate in \
  "${SELF_DIR}/otitenet-streamlit-runtime/otitenet-streamlit" \
  "${SELF_DIR}/../lib/Otitenet/binaries/otitenet-streamlit-runtime/otitenet-streamlit" \
  "/usr/lib/Otitenet/binaries/otitenet-streamlit-runtime/otitenet-streamlit"
do
  if [[ -x "$candidate" ]]; then
    exec "$candidate" "$@"
  fi
done

echo "Could not find otitenet-streamlit runtime next to the Tauri sidecar." >&2
exit 127
EOF
    chmod +x "desktop/src-tauri/binaries/otitenet-streamlit-${HOST_TRIPLET}"

    VARIANT_RUNTIME_DIR="desktop/src-tauri/binaries/otitenet-streamlit-runtime-${VARIANT}"
    rm -rf "$VARIANT_RUNTIME_DIR"
    cp -R "$DIST_DIR" "$VARIANT_RUNTIME_DIR"

    echo "Updated desktop/src-tauri/binaries/otitenet-streamlit-${HOST_TRIPLET}"
    echo "Updated desktop/src-tauri/binaries/otitenet-streamlit-runtime"
    echo "Saved desktop/src-tauri/binaries/otitenet-streamlit-runtime-${VARIANT}"
    echo "Active sidecar variant: ${VARIANT}"
  fi
else
  echo "Built ${SIDECAR_EXE}. Install Rust to auto-copy it into desktop/src-tauri/binaries."
fi
