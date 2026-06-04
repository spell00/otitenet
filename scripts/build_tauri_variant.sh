#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-compact}"

bash -n \
  "$ROOT_DIR/scripts/ensure_desktop_python.sh" \
  "$ROOT_DIR/scripts/prepare_desktop_variant.sh" \
  "$ROOT_DIR/scripts/build_streamlit_sidecar.sh" \
  "$ROOT_DIR/scripts/build_tauri_variant.sh"

if [[ "$VARIANT" != "compact" && "$VARIANT" != "exact" && "$VARIANT" != "full" && "$VARIANT" != "torch" ]]; then
  echo "Unknown Tauri build variant: ${VARIANT}" >&2
  echo "Use compact or exact." >&2
  exit 1
fi

if [[ "$VARIANT" == "full" || "$VARIANT" == "torch" ]]; then
  VARIANT="exact"
fi

cd "$ROOT_DIR"

"$ROOT_DIR/scripts/ensure_desktop_python.sh" tauri "$VARIANT"

npm run "desktop:prepare:${VARIANT}"
npm run "desktop:sidecar:${VARIANT}"

cd "$ROOT_DIR/desktop"
npm run tauri:build -- --bundles deb

DEB_DIR="$ROOT_DIR/desktop/src-tauri/target/release/bundle/deb"
mapfile -t DEB_CANDIDATES < <(
  find "$DEB_DIR" -maxdepth 1 -type f -name 'Otitenet_*_amd64.deb' \
    ! -name '*_compact.deb' \
    ! -name '*_exact.deb' \
    -printf '%T@ %p\n' \
    | sort -nr
)

DEB_PATH=""
if [[ "${#DEB_CANDIDATES[@]}" -gt 0 ]]; then
  DEB_PATH="${DEB_CANDIDATES[0]#* }"
fi

if [[ -z "$DEB_PATH" || ! -f "$DEB_PATH" ]]; then
  echo "Could not find the Tauri .deb output in ${DEB_DIR}" >&2
  exit 1
fi

VARIANT_DEB="${DEB_PATH%.deb}_${VARIANT}.deb"
cp -f "$DEB_PATH" "$VARIANT_DEB"

echo "Built active Tauri .deb: ${DEB_PATH}"
echo "Saved ${VARIANT} Tauri .deb: ${VARIANT_DEB}"
