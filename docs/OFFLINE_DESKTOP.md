# Offline Desktop Build

The offline desktop app is separate from the online Streamlit app.

- Online app: `app.py`, with admin or client behavior depending on who logs in.
- Offline app: `app_offline.py`, the minimal client-shaped app with only `New Analysis` and `Historics`.

The offline app intentionally does not import the full online app. It reuses the same deployment/model runtime concepts where practical, but avoids MySQL, login, admin controls, leaderboards, training logs, and all learned model history. It only runs the current production deployment in `data/mobile_deployments/current` and stores history locally on the device.

The intended workflow is:

1. Use the online app to choose the production model.
2. Run the deployment export step so `data/mobile_deployments/current` contains `manifest.json` plus the production model and any required reference arrays.
3. Build the offline desktop app. PyInstaller bundles the offline app code and `data/mobile_deployments/current`.

## Deployment Payload

The offline app reads the deployment files referenced by:

```text
data/mobile_deployments/current/
├── manifest.json
├── embedding_model.onnx
└── prototypes.npz
```

Compact production runtime:

- `model_type: "onnx_embedding_prototype"`
- `runtime: "onnxruntime"`
- `head_type: "prototype"`
- `files.model: "embedding_model.onnx"`
- `files.prototypes: "prototypes.npz"`

The Python offline code still has a PyTorch fallback for exact/full builds, but the compact desktop build is intended to package only ONNX Runtime. Do not build the compact executable from a deployment manifest that still points to `model.pth`.

Future `knn_embedding` payloads can be added under the same manifest contract by including the embedding model plus reference embeddings/labels.

The offline app does not copy all of `data/`, `logs/`, `best_models_old/`, or `output/`.

## Create or Refresh the Current Deployment

From the online app, click the production-model action first. Then run:

```bash
/home/simon/otitenet/.conda/bin/python scripts/create_mobile_deployment.py
```

You can also pass a model file explicitly:

```bash
/home/simon/otitenet/.conda/bin/python scripts/create_mobile_deployment.py \
  --model-file path/to/model.pth \
  --model-id 123 \
  --model-name resnet18
```

For compact packaging, export the current production embedding model to ONNX while preserving the prototype head:

```bash
/home/simon/otitenet/.conda/bin/python -m pip install -r requirements-export.txt
/home/simon/otitenet/.conda/bin/python scripts/export_offline_onnx_model.py --embedding-output --no-quantize --keep-pytorch
```

The export script:

- reads `data/mobile_deployments/current/manifest.json`
- exports the deployed PyTorch state dict to ONNX
- dynamically quantizes the ONNX weights to UINT8 by default
- runs a one-sample ONNX Runtime check against the PyTorch model
- backs up the old manifest to `manifest.pytorch.json`
- rewrites `manifest.json` to use `onnx_embedding_prototype` when the production model uses a prototype head
- removes the old PyTorch model unless `--keep-pytorch` is passed

Use `--no-quantize` for the closest compact predictions. Dynamic quantization is smaller but may move scores more.

Use `--skip-check` only when ONNX Runtime is not available in the export environment.

## Build the Streamlit Sidecar

Install the standalone desktop dependencies:

```bash
/home/simon/otitenet/.conda/bin/python -m pip install -r requirements-desktop.txt
/home/simon/otitenet/.conda/bin/python -m pip install -r requirements-export.txt
```

`requirements-desktop.txt` is intentionally not based on `requirements.txt`. It excludes most training, analysis, test, experiment tracking, and plotting packages such as scikit-optimize, matplotlib, seaborn, tensorboard, comet-ml, pytest, and TensorFlow.

The compact PyInstaller spec excludes PyTorch and torchvision from the packaged app. The exact build includes them and is much larger.

All desktop build scripts run `scripts/ensure_desktop_python.sh` before doing work. The required interpreter is:

```text
/home/simon/otitenet/.conda/bin/python
```

The scripts fail if `PYTHON` or plain `python` resolves to another environment, for example the base conda Python 3.12 or an old Python 3.10 environment. Verify the build interpreter before long runs:

```bash
/home/simon/otitenet/.conda/bin/python --version
scripts/ensure_desktop_python.sh sidecar compact
```

Build compact, recommended:

```bash
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:prepare:compact
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:sidecar:compact
```

This preserves a compact sidecar at:

```text
dist/otitenet-streamlit-compact/
```

From `desktop/`, run the equivalent command:

```bash
PYTHON=/home/simon/otitenet/.conda/bin/python npm run prepare:compact
PYTHON=/home/simon/otitenet/.conda/bin/python npm run sidecar:build:compact
```

Build exact/full:

```bash
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:prepare:exact
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:sidecar:exact
```

This preserves an exact sidecar at:

```text
dist/otitenet-streamlit-exact/
```

Each sidecar build also updates the active Tauri sidecar under `desktop/src-tauri/binaries/`. The last sidecar variant you build is the variant that `npm run tauri:build` packages. The `dist/otitenet-streamlit-compact/` and `dist/otitenet-streamlit-exact/` folders remain available for direct side-by-side testing.

Smoke test on a free port:

```bash
OTITENET_STREAMLIT_PORT=8502 ./dist/otitenet-streamlit/otitenet-streamlit
```

Then open `http://127.0.0.1:8502`.

Compare compact and exact at the same time:

```bash
OTITENET_STREAMLIT_PORT=8502 ./dist/otitenet-streamlit-compact/otitenet-streamlit
OTITENET_STREAMLIT_PORT=8503 ./dist/otitenet-streamlit-exact/otitenet-streamlit
```

## Copy the Sidecar into Tauri

```bash
npm run desktop:sidecar:compact
```

On Windows, copy `dist/otitenet-streamlit/otitenet-streamlit.exe` to `desktop/src-tauri/binaries/otitenet-streamlit-x86_64-pc-windows-msvc.exe`.

## Build the Desktop App

```bash
cd desktop
npm install
npm run tauri:build
```

To produce variant-suffixed `.deb` installers from the repository root:

```bash
npm run desktop:tauri:compact
npm run desktop:tauri:exact
```

Those commands write:

```text
desktop/src-tauri/target/release/bundle/deb/Otitenet_0.3.0_amd64_compact.deb
desktop/src-tauri/target/release/bundle/deb/Otitenet_0.3.0_amd64_exact.deb
```

From `desktop/`, the equivalent commands are:

```bash
npm run tauri:build:compact
npm run tauri:build:exact
```

Each variant build also refreshes Tauri's normal `.deb` output, such as `Otitenet_0.3.0_amd64.deb`. The suffixed files are copied from that output and preserved so both installers can coexist.

The platform installer/app bundle will be under `desktop/src-tauri/target/release/bundle/`.
