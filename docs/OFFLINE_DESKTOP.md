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

The offline app reads:

```text
data/mobile_deployments/current/
├── manifest.json
└── model.onnx
```

Production runtime:

- `model_type: "onnx_classifier"`
- `runtime: "onnxruntime"`
- `quantization: "dynamic_uint8"`
- `files.model: "model.onnx"`

The Python offline code still has a PyTorch fallback for development, but the desktop build is intended to package only ONNX Runtime. Do not build the production executable from a deployment manifest that still points to `model.pth`.

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

Then export the current production deployment to ONNX:

```bash
/home/simon/otitenet/.conda/bin/python -m pip install -r requirements-export.txt
/home/simon/otitenet/.conda/bin/python scripts/export_offline_onnx_model.py
```

The export script:

- reads `data/mobile_deployments/current/manifest.json`
- exports the deployed PyTorch state dict to ONNX
- dynamically quantizes the ONNX weights to UINT8 by default
- runs a one-sample ONNX Runtime check against the PyTorch model
- backs up the old manifest to `manifest.pytorch.json`
- rewrites `manifest.json` to use `onnx_classifier`
- removes the old PyTorch model unless `--keep-pytorch` is passed

Use `--no-quantize` if you need a full-precision ONNX file for debugging. Use `--keep-float` if you want to keep the intermediate full-precision ONNX file next to the quantized production file.

Use `--skip-check` only when ONNX Runtime is not available in the export environment.

## Build the Streamlit Sidecar

Install the standalone desktop dependencies:

```bash
/home/simon/otitenet/.conda/bin/python -m pip install -r requirements-desktop.txt
```

`requirements-desktop.txt` is intentionally not based on `requirements.txt`. It excludes training, analysis, test, experiment tracking, and plotting packages such as scikit-optimize, matplotlib, seaborn, tensorboard, comet-ml, pytest, and TensorFlow.

It also excludes PyTorch and torchvision. The offline executable should use `onnxruntime` with `data/mobile_deployments/current/model.onnx`.

Build:

```bash
/home/simon/otitenet/.conda/bin/python -m PyInstaller packaging/pyinstaller/otitenet_streamlit.spec --clean -y
```

Smoke test on a free port:

```bash
OTITENET_STREAMLIT_PORT=8502 ./dist/otitenet-streamlit/otitenet-streamlit
```

Then open `http://127.0.0.1:8502`.

## Copy the Sidecar into Tauri

```bash
HOST_TRIPLET="$(rustc -Vv | sed -n 's/^host: //p')"
mkdir -p desktop/src-tauri/binaries
cp dist/otitenet-streamlit/otitenet-streamlit "desktop/src-tauri/binaries/otitenet-streamlit-${HOST_TRIPLET}"
chmod +x "desktop/src-tauri/binaries/otitenet-streamlit-${HOST_TRIPLET}"
```

On Windows, copy `dist/otitenet-streamlit/otitenet-streamlit.exe` to `desktop/src-tauri/binaries/otitenet-streamlit-x86_64-pc-windows-msvc.exe`.

## Build the Desktop App

```bash
cd desktop
npm install
npm run tauri:build
```

The platform installer/app bundle will be under `desktop/src-tauri/target/release/bundle/`.
