# Otitenet

Otitenet is a framework for otitis classification with domain-generalized image embeddings, prototype/KNN heads, SHAP/Grad-CAM interpretability, a Streamlit web app, an offline desktop app, and an Android client.

## Quick Start

### 1. Install Python Dependencies

Use the project Python environment. The desktop build scripts expect:

```text
/home/simon/otitenet/.conda/bin/python
```

For the research/web environment:

```bash
python -m pip install -r requirements.txt
python setup.py
```

For offline desktop packaging:

```bash
/home/simon/otitenet/.conda/bin/python -m pip install -r requirements-desktop.txt
/home/simon/otitenet/.conda/bin/python -m pip install -r requirements-export.txt
```

### DVC Workflow (Recommended)

This repo is initialized with DVC to version large datasets and artifacts outside Git history.

Current default remote:

```text
localstorage -> /home/simon/dvc-storage/otitenet
```

Initial tracked paths (practical baseline):

```bash
/home/simon/otitenet/.conda/bin/dvc add -f data/datasets logs/progresses
/home/simon/otitenet/.conda/bin/dvc push
git add data.dvc logs/progresses.dvc .dvc/config .gitignore
```

Daily commands:

```bash
# pull data/artifacts referenced by .dvc files
/home/simon/otitenet/.conda/bin/dvc pull

# if .dvc files changed, restore workspace state
/home/simon/otitenet/.conda/bin/dvc checkout

# after updating tracked data/artifacts
/home/simon/otitenet/.conda/bin/dvc add -f data/datasets logs/progresses
/home/simon/otitenet/.conda/bin/dvc push
git add data.dvc logs/progresses.dvc
```

To switch from local storage to shared/on-prem/cloud storage later:

```bash
/home/simon/otitenet/.conda/bin/dvc remote add -d <remote_name> <remote_url_or_path>
```

### 2. Initialize MySQL

The online app stores users, model selections, and analysis results in MySQL.

```sql
CREATE DATABASE results_db;
GRANT ALL PRIVILEGES ON results_db.* TO 'y_user'@'%' IDENTIFIED BY 'password';
FLUSH PRIVILEGES;
```

```bash
python scripts/utils/init_db.py
```

The offline desktop app does not require MySQL. It stores users and history locally under `~/.otitenet/offline/`.

### 3. Build the Dataset

Raw source datasets go under:

```text
data/datasets/<dataset_name>/
```

Build the processed dataset from the config:

```bash
python scripts/preprocessing/build_dataset.py --config configs/preprocessing_config.json
```

Typical output:

```text
data/otite_ds_64/USA_Turquie_Chili_GMFUNL/
```

The preprocessing config controls included/excluded datasets, output prefix, and image size.
The output subfolder is derived from `include_datasets`, for example `USA_Turquie_Chili_GMFUNL`.
Preprocessing preserves the original dataset class in `raw_label` and writes the canonical class in `label`.

### 4. Train a Model

Recommended deterministic split:

```bash
python -m otitenet.train.train_triplet_new \
  --path=./data/otite_ds_64/USA_Turquie_Chili_GMFUNL \
  --groupkfold=1 \
  --valid_dataset=Banque_Viscaino_Chili_2020 \
  --test_dataset=GMFUNL_jan2023 \
  --train_datasets=Banque_Comert_Turquie_2020_jpg,Banque_Calaman_USA_2020_trie_CM,GMFUNL_jan2023 \
  --task=otite_four_class
```

Use `--task=notNormal` for the old binary `Normal` / `NotNormal` labels.
Use `--task=otite_four_class` for `Normal` / `NotNormal` / `Wax` / `Tube`.

Check the logs for `[SplitDebug]` lines confirming the validation and test datasets. If `--test_dataset` is missing from the processed dataset, the loader falls back by splitting the validation dataset.

Batch launchers:

```bash
./launch.sh --test
./launch.sh --jobs=4
./launch_optimize.sh
```

### 5. Run the Online App

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

Use the online app to select the production model before building offline/mobile deployments.

### 6. Export the Current Deployment

Create or refresh the deployment used by Android and the offline desktop app:

```bash
/home/simon/otitenet/.conda/bin/python scripts/create_mobile_deployment.py --task otite_four_class
```

For the compact offline desktop runtime, export the current production embedding model to ONNX:

```bash
/home/simon/otitenet/.conda/bin/python scripts/export_offline_onnx_model.py --embedding-output --no-quantize --keep-pytorch
```

The active deployment lives in:

```text
data/mobile_deployments/current/
```

### 7. Build the Offline Desktop App

Compact build, recommended:

```bash
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:prepare:compact
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:sidecar:compact
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:tauri:compact
```

Exact/full PyTorch build:

```bash
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:prepare:exact
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:sidecar:exact
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:tauri:exact
```

Installer outputs:

```text
desktop/src-tauri/target/release/bundle/deb/Otitenet_0.3.0_amd64_compact.deb
desktop/src-tauri/target/release/bundle/deb/Otitenet_0.3.0_amd64_exact.deb
```

Detailed desktop packaging notes are in [docs/OFFLINE_DESKTOP.md](docs/OFFLINE_DESKTOP.md).

### 8. Generate Analysis Outputs

```bash
python scripts/analysis/generate_paper_analysis.py
```

Outputs are written under:

```text
output/analysis/
```

## Notes

- GitHub Actions builds the Dockerfile and runs `bash scripts/test/unit.sh` plus `bash scripts/test/smoketest.sh` inside that image.
- The app can keep a separate production model for each labeling scenario.
- `best_models_registry.mcc` is the training-time best validation MCC.
- The Streamlit learned-embedding section computes post-hoc validation scores with downstream classifiers. These scores are expected to differ from the registry MCC.
- When split manifests exist in a selected model directory, the app reuses them so train/valid/test membership matches the original run.
- `requirements-desktop.txt` is intentionally separate from `requirements.txt`; compact desktop builds avoid bundling PyTorch.

## Project Structure

```text
app.py                         Online Streamlit app
app_offline.py                 Offline desktop Streamlit app
src/otitenet/                  Main Python package
scripts/preprocessing/         Dataset build scripts
scripts/analysis/              Paper/analysis generation scripts
scripts/migrations/            Database migrations
scripts/utils/                 Database and maintenance utilities
desktop/                       Tauri desktop wrapper
app/                           Android app
docs/                          Detailed workflow documentation
output/                        Generated reports and artifacts
logs/                          Training logs and completion markers
```

## More Documentation

- [Offline desktop build](docs/OFFLINE_DESKTOP.md)
- [Example commands](docs/EXAMPLE_COMMANDS.md)
- [Parameters and classifiers](docs/PARAMETERS_AND_CLASSIFIERS.md)
- [Technical summary](docs/TECHNICAL_SUMMARY.md)
