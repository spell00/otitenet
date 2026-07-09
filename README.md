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
/home/simon/otitenet/.conda/bin/dvc add -f data logs/progresses logs/best_models
/home/simon/otitenet/.conda/bin/dvc push
```

Daily commands:

```bash
# pull data/artifacts referenced by .dvc files
/home/simon/otitenet/.conda/bin/dvc pull

# if .dvc files changed, restore workspace state
/home/simon/otitenet/.conda/bin/dvc checkout

# after updating tracked data/artifacts
/home/simon/otitenet/.conda/bin/dvc add -f data logs/progresses logs/best_models
/home/simon/otitenet/.conda/bin/dvc push
git add data.dvc logs/progresses.dvc logs/best_models.dvc configs/datasets.csv configs/best_models.csv
```

To switch from local storage to shared/on-prem/cloud storage later:

```bash
/home/simon/otitenet/.conda/bin/dvc remote add -d <remote_name> <remote_url_or_path>
```

### DVCLive Tracking

Training uses DVCLive by default and does not import Comet unless `--log_comet 1` is passed.
Each training run writes DVCLive metrics and params under:

```text
logs/<task>/<run_uuid>/dvclive/
```

The DVCLive params include:

- CLI args and optimized hyperparameters
- DVC pointer hashes for tracked data/model registry outputs
- Git commit, branch, dirty status, Python/platform, and `requirements.txt` hash
- run paths and split/dataset settings

Useful flags:

```bash
--log_dvclive 1              # default
--dvclive_save_dvc_exp 1     # default
--dvclive_monitor_system 1   # default
--log_comet 0                # default
```

### Artifact Registries and Database

The app should treat the database as the canonical index for production and best-model selection. The CSV files under `configs/` are portable snapshots that can be reviewed, committed, rebuilt, and cited in a paper:

- `configs/datasets.csv` records raw/processed datasets, ignored raw folders, pixel size, source combination, inference inclusion, sample count, and DVC target path.
- `configs/best_models.csv` records each best-model artifact, its metrics/parameters, the mirrored best-model directory, and the original source run path when available.
- `best_models_registry` in SQL stores the same model pointers for the app, including `artifact_id`, `best_model_dir`, and `source_run_log_path`.

`logs/best_models/` is kept as a convenient mirror and emergency fallback, but app loading should prefer `source_run_log_path` when that directory still contains `model.pth` and `prototypes.pkl`. This avoids making the Streamlit app depend on a fragile parameter-encoded directory tree while preserving a human-readable backup location.

Rebuild and sync the registries after adding datasets or promoting models:

```bash
python scripts/build_artifact_registries.py
python scripts/sync_best_model_artifacts_to_db.py
```

For manuscript methods/reproducibility, report the dataset registry row, DVC pointer hash, DVCLive run directory, Git commit/dirty status, and best-model registry row or `artifact_id` used for each result.

### 2. Initialize MySQL

The online app stores users, model selections, and analysis results in MySQL.

```sql
CREATE DATABASE results_db;
CREATE USER IF NOT EXISTS 'y_user'@'localhost' IDENTIFIED BY 'password';
CREATE USER IF NOT EXISTS 'y_user'@'%' IDENTIFIED BY 'password';
ALTER USER 'y_user'@'localhost' IDENTIFIED BY 'password';
ALTER USER 'y_user'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON results_db.* TO 'y_user'@'localhost';
GRANT ALL PRIVILEGES ON results_db.* TO 'y_user'@'%';
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
#cd /home/simon/otitenet/desktop
#PYTHON=/home/simon/otitenet/.conda/bin/python npm run sidecar:build:compact
#PYTHON=/home/simon/otitenet/.conda/bin/python npm run tauri:build:compact
```

Exact/full PyTorch build:

```bash
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:prepare:exact
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:sidecar:exact
PYTHON=/home/simon/otitenet/.conda/bin/python npm run desktop:tauri:exact
```

Installer outputs:

```text
desktop/src-tauri/target/release/bundle/deb/Otitenet_0.7.1_amd64_compact.deb
desktop/src-tauri/target/release/bundle/deb/Otitenet_0.7.1_amd64_exact.deb
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
