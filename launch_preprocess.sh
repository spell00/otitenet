#!/usr/bin/env bash

set -euo pipefail

# Easy-to-change config block.
PREPROCESS_CONFIG="configs/preprocessing_config.json"
if [ ! -f "$PREPROCESS_CONFIG" ] && [ -f "data/datasets/preprocessing_config.json" ]; then
    PREPROCESS_CONFIG="data/datasets/preprocessing_config.json"
fi
VALID_DATASET="Banque_Viscaino_Chili_2020"
# Use the same dataset as validation to force deterministic fallback split:
# valid stays valid, and test is sampled from the same dataset instead of GMF.
TEST_DATASET="Banque_Viscaino_Chili_2020"
# Optional explicit training datasets (comma-separated). Empty = auto.
TRAIN_DATASETS=""

resolve_dataset_name_from_config() {
    local config_path="$1"
    python - "$config_path" <<'PY'
import json
import os
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

output_base = str(cfg.get("output_base", "data/otite_ds"))
image_size = int(cfg.get("image_size", 64))
base_name = os.path.basename(output_base.rstrip("/"))
print(f"{base_name}_{image_size}")
PY
}

echo "Preprocessing dataset using config: $PREPROCESS_CONFIG"
python scripts/preprocessing/build_dataset.py --config "$PREPROCESS_CONFIG"

dataset_name="$(resolve_dataset_name_from_config "$PREPROCESS_CONFIG")"
echo "Resolved dataset for training: $dataset_name"

launch_args=(
    --dataset="$dataset_name"
    --valid-dataset="$VALID_DATASET"
    --test-dataset="$TEST_DATASET"
)

if [ -n "$TRAIN_DATASETS" ]; then
    launch_args+=(--train-datasets="$TRAIN_DATASETS")
fi

# Forward any extra flags to launch.sh (for example: --test, --jobs=4, --force).
if [ "$#" -gt 0 ]; then
    launch_args+=("$@")
fi

