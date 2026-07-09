"""Helpers for the Past Analysis Results Streamlit tab."""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd

from otitenet.app.utils import ensure_int, extract_params_from_log_path


PAST_RESULTS_COLUMNS = [
    "Id",
    "Model_ID",
    "Filename",
    "Model Name",
    "Task",
    "Confidence",
    "Timestamp",
    "Pred_Label",
    "Log Path",
    "NSize",
    "FGSM",
    "Normalize",
    "N_Calibration",
    "Classif_Loss",
    "DLoss",
    "Dist_Fct",
    "Prototypes",
    "NPos",
    "NNeg",
    "N_Neighbors",
]


def past_results_dataframe(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=PAST_RESULTS_COLUMNS)


def latest_results_for_image(results_df: pd.DataFrame, filename: str) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame(columns=PAST_RESULTS_COLUMNS)
    image_results_full = results_df[results_df["Filename"] == filename].copy().reset_index(drop=True)
    image_results_sorted = image_results_full.sort_values("Timestamp", ascending=False)
    return image_results_sorted.drop_duplicates(subset=["Model_ID"], keep="first").reset_index(drop=True)


def args_from_model_row(base_args: Any, row_dict: dict[str, Any]):
    """Build analysis args from a best-model/result table row."""
    local_args = argparse.Namespace(**vars(base_args))
    local_args.model_name = row_dict.get("Model Name", base_args.model_name)
    local_args.new_size = ensure_int(row_dict.get("NSize", base_args.new_size))
    local_args.fgsm = row_dict.get("FGSM", base_args.fgsm)
    local_args.normalize = row_dict.get("Normalize", base_args.normalize)
    local_args.n_calibration = row_dict.get("N_Calibration", base_args.n_calibration)
    local_args.classif_loss = row_dict.get("Classif_Loss", base_args.classif_loss)
    local_args.dloss = row_dict.get("DLoss", base_args.dloss)
    local_args.dist_fct = row_dict.get("Dist_Fct", base_args.dist_fct)
    local_args.prototypes_to_use = row_dict.get("Prototypes", base_args.prototypes_to_use)
    local_args.n_positives = ensure_int(row_dict.get("NPos", base_args.n_positives))
    local_args.n_negatives = ensure_int(row_dict.get("NNeg", base_args.n_negatives))
    local_args.n_neighbors = ensure_int(row_dict.get("N_Neighbors", base_args.n_neighbors))
    local_args.model_id = row_dict.get("Model ID") or row_dict.get("Model_ID") or getattr(base_args, "model_id", None)
    return local_args


def args_from_result_row(
    base_args: Any,
    row_dict: dict[str, Any],
    log_path: str | None = None,
    data_dir: str = "data",
    device_default: str = "cpu",
):
    """Build recomputation args from a selected historical result row."""
    local_args = args_from_model_row(base_args, row_dict)
    local_args.task = row_dict.get("Task", getattr(base_args, "task", None))
    local_args.model_id = row_dict.get("Model ID") or row_dict.get("Model_ID") or getattr(base_args, "model_id", None)

    if not hasattr(local_args, "auto_select_k"):
        local_args.auto_select_k = 0
    if not hasattr(local_args, "random_recs"):
        local_args.random_recs = 0
    if not hasattr(local_args, "seed"):
        local_args.seed = 1
    if not hasattr(local_args, "groupkfold"):
        local_args.groupkfold = 1
    if not hasattr(local_args, "valid_dataset"):
        local_args.valid_dataset = ""
    if not hasattr(local_args, "device"):
        local_args.device = device_default

    dataset = dataset_from_log_path(log_path or row_dict.get("Log Path"))
    if dataset:
        local_args.path = os.path.join(data_dir, dataset)

    return local_args


def dataset_from_log_path(log_path: str | None) -> str | None:
    """Extract the dataset segment from a best-model log path."""
    p_extra = extract_params_from_log_path(log_path)
    if p_extra.get("Dataset"):
        return p_extra["Dataset"]
    if not log_path:
        return None
    parts = str(log_path).strip("/").split("/")
    try:
        base_idx = parts.index("best_models")
    except ValueError:
        return None
    dataset_idx = base_idx + 3
    if len(parts) > dataset_idx and not parts[dataset_idx].startswith("nsize"):
        return parts[dataset_idx]
    return None


def query_image_path(filename: str, query_dir: str = "data/queries") -> str:
    """Return the query image path used by past-results rerun actions."""
    return os.path.join(query_dir, str(filename).split("/")[-1])
