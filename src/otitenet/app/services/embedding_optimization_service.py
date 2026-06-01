# /home/simon/otitenet/otitenet/app/services/embedding_optimization_service.py

from __future__ import annotations

import argparse
import os
from copy import deepcopy

import numpy as np
import pandas as pd


def _ensure_int(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _extract_params_from_log_path(log_path: str) -> dict:
    """
    Defensive fallback parser.

    Your project may already have extract_params_from_log_path elsewhere, but
    this keeps the service from crashing if that import changes.
    """
    if not log_path:
        return {}

    out = {}

    parts = str(log_path).strip("/").replace("\\", "/").split("/")

    try:
        base_idx = parts.index("best_models")
    except ValueError:
        base_idx = None

    if base_idx is not None:
        if len(parts) > base_idx + 1:
            out["Task"] = parts[base_idx + 1]
        if len(parts) > base_idx + 2:
            out["Model Name"] = parts[base_idx + 2]
        if len(parts) > base_idx + 3:
            out["Dataset"] = parts[base_idx + 3]
        if len(parts) > base_idx + 4 and str(parts[base_idx + 4]).startswith("split_"):
            out["Split Segment"] = parts[base_idx + 4]
            out["_split_config_in_path"] = True
        offset = 1 if out.get("_split_config_in_path") else 0
        if len(parts) > base_idx + 7 + offset:
            out["Classif_Loss"] = parts[base_idx + 7 + offset]
        if len(parts) > base_idx + 8 + offset:
            out["DLoss"] = parts[base_idx + 8 + offset]

    for p in parts:
        low = p.lower()

        if low.startswith("nsize") or low.startswith("size"):
            digits = "".join(ch for ch in p if ch.isdigit())
            if digits:
                out["NSize"] = int(digits)
        elif low.startswith("fgsm"):
            digits = "".join(ch for ch in p if ch.isdigit())
            if digits:
                out["FGSM"] = int(digits)
        elif low.startswith("ncal"):
            digits = "".join(ch for ch in p if ch.isdigit())
            if digits:
                out["N_Calibration"] = int(digits)
        elif low.startswith("prototypes_"):
            out["Prototypes"] = p[len("prototypes_"):]
        elif low.startswith("norm"):
            out["Normalize"] = p[len("norm"):]
        elif low.startswith("dist_"):
            out["Dist_Fct"] = p[len("dist_"):]
        elif low.startswith("knn"):
            digits = "".join(ch for ch in p if ch.isdigit())
            if digits:
                out["N_Neighbors"] = int(digits)
        elif low.startswith("protoagg_"):
            agg_parts = p.split("_")
            if len(agg_parts) >= 2 and agg_parts[1]:
                out["Proto_Strat"] = agg_parts[1]
                out["prototype_strategy"] = agg_parts[1]
            if len(agg_parts) >= 3 and agg_parts[2]:
                out["Proto_Comp"] = agg_parts[2]
                out["prototype_components"] = agg_parts[2]

        if low.startswith("npos"):
            digits = "".join(ch for ch in p if ch.isdigit())
            if digits:
                out["NPos"] = int(digits)

        if low.startswith("nneg"):
            digits = "".join(ch for ch in p if ch.isdigit())
            if digits:
                out["NNeg"] = int(digits)

    return out


def args_from_model_row(base_args, row_dict: dict):
    """
    Build an args namespace from one best_models_registry row.

    This is used by the learned embedding page to reconstruct the correct
    model configuration before enumerating KNN / baseline / prototype heads.
    """
    local_args = argparse.Namespace(**vars(base_args))
    row = dict(row_dict or {})

    log_path = (
        row.get("Log Path")
        or row.get("log_path")
        or row.get("path")
        or getattr(local_args, "log_path", None)
    )

    parsed = _extract_params_from_log_path(log_path or "")
    row.update({k: v for k, v in parsed.items() if v is not None})

    string_mapping = {
        "Task": "task",
        "task": "task",
        "Model Name": "model_name",
        "model_name": "model_name",
        "FGSM": "fgsm",
        "fgsm": "fgsm",
        "Normalize": "normalize",
        "normalize": "normalize",
        "Classif_Loss": "classif_loss",
        "classif_loss": "classif_loss",
        "DLoss": "dloss",
        "dloss": "dloss",
        "Dist_Fct": "dist_fct",
        "dist_fct": "dist_fct",
        "Prototypes": "prototypes_to_use",
        "prototypes": "prototypes_to_use",
        "Proto_Strat": "prototype_strategy",
        "prototype_strategy": "prototype_strategy",
    }

    for src, dst in string_mapping.items():
        if row.get(src) is not None:
            setattr(local_args, dst, row.get(src))

    int_mapping = {
        "Model ID": "model_id",
        "id": "model_id",
        "NSize": "new_size",
        "nsize": "new_size",
        "NPos": "n_positives",
        "npos": "n_positives",
        "NNeg": "n_negatives",
        "nneg": "n_negatives",
        "N_Calibration": "n_calibration",
        "n_calibration": "n_calibration",
        "N_Neighbors": "n_neighbors",
        "n_neighbors": "n_neighbors",
        "Proto_Comp": "prototype_components",
        "prototype_components": "prototype_components",
    }

    for src, dst in int_mapping.items():
        if row.get(src) is not None:
            val = _ensure_int(row.get(src), getattr(local_args, dst, None))
            if val is not None:
                setattr(local_args, dst, val)

    for src, dst in {
        "train_datasets": "train_datasets",
        "Train Datasets": "train_datasets",
        "valid_dataset": "valid_dataset",
        "Valid Dataset": "valid_dataset",
        "test_dataset": "test_dataset",
        "Test Dataset": "test_dataset",
    }.items():
        if row.get(src) is not None:
            setattr(local_args, dst, row.get(src))

    if row.get("_split_config_in_path") or row.get("Split Segment") or row.get("split_config_key"):
        setattr(local_args, "split_config_in_path", True)
        setattr(local_args, "_split_config_in_path", True)

    if log_path:
        local_args.log_path = log_path

    dataset = row.get("Dataset") or row.get("dataset")
    if dataset:
        dataset = str(dataset)
        local_args.path = dataset if dataset.startswith("data/") else os.path.join("data", dataset)

    return local_args


def fetch_best_model_rows(cursor, limit: int = 10) -> pd.DataFrame:
    """
    Read top models from best_models_registry.

    This tries the column names used in your OtiteNet app. If a branch has
    slightly different names, the fallback query still returns the full rows.
    """
    limit = int(limit)

    preferred_query = """
        SELECT
            id AS `Model ID`,
            task AS `Task`,
            model_name AS `Model Name`,
            nsize AS `NSize`,
            fgsm AS `FGSM`,
            prototypes AS `Prototypes`,
            npos AS `NPos`,
            nneg AS `NNeg`,
            dloss AS `DLoss`,
            dist_fct AS `Dist_Fct`,
            classif_loss AS `Classif_Loss`,
            n_calibration AS `N_Calibration`,
            accuracy AS `Accuracy`,
            mcc AS `MCC`,
            normalize AS `Normalize`,
            n_neighbors AS `N_Neighbors`,
            train_datasets AS `train_datasets`,
            valid_dataset AS `valid_dataset`,
            test_dataset AS `test_dataset`,
            split_config_key AS `split_config_key`,
            log_path AS `Log Path`
        FROM best_models_registry
        ORDER BY mcc DESC
        LIMIT %s
    """

    try:
        cursor.execute(preferred_query, (limit,))
        rows = cursor.fetchall()

        columns = [
            "Model ID",
            "Task",
            "Model Name",
            "NSize",
            "FGSM",
            "Prototypes",
            "NPos",
            "NNeg",
            "DLoss",
            "Dist_Fct",
            "Classif_Loss",
            "N_Calibration",
            "Accuracy",
            "MCC",
            "Normalize",
            "N_Neighbors",
            "train_datasets",
            "valid_dataset",
            "test_dataset",
            "split_config_key",
            "Log Path",
        ]

        return pd.DataFrame(rows, columns=columns)

    except Exception:
        cursor.execute(
            """
            SELECT *
            FROM best_models_registry
            ORDER BY mcc DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        if isinstance(rows[0], dict):
            return pd.DataFrame(rows)

        colnames = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=colnames)


def what_is_missing(result) -> list[str]:
    """
    Small helper used by the page when showing optional recompute results.
    """
    if not isinstance(result, dict):
        return ["unknown"]

    missing = []

    if not result.get("knn") and not result.get("mcc_per_k"):
        missing.append("knn")

    if not result.get("baselines"):
        missing.append("baselines")

    if not result.get("prototypes"):
        missing.append("prototypes")

    return missing


def optimize_and_cache(args, n_aug: int, min_k: int = 1, max_k: int = 20, force: bool = False):
    """
    Optional recompute hook.

    The current learned_embedding page primarily displays existing heads from
    enumerate_classification_heads(). This function is kept so the page imports
    cleanly. We can wire it to your original full optimization functions after
    we confirm where those functions live in your current repo.
    """
    raise NotImplementedError(
        "Recompute is not wired yet in embedding_optimization_service.py. "
        "Existing KNN/baseline/prototype heads should still be displayed using "
        "enumerate_classification_heads() in learned_embedding.py."
    )
