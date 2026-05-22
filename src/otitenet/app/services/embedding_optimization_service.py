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

    parts = str(log_path).replace("\\", "/").split("/")

    for p in parts:
        low = p.lower()

        if low.startswith("nsize") or low.startswith("size"):
            digits = "".join(ch for ch in p if ch.isdigit())
            if digits:
                out["NSize"] = int(digits)

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
    }

    for src, dst in int_mapping.items():
        if row.get(src) is not None:
            val = _ensure_int(row.get(src), getattr(local_args, dst, None))
            if val is not None:
                setattr(local_args, dst, val)

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
