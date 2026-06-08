import argparse
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc as sk_auc, matthews_corrcoef, roc_curve
from sklearn.preprocessing import LabelEncoder

from otitenet.app.display_metrics import (
    _arrow_safe_dataframe,
    _head_config_label_global,
)
from otitenet.app.model_loading import load_model_and_prototypes
from otitenet.app.artifact_registry import (
    allow_legacy_best_models_artifacts,
    preferred_model_artifact_dir,
    scan_best_models_registry,
)
from otitenet.app.services.inference_results_service import labels_match
from otitenet.app.utils import (
    _make_model_selection_key,
    _unique_preserve_order,
    apply_selection_keys_to_models_df,
    attach_task_column,
    best_cached_head_metrics_for_model_row,
    ensure_int,
    extract_params_from_log_path,
    filter_models_df_by_task,
    format_classifier_config,
    get_calibration_metrics,
    get_model_order_metric,
    get_model_split_config,
    get_split_mcc_metrics,
    metric_value_from_mapping,
    resolve_best_classifier_config,
    sort_dataframe_by_model_metric,
    split_combo_key_from_row,
    split_config_key,
)
from otitenet.app.utils_dataset_names import get_short_dataset_name, get_short_dataset_names
from otitenet.data.data_getters import get_images_loaders
from otitenet.logging.metrics import MCC
from otitenet.train.train_triplet_new import TrainAE
from otitenet.utils.encoding_utils import (
    compute_prototypes_by_strategy,
    flatten_prototype_dict,
)
from otitenet.utils.utils import get_empty_traces


# -------------------------------------------------
# Shared helpers
# -------------------------------------------------

def _k(page_key: str, key: str) -> str:
    """Namespace Streamlit keys so this page can be rendered in multiple tabs."""
    return f"{page_key}_{key}"


def _safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_metric(value, digits: int = 4) -> str:
    value = _safe_float(value)
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def _clone_args(args):
    """Copy argparse-like args safely."""
    try:
        return argparse.Namespace(**vars(args))
    except Exception:
        return argparse.Namespace()


def _apply_model_row_to_args(args, row_dict: Dict[str, Any]):
    """Return a local args object updated from a leaderboard/model row."""
    local_args = _clone_args(args)
    row = dict(row_dict or {})

    parsed = extract_params_from_log_path(
        row.get("Best Model Dir")
        or row.get("best_model_dir")
        or row.get("Log Path")
        or row.get("log_path")
        or ""
    )
    row.update({k: v for k, v in parsed.items() if v is not None})

    mapping = {
        "Model Name": "model_name",
        "FGSM": "fgsm",
        "Normalize": "normalize",
        "N_Calibration": "n_calibration",
        "Classif_Loss": "classif_loss",
        "DLoss": "dloss",
        "Dist_Fct": "dist_fct",
        "Prototypes": "prototypes_to_use",
    }

    for row_key, arg_key in mapping.items():
        if row.get(row_key) is not None:
            setattr(local_args, arg_key, row.get(row_key))

    if row.get("NSize") is not None:
        local_args.new_size = ensure_int(row.get("NSize"))
    if row.get("NPos") is not None:
        local_args.n_positives = ensure_int(row.get("NPos"))
    if row.get("NNeg") is not None:
        local_args.n_negatives = ensure_int(row.get("NNeg"))
    if row.get("N_Neighbors") is not None:
        local_args.n_neighbors = ensure_int(row.get("N_Neighbors"))

    proto_strat = row.get("Proto_Strat") or row.get("prototype_strategy")
    proto_comp = row.get("Proto_Comp") or row.get("prototype_components")
    if proto_strat is not None:
        local_args.prototype_strategy = proto_strat
    if proto_comp is not None:
        local_args.prototype_components = ensure_int(proto_comp)

    for row_key, arg_key in {
        "train_datasets": "train_datasets",
        "Train Datasets": "train_datasets",
        "valid_dataset": "valid_dataset",
        "Valid Dataset": "valid_dataset",
        "test_dataset": "test_dataset",
        "Test Dataset": "test_dataset",
    }.items():
        if row.get(row_key) is not None:
            setattr(local_args, arg_key, row.get(row_key))

    if row.get("_split_config_in_path") or row.get("Split Segment") or row.get("split_config_key"):
        local_args.split_config_in_path = True
        local_args._split_config_in_path = True

    dataset = row.get("Dataset") or parsed.get("Dataset")
    if dataset:
        dataset = str(dataset)
        local_args.path = dataset if dataset.startswith("data/") else os.path.join("data", dataset)

    return local_args


def top_models_head_label(row_dict, args):
    """Return the learned-embedding head currently/best applied for this model row."""
    try:
        local_args = _apply_model_row_to_args(args, row_dict)
        row = dict(row_dict or {})

        head_config = row.get("Head Config")
        if head_config is not None and str(head_config).strip() not in {"", "nan", "None"}:
            stored_head = row.get("Head")
            if stored_head is not None and str(stored_head).strip() not in {"", "nan", "None", "—"}:
                return str(stored_head)
        if head_config is None or str(head_config).strip() in {"", "nan", "None"}:
            head_config = resolve_best_classifier_config(local_args, use_optimized=True)

        metadata = _load_model_row_metadata(row)
        metadata_args = metadata.get("args", {}) if isinstance(metadata, dict) else {}
        kind = str(metadata_args.get("kind") or metadata.get("kind", "") if isinstance(metadata, dict) else "").strip().lower()
        variant = str(metadata_args.get("variant") or metadata.get("variant", "") if isinstance(metadata, dict) else "").strip().lower()
        exp_id = str(metadata_args.get("exp_id") or "").strip().lower()
        complete_log_path = str(metadata.get("complete_log_path", "") if isinstance(metadata, dict) else "").strip().lower()
        siamese_inference = str(metadata_args.get("siamese_inference") or getattr(local_args, "siamese_inference", "linearsvc")).strip().lower()

        classif_loss = str(getattr(local_args, "classif_loss", "")).lower()
        is_cnn_mlp = (
            variant in {"cnn_transfer", "cnn_scratch", "mlp"}
            or "cnn_mlp_compare" in complete_log_path
            or ("_cnn_" in exp_id and "_siamese_" not in exp_id)
            or ("_mlp_" in exp_id and "_siamese_" not in exp_id)
        )
        if is_cnn_mlp:
            variant_label = {
                "cnn_transfer": "CNN transfer",
                "cnn_scratch": "CNN scratch",
                "mlp": "MLP",
            }.get(variant, "CNN/MLP")
            if "_cnn_" in exp_id:
                variant_label = "CNN transfer"
            elif "_mlp_" in exp_id:
                variant_label = "MLP"
            if classif_loss in {"ce", "cross_entropy", "cross-entropy"}:
                return f"{variant_label} CE head"
            if classif_loss == "hinge":
                return f"{variant_label} hinge head"
            return f"{variant_label} classifier head"

        if classif_loss in {"ce", "cross_entropy", "cross-entropy"} and str(head_config) in {"0", "0.0", "knn0", "None", ""}:
            if siamese_inference == "knn":
                return f"KNN (nn={ensure_int(row.get('N_Neighbors'))})"
            if siamese_inference == "logisticregression":
                return "Logistic Regression"
            return "Linear SVC"

        if head_config is None or str(head_config).strip() in {"", "nan", "None"}:
            if siamese_inference == "knn":
                return f"KNN (nn={ensure_int(row.get('N_Neighbors'))})"
            if siamese_inference == "logisticregression":
                return "Logistic Regression"
            if siamese_inference == "mlp_head":
                return "MLP Head"
            return "Linear SVC"

        label = format_classifier_config(head_config)
        if not label or str(label).strip() in {"", "—", "None", "nan"}:
            label = _head_config_label_global(head_config)
        label = label.replace("KNN (k=", "KNN (nn=")
        label = label.replace("KNN k=", "KNN (nn=")
        return label
    except Exception:
        try:
            nn = ensure_int(row_dict.get("N_Neighbors"))
            return f"KNN (nn={nn})"
        except Exception:
            return "—"


def _load_model_row_metadata(row: dict) -> dict:
    log_path = row.get("Log Path") or row.get("log_path")
    if not log_path:
        return {}

    candidates = [
        os.path.join(str(log_path), "run_metadata.json"),
        os.path.join(str(log_path), "run_summary.json"),
    ]

    for path in candidates:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
        except Exception:
            continue

    return {}


# -------------------------------------------------
# Data loading / dataframe construction
# -------------------------------------------------

def _query_best_models(cursor) -> Tuple[List[tuple], bool]:
    """Fetch model registry rows. Prefer model_rank when available."""
    try:
        cursor.execute(
            """
            SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg,
                   dloss, dist_fct, classif_loss, n_calibration,
                   accuracy, mcc,
                   train_mcc, valid_mcc, test_mcc, all_mcc,
                   valid_accuracy, train_auc, valid_auc, test_auc, all_auc,
                   ece, brier,
                   best_head_config, best_head_name, best_head_family, best_head_n_aug,
                   normalize, n_neighbors, log_path, model_rank,
                   prototype_strategy, prototype_components,
                   train_datasets, valid_dataset, test_dataset, split_config_key,
                   artifact_id, best_model_dir, source_run_log_path
            FROM best_models_registry
            ORDER BY (model_rank IS NULL) ASC, model_rank ASC, mcc DESC
            """
        )
        return cursor.fetchall() or [], True
    except Exception:
        try:
            try:
                cursor.execute(
                    """
                    SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg,
                           dloss, dist_fct, classif_loss, n_calibration,
                           accuracy, mcc, normalize, n_neighbors, log_path,
                           prototype_strategy, prototype_components,
                           train_datasets, valid_dataset, test_dataset, split_config_key,
                           artifact_id, best_model_dir, source_run_log_path
                    FROM best_models_registry
                    ORDER BY mcc DESC
                    """
                )
                return cursor.fetchall() or [], False
            except Exception:
                cursor.execute(
                    """
                    SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg,
                           dloss, dist_fct, classif_loss, n_calibration,
                           accuracy, mcc, normalize, n_neighbors, log_path,
                           prototype_strategy, prototype_components,
                           train_datasets, valid_dataset, test_dataset, split_config_key
                    FROM best_models_registry
                    ORDER BY mcc DESC
                    """
                )
                rows = cursor.fetchall() or []
                return [tuple(list(row) + [None, None, None]) for row in rows], False
        except Exception:
            cursor.execute(
                """
                SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg,
                       dloss, dist_fct, classif_loss, n_calibration,
                       accuracy, mcc, normalize, n_neighbors, log_path,
                       prototype_strategy, prototype_components
                FROM best_models_registry
                ORDER BY mcc DESC
                """
            )
            rows = cursor.fetchall() or []
            return [tuple(list(row) + [None, None, None, None, None, None, None]) for row in rows], False


def _models_dataframe_from_rows(rows: List[tuple], use_db_rank: bool) -> pd.DataFrame:
    if use_db_rank:
        columns = [
            "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos",
            "NNeg", "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration",
            "Accuracy", "MCC",
            "Train MCC", "Valid MCC", "Test MCC", "All MCC",
            "Valid Accuracy", "Train AUC", "Valid AUC", "Test AUC", "All AUC",
            "ECE", "Brier",
            "Head Config", "Head", "Head Family", "Head N Aug",
            "Normalize", "N_Neighbors", "Log Path", "#",
            "Proto_Strat", "Proto_Comp",
            "train_datasets", "valid_dataset", "test_dataset", "split_config_key",
            "Artifact ID", "Best Model Dir", "Source Run Path",
        ]
    else:
        columns = [
            "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos",
            "NNeg", "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration",
            "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path",
            "Proto_Strat", "Proto_Comp",
            "train_datasets", "valid_dataset", "test_dataset", "split_config_key",
            "Artifact ID", "Best Model Dir", "Source Run Path",
        ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        return df

    # Preserve task information from the original registry path before Log Path is
    # swapped to a source-run artifact path that no longer encodes best_models/<task>/.
    df = attach_task_column(df)

    if "Best Model Dir" not in df.columns:
        df["Best Model Dir"] = df.get("Log Path", "")
    if "Source Run Path" not in df.columns:
        df["Source Run Path"] = ""
    for idx, row in df.iterrows():
        preferred = preferred_model_artifact_dir(row.to_dict())
        if preferred:
            df.at[idx, "Artifact Log Path"] = preferred
            df.at[idx, "Log Path"] = preferred

    if "MCC" in df.columns:
        df["_mcc_numeric"] = pd.to_numeric(df["MCC"], errors="coerce").fillna(float("-inf"))
        df = df.sort_values("_mcc_numeric", ascending=False).drop(columns=["_mcc_numeric"])

    if "Log Path" in df.columns:
        df = df.dropna(subset=["Log Path"])
        df = df[df["Log Path"].astype(str) != ""]

    for idx, row in df.iterrows():
        split_config = get_model_split_config(row.get("Log Path"))
        for col in ["train_datasets", "valid_dataset", "test_dataset", "split_config_key"]:
            if col in df.columns and (pd.isna(row.get(col)) or str(row.get(col)).strip() in {"", "None", "nan"}):
                df.at[idx, col] = split_config.get(col)

    return df.reset_index(drop=True)


def _models_dataframe_from_log_tree() -> pd.DataFrame:
    """Scan logs/best_models so freshly mirrored artifacts can appear before DB sync."""
    if not allow_legacy_best_models_artifacts():
        return pd.DataFrame()
    try:
        artifacts = scan_best_models_registry("logs/best_models")
    except Exception:
        return pd.DataFrame()

    if artifacts is None or artifacts.empty:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Model ID": "",
            "Model Name": artifacts.get("model_name", ""),
            "NSize": artifacts.get("nsize", ""),
            "FGSM": artifacts.get("fgsm", ""),
            "Prototypes": artifacts.get("prototypes", ""),
            "NPos": artifacts.get("npos", ""),
            "NNeg": artifacts.get("nneg", ""),
            "DLoss": artifacts.get("dloss", ""),
            "Dist_Fct": artifacts.get("dist_fct", ""),
            "Classif_Loss": artifacts.get("classif_loss", ""),
            "N_Calibration": artifacts.get("n_calibration", ""),
            "Accuracy": np.nan,
            "MCC": np.nan,
            "Normalize": artifacts.get("normalize", ""),
            "N_Neighbors": artifacts.get("n_neighbors", ""),
            "Log Path": artifacts.get("model_dir", ""),
            "Proto_Strat": artifacts.get("prototype_strategy", ""),
            "Proto_Comp": artifacts.get("prototype_components", ""),
            "train_datasets": "",
            "valid_dataset": "",
            "test_dataset": "",
            "split_config_key": "",
            "Artifact ID": artifacts.get("artifact_id", ""),
            "Best Model Dir": artifacts.get("model_dir", ""),
            "Source Run Path": artifacts.get("run_log_path", ""),
            "train_encodings_path": artifacts.get("train_encodings_path", ""),
            "valid_encodings_path": artifacts.get("valid_encodings_path", ""),
            "test_encodings_path": artifacts.get("test_encodings_path", ""),
            "Source": "logs/best_models",
            "Task": artifacts.get("task", ""),
        }
    )

    for idx, row in df.iterrows():
        split_config = get_model_split_config(row.get("Log Path"))
        for col in ["train_datasets", "valid_dataset", "test_dataset", "split_config_key"]:
            value = row.get(col)
            if pd.isna(value) or str(value).strip() in {"", "None", "nan"}:
                df.at[idx, col] = split_config.get(col, "")

    return df


def _dataset_path_from_manifest_value(value: Any) -> str:
    dataset = str(value or "").strip().replace("\\", "/")
    if dataset and "/" not in dataset and dataset.startswith("otite_ds_"):
        parts = dataset.split("_", 3)
        if len(parts) == 4:
            dataset = f"{parts[0]}_{parts[1]}_{parts[2]}/{parts[3]}"
    return dataset


def _progress_manifest_models_dataframe(task: Optional[str] = None) -> pd.DataFrame:
    task_text = str(task or "").strip()
    roots = []
    if task_text:
        roots.append(os.path.join("logs", "progresses", task_text))
    else:
        roots.append(os.path.join("logs", "progresses"))

    rows: List[Dict[str, Any]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            manifest_names = [
                name for name in filenames
                if name.startswith("PROD_") and name.endswith("_job_manifest.csv")
            ]
            if not manifest_names:
                continue
            for manifest_name in manifest_names:
                manifest_path = os.path.join(dirpath, manifest_name)
                try:
                    manifest_df = pd.read_csv(manifest_path, dtype=str).fillna("")
                except Exception:
                    continue
                if "job_state" in manifest_df.columns:
                    manifest_df = manifest_df[manifest_df["job_state"].astype(str).str.lower() == "done"]
                if task_text and "task" in manifest_df.columns:
                    manifest_df = manifest_df[manifest_df["task"].astype(str) == task_text]
                for _, row in manifest_df.iterrows():
                    row_task = str(row.get("task") or task_text).strip()
                    if not row_task:
                        continue
                    uuid = str(row.get("uuid") or "").strip()
                    source_run_path = os.path.join("logs", row_task, uuid) if uuid else ""
                    if source_run_path and not os.path.isdir(source_run_path):
                        source_run_path = ""
                    log_path = source_run_path or str(row.get("stdout_file") or row.get("log_file") or "").strip()
                    if not log_path:
                        continue
                    classif_loss = str(row.get("classif_loss") or "").strip() or str(row.get("loss") or "").strip()
                    split_key = split_config_key(row.get("train_datasets"), row.get("valid_dataset"), row.get("test_dataset"))
                    rows.append(
                        {
                            "Model ID": str(row.get("exp_id") or row.get("job_id") or uuid or "").strip(),
                            "Model Name": str(row.get("model") or "").strip(),
                            "NSize": str(row.get("new_size") or "").strip(),
                            "FGSM": str(row.get("fgsm") or "").strip(),
                            "Prototypes": str(row.get("prototype") or row.get("prototypes") or "").strip(),
                            "NPos": str(row.get("n_positives") or "").strip(),
                            "NNeg": str(row.get("n_negatives") or "").strip(),
                            "DLoss": str(row.get("dloss") or "").strip(),
                            "Dist_Fct": str(row.get("dist_fct") or "").strip(),
                            "Classif_Loss": classif_loss,
                            "N_Calibration": str(row.get("n_calibration") or "").strip(),
                            "Accuracy": np.nan,
                            "MCC": np.nan,
                            "Normalize": str(row.get("normalize") or "").strip(),
                            "N_Neighbors": str(row.get("knn") or row.get("n_neighbors") or "").strip(),
                            "Log Path": log_path,
                            "Proto_Strat": str(row.get("prototype_strategy") or "").strip(),
                            "Proto_Comp": str(row.get("prototype_components") or "").strip(),
                            "train_datasets": str(row.get("train_datasets") or "").strip(),
                            "valid_dataset": str(row.get("valid_dataset") or "").strip(),
                            "test_dataset": str(row.get("test_dataset") or "").strip(),
                            "split_config_key": split_key,
                            "Artifact ID": uuid,
                            "Best Model Dir": source_run_path or log_path,
                            "Source Run Path": source_run_path,
                            "Dataset": _dataset_path_from_manifest_value(row.get("dataset_name") or row.get("dataset_key")),
                            "Source": "progress manifest",
                            "Task": row_task,
                        }
                    )
    return pd.DataFrame(rows)


def _merge_db_and_log_tree_models(db_df: pd.DataFrame) -> pd.DataFrame:
    tree_df = _models_dataframe_from_log_tree()
    if tree_df.empty:
        out = db_df
    elif db_df is None or db_df.empty:
        out = tree_df
    else:
        db_df = db_df.copy()
        if "Source" not in db_df.columns:
            db_df["Source"] = "database"

        known_paths = set()
        for col in ["Log Path", "Best Model Dir", "Source Run Path"]:
            if col in db_df.columns:
                known_paths.update(
                    str(v).strip()
                    for v in db_df[col].dropna().astype(str)
                    if str(v).strip()
                )

        tree_df = tree_df[
            ~tree_df["Log Path"].astype(str).str.strip().isin(known_paths)
            & ~tree_df["Best Model Dir"].astype(str).str.strip().isin(known_paths)
        ]
        out = db_df if tree_df.empty else pd.concat([db_df, tree_df], ignore_index=True, sort=False)

    manifest_df = _progress_manifest_models_dataframe()
    if manifest_df.empty:
        return out
    if out is None or out.empty:
        return manifest_df

    known_paths = set()
    for col in ["Log Path", "Best Model Dir", "Source Run Path"]:
        if col in out.columns:
            known_paths.update(
                str(v).strip()
                for v in out[col].dropna().astype(str)
                if str(v).strip()
            )
    manifest_df = manifest_df[
        ~manifest_df["Log Path"].astype(str).str.strip().isin(known_paths)
        & ~manifest_df["Best Model Dir"].astype(str).str.strip().isin(known_paths)
    ]
    if manifest_df.empty:
        return out
    return pd.concat([out, manifest_df], ignore_index=True, sort=False)


def _recent_trained_heads_for_model(row: pd.Series) -> List[Dict[str, Any]]:
    payload = st.session_state.get("learned_last_training_payload") or {}
    recent_rows = payload.get("trained_head_rows") or []
    if not recent_rows:
        return []

    model_id = row.get("Model ID") or row.get("id") or row.get("model_id")
    log_path = str(row.get("Log Path") or row.get("log_path") or "").strip()
    best_model_dir = str(row.get("Best Model Dir") or row.get("best_model_dir") or "").strip()
    candidates = []

    for recent in recent_rows:
        if not isinstance(recent, dict):
            continue
        recent_model_id = recent.get("Model ID") or recent.get("id") or recent.get("model_id")
        recent_log_path = str(recent.get("Log Path") or recent.get("log_path") or "").strip()
        model_matches = model_id is not None and recent_model_id is not None and str(model_id) == str(recent_model_id)
        path_matches = bool(recent_log_path and recent_log_path in {log_path, best_model_dir})
        if model_matches or path_matches:
            candidates.append(recent)

    return candidates


def _active_calibration_split(page_key: str | None = None) -> str:
    if page_key:
        return str(st.session_state.get(_k(page_key, "calibration_split"), "valid"))
    return str(st.session_state.get("leaderboard_calibration_split", "valid"))


def _attach_metrics(df: pd.DataFrame, calibration_split: str = "valid") -> pd.DataFrame:
    df = df.copy()

    metric_cols = [
        "Train MCC", "Valid MCC", "Test MCC",
        "Train AUC", "Valid AUC", "Test AUC",
        "ECE", "Brier",
    ]

    for col in metric_cols:
        if col not in df.columns:
            df[col] = np.nan

    # --- Prefer classifier head metrics from optimization cache if available ---
    try:
        from otitenet.app.pages.learned_embedding import _get_heads_for_model_args, args_from_model_row
    except ImportError:
        _get_heads_for_model_args = None
        args_from_model_row = None

    for idx, row in df.iterrows():
        log_path = row.get("Log Path")
        best_model_dir = row.get("Best Model Dir") or row.get("best_model_dir") or log_path
        head_metrics = None
        registry_head_metrics = None
        registry_head_config = row.get("Head Config") or row.get("best_head_config")
        registry_valid_mcc = pd.to_numeric(pd.Series([row.get("Valid MCC")]), errors="coerce").iloc[0]
        if registry_head_config is not None and str(registry_head_config).strip() not in {"", "nan", "None"} and pd.notna(registry_valid_mcc):
            registry_head_metrics = {
                "Config": registry_head_config,
                "Head": row.get("Head") or row.get("best_head_name") or format_classifier_config(registry_head_config),
                "N Aug": row.get("Head N Aug") or row.get("best_head_n_aug"),
                "Train MCC": row.get("Train MCC"),
                "Valid MCC": registry_valid_mcc,
                "Test MCC": row.get("Test MCC"),
                "All MCC": row.get("All MCC"),
                "Valid Accuracy": row.get("Valid Accuracy"),
                "Train AUC": row.get("Train AUC"),
                "Valid AUC": row.get("Valid AUC"),
                "Test AUC": row.get("Test AUC"),
                "All AUC": row.get("All AUC"),
                "ECE": np.nan,
                "Brier": np.nan,
            }
        head_candidates = []
        cached_head_metrics = best_cached_head_metrics_for_model_row(row.to_dict())
        if cached_head_metrics:
            head_candidates.append(cached_head_metrics)
        if _get_heads_for_model_args and args_from_model_row:
            try:
                row_for_head_lookup = row.to_dict()
                row_for_head_lookup["Log Path"] = best_model_dir
                model_args = args_from_model_row(argparse.Namespace(), row_for_head_lookup)
                head_rows, _ = _get_heads_for_model_args(model_args, row_for_head_lookup)
                if head_rows:
                    head_candidates.extend(head_rows)
            except Exception:
                pass
        head_candidates.extend(_recent_trained_heads_for_model(row))
        # The optimization cache and just-trained rows are the source of truth
        # after Tab 2 retraining. Persisted registry values are only a fallback.
        if head_candidates:
            head_metrics = max(
                head_candidates,
                key=lambda r: metric_value_from_mapping(r, default=float("-inf")),
            )
        if head_metrics is None:
            head_metrics = registry_head_metrics
        if head_metrics:
            head_config = head_metrics.get("Config") or head_metrics.get("config")
            if head_config is not None:
                df.at[idx, "Head Config"] = str(head_config)
                df.at[idx, "Head"] = head_metrics.get("Head") or format_classifier_config(head_config)
            if head_metrics.get("N Aug") is not None:
                df.at[idx, "Head N Aug"] = head_metrics.get("N Aug")
            df.at[idx, "Train MCC"] = head_metrics.get("Train MCC", head_metrics.get("train_mcc", np.nan))
            df.at[idx, "Valid MCC"] = head_metrics.get("Valid MCC", head_metrics.get("Best Valid MCC", head_metrics.get("valid_mcc", np.nan)))
            df.at[idx, "Test MCC"] = head_metrics.get("Test MCC", head_metrics.get("Best Test MCC", head_metrics.get("test_mcc", np.nan)))
            df.at[idx, "All MCC"] = head_metrics.get("All MCC", head_metrics.get("all_mcc", np.nan))
            df.at[idx, "Valid Accuracy"] = head_metrics.get("Valid Accuracy", head_metrics.get("valid_accuracy", np.nan))
            df.at[idx, "Train AUC"] = head_metrics.get("Train AUC", head_metrics.get("train_auc", np.nan))
            df.at[idx, "Valid AUC"] = head_metrics.get("Valid AUC", head_metrics.get("Best Valid AUC", head_metrics.get("valid_auc", np.nan)))
            df.at[idx, "Test AUC"] = head_metrics.get("Test AUC", head_metrics.get("Best Test AUC", head_metrics.get("test_auc", np.nan)))
            df.at[idx, "All AUC"] = head_metrics.get("All AUC", head_metrics.get("all_auc", np.nan))
            df.at[idx, "ECE"] = np.nan
            df.at[idx, "Brier"] = np.nan
            # Add any extra metrics (F1, Recall, etc.)
            for metric_name, metric_value in head_metrics.items():
                if " F1 " in metric_name or " Recall " in metric_name or " Precision " in metric_name or " Support " in metric_name:
                    if metric_name not in df.columns:
                        df[metric_name] = np.nan
                    df.at[idx, metric_name] = metric_value
        else:
            # Fallback to log summary metrics
            split_metrics = get_split_mcc_metrics(log_path)
            if split_metrics:
                df.at[idx, "Train MCC"] = split_metrics.get("train_mcc", np.nan)
                df.at[idx, "Valid MCC"] = split_metrics.get("valid_mcc", np.nan)
                df.at[idx, "Test MCC"] = split_metrics.get("test_mcc", np.nan)
                df.at[idx, "Train AUC"] = split_metrics.get("train_auc", np.nan)
                df.at[idx, "Valid AUC"] = split_metrics.get("valid_auc", np.nan)
                df.at[idx, "Test AUC"] = split_metrics.get("test_auc", np.nan)
                for metric_name, metric_value in split_metrics.items():
                    if " F1 " in metric_name or " Recall " in metric_name or " Precision " in metric_name or " Support " in metric_name:
                        if metric_name not in df.columns:
                            df[metric_name] = np.nan
                        df.at[idx, metric_name] = metric_value

        calibration_metrics = get_calibration_metrics(log_path, split=calibration_split)
        if calibration_metrics and not calibration_metrics.get("error"):
            df.at[idx, "ECE"] = calibration_metrics.get("ece", np.nan)
            df.at[idx, "Brier"] = calibration_metrics.get("brier", np.nan)

    if "Valid MCC" in df.columns and "MCC" in df.columns:
        df["Valid MCC"] = pd.to_numeric(df["Valid MCC"], errors="coerce").fillna(
            pd.to_numeric(df["MCC"], errors="coerce")
        )

    valid_sort = pd.to_numeric(df.get("Valid MCC"), errors="coerce")
    mcc_sort = pd.to_numeric(df.get("MCC"), errors="coerce")

    df = df.assign(
        _valid_sort=valid_sort.fillna(float("-inf")),
        _mcc_sort=mcc_sort.fillna(float("-inf")),
    )

    df = df.sort_values(
        ["_valid_sort", "_mcc_sort"],
        ascending=[False, False],
    ).reset_index(drop=True)

    df = df.drop(columns=["_valid_sort", "_mcc_sort"], errors="ignore")
    df["#"] = range(1, len(df) + 1)

    return df


def _order_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Accuracy", "MCC",
        "Train MCC", "Valid MCC", "Test MCC",
        "Train AUC", "Valid AUC", "Test AUC",
        "ECE", "Brier",
    ]
    per_class_cols = [
        c for c in df.columns
        if " F1 " in c or " Recall " in c or " Precision " in c or " Support " in c
    ]

    ordered_cols = []
    if "#" in df.columns:
        ordered_cols.append("#")

    preferred_non_metric = [
        "Model ID", "Task", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
        "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize",
        "N_Neighbors", "Proto_Strat", "Proto_Comp",
        "train_datasets", "valid_dataset", "test_dataset",
        "Source", "Log Path",
    ]

    ordered_cols += [c for c in preferred_non_metric if c in df.columns and c not in ordered_cols]
    ordered_cols += [c for c in metric_cols if c in df.columns and c not in ordered_cols]
    ordered_cols += [c for c in sorted(per_class_cols) if c not in ordered_cols]
    ordered_cols += [c for c in df.columns if c not in ordered_cols]

    return df[ordered_cols]


def _split_combo_key_from_row(row) -> str:
    return split_combo_key_from_row(row)


def _filter_models_df_by_sidebar_split(models_df: pd.DataFrame) -> pd.DataFrame:
    if models_df is None or models_df.empty:
        return models_df

    selected_split_key = st.session_state.get("sidebar_split_combo_key")
    if not selected_split_key:
        return models_df

    df = models_df.copy()
    df["_split_combo_key"] = df.apply(_split_combo_key_from_row, axis=1)
    filtered_df = df[df["_split_combo_key"] == str(selected_split_key)].drop(columns=["_split_combo_key"])
    return filtered_df.reset_index(drop=True)


def load_best_models_table(cursor, task: Optional[str] = None, calibration_split: str = "valid") -> pd.DataFrame:
    rows, use_db_rank = _query_best_models(cursor)
    df = _models_dataframe_from_rows(rows, use_db_rank)
    df = _merge_db_and_log_tree_models(df)
    if df.empty:
        return df

    df = attach_task_column(df)
    df = filter_models_df_by_task(df, task)
    if df.empty:
        return df

    df = _filter_models_df_by_sidebar_split(df)
    if df.empty:
        return df

    df = _attach_metrics(df, calibration_split=calibration_split)
    df = _order_model_columns(df)
    return df


def _update_model_number_map(df: pd.DataFrame) -> None:
    model_number_map = {}

    for _, row in df.iterrows():
        rd = row.to_dict()
        selection_key = _make_model_selection_key(rd)
        model_number_map[selection_key] = rd.get("#", "?")

    st.session_state["model_number_map"] = model_number_map
    st.session_state["best_models_table"] = df.copy()


# -------------------------------------------------
# Prototype/PCA computation
# -------------------------------------------------

def compute_best_proto_mcc_for_args(
    _args,
    strategy: str,
    min_components: int = 1,
    max_components: int = 5,
    random_state: int = 1,
):
    """Compute the best validation MCC for a prototype strategy."""
    model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)

    train = TrainAE(
        _args,
        _args.path,
        load_tb=False,
        log_metrics=False,
        keep_models=True,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_tracking=False,
        log_mlflow=False,
        groupkfold=getattr(_args, "groupkfold", 1),
    )

    train.n_batches = len(unique_batches)
    train.n_cats = len(unique_labels)
    train.unique_batches = unique_batches
    train.unique_labels = unique_labels
    train._batch_encoder = LabelEncoder().fit(np.asarray(unique_batches))
    train.epoch = 1
    train.model = model
    train.params = {"n_neighbors": int(getattr(_args, "n_neighbors", 1))}
    train.set_arcloss()

    lists, traces = get_empty_traces()

    loaders = get_images_loaders(
        data=data,
        random_recs=getattr(_args, "random_recs", 0),
        weighted_sampler=0,
        is_transform=0,
        samples_weights=None,
        epoch=1,
        unique_labels=unique_labels,
        triplet_dloss=getattr(_args, "dloss", "triplet"),
        bs=getattr(_args, "bs", 32),
        prototypes_to_use=getattr(_args, "prototypes_to_use", "class"),
        prototypes=prototypes,
        size=getattr(_args, "new_size", 64),
        normalize=getattr(_args, "normalize", "no"),
        batch_encoder=train._batch_encoder,
    )

    with torch.no_grad():
        _, lists, _ = train.loop("train", None, 0, loaders["train"], lists, traces)
        _, lists, _ = train.loop("valid", None, 0, loaders["valid"], lists, traces)

    train_encs = np.concatenate(lists["train"]["encoded_values"])
    train_cats = np.concatenate(lists["train"]["cats"])
    valid_encs = np.concatenate(lists["valid"]["encoded_values"])
    valid_cats = np.concatenate(lists["valid"]["cats"])

    best_mcc = None
    best_n_components = None
    best_n_prototypes = 0
    per_components = []

    for n_components in range(min_components, max_components + 1):
        proto_dict = compute_prototypes_by_strategy(
            train_encs,
            train_cats,
            strategy,
            n_components,
            random_state,
        )
        proto_vecs, proto_labels = flatten_prototype_dict(proto_dict)

        if len(proto_vecs) == 0:
            continue

        dists = np.linalg.norm(valid_encs[:, None, :] - proto_vecs[None, :, :], axis=2)
        proto_preds = proto_labels[np.argmin(dists, axis=1)]
        proto_mcc = float(MCC(valid_cats, proto_preds))

        per_components.append(
            {
                "n_components": n_components,
                "mcc": proto_mcc,
                "n_prototypes": len(proto_vecs),
            }
        )

        if best_mcc is None or proto_mcc > best_mcc:
            best_mcc = proto_mcc
            best_n_components = n_components
            best_n_prototypes = len(proto_vecs)

    return {
        "best_mcc": best_mcc,
        "best_n_components": best_n_components,
        "n_prototypes": best_n_prototypes,
        "per_components": per_components,
    }


def compute_pca_for_args(_args, proto_strategies=None, proto_components=1):
    """Compute PCA with prototypes for one or multiple strategies."""
    if proto_strategies is None:
        proto_strategies = ["mean"]
    elif isinstance(proto_strategies, str):
        proto_strategies = [proto_strategies]

    model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)

    train = TrainAE(
        _args,
        _args.path,
        load_tb=False,
        log_metrics=False,
        keep_models=True,
        log_inputs=False,
        log_plots=False,
        log_tb=False,
        log_tracking=False,
        log_mlflow=False,
        groupkfold=getattr(_args, "groupkfold", 1),
    )

    train.n_batches = len(unique_batches)
    train.n_cats = len(unique_labels)
    train.unique_batches = unique_batches
    train.unique_labels = unique_labels
    train._batch_encoder = LabelEncoder().fit(np.asarray(unique_batches))
    train.epoch = 1
    train.model = model
    train.params = {"n_neighbors": int(getattr(_args, "n_neighbors", 1))}
    train.set_arcloss()

    lists, traces = get_empty_traces()

    loaders = get_images_loaders(
        data=data,
        random_recs=getattr(_args, "random_recs", 0),
        weighted_sampler=0,
        is_transform=0,
        samples_weights=None,
        epoch=1,
        unique_labels=unique_labels,
        triplet_dloss=getattr(_args, "dloss", "triplet"),
        bs=getattr(_args, "bs", 32),
        prototypes_to_use=getattr(_args, "prototypes_to_use", "class"),
        prototypes=prototypes,
        size=getattr(_args, "new_size", 64),
        normalize=getattr(_args, "normalize", "no"),
        batch_encoder=train._batch_encoder,
    )

    with torch.no_grad():
        for group in ["train", "valid", "test", "calibration"]:
            if group not in loaders:
                continue
            try:
                _, lists, _ = train.loop(group, None, 0, loaders[group], lists, traces)
            except Exception:
                if group in {"train", "valid"}:
                    raise

    encs = []
    cats = []
    datasets = []
    splits = []

    for grp in ["train", "valid", "test", "calibration"]:
        if lists.get(grp, {}).get("encoded_values"):
            grp_encs = np.concatenate(lists[grp]["encoded_values"])
            grp_cats = np.concatenate(lists[grp]["cats"])
            encs.append(grp_encs)
            cats.append(grp_cats)

            try:
                grp_datasets = np.concatenate(lists[grp]["domains"]).astype(str)
            except Exception:
                grp_datasets = np.array([grp] * len(grp_cats), dtype=str)
            datasets.append(grp_datasets)
            splits.append(np.array([grp] * len(grp_cats), dtype=str))

    if not encs:
        raise RuntimeError("No embeddings available to plot.")

    all_encs = np.concatenate(encs)
    all_cats = np.concatenate(cats)
    all_datasets = np.concatenate(datasets) if datasets else np.array(["unknown"] * len(all_cats), dtype=str)
    all_splits = np.concatenate(splits) if splits else np.array(["unknown"] * len(all_cats), dtype=str)

    n_comp = min(3, all_encs.shape[1])
    pca = PCA(n_components=n_comp)
    encs_pca = pca.fit_transform(all_encs)
    explained = pca.explained_variance_ratio_ * 100.0

    n_strategies = len(proto_strategies)
    fig, axes = plt.subplots(1, n_strategies, figsize=(6 * n_strategies, 5))

    if n_strategies == 1:
        axes = [axes]

    strategy_markers = {"mean": "X", "kmeans": "*", "gmm": "P"}
    strategy_sizes = {"mean": 300, "kmeans": 500, "gmm": 400}
    split_markers = {
        "train": "o",
        "valid": "^",
        "test": "s",
        "calibration": "D",
    }
    split_sizes = {
        "train": 24,
        "valid": 32,
        "test": 32,
        "calibration": 58,
    }
    dataset_values = sorted(pd.unique(pd.Series(all_datasets).astype(str)))
    dataset_cmap = plt.get_cmap("tab20", max(1, len(dataset_values)))
    dataset_colors = {
        dataset: dataset_cmap(idx % dataset_cmap.N)
        for idx, dataset in enumerate(dataset_values)
    }

    for ax_idx, proto_strategy in enumerate(proto_strategies):
        ax = axes[ax_idx]

        proto_dict = compute_prototypes_by_strategy(
            all_encs,
            all_cats,
            proto_strategy,
            proto_components,
            random_state=1,
        )

        proto_arr = None
        proto_colors = None

        if proto_dict:
            proto_list = []
            proto_colors = []

            for cls_id in sorted(proto_dict.keys()):
                for proto_vec, _comp_idx in proto_dict[cls_id]:
                    proto_list.append(proto_vec)
                    proto_colors.append(cls_id)

            if proto_list:
                proto_arr = np.stack(proto_list)
                proto_colors = np.array(proto_colors)

        proto_pca = pca.transform(proto_arr) if proto_arr is not None else None

        for dataset in dataset_values:
            dataset_mask = all_datasets.astype(str) == str(dataset)
            for split, marker in split_markers.items():
                mask = dataset_mask & (all_splits == split)
                if not np.any(mask):
                    continue
                is_calibration = split == "calibration"
                ax.scatter(
                    encs_pca[mask, 0],
                    encs_pca[mask, 1],
                    marker=marker,
                    color=dataset_colors[dataset],
                    alpha=0.82 if is_calibration else 0.52,
                    s=split_sizes.get(split, 28),
                    edgecolors="black" if is_calibration else "none",
                    linewidths=1.1 if is_calibration else 0.0,
                    zorder=4 if is_calibration else 2,
                )

        if proto_pca is not None:
            marker = strategy_markers.get(proto_strategy, "*")
            marker_size = strategy_sizes.get(proto_strategy, 500)

            ax.scatter(
                proto_pca[:, 0],
                proto_pca[:, 1],
                marker=marker,
                c=proto_colors,
                cmap="tab20",
                s=marker_size,
                edgecolors="black",
                linewidths=1.5,
                zorder=5,
            )

            ax.set_title(
                f"{proto_strategy.upper()}\n"
                f"({len(proto_pca)} prototypes total, {proto_components} per class)"
            )
        else:
            ax.set_title(f"{proto_strategy.upper()}\n({proto_components} per class)")

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        if n_comp > 1:
            ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.grid(True, alpha=0.3)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=dataset_colors[dataset],
            markeredgecolor="none",
            markersize=7,
            label=get_short_dataset_names(str(dataset)),
        )
        for dataset in dataset_values
    ]
    split_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black" if split == "calibration" else "none",
            markersize=8 if split == "calibration" else 7,
            label=split,
        )
        for split, marker in split_markers.items()
        if np.any(all_splits == split)
    ]
    dataset_legend = None
    if dataset_handles:
        dataset_legend = axes[0].legend(handles=dataset_handles, title="Dataset", loc="upper left", fontsize=8, title_fontsize=9)
    if split_handles:
        if dataset_legend is not None and axes[0] is axes[-1]:
            axes[0].add_artist(dataset_legend)
        axes[-1].legend(handles=split_handles, title="Split", loc="upper right", fontsize=8, title_fontsize=9)

    fig.suptitle("PCA with Prototypes (dataset color, split shape, calibration outlined)", fontsize=14, y=1.02)
    plt.tight_layout()

    fig_bytes = io.BytesIO()
    fig.savefig(fig_bytes, format="png", dpi=100, bbox_inches="tight")
    fig_bytes.seek(0)
    plt.close(fig)

    return fig_bytes.getvalue()


# -------------------------------------------------
# Rendering sections
# -------------------------------------------------

def _render_top_models_table(models_df: pd.DataFrame, args) -> None:
    st.write("**Top Models (all registry rows):**")
    st.markdown("**Table:** Top Models")

    display_columns = [
        col for col in models_df.columns
        if col not in {"Log Path", "N_Neighbors"}
    ]

    display_df = models_df[display_columns].copy()
    display_df["Head"] = models_df.apply(
        lambda r: top_models_head_label(r.to_dict(), args),
        axis=1,
    )
    # Add short name columns for datasets if present
    for col in ["train_datasets", "valid_dataset", "test_dataset", "Train Datasets", "Valid Dataset", "Test Dataset"]:
        if col in display_df.columns:
            short_col = f"Short {col.replace('_', ' ').title()}"
            display_df[short_col] = display_df[col].apply(get_short_dataset_names)

    cols_now = list(display_df.columns)
    if "Head" in cols_now:
        cols_now.remove("Head")
        insert_at = cols_now.index("Normalize") + 1 if "Normalize" in cols_now else min(12, len(cols_now))
        cols_now.insert(insert_at, "Head")
        display_df = display_df[cols_now]

    st.dataframe(_arrow_safe_dataframe(display_df), use_container_width=True)


def _prediction_csv_candidates(row_dict: Dict[str, Any], split: str) -> List[str]:
    candidates: List[str] = []
    for key in ["Artifact Log Path", "Best Model Dir", "Log Path", "Source Run Path"]:
        path = str(row_dict.get(key) or "").strip()
        if not path:
            continue
        if os.path.isfile(path):
            path = os.path.dirname(path)
        candidates.append(os.path.join(path, f"{split}_predictions.csv"))
        parent = path
        for _ in range(3):
            parent = os.path.dirname(parent)
            if not parent or parent == ".":
                break
            candidates.append(os.path.join(parent, f"{split}_predictions.csv"))
    return _unique_preserve_order(candidates)


def _load_prediction_csv_for_row(row_dict: Dict[str, Any], split: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    for path in _prediction_csv_candidates(row_dict, split):
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            return df, path
    return None, None


def _validation_threshold_decision_for_row(
    row,
    confidence_threshold,
    vote_threshold_pct,
    require_both_thresholds,
):
    try:
        confidence_value = float(row.get("Confidence"))
    except Exception:
        confidence_value = np.nan
    try:
        ensemble_vote_pct = float(row.get("Ensemble Raw Vote %"))
    except Exception:
        ensemble_vote_pct = np.nan

    selected_prediction = row.get("Prediction")
    ensemble_prediction = row.get("Ensemble Raw Prediction", "Unknown")
    selected_passes = pd.notna(confidence_value) and confidence_value >= float(confidence_threshold)
    ensemble_passes = pd.notna(ensemble_vote_pct) and ensemble_vote_pct >= float(vote_threshold_pct)

    if require_both_thresholds:
        return selected_prediction if selected_passes and ensemble_passes else "Unknown"
    if selected_passes:
        return selected_prediction
    return ensemble_prediction if ensemble_passes and str(ensemble_prediction or "").strip() else "Unknown"


def _threshold_metric_value(eval_df: pd.DataFrame, metric_name: str) -> float:
    kept_df = eval_df[~eval_df["_Decision"].astype(str).str.lower().isin(["", "unknown", "na", "nan", "none"])].copy()
    if metric_name == "Coverage":
        return float(len(kept_df) / len(eval_df)) if len(eval_df) else np.nan
    if metric_name.endswith("kept"):
        if kept_df.empty:
            return np.nan
        if metric_name.startswith("ACC"):
            return float(np.mean([
                labels_match(pred, truth)
                for pred, truth in zip(kept_df["_Decision"], kept_df["Ground Truth"])
            ]))
        try:
            return float(matthews_corrcoef(
                kept_df["Ground Truth"].astype(str).str.lower().values,
                kept_df["_Decision"].astype(str).str.lower().values,
            ))
        except Exception:
            return np.nan

    pred = eval_df["_Decision"].astype(str).str.lower().replace({"unknown": "__unknown__"})
    truth = eval_df["Ground Truth"].astype(str).str.lower()
    if metric_name.startswith("ACC"):
        return float(accuracy_score(truth.values, pred.values))
    try:
        return float(matthews_corrcoef(truth.values, pred.values))
    except Exception:
        return np.nan


def _threshold_metric_from_arrays(decisions: np.ndarray, truth: np.ndarray, metric_name: str) -> float:
    decision_text = pd.Series(decisions).fillna("").astype(str).str.strip().str.lower().to_numpy()
    truth_text = pd.Series(truth).fillna("").astype(str).str.strip().str.lower().to_numpy()
    kept = ~np.isin(decision_text, ["", "unknown", "na", "nan", "none"])
    if metric_name == "Coverage":
        return float(np.mean(kept)) if len(kept) else np.nan
    if metric_name.endswith("kept"):
        if not np.any(kept):
            return np.nan
        if metric_name.startswith("ACC"):
            return float(np.mean([
                labels_match(pred, actual)
                for pred, actual in zip(decision_text[kept], truth_text[kept])
            ]))
        try:
            return float(matthews_corrcoef(truth_text[kept], decision_text[kept]))
        except Exception:
            return np.nan

    scored_decisions = np.where(kept, decision_text, "__unknown__")
    if metric_name.startswith("ACC"):
        return float(np.mean([
            labels_match(pred, actual)
            for pred, actual in zip(scored_decisions, truth_text)
        ]))
    try:
        return float(matthews_corrcoef(truth_text, scored_decisions))
    except Exception:
        return np.nan


def _compute_threshold_heat_arrays(
    decision_df: pd.DataFrame,
    metric_name: str,
    coverage_metric_name: str,
    require_both_thresholds: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    conf_values = np.round(np.linspace(0.0, 1.0, 21), 2)
    vote_values = np.arange(0.0, 101.0, 5.0)
    heat = np.full((len(vote_values), len(conf_values)), np.nan, dtype=float)
    coverage_metric_heat = np.full((len(vote_values), len(conf_values)), np.nan, dtype=float)
    coverage_heat = np.full((len(vote_values), len(conf_values)), np.nan, dtype=float)

    valid_truth = ~decision_df["Ground Truth"].astype(str).str.lower().isin(["", "unknown", "na", "nan", "none"])
    base_df = decision_df[valid_truth].copy()
    if base_df.empty:
        return conf_values, vote_values, heat, coverage_metric_heat, coverage_heat

    truth = base_df["Ground Truth"].astype(str).to_numpy()
    selected_pred = base_df["Prediction"].fillna("").astype(str).to_numpy()
    ensemble_pred = base_df["Ensemble Raw Prediction"].fillna("").astype(str).to_numpy()
    ensemble_has_prediction = ~np.isin(
        pd.Series(ensemble_pred).str.strip().str.lower().to_numpy(),
        ["", "unknown", "na", "nan", "none"],
    )
    confidence = pd.to_numeric(base_df["Confidence"], errors="coerce").to_numpy(dtype=float)
    ensemble_vote = pd.to_numeric(base_df["Ensemble Raw Vote %"], errors="coerce").to_numpy(dtype=float)

    for y_idx, vote_threshold in enumerate(vote_values):
        ensemble_passes = np.isfinite(ensemble_vote) & (ensemble_vote >= float(vote_threshold))
        for x_idx, confidence_threshold in enumerate(conf_values):
            selected_passes = np.isfinite(confidence) & (confidence >= float(confidence_threshold))
            if require_both_thresholds:
                decisions = np.where(selected_passes & ensemble_passes, selected_pred, "Unknown")
            else:
                decisions = np.where(
                    selected_passes,
                    selected_pred,
                    np.where(ensemble_passes & ensemble_has_prediction, ensemble_pred, "Unknown"),
                )
            heat[y_idx, x_idx] = _threshold_metric_from_arrays(decisions, truth, metric_name)
            coverage_metric_heat[y_idx, x_idx] = _threshold_metric_from_arrays(decisions, truth, coverage_metric_name)
            coverage_heat[y_idx, x_idx] = _threshold_metric_from_arrays(decisions, truth, "Coverage")

    return conf_values, vote_values, heat, coverage_metric_heat, coverage_heat


def _render_coverage_threshold_plot(
    conf_values,
    vote_values,
    metric_heat,
    coverage_heat,
    metric_name: str,
    title_prefix: str,
    x_label: str,
) -> None:
    x_grid, y_grid = np.meshgrid(conf_values, vote_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        metric_heat,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            float(np.min(conf_values)),
            float(np.max(conf_values)),
            float(np.min(vote_values)),
            float(np.max(vote_values)),
        ],
        interpolation="nearest",
    )
    finite_metric = metric_heat[np.isfinite(metric_heat)]
    if finite_metric.size:
        im.set_clim(float(np.nanmin(finite_metric)), float(np.nanmax(finite_metric)))
    finite_coverage = coverage_heat[np.isfinite(coverage_heat)]
    if finite_coverage.size:
        levels = [
            level for level in [0.25, 0.50, 0.75, 0.90]
            if finite_coverage.min() <= level <= finite_coverage.max()
        ]
        if levels:
            contours = ax.contour(x_grid, y_grid, coverage_heat, levels=levels, colors="black", linewidths=1.1)
            ax.clabel(contours, inline=True, fontsize=8, fmt=lambda value: f"{100.0 * value:.0f}%")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Top-N validation ensemble vote threshold (%)")
    ax.set_title(f"{title_prefix}: {metric_name} heatmap with coverage contours")
    fig.colorbar(im, ax=ax, label=metric_name)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _prediction_frame_for_thresholds(pred_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    if pred_df is None or pred_df.empty or "label" not in pred_df.columns:
        return pd.DataFrame()

    prob_cols = [c for c in pred_df.columns if str(c).startswith("probs_")]
    if not prob_cols:
        return pd.DataFrame()

    probs = pred_df[prob_cols].apply(pd.to_numeric, errors="coerce")
    if probs.empty:
        return pd.DataFrame()

    labels = [str(c)[len("probs_"):] for c in prob_cols]
    best_idx = probs.to_numpy(dtype=float).argmax(axis=1)
    confidence = probs.max(axis=1)
    pred_from_probs = pd.Series([labels[i] for i in best_idx], index=pred_df.index)
    prediction = pred_df["pred"] if "pred" in pred_df.columns else pred_from_probs
    if "name" in pred_df.columns:
        sample_key = pred_df["name"].astype(str)
    else:
        sample_key = pd.Series([str(i) for i in range(len(pred_df))], index=pred_df.index)

    return pd.DataFrame(
        {
            "_sample_key": sample_key.astype(str),
            "Ground Truth": pred_df["label"].astype(str),
            f"Prediction {model_key}": prediction.astype(str),
            f"Confidence {model_key}": pd.to_numeric(confidence, errors="coerce"),
        }
    )


def _build_validation_threshold_frame(model_rows: pd.DataFrame, selected_idx: int, max_models: int) -> pd.DataFrame:
    if model_rows is None or model_rows.empty:
        return pd.DataFrame()

    selected_idx = max(0, min(int(selected_idx), len(model_rows) - 1))
    rows = model_rows.head(int(max_models)).copy().reset_index(drop=True)
    if selected_idx >= len(rows):
        selected_idx = 0

    frames = []
    model_keys = []
    for idx, row in rows.iterrows():
        pred_df, _path = _load_prediction_csv_for_row(row.to_dict(), "valid")
        frame = _prediction_frame_for_thresholds(pred_df, f"m{idx}")
        if frame.empty:
            continue
        frames.append(frame)
        model_keys.append(f"m{idx}")

    if not frames or f"m{selected_idx}" not in model_keys:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame.drop(columns=["Ground Truth"], errors="ignore"),
            on="_sample_key",
            how="inner",
        )
    if merged.empty:
        return pd.DataFrame()

    selected_key = f"m{selected_idx}"
    prediction_cols = [f"Prediction {key}" for key in model_keys if f"Prediction {key}" in merged.columns]
    if not prediction_cols:
        return pd.DataFrame()

    ensemble_predictions = []
    ensemble_votes = []
    for _, row in merged.iterrows():
        votes = [
            str(row.get(col) or "").strip()
            for col in prediction_cols
            if str(row.get(col) or "").strip()
        ]
        if not votes:
            ensemble_predictions.append("Unknown")
            ensemble_votes.append(np.nan)
            continue
        counts = pd.Series(votes).value_counts()
        winner = str(counts.index[0])
        ensemble_predictions.append(winner)
        ensemble_votes.append(100.0 * float(counts.iloc[0]) / float(len(votes)))

    return pd.DataFrame(
        {
            "Ground Truth": merged["Ground Truth"],
            "Prediction": merged[f"Prediction {selected_key}"],
            "Confidence": merged[f"Confidence {selected_key}"],
            "Ensemble Raw Prediction": ensemble_predictions,
            "Ensemble Raw Vote %": ensemble_votes,
        }
    )


def _render_threshold_heatmap_from_decisions(
    decision_df: pd.DataFrame,
    page_key: str,
    require_both_thresholds: bool,
    metric_name: str,
    coverage_metric_name: str,
) -> None:
    if decision_df is None or decision_df.empty:
        st.info("No validation predictions available for threshold heatmap.")
        return

    conf_values, vote_values, heat, coverage_metric_heat, coverage_heat = _compute_threshold_heat_arrays(
        decision_df,
        metric_name=metric_name,
        coverage_metric_name=coverage_metric_name,
        require_both_thresholds=require_both_thresholds,
    )
    if np.all(np.isnan(heat)):
        st.info("No validation labels available for threshold heatmap.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(conf_values))[::2])
    ax.set_xticklabels([f"{v:.1f}" for v in conf_values[::2]])
    ax.set_yticks(np.arange(len(vote_values))[::2])
    ax.set_yticklabels([f"{v:.0f}" for v in vote_values[::2]])
    ax.set_xlabel("Selected model validation confidence threshold")
    ax.set_ylabel("Top-N validation ensemble vote threshold (%)")
    ax.set_title(metric_name)
    fig.colorbar(im, ax=ax, label=metric_name)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    _render_coverage_threshold_plot(
        conf_values,
        vote_values,
        coverage_metric_heat,
        coverage_heat,
        coverage_metric_name,
        "Coverage-aware validation threshold plot",
        x_label="Selected model validation confidence threshold",
    )


def render_validation_threshold_heatmap(models_df: pd.DataFrame, page_key: str = "leaderboard") -> None:
    if models_df is None or models_df.empty:
        return

    with st.expander("Threshold Sensitivity Heatmap (validation)", expanded=False):
        max_default = min(5, len(models_df))
        metric_options = ["ACC kept", "ACC unknown=wrong", "Coverage", "MCC kept", "MCC unknown=wrong"]
        coverage_metric_options = ["ACC kept", "MCC kept", "ACC unknown=wrong", "MCC unknown=wrong"]

        with st.form(key=_k(page_key, "validation_threshold_form")):
            max_models = st.number_input(
                "Validation Top-N models",
                min_value=1,
                max_value=min(25, len(models_df)),
                value=max_default,
                step=1,
                key=_k(page_key, "validation_threshold_top_n"),
            )
            top_df = models_df.head(int(max_models)).copy().reset_index(drop=True)
            top_df["_validation_selection_key"] = top_df.apply(
                lambda row: _make_model_selection_key(row.to_dict()),
                axis=1,
            )
            selection_options = top_df["_validation_selection_key"].tolist()
            key_to_idx = {key: idx for idx, key in enumerate(selection_options)}
            selected_model_key = _k(page_key, "validation_threshold_selected_model")
            if st.session_state.get(selected_model_key) not in selection_options:
                st.session_state[selected_model_key] = selection_options[0] if selection_options else None
            selected_key = st.selectbox(
                "Selected validation model",
                options=selection_options,
                format_func=lambda key: (
                    f"#{top_df.iloc[key_to_idx[key]].get('#', key_to_idx[key] + 1)} | "
                    f"ID {top_df.iloc[key_to_idx[key]].get('Model ID', '?')}"
                ),
                key=selected_model_key,
            )
            require_both = st.checkbox(
                "Require both thresholds",
                value=bool(st.session_state.get("production_require_both_thresholds", False)),
                key=_k(page_key, "validation_threshold_require_both"),
            )
            metric_name = st.selectbox(
                "Heatmap metric",
                options=metric_options,
                index=0,
                key=_k(page_key, "validation_threshold_heatmap_metric"),
            )
            coverage_metric_name = st.selectbox(
                "Coverage-aware plot metric",
                options=coverage_metric_options,
                index=0,
                key=_k(page_key, "validation_threshold_coverage_plot_metric"),
            )
            render_clicked = st.form_submit_button("Render / update heatmap")

        cache_key = _k(page_key, "validation_threshold_cached")
        if render_clicked:
            selected_idx = key_to_idx.get(selected_key, 0)
            top_df = top_df.drop(columns=["_validation_selection_key"], errors="ignore")
            model_signature = tuple(_make_model_selection_key(row.to_dict()) for _, row in top_df.iterrows())
            config_signature = (
                model_signature,
                int(selected_idx),
                int(max_models),
                bool(require_both),
                str(metric_name),
                str(coverage_metric_name),
            )
            with st.spinner("Computing threshold heatmaps..."):
                decision_df = _build_validation_threshold_frame(top_df, int(selected_idx), int(max_models))
                conf_values, vote_values, heat, coverage_metric_heat, coverage_heat = _compute_threshold_heat_arrays(
                    decision_df,
                    metric_name=str(metric_name),
                    coverage_metric_name=str(coverage_metric_name),
                    require_both_thresholds=bool(require_both),
                )
            st.session_state[cache_key] = {
                "signature": config_signature,
                "metric_name": str(metric_name),
                "coverage_metric_name": str(coverage_metric_name),
                "conf_values": conf_values,
                "vote_values": vote_values,
                "heat": heat,
                "coverage_metric_heat": coverage_metric_heat,
                "coverage_heat": coverage_heat,
            }

        cached = st.session_state.get(cache_key)
        if not cached:
            st.info("Choose settings, then click Render / update heatmap.")
            return

        metric_name = cached["metric_name"]
        coverage_metric_name = cached["coverage_metric_name"]
        conf_values = cached["conf_values"]
        vote_values = cached["vote_values"]
        heat = cached["heat"]
        coverage_metric_heat = cached["coverage_metric_heat"]
        coverage_heat = cached["coverage_heat"]
        if np.all(np.isnan(heat)):
            st.info("No validation labels available for threshold heatmap.")
            return

        st.subheader("Threshold Sensitivity Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(conf_values))[::2])
        ax.set_xticklabels([f"{v:.1f}" for v in conf_values[::2]])
        ax.set_yticks(np.arange(len(vote_values))[::2])
        ax.set_yticklabels([f"{v:.0f}" for v in vote_values[::2]])
        ax.set_xlabel("Selected model validation confidence threshold")
        ax.set_ylabel("Top-N validation ensemble vote threshold (%)")
        ax.set_title(metric_name)
        fig.colorbar(im, ax=ax, label=metric_name)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        _render_coverage_threshold_plot(
            conf_values,
            vote_values,
            coverage_metric_heat,
            coverage_heat,
            coverage_metric_name,
            "Coverage-aware validation threshold plot",
            x_label="Selected model validation confidence threshold",
        )
        return


def _render_prediction_roc_curves(pred_df: pd.DataFrame, title: str) -> bool:
    if pred_df is None or pred_df.empty or "label" not in pred_df.columns:
        st.info("No prediction labels available for ROC plotting.")
        return False

    prob_cols = [c for c in pred_df.columns if str(c).startswith("probs_")]
    if not prob_cols:
        st.info("No saved probability columns available for ROC plotting.")
        return False

    fig, ax = plt.subplots(figsize=(7, 4.2))
    curves = 0
    truth = pred_df["label"].astype(str).str.strip()
    for col in prob_cols:
        label = str(col)[len("probs_"):]
        scores = pd.to_numeric(pred_df[col], errors="coerce")
        valid = scores.notna() & truth.ne("")
        if valid.sum() < 2:
            continue
        y_true = truth[valid].str.lower().eq(str(label).strip().lower()).astype(int).values
        if len(np.unique(y_true)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, scores[valid].astype(float).values)
            auc_value = sk_auc(fpr, tpr)
        except Exception:
            continue
        ax.plot(fpr, tpr, linewidth=1.7, label=f"{label} vs rest (AUC={auc_value:.3f})")
        curves += 1

    if curves == 0:
        plt.close(fig)
        st.info("Not enough positive/negative samples to draw ROC curves.")
        return False

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.5")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    return True


def _render_best_models_auc_curves(models_df: pd.DataFrame, page_key: str, split: str = "valid") -> None:
    if models_df is None or models_df.empty:
        return

    st.subheader("AUC / ROC Curves")
    options, key_to_row, key_to_label = _make_model_selection_options(models_df)
    if not options:
        st.info("No selectable model rows available for ROC curves.")
        return

    selected_key = st.session_state.get(_k(page_key, "best_model_key"))
    if selected_key not in options:
        selected_key = st.session_state.get("selected_model_selection_key")
    if selected_key not in options:
        selected_key = options[0]

    row = key_to_row.get(selected_key)
    row_dict = row.to_dict() if row is not None else {}
    pred_df, path = _load_prediction_csv_for_row(row_dict, split)
    if pred_df is None:
        st.info(f"No `{split}_predictions.csv` artifact found for this model.")
        return
    st.caption(f"Using `{path}`")
    _render_prediction_roc_curves(pred_df, f"Model #{row_dict.get('#', '?')} {split} ROC curves")


def _render_top_models_filter_controls(models_df: pd.DataFrame, page_key: str) -> None:
    if models_df is None or models_df.empty:
        return

    excluded = {"Log Path", "Artifact Log Path", "Best Model Dir", "Source Run Path", "split_config_key"}
    candidate_columns = [
        col for col in models_df.columns
        if col not in excluded and models_df[col].nunique(dropna=True) > 1
    ]
    if not candidate_columns:
        return

    preferred = [
        "N_Calibration", "NNeg", "NPos", "Classif_Loss", "DLoss", "Prototypes",
        "Normalize", "N_Neighbors", "Model Name", "Task", "NSize", "FGSM",
        "Dist_Fct", "Proto_Strat", "Proto_Comp", "train_datasets",
        "valid_dataset", "test_dataset",
    ]
    ordered_columns = [c for c in preferred if c in candidate_columns]
    ordered_columns += [c for c in candidate_columns if c not in ordered_columns]

    def _is_integer_series(series: pd.Series) -> bool:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return False
        return bool(np.all(np.isclose(numeric, np.round(numeric), rtol=0.0, atol=1e-12)))

    def _is_numeric_filter_column(series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return False
        numeric = pd.to_numeric(non_null, errors="coerce")
        return bool(numeric.notna().mean() >= 0.9)

    def _filter_key(col: str, suffix: str) -> str:
        safe_col = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(col))
        return _k(page_key, f"top_models_filter_{safe_col}_{suffix}")

    with st.expander("Filter Top Models", expanded=False):
        filter_columns = st.multiselect(
            "Restrict by columns",
            options=ordered_columns,
            default=st.session_state.get(_k(page_key, "top_models_filter_columns"), []),
            key=_k(page_key, "top_models_filter_columns"),
        )

        filtered_df = models_df.copy()
        for col in filter_columns:
            if col not in filtered_df.columns:
                continue
            if _is_numeric_filter_column(filtered_df[col]):
                numeric_values = pd.to_numeric(filtered_df[col], errors="coerce")
                available = numeric_values.dropna()
                if available.empty:
                    continue
                min_value = float(available.min())
                max_value = float(available.max())
                is_integer = _is_integer_series(filtered_df[col])
                operators = [">=", "<="]
                if is_integer:
                    operators.append("=")
                operator = st.selectbox(
                    f"{col} filter",
                    options=operators,
                    index=0,
                    key=_filter_key(col, "operator"),
                )
                default_value = min_value if operator == ">=" else max_value
                if operator == "=" and is_integer:
                    value = st.number_input(
                        f"{col} value",
                        value=int(round(default_value)),
                        min_value=int(np.floor(min_value)),
                        max_value=int(np.ceil(max_value)),
                        step=1,
                        key=_filter_key(col, "value"),
                    )
                    filtered_df = filtered_df[numeric_values.eq(float(value))]
                else:
                    value = st.number_input(
                        f"{col} value",
                        value=float(default_value),
                        min_value=float(min_value),
                        max_value=float(max_value),
                        step=float(max((max_value - min_value) / 100.0, 0.0001)),
                        format="%.6f",
                        key=_filter_key(col, "value"),
                    )
                    if operator == ">=":
                        filtered_df = filtered_df[numeric_values >= float(value)]
                    else:
                        filtered_df = filtered_df[numeric_values <= float(value)]
            else:
                values = filtered_df[col].dropna().map(str).sort_values().unique().tolist()
                if not values:
                    continue
                selected_values = st.multiselect(
                    f"{col} values",
                    options=values,
                    default=values,
                    key=_filter_key(col, "values"),
                )
                if selected_values:
                    filtered_df = filtered_df[filtered_df[col].map(str).isin(set(selected_values))]
                else:
                    filtered_df = filtered_df.iloc[0:0]

        if filter_columns:
            st.caption(f"Draft restriction matches {len(filtered_df)} / {len(models_df)} models. Click Apply restrictions to update the table.")

        col_apply, col_clear = st.columns([1, 1])
        with col_apply:
            apply_clicked = st.button("Apply restrictions", key=_k(page_key, "top_models_filter_apply"))
        with col_clear:
            clear_clicked = st.button("Clear restrictions", key=_k(page_key, "top_models_filter_clear"))

        if clear_clicked:
            st.session_state.pop("top_models_filtered_selection_keys", None)
            st.session_state["top_models_filters_active"] = False
            st.rerun()
        elif apply_clicked:
            if filter_columns:
                st.session_state["top_models_filtered_selection_keys"] = [
                    _make_model_selection_key(row.to_dict()) for _, row in filtered_df.iterrows()
                ]
                st.session_state["top_models_filters_active"] = True
            else:
                st.session_state.pop("top_models_filtered_selection_keys", None)
                st.session_state["top_models_filters_active"] = False
            st.rerun()

        if st.session_state.get("top_models_filters_active"):
            active_df = apply_active_top_models_selection(models_df)
            st.caption(f"Showing {len(active_df)} / {len(models_df)} models after applied filters.")


def apply_active_top_models_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the user-pruned Top Models selection to another dataframe."""
    return apply_selection_keys_to_models_df(
        df,
        st.session_state.get("top_models_filtered_selection_keys"),
        filters_active=bool(st.session_state.get("top_models_filters_active")),
    )


def _render_calibration_vs_performance(models_df: pd.DataFrame, split: str = "valid") -> None:
    st.markdown("---")
    st.subheader("📈 Calibration vs Performance")

    plot_df = models_df.copy()
    calibration_errors = []
    calibration_missing = 0
    # Ensure ECE and Brier columns exist and are numeric, compute if missing
    for idx, row in plot_df.iterrows():
        for col, metric_key in [("ECE", "ece"), ("Brier", "brier")]:
            val = row.get(col, np.nan)
            if pd.isna(val):
                log_path = row.get("Log Path")
                if log_path:
                    metrics = get_calibration_metrics(log_path, split=split)
                    if metrics and not metrics.get("error"):
                        plot_df.at[idx, "ECE"] = metrics.get("ece", np.nan)
                        plot_df.at[idx, "Brier"] = metrics.get("brier", np.nan)
                    elif metrics and metrics.get("error"):
                        calibration_errors.append({
                            "Model ID": row.get("Model ID"),
                            "Model": row.get("Model Name"),
                            "Reason": metrics.get("error"),
                            "Log Path": log_path,
                        })
                    else:
                        calibration_missing += 1
                else:
                    calibration_missing += 1
    for col in ["ECE", "Brier"]:
        if col not in plot_df.columns:
            plot_df[col] = np.nan
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    if "Valid MCC" in plot_df.columns:
        plot_df["Calibration Plot MCC"] = pd.to_numeric(plot_df["Valid MCC"], errors="coerce")
    else:
        plot_df["Calibration Plot MCC"] = np.nan
    if "MCC" in plot_df.columns:
        plot_df["Calibration Plot MCC"] = plot_df["Calibration Plot MCC"].fillna(
            pd.to_numeric(plot_df["MCC"], errors="coerce")
        )
    plot_df = plot_df.dropna(subset=["Calibration Plot MCC", "ECE", "Brier"]).copy()

    st.caption(
        f"Calibration points available: {len(plot_df)} / {len(models_df)} models. "
        f"A point requires saved {split} probabilities in {split}_predictions.csv."
    )

    if plot_df.empty:
        st.info("Calibration metrics not available for plotting. Run models to compute ECE and Brier scores.")
        if calibration_errors or calibration_missing:
            with st.expander("Calibration rows skipped"):
                rows = calibration_errors[:]
                if calibration_missing:
                    rows.append({"Reason": f"{calibration_missing} rows had no {split}_predictions.csv calibration artifact"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Valid MCC vs {split} ECE**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(plot_df["Calibration Plot MCC"], plot_df["ECE"], alpha=0.6, s=50)
        ax.set_xlabel("Valid MCC (higher is better)", fontsize=10)
        ax.set_ylabel("ECE (lower is better)", fontsize=10)
        ax.set_title(f"{split.title()} Expected Calibration Error vs Valid MCC", fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if calibration_errors or calibration_missing:
        with st.expander("Calibration rows skipped"):
            rows = calibration_errors[:]
            if calibration_missing:
                rows.append({"Reason": f"{calibration_missing} rows had no {split}_predictions.csv calibration artifact"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with col2:
        st.markdown(f"**Valid MCC vs {split} Brier Score**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(plot_df["Calibration Plot MCC"], plot_df["Brier"], alpha=0.6, s=50)
        ax.set_xlabel("Valid MCC (higher is better)", fontsize=10)
        ax.set_ylabel("Brier Score (lower is better)", fontsize=10)
        ax.set_title(f"{split.title()} Brier Score vs Valid MCC", fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _make_model_selection_options(models_df: pd.DataFrame):
    key_to_row = {}
    key_to_label = {}

    for _, row in models_df.iterrows():
        row_dict = row.to_dict()
        selection_key = _make_model_selection_key(row_dict)

        if selection_key in key_to_row:
            continue

        key_to_row[selection_key] = row

        model_num = row_dict.get("#", "?")
        model_name = row_dict.get("Model Name")
        mcc = _safe_metric(row_dict.get("MCC"), digits=3)
        valid_mcc = _safe_metric(row_dict.get("Valid MCC"), digits=3)
        dist_fct = row_dict.get("Dist_Fct")
        normalize = row_dict.get("Normalize")

        key_to_label[selection_key] = (
            f"#{model_num} - {model_name} "
            f"(MCC={mcc}, valid={valid_mcc}, dist_fct={dist_fct}, normalize={normalize})"
        )

    options = _unique_preserve_order(list(key_to_row.keys()))
    return options, key_to_row, key_to_label


def _render_selected_model_calibration(row_dict: Dict[str, Any], split: str = "valid") -> None:
    log_path = row_dict.get("Log Path")

    if not log_path:
        return

    st.subheader(f"📈 Calibration Curve (Model #{row_dict.get('#', '?')}, {split})")

    metrics = get_calibration_metrics(log_path, split=split)

    if metrics is None:
        st.info("No calibration metrics available for this model.")
        return

    if metrics.get("error"):
        st.warning(f"Could not load calibration metrics: {metrics['error']}")
        return

    ece_val = metrics.get("ece")
    brier_val = metrics.get("brier")
    prob_true = np.array(metrics.get("prob_true", []), dtype=float)
    prob_pred = np.array(metrics.get("prob_pred", []), dtype=float)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Expected Calibration Error (ECE)", _safe_metric(ece_val))
    with c2:
        st.metric("Brier Score", _safe_metric(brier_val))

    if ece_val is not None:
        if ece_val < 0.05:
            st.success("✅ The model is well-calibrated.")
        elif ece_val < 0.15:
            st.warning("⚠️ The model shows moderate calibration error.")
        else:
            st.error("❌ The model is poorly calibrated.")

    if len(prob_true) < 2 or len(prob_pred) < 2:
        st.info("Not enough bins to render a calibration curve (need at least 2 points).")
        return

    df_curve = pd.DataFrame(
        {
            "prob_pred": prob_pred,
            "prob_true": prob_true,
        }
    ).sort_values("prob_pred")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(
        df_curve["prob_pred"],
        df_curve["prob_true"],
        marker="o",
        linewidth=1.5,
        label=f"Model #{row_dict.get('#', '?')} (ECE: {_safe_metric(ece_val)}, Brier: {_safe_metric(brier_val)})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _set_selected_model(row_dict: Dict[str, Any], selected_key: str) -> None:
    row_dict = dict(row_dict)
    row_dict.update(extract_params_from_log_path(row_dict.get("Best Model Dir") or row_dict.get("Log Path")))

    row_dict["model_id"] = row_dict.get("Model ID")

    if "N_Neighbors" in row_dict:
        row_dict["n_neighbors"] = row_dict["N_Neighbors"]
    if "NSize" in row_dict:
        row_dict["new_size"] = row_dict["NSize"]
    if "Dist_Fct" in row_dict:
        row_dict["dist_fct"] = row_dict["Dist_Fct"]
    if "Classif_Loss" in row_dict:
        row_dict["classif_loss"] = row_dict["Classif_Loss"]

    st.session_state.selected_model_params = row_dict
    st.session_state.selected_params_version = st.session_state.get("selected_params_version", 0) + 1
    st.session_state.selected_model_log_path = row_dict.get("Log Path")
    st.session_state.selected_model_selection_key = selected_key
    st.session_state.selected_model_version = st.session_state.get("selected_model_version", 0) + 1


def _render_model_selector(models_df: pd.DataFrame, page_key: str, calibration_split: str = "valid") -> None:
    if models_df.empty:
        st.warning("No models found in leaderboard.")
        return

    st.markdown("---")

    options, key_to_row, key_to_label = _make_model_selection_options(models_df)

    if not options:
        st.warning("No selectable models found.")
        return

    widget_key = _k(page_key, "best_model_key")

    canonical_key = st.session_state.get("selected_model_selection_key")
    if canonical_key and canonical_key in options:
        last = st.session_state.get(_k(page_key, "best_model_last_sync"))
        version = st.session_state.get("selected_model_version")
        if version is not None and version != last:
            st.session_state[widget_key] = canonical_key
            st.session_state[_k(page_key, "best_model_last_sync")] = version

    current_value = st.session_state.get(widget_key)
    if current_value not in options:
        st.session_state[widget_key] = options[0]

    selected_key = st.selectbox(
        "Select a model to use:",
        options=options,
        format_func=lambda k: key_to_label.get(k, str(k)),
        key=widget_key,
    )

    row = key_to_row.get(selected_key)
    if row is not None:
        row_dict = row.to_dict()
        _render_selected_model_calibration(row_dict, split=calibration_split)

    if st.button("✅ Use Selected Model", key=_k(page_key, "use_selected_model_btn")):
        row = key_to_row.get(selected_key)

        if row is None:
            st.error("Could not resolve selected model.")
        else:
            row_dict = row.to_dict()
            _set_selected_model(row_dict, selected_key)

            st.success(f"✅ Selected model #{row_dict.get('#', '?')}: {row_dict.get('Model Name')}")
            st.info("Switch to '🔬 New Analysis' tab to upload an image and run analysis with this model.")
            st.rerun()

    if st.session_state.get("selected_model_params"):
        st.info("ℹ️ Model parameters loaded. Check sidebar for current settings. Go to '🔬 New Analysis' tab to run analysis.")


def _render_cached_optimization(page_key: str) -> None:
    current_model_key = (
        st.session_state.get("selected_model_selection_key")
        or st.session_state.get(_k(page_key, "best_model_key"))
        or st.session_state.get("sidebar_best_model_key")
    )

    if st.session_state.get(_k(page_key, "k_opt_model_key")) not in (None, current_model_key):
        st.session_state.pop(_k(page_key, "optimized_k_value"), None)
        st.session_state.pop(_k(page_key, "k_opt_best_mcc"), None)
        st.session_state.pop(_k(page_key, "k_opt_curve"), None)
        st.session_state[_k(page_key, "k_opt_model_key")] = current_model_key

    # Backward-compatible fallback to old global keys.
    optimized_value = st.session_state.get(
        _k(page_key, "optimized_k_value"),
        st.session_state.get("optimized_k_value"),
    )
    best_mcc = st.session_state.get(
        _k(page_key, "k_opt_best_mcc"),
        st.session_state.get("k_opt_best_mcc"),
    )
    mcc_curve = st.session_state.get(
        _k(page_key, "k_opt_curve"),
        st.session_state.get("k_opt_curve", []),
    )
    proto_results = st.session_state.get(
        _k(page_key, "k_opt_proto_results"),
        st.session_state.get("k_opt_proto_results", {}),
    )

    if optimized_value is None:
        st.info(
            "No cached KNN/prototype optimization found yet. "
            "If your original optimizer button was in the old app.py, move that block here next."
        )
        return

    if best_mcc is not None:
        st.success(f"✅ Previous Optimization Found: {optimized_value} (Validation MCC: {best_mcc:.3f})")
    else:
        st.success(f"✅ Previous Optimization Found: {optimized_value}")

    if not (mcc_curve and proto_results):
        return

    try:
        curve_df = pd.DataFrame(mcc_curve).sort_values("k")

        if curve_df.empty or "mcc" not in curve_df.columns:
            return

        fig, ax = plt.subplots(figsize=(6, 3.5))

        ax.plot(
            curve_df["k"],
            curve_df["mcc"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            label="KNN",
            zorder=3,
        )

        for strategy in ["mean", "kmeans", "gmm"]:
            result = proto_results.get(strategy, {})
            per_components = result.get("per_components", [])

            if not per_components:
                continue

            per_df = pd.DataFrame(per_components).sort_values("n_components")

            ax.plot(
                per_df["n_components"],
                per_df["mcc"],
                marker="s",
                linewidth=2.5,
                markersize=7,
                label=f"{strategy.capitalize()}",
                zorder=2,
            )

        if best_mcc is not None:
            ax.axhline(
                y=best_mcc,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                zorder=1,
            )

        ax.set_xlabel("k (KNN) / n_components (Prototypes)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Validation MCC", fontsize=12, fontweight="bold")
        ax.set_title("KNN vs Prototype Strategies: MCC Comparison", fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        all_mccs = list(curve_df["mcc"].dropna().values)
        for result in proto_results.values():
            if result.get("best_mcc") is not None:
                all_mccs.append(result.get("best_mcc"))

        if all_mccs:
            ymin = max(-1.0, min(all_mccs) - 0.05)
            ymax = min(1.0, max(all_mccs) + 0.1)
            if ymin < ymax:
                ax.set_ylim([ymin, ymax])

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except Exception as e:
        st.warning(f"Could not display cached chart: {e}")


def _render_pca_with_prototypes(args, page_key: str) -> None:
    with st.expander("🧭 PCA with Prototypes", expanded=False):
        st.caption("Compute PCA of encoded representations for the current sidebar model and overlay class prototypes.")

        pca_cols = st.columns([2, 2, 2])

        with pca_cols[0]:
            proto_strategies = st.multiselect(
                "Prototype Aggregation",
                options=["mean", "kmeans", "gmm"],
                default=["mean"],
                key=_k(page_key, "pca_proto_strategy"),
                help=(
                    "Select one or more: mean (single average per class), "
                    "kmeans (k centers per class), or gmm (gaussian mixture components per class)."
                ),
            )

        with pca_cols[1]:
            proto_components = st.slider(
                "Components per Class",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                key=_k(page_key, "pca_proto_components"),
                help="Number of prototypes/centroids/components per class.",
            )

        with pca_cols[2]:
            st.empty()

        if not proto_strategies:
            st.warning("Please select at least one prototype aggregation strategy.")
            proto_strategies = ["mean"]

        strategies_str = "_".join(sorted(proto_strategies))
        model_path_base = os.path.basename(getattr(args, "path", "") or "no_path")

        current_model_key = (
            f"{getattr(args, 'task', 'task')}_"
            f"{model_path_base}_"
            f"{getattr(args, 'new_size', 'size')}_"
            f"{getattr(args, 'n_neighbors', 'nn')}_"
            f"{getattr(args, 'dist_fct', 'dist')}_"
            f"{strategies_str}_"
            f"{proto_components}"
        )

        cache_key_model = _k(page_key, "pca_model_key")
        cache_key_fig = _k(page_key, "pca_fig_bytes")

        has_pca_cache = (
            st.session_state.get(cache_key_model) == current_model_key
            and st.session_state.get(cache_key_fig)
        )

        col_pca_btn, col_pca_status = st.columns([3, 2])

        with col_pca_btn:
            if st.button("Compute PCA (encodings + prototypes)", key=_k(page_key, "compute_pca")):
                st.session_state[cache_key_fig] = None
                st.session_state[cache_key_model] = None

                with st.spinner("Computing PCA on encodings..."):
                    try:
                        fig_bytes = compute_pca_for_args(
                            args,
                            proto_strategies=proto_strategies,
                            proto_components=proto_components,
                        )
                        st.session_state[cache_key_fig] = fig_bytes
                        st.session_state[cache_key_model] = current_model_key
                        st.rerun()
                    except Exception as e:
                        st.error(f"PCA failed: {e}")

        with col_pca_status:
            if has_pca_cache:
                st.success("✅ Cached")

        if has_pca_cache:
            st.info("Displaying cached PCA from previous run.")
            st.image(st.session_state[cache_key_fig], use_container_width=True)


def _render_model_usage_summary(cursor) -> None:
    st.subheader("📊 Models Used for Analysis")

    try:
        # Get the active task from session state or args if available
        active_task = st.session_state.get("production_task") or None

        cursor.execute(
            """
            SELECT (
                       SELECT bmr.id
                       FROM best_models_registry bmr
                       WHERE bmr.model_name = mus.model_name
                         AND bmr.task = mus.task
                         AND CAST(bmr.nsize AS CHAR) = CAST(mus.nsize AS CHAR)
                         AND CAST(bmr.fgsm AS CHAR) = CAST(mus.fgsm AS CHAR)
                         AND CAST(bmr.normalize AS CHAR) = CAST(mus.normalize AS CHAR)
                         AND CAST(bmr.n_calibration AS CHAR) = CAST(mus.n_calibration AS CHAR)
                         AND CAST(bmr.classif_loss AS CHAR) = CAST(mus.classif_loss AS CHAR)
                         AND CAST(bmr.dloss AS CHAR) = CAST(mus.dloss AS CHAR)
                         AND CAST(bmr.prototypes AS CHAR) = CAST(mus.prototypes AS CHAR)
                         AND CAST(bmr.npos AS CHAR) = CAST(mus.npos AS CHAR)
                         AND CAST(bmr.nneg AS CHAR) = CAST(mus.nneg AS CHAR)
                         AND CAST(bmr.n_neighbors AS CHAR) = CAST(mus.n_neighbors AS CHAR)
                       ORDER BY bmr.mcc DESC, bmr.id ASC
                       LIMIT 1
                   ) AS model_id,
                   mus.model_name, mus.task, mus.nsize, mus.fgsm, mus.normalize,
                   mus.n_calibration, mus.classif_loss, mus.dloss, mus.prototypes,
                   mus.npos, mus.nneg, mus.n_neighbors, mus.num_samples_analyzed,
                   mus.last_used
            FROM model_usage_summary mus
            ORDER BY mus.last_used DESC
            """
        )

        usage_rows = cursor.fetchall() or []

        if not usage_rows:
            st.info("No models have been used for analysis yet.")
            return

        usage_columns = [
            "Model ID", "Model", "Task", "Size", "FGSM", "Normalize", "N_Cal",
            "Loss", "DLoss", "Prototypes", "NPos", "NNeg",
            "N_Neighbors", "Samples", "Last Used",
        ]

        usage_df = pd.DataFrame(usage_rows, columns=usage_columns)

        # Filter by active task if set
        if active_task is not None:
            usage_df = usage_df[usage_df["Task"] == active_task]

        st.markdown("**Table:** Model Usage Summary")
        st.dataframe(_arrow_safe_dataframe(usage_df), use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load model usage summary: {e}")


# -------------------------------------------------
# Public render entrypoint
# -------------------------------------------------

def render(
    ctx,
    page_key: str = "leaderboard",
    title: str = "🏆 Best Models",
    include_model_table: bool = True,
    include_calibration: bool = True,
    include_model_selector: bool = True,
    include_embedding_tools: bool = False,
    include_usage_summary: bool = True,
):
    """
    Render the leaderboard / learned-embedding page.

    page_key is important: it namespaces Streamlit widget keys so this function
    can be called from multiple tabs without DuplicateWidgetID errors.
    """
    cursor = ctx.cursor
    args = ctx.args
    active_task = st.session_state.get("production_task") or getattr(args, "task", None)

    st.header(title)
    if active_task:
        st.caption(f"Task: {active_task}")

    if include_model_table or include_calibration or include_model_selector:
        try:
            calibration_split_key = _k(page_key, "calibration_split")
            calibration_split_options = ["valid", "test", "train"]
            calibration_split = str(st.session_state.get(calibration_split_key, "valid"))
            if calibration_split not in calibration_split_options:
                calibration_split = "valid"
            models_df = load_best_models_table(cursor, task=active_task, calibration_split=calibration_split)
            if models_df.empty:
                if active_task:
                    st.warning(f"No models found in leaderboard for task {active_task}.")
                else:
                    st.warning("No models found in leaderboard.")
            else:
                _update_model_number_map(models_df)

                display_models_df = apply_active_top_models_selection(models_df).reset_index(drop=True)
                if display_models_df.empty:
                    st.info("No models match the selected Top Models filters.")
                    _render_top_models_filter_controls(models_df, page_key)
                else:
                    _update_model_number_map(display_models_df)

                    if include_model_table:
                        _render_top_models_table(display_models_df, args)

                    _render_top_models_filter_controls(models_df, page_key)

                    calibration_split = st.selectbox(
                        "Calibration / ROC split",
                        options=calibration_split_options,
                        index=calibration_split_options.index(calibration_split),
                        key=calibration_split_key,
                    )
                    if calibration_split_key != "leaderboard_calibration_split":
                        st.session_state["leaderboard_calibration_split"] = calibration_split

                    if include_model_selector:
                        _render_model_selector(display_models_df, page_key, calibration_split=calibration_split)

                    if include_model_table:
                        _render_best_models_auc_curves(display_models_df, page_key, split=calibration_split)
                        render_validation_threshold_heatmap(display_models_df, page_key)

                    if include_calibration:
                        _render_calibration_vs_performance(display_models_df, split=calibration_split)

        except Exception as e:
            st.error(f"Could not load best models leaderboard: {e}")

    if include_embedding_tools:
        st.divider()
        st.subheader("🔧 KNN Optimization")
        st.caption("Optimize k/prototype heads on the validation split for the currently selected sidebar model.")

        _render_cached_optimization(page_key)

        st.divider()
        _render_pca_with_prototypes(args, page_key)

    if include_usage_summary:
        st.divider()
        _render_model_usage_summary(cursor)
