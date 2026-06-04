"""
Utility functions for the Streamlit app.

This module contains helper functions for:
- String/filename processing
- Model parameter path construction
- Database operations
- Calibration metrics computation
- Model selection key generation
"""

import os
import json
import math
import random
import hashlib
import re
import glob
import numpy as np
import pandas as pd
import streamlit as st
import pickle

from otitenet.app.artifact_registry import preferred_model_artifact_dir

# ---- String/Filename Utilities ---- #

def set_random_seeds(seed=1, deterministic=False):
    """Set app-level random seeds without importing training-only dependencies."""
    try:
        seed_int = int(seed)
    except Exception:
        seed_int = 1

    random.seed(seed_int)
    np.random.seed(seed_int)

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def strip_extension(filename):
    """Remove common image extensions from filename."""
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.JPG', '.PNG', '.JPEG']:
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename


def ensure_int(val):
    """Safely convert value to int by stripping common parameter prefixes."""
    if val is None:
        return 0
    s = str(val).lower()
    for prefix in ['npos', 'nneg', 'nsize', 'fgsm', 'ncal', 'n_neighbors', 'bs']:
        if s.startswith(prefix):
            s = s[len(prefix):]
    try:
        return int(float(s))
    except ValueError:
        return 0


def _unique_preserve_order(items):
    """Return a list of unique items preserving first-seen order."""
    return list(dict.fromkeys(items))


def normalize_train_datasets(value) -> str:
    """Canonical comma-separated training dataset combination."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [str(x).strip() for x in value]
    else:
        parts = [x.strip() for x in str(value).replace(";", ",").split(",")]
    return ",".join(_unique_preserve_order([x for x in parts if x and x.lower() not in {"none", "nan", "null"}]))


LEGACY_TRAIN_DATASETS = "Banque_Comert_Turquie_2020_jpg,Banque_Calaman_USA_2020_trie_CM"
LEGACY_VALID_DATASET = "Banque_Viscaino_Chili_2020"


def legacy_split_config(valid_dataset="") -> dict:
    valid = str(valid_dataset or LEGACY_VALID_DATASET).strip()
    train = normalize_train_datasets(LEGACY_TRAIN_DATASETS)
    return {
        "train_datasets": train,
        "valid_dataset": valid,
        "test_dataset": valid,
        "split_config_key": split_config_key(train, valid, valid),
    }


def split_config_values(_args) -> dict:
    """Return train/valid/test dataset identifiers for a model run."""
    return {
        "train_datasets": normalize_train_datasets(getattr(_args, "train_datasets", "")),
        "valid_dataset": str(getattr(_args, "valid_dataset", "") or "").strip(),
        "test_dataset": str(getattr(_args, "test_dataset", "") or "").strip(),
    }


def split_config_key(train_datasets="", valid_dataset="", test_dataset="") -> str:
    return "|".join([
        normalize_train_datasets(train_datasets),
        str(valid_dataset or "").strip(),
        str(test_dataset or "").strip(),
    ])


def split_combo_key_from_row(row) -> str:
    if row is None:
        return ""
    getter = row.get if hasattr(row, "get") else lambda _key, _default=None: _default
    key = getter("split_config_key")
    if key and str(key).strip() not in {"", "None", "nan", "null"}:
        return str(key).strip().replace(";", ",")
    return split_config_key(
        getter("train_datasets") or getter("Train Datasets") or "",
        getter("valid_dataset") or getter("Valid Dataset") or "",
        getter("test_dataset") or getter("Test Dataset") or "",
    )


def split_combo_key_to_values(combo_key: str) -> tuple[str, str, str]:
    return tuple((str(combo_key or "").split("|") + ["", "", ""])[:3])


def split_config_segment(train_datasets="", valid_dataset="", test_dataset="") -> str:
    """Compact path segment that separates best-model artifacts by split combo."""
    key = split_config_key(train_datasets, valid_dataset, test_dataset)
    if not key.replace("|", ""):
        return ""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"split_{digest}"


# ---- Model Parameter Path Construction ---- #

def dataset_path_segment(path: str) -> str:
    """Return the dataset path segment to store under best_models."""
    text = str(path or "").strip().replace("\\", "/").strip("/")
    for prefix in ("./data/", "data/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    # Preserve dataset subdirectory if present (e.g., otite_ds_64/USA_Turquie_Chili)
    return text or "otite_ds_64"

def get_model_params_path(_args):
    """Construct the standardized relative path for model parameters folders."""
    nsize = ensure_int(_args.new_size)
    fgsm = ensure_int(_args.fgsm)
    ncal = ensure_int(_args.n_calibration)
    npos = ensure_int(_args.n_positives)
    nneg = ensure_int(_args.n_negatives)
    n_neighbors = ensure_int(_args.n_neighbors)
    
    dataset_name = dataset_path_segment(_args.path)
    
    # Strip "prototypes_" prefix if present (training may add it)
    proto_val = str(_args.prototypes_to_use)
    if proto_val.startswith("prototypes_"):
        proto_val = proto_val[len("prototypes_"):]
    
    # Get distance function and normalize values
    dist_fct_val = str(getattr(_args, 'dist_fct', 'euclidean'))
    normalize_val = str(getattr(_args, 'normalize', 'no'))
    
    # Strategy and components for prototypes
    proto_strat = getattr(_args, 'prototype_strategy', 'mean')
    proto_comp = getattr(_args, 'prototype_components', 1)

    split_segment = ""
    if bool(getattr(_args, "split_config_in_path", False) or getattr(_args, "_split_config_in_path", False)):
        split_vals = split_config_values(_args)
        split_segment = split_config_segment(**split_vals)

    params = f'{dataset_name}/'
    if split_segment:
        params += f'{split_segment}/'
    params += f'nsize{nsize}/fgsm{fgsm}/ncal{ncal}/' \
             f'{_args.classif_loss}/{_args.dloss}/prototypes_{proto_val}/' \
             f'npos{npos}/nneg{nneg}/' \
             f'protoagg_{proto_strat}_{proto_comp}/' \
             f'norm{normalize_val}/' \
             f'dist_{dist_fct_val}/' \
             f'knn{n_neighbors}'
    return params


def extract_params_from_log_path(log_path: str):
    """Derive leaderboard defaults from the stored best-model log path."""
    params = {}
    if not log_path:
        return params
    parts = log_path.strip("/").split("/")
    try:
        base_idx = parts.index("best_models")
    except ValueError:
        return params

    if len(parts) > base_idx + 1:
        params["Task"] = parts[base_idx + 1]
    if len(parts) > base_idx + 2:
        params["Model Name"] = parts[base_idx + 2]

    raw_data_parts = parts[base_idx + 3:]
    dataset_parts = []
    data_parts = []
    for p in raw_data_parts:
        if p.startswith("nsize"):
            data_parts.append(p)
            continue
        if p.startswith("split_") and re.match(r"^split_[0-9a-f]{12}$", p):
            params["Split Segment"] = p
            params["_split_config_in_path"] = True
            data_parts.append(p)
            continue
        if data_parts:
            data_parts.append(p)
        else:
            dataset_parts.append(p)

    if dataset_parts:
        joined = "/".join(dataset_parts)
        if joined != "otite_ds_-1":
            params["Dataset"] = joined

    # Rest of the segments are positional but we can be more robust by checking prefixes.
    for p in data_parts:
        if p.startswith("otite_ds_") and p != "otite_ds_-1":
            params["Dataset"] = p
        elif p.startswith("nsize"):
            params["new_size"] = p[len("nsize"):]
        elif p.startswith("fgsm"):
            params["FGSM"] = p[len("fgsm"):]
        elif p.startswith("ncal"):
            params["N_Calibration"] = p[len("ncal"):]
        elif p.startswith("prototypes_"):
            params["Prototypes"] = p[len("prototypes_"):]
        elif p.startswith("npos"):
            params["NPos"] = p[len("npos"):]
        elif p.startswith("nneg"):
            params["NNeg"] = p[len("nneg"):]
        elif p.startswith("norm"):
            params["Normalize"] = p[len("norm"):]
        elif p.startswith("dist_"):
            params["Dist_Fct"] = p[len("dist_"):]
        elif p.startswith("knn"):
            params["N_Neighbors"] = p[len("knn"):]
        elif p.startswith("protoagg_"):
            # New segment! Format: protoagg_strategy_components
            agg_parts = p.split("_")
            if len(agg_parts) >= 2:
                params["prototype_strategy"] = agg_parts[1]
            if len(agg_parts) >= 3:
                params["prototype_components"] = agg_parts[2]
        elif p in ["triplet", "arcface", "softmax_contrastive", "ce"]:
            # This is likely classif_loss or dloss. 
            # In the current structure, classif_loss is at index 4 (relative to dataset)
            # and dloss is at index 5.
            pass

    # Fallback to positional indices for fixed-position fields without unique prefixes.
    param_parts = [p for p in data_parts if not re.match(r"^split_[0-9a-f]{12}$", p)]
    if len(param_parts) > 3:
        params.setdefault("classif_loss", param_parts[3])
    if len(param_parts) > 4:
        params.setdefault("DLoss", param_parts[4])
    
    return params


def task_from_model_row(row, default=None):
    """Return a model row task, falling back to the task encoded in log_path."""
    if row is None:
        return default

    getter = row.get if hasattr(row, "get") else None
    if getter is None:
        return default

    task = getter("Task") or getter("task") or getter("label_task")
    if task:
        return str(task)

    parsed = extract_params_from_log_path(
        getter("Log Path") or getter("log_path") or getter("path") or ""
    )
    return parsed.get("Task") or default


def attach_task_column(models_df: pd.DataFrame, default=None) -> pd.DataFrame:
    """Ensure leaderboard-style dataframes have a Task column."""
    if models_df is None or models_df.empty:
        return models_df

    df = models_df.copy()
    if "Task" not in df.columns:
        df["Task"] = None

    df["Task"] = df.apply(lambda r: task_from_model_row(r, default=default), axis=1)
    return df


def filter_models_df_by_task(models_df: pd.DataFrame, task: str | None) -> pd.DataFrame:
    """Filter best-model rows to the active task without requiring a DB task column."""
    if models_df is None or models_df.empty or not task:
        return models_df

    df = attach_task_column(models_df)
    task_text = str(task)
    return df[df["Task"].astype(str) == task_text].reset_index(drop=True)


LEADERBOARD_DONE_MANIFEST_CACHE_KEY = "done_manifest_config_cache"


def _normalize_manifest_value(value):
    """Normalize values before comparing registry rows to manifest rows."""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in {"nan", "none"}:
        return ""
    return text


def _build_manifest_config_key(task, model_name, fgsm, n_calibration, classif_loss, dloss, prototypes, n_positives, n_negatives, normalize):
    """Build a stable key shared by manifest rows and leaderboard rows."""
    return "|".join([
        _normalize_manifest_value(task),
        _normalize_manifest_value(model_name),
        _normalize_manifest_value(fgsm),
        _normalize_manifest_value(n_calibration),
        _normalize_manifest_value(classif_loss),
        _normalize_manifest_value(dloss),
        _normalize_manifest_value(prototypes),
        _normalize_manifest_value(n_positives),
        _normalize_manifest_value(n_negatives),
        _normalize_manifest_value(normalize),
    ])


def _resolve_manifest_path(task_name: str) -> str | None:
    """Resolve the manifest CSV path for a task across legacy and per-dataset layouts."""
    if not task_name:
        return None

    # Legacy layout: logs/progresses/<task>/csv/PROD_<task>_job_manifest.csv
    legacy_path = os.path.join("logs", "progresses", task_name, "csv", f"PROD_{task_name}_job_manifest.csv")
    if os.path.exists(legacy_path):
        return legacy_path

    # New layout: logs/progresses/<task>/<dataset_key>/csv/PROD_<task>_job_manifest.csv
    pattern = os.path.join("logs", "progresses", task_name, "*", "csv", f"PROD_{task_name}_job_manifest.csv")
    matches = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if not matches:
        return None

    # Prefer the most recently modified manifest when multiple dataset-key folders exist.
    return max(matches, key=os.path.getmtime)


def _get_done_manifest_config_keys(task: str):
    """Return the set of completed manifest config keys for a task, or None if unavailable."""
    task_name = _normalize_manifest_value(task)
    if not task_name:
        return None

    cache = st.session_state.setdefault(LEADERBOARD_DONE_MANIFEST_CACHE_KEY, {})
    if task_name in cache:
        return cache[task_name]

    manifest_path = _resolve_manifest_path(task_name)
    if not os.path.exists(manifest_path):
        cache[task_name] = None
        return None

    try:
        manifest_df = pd.read_csv(manifest_path)
    except Exception:
        cache[task_name] = None
        return None

    if "job_state" not in manifest_df.columns:
        cache[task_name] = None
        return None

    done_df = manifest_df[manifest_df["job_state"].astype(str).str.strip().str.lower() == "done"].copy()
    if "task" in done_df.columns:
        done_df = done_df[done_df["task"].astype(str).str.strip().str.lower() == task_name]
    done_keys = set()
    for _, row in done_df.iterrows():
        loss_value = row.get("classif_loss")
        if _normalize_manifest_value(loss_value) == "":
            loss_value = row.get("loss")
        done_keys.add(
            _build_manifest_config_key(
                task=row.get("task", task_name),
                model_name=row.get("model"),
                fgsm=row.get("fgsm"),
                n_calibration=row.get("n_calibration"),
                classif_loss=loss_value,
                dloss=row.get("dloss"),
                prototypes=row.get("prototype"),
                n_positives=row.get("n_positives"),
                n_negatives=row.get("n_negatives"),
                normalize=row.get("normalize"),
            )
        )

    cache[task_name] = done_keys
    return done_keys


def is_done_manifest_model_row(row):
    """Return True when a leaderboard row corresponds to a done manifest job."""
    if row is None:
        return False

    log_path = row.get("Log Path") if hasattr(row, "get") else None
    parsed = extract_params_from_log_path(log_path)
    task = row.get("Task") if hasattr(row, "get") else None
    if not task:
        task = parsed.get("Task")

    done_keys = _get_done_manifest_config_keys(task)
    if done_keys is None:
        return True

    row_key = _build_manifest_config_key(
        task=task,
        model_name=row.get("Model Name") if hasattr(row, "get") else None,
        fgsm=row.get("FGSM") if hasattr(row, "get") else None,
        n_calibration=row.get("N_Calibration") if hasattr(row, "get") else None,
        classif_loss=row.get("Classif_Loss") if hasattr(row, "get") else None,
        dloss=row.get("DLoss") if hasattr(row, "get") else None,
        prototypes=row.get("Prototypes") if hasattr(row, "get") else None,
        n_positives=row.get("NPos") if hasattr(row, "get") else None,
        n_negatives=row.get("NNeg") if hasattr(row, "get") else None,
        normalize=row.get("Normalize") if hasattr(row, "get") else None,
    )
    return row_key in done_keys


def get_done_manifest_models_df(task: str):
    """Load manifest and return dataframe of only 'done' models with key columns for filtering registry.
    
    Returns a dataframe with columns: model, fgsm, n_calibration, loss, dloss, prototype, n_positives, n_negatives, normalize
    or None if manifest not available.
    """
    task_name = _normalize_manifest_value(task)
    if not task_name:
        return None
    
    manifest_path = _resolve_manifest_path(task_name)
    if not manifest_path or not os.path.exists(manifest_path):
        return None
    
    try:
        manifest_df = pd.read_csv(manifest_path)
    except Exception:
        return None
    
    if "job_state" not in manifest_df.columns:
        return None
    
    done_df = manifest_df[manifest_df["job_state"].astype(str).str.strip().str.lower() == "done"].copy()
    if "task" in done_df.columns:
        done_df = done_df[done_df["task"].astype(str).str.strip().str.lower() == task_name]
    if done_df.empty:
        return None
    
    # Select and rename columns to match registry parameter names
    try:
        result = done_df[[
            "model", "fgsm", "n_calibration", "dloss", "prototype", "n_positives", "n_negatives", "normalize"
        ]].copy()
        
        # Add classif_loss (use "classif_loss" if present, otherwise "loss")
        if "classif_loss" in manifest_df.columns:
            result["classif_loss"] = done_df["classif_loss"]
        elif "loss" in manifest_df.columns:
            result["classif_loss"] = done_df["loss"]
        else:
            result["classif_loss"] = None
            
        return result
    except KeyError:
        return None


def filter_models_df_by_done_manifest(models_df: pd.DataFrame, task: str):
    """Filter models_df to only rows matching done manifest models.
    
    Returns filtered dataframe or original if manifest unavailable.
    """
    done_manifest_df = get_done_manifest_models_df(task)
    if done_manifest_df is None or done_manifest_df.empty:
        return models_df  # Fallback to all models if manifest not available
    
    # Normalize column names in models_df for comparison
    models_df = models_df.copy()
    
    # Build a mask by checking if each row matches a done manifest row
    def row_in_done_manifest(row):
        # Extract the key parameters from the row
        row_model = _normalize_manifest_value(row.get("Model Name"))
        row_fgsm = _normalize_manifest_value(row.get("FGSM"))
        row_ncal = _normalize_manifest_value(row.get("N_Calibration"))
        row_dloss = _normalize_manifest_value(row.get("DLoss"))
        row_proto = _normalize_manifest_value(row.get("Prototypes"))
        row_npos = _normalize_manifest_value(row.get("NPos"))
        row_nneg = _normalize_manifest_value(row.get("NNeg"))
        row_norm = _normalize_manifest_value(row.get("Normalize"))
        row_classif_loss = _normalize_manifest_value(row.get("Classif_Loss"))
        
        # Check if this row matches any done manifest entry
        for _, manifest_row in done_manifest_df.iterrows():
            m_model = _normalize_manifest_value(manifest_row.get("model"))
            m_fgsm = _normalize_manifest_value(manifest_row.get("fgsm"))
            m_ncal = _normalize_manifest_value(manifest_row.get("n_calibration"))
            m_dloss = _normalize_manifest_value(manifest_row.get("dloss"))
            m_proto = _normalize_manifest_value(manifest_row.get("prototype"))
            m_npos = _normalize_manifest_value(manifest_row.get("n_positives"))
            m_nneg = _normalize_manifest_value(manifest_row.get("n_negatives"))
            m_norm = _normalize_manifest_value(manifest_row.get("normalize"))
            m_classif_loss = _normalize_manifest_value(manifest_row.get("classif_loss"))
            
            if (row_model == m_model and row_fgsm == m_fgsm and row_ncal == m_ncal and
                row_dloss == m_dloss and row_proto == m_proto and row_npos == m_npos and
                row_nneg == m_nneg and row_norm == m_norm and row_classif_loss == m_classif_loss):
                return True
        return False
    
    mask = models_df.apply(row_in_done_manifest, axis=1)
    return models_df[mask].copy()


def build_params_from_args(_args, keys):
    """Build parameter string from args and key list."""
    parts = []
    for key in keys:
        value = getattr(_args, key, None)
        if value is not None:
            parts.append(f"{key}{value}")
    return "/".join(parts)


# ---- Calibration Metrics ---- #

LEADERBOARD_CAL_CACHE_KEY = "calibration_metrics_cache"
LEADERBOARD_CAL_POS_LABEL = "NotNormal"
LEADERBOARD_CAL_N_BINS = 10


def _resolve_valid_predictions_csv(log_path: str):
    """Return the best available path to validation predictions, if any."""
    if not log_path:
        return None

    direct_csv_path = os.path.join(log_path, "valid_predictions.csv")
    if os.path.exists(direct_csv_path):
        return direct_csv_path

    metadata_csv_path = _resolve_predictions_csv_from_metadata(log_path, "valid")
    if metadata_csv_path:
        return metadata_csv_path

    return _resolve_predictions_csv_from_models_csv(log_path, "valid")


def _resolve_predictions_csv_from_metadata(log_path: str, split: str):
    """Find a predictions CSV referenced by run metadata/summary files."""
    if not log_path:
        return None

    summary_path = os.path.join(log_path, "run_summary.json")
    try:
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as summary_file:
                summary = json.load(summary_file)

            artifacts = summary.get("artifacts") or {}
            artifact_csv_path = artifacts.get(f"{split}_predictions_csv")
            if artifact_csv_path:
                if not os.path.isabs(artifact_csv_path):
                    artifact_csv_path = os.path.join(log_path, artifact_csv_path)
                if os.path.exists(artifact_csv_path):
                    return artifact_csv_path
    except Exception:
        pass

    metadata_path = os.path.join(log_path, "run_metadata.json")
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            complete_log_path = metadata.get("complete_log_path")
            if complete_log_path and complete_log_path != log_path:
                direct_path = os.path.join(complete_log_path, f"{split}_predictions.csv")
                if os.path.exists(direct_path):
                    return direct_path
                return _resolve_predictions_csv_from_metadata(complete_log_path, split)
    except Exception:
        pass

    return None


def _norm_model_csv_value(value):
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""
    return text


def _norm_model_csv_token(value):
    text = _norm_model_csv_value(value)
    if "/" in text and text.endswith("_inference"):
        text = text[: -len("_inference")]
    for prefix in ["prototypes_", "nsize", "fgsm", "fsgm", "ncal", "npos", "nneg", "dist_"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text


def _best_models_csv_match_score(row, parsed):
    """Score how well one legacy models.csv row matches a best-model log path."""
    checks = [
        ("model", parsed.get("Model Name"), False),
        ("path", parsed.get("Dataset"), False),
        ("n_neighbors", parsed.get("N_Neighbors"), False),
        ("nsize", parsed.get("new_size"), False),
        ("fgsm", parsed.get("FGSM"), False),
        ("n_calibration", parsed.get("N_Calibration"), False),
        ("loss", parsed.get("classif_loss"), True),
        ("dloss", parsed.get("DLoss"), True),
        ("dist_fct", parsed.get("Dist_Fct"), True),
        ("prototype", parsed.get("Prototypes"), False),
        ("n_positives", parsed.get("NPos"), False),
        ("n_negatives", parsed.get("NNeg"), False),
        ("normalize", parsed.get("Normalize"), True),
    ]

    score = 0
    required_misses = 0
    for csv_col, parsed_value, raw_compare in checks:
        if parsed_value is None:
            continue
        row_value = row.get(csv_col)
        left = _norm_model_csv_value(row_value) if raw_compare else _norm_model_csv_token(row_value)
        right = _norm_model_csv_value(parsed_value) if raw_compare else _norm_model_csv_token(parsed_value)
        if not left or not right:
            continue
        if left == right:
            score += 1
        elif csv_col in {"model", "path"}:
            required_misses += 1

    if required_misses:
        return -1
    return score


def _resolve_predictions_csv_from_models_csv(log_path: str, split: str):
    """Use legacy best_models/<task>/models.csv to find original run artifacts."""
    parsed = extract_params_from_log_path(log_path)
    task = parsed.get("Task")
    if not task:
        return None

    models_csv_path = os.path.join("logs", "best_models", task, "models.csv")
    if not os.path.exists(models_csv_path):
        return None

    try:
        models_df = pd.read_csv(models_csv_path)
    except Exception:
        return None

    if models_df.empty or "complete_log_path" not in models_df.columns:
        return None

    scored_rows = []
    for _, row in models_df.iterrows():
        score = _best_models_csv_match_score(row, parsed)
        if score < 0:
            continue
        complete_log_path = row.get("complete_log_path")
        if not complete_log_path or _norm_model_csv_value(complete_log_path) == "":
            continue
        csv_path = os.path.join(str(complete_log_path), f"{split}_predictions.csv")
        if os.path.exists(csv_path):
            scored_rows.append((score, csv_path))

    if not scored_rows:
        return None

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    return scored_rows[0][1]


def _compute_calibration_metrics(log_path: str):
    """Compute calibration metrics from the saved validation predictions."""
    from sklearn.calibration import calibration_curve

    from otitenet.logging.metrics import expected_calibration_error, brier_score

    if not log_path:
        return None

    csv_path = _resolve_valid_predictions_csv(log_path)
    if csv_path is None:
        return None

    try:
        df_cal = pd.read_csv(csv_path)
    except Exception as exc:
        return {"error": f"Could not read {os.path.basename(csv_path)}: {exc}"}

    if "label" not in df_cal.columns:
        return {"error": "Missing columns: label"}

    prob_cols = [col for col in df_cal.columns if col.startswith("probs_")]
    if not prob_cols:
        return {"error": "Missing probability columns: expected columns named probs_<label>"}

    labels = [col[len("probs_"):] for col in prob_cols]
    y_label = df_cal["label"].astype(str).to_numpy()
    probs = df_cal[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_rows = np.isfinite(probs).all(axis=1)
    known_rows = np.isin(y_label, labels)
    valid_mask = finite_rows & known_rows

    y_label = y_label[valid_mask]
    probs = probs[valid_mask]
    if len(y_label) == 0:
        return {"error": "No valid labels/probabilities for calibration", "n_samples": 0}

    row_sums = probs.sum(axis=1, keepdims=True)
    probs = np.divide(
        probs,
        row_sums,
        out=np.full_like(probs, 1.0 / len(labels), dtype=float),
        where=row_sums != 0,
    )

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_idx = np.asarray([label_to_idx[label] for label in y_label], dtype=int)

    if len(labels) == 2:
        pos_label = LEADERBOARD_CAL_POS_LABEL if LEADERBOARD_CAL_POS_LABEL in label_to_idx else labels[1]
        pos_idx = label_to_idx[pos_label]
        y_true_filt = (y_idx == pos_idx).astype(int)
        y_prob_filt = np.clip(probs[:, pos_idx], 0.0, 1.0)
        if set(np.unique(y_true_filt)) != {0, 1}:
            return {
                "error": "Calibration requires both binary classes in validation labels",
                "n_samples": int(len(y_true_filt)),
            }
    else:
        y_prob_filt = np.clip(probs[np.arange(len(y_idx)), y_idx], 0.0, 1.0)
        y_true_filt = (np.argmax(probs, axis=1) == y_idx).astype(int)
        if len(np.unique(y_true_filt)) < 2:
            return {
                "error": "Multiclass calibration requires at least one correct and one incorrect validation prediction",
                "n_samples": int(len(y_true_filt)),
            }

    try:
        prob_true, prob_pred = calibration_curve(
            y_true_filt,
            y_prob_filt,
            n_bins=LEADERBOARD_CAL_N_BINS,
        )
        ece_val = expected_calibration_error(
            y_prob_filt,
            y_true_filt,
            n_bins=LEADERBOARD_CAL_N_BINS,
        )
        brier_val = brier_score(y_prob_filt, y_true_filt)
    except Exception as exc:
        return {"error": f"Calibration computation failed: {exc}"}

    return {
        "error": None,
        "ece": float(ece_val),
        "brier": float(brier_val),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "n_points": int(len(prob_true)),
        "n_samples": int(len(y_true_filt)),
        "n_classes": int(len(labels)),
        "class_labels": labels,
    }


def get_calibration_metrics(log_path: str):
    """Return cached calibration metrics for a given log path, computing if needed."""
    if not log_path:
        return None
    cache = st.session_state.setdefault(LEADERBOARD_CAL_CACHE_KEY, {})
    if log_path in cache:
        return cache[log_path]
    metrics = _compute_calibration_metrics(log_path)
    cache[log_path] = metrics
    return metrics


def _compute_auc_from_predictions_csv(csv_path: str, pos_label: str = LEADERBOARD_CAL_POS_LABEL):
    """Compute binary AUC from a saved predictions CSV.

    Expected columns:
      - label
      - probs_<pos_label>, for example probs_NotNormal
    """
    if not csv_path or not os.path.exists(csv_path):
        return np.nan

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return np.nan

    prob_col = f"probs_{pos_label}"

    if "label" not in df.columns or prob_col not in df.columns:
        return np.nan

    y_true = (df["label"].astype(str) == str(pos_label)).astype(int).to_numpy()
    y_score = pd.to_numeric(df[prob_col], errors="coerce").to_numpy()

    valid_mask = np.isfinite(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    # AUC is undefined if only one class is present
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def _prediction_class_labels(df: pd.DataFrame):
    """Return class labels visible in a saved predictions CSV."""
    labels = []

    for col in df.columns:
        if col.startswith("probs_"):
            labels.append(str(col[len("probs_"):]))

    for col in ["label", "pred"]:
        if col in df.columns:
            values = df[col].dropna().astype(str).tolist()
            labels.extend([v for v in values if v and v.lower() not in {"nan", "none"}])

    preferred = ["Normal", "NotNormal", "Wax", "Tube"]
    labels = _unique_preserve_order(labels)
    return [c for c in preferred if c in labels] + [c for c in labels if c not in preferred]


def _compute_per_class_metrics_from_predictions_csv(csv_path: str, split: str):
    """Compute per-class one-vs-rest scores from a saved predictions CSV."""
    if not csv_path or not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    if "label" not in df.columns or "pred" not in df.columns:
        return {}

    labels = _prediction_class_labels(df)
    if not labels:
        return {}

    y_true = df["label"].fillna("").astype(str)
    y_pred = df["pred"].fillna("").astype(str)
    prefix = str(split).strip().title()
    out = {}

    for label in labels:
        true_pos = (y_true == label)
        pred_pos = (y_pred == label)
        tp = int((true_pos & pred_pos).sum())
        fp = int((~true_pos & pred_pos).sum())
        fn = int((true_pos & ~pred_pos).sum())
        support = int(true_pos.sum())

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan

        out[f"{prefix} F1 {label}"] = float(f1) if np.isfinite(f1) else np.nan
        out[f"{prefix} Recall {label}"] = float(recall) if np.isfinite(recall) else np.nan
        out[f"{prefix} Precision {label}"] = float(precision) if np.isfinite(precision) else np.nan
        out[f"{prefix} Support {label}"] = support

    return out


def _resolve_split_predictions_csv(log_path: str, split: str):
    """Find train/valid/test predictions CSV from common locations."""
    if not log_path:
        return None

    direct_path = os.path.join(log_path, f"{split}_predictions.csv")
    if os.path.exists(direct_path):
        return direct_path

    metadata_path = _resolve_predictions_csv_from_metadata(log_path, split)
    if metadata_path:
        return metadata_path

    return _resolve_predictions_csv_from_models_csv(log_path, split)


def _load_split_mcc_metrics(log_path: str):
    """Load train/valid/test MCC and AUC from run artifacts when available."""
    empty = {
        "train_mcc": np.nan,
        "valid_mcc": np.nan,
        "test_mcc": np.nan,
        "train_auc": np.nan,
        "valid_auc": np.nan,
        "test_auc": np.nan,
    }

    if not log_path:
        return empty

    summary_candidates = [os.path.join(log_path, "run_summary.json")]
    run_metadata_path = os.path.join(log_path, "run_metadata.json")

    try:
        if os.path.exists(run_metadata_path):
            with open(run_metadata_path, "r", encoding="utf-8") as f:
                run_metadata = json.load(f)
            original_log_path = run_metadata.get("complete_log_path")
            if original_log_path:
                summary_candidates.append(os.path.join(original_log_path, "run_summary.json"))
    except Exception:
        pass

    out = empty.copy()

    # First: get MCC, and AUC too if already saved in run_summary.json
    for summary_path in summary_candidates:
        try:
            if not summary_path or not os.path.exists(summary_path):
                continue

            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            best_values = payload.get("best_values") or {}

            out.update({
                "train_mcc": best_values.get("train", {}).get("mcc", np.nan),
                "valid_mcc": best_values.get("valid", {}).get("mcc", np.nan),
                "test_mcc": best_values.get("test", {}).get("mcc", np.nan),

                # These will work if you later save auc directly in run_summary.json
                "train_auc": best_values.get("train", {}).get("auc", np.nan),
                "valid_auc": best_values.get("valid", {}).get("auc", np.nan),
                "test_auc": best_values.get("test", {}).get("auc", np.nan),
            })
            break
        except Exception:
            continue

    # Second: compute AUC and per-class scores from saved prediction CSVs.
    for split in ["train", "valid", "test"]:
        key = f"{split}_auc"
        csv_path = _resolve_split_predictions_csv(log_path, split)

        if pd.isna(out.get(key, np.nan)):
            # Also check original_log_path if needed
            if csv_path is None:
                try:
                    if os.path.exists(run_metadata_path):
                        with open(run_metadata_path, "r", encoding="utf-8") as f:
                            run_metadata = json.load(f)
                        original_log_path = run_metadata.get("complete_log_path")
                        if original_log_path:
                            csv_path = _resolve_split_predictions_csv(original_log_path, split)
                except Exception:
                    pass

            out[key] = _compute_auc_from_predictions_csv(csv_path)

        if csv_path is not None:
            out.update(_compute_per_class_metrics_from_predictions_csv(csv_path, split))

    return out

def get_split_mcc_metrics(log_path: str):
    """Return cached train/valid/test MCC metrics for a given model log path."""
    if not log_path:
        return None
    cache = st.session_state.setdefault('leaderboard_split_mcc_cache', {})
    if log_path in cache:
        return cache[log_path]
    metrics = _load_split_mcc_metrics(log_path)
    cache[log_path] = metrics
    return metrics


def _load_run_n_aug(log_path: str):
    """Load the effective n_aug value for a best-model folder from saved metadata."""
    if not log_path:
        return np.nan

    run_metadata_path = os.path.join(log_path, 'run_metadata.json')
    summary_candidates = [os.path.join(log_path, 'run_summary.json')]

    try:
        if os.path.exists(run_metadata_path):
            with open(run_metadata_path, 'r', encoding='utf-8') as f:
                run_metadata = json.load(f)

            optimized_n_aug = (run_metadata.get('optimized_params') or {}).get('n_aug')
            if optimized_n_aug is not None:
                return optimized_n_aug

            args_n_aug = (run_metadata.get('args') or {}).get('n_aug')
            if args_n_aug is not None:
                return args_n_aug

            original_log_path = run_metadata.get('complete_log_path')
            if original_log_path:
                summary_candidates.append(os.path.join(original_log_path, 'run_summary.json'))
    except Exception:
        pass

    for summary_path in summary_candidates:
        try:
            if not summary_path or not os.path.exists(summary_path):
                continue
            with open(summary_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            params = payload.get('params') or {}
            if params.get('n_aug') is not None:
                return params.get('n_aug')
        except Exception:
            continue

    return np.nan


def get_run_n_aug(log_path: str):
    """Return cached n_aug metadata for a given model log path."""
    if not log_path:
        return np.nan
    cache = st.session_state.setdefault('leaderboard_n_aug_cache', {})
    if log_path in cache:
        return cache[log_path]
    value = _load_run_n_aug(log_path)
    cache[log_path] = value
    return value


def _load_run_split_config(log_path: str) -> dict:
    if not log_path:
        return {}

    candidates = [
        os.path.join(log_path, "run_summary.json"),
        os.path.join(log_path, "run_metadata.json"),
    ]
    try:
        metadata_path = os.path.join(log_path, "run_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            original_log_path = metadata.get("complete_log_path")
            if original_log_path:
                candidates.extend([
                    os.path.join(original_log_path, "run_summary.json"),
                    os.path.join(original_log_path, "run_metadata.json"),
                ])
    except Exception:
        pass

    for path in candidates:
        try:
            if not path or not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            split_config = payload.get("split_config") or {}
            args_payload = payload.get("args") or {}
            out = {
                "train_datasets": split_config.get("train_datasets") or args_payload.get("train_datasets"),
                "valid_dataset": split_config.get("valid_dataset") or args_payload.get("valid_dataset"),
                "test_dataset": split_config.get("test_dataset") or args_payload.get("test_dataset"),
            }
            if any(out.values()):
                out["train_datasets"] = normalize_train_datasets(out.get("train_datasets"))
                out["valid_dataset"] = str(out.get("valid_dataset") or "").strip()
                out["test_dataset"] = str(out.get("test_dataset") or "").strip()
                if not out["train_datasets"]:
                    out["train_datasets"] = normalize_train_datasets(LEGACY_TRAIN_DATASETS)
                if not out["valid_dataset"]:
                    out["valid_dataset"] = LEGACY_VALID_DATASET
                if not out["test_dataset"]:
                    out["test_dataset"] = out["valid_dataset"]
                out["split_config_key"] = split_config_key(
                    out["train_datasets"],
                    out["valid_dataset"],
                    out["test_dataset"],
                )
                return out
        except Exception:
            continue
    return legacy_split_config()


def get_model_split_config(log_path: str) -> dict:
    if not log_path:
        return {}
    cache = st.session_state.setdefault("leaderboard_split_config_cache", {})
    if log_path in cache:
        return cache[log_path]
    value = _load_run_split_config(log_path)
    cache[log_path] = value
    return value


# ---- Model Selection Key Generation ---- #

def _normalize_value(v):
    """Normalize values to stable strings for key generation."""
    try:
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        if isinstance(v, (np.floating,)) and np.isnan(v):
            return ""
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if float(v).is_integer():
                return str(int(v))
            s = ("%g" % float(v)).strip()
            return s.lower()
        s = str(v).strip()
        if s.lower() in {"nan", "none"}:
            return ""
        return s.lower()
    except Exception:
        return ""


def _make_model_selection_key(row_dict: dict) -> str:
    """Create a stable unique key for a model parameter combination."""
    parts = [
        _normalize_value(row_dict.get("Model Name", "")),
        _normalize_value(row_dict.get("NSize", "")),
        _normalize_value(row_dict.get("FGSM", "")),
        _normalize_value(row_dict.get("Prototypes", "")),
        _normalize_value(row_dict.get("NPos", "")),
        _normalize_value(row_dict.get("NNeg", "")),
        _normalize_value(row_dict.get("DLoss", "")),
        _normalize_value(row_dict.get("Dist_Fct", "")),
        _normalize_value(row_dict.get("Classif_Loss", "")),
        _normalize_value(row_dict.get("N_Calibration", "")),
        _normalize_value(row_dict.get("Normalize", "")),
        _normalize_value(row_dict.get("N_Neighbors", "")),
        _normalize_value(row_dict.get("train_datasets", row_dict.get("Train Datasets", ""))),
        _normalize_value(row_dict.get("valid_dataset", row_dict.get("Valid Dataset", ""))),
        _normalize_value(row_dict.get("test_dataset", row_dict.get("Test Dataset", ""))),
    ]
    return "|".join(parts)


def _lookup_model_number(mapping_rd: dict, model_number_map: dict) -> str:
    """Resolve the model number from the map with graceful fallback."""
    # First, attempt exact match
    exact_key = _make_model_selection_key(mapping_rd)
    num = model_number_map.get(exact_key)
    if num is not None:
        return num

    # Fallback: fuzzy match on all fields except N_Neighbors
    expected_parts_wo_neighbors = [
        _normalize_value(mapping_rd.get("Model Name", "")),
        _normalize_value(mapping_rd.get("NSize", "")),
        _normalize_value(mapping_rd.get("FGSM", "")),
        _normalize_value(mapping_rd.get("Prototypes", "")),
        _normalize_value(mapping_rd.get("NPos", "")),
        _normalize_value(mapping_rd.get("NNeg", "")),
        _normalize_value(mapping_rd.get("DLoss", "")),
        _normalize_value(mapping_rd.get("Dist_Fct", "")),
        _normalize_value(mapping_rd.get("Classif_Loss", "")),
        _normalize_value(mapping_rd.get("N_Calibration", "")),
        _normalize_value(mapping_rd.get("Normalize", "")),
    ]
    want_neighbors = _normalize_value(mapping_rd.get("N_Neighbors", ""))

    for k, v in model_number_map.items():
        parts = k.split("|")
        if len(parts) < 12:
            continue
        if parts[:11] == expected_parts_wo_neighbors:
            if want_neighbors == "":
                return v
            if parts[11] == want_neighbors:
                return v

    return "?"


def _ensure_model_number_map(cursor):
    """Ensure model_number_map and best_models_table exist in session state."""
    model_number_map = st.session_state.get('model_number_map', {})
    best_models_table = st.session_state.get('best_models_table', None)
    if model_number_map and best_models_table is not None:
        return model_number_map, best_models_table

    try:
        try:
            cursor.execute(
                """
                 SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct,
                     classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path, model_rank,
                     prototype_strategy, prototype_components, test_mcc, valid_auc, test_auc,
                     train_datasets, valid_dataset, test_dataset, split_config_key,
                     artifact_id, best_model_dir, source_run_log_path
                FROM best_models_registry
                ORDER BY (model_rank IS NULL) ASC, model_rank ASC, mcc DESC
                """
            )
            model_rows = cursor.fetchall()
        except Exception:
            cursor.execute(
                """
                 SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct,
                     classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path, model_rank,
                     prototype_strategy, prototype_components, test_mcc, valid_auc, test_auc,
                     train_datasets, valid_dataset, test_dataset, split_config_key
                FROM best_models_registry
                ORDER BY (model_rank IS NULL) ASC, model_rank ASC, mcc DESC
                """
            )
            model_rows = [tuple(list(row) + [None, None, None]) for row in (cursor.fetchall() or [])]
        use_db_rank = True
    except Exception:
        try:
            try:
                cursor.execute(
                    """
                     SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct,
                         classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path,
                         prototype_strategy, prototype_components, test_mcc, valid_auc, test_auc,
                         train_datasets, valid_dataset, test_dataset, split_config_key,
                         artifact_id, best_model_dir, source_run_log_path
                    FROM best_models_registry
                    ORDER BY mcc DESC
                    """
                )
                model_rows = cursor.fetchall()
            except Exception:
                cursor.execute(
                    """
                     SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct,
                         classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path,
                         prototype_strategy, prototype_components, test_mcc, valid_auc, test_auc,
                         train_datasets, valid_dataset, test_dataset, split_config_key
                    FROM best_models_registry
                    ORDER BY mcc DESC
                    """
                )
                model_rows = [tuple(list(row) + [None, None, None]) for row in (cursor.fetchall() or [])]
        except Exception:
            cursor.execute(
                """
                 SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct,
                     classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path,
                     prototype_strategy, prototype_components, test_mcc, valid_auc, test_auc
                FROM best_models_registry
                ORDER BY mcc DESC
                """
            )
            model_rows = [tuple(list(row) + [None, None, None, None, None, None, None]) for row in (cursor.fetchall() or [])]
        use_db_rank = False

    if not model_rows:
        st.session_state['model_number_map'] = model_number_map
        return model_number_map, best_models_table

    if use_db_rank:
        cols = [
            "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
            "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", "#",
            "Proto_Strat", "Proto_Comp", "Test_MCC", "Valid_AUC", "Test_AUC",
            "train_datasets", "valid_dataset", "test_dataset", "split_config_key",
            "Artifact ID", "Best Model Dir", "Source Run Path",
        ]
    else:
        cols = [
            "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
            "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path",
            "Proto_Strat", "Proto_Comp", "Test_MCC", "Valid_AUC", "Test_AUC",
            "train_datasets", "valid_dataset", "test_dataset", "split_config_key",
            "Artifact ID", "Best Model Dir", "Source Run Path",
        ]
    df = pd.DataFrame(model_rows, columns=cols)
    if "Best Model Dir" not in df.columns:
        df["Best Model Dir"] = df.get("Log Path", "")
    if "Source Run Path" not in df.columns:
        df["Source Run Path"] = ""
    for idx, row in df.iterrows():
        preferred = preferred_model_artifact_dir(row.to_dict())
        if preferred:
            df.at[idx, "Artifact Log Path"] = preferred
            df.at[idx, "Log Path"] = preferred
    group_cols = [
        "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
        "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize", "N_Neighbors",
        "train_datasets", "valid_dataset", "test_dataset",
    ]
    _dedupe_frame = df[group_cols].copy().fillna("").astype(str)
    df["_dedupe_key"] = _dedupe_frame.apply(lambda r: "|".join(r.values.tolist()), axis=1)
    df = df.sort_values("MCC", ascending=False)
    df = df.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"])
    df = df.dropna(subset=["Log Path"])
    df = df[df["Log Path"].astype(str) != ""]
    df = df.reset_index(drop=True)
    df["N_Aug"] = np.nan
    df["Train MCC"] = np.nan
    df["Valid MCC"] = np.nan
    df["Test MCC"] = np.nan
    df["Train AUC"] = np.nan
    df["Valid AUC"] = np.nan
    df["Test AUC"] = np.nan
    for idx, row in df.iterrows():
        df.at[idx, "N_Aug"] = get_run_n_aug(row.get("Log Path"))
        metrics = get_split_mcc_metrics(row.get("Log Path"))
        if metrics:
            df.at[idx, "Train MCC"] = metrics.get('train_mcc', np.nan)
            df.at[idx, "Valid MCC"] = metrics.get('valid_mcc', np.nan)
            df.at[idx, "Test MCC"] = metrics.get('test_mcc', np.nan)
            df.at[idx, "Train AUC"] = metrics.get('train_auc', np.nan)
            df.at[idx, "Valid AUC"] = metrics.get('valid_auc', np.nan)
            df.at[idx, "Test AUC"] = metrics.get('test_auc', np.nan)
            for metric_name, metric_value in metrics.items():
                if " F1 " in metric_name or " Recall " in metric_name or " Precision " in metric_name or " Support " in metric_name:
                    if metric_name not in df.columns:
                        df[metric_name] = np.nan
                    df.at[idx, metric_name] = metric_value

    valid_sort = pd.to_numeric(df.get("Valid MCC"), errors="coerce")
    mcc_sort = pd.to_numeric(df.get("MCC"), errors="coerce")
    df = df.assign(
        _valid_sort=valid_sort.fillna(-np.inf),
        _mcc_sort=mcc_sort.fillna(-np.inf),
    )
    df = df.sort_values(["_valid_sort", "_mcc_sort"], ascending=[False, False]).reset_index(drop=True)
    df = df.drop(columns=["_valid_sort", "_mcc_sort"])

    df["#"] = np.arange(1, len(df) + 1)
    preferred_metrics = [
        "Accuracy", "MCC",
        "Train MCC", "Valid MCC", "Test MCC",
        "Train AUC", "Valid AUC", "Test AUC",
    ]
    per_class_cols = [
        c for c in df.columns
        if " F1 " in c or " Recall " in c or " Precision " in c or " Support " in c
    ]
    preferred = ["#"] + [c for c in df.columns if c != "#" and c not in preferred_metrics and c not in per_class_cols]
    preferred += [c for c in preferred_metrics if c in df.columns]
    preferred += sorted(per_class_cols)
    preferred += [c for c in df.columns if c not in preferred]
    df = df[preferred]

    model_number_map = {}
    for _, r in df.iterrows():
        rd = r.to_dict()
        selection_key = _make_model_selection_key(rd)
        model_number_map[selection_key] = rd.get("#", "?")

    st.session_state['model_number_map'] = model_number_map
    st.session_state['best_models_table'] = df.copy()
    return model_number_map, df.copy()


_BASELINE_DISPLAY_NAMES = {
    'logreg': 'Logistic Regression',
    'mlp_head': 'MLP Head',
    'ridge': 'Ridge Classifier',
    'naive_bayes': 'Naive Bayes',
    'linear_svc': 'Linear SVC',
    'rbf_svc': 'RBF SVC',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'decision_tree': 'Decision Tree',
    'lda': 'Linear Discriminant',
    'qda': 'Quadratic Discriminant',
}


def format_classifier_config(best_config) -> str:
    """Human-readable label for a learned-embedding classifier config."""
    if best_config is None:
        return "—"
    best_k_str = str(best_config).strip()
    if not best_k_str:
        return "—"
    if best_k_str.startswith('protot_'):
        parts = best_k_str.split('_')
        strategy = parts[1] if len(parts) > 1 else 'unknown'
        n_comp = parts[2] if len(parts) > 2 else '?'
        return f"Prototype: {strategy.upper()} (n_comp={n_comp})"
    if best_k_str.startswith('baseline_'):
        baseline_name = best_k_str.replace('baseline_', '')
        return _BASELINE_DISPLAY_NAMES.get(baseline_name, baseline_name.replace('_', ' ').title())
    if best_k_str.startswith('kde'):
        return f"KDE ({best_k_str})"
    try:
        int(best_k_str)
        return f"KNN (k={best_k_str})"
    except ValueError:
        return best_k_str


def best_display_classifier_heads(rows: list[dict]) -> list[dict]:
    """Keep one best display row per model/n_aug/classifier family."""
    if not rows:
        return []

    df = pd.DataFrame(rows)
    if "Valid MCC" not in df.columns:
        return rows

    df["Valid MCC"] = pd.to_numeric(df["Valid MCC"], errors="coerce")
    group_cols = [col for col in ["Model ID", "N Aug", "Family"] if col in df.columns]
    if not group_cols:
        return df.sort_values("Valid MCC", ascending=False, na_position="last").to_dict("records")

    return (
        df.sort_values("Valid MCC", ascending=False, na_position="last")
        .groupby(group_cols, as_index=False, dropna=False)
        .first()
        .to_dict("records")
    )


def parse_classifier_config(best_config) -> dict:
    """Return structured metadata for a learned-embedding classifier config."""
    config = "" if best_config is None else str(best_config).strip()
    if not config or config in {"None", "nan", "—"}:
        return {"config": config, "family": "unknown"}

    if config.startswith("protot_"):
        parts = config.split("_")
        strategy = parts[1] if len(parts) > 1 and parts[1] else "mean"
        try:
            components = int(parts[2]) if len(parts) > 2 else 1
        except Exception:
            components = 1
        return {
            "config": config,
            "family": "prototype",
            "strategy": strategy,
            "components": max(1, components),
        }

    if config.startswith("baseline_"):
        return {
            "config": config,
            "family": "baseline",
            "name": config.replace("baseline_", "", 1),
        }

    if config.startswith("kde"):
        return {"config": config, "family": "kde"}

    try:
        return {"config": config, "family": "knn", "k": int(config)}
    except ValueError:
        return {"config": config, "family": "unknown"}


def get_optimization_cache_file_path(_args) -> str:
    """Path to knn_optimization_cache.pkl (same layout as Tab 1 Learned Embedding)."""
    params = get_model_params_path(_args)
    parts = params.split('/')
    base_params = '/'.join(parts[:-3]) if len(parts) > 3 else params
    normalize_val = str(getattr(_args, 'normalize', 'no'))
    return os.path.join(
        f'logs/best_models/{_args.task}/{_args.model_name}',
        base_params,
        f'norm{normalize_val}',
        'knn_optimization_cache.pkl',
    )


def _optimization_cache_file_paths(_args) -> list[str]:
    """Return split-aware and legacy optimization cache locations for an args namespace."""
    paths = [get_optimization_cache_file_path(_args)]
    if bool(getattr(_args, "split_config_in_path", False) or getattr(_args, "_split_config_in_path", False)):
        old_split_flag = getattr(_args, "split_config_in_path", False)
        old_private_flag = getattr(_args, "_split_config_in_path", False)
        try:
            _args.split_config_in_path = False
            _args._split_config_in_path = False
            legacy_path = get_optimization_cache_file_path(_args)
            if legacy_path not in paths:
                paths.append(legacy_path)
        finally:
            _args.split_config_in_path = old_split_flag
            _args._split_config_in_path = old_private_flag
    return paths


def _cache_result_matches_args_split(result: dict, _args) -> bool:
    target_key = split_config_key(**split_config_values(_args))
    if not target_key.replace("|", "").strip():
        return True

    result_key = split_config_key(
        result.get("train_datasets"),
        result.get("valid_dataset"),
        result.get("test_dataset"),
    )
    if not result_key.replace("|", "").strip():
        return not bool(getattr(_args, "split_config_in_path", False) or getattr(_args, "_split_config_in_path", False))
    return result_key == target_key


def _merge_optimization_cache_entries(
    target_cache: dict,
    source_cache: dict,
    _args=None,
    replace_existing: bool = False,
    allow_no_split_fallback: bool = False,
) -> dict:
    """Merge source cache into target.

    Split-aware cache files are loaded before legacy files. Existing entries
    therefore win by default so older no-split caches cannot override freshly
    retrained split-specific head metrics for the same n_aug.
    """
    if not isinstance(source_cache, dict):
        return target_cache
    for n_aug_key, result in source_cache.items():
        try:
            n_aug_int = int(n_aug_key)
        except Exception:
            continue
        if _args is not None and isinstance(result, dict) and not _cache_result_matches_args_split(result, _args):
            result_key = split_config_key(
                result.get("train_datasets"),
                result.get("valid_dataset"),
                result.get("test_dataset"),
            )
            if not (allow_no_split_fallback and not result_key.replace("|", "").strip()):
                continue
        current = target_cache.get(n_aug_int)
        if current is None:
            target_cache[n_aug_int] = result
            continue
        if not replace_existing:
            continue
        current_mcc = float(current.get('best_mcc', -1)) if isinstance(current, dict) else -1
        new_mcc = float(result.get('best_mcc', -1)) if isinstance(result, dict) else -1
        if new_mcc > current_mcc:
            target_cache[n_aug_int] = result
    return target_cache


def load_optimization_cache_dict(_args) -> dict:
    """Load merged optimization cache (unified + legacy paths)."""
    import glob
    merged_cache = {}
    cache_dirs = []
    for cache_idx, cache_path in enumerate(_optimization_cache_file_paths(_args)):
        cache_dir = os.path.dirname(cache_path)
        if cache_dir not in cache_dirs:
            cache_dirs.append(cache_dir)
        if not os.path.exists(cache_path):
            continue
        try:
            with open(cache_path, 'rb') as f:
                merged_cache = _merge_optimization_cache_entries(
                    merged_cache,
                    pickle.load(f),
                    _args=_args,
                    replace_existing=(cache_idx == 0),
                    allow_no_split_fallback=(cache_idx > 0),
                )
        except Exception as e:
            print(f"[Cache] Could not load optimization cache {cache_path}: {e}")
    for cache_dir in cache_dirs:
        legacy_pattern = os.path.join(cache_dir, 'dist_*', 'knn*', 'knn_optimization_cache.pkl')
        for legacy_file in glob.glob(legacy_pattern):
            try:
                with open(legacy_file, 'rb') as f:
                    merged_cache = _merge_optimization_cache_entries(
                        merged_cache,
                        pickle.load(f),
                        _args=_args,
                        allow_no_split_fallback=True,
                    )
            except Exception as e:
                print(f"[Cache] Could not load legacy cache {legacy_file}: {e}")
    return merged_cache


def load_best_classifier_from_cache(_args):
    """Load the best classifier configuration from the optimization cache.
    
    Returns:
        dict: {
            'best_config': str (e.g. 'baseline_xgboost', '5', 'protot_kmeans_5'),
            'best_mcc': float,
            'n_aug': int,
            'all_results': dict
        } or None if cache not found
    """
    try:
        cache = load_optimization_cache_dict(_args)
        if not cache:
            return None

        best_overall = None
        best_mcc = -1.0
        best_n_aug = 0

        for n_aug, result in cache.items():
            if not isinstance(result, dict):
                continue
            mcc = float(result.get('best_mcc', -1))
            if mcc > best_mcc:
                best_mcc = mcc
                best_overall = result
                best_n_aug = int(n_aug)

        if best_overall and best_overall.get('best_k') is not None:
            return {
                'best_config': best_overall.get('best_k'),
                'best_mcc': best_mcc,
                'n_aug': best_n_aug,
                'all_results': best_overall,
            }
    except Exception as e:
        print(f"Error loading optimization cache: {e}")

    return None


def _update_classification_head(
    best_by_config: dict,
    config: str,
    mcc: float,
    details: str = "",
    dedupe_key: str | None = None,
    **metrics,
):
    """Keep the highest-MCC entry per classification-head config key."""
    if config is None or mcc is None:
        return
    try:
        mcc_f = float(mcc)
    except (TypeError, ValueError):
        return
    if mcc_f < 0:
        return
    key = str(dedupe_key if dedupe_key is not None else config)
    current = best_by_config.get(key)
    if current is None or mcc_f > current.get("mcc", -1):
        entry = {"config": str(config), "mcc": mcc_f, "valid_mcc": mcc_f, "details": details}
        entry.update({k: v for k, v in metrics.items() if v is not None})
        best_by_config[key] = entry


def _classification_head_extra_metrics(source: dict | None) -> dict:
    if not isinstance(source, dict):
        return {}
    allowed = {
        "train_mcc",
        "valid_mcc",
        "test_mcc",
        "all_mcc",
        "train_accuracy",
        "valid_accuracy",
        "test_accuracy",
        "train_auc",
        "valid_auc",
        "test_auc",
        "all_auc",
        "ece",
        "brier",
        "valid_ece",
        "valid_brier",
    }
    out = {}
    for key, value in source.items():
        if (
            key in allowed
            or " F1 " in str(key)
            or " Recall " in str(key)
            or " Precision " in str(key)
            or " Support " in str(key)
        ):
            out[key] = value
    return out


def enumerate_classification_heads(_args, include_all_n_aug: bool = False):
    """List classification heads from the learned-embedding optimization cache.

    Returns one entry per approach by default (best validation MCC across n_aug),
    or one entry per approach+n_aug when include_all_n_aug=True.
    Each item: {config, label, mcc, details}.
    """
    cache = load_optimization_cache_dict(_args)
    if not cache:
        return []

    best_by_config = {}

    for n_aug_val, result in cache.items():
        if not isinstance(result, dict):
            continue
        try:
            n_aug = int(n_aug_val)
        except (TypeError, ValueError):
            n_aug = n_aug_val
        head_cache_version = result.get("head_cache_version")
        split_metrics = {
            "train_datasets": result.get("train_datasets"),
            "valid_dataset": result.get("valid_dataset"),
            "test_dataset": result.get("test_dataset"),
        }

        knn_data = result.get("knn", {}) or {}
        mcc_list = knn_data.get("mcc_per_k", []) or []
        for k_idx, item in enumerate(mcc_list):
            if isinstance(item, dict):
                k_val = item.get("k", k_idx + 1)
                mcc_val = item.get("valid_mcc", item.get("mcc"))
            elif isinstance(item, (int, float)):
                k_val = k_idx + 1
                mcc_val = item
            else:
                continue
            if mcc_val is not None:
                config = str(int(k_val))
                _update_classification_head(
                    best_by_config,
                    config,
                    mcc_val,
                    f"k={int(k_val)}, n_aug={n_aug}",
                    dedupe_key=f"{config}|n_aug={n_aug}" if include_all_n_aug else config,
                    n_aug=n_aug,
                    head_cache_version=head_cache_version,
                    **_classification_head_extra_metrics(item),
                    **split_metrics,
                )

        best_k_overall = result.get("best_k")
        best_mcc_overall = result.get("best_mcc")
        if isinstance(best_k_overall, (int, float)) and best_mcc_overall is not None:
            if not mcc_list:
                _update_classification_head(
                    best_by_config,
                    str(int(best_k_overall)),
                    best_mcc_overall,
                    f"k={int(best_k_overall)}, n_aug={n_aug}",
                    dedupe_key=f"{int(best_k_overall)}|n_aug={n_aug}" if include_all_n_aug else str(int(best_k_overall)),
                    n_aug=n_aug,
                    head_cache_version=head_cache_version,
                    **split_metrics,
                )

        proto_data = result.get("prototypes", {}) or {}
        for strategy in ("mean", "kmeans", "gmm"):
            strat_data = proto_data.get(strategy, {}) or {}
            mcc_val = strat_data.get("best_mcc")
            if mcc_val is None:
                continue
            n_comp = strat_data.get("best_n_components", 1)
            config = f"protot_{strategy}_{n_comp}"
            _update_classification_head(
                best_by_config,
                config,
                mcc_val,
                f"n_comp={n_comp}, n_aug={n_aug}",
                dedupe_key=f"{config}|n_aug={n_aug}" if include_all_n_aug else config,
                n_aug=n_aug,
                head_cache_version=head_cache_version,
                **_classification_head_extra_metrics(strat_data),
                **split_metrics,
            )

        baseline_data = result.get("baselines", {}) or {}
        if not isinstance(baseline_data, dict):
            baseline_data = {}
        for baseline_name in _BASELINE_DISPLAY_NAMES:
            if baseline_name in ("label_map", "label_encoder"):
                continue
            b_data = baseline_data.get(baseline_name, {}) or {}
            if not isinstance(b_data, dict):
                continue
            mcc_val = b_data.get("mcc")
            if mcc_val is None:
                continue
            config = f"baseline_{baseline_name}"
            _update_classification_head(
                best_by_config,
                config,
                mcc_val,
                f"n_aug={n_aug}",
                dedupe_key=f"{config}|n_aug={n_aug}" if include_all_n_aug else config,
                n_aug=n_aug,
                head_cache_version=head_cache_version,
                **_classification_head_extra_metrics(b_data),
                **split_metrics,
            )

    heads = []
    for entry in best_by_config.values():
        cfg = entry["config"]
        head = {
            "config": cfg,
            "label": format_classifier_config(cfg),
            "mcc": entry["mcc"],
            "valid_mcc": entry.get("valid_mcc", entry["mcc"]),
            "train_mcc": entry.get("train_mcc"),
            "test_mcc": entry.get("test_mcc"),
            "valid_auc": entry.get("valid_auc"),
            "train_auc": entry.get("train_auc"),
            "test_auc": entry.get("test_auc"),
            "n_aug": entry.get("n_aug"),
            "head_cache_version": entry.get("head_cache_version"),
            "train_datasets": entry.get("train_datasets"),
            "valid_dataset": entry.get("valid_dataset"),
            "test_dataset": entry.get("test_dataset"),
            "details": entry.get("details", ""),
        }
        head.update(_classification_head_extra_metrics(entry))
        heads.append(head)
    return sorted(heads, key=lambda h: h["mcc"], reverse=True)


def resolve_best_classifier_config(_args, use_optimized: bool = True):
    """Resolve classifier config from optimization cache or training n_neighbors."""
    default = str(getattr(_args, "n_neighbors", 1))
    if not use_optimized:
        return default
    heads = enumerate_classification_heads(_args)
    if heads:
        return heads[0]["config"]
    best_info = load_best_classifier_from_cache(_args)
    if best_info and best_info.get("best_config") is not None:
        return str(best_info["best_config"])
    return default


def get_model_cache_key(args):
    """Build a stable cache key for the current model selection.
    
    This key is used to cache artifacts like fitted KNN classifiers, embeddings, etc.
    that depend on the specific model and configuration.
    
    Args:
        args: Arguments namespace with model configuration parameters
        
    Returns:
        str: Stable cache key combining all model/config parameters
        
    Example:
        key = get_model_cache_key(args)
        if key in cache:
            knn = cache[key]['knn']
    """
    params = [
        str(getattr(args, 'path', '')),
        str(getattr(args, 'device', 'cpu')),
        str(getattr(args, 'model_name', '')),
        str(getattr(args, 'new_size', 224)),
        str(getattr(args, 'fgsm', 0)),
        str(getattr(args, 'prototypes_to_use', 'combined')),
        str(getattr(args, 'n_positives', 1)),
        str(getattr(args, 'n_negatives', 1)),
        str(getattr(args, 'dloss', 'arcface')),
        str(getattr(args, 'dist_fct', 'euclidean')),
        str(getattr(args, 'classif_loss', 'ce')),
        str(getattr(args, 'n_calibration', 0)),
        str(getattr(args, 'normalize', 'no')),
        str(getattr(args, 'n_neighbors', 5)),
    ]
    return "|".join(params)
