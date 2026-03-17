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
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from otitenet.logging.metrics import expected_calibration_error, brier_score
from otitenet.utils.utils import set_random_seeds  # Import from shared utils


# ---- String/Filename Utilities ---- #

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


# ---- Model Parameter Path Construction ---- #

def get_model_params_path(_args):
    """Construct the standardized relative path for model parameters folders."""
    nsize = ensure_int(_args.new_size)
    fgsm = ensure_int(_args.fgsm)
    ncal = ensure_int(_args.n_calibration)
    npos = ensure_int(_args.n_positives)
    nneg = ensure_int(_args.n_negatives)
    n_neighbors = ensure_int(_args.n_neighbors)
    
    dataset_name = _args.path.split("/")[-1]
    
    # Strip "prototypes_" prefix if present (training may add it)
    proto_val = str(_args.prototypes_to_use)
    if proto_val.startswith("prototypes_"):
        proto_val = proto_val[len("prototypes_"):]
    
    # Get distance function and normalize values
    dist_fct_val = str(getattr(_args, 'dist_fct', 'euclidean'))
    normalize_val = str(getattr(_args, 'normalize', 'no'))
    
    params = f'{dataset_name}/nsize{nsize}/fgsm{fgsm}/ncal{ncal}/' \
             f'{_args.classif_loss}/{_args.dloss}/prototypes_{proto_val}/' \
             f'npos{npos}/nneg{nneg}/' \
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
    if len(parts) > base_idx + 3:
        params["Dataset"] = parts[base_idx + 3]
    if len(parts) > base_idx + 4:
        nsize_part = parts[base_idx + 4]
        if nsize_part.startswith("nsize"):
            params["new_size"] = nsize_part[len("nsize"):]
    if len(parts) > base_idx + 5:
        params.setdefault("FGSM", parts[base_idx + 5])
    if len(parts) > base_idx + 6:
        params.setdefault("N_Calibration", parts[base_idx + 6])
    if len(parts) > base_idx + 7:
        params.setdefault("classif_loss", parts[base_idx + 7])
    if len(parts) > base_idx + 8:
        params.setdefault("DLoss", parts[base_idx + 8])
    if len(parts) > base_idx + 9:
        params.setdefault("Prototypes", parts[base_idx + 9])
    if len(parts) > base_idx + 10:
        params.setdefault("NPos", parts[base_idx + 10])
    if len(parts) > base_idx + 11:
        params.setdefault("NNeg", parts[base_idx + 11])
    if len(parts) > base_idx + 12:
        norm_part = parts[base_idx + 12]
        if norm_part.startswith("norm"):
            params.setdefault("Normalize", norm_part[len("norm"):])
        else:
            params.setdefault("Normalize", norm_part)
    if len(parts) > base_idx + 13:
        dist_part = parts[base_idx + 13]
        if dist_part.startswith("dist_"):
            params.setdefault("Dist_Fct", dist_part[len("dist_"):])
        else:
            params.setdefault("Dist_Fct", dist_part)
    if len(parts) > base_idx + 14:
        knn_part = parts[base_idx + 14]
        if knn_part.startswith("knn"):
            params.setdefault("N_Neighbors", knn_part[len("knn"):])
        else:
            params.setdefault("N_Neighbors", knn_part)
    return params


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


def _compute_calibration_metrics(log_path: str):
    """Compute calibration metrics from the saved validation predictions."""
    if not log_path:
        return {"error": "Missing log path"}
    csv_path = os.path.join(log_path, "valid_predictions.csv")
    if not os.path.exists(csv_path):
        return {"error": "valid_predictions.csv not found"}

    try:
        df_cal = pd.read_csv(csv_path)
    except Exception as exc:
        return {"error": f"Could not read valid_predictions.csv: {exc}"}

    required_cols = {"label", f"probs_{LEADERBOARD_CAL_POS_LABEL}"}
    missing = required_cols.difference(df_cal.columns)
    if missing:
        return {"error": f"Missing columns: {', '.join(sorted(missing))}"}

    y_true = (df_cal["label"] == LEADERBOARD_CAL_POS_LABEL).astype(int)
    y_prob = df_cal[f"probs_{LEADERBOARD_CAL_POS_LABEL}"].astype(float)
    valid_mask = ~(np.isnan(y_prob.values) | np.isinf(y_prob.values) | np.isnan(y_true.values.astype(float)))
    y_true_filt = y_true.values[valid_mask].astype(int)
    y_prob_filt = np.clip(y_prob.values[valid_mask], 0.0, 1.0).astype(float)

    uniq_vals = np.unique(y_true_filt)
    if len(y_true_filt) == 0 or set(uniq_vals) != {0, 1}:
        return {
            "error": "Calibration requires binary labels and valid probabilities",
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
        cursor.execute(
            """
             SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct,
                 classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path, model_rank
            FROM best_models_registry
            WHERE model_rank IS NOT NULL
            ORDER BY model_rank ASC
            """
        )
        model_rows = cursor.fetchall()
        use_db_rank = True
    except Exception:
        cursor.execute(
            """
             SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct,
                 classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path
            FROM best_models_registry
            ORDER BY mcc DESC
            """
        )
        model_rows = cursor.fetchall()
        use_db_rank = False

    if not model_rows:
        st.session_state['model_number_map'] = model_number_map
        return model_number_map, best_models_table

    if use_db_rank:
        cols = [
            "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
            "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", "#"
        ]
    else:
        cols = [
            "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
            "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path"
        ]
    df = pd.DataFrame(model_rows, columns=cols)
    group_cols = [
        "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
        "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize", "N_Neighbors",
    ]
    _dedupe_frame = df[group_cols].copy().fillna("").astype(str)
    df["_dedupe_key"] = _dedupe_frame.agg("|".join, axis=1)
    df = df.sort_values("MCC", ascending=False)
    df = df.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"])
    df = df.dropna(subset=["Log Path"])
    df = df[df["Log Path"].astype(str) != ""]
    df = df.reset_index(drop=True)
    if "#" not in df.columns:
        df.insert(0, "#", range(1, len(df) + 1))

    model_number_map = {}
    for _, r in df.iterrows():
        rd = r.to_dict()
        selection_key = _make_model_selection_key(rd)
        model_number_map[selection_key] = rd.get("#", "?")

    st.session_state['model_number_map'] = model_number_map
    st.session_state['best_models_table'] = df.copy()
    return model_number_map, df.copy()


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

