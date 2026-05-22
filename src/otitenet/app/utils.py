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
import pickle
# from otitenet.utils.utils import set_random_seeds  # Import from shared utils

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
    
    # Strategy and components for prototypes
    proto_strat = getattr(_args, 'prototype_strategy', 'mean')
    proto_comp = getattr(_args, 'prototype_components', 1)

    params = f'{dataset_name}/nsize{nsize}/fgsm{fgsm}/ncal{ncal}/' \
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
    
    # Rest of the segments are positional but we can be more robust by checking prefixes
    # Segments start after model_name (base_idx + 3)
    data_parts = parts[base_idx + 3:]
    for p in data_parts:
        if p.startswith("otite_ds_"):
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

    # Fallback to positional indices for fixed-position fields without unique prefixes
    if len(parts) > base_idx + 7:
        params.setdefault("classif_loss", parts[base_idx + 7])
    if len(parts) > base_idx + 8:
        params.setdefault("DLoss", parts[base_idx + 8])
    
    return params


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


def _get_done_manifest_config_keys(task: str):
    """Return the set of completed manifest config keys for a task, or None if unavailable."""
    task_name = _normalize_manifest_value(task)
    if not task_name:
        return None

    cache = st.session_state.setdefault(LEADERBOARD_DONE_MANIFEST_CACHE_KEY, {})
    if task_name in cache:
        return cache[task_name]

    manifest_path = os.path.join("logs", "progresses", task_name, "csv", f"PROD_{task_name}_job_manifest.csv")
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
    
    manifest_path = os.path.join("logs", "progresses", task_name, "csv", f"PROD_{task_name}_job_manifest.csv")
    if not os.path.exists(manifest_path):
        return None
    
    try:
        manifest_df = pd.read_csv(manifest_path)
    except Exception:
        return None
    
    if "job_state" not in manifest_df.columns:
        return None
    
    done_df = manifest_df[manifest_df["job_state"].astype(str).str.strip().str.lower() == "done"].copy()
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

    summary_path = os.path.join(log_path, "run_summary.json")
    if not os.path.exists(summary_path):
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
    except Exception:
        return None

    artifacts = summary.get("artifacts") or {}
    artifact_csv_path = artifacts.get("valid_predictions_csv")
    if not artifact_csv_path:
        return None

    if not os.path.isabs(artifact_csv_path):
        artifact_csv_path = os.path.join(log_path, artifact_csv_path)

    if os.path.exists(artifact_csv_path):
        return artifact_csv_path

    return None


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


def _resolve_split_predictions_csv(log_path: str, split: str):
    """Find train/valid/test predictions CSV from common locations."""
    if not log_path:
        return None

    direct_path = os.path.join(log_path, f"{split}_predictions.csv")
    if os.path.exists(direct_path):
        return direct_path

    summary_path = os.path.join(log_path, "run_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            artifacts = summary.get("artifacts") or {}
            artifact_path = artifacts.get(f"{split}_predictions_csv")

            if artifact_path:
                if not os.path.isabs(artifact_path):
                    artifact_path = os.path.join(log_path, artifact_path)
                if os.path.exists(artifact_path):
                    return artifact_path
        except Exception:
            pass

    return None


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

    # Second: compute AUC from saved prediction CSVs if summary did not contain it
    for split in ["train", "valid", "test"]:
        key = f"{split}_auc"

        if pd.isna(out.get(key, np.nan)):
            csv_path = _resolve_split_predictions_csv(log_path, split)

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
                 classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path, model_rank,
                 prototype_strategy, prototype_components, test_mcc, valid_auc, test_auc
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
                 classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path,
                 prototype_strategy, prototype_components, test_mcc, valid_auc, test_auc
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
            "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", "#",
            "Proto_Strat", "Proto_Comp", "Test_MCC", "Valid_AUC", "Test_AUC",
        ]
    else:
        cols = [
            "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
            "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path",
            "Proto_Strat", "Proto_Comp", "Test_MCC", "Valid_AUC", "Test_AUC",
        ]
    df = pd.DataFrame(model_rows, columns=cols)
    group_cols = [
        "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
        "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize", "N_Neighbors",
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
    df["Valid AUC"] = np.nan
    df["Test AUC"] = np.nan
    for idx, row in df.iterrows():
        df.at[idx, "N_Aug"] = get_run_n_aug(row.get("Log Path"))
        metrics = get_split_mcc_metrics(row.get("Log Path"))
        if metrics:
            df.at[idx, "Train MCC"] = metrics.get('train_mcc', np.nan)
            df.at[idx, "Valid MCC"] = metrics.get('valid_mcc', np.nan)
            df.at[idx, "Test MCC"] = metrics.get('test_mcc', np.nan)
            df.at[idx, "Valid AUC"] = metrics.get('valid_auc', np.nan)
            df.at[idx, "Test AUC"] = metrics.get('test_auc', np.nan)

    valid_sort = pd.to_numeric(df.get("Valid MCC"), errors="coerce")
    mcc_sort = pd.to_numeric(df.get("MCC"), errors="coerce")
    df = df.assign(
        _valid_sort=valid_sort.fillna(-np.inf),
        _mcc_sort=mcc_sort.fillna(-np.inf),
    )
    df = df.sort_values(["_valid_sort", "_mcc_sort"], ascending=[False, False]).reset_index(drop=True)
    df = df.drop(columns=["_valid_sort", "_mcc_sort"])

    df["#"] = np.arange(1, len(df) + 1)
    df = df[["#"] + [c for c in df.columns if c != "#"]]

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


def _merge_optimization_cache_entries(target_cache: dict, source_cache: dict) -> dict:
    """Merge source cache into target, keeping best MCC per n_aug."""
    if not isinstance(source_cache, dict):
        return target_cache
    for n_aug_key, result in source_cache.items():
        try:
            n_aug_int = int(n_aug_key)
        except Exception:
            continue
        current = target_cache.get(n_aug_int)
        if current is None:
            target_cache[n_aug_int] = result
            continue
        current_mcc = float(current.get('best_mcc', -1)) if isinstance(current, dict) else -1
        new_mcc = float(result.get('best_mcc', -1)) if isinstance(result, dict) else -1
        if new_mcc > current_mcc:
            target_cache[n_aug_int] = result
    return target_cache


def load_optimization_cache_dict(_args) -> dict:
    """Load merged optimization cache (unified + legacy paths)."""
    import glob
    cache_path = get_optimization_cache_file_path(_args)
    cache_dir = os.path.dirname(cache_path)
    merged_cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                merged_cache = _merge_optimization_cache_entries(merged_cache, pickle.load(f))
        except Exception as e:
            print(f"[Cache] Could not load optimization cache: {e}")
    legacy_pattern = os.path.join(cache_dir, 'dist_*', 'knn*', 'knn_optimization_cache.pkl')
    for legacy_file in glob.glob(legacy_pattern):
        try:
            with open(legacy_file, 'rb') as f:
                merged_cache = _merge_optimization_cache_entries(merged_cache, pickle.load(f))
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


def _update_classification_head(best_by_config: dict, config: str, mcc: float, details: str = ""):
    """Keep the highest-MCC entry per classification-head config key."""
    if config is None or mcc is None:
        return
    try:
        mcc_f = float(mcc)
    except (TypeError, ValueError):
        return
    if mcc_f < 0:
        return
    current = best_by_config.get(config)
    if current is None or mcc_f > current.get("mcc", -1):
        best_by_config[config] = {"config": str(config), "mcc": mcc_f, "details": details}


def enumerate_classification_heads(_args):
    """List classification heads from the learned-embedding optimization cache.

    Returns one entry per approach (best validation MCC across n_aug), sorted by MCC descending.
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
                _update_classification_head(
                    best_by_config, str(int(k_val)), mcc_val, f"k={int(k_val)}, n_aug={n_aug}"
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
                best_by_config, config, mcc_val, f"n_aug={n_aug}"
            )

    heads = []
    for entry in best_by_config.values():
        cfg = entry["config"]
        heads.append({
            "config": cfg,
            "label": format_classifier_config(cfg),
            "mcc": entry["mcc"],
            "details": entry.get("details", ""),
        })
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
