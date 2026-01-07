import os
import json
import tempfile
import glob
import time

# Disable Streamlit file watcher early to avoid importing/inspecting heavy packages
# (prevents Streamlit from touching packages like torch._classes which can raise)
os.environ.setdefault("STREAMLIT_FILE_WATCHER_TYPE", "none")

import streamlit as st
import debugpy
if 'debugger_attached' not in st.session_state:
    try:
        debugpy.listen(5679)
        print("⏳ Waiting for debugger attach on port 5679...")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        st.session_state.debugger_attached = True
    except Exception as e:
        print("❌ debugpy.listen failed:", e)

# os.environ["STREAMLIT_SECRETS_LOAD_MODE"] = "read_only"
import torch
import pickle
import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd
from PIL import Image
from otitenet.train.train_triplet_new import TrainAE  # If needed for params
from otitenet.data.data_getters import GetData, get_images_loaders, get_images, PerImageNormalize  # Update import path
from otitenet.models.cnn import Net, Net_shap  # Update path if needed
from otitenet.utils.utils import get_empty_traces
from otitenet.logging.shap import log_shap_gradients_only, log_shap_knn_or_deep
from otitenet.logging.grad_cam import (
    log_grad_cam_similarity,
    log_grad_cam_all_classes,
    save_overlay_from_heatmap,
)
from otitenet.logging.metrics import MCC, expected_calibration_error, brier_score
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
except ImportError:
    UMAP = None
import seaborn as sns
from otitenet.utils.update_model_ranks import update_model_ranks
from otitenet.utils.update_model_ranks import update_model_ranks

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
        # Handle cases like 'npos1' or already clean numbers
        return int(float(s))
    except ValueError:
        return 0

def get_model_params_path(_args):
    """Construct the standardized relative path for model parameters folders."""
    # Ensure every numeric param is clean before adding prefix
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

# Your model imports
import argparse
import random

# Set random seeds for reproducibility (must match training)
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

# ---- Load datasets from /data ---- #
data_dir = './data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- MySQL Database Setup ---- #
@st.cache_resource
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="y_user",
            password="password",
            database="results_db",
            buffered=True,
            autocommit=True
        )
        return conn
    except Error as e:
        st.error(f"❌ Database connection error: {e}")
        st.stop()

def create_db():
    try:
        conn = get_db_connection()
        if not conn.is_connected():
            conn.reconnect(attempts=3, delay=1)
        
        # Ping the server to ensure connection is alive
        conn.ping(reconnect=True, attempts=3, delay=1)
        
        cursor = conn.cursor(buffered=True)
        # Minimal check to ensure it works
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return conn, cursor
    except Exception as e:
        # If the cached connection is broken, clear cache and try one more time
        try:
            get_db_connection.clear()
            conn = get_db_connection()
            if not conn.is_connected():
                conn.reconnect(attempts=3, delay=1)
            conn.ping(reconnect=True, attempts=3, delay=1)
            cursor = conn.cursor(buffered=True)
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return conn, cursor
        except Exception as e2:
            if "Too many connections" in str(e2):
                 st.error("❌ MySQL has too many connections. Please restart MySQL service or wait for connections to timeout.")
            else:
                 st.error(f"❌ Database error: {e2}")
            st.stop()


def ensure_results_model_id(conn, cursor):
    """Ensure `results.model_id` exists (migration fallback)."""
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'results'
              AND COLUMN_NAME = 'model_id'
            """
        )
        has_col = cursor.fetchone()[0] > 0
        if not has_col:
            cursor.execute("ALTER TABLE results ADD COLUMN model_id INT NULL AFTER person_id")
            try:
                cursor.execute(
                    """
                    ALTER TABLE results
                    ADD CONSTRAINT fk_results_model_id
                    FOREIGN KEY (model_id) REFERENCES best_models_registry(id)
                    ON DELETE SET NULL
                    """
                )
            except Exception:
                # If FK already exists or add fails, continue without blocking runtime
                pass
            conn.commit()
    except Exception:
        # Soft-fail: do not crash app if schema check fails
        conn.rollback()

def ensure_best_models_registry_nsize(conn, cursor):
    """Ensure `best_models_registry.nsize` exists and backfill from log_path if missing.
    Adds column as INT NULL and parses size from the standard best_models path.
    """
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'best_models_registry'
              AND COLUMN_NAME = 'nsize'
            """
        )
        has_col = cursor.fetchone()[0] > 0
    except Exception:
        # If schema introspection fails, do not block the app
        return

    if not has_col:
        try:
            cursor.execute("ALTER TABLE best_models_registry ADD COLUMN nsize INT NULL")
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            # Could not add column; bail out quietly
            return

    # Backfill missing values from log_path
    try:
        cursor.execute(
            """
            SELECT id, log_path FROM best_models_registry
            WHERE (nsize IS NULL) AND log_path IS NOT NULL AND log_path <> ''
            """
        )
        rows = cursor.fetchall() or []
        for rid, lp in rows:
            try:
                params = extract_params_from_log_path(lp)
                ns = params.get('new_size')
                if ns is None:
                    continue
                val = int(float(ns))
                cursor.execute("UPDATE best_models_registry SET nsize=%s WHERE id=%s", (val, rid))
            except Exception:
                # Skip unparseable rows
                continue
        if rows:
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

def get_image(path, size=-1, normalize='no'):
    ops = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
    if str(normalize).lower() in ['yes', 'true', '1']:
        ops.append(PerImageNormalize())
    
    transform = transforms.Compose(ops)
    original = Image.open(path).convert('RGB')
    if size != -1:
        png = transforms.Resize((size, size))(original)
    else:
        png = original
    print(size, png)
    
    return transform(original).unsqueeze(0), transform(png).unsqueeze(0)


def choose_dataset(label, datasets, default=None, key=None):
    """Return a dataset choice while keeping session_state and defaults in sync."""
    if len(datasets) == 0:
        st.warning(f"No datasets found in {data_dir}.")
        return None

    # Clear stale session values that are no longer valid to avoid selectbox errors.
    if key and key in st.session_state and st.session_state[key] not in datasets:
        st.session_state.pop(key, None)

    # Prefer an existing, valid session value if present; otherwise fall back to default/first.
    current = None
    if key and key in st.session_state and st.session_state[key] in datasets:
        current = st.session_state[key]
    elif default in datasets:
        current = default
    elif datasets:
        current = datasets[0]

    index = datasets.index(current) if current in datasets else 0

    if len(datasets) <= 3:
        return st.radio(label, datasets, index=index, key=key)
    else:
        return st.selectbox(label, datasets, index=index, key=key)


# ---- Database Lookup ---- #
def check_ds_exists(cursor, filename, args):
    cursor.execute(''' 
        SELECT pred_label, confidence, log_path FROM results
        WHERE filename=%s AND person_id=%s AND model_name=%s AND task=%s AND path=%s
              AND n_neighbors=%s AND nsize=%s AND fgsm=%s AND normalize=%s AND
              n_calibration=%s AND classif_loss=%s AND dloss=%s AND dist_fct=%s AND prototypes=%s
              AND npos=%s AND nneg=%s
    ''', (
        filename,
        st.session_state.person_id,
        args.model_name,
        args.task,
        args.path,
        str(args.n_neighbors),
        str(args.new_size),
        str(args.fgsm),
        args.normalize,
        str(args.n_calibration),
        args.classif_loss,
        args.dloss,
        args.dist_fct,
        args.prototypes_to_use,
        str(args.n_positives),
        str(args.n_negatives),
    ))

    row = cursor.fetchone()
    return row


def list_image_results(cursor, person_id, filename):
    """Return all stored analyses for this image and person."""
    cursor.execute('''
        SELECT model_name, task, pred_label, confidence, log_path, timestamp,
               nsize, fgsm, normalize, n_calibration, classif_loss, dloss, dist_fct,
               prototypes, npos, nneg, n_neighbors, model_id
        FROM results
        WHERE filename=%s AND person_id=%s
        ORDER BY timestamp DESC
    ''', (filename, person_id))
    return cursor.fetchall()

def build_params_from_args(_args, keys):
    parts = []
    for key in keys:
        value = getattr(_args, key, None)
        if value is not None:
            parts.append(f"{key}{value}")
    return "/".join(parts)


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


def fetch_model_by_log_path(cursor, log_path: str):
    cursor.execute(
        """
        SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
               n_calibration, accuracy, mcc, normalize, n_neighbors, log_path
        FROM best_models_registry
        WHERE log_path=%s
        LIMIT 1
        """,
        (log_path,),
    )
    return cursor.fetchone()


def _resolve_model_id(cursor, args, log_path: str):
    """Best-effort lookup of best_models_registry.id for the current run."""
    try:
        explicit = getattr(args, "model_id", None)
        if explicit not in (None, "", "None"):
            return int(explicit)
    except Exception:
        pass

    if log_path:
        candidates = [log_path]
        lp_trim = str(log_path).rstrip("/")
        if lp_trim.endswith("/queries"):
            candidates.append(lp_trim[: -len("/queries")])
        for lp in _unique_preserve_order(candidates):
            try:
                cursor.execute("SELECT id FROM best_models_registry WHERE log_path=%s LIMIT 1", (lp,))
                row = cursor.fetchone()
                if row:
                    return int(row[0])
            except Exception:
                pass

    try:
        cursor.execute(
            """
            SELECT id FROM best_models_registry
            WHERE model_name=%s AND nsize=%s AND fgsm=%s AND prototypes=%s AND npos=%s AND nneg=%s AND dloss=%s
                  AND dist_fct=%s AND classif_loss=%s AND n_calibration=%s AND normalize=%s AND n_neighbors=%s
            ORDER BY id ASC
            LIMIT 1
            """,
            (
                getattr(args, "model_name", None),
                str(getattr(args, "new_size", None)),
                str(getattr(args, "fgsm", None)),
                getattr(args, "prototypes_to_use", None),
                str(getattr(args, "n_positives", None)),
                str(getattr(args, "n_negatives", None)),
                getattr(args, "dloss", None),
                getattr(args, "dist_fct", None),
                getattr(args, "classif_loss", None),
                str(getattr(args, "n_calibration", None)),
                getattr(args, "normalize", None),
                str(getattr(args, "n_neighbors", None)),
            ),
        )
        row = cursor.fetchone()
        if row:
            return int(row[0])
    except Exception:
        return None

    return None


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


def _unique_preserve_order(items):
    """Return a list of unique items preserving first-seen order."""
    return list(dict.fromkeys(items))


def _make_model_selection_key(row_dict: dict) -> str:
    """Create a stable unique key for a model parameter combination.

    We cannot rely on `log_path` being unique (DB can contain multiple rows with
    different params but same log_path), so selectors must key off the full
    parameter combination.
    """
    import math
    import numpy as np

    def norm(v):
        """Normalize values to stable strings for key generation.

        - None/NaN → ""
        - Numeric → canonical integer string (e.g., 224, 32)
        - String → trimmed, lowercased
        """
        try:
            if v is None:
                return ""
            # Handle pandas/numpy NaN and regular float('nan')
            if isinstance(v, float) and math.isnan(v):
                return ""
            if isinstance(v, (np.floating,)) and np.isnan(v):
                return ""
            # Numeric to canonical int string when applicable
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                # If integer-like float, cast to int
                if float(v).is_integer():
                    return str(int(v))
                # Else keep compact float string
                s = ("%g" % float(v)).strip()
                return s.lower()
            # Strings: trim + lowercase
            s = str(v).strip()
            if s.lower() in {"nan", "none"}:
                return ""
            return s.lower()
        except Exception:
            return ""

    parts = [
        norm(row_dict.get("Model Name", "")),
        norm(row_dict.get("NSize", "")),
        norm(row_dict.get("FGSM", "")),
        norm(row_dict.get("Prototypes", "")),
        norm(row_dict.get("NPos", "")),
        norm(row_dict.get("NNeg", "")),
        norm(row_dict.get("DLoss", "")),
        norm(row_dict.get("Dist_Fct", "")),
        norm(row_dict.get("Classif_Loss", "")),
        norm(row_dict.get("N_Calibration", "")),
        norm(row_dict.get("Normalize", "")),
        norm(row_dict.get("N_Neighbors", "")),
    ]
    # Do NOT include log_path in the selection key. It can legitimately differ
    # across tables for the same parameter combination and would break mapping.
    # Keep keys strictly based on parameter values to ensure stable cross-view mapping.
    return "|".join(parts)

def _lookup_model_number(mapping_rd: dict, model_number_map: dict) -> str:
    """Resolve the model number from the map with graceful fallback.

    1) Try exact key match with all parameters.
    2) If not found, try a fuzzy match ignoring `N_Neighbors` when it's missing
       in past results, matching on the other parameters only.
    """
    # First, attempt exact match
    exact_key = _make_model_selection_key(mapping_rd)
    num = model_number_map.get(exact_key)
    if num is not None:
        return num

    # Fallback: fuzzy match on all fields except N_Neighbors
    import math
    import numpy as np
    def norm(v):
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

    expected_parts_wo_neighbors = [
        norm(mapping_rd.get("Model Name", "")),
        norm(mapping_rd.get("NSize", "")),
        norm(mapping_rd.get("FGSM", "")),
        norm(mapping_rd.get("Prototypes", "")),
        norm(mapping_rd.get("NPos", "")),
        norm(mapping_rd.get("NNeg", "")),
        norm(mapping_rd.get("DLoss", "")),
        norm(mapping_rd.get("Dist_Fct", "")),
        norm(mapping_rd.get("Classif_Loss", "")),
        norm(mapping_rd.get("N_Calibration", "")),
        norm(mapping_rd.get("Normalize", "")),
    ]
    want_neighbors = norm(mapping_rd.get("N_Neighbors", ""))

    for k, v in model_number_map.items():
        parts = k.split("|")
        # keys contain 12 parts (including neighbors) in the same order
        if len(parts) < 12:
            continue
        # Compare without neighbors
        if parts[:11] == expected_parts_wo_neighbors:
            # If past result has no neighbors recorded, accept first match
            if want_neighbors == "":
                return v
            # Else require neighbors to match too
            if parts[11] == want_neighbors:
                return v

    # No match found
    return "?"


def _ensure_model_number_map(cursor):
    """Ensure model_number_map and best_models_table exist in session state.

    Returns (model_number_map, best_models_table).
    """
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

def get_args(selected_params=None):
    if selected_params is None:
        selected_params = st.session_state.get('selected_model_params', {})
    selected_params_version = st.session_state.get('selected_params_version')
    last_synced_version = st.session_state.get('selected_params_last_sync')
    should_sync = bool(selected_params) and selected_params_version != last_synced_version
    if should_sync:
        st.session_state['selected_params_last_sync'] = selected_params_version

    def sync_value(key, value):
        if should_sync and value is not None:
            st.session_state[key] = value
    # ---- User selects paths ---- #
    with st.sidebar:
        st.header("Model Parameters")
        
        # ---- Model Selection from Leaderboard ---- #
        st.subheader("📋 Quick Model Selection")
        try:
            model_rows = []
            use_db_rank = False
            # Try to use model_rank if available
            try:
                update_model_ranks()
                cursor.execute(
                    """
                    SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
                           n_calibration, accuracy, mcc, normalize, n_neighbors, log_path, model_rank
                    FROM best_models_registry
                    WHERE model_rank IS NOT NULL
                    ORDER BY model_rank ASC
                    """
                )
                model_rows = cursor.fetchall()
                use_db_rank = True
            except Exception:
                # Fallback if model_rank column doesn't exist yet
                cursor.execute(
                    """
                    SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
                           n_calibration, accuracy, mcc, normalize, n_neighbors, log_path
                    FROM best_models_registry
                    ORDER BY mcc DESC
                    """
                )
                model_rows = cursor.fetchall()
                use_db_rank = False

            if model_rows:
                # Build the same deduped view as Tab1, then take the top unique rows.
                if use_db_rank:
                    _cols = [
                        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                        "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", "#"
                    ]
                else:
                    _cols = [
                        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                        "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path"
                    ]
                _df = pd.DataFrame(model_rows, columns=_cols)
                
                _group_cols = [
                    "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
                    "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize", "N_Neighbors",
                ]
                _dedupe_frame = _df[_group_cols].copy().fillna("").astype(str)
                # Use row-wise join to ensure a Series (avoids pandas agg returning a DataFrame)
                _df["_dedupe_key"] = _dedupe_frame.apply(lambda r: "|".join(r.values.tolist()), axis=1)
                _df = _df.sort_values("MCC", ascending=False)
                _df = _df.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"])
                _df = _df.dropna(subset=["Log Path"])
                _df = _df[_df["Log Path"].astype(str) != ""]
                _df = _df.reset_index(drop=True)
                
                # model_rank is already in the dataframe from the database query (if available)
                # Otherwise add dynamic numbering
                if "#" in _df.columns:
                    # Move it to the first column
                    cols = ["#"] + [col for col in _df.columns if col != "#"]
                    _df = _df[cols]
                else:
                    # Add dynamic numbering as fallback
                    _df.insert(0, "#", range(1, len(_df) + 1))

                key_to_row = {}
                key_to_label = {}
                model_number_map = {}
                for _, r in _df.iterrows():
                    rd = r.to_dict()
                    selection_key = _make_model_selection_key(rd)
                    if selection_key in key_to_row:
                        continue
                    key_to_row[selection_key] = rd
                    model_num = rd.get("#", "?")
                    model_number_map[selection_key] = model_num
                    try:
                        key_to_label[selection_key] = (
                            f"#{model_num} - {rd.get('Model Name')} (Size:{rd.get('NSize')}, MCC:{float(rd.get('MCC')):.3f}, Dist:{rd.get('Dist_Fct')}, Norm:{rd.get('Normalize')})"
                        )
                    except Exception:
                        key_to_label[selection_key] = (
                            f"#{model_num} - {rd.get('Model Name')} (Size:{rd.get('NSize')}, MCC:{rd.get('MCC')}, Dist:{rd.get('Dist_Fct')}, Norm:{rd.get('Normalize')})"
                        )
                st.session_state['model_number_map'] = model_number_map

                # Auto-apply the top-ranked model so defaults come from Quick Model Selection
                if not selected_params and len(_df) > 0:
                    try:
                        best_row = _df.iloc[0].to_dict()
                        best_row.update(extract_params_from_log_path(best_row.get("Log Path")))
                        if "N_Neighbors" in best_row:
                            best_row["n_neighbors"] = best_row["N_Neighbors"]
                        if "NSize" in best_row:
                            best_row["new_size"] = best_row["NSize"]
                        if "Dist_Fct" in best_row:
                            best_row["dist_fct"] = best_row["Dist_Fct"]
                        if "Classif_Loss" in best_row:
                            best_row["classif_loss"] = best_row["Classif_Loss"]
                        best_row["model_id"] = best_row.get("Model ID")

                        selected_params = best_row
                        st.session_state.selected_model_params = best_row
                        st.session_state.selected_params_version = st.session_state.get('selected_params_version', 0) + 1
                        st.session_state.selected_model_log_path = best_row.get('Log Path')
                        best_key = _make_model_selection_key(best_row)
                        st.session_state.selected_model_selection_key = best_key
                        st.session_state.selected_model_version = st.session_state.get('selected_model_version', 0) + 1
                        st.session_state.sidebar_best_model_key = best_key
                        should_sync = True
                    except Exception as auto_exc:
                        st.warning(f"Could not auto-apply best model: {auto_exc}")

                sidebar_options = _unique_preserve_order(list(key_to_row.keys()))

                # Sync this widget to the canonical selection (log_path)
                canonical_key = st.session_state.get('selected_model_selection_key')
                if canonical_key and canonical_key in sidebar_options:
                    last = st.session_state.get('sidebar_best_model_last_sync')
                    ver = st.session_state.get('selected_model_version')
                    if ver is not None and ver != last:
                        st.session_state['sidebar_best_model_key'] = canonical_key
                        st.session_state['sidebar_best_model_last_sync'] = ver

                # Ensure the selectbox value is valid before rendering to avoid "not in iterable" errors.
                current_sidebar_key = st.session_state.get('sidebar_best_model_key')
                if current_sidebar_key not in sidebar_options:
                    st.session_state['sidebar_best_model_key'] = sidebar_options[0] if sidebar_options else None

                selected_key = st.selectbox(
                    "Select from best models:",
                    options=sidebar_options,
                    format_func=lambda k: key_to_label.get(k, str(k)),
                    index=0,
                    key="sidebar_best_model_key",
                )

                if selected_key and st.button("✅ Apply Selected Model", key="apply_sidebar_model"):
                    rd = key_to_row.get(selected_key)
                    if rd:
                        model_dict = {
                            'Model ID': rd.get('Model ID'),
                            'model_id': rd.get('Model ID'),
                            'Model Name': rd.get('Model Name'),
                            'NSize': rd.get('NSize'),
                            'FGSM': rd.get('FGSM'),
                            'Prototypes': rd.get('Prototypes'),
                            'NPos': rd.get('NPos'),
                            'NNeg': rd.get('NNeg'),
                            'DLoss': rd.get('DLoss'),
                            'Dist_Fct': rd.get('Dist_Fct'),
                            'Classif_Loss': rd.get('Classif_Loss'),
                            'N_Calibration': rd.get('N_Calibration'),
                            'Accuracy': rd.get('Accuracy'),
                            'MCC': rd.get('MCC'),
                            'Normalize': rd.get('Normalize'),
                            'N_Neighbors': rd.get('N_Neighbors'),
                            'Log Path': rd.get('Log Path'),
                        }
                        model_dict.update(extract_params_from_log_path(model_dict.get("Log Path")))
                        if "N_Neighbors" in model_dict:
                            model_dict["n_neighbors"] = model_dict["N_Neighbors"]
                        if "NSize" in model_dict:
                            model_dict["new_size"] = model_dict["NSize"]
                        if "Dist_Fct" in model_dict:
                            model_dict["dist_fct"] = model_dict["Dist_Fct"]
                        if "Classif_Loss" in model_dict:
                            model_dict["classif_loss"] = model_dict["Classif_Loss"]

                        st.session_state.selected_model_params = model_dict
                        st.session_state.selected_params_version = st.session_state.get('selected_params_version', 0) + 1
                        st.session_state.selected_model_log_path = model_dict.get('Log Path')
                        st.session_state.selected_model_selection_key = selected_key
                        st.session_state.selected_model_version = st.session_state.get('selected_model_version', 0) + 1

                        # Eagerly sync key UI widgets so the next run reflects selection immediately
                        try:
                            if model_dict.get('Dataset'):
                                st.session_state['dataset_selectbox'] = model_dict['Dataset']
                        except Exception:
                            pass
                        try:
                            if model_dict.get('Task'):
                                st.session_state['task_selectbox'] = model_dict['Task']
                        except Exception:
                            pass
                        try:
                            if model_dict.get('new_size') is not None:
                                st.session_state['new_size_input'] = int(float(model_dict['new_size']))
                        except Exception:
                            pass
                        try:
                            if model_dict.get('n_neighbors') is not None:
                                st.session_state['n_neighbors_input'] = int(float(model_dict['n_neighbors']))
                        except Exception:
                            pass
                        try:
                            if model_dict.get('Dist_Fct'):
                                st.session_state['dist_fct_selectbox'] = str(model_dict.get('Dist_Fct')).strip().lower()
                        except Exception:
                            pass
                        # Other selectors (best-effort; choose_dataset etc. will validate on rerun)
                        for key_name, dict_key in [
                            ('model_name_selectbox', 'Model Name'),
                            ('fgsm_selectbox', 'FGSM'),
                            ('n_calibration_selectbox', 'N_Calibration'),
                            ('classif_loss_selectbox', 'Classif_Loss'),
                            ('dloss_selectbox', 'DLoss'),
                            ('prototypes_selectbox', 'Prototypes'),
                            ('npos_selectbox', 'NPos'),
                            ('nneg_selectbox', 'NNeg'),
                        ]:
                            try:
                                v = model_dict.get(dict_key)
                                if v is not None:
                                    st.session_state[key_name] = v
                            except Exception:
                                pass
                        try:
                            if model_dict.get('Normalize') in ['yes', 'no']:
                                st.session_state['normalize_input'] = model_dict.get('Normalize')
                        except Exception:
                            pass
                        st.success(f"✅ Applied: {model_dict.get('Model Name')}")
                        st.rerun()
        except Exception as e:
            st.warning(f"Could not load models: {e}")

        st.divider()
        
        available_datasets = sorted([
            name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))
        ])
        dataset_default = selected_params.get('Dataset')
        if dataset_default not in available_datasets:
            dataset_default = 'otite_ds_64' if 'otite_ds_64' in available_datasets else (available_datasets[0] if available_datasets else None)
        selected_path = choose_dataset(
            "Select --path dataset", available_datasets, default=dataset_default, key="dataset_selectbox"
        )
        # selected_path = choose_dataset(
        #     "Select --path dataset", available_datasets, default='otite_ds_64'
        # )  # No key needed, handled in choose_dataset
        # selected_path_original = choose_dataset(
        #     "Select --path_original dataset", available_datasets, default='otite_ds_-1'
        # )

        # ---- Other args with defaults ---- #
        # Prefer explicit new_size from selection; else infer from dataset suffix; fallback to 224
        new_size_raw = selected_params.get('new_size')
        if new_size_raw is None and selected_path:
            try:
                new_size = int(str(selected_path).split('_')[-1])
            except Exception:
                new_size = 224
        else:
            try:
                new_size = int(new_size_raw) if new_size_raw is not None else 224
            except Exception:
                new_size = 224
        # Expose new_size control like other parameters
        if should_sync and new_size is not None:
            st.session_state['new_size_input'] = int(new_size)
        new_size = st.number_input("new_size", value=int(st.session_state.get('new_size_input', new_size)), step=1, key="new_size_input")

        task_root = 'logs/best_models'
        task_list = sorted(os.listdir(task_root)) if os.path.isdir(task_root) else []
        if not task_list:
            st.error("No tasks found under logs/best_models.")
            st.stop()
        task_default = selected_params.get("Task")
        if task_default not in task_list:
            task_default = task_list[0]
        sync_value('task_selectbox', task_default)
        task = st.selectbox("task", task_list, key="task_selectbox")

        model_dir = os.path.join(task_root, task)
        model_name_list = [name for name in sorted(os.listdir(model_dir)) if not name.endswith('.csv')] if os.path.isdir(model_dir) else []
        if not model_name_list:
            st.error(f"No models found for task {task}.")
            st.stop()
        model_name_default = selected_params.get("Model Name")
        # remove files ending in .json or any extension really
        model_name_list = [name for name in model_name_list if not any(name.endswith(ext) for ext in ['.json', '.csv', '.txt'])]
        if model_name_default not in model_name_list:
            model_name_default = model_name_list[0]
        sync_value('model_name_selectbox', model_name_default)
        model_name = st.selectbox("Model Name", model_name_list, key="model_name_selectbox")

        fgsm_dir = os.path.join(model_dir, model_name, selected_path if selected_path else 'otite_ds_64', f'nsize{new_size}')
        fgsm_list = sorted(os.listdir(fgsm_dir)) if os.path.isdir(fgsm_dir) else []
        if not fgsm_list:
            st.error(f"No FGSM entries found in {fgsm_dir}.")
            st.stop()
        fgsm_default = selected_params.get("FGSM")
        if fgsm_default not in fgsm_list:
            fgsm_default = fgsm_list[0]
        sync_value('fgsm_selectbox', fgsm_default)
        fgsm = st.selectbox("fgsm", fgsm_list, key="fgsm_selectbox")

        n_cal_dir = os.path.join(fgsm_dir, fgsm)
        n_calibration_list = sorted(os.listdir(n_cal_dir)) if os.path.isdir(n_cal_dir) else []
        if not n_calibration_list:
            st.error(f"No calibration folders found in {n_cal_dir}.")
            st.stop()
        n_calibration_default = selected_params.get("N_Calibration")
        if n_calibration_default not in n_calibration_list:
            n_calibration_default = n_calibration_list[0]
        sync_value('n_calibration_selectbox', n_calibration_default)
        n_calibration = st.selectbox("n_calibration", n_calibration_list, key="n_calibration_selectbox")

        classif_dir = os.path.join(n_cal_dir, n_calibration)
        classif_loss_list = sorted(os.listdir(classif_dir)) if os.path.isdir(classif_dir) else []
        if not classif_loss_list:
            st.error(f"No classif_loss folders found in {classif_dir}.")
            st.stop()
        classif_loss_default = selected_params.get("classif_loss")
        if classif_loss_default not in classif_loss_list:
            classif_loss_default = classif_loss_list[0]
        sync_value('classif_loss_selectbox', classif_loss_default)
        classif_loss = st.selectbox("classif_loss", classif_loss_list, key="classif_loss_selectbox")

        dloss_dir = os.path.join(classif_dir, classif_loss)
        dloss_list = sorted(os.listdir(dloss_dir)) if os.path.isdir(dloss_dir) else []
        if not dloss_list:
            st.error(f"No dloss folders found in {dloss_dir}.")
            st.stop()
        dloss_default = selected_params.get("DLoss")
        if dloss_default not in dloss_list:
            dloss_default = dloss_list[0]
        sync_value('dloss_selectbox', dloss_default)
        dloss = st.selectbox("dloss", dloss_list, key="dloss_selectbox")

        proto_dir = os.path.join(dloss_dir, dloss)
        prototypes_list = sorted(os.listdir(proto_dir)) if os.path.isdir(proto_dir) else []
        if not prototypes_list:
            st.error(f"No prototype folders found in {proto_dir}.")
            st.stop()
        prototypes_default = selected_params.get("Prototypes")
        if prototypes_default not in prototypes_list:
            prototypes_default = prototypes_list[0]
        sync_value('prototypes_selectbox', prototypes_default)
        prototypes_to_use = st.selectbox("prototypes_to_use", prototypes_list, key="prototypes_selectbox")

        npos_dir = os.path.join(proto_dir, prototypes_to_use)
        n_positives_list = sorted(os.listdir(npos_dir)) if os.path.isdir(npos_dir) else []
        if not n_positives_list:
            st.error(f"No npos folders found in {npos_dir}.")
            st.stop()
        npos_default = str(selected_params.get("NPos")) if selected_params.get("NPos") is not None else None
        if npos_default not in n_positives_list:
            npos_default = n_positives_list[0]
        sync_value('npos_selectbox', npos_default)
        n_positives = st.selectbox('npos', n_positives_list, key="npos_selectbox")

        nneg_dir = os.path.join(npos_dir, n_positives)
        n_negatives_list = sorted(os.listdir(nneg_dir)) if os.path.isdir(nneg_dir) else []
        if not n_negatives_list:
            st.error(f"No nneg folders found in {nneg_dir}.")
            st.stop()
        nneg_default = str(selected_params.get("NNeg")) if selected_params.get("NNeg") is not None else None
        if nneg_default not in n_negatives_list:
            nneg_default = n_negatives_list[0]
        sync_value('nneg_selectbox', nneg_default)
        n_negatives = st.selectbox('nneg', n_negatives_list, key="nneg_selectbox")

        # Apply optimized k from Tab 1 KNN Optimization if available
        if 'optimized_k_value' in st.session_state:
            n_neighbors_default = st.session_state.pop('optimized_k_value')
            should_sync = True  # Force sync so the widget picks up the new value

        n_neighbors_default = selected_params.get("n_neighbors")
        if should_sync and n_neighbors_default is not None:
            try:
                st.session_state['n_neighbors_input'] = int(n_neighbors_default)
            except (TypeError, ValueError):
                st.session_state['n_neighbors_input'] = 1
        n_neighbors_default_value = st.session_state.get('n_neighbors_input', 1)
        n_neighbors = st.number_input("n_neighbors", value=int(n_neighbors_default_value), step=1, key="n_neighbors_input")

        normalize_default = selected_params.get("Normalize", "no")
        if should_sync and normalize_default is not None:
            st.session_state['normalize_input'] = normalize_default
        normalize = st.selectbox("normalize", ['yes', 'no'], index=1 if normalize_default == "no" else 0, key="normalize_input")

        device = st.selectbox("device", ['cpu', 'cuda'], index=1, key="device_selectbox")
        valid_dataset_default = selected_params.get('valid_dataset', 'Banque_Viscaino_Chili_2020')
        if should_sync and valid_dataset_default is not None:
            st.session_state['valid_dataset_input'] = valid_dataset_default
        valid_dataset = st.text_input("valid_dataset", value=st.session_state.get('valid_dataset_input', valid_dataset_default), key="valid_dataset_input")
        dist_fct_options = ['euclidean', 'cosine']
        dist_fct_default = str(selected_params.get('dist_fct', 'euclidean')).strip().lower()
        if dist_fct_default not in dist_fct_options:
            dist_fct_default = 'euclidean'

        # If Streamlit has a stale value for this widget, it can crash with
        # "<value> is not in iterable" during serialization. Repair it before rendering.
        current_dist_fct = st.session_state.get('dist_fct_selectbox')
        if current_dist_fct is not None:
            current_dist_fct = str(current_dist_fct).strip().lower()
        if current_dist_fct not in dist_fct_options:
            st.session_state['dist_fct_selectbox'] = dist_fct_default

        if should_sync and dist_fct_default is not None:
            st.session_state['dist_fct_selectbox'] = dist_fct_default

        dist_fct = st.selectbox(
            "dist_fct",
            dist_fct_options,
            index=dist_fct_options.index(st.session_state['dist_fct_selectbox']),
            key="dist_fct_selectbox",
        )
        # new_size = st.number_input("new_size", value=224, step=1)
        # seed = st.number_input("seed", value=42, step=1)
        # bs = st.number_input("bs", value=32, step=1)
        # groupkfold = st.number_input("groupkfold", value=1, step=1)
        # random_recs = st.number_input("random_recs", value=0, step=1)

        # Explanation parameters are fixed here to avoid cluttering the sidebar UI.
        grad_cam_layer = int(st.session_state.get('grad_cam_layer', 7))
        grad_cam_alpha = float(st.session_state.get('grad_cam_alpha', 0.55))
        shap_layer = int(st.session_state.get('shap_layer', 4))

        st.session_state['grad_cam_layer'] = grad_cam_layer
        st.session_state['grad_cam_alpha'] = grad_cam_alpha
        st.session_state['shap_layer'] = shap_layer

    # ---- Build args Namespace ---- #
    args = argparse.Namespace(
        model_name=model_name,
        fgsm=fgsm,
        n_calibration=n_calibration,
        n_neighbors=n_neighbors,
        dloss=dloss,
        prototypes_to_use=prototypes_to_use,
        n_positives=n_positives,
        n_negatives=n_negatives,
        # new_size=new_size,
        device=device,
        task=task,
        classif_loss=classif_loss,
        # seed=seed,
        path=os.path.join(data_dir, selected_path) if selected_path else None,
        # path_original=os.path.join(data_dir, selected_path_original) if selected_path_original else None,
        # bs=bs,
        # groupkfold=groupkfold,
        # random_recs=random_recs,
        valid_dataset=valid_dataset,
        dist_fct=dist_fct,
        grad_cam_layer=grad_cam_layer,
        grad_cam_alpha=grad_cam_alpha,
        shap_layer=shap_layer
    )

    # TODO REMOVE THESE PARAMETERS
    args.new_size = new_size
    args.bs = 32
    args.groupkfold = 1
    args.random_recs = 0
    args.seed = 42
    args.normalize = normalize
    args.model_id = selected_params.get('model_id') or selected_params.get('Model ID')

    return args


# ---- Load Model and Prototypes ---- #
@st.cache_resource
def _resolve_model_paths(base_dir: str, normalize_val: str, dist_fct_val: str, requested_k: int):
    """Find model/prototype paths, falling back to available knn folders if the requested one is missing."""
    norm_dir = os.path.join(base_dir, f'norm{normalize_val}', f'dist_{dist_fct_val}')
    requested_dir = os.path.join(norm_dir, f'knn{requested_k}')
    model_path = os.path.join(requested_dir, 'model.pth')
    proto_path = os.path.join(requested_dir, 'prototypes.pkl')
    if os.path.exists(model_path) and os.path.exists(proto_path):
        return model_path, proto_path, requested_k

    # Fallback: pick the highest available knn folder under this dist
    try:
        candidates = []
        for entry in os.listdir(norm_dir):
            if entry.startswith('knn'):
                try:
                    k_val = int(entry[len('knn'):])
                    cand_model = os.path.join(norm_dir, entry, 'model.pth')
                    cand_proto = os.path.join(norm_dir, entry, 'prototypes.pkl')
                    if os.path.exists(cand_model) and os.path.exists(cand_proto):
                        candidates.append((k_val, cand_model, cand_proto))
                except Exception:
                    continue
        if candidates:
            # choose the closest >= requested, else max
            candidates.sort(key=lambda x: (abs(x[0]-requested_k), -x[0]))
            best_k, best_model, best_proto = candidates[0]
            return best_model, best_proto, best_k
    except Exception:
        pass

    return model_path, proto_path, requested_k


def load_model_and_prototypes(_args):
    # Ensure seeds are set for reproducible split
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    
    data_getter = GetData(_args.path, _args.valid_dataset, _args)
    data, unique_labels, unique_batches = data_getter.get_variables()
    n_cats = len(unique_labels)
    n_batches = len(unique_batches)

    params = get_model_params_path(_args)
    parts = params.split('/')
    # remove the last three components: norm..., dist_..., knn...
    base_params = '/'.join(parts[:-3])
    base_dir = f'logs/best_models/{_args.task}/{_args.model_name}/{base_params}'
    model_path, proto_path, resolved_k = _resolve_model_paths(base_dir, _args.normalize, _args.dist_fct, int(_args.n_neighbors))

    if not (os.path.exists(model_path) and os.path.exists(proto_path)):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # If we had to fall back, keep args in sync so downstream cache keys are correct
    try:
        _args.n_neighbors = resolved_k
    except Exception:
        pass

    try:
        model = Net(_args.device, n_cats=n_cats, n_batches=n_batches,
                    model_name=_args.model_name, is_stn=0,
                    n_subcenters=n_batches)
        model.load_state_dict(torch.load(model_path, map_location=_args.device))
        model.to(_args.device)
        model.eval()

        shap_model = Net_shap(_args.device, n_cats=n_cats, n_batches=n_batches,
                              model_name=_args.model_name, is_stn=0,
                              n_subcenters=n_batches)
        shap_model.load_state_dict(torch.load(model_path, map_location=_args.device))
        shap_model.to(_args.device)
        shap_model.eval()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
            print("Warning: GPU OOM during model loading. Falling back to CPU.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            _args.device = 'cpu'
            device_str = 'cpu' # Update local var if needed, though _args is used
            
            model = Net('cpu', n_cats=n_cats, n_batches=n_batches,
                        model_name=_args.model_name, is_stn=0,
                        n_subcenters=n_batches)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.to('cpu')
            model.eval()

            shap_model = Net_shap('cpu', n_cats=n_cats, n_batches=n_batches,
                                  model_name=_args.model_name, is_stn=0,
                                  n_subcenters=n_batches)
            shap_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            shap_model.to('cpu')
            shap_model.eval()
        else:
            raise e

    with open(proto_path, 'rb') as f:
        proto_obj = pickle.load(f)
    prototypes = {'combined': proto_obj.prototypes, 'class': proto_obj.class_prototypes, 'batch': proto_obj.batch_prototypes}

    return model, shap_model, prototypes, _args.new_size, _args.device, data, unique_labels, unique_batches, data_getter

# Safe cache clear for the model/prototype loader
def _clear_cached_model():
    try:
            _clear_cached_model()
    except Exception:
        pass

# ---- Fast KNN Cache for Inference ---- #
def _get_model_cache_key_from_args(_args):
    """Build a stable cache key for the current model selection."""
    return "|".join([
        str(_args.path),
        str(_args.device),
        str(_args.model_name),
        str(_args.new_size),
        str(_args.fgsm),
        str(_args.prototypes_to_use),
        str(_args.n_positives),
        str(_args.n_negatives),
        str(_args.dloss),
        str(_args.dist_fct),
        str(_args.classif_loss),
        str(_args.n_calibration),
        str(_args.normalize),
        str(_args.n_neighbors),
    ])

def get_or_build_knn(_args, data, unique_labels, unique_batches, prototypes):
    """Return a cached KNN for the selected model; build once if missing.

    This avoids re-running predict loops for 'valid'/'test' during inference.
    """
    cache = st.session_state.setdefault('knn_cache', {})
    key = _get_model_cache_key_from_args(_args)
    if key in cache:
        return cache[key]['knn'], cache[key]['unique_labels']

    # Build loaders and encode 'train' set once to fit KNN
    train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                    log_inputs=False, log_plots=False, log_tb=False, log_neptune=False,
                    log_mlflow=False, groupkfold=_args.groupkfold)
    train.n_batches = len(unique_batches)
    train.n_cats = len(unique_labels)
    train.unique_batches = unique_batches
    train.unique_labels = unique_labels
    train.epoch = 1
    train.params = {
        'n_neighbors': _args.n_neighbors,
        'lr': 0,
        'wd': 0,
        'smoothing': 0,
        'is_transform': 0,
        'valid_dataset': _args.valid_dataset
    }
    train.set_arcloss()

    lists, traces = get_empty_traces()
    loaders = get_images_loaders(
        data=data,
        random_recs=_args.random_recs,
        weighted_sampler=0,
        is_transform=0,
        samples_weights=None,
        epoch=1,
        unique_labels=unique_labels,
        triplet_dloss=_args.dloss, bs=_args.bs,
        prototypes_to_use=_args.prototypes_to_use,
        prototypes=prototypes,
        size=_args.new_size,
        normalize=_args.normalize,
    )

    # Encode the train set
    with torch.no_grad():
        try:
            # Minimal model load just for encoding: reuse cached model via load_model_and_prototypes
            model, _, _, _, _, _, _, _, _ = load_model_and_prototypes(_args)
            train.model = model
            _, lists, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
        except Exception as e:
            st.error(f"❌ Could not encode training set for KNN: {e}")
            raise

    # Fit KNN
    import numpy as np
    train_encs = np.concatenate(lists['train']['encoded_values'])
    train_cats = np.concatenate(lists['train']['cats'])
    if _args.classif_loss not in ['ce', 'hinge']:
        nn_count = int(_args.n_neighbors) if train_encs.shape[0] >= int(_args.n_neighbors) else train_encs.shape[0]
        knn = KNN(n_neighbors=nn_count, metric='minkowski')
        knn.fit(train_encs, train_cats)
    else:
        # If classification is CE/hinge, use linear head via model for probabilities; fallback to 1-NN
        knn = KNN(n_neighbors=1, metric='minkowski')
        knn.fit(train_encs, train_cats)

    cache[key] = {'knn': knn, 'unique_labels': unique_labels}
    return knn, unique_labels

# ---- Image Preprocessing ---- #
def preprocess_image(img: Image.Image, size: int, normalize='no'):
    img = img.resize((size, size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC → CHW
    tensor = torch.tensor(img_array)
    
    if str(normalize).lower() in ['yes', 'true', '1']:
        per_img_norm = PerImageNormalize()
        tensor = per_img_norm(tensor)
    
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

# ---- Database Insertion Function ---- #
def insert_score(cursor, conn, filename, args, pred_label, confidence, log_path):
    """Upsert the result row (respecting unique constraint) and upsert usage summary."""
    model_id = _resolve_model_id(cursor, args, log_path)
    query = '''
        INSERT INTO results (
            filename, model_name, task, path, n_neighbors, nsize, fgsm, normalize, n_calibration, classif_loss,
            dloss, dist_fct, prototypes, npos, nneg, pred_label, confidence, log_path, person_id, model_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            pred_label = VALUES(pred_label),
            confidence = VALUES(confidence),
            log_path = VALUES(log_path),
            person_id = VALUES(person_id),
            model_id = VALUES(model_id),
            timestamp = CURRENT_TIMESTAMP
    '''
    values = (
        filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
        args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
        str(args.n_positives), str(args.n_negatives), pred_label, confidence, log_path,
        st.session_state.person_id, model_id
    )
    try:
        cursor.execute(query, values)
    except mysql.connector.errors.IntegrityError:
        # If the unique key blocks insertion, perform an explicit update to refresh timestamp and values
        update_q = '''
            UPDATE results
            SET pred_label=%s, confidence=%s, log_path=%s, person_id=%s, model_id=%s, timestamp=CURRENT_TIMESTAMP
            WHERE filename=%s AND model_name=%s AND task=%s AND path=%s AND n_neighbors=%s AND nsize=%s AND fgsm=%s
                  AND normalize=%s AND n_calibration=%s AND classif_loss=%s AND dloss=%s AND dist_fct=%s AND prototypes=%s
                  AND npos=%s AND nneg=%s
        '''
        update_vals = (
            pred_label, confidence, log_path, st.session_state.person_id, model_id,
            filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
            args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
            str(args.n_positives), str(args.n_negatives)
        )
        cursor.execute(update_q, update_vals)

    summary_query = '''
        INSERT INTO model_usage_summary (
            model_name, task, path, nsize, fgsm, normalize, n_calibration, classif_loss,
            dloss, dist_fct, prototypes, npos, nneg, n_neighbors, num_samples_analyzed
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            num_samples_analyzed = num_samples_analyzed + 1,
            last_used = CURRENT_TIMESTAMP
    '''
    summary_values = (
        args.model_name, args.task, args.path, str(args.new_size), str(args.fgsm),
        args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
        str(args.n_positives), str(args.n_negatives), str(args.n_neighbors), 1
    )
    cursor.execute(summary_query, summary_values)
    conn.commit()

# ---- Helper: Predict label from class prototypes ---- #
def _predict_label_from_prototypes(embedding_tensor: torch.Tensor, class_prototypes: dict, dist_fct_name: str = 'euclidean'):
    """Return the nearest class label to the embedding using provided prototypes.
    Works with either Euclidean or cosine distance, averaging over multiple prototypes per class.
    """
    try:
        emb = embedding_tensor.detach().cpu()
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)

        # For cosine, we maximize similarity; for euclidean, we minimize distance
        best_label = None
        best_score = -float('inf') if dist_fct_name == 'cosine' else float('inf')

        if str(dist_fct_name).lower() == 'cosine':
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        for label, proto in class_prototypes.items():
            if proto is None:
                continue
            proto_t = torch.as_tensor(proto, dtype=emb.dtype)
            if proto_t.ndim == 1:
                proto_t = proto_t.unsqueeze(0)

            if str(dist_fct_name).lower() == 'cosine':
                proto_t = torch.nn.functional.normalize(proto_t, p=2, dim=1)
                sim = torch.mm(emb, proto_t.t())  # (1,k)
                score = torch.max(sim).item()  # higher is better
                if score > best_score:
                    best_score = score
                    best_label = label
            else:
                dists = torch.cdist(emb, proto_t)  # (1,k)
                score = torch.min(dists).item()  # lower is better
                if score < best_score:
                    best_score = score
                    best_label = label

        return best_label
    except Exception:
        return None

def _predict_with_prototype_distance_ratio(embedding_tensor: torch.Tensor, class_prototypes: dict, dist_fct_name: str = 'euclidean'):
    """
    Predict using distance ratio to class prototypes.
    Probability is calculated as inverse distance ratio: prob(class) = (1/dist_to_proto) / sum(1/dist_to_all_protos)
    Returns the class with highest probability.
    """
    try:
        emb = embedding_tensor.detach().cpu()
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)

        # Compute distances to each class prototype
        distances = {}
        for label, proto in class_prototypes.items():
            if proto is None:
                continue
            proto_t = torch.as_tensor(proto, dtype=emb.dtype)
            if proto_t.ndim == 1:
                proto_t = proto_t.unsqueeze(0)

            if str(dist_fct_name).lower() == 'cosine':
                # For cosine distance, use 1 - similarity
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
                proto_norm = torch.nn.functional.normalize(proto_t, p=2, dim=1)
                sim = torch.mm(emb_norm, proto_norm.t())
                dist = (1.0 - torch.max(sim)).item()
            else:
                # For euclidean distance
                dists = torch.cdist(emb, proto_t)
                dist = torch.min(dists).item()
            
            distances[label] = dist

        # Compute inverse distance probabilities
        inv_distances = {label: 1.0 / (dist + 1e-8) for label, dist in distances.items()}
        total_inv = sum(inv_distances.values())
        probas = {label: inv_dist / total_inv for label, inv_dist in inv_distances.items()}
        
        # Return class with highest probability
        best_label = max(probas, key=probas.get)
        return best_label
    except Exception as e:
        print(f"Error in prototype distance ratio prediction: {e}")
        return None

# ---- Cached loader for single model by log_path (used by Ensemble tab) ---- #
@st.cache_resource
def load_model_for_log_path(log_path: str, model_name: str, device: str = 'cpu'):
    try:
        model_pth = os.path.join(log_path, 'model.pth')
        proto_pkl = os.path.join(log_path, 'prototypes.pkl')
        if not os.path.exists(model_pth) or not os.path.exists(proto_pkl):
            raise FileNotFoundError(f"Missing model or prototypes under {log_path}")

        state_dict = torch.load(model_pth, map_location=device)
        n_batches = state_dict['dann.weight'].shape[0]
        n_cats = state_dict['linear.weight'].shape[0]
        n_subcenters = state_dict['subcenters'].shape[1]

        model = Net(device, n_cats=n_cats, n_batches=n_batches,
                    model_name=model_name, is_stn=0, n_subcenters=n_subcenters)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        with open(proto_pkl, 'rb') as f:
            proto_obj = pickle.load(f)
        class_protos = proto_obj.class_prototypes['train']

        return model, class_protos
    except Exception as e:
        raise RuntimeError(f"Failed to load model for ensemble from {log_path}: {e}")

# ---- Reusable Analysis Function ---- #
def run_analysis_on_file(filename, file_bytes, _args, force_reanalyze=False, show_validation_metrics=True, fast_infer=False):
    """Run complete analysis on a file and return results."""
    # Make sure model numbering is available even if user didn't open Tab 1/2 first
    model_number_map, best_models_table = _ensure_model_number_map(cursor)
    params = get_model_params_path(_args)
    complete_log_path = f"logs/best_models/{_args.task}/{_args.model_name}/{params}/queries"
    
    # Check if already analyzed
    exists = None if force_reanalyze else check_ds_exists(cursor, filename, _args)
    
    if exists is not None and not force_reanalyze:
        # Load previous results (indices match SELECT pred_label, confidence, log_path)
        pred_label = exists[0]
        pred_confidence = exists[1]
        log_path = exists[2]
        st.info(
            f"✅ Already analyzed with this exact model & params → Pred: {pred_label} (conf {pred_confidence:.2f})."
        )
        return pred_label, pred_confidence, log_path, exists
    
    # Run fresh analysis
    st.info("🔄 Running analysis...")
    with st.spinner("Loading model and processing image..."):
        # Ensure reproducible split: reset seeds before data loading
        random.seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)
        
        model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
            load_model_and_prototypes(_args)
        
        save_path = os.path.join('data/queries', filename)
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
        
        # Load saved hyperparameters
        params_path = os.path.join(f'logs/best_models/{_args.task}/{_args.model_name}',
                                   f'{_args.path.split("/")[-1]}/nsize{_args.new_size}/{_args.fgsm}/{_args.n_calibration}/'
                                   f'{_args.classif_loss}/{_args.dloss}/{_args.prototypes_to_use}/'
                                   f'{_args.n_positives}/{_args.n_negatives}',
                                   'params.json')
        saved_search = {}
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                    saved_search = payload.get('search_params', {}) or {}
            except Exception:
                saved_search = {}
        
        # Do not override sidebar-selected n_neighbors from saved search params.
        # The user's current selection should take precedence over training-time defaults.
        
        # Prepare inference resources
        params = get_model_params_path(_args)
        train_complete_log_path = f'logs/best_models/{_args.task}/{_args.model_name}/{params}'

        if not fast_infer:
            loaders = get_images_loaders(data=data,
                                         random_recs=_args.random_recs,
                                         weighted_sampler=0,
                                         is_transform=0,
                                         samples_weights=None,
                                         epoch=1,
                                         unique_labels=unique_labels,
                                         triplet_dloss=_args.dloss, bs=_args.bs,
                                         prototypes_to_use=_args.prototypes_to_use,
                                         prototypes=prototypes,
                                         size=_args.new_size,
                                         normalize=_args.normalize)
            # Build or reuse cached KNN once per model selection
            knn, unique_labels_knn = get_or_build_knn(_args, data, unique_labels, unique_batches, prototypes)
            nets = {'cnn': shap_model, 'knn': knn}
        else:
            loaders = None
            unique_labels_knn = unique_labels
            nets = {'cnn': shap_model, 'knn': None}
        
        original, image = get_image(f'data/queries/{filename}', size=image_size, normalize=_args.normalize)
        
        # Optionally fetch cached validation metrics; bulk analysis can skip this entirely
        valid_acc = None
        valid_mcc = None
        valid_metrics_source = None
        if show_validation_metrics:
            # Resolve the correct model_id based on log_path or model parameters
            resolved_model_id = _resolve_model_id(cursor, _args, train_complete_log_path)
            
            # Prefer metrics that match the exact model ID we are using right now
            current_model_row = None
            try:
                if best_models_table is not None and not best_models_table.empty and resolved_model_id is not None:
                    match = best_models_table[best_models_table["Model ID"] == resolved_model_id]
                    if not match.empty:
                        current_model_row = match.iloc[0].to_dict()
            except Exception:
                current_model_row = None

            if current_model_row:
                try:
                    reg_acc = current_model_row.get('Accuracy')
                    reg_mcc = current_model_row.get('MCC')
                    if reg_acc is not None and reg_mcc is not None:
                        valid_acc = float(reg_acc)
                        valid_mcc = float(reg_mcc)
                        valid_metrics_source = 'registry'
                except Exception:
                    pass

            params_path_metrics = os.path.join(train_complete_log_path, "params.json")
            if valid_metrics_source is None and os.path.exists(params_path_metrics):
                try:
                    with open(params_path_metrics, "r", encoding="utf-8") as f:
                        params_payload = json.load(f)
                    best_metrics = params_payload.get("best_metrics", {})
                    valid_metrics = best_metrics.get("valid", {})
                    acc_list = valid_metrics.get("acc") or []
                    mcc_list = valid_metrics.get("mcc") or []
                    if isinstance(acc_list, list) and acc_list:
                        valid_acc = float(acc_list[-1])
                    if isinstance(mcc_list, list) and mcc_list:
                        valid_mcc = float(mcc_list[-1])
                    if valid_acc is not None and valid_mcc is not None:
                        valid_metrics_source = 'params.json'
                except Exception as e:
                    st.warning(f"Could not read saved validation metrics: {e}")
            if valid_acc is None or valid_mcc is None:
                valid_metrics_source = None
        
        # Get prediction (fast prototype-based or KNN)
        with torch.no_grad():
            emb_tensor = nets['cnn'](image.to(device_str))
            if fast_infer:
                class_protos = prototypes.get('class', {}).get('train', {}) if isinstance(prototypes, dict) else {}
                pred_lbl_fast = _predict_with_prototype_distance_ratio(emb_tensor, class_protos, dist_fct_name=str(_args.dist_fct).lower())
                if pred_lbl_fast is None:
                    pred_label = unique_labels[0]
                    pred_confidence = 0.0
                else:
                    pred_label = pred_lbl_fast
                    pred_confidence = 1.0
            else:
                embedding = emb_tensor.detach().cpu().numpy()
                pred_probs = nets['knn'].predict_proba(embedding)
                pred_class = int(np.argmax(pred_probs, axis=1)[0])
                pred_confidence = float(pred_probs[0, pred_class]) if pred_probs.ndim == 2 else float(np.max(pred_probs))
                pred_label = unique_labels[pred_class]
        
        if show_validation_metrics:
            if valid_metrics_source == 'registry':
                st.write(f"**Validation Accuracy (from registry):** {valid_acc:.3f}")
                st.write(f"**Validation MCC (from registry):** {valid_mcc:.3f}")
            elif valid_metrics_source == 'params.json':
                st.write(f"**Validation Accuracy (from params.json):** {valid_acc:.3f}")
                st.write(f"**Validation MCC (from params.json):** {valid_mcc:.3f}")
            else:
                st.info("Validation metrics unavailable (skipped recomputation to speed up inference).")
        st.write(f"**Predicted Label:** {pred_label} ({pred_confidence:.2f} confidence)")
        st.caption(
            f"Model run → id: {_args.model_id}, name: {_args.model_name}, size: {_args.new_size}, fgsm: {_args.fgsm}, dist: {_args.dist_fct}, protos: {_args.prototypes_to_use}, normalize: {_args.normalize}"
        )
        
        # Insert into database
        insert_score(cursor, conn, filename, _args, pred_label, pred_confidence, complete_log_path)
        st.success("✅ Results saved to database.")

        # Track the model number used for this run so the UI can show it immediately
        selection_key = _make_model_selection_key({
            "Model Name": _args.model_name,
            "NSize": _args.new_size,
            "FGSM": _args.fgsm,
            "Prototypes": _args.prototypes_to_use,
            "NPos": _args.n_positives,
            "NNeg": _args.n_negatives,
            "DLoss": _args.dloss,
            "Dist_Fct": _args.dist_fct,
            "Classif_Loss": _args.classif_loss,
            "N_Calibration": _args.n_calibration,
            "Normalize": _args.normalize,
            "N_Neighbors": getattr(_args, "n_neighbors", None),
        })
        model_number = model_number_map.get(selection_key, "?")
        if model_number == "?" and best_models_table is not None and not best_models_table.empty:
            try:
                match = best_models_table[best_models_table["Log Path"] == complete_log_path]
                if not match.empty:
                    model_number = match.iloc[0].get("#", model_number)
            except Exception:
                pass
        st.session_state['last_model_number'] = model_number
    
    return pred_label, pred_confidence, complete_log_path, None

# ---- Streamlit UI ---- #

conn, cursor = create_db()
ensure_results_model_id(conn, cursor)
ensure_best_models_registry_nsize(conn, cursor)

# Ensure model ranks are computed once per session so Quick Model Selection has fresh ranks
try:
    if not st.session_state.get('ranks_initialized', False):
        update_model_ranks()
        st.session_state['ranks_initialized'] = True
except Exception as _rank_exc:
    # Don't block UI if rank update fails; fallback queries handle missing ranks
    pass

if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'person_id' not in st.session_state:
    st.session_state.person_id = None

if st.session_state.user_email is None:
    st.title("🔒 Login Required")
    email = st.text_input("Enter your email to log in or sign up:")
    if st.button("Continue") and email:
        try:
            cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
            row = cursor.fetchone()
            if row:  # earlier I had a row here with just 1 in 
                st.session_state.user_id = row[0]
            else:
                cursor.execute("INSERT INTO users (email) VALUES (%s)", (email,))
                conn.commit()
                st.session_state.user_id = cursor.lastrowid
            st.session_state.user_email = email
            st.rerun()
        except Error as e:
            st.error(f"❌ Database error: {e}")
            st.stop()
    st.stop()

# ---- From here onward, user is guaranteed to be logged in ---- #
st.title("Ear Health Classifier with SHAP 👂")

# ---- Require person selection before using the app ---- #
st.sidebar.header("👤 Select a Family Member")
cursor.execute("SELECT id, name FROM people WHERE user_id=%s", (st.session_state.user_id,))
people = cursor.fetchall()
person_options = [p[1] for p in people]
person_ids = {p[1]: p[0] for p in people}

selected_person = st.sidebar.selectbox("Choose a person", person_options)

if selected_person:
    st.session_state.person_id = person_ids[selected_person]

with st.sidebar.expander("➕ Add a new person"):
    new_name = st.text_input("Person's Name", key="new_person_name")
    if st.button("Add Person") and new_name:
        try:
            cursor.execute("INSERT INTO people (user_id, name) VALUES (%s, %s)", (st.session_state.user_id, new_name))
            conn.commit()
            st.rerun()
        except Error as e:
            st.error(f"❌ Could not add person: {e}")

with st.sidebar.expander("❌ Remove person"):
    if len(person_options) > 0:
        person_to_remove = st.selectbox("Select person to delete", person_options, key="remove_person")
        if st.button("Delete Person"):
            try:
                person_id = person_ids[person_to_remove]
                cursor.execute("DELETE FROM results WHERE person_id = %s", (person_id,))
                cursor.execute("DELETE FROM people WHERE id = %s", (person_id,))
                conn.commit()
                if st.session_state.person_id == person_id:
                    st.session_state.person_id = None
                st.success(f"Deleted {person_to_remove}")
                st.rerun()
            except Error as e:
                st.error(f"❌ Could not delete person: {e}")

with st.sidebar.expander("🚫 Delete your account"):
    if st.button("Delete My Account"):
        try:
            uid = st.session_state.user_id
            cursor.execute("DELETE FROM results WHERE person_id IN (SELECT id FROM people WHERE user_id = %s)", (uid,))
            cursor.execute("DELETE FROM people WHERE user_id = %s", (uid,))
            cursor.execute("DELETE FROM users WHERE id = %s", (uid,))
            conn.commit()
            st.success("Your account and all associated data has been deleted.")
            st.session_state.clear()
            st.rerun()
        except Error as e:
            st.error(f"❌ Could not delete account: {e}")

# ---- Remove a result ---- #
with st.sidebar.expander("🗑️ Remove a result"):
    cursor.execute("""
        SELECT id, filename FROM results
        WHERE person_id = %s
        ORDER BY timestamp DESC
    """, (st.session_state.person_id,))
    results = cursor.fetchall()
    if results:
        result_names = [r[1] for r in results]
        result_map = {r[1]: r[0] for r in results}
        result_to_delete = st.selectbox("Select result to delete", result_names, key="delete_result")
        if st.button("Delete Result"):
            try:
                cursor.execute("DELETE FROM results WHERE id = %s", (result_map[result_to_delete],))
                conn.commit()
                st.success(f"Deleted result: {result_to_delete}")
                st.rerun()
            except Error as e:
                st.error(f"❌ Could not delete result: {e}")

    with st.sidebar:
        if st.button("🚪 Log Out"):
            st.session_state.user_email = None
            st.session_state.user_id = None
            st.session_state.person_id = None
            st.success("Logged out successfully.")
            st.rerun()

if st.session_state.person_id is None:
    st.warning("👤 Please select a family member to proceed.")
    st.stop()

# ---- Load sidebar parameters BEFORE tabs (always visible) ---- #
args = get_args()

# ---- Create Tabs ---- #
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Model Selection", "📂 Past Results", "🔬 New Analysis", "🤝 Ensemble", "🖼️ Grad-CAM Gallery"])

# ========================= TAB 1: Model Selection ========================= #
with tab1:
    st.header("🏆 Best Models Leaderboard")
    try:
        # Try to use model_rank if available, otherwise fall back to dynamic numbering
        try:
            cursor.execute("""
                SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path, model_rank
                FROM best_models_registry
                WHERE model_rank IS NOT NULL
                ORDER BY model_rank ASC
            """)
            use_db_rank = True
        except:
            # Fallback if model_rank column doesn't exist yet
            cursor.execute("""
                SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path
                FROM best_models_registry
                ORDER BY mcc DESC
            """)
            use_db_rank = False
            
        models = cursor.fetchall()
        import pandas as pd
        
        if use_db_rank:
            columns = [
                "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", "#"
            ]
        else:
            columns = [
                "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path"
            ]
        models_df = pd.DataFrame(models, columns=columns)

        # Keep only the best (highest MCC) for each unique configuration.
        # NOTE: pandas considers NaN values as distinct for drop_duplicates; build a stable key.
        group_cols = [
            "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
            "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize", "N_Neighbors"
        ]

        # Build a dedupe key that treats missing values consistently
        _dedupe_frame = models_df[group_cols].copy()
        _dedupe_frame = _dedupe_frame.fillna("").astype(str)
        # Build a robust single-column key even with mixed types
        models_df["_dedupe_key"] = _dedupe_frame.apply(lambda r: "|".join(r.values.tolist()), axis=1)

        # Sort by MCC descending, then drop duplicates keeping the first (best MCC)
        models_df = models_df.sort_values("MCC", ascending=False)
        models_df = models_df.drop_duplicates(subset=["_dedupe_key"], keep="first")
        models_df = models_df.drop(columns=["_dedupe_key"])

        # Require a non-empty log_path (used for loading), but do NOT dedupe by log_path.
        # Some DB rows can legitimately share the same log_path while differing in params
        # (e.g., dist_fct); selectors use the full parameter-combination key instead.
        models_df = models_df.dropna(subset=["Log Path"])
        models_df = models_df[models_df["Log Path"].astype(str) != ""]
        models_df = models_df.reset_index(drop=True)

        # Attach cached calibration metrics (ECE/Brier) per model entry.
        models_df["ECE"] = np.nan
        models_df["Brier"] = np.nan
        for idx, row in models_df.iterrows():
            log_path = row.get("Log Path")
            metrics = get_calibration_metrics(log_path)
            if metrics and not metrics.get("error"):
                models_df.at[idx, "ECE"] = metrics.get("ece")
                models_df.at[idx, "Brier"] = metrics.get("brier")

        # Reorder columns to surface calibration next to accuracy/MCC for the table display
        metric_cols = ["Accuracy", "MCC", "ECE", "Brier"]
        base_order = [col for col in models_df.columns if col not in metric_cols and col != "Log Path" and col != "#"]
        ordered_cols = ["#"] if "#" in models_df.columns else []
        ordered_cols += [c for c in base_order if c not in metric_cols]
        ordered_cols += [c for c in metric_cols if c in models_df.columns]
        # Keep log path hidden from the display table but available in the stored df
        ordered_cols += [c for c in models_df.columns if c not in ordered_cols]
        models_df = models_df[ordered_cols]
        
        # model_rank is already in the dataframe from the database query
        # Move it to the first column for display
        if "#" in models_df.columns:
            cols = ["#"] + [col for col in models_df.columns if col != "#"]
            models_df = models_df[cols]
        
        # Store model number mapping in session state for use elsewhere
        model_number_map = {}
        for idx, row in models_df.iterrows():
            rd = row.to_dict()
            selection_key = _make_model_selection_key(rd)
            model_number_map[selection_key] = rd.get("#", "?")
        st.session_state['model_number_map'] = model_number_map

        st.write("**Top Models (best per parameter combination):**")
        st.markdown("**Table:** Top Models")
        display_columns = [col for col in models_df.columns if col != "Log Path"]
        display_df = models_df[display_columns].copy()
        # Keep full table (including Log Path and #) in session for cross-tab mapping
        st.session_state['best_models_table'] = models_df.copy()
        st.dataframe(display_df, use_container_width=True)

        # Calibration vs Performance Plots
        st.markdown("---")
        st.subheader("📈 Calibration vs Performance")
        
        # Filter to rows with valid metrics
        plot_df = models_df.dropna(subset=["MCC", "ECE", "Brier"]).copy()
        
        if len(plot_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**MCC vs ECE**")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.scatter(plot_df["MCC"], plot_df["ECE"], alpha=0.6, s=50)
                ax1.set_xlabel("MCC (higher is better)", fontsize=10)
                ax1.set_ylabel("ECE (lower is better)", fontsize=10)
                ax1.set_title("Expected Calibration Error vs MCC", fontsize=11)
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                plt.close(fig1)
            
            with col2:
                st.markdown("**MCC vs Brier Score**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.scatter(plot_df["MCC"], plot_df["Brier"], alpha=0.6, s=50, color='orange')
                ax2.set_xlabel("MCC (higher is better)", fontsize=10)
                ax2.set_ylabel("Brier Score (lower is better)", fontsize=10)
                ax2.set_title("Brier Score vs MCC", fontsize=11)
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                plt.close(fig2)
        else:
            st.info("Calibration metrics not available for plotting. Run models to compute ECE and Brier scores.")
        
        st.markdown("---")

        # Dropdown menu to select model
        if len(models_df) > 0:
            key_to_row = {}
            key_to_label = {}
            for _, r in models_df.iterrows():
                rd = r.to_dict()
                selection_key = _make_model_selection_key(rd)
                if selection_key in key_to_row:
                    continue
                key_to_row[selection_key] = r
                model_num = rd.get("#", "?")
                try:
                    key_to_label[selection_key] = (
                        f"#{model_num} - {rd.get('Model Name')} (MCC={float(rd.get('MCC')):.3f}, dist_fct={rd.get('Dist_Fct')}, normalize={rd.get('Normalize')})"
                    )
                except Exception:
                    key_to_label[selection_key] = (
                        f"#{model_num} - {rd.get('Model Name')} (MCC={rd.get('MCC')}, dist_fct={rd.get('Dist_Fct')}, normalize={rd.get('Normalize')})"
                    )

            tab1_options = _unique_preserve_order(list(key_to_row.keys()))

            # Sync this widget to the canonical selection (log_path)
            canonical_key = st.session_state.get('selected_model_selection_key')
            if canonical_key and canonical_key in tab1_options:
                last = st.session_state.get('tab1_best_model_last_sync')
                ver = st.session_state.get('selected_model_version')
                if ver is not None and ver != last:
                    st.session_state['tab1_best_model_key'] = canonical_key
                    st.session_state['tab1_best_model_last_sync'] = ver

            selected_key = st.selectbox(
                "Select a model to use:",
                options=tab1_options,
                format_func=lambda k: key_to_label.get(k, str(k)),
                index=0,
                key="tab1_best_model_key",
            )

            if selected_key:
                row = key_to_row.get(selected_key)
                if row is not None:
                    row_dict = row.to_dict()
                    log_path = row_dict.get("Log Path")
                    if log_path:
                        st.subheader(f"📈 Calibration Curve (Model #{row_dict.get('#', '?')})")
                        metrics = get_calibration_metrics(log_path)
                        if metrics is None:
                            st.info("No calibration metrics available for this model.")
                        elif metrics.get("error"):
                            st.warning(f"Could not load calibration metrics: {metrics['error']}")
                        else:
                            ece_val = metrics.get("ece")
                            brier_val = metrics.get("brier")
                            prob_true = np.array(metrics.get("prob_true", []), dtype=float)
                            prob_pred = np.array(metrics.get("prob_pred", []), dtype=float)

                            c1, c2 = st.columns(2)
                            with c1:
                                st.metric("Expected Calibration Error (ECE)", f"{ece_val:.4f}")
                            with c2:
                                st.metric("Brier Score", f"{brier_val:.4f}")

                            if ece_val < 0.05:
                                st.success("✅ The model is well-calibrated.")
                            elif ece_val < 0.15:
                                st.warning("⚠️ The model shows moderate calibration error.")
                            else:
                                st.error("❌ The model is poorly calibrated.")

                            if len(prob_true) >= 2 and len(prob_pred) >= 2:
                                # Sort by predicted prob for a smooth curve
                                df_curve = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true}).sort_values("prob_pred")

                                # Matplotlib plot
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.plot(df_curve["prob_pred"], df_curve["prob_true"], marker='o', linewidth=1.5,
                                        label=f"Model #{row_dict.get('#', '?')} (ECE: {ece_val:.4f}, Brier: {brier_val:.4f})")
                                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly calibrated")
                                ax.set_xlabel("Mean predicted probability")
                                ax.set_ylabel("Fraction of positives")
                                ax.set_title("Calibration Curve (Validation Set)")
                                ax.legend()
                                ax.grid(alpha=0.3)
                                st.pyplot(fig)

                                # Alt: simple line chart for quick glance
                                st.line_chart(df_curve.set_index('prob_pred'))
                            else:
                                st.info("Not enough bins to render a calibration curve (need at least 2 points).")

            if st.button("✅ Use Selected Model", key="use_selected_model_btn"):
                row = key_to_row.get(selected_key)
                if row is None:
                    st.error("Could not resolve selected model.")
                else:
                    row_dict = row.to_dict()
                    row_dict.update(extract_params_from_log_path(row_dict.get("Log Path")))
                    row_dict["model_id"] = row_dict.get("Model ID")
                    # Normalize key names to match get_args() expectations
                    if "N_Neighbors" in row_dict:
                        row_dict["n_neighbors"] = row_dict["N_Neighbors"]
                    if "NSize" in row_dict:
                        row_dict["new_size"] = row_dict["NSize"]
                    if "Dist_Fct" in row_dict:
                        row_dict["dist_fct"] = row_dict["Dist_Fct"]
                    if "Classif_Loss" in row_dict:
                        row_dict["classif_loss"] = row_dict["Classif_Loss"]

                    st.session_state.selected_model_params = row_dict
                    st.session_state.selected_params_version = st.session_state.get('selected_params_version', 0) + 1
                    st.session_state.selected_model_log_path = row_dict.get('Log Path')
                    st.session_state.selected_model_selection_key = selected_key
                    st.session_state.selected_model_version = st.session_state.get('selected_model_version', 0) + 1

                    st.success(f"✅ Selected model #{row_dict.get('#', '?')}: {row_dict.get('Model Name')}")
                    st.info("Switch to '🔬 New Analysis' tab to upload an image and run analysis with this model.")
                    st.rerun()
                
            if st.session_state.get('selected_model_params'):
                st.info("ℹ️ Model parameters loaded. Check sidebar for current settings. Go to '🔬 New Analysis' tab to run analysis.")
        else:
            st.warning("No models found in leaderboard.")
    except Exception as e:
        st.error(f"Could not load best models leaderboard: {e}")
    
    st.divider()

    # ---- KNN Optimization (Tab 1) ---- #
    st.subheader("🔧 KNN Optimization")
    st.caption("Optimize k (1–20) on the validation split for the currently selected sidebar model.")

    # Clear stale optimization results if the selected model changed
    current_model_key = (
        st.session_state.get('selected_model_selection_key')
        or st.session_state.get('tab1_best_model_key')
        or st.session_state.get('sidebar_best_model_key')
    )
    if st.session_state.get('k_opt_model_key') not in (None, current_model_key):
        st.session_state.pop('optimized_k_value', None)
        st.session_state.pop('k_opt_best_mcc', None)
        st.session_state.pop('k_opt_curve', None)
        st.session_state['k_opt_model_key'] = current_model_key

    def _optimize_k_for_args(_args, min_k: int = 1, max_k: int = 20):
        # Load model + datasets
        model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)

        # Prepare a minimal TrainAE wrapper to encode sets (reuse utilities used elsewhere)
        train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                        log_inputs=False, log_plots=False, log_tb=False, log_neptune=False,
                        log_mlflow=False, groupkfold=_args.groupkfold)
        train.n_batches = len(unique_batches)
        train.n_cats = len(unique_labels)
        train.unique_batches = unique_batches
        train.unique_labels = unique_labels
        train.epoch = 1
        train.model = model
        train.params = {'n_neighbors': int(_args.n_neighbors)}
        train.set_arcloss()

        lists, traces = get_empty_traces()
        loaders = get_images_loaders(
            data=data,
            random_recs=_args.random_recs,
            weighted_sampler=0,
            is_transform=0,
            samples_weights=None,
            epoch=1,
            unique_labels=unique_labels,
            triplet_dloss=_args.dloss, bs=_args.bs,
            prototypes_to_use=_args.prototypes_to_use,
            prototypes=prototypes,
            size=_args.new_size,
            normalize=_args.normalize,
        )

        # Encode train and valid sets
        with torch.no_grad():
            _, lists, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
            _, lists, _ = train.loop('valid', None, 0, loaders['valid'], lists, traces)

        train_encs = np.concatenate(lists['train']['encoded_values'])
        train_cats = np.concatenate(lists['train']['cats'])
        valid_encs = np.concatenate(lists['valid']['encoded_values'])
        valid_cats = np.concatenate(lists['valid']['cats'])

        best_k, best_mcc = 1, -1.0
        max_k = min(max_k + 1, train_encs.shape[0] + 1)
        mcc_per_k = []
        for k in range(min_k, max_k):
            knn_temp = KNN(n_neighbors=k, metric='minkowski')
            knn_temp.fit(train_encs, train_cats)
            preds = knn_temp.predict(valid_encs)
            mcc_val = MCC(valid_cats, preds)
            mcc_per_k.append({'k': k, 'mcc': float(mcc_val)})
            if mcc_val > best_mcc:
                best_k, best_mcc = k, mcc_val
        # Prototype proximity baseline (if available)
        proto_mcc = None
        try:
            proto_train = prototypes.get('class', {}).get('train', {}) if isinstance(prototypes, dict) else {}
            if proto_train and len(proto_train.keys()) > 0:
                proto_vecs = []
                proto_labels = []
                for lbl, vec in proto_train.items():
                    arr = np.asarray(vec)
                    if arr.ndim > 1:
                        arr = arr[0]
                    proto_vecs.append(arr)
                    proto_labels.append(lbl)
                proto_vecs = np.stack(proto_vecs)
                # compute nearest prototype per validation sample
                dists = np.linalg.norm(valid_encs[:, None, :] - proto_vecs[None, :, :], axis=2)
                proto_preds = np.take(proto_labels, np.argmin(dists, axis=1))
                proto_mcc = float(MCC(valid_cats, proto_preds))
        except Exception:
            proto_mcc = None

        return best_k, float(best_mcc), mcc_per_k, proto_mcc

    col_k_range, col_k_btn = st.columns([1, 1])
    with col_k_range:
        min_k = st.number_input("Min k", min_value=1, max_value=50, value=1, step=1, key="tab1_min_k")
        max_k = st.number_input("Max k", min_value=2, max_value=200, value=20, step=1, key="tab1_max_k")
        if max_k <= min_k:
            st.warning("Max k must be greater than min k")
    with col_k_btn:
        if st.button("Optimize k on validation", key="tab1_optimize_k_button"):
            try:
                best_k, best_mcc, mcc_curve, proto_mcc = _optimize_k_for_args(args, int(min_k), int(max_k))
                st.success(f"✅ Best k = {best_k} (Validation MCC: {best_mcc:.3f})")
                # Store result in a non-widget key and trigger rerun to apply it
                st.session_state['optimized_k_value'] = int(best_k)
                st.session_state['k_opt_best_mcc'] = float(best_mcc)
                st.session_state['k_opt_curve'] = mcc_curve
                st.session_state['k_opt_proto_mcc'] = proto_mcc
                st.session_state['k_opt_model_key'] = current_model_key
                st.rerun()
            except Exception as e:
                st.error(f"K optimization failed: {e}")
    st.caption("Uses current sidebar model settings (dataset, size, normalize, etc.)")

    if 'optimized_k_value' in st.session_state and 'k_opt_best_mcc' in st.session_state:
        msg = f"Last optimized k = {st.session_state['optimized_k_value']} (Validation MCC: {st.session_state['k_opt_best_mcc']:.3f})"
        proto_mcc = st.session_state.get('k_opt_proto_mcc')
        if proto_mcc is not None:
            msg += f" | Prototype-only MCC: {proto_mcc:.3f}"
        st.info(msg)

    if st.session_state.get('k_opt_curve'):
        try:
            curve_df = pd.DataFrame(st.session_state['k_opt_curve'])
            curve_df = curve_df.sort_values('k')
            if 'optimized_k_value' in st.session_state:
                best_row = curve_df[curve_df['k'] == st.session_state['optimized_k_value']]
                if not best_row.empty:
                    st.markdown(f"⭐ Best k = {int(best_row.iloc[0]['k'])} (MCC {best_row.iloc[0]['mcc']:.3f})")
            st.line_chart(curve_df.set_index('k'))
            st.dataframe(curve_df.assign(best=curve_df['k'] == st.session_state.get('optimized_k_value', -1)), use_container_width=True)
        except Exception:
            pass


    # Define PCA computation function
    def _compute_pca_for_args(_args):
        # Load model + datasets
        model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)

        # Minimal TrainAE to encode sets
        train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                        log_inputs=False, log_plots=False, log_tb=False, log_neptune=False,
                        log_mlflow=False, groupkfold=_args.groupkfold)
        train.n_batches = len(unique_batches)
        train.n_cats = len(unique_labels)
        train.unique_batches = unique_batches
        train.unique_labels = unique_labels
        train.epoch = 1
        train.model = model
        train.params = {'n_neighbors': int(_args.n_neighbors)}
        train.set_arcloss()

        lists, traces = get_empty_traces()
        loaders = get_images_loaders(
            data=data,
            random_recs=_args.random_recs,
            weighted_sampler=0,
            is_transform=0,
            samples_weights=None,
            epoch=1,
            unique_labels=unique_labels,
            triplet_dloss=_args.dloss, bs=_args.bs,
            prototypes_to_use=_args.prototypes_to_use,
            prototypes=prototypes,
            size=_args.new_size,
            normalize=_args.normalize,
        )

        with torch.no_grad():
            _, lists, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
            if 'valid' in loaders:
                _, lists, _ = train.loop('valid', None, 0, loaders['valid'], lists, traces)
            if 'test' in loaders:
                try:
                    _, lists, _ = train.loop('test', None, 0, loaders['test'], lists, traces)
                except Exception:
                    pass

        # Collect encodings and metadata
        encs = []
        cats = []
        batches = []
        for grp in ['train', 'valid', 'test']:
            if lists[grp]['encoded_values']:
                encs.append(np.concatenate(lists[grp]['encoded_values']))
                cats.append(np.concatenate(lists[grp]['cats']))
                # domains/batches stored in lists[group]['domains'] as array of batch names
                try:
                    batches.append(np.concatenate(lists[grp]['domains']))
                except Exception:
                    batches.append(np.array([grp] * len(lists[grp]['encoded_values'][0])))

        if not encs:
            raise RuntimeError("No embeddings available to plot.")

        all_encs = np.concatenate(encs)
        all_cats = np.concatenate(cats)
        all_batches = np.concatenate(batches) if batches else np.zeros(len(all_cats))

        # Prototypes array (optional)
        proto_arr = None
        if isinstance(prototypes, dict):
            try:
                class_train = prototypes.get('class', {}).get('train', {})
                proto_list = []
                for lbl, vec in class_train.items():
                    arr = np.asarray(vec)
                    if arr.ndim > 1:
                        arr = arr[0]
                    proto_list.append(arr)
                if proto_list:
                    proto_arr = np.stack(proto_list)
            except Exception:
                proto_arr = None

        # PCA fit
        n_comp = min(3, all_encs.shape[1])
        pca = PCA(n_components=n_comp)
        encs_pca = pca.fit_transform(all_encs)
        explained = pca.explained_variance_ratio_ * 100.0
        proto_pca = pca.transform(proto_arr) if proto_arr is not None else None

        # Plot and store in session_state
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(encs_pca[:, 0], encs_pca[:, 1], c=all_cats, cmap='tab20', alpha=0.7, s=20)
        if proto_pca is not None:
            ax.scatter(proto_pca[:, 0], proto_pca[:, 1], marker='x', color='red', s=90, label='Prototypes')
        ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)')
        if n_comp > 1:
            ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)')
        ax.set_title('PCA of encodings (train/valid/test)')
        ax.legend(loc='best')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class id')
        
        # Store figure bytes in session state for persistence
        import io
        fig_bytes = io.BytesIO()
        fig.savefig(fig_bytes, format='png', dpi=100, bbox_inches='tight')
        fig_bytes.seek(0)
        st.session_state['tab1_pca_fig_bytes'] = fig_bytes.getvalue()
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ---- PCA (Tab 1) ---- #
    with st.expander("🧭 PCA with Prototypes", expanded=False):
        st.caption("Compute PCA of encoded representations for the current sidebar model and overlay class prototypes.")

        # Check if we have PCA cached for the current model
        current_model_key = f"{args.task}_{os.path.basename(args.path) if args.path else 'no_path'}_{args.new_size}_{args.n_neighbors}_{args.dist_fct}"
        has_pca_cache = (st.session_state.get('tab1_pca_model_key') == current_model_key and 
                         st.session_state.get('tab1_pca_fig_bytes'))
        
        col_pca_btn, col_pca_status = st.columns([3, 2])
        with col_pca_btn:
            if st.button("Compute PCA (encodings + prototypes)", key="tab1_compute_pca"):
                # Clear cache when button clicked to force recomputation
                st.session_state['tab1_pca_fig_bytes'] = None
                st.session_state['tab1_pca_model_key'] = None
                with st.spinner("Computing PCA on encodings..."):
                    try:
                        _compute_pca_for_args(args)
                        st.session_state['tab1_pca_model_key'] = current_model_key
                    except Exception as e:
                        st.error(f"PCA failed: {e}")
        
        with col_pca_status:
            if has_pca_cache:
                st.success("✅ Cached")
        
        # Display cached PCA if available
        if has_pca_cache:
            st.info("Displaying cached PCA from previous run")
            st.image(st.session_state['tab1_pca_fig_bytes'], use_column_width=True)

    # ---- Comprehensive EDA Suite ---- #
    st.divider()
    st.subheader("📊 Comprehensive EDA Suite")
    st.caption("Advanced analysis: Raw PCA, t-SNE, UMAP, distributions, embeddings stats")

    def _run_comprehensive_eda(_args):
        """Zealous ML engineer's comprehensive EDA analysis."""
        with st.spinner("Loading model and data..."):
            model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)
            train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                            log_inputs=False, log_plots=False, log_tb=False, log_neptune=False,
                            log_mlflow=False, groupkfold=_args.groupkfold)
            train.n_batches = len(unique_batches)
            train.n_cats = len(unique_labels)
            train.unique_batches = unique_batches
            train.unique_labels = unique_labels
            train.epoch = 1
            train.model = model
            train.params = {'n_neighbors': int(_args.n_neighbors)}
            train.set_arcloss()

        lists, traces = get_empty_traces()
        loaders = get_images_loaders(
            data=data, random_recs=_args.random_recs, weighted_sampler=0, is_transform=0,
            samples_weights=None, epoch=1, unique_labels=unique_labels,
            triplet_dloss=_args.dloss, bs=_args.bs, prototypes_to_use=_args.prototypes_to_use,
            prototypes=prototypes, size=_args.new_size, normalize=_args.normalize,
        )

        # Encode all sets
        with st.spinner("Encoding all samples..."):
            with torch.no_grad():
                for grp in ['train', 'valid', 'test']:
                    if grp in loaders:
                        try:
                            _, lists, _ = train.loop(grp, None, 0, loaders[grp], lists, traces)
                        except:
                            pass

        # Collect data
        encs, cats, batches, raw_imgs = [], [], [], []
        grp_labels = []
        
        # Load raw images from data/queries if available
        import glob as glob_module
        query_imgs = sorted(glob_module.glob('data/queries/*'))
        if query_imgs:
            with st.spinner("Loading raw query images..."):
                for img_path in query_imgs[:500]:  # Limit to 500
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_arr = np.array(img) / 255.0
                        raw_imgs.append(img_arr)
                    except:
                        pass
        
        for grp in ['train', 'valid', 'test']:
            if lists[grp].get('encoded_values'):
                encs.append(np.concatenate(lists[grp]['encoded_values']))
                cats.append(np.concatenate(lists[grp]['cats']))
                grp_labels.append(np.array([grp] * len(lists[grp]['encoded_values'])))

        if not encs:
            raise RuntimeError("No encodings available")

        all_encs = np.concatenate(encs)
        all_cats = np.concatenate(cats)
        all_grps = np.concatenate(grp_labels) if grp_labels else np.zeros(len(all_cats), dtype=object)
        all_raw = np.concatenate(raw_imgs) if raw_imgs else None

        # Create tabs for different analyses
        eda_tabs = st.tabs(["📈 Distributions", "🔍 Raw PCA", "🌐 t-SNE", "🎨 UMAP", "📊 Statistics", "🔗 Correlations", "🎯 Prototypes"])

        # TAB 1: Distributions
        with eda_tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Class Distribution**")
                class_counts = pd.Series(all_cats).value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(6, 4))
                class_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title('Samples per Class')
                ax.set_ylabel('Count')
                ax.set_xlabel('Class ID')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            with col2:
                st.write("**Group Distribution**")
                grp_counts = pd.Series(all_grps).value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                grp_counts.plot(kind='bar', ax=ax, color='coral')
                ax.set_title('Samples per Split')
                ax.set_ylabel('Count')
                st.pyplot(fig, use_container_width=True)

        # TAB 2: Raw PCA (on pixel data)
        with eda_tabs[1]:
            if all_raw is not None and len(all_raw) > 0:
                with st.spinner("Computing PCA on raw pixels..."):
                    max_samples = min(1000, len(all_raw))
                    raw_flat = all_raw[:max_samples].reshape(max_samples, -1)  # Limit to available or 1000
                    raw_cats_subset = all_cats[:max_samples]
                    pca_raw = PCA(n_components=2)
                    raw_pca_2d = pca_raw.fit_transform(raw_flat)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(raw_pca_2d[:, 0], raw_pca_2d[:, 1], c=raw_cats_subset, cmap='tab20', alpha=0.7, s=20)
                    ax.set_xlabel(f'PC1 ({pca_raw.explained_variance_ratio_[0]*100:.1f}%)')
                    ax.set_ylabel(f'PC2 ({pca_raw.explained_variance_ratio_[1]*100:.1f}%)')
                    ax.set_title(f'Raw Pixel PCA (first {max_samples} samples)')
                    plt.colorbar(scatter, ax=ax, label='Class')
                    st.pyplot(fig, use_container_width=True)
            else:
                st.info("Raw pixel data not available")

        # TAB 3: t-SNE
        with eda_tabs[2]:
            with st.spinner("Computing t-SNE (this may take a moment)..."):
                try:
                    sample_size = min(500, len(all_encs))
                    sample_idx = np.random.choice(len(all_encs), sample_size, replace=False)
                    encs_sample = all_encs[sample_idx]
                    cats_sample = all_cats[sample_idx]
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                    encs_tsne = tsne.fit_transform(encs_sample)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(encs_tsne[:, 0], encs_tsne[:, 1], c=cats_sample, cmap='tab20', alpha=0.7, s=20)
                    ax.set_title('t-SNE of Embeddings (sampled)')
                    plt.colorbar(scatter, ax=ax, label='Class')
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"t-SNE failed: {e}")

        # TAB 4: UMAP
        with eda_tabs[3]:
            if UMAP is not None:
                with st.spinner("Computing UMAP..."):
                    try:
                        sample_size = min(500, len(all_encs))
                        sample_idx = np.random.choice(len(all_encs), sample_size, replace=False)
                        encs_sample = all_encs[sample_idx]
                        cats_sample = all_cats[sample_idx]
                        umap = UMAP(n_components=2, random_state=42)
                        encs_umap = umap.fit_transform(encs_sample)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        scatter = ax.scatter(encs_umap[:, 0], encs_umap[:, 1], c=cats_sample, cmap='tab20', alpha=0.7, s=20)
                        ax.set_title('UMAP of Embeddings (sampled)')
                        plt.colorbar(scatter, ax=ax, label='Class')
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"UMAP failed: {e}")
            else:
                st.info("UMAP not installed. Install with: pip install umap-learn")

        # TAB 5: Embedding Statistics
        with eda_tabs[4]:
            stats_data = []
            for cls_id in sorted(np.unique(all_cats)):
                mask = all_cats == cls_id
                enc_subset = all_encs[mask]
                stats_data.append({
                    'Class': cls_id,
                    'N Samples': len(enc_subset),
                    'Mean Norm': np.linalg.norm(enc_subset.mean(axis=0)),
                    'Std Norm': np.linalg.norm(enc_subset.std(axis=0)),
                    'Min': enc_subset.min(),
                    'Max': enc_subset.max(),
                    'Median': np.median(enc_subset),
                })
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

            # Embedding norm distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            for cls_id in sorted(np.unique(all_cats)):
                mask = all_cats == cls_id
                norms = np.linalg.norm(all_encs[mask], axis=1)
                ax.hist(norms, alpha=0.6, label=f'Class {cls_id}', bins=30)
            ax.set_xlabel('Embedding L2 Norm')
            ax.set_ylabel('Frequency')
            ax.set_title('Embedding Norm Distribution by Class')
            ax.legend()
            st.pyplot(fig, use_container_width=True)

        # TAB 6: Feature Correlations
        with eda_tabs[5]:
            st.write("**Embedding Dimension Correlations (first 20 dims)**")
            n_dims = min(20, all_encs.shape[1])
            corr_matrix = np.corrcoef(all_encs[:, :n_dims].T)
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Embedding Dimension Correlations')
            st.pyplot(fig, use_container_width=True)

        # TAB 7: Prototype Analysis
        with eda_tabs[6]:
            proto_info = []
            if isinstance(prototypes, dict):
                class_train = prototypes.get('class', {}).get('train', {})
                for cls_id, proto in class_train.items():
                    proto_arr = np.asarray(proto)
                    if proto_arr.ndim > 1:
                        proto_arr = proto_arr[0]
                    mask = all_cats == cls_id
                    enc_cls = all_encs[mask]
                    if len(enc_cls) == 0:  # Skip empty classes
                        continue
                    distances = np.linalg.norm(enc_cls - proto_arr, axis=1)
                    if len(distances) > 0:
                        proto_info.append({
                            'Class': cls_id,
                            'Proto Norm': np.linalg.norm(proto_arr),
                            'Mean Sample Dist': distances.mean(),
                            'Std Sample Dist': distances.std(),
                            'Min Dist': distances.min(),
                            'Max Dist': distances.max(),
                        })
            if proto_info:
                proto_df = pd.DataFrame(proto_info)
                st.dataframe(proto_df, use_container_width=True)
                st.write("**Sample-to-Prototype Distances**")
                fig, ax = plt.subplots(figsize=(8, 5))
                for i, info in enumerate(proto_info):
                    cls_id = info['Class']
                    mask = all_cats == cls_id
                    enc_cls = all_encs[mask]
                    if len(enc_cls) == 0:
                        continue
                    proto_arr = np.asarray(class_train[cls_id])
                    if proto_arr.ndim > 1:
                        proto_arr = proto_arr[0]
                    distances = np.linalg.norm(enc_cls - proto_arr, axis=1)
                    if len(distances) > 0:
                        ax.hist(distances, alpha=0.6, label=f'Class {cls_id}', bins=30)
                ax.set_xlabel('Distance to Class Prototype')
                ax.set_ylabel('Frequency')
                ax.set_title('Sample Distances to Class Prototypes')
                ax.legend()
                st.pyplot(fig, use_container_width=True)

    # Reset per-run render flag
    st.session_state['tab1_eda_rendered_this_run'] = False

    # EDA Suite button
    if st.button("🧪 Run Full EDA Suite", key="tab1_run_eda"):
        try:
            _run_comprehensive_eda(args)
            st.session_state['tab1_eda_model_key'] = current_model_key
            st.session_state['tab1_eda_keep'] = True
            st.session_state['tab1_eda_rendered_this_run'] = True
        except Exception as e:
            st.error(f"EDA failed: {e}")
            import traceback
            st.error(traceback.format_exc())
    
    # Auto-display EDA if previously run for this model and pinned
    if st.session_state.get('tab1_eda_model_key') == current_model_key:
        keep = st.checkbox(
            "📌 Pin EDA results",
            value=st.session_state.get('tab1_eda_keep', True),
            key="tab1_keep_eda_checkbox",
            help="Keep EDA visible across other interactions."
        )
        st.session_state['tab1_eda_keep'] = keep
        if keep:
            st.caption("💾 EDA will re-display after other actions")
            if not st.session_state.get('tab1_eda_rendered_this_run', False):
                try:
                    _run_comprehensive_eda(args)
                    st.session_state['tab1_eda_rendered_this_run'] = True
                except Exception as e:
                    st.error(f"EDA failed: {e}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # ---- Model Usage Summary (moved from Tab 2) ---- #
    st.subheader("📊 Models Used for Analysis")
    try:
        cursor.execute("""
            SELECT model_name, task, nsize, fgsm, normalize, n_calibration, classif_loss, dloss, prototypes, 
                   npos, nneg, n_neighbors, num_samples_analyzed, last_used
            FROM model_usage_summary
            ORDER BY last_used DESC
        """)
        usage_rows = cursor.fetchall()
        if usage_rows:
            usage_columns = [
                "Model", "Task", "Size", "FGSM", "Normalize", "N_Cal", "Loss", "DLoss", "Prototypes", 
                "NPos", "NNeg", "N_Neighbors", "Samples", "Last Used"
            ]
            usage_df = pd.DataFrame(usage_rows, columns=usage_columns)
            st.markdown("**Table:** Model Usage Summary")
            st.dataframe(usage_df, use_container_width=True)
        else:
            st.info("No models have been used for analysis yet.")
    except Exception as e:
        st.warning(f"Could not load model usage summary: {e}")

# ========================= TAB 2: Past Results ========================= #
with tab2:
    st.header("📂 Past Analysis Results")
    
    # Query ALL results for this person (don't deduplicate - we want to see all models tried)
    query = '''
         SELECT id, model_id, filename, model_name, task, confidence, timestamp, pred_label, log_path,
             nsize, fgsm, normalize, n_calibration, classif_loss, dloss, dist_fct, prototypes, npos, nneg, n_neighbors
        FROM results
        WHERE person_id=%s
        ORDER BY timestamp DESC
    '''
    cursor.execute(query, (st.session_state.person_id,))
    rows = cursor.fetchall()

    import pandas as pd
    df = pd.DataFrame(rows, columns=[
        "Id", "Model_ID", "Filename", "Model Name", "Task", "Confidence", "Timestamp", "Pred_Label", "Log Path",
        "NSize", "FGSM", "Normalize", "N_Calibration", "Classif_Loss", "DLoss", "Dist_Fct", "Prototypes", "NPos", "NNeg", "N_Neighbors"
    ])

    if len(df) > 0:
        # Get unique images (filenames)
        unique_images = df['Filename'].unique()
        
        # Dropdown to select an image
        selected_image_idx = st.selectbox(
            "Select an image to view all analysis results:",
            options=range(len(unique_images)),
            format_func=lambda i: unique_images[i],
            key="image_selectbox_tab2"
        )
        selected_filename_tab2 = unique_images[selected_image_idx]

        # ---- Helpers for bulk analysis (hoisted for reuse) ---- #
        def _args_from_model_row(row_dict):
            local_args = argparse.Namespace(**vars(args))
            local_args.model_name = row_dict.get("Model Name", args.model_name)
            local_args.new_size = ensure_int(row_dict.get("NSize", args.new_size))
            local_args.fgsm = row_dict.get("FGSM", args.fgsm)
            local_args.normalize = row_dict.get("Normalize", args.normalize)
            local_args.n_calibration = row_dict.get("N_Calibration", args.n_calibration)
            local_args.classif_loss = row_dict.get("Classif_Loss", args.classif_loss)
            local_args.dloss = row_dict.get("DLoss", args.dloss)
            local_args.dist_fct = row_dict.get("Dist_Fct", args.dist_fct)
            local_args.prototypes_to_use = row_dict.get("Prototypes", args.prototypes_to_use)
            local_args.n_positives = ensure_int(row_dict.get("NPos", args.n_positives))
            local_args.n_negatives = ensure_int(row_dict.get("NNeg", args.n_negatives))
            local_args.n_neighbors = ensure_int(row_dict.get("N_Neighbors", args.n_neighbors))
            local_args.model_id = row_dict.get("Model ID") or row_dict.get("Model_ID") or args.model_id
            return local_args

        def _run_models_on_image(filename: str, model_df: pd.DataFrame, label: str, show_val: bool, fast_infer: bool):
            file_path = os.path.join("data/queries", filename.split("/")[-1])
            if not os.path.exists(file_path):
                st.error(f"File not found for {label}: {file_path}")
                return
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
            except Exception as e:
                st.error(f"Could not read {file_path}: {e}")
                return

            total = len(model_df)
            if total == 0:
                st.info("No models available to run.")
                return

            progress = st.progress(0.0)
            status = st.empty()
            failures = []
            for idx, row in model_df.iterrows():
                status.write(f"Analyzing {filename} with model {row.get('Model ID', row.get('Model_ID'))} ({row.get('Model Name')})")
                try:
                    local_args_bulk = _args_from_model_row(row.to_dict())
                    run_analysis_on_file(filename, file_bytes, local_args_bulk, force_reanalyze=False, show_validation_metrics=show_val, fast_infer=fast_infer)
                except Exception as e:  # keep iterating on failure
                    failures.append(f"Model {row.get('Model ID', row.get('Model_ID'))}: {e}")
                progress.progress((idx + 1) / total)

            status.write(f"Finished {label}")
            if failures:
                st.warning("Some analyses failed:\n" + "\n".join(failures))
            else:
                st.success("✅ Completed without errors")

        # ---- Bulk Analysis: all images x all best models ---- #
        st.markdown("---")
        st.subheader("Bulk Analysis")
        if 'best_models_table' in st.session_state and st.session_state['best_models_table'] is not None and not st.session_state['best_models_table'].empty:
            bm_table_tab2 = st.session_state['best_models_table']
            show_val_bulk = st.checkbox("Show validation metrics during bulk runs", value=False, key="bulk_show_val")
            fast_infer_bulk = st.checkbox("Use fast prototype inference (skip KNN fit)", value=True, key="bulk_fast_infer")
            if st.button("🌀 Analyze ALL images with ALL models", key="bulk_all_images_top"):
                total_tasks = len(unique_images) * len(bm_table_tab2)
                if total_tasks == 0:
                    st.info("Nothing to analyze.")
                else:
                    progress_all = st.progress(0.0)
                    status_all = st.empty()
                    failures_all = []
                    task_idx = 0

                    for fname in unique_images:
                        file_path = f"data/queries/{fname.split('/')[-1]}"
                        if not os.path.exists(file_path):
                            failures_all.append(f"{fname}: file not found")
                            task_idx += len(bm_table_tab2)
                            progress_all.progress(min(1.0, task_idx / total_tasks))
                            continue
                        try:
                            with open(file_path, 'rb') as f:
                                file_bytes_all = f.read()
                        except Exception as e:
                            failures_all.append(f"{fname}: read error {e}")
                            task_idx += len(bm_table_tab2)
                            progress_all.progress(min(1.0, task_idx / total_tasks))
                            continue

                        for _, row in bm_table_tab2.iterrows():
                            status_all.write(f"Analyzing {fname} with model {row.get('Model ID')}")
                            try:
                                local_args_bulk = _args_from_model_row(row.to_dict())
                                run_analysis_on_file(fname, file_bytes_all, local_args_bulk, force_reanalyze=False, show_validation_metrics=show_val_bulk, fast_infer=fast_infer_bulk)
                            except Exception as e:
                                failures_all.append(f"{fname} / model {row.get('Model ID')}: {e}")
                            task_idx += 1
                            progress_all.progress(min(1.0, task_idx / total_tasks))

                    status_all.write("Finished all images/models")
                    if failures_all:
                        st.warning("Some analyses failed:\n" + "\n".join(failures_all))
                    else:
                        st.success("✅ Completed all analyses")
        else:
            st.info("No best models available for bulk analysis yet.")

        st.markdown("---")

        ran_sidebar_analysis = False
        # Run this image with the model currently selected in the left sidebar
        cols_actions = st.columns(2)
        with cols_actions[0]:
            if st.button("▶️ Analyze with sidebar model", key=f"analyze_sidebar_model_tab2_{selected_filename_tab2}"):
                file_path_sidebar = f"data/queries/{selected_filename_tab2.split('/')[-1]}"
                if os.path.exists(file_path_sidebar):
                    with open(file_path_sidebar, 'rb') as f:
                        file_bytes_sidebar = f.read()
                    run_analysis_on_file(selected_filename_tab2, file_bytes_sidebar, args, force_reanalyze=True, show_validation_metrics=True)
                    st.success("Analysis queued with current sidebar model settings.")
                    ran_sidebar_analysis = True
                else:
                    st.error(f"❌ File not found: {file_path_sidebar}")
        with cols_actions[1]:
            if st.button("▶️ Analyze selected image with all models", key=f"bulk_single_{selected_filename_tab2}"):
                if 'best_models_table' in st.session_state and st.session_state['best_models_table'] is not None and not st.session_state['best_models_table'].empty:
                    show_val_bulk = st.session_state.get("bulk_show_val", False)
                    fast_infer_bulk = st.session_state.get("bulk_fast_infer", True)
                    _run_models_on_image(selected_filename_tab2, st.session_state['best_models_table'], label=selected_filename_tab2, show_val=show_val_bulk, fast_infer=fast_infer_bulk)
                else:
                    st.info("No best models available to run bulk analysis.")
        
        # Refresh results if a new sidebar analysis just ran
        if ran_sidebar_analysis:
            cursor.execute(query, (st.session_state.person_id,))
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=[
                "Id", "Model_ID", "Filename", "Model Name", "Task", "Confidence", "Timestamp", "Pred_Label", "Log Path",
                "NSize", "FGSM", "Normalize", "N_Calibration", "Classif_Loss", "DLoss", "Dist_Fct", "Prototypes", "NPos", "NNeg", "N_Neighbors"
            ])

        # Filter results for the selected image; keep only the latest entry per Model_ID
        image_results_full = df[df['Filename'] == selected_filename_tab2].copy().reset_index(drop=True)
        image_results_sorted = image_results_full.sort_values("Timestamp", ascending=False)
        image_results = image_results_sorted.drop_duplicates(subset=["Model_ID"], keep="first").reset_index(drop=True)
        
        st.write(f"**All models tried on: {selected_filename_tab2}**")
        st.write(f"Total unique models: {len(image_results)}")

        # Pull best_models_table early for display consistency; if missing, try to rebuild lightweight
        best_models_table = st.session_state.get('best_models_table', None)
        if best_models_table is None:
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
                cols = [
                    "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                    "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", "#"
                ]
            except Exception:
                try:
                    cursor.execute(
                        """
                        SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct,
                               classif_loss, n_calibration, accuracy, mcc, normalize, n_neighbors, log_path
                        FROM best_models_registry
                        ORDER BY mcc DESC
                        """
                    )
                    model_rows = cursor.fetchall()
                    cols = [
                        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                        "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path"
                    ]
                except Exception:
                    model_rows = []
                    cols = []

            if model_rows:
                try:
                    import pandas as pd
                    df_tmp = pd.DataFrame(model_rows, columns=cols)
                    if "#" not in df_tmp.columns:
                        df_tmp.insert(0, "#", range(1, len(df_tmp) + 1))
                    best_models_table = df_tmp[[c for c in df_tmp.columns if c != "Log Path"]]
                except Exception:
                    best_models_table = None

        if st.button("🗑️ Delete All Analyses For This Image", key="delete_all_results_tab2"):
            try:
                cursor.execute(
                    "DELETE FROM results WHERE filename=%s AND person_id=%s",
                    (selected_filename_tab2, st.session_state.person_id),
                )
                conn.commit()
                st.success(f"All analyses for {selected_filename_tab2} deleted.")
                st.rerun()
            except Error as e:
                st.error(f"❌ Could not delete analyses: {e}")

        if best_models_table is not None and len(best_models_table) > 0:
            st.caption("Best models (same as 'Select from best models') to keep numbering consistent:")
            st.markdown("**Table:** Best Models Reference")
            display_cols = [col for col in best_models_table.columns if col != "Log Path"]
            st.dataframe(best_models_table[display_cols], use_container_width=True)
        
        # Display table of all models tried on this image (sorted newest first)
        st.markdown("**Table:** Analyses for Selected Image")
        display_cols = ["Model_ID", "Model Name", "Pred_Label", "Confidence", "Timestamp", "NSize", "FGSM", "Normalize"]
        image_results_display = image_results.sort_values("Timestamp", ascending=False)
        st.dataframe(image_results_display[display_cols], use_container_width=True)

        # Dropdown to select which model result to view in detail
        selected_result_idx = st.selectbox(
            "Select a model result to view details:",
            options=range(len(image_results)),
            format_func=lambda i: (
                f"Model {image_results.iloc[i]['Model_ID']} - {image_results.iloc[i]['Model Name']} | {image_results.iloc[i]['Pred_Label']} ({float(image_results.iloc[i]['Confidence']):.2f})"
                if image_results.iloc[i]['Confidence'] is not None
                else f"Model {image_results.iloc[i]['Model_ID']} - {image_results.iloc[i]['Model Name']} | {image_results.iloc[i]['Pred_Label']}"
            ),
            key="result_detail_selectbox_tab2"
        )
        
        row_tab2 = image_results.iloc[selected_result_idx]
        selected_result_id_tab2 = int(row_tab2['Id'])
        pred_label = row_tab2['Pred_Label']
        confidence = row_tab2['Confidence']
        log_path = row_tab2['Log Path']
        selected_model_id_tab2 = row_tab2.get('Model_ID')

        # (Apply selected model button removed by user request)
        
        # Create a local args object for re-computation that matches the selected result
        local_args_tab2 = argparse.Namespace(**vars(args))
        local_args_tab2.task = row_tab2['Task']
        local_args_tab2.model_name = row_tab2['Model Name']
        local_args_tab2.new_size = ensure_int(row_tab2['NSize'])
        local_args_tab2.fgsm = row_tab2['FGSM']
        local_args_tab2.normalize = row_tab2['Normalize']
        local_args_tab2.n_calibration = row_tab2['N_Calibration']
        local_args_tab2.classif_loss = row_tab2['Classif_Loss']
        local_args_tab2.dloss = row_tab2['DLoss']
        local_args_tab2.prototypes_to_use = row_tab2['Prototypes']
        local_args_tab2.n_positives = ensure_int(row_tab2['NPos'])
        local_args_tab2.n_negatives = ensure_int(row_tab2['NNeg'])
        local_args_tab2.n_neighbors = ensure_int(row_tab2['N_Neighbors'])
        local_args_tab2.model_id = selected_model_id_tab2
        
        # Extract the dataset name from log_path to set local_args_tab2.path correctly
        p_extra = extract_params_from_log_path(log_path)
        if 'Dataset' in p_extra:
             local_args_tab2.path = os.path.join('data', p_extra['Dataset'])

        # Do not auto-run analysis when switching past results; user can trigger manually if needed.
        st.session_state['tab2_last_result_id'] = selected_result_id_tab2
        
        # Add Delete Selected Result button
        if st.button("🗑️ Delete Selected Result", key="delete_selected_result_tab2"):
            try:
                cursor.execute("DELETE FROM results WHERE id=%s AND person_id=%s", (selected_result_id_tab2, st.session_state.person_id))
                conn.commit()
                st.success(f"Result deleted.")
                st.rerun()
            except Error as e:
                st.error(f"❌ Could not delete result: {e}")
        
        # Display the selected result details
        st.write(f"**Prediction:** {pred_label} ({confidence:.2f} confidence)")
        
        def on_shap_layer_change():
            st.session_state['shap_layer'] = st.session_state[f"adj_shap_layer_{selected_filename_tab2}"]

        with st.expander("🛠️ Adjust SHAP Layer (Recompute Below)"):
            st.number_input("SHAP Layer", value=args.shap_layer, step=1, 
                            key=f"adj_shap_layer_{selected_filename_tab2}",
                            on_change=on_shap_layer_change)
        
        # Add action buttons
        cols = st.columns(4)
        with cols[0]:
            if st.button("▶️ Run Analysis", key=f"run_analysis_tab2_{selected_filename_tab2}"):
                file_path = f'data/queries/{selected_filename_tab2.split("/")[-1]}'
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                    run_analysis_on_file(
                        selected_filename_tab2, file_bytes, local_args_tab2, force_reanalyze=True
                    )
                    st.rerun()
                else:
                    st.error(f"❌ File not found: {file_path}")
        with cols[1]:
            if st.button("🔄 Force Re-analysis", key=f"force_reanalyze_tab2_{selected_filename_tab2}"):
                    # Load the file from data/queries directory
                    file_path = f'data/queries/{selected_filename_tab2.split("/")[-1]}'
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            file_bytes = f.read()
                            pred_label_new, pred_confidence_new, log_path_new, _ = run_analysis_on_file(
                                selected_filename_tab2, file_bytes, local_args_tab2, force_reanalyze=True
                            )
                        st.rerun()
                    else:
                        st.error(f"❌ File not found: {file_path}")
        with cols[2]:
            if st.button("🧠 Compute SHAP Gradients", key=f"compute_grad_shap_tab2_{selected_filename_tab2}"):
                    with st.spinner("Computing SHAP gradients... this may take a moment"):
                        try:
                            # Load model and data
                            random.seed(1)
                            torch.manual_seed(1)
                            np.random.seed(1)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(1)
                            
                            model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
                                load_model_and_prototypes(local_args_tab2)
                            
                            train = TrainAE(local_args_tab2, local_args_tab2.path, load_tb=False, log_metrics=True, keep_models=True,
                                          log_inputs=False, log_plots=True, log_tb=False, log_neptune=True,
                                          log_mlflow=False, groupkfold=local_args_tab2.groupkfold)
                            train.n_batches = len(unique_batches)
                            train.n_cats = len(unique_labels)
                            train.unique_batches = unique_batches
                            train.unique_labels = unique_labels
                            train.epoch = 1
                            train.model = model
                            # TrainAE.predict() writes artifacts under self.complete_log_path; set it for this ad-hoc run
                            train.complete_log_path = log_path
                            train.params = {
                                'n_neighbors': local_args_tab2.n_neighbors,
                                'lr': 0,
                                'wd': 0,
                                'smoothing': 0,
                                'is_transform': 0,
                                'valid_dataset': local_args_tab2.valid_dataset
                            }
                            train.set_arcloss()
                            
                            lists, traces = get_empty_traces()
                            train.complete_log_path = log_path
                            loaders = get_images_loaders(data=data,
                                                        random_recs=local_args_tab2.random_recs,
                                                        weighted_sampler=0,
                                                        is_transform=0,
                                                        samples_weights=None,
                                                        epoch=1,
                                                        unique_labels=unique_labels,
                                                        triplet_dloss=local_args_tab2.dloss, bs=local_args_tab2.bs,
                                                        prototypes_to_use=local_args_tab2.prototypes_to_use,
                                                        prototypes=prototypes,
                                                        size=local_args_tab2.new_size,
                                                        normalize=local_args_tab2.normalize)
                            
                            with torch.no_grad():
                                _, best_lists1, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
                                for group in ["train", "valid", "test"]:
                                    _, best_lists2, traces, knn = train.predict(group, loaders[group], lists, traces)
                            
                            # best_lists = {**best_lists1, **best_lists2}
                            best_lists = best_lists2
                            nets = {'cnn': shap_model, 'knn': knn}
                            
                            original, image = get_image(f'data/queries/{selected_filename_tab2.split("/")[-1]}', size=image_size, normalize=args.normalize)
                            inputs = {
                                'queries': {"inputs": [image]},
                                'train': {
                                    "inputs": [
                                        torch.concatenate(best_lists['train']['inputs']),
                                        torch.concatenate(best_lists['valid']['inputs'])
                                    ],
                                },
                            }
                            
                            complete_log_path = log_path
                            os.makedirs(f'{complete_log_path}/gradients_shap', exist_ok=True)
                            
                            base_filename = selected_filename_tab2.split("/")[-1]
                            base_filename = strip_extension(selected_filename_tab2.split("/")[-1])
                            
                            log_shap_gradients_only(
                                nets, i=0, inputs=inputs, group='queries', name=base_filename, log_path=complete_log_path,
                                layer=args.shap_layer
                            )
                            st.success("✅ SHAP gradients computed and saved!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error computing SHAP gradients: {e}")

        with cols[3]:
            if st.button("🧠 Compute KNN SHAP", key=f"compute_knn_shap_tab2_{selected_filename_tab2}"):
                    with st.spinner("Computing KNN SHAP explanations... this may take a moment"):
                        try:
                            # Load model and data
                            random.seed(1)
                            torch.manual_seed(1)
                            np.random.seed(1)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(1)

                            model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
                                load_model_and_prototypes(local_args_tab2)

                            train = TrainAE(local_args_tab2, local_args_tab2.path, load_tb=False, log_metrics=True, keep_models=True,
                                          log_inputs=False, log_plots=True, log_tb=False, log_neptune=True,
                                          log_mlflow=False, groupkfold=local_args_tab2.groupkfold)
                            train.n_batches = len(unique_batches)
                            train.n_cats = len(unique_labels)
                            train.unique_batches = unique_batches
                            train.unique_labels = unique_labels
                            train.epoch = 1
                            train.model = model
                            # TrainAE.predict() expects self.complete_log_path for artifact logging
                            train.complete_log_path = log_path
                            train.params = {
                                'n_neighbors': local_args_tab2.n_neighbors,
                                'lr': 0,
                                'wd': 0,
                                'smoothing': 0,
                                'is_transform': 0,
                                'valid_dataset': local_args_tab2.valid_dataset
                            }
                            train.set_arcloss()

                            lists, traces = get_empty_traces()
                            loaders = get_images_loaders(data=data,
                                                        random_recs=local_args_tab2.random_recs,
                                                        weighted_sampler=0,
                                                        is_transform=0,
                                                        samples_weights=None,
                                                        epoch=1,
                                                        unique_labels=unique_labels,
                                                        triplet_dloss=local_args_tab2.dloss, bs=local_args_tab2.bs,
                                                        prototypes_to_use=local_args_tab2.prototypes_to_use,
                                                        prototypes=prototypes,
                                                        size=local_args_tab2.new_size,
                                                        normalize=local_args_tab2.normalize)

                            with torch.no_grad():
                                _, best_lists1, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
                                for group in ["train", "valid", "test"]:
                                    _, best_lists2, traces, knn = train.predict(group, loaders[group], lists, traces)

                            best_lists = {**best_lists1, **best_lists2}
                            nets = {'cnn': shap_model, 'knn': knn}

                            _, image = get_image(
                                f'data/queries/{selected_filename_tab2.split("/")[-1]}',
                                size=image_size,
                                normalize=local_args_tab2.normalize,
                            )
                            inputs = {
                                'queries': {"inputs": [image]},
                                'train': {
                                    "inputs": [
                                        torch.concatenate(best_lists['train']['inputs']),
                                        torch.concatenate(best_lists['valid']['inputs'])
                                    ],
                                },
                            }

                            complete_log_path = log_path
                            os.makedirs(f'{complete_log_path}/knn_shap', exist_ok=True)

                            base_filename = selected_filename_tab2.split("/")[-1]
                            base_filename = strip_extension(selected_filename_tab2.split("/")[-1])

                            log_shap_knn_or_deep(
                                nets, i=0, inputs=inputs, group='queries', name=base_filename, log_path=complete_log_path,
                                layer=args.shap_layer
                            )
                            st.success("✅ KNN SHAP computed and saved!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error computing KNN SHAP: {e}")

        
        # Display SHAP explanations
        base_name = strip_extension(selected_filename_tab2.split("/")[-1])

        # Show original uploaded image
        raw_image_path = os.path.join('data/queries', selected_filename_tab2.split("/")[-1])
        if os.path.exists(raw_image_path):
            try:
                raw_img = Image.open(raw_image_path).convert('RGB')
                st.image(raw_img, caption="Original uploaded image", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load original image: {e}")
        else:
            st.info(f"Original image not found at {raw_image_path}")
        
        # Try to display SHAP gradient explanation
        grad_shap_path = f'{log_path}/gradients_shap/queries_{base_name}_layer{args.shap_layer}.png'
        if os.path.exists(grad_shap_path):
            fig = plt.imread(grad_shap_path)
            st.image(fig, caption=f"SHAP Gradient Explanation (layer {args.shap_layer})", use_container_width=True)
        else:
            st.info(f"SHAP gradient explanation not found at: {grad_shap_path}")
        
        # Try to display KNN SHAP explanation
        knn_shap_path = f'{log_path}/knn_shap/queries_{base_name}_layer{args.shap_layer}.png'
        if os.path.exists(knn_shap_path):
            fig = plt.imread(knn_shap_path)
            st.image(fig, caption=f"KNN SHAP Explanation (layer {args.shap_layer})", use_container_width=True)
        else:
            st.info(f"KNN SHAP explanation not found")
        
        # PCA visualization with prototype overlay (if available)
        st.divider()
        st.subheader("🧭 PCA with Prototypes")
        pca_candidates = [
            ("labels", os.path.join(log_path, "labels_PCA.png")),
            ("clusters", os.path.join(log_path, "clusters_PCA.png")),
            ("batches", os.path.join(log_path, "batches_PCA.png")),
            ("subcenters", os.path.join(log_path, "subcenters_PCA.png")),
        ]
        pca_found = False
        for tag, ppath in pca_candidates:
            if os.path.exists(ppath):
                pca_found = True
                st.image(ppath, caption=f"PCA ({tag}) with prototypes overlay", use_container_width=True)
        if not pca_found:
            st.info("No PCA plots found for this result. Generate them during training or place *_PCA.png files in the run folder.")

        # (Removed duplicate) Do not display KNN SHAP main image to avoid duplication

        # Grad-CAM Display with local controls
        st.divider()
        st.subheader("💡 Grad-CAM Visualization")
        
        # Determine the image-specific subdirectory for this result
        base_name = strip_extension(selected_filename_tab2.split("/")[-1])
        grad_cam_dir = os.path.join(log_path, base_name)
        
        def on_gc_params_change_t2():
            st.session_state['grad_cam_layer'] = st.session_state[f"gc_layer_t2_{selected_filename_tab2}"]
            st.session_state['grad_cam_alpha'] = st.session_state[f"gc_alpha_t2_{selected_filename_tab2}"]

        gc_cols = st.columns([1, 2])
        with gc_cols[0]:
            st.number_input("Layer", value=args.grad_cam_layer, step=1, 
                            key=f"gc_layer_t2_{selected_filename_tab2}",
                            on_change=on_gc_params_change_t2)
            st.slider("Alpha", 0.0, 1.0, args.grad_cam_alpha, 0.05, 
                      key=f"gc_alpha_t2_{selected_filename_tab2}",
                      on_change=on_gc_params_change_t2)
            
            if st.button("🧠 Compute Grad-CAM", key=f"compute_grad_cam_inline_t2_{selected_filename_tab2}"):
                with st.spinner("Computing Grad-CAM..."):
                    try:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # Deterministic
                        random.seed(1)
                        torch.manual_seed(1)
                        np.random.seed(1)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(1)

                        # Read current layer/alpha from UI state, not from args
                        current_layer = st.session_state.get(f"gc_layer_t2_{selected_filename_tab2}", args.grad_cam_layer)
                        current_alpha = st.session_state.get(f"gc_alpha_t2_{selected_filename_tab2}", args.grad_cam_alpha)

                        # CRITICAL: Clear model cache to ensure we load the correct model for this result
                        _clear_cached_model()
                        
                        # Load model and data for THIS specific result's parameters
                        model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
                            load_model_and_prototypes(local_args_tab2)
                        st.session_state['last_image_size_tab2'] = image_size
                        st.session_state['last_normalize_tab2'] = local_args_tab2.normalize
                        
                        # Display which model is being used for verification
                        st.info(f"Using model: {local_args_tab2.model_name} (size: {local_args_tab2.new_size}, prototypes: {local_args_tab2.prototypes_to_use})")

                        # Prepare input
                        _, image = get_image(
                            f'data/queries/{selected_filename_tab2.split("/")[-1]}',
                            size=image_size,
                            normalize=local_args_tab2.normalize
                        )
                        inputs = { 'queries': { 'inputs': [image] } }

                        # Generate Grad-CAM for all classes
                        base_name = strip_extension(selected_filename_tab2.split("/")[-1])
                        # Organize by image: create subdirectory per image to avoid collisions
                        image_output_dir = os.path.join(log_path, base_name)
                        os.makedirs(image_output_dir, exist_ok=True)
                        log_grad_cam_all_classes(
                            model,
                            0,
                            inputs,
                            'queries',
                            image_output_dir,
                            base_name,
                            prototypes['class']['train'],
                            device=device_str,
                            layer=current_layer,
                            alpha=current_alpha
                        )
                        
                        # Create montage from individual class images
                        class_labels = sorted(prototypes['class']['train'].keys())
                        class_images = []
                        for lbl in class_labels:
                            class_img_path = os.path.join(image_output_dir, f"{base_name}_class{lbl}.png")
                            if os.path.exists(class_img_path):
                                class_images.append(plt.imread(class_img_path))
                        
                        if class_images:
                            # Create horizontal montage
                            fig_montage, axes_montage = plt.subplots(1, len(class_images), figsize=(5 * len(class_images), 5))
                            if len(class_images) == 1:
                                axes_montage = [axes_montage]
                            for ax_m, img_m, lbl_m in zip(axes_montage, class_images, class_labels):
                                ax_m.imshow(img_m)
                                ax_m.set_title(f"Class: {lbl_m}")
                                ax_m.axis('off')
                            plt.tight_layout()
                            montage_path = os.path.join(image_output_dir, f'{base_name}_grad_cam_all_classes_layer{current_layer}.png')
                            plt.savefig(montage_path, dpi=150, bbox_inches='tight')
                            plt.close()
                        
                        # Clear the display cache so the newly computed image shows
                        display_key = f"tab2_grad_cam_{selected_result_id_tab2}_{current_layer}"
                        st.session_state[display_key] = None
                        st.success("✅ Grad-CAM generated for all classes.")
                        st.rerun()

                        # Cleanup
                        del model, shap_model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        st.error(f"❌ Error computing Grad-CAM: {e}")
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        # Read current layer from UI state (not from args, which may be stale)
        def rebuild_grad_cam_from_cache_tab2(layer: int, alpha: float) -> bool:
            """Re-render overlays from cached heatmaps so alpha tweaks don't trigger full recompute."""
            if not log_path or not os.path.isdir(log_path):
                return False

            # Look in the image-specific subdirectory
            image_grad_cam_dir = os.path.join(log_path, base_name)
            if not os.path.isdir(image_grad_cam_dir):
                return False

            prefix = f"{base_name}_class"
            # Filter heatmap files by the specific layer
            layer_suffix = f"_layer{layer}_heatmap.npy"
            heatmap_files = [f for f in os.listdir(image_grad_cam_dir) if f.startswith(prefix) and f.endswith(layer_suffix)]
            if not heatmap_files:
                return False

            image_size_cached = st.session_state.get('last_image_size_tab2', args.new_size)
            normalize_cached = st.session_state.get('last_normalize_tab2', args.normalize)
            img_path = f'data/queries/{selected_filename_tab2.split("/")[-1]}'

            try:
                _, image = get_image(img_path, size=image_size_cached, normalize=normalize_cached)
            except Exception:
                return False

            def _parse_label(fname: str) -> str:
                # Extract label from: {base}_class{LABEL}_layer{N}_heatmap.npy
                core = fname[len(prefix):]
                core = core.split("_layer")[0]  # Remove _layerN_heatmap.npy
                return core

            class_labels = sorted(_parse_label(f) for f in heatmap_files)
            class_images = []
            for lbl in class_labels:
                heatmap_path = os.path.join(image_grad_cam_dir, f"{base_name}_class{lbl}_layer{layer}_heatmap.npy")
                if not os.path.exists(heatmap_path):
                    continue
                overlay_path = os.path.join(image_grad_cam_dir, f"{base_name}_class{lbl}_layer{layer}.png")
                try:
                    heatmap = np.load(heatmap_path)
                    save_overlay_from_heatmap(image, heatmap, overlay_path, alpha=alpha)
                    class_images.append((lbl, plt.imread(overlay_path)))
                except Exception:
                    continue

            if not class_images:
                return False

            fig_montage, axes_montage = plt.subplots(1, len(class_images), figsize=(5 * len(class_images), 5))
            if len(class_images) == 1:
                axes_montage = [axes_montage]
            for ax_m, (lbl_m, img_m) in zip(axes_montage, class_images):
                ax_m.imshow(img_m)
                ax_m.set_title(f"Class: {lbl_m}")
                ax_m.axis('off')
            plt.tight_layout()
            montage_path = os.path.join(image_grad_cam_dir, f'{base_name}_grad_cam_all_classes_layer{layer}.png')
            plt.savefig(montage_path, dpi=150, bbox_inches='tight')
            plt.close()
            return True

        current_display_layer = st.session_state.get(f"gc_layer_t2_{selected_filename_tab2}", args.grad_cam_layer)
        current_display_alpha = st.session_state.get(f"gc_alpha_t2_{selected_filename_tab2}", args.grad_cam_alpha)
        # Display all-classes montage (original + all class Grad-CAMs)
        image_grad_cam_dir = os.path.join(log_path, base_name)
        grad_cam_all_path = os.path.join(image_grad_cam_dir, f'{base_name}_grad_cam_all_classes_layer{current_display_layer}.png')
        
        # Re-render overlays from cached heatmaps when alpha changes
        rebuild_grad_cam_from_cache_tab2(current_display_layer, current_display_alpha)
        
        with gc_cols[1]:
            if os.path.exists(grad_cam_all_path):
                fig = plt.imread(grad_cam_all_path)
                st.image(fig, caption=f"Grad-CAM All Classes (Layer {current_display_layer})", use_container_width=True)
            else:
                st.info(f"Grad-CAM layer {current_display_layer} not computed for this result. Click 'Compute Grad-CAM' button above.")
    else:
        st.info("No past results found for this family member.")


# ========================= TAB 5: Grad-CAM Gallery ========================= #
with tab5:
    st.header("🖼️ Grad-CAM Gallery")

    cursor.execute(
        '''
         SELECT id, model_id, filename, model_name, task, confidence, timestamp, pred_label, log_path,
             nsize, fgsm, normalize, n_calibration, classif_loss, dloss, dist_fct, prototypes, npos, nneg, n_neighbors
        FROM results
        WHERE person_id=%s
        ORDER BY timestamp DESC
        ''',
        (st.session_state.person_id,),
    )
    gallery_rows = cursor.fetchall()

    df_gallery = pd.DataFrame(
        gallery_rows,
        columns=[
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
        ],
    )

    if len(df_gallery) == 0:
        st.info("No analyses available yet. Run a model to populate the gallery.")
    else:
        df_gallery = df_gallery.sort_values("Timestamp", ascending=False)
        df_gallery = df_gallery.drop_duplicates(subset=["Model_ID", "Filename"], keep="first").reset_index(drop=True)

        # Allow focusing the gallery on a specific model
        model_choices = df_gallery[["Model_ID", "Model Name", "NSize", "Dist_Fct", "Normalize"]].drop_duplicates().sort_values("Model_ID")
        
        # Get MCC info from best models table for richer display
        model_number_map, best_models_table = _ensure_model_number_map(cursor)
        
        # Build model info with additional details
        model_info_list = []
        for idx, (_, row) in enumerate(model_choices.iterrows()):
            # Handle missing/NaN Model_IDs safely (skip rows without a valid model id)
            model_id_val = row.get("Model_ID")
            if model_id_val is None or (isinstance(model_id_val, float) and np.isnan(model_id_val)):
                continue
            try:
                model_id = int(model_id_val)
            except Exception:
                continue
            model_name = row.get("Model Name")
            nsize = row.get("NSize")
            dist_fct = row.get("Dist_Fct")
            # Normalize may be stored as boolean or string; map robustly to 'yes'/'no'
            norm_val = row.get("Normalize")
            if isinstance(norm_val, str):
                normalize = 'yes' if norm_val.strip().lower() in ['yes', 'true', '1'] else 'no'
            else:
                normalize = 'yes' if bool(norm_val) else 'no'
            
            # Try to get MCC from best_models_table
            mcc = "?"
            model_num = idx + 1
            if best_models_table is not None and not best_models_table.empty:
                # Try both "Model ID" (with space) and "Model_ID" (with underscore)
                model_id_col = None
                if "Model ID" in best_models_table.columns:
                    model_id_col = "Model ID"
                elif "Model_ID" in best_models_table.columns:
                    model_id_col = "Model_ID"
                
                if model_id_col:
                    match = best_models_table[best_models_table[model_id_col] == model_id]
                    if not match.empty:
                        mcc = f"{float(match.iloc[0].get('MCC', 0)):.3f}"
                        model_num = match.iloc[0].get("#", idx + 1)
            
            model_info = f"#{model_num} - {model_name} (Size:{nsize}, MCC:{mcc}, Dist:{dist_fct}, Norm:{normalize})"
            model_info_list.append({
                "label": model_info,
                "id": model_id,
                "num": model_num
            })
        
        # Allow limiting number of models displayed (slider needs min < max)
        if len(model_info_list) > 1:
            max_models_to_display = st.slider(
                "Number of models to display",
                min_value=1,
                max_value=len(model_info_list),
                value=min(5, len(model_info_list)),
                key="gallery_max_models_slider"
            )
            model_info_list = model_info_list[:max_models_to_display]
        elif len(model_info_list) == 0:
            st.info("No Grad-CAM results available in the gallery yet.")
        
        # Handle pending auto-select (from batch compute) before rendering widgets
        if 'gallery_pending_select_all' in st.session_state:
            for model_info in model_info_list:
                st.session_state[f"model_select_{model_info['id']}"] = True
            st.session_state.pop('gallery_pending_select_all')

        # Multi-select with checkboxes
        st.write("**Select models to view (multiple selection):**")
        selected_model_ids = []
        for model_info in model_info_list:
            if st.checkbox(model_info["label"], value=(model_info["num"] == 1), key=f"model_select_{model_info['id']}"):
                selected_model_ids.append(model_info["id"])
        
        if len(selected_model_ids) > 0:
            df_gallery_view = df_gallery[df_gallery["Model_ID"].isin(selected_model_ids)].copy()
        else:
            df_gallery_view = df_gallery.copy()

        def _has_grad_cam(row) -> bool:
            log_path_row = row.get("Log Path") or ""
            if not log_path_row or not os.path.exists(log_path_row):
                return False
            base = strip_extension(str(row.get("Filename", "")).split("/")[-1])
            # Check in the image-specific subdirectory
            image_dir = os.path.join(log_path_row, base)
            if not os.path.exists(image_dir):
                return False
            pattern = os.path.join(image_dir, f"{base}_class*.png")
            return len(glob.glob(pattern)) > 0

        df_gallery_view["Has_GradCAM"] = df_gallery_view.apply(_has_grad_cam, axis=1)
        per_image_status = df_gallery_view.groupby("Filename")["Has_GradCAM"].any().to_dict()

        st.markdown(
            f"Total results: {len(df_gallery_view)} | Images: {len(per_image_status)} | Missing Grad-CAM: {int((~df_gallery_view['Has_GradCAM']).sum())}"
        )

        unique_images = list(df_gallery_view["Filename"].unique())
        max_images_default = min(12, len(unique_images)) if len(unique_images) > 0 else 0
        # Layer selection for gallery
        gallery_layer = st.slider("Grad-CAM Layer to Display", 3, 7, st.session_state.get('grad_cam_layer', 7), key="gallery_layer_select")
        
        # Only show image count slider if there are 2+ images
        if len(unique_images) > 1:
            max_images_to_show = st.slider("How many images to show", 1, len(unique_images), max_images_default, key="gallery_max_images")
        elif len(unique_images) == 1:
            max_images_to_show = 1
        else:
            max_images_to_show = 0
        
        num_cols = st.slider("Columns", 2, 5, 3, key="gallery_num_cols") if len(unique_images) > 0 else 0
        
        # Add control for max models to display grad-cam for (only if 2+ models exist)
        if len(model_info_list) > 1:
            max_models_grad_cam = st.slider("Max models to show Grad-CAM", 1, len(model_info_list), 
                         min(3, len(model_info_list)), key="gallery_max_models_grad_cam")
        elif len(model_info_list) == 1:
            max_models_grad_cam = 1
        else:
            max_models_grad_cam = 1

        def _list_overlays(fname: str, layer: int):
            """List overlays for an image across all selected models, filtered by layer."""
            rows_for_image = df_gallery_view[df_gallery_view["Filename"] == fname]
            base = strip_extension(str(fname).split("/")[-1])
            model_overlays = {}  # {model_id: [(overlay_path, class_label), ...]}
            
            for _, r in rows_for_image.iterrows():
                model_id = r.get("Model_ID")
                model_name = r.get("Model Name", f"Model {model_id}")
                log_path_row = r.get("Log Path") or ""
                if not log_path_row or not os.path.isdir(log_path_row):
                    continue
                # Look in the image-specific subdirectory
                image_dir = os.path.join(log_path_row, base)
                if not os.path.isdir(image_dir):
                    continue
                # Filter by layer in the glob pattern
                pattern = sorted(glob.glob(os.path.join(image_dir, f"{base}_class*_layer{layer}.png")))
                if pattern:
                    if model_id not in model_overlays:
                        model_overlays[model_id] = []
                    for p in pattern:
                        # Extract label from filename: base_classLABEL_layerN.png
                        parts = p.split("_class")[-1].split("_layer")
                        lbl = parts[0] if len(parts) > 0 else "?"
                        model_overlays[model_id].append((p, lbl, model_name))
            
            return model_overlays  # Returns {model_id: [(path, label, model_name), ...]}

        def _get_pred_label_for_image(fname: str):
            """Get the predicted label for an image from the most recent result."""
            rows_for_image = df_gallery_view[df_gallery_view["Filename"] == fname]
            if len(rows_for_image) > 0:
                # Return pred_label and confidence from most recent
                return str(rows_for_image.iloc[0].get("Pred_Label", "?")), rows_for_image.iloc[0].get("Confidence")
            return None, None

        if max_images_to_show > 0:
            st.subheader("Gallery")
            cols = st.columns(num_cols)
            for idx, fname in enumerate(unique_images[:max_images_to_show]):
                with cols[idx % num_cols]:
                    img_path = os.path.join("data/queries", str(fname).split("/")[-1])
                    model_overlays = _list_overlays(fname, gallery_layer)
                    has_gc = per_image_status.get(fname, False) and len(model_overlays) > 0
                    status_icon = "✅" if has_gc else "⏳"

                    st.markdown(f"**{fname}**")
                    st.caption(f"{status_icon} Grad-CAM {'ready' if has_gc else 'missing'}")

                    # Show original image
                    if os.path.exists(img_path):
                        try:
                            st.image(Image.open(img_path).convert("RGB"), caption="Original", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not load image {fname}: {e}")
                    else:
                        st.info(f"Image not found at {img_path}")

                    # Show overlays for each selected model (limited by max_models_grad_cam)
                    if has_gc:
                        try:
                            pred_lbl, conf = _get_pred_label_for_image(fname)
                            model_ids_sorted = sorted(model_overlays.keys())
                            for idx, model_id in enumerate(model_ids_sorted[:max_models_grad_cam]):
                                overlays = model_overlays[model_id]
                                model_name = overlays[0][2] if overlays else f"Model {model_id}"
                                with st.expander(f"📊 {model_name}", expanded=True):
                                    ov_cols = st.columns(len(overlays))
                                    for ov_col, (ov_path, lbl, _) in zip(ov_cols, overlays):
                                        with ov_col:
                                            # Highlight correct prediction
                                            is_correct = (str(lbl) == pred_lbl)
                                            caption_text = f"Class {lbl}"
                                            if is_correct and conf is not None:
                                                caption_text += f" ✅ ({conf:.2f})"
                                            elif is_correct:
                                                caption_text += " ✅"
                                            st.image(ov_path, caption=caption_text, use_container_width=True)
                            # Show info if more models exist but not displayed
                            if len(model_ids_sorted) > max_models_grad_cam:
                                st.caption(f"ℹ️ {len(model_ids_sorted) - max_models_grad_cam} more model(s) available - adjust slider above")
                        except Exception as e:
                            st.warning(f"Could not load Grad-CAM for {fname}: {e}")
                    else:
                            st.info("Grad-CAM not available yet")

        # Batch compute missing Grad-CAMs (respect current model filter)
        missing_df = df_gallery_view[~df_gallery_view["Has_GradCAM"]].reset_index(drop=True)
        st.markdown("---")
        st.subheader("Compute Missing Grad-CAMs")
        st.write(f"Queued items: {len(missing_df)}")

        # Controls: choose which layer to compute for batch actions in this tab
        def on_gc_params_change_tab5():
            st.session_state['grad_cam_layer'] = st.session_state.get('gc_layer_tab5', st.session_state.get('grad_cam_layer', 7))
            st.session_state['grad_cam_alpha'] = st.session_state.get('gc_alpha_tab5', st.session_state.get('grad_cam_alpha', 0.55))

        gc_control_cols = st.columns(2)
        with gc_control_cols[0]:
            st.number_input(
                "Grad-CAM layer to compute",
                value=int(st.session_state.get('grad_cam_layer', 7)),
                step=1,
                key="gc_layer_tab5",
                on_change=on_gc_params_change_tab5,
            )
        with gc_control_cols[1]:
            st.slider(
                "Alpha",
                0.0,
                1.0,
                float(st.session_state.get('grad_cam_alpha', 0.55)),
                0.05,
                key="gc_alpha_tab5",
                on_change=on_gc_params_change_tab5,
            )

        def _dataset_from_log_path(log_path: str):
            """Extract dataset folder from a best-model log path."""
            try:
                parts = str(log_path).strip("/").split("/")
                # logs/best_models/<task>/<model>/<dataset>/nsize...
                if len(parts) >= 5:
                    return parts[4]
            except Exception:
                return None
            return None

        def _build_args_from_row(row):
            """Build args namespace from a database result row (pandas Series or dict)."""
            local_args = argparse.Namespace(**vars(args))
            
            # Helper to safely get values from pandas Series or dict
            def safe_get(obj, key, default=None):
                try:
                    if hasattr(obj, 'get'):
                        return obj.get(key, default)
                    else:
                        return obj[key] if key in obj else default
                except (KeyError, AttributeError):
                    return default
            
            local_args.task = safe_get(row, "Task", args.task)
            local_args.model_name = safe_get(row, "Model Name", args.model_name)
            # Prefer dataset inferred from this row's log_path
            ds_name = _dataset_from_log_path(safe_get(row, "Log Path"))
            if ds_name:
                local_args.path = os.path.join(data_dir, ds_name)
            local_args.new_size = ensure_int(safe_get(row, "NSize", args.new_size))
            local_args.fgsm = safe_get(row, "FGSM", args.fgsm)
            local_args.normalize = safe_get(row, "Normalize", args.normalize)
            local_args.n_calibration = safe_get(row, "N_Calibration", args.n_calibration)
            local_args.classif_loss = safe_get(row, "Classif_Loss", args.classif_loss)
            local_args.dloss = safe_get(row, "DLoss", args.dloss)
            local_args.dist_fct = safe_get(row, "Dist_Fct", getattr(args, 'dist_fct', 'euclidean'))
            local_args.prototypes_to_use = safe_get(row, "Prototypes", args.prototypes_to_use)
            local_args.n_positives = ensure_int(safe_get(row, "NPos", args.n_positives))
            local_args.n_negatives = ensure_int(safe_get(row, "NNeg", args.n_negatives))
            # Use sidebar n_neighbors value (allows user to override for Grad-CAM recomputation)
            local_args.n_neighbors = ensure_int(args.n_neighbors)
            local_args.model_id = safe_get(row, "Model_ID", args.model_id)
            local_args.grad_cam_layer = int(st.session_state.get("grad_cam_layer", args.grad_cam_layer))
            local_args.grad_cam_alpha = float(st.session_state.get("grad_cam_alpha", args.grad_cam_alpha))
            return local_args

        def _compute_grad_cam_row_layers(row, layers):
            """Compute Grad-CAM overlays for one row (image+model) across multiple layers.
            Loads the model once, then iterates layers to avoid redundant loads.
            """
            local_args = _build_args_from_row(row)
            # Clear cached model to avoid reusing a previous model/config for other rows
            _clear_cached_model()
            model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = load_model_and_prototypes(local_args)
            image_path = os.path.join("data/queries", str(row.get("Filename", "")).split("/")[-1])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            _, image_tensor = get_image(image_path, size=image_size, normalize=local_args.normalize)

            # Validate image tensor dimensions
            if image_tensor.shape[1] <= 0 or image_tensor.shape[2] <= 0:
                raise ValueError(f"Image has invalid dimensions after loading: {image_tensor.shape}. Cannot compute Grad-CAM.")

            inputs = {'queries': {'inputs': [image_tensor]}}

            try:
                class_protos = prototypes.get('class', {}).get('train', {})
            except Exception:
                class_protos = {}
            if not class_protos:
                raise RuntimeError("Missing class prototypes for Grad-CAM computation")

            base_name = strip_extension(str(row.get("Filename", "")).split("/")[-1])
            log_path_base = row.get("Log Path") or local_args.path or "logs"
            output_dir = os.path.join(log_path_base, base_name)
            os.makedirs(output_dir, exist_ok=True)

            # Ensure layers is iterable
            if isinstance(layers, (int, float)):
                layers_iter = [int(layers)]
            else:
                layers_iter = [int(l) for l in layers]

            for lyr in layers_iter:
                log_grad_cam_all_classes(
                    model,
                    0,
                    inputs,
                    'queries',
                    output_dir,
                    base_name,
                    class_protos,
                    device=device_str,
                    layer=int(lyr),
                    alpha=float(local_args.grad_cam_alpha),
                )

                # Optional montage for quick viewing per layer
                class_labels = sorted(class_protos.keys())
                class_images = []
                for lbl in class_labels:
                    class_img_path = os.path.join(output_dir, f"{base_name}_class{lbl}.png")
                    if os.path.exists(class_img_path):
                        class_images.append(plt.imread(class_img_path))
                if class_images:
                    fig_montage, axes_montage = plt.subplots(1, len(class_images), figsize=(4 * len(class_images), 4))
                    if len(class_images) == 1:
                        axes_montage = [axes_montage]
                    for ax_m, img_m, lbl_m in zip(axes_montage, class_images, class_labels):
                        ax_m.imshow(img_m)
                        ax_m.set_title(f"Class: {lbl_m}")
                        ax_m.axis('off')
                    plt.tight_layout()
                    montage_path = os.path.join(output_dir, f"{base_name}_grad_cam_all_classes_layer{int(lyr)}.png")
                    plt.savefig(montage_path, dpi=150, bbox_inches='tight')
                    plt.close()

        def _compute_grad_cam_row(row):
            # Backward-compatible single-layer wrapper
            target_layer = int(st.session_state.get("grad_cam_layer", args.grad_cam_layer))
            _compute_grad_cam_row_layers(row, [target_layer])

        if st.button("🧠 Compute remaining Grad-CAMs", key="compute_missing_gradcam_tab5"):
            if len(missing_df) == 0:
                st.info("All displayed results already have Grad-CAM overlays.")
            else:
                progress = st.progress(0.0)
                status_placeholder = st.empty()
                failures = []
                total = len(missing_df)
                batch_start = time.time()
                for i, (_, row) in enumerate(missing_df.iterrows()):
                    status_placeholder.write(
                        f"Computing Grad-CAM for {row.get('Filename')} (model {row.get('Model_ID')})"
                    )
                    try:
                        _compute_grad_cam_row(row)
                    except Exception as e:
                        failures.append((row.get('Filename'), row.get('Model_ID'), str(e)))
                    progress.progress((i + 1) / total)

                batch_elapsed = time.time() - batch_start
                status_placeholder.write(f"✅ Batch complete ({batch_elapsed:.2f}s)")
                if failures:
                    fail_lines = "\n".join([f"  • {f} (model {m}): {err[:60]}..." if len(err) > 60 else f"  • {f} (model {m}): {err}" for f, m, err in failures])
                    st.warning(f"⚠️ {len(failures)}/{total} Grad-CAMs skipped:\n{fail_lines}")
                    st.caption(f"⏱️ Computed {total - len(failures)}/{total} in {batch_elapsed:.2f}s")
                else:
                    st.success("✅ All missing Grad-CAMs computed")
                    st.caption(f"⏱️ Computed {total} Grad-CAMs in {batch_elapsed:.2f}s")
                    st.rerun()
        
        # Option to recompute all Grad-CAMs (including existing ones)
        if st.button("🔄 Recompute all Grad-CAMs for displayed results", key="recompute_all_gradcam_tab5"):
            progress = st.progress(0.0)
            status_placeholder = st.empty()
            failures = []
            total = len(df_gallery_view)
            batch_start = time.time()
            for i, (_, row) in enumerate(df_gallery_view.iterrows()):
                status_placeholder.write(
                    f"Computing Grad-CAM for {row.get('Filename')} (model {row.get('Model_ID')})"
                )
                try:
                    _compute_grad_cam_row(row)
                except Exception as e:
                    failures.append((row.get('Filename'), row.get('Model_ID'), str(e)))
                progress.progress((i + 1) / total)

            batch_elapsed = time.time() - batch_start
            status_placeholder.write(f"✅ Batch complete ({batch_elapsed:.2f}s)")
            if failures:
                fail_lines = "\n".join([f"  • {f} (model {m}): {err[:60]}..." if len(err) > 60 else f"  • {f} (model {m}): {err}" for f, m, err in failures])
                st.warning(f"⚠️ {len(failures)}/{total} Grad-CAMs skipped:\n{fail_lines}")
                st.caption(f"⏱️ Computed {total - len(failures)}/{total} in {batch_elapsed:.2f}s")
            else:
                st.success(f"✅ All {total} Grad-CAMs computed")
                st.caption(f"⏱️ Computed {total} Grad-CAMs in {batch_elapsed:.2f}s")
                st.rerun()

        # Option to compute all Grad-CAMs for ALL models (unfiltered) but skip existing overlays
        if st.button("🚀 Compute all missing Grad-CAMs for ALL models", key="compute_all_all_models_gradcam_tab5"):
            missing_all_df = df_gallery[~df_gallery.apply(_has_grad_cam, axis=1)].copy()
            if len(missing_all_df) == 0:
                st.info("All models/images already have Grad-CAM overlays.")
            else:
                progress = st.progress(0.0)
                status_placeholder = st.empty()
                failures = []
                total = len(missing_all_df)
                batch_start = time.time()
                for i, (_, row) in enumerate(missing_all_df.iterrows()):
                    status_placeholder.write(
                        f"Computing Grad-CAM for {row.get('Filename')} (model {row.get('Model_ID')})"
                    )
                    try:
                        _compute_grad_cam_row(row)
                    except Exception as e:
                        failures.append((row.get('Filename'), row.get('Model_ID'), str(e)))
                    progress.progress((i + 1) / total)

                batch_elapsed = time.time() - batch_start
                status_placeholder.write(f"✅ Batch complete - processed all missing overlays ({batch_elapsed:.2f}s)")
                if failures:
                    fail_lines = "\n".join([f"  • {f} (model {m}): {err[:60]}..." if len(err) > 60 else f"  • {f} (model {m}): {err}" for f, m, err in failures])
                    st.warning(f"⚠️ {len(failures)}/{total} Grad-CAMs skipped:\n{fail_lines}")
                    st.caption(f"⏱️ Computed {total - len(failures)}/{total} in {batch_elapsed:.2f}s")
                else:
                    st.success(f"✅ All {total} missing Grad-CAMs computed for all models")
                    st.caption(f"⏱️ Computed {total} Grad-CAMs in {batch_elapsed:.2f}s")
                    # Mark pending auto-select for next render
                    st.session_state['gallery_pending_select_all'] = True
                    st.rerun()

        # Option to compute ALL layers for ALL images and ALL models
        st.markdown("---")
        st.subheader("Compute ALL Layers for ALL Images/Models")
        st.caption("Runs Grad-CAM for every selected layer on each result. Heavy operation.")
        default_layers_text = st.session_state.get('gallery_layers_to_compute', '3,4,5,6,7')
        layers_text = st.text_input(
            "Layers to compute (comma-separated)",
            value=default_layers_text,
            key="gallery_layers_to_compute",
        )
        # Parse layer list safely
        def _parse_layers(txt: str):
            vals = []
            for tok in str(txt).split(','):
                tok = tok.strip()
                if not tok:
                    continue
                # support simple ranges like 0-7
                if '-' in tok:
                    try:
                        a, b = tok.split('-', 1)
                        a_i, b_i = int(a), int(b)
                        step = 1 if a_i <= b_i else -1
                        vals.extend(list(range(a_i, b_i + step, step)))
                        continue
                    except Exception:
                        pass
                try:
                    vals.append(int(tok))
                except Exception:
                    continue
            # de-duplicate and sort
            return sorted(set(vals))

        if st.button("🧠 Compute ALL layers for ALL images and models", key="compute_all_layers_all_models_tab5"):
            layer_list = _parse_layers(layers_text)
            if not layer_list:
                st.warning("Please provide at least one valid layer index.")
            else:
                progress = st.progress(0.0)
                status_placeholder = st.empty()
                failures = []
                # Use df_gallery (deduped by model/image earlier) to cover all models
                total = len(df_gallery)
                batch_start = time.time()
                for i, (_, row) in enumerate(df_gallery.iterrows()):
                    status_placeholder.write(
                        f"Computing layers {layer_list} for {row.get('Filename')} (model {row.get('Model_ID')})"
                    )
                    try:
                        _compute_grad_cam_row_layers(row, layer_list)
                    except Exception as e:
                        failures.append((row.get('Filename'), row.get('Model_ID'), str(e)))
                    progress.progress((i + 1) / max(1, total))

                batch_elapsed = time.time() - batch_start
                status_placeholder.write(f"✅ Batch complete - processed all layers for all results ({batch_elapsed:.2f}s)")
                if failures:
                    fail_lines = "\n".join([f"  • {f} (model {m}): {err[:60]}..." if len(err) > 60 else f"  • {f} (model {m}): {err}" for f, m, err in failures])
                    st.warning(f"⚠️ {len(failures)}/{total} items failed:\n{fail_lines}")
                    st.caption(f"⏱️ Completed {total - len(failures)}/{total} in {batch_elapsed:.2f}s")
                else:
                    st.success(f"✅ Computed layers {layer_list} for all {total} results")
                    st.caption(f"⏱️ Completed in {batch_elapsed:.2f}s")
                    # Mark pending auto-select for next render
                    st.session_state['gallery_pending_select_all'] = True
                    st.rerun()


# ========================= TAB 3: New Analysis ========================= #
with tab3:
    st.header("🔬 New Analysis")
    
    # ---- File Upload Section ---- #
    # Initialize mode tracking if not present
    if 'last_upload_mode' not in st.session_state:
        st.session_state['last_upload_mode'] = "Single File"
    
    upload_mode = st.radio(
        "Select input mode:",
        options=["Single File", "Multiple Files", "Entire Folder"],
        horizontal=True,
        key="upload_mode_tab3"
    )
    
    uploaded_files = []
    
    # Check if we should restore from previous upload (same mode + files in session)
    should_restore = False
    if st.session_state.get('last_upload_mode') == upload_mode and st.session_state.get('last_uploaded_files'):
        should_restore = True
    
    if upload_mode == "Single File":
        uploaded_file = st.file_uploader("Upload an ear image", type=["jpg", "jpeg", "png"], key="upload_tab3_single")
        if uploaded_file is not None:
            uploaded_files = [uploaded_file]
            st.session_state['last_uploaded_files'] = uploaded_files
            st.session_state['last_upload_mode'] = "Single File"
            should_restore = False
        elif should_restore:
            # Restore previous single file upload
            uploaded_files = st.session_state['last_uploaded_files']
            st.caption(f"📎 Restoring: {uploaded_files[0].name}")
    
    elif upload_mode == "Multiple Files":
        uploaded_file_list = st.file_uploader(
            "Upload multiple ear images", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            key="upload_tab3_multi"
        )
        if uploaded_file_list:
            uploaded_files = uploaded_file_list
            st.session_state['last_uploaded_files'] = uploaded_files
            st.session_state['last_upload_mode'] = "Multiple Files"
            should_restore = False
        elif should_restore:
            # Restore previous multiple files upload
            uploaded_files = st.session_state['last_uploaded_files']
            st.caption(f"📎 Restoring {len(uploaded_files)} files")
    
    elif upload_mode == "Entire Folder":
        st.info("📁 Enter the folder path containing your ear images")
        folder_path = st.text_input(
            "Folder path (e.g., /path/to/images or ./my_images)",
            key="folder_path_tab3"
        )
        if folder_path and os.path.isdir(folder_path):
            # Collect all image files from folder
            image_extensions = ('.jpg', '.jpeg', '.png')
            folder_images = []
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith(image_extensions):
                    full_path = os.path.join(folder_path, filename)
                    if os.path.isfile(full_path):
                        folder_images.append(full_path)
            
            if folder_images:
                st.success(f"✅ Found {len(folder_images)} image(s) in folder")
                st.write("Images to process:")
                for img_path in folder_images[:10]:
                    st.caption(os.path.basename(img_path))
                if len(folder_images) > 10:
                    st.caption(f"... and {len(folder_images) - 10} more")
                
                # Store folder images as file-like objects for processing
                folder_file_objects = []
                for img_path in folder_images:
                    with open(img_path, 'rb') as f:
                        import io
                        file_obj = io.BytesIO(f.read())
                        file_obj.name = os.path.basename(img_path)
                        folder_file_objects.append(file_obj)
                uploaded_files = folder_file_objects
                st.session_state['last_uploaded_files'] = uploaded_files
                st.session_state['last_upload_mode'] = "Entire Folder"
                should_restore = False
            elif should_restore:
                # Restore previous folder upload
                uploaded_files = st.session_state['last_uploaded_files']
                st.caption(f"📁 Restoring {len(uploaded_files)} files from previous folder")
            else:
                st.warning("No image files found in the selected folder")
        elif folder_path:
            st.error(f"Folder not found: {folder_path}")
    
    uploaded_file = uploaded_files[0] if uploaded_files else None
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded. Ready to run analysis.")
        
        # Display previews of uploaded images
        if len(uploaded_files) == 1:
            # Single file - show detailed preview
            from PIL import Image
            import io
            uploaded_bytes = uploaded_files[0].read()
            uploaded_files[0].seek(0)
            img = Image.open(io.BytesIO(uploaded_bytes)).convert('RGB')
            st.image(img, caption="Original (non-normalized) image", use_container_width=True)
        else:
            # Multiple files - show grid of thumbnails
            from PIL import Image
            import io
            cols = st.columns(min(5, len(uploaded_files)))
            for idx, file_obj in enumerate(uploaded_files[:20]):
                with cols[idx % len(cols)]:
                    try:
                        file_bytes = file_obj.read()
                        file_obj.seek(0)
                        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
                        st.image(img, caption=os.path.basename(file_obj.name), width=150)
                    except Exception as e:
                        st.warning(f"Could not load {file_obj.name}: {e}")
            if len(uploaded_files) > 20:
                st.caption(f"... and {len(uploaded_files) - 20} more files")
        
        # Show previous analyses for this image (current person only) - only for single file
        if len(uploaded_files) == 1:
            try:
                previous = list_image_results(cursor, st.session_state.person_id, uploaded_file.name)
                if previous:
                    latest_pred = previous[0]
                    st.info(
                        f"This image was already analyzed {len(previous)} time(s). Latest: "
                        f"{latest_pred[2]} (conf {float(latest_pred[3]):.2f}) by model {latest_pred[0]} on {latest_pred[5]}"
                    )
                    import pandas as pd
                    cols = [
                        "Model", "Task", "Pred", "Conf", "Log Path", "Timestamp",
                        "NSize", "FGSM", "Normalize", "N_Calibration", "Classif_Loss",
                        "DLoss", "Dist_Fct", "Prototypes", "NPos", "NNeg", "N_Neighbors", "Model_ID"
                    ]
                    df_prev = pd.DataFrame(previous, columns=cols)
                    
                    # Add model numbers (#) if available
                    model_number_map, best_models_table = _ensure_model_number_map(cursor)
                    model_nums = []
                    for _, row in df_prev.iterrows():
                        rd = row.to_dict()
                        # Align keys with _make_model_selection_key expectations
                        key_dict = {
                            "Model Name": rd.get("Model"),
                            "NSize": rd.get("NSize"),
                            "FGSM": rd.get("FGSM"),
                            "Prototypes": rd.get("Prototypes"),
                            "NPos": rd.get("NPos"),
                            "NNeg": rd.get("NNeg"),
                            "DLoss": rd.get("DLoss"),
                            "Dist_Fct": rd.get("Dist_Fct"),
                            "Classif_Loss": rd.get("Classif_Loss"),
                            "N_Calibration": rd.get("N_Calibration"),
                            "Normalize": rd.get("Normalize"),
                            "N_Neighbors": rd.get("N_Neighbors"),
                            "Log Path": rd.get("Log Path")
                        }
                        selection_key = _make_model_selection_key(key_dict)
                        model_num = model_number_map.get(selection_key, "?")
                        if model_num == "?" and best_models_table is not None and not best_models_table.empty:
                            try:
                                match = best_models_table[best_models_table["Log Path"] == rd.get("Log Path")]
                                if not match.empty:
                                    model_num = match.iloc[0].get("#", model_num)
                            except Exception:
                                pass
                        model_nums.append(model_num)
                    df_prev.insert(0, "#", model_nums)
                    st.markdown("**Table:** Previous Analyses for This Image")
                    st.dataframe(df_prev, use_container_width=True)
                else:
                    st.info("No previous analyses found for this image.")
            except Exception as e:
                st.warning(f"Could not load past analyses for this image: {e}")

        # Inference method selector
        st.markdown("---")
        st.subheader("⚙️ Inference Settings")
        infer_method_tab3 = st.selectbox(
            "Inference Method",
            options=['majority_vote', 'prototypes', 'prototype_distance'],
            index=0,
            help="majority_vote: Majority voting of class predictions\nprototypes: Euclidean/cosine distance to prototypes\nprototype_distance: Inverse distance ratio method",
            key="analysis_infer_method"
        )
        
        if infer_method_tab3 == 'prototype_distance':
            dist_metric_tab3 = st.selectbox(
                "Distance Metric",
                options=['euclidean', 'cosine'],
                index=0,
                key="analysis_dist_metric"
            )
        else:
            dist_metric_tab3 = 'euclidean'

        # Speed optimization options
        col_speed1, col_speed2 = st.columns(2)
        with col_speed1:
            skip_validation_tab3 = st.checkbox(
                "⚡ Skip validation metrics (faster)",
                value=False,
                help="Skip loading validation metrics to speed up inference"
            )
        with col_speed2:
            fast_infer_tab3 = st.checkbox(
                "⚡ Fast inference (skip KNN building)",
                value=False,
                help="Use prototype distance directly instead of building KNN"
            )

        run_analysis = st.button("▶️ Run Analysis", key="run_analysis_tab3")
        force_analysis = st.button("🔄 Force New Analysis", key="force_analysis_tab3")
        should_run_analysis = run_analysis or force_analysis
        
        # Show batch processing info
        if len(uploaded_files) > 1:
            st.info(f"📊 Batch mode: {len(uploaded_files)} files will be processed")
    else:
        should_run_analysis = False

# ---- Analysis Logic for Tab3 ---- #
if uploaded_files and should_run_analysis:
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    results_list = []
    batch_start_time = time.time()
    
    for file_idx, file_obj in enumerate(uploaded_files):
        file_start_time = time.time()
        status_text.write(f"Processing {file_idx + 1}/{len(uploaded_files)}: {file_obj.name}")
        progress_bar.progress((file_idx) / len(uploaded_files))
        
        uploaded_bytes = file_obj.read()
        file_obj.seek(0)
        
        try:
            pred_label, pred_confidence, complete_log_path, _ = run_analysis_on_file(
                file_obj.name, uploaded_bytes, args, force_reanalyze=force_analysis,
                show_validation_metrics=not skip_validation_tab3, fast_infer=fast_infer_tab3
            )
            
            # For single file, cache for display
            if len(uploaded_files) == 1:
                st.session_state['last_uploaded_bytes'] = uploaded_bytes
                st.session_state['last_uploaded_name'] = file_obj.name
                st.session_state['last_complete_log_path'] = complete_log_path
                st.session_state['last_base_name'] = strip_extension(file_obj.name.split("/")[-1])
                st.session_state['last_pred_label'] = pred_label
                st.session_state['last_pred_confidence'] = pred_confidence
                st.session_state['last_image_size'] = args.new_size
                st.session_state['last_normalize'] = args.normalize
            
            file_elapsed = time.time() - file_start_time
            results_list.append({
                'filename': file_obj.name,
                'prediction': pred_label,
                'confidence': pred_confidence,
                'log_path': complete_log_path,
                'elapsed': file_elapsed
            })
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                status_text.write(f"⚠️ GPU OOM for {file_obj.name}. Retrying on CPU...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                load_model_and_prototypes.clear()
                st.session_state.pop('knn_cache', None)
                
                args.device = 'cpu'
                
                pred_label, pred_confidence, complete_log_path, _ = run_analysis_on_file(
                    file_obj.name, uploaded_bytes, args, force_reanalyze=force_analysis,
                    show_validation_metrics=not skip_validation_tab3, fast_infer=fast_infer_tab3
                )
                
                if len(uploaded_files) == 1:
                    st.session_state['last_complete_log_path'] = complete_log_path
                    st.session_state['last_base_name'] = strip_extension(file_obj.name.split("/")[-1])
                    st.session_state['last_pred_label'] = pred_label
                    st.session_state['last_pred_confidence'] = pred_confidence
                
                file_elapsed = time.time() - file_start_time
                results_list.append({
                    'filename': file_obj.name,
                    'prediction': pred_label,
                    'confidence': pred_confidence,
                    'log_path': complete_log_path,
                    'elapsed': file_elapsed
                })
            else:
                file_elapsed = time.time() - file_start_time
                results_list.append({
                    'filename': file_obj.name,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'log_path': '',
                    'error': str(e),
                    'elapsed': file_elapsed
                })
    
    progress_bar.progress(1.0)
    batch_elapsed = time.time() - batch_start_time
    status_text.write(f"✅ Batch processing complete! ({batch_elapsed:.2f}s)")
    
    # Store results in session state for persistence
    st.session_state['last_batch_results'] = results_list
    st.session_state['last_batch_elapsed'] = batch_elapsed
    
    # Show results table for batch
    if len(results_list) > 1:
        import pandas as pd
        df_results = pd.DataFrame(results_list)
        st.markdown("---")
        st.subheader("📋 Batch Results")
        st.dataframe(df_results, use_container_width=True)
        st.caption(f"⏱️ Total batch time: {batch_elapsed:.2f}s | Average: {batch_elapsed/len(results_list):.2f}s per file")

# ---- Display persisted batch results ---- #
if st.session_state.get('last_batch_results'):
    import pandas as pd
    st.markdown("---")
    st.subheader("📋 Batch Results (from previous run)")
    df_results = pd.DataFrame(st.session_state['last_batch_results'])
    st.dataframe(df_results, use_container_width=True)
    batch_elapsed = st.session_state.get('last_batch_elapsed', 0)
    results_count = len(st.session_state['last_batch_results'])
    st.caption(f"⏱️ Total batch time: {batch_elapsed:.2f}s | Average: {batch_elapsed/results_count:.2f}s per file")
    if st.button("Clear batch results", key="clear_batch_results"):
        st.session_state['last_batch_results'] = None
        st.rerun()

# ---- Display Logic for Tab3 (persistent across parameter changes) ---- #
if uploaded_file is not None and st.session_state.get('last_uploaded_name') == uploaded_file.name:
    complete_log_path = st.session_state.get('last_complete_log_path')
    base_name = st.session_state.get('last_base_name')
    
    if complete_log_path and base_name:
        st.subheader("🧠 Explanations (on-demand)")
        exp_cols = st.columns(2)
        img_filename = st.session_state.get('last_uploaded_name') or uploaded_file.name
        img_path = f"data/queries/{img_filename.split('/')[-1]}"

        with exp_cols[0]:
            if st.button("Compute SHAP Gradients", key=f"compute_grad_shap_tab3_{base_name}"):
                with st.spinner("Computing SHAP gradients..."):
                    try:
                        random.seed(1)
                        torch.manual_seed(1)
                        np.random.seed(1)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(1)

                        _clear_cached_model()
                        model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
                            load_model_and_prototypes(args)

                        train = TrainAE(args, args.path, load_tb=False, log_metrics=True, keep_models=True,
                                      log_inputs=False, log_plots=True, log_tb=False, log_neptune=True,
                                      log_mlflow=False, groupkfold=args.groupkfold)
                        train.n_batches = len(unique_batches)
                        train.n_cats = len(unique_labels)
                        train.unique_batches = unique_batches
                        train.unique_labels = unique_labels
                        train.epoch = 1
                        train.model = model
                        train.complete_log_path = complete_log_path
                        train.params = {
                            'n_neighbors': args.n_neighbors,
                            'lr': 0,
                            'wd': 0,
                            'smoothing': 0,
                            'is_transform': 0,
                            'valid_dataset': args.valid_dataset
                        }
                        train.set_arcloss()

                        lists, traces = get_empty_traces()
                        loaders = get_images_loaders(data=data,
                                                    random_recs=args.random_recs,
                                                    weighted_sampler=0,
                                                    is_transform=0,
                                                    samples_weights=None,
                                                    epoch=1,
                                                    unique_labels=unique_labels,
                                                    triplet_dloss=args.dloss, bs=args.bs,
                                                    prototypes_to_use=args.prototypes_to_use,
                                                    prototypes=prototypes,
                                                    size=args.new_size,
                                                    normalize=args.normalize)

                        with torch.no_grad():
                            _, best_lists1, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
                            for group in ["train", "valid", "test"]:
                                _, best_lists2, traces, knn = train.predict(group, loaders[group], lists, traces)

                        best_lists = {**best_lists1, **best_lists2}
                        nets = {'cnn': shap_model, 'knn': knn}

                        if not os.path.exists(img_path):
                            st.error(f"Image not found at {img_path}. Please run analysis first.")
                        else:
                            _, image = get_image(img_path, size=image_size, normalize=args.normalize)
                            inputs = {
                                'queries': {"inputs": [image]},
                                'train': {
                                    "inputs": [
                                        torch.concatenate(best_lists['train']['inputs']),
                                        torch.concatenate(best_lists['valid']['inputs'])
                                    ],
                                },
                            }

                            os.makedirs(f'{complete_log_path}/gradients_shap', exist_ok=True)
                            log_shap_gradients_only(
                                nets, i=0, inputs=inputs, group='queries', name=base_name, log_path=complete_log_path,
                                layer=args.shap_layer
                            )
                            st.success("✅ SHAP gradients generated.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error computing SHAP gradients: {e}")

        with exp_cols[1]:
            if st.button("Compute KNN SHAP", key=f"compute_knn_shap_tab3_{base_name}"):
                with st.spinner("Computing KNN SHAP explanations..."):
                    try:
                        random.seed(1)
                        torch.manual_seed(1)
                        np.random.seed(1)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(1)

                        _clear_cached_model()
                        model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
                            load_model_and_prototypes(args)

                        train = TrainAE(args, args.path, load_tb=False, log_metrics=True, keep_models=True,
                                      log_inputs=False, log_plots=True, log_tb=False, log_neptune=True,
                                      log_mlflow=False, groupkfold=args.groupkfold)
                        train.n_batches = len(unique_batches)
                        train.n_cats = len(unique_labels)
                        train.unique_batches = unique_batches
                        train.unique_labels = unique_labels
                        train.epoch = 1
                        train.model = model
                        train.complete_log_path = complete_log_path
                        train.params = {
                            'n_neighbors': args.n_neighbors,
                            'lr': 0,
                            'wd': 0,
                            'smoothing': 0,
                            'is_transform': 0,
                            'valid_dataset': args.valid_dataset
                        }
                        train.set_arcloss()

                        lists, traces = get_empty_traces()
                        loaders = get_images_loaders(data=data,
                                                    random_recs=args.random_recs,
                                                    weighted_sampler=0,
                                                    is_transform=0,
                                                    samples_weights=None,
                                                    epoch=1,
                                                    unique_labels=unique_labels,
                                                    triplet_dloss=args.dloss, bs=args.bs,
                                                    prototypes_to_use=args.prototypes_to_use,
                                                    prototypes=prototypes,
                                                    size=args.new_size,
                                                    normalize=args.normalize)

                        with torch.no_grad():
                            _, best_lists1, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
                            for group in ["train", "valid", "test"]:
                                _, best_lists2, traces, knn = train.predict(group, loaders[group], lists, traces)

                        best_lists = {**best_lists1, **best_lists2}
                        nets = {'cnn': shap_model, 'knn': knn}

                        if not os.path.exists(img_path):
                            st.error(f"Image not found at {img_path}. Please run analysis first.")
                        else:
                            _, image = get_image(img_path, size=image_size, normalize=args.normalize)
                            inputs = {
                                'queries': {"inputs": [image]},
                                'train': {
                                    "inputs": [
                                        torch.concatenate(best_lists['train']['inputs']),
                                        torch.concatenate(best_lists['valid']['inputs'])
                                    ],
                                },
                            }

                            os.makedirs(f'{complete_log_path}/knn_shap', exist_ok=True)
                            log_shap_knn_or_deep(
                                nets, i=0, inputs=inputs, group='queries', name=base_name, log_path=complete_log_path,
                                layer=args.shap_layer
                            )
                            st.success("✅ KNN SHAP explanations generated.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error computing KNN SHAP: {e}")

        st.markdown("---")
        # Display SHAP explanations
        
        # Try to display SHAP gradient explanation
        grad_shap_path = f'{complete_log_path}/gradients_shap/queries_{base_name}_layer{args.shap_layer}.png'
        if os.path.exists(grad_shap_path):
            fig = plt.imread(grad_shap_path)
            st.image(fig, caption=f"SHAP Gradient Explanation (layer {args.shap_layer})", use_container_width=True)
        else:
            st.info("SHAP gradients not computed yet. Use the button above to generate.")

        # Try to display KNN SHAP explanation
        knn_shap_path = f'{complete_log_path}/knn_shap/queries_{base_name}_layer{args.shap_layer}.png'
        if os.path.exists(knn_shap_path):
            fig = plt.imread(knn_shap_path)
            st.image(fig, caption=f"KNN SHAP Gradient Explanation (layer {args.shap_layer})", use_container_width=True)
        else:
            st.info("KNN SHAP (layer) not computed yet. Use the button above to generate.")

        # Try to display KNN SHAP explanation (main)
        knn_shap_path = f'{complete_log_path}/knn_shap/queries_{base_name}.png'
        if os.path.exists(knn_shap_path):
            fig = plt.imread(knn_shap_path)
            st.image(fig, caption="KNN SHAP Gradient Explanation", use_container_width=True)
        else:
            st.info("KNN SHAP not computed yet. Use the button above to generate.")

        # Try to display Grad-CAM
        st.divider()
        st.subheader("💡 Grad-CAM Visualization")

        def rebuild_grad_cam_from_cache(base_name: str, log_dir: str, img_path: str, layer: int, alpha: float,
                                         image_size: int, normalize_flag: bool) -> bool:
            """Re-render overlays from cached heatmaps so alpha tweaks don't recompute Grad-CAM."""

            # Look in the image-specific subdirectory
            image_grad_cam_dir = os.path.join(log_dir, base_name)
            if not os.path.isdir(image_grad_cam_dir):
                return False

            prefix = f"{base_name}_class"
            heatmap_files = [f for f in os.listdir(image_grad_cam_dir) if f.startswith(prefix) and f.endswith("_heatmap.npy")]
            if not heatmap_files:
                return False

            try:
                _, image = get_image(img_path, size=image_size, normalize=normalize_flag)
            except Exception:
                return False

            def _parse_label(fname: str) -> str:
                core = fname[len(prefix):]
                core = core.replace("_heatmap.npy", "")
                core = core.replace(".npy", "")
                return core

            class_labels = sorted(_parse_label(f) for f in heatmap_files)
            class_images = []
            for lbl in class_labels:
                heatmap_path = os.path.join(image_grad_cam_dir, f"{base_name}_class{lbl}_heatmap.npy")
                if not os.path.exists(heatmap_path):
                    continue
                overlay_path = os.path.join(image_grad_cam_dir, f"{base_name}_class{lbl}.png")
                try:
                    heatmap = np.load(heatmap_path)
                    save_overlay_from_heatmap(image, heatmap, overlay_path, alpha=alpha)
                    class_images.append((lbl, plt.imread(overlay_path)))
                except Exception:
                    continue

            if not class_images:
                return False

            fig, axes = plt.subplots(1, len(class_images), figsize=(5 * len(class_images), 5))
            if len(class_images) == 1:
                axes = [axes]
            for ax, (lbl, img) in zip(axes, class_images):
                ax.imshow(img)
                ax.set_title(f"Class: {lbl}")
                ax.axis('off')
            plt.tight_layout()
            montage_path = os.path.join(image_grad_cam_dir, f'{base_name}_grad_cam_all_classes_layer{layer}.png')
            plt.savefig(montage_path, dpi=150, bbox_inches='tight')
            plt.close()
            return True
        
        gc_cols_t3 = st.columns([1, 2])
        with gc_cols_t3[0]:
            layer_input = st.number_input(
                "Layer", value=args.grad_cam_layer, step=1, key=f"gc_layer_t3_{base_name}"
            )
            alpha_input = st.slider(
                "Alpha", 0.0, 1.0, args.grad_cam_alpha, 0.05, key=f"gc_alpha_t3_{base_name}"
            )

        # Action: compute Grad-CAM on demand
        with gc_cols_t3[1]:
            if st.button("🧠 Compute Grad-CAM", key=f"compute_grad_cam_t3_{base_name}"):
                with st.spinner("Computing Grad-CAM..."):
                    try:
                        # Read current layer/alpha from UI state, not from args
                        current_layer = st.session_state.get(f"gc_layer_t3_{base_name}", args.grad_cam_layer)
                        current_alpha = st.session_state.get(f"gc_alpha_t3_{base_name}", args.grad_cam_alpha)

                        # Ensure deterministic behavior
                        random.seed(1)
                        torch.manual_seed(1)
                        np.random.seed(1)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(1)

                        # Load model and prototypes (no need to recompute predictions)
                        model, _, prototypes, image_size, device_str, _, _, _, _ = \
                            load_model_and_prototypes(args)
                        st.session_state['last_image_size'] = image_size
                        st.session_state['last_normalize'] = args.normalize

                        # Prepare single-image input tensor from saved upload
                        img_filename = st.session_state.get('last_uploaded_name') or uploaded_file.name
                        img_path = f"data/queries/{img_filename.split('/')[-1]}"
                        if not os.path.exists(img_path):
                            st.error(f"Image not found at {img_path}. Please run analysis first.")
                        else:
                            _, image = get_image(img_path, size=image_size, normalize=args.normalize)
                            inputs = { 'queries': { 'inputs': [image] } }

                            # Generate Grad-CAM for all classes
                            # Organize by image: create subdirectory per image
                            image_output_dir = os.path.join(complete_log_path, base_name)
                            os.makedirs(image_output_dir, exist_ok=True)
                            log_grad_cam_all_classes(
                                model,
                                0,
                                inputs,
                                'queries',
                                image_output_dir,
                                base_name,
                                prototypes['class']['train'],
                                device=device_str,
                                layer=current_layer,
                                alpha=current_alpha
                            )
                            
                            # Create montage from individual class images
                            class_labels = sorted(prototypes['class']['train'].keys())
                            class_images = []
                            for lbl in class_labels:
                                class_img_path = os.path.join(image_output_dir, f"{base_name}_class{lbl}.png")
                                if os.path.exists(class_img_path):
                                    class_images.append(plt.imread(class_img_path))
                            
                            if class_images:
                                # Create horizontal montage
                                fig, axes = plt.subplots(1, len(class_images), figsize=(5 * len(class_images), 5))
                                if len(class_images) == 1:
                                    axes = [axes]
                                for ax, img, lbl in zip(axes, class_images, class_labels):
                                    ax.imshow(img)
                                    ax.set_title(f"Class: {lbl}")
                                    ax.axis('off')
                                plt.tight_layout()
                                montage_path = os.path.join(image_output_dir, f'{base_name}_grad_cam_all_classes_layer{current_layer}.png')
                                plt.savefig(montage_path, dpi=150, bbox_inches='tight')
                                plt.close()
                            
                            st.success("✅ Grad-CAM generated for all classes.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error computing Grad-CAM: {e}")

        layer_changed = layer_input != args.grad_cam_layer
        alpha_changed = alpha_input != args.grad_cam_alpha
        if layer_changed:
            # Only update displayed layer/alpha; do NOT rerun analysis unless user forces it.
            args.grad_cam_layer = layer_input
            args.grad_cam_alpha = alpha_input
        elif alpha_changed:
            args.grad_cam_alpha = alpha_input
            image_size_cached = st.session_state.get('last_image_size', args.new_size)
            normalize_cached = st.session_state.get('last_normalize', args.normalize)
            rebuilt = rebuild_grad_cam_from_cache(
                base_name,
                complete_log_path,
                img_path,
                layer_input,
                alpha_input,
                image_size_cached,
                normalize_cached,
            )
            if rebuilt:
                st.info("Re-rendered Grad-CAM with new alpha (no recompute).")

        # Display all-classes montage (original + all class Grad-CAMs)
        image_grad_cam_dir = os.path.join(complete_log_path, base_name)
        grad_cam_all_path = os.path.join(image_grad_cam_dir, f'{base_name}_grad_cam_all_classes_layer{layer_input}.png')
        if os.path.exists(grad_cam_all_path):
            fig = plt.imread(grad_cam_all_path)
            st.image(fig, caption=f"Grad-CAM All Classes (Layer {layer_input})", use_container_width=True)
        else:
            st.info(f"Grad-CAM layer {layer_input} not computed for this analysis. Click 'Compute Grad-CAM' button above.")

# Keep connection open for Streamlit session; closing can break callbacks
# conn.close()

# ========================= TAB 4: Ensemble ========================= #
with tab4:
    st.header("🤝 Ensemble Inference (Top-N Models)")

    # Controls
    ens_cols = st.columns(3)
    with ens_cols[0]:
        top_k = st.number_input("# Models to use", min_value=1, max_value=50, value=10, step=1, key="ensemble_top_k")
    with ens_cols[1]:
        ens_device = st.selectbox("Device", ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'], index=0, key="ensemble_device")
    with ens_cols[2]:
        dataset_filter = None
        try:
            # Prefer dataset from sidebar selection
            if args and args.path:
                dataset_filter = os.path.basename(args.path)
        except Exception:
            dataset_filter = None
        st.write(f"Dataset filter: {dataset_filter or 'any'}")

    # Fetch top models (best unique per param combo), filter by dataset
    try:
        try:
            cursor.execute(
                """
                SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
                       n_calibration, accuracy, mcc, normalize, n_neighbors, log_path
                FROM best_models_registry
                ORDER BY mcc DESC
                """
            )
            rows = cursor.fetchall()
        except Exception as e:
            rows = []

        if rows:
            cols = [
                "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path"
            ]
            df_all = pd.DataFrame(rows, columns=cols)

            # Deduplicate per unique param combo keeping best MCC
            group_cols = [
                "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
                "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize", "N_Neighbors"
            ]
            _dedupe_frame = df_all[group_cols].copy().fillna("").astype(str)
            df_all["_dedupe_key"] = _dedupe_frame.agg("|".join, axis=1)
            df_all = df_all.sort_values("MCC", ascending=False)
            df_uniq = df_all.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"]).copy()

            # Filter by dataset (extracted from log_path)
            if dataset_filter:
                def _path_ds(lp: str):
                    try:
                        p = extract_params_from_log_path(lp or "")
                        return p.get("Dataset")
                    except Exception:
                        return None
                df_uniq = df_uniq[df_uniq["Log Path"].apply(lambda lp: _path_ds(lp) == dataset_filter)]

            # Take top-K
            df_top = df_uniq.head(int(top_k)).reset_index(drop=True)

            if len(df_top) == 0:
                st.info("No models found matching the current dataset.")
            else:
                st.write("Using the following models:")
                st.markdown("**Table:** Ensemble Model Lineup")
                st.dataframe(df_top[["Model Name", "NSize", "MCC", "Dist_Fct", "Normalize", "Log Path"]], use_container_width=True)

                # Inference method selector
                infer_method = st.selectbox(
                    "Inference Method",
                    options=['majority_vote', 'prototypes', 'prototype_distance'],
                    index=0,
                    help="majority_vote: Majority voting of class predictions\nprototypes: Euclidean/cosine distance to prototypes\nprototype_distance: Inverse distance ratio method",
                    key="ensemble_infer_method"
                )
                
                # Get best model's distance function as default
                best_model_dist = str(df_top.iloc[0]['Dist_Fct']).lower() if df_top.iloc[0]['Dist_Fct'] is not None else 'euclidean'
                best_model_dist = best_model_dist if best_model_dist in ['euclidean', 'cosine'] else 'euclidean'
                dist_metric_default_idx = 0 if best_model_dist == 'euclidean' else 1
                
                if infer_method == 'prototype_distance':
                    dist_metric = st.selectbox(
                        "Distance Metric",
                        options=['euclidean', 'cosine'],
                        index=dist_metric_default_idx,
                        help=f"Best model uses: {best_model_dist}",
                        key="ensemble_dist_metric"
                    )
                else:
                    dist_metric = 'euclidean'

                # Upload image for ensemble prediction
                uploaded_file_ens = st.file_uploader("Upload an ear image for ensemble", type=["jpg", "jpeg", "png"], key="upload_tab4")
                run_ens = st.button("▶️ Run Ensemble", key="run_ensemble_btn")

                if uploaded_file_ens is not None and run_ens:
                    import io
                    img = Image.open(io.BytesIO(uploaded_file_ens.read())).convert('RGB')

                    from collections import Counter
                    votes = []
                    per_model_rows = []

                    for idx, r in df_top.iterrows():
                        try:
                            log_path = r["Log Path"]
                            model_name = str(r["Model Name"]) if r["Model Name"] is not None else 'resnet18'
                            dist_fct = str(r["Dist_Fct"]).lower() if r["Dist_Fct"] is not None else 'euclidean'
                            normalize_flag = str(r["Normalize"]) if r["Normalize"] is not None else 'no'

                            # Determine image size and dataset from path
                            p = extract_params_from_log_path(log_path or "")
                            try:
                                im_size = int(p.get("new_size", r["NSize"])) if p.get("new_size") is not None else int(r["NSize"]) if r["NSize"] is not None else 224
                            except Exception:
                                im_size = 224

                            # Load model + prototypes (cached)
                            model, class_protos = load_model_for_log_path(log_path, model_name, ens_device)

                            # Preprocess image for this model
                            _, img_tensor = get_image(f"data/queries/{uploaded_file_ens.name}", size=im_size, normalize=normalize_flag)
                            # If file not yet saved in data/queries, process from PIL directly
                            if img_tensor is None or img_tensor.numel() == 0 or not os.path.exists(f"data/queries/{uploaded_file_ens.name}"):
                                # Fallback to on-the-fly preprocessing
                                img_resized = img.resize((im_size, im_size))
                                arr = np.array(img_resized).astype(np.float32) / 255.0
                                arr = arr.transpose(2, 0, 1)
                                img_tensor = torch.tensor(arr).unsqueeze(0)
                                if str(normalize_flag).lower() in ['yes', 'true', '1']:
                                    img_tensor = PerImageNormalize()(img_tensor.squeeze(0)).unsqueeze(0)
                            img_tensor = img_tensor.to(ens_device)

                            with torch.no_grad():
                                out = model(img_tensor)
                                emb = out[0] if isinstance(out, tuple) else out
                            
                            # Use selected inference method
                            if infer_method == 'prototype_distance':
                                pred_lbl = _predict_with_prototype_distance_ratio(emb, class_protos, dist_fct_name=dist_metric)
                            else:  # majority_vote or prototypes
                                pred_lbl = _predict_label_from_prototypes(emb, class_protos, dist_fct_name=dist_fct)
                            
                            votes.append(str(pred_lbl))
                            per_model_rows.append({
                                'Model': model_name,
                                'MCC': r['MCC'],
                                'Size': im_size,
                                'Dist': dist_fct,
                                'Normalize': normalize_flag,
                                'Pred': str(pred_lbl)
                            })
                        except Exception as e:
                            per_model_rows.append({
                                'Model': r.get('Model Name', 'unknown'),
                                'MCC': r.get('MCC'),
                                'Size': r.get('NSize'),
                                'Dist': r.get('Dist_Fct'),
                                'Normalize': r.get('Normalize'),
                                'Pred': f"ERR: {e}"
                            })

                    if votes:
                        cnt = Counter(votes)
                        majority_label, count = cnt.most_common(1)[0]
                        consensus = count / max(1, len(votes))
                        
                        # Store ensemble results in session state
                        if 'ensemble_single_cache' not in st.session_state:
                            st.session_state['ensemble_single_cache'] = {}
                        st.session_state['ensemble_single_cache']['img'] = img
                        st.session_state['ensemble_single_cache']['uploaded_name'] = uploaded_file_ens.name
                        st.session_state['ensemble_single_cache']['df_top'] = df_top
                        st.session_state['ensemble_single_cache']['device'] = ens_device
                        st.session_state['ensemble_single_cache']['per_model_rows'] = per_model_rows
                        st.session_state['ensemble_single_cache']['majority_label'] = majority_label
                        st.session_state['ensemble_single_cache']['consensus'] = consensus
                        st.session_state['ensemble_single_cache']['infer_method'] = infer_method
                        st.session_state['ensemble_single_cache']['dist_metric'] = dist_metric
                        
                        st.success("✅ Ensemble results computed and cached!")
                    else:
                        st.warning("No predictions generated.")
                
                # Render cached ensemble results if available
                if 'ensemble_single_cache' in st.session_state and st.session_state['ensemble_single_cache']:
                    cache = st.session_state['ensemble_single_cache']
                    
                    # Display cached ensemble results
                    if 'majority_label' in cache and 'consensus' in cache:
                        st.markdown("---")
                        st.subheader("Ensemble Result")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Majority Label", cache['majority_label'])
                        with c2:
                            st.metric("Consensus", f"{cache['consensus']:.2f}")

                        st.subheader("Per-model Predictions")
                        st.markdown("**Table:** Ensemble Per-model Predictions")
                        st.dataframe(pd.DataFrame(cache['per_model_rows']), use_container_width=True)
                    
                    # Grad-CAM section
                    img_cached = cache['img']
                    uploaded_name = cache['uploaded_name']
                    df_top_cached = cache['df_top']
                    device_cached = cache['device']
                    per_model_rows_cached = cache['per_model_rows']
                    
                    st.markdown("---")
                    st.subheader("🔥 Grad-CAM for Uploaded Image")
                    
                    # Model selector (default to best model)
                    model_options = [f"{i+1}. {row['Model']} (MCC: {row['MCC']:.3f}, Size: {row['Size']})" 
                                   for i, row in enumerate(per_model_rows_cached)]
                    selected_model_idx = st.selectbox(
                        "Select model to explain",
                        range(len(model_options)),
                        index=0,
                        format_func=lambda x: model_options[x],
                        key="ensemble_single_gradcam_model"
                    )
                    
                    col_layer, col_alpha = st.columns(2)
                    with col_layer:
                        gc_layer = st.number_input("Layer index", min_value=0, max_value=20, value=7, key="ensemble_single_gc_layer")
                    with col_alpha:
                        gc_alpha = st.slider("Alpha (overlay)", 0.0, 1.0, 0.5, key="ensemble_single_gc_alpha")
                    compute_all_max_layer = st.number_input("Max layer for 'Compute All Layers'", min_value=0, max_value=20, value=7, key="ensemble_single_gc_all_max")
                    
                    def _compute_gradcam(layer_value: int):
                        selected_row = df_top_cached.iloc[selected_model_idx]
                        log_path = selected_row["Log Path"]
                        model_name = str(selected_row["Model Name"]) if selected_row["Model Name"] is not None else 'resnet18'
                        normalize_flag = str(selected_row["Normalize"]) if selected_row["Normalize"] is not None else 'no'

                        p = extract_params_from_log_path(log_path or "")
                        try:
                            im_size = int(p.get("new_size", selected_row["NSize"])) if p.get("new_size") is not None else int(selected_row["NSize"]) if selected_row["NSize"] is not None else 224
                        except Exception:
                            im_size = 224

                        model, class_protos = load_model_for_log_path(log_path, model_name, device_cached)

                        temp_img_path = f"data/queries/{uploaded_name}"
                        os.makedirs("data/queries", exist_ok=True)
                        img_resized = img_cached.resize((im_size, im_size))
                        img_resized.save(temp_img_path)

                        _, img_tensor = get_image(temp_img_path, size=im_size, normalize=normalize_flag)
                        if img_tensor is None or img_tensor.numel() == 0:
                            arr = np.array(img_resized).astype(np.float32) / 255.0
                            arr = arr.transpose(2, 0, 1)
                            img_tensor = torch.tensor(arr).unsqueeze(0)
                            if str(normalize_flag).lower() in ['yes', 'true', '1']:
                                img_tensor = PerImageNormalize()(img_tensor.squeeze(0)).unsqueeze(0)
                        img_tensor = img_tensor.to(device_cached)

                        with torch.no_grad():
                            out = model(img_tensor)
                            emb = out[0] if isinstance(out, tuple) else out

                        gc_inputs = {'queries': {'inputs': [img_tensor.cpu()]}}
                        temp_gc_dir = tempfile.mkdtemp()
                        gc_filename = f"grad_cam_{selected_model_idx}_layer{layer_value}"
                        log_grad_cam_all_classes(
                            model,
                            0,
                            gc_inputs,
                            'queries',
                            temp_gc_dir,
                            gc_filename,
                            class_protos,
                            device=device_cached,
                            layer=layer_value,
                            alpha=gc_alpha
                        )

                        montage_files = [f for f in os.listdir(temp_gc_dir) if gc_filename in f and f.endswith('.png')]
                        if not montage_files:
                            raise RuntimeError("Grad-CAM was computed but no output image found")
                        montage_path = os.path.join(temp_gc_dir, montage_files[0])
                        st.session_state['ensemble_single_gc_path'] = montage_path
                        st.session_state['ensemble_single_gc_meta'] = {
                            'model_idx': selected_model_idx,
                            'layer': layer_value,
                            'alpha': gc_alpha,
                        }
                        return montage_path

                    prev_meta = st.session_state.get('ensemble_single_gc_meta', {})
                    need_compute = (
                        'ensemble_single_gc_path' not in st.session_state
                        or prev_meta.get('model_idx') != selected_model_idx
                        or prev_meta.get('layer') != gc_layer
                        or abs(prev_meta.get('alpha', -1.0) - gc_alpha) > 1e-6
                    )

                    if need_compute:
                        with st.spinner("Generating Grad-CAM..."):
                            try:
                                _compute_gradcam(gc_layer)
                                st.success("✅ Grad-CAM computed successfully!")
                            except Exception as e:
                                st.error(f"Error generating Grad-CAM: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                    if st.button("🔁 Compute/Recompute Grad-CAM", key="ensemble_single_compute_gc"):
                        with st.spinner("Generating Grad-CAM..."):
                            try:
                                _compute_gradcam(gc_layer)
                                st.success("✅ Grad-CAM computed successfully!")
                            except Exception as e:
                                st.error(f"Error generating Grad-CAM: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                    if st.button("📚 Compute All Layers", key="ensemble_single_compute_all_gc"):
                        with st.spinner("Computing Grad-CAM for all layers..."):
                            all_paths = []
                            try:
                                for l in range(int(compute_all_max_layer) + 1):
                                    try:
                                        path_l = _compute_gradcam(l)
                                        all_paths.append((l, path_l))
                                    except Exception as inner_e:
                                        st.warning(f"Layer {l}: {inner_e}")
                                st.session_state['ensemble_single_gc_all_paths'] = all_paths
                                st.success("✅ All layers Grad-CAM computed!")
                            except Exception as e:
                                st.error(f"Error computing all layers: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                    # Display cached Grad-CAM if available (without losing results)
                    if 'ensemble_single_gc_path' in st.session_state and st.session_state['ensemble_single_gc_path']:
                        gc_path = st.session_state['ensemble_single_gc_path']
                        if os.path.exists(gc_path):
                            st.markdown("---")
                            st.subheader("✅ Grad-CAM Result")
                            st.image(gc_path, caption=f"Grad-CAM for {model_options[selected_model_idx]}", use_container_width=True)

                    # Display all layers if computed
                    if 'ensemble_single_gc_all_paths' in st.session_state:
                        all_paths = st.session_state['ensemble_single_gc_all_paths']
                        if all_paths:
                            st.markdown("---")
                            st.subheader("✅ Grad-CAM Results - All Layers")
                            for l, pth in all_paths:
                                if os.path.exists(pth):
                                    st.image(pth, caption=f"Layer {l}", use_container_width=True)

                # On-demand: compute ensemble validation metrics over current valid_dataset
                if st.button("📊 Compute Validation Ensemble", key="compute_ensemble_valid"):
                    try:
                        # Load validation set for current dataset selection
                        local_args = argparse.Namespace(**vars(args))
                        data_getter = GetData(local_args.path, local_args.valid_dataset, local_args)
                        data, unique_labels, unique_batches = data_getter.get_variables()
                        label_to_index = {str(lbl): i for i, lbl in enumerate(unique_labels)}
                        pos_label = 'NotNormal' if 'NotNormal' in unique_labels else str(unique_labels[0]) if len(unique_labels) > 0 else None

                        valid_inputs = data['inputs']['valid']
                        valid_labels = data['labels']['valid']
                        if valid_inputs is None or len(valid_inputs) == 0:
                            st.info("Validation set is empty.")
                        else:
                            # Preload top-N models for faster evaluation
                            models_cfg = []
                            for _, r in df_top.iterrows():
                                try:
                                    lp = r["Log Path"]
                                    mn = str(r["Model Name"]) if r["Model Name"] is not None else 'resnet18'
                                    dfct = str(r["Dist_Fct"]).lower() if r["Dist_Fct"] is not None else 'euclidean'
                                    normf = str(r["Normalize"]) if r["Normalize"] is not None else 'no'
                                    p = extract_params_from_log_path(lp or "")
                                    try:
                                        im_size = int(p.get("new_size", r["NSize"])) if p.get("new_size") is not None else int(r["NSize"]) if r["NSize"] is not None else 224
                                    except Exception:
                                        im_size = 224
                                    model, class_protos = load_model_for_log_path(lp, mn, ens_device)
                                    models_cfg.append({
                                        'model': model,
                                        'protos': class_protos,
                                        'size': im_size,
                                        'dist': dfct,
                                        'normalize': normf
                                    })
                                except Exception as e:
                                    st.warning(f"Skipping a model due to load error: {e}")

                            ens_preds = []
                            ens_true = []
                            ens_true_labels = []
                            ens_pred_labels = []
                            ens_prob_pos = []
                            ens_records = []
                            from collections import Counter
                            for i in range(len(valid_inputs)):
                                raw_arr = valid_inputs[i]
                                # Convert HWC float [0,1] to PIL, then resize per-model and run prediction
                                try:
                                    arr_uint8 = (np.clip(raw_arr, 0.0, 1.0) * 255).astype(np.uint8)
                                    pil_img = Image.fromarray(arr_uint8).convert('RGB')
                                except Exception:
                                    # Fallback: construct grayscale if shape unexpected
                                    if raw_arr.ndim == 2:
                                        arr_uint8 = (np.clip(raw_arr, 0.0, 1.0) * 255).astype(np.uint8)
                                        pil_img = Image.fromarray(arr_uint8).convert('RGB')
                                    else:
                                        # Last resort: skip sample
                                        continue

                                votes = []
                                for cfg in models_cfg:
                                    try:
                                        sz = int(cfg['size'])
                                        img_resized = pil_img.resize((sz, sz))
                                        arr = np.array(img_resized, dtype=np.float32) / 255.0
                                        # HWC -> CHW
                                        chw = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0)
                                        if str(cfg['normalize']).lower() in ['yes', 'true', '1']:
                                            chw = PerImageNormalize()(chw.squeeze(0)).unsqueeze(0)
                                        chw = chw.to(ens_device)
                                        with torch.no_grad():
                                            out = cfg['model'](chw)
                                            emb = out[0] if isinstance(out, tuple) else out
                                        pred_lbl = _predict_label_from_prototypes(emb, cfg['protos'], dist_fct_name=cfg['dist'])
                                        votes.append(str(pred_lbl))
                                    except Exception:
                                        continue
                                if votes:
                                    cnt = Counter(votes)
                                    maj_lbl = cnt.most_common(1)[0][0]
                                    true_lbl_str = str(valid_labels[i])
                                    pred_lbl_str = str(maj_lbl)
                                    ens_preds.append(label_to_index.get(pred_lbl_str))
                                    ens_true.append(label_to_index.get(true_lbl_str))
                                    ens_true_labels.append(true_lbl_str)
                                    ens_pred_labels.append(pred_lbl_str)
                                    if pos_label is not None:
                                        ens_prob_pos.append(float(cnt.get(str(pos_label), 0) / len(votes)))
                                    ens_records.append({
                                        'name': data['names']['valid'][i] if i < len(data['names']['valid']) else str(i),
                                        'true_label': true_lbl_str,
                                        'pred_label': pred_lbl_str,
                                        'prob_pos': float(cnt.get(str(pos_label), 0) / len(votes)) if pos_label is not None else None
                                    })

                            # Filter out None indices
                            ens_preds = [p for p in ens_preds if p is not None]
                            ens_true = [t for t in ens_true if t is not None]
                            if len(ens_true) == 0 or len(ens_preds) != len(ens_true):
                                st.warning("Could not compute metrics; missing labels or predictions.")
                            else:
                                acc = float(np.mean(np.array(ens_preds) == np.array(ens_true)))
                                mcc_val = MCC(np.array(ens_true), np.array(ens_preds))
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric("Validation Accuracy (ensemble)", f"{acc:.3f}")
                                with c2:
                                    st.metric("Validation MCC (ensemble)", f"{mcc_val:.3f}")

                                # Calibration curve (binary vs pos_label)
                                if pos_label is not None and len(ens_prob_pos) == len(ens_true_labels):
                                    y_true_bin = np.array([1 if lbl == str(pos_label) else 0 for lbl in ens_true_labels], dtype=np.int32)
                                    y_prob = np.array(ens_prob_pos, dtype=np.float64)
                                    
                                    # Filter out NaN/inf values and clip to [0, 1]
                                    valid_mask = ~(np.isnan(y_prob) | np.isinf(y_prob) | np.isnan(y_true_bin.astype(float)))
                                    y_true_filt = y_true_bin[valid_mask].astype(int)  # Use plain int
                                    y_prob_filt = np.clip(y_prob[valid_mask], 0.0, 1.0).astype(float)  # Use plain float
                                    
                                    # Ensure binary labels and continuous probabilities
                                    uniq_vals = np.unique(y_true_filt)
                                    uniq_probs = np.unique(y_prob_filt)
                                    # Only compute calibration if: 2 binary classes AND probabilities are truly continuous (many unique values)
                                    if len(uniq_vals) == 2 and set(uniq_vals) == {0, 1} and len(uniq_probs) > 3 and len(y_true_filt) > 0:
                                        try:
                                            prob_true, prob_pred = calibration_curve(y_true_filt, y_prob_filt, n_bins=10)
                                            fig, ax = plt.subplots(figsize=(5, 4))
                                            ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f"Ensemble (pos={pos_label})")
                                            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect")
                                            ax.set_xlabel("Mean predicted probability")
                                            ax.set_ylabel("Fraction of positives")
                                            ax.set_title("Validation Calibration (Ensemble)")
                                            ax.legend()
                                            ax.grid(alpha=0.3)
                                            st.pyplot(fig)
                                        except Exception as e:
                                            prob_min, prob_max = y_prob_filt.min(), y_prob_filt.max()
                                            st.warning(f"Calibration curve unavailable for labels {list(uniq_vals)} (pos={pos_label}): {e}\n\nDiagnostics: dtypes y_true={y_true_filt.dtype}, y_prob={y_prob_filt.dtype}, unique_vals={set(uniq_vals)}, prob range [{prob_min:.4f}, {prob_max:.4f}], n_samples={len(y_true_filt)}")
                                    else:
                                        st.warning(f"Calibration curve requires exactly 2 classes [0, 1], but found unique values {set(uniq_vals)}")


                                # Missed samples table
                                mis_rows = [r for r in ens_records if r['true_label'] != r['pred_label']]
                                # Cache results + model configs to persist across reruns
                                st.session_state['ensemble_valid_cache'] = {
                                    'mis_rows': mis_rows,
                                    'dataset_path': local_args.path,
                                    'models_cfg': models_cfg,
                                    'df_top': df_top,
                                    'ens_device': ens_device
                                }
                                if mis_rows:
                                    st.subheader("Misclassified Validation Samples")
                                    st.markdown("**Table:** Validation Misclassifications")
                                    st.dataframe(pd.DataFrame(mis_rows), use_container_width=True)

                                    # Dropdown to view individual images
                                    names_sorted = [r['name'] for r in mis_rows]
                                    selected_name = st.selectbox("View misclassified image", names_sorted, key="ensemble_missed_select")
                                    if selected_name:
                                        img_path = os.path.join(local_args.path, selected_name)
                                        if os.path.exists(img_path):
                                            try:
                                                true_lbl_sel = next((r['true_label'] for r in mis_rows if r['name'] == selected_name), '?')
                                                pred_lbl_sel = next((r['pred_label'] for r in mis_rows if r['name'] == selected_name), '?')
                                                st.image(Image.open(img_path).convert('RGB'), caption=f"{selected_name} (true: {true_lbl_sel}, pred: {pred_lbl_sel})", use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"Could not open image {selected_name}: {e}")
                                        else:
                                            st.info(f"Image file not found at {img_path}")
                                else:
                                    st.info("No misclassified samples in validation set.")
                    except Exception as e:
                        st.error(f"Error computing validation ensemble: {e}")

                # Re-render last cached misclassifications on rerun (e.g., after dropdown change)
                cache = st.session_state.get('ensemble_valid_cache')
                if cache and cache.get('mis_rows'):
                    mis_rows = cache['mis_rows']
                    st.subheader("Misclassified Validation Samples (cached)")
                    st.markdown("**Table:** Validation Misclassifications (Cached)")
                    st.dataframe(pd.DataFrame(mis_rows), use_container_width=True)
                    names_sorted = [r['name'] for r in mis_rows]
                    selected_name = st.selectbox("View misclassified image", names_sorted, key="ensemble_missed_select_cached")
                    if selected_name:
                        img_path = os.path.join(cache.get('dataset_path', ''), selected_name)
                        if os.path.exists(img_path):
                            try:
                                true_lbl_sel = next((r['true_label'] for r in mis_rows if r['name'] == selected_name), '?')
                                pred_lbl_sel = next((r['pred_label'] for r in mis_rows if r['name'] == selected_name), '?')
                                st.image(Image.open(img_path).convert('RGB'), caption=f"{selected_name} (true: {true_lbl_sel}, pred: {pred_lbl_sel})", use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not open image {selected_name}: {e}")
                        else:
                            st.info(f"Image file not found at {img_path}")
                    
                    # Grad-CAM section for misclassified image
                    st.divider()
                    st.subheader("💡 Grad-CAM for Misclassified Image")
                    
                    # Select model for Grad-CAM
                    models_cfg = cache.get('models_cfg', [])
                    df_top_cache = cache.get('df_top')
                    ens_device_cache = cache.get('ens_device', 'cpu')
                    
                    if models_cfg and df_top_cache is not None and len(df_top_cache) > 0:
                        model_options = [f"{i+1}. {r['Model Name']} (MCC: {r['MCC']:.3f})" for i, r in enumerate(df_top_cache.itertuples())]
                        selected_model_idx = st.selectbox("Select model for Grad-CAM", range(len(model_options)), format_func=lambda i: model_options[i], key="ensemble_gradcam_model")
                        
                        gc_layer = st.number_input("Grad-CAM Layer", value=7, step=1, key="ensemble_gc_layer")
                        gc_alpha = st.slider("Grad-CAM Alpha", 0.0, 1.0, 0.55, 0.05, key="ensemble_gc_alpha")
                        
                        if st.button("🧠 Compute Grad-CAM", key="ensemble_compute_gradcam"):
                            with st.spinner("Computing Grad-CAM..."):
                                try:
                                    import gc as gc_module
                                    random.seed(1)
                                    torch.manual_seed(1)
                                    np.random.seed(1)
                                    
                                    # Get selected model config and corresponding model row
                                    model_cfg = models_cfg[selected_model_idx]
                                    model_row = df_top_cache.iloc[selected_model_idx]
                                    
                                    # Use cached model from config (already loaded)
                                    model = model_cfg['model']
                                    class_protos = model_cfg['protos']
                                    
                                    # Load the image
                                    img_path = os.path.join(cache.get('dataset_path', ''), selected_name)
                                    if os.path.exists(img_path):
                                        pil_img = Image.open(img_path).convert('RGB')
                                        im_size = model_cfg['size']
                                        pil_img_resized = pil_img.resize((im_size, im_size))
                                        arr = np.array(pil_img_resized, dtype=np.float32) / 255.0
                                        chw = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0)
                                        if str(model_cfg['normalize']).lower() in ['yes', 'true', '1']:
                                            chw = PerImageNormalize()(chw.squeeze(0)).unsqueeze(0)
                                        
                                        inputs = {'queries': {'inputs': [chw]}}
                                        
                                        # Compute Grad-CAM for all classes
                                        # Organize by image: create subdirectory per image
                                        base_name = os.path.splitext(selected_name)[0]
                                        output_base_dir = os.path.join(cache.get('dataset_path', ''), f"ensemble_gradcam_{selected_model_idx}")
                                        output_dir = os.path.join(output_base_dir, base_name)
                                        os.makedirs(output_dir, exist_ok=True)
                                        
                                        log_grad_cam_all_classes(
                                            model, 0, inputs, 'queries', output_dir, base_name,
                                            class_protos, device=ens_device_cache, layer=int(gc_layer), alpha=gc_alpha
                                        )
                                        st.success(f"✅ Grad-CAM computed for layer {gc_layer}")
                                        st.session_state['ensemble_gc_path'] = os.path.join(output_dir, f"{base_name}_grad_cam_all_classes_layer{gc_layer}.png")
                                        st.rerun()
                                    else:
                                        st.error(f"Image not found: {img_path}")
                                except Exception as e:
                                    st.error(f"Error computing Grad-CAM: {e}")
                        
                        # Display cached Grad-CAM if available
                        gc_path = st.session_state.get('ensemble_gc_path')
                        if gc_path and os.path.exists(gc_path):
                            try:
                                fig = plt.imread(gc_path)
                                st.image(fig, caption=f"Grad-CAM Layer {gc_layer} - {model_options[selected_model_idx]}", use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not display Grad-CAM: {e}")
                    else:
                        st.info("Run validation ensemble first to enable Grad-CAM analysis.")
        else:
            st.info("No models found in registry.")
    except Exception as e:
        st.error(f"Ensemble tab error: {e}")
