import os
import json
import tempfile
import glob
import time
import gc
import traceback
import re
from tqdm import tqdm
from otitenet.app.ui_helpers import plot_knn_mcc_curves, plot_prototype_mcc_curves

# Disable Streamlit file watcher early to avoid importing/inspecting heavy packages
# (prevents Streamlit from touching packages like torch._classes which can raise)
os.environ.setdefault("STREAMLIT_FILE_WATCHER_TYPE", "none")

import streamlit as st
st.set_page_config(layout="wide")
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
from otitenet.utils.prototypes import Prototypes
from otitenet.utils.encoding_utils import (
    get_base_transform, get_knn_augmentation_transform,
    encode_split_with_augmentation, compute_prototypes_by_strategy, flatten_prototype_dict
)
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
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score, matthews_corrcoef, roc_auc_score
from otitenet.ml import (
    find_best_classifier,
    fit_knn_classifier,
    evaluate_knn_with_k_search,
    fit_baseline_classifiers,
    optimize_prototype_components,
)
from otitenet.ml.evaluation import evaluate_baseline_classifiers
from sklearn.manifold import TSNE
from datetime import datetime
try:
    from umap import UMAP
except ImportError:
    UMAP = None
import seaborn as sns
import joblib
from otitenet.utils.update_model_ranks import update_model_ranks
from otitenet.utils.kde import make_kde_classifier
from otitenet.app.utils import (
    strip_extension,
    ensure_int,
    get_model_params_path,
    extract_params_from_log_path,
    build_params_from_args,
    get_calibration_metrics,
    _make_model_selection_key,
    _lookup_model_number,
    _ensure_model_number_map,
    _unique_preserve_order,
    get_split_mcc_metrics,
    format_classifier_config,
    enumerate_classification_heads,
    resolve_best_classifier_config,
    set_random_seeds,
)
from otitenet.app.database import (
    create_db,
    ensure_results_model_id,
    ensure_best_models_registry_nsize,
    ensure_production_model_table,
    get_production_model,
    set_production_model,
    check_ds_exists,
    list_image_results,
    fetch_model_by_log_path,
    resolve_model_id,
    insert_score,
)
from otitenet.app.image_processing import get_image, preprocess_image
from otitenet.app.model_loading import (
    resolve_model_paths,
    load_saved_search_params,
    load_model_parameters,
    load_model_and_prototypes,
    load_model_for_log_path,
    clear_cached_model,
)
from otitenet.app.inference import (
    predict_label_from_prototypes as _predict_label_from_prototypes,
    predict_with_prototype_distance_ratio as _predict_with_prototype_distance_ratio,
    predict_with_kde as _predict_with_kde,
)
from otitenet.app.ui_helpers import choose_dataset
from otitenet.app.args import get_args, build_args_from_sidebar
from otitenet.app.analysis import get_or_build_knn, run_analysis_on_file

# Your model imports
import argparse
import random

# Set random seeds for reproducibility (must match training)
set_random_seeds(1)

# ---- Load datasets from /data ---- #
data_dir = './data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _arrow_safe_dataframe(df):
    """Return a Streamlit/Arrow-friendly copy of a dataframe.

    PyArrow is strict when an object column mixes strings and ints, which can
    happen for display columns such as '#'. For UI tables, normalize object-like
    columns to strings and keep numeric columns numeric.
    """
    if df is None:
        return df
    out = df.copy()
    for col in out.columns:
        try:
            if str(col) == "#":
                numeric = pd.to_numeric(out[col], errors="coerce")
                if numeric.notna().all():
                    out[col] = numeric.astype("Int64")
                else:
                    out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
                continue
            if out[col].dtype == "object":
                out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
        except Exception:
            out[col] = out[col].astype(str)
    return out


def _production_head_config(prod, fallback_args=None):
    """Return the classification head stored with the production model, or resolve it."""
    if not prod:
        return None
    for key in ["Head Config", "head_config", "classification_head_config", "best_classifier_config"]:
        val = prod.get(key) if isinstance(prod, dict) else None
        if val is not None and str(val).strip() not in {"", "nan", "None"}:
            return str(val)
    if fallback_args is not None:
        try:
            return str(resolve_best_classifier_config(fallback_args, use_optimized=True))
        except Exception:
            return None
    return None


def _production_head_label(prod, fallback_args=None):
    """Return a readable production head label."""
    if not prod:
        return "—"
    for key in ["Head", "head_name", "learned_classifier_label", "classification_head_name"]:
        val = prod.get(key) if isinstance(prod, dict) else None
        if val is not None and str(val).strip() not in {"", "nan", "None"}:
            return str(val)
    cfg = _production_head_config(prod, fallback_args=fallback_args)
    return format_classifier_config(cfg) if cfg else "—"

# ---- Inference ground truth and head-specific metrics helpers ---- #
def _diagnostic_to_binary_label_global(value):
    """Map Sheet3 diagnostic text to app labels: Normal / NotNormal.

    Excel rule for inference ground truth:
      - exactly T. G normal or T. D normal => Normal
      - any other non-empty diagnostic text => NotNormal

    This is intentionally strict because strings like
    "T. D ANORMAL - otite moyenne aigue" contain the substring
    "normal" inside "anormal" and must still be NotNormal.
    """
    import unicodedata
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    simplified = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    simplified = " ".join(simplified.replace("\xa0", " ").split())
    normal_values = {"t. g normal", "t.g normal", "tg normal", "t. d normal", "t.d normal", "td normal"}
    if simplified in normal_values:
        return "Normal"
    return "NotNormal"


def _read_xlsx_sheet_values_global(xlsx_path: str, sheet_name: str = "Sheet3"):
    """Read an xlsx sheet with the stdlib so we do not depend on pandas/openpyxl versions."""
    import zipfile
    import xml.etree.ElementTree as ET
    import posixpath
    import re

    if not xlsx_path or not os.path.exists(xlsx_path):
        return []

    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    def _xlsx_col_to_idx(cell_ref: str) -> int:
        letters = "".join(ch for ch in str(cell_ref) if ch.isalpha()).upper()
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return max(0, idx - 1)

    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.findall(".//main:si", ns):
                texts = [t.text or "" for t in si.findall(".//main:t", ns)]
                shared_strings.append("".join(texts))

        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib.get("Id"): rel.attrib.get("Target") for rel in rel_root.findall("pkgrel:Relationship", ns)}
        target = None
        for sheet in wb_root.findall(".//main:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                rid = sheet.attrib.get("{%s}id" % ns["rel"])
                target = rel_targets.get(rid)
                break
        if target is None:
            return []

        sheet_path = posixpath.normpath(posixpath.join("xl", target))
        if sheet_path.startswith("/"):
            sheet_path = sheet_path[1:]
        sheet_root = ET.fromstring(zf.read(sheet_path))
        rows_out = []
        for row in sheet_root.findall(".//main:sheetData/main:row", ns):
            vals = []
            for cell in row.findall("main:c", ns):
                col_idx = _xlsx_col_to_idx(cell.attrib.get("r", "A1"))
                while len(vals) <= col_idx:
                    vals.append(None)
                cell_type = cell.attrib.get("t")
                value_el = cell.find("main:v", ns)
                inline_el = cell.find("main:is/main:t", ns)
                if cell_type == "s" and value_el is not None:
                    value = shared_strings[int(value_el.text)]
                elif cell_type == "inlineStr" and inline_el is not None:
                    value = inline_el.text or ""
                elif value_el is not None:
                    value = value_el.text
                else:
                    value = None
                vals[col_idx] = value
            rows_out.append(vals)
        return rows_out


@st.cache_data(show_spinner=False)
def _load_inference_ground_truth_map_global(xlsx_path: str, mtime: float):
    rows = _read_xlsx_sheet_values_global(xlsx_path, "Sheet3")
    if not rows:
        return {}
    headers = [str(v).strip() if v is not None else "" for v in rows[0]]

    def _find_col(contains_all):
        for i, header in enumerate(headers):
            low = header.lower()
            if all(tok.lower() in low for tok in contains_all):
                return i
        return None

    def _find_diag_col(side):
        # Avoid matching the "d" in "diagnostic" when looking for the D column.
        import re
        target = str(side).strip().lower()
        for i, header in enumerate(headers):
            low = " ".join(str(header).lower().replace("_", " ").split())
            if "diagnostic" not in low:
                continue
            if re.search(rf"(?:^|\b)tympan\s+{target}(?:\b|$)", low):
                return i
            if re.search(rf"(?:^|\b){target}(?:\b|$)", low.replace("diagnostic", " ")):
                return i
        return None

    id_col = _find_col(["id"])
    diag_g_col = _find_diag_col("g")
    diag_d_col = _find_diag_col("d")
    if id_col is None:
        return {}

    import re
    truth_map = {}
    for row in rows[1:]:
        if id_col >= len(row) or row[id_col] in (None, ""):
            continue
        match = re.search(r"(\d{5})", str(row[id_col]))
        if not match:
            continue
        img_id = match.group(1)
        if diag_g_col is not None and diag_g_col < len(row):
            truth_map[(img_id, "G")] = _diagnostic_to_binary_label_global(row[diag_g_col])
        if diag_d_col is not None and diag_d_col < len(row):
            truth_map[(img_id, "D")] = _diagnostic_to_binary_label_global(row[diag_d_col])
    return truth_map


def _inference_ground_truth_global(filename: str, inference_dir: str = "data/datasets/inference"):
    import re
    base = os.path.basename(str(filename))
    id_match = re.search(r"(\d{5})", base)
    side_match = re.search(r"(?:^|[-_])(G|D)(?:[-_.]|$)", base, flags=re.IGNORECASE)
    if not id_match or not side_match:
        return "Unknown"
    xlsx_path = os.path.join(inference_dir, "LOG_Entrainement_IA.xlsx")
    if not os.path.exists(xlsx_path):
        return "Unknown"
    truth_map = _load_inference_ground_truth_map_global(xlsx_path, os.path.getmtime(xlsx_path))
    return truth_map.get((id_match.group(1), side_match.group(1).upper()), "Unknown")


def _binary_label_global(value):
    value_l = str(value).strip().lower()
    if value_l == "notnormal":
        return 1
    if value_l == "normal":
        return 0
    return None


def _notnormal_score_from_prediction_global(pred_label, confidence):
    try:
        conf = float(confidence)
    except Exception:
        return np.nan
    pred_bin = _binary_label_global(pred_label)
    if pred_bin is None:
        return np.nan
    return conf if pred_bin == 1 else 1.0 - conf


def _metrics_from_binary_predictions_global(y_true_labels, y_pred_labels, scores=None):
    y_true_bin = np.array([_binary_label_global(v) for v in y_true_labels], dtype=object)
    y_pred_bin = np.array([_binary_label_global(v) for v in y_pred_labels], dtype=object)
    valid = np.array([(yt is not None and yp is not None) for yt, yp in zip(y_true_bin, y_pred_bin)])
    if valid.size == 0 or not valid.any():
        return {"inf_acc": np.nan, "inf_mcc": np.nan, "inf_AUC": np.nan, "N": 0}
    yt = y_true_bin[valid].astype(int)
    yp = y_pred_bin[valid].astype(int)
    out = {
        "inf_acc": float(accuracy_score(yt, yp)),
        "inf_mcc": float(matthews_corrcoef(yt, yp)) if len(np.unique(yt)) > 1 and len(np.unique(yp)) > 1 else 0.0,
        "inf_AUC": np.nan,
        "N": int(len(yt)),
    }
    if scores is not None:
        try:
            sc = np.asarray(scores, dtype=float)[valid]
            finite = np.isfinite(sc)
            if finite.any() and len(np.unique(yt[finite])) > 1:
                out["inf_AUC"] = float(roc_auc_score(yt[finite], sc[finite]))
        except Exception:
            pass
    return out


def _head_config_label_global(head_config):
    try:
        return format_classifier_config(head_config)
    except Exception:
        return str(head_config) if head_config is not None else "—"


def _best_head_config_for_args_global(_args):
    try:
        return str(resolve_best_classifier_config(_args, use_optimized=True))
    except Exception:
        # For per-row tables, do not fall back to a global selected head stored
        # on args.best_classifier_config, because that makes every model display
        # the same head. If no optimized head can be resolved for this row, use
        # that row's own n_neighbors as the KNN fallback.
        return str(getattr(_args, "n_neighbors", 1))


def _set_classifier_head_on_args_global(_args, head_config):
    _args.best_classifier_config = str(head_config)
    _args.learned_classifier_label = _head_config_label_global(head_config)
    try:
        if str(head_config).isdigit():
            _args.n_neighbors = int(head_config)
    except Exception:
        pass
    return _args


def _best_head_entry_for_args_global(_args):
    """Return the best cached classification-head entry for one model args object."""
    try:
        heads = enumerate_classification_heads(_args)
    except Exception:
        heads = []
    if heads:
        return heads[0]
    fallback = str(getattr(_args, "n_neighbors", 1))
    return {
        "config": fallback,
        "label": _head_config_label_global(fallback),
        "mcc": np.nan,
        "details": "fallback from N_Neighbors",
    }


def _classification_head_options_for_args_global(_args):
    """Return classification-head options for a model, including a KNN fallback."""
    try:
        heads = list(enumerate_classification_heads(_args) or [])
    except Exception:
        heads = []
    fallback = str(getattr(_args, "n_neighbors", 1))
    if fallback and fallback not in {str(h.get("config")) for h in heads}:
        heads.append({
            "config": fallback,
            "label": _head_config_label_global(fallback),
            "mcc": np.nan,
            "details": "fallback from N_Neighbors",
        })
    return heads


def _compute_batch_effect_from_predictions(preds, batch_labels):
    preds = np.asarray(preds)
    batch_labels = np.asarray(batch_labels)
    if preds.shape[0] == 0 or batch_labels.shape[0] == 0 or preds.shape[0] != batch_labels.shape[0]:
        return {
            'batch_entropy_norm': np.nan,
            'batch_nmi': np.nan,
            'batch_ari': np.nan,
        }

    unique_batches = np.unique(batch_labels)
    unique_preds = np.unique(preds)

    # Early exit only if we have <= 1 batch (nothing to measure mix across)
    if unique_batches.size <= 1:
        return {
            'batch_entropy_norm': np.nan,
            'batch_nmi': np.nan,
            'batch_ari': np.nan,
        }

    contingency = np.zeros((unique_preds.size, unique_batches.size), dtype=float)
    for i, pred_val in enumerate(unique_preds):
        pred_mask = preds == pred_val
        for j, batch_val in enumerate(unique_batches):
            contingency[i, j] = np.sum(pred_mask & (batch_labels == batch_val))

    row_sums = contingency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = contingency / row_sums
    ent = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    max_ent = np.log(unique_batches.size)
    entropy_norm = float(np.mean(ent / (max_ent + 1e-12)))

    # Handle degenerate single-cluster prediction case explicitly
    if unique_preds.size <= 1:
        nmi = 0.0
        ari = 0.0
    else:
        nmi = float(normalized_mutual_info_score(batch_labels, preds))
        ari = float(adjusted_rand_score(batch_labels, preds))

    return {
        'batch_entropy_norm': entropy_norm,
        'batch_nmi': nmi,
        'batch_ari': ari,
    }

# ---- Classifier Display Names and Colors ---- #
BASELINE_DISPLAY_NAMES = {
    'logreg': 'Logistic Regression',
    'ridge': 'Ridge Classifier',
    'naive_bayes': 'Naive Bayes',
    'linear_svc': 'Linear SVC',
    'rbf_svc': 'RBF SVC',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'decision_tree': 'Decision Tree',
    'lda': 'Linear Discriminant',
    'qda': 'Quadratic Discriminant'
}

BASELINE_DISPLAY_SHORT = {
    'logreg': 'LogReg',
    'ridge': 'Ridge',
    'naive_bayes': 'NaiveBayes',
    'linear_svc': 'LinearSVC',
    'rbf_svc': 'RBF_SVC',
    'random_forest': 'RandForest',
    'gradient_boosting': 'GradBoost',
    'decision_tree': 'DecTree',
    'lda': 'LDA',
    'qda': 'QDA'
}

CLASSIFIER_COLORS = {
    'knn': '#e74c3c',
    'mean': '#3498db',
    'kmeans': '#f39c12',
    'gmm': '#2ecc71',
    'logreg': '#9b59b6',
    'ridge': '#e8daef',
    'naive_bayes': '#1abc9c',
    'linear_svc': '#e67e22',
    'rbf_svc': '#d35400',
    'random_forest': '#16a085',
    'gradient_boosting': '#27ae60',
    'decision_tree': '#2ecc71',
    'lda': '#3498db',
    'qda': '#5dade2'
}


# ---- Load Model and Prototypes ---- #


# ---- Fast KNN Cache for Inference ---- #
# (Imported from otitenet.app.analysis module)


# ---- Streamlit UI ---- #

conn, cursor = create_db()
ensure_results_model_id(conn, cursor)
ensure_best_models_registry_nsize(conn, cursor)
ensure_production_model_table(conn, cursor)

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

# Admin check
ADMIN_EMAIL = "simonjpelletier@gmail.com"
is_admin = st.session_state.user_email == ADMIN_EMAIL

# Load production model from database (applies to all users)
production_model_db = get_production_model(cursor)
if production_model_db:
    st.session_state['production_model'] = production_model_db

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

with st.sidebar.expander("🧹 Clear a table"):
    try:
        cursor.execute("SHOW TABLES")
        _all_tables = [row[0] for row in cursor.fetchall()]
        _clearable_tables = [t for t in _all_tables if t != "best_models_registry"]

        if not _clearable_tables:
            st.info("No clearable tables found.")
        else:
            selected_table_to_clear = st.selectbox(
                "Select table to clear",
                options=sorted(_clearable_tables),
                key="clear_table_selectbox",
            )
            st.caption("`best_models_registry` is protected and cannot be cleared from this option.")
            
            clear_action = st.radio(
                "Select action",
                options=["Clear data (TRUNCATE)", "Delete table (DROP)"],
                key="clear_action_radio",
                help="TRUNCATE removes all rows but keeps table structure. DROP completely removes the table from database."
            )

            if st.button("Proceed", key="clear_table_btn"):
                st.session_state['pending_clear_table'] = selected_table_to_clear
                st.session_state['pending_clear_action'] = clear_action

            pending_table = st.session_state.get('pending_clear_table')
            pending_action = st.session_state.get('pending_clear_action', 'Clear data (TRUNCATE)')
            if pending_table:
                action_desc = "clear all data from" if "TRUNCATE" in pending_action else "completely delete"
                st.warning(f"⚠️ Are you sure you want to {action_desc} table `{pending_table}`?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, proceed", key="confirm_clear_table_btn"):
                        try:
                            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                            if "DROP" in pending_action:
                                cursor.execute(f"DROP TABLE `{pending_table}`")
                                success_msg = f"✅ Deleted table: {pending_table}"
                            else:
                                cursor.execute(f"TRUNCATE TABLE `{pending_table}`")
                                success_msg = f"✅ Cleared table: {pending_table}"
                            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
                            conn.commit()
                            st.success(success_msg)
                            st.session_state.pop('pending_clear_table', None)
                            st.session_state.pop('pending_clear_action', None)
                            st.rerun()
                        except Error as e:
                            try:
                                cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
                            except Exception:
                                pass
                            st.error(f"❌ Could not {action_desc} table {pending_table}: {e}")
                with c2:
                    if st.button("Cancel", key="cancel_clear_table_btn"):
                        st.session_state.pop('pending_clear_table', None)
                        st.session_state.pop('pending_clear_action', None)
                        st.info("Operation canceled.")
    except Error as e:
        st.error(f"❌ Could not load table list: {e}")

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

# ---- Display Current Optimization Selection in Sidebar ---- #
st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Current Optimization:**")
if 'k_opt_current_selection' in st.session_state:
    current_selection = st.session_state['k_opt_current_selection']
    if is_admin:
        # Admin: Show only model number
        if current_selection.get('type') == 'knn':
            st.sidebar.success(f"🎯 Model #{st.session_state.get('selected_model_version', 0)}")
        else:
            st.sidebar.success(f"🎯 Model #{st.session_state.get('selected_model_version', 0)}")
    else:
        # Client: Show full details (read-only)
        if current_selection.get('type') == 'knn':
            st.sidebar.success(f"✅ KNN with k={current_selection.get('k')}\n\nMCC: {current_selection.get('mcc', '?'):.4f}")
        else:
            strategy = current_selection.get('strategy', '?').upper()
            n_comp = current_selection.get('n_comp', '?')
            st.sidebar.success(f"✅ {strategy} Strategy\nn_comp={n_comp}\n\nMCC: {current_selection.get('mcc', '?'):.4f}")
else:
    st.sidebar.info("No optimization applied yet.\nRun 'Optimize k' in Tab 1.")

# ---- Load sidebar parameters BEFORE tabs (always visible) ---- #
if is_admin:
    args = build_args_from_sidebar(cursor, conn, is_admin, data_dir)
else:
    # For non-admin, start from minimal args, then replace them with the admin-selected production model.
    args = build_args_from_sidebar(cursor, conn, is_admin, data_dir)


def _apply_production_model_to_args(_args, production_model, data_dir='./data'):
    """Use the admin-selected production model for client-side inference."""
    if not production_model:
        return _args
    try:
        prod = dict(production_model)
        prod.update(extract_params_from_log_path(prod.get('Log Path') or prod.get('log_path') or ''))
        dataset = prod.get('Dataset') or prod.get('path') or 'otite_ds_64'
        _args.path = dataset if isinstance(dataset, str) and dataset.startswith('data/') else os.path.join(data_dir, str(dataset))
        _args.task = str(prod.get('Task') or prod.get('task') or getattr(_args, 'task', 'otite'))
        _args.model_name = prod.get('Model Name') or prod.get('model_name') or getattr(_args, 'model_name', None)
        _args.new_size = ensure_int(prod.get('NSize') or prod.get('new_size') or getattr(_args, 'new_size', 64))
        _args.fgsm = str(prod.get('FGSM') or prod.get('fgsm') or '0')
        _args.n_calibration = str(prod.get('N_Calibration') or prod.get('n_calibration') or '0')
        _args.classif_loss = str(prod.get('Classif_Loss') or prod.get('classif_loss') or 'triplet')
        _args.dloss = str(prod.get('DLoss') or prod.get('dloss') or 'triplet')
        _args.prototypes_to_use = str(prod.get('Prototypes') or prod.get('prototypes_to_use') or 'class')
        _args.n_positives = str(prod.get('NPos') or prod.get('n_positives') or '1')
        _args.n_negatives = str(prod.get('NNeg') or prod.get('n_negatives') or '1')
        _args.normalize = str(prod.get('Normalize') or prod.get('normalize') or 'no')
        _args.dist_fct = str(prod.get('Dist_Fct') or prod.get('dist_fct') or 'euclidean').strip().lower()
        _args.n_neighbors = int(float(prod.get('N_Neighbors') or prod.get('n_neighbors') or 1))
        _args.model_id = prod.get('Model ID') or prod.get('model_id')
        _args.prototype_strategy = str(prod.get('Proto_Strat') or prod.get('prototype_strategy') or 'mean')
        try:
            _args.prototype_components = int(prod.get('Proto_Comp') or prod.get('prototype_components') or 1)
        except Exception:
            _args.prototype_components = 1
        head_config = _production_head_config(prod, fallback_args=_args)
        if not head_config:
            head_config = str(resolve_best_classifier_config(_args, use_optimized=True))
        _args.best_classifier_config = str(head_config)
        _args.learned_classifier_label = _production_head_label(prod, fallback_args=_args)
        if _args.learned_classifier_label in {None, "", "—"}:
            _args.learned_classifier_label = format_classifier_config(_args.best_classifier_config)
        st.session_state['client_production_model_id'] = _args.model_id
        st.session_state['client_production_head_config'] = _args.best_classifier_config
        st.session_state['client_production_head_label'] = _args.learned_classifier_label
        st.session_state['client_production_model_label'] = f"{_args.model_name} (ID: {_args.model_id}) - {_args.learned_classifier_label}"
        st.session_state['optimized_k_value'] = _args.best_classifier_config
        st.session_state['learned_classifier_label'] = _args.learned_classifier_label
    except Exception as exc:
        st.warning(f"Could not apply production model for client inference: {exc}")
    return _args


if not is_admin:
    args = _apply_production_model_to_args(args, st.session_state.get('production_model'), data_dir)

# ---- Create Tabs ---- #

# ---- Shared optimization helpers ----
def _get_aug_cache_dir(_args, split_name: str):
    """Directory storing cached per-round augmentations for one split."""
    params = get_model_params_path(_args)
    parts = params.split('/')
    base_params = '/'.join(parts[:-3])
    base_dir = f'logs/best_models/{_args.task}/{_args.model_name}/{base_params}'
    cache_dir = os.path.join(base_dir, f"norm{_args.normalize}", 'aug_cache', split_name)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def _load_or_build_aug_round_cache(_args, split_name: str, inputs, aug_idx: int, aug_transform):
    """Load one augmentation round from disk, or compute and save it once."""
    cache_dir = _get_aug_cache_dir(_args, split_name)
    cache_file = os.path.join(cache_dir, f"aug_{int(aug_idx)}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                obj = pickle.load(f)
            arr = obj.get('data') if isinstance(obj, dict) and 'data' in obj else obj
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr)
            if arr.shape[0] == len(inputs):
                print(f"[AugCache] Loaded {split_name} aug_{aug_idx} from {cache_file}")
                return arr
            print(f"[AugCache] Shape mismatch for {cache_file}: expected {len(inputs)} samples, got {arr.shape[0]}. Rebuilding.")
        except Exception as e:
            print(f"[AugCache] Failed loading {cache_file}: {e}. Rebuilding.")

    print(f"[AugCache] Building {split_name} aug_{aug_idx} cache ({len(inputs)} samples)...")
    augmented = []
    pbar = tqdm(range(len(inputs)),
                total=len(inputs),
                desc=f"[AugCache] {split_name} aug_{aug_idx}",
                unit="img",
                dynamic_ncols=True)
    for idx in pbar:
        sample = aug_transform(inputs[idx])
        augmented.append(sample.detach().cpu().numpy().astype(np.float16))

    arr = np.stack(augmented, axis=0) if augmented else np.empty((0,), dtype=np.float16)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'data': arr, 'dtype': str(arr.dtype), 'shape': arr.shape}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[AugCache] Saved {split_name} aug_{aug_idx} to {cache_file}")
    except Exception as e:
        print(f"[AugCache] Failed saving {cache_file}: {e}")
    return arr

# Use shared prototype computation from encoding_utils
def _optimize_k_for_args(_args, min_k: int = 1, max_k: int = 20,
                          skip_knn: bool = False, skip_baselines: bool = False,
                          skip_prototypes: bool = False):
    # Load model + datasets and prototypes with n_aug
    n_aug = getattr(_args, 'n_aug', 1)
    
    # Try to load prototypes with n_aug, compute if not exists
    # ---- Device: always prefer CUDA when available ----
    effective_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if getattr(_args, 'device', 'cpu') != effective_device:
        print(f"[Device] ⚠️  args.device='{getattr(_args, 'device', 'cpu')}' overridden → '{effective_device}'")
    _args.device = effective_device
    if effective_device == 'cuda':
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"[Device] ✅ Using GPU: {dev_name} (CUDA:{torch.cuda.current_device()})")
    else:
        print("[Device] ⚠️  Using CPU — no CUDA device detected!")

    try:
        prototypes = _compute_and_save_prototypes_with_naug(_args, n_aug, force_recompute=False)
    except Exception as e:
        st.warning(f"Could not load/compute prototypes with n_aug={n_aug}: {e}. Using default prototypes.")
        print(f"[Encode] Loading model (fallback — no cached prototypes)...")
        model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)
    else:
        print(f"[Encode] Loading model for embedding pass...")
        model, _, _, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)
        print(f"[Encode] Model loaded. train={len(data['inputs']['train'])} valid={len(data['inputs']['valid'])} samples")

    # Prepare a minimal TrainAE wrapper to encode sets (reuse utilities used elsewhere)
    train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                    log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
                    log_mlflow=False, groupkfold=_args.groupkfold)
    train.n_batches = len(unique_batches)
    train.n_cats = len(unique_labels)
    train.unique_batches = unique_batches
    train.unique_labels = unique_labels
    train.epoch = 1
    train.model = model
    train.params = {'n_neighbors': int(_args.n_neighbors)}
    train.set_arcloss()

    # --- Encode train/valid with optional augmentation (train only) ---
    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    knn_aug_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomApply(
            [transforms.RandomResizedCrop(
                size=_args.new_size,
                scale=(0.8, 1.0),
                ratio=(0.8, 1.2)
            )], p=0.5),
    ])

    def _encode_split(split_name: str, n_aug_split: int = 1):
        inputs = data['inputs'][split_name]
        labels = data['labels'][split_name]
        batches = data['batches'][split_name]
        n_total = len(inputs)
        n_batches_enc = max(1, (n_total + _args.bs - 1) // _args.bs)

        # Build/load each augmentation round once, then reuse by sample index.
        aug_round_cache = {}
        if split_name == 'train' and int(n_aug_split) > 0:
            for aug_idx in range(1, int(n_aug_split) + 1):
                aug_round_cache[aug_idx] = _load_or_build_aug_round_cache(
                    _args, split_name, inputs, aug_idx, knn_aug_transform
                )

        encs = []
        labs = []
        doms = []
        _enc_pbar = tqdm(
            range(0, n_total, _args.bs),
            total=n_batches_enc,
            desc=f"[Encode] {split_name}",
            unit="batch",
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for bidx, i in enumerate(_enc_pbar):
                _enc_pbar.set_postfix(samples=f"{min(i+_args.bs, n_total)}/{n_total}")
                batch_inputs = inputs[i:i + _args.bs]
                batch_labels = labels[i:i + _args.bs]
                batch_batches = batches[i:i + _args.bs]
                transformed = []
                labs_batch = []
                doms_batch = []
                for j in range(len(batch_inputs)):
                    # Always include original; n_aug_split is number of extra augmentations
                    repeats = 1 + max(0, n_aug_split) if split_name == 'train' else 1
                    for r in range(repeats):
                        # r == 0 -> original; r > 0 -> augmented (when n_aug_split > 0)
                        if split_name == 'train' and r > 0:
                            cache_arr = aug_round_cache.get(r)
                            cache_idx = i + j
                            if isinstance(cache_arr, np.ndarray) and cache_idx < cache_arr.shape[0]:
                                sample = torch.from_numpy(cache_arr[cache_idx]).float()
                            else:
                                sample = knn_aug_transform(batch_inputs[j])
                        else:
                            sample = base_transform(batch_inputs[j])
                        transformed.append(sample)
                        labs_batch.append(batch_labels[j])
                        doms_batch.append(batch_batches[j])
                if not transformed:
                    continue
                tensor_batch = torch.stack(transformed).to(_args.device)
                encoded, _ = model(tensor_batch)
                encs.append(encoded.detach().cpu().numpy())
                labs.extend(labs_batch)
                doms.extend(doms_batch)
        return np.concatenate(encs), np.array(labs), np.array(doms)

    train_encs, train_cats, train_batches = _encode_split('train', n_aug_split=n_aug)
    valid_encs, valid_cats, valid_batches = _encode_split('valid', n_aug_split=1)
    try:
        test_encs, test_cats, test_batches = _encode_split('test', n_aug_split=1)
    except Exception:
        test_encs = np.empty((0, valid_encs.shape[1]), dtype=valid_encs.dtype)
        test_batches = np.empty((0,), dtype=valid_batches.dtype)
        test_cats = np.empty((0,), dtype=valid_cats.dtype)

    all_encs_parts = [arr for arr in [train_encs, valid_encs, test_encs] if isinstance(arr, np.ndarray) and arr.shape[0] > 0]
    all_batches_parts = [arr for arr in [train_batches, valid_batches, test_batches] if isinstance(arr, np.ndarray) and arr.shape[0] > 0]
    if all_encs_parts and all_batches_parts:
        all_dataset_encs = np.concatenate(all_encs_parts, axis=0)
        all_dataset_batches = np.concatenate(all_batches_parts, axis=0)
    else:
        all_dataset_encs = valid_encs
        all_dataset_batches = valid_batches

    def _safe_split_metrics(y_true, y_pred, y_score=None, pos_label="NotNormal"):
        """Return acc/mcc/auc for one split, with AUC as NaN when unavailable."""
        out = {"acc": np.nan, "mcc": np.nan, "auc": np.nan}
        try:
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            if y_true_arr.size == 0 or y_pred_arr.size == 0 or y_true_arr.shape[0] != y_pred_arr.shape[0]:
                return out
            out["acc"] = float(accuracy_score(y_true_arr, y_pred_arr))
            out["mcc"] = float(matthews_corrcoef(y_true_arr, y_pred_arr))
        except Exception:
            return out

        if y_score is not None:
            try:
                y_score_arr = np.asarray(y_score, dtype=float)
                valid = np.isfinite(y_score_arr)
                y_true_bin = (y_true_arr.astype(str) == str(pos_label)).astype(int)
                if valid.any() and len(np.unique(y_true_bin[valid])) == 2:
                    out["auc"] = float(roc_auc_score(y_true_bin[valid], y_score_arr[valid]))
            except Exception:
                pass
        return out

    def _infer_prediction_label_mapping(raw_train_preds, y_train, max_classes=8):
        """Infer a mapping from encoded classifier outputs back to dataset labels.

        Some baseline classifiers / wrappers return encoded labels (0/1) even though
        the dataset labels are strings (Normal/NotNormal). Comparing those directly
        makes ACC/MCC collapse to 0. We infer the mapping on the train split by
        choosing the label assignment with the best train accuracy, then reuse it
        for valid/test/inference metrics.
        """
        try:
            raw_train_preds = np.asarray(raw_train_preds)
            y_train = np.asarray(y_train)
            if raw_train_preds.size == 0 or y_train.size == 0:
                return None

            pred_values = list(dict.fromkeys(raw_train_preds.tolist()))
            true_values = list(dict.fromkeys(y_train.tolist()))

            # Already in the same label space.
            if set(map(str, pred_values)).issubset(set(map(str, true_values))):
                return None

            if len(pred_values) == 0 or len(true_values) == 0:
                return None
            if len(pred_values) > max_classes or len(true_values) > max_classes:
                return None

            import itertools
            best_mapping = None
            best_acc = -1.0
            for perm in itertools.permutations(true_values, min(len(pred_values), len(true_values))):
                mapping = {pred_values[i]: perm[i] for i in range(len(perm))}
                mapped = np.asarray([mapping.get(v, v) for v in raw_train_preds])
                acc = float(np.mean(mapped.astype(str) == y_train.astype(str)))
                if acc > best_acc:
                    best_acc = acc
                    best_mapping = mapping
            return best_mapping
        except Exception:
            return None

    def _apply_prediction_label_mapping(preds, mapping):
        if not mapping:
            return preds
        try:
            preds_arr = np.asarray(preds)
            return np.asarray([mapping.get(v, v) for v in preds_arr])
        except Exception:
            return preds

    def _score_from_estimator(clf_obj, X, pos_label="NotNormal", label_mapping=None):
        """Return a NotNormal-like score from predict_proba or decision_function."""
        try:
            if hasattr(clf_obj, "predict_proba"):
                proba = clf_obj.predict_proba(X)
                if proba.ndim == 2:
                    classes = getattr(clf_obj, "classes_", None)
                    if classes is not None:
                        classes_list = list(classes)
                        classes_str = [str(c) for c in classes_list]
                        if str(pos_label) in classes_str:
                            return proba[:, classes_str.index(str(pos_label))]
                        if label_mapping:
                            mapped_classes = [str(label_mapping.get(c, c)) for c in classes_list]
                            if str(pos_label) in mapped_classes:
                                return proba[:, mapped_classes.index(str(pos_label))]
                    return np.max(proba, axis=1)
            if hasattr(clf_obj, "decision_function"):
                score = clf_obj.decision_function(X)
                score_arr = np.asarray(score)
                if score_arr.ndim == 1:
                    classes = getattr(clf_obj, "classes_", None)
                    if classes is not None and len(classes) == 2 and label_mapping:
                        positive_raw_class = classes[1]
                        positive_mapped = str(label_mapping.get(positive_raw_class, positive_raw_class))
                        if positive_mapped != str(pos_label):
                            return -score_arr
                    return score_arr
                if classes is not None and label_mapping:
                    mapped_classes = [str(label_mapping.get(c, c)) for c in list(classes)]
                    if str(pos_label) in mapped_classes:
                        return score_arr[:, mapped_classes.index(str(pos_label))]
                return np.max(score_arr, axis=1)
        except Exception:
            return None
        return None

    def _split_metrics_from_estimator(clf_obj):
        """Evaluate one fitted classifier on train/valid/test embeddings."""
        metrics = {}
        label_mapping = None
        try:
            raw_train_preds = clf_obj.predict(train_encs)
            label_mapping = _infer_prediction_label_mapping(raw_train_preds, train_cats)
        except Exception:
            label_mapping = None

        for split_name, X_split, y_split in [
            ("train", train_encs, train_cats),
            ("valid", valid_encs, valid_cats),
            ("test", test_encs, test_cats),
        ]:
            try:
                if X_split is None or len(X_split) == 0:
                    metrics[split_name] = {"acc": np.nan, "mcc": np.nan, "auc": np.nan}
                    continue
                raw_preds = clf_obj.predict(X_split)
                preds = _apply_prediction_label_mapping(raw_preds, label_mapping)
                score = _score_from_estimator(clf_obj, X_split, label_mapping=label_mapping)
                metrics[split_name] = _safe_split_metrics(y_split, preds, score)
            except Exception:
                metrics[split_name] = {"acc": np.nan, "mcc": np.nan, "auc": np.nan}
        return metrics

    def _flatten_split_metrics(metrics):
        flat = {}
        for split_name in ["train", "valid", "test"]:
            m = (metrics or {}).get(split_name, {}) or {}
            flat[f"{split_name}_acc"] = m.get("acc", np.nan)
            flat[f"{split_name}_mcc"] = m.get("mcc", np.nan)
            flat[f"{split_name}_AUC"] = m.get("auc", np.nan)
            flat[f"{split_name}_auc"] = m.get("auc", np.nan)
        return flat

    def _prototype_predict_and_score(proto_dict, X_split, strategy_name="mean"):
        if X_split is None or len(X_split) == 0:
            return np.array([]), None
        proto_vecs, proto_labels = flatten_prototype_dict(proto_dict)
        if proto_vecs.size == 0:
            return np.array([]), None
        proto_knn = fit_knn_classifier(proto_vecs, proto_labels, n_neighbors=1, metric='minkowski')
        preds = proto_knn.predict(X_split)
        score = None
        try:
            dists = proto_knn.kneighbors(X_split, return_distance=True)[0][:, 0]
            # Use inverse distance as a weak confidence score for AUC.
            score = 1.0 / (dists + 1e-12)
            # If nearest label is Normal, invert so high score means NotNormal.
            preds_str = np.asarray(preds).astype(str)
            score = np.where(preds_str == "NotNormal", score, -score)
        except Exception:
            pass
        return preds, score

    # Use ML module for comprehensive classifier search
    include_proto = st.session_state.get('include_prototypes', True)
    best_k_final, best_mcc_final, all_results = find_best_classifier(
        train_encs, train_cats,
        valid_encs, valid_cats,
        min_k=min_k,
        max_k=max_k,
        include_knn=(not skip_knn),
        include_baselines=(not skip_baselines),
        include_prototypes=(include_proto and not skip_prototypes),
        prototype_strategies=['mean', 'kmeans', 'gmm'],
        max_components=10
    )
    
    # Compute train/valid/test metrics for every prediction path and save them into all_results.
    split_metrics = {'knn_per_k': [], 'baselines': {}, 'prototypes': {}}

    if not skip_knn:
        max_k_eff_metrics = min(int(max_k), int(train_encs.shape[0]))
        for k in range(int(min_k), max_k_eff_metrics + 1):
            try:
                knn_model_metrics = fit_knn_classifier(train_encs, train_cats, n_neighbors=k, metric='minkowski')
                flat = _flatten_split_metrics(_split_metrics_from_estimator(knn_model_metrics))
                flat['k'] = int(k)
                split_metrics['knn_per_k'].append(flat)
            except Exception:
                split_metrics['knn_per_k'].append({'k': int(k)})

        # Attach metrics directly to the KNN curve entries used by the UI.
        try:
            knn_curve = all_results.setdefault('knn', {}).setdefault('mcc_per_k', [])
            by_k = {int(m.get('k')): m for m in split_metrics['knn_per_k'] if isinstance(m, dict) and m.get('k') is not None}
            for idx, item in enumerate(knn_curve):
                k_val = item.get('k', idx + 1) if isinstance(item, dict) else idx + 1
                extra = by_k.get(int(k_val), {})
                if isinstance(item, dict):
                    item.update(extra)
                    if 'valid_mcc' not in item and 'valid_mcc' in extra:
                        item['valid_mcc'] = extra.get('valid_mcc')
                else:
                    knn_curve[idx] = {'k': int(k_val), 'valid_mcc': float(item), **extra}
        except Exception:
            pass

    if not skip_baselines:
        baseline_results_for_metrics = all_results.get('baselines', {})
        for baseline_name, baseline_data in baseline_results_for_metrics.items():
            clf_entry = baseline_data.get('classifier', None) if isinstance(baseline_data, dict) else None
            clf_obj = clf_entry.get('classifier', None) if isinstance(clf_entry, dict) else clf_entry
            if clf_obj is None:
                continue
            try:
                flat = _flatten_split_metrics(_split_metrics_from_estimator(clf_obj))
                split_metrics['baselines'][baseline_name] = flat
                if isinstance(baseline_data, dict):
                    baseline_data.update(flat)
                    # Keep old names too, because the UI historically reads mcc/acc.
                    baseline_data.setdefault('mcc', flat.get('valid_mcc'))
                    baseline_data.setdefault('acc', flat.get('valid_acc'))
            except Exception:
                continue

    if not skip_prototypes:
        proto_results_for_metrics = all_results.get('prototypes', {})
    else:
        proto_results_for_metrics = {}
    for strategy_name, strategy_data in proto_results_for_metrics.items():
        if not isinstance(strategy_data, dict):
            continue
        best_n_components = strategy_data.get('best_n_components', None)
        if best_n_components is None:
            continue
        try:
            proto_dict_metrics = compute_prototypes_by_strategy(
                train_encs,
                train_cats,
                strategy=strategy_name,
                n_components=int(best_n_components),
                random_state=1,
            )
            metrics = {}
            for split_name, X_split, y_split in [
                ("train", train_encs, train_cats),
                ("valid", valid_encs, valid_cats),
                ("test", test_encs, test_cats),
            ]:
                preds, score = _prototype_predict_and_score(proto_dict_metrics, X_split, strategy_name)
                metrics[split_name] = _safe_split_metrics(y_split, preds, score)
            flat = _flatten_split_metrics(metrics)
            split_metrics['prototypes'][strategy_name] = flat
            strategy_data.update(flat)
            strategy_data.setdefault('best_mcc', flat.get('valid_mcc'))
        except Exception:
            continue

    all_results['split_metrics'] = split_metrics

    # Compute inference metrics for every classification head on the inference folder.
    # These metrics are keyed by classification head config, not by deep model only.
    # This prevents the UI from showing the same inf_* values for KNN, prototypes,
    # and baselines. Values are saved in the optimization cache.
    inference_metrics_by_head = {}
    try:
        inference_dir_metrics = os.path.join("data", "datasets", "inference")
        inference_paths = []
        for _ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.PNG", "*.JPEG"]:
            inference_paths.extend(glob.glob(os.path.join(inference_dir_metrics, _ext)))
        inference_paths = sorted(dict.fromkeys(inference_paths))

        inf_embs, inf_labels, inf_names = [], [], []
        if inference_paths:
            with torch.no_grad():
                for img_path in inference_paths:
                    fname = os.path.basename(img_path)
                    gt = _inference_ground_truth_global(fname, inference_dir_metrics)
                    if gt == "Unknown":
                        continue
                    try:
                        from PIL import Image as _PILImage
                        pil_img = _PILImage.open(img_path).convert("RGB")
                        img_tensor = preprocess_image(
                            pil_img,
                            size=int(_args.new_size),
                            normalize=getattr(_args, "normalize", "no"),
                            device=getattr(_args, "device", "cpu"),
                        )
                        encoded_inf, _ = model(img_tensor)
                        inf_embs.append(encoded_inf.detach().cpu().numpy()[0])
                        inf_labels.append(gt)
                        inf_names.append(fname)
                    except Exception:
                        continue
        if inf_embs:
            inf_encs = np.asarray(inf_embs)
            inf_labels = np.asarray(inf_labels)

            # KNN heads: one entry per tested k.
            if not skip_knn:
                max_k_eff_inf = min(int(max_k), int(train_encs.shape[0]))
                for k in range(int(min_k), max_k_eff_inf + 1):
                    try:
                        knn_model_inf = fit_knn_classifier(train_encs, train_cats, n_neighbors=k, metric='minkowski')
                        pred_inf = knn_model_inf.predict(inf_encs)
                        score_inf = None
                        if hasattr(knn_model_inf, "predict_proba"):
                            proba = knn_model_inf.predict_proba(inf_encs)
                            classes = [str(c) for c in getattr(knn_model_inf, "classes_", [])]
                            if "NotNormal" in classes:
                                score_inf = proba[:, classes.index("NotNormal")]
                            elif proba.ndim == 2:
                                score_inf = np.max(proba, axis=1)
                        inference_metrics_by_head[str(int(k))] = _metrics_from_binary_predictions_global(
                            inf_labels, pred_inf, score_inf
                        )
                    except Exception:
                        continue

            # Baseline heads: use the fitted estimator objects before they are stripped.
            if not skip_baselines:
                for baseline_name, baseline_data in (all_results.get('baselines', {}) or {}).items():
                    try:
                        clf_entry = baseline_data.get('classifier', None) if isinstance(baseline_data, dict) else None
                        clf_obj = clf_entry.get('classifier', None) if isinstance(clf_entry, dict) else clf_entry
                        if clf_obj is None:
                            continue
                        pred_inf = clf_obj.predict(inf_encs)
                        score_inf = _score_from_estimator(clf_obj, inf_encs)
                        inference_metrics_by_head[f"baseline_{baseline_name}"] = _metrics_from_binary_predictions_global(
                            inf_labels, pred_inf, score_inf
                        )
                    except Exception:
                        continue

            # Prototype heads: one entry per strategy/best component count.
            if not skip_prototypes:
                for strategy_name, strategy_data in (all_results.get('prototypes', {}) or {}).items():
                    if not isinstance(strategy_data, dict):
                        continue
                    best_n_components = strategy_data.get('best_n_components', None)
                    if best_n_components is None:
                        continue
                    try:
                        proto_dict_inf = compute_prototypes_by_strategy(
                            train_encs,
                            train_cats,
                            strategy=strategy_name,
                            n_components=int(best_n_components),
                            random_state=1,
                        )
                        pred_inf, score_inf = _prototype_predict_and_score(proto_dict_inf, inf_encs, strategy_name)
                        inference_metrics_by_head[f"protot_{strategy_name}_{int(best_n_components)}"] = _metrics_from_binary_predictions_global(
                            inf_labels, pred_inf, score_inf
                        )
                    except Exception:
                        continue
    except Exception as _inf_metrics_exc:
        print(f"[InferenceMetrics] Could not compute head-specific inference metrics: {_inf_metrics_exc}")

    all_results['inference_metrics_by_head'] = inference_metrics_by_head

    # Compute batch-effect metrics for every prediction path (incl. all KNN k)
    batch_effects = {'knn_per_k': [], 'baselines': {}, 'prototypes': {}}

    if not skip_knn:
        max_k_eff = min(int(max_k), int(train_encs.shape[0]))
        for k in tqdm(range(int(min_k), max_k_eff + 1),
                      total=max_k_eff - int(min_k) + 1,
                      desc=f"[BatchEff] KNN k={min_k}..{max_k_eff}",
                      unit="k",
                      dynamic_ncols=True):
            knn_model = fit_knn_classifier(train_encs, train_cats, n_neighbors=k, metric='minkowski')
            knn_preds = knn_model.predict(all_dataset_encs)
            metrics_k = _compute_batch_effect_from_predictions(knn_preds, all_dataset_batches)
            batch_effects['knn_per_k'].append({'k': int(k), **metrics_k})

    if not skip_baselines:
        baseline_results = all_results.get('baselines', {})
        for baseline_name, baseline_data in baseline_results.items():
            clf_entry = baseline_data.get('classifier', None) if isinstance(baseline_data, dict) else None
            clf_obj = clf_entry.get('classifier', None) if isinstance(clf_entry, dict) else clf_entry
            if clf_obj is None:
                continue
            try:
                baseline_preds = clf_obj.predict(all_dataset_encs)
                batch_effects['baselines'][baseline_name] = _compute_batch_effect_from_predictions(
                    baseline_preds, all_dataset_batches
                )
            except Exception:
                continue

    if not skip_prototypes:
        proto_results_be = all_results.get('prototypes', {})
    else:
        proto_results_be = {}
    for strategy_name, strategy_data in proto_results_be.items():
        if not isinstance(strategy_data, dict):
            continue
        best_n_components = strategy_data.get('best_n_components', None)
        if best_n_components is None:
            continue
        try:
            proto_dict = compute_prototypes_by_strategy(
                train_encs,
                train_cats,
                strategy=strategy_name,
                n_components=int(best_n_components),
                random_state=1,
            )
            proto_vecs, proto_labels = flatten_prototype_dict(proto_dict)
            if proto_vecs.size == 0:
                continue
            proto_knn = fit_knn_classifier(proto_vecs, proto_labels, n_neighbors=1, metric='minkowski')
            proto_preds = proto_knn.predict(all_dataset_encs)
            batch_effects['prototypes'][strategy_name] = _compute_batch_effect_from_predictions(
                proto_preds, all_dataset_batches
            )
        except Exception:
            continue

    all_results['batch_effects'] = batch_effects

    # Extract detailed results for backward compatibility
    mcc_per_k = all_results.get('knn', {}).get('mcc_per_k', [])
    proto_results = all_results.get('prototypes', {})
    baseline_results = all_results.get('baselines', {})

    return best_k_final, float(best_mcc_final), mcc_per_k, all_results

def _compute_and_save_prototypes_with_naug(_args, n_aug: int, force_recompute: bool = False):
    """Compute prototypes with specific n_aug and save them for future use.
    
    Args:
        _args: Arguments with model configuration
        n_aug: Number of augmentations per image
        force_recompute: If True, recompute even if cached prototypes exist
        
    Returns:
        prototypes: Dict with 'combined', 'class', 'batch' prototypes
    """
    # ---- Device: always prefer CUDA when available ----
    effective_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if getattr(_args, 'device', 'cpu') != effective_device:
        print(f"[Device] ⚠️  args.device='{getattr(_args, 'device', 'cpu')}' overridden → '{effective_device}'")
    _args.device = effective_device
    if effective_device == 'cuda':
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"[Device] ✅ Using GPU: {dev_name} (CUDA:{torch.cuda.current_device()})")
    else:
        print("[Device] ⚠️  Using CPU — no CUDA device detected!")

    # Build path for prototypes with n_aug
    params = get_model_params_path(_args)
    parts = params.split('/')
    base_params = '/'.join(parts[:-3])
    base_dir = f'logs/best_models/{_args.task}/{_args.model_name}/{base_params}'
    model_path, _, resolved_k = resolve_model_paths(base_dir, _args.normalize, _args.dist_fct, int(_args.n_neighbors))
    
    # Get directory from model path
    model_dir = os.path.dirname(model_path)
    proto_path_with_naug = os.path.join(model_dir, f'prototypes_naug{n_aug}.pkl')
    
    print(f"[Proto] Looking for prototypes at: {proto_path_with_naug}")
    
    # Load existing prototypes if available and not forcing recompute
    if os.path.exists(proto_path_with_naug) and not force_recompute:
        print(f"[Proto] Found cached prototypes, loading from {proto_path_with_naug}")
        with open(proto_path_with_naug, 'rb') as f:
            proto_obj = pickle.load(f)

        # Support both legacy Prototypes objects and dict-only saves
        if isinstance(proto_obj, dict):
            prototypes = proto_obj
        else:
            prototypes = {
                'combined': proto_obj.prototypes,
                'class': proto_obj.class_prototypes,
                'batch': proto_obj.batch_prototypes
            }
        print(f"[Proto] Successfully loaded prototypes with n_aug={n_aug}")
        return prototypes
    
    print(f"[Proto] Computing new prototypes with n_aug={n_aug}...")
    
    # Otherwise, compute new prototypes with n_aug
    # Load model and data
    print(f"[Proto] Loading model and dataset...")
    model, _, _, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)
    print(f"[Proto] Model loaded. Training samples: {len(data['inputs']['train'])}")
    
    # Prepare TrainAE wrapper to encode with augmentation
    train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                    log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
                    log_mlflow=False, groupkfold=_args.groupkfold)
    train.n_batches = len(unique_batches)
    train.n_cats = len(unique_labels)
    train.unique_batches = unique_batches
    train.unique_labels = unique_labels
    train.epoch = 1
    train.model = model
    train.params = {'n_aug': n_aug}
    train.set_arcloss()
    
    # Initialize Prototypes object with strategy from args
    proto_strategy = getattr(_args, 'prototype_strategy', 'mean')
    proto_components = getattr(_args, 'prototype_components', 1)
    proto_seed = getattr(_args, 'seed', 1)
    proto_obj = Prototypes(
        unique_labels,
        unique_batches,
        strategy=proto_strategy,
        components=proto_components,
        random_state=proto_seed,
    )
    
    lists, traces = get_empty_traces()
    
    # Apply augmentation and encode like make_encoded_values does
    print(f"[Proto] Encoding train set with n_aug={n_aug}...")
    
    # Base transform (no augmentation)
    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Augmentations dedicated to KNN expansion
    knn_aug_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomApply(
            [transforms.RandomResizedCrop(
                size=_args.new_size,
                scale=(0.8, 1.0),
                ratio=(0.8, 1.2)
            )], p=0.5),
    ])
    
    # Collect encoded samples and metadata
    all_encoded = []
    all_labels = []
    all_domains = []
    
    train_inputs = data['inputs']['train']
    train_labels = data['labels']['train']
    train_batches = data['batches']['train']

    # Build/load each augmentation round once, then reuse by sample index.
    aug_round_cache = {}
    if int(n_aug) > 0:
        for aug_idx in range(1, int(n_aug) + 1):
            aug_round_cache[aug_idx] = _load_or_build_aug_round_cache(
                _args, 'train', train_inputs, aug_idx, knn_aug_transform
            )
    
    # Process in batches
    n_train = len(train_inputs)
    n_batches_proto = max(1, (n_train + _args.bs - 1) // _args.bs)
    _proto_pbar = tqdm(
        range(0, n_train, _args.bs),
        total=n_batches_proto,
        desc=f"[Proto] Encoding n_aug={n_aug}",
        unit="batch",
        dynamic_ncols=True,
    )
    for batch_idx, i in enumerate(_proto_pbar):
        _proto_pbar.set_postfix(samples=f"{min(i+_args.bs, n_train)}/{n_train}")
        batch_inputs = train_inputs[i:i + _args.bs]
        batch_labels = train_labels[i:i + _args.bs]
        batch_batches = train_batches[i:i + _args.bs]
        
        transformed_samples = []
        labels_for_batch = []
        batches_for_batch = []
        
        for j in range(len(batch_inputs)):
            # Always include original + n_aug extra augmented copies
            repeats = 1 + max(0, n_aug)
            for r in range(repeats):
                if r == 0:
                    sample = base_transform(batch_inputs[j])
                else:
                    cache_arr = aug_round_cache.get(r)
                    cache_idx = i + j
                    if isinstance(cache_arr, np.ndarray) and cache_idx < cache_arr.shape[0]:
                        sample = torch.from_numpy(cache_arr[cache_idx]).float()
                    else:
                        sample = knn_aug_transform(batch_inputs[j])
                transformed_samples.append(sample)
                labels_for_batch.append(batch_labels[j])
                batches_for_batch.append(batch_batches[j])
        
        if not transformed_samples:
            continue
        
        # Stack and encode
        transformed_tensor = torch.stack(transformed_samples).to(_args.device)
        with torch.no_grad():
            encoded, _ = model(transformed_tensor)
        
        all_encoded.append(encoded.detach().cpu().numpy())
        all_labels.extend(labels_for_batch)
        all_domains.extend(batches_for_batch)
    
    # Prepare lists structure for prototypes computation
    lists['train']['encoded_values'] = all_encoded
    lists['train']['labels'] = [np.array(all_labels)]
    lists['train']['domains'] = [np.array(all_domains)]
    
    print(f"[Proto] Computing prototypes from {len(all_encoded)} batches with {sum(len(b) for b in all_encoded)} total samples...")
    # Compute prototypes from encoded values
    proto_obj.set_prototypes('train', lists)
    
    # Ensure directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Save prototypes with n_aug suffix (store plain dict to avoid pickle class issues)
    prototypes = {
        'combined': proto_obj.prototypes,
        'class': proto_obj.class_prototypes,
        'batch': proto_obj.batch_prototypes
    }

    print(f"[Proto] Saving prototypes to {proto_path_with_naug}...")
    try:
        with open(proto_path_with_naug, 'wb') as f:
            pickle.dump(prototypes, f)
        print(f"[Proto] ✅ Successfully saved prototypes to {proto_path_with_naug}")
    except Exception as e:
        print(f"[Proto] ❌ Failed to save prototypes: {e}")
        raise

    return prototypes

def _get_optimization_cache_file(_args):
    """Get path to optimization cache file for this model."""
    params = get_model_params_path(_args)
    parts = params.split('/')
    base_params = '/'.join(parts[:-3])
    base_dir = f'logs/best_models/{_args.task}/{_args.model_name}/{base_params}'
    # Keep a single cache per model setup and normalization.
    # Older runs stored cache under dist_*/knn*/; _load_optimization_cache merges them.
    cache_dir = os.path.join(base_dir, f"norm{_args.normalize}")
    return os.path.join(cache_dir, 'knn_optimization_cache.pkl')

def _merge_optimization_cache(target_cache, source_cache):
    """Merge source cache into target cache, keeping best MCC per n_aug."""
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

def _load_optimization_cache(_args):
    """Load cached optimization results for different n_aug values.

    Also merges legacy caches saved under dist_*/knn*/ subfolders.
    """
    cache_file = _get_optimization_cache_file(_args)
    cache_dir = os.path.dirname(cache_file)
    merged_cache = {}

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                merged_cache = _merge_optimization_cache(merged_cache, pickle.load(f))
        except Exception as e:
            print(f"[Cache] Could not load optimization cache: {e}")

    # Backward compatibility: merge older split cache files.
    legacy_pattern = os.path.join(cache_dir, 'dist_*', 'knn*', 'knn_optimization_cache.pkl')
    legacy_files = glob.glob(legacy_pattern)
    merged_from_legacy = False
    for legacy_file in legacy_files:
        try:
            with open(legacy_file, 'rb') as f:
                legacy_cache = pickle.load(f)
            merged_cache = _merge_optimization_cache(merged_cache, legacy_cache)
            merged_from_legacy = True
        except Exception as e:
            print(f"[Cache] Could not load legacy cache {legacy_file}: {e}")

    # Persist merged legacy cache into the unified location.
    if merged_from_legacy and merged_cache:
        try:
            _save_optimization_cache(_args, merged_cache)
            print(f"[Cache] ✅ Merged {len(legacy_files)} legacy cache file(s) into unified cache")
        except Exception as e:
            print(f"[Cache] Could not persist merged cache: {e}")

    return merged_cache

def _save_optimization_cache(_args, cache_data):
    """Save optimization results cache."""
    cache_file = _get_optimization_cache_file(_args)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"[Cache] ✅ Saved optimization cache for {len(cache_data)} n_aug values")

def _strip_classifier_objects(baseline_results):
    """Strip unpicklable classifier objects, keeping only metrics."""
    clean = {}
    for name, result in baseline_results.items():
        if not isinstance(result, dict):
            continue
        metric_keys = [
            'mcc', 'acc', 'best_params',
            'train_acc', 'train_mcc', 'train_AUC', 'train_auc',
            'valid_acc', 'valid_mcc', 'valid_AUC', 'valid_auc',
            'test_acc', 'test_mcc', 'test_AUC', 'test_auc',
        ]
        clean[name] = {k: result.get(k) for k in metric_keys if k in result}
    return clean

def _strip_prototype_results(proto_results):
    """Strip potentially problematic objects from prototype results."""
    clean = {}
    for strategy, result in proto_results.items():
        if not isinstance(result, dict):
            continue
        metric_keys = [
            'best_mcc', 'best_n_components', 'n_prototypes', 'per_components',
            'train_acc', 'train_mcc', 'train_AUC', 'train_auc',
            'valid_acc', 'valid_mcc', 'valid_AUC', 'valid_auc',
            'test_acc', 'test_mcc', 'test_AUC', 'test_auc',
        ]
        clean[strategy] = {k: result.get(k) for k in metric_keys if k in result}
        clean[strategy].setdefault('per_components', result.get('per_components', []))
    return clean

def _save_optimization_result(_args, n_aug, best_k, best_mcc, all_results):
    """Save optimization result including all classifier approaches with n_aug tracking.
    
    Args:
        _args: Model arguments
        n_aug: Data augmentation count used for this optimization
        best_k: Best k value found
        best_mcc: Best MCC score achieved
        all_results: Dict with results from find_best_classifier containing:
            - 'knn': KNN results with best_k and mcc_per_k
            - 'baselines': NB, LogReg, SVC results
            - 'prototypes': Prototype strategy results (mean, kmeans, gmm)
    """
    cache = _load_optimization_cache(_args)
    
    # Extract classifier-specific results for explicit n_aug tracking
    baseline_results = all_results.get('baselines', {})
    proto_results = all_results.get('prototypes', {})
    knn_results = all_results.get('knn', {})
    
    # Extract batch effects from all_results
    batch_effects = all_results.get('batch_effects', {})
    
    # Strip unpicklable objects (classifier instances) before saving
    clean_baselines = _strip_classifier_objects(baseline_results)
    clean_prototypes = _strip_prototype_results(proto_results)
    
    # Store with explicit classifier type and n_aug
    cache[int(n_aug)] = {
        'cache_version': 2,  # Version the cache for future compatibility
        'n_aug': int(n_aug),
        'best_k': best_k,  # Best overall
        'best_mcc': float(best_mcc),  # Best overall
        'timestamp': datetime.now().isoformat(),
        # Baselines - only plain metrics, no classifier objects
        'baselines': clean_baselines,
        # KNN results
        'knn': {
            'n_aug_applied': int(n_aug),
            'best_k': knn_results.get('best_k', best_k),
            'mcc_per_k': knn_results.get('mcc_per_k', []),
        },
        # Prototypes results - only metrics
        'prototypes': clean_prototypes,
        # Batch effects - containing metrics for all approaches
        'batch_effects': batch_effects,
        # Train/valid/test metrics for all approaches
        'split_metrics': all_results.get('split_metrics', {}),
        # Inference metrics keyed by classification head config, e.g. '11',
        # 'baseline_logreg', 'protot_kmeans_3'.
        'inference_metrics_by_head': all_results.get('inference_metrics_by_head', {}),
    }
    _save_optimization_cache(_args, cache)

def _is_cache_complete(entry, max_k):
    """Return True only if all expected model types are present and KNN covers up to max_k.

    An entry is considered incomplete if:
    - It has no knn.mcc_per_k (KNN was never computed)
    - The highest k in mcc_per_k is below max_k (user increased batch_k_max)
    - It has no baselines (e.g. very old cache format)
    - It has no prototypes (e.g. very old cache format)
    """
    if not isinstance(entry, dict):
        return False
    # --- KNN check ---
    knn = entry.get('knn', {})
    mcc_per_k = knn.get('mcc_per_k', []) if isinstance(knn, dict) else []
    if not mcc_per_k:
        return False
    # Determine the highest k that was evaluated
    highest_k = 0
    for item in mcc_per_k:
        if isinstance(item, dict):
            k_val = item.get('k')
            if k_val is not None:
                highest_k = max(highest_k, int(k_val))
        else:
            # Legacy: list of plain floats indexed by k-1
            highest_k = len(mcc_per_k)
    if highest_k < int(max_k):
        return False
    # --- Baselines check ---
    if not entry.get('baselines'):
        return False
    # --- Prototypes check ---
    if not entry.get('prototypes'):
        return False
    return True

def _what_is_missing(entry, max_k):
    """Return a dict describing which parts are absent/incomplete in a cache entry.

    Returns:
        {'knn': bool, 'baselines': bool, 'prototypes': bool}
        True means the part needs to be computed.
    """
    if not isinstance(entry, dict):
        return {'knn': True, 'baselines': True, 'prototypes': True}
    missing = {'knn': False, 'baselines': False, 'prototypes': False}
    # --- KNN ---
    knn = entry.get('knn', {})
    mcc_per_k = knn.get('mcc_per_k', []) if isinstance(knn, dict) else []
    if not mcc_per_k:
        missing['knn'] = True
    else:
        highest_k = 0
        for item in mcc_per_k:
            if isinstance(item, dict):
                k_val = item.get('k')
                if k_val is not None:
                    highest_k = max(highest_k, int(k_val))
            else:
                highest_k = len(mcc_per_k)
        if highest_k < int(max_k):
            missing['knn'] = True
    # --- Baselines ---
    if not entry.get('baselines'):
        missing['baselines'] = True
    # --- Prototypes ---
    if not entry.get('prototypes'):
        missing['prototypes'] = True
    return missing

def _merge_partial_optimization_result(_args, n_aug, new_all_results):
    """Merge partial new results into an existing cache entry, recomputing best_k/best_mcc.

    Only the sub-sections present in new_all_results are updated; existing sections
    (knn / baselines / prototypes) are preserved when the new results dict omits them.
    """
    cache = _load_optimization_cache(_args)
    existing = cache.get(int(n_aug), {})

    # --- Merge KNN ---
    new_knn = new_all_results.get('knn', {})
    if new_knn and new_knn.get('mcc_per_k'):
        existing['knn'] = {
            'n_aug_applied': int(n_aug),
            'best_k': new_knn.get('best_k'),
            'mcc_per_k': new_knn.get('mcc_per_k', []),
        }

    # --- Merge baselines ---
    new_baselines = new_all_results.get('baselines', {})
    if new_baselines:
        existing['baselines'] = _strip_classifier_objects(new_baselines)

    # --- Merge prototypes ---
    new_prototypes = new_all_results.get('prototypes', {})
    if new_prototypes:
        existing['prototypes'] = _strip_prototype_results(new_prototypes)

    # --- Merge head-specific inference metrics ---
    new_inf = new_all_results.get('inference_metrics_by_head', {})
    if new_inf:
        existing_inf = existing.get('inference_metrics_by_head', {}) if isinstance(existing.get('inference_metrics_by_head', {}), dict) else {}
        existing_inf.update(new_inf)
        existing['inference_metrics_by_head'] = existing_inf

    # --- Merge split metrics ---
    new_split = new_all_results.get('split_metrics', {})
    if new_split:
        existing_split = existing.get('split_metrics', {}) if isinstance(existing.get('split_metrics', {}), dict) else {}
        for key, value in new_split.items():
            if isinstance(value, dict) and isinstance(existing_split.get(key), dict):
                existing_split[key].update(value)
            else:
                existing_split[key] = value
        existing['split_metrics'] = existing_split

    # --- Merge batch effects ---
    new_be = new_all_results.get('batch_effects', {})
    existing_be = existing.get('batch_effects', {'knn_per_k': [], 'baselines': {}, 'prototypes': {}})
    if new_be.get('knn_per_k'):
        existing_be['knn_per_k'] = new_be['knn_per_k']
    if new_be.get('baselines'):
        existing_be.setdefault('baselines', {}).update(new_be['baselines'])
    if new_be.get('prototypes'):
        existing_be.setdefault('prototypes', {}).update(new_be['prototypes'])
    existing['batch_effects'] = existing_be

    # --- Recompute best_k / best_mcc from all available data ---
    best_k_overall, best_mcc_overall = None, -1.0
    # From KNN
    for item in existing.get('knn', {}).get('mcc_per_k', []):
        if isinstance(item, dict):
            k_val = item.get('k'); mcc = item.get('mcc', -1)
            if mcc > best_mcc_overall:
                best_mcc_overall = mcc
                best_k_overall = k_val
    # From prototypes
    for strategy, strat_data in existing.get('prototypes', {}).items():
        if not isinstance(strat_data, dict):
            continue
        mcc = strat_data.get('best_mcc', -1) or -1
        if mcc > best_mcc_overall:
            best_mcc_overall = mcc
            best_k_overall = f'protot_{strategy}_{strat_data.get("best_n_components")}'
    # From baselines
    for bl_name, bl_data in existing.get('baselines', {}).items():
        if not isinstance(bl_data, dict):
            continue
        mcc = bl_data.get('mcc', -1) or -1
        if mcc > best_mcc_overall:
            best_mcc_overall = mcc
            best_k_overall = f'baseline_{bl_name}'

    existing.update({
        'cache_version': 2,
        'n_aug': int(n_aug),
        'best_k': best_k_overall,
        'best_mcc': float(best_mcc_overall) if best_mcc_overall > -1 else 0.0,
        'timestamp': datetime.now().isoformat(),
    })
    cache[int(n_aug)] = existing
    _save_optimization_cache(_args, cache)


# Show different tabs based on admin role
if is_admin:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🏆 Model Selection",
        "🧠 Learned Embedding",
        "📂 Past Results",
        "🔬 New Analysis",
        "🤝 Ensemble",
        "🖼️ Grad-CAM Gallery",
        "🖼️ Raw Pixel Classification",
        "📊 Admin Analytics",
        "📈 Inference Results",
    ])
else:
    # Client view - restricted tabs only
    tab3, tab4, tab9 = st.tabs([
        "📂 Past Results",
        "🔬 New Analysis",
        "📈 Inference Results",
    ])

if is_admin:
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
    
            # Attach cached split metrics + calibration metrics per model entry.
            # Keep the registry Accuracy/MCC as-is, but also expose split-specific values
            # from run_summary.json / prediction CSVs when available.
            for col in [
                "Train MCC", "Valid MCC", "Test MCC",
                "Train AUC", "Valid AUC", "Test AUC",
                "ECE", "Brier",
            ]:
                models_df[col] = np.nan

            for idx, row in models_df.iterrows():
                log_path = row.get("Log Path")

                split_metrics = get_split_mcc_metrics(log_path)
                if split_metrics:
                    models_df.at[idx, "Train MCC"] = split_metrics.get("train_mcc", np.nan)
                    models_df.at[idx, "Valid MCC"] = split_metrics.get("valid_mcc", np.nan)
                    models_df.at[idx, "Test MCC"] = split_metrics.get("test_mcc", np.nan)
                    models_df.at[idx, "Train AUC"] = split_metrics.get("train_auc", np.nan)
                    models_df.at[idx, "Valid AUC"] = split_metrics.get("valid_auc", np.nan)
                    models_df.at[idx, "Test AUC"] = split_metrics.get("test_auc", np.nan)

                metrics = get_calibration_metrics(log_path)
                if metrics and not metrics.get("error"):
                    models_df.at[idx, "ECE"] = metrics.get("ece")
                    models_df.at[idx, "Brier"] = metrics.get("brier")

            # Rank and number the leaderboard with validation MCC first, then registry MCC.
            valid_sort = pd.to_numeric(models_df.get("Valid MCC"), errors="coerce")
            mcc_sort = pd.to_numeric(models_df.get("MCC"), errors="coerce")
            models_df = models_df.assign(
                _valid_sort=valid_sort.fillna(float("-inf")),
                _mcc_sort=mcc_sort.fillna(float("-inf")),
            )
            models_df = models_df.sort_values(["_valid_sort", "_mcc_sort"], ascending=[False, False]).reset_index(drop=True)
            models_df = models_df.drop(columns=["_valid_sort", "_mcc_sort"])
            models_df["#"] = range(1, len(models_df) + 1)
    
            # Reorder columns to surface split metrics and calibration next to registry metrics
            metric_cols = [
                "Accuracy", "MCC",
                "Train MCC", "Valid MCC", "Test MCC",
                "Train AUC", "Valid AUC", "Test AUC",
                "ECE", "Brier",
            ]
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

            def _top_models_head_label(row_dict):
                """Return the learned-embedding head currently/best applied for this model row."""
                try:
                    local_args = argparse.Namespace(**vars(args))
                    row = dict(row_dict)
                    parsed = extract_params_from_log_path(row.get("Log Path") or "")
                    row.update({k: v for k, v in parsed.items() if v is not None})

                    if row.get("Model Name") is not None:
                        local_args.model_name = row.get("Model Name")
                    if row.get("NSize") is not None:
                        local_args.new_size = ensure_int(row.get("NSize"))
                    if row.get("FGSM") is not None:
                        local_args.fgsm = row.get("FGSM")
                    if row.get("Normalize") is not None:
                        local_args.normalize = row.get("Normalize")
                    if row.get("N_Calibration") is not None:
                        local_args.n_calibration = row.get("N_Calibration")
                    if row.get("Classif_Loss") is not None:
                        local_args.classif_loss = row.get("Classif_Loss")
                    if row.get("DLoss") is not None:
                        local_args.dloss = row.get("DLoss")
                    if row.get("Dist_Fct") is not None:
                        local_args.dist_fct = row.get("Dist_Fct")
                    if row.get("Prototypes") is not None:
                        local_args.prototypes_to_use = row.get("Prototypes")
                    if row.get("NPos") is not None:
                        local_args.n_positives = ensure_int(row.get("NPos"))
                    if row.get("NNeg") is not None:
                        local_args.n_negatives = ensure_int(row.get("NNeg"))
                    if row.get("N_Neighbors") is not None:
                        local_args.n_neighbors = ensure_int(row.get("N_Neighbors"))

                    dataset = row.get("Dataset") or parsed.get("Dataset")
                    if dataset:
                        local_args.path = os.path.join("data", dataset) if not str(dataset).startswith("data/") else dataset

                    # This must be resolved per model row. Do NOT use
                    # local_args.best_classifier_config here, because local_args is
                    # copied from the currently selected sidebar model and would
                    # make every leaderboard row show the same head.
                    head_config = row.get("Head Config")
                    if head_config is None or str(head_config).strip() in {"", "nan", "None"}:
                        head_config = _best_head_config_for_args_global(local_args)

                    label = _head_config_label_global(head_config)
                    # User-facing convention for this table: KNN (nn=11), not raw N_Neighbors.
                    label = label.replace("KNN (k=", "KNN (nn=")
                    label = label.replace("KNN k=", "KNN (nn=")
                    return label
                except Exception:
                    try:
                        nn = ensure_int(row_dict.get("N_Neighbors"))
                        return f"KNN (nn={nn})"
                    except Exception:
                        return "—"

            display_columns = [col for col in models_df.columns if col not in {"Log Path", "N_Neighbors"}]
            display_df = models_df[display_columns].copy()
            display_df["Head"] = models_df.apply(lambda r: _top_models_head_label(r.to_dict()), axis=1)
            # Put Head where N_Neighbors used to be: after Normalize when possible.
            cols_now = list(display_df.columns)
            if "Head" in cols_now:
                cols_now.remove("Head")
                insert_at = cols_now.index("Normalize") + 1 if "Normalize" in cols_now else min(12, len(cols_now))
                cols_now.insert(insert_at, "Head")
                display_df = display_df[cols_now]

            # Keep full table (including Log Path and raw N_Neighbors) in session for cross-tab mapping/internal args
            st.session_state['best_models_table'] = models_df.copy()
            st.dataframe(_arrow_safe_dataframe(display_df), use_container_width=True)
    
            # Calibration vs Performance Plots
            st.markdown("---")
            st.subheader("📈 Calibration vs Performance")
            
            # Filter to rows with valid metrics
            plot_df = models_df.dropna(subset=["MCC", "ECE", "Brier"]).copy()
            
            if len(plot_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**MCC vs ECE**")
                    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
                    ax1.scatter(plot_df["MCC"], plot_df["ECE"], alpha=0.6, s=50)
                    ax1.set_xlabel("MCC (higher is better)", fontsize=10)
                    ax1.set_ylabel("ECE (lower is better)", fontsize=10)
                    ax1.set_title("Expected Calibration Error vs MCC", fontsize=11)
                    ax1.grid(True, alpha=0.3)
                    st.pyplot(fig1, use_container_width=True)
                    plt.close(fig1)
                with col2:
                    st.markdown("**MCC vs Brier Score**")
                    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                    ax2.scatter(plot_df["MCC"], plot_df["Brier"], alpha=0.6, s=50, color='orange')
                    ax2.set_xlabel("MCC (higher is better)", fontsize=10)
                    ax2.set_ylabel("Brier Score (lower is better)", fontsize=10)
                    ax2.set_title("Brier Score vs MCC", fontsize=11)
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2, use_container_width=True)
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
    
                                    col_curve, col_line = st.columns(2)
                                    with col_curve:
                                        fig, ax = plt.subplots(figsize=(6, 3.5))
                                        ax.plot(df_curve["prob_pred"], df_curve["prob_true"], marker='o', linewidth=1.5,
                                            label=f"Model #{row_dict.get('#', '?')} (ECE: {ece_val:.4f}, Brier: {brier_val:.4f})")
                                        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly calibrated")
                                        ax.set_xlabel("Mean predicted probability")
                                        ax.set_ylabel("Fraction of positives")
                                        ax.set_title("Calibration Curve")
                                        ax.legend()
                                        ax.grid(alpha=0.3)
                                        st.pyplot(fig, use_container_width=True)
                                        plt.close(fig)
                                    # with col_line:
                                    #     st.line_chart(df_curve.set_index('prob_pred'))
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
                        # Only update session state and rerun; defer any heavy computation until user interacts with analysis or optimization controls
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
    
        # Display cached results immediately if they exist for current model
        if 'optimized_k_value' in st.session_state and st.session_state.get('k_opt_model_key') == current_model_key:
            best_val = st.session_state['optimized_k_value']
            mcc_val = st.session_state['k_opt_best_mcc']
            st.success(f"✅ Previous Optimization Found: {best_val} (Validation MCC: {mcc_val:.3f})")
            
            # Display cached comparison plot if available
            proto_results = st.session_state.get('k_opt_proto_results', {})
            mcc_curve = st.session_state.get('k_opt_curve', [])
            
            if mcc_curve and proto_results:
                try:
                    fig_compare, ax_compare = plt.subplots(figsize=(6, 3.5))
                    
                    # Plot KNN curve
                    curve_df = pd.DataFrame(mcc_curve)
                    curve_df = curve_df.sort_values('k')
                    ax_compare.plot(curve_df['k'], curve_df['mcc'], marker='o', linewidth=2.5, 
                                   markersize=7, label='KNN', color='#e74c3c', zorder=3)
                    
                    # Plot prototype strategies
                    colors = {'mean': '#3498db', 'kmeans': '#f39c12', 'gmm': '#2ecc71'}
                    for strategy in ['mean', 'kmeans', 'gmm']:
                        result = proto_results.get(strategy, {})
                        per_components = result.get('per_components', [])
                        if per_components:
                            per_df = pd.DataFrame(per_components).sort_values('n_components')
                            ax_compare.plot(per_df['n_components'], per_df['mcc'], marker='s', linewidth=2.5,
                                          markersize=7, label=f'{strategy.capitalize()}', 
                                          color=colors.get(strategy, '#95a5a6'), zorder=2)
                    
                    best_knn_mcc = st.session_state.get('k_opt_best_mcc', 0)
                    ax_compare.axhline(y=best_knn_mcc, color='#e74c3c', linestyle='--', linewidth=1.5, 
                                      alpha=0.6, zorder=1)
                    
                    ax_compare.set_xlabel('k (KNN) / n_components (Prototypes)', fontsize=12, fontweight='bold')
                    ax_compare.set_ylabel('Validation MCC', fontsize=12, fontweight='bold')
                    ax_compare.set_title('KNN vs Prototype Strategies: MCC Comparison', fontsize=13, fontweight='bold')
                    ax_compare.legend(loc='best', fontsize=11, framealpha=0.95)
                    ax_compare.grid(True, alpha=0.3)
                    ax_compare.set_ylim([min(curve_df['mcc'].min() - 0.05, min([r.get('best_mcc', 1) for r in proto_results.values() if r.get('best_mcc')]) - 0.05), 
                                        min(1.0, max(curve_df['mcc'].max(), max([r.get('best_mcc', 0) for r in proto_results.values() if r.get('best_mcc')]) + 0.1))])
                    
                    plt.tight_layout()
                    st.pyplot(fig_compare, use_container_width=True)
                    plt.close(fig_compare)
                except Exception as e:
                    st.warning(f"Could not display cached chart: {e}")
            
            st.divider()
    
        def _compute_best_proto_mcc_for_args(_args, strategy: str, min_components: int = 1, max_components: int = 5, random_state: int = 1):
            """Compute the best validation MCC for a given prototype strategy over a range of components per class.
    
            Returns a dict with keys: best_mcc, best_n_components, n_prototypes, per_components.
            """
            # Load model + datasets
            model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)
    
            # Prepare a minimal TrainAE wrapper
            train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                            log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
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
    
            best_mcc = None
            best_n_components = None
            best_n_prototypes = 0
            per_components = []
    
            for n_components in range(min_components, max_components + 1):
                proto_dict = compute_prototypes_by_strategy(train_encs, train_cats, strategy, n_components, random_state)
    
                proto_vecs, proto_labels = flatten_prototype_dict(proto_dict)
    
                if len(proto_vecs) == 0:
                    continue
    
                dists = np.linalg.norm(valid_encs[:, None, :] - proto_vecs[None, :, :], axis=2)
                proto_preds = proto_labels[np.argmin(dists, axis=1)]
                proto_mcc = float(MCC(valid_cats, proto_preds))
                per_components.append({'n_components': n_components, 'mcc': proto_mcc, 'n_prototypes': len(proto_vecs)})
    
                if (best_mcc is None) or (proto_mcc > best_mcc):
                    best_mcc = proto_mcc
                    best_n_components = n_components
                    best_n_prototypes = len(proto_vecs)
    
            return {
                'best_mcc': best_mcc,
                'best_n_components': best_n_components,
                'n_prototypes': best_n_prototypes,
                'per_components': per_components,
            }
    
        # Define PCA computation function - now handles multiple strategies
        def _compute_pca_for_args(_args, proto_strategies=None, proto_components=1):
            """Compute PCA with prototypes for one or multiple strategies.
            
            Args:
                _args: Model configuration
                proto_strategies: list of strategy names or single strategy string
                proto_components: number of components per class
            """
            if proto_strategies is None:
                proto_strategies = ['mean']
            elif isinstance(proto_strategies, str):
                proto_strategies = [proto_strategies]
            
            # Load model + datasets
            model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)
    
            # Minimal TrainAE to encode sets
    
            train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                            log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
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
    
            # PCA fit (same for all strategies)
            n_comp = min(3, all_encs.shape[1])
            pca = PCA(n_components=n_comp)
            encs_pca = pca.fit_transform(all_encs)
            explained = pca.explained_variance_ratio_ * 100.0
    
            # Create subplots for each strategy
            n_strategies = len(proto_strategies)
            fig, axes = plt.subplots(1, n_strategies, figsize=(6 * n_strategies, 5))
            
            # Handle single subplot case
            if n_strategies == 1:
                axes = [axes]
            
            # Different markers for each strategy to make differences visible
            strategy_markers = {'mean': 'X', 'kmeans': '*', 'gmm': 'P'}
            strategy_sizes = {'mean': 300, 'kmeans': 500, 'gmm': 400}
            
            for ax_idx, proto_strategy in enumerate(proto_strategies):
                ax = axes[ax_idx]
                
                # Compute prototypes for this strategy
                proto_dict = compute_prototypes_by_strategy(
                    all_encs, all_cats, proto_strategy, proto_components, random_state=1
                )
                
                # Flatten prototypes for PCA
                proto_arr = None
                proto_colors = None
                if proto_dict:
                    proto_list = []
                    proto_colors = []
                    for cls_id in sorted(proto_dict.keys()):
                        for proto_vec, comp_idx in proto_dict[cls_id]:
                            proto_list.append(proto_vec)
                            proto_colors.append(cls_id)
                    if proto_list:
                        proto_arr = np.stack(proto_list)
                        proto_colors = np.array(proto_colors)
                
                proto_pca = pca.transform(proto_arr) if proto_arr is not None else None
                
                # Plot encodings
                scatter = ax.scatter(encs_pca[:, 0], encs_pca[:, 1], c=all_cats, cmap='tab20', alpha=0.5, s=20)
                
                # Plot prototypes with strategy-specific markers and color by class
                if proto_pca is not None:
                    marker = strategy_markers.get(proto_strategy, '*')
                    marker_size = strategy_sizes.get(proto_strategy, 500)
                    ax.scatter(proto_pca[:, 0], proto_pca[:, 1], marker=marker, c=proto_colors, 
                              cmap='tab20', s=marker_size, edgecolors='black', linewidths=1.5, zorder=5)
                    # Add count info to title
                    n_protos = len(proto_pca)
                    ax.set_title(f'{proto_strategy.upper()}\n({n_protos} prototypes total, {proto_components} per class)')
                else:
                    ax.set_title(f'{proto_strategy.upper()}\n({proto_components} per class)')
                
                ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)')
                if n_comp > 1:
                    ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)')
                ax.grid(True, alpha=0.3)
            
            # Add colorbar for the whole figure
            cbar = plt.colorbar(scatter, ax=axes[-1], pad=0.02)
            cbar.set_label('Class ID')
            
            fig.suptitle(f'PCA with Prototypes (train/valid/test)', fontsize=14, y=1.02)
            plt.tight_layout()
            
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
    
            # Prototype strategy controls
            pca_cols = st.columns([2, 2, 2])
            with pca_cols[0]:
                proto_strategies = st.multiselect(
                    "Prototype Aggregation",
                    options=["mean", "kmeans", "gmm"],
                    default=["mean"],
                    key="pca_proto_strategy",
                    help="Select one or more: mean (single average per class) | kmeans (k centers per class) | gmm (gaussian mixture components per class)"
                )
            with pca_cols[1]:
                proto_components = st.slider(
                    "Components per Class",
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1,
                    key="pca_proto_components",
                    help="Number of prototypes (centroids/components) per class. Set > 1 to see differences between strategies"
                )
            with pca_cols[2]:
                st.empty()  # Spacer
    
            # Ensure at least one strategy is selected
            if not proto_strategies:
                st.warning("Please select at least one prototype aggregation strategy")
                proto_strategies = ["mean"]
    
            # Check if we have PCA cached for the current model
            strategies_str = "_".join(sorted(proto_strategies))
            current_model_key = f"{args.task}_{os.path.basename(args.path) if args.path else 'no_path'}_{args.new_size}_{args.n_neighbors}_{args.dist_fct}_{strategies_str}_{proto_components}"
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
                            _compute_pca_for_args(args, proto_strategies=proto_strategies, proto_components=proto_components)
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
    
    # ========================= TAB 2: Learned Embedding Classification ========================= #
    with tab2:
            # ========================= EMBEDDING-BASED CLASSIFIER OPTIMIZATION ========================= #
            st.header("🧠 Learned Embedding Classification")
            st.caption("This section optimizes classifiers using embeddings from the trained deep learning model.")
            
            # Admin: Button for learning with top N models
            if is_admin:
                st.divider()
                col_top_n_label, col_top_n_input, col_top_n_force, col_top_n_btn = st.columns([1.5, 1, 1, 2])
                with col_top_n_label:
                    st.markdown("**🎯 Learn with Top N Models:**")
                with col_top_n_input:
                    top_n = st.number_input("Number of top models", min_value=1, max_value=50, value=3, step=1, key="top_n_models")
                with col_top_n_force:
                    st.checkbox("Force", value=False, key="top_n_force_learning", help="Recompute even when a cache entry exists.")
                with col_top_n_btn:
                    if st.button("🚀 Run Learning for Top N Models", key="learn_top_n"):
                        st.session_state["run_learning_top_n_requested"] = int(top_n)
                        st.success(f"✅ Queued learning for top {top_n} models. It will start below using the selected n_aug values.")
            n_aug_values = []
            batch_k_max = 20
            
            # Add n_aug control with comma-separated custom values
            # Only show N_Aug controls if in N_Aug Mode
            if st.session_state.get('opt_mode', 'N_Aug Mode') == 'N_Aug Mode':
                col_naug_label, col_naug_input = st.columns([1.5, 2])
                
                with col_naug_label:
                    st.markdown("**N_Aug Values (comma-separated):**")
                
                with col_naug_input:
                    n_aug_input = st.text_input(
                        "Enter augmentation levels",
                        value="0,1,10,100",
                        key="tab1_n_aug_input",
                        help="Enter comma-separated values (e.g., 0,1,5,10,50,100). Will test each value."
                    )
                
                # Parse and validate input
                try:
                    n_aug_values = [int(x.strip()) for x in n_aug_input.split(',') if x.strip()]
                    n_aug_values = sorted(list(set(n_aug_values)))  # Remove duplicates and sort
                    if not n_aug_values or any(v < 0 for v in n_aug_values):
                        st.error("N_Aug values must be non-negative integers")
                        n_aug_values = []
                    else:
                        st.success(f"✅ Will test n_aug = {n_aug_values}")
                except ValueError:
                    st.error("Invalid input. Please enter comma-separated integers (e.g., 0,1,10,100)")
                    n_aug_values = []
            
            # Buttons for batch operations
            col_k_cfg_label, col_k_cfg_input = st.columns([1.5, 2])
            with col_k_cfg_label:
                st.markdown("**KNN Max k (batch optimize):**")
            with col_k_cfg_input:
                batch_k_max = int(st.number_input(
                    "Max k for Optimize All / Force Reoptimize",
                    min_value=2,
                    max_value=500,
                    value=20,
                    step=1,
                    key="tab1_batch_k_max",
                    help="Default is 20. Increase this to test more neighbors in batch optimization."
                ))

            def _args_from_tab2_model_row(row_dict):
                local_args = argparse.Namespace(**vars(args))
                row = dict(row_dict)
                parsed = extract_params_from_log_path(row.get("Log Path") or "")
                row.update({k: v for k, v in parsed.items() if v is not None})
                if row.get("Model Name") is not None:
                    local_args.model_name = row.get("Model Name")
                if row.get("NSize") is not None:
                    local_args.new_size = ensure_int(row.get("NSize"))
                if row.get("FGSM") is not None:
                    local_args.fgsm = row.get("FGSM")
                if row.get("Normalize") is not None:
                    local_args.normalize = row.get("Normalize")
                if row.get("N_Calibration") is not None:
                    local_args.n_calibration = row.get("N_Calibration")
                if row.get("Classif_Loss") is not None:
                    local_args.classif_loss = row.get("Classif_Loss")
                if row.get("DLoss") is not None:
                    local_args.dloss = row.get("DLoss")
                if row.get("Dist_Fct") is not None:
                    local_args.dist_fct = row.get("Dist_Fct")
                if row.get("Prototypes") is not None:
                    local_args.prototypes_to_use = row.get("Prototypes")
                if row.get("NPos") is not None:
                    local_args.n_positives = ensure_int(row.get("NPos"))
                if row.get("NNeg") is not None:
                    local_args.n_negatives = ensure_int(row.get("NNeg"))
                if row.get("N_Neighbors") is not None:
                    local_args.n_neighbors = ensure_int(row.get("N_Neighbors"))
                local_args.model_id = row.get("Model ID") or row.get("Model_ID") or getattr(local_args, "model_id", None)
                dataset = row.get("Dataset") or parsed.get("Dataset")
                if dataset:
                    local_args.path = os.path.join("data", dataset) if not str(dataset).startswith("data/") else dataset
                return local_args

            if is_admin and st.session_state.pop("run_learning_top_n_requested", None):
                requested_top_n = int(st.session_state.get("top_n_models", 3))
                force_top_n = bool(st.session_state.get("top_n_force_learning", False))
                if not n_aug_values:
                    st.warning("No valid n_aug values selected. Enter values like 0,1,10,100 first.")
                else:
                    bm_table = st.session_state.get("best_models_table")
                    if bm_table is None or len(bm_table) == 0:
                        st.warning("No best_models_table available yet. Apply/select a model first so Quick Model Selection builds it.")
                    else:
                        top_models_df = bm_table.head(requested_top_n).copy()
                        total_jobs = len(top_models_df) * len(n_aug_values)
                        st.success(f"✅ Starting learning for top {len(top_models_df)} models × {len(n_aug_values)} n_aug values ({total_jobs} jobs)...")
                        progress_topn = st.progress(0.0)
                        status_topn = st.empty()
                        log_topn = st.empty()
                        failures_topn = []
                        topn_logs = []
                        done_jobs = 0

                        def _topn_log(message: str):
                            """Write Top-N learning progress to Streamlit and terminal immediately."""
                            text = str(message)
                            print(text, flush=True)
                            topn_logs.append(text)
                            # Keep UI lightweight but still visibly updating.
                            log_topn.code("\n".join(topn_logs[-30:]))

                        _topn_log(
                            f"[TopN Learning] Starting top {len(top_models_df)} models × "
                            f"{len(n_aug_values)} n_aug values ({total_jobs} jobs), "
                            f"force={force_top_n}, max_k={int(batch_k_max)}"
                        )

                        for model_idx, (_, model_row) in enumerate(top_models_df.iterrows(), start=1):
                            local_args_topn = _args_from_tab2_model_row(model_row.to_dict())
                            model_num = model_row.get("#", model_idx)
                            model_id = getattr(local_args_topn, 'model_id', '?')
                            model_name = getattr(local_args_topn, 'model_name', '?')
                            model_label = f"model {model_idx}/{len(top_models_df)} | #{model_num} | id={model_id} | {model_name}"
                            _topn_log(f"[TopN Learning] Entering {model_label}")

                            for n_aug_opt in n_aug_values:
                                job_num = done_jobs + 1
                                job_prefix = f"[TopN Learning] Job {job_num}/{total_jobs} | {model_label} | n_aug={n_aug_opt}"
                                status_topn.info(job_prefix)
                                _topn_log(f"{job_prefix} | checking cache...")

                                try:
                                    local_args_topn.n_aug = int(n_aug_opt)
                                    cached_entry = None
                                    if not force_top_n:
                                        try:
                                            cache_topn = _load_optimization_cache(local_args_topn)
                                            cached_entry = cache_topn.get(int(n_aug_opt))
                                            _topn_log(
                                                f"{job_prefix} | cache entry: "
                                                f"{'found' if cached_entry is not None else 'missing'}"
                                            )
                                        except Exception as cache_exc:
                                            cached_entry = None
                                            _topn_log(f"{job_prefix} | cache read failed, will compute: {cache_exc}")
                                    else:
                                        _topn_log(f"{job_prefix} | force=True, skipping cache check result and recomputing")

                                    if cached_entry is not None:
                                        cache_complete = _is_cache_complete(cached_entry, int(batch_k_max))
                                        _topn_log(f"{job_prefix} | cache_complete={cache_complete}")
                                        if cache_complete:
                                            best_cached = cached_entry.get('best_k', '?') if isinstance(cached_entry, dict) else '?'
                                            mcc_cached = cached_entry.get('best_mcc', '?') if isinstance(cached_entry, dict) else '?'
                                            _topn_log(f"{job_prefix} | skipped cached result: best={best_cached}, MCC={mcc_cached}")
                                            done_jobs += 1
                                            progress_topn.progress(min(1.0, done_jobs / max(1, total_jobs)))
                                            status_topn.success(f"Skipped cached {done_jobs}/{total_jobs}: {model_name}, n_aug={n_aug_opt}")
                                            continue

                                        missing = _what_is_missing(cached_entry, int(batch_k_max)) if isinstance(cached_entry, dict) else {}
                                        _topn_log(f"{job_prefix} | cache incomplete, missing={missing}. Computing missing/full job.")

                                    _topn_log(f"{job_prefix} | computing prototypes...")
                                    _compute_and_save_prototypes_with_naug(local_args_topn, int(n_aug_opt), force_recompute=force_top_n)

                                    _topn_log(f"{job_prefix} | optimizing learned heads...")
                                    best_k, best_mcc, mcc_curve, all_results = _optimize_k_for_args(
                                        local_args_topn, 1, int(batch_k_max)
                                    )
                                    _save_optimization_result(local_args_topn, n_aug_opt, best_k, best_mcc, all_results)
                                    _topn_log(f"{job_prefix} | saved: best={best_k}, MCC={best_mcc:.4f}")

                                except Exception as exc:
                                    err_msg = (
                                        f"model_id={getattr(local_args_topn, 'model_id', '?')}, "
                                        f"model={getattr(local_args_topn, 'model_name', '?')}, "
                                        f"n_aug={n_aug_opt}: {exc}"
                                    )
                                    failures_topn.append(err_msg)
                                    _topn_log(f"{job_prefix} | FAILED: {exc}")

                                done_jobs += 1
                                progress_topn.progress(min(1.0, done_jobs / max(1, total_jobs)))
                                status_topn.info(f"Processed {done_jobs}/{total_jobs} Top-N learning jobs")

                        progress_topn.progress(1.0)
                        _topn_log(f"[TopN Learning] Finished. processed={done_jobs}/{total_jobs}, failures={len(failures_topn)}")
                        if failures_topn:
                            st.warning("Some Top-N learning jobs failed:\n" + "\n".join(failures_topn[:20]))
                        else:
                            st.success("✅ Top-N learning completed.")
                        st.rerun()

        
            col_btn_recompute, col_btn_optimize, col_btn_force = st.columns([1, 1, 1])
            
            with col_btn_recompute:
                if st.button("🔄 Recompute All", key="tab1_recompute_all",
                             help="Recompute prototypes for all n_aug values"):
                    if n_aug_values:
                        with st.spinner(f"Computing prototypes for n_aug = {n_aug_values}..."):
                            for n_aug_opt in n_aug_values:
                                try:
                                    args.n_aug = int(n_aug_opt)
                                    _compute_and_save_prototypes_with_naug(args, int(n_aug_opt), force_recompute=True)
                                    st.success(f"✅ n_aug={n_aug_opt} done")
                                except Exception as e:
                                    st.error(f"❌ n_aug={n_aug_opt} failed: {e}")
                        # Force rerun to refresh UI and show latest optimization results
                        st.rerun()
                    else:
                        st.warning("No valid n_aug values to process")
            
            with col_btn_optimize:
                if st.button("⚡ Optimize All", key="tab1_optimize_all",
                             help="Run KNN optimization (k=1..Max k) + all classifiers for all n_aug values (skips fully cached)"):
                    if n_aug_values:
                        with st.spinner(f"Optimizing for n_aug = {n_aug_values}..."):
                            cache = _load_optimization_cache(args)
                            skipped = 0
                            computed = 0
                            for n_aug_opt in n_aug_values:
                                cached_entry = cache.get(int(n_aug_opt))
                                # Check per-component what's missing
                                missing = _what_is_missing(cached_entry, int(batch_k_max)) if cached_entry is not None else {'knn': True, 'baselines': True, 'prototypes': True}
                                if not any(missing.values()):
                                    st.info(f"↩️  n_aug={n_aug_opt}: Already fully cached (MCC={cached_entry.get('best_mcc'):.3f})")
                                    skipped += 1
                                    continue
        
                                # Report exactly what will be computed
                                parts_to_compute = [k for k, v in missing.items() if v]
                                if cached_entry is not None:
                                    st.info(f"⚠️  n_aug={n_aug_opt}: Computing missing parts: {parts_to_compute}")
                                
                                try:
                                    args.n_aug = int(n_aug_opt)
                                    _compute_and_save_prototypes_with_naug(args, int(n_aug_opt), force_recompute=False)
                                    # Only run the parts that are actually missing
                                    _, _, _, partial_results = _optimize_k_for_args(
                                        args, 1, int(batch_k_max),
                                        skip_knn=not missing['knn'],
                                        skip_baselines=not missing['baselines'],
                                        skip_prototypes=not missing['prototypes'],
                                    )
                                    # Merge partial results into existing cache entry (preserves existing parts)
                                    _merge_partial_optimization_result(args, n_aug_opt, partial_results)
                                    # Re-read to show updated best MCC
                                    updated_cache = _load_optimization_cache(args)
                                    updated_entry = updated_cache.get(int(n_aug_opt), {})
                                    st.success(f"✅ n_aug={n_aug_opt}: computed {parts_to_compute} → best MCC={updated_entry.get('best_mcc', '?')}")
                                    computed += 1
                                except Exception as e:
                                    st.error(f"❌ n_aug={n_aug_opt} failed: {e}")
                        st.info(f"💾 Done! Computed: {computed} new/partial, Skipped: {skipped} cached")
                    else:
                        st.warning("No valid n_aug values to process")
            
            with col_btn_force:
                if st.button("🔥 Force Reoptimize", key="tab1_force_optimize",
                             help="Force recompute ALL optimizations (KNN k=1..Max k + all classifiers) even if cached"):
                    if n_aug_values:
                        with st.spinner(f"Force optimizing for n_aug = {n_aug_values}..."):
                            computed = 0
                            for n_aug_opt in n_aug_values:
                                try:
                                    args.n_aug = int(n_aug_opt)
                                    # Ensure prototypes exist
                                    _compute_and_save_prototypes_with_naug(args, int(n_aug_opt), force_recompute=True)
                                    # Run optimization: tests KNN k=1..batch_k_max, all baselines, all prototype strategies
                                    best_k, best_mcc, mcc_curve, all_results = _optimize_k_for_args(args, 1, int(batch_k_max))
                                    # Save result with all classifier results (overwrites cache)
                                    _save_optimization_result(args, n_aug_opt, best_k, best_mcc, all_results)
                                    st.success(f"✅ n_aug={n_aug_opt}: {best_k} (MCC={best_mcc:.3f})")
                                    computed += 1
                                except Exception as e:
                                    st.error(f"❌ n_aug={n_aug_opt} failed: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                        st.success(f"💾 Done! Computed: {computed} optimizations")
                        st.rerun()
                    else:
                        st.warning("No valid n_aug values to process")
        
            
            # Display cached best scores for all n_aug values
            cache = _load_optimization_cache(args)
            if cache:
                st.markdown("---")
                st.markdown("**📊 Optimization Results Summary (Learned Embeddings):**")
                st.caption("Results based on deep learning model embeddings. For raw pixel classification, see the '🖼️ Raw Pixel Classification' tab.")
                
                # Find the absolute best across all n_aug values
                best_overall = None
                best_overall_mcc = -1
                for n_aug_val, result in cache.items():
                    mcc = result.get('best_mcc', -1)
                    if mcc > best_overall_mcc:
                        best_overall_mcc = mcc
                        best_overall = (n_aug_val, result)
                
                # Display best overall result prominently
                if best_overall:
                    n_aug_best, result_best = best_overall
                    best_k_str = str(result_best.get('best_k', '?'))
                    
                    # Extract strategy info
                    if isinstance(best_k_str, str) and best_k_str.startswith('protot_'):
                        parts = best_k_str.split('_')
                        strategy = parts[1] if len(parts) > 1 else 'unknown'
                        n_comp = parts[2] if len(parts) > 2 else '?'
                        approach_label = f"{strategy.upper()} Prototype (n_comp={n_comp})"
                    elif isinstance(best_k_str, str) and best_k_str.startswith('baseline_'):
                        baseline_name = best_k_str.replace('baseline_', '')
                        approach_label = BASELINE_DISPLAY_NAMES.get(baseline_name, baseline_name.upper())
                    else:
                        approach_label = f"KNN (k={best_k_str})"
                    
                    st.success(f"🏆 **Best Overall:** n_aug={n_aug_best} using {approach_label} → **MCC: {best_overall_mcc:.4f}**")
                    st.caption(f"Tested on: {result_best.get('timestamp', 'unknown')[:10]}")
                    
                    st.markdown("**Select plot to display:**")
                    plot_options = [
                        ("KNN", "knn"),
                        ("Prototypes KMEANS", "kmeans"),
                        ("Prototypes GMM", "gmm"),
                        ("Prototypes MEAN", "mean"),
                    ]
                    plot_labels = [x[0] for x in plot_options]
                    plot_keys = [x[1] for x in plot_options]
                    selected_plot = st.radio("Plot type", plot_labels, index=0, horizontal=True, key="knn_proto_plot_radio")
                    plot_idx = plot_labels.index(selected_plot)
                    plot_key = plot_keys[plot_idx]
                    
                    # Respect exactly the n_aug values entered by the user.
                    cache_by_int = {}
                    for k, v in cache.items():
                        try:
                            cache_by_int[int(k)] = v
                        except Exception:
                            continue
        
                    if n_aug_values:
                        filtered_cache = {k: cache_by_int[k] for k in n_aug_values if k in cache_by_int}
                        missing_n_aug = [k for k in n_aug_values if k not in cache_by_int]
                        if missing_n_aug:
                            st.warning(f"No cached result for selected n_aug: {missing_n_aug}")
                    else:
                        filtered_cache = cache_by_int
        
                    if not filtered_cache:
                        st.info("No matching cached n_aug values to plot for current selection.")
        
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        if plot_key == "knn" and filtered_cache:
                            plot_knn_mcc_curves(filtered_cache)
                        elif filtered_cache:
                            plot_prototype_mcc_curves(filtered_cache, plot_key)
                
                # Show summary table: Best per model type (including all variants)
                st.markdown("**Best Results per Model Type:**")
                
                # Aggregate best results per approach type across all n_aug values
                # Pre-populate ALL model types (even if not tested)
                approach_best = {}
                
                # Initialize KNN
                approach_best['KNN'] = {
                    'Approach': 'KNN',
                    'MCC': -1,
                    'Details': 'Not tested',
                    'Date': '-',
                    'Is Best': False,
                    'Batch Entropy': np.nan,
                    'Batch NMI': np.nan,
                    'Batch ARI': np.nan
                }
                
                # Initialize all prototype strategies
                for strategy in ['mean', 'kmeans', 'gmm']:
                    approach_best[f"Prototype: {strategy.upper()}"] = {
                        'Approach': f"Prototype: {strategy.upper()}",
                        'MCC': -1,
                        'Details': 'Not tested',
                        'Date': '-',
                        'Is Best': False,
                        'Batch Entropy': np.nan,
                        'Batch NMI': np.nan,
                        'Batch ARI': np.nan
                    }
                
                # Initialize all baselines
                for baseline_name, baseline_display in BASELINE_DISPLAY_NAMES.items():
                    approach_best[baseline_display] = {
                        'Approach': baseline_display,
                        'MCC': -1,
                        'Details': 'Not tested',
                        'Date': '-',
                        'Is Best': False,
                        'Batch Entropy': np.nan,
                        'Batch NMI': np.nan,
                        'Batch ARI': np.nan
                    }
                
                def _metric_payload(src):
                    src = src or {}
                    return {
                        'valid_mcc': src.get('valid_mcc', src.get('mcc', np.nan)),
                        'valid_acc': src.get('valid_acc', src.get('acc', np.nan)),
                        'test_acc': src.get('test_acc', np.nan),
                        'test_mcc': src.get('test_mcc', np.nan),
                        'valid_AUC': src.get('valid_AUC', src.get('valid_auc', np.nan)),
                        'test_AUC': src.get('test_AUC', src.get('test_auc', np.nan)),
                    }

                # Now populate with actual results from cache
                for n_aug_val, result in cache.items():
                    timestamp = result.get('timestamp', 'unknown')
                    best_k_overall = result.get('best_k')
                    best_mcc_overall = result.get('best_mcc')
                    
                    # Check KNN from detailed results
                    knn_data = result.get('knn', {})
                    mcc_list = knn_data.get('mcc_per_k', [])
                    batch_effects_data = result.get('batch_effects', {})
                    if mcc_list:
                        # Handle both formats: list of dicts [{'k': 1, 'mcc': 0.5}, ...] or list of floats [0.5, 0.6, ...]
                        valid_mccs_with_k = []
                        for k_idx, item in enumerate(mcc_list):
                            if isinstance(item, dict):
                                # Dict format: {'k': 1, 'valid_mcc': 0.5, 'train_mcc': 0.5}
                                mcc_val = item.get('valid_mcc')
                                k_val = item.get('k', k_idx + 1)
                                if mcc_val is not None and isinstance(mcc_val, (int, float)):
                                    valid_mccs_with_k.append((float(mcc_val), int(k_val)))
                            elif isinstance(item, (int, float)) and item is not None:
                                # Plain number format
                                valid_mccs_with_k.append((float(item), k_idx + 1))
                        
                        if valid_mccs_with_k:
                            best_knn_mcc, best_k = max(valid_mccs_with_k, key=lambda x: x[0])
                            best_knn_item = {}
                            for k_idx2, item2 in enumerate(mcc_list):
                                item_k = item2.get('k', k_idx2 + 1) if isinstance(item2, dict) else k_idx2 + 1
                                if int(item_k) == int(best_k):
                                    best_knn_item = item2 if isinstance(item2, dict) else {'valid_mcc': item2, 'k': int(best_k)}
                                    break
                            if best_knn_mcc > approach_best['KNN']['MCC']:
                                # Extract batch metrics for this k
                                batch_metrics = {'batch_entropy_norm': np.nan, 'batch_nmi': np.nan, 'batch_ari': np.nan}
                                knn_batch_list = batch_effects_data.get('knn_per_k', [])
                                for batch_item in knn_batch_list:
                                    if isinstance(batch_item, dict) and batch_item.get('k') == best_k:
                                        batch_metrics = {
                                            'batch_entropy_norm': batch_item.get('batch_entropy_norm', np.nan),
                                            'batch_nmi': batch_item.get('batch_nmi', np.nan),
                                            'batch_ari': batch_item.get('batch_ari', np.nan)
                                        }
                                        break
                                
                                approach_best['KNN'] = {
                                    'Approach': 'KNN',
                                    'MCC': float(best_knn_mcc),
                                    'Details': f"k={best_k}, n_aug={n_aug_val}",
                                    'Date': timestamp[:10],
                                    'Is Best': False,
                                    'Batch Entropy': batch_metrics['batch_entropy_norm'],
                                    'Batch NMI': batch_metrics['batch_nmi'],
                                    'Batch ARI': batch_metrics['batch_ari'],
                                    **_metric_payload(best_knn_item)
                                }
                    # Fallback: Check if best_k is a plain number (KNN was overall winner)
                    elif isinstance(best_k_overall, (int, float)) and best_mcc_overall is not None:
                        if float(best_mcc_overall) > approach_best['KNN']['MCC']:
                            # Extract batch metrics for this k
                            batch_metrics = {'batch_entropy_norm': np.nan, 'batch_nmi': np.nan, 'batch_ari': np.nan}
                            knn_batch_list = batch_effects_data.get('knn_per_k', [])
                            for batch_item in knn_batch_list:
                                if isinstance(batch_item, dict) and batch_item.get('k') == int(best_k_overall):
                                    batch_metrics = {
                                        'batch_entropy_norm': batch_item.get('batch_entropy_norm', np.nan),
                                        'batch_nmi': batch_item.get('batch_nmi', np.nan),
                                        'batch_ari': batch_item.get('batch_ari', np.nan)
                                    }
                                    break
                            
                            approach_best['KNN'] = {
                                'Approach': 'KNN',
                                'MCC': float(best_mcc_overall),
                                'Details': f"k={int(best_k_overall)}, n_aug={n_aug_val}",
                                'Date': timestamp[:10],
                                'Is Best': False,
                                'Batch Entropy': batch_metrics['batch_entropy_norm'],
                                'Batch NMI': batch_metrics['batch_nmi'],
                                'Batch ARI': batch_metrics['batch_ari'],
                                **_metric_payload((result.get('split_metrics', {}).get('knn_per_k', [{}]) or [{}])[0])
                            }
                    
                    # Check all prototype strategies
                    proto_data = result.get('prototypes', {})
                    for strategy in ['mean', 'kmeans', 'gmm']:
                        strat_data = proto_data.get(strategy, {})
                        if strat_data and 'best_mcc' in strat_data:
                            mcc_val = strat_data.get('best_mcc')
                            best_n_comp = strat_data.get('best_n_components', '?')
                            if mcc_val is not None and isinstance(mcc_val, (int, float)):
                                approach_key = f"Prototype: {strategy.upper()}"
                                if float(mcc_val) > approach_best[approach_key]['MCC']:
                                    # Extract batch metrics for this prototype strategy
                                    proto_batch_data = batch_effects_data.get('prototypes', {}).get(strategy, {})
                                    batch_metrics = {
                                        'batch_entropy_norm': proto_batch_data.get('batch_entropy_norm', np.nan),
                                        'batch_nmi': proto_batch_data.get('batch_nmi', np.nan),
                                        'batch_ari': proto_batch_data.get('batch_ari', np.nan)
                                    }
                                    
                                    approach_best[approach_key] = {
                                        'Approach': approach_key,
                                        'MCC': float(mcc_val),
                                        'Details': f"n_comp={best_n_comp}, n_aug={n_aug_val}",
                                        'Date': timestamp[:10],
                                        'Is Best': False,
                                        'Batch Entropy': batch_metrics['batch_entropy_norm'],
                                        'Batch NMI': batch_metrics['batch_nmi'],
                                        'Batch ARI': batch_metrics['batch_ari'],
                                        **_metric_payload(strat_data)
                                    }
                    
                    # Check all baselines
                    baseline_data = result.get('baselines', {})
                    for baseline_name, baseline_display in BASELINE_DISPLAY_NAMES.items():
                        b_data = baseline_data.get(baseline_name, {})
                        if b_data and 'mcc' in b_data:
                            mcc_val = b_data.get('mcc')
                            if mcc_val is not None and isinstance(mcc_val, (int, float)):
                                if float(mcc_val) > approach_best[baseline_display]['MCC']:
                                    # Extract batch metrics for this baseline
                                    baseline_batch_data = batch_effects_data.get('baselines', {}).get(baseline_name, {})
                                    batch_metrics = {
                                        'batch_entropy_norm': baseline_batch_data.get('batch_entropy_norm', np.nan),
                                        'batch_nmi': baseline_batch_data.get('batch_nmi', np.nan),
                                        'batch_ari': baseline_batch_data.get('batch_ari', np.nan)
                                    }
                                    
                                    approach_best[baseline_display] = {
                                        'Approach': baseline_display,
                                        'MCC': float(mcc_val),
                                        'Details': f"n_aug={n_aug_val}",
                                        'Date': timestamp[:10],
                                        'Is Best': False,
                                        'Batch Entropy': batch_metrics['batch_entropy_norm'],
                                        'Batch NMI': batch_metrics['batch_nmi'],
                                        'Batch ARI': batch_metrics['batch_ari'],
                                        **_metric_payload(b_data)
                                    }
                
                # Also check session state for recent optimization results (from "Optimize k on validation")
                if 'optimized_k_value' in st.session_state and 'k_opt_best_mcc' in st.session_state:
                    session_k = st.session_state.get('optimized_k_value')
                    session_mcc = st.session_state.get('k_opt_best_mcc')
                    session_n_aug = getattr(args, 'n_aug', 1)
                    
                    # Get all results including batch_effects from session state
                    session_all_results = st.session_state.get('k_opt_proto_results', {})
                    
                    # Update KNN if session state has better result
                    if isinstance(session_k, int) and session_mcc is not None:
                        if float(session_mcc) > approach_best['KNN']['MCC']:
                            # Extract batch metrics from session state
                            session_batch_effects = session_all_results.get('batch_effects', {})
                            knn_batch_list = session_batch_effects.get('knn_per_k', [])
                            batch_metrics = {'batch_entropy_norm': np.nan, 'batch_nmi': np.nan, 'batch_ari': np.nan}
                            for batch_item in knn_batch_list:
                                if isinstance(batch_item, dict) and batch_item.get('k') == session_k:
                                    batch_metrics = {
                                        'batch_entropy_norm': batch_item.get('batch_entropy_norm', np.nan),
                                        'batch_nmi': batch_item.get('batch_nmi', np.nan),
                                        'batch_ari': batch_item.get('batch_ari', np.nan)
                                    }
                                    break
                            
                            approach_best['KNN'] = {
                                'Approach': 'KNN',
                                'MCC': float(session_mcc),
                                'Details': f"k={session_k}, n_aug={session_n_aug}",
                                'Date': 'Recent',
                                'Is Best': False,
                                'Batch Entropy': batch_metrics['batch_entropy_norm'],
                                'Batch NMI': batch_metrics['batch_nmi'],
                                'Batch ARI': batch_metrics['batch_ari']
                            }
                    
                    # Check if session state has prototype results
                    if isinstance(session_all_results, dict):
                        proto_results = session_all_results.get('prototypes', {})
                        session_batch_effects = session_all_results.get('batch_effects', {})
                        for strategy in ['mean', 'kmeans', 'gmm']:
                            result = proto_results.get(strategy, {})
                            mcc_val = result.get('best_mcc')
                            best_n_comp = result.get('best_n_components', '?')
                            if mcc_val is not None and isinstance(mcc_val, (int, float)):
                                approach_key = f"Prototype: {strategy.upper()}"
                                if float(mcc_val) > approach_best[approach_key]['MCC']:
                                    # Extract batch metrics for this prototype strategy
                                    proto_batch_data = session_batch_effects.get('prototypes', {}).get(strategy, {})
                                    batch_metrics = {
                                        'batch_entropy_norm': proto_batch_data.get('batch_entropy_norm', np.nan),
                                        'batch_nmi': proto_batch_data.get('batch_nmi', np.nan),
                                        'batch_ari': proto_batch_data.get('batch_ari', np.nan)
                                    }
                                    
                                    approach_best[approach_key] = {
                                        'Approach': approach_key,
                                        'MCC': float(mcc_val),
                                        'Details': f"n_comp={best_n_comp}, n_aug={session_n_aug}",
                                        'Date': 'Recent',
                                        'Is Best': False,
                                        'Batch Entropy': batch_metrics['batch_entropy_norm'],
                                        'Batch NMI': batch_metrics['batch_nmi'],
                                        'Batch ARI': batch_metrics['batch_ari']
                                    }
                        
                        # Check baseline results from session state
                        baseline_results = session_all_results.get('baselines', {})
                        for baseline_name, baseline_display in BASELINE_DISPLAY_NAMES.items():
                            b_data = baseline_results.get(baseline_name, {})
                            if b_data and 'mcc' in b_data:
                                mcc_val = b_data.get('mcc')
                                if mcc_val is not None and isinstance(mcc_val, (int, float)):
                                    if float(mcc_val) > approach_best[baseline_display]['MCC']:
                                        # Extract batch metrics for this baseline
                                        baseline_batch_data = session_batch_effects.get('baselines', {}).get(baseline_name, {})
                                        batch_metrics = {
                                            'batch_entropy_norm': baseline_batch_data.get('batch_entropy_norm', np.nan),
                                            'batch_nmi': baseline_batch_data.get('batch_nmi', np.nan),
                                            'batch_ari': baseline_batch_data.get('batch_ari', np.nan)
                                        }
                                        
                                        approach_best[baseline_display] = {
                                            'Approach': baseline_display,
                                            'MCC': float(mcc_val),
                                            'Details': f"n_aug={session_n_aug}",
                                            'Date': 'Recent',
                                            'Is Best': False,
                                            'Batch Entropy': batch_metrics['batch_entropy_norm'],
                                            'Batch NMI': batch_metrics['batch_nmi'],
                                            'Batch ARI': batch_metrics['batch_ari']
                                        }
                
                # Mark overall best
                for approach_key in approach_best:
                    if approach_best[approach_key]['MCC'] > 0 and abs(approach_best[approach_key]['MCC'] - best_overall_mcc) < 1e-4:
                        approach_best[approach_key]['Is Best'] = '🏆'
                    else:
                        approach_best[approach_key]['Is Best'] = ''
                
                # Filter out models that were not tested (MCC = -1) and convert to DataFrame
                tested_approaches = {k: v for k, v in approach_best.items() if v['MCC'] >= 0}
                
                if tested_approaches:
                    approach_df = pd.DataFrame(list(tested_approaches.values()))

                    # Expose the metrics expected by the model-type summary table.
                    # The existing MCC in this table is the validation-selection MCC, so keep
                    # it as valid_mcc when no explicit split metric exists in the cache.
                    required_metric_cols = [
                        "valid_mcc", "valid_acc", "test_acc", "test_mcc",
                        "valid_AUC", "test_AUC",
                        "inf_mcc", "inf_acc", "inf_AUC",
                    ]
                    for col in required_metric_cols:
                        if col not in approach_df.columns:
                            approach_df[col] = np.nan
                    approach_df["valid_mcc"] = pd.to_numeric(approach_df["valid_mcc"], errors="coerce")
                    approach_df["valid_mcc"] = approach_df["valid_mcc"].fillna(pd.to_numeric(approach_df["MCC"], errors="coerce"))

                    def _head_config_from_approach_row(row):
                        import re
                        approach = str(row.get("Approach", ""))
                        details = str(row.get("Details", ""))
                        if approach == "KNN":
                            m = re.search(r"k\s*=\s*(\d+)", details)
                            return m.group(1) if m else None
                        if approach.startswith("Prototype:"):
                            strategy = approach.split(":", 1)[1].strip().lower()
                            m = re.search(r"n_comp\s*=\s*(\d+)", details)
                            n_comp = m.group(1) if m else "1"
                            return f"protot_{strategy}_{n_comp}"
                        reverse_baselines = {v: k for k, v in BASELINE_DISPLAY_NAMES.items()}
                        if approach in reverse_baselines:
                            return f"baseline_{reverse_baselines[approach]}"
                        return None

                    def _n_aug_from_details(details):
                        import re
                        m = re.search(r"n_aug\s*=\s*(\d+)", str(details))
                        return int(m.group(1)) if m else None

                    def _lookup_head_inference_metrics(head_config, n_aug_hint=None):
                        """Return inference metrics for this exact classification head.

                        Never fill from the average of another table. If the head-specific
                        metric does not exist in the optimization cache, leave inf_* empty.
                        """
                        if not head_config:
                            return {}
                        candidates = []
                        for n_aug_key, result in cache.items():
                            if not isinstance(result, dict):
                                continue
                            if n_aug_hint is not None:
                                try:
                                    if int(n_aug_key) != int(n_aug_hint):
                                        continue
                                except Exception:
                                    pass
                            inf_by_head = result.get("inference_metrics_by_head", {}) or {}
                            if str(head_config) in inf_by_head and isinstance(inf_by_head[str(head_config)], dict):
                                candidates.append((n_aug_key, inf_by_head[str(head_config)]))
                        if not candidates:
                            return {}
                        # Prefer the candidate with highest N, then highest inf_mcc.
                        def _rank_candidate(item):
                            metrics = item[1]
                            try:
                                n_val = float(metrics.get("N", 0) or 0)
                            except Exception:
                                n_val = 0
                            try:
                                mcc_val = float(metrics.get("inf_mcc", -999) if metrics.get("inf_mcc") is not None else -999)
                            except Exception:
                                mcc_val = -999
                            return (n_val, mcc_val)
                        return sorted(candidates, key=_rank_candidate, reverse=True)[0][1]

                    approach_df["Head Config"] = approach_df.apply(_head_config_from_approach_row, axis=1)
                    approach_df["Classification Head"] = approach_df["Head Config"].apply(_head_config_label_global)
                    for idx, row in approach_df.iterrows():
                        head_config = row.get("Head Config")
                        n_aug_hint = _n_aug_from_details(row.get("Details"))
                        inf_metrics = _lookup_head_inference_metrics(head_config, n_aug_hint=n_aug_hint)
                        if inf_metrics:
                            approach_df.at[idx, "inf_mcc"] = inf_metrics.get("inf_mcc", np.nan)
                            approach_df.at[idx, "inf_acc"] = inf_metrics.get("inf_acc", np.nan)
                            approach_df.at[idx, "inf_AUC"] = inf_metrics.get("inf_AUC", np.nan)
                            approach_df.at[idx, "inf_N"] = inf_metrics.get("N", np.nan)

                    preferred_cols = [
                        "Is Best", "Approach", "Classification Head", "Head Config", "Details",
                        "valid_mcc", "valid_acc", "test_acc", "test_mcc",
                        "valid_AUC", "test_AUC",
                        "inf_mcc", "inf_acc", "inf_AUC", "inf_N",
                        "MCC", "Date", "Batch Entropy", "Batch NMI", "Batch ARI",
                    ]
                    display_cols = [c for c in preferred_cols if c in approach_df.columns] + [
                        c for c in approach_df.columns if c not in preferred_cols
                    ]
                    approach_df = approach_df.sort_values('valid_mcc', ascending=False)
                    st.dataframe(approach_df[display_cols], use_container_width=True, hide_index=True)
                    st.caption(f"💾 Showing best result per model type from {len(cache)} cached optimization(s). Split columns are filled when present in cache; inference columns are filled after inference metrics are available.")
                else:
                    st.info("No model results available yet.")
                
                # Also show all n_aug configurations in expandable section
                with st.expander("📋 View All N_Aug Configurations"):
                    cache_data = []
                    for n_aug_val, result in sorted(cache.items()):
                        best_k = result.get('best_k', '?')
                        mcc = result.get('best_mcc', 0)
                        
                        # Extract strategy
                        if isinstance(best_k, str) and best_k.startswith('protot_'):
                            parts = best_k.split('_')
                            strategy = parts[1] if len(parts) > 1 else 'unknown'
                            approach = f"{strategy.upper()}"
                        elif isinstance(best_k, str) and best_k.startswith('baseline_'):
                            baseline_name = best_k.replace('baseline_', '')
                            approach = baseline_name.upper().replace('_', ' ')
                        else:
                            approach = "KNN"
                        
                        is_best = (mcc == best_overall_mcc)
                        
                        cache_data.append({
                            'N_Aug': n_aug_val,
                            'Approach': approach,
                            'Best K/Strategy': str(best_k),
                            'MCC': float(mcc),
                            'Is Best': '🏆' if is_best else '',
                            'Date': result.get('timestamp', 'unknown')[:10]
                        })
                    
                    cache_df = pd.DataFrame(cache_data)
                    cache_df = cache_df.sort_values('MCC', ascending=False)
                    st.dataframe(cache_df, use_container_width=True, hide_index=True)
                
                # Show detailed results for ALL models/approaches across all n_aug
                with st.expander("🔍 View All Models/Approaches (Detailed Results)"):
                    all_models_data = []
                    
                    for n_aug_val in sorted(cache.keys()):
                        result = cache[n_aug_val]
                        batch_effects = result.get('batch_effects', {})
                        
                        # KNN results (all k values tested)
                        knn_data = result.get('knn', {})
                        mcc_per_k = knn_data.get('mcc_per_k', [])
                        knn_batch_by_k = {
                            int(item.get('k')): item
                            for item in batch_effects.get('knn_per_k', [])
                            if isinstance(item, dict) and item.get('k') is not None
                        }
                        for k_idx, item in enumerate(mcc_per_k):
                            if isinstance(item, dict):
                                mcc_val = item.get('valid_mcc')
                                k_val = item.get('k', k_idx + 1)
                            else:
                                mcc_val = item
                                k_val = k_idx + 1
                            if isinstance(mcc_val, (int, float)):
                                bm = knn_batch_by_k.get(int(k_val), {})
                                all_models_data.append({
                                    'N_Aug': n_aug_val,
                                    'Model Type': 'KNN',
                                    'Configuration': f'k={int(k_val)}',
                                    'MCC': float(mcc_val),
                                    'Batch Entropy': bm.get('batch_entropy_norm', np.nan),
                                    'Batch NMI': bm.get('batch_nmi', np.nan),
                                    'Batch ARI': bm.get('batch_ari', np.nan),
                                    'Date': result.get('timestamp', 'unknown')[:10]
                                })
                        
                        # Prototype results (all strategies and components)
                        proto_data = result.get('prototypes', {})
                        for strategy in ['mean', 'kmeans', 'gmm']:
                            strat_data = proto_data.get(strategy, {})
                            if strat_data:
                                proto_bm = batch_effects.get('prototypes', {}).get(strategy, {})
                                per_components = strat_data.get('per_components', [])
                                if not per_components:
                                    per_components = strat_data.get('mcc_per_n_components', [])
                                for comp_info in per_components:
                                    if isinstance(comp_info, dict):
                                        n_comp = comp_info.get('n_components', '?')
                                        mcc_val = comp_info.get('mcc', 0)
                                        all_models_data.append({
                                            'N_Aug': n_aug_val,
                                            'Model Type': f'{strategy.upper()} Prototype',
                                            'Configuration': f'n_comp={n_comp}',
                                            'MCC': float(mcc_val),
                                            'Batch Entropy': proto_bm.get('batch_entropy_norm', np.nan),
                                            'Batch NMI': proto_bm.get('batch_nmi', np.nan),
                                            'Batch ARI': proto_bm.get('batch_ari', np.nan),
                                            'Date': result.get('timestamp', 'unknown')[:10]
                                        })
                        
                        # Baseline results
                        baseline_data = result.get('baselines', {})
                        for baseline_name in BASELINE_DISPLAY_NAMES.keys():
                            b_data = baseline_data.get(baseline_name, {})
                            if b_data and 'mcc' in b_data:
                                base_bm = batch_effects.get('baselines', {}).get(baseline_name, {})
                                all_models_data.append({
                                    'N_Aug': n_aug_val,
                                    'Model Type': BASELINE_DISPLAY_NAMES.get(baseline_name, baseline_name.upper()),
                                    'Configuration': '-',
                                    'MCC': float(b_data.get('mcc', 0)),
                                    'Batch Entropy': base_bm.get('batch_entropy_norm', np.nan),
                                    'Batch NMI': base_bm.get('batch_nmi', np.nan),
                                    'Batch ARI': base_bm.get('batch_ari', np.nan),
                                    'Date': result.get('timestamp', 'unknown')[:10]
                                })
                    
                    if all_models_data:
                        all_models_df = pd.DataFrame(all_models_data)
                        all_models_df = all_models_df.sort_values(['MCC', 'Model Type'], ascending=[False, True])
                        st.dataframe(all_models_df, use_container_width=True, hide_index=True)
                        st.caption(f"📊 Showing {len(all_models_data)} total configurations across {len(cache)} n_aug value(s)")
                    else:
                        st.info("No detailed model data available yet.")
                
            # Create visualization: Best MCC per model type (across all n_aug values)
            if len(cache) > 0:
                st.markdown("---")
                st.markdown("**📈 Best Performance per Model (Optimal N_Aug) - Using Learned Embeddings:**")
                st.caption("These results use embeddings from the trained deep learning model, not raw pixel data.")
                col_perf, col_empty = st.columns(2)
                with col_perf:
                    fig_naug, ax_naug = plt.subplots(figsize=(12, 7))
                    color_map = CLASSIFIER_COLORS
                    model_types = ['knn', 'mean', 'kmeans', 'gmm'] + sorted(BASELINE_DISPLAY_NAMES.keys())
                    model_best = {}
                    for model_type in model_types:
                        best_mcc = -1
                        best_n_aug = None
                        for n_aug_val in cache.keys():
                            result = cache[n_aug_val]
                            mcc = None
                            if model_type == 'knn':
                                knn_data = result.get('knn', {})
                                mcc_list = knn_data.get('mcc_per_k', [])
                                if mcc_list and isinstance(mcc_list, list):
                                    valid_mccs = []
                                    for item in mcc_list:
                                        if isinstance(item, dict):
                                            mcc_val = item.get('valid_mcc')
                                            if mcc_val is not None and isinstance(mcc_val, (int, float)):
                                                valid_mccs.append(float(mcc_val))
                                        elif isinstance(item, (int, float)) and item is not None:
                                            valid_mccs.append(float(item))
                                    if valid_mccs:
                                        mcc = max(valid_mccs)
                            elif model_type in ['mean', 'kmeans', 'gmm']:
                                proto_data = result.get('prototypes', {})
                                strat_data = proto_data.get(model_type, {})
                                if strat_data and 'best_mcc' in strat_data:
                                    mcc_val = strat_data.get('best_mcc')
                                    if mcc_val is not None and isinstance(mcc_val, (int, float)):
                                        mcc = float(mcc_val)
                            else:
                                baseline_data = result.get('baselines', {})
                                b_data = baseline_data.get(model_type, {})
                                if b_data and 'mcc' in b_data:
                                    mcc_val = b_data.get('mcc')
                                    if mcc_val is not None and isinstance(mcc_val, (int, float)):
                                        mcc = float(mcc_val)
                            if mcc is not None and mcc > best_mcc:
                                best_mcc = mcc
                                best_n_aug = n_aug_val
                        model_best[model_type] = {
                            'mcc': best_mcc if best_mcc >= 0 else 0,
                            'n_aug': best_n_aug
                        }
                    labels = []
                    mccs = []
                    colors = []
                    n_aug_labels = []
                    for model_type in model_types:
                        best_info = model_best[model_type]
                        mccs.append(best_info['mcc'])
                        n_aug_labels.append(f"n_aug={best_info['n_aug']}" if best_info['n_aug'] is not None else "N/A")
                        colors.append(color_map.get(model_type, '#95a5a6'))
                        if model_type == 'knn':
                            labels.append('KNN')
                        elif model_type in ['mean', 'kmeans', 'gmm']:
                            labels.append(f'{model_type.capitalize()}\nPrototype')
                        else:
                            labels.append(BASELINE_DISPLAY_SHORT.get(model_type, model_type.upper()))
                    x = np.arange(len(labels))
                    bars = ax_naug.bar(x, mccs, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
                    for i, (bar, mcc, n_aug_label) in enumerate(zip(bars, mccs, n_aug_labels)):
                        height = bar.get_height()
                        y_offset = 0.01 if height >= 0 else -0.01
                        v_align = 'bottom' if height >= 0 else 'top'
                        ax_naug.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{mcc:.3f}\n{n_aug_label}',
                                   ha='center', va=v_align, fontsize=9, fontweight='bold')
                    ax_naug.set_xticks(x)
                    ax_naug.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
                    ax_naug.set_xlabel('Model Type', fontsize=12, fontweight='bold')
                    ax_naug.set_ylabel('Best MCC Score', fontsize=12, fontweight='bold')
                    ax_naug.set_title('Model Comparison: Best MCC Achieved (with Optimal N_Aug)', fontsize=13, fontweight='bold')
                    if mccs:
                        y_min = min(mccs)
                        y_max = max(mccs)
                        if y_min == y_max:
                            pad = 0.1 if y_max == 0 else abs(y_max) * 0.15
                        else:
                            pad = (y_max - y_min) * 0.15
                        ax_naug.set_ylim([y_min - pad, y_max + pad])
                    else:
                        ax_naug.set_ylim([-0.1, 1.05])
                    ax_naug.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    st.pyplot(fig_naug, use_container_width=True)
                    plt.close(fig_naug)
                with col_empty:
                    pass
            else:
                # Single KNN Mode: simplified interface
                st.info("ℹ️ **Single KNN Mode** - Running one-shot optimization on current settings")
            col_k_range, col_k_btn = st.columns([1, 1])
            with col_k_range:
                min_k = st.number_input("Min k", min_value=1, max_value=50, value=1, step=1, key="tab1_min_k")
                max_k = st.number_input("Max k", min_value=2, max_value=200, value=20, step=1, key="tab1_max_k")
                if max_k <= min_k:
                    st.warning("Max k must be greater than min k")
            with col_k_btn:
                if st.button("Optimize k on validation", key="tab1_optimize_k_button"):
                    if not n_aug_values:
                        st.error("No valid n_aug values specified")
                    else:
                        try:
                            # Optimize for the first n_aug value in the list (for single optimization)
                            n_aug_single = n_aug_values[0]
                            args.n_aug = int(n_aug_single)
                            
                            # Ensure prototypes exist for this n_aug value
                            with st.spinner(f"Ensuring prototypes exist for n_aug={n_aug_single}..."):
                                _compute_and_save_prototypes_with_naug(args, int(n_aug_single), force_recompute=False)
                            
                            # Now run KNN optimization with the n_aug-specific prototypes
                            best_k, best_mcc, mcc_curve, all_results = _optimize_k_for_args(args, int(min_k), int(max_k))
                            # best_k is either an int (KNN k) or a string like "protot_kmeans_3"
                            st.success(f"✅ Best approach = {best_k} (Validation MCC: {best_mcc:.3f})")
                            
                            # Save optimization result to cache with all classifier results
                            _save_optimization_result(args, n_aug_single, best_k, best_mcc, all_results)
                            st.info("💾 Result saved to optimization cache!")
                            
                            # Store result in a non-widget key and trigger rerun to apply it
                            st.session_state['optimized_k_value'] = best_k  # Keep as original type (int or str)
                            st.session_state['k_opt_best_mcc'] = float(best_mcc)
                            st.session_state['k_opt_curve'] = mcc_curve
                            st.session_state['k_opt_model_key'] = current_model_key
                            st.session_state['k_opt_proto_results'] = all_results  # Store full results including batch_effects
                            
                            # Automatically apply the best result to sidebar
                            if isinstance(best_k, int) or (isinstance(best_k, str) and best_k.isdigit()):
                                # KNN is best
                                st.session_state['k_opt_current_selection'] = {
                                    'type': 'knn',
                                    'k': int(best_k),
                                    'mcc': float(best_mcc)
                                }
                            else:
                                # Prototype strategy is best (e.g., "protot_kmeans_3")
                                parts = best_k.split('_')
                                if len(parts) >= 3:
                                    strategy = parts[1]
                                    n_comp = int(parts[2])
                                    st.session_state['k_opt_current_selection'] = {
                                        'type': 'prototype',
                                        'strategy': strategy,
                                        'n_comp': n_comp,
                                        'mcc': float(best_mcc)
                                    }
                            
                            st.info("✅ Best result automatically applied! Check sidebar for details.")
                        except Exception as e:
                            st.error(f"K optimization failed: {e}")
            st.caption("Uses current sidebar model settings (dataset, size, normalize, etc.)")
        
            if 'optimized_k_value' in st.session_state and 'k_opt_best_mcc' in st.session_state:
                best_val = st.session_state['optimized_k_value']
                mcc_val = st.session_state['k_opt_best_mcc']
                msg = f"Last optimized: {best_val} (Validation MCC: {mcc_val:.3f})"
                st.info(msg)
                
                # Display prototype strategy results
                all_results = st.session_state.get('k_opt_proto_results', {})
                proto_results = all_results.get('prototypes', {}) if isinstance(all_results, dict) else {}
                baseline_results = all_results.get('baselines', {}) if isinstance(all_results, dict) else {}
                
                if proto_results or baseline_results:
                    st.markdown("**Classification Approaches Comparison:**")
                    
                    # Collect all results into one table
                    comparison_data = []
                    
                    # Add KNN result (from k optimization)
                    knn_mcc = st.session_state.get('k_opt_best_mcc')
                    best_k = st.session_state.get('optimized_k_value')
                    if knn_mcc is not None and best_k is not None:
                        comparison_data.append({
                            'Approach': 'KNN',
                            'Type': 'KNN',
                            'MCC': float(knn_mcc),
                            'Details': f'k={best_k}'
                        })
                    
                    # Add baseline results
                    for clf_name, result in baseline_results.items():
                        mcc_val = result.get('mcc')
                        if mcc_val is not None:
                            comparison_data.append({
                                'Approach': BASELINE_DISPLAY_NAMES.get(clf_name, clf_name.upper()),
                                'Type': 'Baseline',
                                'MCC': float(mcc_val),
                                'Details': '-'
                            })
                    
                    # Add prototype results
                    for strategy in ['mean', 'kmeans', 'gmm']:
                        result = proto_results.get(strategy, {})
                        mcc_val = result.get('best_mcc')
                        n_protos = result.get('n_prototypes', 0)
                        best_n_components = result.get('best_n_components')
        
                        # If best_mcc not computed yet, compute on-demand now
                        if mcc_val is None:
                            try:
                                best_info = _compute_best_proto_mcc_for_args(args, strategy, min_components=1, max_components=5)
                                mcc_val = best_info.get('best_mcc')
                                n_protos = best_info.get('n_prototypes', n_protos)
                                best_n_components = best_info.get('best_n_components', best_n_components)
                                # Persist the computed best into session to avoid recomputation
                                proto_results[strategy] = best_info
                                st.session_state['k_opt_proto_results'] = proto_results
                            except Exception:
                                mcc_val = None
        
                        if mcc_val is not None:
                            comparison_data.append({
                                'Approach': f'{strategy.upper()} Prototype',
                                'Type': 'Prototype',
                                'MCC': float(mcc_val),
                                'Details': f'n_comp={best_n_components}, n_proto={n_protos}'
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        comparison_df = comparison_df.sort_values('MCC', ascending=False)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # Create line plot comparing KNN curve with all strategies
                        mcc_curve = st.session_state.get('k_opt_curve', [])
                        if mcc_curve:
                            fig_compare, ax_compare = plt.subplots(figsize=(6, 3.5))
                            
                            # Plot KNN curve
                            curve_df = pd.DataFrame(mcc_curve)
                            curve_df = curve_df.sort_values('k')
                            ax_compare.plot(curve_df['k'], curve_df['mcc'], marker='o', linewidth=2.5, 
                                           markersize=7, label='KNN', color='#e74c3c', zorder=3)
                            
                            # Plot prototype strategies
                            colors = {'mean': '#3498db', 'kmeans': '#f39c12', 'gmm': '#2ecc71'}
                            for strategy in ['mean', 'kmeans', 'gmm']:
                                result = proto_results.get(strategy, {})
                                per_components = result.get('per_components', [])
                                if per_components:
                                    per_df = pd.DataFrame(per_components).sort_values('n_components')
                                    ax_compare.plot(per_df['n_components'], per_df['mcc'], marker='s', linewidth=2.5,
                                                  markersize=7, label=f'{strategy.capitalize()} Prototype', 
                                                  color=colors.get(strategy, '#95a5a6'), zorder=2)
                            
                            # Plot baseline classifiers as horizontal lines
                            for clf_name, result in baseline_results.items():
                                mcc_val = result.get('mcc')
                                if mcc_val is not None:
                                    ax_compare.axhline(y=mcc_val, color=CLASSIFIER_COLORS.get(clf_name, '#95a5a6'),
                                                     linestyle=':', linewidth=2, alpha=0.7,
                                                     label=BASELINE_DISPLAY_SHORT.get(clf_name, clf_name), zorder=1)
                            
                            best_knn_mcc = st.session_state.get('k_opt_best_mcc', 0)
                            ax_compare.axhline(y=best_knn_mcc, color='#e74c3c', linestyle='--', linewidth=1.5, 
                                              alpha=0.6, zorder=1)
                            
                            ax_compare.set_xlabel('k (KNN) / n_components (Prototypes)', fontsize=12, fontweight='bold')
                            ax_compare.set_ylabel('Validation MCC', fontsize=12, fontweight='bold')
                            ax_compare.set_title('All Classification Approaches: MCC Comparison', fontsize=13, fontweight='bold')
                            ax_compare.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2)
                            ax_compare.grid(True, alpha=0.3)
                            ax_compare.set_ylim([min(curve_df['mcc'].min() - 0.05, min([r.get('best_mcc', 1) for r in proto_results.values() if r.get('best_mcc')]) - 0.05), 
                                                min(1.0, max(curve_df['mcc'].max(), max([r.get('best_mcc', 0) for r in proto_results.values() if r.get('best_mcc')]) + 0.1))])
                            
                            plt.tight_layout()
                            st.pyplot(fig_compare, use_container_width=True)
                            plt.close(fig_compare)
                        
                        # Show detailed table
                        with st.expander("📊 Detailed Strategy Comparison"):
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
            if st.session_state.get('k_opt_curve') and st.session_state.get('k_opt_model_key') == current_model_key:
                try:
                    opt_val = st.session_state.get('optimized_k_value')
                    curve_df = pd.DataFrame(st.session_state['k_opt_curve'])
                    curve_df = curve_df.sort_values('k')
                    
                    all_results = st.session_state.get('k_opt_proto_results', {})
                    proto_results = all_results.get('prototypes', {}) if isinstance(all_results, dict) else {}
                    baseline_results = all_results.get('baselines', {}) if isinstance(all_results, dict) else {}
                    best_overall_mcc = st.session_state.get('k_opt_best_mcc', -1)
                    
                    # Create unified plot: KNN curve + prototype curves + baselines
                    fig_unified, ax_unified = plt.subplots(figsize=(6, 3.5))
                    
                    # Plot KNN curve
                    ax_unified.plot(curve_df['k'], curve_df['mcc'], marker='o', linewidth=2.5, 
                                   markersize=7, label='KNN', color='#e74c3c', zorder=3, alpha=0.9)
                    
                    # Plot prototype strategy curves
                    colors = {'mean': '#3498db', 'kmeans': '#f39c12', 'gmm': '#2ecc71'}
                    for strategy in ['mean', 'kmeans', 'gmm']:
                        result = proto_results.get(strategy, {})
                        per_components = result.get('per_components', [])
                        if per_components:
                            per_df = pd.DataFrame(per_components).sort_values('n_components')
                            ax_unified.plot(per_df['n_components'], per_df['mcc'], marker='s', linewidth=2.5,
                                          markersize=7, label=f'{strategy.capitalize()} Prototype', 
                                          color=colors.get(strategy, '#95a5a6'), zorder=2, alpha=0.9)
                    
                    # Add markers for baseline classifiers at x=0 (no hyperparameter)
                    for clf_name, result in baseline_results.items():
                        mcc_val = result.get('mcc')
                        if mcc_val is not None:
                            ax_unified.scatter([0], [mcc_val], s=200, marker='D',
                                              label=f'{BASELINE_DISPLAY_SHORT.get(clf_name, clf_name)}: {mcc_val:.3f}',
                                              color=CLASSIFIER_COLORS.get(clf_name, 'gray'), alpha=0.9, 
                                              zorder=4, edgecolors='black', linewidths=1.5)
                    
                    # Highlight overall best with a star marker
                    if isinstance(opt_val, int) or (isinstance(opt_val, str) and opt_val.isdigit()):
                        best_k_int = int(opt_val)
                        best_row = curve_df[curve_df['k'] == best_k_int]
                        if not best_row.empty and abs(float(best_row.iloc[0]['mcc']) - best_overall_mcc) < 1e-4:
                            ax_unified.scatter([best_k_int], [best_overall_mcc], color='gold', s=400, marker='*', 
                                             zorder=5, edgecolors='darkred', linewidths=2, label=f'⭐ Best: k={best_k_int} (MCC={best_overall_mcc:.3f})')
                    elif isinstance(opt_val, str) and opt_val.startswith('protot_'):
                        # Mark best prototype configuration
                        parts = opt_val.split('_')
                        if len(parts) >= 3:
                            strategy = parts[1]
                            n_comp = int(parts[2])
                            ax_unified.scatter([n_comp], [best_overall_mcc], color='gold', s=400, marker='*',
                                             zorder=5, edgecolors='darkred', linewidths=2, 
                                             label=f'⭐ Best: {strategy.capitalize()} n_comp={n_comp} (MCC={best_overall_mcc:.3f})')
                    elif isinstance(opt_val, str) and opt_val.startswith('baseline_'):
                        # Mark best baseline
                        baseline_name = opt_val.replace('baseline_', '')
                        ax_unified.scatter([0], [best_overall_mcc], color='gold', s=500, marker='*',
                                         zorder=6, edgecolors='darkred', linewidths=2.5,
                                         label=f'⭐ Best: {BASELINE_DISPLAY_SHORT.get(baseline_name, baseline_name)} (MCC={best_overall_mcc:.3f})')
                    
                    ax_unified.set_xlabel('k (KNN) / n_components (Prototypes) / 0 (Baselines)', fontsize=12, fontweight='bold')
                    ax_unified.set_ylabel('Validation MCC', fontsize=12, fontweight='bold')
                    ax_unified.set_title('All Classification Approaches: Performance Comparison', fontsize=14, fontweight='bold')
                    ax_unified.grid(True, alpha=0.3)
                    ax_unified.legend(loc='best', fontsize=9, framealpha=0.95, ncol=2)
                    
                    # Calculate ylim dynamically from all data
                    all_mccs = list(curve_df['mcc'])
                    for strategy in ['mean', 'kmeans', 'gmm']:
                        result = proto_results.get(strategy, {})
                        per_components = result.get('per_components', [])
                        if per_components:
                            all_mccs.extend([x['mcc'] for x in per_components])
                    for result in baseline_results.values():
                        if result.get('mcc') is not None:
                            all_mccs.append(result['mcc'])
                    
                    if all_mccs:
                        ax_unified.set_ylim([min(all_mccs) - 0.05, min(1.0, max(all_mccs) + 0.1)])
                    
                    plt.tight_layout()
                    st.pyplot(fig_unified, use_container_width=True)
                    plt.close(fig_unified)
                    
                    # Show results summary
                    st.markdown("**Optimization Results Summary:**")
                    summary_data = []
                    
                    # Add KNN data
                    best_knn_row = curve_df.loc[curve_df['mcc'].idxmax()]
                    summary_data.append({
                        'Approach': 'KNN',
                        'Type': 'Neighbor-based',
                        'Best Value': f"k={int(best_knn_row['k'])}",
                        'MCC': float(best_knn_row['mcc']),
                        'Is Overall Best': best_knn_row['mcc'] == best_overall_mcc
                    })
                    
                    # Add baseline classifiers
                    for clf_name, result in baseline_results.items():
                        mcc_val = result.get('mcc')
                        if mcc_val is not None:
                            summary_data.append({
                                'Approach': BASELINE_DISPLAY_NAMES.get(clf_name, clf_name.upper()),
                                'Type': 'Baseline',
                                'Best Value': '-',
                                'MCC': float(mcc_val),
                                'Is Overall Best': abs(mcc_val - best_overall_mcc) < 1e-4
                            })
                    
                    # Add prototype data
                    for strategy in ['mean', 'kmeans', 'gmm']:
                        result = proto_results.get(strategy, {})
                        mcc_val = result.get('best_mcc')
                        if mcc_val is not None:
                            n_comp = result.get('best_n_components', '?')
                            summary_data.append({
                                'Approach': f'{strategy.capitalize()} Prototype',
                                'Type': 'Prototype',
                                'Best Value': f'n_comp={n_comp}',
                                'MCC': float(mcc_val),
                                'Is Overall Best': abs(mcc_val - best_overall_mcc) < 1e-4
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Selection widget to choose which approach to use
                    st.markdown("---")
                    st.markdown("**🎯 Select Approach to Use:**")
                    
                    # Create options for selectbox
                    default_idx = 0
                    options_list = []
                    option_to_config = {}
                    
                    for idx, row in summary_df.iterrows():
                        approach = row['Approach']
                        approach_type = row['Type']
                        best_val = row['Best Value']
                        mcc = row['MCC']
                        is_best = row['Is Overall Best']
                        
                        # Create display label
                        best_marker = "✅ " if is_best else ""
                        if best_val == '-':
                            label = f"{best_marker}{approach} (MCC: {mcc:.4f})"
                        else:
                            label = f"{best_marker}{approach} - {best_val} (MCC: {mcc:.4f})"
                        options_list.append(label)
                        
                        # Store config for this option
                        if approach == 'KNN':
                            k_val = int(best_val.split('=')[1])
                            option_to_config[label] = {'type': 'knn', 'k': k_val}
                        elif approach_type == 'Baseline':
                            # Baseline classifiers - store for reference but won't apply to sidebar
                            baseline_key = approach.lower().replace(' ', '_').replace('logistic_regression', 'logreg').replace('naive_bayes', 'naive_bayes').replace('linear_svc', 'linear_svc')
                            option_to_config[label] = {'type': 'baseline', 'name': baseline_key}
                        elif 'Prototype' in approach:
                            strategy_name = approach.split()[0].lower()
                            n_comp = int(best_val.split('=')[1])
                            option_to_config[label] = {'type': 'prototype', 'strategy': strategy_name, 'n_comp': n_comp}
                        
                        if is_best:
                            default_idx = idx
                    
                    selected_option = st.selectbox(
                        "Choose which optimization result to apply:",
                        options=options_list,
                        index=default_idx,
                        key='opt_result_selector'
                    )
                    
                    # Get the config for selected option
                    selected_config = option_to_config.get(selected_option, {})
                    
                    if selected_config:
                        # Display info about selected choice
                        col_info1, col_info2 = st.columns([2, 1])
                        with col_info1:
                            if selected_config['type'] == 'knn':
                                st.info(f"📊 **Selected:** KNN with k={selected_config['k']}")
                                if st.button("✅ Apply KNN k to sidebar", key="apply_knn_btn"):
                                    st.session_state['n_neighbors_input'] = selected_config['k']
                                    st.session_state['k_opt_current_selection'] = {
                                        'type': 'knn',
                                        'k': selected_config['k'],
                                        'mcc': summary_df.iloc[summary_df.index[summary_df['Approach'] == 'KNN'].tolist()[0]]['MCC'] if 'KNN' in summary_df['Approach'].values else 0
                                    }
                                    st.session_state.selected_model_version = st.session_state.get('selected_model_version', 0) + 1
                                    st.success(f"✅ Applied: k={selected_config['k']} (now visible in sidebar)")
                            else:
                                st.info(f"📊 **Selected:** {selected_config['strategy'].upper()} Strategy with n_comp={selected_config['n_comp']}")
                                if st.button("✅ Apply Prototype Strategy", key="apply_proto_btn"):
                                    st.session_state['proto_strategies_checkbox'] = [selected_config['strategy']]
                                    st.session_state['pca_proto_components'] = selected_config['n_comp']
                                    st.session_state['selected_prototype_approach'] = f"protot_{selected_config['strategy']}_{selected_config['n_comp']}"
                                    proto_mcc = next((row['MCC'] for _, row in summary_df.iterrows() 
                                                    if row['Approach'] == f"{selected_config['strategy'].capitalize()} Strategy"), 0)
                                    st.session_state['k_opt_current_selection'] = {
                                        'type': 'prototype',
                                        'strategy': selected_config['strategy'],
                                        'n_comp': selected_config['n_comp'],
                                        'mcc': proto_mcc
                                    }
                                    st.success(f"✅ Applied: {selected_config['strategy']} with n_comp={selected_config['n_comp']} (now visible in sidebar)")
                    
                    # Expandable table with all k values
                    with st.expander("📋 KNN All k Values"):
                        display_df = curve_df.copy()
                        display_df['Is Best'] = (display_df['mcc'] == curve_df['mcc'].max())
                        st.dataframe(_arrow_safe_dataframe(display_df), use_container_width=True, hide_index=True)
                except Exception:
                    pass
            elif st.session_state.get('k_opt_curve'):
                st.info("k optimization shown is from a previous model selection. Re-run 'Optimize k' to refresh.")
        
        
        
        # ========================= TAB 7: Raw Pixel Data Classification ========================= #
    with tab7:
            
            # Raw Data Analysis Section (shown regardless of mode)
            # ============================================
            # Second visualization: Best Performance per Model (with Raw Data)
            st.header("🖼️ Raw Pixel Data Classification")
            st.markdown("**📊 Best Performance per Model (Raw Data - No Embeddings):**")
            st.caption("This section tests classifiers directly on raw pixel values, bypassing the deep learning model.")
            
            # ---- Raw Data Computation Options ---- #
            st.markdown("**⚙️ Raw Data Options:**")
            raw_opt_cols = st.columns([2, 2, 2, 2])
        
            with raw_opt_cols[0]:
                include_knn = st.checkbox(
                    "Include KNN",
                    value=st.session_state.get('raw_include_knn', True),
                    key="raw_include_knn_checkbox",
                    help="Test KNN classifier (k=1-20)"
                )
                st.session_state['raw_include_knn'] = include_knn
        
            with raw_opt_cols[1]:
                include_proto = st.checkbox(
                    "Include Prototypes",
                    value=st.session_state.get('include_prototypes', True),
                    key="raw_include_proto_checkbox",
                    help="Test mean/kmeans/gmm prototype strategies"
                )
                st.session_state['include_prototypes'] = include_proto
        
            with raw_opt_cols[2]:
                include_baseline = st.checkbox(
                    "Include Baselines",
                    value=st.session_state.get('include_baselines', True),
                    key="raw_include_baseline_checkbox",
                    help="Test 10 baseline classifiers (LogReg, NB, SVC, etc.)"
                )
                st.session_state['include_baselines'] = include_baseline
        
            with raw_opt_cols[3]:
                raw_n_aug_input = st.text_input(
                    "Raw N_Aug values",
                    value=st.session_state.get('raw_n_aug_input', "0"),
                    key="raw_n_aug_input",
                    help="Comma-separated n_aug values to test, e.g. 0,1,10"
                )
        
            # Parse raw n_aug values
            try:
                raw_n_aug_values = sorted(set(int(x.strip()) for x in str(raw_n_aug_input).split(",") if x.strip()))
                if any(v < 0 for v in raw_n_aug_values):
                    st.error("Raw N_Aug values must be non-negative integers")
                    raw_n_aug_values = []
                elif raw_n_aug_values:
                    st.success(f"✅ Raw n_aug values to test: {raw_n_aug_values}")
                else:
                    st.warning("Please enter at least one raw n_aug value")
            except ValueError:
                st.error("Invalid Raw N_Aug input. Example: 0,1,10")
                raw_n_aug_values = []
        
            # Advanced Options Section
            with st.expander("⚙️ Advanced Model Options", expanded=False):
                st.markdown("**Model Selection:**")
                model_options = [
                    'knn', 'mean', 'kmeans', 'gmm',
                    'decision_tree', 'gradient_boosting', 'lda', 'linear_svc', 'svc',
                    'logreg', 'naive_bayes', 'qda', 'random_forest', 'ridge'
                ]
                model_display_names = {
                    'knn': 'KNN',
                    'mean': 'Mean Prototype',
                    'kmeans': 'KMeans Prototype',
                    'gmm': 'GMM Prototype',
                    'decision_tree': 'Decision Tree',
                    'gradient_boosting': 'Gradient Boosting',
                    'lda': 'LDA',
                    'linear_svc': 'Linear SVC',
                    'svc': 'SVC',
                    'logreg': 'Logistic Regression',
                    'naive_bayes': 'Naive Bayes',
                    'qda': 'QDA',
                    'random_forest': 'Random Forest',
                    'ridge': 'Ridge Classifier'
                }
                # Use session_state to track selected models
                if 'advanced_selected_models' not in st.session_state:
                    # Default to KNN and basic classifiers for first use
                    st.session_state['advanced_selected_models'] = ['knn', 'logreg', 'ridge', 'naive_bayes']
                selected_models = st.session_state['advanced_selected_models']
                # Render model selection as checkboxes
                checkbox_cols = st.columns(4)
                for idx, model in enumerate(model_options):
                    col = checkbox_cols[idx % 4]
                    is_selected = model in selected_models
                    checked = col.checkbox(model_display_names.get(model, model), value=is_selected, key=f"model_checkbox_{model}")
                    if checked and not is_selected:
                        selected_models.append(model)
                    elif not checked and is_selected:
                        selected_models.remove(model)
                st.session_state['advanced_selected_models'] = selected_models.copy()
                # Show selected models as chips
                if selected_models:
                    st.markdown("Selected: " + ", ".join([model_display_names.get(m, m) for m in selected_models]))
                else:
                    st.markdown("No models selected.")
        
                # ...existing code...
            # Render hyperparameter menus for each selected model after the Advanced Model Options menu
            from otitenet.app.classifier_param_ui import (
                random_forest_params_ui,
                knn_params_ui,
                logreg_params_ui,
                ridge_params_ui,
                naive_bayes_params_ui,
                mean_prototype_params_ui,
                kmeans_prototype_params_ui,
                gmm_prototype_params_ui,
                decision_tree_params_ui,
                gradient_boosting_params_ui,
                lda_params_ui,
                linear_svc_params_ui,
                svc_params_ui,
                qda_params_ui
            )
            classifier_params = {}
            for model in selected_models:
                if is_admin:
                    # Admin: Read-only dropdown for model details
                    with st.expander(f"📋 {model_display_names.get(model, model)} Details (Read-Only)", expanded=False):
                        st.json({
                            'model': model,
                            'display_name': model_display_names.get(model, model),
                            'status': 'active' if model in selected_models else 'inactive'
                        })
                else:
                    # Non-admin: skip model details entirely
                    continue
                params = None
                type_map = None
                if model == 'knn':
                    params = knn_params_ui()
                    type_map = {'n_neighbors': int}
                elif model == 'mean':
                    params = mean_prototype_params_ui()
                elif model == 'kmeans':
                    params = kmeans_prototype_params_ui()
                elif model == 'gmm':
                    params = gmm_prototype_params_ui()
                elif model == 'decision_tree':
                    params = decision_tree_params_ui()
                    type_map = {'max_depth': int, 'min_samples_split': int, 'min_samples_leaf': int, 'max_features': str}
                elif model == 'gradient_boosting':
                    params = gradient_boosting_params_ui()
                    type_map = {'n_estimators': int, 'learning_rate': float, 'max_depth': int}
                elif model == 'lda':
                    params = lda_params_ui()
                elif model == 'linear_svc':
                    params = linear_svc_params_ui()
                    type_map = {'max_iter': int}
                elif model == 'svc':
                    params = svc_params_ui()
                    type_map = {'max_iter': int}
                elif model == 'logreg':
                    params = logreg_params_ui()
                    type_map = {'logreg_max_iter': int, 'C': float}
                elif model == 'naive_bayes':
                    params = naive_bayes_params_ui()
                    type_map = {'var_smoothing': float}
                elif model == 'qda':
                    params = qda_params_ui()
                    type_map = {'reg_param': float}
                elif model == 'random_forest':
                    params = random_forest_params_ui()
                    type_map = {'n_estimators': int, 'max_depth': int, 'min_samples_split': int, 'min_samples_leaf': int, 'max_features': str, 'random_state': int}
                elif model == 'ridge':
                    params = ridge_params_ui()
                    type_map = {'alpha': float}
                # Convert param values to correct types if type_map is provided
                if params:
                    typed_params = {}
                    for k, v in params.items():
                        if type_map and k in type_map:
                            try:
                                typed_params[k] = type_map[k](v) if v != '' else None
                            except Exception:
                                typed_params[k] = v
                        else:
                            typed_params[k] = v
                    # Force number of prototypes to be a string entry if present
                    for proto_key in ['n_prototypes', 'num_prototypes', 'n_components', 'prototype_components']:
                        if proto_key in typed_params:
                            typed_params[proto_key] = str(typed_params[proto_key])
                    
                    # Extract special parameters that need to be at top level for classifiers.py
                    # (e.g., max_iter for various classifiers, kernel for SVC, etc.)
                    if model == 'logreg' and 'logreg_max_iter' in typed_params:
                        classifier_params['logreg_max_iter'] = typed_params.pop('logreg_max_iter')
                    elif model == 'svc':
                        if 'max_iter' in typed_params:
                            classifier_params['svc_max_iter'] = typed_params.pop('max_iter')
                        if 'kernel' in typed_params:
                            classifier_params['svc_kernel'] = typed_params.pop('kernel')
                    elif model == 'linear_svc' and 'max_iter' in typed_params:
                        classifier_params['linearsvc_max_iter'] = typed_params.pop('max_iter')
                    elif model == 'random_forest':
                        if 'n_estimators' in typed_params:
                            classifier_params['rfc_n_estimators'] = typed_params.pop('n_estimators')
                        # Remove random_state as classifiers.py hardcodes it to 1
                        typed_params.pop('random_state', None)
                    elif model == 'gradient_boosting':
                        if 'n_estimators' in typed_params:
                            classifier_params['gbc_n_estimators'] = typed_params.pop('n_estimators')
                        
                    classifier_params[f'{model}_params'] = typed_params
    
            # Example: Split comma-separated values for grid search
            def get_grid_search_param_combinations(params_dict, type_map=None):
                """
                Given a dict of param_name: str_value (comma-separated),
                return a dict of param_name: list of values for grid search, converting to correct types if type_map is provided.
                type_map: dict of param_name: type (e.g., int, float, str)
                """
                grid_params = {}
                for k, v in params_dict.items():
                    vals = [s.strip() for s in v.split(',') if s.strip()]
                    if type_map and k in type_map:
                        try:
                            grid_params[k] = [type_map[k](x) for x in vals]
                        except Exception:
                            grid_params[k] = vals
                    else:
                        grid_params[k] = vals
                return grid_params
        
            # Example usage for grid search:
            # type_map = {'n_neighbors': int, 'max_depth': int, 'learning_rate': float, ...}
            # param_grid = get_grid_search_param_combinations(params_dict, type_map)
        
            
            col_raw_info, col_raw_btn = st.columns([3, 1])
            
            with col_raw_info:
                st.info("ℹ️ This section compares classifier performance on raw image data (no learned embeddings).")
            
            # Helpers for raw-data model caching
            def _raw_cache_dir(local_args):
                try:
                    ds_name = os.path.basename(str(getattr(local_args, 'path', './data')).rstrip('/'))
                    nsize = ensure_int(getattr(local_args, 'new_size', None))
                    base = os.path.join('logs', 'raw_data', ds_name)
                    if nsize:
                        base = os.path.join(base, f"size_{nsize}")
                    os.makedirs(base, exist_ok=True)
                    return base
                except Exception:
                    return os.path.join('logs', 'raw_data')
        
            def _save_baseline_models(cache_dir, baselines_dict):
                """Save baseline classifiers as they become available."""
                for name, data_b in baselines_dict.items():
                    try:
                        clf = data_b.get('classifier')
                        if clf is not None:
                            outp = os.path.join(cache_dir, f"baseline_{name}.joblib")
                            joblib.dump(clf, outp)
                    except Exception as e:
                        st.warning(f"Could not save baseline {name}: {e}")
        
            def _save_knn_model(cache_dir, train_raw, train_labels, best_k_val):
                """Fit and save KNN model immediately."""
                try:
                    if best_k_val is not None:
                        knn = fit_knn_classifier(train_raw, train_labels, n_neighbors=int(best_k_val))
                        outp_knn = os.path.join(cache_dir, f"knn_k{int(best_k_val)}.joblib")
                        joblib.dump(knn, outp_knn)
                        return outp_knn
                except Exception as e:
                    st.warning(f"Could not save KNN: {e}")
                return None
        
            def _save_raw_summary(cache_dir, n_aug, knn_results, baseline_results, proto_results, batch_effects=None):
                """Save raw-data results under a given n_aug key."""
                try:
                    pkl = os.path.join(cache_dir, 'raw_results.pkl')
        
                    if os.path.exists(pkl):
                        with open(pkl, 'rb') as fh:
                            full_cache = pickle.load(fh)
                    else:
                        full_cache = {}
        
                    # Backward compatibility: if old cache format is flat, wrap it
                    if not isinstance(full_cache, dict) or any(k in full_cache for k in ['knn', 'baselines', 'prototypes']):
                        full_cache = {}
        
                    full_cache[int(n_aug)] = {
                        'timestamp': datetime.now().isoformat(),
                        'n_aug': int(n_aug),
                        'knn': knn_results or {},
                        'baselines': {k: {'mcc': v.get('mcc')} for k, v in (baseline_results or {}).items()},
                        'prototypes': proto_results or {},
                        'batch_effects': batch_effects or {},
                    }
        
                    with open(pkl, 'wb') as fh:
                        pickle.dump(full_cache, fh)
        
                except Exception as save_exc:
                    st.warning(f"Could not save raw results summary: {save_exc}")
        
            def _get_progress_file(cache_dir):
                """Path to progress checkpoint file."""
                return os.path.join(cache_dir, '.progress.pkl')
        
            def _save_progress(cache_dir, stage, data):
                """Save progress checkpoint for resumable computation."""
                try:
                    with open(_get_progress_file(cache_dir), 'wb') as fh:
                        pickle.dump({'stage': stage, **data}, fh)
                except Exception:
                    pass
        
            def _clear_progress(cache_dir):
                """Clear progress checkpoint when computation completes."""
                try:
                    prog_file = _get_progress_file(cache_dir)
                    if os.path.exists(prog_file):
                        os.remove(prog_file)
                except Exception:
                    pass
        
            def _load_raw_results(cache_dir):
                try:
                    pkl = os.path.join(cache_dir, 'raw_results.pkl')
                    if os.path.exists(pkl):
                        with open(pkl, 'rb') as fh:
                            cache = pickle.load(fh)
        
                        # Convert legacy flat format to empty or wrapped form
                        if isinstance(cache, dict) and any(k in cache for k in ['knn', 'baselines', 'prototypes']):
                            cache = {
                                0: {
                                    'timestamp': 'legacy',
                                    'n_aug': 0,
                                    'knn': cache.get('knn', {}),
                                    'baselines': cache.get('baselines', {}),
                                    'prototypes': cache.get('prototypes', {}),
                                    'batch_effects': cache.get('batch_effects', {}),
                                }
                            }
                        return cache
                except Exception as load_exc:
                    st.warning(f"Could not load raw results cache: {load_exc}")
                return {}
        
            with col_raw_btn:
                do_skip = st.button("⚡ Optimize Raw (Skip Cached)", key="compute_raw_skip_cached_btn",
                                    help="Load cached raw results if available; otherwise compute and save")
                do_force = st.button("🔥 Force Retrain Raw", key="compute_raw_force_btn",
                                     help="Recompute raw classifiers and overwrite cache/models")
        
                if do_skip or do_force:
                    # Validate that models are selected
                    selected_models = st.session_state.get('advanced_selected_models', [])
                    if not selected_models:
                        st.error("❌ No models selected! Please select at least one model type (KNN, Baselines, or Prototypes) in the Advanced Model Options above.")
                        st.stop()
        
                    if not raw_n_aug_values:
                        st.error("❌ No valid raw n_aug values provided.")
                        st.stop()
                    
                    progress_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    try:
                        cache_dir = _raw_cache_dir(args)
                        existing_cache = _load_raw_results(cache_dir)
                        all_run_results = {}
                        total_runs = len(raw_n_aug_values)
        
                        for run_idx, n_aug_raw in enumerate(raw_n_aug_values):
                            run_label = f"({run_idx + 1}/{total_runs})"
                            stage_idx = 0
        
                            run_has_knn = 'knn' in selected_models and st.session_state.get('raw_include_knn', True)
                            baseline_selected = [m for m in selected_models if m in BASELINE_DISPLAY_NAMES]
                            run_has_baselines = bool(baseline_selected) and st.session_state.get('include_baselines', True)
                            proto_selected = [m for m in selected_models if m in ['mean', 'kmeans', 'gmm']]
                            run_has_prototypes = bool(proto_selected) and st.session_state.get('include_prototypes', True)
                            stage_count = 2 + int(run_has_knn) + int(run_has_baselines) + int(run_has_prototypes)
        
                            progress_placeholder.info(
                                f"📦 Processing raw n_aug={n_aug_raw} {run_label} | stage {stage_idx}/{stage_count}: preparing"
                            )
                            progress_bar.progress(int(((run_idx + (stage_idx / stage_count)) / total_runs) * 100))
        
                            if (not do_force) and int(n_aug_raw) in existing_cache:
                                progress_placeholder.info(
                                    f"↩️ Using cached result for raw n_aug={n_aug_raw} {run_label}"
                                )
                                all_run_results[int(n_aug_raw)] = existing_cache[int(n_aug_raw)]
                                progress_bar.progress(int((run_idx + 1) / total_runs * 100))
                                continue
        
                            progress_placeholder.info(
                                f"📥 Loading model/data for raw n_aug={n_aug_raw} {run_label}"
                            )
                            _, _, _, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(args)
                            stage_idx += 1
                            progress_bar.progress(int(((run_idx + (stage_idx / stage_count)) / total_runs) * 100))
        
                            train_raw = data['inputs']['train'].reshape(len(data['inputs']['train']), -1)
                            valid_raw = data['inputs']['valid'].reshape(len(data['inputs']['valid']), -1)
                            train_labels = data['labels']['train']
                            valid_labels = data['labels']['valid']
                            train_batches = np.asarray(data['batches']['train'])
                            valid_batches = np.asarray(data['batches']['valid'])
                            test_raw_data = data['inputs'].get('test', None)
                            test_raw = test_raw_data.reshape(len(test_raw_data), -1) if test_raw_data is not None and len(test_raw_data) > 0 else np.empty((0, train_raw.shape[1]), dtype=train_raw.dtype)
                            test_batches_data = data['batches'].get('test', None)
                            test_batches = np.asarray(test_batches_data) if test_batches_data is not None else np.empty((0,), dtype=valid_batches.dtype)
        
                            all_raw_parts = [arr for arr in [train_raw, valid_raw, test_raw] if isinstance(arr, np.ndarray) and arr.shape[0] > 0]
                            all_batch_parts = [arr for arr in [train_batches, valid_batches, test_batches] if isinstance(arr, np.ndarray) and arr.shape[0] > 0]
                            all_dataset_raw = np.concatenate(all_raw_parts, axis=0) if all_raw_parts else valid_raw
                            all_dataset_batches = np.concatenate(all_batch_parts, axis=0) if all_batch_parts else valid_batches
        
                            all_results_raw = {
                                'baselines': {},
                                'prototypes': {},
                                'knn': {},
                                'batch_effects': {'knn_per_k': [], 'baselines': {}, 'prototypes': {}}
                            }
        
                            if run_has_knn:
                                progress_placeholder.info(
                                    f"🤖 Running KNN for raw n_aug={n_aug_raw} {run_label}"
                                )
                                best_k_knn, best_mcc_knn, mcc_per_k = evaluate_knn_with_k_search(
                                    train_raw, train_labels, valid_raw, valid_labels, min_k=1, max_k=20
                                )
                                all_results_raw['knn'] = {'best_k': best_k_knn, 'best_mcc': best_mcc_knn, 'mcc_per_k': mcc_per_k}
        
                                max_k_eff = min(20, int(train_raw.shape[0]))
                                for k in range(1, max_k_eff + 1):
                                    try:
                                        knn_model = fit_knn_classifier(train_raw, train_labels, n_neighbors=k, metric='minkowski')
                                        knn_preds = knn_model.predict(all_dataset_raw)
                                        metrics_k = _compute_batch_effect_from_predictions(knn_preds, all_dataset_batches)
                                        all_results_raw['batch_effects']['knn_per_k'].append({'k': int(k), **metrics_k})
                                    except Exception:
                                        continue
                                _save_knn_model(cache_dir, train_raw, train_labels, best_k_knn)
                                stage_idx += 1
                                progress_bar.progress(int(((run_idx + (stage_idx / stage_count)) / total_runs) * 100))
        
                            if run_has_baselines:
                                progress_placeholder.info(
                                    f"🧪 Running baselines ({', '.join(baseline_selected)}) for raw n_aug={n_aug_raw} {run_label}"
                                )
                                baseline_classifier_params = {}
                                for model in baseline_selected:
                                    params_key = f"{model}_params"
                                    if params_key in classifier_params:
                                        baseline_classifier_params[params_key] = classifier_params[params_key]
                                    if model == 'logreg' and 'logreg_max_iter' in classifier_params:
                                        baseline_classifier_params['logreg_max_iter'] = classifier_params['logreg_max_iter']
                                    elif model == 'svc':
                                        if 'svc_max_iter' in classifier_params:
                                            baseline_classifier_params['svc_max_iter'] = classifier_params['svc_max_iter']
                                        if 'svc_kernel' in classifier_params:
                                            baseline_classifier_params['svc_kernel'] = classifier_params['svc_kernel']
                                    elif model == 'linear_svc' and 'linearsvc_max_iter' in classifier_params:
                                        baseline_classifier_params['linearsvc_max_iter'] = classifier_params['linearsvc_max_iter']
                                    elif model == 'random_forest' and 'rfc_n_estimators' in classifier_params:
                                        baseline_classifier_params['rfc_n_estimators'] = classifier_params['rfc_n_estimators']
                                    elif model == 'gradient_boosting' and 'gbc_n_estimators' in classifier_params:
                                        baseline_classifier_params['gbc_n_estimators'] = classifier_params['gbc_n_estimators']
        
                                baseline_results = evaluate_baseline_classifiers(
                                    train_raw,
                                    train_labels,
                                    valid_raw,
                                    valid_labels,
                                    progress_placeholder,
                                    baseline_classifier_params
                                )
                                all_results_raw['baselines'] = {k: v for k, v in baseline_results.items() if k in baseline_selected}
                                for baseline_name, baseline_data in all_results_raw['baselines'].items():
                                    clf_obj = baseline_data.get('classifier', None) if isinstance(baseline_data, dict) else None
                                    if clf_obj is None:
                                        continue
                                    try:
                                        baseline_preds = clf_obj.predict(all_dataset_raw)
                                        all_results_raw['batch_effects']['baselines'][baseline_name] = _compute_batch_effect_from_predictions(
                                            baseline_preds, all_dataset_batches
                                        )
                                    except Exception:
                                        continue
                                _save_baseline_models(cache_dir, all_results_raw['baselines'])
                                stage_idx += 1
                                progress_bar.progress(int(((run_idx + (stage_idx / stage_count)) / total_runs) * 100))
        
                            if run_has_prototypes:
                                progress_placeholder.info(
                                    f"🧬 Running prototypes ({', '.join(proto_selected)}) for raw n_aug={n_aug_raw} {run_label}"
                                )
                                proto_results = optimize_prototype_components(
                                    train_raw, train_labels, valid_raw, valid_labels,
                                    strategies=proto_selected,
                                    max_components=10
                                )
                                all_results_raw['prototypes'] = {k: v for k, v in proto_results.items() if k in proto_selected}
                                for strategy_name, strategy_data in all_results_raw['prototypes'].items():
                                    if not isinstance(strategy_data, dict):
                                        continue
                                    best_n_components = strategy_data.get('best_n_components', None)
                                    if best_n_components is None:
                                        continue
                                    try:
                                        proto_dict = compute_prototypes_by_strategy(
                                            train_raw,
                                            train_labels,
                                            strategy=strategy_name,
                                            n_components=int(best_n_components),
                                            random_state=1,
                                        )
                                        proto_vecs, proto_labels = flatten_prototype_dict(proto_dict)
                                        if proto_vecs.size == 0:
                                            continue
                                        proto_knn = fit_knn_classifier(proto_vecs, proto_labels, n_neighbors=1, metric='minkowski')
                                        proto_preds = proto_knn.predict(all_dataset_raw)
                                        all_results_raw['batch_effects']['prototypes'][strategy_name] = _compute_batch_effect_from_predictions(
                                            proto_preds, all_dataset_batches
                                        )
                                    except Exception:
                                        continue
                                stage_idx += 1
                                progress_bar.progress(int(((run_idx + (stage_idx / stage_count)) / total_runs) * 100))
        
                            progress_placeholder.info(
                                f"💾 Saving cache entry for raw n_aug={n_aug_raw} {run_label}"
                            )
                            _save_raw_summary(
                                cache_dir=cache_dir,
                                n_aug=n_aug_raw,
                                knn_results=all_results_raw['knn'],
                                baseline_results=all_results_raw['baselines'],
                                proto_results=all_results_raw['prototypes'],
                                batch_effects=all_results_raw['batch_effects'],
                            )
                            stage_idx += 1
                            progress_bar.progress(int(((run_idx + (stage_idx / stage_count)) / total_runs) * 100))
        
                            all_run_results[int(n_aug_raw)] = {
                                'timestamp': datetime.now().isoformat(),
                                'n_aug': int(n_aug_raw),
                                **all_results_raw,
                            }
                            progress_bar.progress(int((run_idx + 1) / total_runs * 100))
        
                        _clear_progress(cache_dir)
                        st.session_state['raw_data_results'] = all_run_results
                        progress_placeholder.empty()
                        st.success("✅ Raw data classification complete for all requested n_aug values.")
                        st.rerun()
        
                    except Exception as e:
                        progress_placeholder.empty()
                        progress_bar.empty()
                        st.error(f"❌ Error computing raw data metrics: {e}")
                        st.error(traceback.format_exc())
                
                # Visualization for raw data results
                # Visualization for raw data results
                # Original single-run plot
                fig_raw, ax_raw = plt.subplots(figsize=(6, 3.5))
                cache = _load_raw_results(_raw_cache_dir(args))
                model_types = ['knn', 'mean', 'kmeans', 'gmm'] + sorted(BASELINE_DISPLAY_NAMES.keys())
                labels = []
                mccs = []
                colors = []
                n_aug_labels = []
                has_valid_mcc = False
                if cache:
                    for model_type in model_types:
                        best_mcc = -1
                        best_n_aug = None
                        for n_aug, result in cache.items():
                            mcc = None
                            # Defensive: ensure result is a dict
                            if not isinstance(result, dict):
                                continue
                            if model_type == 'knn':
                                knn_data = result.get('knn', {})
                                mcc_list = knn_data.get('mcc_per_k', [])
                                if mcc_list:
                                    valid_mccs = [float(item.get('valid_mcc')) for item in mcc_list if isinstance(item, dict) and item.get('valid_mcc') is not None]
                                    if valid_mccs:
                                        mcc = max(valid_mccs)
                                # fallback: try best_mcc
                                if mcc is None:
                                    mcc = knn_data.get('best_mcc')
                                    if mcc is not None:
                                        mcc = float(mcc)
                            elif model_type in ['mean', 'kmeans', 'gmm']:
                                proto_data = result.get('prototypes', {})
                                strat_data = proto_data.get(model_type, {})
                                if strat_data and 'best_mcc' in strat_data:
                                    val = strat_data.get('best_mcc')
                                    if val is not None and isinstance(val, (int, float)):
                                        mcc = float(val)
                            else:
                                baseline_data = result.get('baselines', {})
                                b_data = baseline_data.get(model_type, {})
                                if b_data and 'mcc' in b_data:
                                    val = b_data.get('mcc')
                                    if val is not None and isinstance(val, (int, float)):
                                        mcc = float(val)
                            if mcc is not None:
                                has_valid_mcc = True
                            if mcc is not None and mcc > best_mcc:
                                best_mcc = mcc
                                best_n_aug = n_aug
                        mccs.append(best_mcc if best_mcc > -1 else 0)
                        n_aug_labels.append(f"n_aug={best_n_aug}" if best_n_aug is not None else "n_aug=N/A")
                        colors.append(CLASSIFIER_COLORS.get(model_type, '#95a5a6'))
                        if model_type == 'knn':
                            labels.append('KNN')
                        elif model_type in ['mean', 'kmeans', 'gmm']:
                            labels.append(f'{model_type.capitalize()}\nPrototype')
                        else:
                            labels.append(BASELINE_DISPLAY_SHORT.get(model_type, model_type.upper()))
                    x = np.arange(len(labels))
                    bars = ax_raw.bar(x, mccs, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
                    for bar, mcc, n_aug_label in zip(bars, mccs, n_aug_labels):
                        height = bar.get_height()
                        if height > 0:
                            ax_raw.text(
                                bar.get_x() + bar.get_width()/2.,
                                height,
                                f'{mcc:.3f}\n{n_aug_label}',
                                ha='center',
                                va='bottom',
                                fontsize=8,
                                fontweight='bold'
                            )
                        elif height < 0:
                            ax_raw.text(
                                bar.get_x() + bar.get_width()/2.,
                                height,
                                f'{mcc:.3f}\n{n_aug_label}',
                                ha='center',
                                va='top',
                                fontsize=8,
                                fontweight='bold'
                            )
                    ax_raw.set_xticks(x)
                    ax_raw.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
                    ax_raw.set_xlabel('Model Type', fontsize=12, fontweight='bold')
                    ax_raw.set_ylabel('Best MCC Score', fontsize=12, fontweight='bold')
                    ax_raw.set_title('Model Comparison: Best MCC on Raw Data (Best per Model)', fontsize=13, fontweight='bold')
                    if mccs:
                        y_min = min(mccs)
                        y_max = max(mccs)
                        if y_min == y_max:
                            pad = 0.1 if y_max == 0 else abs(y_max) * 0.15
                        else:
                            pad = (y_max - y_min) * 0.15
                        ax_raw.set_ylim([y_min - pad, y_max + pad])
                    else:
                        ax_raw.set_ylim([-0.1, 0.2])
                    ax_raw.grid(True, alpha=0.3, axis='y')
                    # Show warning only when no valid MCC was found in cache at all
                    if not has_valid_mcc:
                        ax_raw.text(0.5, 0.5, 'Cache exists but no valid MCC scores found.\nCheck if optimization completed successfully.', 
                                   ha='center', va='center', fontsize=11, style='italic',
                                   transform=ax_raw.transAxes, color='red', alpha=0.7, 
                                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
                else:
                    ax_raw.text(0.5, 0.5, 'Raw Data Metrics\nNot Yet Computed\n\nClick "Compute Raw Data Metrics" to start', 
                               ha='center', va='center', fontsize=14, fontweight='bold',
                               transform=ax_raw.transAxes, color='gray', alpha=0.5)
                    ax_raw.set_xlim(0, 1)
                    ax_raw.set_ylim(0, 1)
                    ax_raw.axis('off')
                    ax_raw.set_title('Model Comparison: Best MCC on Raw Data', fontsize=13, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig_raw, use_container_width=True)
                plt.close(fig_raw)
        
                # Additional multi-n_aug plot
                fig_multi_naug, ax_multi_naug = plt.subplots(figsize=(7, 4))
                cache = _load_raw_results(_raw_cache_dir(args))
                n_aug_input = st.session_state.get('raw_n_aug_input', "0")
                if isinstance(n_aug_input, int):
                    n_aug_values = [n_aug_input]
                elif isinstance(n_aug_input, str):
                    n_aug_values = [int(x.strip()) for x in n_aug_input.split(',') if x.strip()]
                else:
                    n_aug_values = []
                model_types = ['knn', 'mean', 'kmeans', 'gmm'] + sorted(BASELINE_DISPLAY_NAMES.keys())
                color_map = CLASSIFIER_COLORS
                if cache and n_aug_values:
                    width = 0.8 / len(n_aug_values) if len(n_aug_values) > 1 else 0.6
                    x = np.arange(len(model_types))
                    has_valid_mcc_multi = False
                    for idx, n_aug in enumerate(n_aug_values):
                        result = cache.get(n_aug)
                        if not result:
                            continue
                        mccs = []
                        for model_type in model_types:
                            mcc = None
                            if model_type == 'knn':
                                knn_data = result.get('knn', {})
                                mcc_list = knn_data.get('mcc_per_k', [])
                                if mcc_list:
                                    valid_mccs = [float(item.get('valid_mcc')) for item in mcc_list if isinstance(item, dict) and item.get('valid_mcc') is not None]
                                    if valid_mccs:
                                        mcc = max(valid_mccs)
                            elif model_type in ['mean', 'kmeans', 'gmm']:
                                proto_data = result.get('prototypes', {})
                                strat_data = proto_data.get(model_type, {})
                                if strat_data and 'best_mcc' in strat_data:
                                    val = strat_data.get('best_mcc')
                                    if val is not None and isinstance(val, (int, float)):
                                        mcc = float(val)
                            else:
                                baseline_data = result.get('baselines', {})
                                b_data = baseline_data.get(model_type, {})
                                if b_data and 'mcc' in b_data:
                                    val = b_data.get('mcc')
                                    if val is not None and isinstance(val, (int, float)):
                                        mcc = float(val)
                            if mcc is not None:
                                has_valid_mcc_multi = True
                            mccs.append(mcc if mcc is not None else 0)
                        bars = ax_multi_naug.bar(x + idx * width, mccs, width=width, color=[color_map.get(mt, '#95a5a6') for mt in model_types], alpha=0.85, edgecolor='black', linewidth=1.2, label=f'n_aug={n_aug}')
                        for bar, mcc in zip(bars, mccs):
                            height = bar.get_height()
                            if height > 0:  # Only show label if bar has height
                                ax_multi_naug.text(bar.get_x() + bar.get_width()/2., height,
                                           f'{mcc:.3f}',
                                           ha='center', va='bottom', fontsize=8, fontweight='bold')
                    ax_multi_naug.set_xticks(x + width * (len(n_aug_values)-1)/2)
                    ax_multi_naug.set_xticklabels([BASELINE_DISPLAY_SHORT.get(mt, mt.upper()) if mt not in ['knn', 'mean', 'kmeans', 'gmm'] else (mt.capitalize() + ('\nPrototype' if mt in ['mean', 'kmeans', 'gmm'] else '')) for mt in model_types], fontsize=10, rotation=45, ha='right')
                    ax_multi_naug.set_xlabel('Model Type', fontsize=12, fontweight='bold')
                    ax_multi_naug.set_ylabel('Best MCC Score', fontsize=12, fontweight='bold')
                    ax_multi_naug.set_title('Model Comparison: Best MCC on Raw Data (Selected n_aug)', fontsize=13, fontweight='bold')
                    if ax_multi_naug.patches:
                        heights = [bar.get_height() for bar in ax_multi_naug.patches]
                        y_min = min(heights)
                        y_max = max(heights)
                        if y_min == y_max:
                            pad = 0.1 if y_max == 0 else abs(y_max) * 0.15
                        else:
                            pad = (y_max - y_min) * 0.15
                        ax_multi_naug.set_ylim([y_min - pad, y_max + pad])
                    else:
                        ax_multi_naug.set_ylim([-0.1, 0.2])
                    ax_multi_naug.grid(True, alpha=0.3, axis='y')
                    ax_multi_naug.legend()
                    # Show warning only when no valid MCC was found in cache at all
                    if not has_valid_mcc_multi:
                        ax_multi_naug.text(0.5, 0.5, 'Cache exists but no valid MCC scores found.\nCheck if optimization completed successfully.', 
                                   ha='center', va='center', fontsize=11, style='italic',
                                   transform=ax_multi_naug.transAxes, color='red', alpha=0.7, 
                                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
                else:
                    ax_multi_naug.text(0.5, 0.5, 'Raw Data Metrics\nNot Yet Computed\n\nSelect n_aug values and run computation', 
                               ha='center', va='center', fontsize=14, fontweight='bold',
                               transform=ax_multi_naug.transAxes, color='gray', alpha=0.5)
                    ax_multi_naug.set_xlim(0, 1)
                    ax_multi_naug.set_ylim(0, 1)
                    ax_multi_naug.axis('off')
                    ax_multi_naug.set_title('Model Comparison: Best MCC on Raw Data', fontsize=13, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig_multi_naug, use_container_width=True)
                plt.close(fig_multi_naug)
                
                # Additional visualization: Best Performance per Model Type (Raw Data) - similar to N_Aug version
            
        
        # ========================= TAB 3: Past Results ========================= #

with tab3:
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
                    run_analysis_on_file(filename, file_bytes, local_args_bulk, cursor, conn, force_reanalyze=False, show_validation_metrics=show_val, fast_infer=fast_infer)
                except Exception as e:  # keep iterating on failure
                    failures.append(f"Model {row.get('Model ID', row.get('Model_ID'))}: {e}")
                progress.progress((idx + 1) / total)

            status.write(f"Finished {label}")
            if failures:
                st.warning("Some analyses failed:\n" + "\n".join(failures))
            else:
                st.success("✅ Completed without errors")

        # ---- Bulk Analysis: all images x all best models (ADMIN ONLY) ---- #
        if is_admin:
            st.markdown("---")
            st.subheader("Bulk Analysis (Admin Only)")
            if 'best_models_table' in st.session_state and st.session_state['best_models_table'] is not None and not st.session_state['best_models_table'].empty:
                bm_table_tab2 = st.session_state['best_models_table']
                show_val_bulk = st.checkbox("Show validation metrics during bulk runs", value=False, key="bulk_show_val")
                fast_infer_bulk = st.checkbox("Use fast prototype inference (skip KNN fit)", value=True, key="bulk_fast_infer")
                compute_gradcam_bulk = st.checkbox("🧠 Compute Grad-CAM for all analyses (Admin Only)", value=True, key="bulk_gradcam")
                force_reanalyze_bulk = st.checkbox("🔄 Force reanalyze all images (skip existing results)", value=False, key="bulk_force_reanalyze")
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
                                    run_analysis_on_file(fname, file_bytes_all, local_args_bulk, cursor, conn, force_reanalyze=force_reanalyze_bulk, show_validation_metrics=show_val_bulk, fast_infer=fast_infer_bulk)
                                    
                                    # Compute Grad-CAM if enabled
                                    if compute_gradcam_bulk:
                                        try:
                                            from otitenet.logging.grad_cam import log_grad_cam_similarity
                                            from PIL import Image as PILImage
                                            import torch
                                            
                                            # Load model and compute Grad-CAM
                                            model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(local_args_bulk)
                                            img = PILImage.open(file_path).convert('RGB')
                                            img_tensor = get_base_transform(local_args_bulk.new_size)(img).unsqueeze(0)
                                            
                                            with torch.no_grad():
                                                # Compute Grad-CAM
                                                log_grad_cam_similarity(
                                                    model, img_tensor, local_args_bulk, 
                                                    output_dir="output/gradcam_bulk",
                                                    fname_prefix=f"bulk_{fname.split('/')[-1]}_{row.get('Model ID')}"
                                                )
                                        except Exception as grad_e:
                                            # Don't fail the whole analysis if Grad-CAM fails
                                            pass
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
                    run_analysis_on_file(selected_filename_tab2, file_bytes_sidebar, args, cursor, conn, force_reanalyze=True, show_validation_metrics=True)
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
            st.dataframe(_arrow_safe_dataframe(best_models_table[display_cols]), use_container_width=True)
        
        # Display table of all models tried on this image (sorted newest first)
        st.markdown("**Table:** Analyses for Selected Image")
        display_cols = ["Model_ID", "Model Name", "Pred_Label", "Confidence", "Timestamp", "NSize", "FGSM", "Normalize"]
        image_results_display = image_results.sort_values("Timestamp", ascending=False)
        st.dataframe(_arrow_safe_dataframe(image_results_display[display_cols]), use_container_width=True)

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
        
        # Add missing attributes for compatibility
        if not hasattr(local_args_tab2, 'auto_select_k'):
            local_args_tab2.auto_select_k = 0
        if not hasattr(local_args_tab2, 'random_recs'):
            local_args_tab2.random_recs = 0
        if not hasattr(local_args_tab2, 'seed'):
            local_args_tab2.seed = 1
        if not hasattr(local_args_tab2, 'groupkfold'):
            local_args_tab2.groupkfold = 1
        if not hasattr(local_args_tab2, 'valid_dataset'):
            local_args_tab2.valid_dataset = 'Banque_Viscaino_Chili_2020'
        if not hasattr(local_args_tab2, 'device'):
            local_args_tab2.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
        cols = st.columns(5)
        with cols[0]:
            if st.button("▶️ Run Analysis", key=f"run_analysis_tab2_{selected_filename_tab2}"):
                file_path = f'data/queries/{selected_filename_tab2.split("/")[-1]}'
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                    run_analysis_on_file(
                        selected_filename_tab2, file_bytes, local_args_tab2, cursor, conn, force_reanalyze=True
                    )
                    st.rerun()
                else:
                    st.error(f"❌ File not found: {file_path}")
        with cols[1]:
            if st.button("📊 Recompute Valid MCC", key=f"recompute_mcc_tab2_{selected_filename_tab2}"):
                with st.spinner("Recomputing validation MCC with training parameters..."):
                    try:
                        random.seed(1)
                        torch.manual_seed(1)
                        np.random.seed(1)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(1)
                        
                        model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
                            load_model_and_prototypes(local_args_tab2)
                        
                        # Load saved training parameters from model directory
                        model_log_dir = f"logs/best_models/{local_args_tab2.task}/{local_args_tab2.model_name}/{get_model_params_path(local_args_tab2)}"
                        saved_search = load_saved_search_params(model_log_dir)
                        
                        # Extract exact training parameters
                        saved_nn = saved_search.get('n_neighbors') if isinstance(saved_search, dict) else None
                        if saved_nn is not None:
                            try:
                                local_args_tab2.n_neighbors = int(saved_nn)
                            except Exception:
                                pass
                        
                        saved_normalize = saved_search.get('normalize') if isinstance(saved_search, dict) else None
                        if saved_normalize is not None:
                            local_args_tab2.normalize = saved_normalize
                        
                        saved_is_transform = saved_search.get('is_transform') if isinstance(saved_search, dict) else None
                        if saved_is_transform is not None:
                            try:
                                local_is_transform = int(saved_is_transform)
                            except Exception:
                                local_is_transform = 1
                        else:
                            local_is_transform = 1
                        
                        train = TrainAE(local_args_tab2, local_args_tab2.path, load_tb=False, log_metrics=True, keep_models=True,
                                      log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
                                      log_mlflow=False, groupkfold=local_args_tab2.groupkfold)
                        train.n_batches = len(unique_batches)
                        train.n_cats = len(unique_labels)
                        train.unique_batches = unique_batches
                        train.unique_labels = unique_labels
                        train.epoch = 1
                        train.model = model
                        train.complete_log_path = log_path
                        train.params = {
                            'n_neighbors': local_args_tab2.n_neighbors,
                            'lr': 0,
                            'wd': 0,
                            'smoothing': 0,
                            'is_transform': local_is_transform,
                            'valid_dataset': local_args_tab2.valid_dataset
                        }
                        train.set_arcloss()
                        
                        lists, traces = get_empty_traces()
                        loaders = get_images_loaders(
                            data=data,
                            random_recs=local_args_tab2.random_recs,
                            weighted_sampler=0,
                            is_transform=local_is_transform,
                            samples_weights=None,
                            epoch=1,
                            unique_labels=unique_labels,
                            triplet_dloss=local_args_tab2.dloss, bs=local_args_tab2.bs,
                            prototypes_to_use=local_args_tab2.prototypes_to_use,
                            prototypes=prototypes,
                            size=local_args_tab2.new_size,
                            normalize=local_args_tab2.normalize
                        )
                        
                        with torch.no_grad():
                            _, best_lists1, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
                            for group in ["valid"]:
                                _, best_lists2, traces, knn = train.predict(group, loaders[group], lists, traces)
                        
                        # Compute MCC on validation set
                        if 'valid' in lists and len(lists['valid'].get('cats', [])) > 0:
                            valid_cats = np.concatenate(lists['valid']['cats'])
                            valid_preds = np.concatenate(lists['valid']['preds']).argmax(1)
                            valid_mcc = MCC(valid_cats, valid_preds)
                            valid_acc = float(np.sum(valid_preds == valid_cats) / len(valid_cats))
                            
                            st.success(f"✅ Validation MCC (Recomputed): **{valid_mcc:.4f}**")
                            st.info(f"Validation Accuracy: **{valid_acc:.4f}**")
                            st.info(f"Model: {local_args_tab2.model_name} | Normalize: {local_args_tab2.normalize} | K: {local_args_tab2.n_neighbors} | Transform: {local_is_transform}")
                        else:
                            st.error("❌ Validation set is empty; could not compute MCC")
                        
                        # Cleanup
                        del model, shap_model, train
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        st.error(f"❌ Error recomputing MCC: {e}")
                        import traceback
                        traceback.print_exc()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        with cols[2]:
            if st.button("🔄 Force Re-analysis", key=f"force_reanalyze_tab2_{selected_filename_tab2}"):
                    # Load the file from data/queries directory
                    file_path = f'data/queries/{selected_filename_tab2.split("/")[-1]}'
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            file_bytes = f.read()
                            pred_label_new, pred_confidence_new, log_path_new, _ = run_analysis_on_file(
                                selected_filename_tab2, file_bytes, local_args_tab2, cursor, conn, force_reanalyze=True
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
                                          log_inputs=False, log_plots=True, log_tb=False, log_tracking=True,
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
                                          log_inputs=False, log_plots=True, log_tb=False, log_tracking=True,
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
        
        # Auto-display cached Grad-CAM results if they exist
        if os.path.isdir(grad_cam_dir):
            cached_montage_files = [f for f in os.listdir(grad_cam_dir) if f.startswith(f"{base_name}_grad_cam_all_classes_layer") and f.endswith(".png")]
            if cached_montage_files:
                st.success("✅ Cached Grad-CAM Results Found")
                # Display most recent (typically highest layer number)
                cached_montage_files.sort(reverse=True)
                for montage_file in cached_montage_files[:3]:  # Show last 3 computed layers
                    montage_path = os.path.join(grad_cam_dir, montage_file)
                    try:
                        st.image(montage_path, caption=f"Grad-CAM {montage_file.replace(base_name + '_', '').replace('.png', '')}", use_container_width=True)
                    except Exception:
                        pass
                st.divider()
        
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
                        clear_cached_model()
                        
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


# ========================= TAB 6: Grad-CAM Gallery ========================= #
if is_admin:
    with tab6:
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
                    "num": model_num,
                    "mcc_value": float(mcc) if mcc != "?" else -1  # Store MCC as float for sorting
                })
            
            # Sort by MCC value in decreasing order (highest MCC first)
            model_info_list.sort(key=lambda x: x["mcc_value"], reverse=True)
            
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
    
            def _get_model_meta(model_id: int):
                """Lookup model rank/label/MCC for display."""
                if best_models_table is None or best_models_table.empty:
                    return None
                model_id_col = "Model ID" if "Model ID" in best_models_table.columns else None
                if model_id_col is None:
                    return None
                match = best_models_table[best_models_table[model_id_col] == model_id]
                if match.empty:
                    return None
                row = match.iloc[0]
                mcc_val = row.get("MCC")
                try:
                    mcc_val = float(mcc_val)
                except Exception:
                    mcc_val = None
                return {
                    "rank": row.get("#"),
                    "mcc": mcc_val,
                    "name": row.get("Model Name"),
                }
    
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
    
                        # List which models have Grad-CAMs for this image
                        rows_for_image = df_gallery_view[df_gallery_view["Filename"] == fname]
                        model_summaries = []
                        seen_models = set()
                        for _, r_img in rows_for_image.iterrows():
                            mid = r_img.get("Model_ID")
                            if mid in seen_models:
                                continue
                            seen_models.add(mid)
                            meta = _get_model_meta(mid) or {}
                            mcc_val = meta.get("mcc")
                            rank_val = meta.get("rank")
                            label_name = r_img.get("Model Name", f"Model {mid}")
                            summary = f"{label_name}"
                            if rank_val is not None:
                                summary = f"#{rank_val} {summary}"
                            if mcc_val is not None:
                                summary += f" (MCC {mcc_val:.3f})"
                            model_summaries.append((mcc_val if mcc_val is not None else -np.inf, summary))
                        if model_summaries:
                            model_summaries.sort(key=lambda x: x[0], reverse=True)
                            st.caption("Models available: " + " | ".join([s for _, s in model_summaries]))
    
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
                clear_cached_model()
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
    
    
    # ========================= TAB 4: New Analysis ========================= #

with tab4:
    st.header("🔬 New Analysis")

    production_model_for_analysis = st.session_state.get('production_model')
    analysis_args = args
    if production_model_for_analysis:
        try:
            analysis_args = argparse.Namespace(**vars(args))
            analysis_args = _apply_production_model_to_args(analysis_args, production_model_for_analysis, data_dir)
            st.caption(f"Production model for new analyses: {st.session_state.get('client_production_model_label', 'selected by admin')}")
        except Exception as prod_exc:
            st.warning(f"Could not prepare production model for new analysis: {prod_exc}")
    else:
        st.warning("No production model is currently selected by the admin. New analyses are disabled until one is set.")
    
    # ---- File Upload Section ---- #
    # Initialize mode tracking if not present
    if 'last_upload_mode' not in st.session_state:
        st.session_state['last_upload_mode'] = "Single File"
    
    if is_admin:
        upload_mode = st.radio(
            "Select input mode:",
            options=["Single File", "Multiple Files", "Entire Folder"],
            horizontal=True,
            key="upload_mode_tab3"
        )
    else:
        upload_mode = "Single File"
        if not st.session_state.get('production_model'):
            st.warning("No production model is currently selected by the admin. Please ask the admin to set one before running analysis.")
    
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
        
        # Admin-only: show previous analyses for this image. Client view stays clean.
        if is_admin and len(uploaded_files) == 1:
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

        if is_admin:
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

            run_analysis = st.button("▶️ Run Analysis", key="run_analysis_tab3", disabled=not bool(st.session_state.get('production_model')))
            force_analysis = st.button("🔄 Force New Analysis", key="force_analysis_tab3", disabled=not bool(st.session_state.get('production_model')))
        else:
            infer_method_tab3 = 'majority_vote'
            dist_metric_tab3 = 'euclidean'
            skip_validation_tab3 = True
            fast_infer_tab3 = False
            force_analysis = False
            run_analysis = st.button("▶️ Run Analysis", key="run_analysis_tab3_client", disabled=not bool(st.session_state.get('production_model')))
            st.caption("Uses the admin-selected production model. If the model changed since the last analysis, a new result is computed; otherwise the saved result is loaded.")

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
                file_obj.name, uploaded_bytes, analysis_args, cursor, conn, force_reanalyze=force_analysis,
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
                st.session_state['last_image_size'] = analysis_args.new_size
                st.session_state['last_normalize'] = analysis_args.normalize
            
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
                
                analysis_args.device = 'cpu'
                
                pred_label, pred_confidence, complete_log_path, _ = run_analysis_on_file(
                    file_obj.name, uploaded_bytes, analysis_args, cursor, conn, force_reanalyze=force_analysis,
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
        st.dataframe(_arrow_safe_dataframe(df_results), use_container_width=True)
        st.caption(f"⏱️ Total batch time: {batch_elapsed:.2f}s | Average: {batch_elapsed/len(results_list):.2f}s per file")

# ---- Display persisted batch results ---- #
if st.session_state.get('last_batch_results'):
    import pandas as pd
    st.markdown("---")
    st.subheader("📋 Batch Results (from previous run)")
    df_results = pd.DataFrame(st.session_state['last_batch_results'])
    st.dataframe(_arrow_safe_dataframe(df_results), use_container_width=True)
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
        if is_admin:
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

                            clear_cached_model()
                            model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
                                load_model_and_prototypes(args)

                            train = TrainAE(args, args.path, load_tb=False, log_metrics=True, keep_models=True,
                                          log_inputs=False, log_plots=True, log_tb=False, log_tracking=True,
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

                            clear_cached_model()
                            model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
                                load_model_and_prototypes(args)

                            train = TrainAE(args, args.path, load_tb=False, log_metrics=True, keep_models=True,
                                          log_inputs=False, log_plots=True, log_tb=False, log_tracking=True,
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

        else:
            st.subheader("💡 Explanation (on-demand)")
            img_filename = st.session_state.get('last_uploaded_name') or uploaded_file.name
            img_path = f"data/queries/{img_filename.split('/')[-1]}"

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
        if is_admin:
            with gc_cols_t3[0]:
                layer_input = st.number_input(
                    "Layer", value=args.grad_cam_layer, step=1, key=f"gc_layer_t3_{base_name}"
                )
                alpha_input = st.slider(
                    "Alpha", 0.0, 1.0, args.grad_cam_alpha, 0.05, key=f"gc_alpha_t3_{base_name}"
                )
        else:
            layer_input = int(args.grad_cam_layer)
            alpha_input = float(args.grad_cam_alpha)
            with gc_cols_t3[0]:
                st.caption("Grad-CAM uses the production-model defaults.")

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

                            # Generate Grad-CAM for all classes. Admin keeps one selected layer;
                            # client computes all layers up to the default layer in one click.
                            image_output_dir = os.path.join(complete_log_path, base_name)
                            os.makedirs(image_output_dir, exist_ok=True)
                            layers_to_compute = [int(current_layer)] if is_admin else list(range(0, int(current_layer) + 1))
                            for layer_to_compute in layers_to_compute:
                                log_grad_cam_all_classes(
                                    model,
                                    0,
                                    inputs,
                                    'queries',
                                    image_output_dir,
                                    base_name,
                                    prototypes['class']['train'],
                                    device=device_str,
                                    layer=int(layer_to_compute),
                                    alpha=current_alpha
                                )
                                
                                class_labels = sorted(prototypes['class']['train'].keys())
                                class_images = []
                                for lbl in class_labels:
                                    class_img_path = os.path.join(image_output_dir, f"{base_name}_class{lbl}.png")
                                    if os.path.exists(class_img_path):
                                        class_images.append(plt.imread(class_img_path))
                                
                                if class_images:
                                    fig, axes = plt.subplots(1, len(class_images), figsize=(5 * len(class_images), 5))
                                    if len(class_images) == 1:
                                        axes = [axes]
                                    for ax, img, lbl in zip(axes, class_images, class_labels):
                                        ax.imshow(img)
                                        ax.set_title(f"Class: {lbl}")
                                        ax.axis('off')
                                    plt.tight_layout()
                                    montage_path = os.path.join(image_output_dir, f'{base_name}_grad_cam_all_classes_layer{int(layer_to_compute)}.png')
                                    plt.savefig(montage_path, dpi=150, bbox_inches='tight')
                                    plt.close()
                            
                            st.success(f"✅ Grad-CAM generated for all classes ({len(layers_to_compute)} layer(s)).")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error computing Grad-CAM: {e}")

        layer_changed = is_admin and layer_input != args.grad_cam_layer
        alpha_changed = is_admin and alpha_input != args.grad_cam_alpha
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

        # Display all-classes montage(s): admin shows selected layer, client shows all computed layers.
        image_grad_cam_dir = os.path.join(complete_log_path, base_name)
        if is_admin:
            grad_cam_paths = [os.path.join(image_grad_cam_dir, f'{base_name}_grad_cam_all_classes_layer{layer_input}.png')]
        else:
            import glob
            grad_cam_paths = sorted(glob.glob(os.path.join(image_grad_cam_dir, f'{base_name}_grad_cam_all_classes_layer*.png')))
        shown_any_gradcam = False
        for grad_cam_all_path in grad_cam_paths:
            if os.path.exists(grad_cam_all_path):
                shown_any_gradcam = True
                try:
                    layer_label = os.path.basename(grad_cam_all_path).split('_layer')[-1].replace('.png', '')
                except Exception:
                    layer_label = str(layer_input)
                fig = plt.imread(grad_cam_all_path)
                st.image(fig, caption=f"Grad-CAM All Classes (Layer {layer_label})", use_container_width=True)
        if not shown_any_gradcam:
            st.info("Grad-CAM not computed for this analysis. Click 'Compute Grad-CAM' button above.")

# Keep connection open for Streamlit session; closing can break callbacks
# conn.close()

# ========================= TAB 5: Ensemble ========================= #
if is_admin:
    with tab5:
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
                    compute_ens = st.button("📊 Compute Validation Ensemble", key="compute_ensemble_valid_tab4")
    
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
                            st.dataframe(_arrow_safe_dataframe(pd.DataFrame(cache['per_model_rows'])), use_container_width=True)
                        
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
                    if compute_ens:
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
                                        try:
                                            model = load_model_for_log_path(lp, mn, ens_device)
                                            class_protos = None
                                        except Exception as e:
                                            model, class_protos = load_model_and_prototypes(lp, mn, ens_device)
                                            st.warning(f"Model {mn} loaded without prototypes: {e}")
            
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
                                                n_bins = st.number_input("Calibration Curve n_bins", min_value=2, max_value=50, value=10, step=1, key="calib_n_bins")
                                                prob_true, prob_pred = calibration_curve(y_true_filt, y_prob_filt, n_bins=int(n_bins))
                                                fig, ax = plt.subplots(figsize=(6, 3.5))
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
                                        st.dataframe(_arrow_safe_dataframe(pd.DataFrame(mis_rows)), use_container_width=True)
    
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
                        st.dataframe(_arrow_safe_dataframe(pd.DataFrame(mis_rows)), use_container_width=True)
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

    
    # ========================= TAB 8: Admin Analytics (PCA/EDA) ========================= #
    with tab8:
        st.header("📊 Admin Analytics")
        st.caption("Advanced analysis: Raw PCA, t-SNE, UMAP, distributions, embeddings stats")
        
        # ---- Comprehensive EDA Suite ---- #
        st.divider()
        st.subheader("📊 Comprehensive EDA Suite")
        st.caption("Advanced analysis: Raw PCA, t-SNE, UMAP, distributions, embeddings stats")
    
        def _run_comprehensive_eda(_args):
            """Zealous ML engineer's comprehensive EDA analysis."""
            with st.spinner("Loading model and data..."):
                model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)
                train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                                log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
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
                            img = img.resize((_args.new_size, _args.new_size))
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
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    class_counts.plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title('Samples per Class')
                    ax.set_ylabel('Count')
                    ax.set_xlabel('Class ID')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                with col2:
                    st.write("**Group Distribution**")
                    grp_counts = pd.Series(all_grps).value_counts()
                    fig, ax = plt.subplots(figsize=(6, 3.5))
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
                        fig, ax = plt.subplots(figsize=(6, 3.5))
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
                        fig, ax = plt.subplots(figsize=(6, 3.5))
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
                            fig, ax = plt.subplots(figsize=(6, 3.5))
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
                fig, ax = plt.subplots(figsize=(6, 3.5))
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
                fig, ax = plt.subplots(figsize=(6, 3.5))
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
                    fig, ax = plt.subplots(figsize=(6, 3.5))
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
        


    # ========================= TAB 9: Inference Results ========================= #
    with tab9:
        st.header("📈 Inference Results")
        st.caption("Analyze images from data/datasets/inference with model performance metrics")

        import os
        import glob

        inference_dir = 'data/datasets/inference'
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        inference_images = []
        for ext in image_extensions:
            inference_images.extend(glob.glob(os.path.join(inference_dir, ext)))
        inference_images = sorted(inference_images)

        def _xlsx_col_to_idx(cell_ref: str) -> int:
            """Convert an Excel cell reference like A1/AB12 to a zero-based column index."""
            import re
            letters = re.sub(r"[^A-Z]", "", str(cell_ref).upper())
            idx = 0
            for ch in letters:
                idx = idx * 26 + (ord(ch) - ord("A") + 1)
            return idx - 1

        def _read_xlsx_sheet_values(xlsx_path: str, sheet_name: str = "Sheet3"):
            """Read a simple XLSX sheet using only stdlib, avoiding pandas/openpyxl version issues."""
            import posixpath
            import zipfile
            import xml.etree.ElementTree as ET

            ns = {
                "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
                "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
            }

            with zipfile.ZipFile(xlsx_path) as zf:
                shared_strings = []
                if "xl/sharedStrings.xml" in zf.namelist():
                    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                    for si in root.findall("main:si", ns):
                        texts = [t.text or "" for t in si.findall(".//main:t", ns)]
                        shared_strings.append("".join(texts))

                wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
                rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
                rel_targets = {
                    rel.attrib.get("Id"): rel.attrib.get("Target")
                    for rel in rels_root.findall("pkgrel:Relationship", ns)
                }

                target = None
                for sheet in wb_root.findall(".//main:sheet", ns):
                    if sheet.attrib.get("name") == sheet_name:
                        rid = sheet.attrib.get("{%s}id" % ns["rel"])
                        target = rel_targets.get(rid)
                        break
                if target is None:
                    return []

                sheet_path = posixpath.normpath(posixpath.join("xl", target))
                if sheet_path.startswith("/"):
                    sheet_path = sheet_path[1:]
                sheet_root = ET.fromstring(zf.read(sheet_path))
                rows_out = []
                for row in sheet_root.findall(".//main:sheetData/main:row", ns):
                    vals = []
                    for cell in row.findall("main:c", ns):
                        col_idx = _xlsx_col_to_idx(cell.attrib.get("r", "A1"))
                        while len(vals) <= col_idx:
                            vals.append(None)

                        cell_type = cell.attrib.get("t")
                        value_el = cell.find("main:v", ns)
                        inline_el = cell.find("main:is/main:t", ns)

                        if cell_type == "s" and value_el is not None:
                            value = shared_strings[int(value_el.text)]
                        elif cell_type == "inlineStr" and inline_el is not None:
                            value = inline_el.text or ""
                        elif value_el is not None:
                            value = value_el.text
                        else:
                            value = None
                        vals[col_idx] = value
                    rows_out.append(vals)
                return rows_out

        def _diagnostic_to_binary_label(value):
            """Map Sheet3 diagnostic text to the model labels.

            Only exact T. G normal / T. D normal entries are Normal.
            Any other non-empty diagnostic text is NotNormal.
            """
            import unicodedata

            if value is None:
                return "Unknown"
            text = str(value).strip()
            if not text:
                return "Unknown"

            simplified = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
            simplified = " ".join(simplified.replace("\xa0", " ").split())
            normal_values = {"t. g normal", "t.g normal", "tg normal", "t. d normal", "t.d normal", "td normal"}
            if simplified in normal_values:
                return "Normal"
            return "NotNormal"

        @st.cache_data(show_spinner=False)
        def _load_inference_ground_truth_map(xlsx_path: str, mtime: float):
            """Load ground-truth labels from Sheet3 of LOG_Entrainement_IA.xlsx."""
            rows = _read_xlsx_sheet_values(xlsx_path, "Sheet3")
            if not rows:
                return {}

            headers = [str(v).strip() if v is not None else "" for v in rows[0]]

            def _find_col(contains_all):
                for i, header in enumerate(headers):
                    low = header.lower()
                    if all(tok.lower() in low for tok in contains_all):
                        return i
                return None

            def _find_diag_col(side):
                # Avoid matching the "d" in "diagnostic" when looking for the D column.
                import re
                target = str(side).strip().lower()
                for i, header in enumerate(headers):
                    low = " ".join(str(header).lower().replace("_", " ").split())
                    if "diagnostic" not in low:
                        continue
                    if re.search(rf"(?:^|\b)tympan\s+{target}(?:\b|$)", low):
                        return i
                    if re.search(rf"(?:^|\b){target}(?:\b|$)", low.replace("diagnostic", " ")):
                        return i
                return None

            id_col = _find_col(["id"])
            diag_g_col = _find_diag_col("g")
            diag_d_col = _find_diag_col("d")
            if id_col is None:
                return {}

            import re
            truth_map = {}
            for row in rows[1:]:
                if id_col >= len(row) or row[id_col] in (None, ""):
                    continue
                match = re.search(r"(\d{5})", str(row[id_col]))
                if not match:
                    continue
                img_id = match.group(1)

                if diag_g_col is not None and diag_g_col < len(row):
                    truth_map[(img_id, "G")] = _diagnostic_to_binary_label(row[diag_g_col])
                if diag_d_col is not None and diag_d_col < len(row):
                    truth_map[(img_id, "D")] = _diagnostic_to_binary_label(row[diag_d_col])

            return truth_map

        def _inference_ground_truth(filename: str):
            """Return Normal/NotNormal from LOG_Entrainement_IA.xlsx based on image ID and G/D side."""
            import re

            base = os.path.basename(str(filename))
            id_match = re.search(r"(\d{5})", base)
            side_match = re.search(r"(?:^|[-_])(G|D)(?:[-_.]|$)", base, flags=re.IGNORECASE)
            if not id_match or not side_match:
                return "Unknown"

            xlsx_path = os.path.join(inference_dir, "LOG_Entrainement_IA.xlsx")
            if not os.path.exists(xlsx_path):
                return "Unknown"

            truth_map = _load_inference_ground_truth_map(xlsx_path, os.path.getmtime(xlsx_path))
            return truth_map.get((id_match.group(1), side_match.group(1).upper()), "Unknown")

        def _labels_match(pred_label, ground_truth):
            if ground_truth == "Unknown":
                return None
            return str(pred_label).strip().lower() == str(ground_truth).strip().lower()

        def _binary_label(value):
            value_l = str(value).strip().lower()
            if value_l == "notnormal":
                return 1
            if value_l == "normal":
                return 0
            return None

        def _notnormal_score(pred_label, confidence):
            """Approximate P(NotNormal) from stored winning-class confidence.

            The database stores only the predicted-class confidence, not a full
            probability vector. For binary Normal/NotNormal AUC this is the best
            available reconstruction: confidence if the prediction is NotNormal,
            otherwise 1 - confidence.
            """
            try:
                conf = float(confidence)
            except Exception:
                return np.nan
            pred_bin = _binary_label(pred_label)
            if pred_bin is None:
                return np.nan
            return conf if pred_bin == 1 else 1.0 - conf

        def _compute_inference_metrics(frame):
            if frame is None or frame.empty:
                return {"ACC": np.nan, "MCC": np.nan, "AUC": np.nan, "N": 0}
            eval_df = frame.copy()
            eval_df["_y_true"] = eval_df["Ground Truth"].apply(_binary_label)
            eval_df["_y_pred"] = eval_df["Prediction"].apply(_binary_label)
            eval_df["_score"] = eval_df.apply(lambda r: _notnormal_score(r["Prediction"], r["Confidence"]), axis=1)
            eval_df = eval_df.dropna(subset=["_y_true", "_y_pred"])
            if eval_df.empty:
                return {"ACC": np.nan, "MCC": np.nan, "AUC": np.nan, "N": 0}
            y_true = eval_df["_y_true"].astype(int).to_numpy()
            y_pred = eval_df["_y_pred"].astype(int).to_numpy()
            metrics = {
                "ACC": float(accuracy_score(y_true, y_pred)),
                "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0,
                "AUC": np.nan,
                "N": int(len(eval_df)),
            }
            try:
                scores = eval_df["_score"].astype(float).to_numpy()
                if len(np.unique(y_true)) > 1 and np.isfinite(scores).all():
                    metrics["AUC"] = float(roc_auc_score(y_true, scores))
            except Exception:
                metrics["AUC"] = np.nan
            return metrics

        def _fmt_metric(value):
            try:
                if pd.isna(value):
                    return ""
                return f"{float(value):.4f}"
            except Exception:
                return ""

        def _fmt_confidence(value):
            """Always display confidence with exactly two decimals, including 100.00."""
            try:
                return f"{float(value):.2f}"
            except Exception:
                return ""

        def _normalize_analysis_result(result):
            """Accept old/new run_analysis_on_file return shapes without crashing the UI.

            Known shapes:
            - (pred_label, confidence, log_path, existing)
            - (pred_label, confidence, log_path, existing, gradcam_path)
            - legacy variants with additional values appended
            """
            if not isinstance(result, (tuple, list)):
                raise ValueError(f"run_analysis_on_file returned unsupported type: {type(result)}")
            if len(result) < 2:
                raise ValueError(f"run_analysis_on_file returned too few values: {len(result)}")
            pred_label = result[0]
            confidence = result[1]
            complete_log_path = result[2] if len(result) > 2 else None
            existing = result[3] if len(result) > 3 else None
            gradcam_path = result[4] if len(result) > 4 else None
            return pred_label, confidence, complete_log_path, existing, gradcam_path

        def _inference_gradcam_dir(log_path_value, filename):
            if not log_path_value:
                return None
            base_name = strip_extension(os.path.basename(str(filename)))
            return os.path.join(str(log_path_value), base_name)

        def _gradcam_layer_from_path(path_value):
            match = re.search(r"_layer(\d+)\.png$", os.path.basename(str(path_value)))
            return int(match.group(1)) if match else -1

        def _find_inference_gradcam_images(log_path_value, filename, n_layers=4):
            """Return only the wide all-class Grad-CAM montage images, newest/highest layers first.

            The single-class overlay PNGs are intentionally excluded here because the
            observer view should only show the composite image containing original,
            Normal, and NotNormal/Anormal panels.
            """
            grad_cam_dir = _inference_gradcam_dir(log_path_value, filename)
            if not grad_cam_dir or not os.path.isdir(grad_cam_dir):
                return []
            base_name = strip_extension(os.path.basename(str(filename)))
            found = glob.glob(os.path.join(grad_cam_dir, f"{base_name}_grad_cam_all_classes_layer*.png"))
            found = sorted(dict.fromkeys(found), key=_gradcam_layer_from_path, reverse=True)
            return found[:int(n_layers)]

        def _args_from_inference_row(row_dict):
            """Build inference args from the same row format used by Quick Model Selection."""
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
            local_args.model_id = row_dict.get("Model ID") or row_dict.get("Model_ID") or getattr(args, "model_id", None)
            return local_args

        if inference_images:
            st.success(f"Found {len(inference_images)} images in {inference_dir}")

            # Use the exact same ranked/deduped table and numbering as Quick Model Selection.
            model_number_map, best_models_table = _ensure_model_number_map(cursor)

            if best_models_table is not None and not best_models_table.empty:
                st.divider()
                st.subheader("🎯 Model Selection")

                inference_model_df = best_models_table.copy().reset_index(drop=True)
                inference_model_df["_selection_key"] = inference_model_df.apply(
                    lambda r: _make_model_selection_key(r.to_dict()), axis=1
                )

                model_options = inference_model_df["_selection_key"].tolist()
                key_to_row = {
                    row["_selection_key"]: row.drop(labels=["_selection_key"]).to_dict()
                    for _, row in inference_model_df.iterrows()
                }

                def _inference_model_label(selection_key):
                    row = key_to_row.get(selection_key, {})
                    rank = row.get("#", model_number_map.get(selection_key, "?"))
                    model_name = row.get("Model Name", "?")
                    model_id_val = row.get("Model ID", "?")
                    valid_mcc = row.get("Valid MCC", np.nan)
                    test_mcc = row.get("Test MCC", np.nan)
                    valid_auc = row.get("Valid AUC", np.nan)
                    test_auc = row.get("Test AUC", np.nan)
                    mcc = row.get("MCC", np.nan)
                    score = valid_mcc if pd.notna(valid_mcc) else mcc
                    metric_bits = []
                    for label, val in [
                        ("valid_mcc", valid_mcc),
                        ("test_mcc", test_mcc),
                        ("valid_auc", valid_auc),
                        ("test_auc", test_auc),
                    ]:
                        try:
                            if pd.notna(val):
                                metric_bits.append(f"{label}: {float(val):.4f}")
                        except Exception:
                            pass
                    if not metric_bits:
                        try:
                            metric_bits.append(f"MCC: {float(score):.4f}")
                        except Exception:
                            pass
                    score_txt = " | " + ", ".join(metric_bits) if metric_bits else ""
                    return f"#{rank} = {model_name} (ID: {model_id_val}){score_txt}"

                # Keep this tab synced to the model selected in Quick Model Selection when possible.
                canonical_key = st.session_state.get("selected_model_selection_key")
                default_index = model_options.index(canonical_key) if canonical_key in model_options else 0

                st.divider()
                st.subheader("📊 Top-N Inference Performance")
                top_n_models = st.number_input(
                    "Number of top models to include",
                    min_value=1,
                    max_value=int(len(inference_model_df)),
                    value=min(5, int(len(inference_model_df))),
                    step=1,
                    key="inference_top_n_models",
                )
                top_n_model_df = inference_model_df.head(int(top_n_models)).copy()
                top_n_model_ids = [
                    int(mid) for mid in top_n_model_df["Model ID"].dropna().tolist()
                ]

                def _fetch_latest_inference_results_for_models(model_ids):
                    if not model_ids or not inference_images:
                        return pd.DataFrame()
                    inference_filenames_local = [os.path.basename(img) for img in inference_images]
                    model_placeholders = ",".join(["%s"] * len(model_ids))
                    file_placeholders = ",".join(["%s"] * len(inference_filenames_local))
                    cursor.execute(
                        f"""
                        SELECT filename, pred_label, confidence, timestamp, log_path, model_id
                        FROM results
                        WHERE person_id=%s
                          AND model_id IN ({model_placeholders})
                          AND filename IN ({file_placeholders})
                        ORDER BY timestamp DESC
                        """,
                        tuple([st.session_state.person_id] + list(model_ids) + inference_filenames_local),
                    )
                    rows = cursor.fetchall()
                    if not rows:
                        return pd.DataFrame()
                    df = pd.DataFrame(
                        rows,
                        columns=["Filename", "Prediction", "Confidence", "Timestamp", "Log Path", "Model ID"],
                    )
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                    df = (
                        df.sort_values("Timestamp", ascending=False)
                        .drop_duplicates(subset=["Model ID", "Filename"], keep="first")
                        .reset_index(drop=True)
                    )
                    df["Ground Truth"] = df["Filename"].apply(_inference_ground_truth)
                    df["Correct"] = df.apply(lambda r: _labels_match(r["Prediction"], r["Ground Truth"]), axis=1)
                    return df

                top_n_existing_df = _fetch_latest_inference_results_for_models(top_n_model_ids)
                performance_rows = []
                if not top_n_model_df.empty:
                    for _, perf_model_row in top_n_model_df.iterrows():
                        perf_model_id = perf_model_row.get("Model ID")
                        model_results_df = top_n_existing_df[top_n_existing_df["Model ID"] == perf_model_id].copy() if not top_n_existing_df.empty else pd.DataFrame()
                        metrics = _compute_inference_metrics(model_results_df)
                        rank_value = perf_model_row.get("#", model_number_map.get(perf_model_row.get("_selection_key"), "?"))
                        try:
                            _tmp_perf_args = _args_from_inference_row(perf_model_row.to_dict())
                            best_head_entry = _best_head_entry_for_args_global(_tmp_perf_args)
                            head_config = str(best_head_entry.get("config", ""))
                            head_label = best_head_entry.get("label") or _head_config_label_global(head_config)
                            head_score = best_head_entry.get("mcc", np.nan)
                        except Exception:
                            head_config = ""
                            head_label = "—"
                            head_score = np.nan
                        performance_rows.append({
                            "#": rank_value,
                            "Model ID": perf_model_id,
                            "Model Name": perf_model_row.get("Model Name"),
                            "Best Classification Head": head_label,
                            "Best Head Config": head_config,
                            "Best Head Score": _fmt_metric(head_score),
                            "ACC": _fmt_metric(metrics["ACC"]),
                            "MCC": _fmt_metric(metrics["MCC"]),
                            "AUC": _fmt_metric(metrics["AUC"]),
                            "N analyzed": metrics["N"],
                        })
                if performance_rows:
                    _top_n_perf_df = pd.DataFrame(performance_rows)
                    st.session_state["inference_topn_performance_df"] = _top_n_perf_df.copy()
                    st.dataframe(_arrow_safe_dataframe(_top_n_perf_df), use_container_width=True)
                else:
                    st.info("No top-N inference results available yet.")

                st.caption(
                    "Top-N computation is also run silently, so the page will not print one prediction trace per image."
                )
                force_all_models_inference = st.checkbox(
                    "🔄 Force recompute existing results when running all selected top models",
                    value=False,
                    key="inference_force_all_top_models",
                )
                if st.button("▶️ Compute inference for all images × selected top models", key="inference_run_all_top_models"):
                    from otitenet.app.analysis import run_analysis_on_file
                    total_jobs = int(len(inference_images) * len(top_n_model_df))
                    progress = st.progress(0)
                    done_jobs = 0
                    try:
                        for _, batch_model_row in top_n_model_df.iterrows():
                            batch_model_dict = batch_model_row.drop(labels=["_selection_key"], errors="ignore").to_dict()
                            batch_model_args = _args_from_inference_row(batch_model_dict)
                            # Use the best learned-embedding classification head for this model,
                            # unless the selected sidebar model/head overrides it elsewhere.
                            batch_head_config = _best_head_config_for_args_global(batch_model_args)
                            _set_classifier_head_on_args_global(batch_model_args, batch_head_config)
                            batch_model_id = batch_model_dict.get("Model ID")
                            batch_filenames = [os.path.basename(img) for img in inference_images]
                            if force_all_models_inference and batch_model_id is not None and batch_filenames:
                                placeholders = ",".join(["%s"] * len(batch_filenames))
                                cursor.execute(
                                    f"""
                                    DELETE FROM results
                                    WHERE person_id=%s AND model_id=%s AND filename IN ({placeholders})
                                    """,
                                    tuple([st.session_state.person_id, batch_model_id] + batch_filenames),
                                )
                                conn.commit()
                            for img_path in inference_images:
                                img_name = os.path.basename(img_path)
                                with open(img_path, "rb") as f:
                                    image_bytes = f.read()
                                run_analysis_on_file(
                                    img_name,
                                    image_bytes,
                                    batch_model_args,
                                    cursor,
                                    conn,
                                    force_reanalyze=force_all_models_inference,
                                    show_validation_metrics=False,
                                    fast_infer=False,
                                    quiet=True,
                                )
                                done_jobs += 1
                                progress.progress(min(1.0, done_jobs / max(total_jobs, 1)))
                        st.success(f"Computed inference for {done_jobs} image/model jobs.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Top-N inference failed: {e}")
                        import traceback
                        st.error(traceback.format_exc())

                selected_model_key = st.selectbox(
                    "Select a model for inference analysis",
                    options=model_options,
                    index=default_index,
                    format_func=_inference_model_label,
                    key="inference_model_select_key",
                )

                selected_model_dict = key_to_row[selected_model_key]
                model_id = selected_model_dict.get("Model ID")
                model_name = selected_model_dict.get("Model Name")
                log_path = selected_model_dict.get("Log Path")
                model_args = _args_from_inference_row(selected_model_dict)

                # Classification head used for this selected model. Default to the
                # best cached head for this model, but let the user override it here.
                head_entries = _classification_head_options_for_args_global(model_args)
                head_options = [str(h.get("config")) for h in head_entries if h.get("config") is not None]
                head_label_map = {}
                for h in head_entries:
                    cfg = str(h.get("config"))
                    label = h.get("label") or _head_config_label_global(cfg)
                    score = h.get("mcc", np.nan)
                    details = h.get("details", "")
                    score_txt = "" if pd.isna(score) else f" | score={float(score):.4f}"
                    details_txt = f" | {details}" if details else ""
                    head_label_map[cfg] = f"{label} ({cfg}){score_txt}{details_txt}"

                default_head_config = _best_head_config_for_args_global(model_args)
                if selected_model_key == st.session_state.get("selected_model_selection_key") and st.session_state.get("sidebar_classification_head_config"):
                    default_head_config = str(st.session_state.get("sidebar_classification_head_config"))
                if str(default_head_config) not in head_options and head_options:
                    default_head_config = head_options[0]

                head_widget_key = f"inference_head_select_{model_id}_{selected_model_key}"
                if st.session_state.get(head_widget_key) not in head_options:
                    st.session_state[head_widget_key] = str(default_head_config)

                selected_head_config = st.selectbox(
                    "Select classification head for inference",
                    options=head_options,
                    key=head_widget_key,
                    format_func=lambda cfg: head_label_map.get(str(cfg), _head_config_label_global(cfg)),
                    help="Defaults to the best cached head found previously for this model, but you can override it for inference.",
                ) if head_options else str(default_head_config)

                _set_classifier_head_on_args_global(model_args, selected_head_config)
                st.caption(f"Classification head applied: {_head_config_label_global(selected_head_config)} (`{selected_head_config}`)")

                # Button to launch inference on all images
                st.divider()
                st.subheader("🚀 Launch Inference")
                st.caption(
                    "Computations keep running if you only switch Streamlit tabs. "
                    "Avoid changing widgets, refreshing, or navigating away during a run, because those can trigger a rerun."
                )
                force_reanalyze_inference = st.checkbox(
                    "🔄 Force re-analyze all images",
                    value=False,
                    key="inference_force_reanalyze",
                )

                if st.button("▶️ Run Inference on All Images", key="inference_run_all"):
                    with st.spinner(f"Running inference on {len(inference_images)} images with {model_name}..."):
                        try:
                            from otitenet.app.analysis import run_analysis_on_file

                            inference_filenames = [os.path.basename(img) for img in inference_images]

                            # Force means replace old rows for this selected model/images, not accumulate duplicates.
                            if force_reanalyze_inference and model_id is not None and inference_filenames:
                                placeholders = ",".join(["%s"] * len(inference_filenames))
                                cursor.execute(
                                    f"""
                                    DELETE FROM results
                                    WHERE person_id=%s AND model_id=%s AND filename IN ({placeholders})
                                    """,
                                    tuple([st.session_state.person_id, model_id] + inference_filenames),
                                )
                                conn.commit()

                            results = []
                            correct_count = 0

                            for img_path in inference_images:
                                img_name = os.path.basename(img_path)
                                with open(img_path, 'rb') as f:
                                    image_bytes = f.read()

                                analysis_result = run_analysis_on_file(
                                    img_name,
                                    image_bytes,
                                    model_args,
                                    cursor,
                                    conn,
                                    force_reanalyze=force_reanalyze_inference,
                                    show_validation_metrics=False,
                                    quiet=True,
                                )
                                pred_label, confidence, complete_log_path, _existing, gradcam_path = _normalize_analysis_result(analysis_result)

                                ground_truth = _inference_ground_truth(img_name)
                                is_correct = _labels_match(pred_label, ground_truth)
                                if is_correct is True:
                                    correct_count += 1

                                results.append({
                                    'Filename': img_name,
                                    'Head': _head_config_label_global(selected_head_config),
                                    'Head Config': selected_head_config,
                                    'Ground Truth': ground_truth,
                                    'Prediction': pred_label,
                                    'Correct': is_correct,
                                    'Confidence': _fmt_confidence(confidence),
                                })

                            accuracy = correct_count / len(results) if results else 0

                            st.divider()
                            st.subheader("📊 Performance Summary")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Accuracy", f"{accuracy:.4f}")
                            with col2:
                                st.metric("Total Images", len(results))

                            st.success(f"Inference completed: {correct_count}/{len(results)} correct ({accuracy:.2%})")

                        except Exception as e:
                            st.error(f"Inference failed: {e}")
                            import traceback
                            st.error(traceback.format_exc())

                # Show existing results for the selected model only.
                st.divider()
                st.subheader("📂 Existing Results for Inference Images")

                inference_filenames = [os.path.basename(img) for img in inference_images]
                existing_results = []

                if model_id is not None and inference_filenames:
                    placeholders = ",".join(["%s"] * len(inference_filenames))
                    cursor.execute(
                        f"""
                        SELECT filename, pred_label, confidence, timestamp, log_path
                        FROM results
                        WHERE person_id=%s AND model_id=%s AND filename IN ({placeholders})
                        ORDER BY timestamp DESC
                        """,
                        tuple([st.session_state.person_id, model_id] + inference_filenames),
                    )
                    existing_results = cursor.fetchall()

                if existing_results:
                    raw_existing_count = len(existing_results)
                    existing_df = pd.DataFrame(
                        existing_results,
                        columns=['Filename', 'Prediction', 'Confidence', 'Timestamp', 'Log Path'],
                    )

                    # Keep only the most recent row per inference filename for display/counts.
                    # The results table may contain many historical rows for the same inference
                    # image/model, so the inference tab must count unique inference images, not raw DB rows.
                    existing_df['Timestamp'] = pd.to_datetime(existing_df['Timestamp'], errors='coerce')
                    existing_df = (
                        existing_df.sort_values('Timestamp', ascending=False)
                        .drop_duplicates(subset=['Filename'], keep='first')
                        .reset_index(drop=True)
                    )
                    st.info(
                        f"Found {len(existing_df)} latest inference image results for {model_name} "
                        f"out of {len(inference_filenames)} inference images."
                    )
                    if raw_existing_count > len(existing_df):
                        st.caption(
                            f"Ignored {raw_existing_count - len(existing_df)} older duplicate DB rows "
                            "for the same inference image/model."
                        )

                    existing_df['Head'] = _head_config_label_global(selected_head_config)
                    existing_df['Head Config'] = selected_head_config
                    existing_df['Ground Truth'] = existing_df['Filename'].apply(_inference_ground_truth)
                    existing_df['Correct'] = existing_df.apply(
                        lambda r: _labels_match(r['Prediction'], r['Ground Truth']),
                        axis=1,
                    )
                    # Add per-image consensus: among the selected Top-N models, what percent were correct?
                    if not top_n_existing_df.empty:
                        consensus = (
                            top_n_existing_df[top_n_existing_df["Correct"].isin([True, False])]
                            .groupby("Filename")["Correct"]
                            .agg(lambda vals: 100.0 * float(np.sum(vals == True)) / float(len(vals)) if len(vals) else np.nan)
                            .to_dict()
                        )
                        existing_df["Top-N Correct %"] = existing_df["Filename"].map(consensus)
                    else:
                        existing_df["Top-N Correct %"] = np.nan
                    existing_df["Top-N Correct %"] = existing_df["Top-N Correct %"].apply(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")

                    selected_metrics = _compute_inference_metrics(existing_df)
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("ACC", _fmt_metric(selected_metrics["ACC"]))
                    metric_cols[1].metric("MCC", _fmt_metric(selected_metrics["MCC"]))
                    metric_cols[2].metric("AUC", _fmt_metric(selected_metrics["AUC"]))
                    metric_cols[3].metric("N analyzed", selected_metrics["N"])

                    existing_df['Confidence'] = existing_df['Confidence'].apply(_fmt_confidence)
                    existing_df = existing_df[
                        ['Filename', 'Head', 'Head Config', 'Ground Truth', 'Prediction', 'Correct', 'Confidence', 'Top-N Correct %', 'Timestamp', 'Log Path']
                    ]

                    st.dataframe(
                        existing_df.drop(columns=['Log Path'], errors='ignore'),
                        use_container_width=True,
                    )

                    st.divider()
                    st.subheader("🔎 Observe One Inference Image")
                    observe_options = existing_df['Filename'].tolist()
                    selected_observe_filename = st.selectbox(
                        "Select an inference image to inspect",
                        options=observe_options,
                        key="inference_observe_image_select",
                    )
                    observe_row = existing_df[existing_df['Filename'] == selected_observe_filename].iloc[0]
                    observe_img_path = os.path.join(inference_dir, selected_observe_filename)

                    obs_cols = st.columns([1, 1])
                    with obs_cols[0]:
                        if os.path.exists(observe_img_path):
                            st.image(observe_img_path, caption=selected_observe_filename, use_container_width=True)
                        else:
                            st.warning(f"Image file not found: {observe_img_path}")
                    with obs_cols[1]:
                        st.markdown(f"**Prediction:** {observe_row['Prediction']}")
                        st.markdown(f"**Confidence:** {observe_row['Confidence']}")
                        st.markdown(f"**Ground Truth:** {observe_row['Ground Truth']}")
                        correct_value = observe_row['Correct']
                        if correct_value is True:
                            st.success("Correct: True")
                        elif correct_value is False:
                            st.error("Correct: False")
                        else:
                            st.info("Correct: Unknown")

                    gradcam_images = _find_inference_gradcam_images(observe_row.get('Log Path'), selected_observe_filename, n_layers=4)
                    st.markdown("**Grad-CAM composite explanations: last 4 layers**")
                    if gradcam_images:
                        for gc_path in gradcam_images:
                            try:
                                st.image(gc_path, caption=os.path.basename(gc_path), use_container_width=True)
                            except Exception as _gc_display_exc:
                                st.warning(f"Could not display {gc_path}: {_gc_display_exc}")
                    else:
                        st.info("No Grad-CAM image found yet for this image/model. New predictions will generate Grad-CAM automatically; the legacy Grad-CAM buttons can still be used for older rows.")
                else:
                    st.info("No existing results found for this selected model")
            else:
                st.warning("No ranked best models found. Run/update the best model registry first.")
        else:
            st.warning(f"No images found in {inference_dir}")
            st.warning(f"No images found in {inference_dir}")