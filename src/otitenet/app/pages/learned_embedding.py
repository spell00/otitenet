
# /home/simon/otitenet/otitenet/app/pages/learned_embedding.py

from __future__ import annotations

import os
import pickle
import re
import traceback
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from otitenet.app.model_loading import load_model_and_prototypes
from otitenet.app.model_loading import clear_cached_model, resolve_model_paths
from otitenet.app.utils_dataset_names import get_short_dataset_name, get_short_dataset_names
from otitenet.app.utils import (
    attach_task_column,
    enumerate_classification_heads,
    filter_models_df_by_task,
    format_classifier_config,
    get_model_params_path,
    get_optimization_cache_file_path,
)
from otitenet.app.services.embedding_optimization_service import (
    args_from_model_row,
    fetch_best_model_rows,
)
from otitenet.logging.metrics import MCC
from otitenet.data.labels import labels_for_task
from otitenet.train.train_triplet_new import TrainAE
from otitenet.utils.encoding_utils import (
    compute_prototypes_by_strategy,
    flatten_prototype_dict,
)


BASELINE_DISPLAY_NAMES = {
    "logreg": "Logistic Regression",
    "ridge": "Ridge Classifier",
    "naive_bayes": "Naive Bayes",
    "linear_svc": "Linear SVC",
    "rbf_svc": "RBF SVC",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "decision_tree": "Decision Tree",
    "lda": "Linear Discriminant",
    "qda": "Quadratic Discriminant",
}

LEARNED_HEAD_CACHE_VERSION = 2


# -------------------------------------------------
# Display helpers from your current page
# -------------------------------------------------

def _safe_float(x):
    try:
        if x is None:
            return None
        val = float(x)
        if pd.isna(val):
            return None
        return val
    except Exception:
        return None


def _is_blank_value(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "nan", "null"}


def _first_nonblank(*values):
    for value in values:
        if not _is_blank_value(value):
            return value
    return None


def _infer_head_family(config_or_label) -> str:
    text = str(config_or_label or "").lower()

    if text.strip().isdigit():
        return "knn"
    if "knn" in text or "neighbor" in text:
        return "knn"
    if "kmeans" in text or "k-means" in text:
        return "kmeans"
    if "gmm" in text or "gaussian" in text:
        return "gmm"
    if re.search(r"\bmean\b", text):
        return "mean"

    for key in BASELINE_DISPLAY_NAMES:
        if key in text:
            return key

    if "logistic" in text:
        return "logreg"
    if "ridge" in text:
        return "ridge"
    if "forest" in text:
        return "random_forest"
    if "svc" in text or "svm" in text:
        return "linear_svc"

    return "other"


def _family_display_name(family: str) -> str:
    if family == "knn":
        return "KNN"
    if family == "mean":
        return "Mean prototypes"
    if family == "kmeans":
        return "KMeans prototypes"
    if family == "gmm":
        return "GMM prototypes"
    return BASELINE_DISPLAY_NAMES.get(family, family)


def _normalize_head_entry(entry, model_row, model_args=None):
    model_id = (
        model_row.get("Model ID")
        or model_row.get("id")
        or model_row.get("model_id")
    )
    model_name = (
        model_row.get("Model Name")
        or model_row.get("model_name")
        or model_row.get("model")
    )
    model_train_mcc = model_row.get("Train MCC") or model_row.get("train_mcc")
    model_valid_mcc = model_row.get("Valid MCC") or model_row.get("valid_mcc") or model_row.get("MCC") or model_row.get("mcc")
    model_test_mcc = model_row.get("Test MCC") or model_row.get("test_mcc")
    log_path = model_row.get("Log Path") or model_row.get("log_path") or getattr(model_args, "log_path", None)
    train_datasets = _first_nonblank(
        model_row.get("train_datasets"),
        model_row.get("Train Datasets"),
        getattr(model_args, "train_datasets", None),
    )
    valid_dataset = _first_nonblank(
        model_row.get("valid_dataset"),
        model_row.get("Valid Dataset"),
        getattr(model_args, "valid_dataset", None),
    )
    test_dataset = _first_nonblank(
        model_row.get("test_dataset"),
        model_row.get("Test Dataset"),
        getattr(model_args, "test_dataset", None),
    )

    n_aug = None

    if not isinstance(entry, dict):
        cfg = str(entry)
        family = _infer_head_family(cfg)
        return {
            "Model ID": model_id,
            "Model": model_name,
            "Log Path": log_path,
            "train_datasets": train_datasets,
            "valid_dataset": valid_dataset,
            "test_dataset": test_dataset,
            "Train Datasets": train_datasets,
            "Valid Dataset": valid_dataset,
            "Test Dataset": test_dataset,
            "N Aug": n_aug,
            "Family": family,
            "Classifier": _family_display_name(family),
            "Head": cfg,
            "Config": cfg,
            "Train MCC": None,
            "Valid MCC": None,
            "Test MCC": None,
            "All MCC": None,
            "Valid Accuracy": None,
            "Train AUC": None,
            "Valid AUC": None,
            "Test AUC": None,
            "All AUC": None,
            "Model Train MCC": _safe_float(model_train_mcc),
            "Model Valid MCC": _safe_float(model_valid_mcc),
            "Model Test MCC": _safe_float(model_test_mcc),
            "Details": "",
        }

    config = (
        entry.get("config")
        or entry.get("head_config")
        or entry.get("classifier_config")
        or entry.get("best_classifier_config")
    )

    label = (
        entry.get("label")
        or entry.get("name")
        or entry.get("head")
        or entry.get("classifier")
    )

    if label is None:
        try:
            label = format_classifier_config(config)
        except Exception:
            label = str(config)

    family = (
        entry.get("family")
        or entry.get("type")
        or _infer_head_family(config or label)
    )

    n_aug = entry.get("n_aug", entry.get("N_Aug", entry.get("N Aug", entry.get("n_augmentations"))))
    valid_mcc = entry.get("valid_mcc", entry.get("Valid MCC", entry.get("mcc", entry.get("MCC"))))
    valid_auc = entry.get("valid_auc", entry.get("Valid AUC", entry.get("auc", entry.get("AUC"))))

    return {
        "Model ID": model_id,
        "Model": model_name,
        "Log Path": log_path,
        "train_datasets": train_datasets,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "Train Datasets": train_datasets,
        "Valid Dataset": valid_dataset,
        "Test Dataset": test_dataset,
        "N Aug": n_aug,
        "Family": family,
        "Classifier": _family_display_name(family),
        "Head": label,
        "Config": config,
        "Train MCC": entry.get("train_mcc", entry.get("Train MCC")),
        "Valid MCC": valid_mcc,
        "Test MCC": entry.get("test_mcc", entry.get("Test MCC")),
        "All MCC": entry.get("all_mcc", entry.get("All MCC")),
        "Valid Accuracy": entry.get("valid_accuracy", entry.get("Valid Accuracy", entry.get("accuracy", entry.get("acc", entry.get("Accuracy"))))),
        "Train AUC": entry.get("train_auc", entry.get("Train AUC")),
        "Valid AUC": valid_auc,
        "Test AUC": entry.get("test_auc", entry.get("Test AUC")),
        "All AUC": entry.get("all_auc", entry.get("All AUC")),
        "Head Cache Version": entry.get("head_cache_version", entry.get("Head Cache Version")),
        "Model Train MCC": _safe_float(model_train_mcc),
        "Model Valid MCC": _safe_float(model_valid_mcc),
        "Model Test MCC": _safe_float(model_test_mcc),
        "Details": entry.get("details", entry.get("source", entry.get("path", ""))),
    }


def _best_display_heads(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one best display row per optimized model/n_aug/family."""
    if not rows:
        return []

    df = pd.DataFrame(rows)
    if "Valid MCC" not in df.columns:
        return rows

    df["Valid MCC"] = pd.to_numeric(df["Valid MCC"], errors="coerce")
    if "Head Cache Version" in df.columns:
        df["_current_head_cache"] = df["Head Cache Version"].eq(LEARNED_HEAD_CACHE_VERSION)
        sort_cols = ["_current_head_cache", "Valid MCC"]
        ascending = [False, False]
    else:
        sort_cols = ["Valid MCC"]
        ascending = [False]

    group_cols = [col for col in ["Model ID", "Log Path", "N Aug", "Family"] if col in df.columns]
    df = df.sort_values(sort_cols, ascending=ascending, na_position="last")
    if group_cols:
        df = df.groupby(group_cols, as_index=False, dropna=False).first()

    df = df.drop(columns=["_current_head_cache"], errors="ignore")
    return df.to_dict("records")


def _get_heads_for_model_args(model_args, model_row):
    try:
        heads = list(enumerate_classification_heads(model_args, include_all_n_aug=True) or [])
    except Exception as e:
        return [], str(e)

    rows = [_normalize_head_entry(h, model_row, model_args=model_args) for h in heads]
    rows = _best_display_heads(rows)
    return rows, None


def _get_heads_for_model(base_args, model_row):
    model_args = args_from_model_row(base_args, model_row)
    return _get_heads_for_model_args(model_args, model_row)


def _has_usable_model_log_paths(df: pd.DataFrame) -> bool:
    if df is None or df.empty or "Log Path" not in df.columns:
        return False
    log_paths = df["Log Path"].astype(str).str.strip().str.lower()
    return bool((df["Log Path"].notna() & ~log_paths.isin({"", "none", "nan", "null"})).any())


def _load_top_models(ctx, top_n):
    active_task = st.session_state.get("production_task") or getattr(ctx.args, "task", None)
    table = st.session_state.get("best_models_table")

    if table is not None:
        try:
            df = attach_task_column(pd.DataFrame(table))
            df = filter_models_df_by_task(df, active_task)
            if len(df) > 0 and _has_usable_model_log_paths(df):
                if int(top_n) >= 99999:
                    return df
                return df.head(int(top_n))
        except Exception:
            pass

    try:
        from otitenet.app.pages.leaderboard import load_best_models_table

        df = load_best_models_table(ctx.cursor, task=active_task)
        df = attach_task_column(df)
        df = filter_models_df_by_task(df, active_task)
        if int(top_n) >= 99999:
            return df
        return df.head(int(top_n))
    except Exception:
        df = fetch_best_model_rows(ctx.cursor, limit=int(top_n))
        df = attach_task_column(df)
        df = filter_models_df_by_task(df, active_task)
        if int(top_n) >= 99999:
            return df
        return df.head(int(top_n))


def _load_existing_heads(ctx, top_n):
    models_df = _load_top_models(ctx, top_n=int(top_n))
    if models_df is None:
        models_df = pd.DataFrame()
    if not models_df.empty:
        if _has_usable_model_log_paths(models_df):
            log_paths = models_df["Log Path"].astype(str).str.strip().str.lower()
            models_df = models_df[
                models_df["Log Path"].notna()
                & ~log_paths.isin({"", "none", "nan", "null"})
            ].copy()
        else:
            models_df = pd.DataFrame()
    try:
        current_row = _current_sidebar_model_row(ctx, int(getattr(ctx.args, "n_aug", 0) or 0))
        current_key = (
            current_row.get("Model ID") or current_row.get("id") or current_row.get("model_id"),
            current_row.get("Log Path") or current_row.get("log_path"),
        )
        if not (_is_blank_value(current_key[0]) and _is_blank_value(current_key[1])):
            existing_keys = set()
            if models_df is not None and len(models_df) > 0:
                for _, existing_row in models_df.iterrows():
                    existing_keys.add(
                        (
                            existing_row.get("Model ID") or existing_row.get("id") or existing_row.get("model_id"),
                            existing_row.get("Log Path") or existing_row.get("log_path"),
                        )
                    )
            if current_key not in existing_keys:
                models_df = pd.concat([models_df, pd.DataFrame([current_row])], ignore_index=True)
    except Exception:
        pass

    all_rows = []
    errors = []

    for _, row in models_df.iterrows():
        model_row = row.to_dict()
        rows, err = _get_heads_for_model(ctx.args, model_row)

        all_rows.extend(rows)

        if err:
            errors.append(
                {
                    "Model ID": model_row.get("Model ID") or model_row.get("id"),
                    "Model": model_row.get("Model Name") or model_row.get("model_name"),
                    "Error": err,
                }
            )

    heads_df = pd.DataFrame(all_rows)
    errors_df = pd.DataFrame(errors)

    if "N Aug" in heads_df.columns:
        n_aug_text = heads_df["N Aug"].astype(str).str.strip().str.lower()
        heads_df = heads_df[
            heads_df["N Aug"].notna()
            & ~n_aug_text.isin({"", "none", "nan", "null"})
        ].copy()

    metric_cols = [
        "Train MCC", "Valid MCC", "Test MCC",
        "All MCC",
        "Valid Accuracy",
        "Train AUC", "Valid AUC", "Test AUC", "All AUC",
        "Model Train MCC", "Model Valid MCC", "Model Test MCC",
    ]
    for col in metric_cols:
        if col in heads_df.columns:
            heads_df[col] = pd.to_numeric(heads_df[col], errors="coerce")

    return heads_df, errors_df


def _best_by_model(heads_df):
    if heads_df is None or len(heads_df) == 0:
        return pd.DataFrame()

    df = heads_df.copy()

    if "Valid MCC" not in df.columns:
        return pd.DataFrame()

    df["Valid MCC"] = pd.to_numeric(df["Valid MCC"], errors="coerce")

    if "Model ID" in df.columns:
        group_cols = ["Model ID"]
    elif "Model" in df.columns:
        group_cols = ["Model"]
    else:
        return df.sort_values("Valid MCC", ascending=False, na_position="last").head(1)

    return (
        df.sort_values("Valid MCC", ascending=False, na_position="last")
        .groupby(group_cols, as_index=False, dropna=False)
        .first()
    )


def _render_metric_row(heads_df):
    if heads_df is None or len(heads_df) == 0:
        return

    best = _best_by_model(heads_df)

    n_models = heads_df["Model ID"].nunique() if "Model ID" in heads_df.columns else 0
    n_heads = len(heads_df)

    best_valid_mcc = None
    if "Valid MCC" in heads_df.columns:
        best_valid_mcc = pd.to_numeric(heads_df["Valid MCC"], errors="coerce").max()

    best_family = "—"
    if len(best) > 0 and "Classifier" in best.columns:
        best_family = str(best.sort_values("Valid MCC", ascending=False).iloc[0]["Classifier"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", n_models)
    c2.metric("Classifier heads", n_heads)
    c3.metric("Best Valid MCC", "—" if pd.isna(best_valid_mcc) else f"{best_valid_mcc:.4f}")
    c4.metric("Top head", best_family)


def _render_charts(heads_df):
    if heads_df is None or len(heads_df) == 0:
        return

    df = heads_df.copy()

    if "Valid MCC" not in df.columns:
        st.info("No Valid MCC column available for plots.")
        return

    df["Valid MCC"] = pd.to_numeric(df["Valid MCC"], errors="coerce")
    df = df.dropna(subset=["Valid MCC"])

    if len(df) == 0:
        st.info("No numeric Valid MCC values available for plots.")
        return

    st.subheader("Visual summaries")

    best = _best_by_model(df)

    if len(best) > 0:
        st.markdown("#### Best classifier head per model")
        plot_df = best[["Model ID", "Valid MCC"]].copy()
        plot_df["Model ID"] = plot_df["Model ID"].astype(str)
        st.bar_chart(plot_df.set_index("Model ID"))

    family_df = (
        df.groupby("Classifier", as_index=False)
        .agg(
            Best_Valid_MCC=("Valid MCC", "max"),
            Mean_Valid_MCC=("Valid MCC", "mean"),
            N=("Valid MCC", "count"),
        )
        .sort_values("Best_Valid_MCC", ascending=False)
    )

    if len(family_df) > 0:
        st.markdown("#### Best Valid MCC by classifier family")
        st.bar_chart(family_df.set_index("Classifier")["Best_Valid_MCC"])

        st.markdown("#### Mean Valid MCC by classifier family")
        st.bar_chart(family_df.set_index("Classifier")["Mean_Valid_MCC"])

    # Show only the best head per Model ID in the comparison table, with short train_datasets column
    if best is not None and len(best) > 0:
        comp_df = best.copy()
        # Add short train_datasets column if present
        if "train_datasets" in comp_df.columns:
            comp_df["Short Train Datasets"] = comp_df["train_datasets"].apply(get_short_dataset_names)
        st.markdown("#### Classifier comparison across models (best head per model)")
        st.dataframe(comp_df, use_container_width=True)


def _render_head_tables(heads_df):
    if heads_df is None or len(heads_df) == 0:
        st.warning(
            "No existing learned classifier heads were found. "
            "This means enumerate_classification_heads() could not resolve the saved KNN, baseline, or prototype results."
        )
        return

    st.subheader("Existing classifier heads")

    display_cols = [
        "Model ID",
        "Model",
        "Train Datasets",
        "Valid Dataset",
        "Test Dataset",
        "Log Path",
        "N Aug",
        "Classifier",
        "Head",
        "Train MCC",
        "Valid MCC",
        "Test MCC",
        "All MCC",
        "Valid Accuracy",
        "Train AUC",
        "Valid AUC",
        "Test AUC",
        "All AUC",
        "Config",
        "Details",
    ]
    display_cols = [c for c in display_cols if c in heads_df.columns]

    display_df = heads_df.copy()
    # Add short name columns for datasets if present
    for col in ["train_datasets", "valid_dataset", "test_dataset", "Train Datasets", "Valid Dataset", "Test Dataset"]:
        if col in display_df.columns:
            short_col = f"Short {col.replace('_', ' ').title()}"
            display_df[short_col] = display_df[col].apply(get_short_dataset_names)

    if "Valid MCC" in display_df.columns:
        display_df = display_df.sort_values(
            by=["Model ID", "Valid MCC"],
            ascending=[True, False],
            na_position="last",
        )

    st.dataframe(display_df[display_cols], use_container_width=True)

    best = _best_by_model(display_df)

    if len(best) > 0:
        st.subheader("Best head by model")

        best_cols = [
            "Model ID",
            "Model",
            "Train Datasets",
            "Valid Dataset",
            "Test Dataset",
            "Log Path",
            "N Aug",
            "Classifier",
            "Head",
            "Train MCC",
            "Valid MCC",
            "Test MCC",
            "All MCC",
            "Valid Accuracy",
            "Train AUC",
            "Valid AUC",
            "Test AUC",
            "All AUC",
            "Config",
        ]
        best_cols = [c for c in best_cols if c in best.columns]

        st.dataframe(best[best_cols], use_container_width=True)


def _render_training_dataset_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "train_datasets" not in df.columns:
        return df

    combo_keys = []
    labels = {}
    for _, row in df.iterrows():
        train_value = str(row.get("train_datasets") or "").strip()
        if not train_value or train_value.lower() in {"none", "nan", "null"}:
            continue
        valid_value = str(row.get("valid_dataset") or "").strip()
        test_value = str(row.get("test_dataset") or "").strip()
        key = (train_value, valid_value, test_value)
        if key not in labels:
            valid_label = get_short_dataset_name(valid_value) if valid_value else "valid: n/a"
            test_label = get_short_dataset_name(test_value) if test_value else "test: n/a"
            labels[key] = f"{get_short_dataset_names(train_value)} | valid: {valid_label} | test: {test_label}"
            combo_keys.append(key)

    if not combo_keys:
        return df

    selected = st.selectbox(
        "Train / valid / test dataset combination",
        options=combo_keys,
        index=0,
        format_func=lambda x: labels.get(x, x),
        key="learned_filter_train_dataset_combo",
        help="Only combinations already present in saved model results are shown.",
    )
    if selected:
        train_value, valid_value, test_value = selected
        st.session_state["selected_train_datasets"] = train_value
        st.session_state["selected_valid_dataset"] = valid_value
        st.session_state["selected_test_dataset"] = test_value
        mask = df["train_datasets"].astype(str) == train_value
        if "valid_dataset" in df.columns:
            mask = mask & (df["valid_dataset"].fillna("").astype(str) == valid_value)
        if "test_dataset" in df.columns:
            mask = mask & (df["test_dataset"].fillna("").astype(str) == test_value)
        df = df[mask]
    return df


def _render_filters(heads_df):
    if heads_df is None or len(heads_df) == 0:
        return heads_df

    df = heads_df.copy()

    st.subheader("Filters")
    df = _render_training_dataset_filter(df)

    c1, c2, c3 = st.columns(3)

    with c1:
        families = sorted([str(x) for x in df["Classifier"].dropna().unique()]) if "Classifier" in df.columns else []
        family_filter_key = "learned_filter_families"
        current_families = st.session_state.get(family_filter_key)
        if current_families is not None:
            current_families = [x for x in current_families if x in families]
            missing_families = [x for x in families if x not in current_families]
            if missing_families:
                st.session_state[family_filter_key] = current_families + missing_families
        selected_families = st.multiselect(
            "Classifier families",
            families,
            default=families,
            key=family_filter_key,
        )

    with c2:
        min_mcc = st.slider(
            "Minimum Valid MCC",
            min_value=-1.0,
            max_value=1.0,
            value=-1.0,
            step=0.01,
            key="learned_filter_min_valid_mcc",
        )

    with c3:
        search = st.text_input(
            "Search head/config",
            value="",
            key="learned_filter_search",
        )

    if selected_families and "Classifier" in df.columns:
        df = df[df["Classifier"].isin(selected_families)]

    if "Valid MCC" in df.columns:
        df["Valid MCC"] = pd.to_numeric(df["Valid MCC"], errors="coerce")
        df = df[df["Valid MCC"].isna() | (df["Valid MCC"] >= min_mcc)]

    if search.strip():
        s = search.strip().lower()
        mask = pd.Series(False, index=df.index)

        for col in ["Head", "Config", "Details", "Classifier", "Model"]:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.lower().str.contains(s, na=False)

        df = df[mask]

    return df


# -------------------------------------------------
# Actual recompute/training of heads
# -------------------------------------------------

def _evaluate_mcc(y_true, y_pred) -> float:
    try:
        return float(MCC(y_true, y_pred))
    except Exception:
        try:
            return float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            return np.nan


def _normalize_auc_scores(score_arr: np.ndarray) -> np.ndarray:
    row_sums = np.nansum(score_arr, axis=1)
    if np.any(score_arr < 0) or not np.allclose(row_sums, 1.0, atol=1e-4):
        shifted = score_arr - np.nanmax(score_arr, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        denom = np.nansum(exp_scores, axis=1, keepdims=True)
        return exp_scores / np.where(denom == 0, 1.0, denom)
    return score_arr / np.where(row_sums[:, None] == 0, 1.0, row_sums[:, None])


def _evaluate_auc_from_scores(y_true, scores, classes=None) -> float:
    try:
        y_arr = np.asarray(y_true)
        score_arr = np.asarray(scores, dtype=float)
        unique_y = np.asarray(sorted(np.unique(y_arr).tolist()))
        if len(unique_y) < 2 or score_arr.size == 0:
            return np.nan

        if score_arr.ndim == 1:
            return float(roc_auc_score(y_arr, score_arr.reshape(-1)))

        if score_arr.ndim == 2 and score_arr.shape[1] > 2:
            if classes is None:
                class_arr = np.arange(score_arr.shape[1])
            else:
                class_arr = np.asarray(classes)
                if len(class_arr) != score_arr.shape[1]:
                    class_arr = np.arange(score_arr.shape[1])

            present_classes = np.asarray([c for c in class_arr if c in set(unique_y.tolist())])
            if len(present_classes) < 2:
                return np.nan

            keep_rows = np.isin(y_arr, present_classes)
            if keep_rows.sum() == 0 or len(np.unique(y_arr[keep_rows])) < 2:
                return np.nan

            col_idx = [int(np.where(class_arr == c)[0][0]) for c in present_classes]
            score_subset = score_arr[keep_rows][:, col_idx]
            y_subset = y_arr[keep_rows]

            if len(present_classes) == 2:
                positive = present_classes[1]
                return float(roc_auc_score((y_subset == positive).astype(int), score_subset[:, 1]))

            score_subset = _normalize_auc_scores(score_subset)
            y_encoded = np.asarray([
                int(np.where(present_classes == label)[0][0])
                for label in y_subset
            ])
            return float(
                roc_auc_score(
                    y_encoded,
                    score_subset,
                    labels=np.arange(len(present_classes)),
                    multi_class="ovr",
                )
            )
        if score_arr.ndim == 2 and score_arr.shape[1] == 2:
            return float(roc_auc_score(y_arr, score_arr[:, 1]))
        return float(roc_auc_score(y_arr, score_arr.reshape(-1)))
    except Exception:
        return np.nan


def _evaluate_classifier_metrics(clf, X, y) -> Dict[str, float]:
    try:
        pred = clf.predict(X)
        mcc = _evaluate_mcc(y, pred)
    except Exception:
        return {"mcc": np.nan, "auc": np.nan}

    auc = np.nan
    try:
        if hasattr(clf, "predict_proba"):
            auc = _evaluate_auc_from_scores(y, clf.predict_proba(X), classes=getattr(clf, "classes_", None))
        elif hasattr(clf, "decision_function"):
            auc = _evaluate_auc_from_scores(y, clf.decision_function(X), classes=getattr(clf, "classes_", None))
    except Exception:
        auc = np.nan

    return {"mcc": float(mcc), "auc": float(auc) if not pd.isna(auc) else np.nan}


def _prototype_metrics(proto_vecs, proto_labels, X, y) -> Dict[str, float]:
    try:
        dists = np.linalg.norm(X[:, None, :] - proto_vecs[None, :, :], axis=2)
        pred = proto_labels[np.argmin(dists, axis=1)]
        labels = np.unique(proto_labels)
        scores = []
        for label in labels:
            mask = proto_labels == label
            label_dist = np.min(dists[:, mask], axis=1) if np.any(mask) else np.full(X.shape[0], np.inf)
            scores.append(-label_dist)
        score_arr = np.vstack(scores).T if scores else np.empty((X.shape[0], 0))
        return {
            "mcc": _evaluate_mcc(y, pred),
            "auc": _evaluate_auc_from_scores(y, score_arr, classes=labels),
        }
    except Exception:
        return {"mcc": np.nan, "auc": np.nan}


def _baseline_models(random_state: int = 1):
    return {
        "logreg": LogisticRegression(max_iter=1000, C=1.0, random_state=random_state),
        "ridge": RidgeClassifier(alpha=1.0),
        "naive_bayes": GaussianNB(),
        "linear_svc": LinearSVC(max_iter=5000, random_state=random_state),
        "rbf_svc": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(),
    }


def _ensure_training_args(_args):
    """Patch missing args fields expected by TrainAE."""
    if not hasattr(_args, "bs"):
        _args.bs = 32
    if not hasattr(_args, "groupkfold"):
        _args.groupkfold = 1
    if not hasattr(_args, "random_recs"):
        _args.random_recs = 0
    if not hasattr(_args, "device"):
        _args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hasattr(_args, "valid_dataset"):
        _args.valid_dataset = ""
    if not hasattr(_args, "prototypes_to_use"):
        _args.prototypes_to_use = "class"
    if not hasattr(_args, "normalize"):
        _args.normalize = "no"
    if not hasattr(_args, "classif_loss"):
        _args.classif_loss = ""
    return _args


def _encode_splits_for_args(_args):
    """Encode train/valid/test embeddings for one trained model."""
    _args = _ensure_training_args(_args)

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
    train.epoch = 1
    train.model = model
    train.params = {
        "n_neighbors": int(getattr(_args, "n_neighbors", 1)),
        "lr": 0,
        "wd": 0,
        "smoothing": 0,
        "is_transform": 0,
        "valid_dataset": getattr(_args, "valid_dataset", ""),
        "n_aug": int(getattr(_args, "n_aug", 0) or 0),
    }
    train.set_arcloss()

    unique_labels_arr = np.asarray(unique_labels)
    groups = []
    for split in ["train", "valid", "test"]:
        split_inputs = data.get("inputs", {}).get(split)
        if split_inputs is None or len(split_inputs) == 0:
            continue

        split_labels = np.asarray(data.get("labels", {}).get(split))
        split_cats = np.array([
            int(np.argwhere(label == unique_labels_arr).flatten()[0])
            for label in split_labels
        ])

        train.all_samples["inputs"][split] = split_inputs
        train.all_samples["labels"][split] = split_labels
        train.all_samples["old_labels"][split] = np.asarray(data.get("old_labels", {}).get(split, split_labels))
        train.all_samples["names"][split] = np.asarray(data.get("names", {}).get(split, np.arange(len(split_labels))))
        train.all_samples["cats"][split] = split_cats
        train.all_samples["batches"][split] = np.asarray(data.get("batches", {}).get(split, np.zeros(len(split_labels))))
        groups.append(split)

    model.eval()
    with torch.no_grad():
        train.make_encoded_values(groups=groups)

    encoded = {}
    for split in groups:
        X = train.all_samples["encoded_values"].get(split)
        y = train.all_samples["cats"].get(split)
        if X is not None and y is not None and len(X) > 0:
            encoded[split] = {"X": np.asarray(X), "y": np.asarray(y)}

    if "train" not in encoded or "valid" not in encoded:
        raise RuntimeError("Could not encode train/valid embeddings.")

    return encoded


def _compute_knn_heads(X_train, y_train, X_valid, y_valid, k_values, X_test=None, y_test=None, on_head_start=None):
    mcc_per_k = []
    best_k = None
    best_mcc = -1.0

    for k in k_values:
        k_eff = max(1, min(int(k), len(X_train)))
        if on_head_start:
            on_head_start(f"KNN k={k_eff}")
        try:
            clf = KNeighborsClassifier(n_neighbors=k_eff, metric="minkowski")
            clf.fit(X_train, y_train)
            train_metrics = _evaluate_classifier_metrics(clf, X_train, y_train)
            valid_metrics = _evaluate_classifier_metrics(clf, X_valid, y_valid)
            test_metrics = _evaluate_classifier_metrics(clf, X_test, y_test) if X_test is not None and y_test is not None else {}
            mcc = valid_metrics["mcc"]

            mcc_per_k.append({
                "k": int(k_eff),
                "train_mcc": train_metrics.get("mcc"),
                "valid_mcc": valid_metrics.get("mcc"),
                "test_mcc": test_metrics.get("mcc"),
                "train_auc": train_metrics.get("auc"),
                "valid_auc": valid_metrics.get("auc"),
                "test_auc": test_metrics.get("auc"),
                "mcc": float(mcc),
            })

            if not pd.isna(mcc) and float(mcc) > best_mcc:
                best_mcc = float(mcc)
                best_k = int(k_eff)
        except Exception as e:
            mcc_per_k.append({"k": int(k_eff), "valid_mcc": np.nan, "mcc": np.nan, "error": str(e)})

    return {
        "best_k": best_k,
        "best_mcc": best_mcc,
        "mcc_per_k": mcc_per_k,
    }


def _compute_prototype_heads(X_train, y_train, X_valid, y_valid, strategies, components, X_test=None, y_test=None, on_head_start=None):
    out = {}

    for strategy in strategies:
        best_mcc = -1.0
        best_n_components = None
        best_n_prototypes = 0
        best_metrics = {}
        per_components = []

        for n_comp in components:
            n_comp = int(n_comp)
            if on_head_start:
                on_head_start(f"Prototype {str(strategy).upper()} n_comp={n_comp}")
            try:
                proto_dict = compute_prototypes_by_strategy(
                    X_train,
                    y_train,
                    strategy,
                    n_comp,
                    random_state=1,
                )
                proto_vecs, proto_labels = flatten_prototype_dict(proto_dict)

                if len(proto_vecs) == 0:
                    continue

                train_metrics = _prototype_metrics(proto_vecs, proto_labels, X_train, y_train)
                valid_metrics = _prototype_metrics(proto_vecs, proto_labels, X_valid, y_valid)
                test_metrics = _prototype_metrics(proto_vecs, proto_labels, X_test, y_test) if X_test is not None and y_test is not None else {}
                mcc = valid_metrics["mcc"]

                per_components.append(
                    {
                        "n_components": n_comp,
                        "mcc": float(mcc),
                        "train_mcc": train_metrics.get("mcc"),
                        "valid_mcc": valid_metrics.get("mcc"),
                        "test_mcc": test_metrics.get("mcc"),
                        "train_auc": train_metrics.get("auc"),
                        "valid_auc": valid_metrics.get("auc"),
                        "test_auc": test_metrics.get("auc"),
                        "n_prototypes": int(len(proto_vecs)),
                    }
                )

                if not pd.isna(mcc) and float(mcc) > best_mcc:
                    best_mcc = float(mcc)
                    best_n_components = n_comp
                    best_n_prototypes = int(len(proto_vecs))
                    best_metrics = {
                        "train_mcc": train_metrics.get("mcc"),
                        "valid_mcc": valid_metrics.get("mcc"),
                        "test_mcc": test_metrics.get("mcc"),
                        "train_auc": train_metrics.get("auc"),
                        "valid_auc": valid_metrics.get("auc"),
                        "test_auc": test_metrics.get("auc"),
                    }

            except Exception as e:
                per_components.append(
                    {
                        "n_components": n_comp,
                        "mcc": np.nan,
                        "n_prototypes": 0,
                        "error": str(e),
                    }
                )

        out[strategy] = {
            "best_mcc": best_mcc if best_n_components is not None else None,
            "best_n_components": best_n_components,
            "n_prototypes": best_n_prototypes,
            "per_components": per_components,
        }
        if best_n_components is not None:
            out[strategy].update(best_metrics)

    return out


def _compute_baseline_heads(X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, on_head_start=None):
    out = {}

    for name, clf in _baseline_models().items():
        if on_head_start:
            on_head_start(format_classifier_config(f"baseline_{name}"))
        try:
            clf.fit(X_train, y_train)
            train_metrics = _evaluate_classifier_metrics(clf, X_train, y_train)
            valid_metrics = _evaluate_classifier_metrics(clf, X_valid, y_valid)
            test_metrics = _evaluate_classifier_metrics(clf, X_test, y_test) if X_test is not None and y_test is not None else {}
            out[name] = {
                "mcc": valid_metrics.get("mcc"),
                "valid_mcc": valid_metrics.get("mcc"),
                "test_mcc": test_metrics.get("mcc"),
                "train_mcc": train_metrics.get("mcc"),
                "valid_auc": valid_metrics.get("auc"),
                "test_auc": test_metrics.get("auc"),
                "train_auc": train_metrics.get("auc"),
            }
        except Exception as e:
            out[name] = {"mcc": None, "error": str(e)}

    return out


def _choose_best_config(result):
    best_config = None
    best_mcc = -1.0

    knn = result.get("knn", {}) or {}
    if knn.get("best_k") is not None and knn.get("best_mcc") is not None:
        best_config = str(knn["best_k"])
        best_mcc = float(knn["best_mcc"])

    for strategy, strat_data in (result.get("prototypes", {}) or {}).items():
        mcc = strat_data.get("best_mcc")
        n_comp = strat_data.get("best_n_components")
        if mcc is not None and n_comp is not None and float(mcc) > best_mcc:
            best_mcc = float(mcc)
            best_config = f"protot_{strategy}_{int(n_comp)}"

    for baseline_name, b_data in (result.get("baselines", {}) or {}).items():
        if not isinstance(b_data, dict):
            continue
        mcc = b_data.get("mcc")
        if mcc is not None and float(mcc) > best_mcc:
            best_mcc = float(mcc)
            best_config = f"baseline_{baseline_name}"

    return best_config, best_mcc


def _metrics_for_config(result, config):
    if config is None:
        return {}
    config = str(config)

    if config.isdigit():
        for item in (result.get("knn", {}) or {}).get("mcc_per_k", []) or []:
            if isinstance(item, dict) and str(item.get("k")) == config:
                return item
        return {}

    if config.startswith("protot_"):
        parts = config.split("_")
        if len(parts) >= 3:
            strategy = parts[1]
            data = (result.get("prototypes", {}) or {}).get(strategy, {}) or {}
            return data

    if config.startswith("baseline_"):
        baseline_name = config[len("baseline_"):]
        data = (result.get("baselines", {}) or {}).get(baseline_name, {}) or {}
        return data if isinstance(data, dict) else {}

    return {}


def train_heads_for_args(
    _args,
    k_values,
    prototype_strategies,
    prototype_components,
    include_knn=True,
    include_prototypes=True,
    include_baselines=True,
    overwrite_current_n_aug=True,
    progress_callback=None,
):
    """Compute learned-embedding classifier heads and write the optimization cache."""
    encoded = _encode_splits_for_args(_args)

    X_train = encoded["train"]["X"]
    y_train = encoded["train"]["y"]
    X_valid = encoded["valid"]["X"]
    y_valid = encoded["valid"]["y"]
    X_test = encoded.get("test", {}).get("X")
    y_test = encoded.get("test", {}).get("y")

    result = {
        "knn": {},
        "prototypes": {},
        "baselines": {},
        "best_k": None,
        "best_config": None,
        "best_mcc": -1.0,
        "best_valid_mcc": -1.0,
        "head_cache_version": LEARNED_HEAD_CACHE_VERSION,
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "n_test": int(len(y_test)) if y_test is not None else 0,
        "train_datasets": getattr(_args, "train_datasets", ""),
        "valid_dataset": getattr(_args, "valid_dataset", ""),
        "test_dataset": getattr(_args, "test_dataset", ""),
    }

    total_heads = 0
    if include_knn:
        total_heads += len(k_values)
    if include_prototypes:
        total_heads += len(prototype_strategies) * len(prototype_components)
    if include_baselines:
        total_heads += len(_baseline_models())
    head_index = 0

    def on_head_start(head_label: str):
        nonlocal head_index
        head_index += 1
        if progress_callback:
            progress_callback(
                {
                    "head": head_label,
                    "head_index": head_index,
                    "total_heads": max(total_heads, 1),
                    "heads_left": max(total_heads - head_index, 0),
                    "n_aug": int(getattr(_args, "n_aug", 0) or 0),
                }
            )

    if include_knn:
        result["knn"] = _compute_knn_heads(
            X_train,
            y_train,
            X_valid,
            y_valid,
            k_values,
            X_test=X_test,
            y_test=y_test,
            on_head_start=on_head_start,
        )

    if include_prototypes:
        result["prototypes"] = _compute_prototype_heads(
            X_train,
            y_train,
            X_valid,
            y_valid,
            strategies=prototype_strategies,
            components=prototype_components,
            X_test=X_test,
            y_test=y_test,
            on_head_start=on_head_start,
        )

    if include_baselines:
        result["baselines"] = _compute_baseline_heads(
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test=X_test,
            y_test=y_test,
            on_head_start=on_head_start,
        )

    best_config, best_mcc = _choose_best_config(result)
    result["best_k"] = best_config
    result["best_config"] = best_config
    result["best_mcc"] = float(best_mcc) if best_config is not None else -1.0
    result["best_valid_mcc"] = result["best_mcc"]
    result["best_head_metrics"] = _metrics_for_config(result, best_config)

    n_aug = int(getattr(_args, "n_aug", 0) or 0)
    cache_path = get_optimization_cache_file_path(_args)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                old_cache = pickle.load(f)
            if isinstance(old_cache, dict):
                cache.update(old_cache)
        except Exception:
            pass

    if overwrite_current_n_aug or n_aug not in cache:
        cache[n_aug] = result
    else:
        # Preserve old entry and put this one under a new numeric key.
        new_key = max([int(k) for k in cache.keys() if str(k).isdigit()] + [n_aug]) + 1
        cache[new_key] = result

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return result, cache_path


def _clear_existing_heads_cache():
    for key in list(st.session_state.keys()):
        if key.startswith("learned_existing_heads_df") or key.startswith("learned_existing_errors"):
            st.session_state.pop(key, None)


def _result_summary_row(model_row, result, cache_path):
    model_id = None
    model_name = None

    if isinstance(model_row, dict):
        model_id = model_row.get("Model ID") or model_row.get("id") or model_row.get("model_id")
        model_name = model_row.get("Model Name") or model_row.get("model_name") or model_row.get("model")

    return {
        "Status": "trained",
        "Model ID": model_id,
        "Model": model_name,
        "N Aug": model_row.get("N_Aug", model_row.get("n_aug")) if isinstance(model_row, dict) else None,
        "Best Config": result.get("best_config"),
        "Best Head": format_classifier_config(result.get("best_config")),
        "Best Valid MCC": result.get("best_valid_mcc", result.get("best_mcc")),
        "Best Test MCC": (result.get("best_head_metrics") or {}).get("test_mcc"),
        "Best Valid AUC": (result.get("best_head_metrics") or {}).get("valid_auc"),
        "Best Test AUC": (result.get("best_head_metrics") or {}).get("test_auc"),
        "Train Datasets": result.get("train_datasets") or (model_row.get("train_datasets") if isinstance(model_row, dict) else None),
        "Valid Dataset": result.get("valid_dataset") or (model_row.get("valid_dataset") if isinstance(model_row, dict) else None),
        "Test Dataset": result.get("test_dataset") or (model_row.get("test_dataset") if isinstance(model_row, dict) else None),
        "N train": result.get("n_train"),
        "N valid": result.get("n_valid"),
        "N test": result.get("n_test"),
        "Cache Path": cache_path,
    }


def _metric_present(value) -> bool:
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except Exception:
        return True


def _load_n_aug_cache_entry(_args, n_aug: int) -> Tuple[Dict[str, Any] | None, str]:
    cache_path = get_optimization_cache_file_path(_args)
    if not os.path.exists(cache_path):
        return None, cache_path

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except Exception:
        return None, cache_path

    if not isinstance(cache, dict):
        return None, cache_path

    n_aug_int = int(n_aug)
    candidates = [cache.get(n_aug_int), cache.get(str(n_aug_int))]
    entry = next((x for x in candidates if isinstance(x, dict)), None)
    return entry, cache_path


def _cache_summary_row(model_row, _args, n_aug: int, status: str) -> Dict[str, Any]:
    entry, cache_path = _load_n_aug_cache_entry(_args, n_aug)
    if not entry:
        return {
            "Status": "error_missing_cache_entry",
            "Model ID": model_row.get("Model ID") if isinstance(model_row, dict) else None,
            "Model": model_row.get("Model Name") if isinstance(model_row, dict) else getattr(_args, "model_name", None),
            "N Aug": int(n_aug),
            "Best Config": None,
            "Best Head": None,
            "Best Valid MCC": None,
            "N train": None,
            "N valid": None,
            "Cache Path": cache_path,
        }
    row = _result_summary_row(model_row, entry, cache_path)
    row["Status"] = status
    row["N Aug"] = int(n_aug)
    return row


def _cache_has_n_aug_entry(_args, n_aug: int) -> Tuple[bool, str]:
    entry, cache_path = _load_n_aug_cache_entry(_args, n_aug)
    if not entry:
        return False, cache_path
    metrics = entry.get("best_head_metrics") or {}
    has_complete_split_metrics = (
        entry.get("best_config") is not None
        and entry.get("head_cache_version") == LEARNED_HEAD_CACHE_VERSION
        and _metric_present(metrics.get("valid_mcc"))
        and _metric_present(metrics.get("valid_auc"))
        and _metric_present(metrics.get("test_mcc"))
        and _metric_present(metrics.get("test_auc"))
    )
    return bool(has_complete_split_metrics), cache_path


def _resolved_model_path_for_args(_args) -> str:
    params = get_model_params_path(_args)
    parts = params.split("/")
    base_params = "/".join(parts[:-3]) if len(parts) > 3 else params
    base_dir = f"logs/best_models/{_args.task}/{_args.model_name}/{base_params}"
    model_path, _proto_path, _resolved_k = resolve_model_paths(
        base_dir,
        str(getattr(_args, "normalize", "no")),
        str(getattr(_args, "dist_fct", "euclidean")),
        int(getattr(_args, "n_neighbors", 1) or 1),
    )
    if (
        not os.path.exists(model_path)
        and bool(getattr(_args, "split_config_in_path", False) or getattr(_args, "_split_config_in_path", False))
    ):
        old_split_flag = getattr(_args, "split_config_in_path", False)
        old_private_flag = getattr(_args, "_split_config_in_path", False)
        try:
            _args.split_config_in_path = False
            _args._split_config_in_path = False
            legacy_params = get_model_params_path(_args)
            legacy_parts = legacy_params.split("/")
            legacy_base_params = "/".join(legacy_parts[:-3]) if len(legacy_parts) > 3 else legacy_params
            legacy_base_dir = f"logs/best_models/{_args.task}/{_args.model_name}/{legacy_base_params}"
            legacy_model_path, _legacy_proto_path, _legacy_k = resolve_model_paths(
                legacy_base_dir,
                str(getattr(_args, "normalize", "no")),
                str(getattr(_args, "dist_fct", "euclidean")),
                int(getattr(_args, "n_neighbors", 1) or 1),
            )
            if os.path.exists(legacy_model_path):
                return legacy_model_path
        finally:
            _args.split_config_in_path = old_split_flag
            _args._split_config_in_path = old_private_flag
    return model_path


def _checkpoint_class_count(model_path: str) -> int | None:
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None

    normalized = {}
    for key, value in state.items():
        norm_key = str(key).replace("module.", "", 1)
        normalized[norm_key] = value

    for key in ("subcenters", "linear.weight", "linear.bias"):
        value = normalized.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
    return None


def _checkpoint_task_mismatch(_args) -> Tuple[bool, str, str | None]:
    try:
        expected_n_cats = len(labels_for_task(getattr(_args, "task", None)))
    except Exception:
        return False, "", None

    try:
        model_path = _resolved_model_path_for_args(_args)
    except Exception as e:
        return False, f"Could not resolve checkpoint path before training: {e}", None

    if not os.path.exists(model_path):
        return False, "", model_path

    try:
        checkpoint_n_cats = _checkpoint_class_count(model_path)
    except Exception as e:
        return False, f"Could not inspect checkpoint class count before training: {e}", model_path

    if checkpoint_n_cats is not None and checkpoint_n_cats != expected_n_cats:
        return (
            True,
            "Model/task class-count mismatch before classifier-head training. "
            f"checkpoint_n_cats={checkpoint_n_cats}, task_n_cats={expected_n_cats}, "
            f"task={getattr(_args, 'task', None)}, model_path={model_path}. "
            "Retrain this neural-network model for the active task before recalculating heads.",
            model_path,
        )
    return False, "", model_path


def _is_class_count_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "model/task class-count mismatch" in message
        or ("checkpoint_n_cats" in message and "dataset_n_cats" in message)
        or "size mismatch for subcenters" in message
    )


def _extract_model_path_from_error(exc: Exception) -> str | None:
    match = re.search(r"model_path=([^\n]+?\.pth)", str(exc))
    if match:
        return match.group(1).strip()
    return None


def _training_status_row(
    model_row: Dict[str, Any],
    model_args,
    n_aug: int,
    status: str,
    message: str,
    cache_path: str | None = None,
) -> Dict[str, Any]:
    model_id = model_row.get("Model ID") if isinstance(model_row, dict) else getattr(model_args, "model_id", None)
    model_name = None
    if isinstance(model_row, dict):
        model_name = model_row.get("Model Name") or model_row.get("model_name") or model_row.get("Model")
    model_name = model_name or getattr(model_args, "model_name", None)

    return {
        "Status": status,
        "Model ID": model_id,
        "Model": model_name,
        "N Aug": int(n_aug),
        "Best Config": None,
        "Best Head": None,
        "Best Valid MCC": None,
        "Best Test MCC": None,
        "Best Valid AUC": None,
        "Best Test AUC": None,
        "N train": None,
        "N valid": None,
        "N test": None,
        "Message": message,
        "Cache Path": cache_path or message,
    }


def _current_sidebar_model_row(ctx, n_aug: int) -> Dict[str, Any]:
    selected = dict(st.session_state.get("selected_model_params", {}) or {})
    selection_key = st.session_state.get("selected_model_selection_key")
    model_number = None
    if selection_key:
        model_number = st.session_state.get("model_number_map", {}).get(selection_key)

    model_id = _first_nonblank(
        selected.get("Model ID"),
        selected.get("model_id"),
        getattr(ctx.args, "model_id", None),
    )
    model_name = _first_nonblank(
        selected.get("Model Name"),
        selected.get("model_name"),
        getattr(ctx.args, "model_name", None),
    )
    log_path = _first_nonblank(
        selected.get("Log Path"),
        selected.get("log_path"),
        getattr(ctx.args, "log_path", None),
    )
    model_number = _first_nonblank(selected.get("#"), selected.get("model_number"), model_number)
    if _is_blank_value(model_number):
        try:
            table_obj = st.session_state.get("best_models_table")
            table = pd.DataFrame(table_obj) if table_obj is not None else pd.DataFrame()
            for _, row in table.iterrows():
                row_id = _first_nonblank(row.get("Model ID"), row.get("id"), row.get("model_id"))
                row_log_path = _first_nonblank(row.get("Log Path"), row.get("log_path"))
                if (not _is_blank_value(model_id) and str(row_id) == str(model_id)) or (
                    not _is_blank_value(log_path) and str(row_log_path) == str(log_path)
                ):
                    model_number = row.get("#")
                    break
        except Exception:
            pass

    return {
        **selected,
        "Model ID": model_id,
        "model_id": model_id,
        "Model Name": model_name,
        "model_name": model_name,
        "#": model_number,
        "Task": selected.get("Task") or selected.get("task") or getattr(ctx.args, "task", None),
        "Log Path": log_path,
        "train_datasets": selected.get("train_datasets") or getattr(ctx.args, "train_datasets", None),
        "valid_dataset": selected.get("valid_dataset") or getattr(ctx.args, "valid_dataset", None),
        "test_dataset": selected.get("test_dataset") or getattr(ctx.args, "test_dataset", None),
        "N_Aug": int(n_aug),
    }


def _model_training_label(model_row, model_args, n_aug_label) -> str:
    model_id = None
    model_name = getattr(model_args, "model_name", None)
    model_number = None
    if isinstance(model_row, dict):
        model_id = _first_nonblank(model_row.get("Model ID"), model_row.get("id"), model_row.get("model_id"))
        model_name = _first_nonblank(model_row.get("Model Name"), model_row.get("model_name"), model_row.get("Model"), model_name)
        model_number = _first_nonblank(model_row.get("#"), model_row.get("model_number"))

    if not _is_blank_value(model_number) and not _is_blank_value(model_id):
        return f"model #{model_number} / ID {model_id} / n_aug={n_aug_label}"
    if not _is_blank_value(model_id):
        return f"model ID {model_id} ({model_name}) / n_aug={n_aug_label}"
    if not _is_blank_value(model_name):
        return f"{model_name} / n_aug={n_aug_label}"
    return f"current sidebar model / n_aug={n_aug_label}"



def _parse_int_list(text, default=None):
    if default is None:
        default = [0]

    if text is None:
        return default

    raw = str(text).strip()
    if not raw:
        return default

    vals = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(float(part)))
        except Exception:
            pass

    return vals or default


def _display_training_results(payload: Dict[str, Any], replay: bool = False):
    if not payload:
        return

    summary_rows = payload.get("summary_rows") or []
    trained_head_rows = payload.get("trained_head_rows") or []
    trained_head_errors = payload.get("trained_head_errors") or []

    st.success("Last classifier-head training finished." if replay else "Finished classifier-head training.")
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    if trained_head_rows:
        st.subheader("Classifier heads now available for trained jobs")
        trained_heads_df = pd.DataFrame(trained_head_rows)
        if "Valid MCC" in trained_heads_df.columns:
            trained_heads_df["Valid MCC"] = pd.to_numeric(trained_heads_df["Valid MCC"], errors="coerce")
            trained_heads_df = trained_heads_df.sort_values(
                ["Model ID", "N Aug", "Valid MCC"],
                ascending=[True, True, False],
                na_position="last",
            )
        trained_cols = [
            "Model ID",
            "Model",
            "N Aug",
            "Classifier",
            "Head",
            "Train MCC",
            "Valid MCC",
            "Test MCC",
            "Train AUC",
            "Valid AUC",
            "Test AUC",
            "Config",
            "Details",
        ]
        trained_cols = [c for c in trained_cols if c in trained_heads_df.columns]
        st.dataframe(trained_heads_df[trained_cols], use_container_width=True)

    if trained_head_errors:
        with st.expander("Trained-job heads that could not be reloaded"):
            st.dataframe(pd.DataFrame(trained_head_errors), use_container_width=True)


def _merge_recent_training_heads(heads_df: pd.DataFrame) -> pd.DataFrame:
    payload = st.session_state.get("learned_last_training_payload") or {}
    recent_rows = payload.get("trained_head_rows") or []
    if not recent_rows:
        return heads_df

    recent_df = pd.DataFrame(recent_rows)
    if recent_df.empty:
        return heads_df

    base_df = heads_df if heads_df is not None else pd.DataFrame()
    combined = pd.concat([base_df, recent_df], ignore_index=True)
    if combined.empty:
        return combined

    dedupe_cols = [
        col
        for col in ["Model ID", "Log Path", "Model", "N Aug", "Family", "Classifier", "Config"]
        if col in combined.columns
    ]
    if dedupe_cols:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    return combined


def _render_training_controls(ctx):
    st.subheader("Train / recalculate classifier heads")

    st.caption(
        "This does not retrain the neural network. It freezes the selected embedding model, "
        "encodes train/valid samples, then fits KNN, prototype, and baseline classifier heads."
    )

    with st.expander("Training settings", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            train_target = st.radio(
                "What to train",
                ["Current sidebar model", "Top N models in bulk"],
                index=0,
                key="learned_train_target",
            )

            bulk_top_n = st.number_input(
                "Bulk: top N models",
                min_value=1,
                max_value=200,
                value=5,
                step=1,
                key="learned_train_bulk_top_n",
                disabled=(train_target != "Top N models in bulk"),
            )

        with c2:
            k_max = st.number_input(
                "Max K for KNN",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                key="learned_train_k_max",
            )

            proto_max = st.number_input(
                "Max prototype components",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="learned_train_proto_max",
            )

        with c3:
            current_n_aug = int(getattr(ctx.args, "n_aug", 0) or 0)
            n_aug_text = st.text_input(
                "n_aug value(s)",
                value=str(current_n_aug),
                key="learned_train_n_aug_values",
                help=(
                    "One or more n_aug values, comma-separated. "
                    "Example: 0,1,2. Each value is saved as a separate key in knn_optimization_cache.pkl."
                ),
            )
            include_knn = st.checkbox("KNN", value=True, key="learned_train_include_knn")
            include_prototypes = st.checkbox("Prototypes", value=True, key="learned_train_include_prototypes")
            include_baselines = st.checkbox("Baselines", value=True, key="learned_train_include_baselines")
            skip_existing = st.checkbox(
                "Skip existing n_aug cache entries",
                value=True,
                key="learned_train_skip_existing",
                help=(
                    "Use this to resume after an interrupted bulk run. "
                    "Models/n_aug values already present in knn_optimization_cache.pkl are skipped."
                ),
            )
            overwrite = st.checkbox("Overwrite selected n_aug cache entry", value=True, key="learned_train_overwrite")

        prototype_strategies = st.multiselect(
            "Prototype strategies",
            ["mean", "kmeans", "gmm"],
            default=["mean", "kmeans", "gmm"],
            key="learned_train_proto_strategies",
            disabled=not include_prototypes,
        )

    run = st.button(
        "🚀 Train / recalculate heads",
        type="primary",
        key="learned_train_run",
    )

    if not run:
        _display_training_results(st.session_state.get("learned_last_training_payload"), replay=True)
        return

    # --- REFRESH LEADERBOARD TABLES after training heads ---
    try:
        from otitenet.app.pages.leaderboard import load_best_models_table, _update_model_number_map
        active_task = st.session_state.get("production_task") or getattr(ctx.args, "task", None)
        leaderboard_df = load_best_models_table(ctx.cursor, task=active_task)
        _update_model_number_map(leaderboard_df)
        # Update session state cache to ensure top models table reflects latest data
        st.session_state["best_models_table"] = leaderboard_df
        # Clear learned embedding heads cache to ensure it syncs with updated models
        _clear_existing_heads_cache()
    except Exception as e:
        st.warning(f"Could not refresh leaderboard table after training heads: {e}")

    clear_cached_model()

    k_values = list(range(1, int(k_max) + 1))
    proto_components = list(range(1, int(proto_max) + 1))
    prototype_strategies = prototype_strategies or ["mean"]
    n_aug_values = _parse_int_list(n_aug_text, default=[int(getattr(ctx.args, "n_aug", 0) or 0)])

    summary_rows = []

    if train_target == "Current sidebar model":
        jobs = []
        for n_aug in n_aug_values:
            row_with_aug = _current_sidebar_model_row(ctx, int(n_aug))
            try:
                model_args = args_from_model_row(ctx.args, row_with_aug)
                model_args.n_aug = int(n_aug)
                jobs.append((row_with_aug, model_args))
            except Exception as e:
                summary_rows.append(
                    {
                        "Status": "error_building_args",
                        "Model ID": row_with_aug.get("Model ID"),
                        "Model": row_with_aug.get("Model Name") or row_with_aug.get("model_name"),
                        "N Aug": int(n_aug),
                        "Best Config": None,
                        "Best Head": None,
                        "Best Valid MCC": None,
                        "N train": None,
                        "N valid": None,
                        "Cache Path": f"ERROR while building args: {e}",
                    }
                )
    else:
        top_models = _load_top_models(ctx, int(bulk_top_n))
        jobs = []

        for _, row in top_models.iterrows():
            row_dict = row.to_dict()
            for n_aug in n_aug_values:
                try:
                    model_args = args_from_model_row(ctx.args, row_dict)
                    model_args.n_aug = int(n_aug)
                    row_with_aug = dict(row_dict)
                    row_with_aug["N_Aug"] = int(n_aug)
                    jobs.append((row_with_aug, model_args))
                except Exception as e:
                    summary_rows.append(
                        {
                            "Status": "error_building_args",
                            "Model ID": row_dict.get("Model ID") or row_dict.get("id"),
                            "Model": row_dict.get("Model Name") or row_dict.get("model_name"),
                            "N Aug": int(n_aug),
                            "Best Config": None,
                            "Best Head": None,
                            "Best Valid MCC": None,
                            "N train": None,
                            "N valid": None,
                            "Cache Path": f"ERROR while building args: {e}",
                        }
                    )

    def _stack_key(model_row, model_args):
        if isinstance(model_row, dict):
            return (
                model_row.get("Model ID")
                or model_row.get("id")
                or model_row.get("Log Path")
                or getattr(model_args, "model_name", "model")
            )
        return getattr(model_args, "model_id", None) or getattr(model_args, "model_name", "model")

    def _stack_meta(model_row, model_args):
        if isinstance(model_row, dict):
            return {
                "Model ID": model_row.get("Model ID") or model_row.get("id"),
                "Model": model_row.get("Model Name") or model_row.get("model_name") or getattr(model_args, "model_name", None),
            }
        return {"Model ID": getattr(model_args, "model_id", None), "Model": getattr(model_args, "model_name", None)}

    stack_totals: Dict[Any, int] = {}
    stack_done: Dict[Any, int] = {}
    stack_status: Dict[Any, str] = {}
    stack_rows: Dict[Any, Dict[str, Any]] = {}
    for model_row, model_args in jobs:
        key = _stack_key(model_row, model_args)
        stack_totals[key] = stack_totals.get(key, 0) + 1
        stack_done.setdefault(key, 0)
        stack_status.setdefault(key, "pending")
        stack_rows.setdefault(key, _stack_meta(model_row, model_args))


    # Always show the tracker table above progress bars
    tracker_placeholder = st.container()
    progress = st.progress(0)
    status = st.empty()
    head_progress = st.progress(0)
    head_tracker = st.empty()
    st.markdown("#### Training updates")
    training_log_placeholder = st.empty()
    training_log_rows: List[Dict[str, Any]] = []

    def _log_training_update(message: str, level: str = "info"):
        training_log_rows.append(
            {
                "Step": len(training_log_rows) + 1,
                "Level": level,
                "Update": message,
            }
        )
        training_log_placeholder.dataframe(
            pd.DataFrame(training_log_rows),
            use_container_width=True,
            hide_index=True,
        )

    def _render_training_tracker(current_key=None):
        if not stack_rows:
            return
        rows = []
        for key, meta in stack_rows.items():
            total = stack_totals.get(key, 0)
            done = stack_done.get(key, 0)
            state = stack_status.get(key, "pending")
            if key == current_key and done < total:
                state = "running"
            rows.append(
                {
                    "Model ID": meta.get("Model ID"),
                    "Model": meta.get("Model"),
                    "Done": int(done),
                    "Total": int(total),
                    "Remaining": int(max(total - done, 0)),
                    "Status": state,
                }
            )
        with tracker_placeholder:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    def _finish_job(key, state: str, idx: int):
        stack_done[key] = min(stack_done.get(key, 0) + 1, stack_totals.get(key, 1))
        stack_status[key] = state
        progress.progress((idx + 1) / max(len(jobs), 1))
        _render_training_tracker()

    # Always render the tracker table before training starts
    _render_training_tracker()

    for idx, (model_row, model_args) in enumerate(jobs):
        n_aug_label = getattr(model_args, "n_aug", model_row.get("N_Aug") if isinstance(model_row, dict) else "?")
        model_label = _model_training_label(model_row, model_args, n_aug_label)
        stack_key = _stack_key(model_row, model_args)
        stack_index = stack_done.get(stack_key, 0) + 1
        stack_total = max(stack_totals.get(stack_key, 1), 1)
        _render_training_tracker(stack_key)
        head_progress.progress(0)
        head_tracker.empty()

        mismatch, mismatch_message, checkpoint_path = _checkpoint_task_mismatch(model_args)
        if mismatch:
            message = (
                f"Skipping classifier-head training for {model_label} "
                f"({idx + 1}/{len(jobs)}, model stack {stack_index}/{stack_total}): "
                f"{mismatch_message}"
            )
            status.warning(message)
            _log_training_update(message, "warning")
            summary_rows.append(
                _training_status_row(
                    model_row,
                    model_args,
                    int(n_aug_label),
                    "skipped_class_mismatch",
                    mismatch_message,
                    cache_path=checkpoint_path,
                )
            )
            _finish_job(stack_key, "skipped_class_mismatch", idx)
            continue

        if skip_existing:
            has_cache_entry, cache_path = _cache_has_n_aug_entry(model_args, int(n_aug_label))
            if has_cache_entry:
                message = (
                    f"Skipping existing classifier heads for {model_label} "
                    f"({idx + 1}/{len(jobs)}, model stack {stack_index}/{stack_total})"
                )
                status.info(message)
                _log_training_update(message)
                summary_rows.append(_cache_summary_row(model_row, model_args, int(n_aug_label), "skipped_existing"))
                _finish_job(stack_key, "skipped_existing", idx)
                continue

        message = (
            f"Encoding embeddings for {model_label} "
            f"({idx + 1}/{len(jobs)}, model/n_aug job {stack_index}/{stack_total})"
        )
        status.info(message)
        _log_training_update(message)
        job_state = "trained"

        def _head_progress_callback(info):
            head_index = int(info.get("head_index", 0) or 0)
            total_heads = max(int(info.get("total_heads", 1) or 1), 1)
            heads_left = max(int(info.get("heads_left", 0) or 0), 0)
            current_head = str(info.get("head") or "")
            message = (
                f"Training {current_head} for {model_label} "
                f"({head_index}/{total_heads} heads, {heads_left} left; "
                f"model/n_aug job {idx + 1}/{len(jobs)})"
            )
            status.info(message)
            _log_training_update(message)
            head_progress.progress(min(head_index / total_heads, 1.0))
            head_tracker.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Model ID": model_row.get("Model ID") if isinstance(model_row, dict) else getattr(model_args, "model_id", None),
                            "Model": (
                                model_row.get("Model Name")
                                or model_row.get("model_name")
                                or getattr(model_args, "model_name", None)
                                if isinstance(model_row, dict)
                                else getattr(model_args, "model_name", None)
                            ),
                            "N Aug": n_aug_label,
                            "Current Head": current_head,
                            "Head": head_index,
                            "Total Heads": total_heads,
                            "Heads Left": heads_left,
                        }
                    ]
                ),
                use_container_width=True,
            )


        try:
            result, cache_path = train_heads_for_args(
                model_args,
                k_values=k_values,
                prototype_strategies=prototype_strategies,
                prototype_components=proto_components,
                include_knn=include_knn,
                include_prototypes=include_prototypes,
                include_baselines=include_baselines,
                overwrite_current_n_aug=overwrite,
                progress_callback=_head_progress_callback,
            )
            summary_rows.append(_result_summary_row(model_row, result, cache_path))
            _log_training_update(
                f"Finished classifier heads for {model_label}: "
                f"{format_classifier_config(result.get('best_config'))} "
                f"(Valid MCC={result.get('best_valid_mcc', result.get('best_mcc'))})",
                "success",
            )

            # Keep sidebar/current inference in sync when training the current model.
            if train_target == "Current sidebar model":
                st.session_state["sidebar_classification_head_config"] = str(result.get("best_config"))
                st.session_state["optimized_k_value"] = str(result.get("best_config"))
                st.session_state["learned_classifier_label"] = format_classifier_config(result.get("best_config"))
                st.session_state["k_opt_best_mcc"] = result.get("best_mcc")
                st.session_state["k_opt_curve"] = result.get("knn", {}).get("mcc_per_k", [])
                st.session_state["k_opt_proto_results"] = result.get("prototypes", {})
                st.session_state["current_n_aug"] = int(getattr(model_args, "n_aug", 0) or 0)

        except Exception as e:
            if _is_class_count_mismatch_error(e):
                # Only recalculate and overwrite classifier heads, do not retrain backbone
                job_state = "recalculated_heads_class_mismatch"
                message = str(e)
                checkpoint_path = _extract_model_path_from_error(e)
                status.warning(
                    f"Class-count mismatch detected for {model_label} (model stack {stack_index}/{stack_total}). "
                    f"The embedding model is unchanged. Old classifier heads are invalid and will be replaced. "
                    f"Recalculating and overwriting classifier heads for the current task/classes.\n\n"
                    f"Details: {message}"
                )
                _log_training_update(
                    f"Class-count mismatch for {model_label}; recalculating heads. Details: {message}",
                    "warning",
                )
                try:
                    # Try head training again, forcing overwrite
                    result, cache_path = train_heads_for_args(
                        model_args,
                        k_values=k_values,
                        prototype_strategies=prototype_strategies,
                        prototype_components=proto_components,
                        include_knn=include_knn,
                        include_prototypes=include_prototypes,
                        include_baselines=include_baselines,
                        overwrite_current_n_aug=True,
                        progress_callback=_head_progress_callback,
                    )
                    summary_rows.append(_result_summary_row(model_row, result, cache_path))
                    _log_training_update(
                        f"Finished recalculated classifier heads for {model_label}: "
                        f"{format_classifier_config(result.get('best_config'))} "
                        f"(Valid MCC={result.get('best_valid_mcc', result.get('best_mcc'))})",
                        "success",
                    )
                    if train_target == "Current sidebar model":
                        st.session_state["sidebar_classification_head_config"] = str(result.get("best_config"))
                        st.session_state["optimized_k_value"] = str(result.get("best_config"))
                        st.session_state["learned_classifier_label"] = format_classifier_config(result.get("best_config"))
                        st.session_state["k_opt_best_mcc"] = result.get("best_mcc")
                        st.session_state["k_opt_curve"] = result.get("knn", {}).get("mcc_per_k", [])
                        st.session_state["k_opt_proto_results"] = result.get("prototypes", {})
                        st.session_state["current_n_aug"] = int(getattr(model_args, "n_aug", 0) or 0)
                except Exception as recalc_exc:
                    status.error(
                        f"Classifier head recalculation failed for {model_label}: {recalc_exc}\n\n"
                        f"Original error: {message}"
                    )
                    _log_training_update(
                        f"Classifier head recalculation failed for {model_label}: {recalc_exc}",
                        "error",
                    )
                    summary_rows.append(
                        _training_status_row(
                            model_row,
                            model_args,
                            int(n_aug_label),
                            "error_head_recalc_failed",
                            f"Head recalculation failed: {recalc_exc}",
                            cache_path=checkpoint_path,
                        )
                    )
            else:
                job_state = "error"
                _log_training_update(f"Classifier-head training failed for {model_label}: {e}", "error")
                summary_rows.append(
                    {
                        "Status": "error",
                        "Model ID": model_row.get("Model ID") if isinstance(model_row, dict) else None,
                        "Model": (
                            model_row.get("Model Name")
                            or model_row.get("model_name")
                            or getattr(model_args, "model_name", None)
                            if isinstance(model_row, dict)
                            else getattr(model_args, "model_name", None)
                        ),
                        "N Aug": getattr(model_args, "n_aug", None),
                        "Best Config": None,
                        "Best Head": None,
                        "Best Valid MCC": None,
                        "N train": None,
                        "N valid": None,
                        "Cache Path": f"ERROR: {e}",
                    }
                )
                with st.expander(f"Traceback for {model_label}"):
                    st.code(traceback.format_exc())

        _finish_job(stack_key, job_state, idx)

    _clear_existing_heads_cache()

    trained_head_rows = []
    trained_head_errors = []
    seen_models = set()
    target_n_aug_values = {int(v) for v in n_aug_values}
    for model_row, _model_args in jobs:
        model_key = (
            model_row.get("Model ID") if isinstance(model_row, dict) else None,
            model_row.get("Log Path") if isinstance(model_row, dict) else None,
        )
        if model_key in seen_models:
            continue
        seen_models.add(model_key)
        rows, err = _get_heads_for_model_args(_model_args, model_row)
        if err:
            trained_head_errors.append(
                {
                    "Model ID": model_row.get("Model ID") if isinstance(model_row, dict) else None,
                    "Model": model_row.get("Model Name") if isinstance(model_row, dict) else None,
                    "Error": err,
                }
            )
            continue
        for row in rows:
            try:
                row_n_aug = int(row.get("N Aug"))
            except Exception:
                continue
            if row_n_aug in target_n_aug_values:
                trained_head_rows.append(row)

    st.session_state["learned_last_training_payload"] = {
        "summary_rows": summary_rows,
        "trained_head_rows": trained_head_rows,
        "trained_head_errors": trained_head_errors,
    }
    st.rerun()


# -------------------------------------------------
# Public page render
# -------------------------------------------------

def render(ctx):
    st.header("🧠 Learned Embedding Classification")

    st.caption(
        "Existing learned-embedding classifier heads from saved results: KNN, baselines, "
        "and prototype heads such as mean, kmeans, and gmm."
    )

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        top_n = st.number_input(
            "Show top N models",
            min_value=1,
            max_value=200,
            value=20,
            step=1,
            key="learned_existing_top_n",
        )

    with c2:
        show_all_models = st.checkbox(
            "Show ALL available models",
            value=False,
            key="learned_show_all_models",
            help="Show all models with trained heads, ignoring the top N filter."
        )

    with c3:
        reload_clicked = st.button(
            "Reload existing classifier heads",
            key="learned_reload_existing_heads",
        )
    
    # Add clear classifier heads button (admin only)
    if ctx.is_admin:
        with st.expander("⚠️ Danger Zone", expanded=False):
            st.warning("Clearing classifier heads will delete all optimization cache files. This action cannot be undone.")
            clear_clicked = st.button(
                "🗑️ Clear ALL classifier heads",
                key="learned_clear_all_heads",
                type="primary"
            )
    # Always update the production model message after any rerun
    if "production_model" in st.session_state:
        prod = st.session_state["production_model"]
        if prod is not None:
            prod_head = (
                prod.get("Head")
                or prod.get("head_name")
                or prod.get("learned_classifier_label")
                or format_classifier_config(
                    prod.get("Head Config")
                    or prod.get("head_config")
                    or prod.get("classification_head_config")
                    or prod.get("best_classifier_config")
                )
            )
            if not prod_head or str(prod_head) in {"None", "nan", "—"}:
                prod_head = "head not stored yet"
            st.info(f"🎯 **Current Production Model:** {prod.get('Model Name')} (#{prod.get('model_number', '?')}) - {prod_head}")

    active_task = st.session_state.get("production_task") or getattr(ctx.args, "task", None)
    load_top_n = 99999 if show_all_models else int(top_n)
    cache_key = f"learned_existing_heads_df_v7_{active_task}_top_{load_top_n}"
    error_key = f"learned_existing_errors_v7_{active_task}_top_{load_top_n}"

    if reload_clicked:
        _clear_existing_heads_cache()
        try:
            from otitenet.app.pages.leaderboard import load_best_models_table, _update_model_number_map
            leaderboard_df = load_best_models_table(ctx.cursor, task=active_task)
            _update_model_number_map(leaderboard_df)
        except Exception as e:
            st.warning(f"Could not refresh leaderboard table after reload: {e}")

        try:
            spinner_text = (
                "Reloading all available classifier heads..."
                if show_all_models
                else "Reloading existing classifier heads..."
            )
            with st.spinner(spinner_text):
                heads_df, errors_df = _load_existing_heads(ctx, top_n=load_top_n)
            st.session_state[cache_key] = heads_df
            st.session_state[error_key] = errors_df
            st.success("Reloaded classifier heads with latest saved metrics.")
        except Exception as e:
            st.error(f"Reload failed: {e}")
    
    # Handle clear classifier heads button (admin only)
    if ctx.is_admin and clear_clicked:
        import glob
        import shutil
        from otitenet.app.utils import get_optimization_cache_file_path, _optimization_cache_file_paths
        
        st.warning("Deleting all optimization cache files...")
        deleted_count = 0
        errors = []
        
        try:
            # Get all optimization cache paths for the current task
            cache_base = f"logs/best_models/{active_task}"
            if os.path.exists(cache_base):
                # Find all knn_optimization_cache.pkl files
                for cache_file in glob.glob(os.path.join(cache_base, "**", "knn_optimization_cache.pkl"), recursive=True):
                    try:
                        os.remove(cache_file)
                        deleted_count += 1
                    except Exception as e:
                        errors.append(f"Failed to delete {cache_file}: {e}")
                
                # Also look for legacy cache files in dist_* directories
                for legacy_file in glob.glob(os.path.join(cache_base, "**", "dist_*", "knn*", "knn_optimization_cache.pkl"), recursive=True):
                    try:
                        os.remove(legacy_file)
                        deleted_count += 1
                    except Exception as e:
                        errors.append(f"Failed to delete {legacy_file}: {e}")
            
            # Clear session state cache
            _clear_existing_heads_cache()
            
            if deleted_count > 0:
                st.success(f"✅ Deleted {deleted_count} optimization cache file(s).")
            else:
                st.info("No optimization cache files found to delete.")
            
            if errors:
                st.error(f"Errors occurred:\n" + "\n".join(errors))
            
            # Force rerun to refresh the UI
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to clear classifier heads: {e}")

    if cache_key not in st.session_state:
        spinner_text = (
            "Loading all available classifier heads..."
            if show_all_models
            else "Loading existing classifier heads..."
        )
        with st.spinner(spinner_text):
            heads_df, errors_df = _load_existing_heads(ctx, top_n=load_top_n)
            st.session_state[cache_key] = heads_df
            st.session_state[error_key] = errors_df

    heads_df = st.session_state.get(cache_key, pd.DataFrame())
    errors_df = st.session_state.get(error_key, pd.DataFrame())
    heads_df = _merge_recent_training_heads(heads_df)

    _render_metric_row(heads_df)

    filtered_df = _render_filters(heads_df)

    _render_charts(filtered_df)

    st.divider()

    _render_head_tables(filtered_df)

    if errors_df is not None and len(errors_df) > 0:
        with st.expander("Models where classifier heads could not be loaded"):
            st.dataframe(errors_df, use_container_width=True)

    st.divider()

    _render_training_controls(ctx)
