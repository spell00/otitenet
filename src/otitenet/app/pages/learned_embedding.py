
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
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from otitenet.app.model_loading import load_model_and_prototypes
from otitenet.app.utils import (
    enumerate_classification_heads,
    format_classifier_config,
    get_optimization_cache_file_path,
)
from otitenet.app.services.embedding_optimization_service import (
    args_from_model_row,
    fetch_best_model_rows,
)
from otitenet.data.data_getters import get_images_loaders
from otitenet.logging.metrics import MCC
from otitenet.train.train_triplet_new import TrainAE
from otitenet.utils.encoding_utils import (
    compute_prototypes_by_strategy,
    flatten_prototype_dict,
)
from otitenet.utils.utils import get_empty_traces


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
    model_mcc = model_row.get("MCC") or model_row.get("mcc")

    n_aug = (
        model_row.get("N_Aug")
        or model_row.get("n_aug")
        or model_row.get("N Aug")
        or getattr(model_args, "n_aug", None)
    )

    if not isinstance(entry, dict):
        cfg = str(entry)
        family = _infer_head_family(cfg)
        return {
            "Model ID": model_id,
            "Model": model_name,
            "N Aug": n_aug,
            "Family": family,
            "Classifier": _family_display_name(family),
            "Head": cfg,
            "Config": cfg,
            "MCC": None,
            "Accuracy": None,
            "AUC": None,
            "Model MCC": _safe_float(model_mcc),
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

    n_aug = entry.get("n_aug", entry.get("N_Aug", entry.get("n_augmentations", n_aug)))

    return {
        "Model ID": model_id,
        "Model": model_name,
        "N Aug": n_aug,
        "Family": family,
        "Classifier": _family_display_name(family),
        "Head": label,
        "Config": config,
        "MCC": entry.get("mcc", entry.get("MCC", entry.get("valid_mcc", entry.get("test_mcc")))),
        "Accuracy": entry.get("accuracy", entry.get("acc", entry.get("Accuracy"))),
        "AUC": entry.get("auc", entry.get("AUC", entry.get("test_auc", entry.get("valid_auc")))),
        "Model MCC": _safe_float(model_mcc),
        "Details": entry.get("details", entry.get("source", entry.get("path", ""))),
    }


def _get_heads_for_model(base_args, model_row):
    model_args = args_from_model_row(base_args, model_row)

    try:
        heads = list(enumerate_classification_heads(model_args) or [])
    except Exception as e:
        return [], str(e)

    rows = [_normalize_head_entry(h, model_row, model_args=model_args) for h in heads]

    if not rows:
        fallback_k = getattr(model_args, "n_neighbors", None)
        if fallback_k is not None:
            cfg = str(fallback_k)
            rows.append(
                {
                    "Model ID": model_row.get("Model ID") or model_row.get("id"),
                    "Model": model_row.get("Model Name") or model_row.get("model_name"),
                    "N Aug": getattr(model_args, "n_aug", model_row.get("N_Aug", model_row.get("n_aug"))),
                    "Family": "knn",
                    "Classifier": "KNN",
                    "Head": f"KNN k={cfg}",
                    "Config": cfg,
                    "MCC": model_row.get("MCC") or model_row.get("mcc"),
                    "Accuracy": model_row.get("Accuracy") or model_row.get("accuracy"),
                    "AUC": model_row.get("Test_AUC") or model_row.get("Valid_AUC"),
                    "Model MCC": _safe_float(model_row.get("MCC") or model_row.get("mcc")),
                    "Details": "fallback from N_Neighbors",
                }
            )

    return rows, None


def _load_top_models(ctx, top_n):
    table = st.session_state.get("best_models_table")

    if table is not None:
        try:
            df = pd.DataFrame(table)
            if len(df) > 0:
                return df.head(int(top_n))
        except Exception:
            pass

    return fetch_best_model_rows(ctx.cursor, limit=int(top_n))


def _load_existing_heads(ctx, top_n):
    models_df = _load_top_models(ctx, top_n=int(top_n))

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

    for col in ["MCC", "Accuracy", "AUC", "Model MCC"]:
        if col in heads_df.columns:
            heads_df[col] = pd.to_numeric(heads_df[col], errors="coerce")

    return heads_df, errors_df


def _best_by_model(heads_df):
    if heads_df is None or len(heads_df) == 0:
        return pd.DataFrame()

    df = heads_df.copy()

    if "MCC" not in df.columns:
        return pd.DataFrame()

    df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce")

    return (
        df.sort_values("MCC", ascending=False, na_position="last")
        .groupby("Model ID", as_index=False)
        .first()
    )


def _render_metric_row(heads_df):
    if heads_df is None or len(heads_df) == 0:
        return

    best = _best_by_model(heads_df)

    n_models = heads_df["Model ID"].nunique() if "Model ID" in heads_df.columns else 0
    n_heads = len(heads_df)

    best_mcc = None
    if "MCC" in heads_df.columns:
        best_mcc = pd.to_numeric(heads_df["MCC"], errors="coerce").max()

    best_family = "—"
    if len(best) > 0 and "Classifier" in best.columns:
        best_family = str(best.sort_values("MCC", ascending=False).iloc[0]["Classifier"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", n_models)
    c2.metric("Classifier heads", n_heads)
    c3.metric("Best MCC", "—" if pd.isna(best_mcc) else f"{best_mcc:.4f}")
    c4.metric("Top head", best_family)


def _render_charts(heads_df):
    if heads_df is None or len(heads_df) == 0:
        return

    df = heads_df.copy()

    if "MCC" not in df.columns:
        st.info("No MCC column available for plots.")
        return

    df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce")
    df = df.dropna(subset=["MCC"])

    if len(df) == 0:
        st.info("No numeric MCC values available for plots.")
        return

    st.subheader("Visual summaries")

    best = _best_by_model(df)

    if len(best) > 0:
        st.markdown("#### Best classifier head per model")
        plot_df = best[["Model ID", "MCC"]].copy()
        plot_df["Model ID"] = plot_df["Model ID"].astype(str)
        st.bar_chart(plot_df.set_index("Model ID"))

    family_df = (
        df.groupby("Classifier", as_index=False)
        .agg(
            Best_MCC=("MCC", "max"),
            Mean_MCC=("MCC", "mean"),
            N=("MCC", "count"),
        )
        .sort_values("Best_MCC", ascending=False)
    )

    if len(family_df) > 0:
        st.markdown("#### Best MCC by classifier family")
        st.bar_chart(family_df.set_index("Classifier")[["Best_MCC"]])

        st.markdown("#### Mean MCC by classifier family")
        st.bar_chart(family_df.set_index("Classifier")[["Mean_MCC"]])

    if "Model ID" in df.columns and "Classifier" in df.columns:
        pivot = df.pivot_table(
            index="Model ID",
            columns="Classifier",
            values="MCC",
            aggfunc="max",
        )

        if len(pivot) > 0:
            st.markdown("#### Classifier comparison across models")
            st.dataframe(pivot, use_container_width=True)
            st.line_chart(pivot)


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
        "N Aug",
        "Classifier",
        "Head",
        "MCC",
        "Accuracy",
        "AUC",
        "Config",
        "Details",
    ]
    display_cols = [c for c in display_cols if c in heads_df.columns]

    display_df = heads_df.copy()

    if "MCC" in display_df.columns:
        display_df = display_df.sort_values(
            by=["Model ID", "MCC"],
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
            "N Aug",
            "Classifier",
            "Head",
            "MCC",
            "Accuracy",
            "AUC",
            "Config",
        ]
        best_cols = [c for c in best_cols if c in best.columns]

        st.dataframe(best[best_cols], use_container_width=True)


def _render_filters(heads_df):
    if heads_df is None or len(heads_df) == 0:
        return heads_df

    df = heads_df.copy()

    st.subheader("Filters")

    c1, c2, c3 = st.columns(3)

    with c1:
        families = sorted([str(x) for x in df["Classifier"].dropna().unique()]) if "Classifier" in df.columns else []
        selected_families = st.multiselect(
            "Classifier families",
            families,
            default=families,
            key="learned_filter_families",
        )

    with c2:
        min_mcc = st.slider(
            "Minimum MCC",
            min_value=-1.0,
            max_value=1.0,
            value=-1.0,
            step=0.01,
            key="learned_filter_min_mcc",
        )

    with c3:
        search = st.text_input(
            "Search head/config",
            value="",
            key="learned_filter_search",
        )

    if selected_families and "Classifier" in df.columns:
        df = df[df["Classifier"].isin(selected_families)]

    if "MCC" in df.columns:
        df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce")
        df = df[df["MCC"].isna() | (df["MCC"] >= min_mcc)]

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
    """Patch missing args fields expected by TrainAE/get_images_loaders."""
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
    }
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
    )

    with torch.no_grad():
        for split in ["train", "valid", "test"]:
            if split in loaders:
                try:
                    _, lists, _ = train.loop(split, None, 0, loaders[split], lists, traces)
                except Exception:
                    if split in {"train", "valid"}:
                        raise

    encoded = {}

    for split in ["train", "valid", "test"]:
        if lists.get(split, {}).get("encoded_values"):
            encoded[split] = {
                "X": np.concatenate(lists[split]["encoded_values"]),
                "y": np.concatenate(lists[split]["cats"]),
            }

    if "train" not in encoded or "valid" not in encoded:
        raise RuntimeError("Could not encode train/valid embeddings.")

    return encoded


def _compute_knn_heads(X_train, y_train, X_valid, y_valid, k_values):
    mcc_per_k = []
    best_k = None
    best_mcc = -1.0

    for k in k_values:
        k_eff = max(1, min(int(k), len(X_train)))
        try:
            clf = KNeighborsClassifier(n_neighbors=k_eff, metric="minkowski")
            clf.fit(X_train, y_train)
            pred = clf.predict(X_valid)
            mcc = _evaluate_mcc(y_valid, pred)

            mcc_per_k.append({"k": int(k_eff), "valid_mcc": float(mcc), "mcc": float(mcc)})

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


def _compute_prototype_heads(X_train, y_train, X_valid, y_valid, strategies, components):
    out = {}

    for strategy in strategies:
        best_mcc = -1.0
        best_n_components = None
        best_n_prototypes = 0
        per_components = []

        for n_comp in components:
            n_comp = int(n_comp)
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

                dists = np.linalg.norm(X_valid[:, None, :] - proto_vecs[None, :, :], axis=2)
                preds = proto_labels[np.argmin(dists, axis=1)]
                mcc = _evaluate_mcc(y_valid, preds)

                per_components.append(
                    {
                        "n_components": n_comp,
                        "mcc": float(mcc),
                        "n_prototypes": int(len(proto_vecs)),
                    }
                )

                if not pd.isna(mcc) and float(mcc) > best_mcc:
                    best_mcc = float(mcc)
                    best_n_components = n_comp
                    best_n_prototypes = int(len(proto_vecs))

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

    return out


def _compute_baseline_heads(X_train, y_train, X_valid, y_valid):
    out = {}

    for name, clf in _baseline_models().items():
        try:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_valid)
            mcc = _evaluate_mcc(y_valid, pred)
            out[name] = {"mcc": float(mcc)}
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


def train_heads_for_args(
    _args,
    k_values,
    prototype_strategies,
    prototype_components,
    include_knn=True,
    include_prototypes=True,
    include_baselines=True,
    overwrite_current_n_aug=True,
):
    """Compute learned-embedding classifier heads and write the optimization cache."""
    encoded = _encode_splits_for_args(_args)

    X_train = encoded["train"]["X"]
    y_train = encoded["train"]["y"]
    X_valid = encoded["valid"]["X"]
    y_valid = encoded["valid"]["y"]

    result = {
        "knn": {},
        "prototypes": {},
        "baselines": {},
        "best_k": None,
        "best_config": None,
        "best_mcc": -1.0,
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
    }

    if include_knn:
        result["knn"] = _compute_knn_heads(X_train, y_train, X_valid, y_valid, k_values)

    if include_prototypes:
        result["prototypes"] = _compute_prototype_heads(
            X_train,
            y_train,
            X_valid,
            y_valid,
            strategies=prototype_strategies,
            components=prototype_components,
        )

    if include_baselines:
        result["baselines"] = _compute_baseline_heads(X_train, y_train, X_valid, y_valid)

    best_config, best_mcc = _choose_best_config(result)
    result["best_k"] = best_config
    result["best_config"] = best_config
    result["best_mcc"] = float(best_mcc) if best_config is not None else -1.0

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
        if key.startswith("learned_existing_heads_df_top_") or key.startswith("learned_existing_errors_top_"):
            st.session_state.pop(key, None)


def _result_summary_row(model_row, result, cache_path):
    model_id = None
    model_name = None

    if isinstance(model_row, dict):
        model_id = model_row.get("Model ID") or model_row.get("id") or model_row.get("model_id")
        model_name = model_row.get("Model Name") or model_row.get("model_name") or model_row.get("model")

    return {
        "Model ID": model_id,
        "Model": model_name,
        "N Aug": model_row.get("N_Aug", model_row.get("n_aug")) if isinstance(model_row, dict) else None,
        "Best Config": result.get("best_config"),
        "Best Head": format_classifier_config(result.get("best_config")),
        "Best MCC": result.get("best_mcc"),
        "N train": result.get("n_train"),
        "N valid": result.get("n_valid"),
        "Cache Path": cache_path,
    }



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
        return

    k_values = list(range(1, int(k_max) + 1))
    proto_components = list(range(1, int(proto_max) + 1))
    prototype_strategies = prototype_strategies or ["mean"]
    n_aug_values = _parse_int_list(n_aug_text, default=[int(getattr(ctx.args, "n_aug", 0) or 0)])

    summary_rows = []

    if train_target == "Current sidebar model":
        jobs = []
        for n_aug in n_aug_values:
            model_args = ctx.args
            model_args.n_aug = int(n_aug)
            jobs.append(({"N_Aug": int(n_aug), "Model Name": "Current sidebar model"}, model_args))
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
                            "Model ID": row_dict.get("Model ID") or row_dict.get("id"),
                            "Model": row_dict.get("Model Name") or row_dict.get("model_name"),
                            "N Aug": int(n_aug),
                            "Best Config": None,
                            "Best Head": None,
                            "Best MCC": None,
                            "N train": None,
                            "N valid": None,
                            "Cache Path": f"ERROR while building args: {e}",
                        }
                    )

    progress = st.progress(0)
    status = st.empty()

    for idx, (model_row, model_args) in enumerate(jobs):
        n_aug_label = getattr(model_args, "n_aug", model_row.get("N_Aug") if isinstance(model_row, dict) else "?")
        model_label = f"current sidebar model / n_aug={n_aug_label}"
        if model_row and model_row.get("Model ID") is not None:
            model_label = f"model #{model_row.get('#', model_row.get('Model ID', '?'))} / ID {model_row.get('Model ID', '?')} / n_aug={n_aug_label}"

        status.info(f"Training classifier heads for {model_label} ({idx + 1}/{len(jobs)})")

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
            )
            summary_rows.append(_result_summary_row(model_row, result, cache_path))

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
            summary_rows.append(
                {
                    "Model ID": model_row.get("Model ID") if isinstance(model_row, dict) else None,
                    "Model": model_row.get("Model Name") if isinstance(model_row, dict) else "Current sidebar model",
                    "N Aug": getattr(model_args, "n_aug", None),
                    "Best Config": None,
                    "Best Head": None,
                    "Best MCC": None,
                    "N train": None,
                    "N valid": None,
                    "Cache Path": f"ERROR: {e}",
                }
            )
            with st.expander(f"Traceback for {model_label}"):
                st.code(traceback.format_exc())

        progress.progress((idx + 1) / max(len(jobs), 1))

    _clear_existing_heads_cache()

    st.success("Finished classifier-head training.")
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

    st.info("Click 'Reload existing classifier heads' above to refresh the displayed tables, or the page will refresh on the next rerun.")


# -------------------------------------------------
# Public page render
# -------------------------------------------------

def render(ctx):
    st.header("🧠 Learned Embedding Classification")

    st.caption(
        "Existing learned-embedding classifier heads from saved results: KNN, baselines, "
        "and prototype heads such as mean, kmeans, and gmm."
    )

    c1, c2 = st.columns([1, 1])

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
        reload_clicked = st.button(
            "Reload existing classifier heads",
            key="learned_reload_existing_heads",
        )

    cache_key = f"learned_existing_heads_df_top_{int(top_n)}"
    error_key = f"learned_existing_errors_top_{int(top_n)}"

    if reload_clicked:
        st.session_state.pop(cache_key, None)
        st.session_state.pop(error_key, None)

    if cache_key not in st.session_state:
        with st.spinner("Loading existing classifier heads..."):
            heads_df, errors_df = _load_existing_heads(ctx, top_n=int(top_n))
            st.session_state[cache_key] = heads_df
            st.session_state[error_key] = errors_df

    heads_df = st.session_state.get(cache_key, pd.DataFrame())
    errors_df = st.session_state.get(error_key, pd.DataFrame())

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