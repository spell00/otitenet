import argparse
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from otitenet.app.display_metrics import (
    _arrow_safe_dataframe,
    _best_head_config_for_args_global,
    _head_config_label_global,
)
from otitenet.app.model_loading import load_model_and_prototypes
from otitenet.app.utils import (
    _make_model_selection_key,
    _unique_preserve_order,
    ensure_int,
    extract_params_from_log_path,
    get_calibration_metrics,
    get_split_mcc_metrics,
)
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

    parsed = extract_params_from_log_path(row.get("Log Path") or row.get("log_path") or "")
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
        if head_config is None or str(head_config).strip() in {"", "nan", "None"}:
            head_config = _best_head_config_for_args_global(local_args)

        classif_loss = str(getattr(local_args, "classif_loss", "")).lower()
        if classif_loss in {"ce", "cross_entropy", "cross-entropy"} and str(head_config) in {"0", "0.0", "knn0", "None", ""}:
            return "CE classifier head"

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
                   accuracy, mcc, normalize, n_neighbors, log_path, model_rank,
                   prototype_strategy, prototype_components
            FROM best_models_registry
            WHERE model_rank IS NOT NULL
            ORDER BY model_rank ASC
            """
        )
        return cursor.fetchall() or [], True
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
        return cursor.fetchall() or [], False


def _models_dataframe_from_rows(rows: List[tuple], use_db_rank: bool) -> pd.DataFrame:
    if use_db_rank:
        columns = [
            "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos",
            "NNeg", "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration",
            "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", "#",
            "Proto_Strat", "Proto_Comp",
        ]
    else:
        columns = [
            "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos",
            "NNeg", "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration",
            "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path",
            "Proto_Strat", "Proto_Comp",
        ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        return df

    group_cols = [
        "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
        "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration",
        "Normalize", "N_Neighbors",
    ]

    for col in group_cols:
        if col not in df.columns:
            df[col] = ""

    dedupe_frame = df[group_cols].copy().fillna("").astype(str)
    df["_dedupe_key"] = dedupe_frame.apply(lambda r: "|".join(r.values.tolist()), axis=1)

    if "MCC" in df.columns:
        df["_mcc_numeric"] = pd.to_numeric(df["MCC"], errors="coerce").fillna(float("-inf"))
        df = df.sort_values("_mcc_numeric", ascending=False).drop(columns=["_mcc_numeric"])

    df = df.drop_duplicates(subset=["_dedupe_key"], keep="first")
    df = df.drop(columns=["_dedupe_key"], errors="ignore")

    if "Log Path" in df.columns:
        df = df.dropna(subset=["Log Path"])
        df = df[df["Log Path"].astype(str) != ""]

    return df.reset_index(drop=True)


def _attach_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    metric_cols = [
        "Train MCC", "Valid MCC", "Test MCC",
        "Train AUC", "Valid AUC", "Test AUC",
        "ECE", "Brier",
    ]

    for col in metric_cols:
        if col not in df.columns:
            df[col] = np.nan

    for idx, row in df.iterrows():
        log_path = row.get("Log Path")

        split_metrics = get_split_mcc_metrics(log_path)
        if split_metrics:
            df.at[idx, "Train MCC"] = split_metrics.get("train_mcc", np.nan)
            df.at[idx, "Valid MCC"] = split_metrics.get("valid_mcc", np.nan)
            df.at[idx, "Test MCC"] = split_metrics.get("test_mcc", np.nan)
            df.at[idx, "Train AUC"] = split_metrics.get("train_auc", np.nan)
            df.at[idx, "Valid AUC"] = split_metrics.get("valid_auc", np.nan)
            df.at[idx, "Test AUC"] = split_metrics.get("test_auc", np.nan)

        calibration_metrics = get_calibration_metrics(log_path)
        if calibration_metrics and not calibration_metrics.get("error"):
            df.at[idx, "ECE"] = calibration_metrics.get("ece", np.nan)
            df.at[idx, "Brier"] = calibration_metrics.get("brier", np.nan)

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

    ordered_cols = []
    if "#" in df.columns:
        ordered_cols.append("#")

    preferred_non_metric = [
        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
        "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize",
        "N_Neighbors", "Proto_Strat", "Proto_Comp", "Log Path",
    ]

    ordered_cols += [c for c in preferred_non_metric if c in df.columns and c not in ordered_cols]
    ordered_cols += [c for c in metric_cols if c in df.columns and c not in ordered_cols]
    ordered_cols += [c for c in df.columns if c not in ordered_cols]

    return df[ordered_cols]


def load_best_models_table(cursor) -> pd.DataFrame:
    rows, use_db_rank = _query_best_models(cursor)
    df = _models_dataframe_from_rows(rows, use_db_rank)

    if df.empty:
        return df

    df = _attach_metrics(df)
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
    )

    with torch.no_grad():
        _, lists, _ = train.loop("train", None, 0, loaders["train"], lists, traces)
        if "valid" in loaders:
            _, lists, _ = train.loop("valid", None, 0, loaders["valid"], lists, traces)
        if "test" in loaders:
            try:
                _, lists, _ = train.loop("test", None, 0, loaders["test"], lists, traces)
            except Exception:
                pass

    encs = []
    cats = []
    batches = []

    for grp in ["train", "valid", "test"]:
        if lists.get(grp, {}).get("encoded_values"):
            encs.append(np.concatenate(lists[grp]["encoded_values"]))
            cats.append(np.concatenate(lists[grp]["cats"]))

            try:
                batches.append(np.concatenate(lists[grp]["domains"]))
            except Exception:
                batches.append(np.array([grp] * len(np.concatenate(lists[grp]["cats"]))))

    if not encs:
        raise RuntimeError("No embeddings available to plot.")

    all_encs = np.concatenate(encs)
    all_cats = np.concatenate(cats)

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

    scatter = None

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

        scatter = ax.scatter(
            encs_pca[:, 0],
            encs_pca[:, 1],
            c=all_cats,
            cmap="tab20",
            alpha=0.5,
            s=20,
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

    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=axes[-1], pad=0.02)
        cbar.set_label("Class ID")

    fig.suptitle("PCA with Prototypes (train/valid/test)", fontsize=14, y=1.02)
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
    st.write("**Top Models (best per parameter combination):**")
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

    cols_now = list(display_df.columns)
    if "Head" in cols_now:
        cols_now.remove("Head")
        insert_at = cols_now.index("Normalize") + 1 if "Normalize" in cols_now else min(12, len(cols_now))
        cols_now.insert(insert_at, "Head")
        display_df = display_df[cols_now]

    st.dataframe(_arrow_safe_dataframe(display_df), use_container_width=True)


def _render_calibration_vs_performance(models_df: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("📈 Calibration vs Performance")

    plot_df = models_df.dropna(subset=["MCC", "ECE", "Brier"]).copy()

    if plot_df.empty:
        st.info("Calibration metrics not available for plotting. Run models to compute ECE and Brier scores.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**MCC vs ECE**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(plot_df["MCC"], plot_df["ECE"], alpha=0.6, s=50)
        ax.set_xlabel("MCC (higher is better)", fontsize=10)
        ax.set_ylabel("ECE (lower is better)", fontsize=10)
        ax.set_title("Expected Calibration Error vs MCC", fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("**MCC vs Brier Score**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(plot_df["MCC"], plot_df["Brier"], alpha=0.6, s=50)
        ax.set_xlabel("MCC (higher is better)", fontsize=10)
        ax.set_ylabel("Brier Score (lower is better)", fontsize=10)
        ax.set_title("Brier Score vs MCC", fontsize=11)
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


def _render_selected_model_calibration(row_dict: Dict[str, Any]) -> None:
    log_path = row_dict.get("Log Path")

    if not log_path:
        return

    st.subheader(f"📈 Calibration Curve (Model #{row_dict.get('#', '?')})")

    metrics = get_calibration_metrics(log_path)

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
    row_dict.update(extract_params_from_log_path(row_dict.get("Log Path")))

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


def _render_model_selector(models_df: pd.DataFrame, page_key: str) -> None:
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
        _render_selected_model_calibration(row_dict)

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
        cursor.execute(
            """
            SELECT model_name, task, nsize, fgsm, normalize, n_calibration,
                   classif_loss, dloss, prototypes, npos, nneg, n_neighbors,
                   num_samples_analyzed, last_used
            FROM model_usage_summary
            ORDER BY last_used DESC
            """
        )

        usage_rows = cursor.fetchall() or []

        if not usage_rows:
            st.info("No models have been used for analysis yet.")
            return

        usage_columns = [
            "Model", "Task", "Size", "FGSM", "Normalize", "N_Cal",
            "Loss", "DLoss", "Prototypes", "NPos", "NNeg",
            "N_Neighbors", "Samples", "Last Used",
        ]

        usage_df = pd.DataFrame(usage_rows, columns=usage_columns)

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
    include_embedding_tools: bool = True,
    include_usage_summary: bool = True,
):
    """
    Render the leaderboard / learned-embedding page.

    page_key is important: it namespaces Streamlit widget keys so this function
    can be called from multiple tabs without DuplicateWidgetID errors.
    """
    cursor = ctx.cursor
    args = ctx.args

    st.header(title)

    models_df = pd.DataFrame()

    if include_model_table or include_calibration or include_model_selector:
        try:
            models_df = load_best_models_table(cursor)

            if models_df.empty:
                st.warning("No models found in leaderboard.")
            else:
                _update_model_number_map(models_df)

                if include_model_table:
                    _render_top_models_table(models_df, args)

                if include_calibration:
                    _render_calibration_vs_performance(models_df)

                if include_model_selector:
                    _render_model_selector(models_df, page_key)

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
