"""
Display‑metrics stub for the Streamlit UI.

The original project provided detailed metric visualizations. For the
current build we only need the function signatures so the app can run.
Each function returns placeholder data or a simple Streamlit output.
"""

import streamlit as st
from typing import Any, Dict, List
import pandas as pd


def compute_classification_metrics(args, predictions):
    """
    Placeholder for computing classification metrics.

    Returns a dict with dummy values.
    """
    return {
        "accuracy": 0.0,
        "mcc": 0.0,
        "auc": 0.0,
    }


def render_metrics_table(metrics: Dict[str, Any]) -> None:
    """Render a simple metrics table in the sidebar."""
    st.subheader("Metrics")
    for name, value in metrics.items():
        st.write(f"**{name}**: {value}")


def plot_roc_curve(metrics: Dict[str, Any]) -> None:
    """Placeholder ROC curve plot."""
    st.subheader("ROC Curve")
    st.line_chart([metrics.get("auc", 0)])


def plot_confusion_matrix(metrics: Dict[str, Any]) -> None:
    """Placeholder confusion matrix plot."""
    st.subheader("Confusion Matrix")
    st.write("Matrix not available in stub.")


def display_all_metrics(args, predictions):
    """
    Convenience wrapper used by inference_results page.
    Computes metrics and renders them using the stubs above.
    """
    metrics = compute_classification_metrics(args, predictions)
    render_metrics_table(metrics)
    plot_roc_curve(metrics)
    plot_confusion_matrix(metrics)

# --- Additional stubs for expected symbols ---

def _arrow_safe_dataframe(df):
    """Return a DataFrame compatible with Streamlit's Arrow display."""
    if df is None:
        return df

    def _safe_display_value(value):
        if value is None:
            return ""
        if isinstance(value, (bytes, bytearray)):
            return f"<bytes {len(value)}>"
        if isinstance(value, (tuple, list, dict, set)):
            return str(value)
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value)

    out = df.copy()
    for col in out.columns:
        series = out[col]

        if series.dtype != "object":
            continue

        out[col] = series.map(_safe_display_value).astype("string")

    return out

def _best_head_config_for_args_global(args):
    """Return the best cached classification head config for given args."""
    entry = _best_head_entry_for_args_global(args)
    config = entry.get("config") if isinstance(entry, dict) else None
    if config is not None:
        return str(config)

    try:
        from otitenet.app.utils import resolve_best_classifier_config

        return str(resolve_best_classifier_config(args, use_optimized=True))
    except Exception:
        return str(getattr(args, "n_neighbors", 1))

def _best_head_entry_for_args_global(args):
    """Return the highest-scoring cached classification head entry."""
    try:
        heads = _classification_head_options_for_args_global(args)
        if heads:
            return heads[0]
    except Exception:
        pass
    try:
        from otitenet.app.utils import resolve_best_classifier_config

        try:
            config = str(resolve_best_classifier_config(args, use_optimized=True))
        except Exception:
            config = str(getattr(args, "n_neighbors", 1))
        score = (
            getattr(args, "valid_mcc", None)
            if getattr(args, "valid_mcc", None) is not None
            else getattr(args, "mcc", None)
        )
        return {
            "config": config,
            "label": _head_config_label_global(config),
            "mcc": score,
            "valid_mcc": score,
            "details": "fallback from selected model",
        }
    except Exception:
        pass
    return {}

def _classification_head_options_for_args_global(args):
    """Return cached classification-head options for the args."""
    try:
        from otitenet.app.utils import enumerate_classification_heads

        return list(enumerate_classification_heads(args) or [])
    except Exception:
        return []

def _head_config_label_global(config):
    """Generate a human-readable label for a head config."""
    if config is None:
        return ""
    try:
        from otitenet.app.utils import format_classifier_config

        return format_classifier_config(str(config))
    except Exception:
        return str(config)

def _set_classifier_head_on_args_global(args, config):
    """Mutate args to set the classifier head config."""
    try:
        config = str(config)
        setattr(args, "classifier_head_config", config)
        setattr(args, "best_classifier_config", config)
        setattr(args, "classification_head_config", config)
        setattr(args, "learned_classifier_label", _head_config_label_global(config))

        from otitenet.app.utils import parse_classifier_config

        head_meta = parse_classifier_config(config)
        setattr(args, "classification_head_family", head_meta.get("family"))
        if head_meta.get("family") == "knn" and head_meta.get("k") is not None:
            setattr(args, "n_neighbors", int(head_meta["k"]))
            setattr(args, "siamese_inference", "knn")
        elif head_meta.get("family") == "prototype":
            setattr(args, "prototypes_to_use", "class")
            if head_meta.get("strategy") is not None:
                setattr(args, "prototype_strategy", str(head_meta["strategy"]))
            if head_meta.get("components") is not None:
                setattr(args, "prototype_components", int(head_meta["components"]))
        elif head_meta.get("family") == "baseline":
            name = str(head_meta.get("name", ""))
            if name in {"linear_svc", "linearsvc"}:
                setattr(args, "siamese_inference", "linearsvc")
            elif name in {"logreg", "logistic_regression"}:
                setattr(args, "siamese_inference", "logisticregression")
            else:
                setattr(args, "siamese_inference", name)
    except Exception:
        pass
