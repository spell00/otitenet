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
    """Return the best classification head config for given args.
    Stub returns None (no specific config).
    """
    return None

def _best_head_entry_for_args_global(args):
    """Return a dict representing the best head entry.
    Stub returns an empty dict.
    """
    return {}

def _classification_head_options_for_args_global(args):
    """Return a list of possible head options for the args.
    Stub returns an empty list – the UI will handle the empty case.
    """
    return []

def _head_config_label_global(config):
    """Generate a human‑readable label for a head config.
    Stub simply returns the string representation.
    """
    return str(config) if config is not None else ""

def _set_classifier_head_on_args_global(args, config):
    """Mutate args to set the classifier head config.
    Stub stores the config on a new attribute if possible.
    """
    try:
        setattr(args, "classifier_head_config", config)
    except Exception:
        # args may be a simple namespace; ignore failures.
        pass
