
# /home/simon/otitenet/otitenet/app/pages/ensemble.py

from __future__ import annotations

import copy
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from otitenet.app.analysis import run_analysis_on_file
from otitenet.app.display_metrics import (
    _arrow_safe_dataframe,
    _best_head_config_for_args_global,
    _head_config_label_global,
    _set_classifier_head_on_args_global,
)
from otitenet.app.services.inference_results_service import (
    args_from_inference_row,
    fmt_confidence,
    normalize_analysis_result,
)
from otitenet.app.utils import _ensure_model_number_map, _make_model_selection_key


IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def _model_label(row: Dict[str, Any]) -> str:
    rank = row.get("#", "?")
    model_id = row.get("Model ID", "?")
    name = row.get("Model Name", "?")
    valid_mcc = row.get("Valid MCC", row.get("MCC", np.nan))
    try:
        score = f"{float(valid_mcc):.4f}"
    except Exception:
        score = str(valid_mcc)
    return f"#{rank} | ID {model_id} | {name} | score={score}"


def _clone_args(args):
    try:
        return copy.copy(args)
    except Exception:
        return args


def _load_ranked_models(cursor) -> pd.DataFrame:
    model_number_map, best_models_table = _ensure_model_number_map(cursor)

    if best_models_table is None or best_models_table.empty:
        return pd.DataFrame()

    df = best_models_table.copy().reset_index(drop=True)
    df["_selection_key"] = df.apply(lambda r: _make_model_selection_key(r.to_dict()), axis=1)

    if "#" not in df.columns:
        df["#"] = range(1, len(df) + 1)

    return df


def _prepare_model_args(base_args, model_row: Dict[str, Any]):
    model_args = args_from_inference_row(base_args, model_row)
    head_config = _best_head_config_for_args_global(model_args)
    _set_classifier_head_on_args_global(model_args, head_config)
    return model_args, str(head_config)


def _delete_existing(ctx, model_id, filenames):
    person_id = st.session_state.get("person_id") or getattr(ctx, "selected_person_id", None)

    if person_id is None or model_id is None or not filenames:
        return

    placeholders = ",".join(["%s"] * len(filenames))
    try:
        ctx.cursor.execute(
            f"""
            DELETE FROM results
            WHERE person_id=%s AND model_id=%s AND filename IN ({placeholders})
            """,
            tuple([int(person_id), model_id] + list(filenames)),
        )
        ctx.conn.commit()
    except Exception:
        try:
            ctx.conn.rollback()
        except Exception:
            pass


def _aggregate(rows: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    if not rows:
        return {
            "Ensemble Prediction": None,
            "Vote Count": 0,
            "Mean Confidence": np.nan,
            "Votes": "",
        }

    scores = defaultdict(float)
    counts = Counter()

    for r in rows:
        pred = str(r.get("Prediction"))
        conf = r.get("Confidence")
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0

        counts[pred] += 1

        if method == "confidence_weighted":
            scores[pred] += conf
        else:
            scores[pred] += 1.0

    best_pred = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    pred_rows = [r for r in rows if str(r.get("Prediction")) == best_pred]
    mean_conf = np.mean([float(r.get("Confidence") or 0) for r in pred_rows]) if pred_rows else np.nan

    votes = ", ".join([f"{k}:{v}" for k, v in counts.most_common()])

    return {
        "Ensemble Prediction": best_pred,
        "Vote Count": counts.get(best_pred, 0),
        "Mean Confidence": mean_conf,
        "Votes": votes,
    }


def _run_ensemble(ctx, uploaded_files, selected_rows: pd.DataFrame, method: str, force: bool):
    filenames = [os.path.basename(f.name) for f in uploaded_files]

    all_rows = []
    summary_rows = []

    progress = st.progress(0)
    total = len(uploaded_files) * len(selected_rows)
    done = 0

    for _, model_row in selected_rows.iterrows():
        model_dict = model_row.drop(labels=["_selection_key"], errors="ignore").to_dict()
        model_args, head_config = _prepare_model_args(ctx.args, model_dict)
        model_id = model_dict.get("Model ID")

        if force:
            _delete_existing(ctx, model_id, filenames)

        for f in uploaded_files:
            filename = os.path.basename(f.name)
            file_bytes = f.read()
            f.seek(0)

            try:
                analysis_result = run_analysis_on_file(
                    filename,
                    file_bytes,
                    model_args,
                    ctx.cursor,
                    ctx.conn,
                    force_reanalyze=force,
                    show_validation_metrics=False,
                    fast_infer=False,
                    quiet=True,
                )
                pred, conf, log_path, existing, gradcam_path = normalize_analysis_result(analysis_result)
                class_scores = analysis_result[5] if isinstance(analysis_result, (tuple, list)) and len(analysis_result) > 5 and isinstance(analysis_result[5], dict) else {}

                result_row = {
                    "Filename": filename,
                    "Model ID": model_id,
                    "Model": model_dict.get("Model Name"),
                    "Rank": model_dict.get("#"),
                    "Head": _head_config_label_global(head_config),
                    "Head Config": head_config,
                    "Prediction": pred,
                    "Confidence": conf,
                    "Existing": existing,
                    "Log Path": log_path,
                }
                for label, score in class_scores.items():
                    result_row[f"Score {label}"] = score
                all_rows.append(result_row)
            except Exception as exc:
                all_rows.append(
                    {
                        "Filename": filename,
                        "Model ID": model_id,
                        "Model": model_dict.get("Model Name"),
                        "Rank": model_dict.get("#"),
                        "Head": _head_config_label_global(head_config),
                        "Head Config": head_config,
                        "Prediction": "ERROR",
                        "Confidence": np.nan,
                        "Existing": False,
                        "Error": str(exc),
                    }
                )

            done += 1
            progress.progress(min(1.0, done / max(total, 1)))

    raw_df = pd.DataFrame(all_rows)

    for filename, g in raw_df.groupby("Filename"):
        valid = g[g["Prediction"].astype(str) != "ERROR"].to_dict("records")
        agg = _aggregate(valid, method)
        summary_rows.append({"Filename": filename, **agg, "Models Used": len(valid)})

    summary_df = pd.DataFrame(summary_rows)
    return raw_df, summary_df


def render(ctx: Any) -> None:
    st.header("👥 Ensemble")
    st.caption("Run several ranked models on the same image(s), then combine their predictions by vote or confidence-weighted vote.")

    person_id = st.session_state.get("person_id") or getattr(ctx, "selected_person_id", None)
    if person_id is None:
        st.warning("Please select a family member from the sidebar before running ensemble inference.")
        return

    models_df = _load_ranked_models(ctx.cursor)
    if models_df.empty:
        st.warning("No ranked models found. Run/update the best model registry first.")
        return

    st.subheader("Model selection")

    c1, c2, c3 = st.columns(3)

    with c1:
        top_n = st.number_input(
            "Use top N models",
            min_value=1,
            max_value=int(len(models_df)),
            value=min(5, int(len(models_df))),
            step=1,
            key="ensemble_top_n",
        )

    with c2:
        method = st.selectbox(
            "Aggregation",
            ["majority_vote", "confidence_weighted"],
            index=0,
            key="ensemble_method",
        )

    with c3:
        force = st.checkbox(
            "Force recompute",
            value=False,
            key="ensemble_force_recompute",
        )

    candidate_df = models_df.head(int(top_n)).copy()

    st.markdown("**Models included**")
    show_cols = [
        "#",
        "Model ID",
        "Model Name",
        "Valid MCC",
        "Test MCC",
        "Valid AUC",
        "Test AUC",
        "MCC",
        "Normalize",
        "N_Neighbors",
    ]
    show_cols = [c for c in show_cols if c in candidate_df.columns]
    st.dataframe(_arrow_safe_dataframe(candidate_df[show_cols]), use_container_width=True)

    uploaded_files = st.file_uploader(
        "Upload image(s) for ensemble inference",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key="ensemble_uploaded_files",
    )

    if uploaded_files:
        with st.expander("Preview uploads", expanded=len(uploaded_files) <= 3):
            cols = st.columns(min(3, len(uploaded_files)))
            for col, f in zip(cols, uploaded_files[:3]):
                with col:
                    st.image(f, caption=f.name, use_container_width=True)
                    f.seek(0)

    if not st.button("▶️ Run Ensemble", disabled=not bool(uploaded_files), type="primary", key="ensemble_run"):
        if "ensemble_summary_df" in st.session_state:
            st.subheader("Last ensemble summary")
            st.dataframe(st.session_state["ensemble_summary_df"], use_container_width=True)
            with st.expander("Last per-model predictions"):
                st.dataframe(st.session_state.get("ensemble_raw_df", pd.DataFrame()), use_container_width=True)
        return

    with st.spinner("Running ensemble inference..."):
        raw_df, summary_df = _run_ensemble(ctx, uploaded_files, candidate_df, method=method, force=force)

    st.session_state["ensemble_raw_df"] = raw_df
    st.session_state["ensemble_summary_df"] = summary_df

    st.success("Ensemble completed.")

    st.subheader("Ensemble summary")
    display_summary = summary_df.copy()
    if "Mean Confidence" in display_summary.columns:
        display_summary["Mean Confidence"] = display_summary["Mean Confidence"].apply(fmt_confidence)
    st.dataframe(_arrow_safe_dataframe(display_summary), use_container_width=True)

    st.subheader("Per-model predictions")
    display_raw = raw_df.copy()
    if "Confidence" in display_raw.columns:
        display_raw["Confidence"] = display_raw["Confidence"].apply(fmt_confidence)
    for col in [c for c in display_raw.columns if str(c).startswith("Score ")]:
        display_raw[col] = display_raw[col].apply(fmt_confidence)
    st.dataframe(_arrow_safe_dataframe(display_raw.drop(columns=["Log Path"], errors="ignore")), use_container_width=True)
