"""Inference Results page."""

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, auc as sk_auc, matthews_corrcoef, roc_curve

from otitenet.app.display_metrics import (
    _arrow_safe_dataframe,
    _best_head_config_for_args_global,
    _best_head_entry_for_args_global,
    _classification_head_options_for_args_global,
    _head_config_label_global,
    _set_classifier_head_on_args_global,
)
from otitenet.app.services.inference_results_service import (
    args_from_inference_row,
    compute_inference_metrics,
    find_inference_gradcam_images,
    fmt_confidence,
    fmt_metric,
    inference_ground_truth,
    labels_match,
    normalize_analysis_result,
)
from otitenet.app.utils import _ensure_model_number_map, _make_model_selection_key

# if "person_id" not in st.session_state:
#     st.session_state.person_id = None

# if st.session_state.person_id is None:
#     st.warning("Please select a person before viewing inference results.")
#     return


def _truth_is_label(truth, label):
    return str(truth or "").strip().lower() == str(label or "").strip().lower()


def _correct_state(prediction, truth):
    if str(prediction or "").strip().lower() in {"", "unknown", "na", "nan", "none"}:
        return "X"
    return "✓" if labels_match(prediction, truth) else ""


def _plot_one_vs_rest_roc(df, title, score_prefix="Score ", prediction_col="Prediction"):
    if df is None or df.empty or "Ground Truth" not in df.columns:
        return False

    score_cols = [c for c in df.columns if str(c).startswith(score_prefix)]
    if not score_cols:
        st.info(f"No per-class scores available for {title}. Recompute inference once so class_scores are stored.")
        return False

    fig, ax = plt.subplots(figsize=(7, 4.2))
    curves = 0
    for col in score_cols:
        label = str(col)[len(score_prefix):]
        scores = pd.to_numeric(df[col], errors="coerce")
        truth = df["Ground Truth"].astype(str)
        valid = scores.notna() & truth.str.strip().ne("")
        if prediction_col in df.columns:
            valid &= ~df[prediction_col].astype(str).str.lower().isin(["unknown", "na", "nan", "none", ""])
        if valid.sum() < 2:
            continue
        y_true = truth[valid].apply(lambda value: 1 if _truth_is_label(value, label) else 0).values
        if len(np.unique(y_true)) < 2:
            continue
        y_score = scores[valid].astype(float).values
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_value = sk_auc(fpr, tpr)
        except Exception:
            continue
        ax.plot(fpr, tpr, linewidth=1.7, label=f"{label} vs rest (AUC={auc_value:.3f})")
        curves += 1

    if curves == 0:
        plt.close(fig)
        st.info(f"Not enough positive/negative samples to draw ROC curves for {title}.")
        return False

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.5")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    return True


def _threshold_decision_for_row(
    row,
    confidence_threshold,
    vote_threshold_pct,
    require_both_thresholds,
):
    try:
        confidence_value = float(row.get("Confidence"))
    except Exception:
        confidence_value = np.nan
    try:
        ensemble_vote_pct = float(row.get("Ensemble Raw Vote %"))
    except Exception:
        ensemble_vote_pct = np.nan

    selected_prediction = row.get("Prediction")
    ensemble_prediction = row.get("Ensemble Raw Prediction", "Unknown")
    selected_passes = pd.notna(confidence_value) and confidence_value >= float(confidence_threshold)
    ensemble_passes = pd.notna(ensemble_vote_pct) and ensemble_vote_pct >= float(vote_threshold_pct)

    if require_both_thresholds:
        return selected_prediction if selected_passes and ensemble_passes else "Unknown"
    if selected_passes:
        return selected_prediction
    return ensemble_prediction if ensemble_passes and str(ensemble_prediction or "").strip() else "Unknown"


def _render_threshold_heatmap(existing_df, require_both_thresholds):
    if existing_df is None or existing_df.empty:
        return
    required = {"Confidence", "Prediction", "Ground Truth", "Ensemble Raw Prediction", "Ensemble Raw Vote %"}
    if not required.issubset(set(existing_df.columns)):
        return

    st.subheader("Threshold Sensitivity Heatmap")
    metric_name = st.selectbox(
        "Heatmap metric",
        options=["ACC kept", "ACC unknown=wrong", "Coverage", "MCC kept", "MCC unknown=wrong"],
        index=0,
        key="inference_threshold_heatmap_metric",
    )
    conf_values = np.round(np.linspace(0.0, 1.0, 21), 2)
    vote_values = np.arange(0.0, 101.0, 5.0)
    heat = np.full((len(vote_values), len(conf_values)), np.nan, dtype=float)

    valid_truth = ~existing_df["Ground Truth"].astype(str).str.lower().isin(["", "unknown", "na", "nan", "none"])
    base_df = existing_df[valid_truth].copy()
    if base_df.empty:
        st.info("No ground-truth labels available for threshold heatmap.")
        return

    for y_idx, vote_threshold in enumerate(vote_values):
        for x_idx, confidence_threshold in enumerate(conf_values):
            decisions = base_df.apply(
                lambda row: _threshold_decision_for_row(
                    row,
                    confidence_threshold,
                    vote_threshold,
                    require_both_thresholds,
                ),
                axis=1,
            )
            eval_df = base_df.assign(_Decision=decisions)
            kept_df = eval_df[~eval_df["_Decision"].astype(str).str.lower().isin(["", "unknown", "na", "nan", "none"])].copy()
            if metric_name == "Coverage":
                value = float(len(kept_df) / len(eval_df)) if len(eval_df) else np.nan
            elif metric_name.endswith("kept"):
                if kept_df.empty:
                    value = np.nan
                elif metric_name.startswith("ACC"):
                    value = float(np.mean([
                        labels_match(pred, truth)
                        for pred, truth in zip(kept_df["_Decision"], kept_df["Ground Truth"])
                    ]))
                else:
                    try:
                        value = float(matthews_corrcoef(
                            kept_df["Ground Truth"].astype(str).str.lower().values,
                            kept_df["_Decision"].astype(str).str.lower().values,
                        ))
                    except Exception:
                        value = np.nan
            else:
                pred = eval_df["_Decision"].astype(str).str.lower().replace({"unknown": "__unknown__"})
                truth = eval_df["Ground Truth"].astype(str).str.lower()
                if metric_name.startswith("ACC"):
                    value = float(accuracy_score(truth.values, pred.values))
                else:
                    try:
                        value = float(matthews_corrcoef(truth.values, pred.values))
                    except Exception:
                        value = np.nan
            heat[y_idx, x_idx] = value

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(conf_values))[::2])
    ax.set_xticklabels([f"{v:.1f}" for v in conf_values[::2]])
    ax.set_yticks(np.arange(len(vote_values))[::2])
    ax.set_yticklabels([f"{v:.0f}" for v in vote_values[::2]])
    ax.set_xlabel("Selected model confidence threshold")
    ax.set_ylabel("Top-N ensemble vote threshold (%)")
    ax.set_title(metric_name)
    fig.colorbar(im, ax=ax, label=metric_name)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def render(ctx):
    import streamlit as st

    if "person_id" not in st.session_state:
        st.session_state.person_id = None

    if st.session_state.person_id is None:
        st.warning("Please select a person before viewing inference results.")
        return

    # existing code continues here
    args = ctx.args
    conn = ctx.conn
    cursor = ctx.cursor
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

            def _as_float_or_nan(value):
                try:
                    if value is None or pd.isna(value):
                        return np.nan
                    return float(value)
                except Exception:
                    return np.nan

            def _candidate_head_cache_paths(row_dict):
                raw_path = (
                    row_dict.get("Artifact Log Path")
                    or row_dict.get("Best Model Dir")
                    or row_dict.get("Log Path")
                    or row_dict.get("log_path")
                    or ""
                )
                path = str(raw_path or "").strip()
                if not path:
                    return []
                candidates = []
                cursor_path = path
                if os.path.isfile(cursor_path):
                    cursor_path = os.path.dirname(cursor_path)
                candidates.append(os.path.join(cursor_path, "knn_optimization_cache.pkl"))
                parts = cursor_path.replace("\\", "/").split("/")
                if "dist_" in "/".join(parts):
                    for idx, part in enumerate(parts):
                        if part.startswith("dist_"):
                            candidates.append(os.path.join("/".join(parts[:idx]), "knn_optimization_cache.pkl"))
                            break
                parent = cursor_path
                for _ in range(3):
                    parent = os.path.dirname(parent)
                    if not parent or parent == ".":
                        break
                    candidates.append(os.path.join(parent, "knn_optimization_cache.pkl"))
                return list(dict.fromkeys(candidates))

            def _head_summary_from_direct_cache(row_dict):
                best = None
                for cache_path in _candidate_head_cache_paths(row_dict):
                    if not os.path.exists(cache_path):
                        continue
                    try:
                        with open(cache_path, "rb") as f:
                            cache = pickle.load(f)
                    except Exception:
                        continue
                    if not isinstance(cache, dict):
                        continue

                    def _consider(config, score, details=""):
                        nonlocal best
                        score_f = _as_float_or_nan(score)
                        if pd.isna(score_f):
                            return
                        if best is None or score_f > best["score"]:
                            best = {
                                "label": _head_config_label_global(config),
                                "config": str(config),
                                "score": score_f,
                                "details": details,
                            }

                    for n_aug, result in cache.items():
                        if not isinstance(result, dict):
                            continue
                        details = f"n_aug={n_aug}"
                        baselines = result.get("baselines") or {}
                        if isinstance(baselines, dict):
                            for baseline_name, baseline_data in baselines.items():
                                if not isinstance(baseline_data, dict):
                                    continue
                                _consider(
                                    f"baseline_{baseline_name}",
                                    baseline_data.get("valid_mcc", baseline_data.get("mcc")),
                                    details,
                                )

                        knn_data = result.get("knn") or {}
                        if isinstance(knn_data, dict):
                            for idx, item in enumerate(knn_data.get("mcc_per_k", []) or []):
                                if isinstance(item, dict):
                                    _consider(item.get("k", idx + 1), item.get("valid_mcc", item.get("mcc")), details)
                                else:
                                    _consider(idx + 1, item, details)

                        proto_data = result.get("prototypes") or {}
                        if isinstance(proto_data, dict):
                            for strategy, strat_data in proto_data.items():
                                if not isinstance(strat_data, dict):
                                    continue
                                n_comp = strat_data.get("best_n_components", 1)
                                _consider(
                                    f"protot_{strategy}_{n_comp}",
                                    strat_data.get("valid_mcc", strat_data.get("best_mcc")),
                                    details,
                                )

                        if result.get("best_config") is not None:
                            _consider(result.get("best_config"), result.get("best_mcc"), details)
                    if best is not None:
                        return best["label"], best["config"], best["score"]
                return None

            def _head_summary_for_row(row_dict):
                direct_summary = _head_summary_from_direct_cache(row_dict)
                if direct_summary is not None:
                    return direct_summary

                tmp_args = args_from_inference_row(args, row_dict)
                try:
                    best_head_entry = _best_head_entry_for_args_global(tmp_args) or {}
                except Exception:
                    best_head_entry = {}

                head_config = best_head_entry.get("config")
                if head_config is None or str(head_config).strip() in {"", "None", "nan"}:
                    try:
                        head_config = _best_head_config_for_args_global(tmp_args)
                    except Exception:
                        head_config = None
                if head_config is None or str(head_config).strip() in {"", "None", "nan"}:
                    head_config = row_dict.get("N_Neighbors") or getattr(tmp_args, "n_neighbors", 1)
                head_config = str(head_config)

                head_label = best_head_entry.get("label") or _head_config_label_global(head_config)
                if not head_label or str(head_label).strip() in {"", "None", "nan"}:
                    head_label = f"KNN (k={head_config})" if str(head_config).isdigit() else str(head_config)

                head_score = _as_float_or_nan(best_head_entry.get("valid_mcc", best_head_entry.get("mcc")))
                if pd.isna(head_score):
                    head_score = _as_float_or_nan(row_dict.get("Valid MCC"))
                if pd.isna(head_score):
                    head_score = _as_float_or_nan(row_dict.get("MCC"))

                return head_label, head_config, head_score

            head_summaries = inference_model_df.apply(
                lambda r: _head_summary_for_row(r.to_dict()), axis=1
            )
            inference_model_df["Best Classification Head"] = head_summaries.map(lambda item: item[0])
            inference_model_df["Best Head Config"] = head_summaries.map(lambda item: item[1])
            inference_model_df["Best Head Score"] = head_summaries.map(lambda item: item[2])

            # Rank inference candidates by optimized-head validation MCC when available.
            # The backbone/run Valid MCC remains available as Backbone Valid MCC.
            inference_model_df["Backbone Valid MCC"] = inference_model_df.get("Valid MCC", np.nan)
            inference_model_df["Display Valid MCC"] = inference_model_df["Best Head Score"].where(
                pd.to_numeric(inference_model_df["Best Head Score"], errors="coerce").notna(),
                inference_model_df.get("Valid MCC", np.nan),
            )
            inference_model_df = (
                inference_model_df.assign(
                    _head_sort=pd.to_numeric(inference_model_df["Display Valid MCC"], errors="coerce").fillna(-np.inf),
                    _mcc_sort=pd.to_numeric(inference_model_df.get("MCC", np.nan), errors="coerce").fillna(-np.inf),
                )
                .sort_values(["_head_sort", "_mcc_sort"], ascending=[False, False])
                .drop(columns=["_head_sort", "_mcc_sort"])
                .reset_index(drop=True)
            )
            inference_model_df["#"] = np.arange(1, len(inference_model_df) + 1)
            model_number_map = {
                row["_selection_key"]: row["#"]
                for _, row in inference_model_df.iterrows()
            }

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
                valid_mcc = row.get("Display Valid MCC", row.get("Valid MCC", np.nan))
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
                df["Model ID"] = pd.to_numeric(df["Model ID"], errors="coerce").astype("Int64")
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                df = (
                    df.sort_values("Timestamp", ascending=False)
                    .drop_duplicates(subset=["Model ID", "Filename"], keep="first")
                    .reset_index(drop=True)
                )
                df["Ground Truth"] = df["Filename"].apply(lambda filename: inference_ground_truth(filename, inference_dir))
                df["Correct"] = df.apply(lambda r: labels_match(r["Prediction"], r["Ground Truth"]), axis=1)
                return df

            def _decision_correct(prediction, truth):
                if str(prediction or "").strip().lower() == "unknown":
                    return False
                return bool(labels_match(prediction, truth))

            def _decision_metrics(decision_df, prediction_col):
                if decision_df is None or decision_df.empty or prediction_col not in decision_df.columns:
                    return {
                        "unknown_wrong": {"ACC": np.nan, "MCC": np.nan, "N": 0},
                        "kept": {"ACC": np.nan, "MCC": np.nan, "N": 0, "Coverage": np.nan},
                    }
                work = decision_df.copy()
                truth = work["Ground Truth"].astype(str).str.strip()
                valid_truth = ~truth.str.lower().isin(["", "unknown", "na", "nan", "none"])
                work = work[valid_truth].copy()
                if work.empty:
                    return {
                        "unknown_wrong": {"ACC": np.nan, "MCC": np.nan, "N": 0},
                        "kept": {"ACC": np.nan, "MCC": np.nan, "N": 0, "Coverage": np.nan},
                    }

                def _calc(df):
                    if df.empty:
                        return {"ACC": np.nan, "MCC": np.nan, "N": 0}
                    y_true = df["Ground Truth"].astype(str).str.lower().values
                    y_pred = df[prediction_col].astype(str).str.lower().values
                    try:
                        acc = float(accuracy_score(y_true, y_pred))
                    except Exception:
                        acc = np.nan
                    try:
                        mcc = float(matthews_corrcoef(y_true, y_pred))
                    except Exception:
                        mcc = np.nan
                    return {"ACC": acc, "MCC": mcc, "N": int(len(df))}

                unknown_wrong_df = work.copy()
                unknown_wrong_df[prediction_col] = unknown_wrong_df[prediction_col].replace(
                    to_replace=["", None],
                    value="Unknown",
                )
                unknown_wrong_df[prediction_col] = unknown_wrong_df[prediction_col].fillna("Unknown")
                unknown_wrong_df[prediction_col] = unknown_wrong_df[prediction_col].mask(
                    unknown_wrong_df[prediction_col].astype(str).str.lower().eq("unknown"),
                    "__unknown__",
                )
                kept_df = work[~work[prediction_col].astype(str).str.lower().isin(["", "unknown", "na", "nan", "none"])].copy()
                unknown_wrong = _calc(unknown_wrong_df)
                kept = _calc(kept_df)
                kept["Coverage"] = float(len(kept_df) / len(work)) if len(work) else np.nan
                return {"unknown_wrong": unknown_wrong, "kept": kept}

            def _show_decision_metrics(title, decision_df, prediction_col):
                metrics = _decision_metrics(decision_df, prediction_col)
                st.markdown(f"**{title}**")
                cols = st.columns(7)
                cols[0].metric("ACC unknown=wrong", fmt_metric(metrics["unknown_wrong"]["ACC"]))
                cols[1].metric("MCC unknown=wrong", fmt_metric(metrics["unknown_wrong"]["MCC"]))
                cols[2].metric("N", metrics["unknown_wrong"]["N"])
                cols[3].metric("Coverage", "" if pd.isna(metrics["kept"]["Coverage"]) else f"{100.0 * metrics['kept']['Coverage']:.1f}%")
                cols[4].metric("ACC kept", fmt_metric(metrics["kept"]["ACC"]))
                cols[5].metric("MCC kept", fmt_metric(metrics["kept"]["MCC"]))
                cols[6].metric("N kept", metrics["kept"]["N"])

            def _build_top_n_votes_df(
                top_n_results_df,
                selected_model_id,
                selected_model_name,
                vote_threshold_pct,
                selected_confidence_threshold,
                selected_model_results_df=None,
            ):
                if top_n_results_df is None or top_n_results_df.empty:
                    return pd.DataFrame()

                vote_source = top_n_results_df.copy()
                vote_source["Model ID"] = pd.to_numeric(vote_source["Model ID"], errors="coerce").astype("Int64")
                vote_source["Prediction"] = vote_source["Prediction"].fillna("Unknown").astype(str)
                try:
                    selected_model_id_int = int(selected_model_id)
                except Exception:
                    selected_model_id_int = None

                vote_counts = (
                    vote_source.groupby(["Filename", "Prediction"])
                    .size()
                    .unstack(fill_value=0)
                    .reset_index()
                )
                class_columns = [c for c in vote_counts.columns if c != "Filename"]
                vote_counts = vote_counts.rename(columns={c: f"Votes {c}" for c in class_columns})

                vote_totals = (
                    vote_source.groupby("Filename")
                    .agg(
                        **{
                            "Models Voted": ("Model ID", "nunique"),
                            "Ground Truth": ("Ground Truth", "first"),
                        }
                    )
                    .reset_index()
                )

                consensus_rows = []
                vote_cols = [f"Votes {c}" for c in class_columns]
                for _, row in vote_counts.iterrows():
                    if not vote_cols:
                        consensus_rows.append({"Filename": row["Filename"], "Ensemble Prediction": "Unknown", "Ensemble Vote %": 0.0, "Top-N Consensus": "", "Top-N Consensus Votes": 0})
                        continue
                    counts = {col[len("Votes "):]: int(row[col]) for col in vote_cols}
                    consensus_label, consensus_votes = max(counts.items(), key=lambda item: item[1])
                    models_voted = int(sum(counts.values()))
                    vote_pct = 100.0 * float(consensus_votes) / float(models_voted) if models_voted else 0.0
                    ensemble_prediction = consensus_label if vote_pct >= float(vote_threshold_pct) else "Unknown"
                    consensus_rows.append(
                        {
                            "Filename": row["Filename"],
                            "Top-N Consensus": consensus_label,
                            "Top-N Consensus Votes": consensus_votes,
                            "Ensemble Vote %": vote_pct,
                            "Ensemble Prediction": ensemble_prediction,
                        }
                    )
                consensus_df = pd.DataFrame(consensus_rows)

                selected_predictions = pd.DataFrame()
                if selected_model_results_df is not None and not selected_model_results_df.empty:
                    selected_predictions = selected_model_results_df.copy()
                    selected_predictions["Model ID"] = pd.to_numeric(
                        selected_predictions["Model ID"],
                        errors="coerce",
                    ).astype("Int64")
                elif selected_model_id_int is not None:
                    selected_predictions = vote_source[vote_source["Model ID"] == selected_model_id_int].copy()

                if not selected_predictions.empty:
                    selected_predictions = selected_predictions[
                        ["Filename", "Prediction", "Confidence", "Correct", "Timestamp"]
                    ].copy()
                    selected_predictions = selected_predictions.rename(
                        columns={
                            "Prediction": "Selected Model Prediction",
                            "Confidence": "Selected Model Confidence",
                            "Correct": "Selected Model Correct",
                            "Timestamp": "Selected Model Timestamp",
                        }
                    )

                votes_df = vote_counts.merge(vote_totals, on="Filename", how="left")
                votes_df = votes_df.merge(consensus_df, on="Filename", how="left")
                if not selected_predictions.empty:
                    votes_df = votes_df.merge(selected_predictions, on="Filename", how="left")
                else:
                    votes_df["Selected Model Prediction"] = ""
                    votes_df["Selected Model Confidence"] = ""
                    votes_df["Selected Model Correct"] = ""

                votes_df["Selected Model ID"] = selected_model_id
                votes_df["Selected Model"] = selected_model_name
                votes_df["Selected Model Confidence Raw"] = pd.to_numeric(
                    votes_df.get("Selected Model Confidence"),
                    errors="coerce",
                )

                def _final_decision(row):
                    selected_prediction = str(row.get("Selected Model Prediction") or "").strip()
                    selected_available = selected_prediction.lower() not in {"", "unknown", "na", "nan", "none"}
                    selected_confidence = row.get("Selected Model Confidence Raw")
                    selected_passes = (
                        selected_available
                        and pd.notna(selected_confidence)
                        and float(selected_confidence) >= float(selected_confidence_threshold)
                    )
                    if selected_passes:
                        return selected_prediction
                    return row.get("Ensemble Prediction", "Unknown")

                def _decision_source(row):
                    selected_prediction = str(row.get("Selected Model Prediction") or "").strip()
                    selected_available = selected_prediction.lower() not in {"", "unknown", "na", "nan", "none"}
                    selected_confidence = row.get("Selected Model Confidence Raw")
                    if (
                        selected_available
                        and pd.notna(selected_confidence)
                        and float(selected_confidence) >= float(selected_confidence_threshold)
                    ):
                        return "Selected model"
                    return "Top-N fallback"

                votes_df["Decision Prediction"] = votes_df.apply(_final_decision, axis=1)
                votes_df["Decision Source"] = votes_df.apply(_decision_source, axis=1)
                votes_df["Selected Model Correct"] = votes_df.apply(
                    lambda r: _correct_state(r.get("Selected Model Prediction"), r.get("Ground Truth")),
                    axis=1,
                )
                votes_df["Ensemble Correct"] = votes_df.apply(
                    lambda r: _correct_state(r.get("Ensemble Prediction"), r.get("Ground Truth")),
                    axis=1,
                )
                votes_df["Decision Correct"] = votes_df.apply(
                    lambda r: _correct_state(r.get("Decision Prediction"), r.get("Ground Truth")),
                    axis=1,
                )
                if "Selected Model Confidence" in votes_df.columns:
                    votes_df["Selected Model Confidence"] = votes_df["Selected Model Confidence"].apply(fmt_confidence)
                return votes_df

            def _render_top_n_vote_table(
                top_n_results_df,
                selected_model_id,
                selected_model_name,
                vote_threshold_pct,
                selected_confidence_threshold,
                selected_model_results_df=None,
            ):
                st.divider()
                st.subheader("Top-N Class Votes / Ensemble Classification")
                st.caption(
                    f"Final decision uses the selected model when confidence is at least "
                    f"{float(selected_confidence_threshold):.2f}. Top-N voting is only used as fallback below that "
                    f"confidence; the fallback winner must reach {float(vote_threshold_pct):.1f}% of votes."
                )

                votes_df = _build_top_n_votes_df(
                    top_n_results_df,
                    selected_model_id,
                    selected_model_name,
                    vote_threshold_pct,
                    selected_confidence_threshold,
                    selected_model_results_df=selected_model_results_df,
                )
                if votes_df.empty:
                    st.info("No Top-N stored inference results available yet. Compute inference for the selected top models first.")
                    return votes_df

                ordered_cols = [
                    "Filename",
                    "Ground Truth",
                    "Decision Prediction",
                    "Decision Correct",
                    "Decision Source",
                    "Selected Model Prediction",
                    "Selected Model Confidence",
                    "Selected Model Correct",
                    "Ensemble Prediction",
                    "Ensemble Correct",
                    "Ensemble Vote %",
                    "Models Voted",
                ]
                ordered_cols += sorted([c for c in votes_df.columns if str(c).startswith("Votes ")])
                ordered_cols += [
                    "Selected Model ID",
                    "Selected Model",
                ]
                ordered_cols += [
                    c for c in votes_df.columns
                    if c not in ordered_cols and c not in {"Selected Model Timestamp", "Selected Model Confidence Raw"}
                ]
                _show_decision_metrics("Final decision metrics", votes_df, "Decision Prediction")
                st.caption(
                    "These metrics use the selected model first and apply the Top-N vote only as fallback."
                )
                if "Selected Model Prediction" in votes_df.columns and votes_df["Selected Model Prediction"].astype(str).str.strip().ne("").any():
                    _show_decision_metrics(
                        f"Selected model metrics ({selected_model_name})",
                        votes_df,
                        "Selected Model Prediction",
                    )
                with st.expander("Fallback Top-N ensemble diagnostics", expanded=False):
                    _show_decision_metrics("Top-N fallback metrics", votes_df, "Ensemble Prediction")
                roc_votes_df = votes_df.copy()
                for vote_col in [c for c in roc_votes_df.columns if str(c).startswith("Votes ")]:
                    roc_votes_df[f"Score {str(vote_col)[len('Votes '):]}"] = (
                        pd.to_numeric(roc_votes_df[vote_col], errors="coerce")
                        / pd.to_numeric(roc_votes_df["Models Voted"], errors="coerce").replace(0, np.nan)
                    )
                _plot_one_vs_rest_roc(
                    roc_votes_df,
                    "Top-N ensemble ROC curves",
                    score_prefix="Score ",
                    prediction_col="Ensemble Prediction",
                )
                display_votes_df = votes_df[ordered_cols].copy()
                display_votes_df["Ensemble Vote %"] = display_votes_df["Ensemble Vote %"].apply(
                    lambda v: "" if pd.isna(v) else f"{float(v):.1f}"
                )
                st.dataframe(
                    _arrow_safe_dataframe(display_votes_df),
                    use_container_width=True,
                    hide_index=True,
                )
                return votes_df

            top_n_existing_df = _fetch_latest_inference_results_for_models(top_n_model_ids)
            performance_rows = []
            if not top_n_model_df.empty:
                for _, perf_model_row in top_n_model_df.iterrows():
                    perf_model_id = perf_model_row.get("Model ID")
                    try:
                        perf_model_id_int = int(perf_model_id)
                    except Exception:
                        perf_model_id_int = None
                    model_results_df = (
                        top_n_existing_df[top_n_existing_df["Model ID"] == perf_model_id_int].copy()
                        if perf_model_id_int is not None and not top_n_existing_df.empty
                        else pd.DataFrame()
                    )
                    metrics = compute_inference_metrics(model_results_df)
                    rank_value = perf_model_row.get("#", model_number_map.get(perf_model_row.get("_selection_key"), "?"))
                    head_label = perf_model_row.get("Best Classification Head") or "—"
                    head_config = perf_model_row.get("Best Head Config") or "—"
                    head_score = perf_model_row.get("Best Head Score", np.nan)
                    perf_row = {
                        "#": rank_value,
                        "Model ID": perf_model_id,
                        "Model Name": perf_model_row.get("Model Name"),
                        "Best Classification Head": head_label,
                        "Best Head Config": head_config,
                        "Best Head Score": fmt_metric(head_score),
                        "ACC": fmt_metric(metrics["ACC"]),
                        "MCC": fmt_metric(metrics["MCC"]),
                        "AUC": fmt_metric(metrics["AUC"]),
                        "N analyzed": metrics["N"],
                    }
                    for metric_name, metric_value in metrics.items():
                        if metric_name in {"ACC", "MCC", "AUC", "N"}:
                            continue
                        if metric_name.startswith(("F1 ", "Recall ", "Precision ")):
                            perf_row[metric_name] = fmt_metric(metric_value)
                        elif metric_name.startswith("Support "):
                            perf_row[metric_name] = metric_value
                    performance_rows.append(perf_row)
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
                        batch_model_args = args_from_inference_row(args, batch_model_dict)
                        batch_model_args.allow_inference_reencode = False
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
                                generate_gradcam=False,
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
            model_args = args_from_inference_row(args, selected_model_dict)

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

            st.divider()
            st.subheader("Decision Thresholds")
            threshold_cols = st.columns(2)
            with threshold_cols[0]:
                selected_confidence_threshold = st.slider(
                    "Selected model confidence threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("production_selected_confidence_threshold", 0.50)),
                    step=0.01,
                    key="inference_selected_confidence_threshold",
                    help="If the selected model confidence is below this value, the decision falls back to the Top-N ensemble vote.",
                )
            with threshold_cols[1]:
                ensemble_vote_threshold_pct = st.slider(
                    "Top-N ensemble vote threshold (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get("production_ensemble_vote_threshold_pct", 80.0)),
                    step=1.0,
                    key="inference_ensemble_vote_threshold_pct",
                    help="If the winning Top-N class has less than this percent of votes, the ensemble decision is Unknown.",
                )
            require_both_thresholds = False
            st.caption(
                "Decision rule: selected model first. Top-N voting is used only when selected-model confidence is below "
                "the selected-model threshold."
            )
            selected_model_existing_df = _fetch_latest_inference_results_for_models([model_id]) if model_id is not None else pd.DataFrame()

            def _generate_gradcams_for_files(filenames, log_path_by_filename=None, skip_existing=True):
                filenames = [str(name) for name in filenames if str(name or "").strip()]
                if not filenames:
                    return {"generated": 0, "skipped": 0, "errors": []}

                from otitenet.app.analysis import _safe_generate_gradcam_for_prediction
                from otitenet.app.image_processing import get_image
                from otitenet.app.model_loading import load_model_and_prototypes

                with st.spinner(f"Generating Grad-CAM for {len(filenames)} image(s)..."):
                    model, _, prototypes, image_size, device_str, _, _, _, _ = load_model_and_prototypes(model_args)
                    generated = 0
                    skipped = 0
                    errors = []

                    for filename in filenames:
                        result_log_path = (log_path_by_filename or {}).get(filename) or os.path.join(
                            "logs",
                            "best_models",
                            str(model_args.task),
                            str(model_args.model_name),
                            "queries",
                        )
                        if skip_existing and find_inference_gradcam_images(result_log_path, filename, n_layers=4):
                            skipped += 1
                            continue

                        image_path = os.path.join(inference_dir, filename)
                        if not os.path.exists(image_path):
                            errors.append(f"{filename}: image file not found")
                            continue

                        try:
                            _, image_tensor = get_image(image_path, size=image_size, normalize=model_args.normalize)
                            out_path = _safe_generate_gradcam_for_prediction(
                                filename,
                                model_args,
                                model,
                                image_tensor,
                                prototypes,
                                device_str,
                                result_log_path,
                            )
                            if out_path:
                                generated += 1
                            else:
                                errors.append(f"{filename}: Grad-CAM generation returned no path")
                        except Exception as exc:
                            errors.append(f"{filename}: {exc}")

                return {"generated": generated, "skipped": skipped, "errors": errors}

            top_n_votes_df = _render_top_n_vote_table(
                top_n_existing_df,
                model_id,
                model_name,
                ensemble_vote_threshold_pct,
                selected_confidence_threshold,
                selected_model_results_df=selected_model_existing_df,
            )

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
                            pred_label, confidence, complete_log_path, _existing, gradcam_path = normalize_analysis_result(analysis_result)
                            class_scores = analysis_result[5] if isinstance(analysis_result, (tuple, list)) and len(analysis_result) > 5 and isinstance(analysis_result[5], dict) else {}

                            ground_truth = inference_ground_truth(img_name, inference_dir)
                            is_correct = labels_match(pred_label, ground_truth)
                            if is_correct is True:
                                correct_count += 1

                            result_row = {
                                'Filename': img_name,
                                'Head': _head_config_label_global(selected_head_config),
                                'Head Config': selected_head_config,
                                'Ground Truth': ground_truth,
                                'Prediction': pred_label,
                                'Correct': is_correct,
                                'Confidence': fmt_confidence(confidence),
                            }
                            for label, score in class_scores.items():
                                result_row[f"Score {label}"] = fmt_confidence(score)
                            results.append(result_row)

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
                try:
                    cursor.execute(
                        f"""
                        SELECT filename, pred_label, confidence, timestamp, log_path, class_scores
                        FROM results
                        WHERE person_id=%s AND model_id=%s AND filename IN ({placeholders})
                        ORDER BY timestamp DESC
                        """,
                        tuple([st.session_state.person_id, model_id] + inference_filenames),
                    )
                except Exception:
                    # Backward-compatible fallback if class_scores column is unavailable.
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
                    columns=['Filename', 'Prediction', 'Confidence', 'Timestamp', 'Log Path', 'Class Scores'] if len(existing_results[0]) >= 6 else ['Filename', 'Prediction', 'Confidence', 'Timestamp', 'Log Path'],
                )

                score_cols = []
                if 'Class Scores' in existing_df.columns:
                    def _parse_scores(value):
                        if isinstance(value, dict):
                            return value
                        if value in [None, "", b""]:
                            return {}
                        try:
                            if isinstance(value, (bytes, bytearray)):
                                value = value.decode('utf-8', errors='ignore')
                            parsed = json.loads(str(value))
                            return parsed if isinstance(parsed, dict) else {}
                        except Exception:
                            return {}

                    parsed_score_series = existing_df['Class Scores'].apply(_parse_scores)
                    score_labels = sorted({str(lbl) for row_scores in parsed_score_series for lbl in row_scores.keys()})
                    for score_label in score_labels:
                        col_name = f"Score {score_label}"
                        existing_df[col_name] = parsed_score_series.apply(
                            lambda row_scores: fmt_confidence(row_scores.get(score_label)) if score_label in row_scores else ""
                        )
                        score_cols.append(col_name)

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
                existing_df['Ground Truth'] = existing_df['Filename'].apply(lambda filename: inference_ground_truth(filename, inference_dir))
                existing_df['Correct'] = existing_df.apply(
                    lambda r: labels_match(r['Prediction'], r['Ground Truth']),
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

                ensemble_prediction_by_filename = {}
                ensemble_vote_pct_by_filename = {}
                ensemble_raw_prediction_by_filename = {}
                if top_n_votes_df is not None and not top_n_votes_df.empty:
                    ensemble_prediction_by_filename = top_n_votes_df.set_index("Filename")["Ensemble Prediction"].to_dict()
                    ensemble_vote_pct_by_filename = top_n_votes_df.set_index("Filename")["Ensemble Vote %"].to_dict()
                    if "Top-N Consensus" in top_n_votes_df.columns:
                        ensemble_raw_prediction_by_filename = top_n_votes_df.set_index("Filename")["Top-N Consensus"].to_dict()

                def _selected_threshold_decision(row):
                    return _threshold_decision_for_row(
                        row,
                        selected_confidence_threshold,
                        ensemble_vote_threshold_pct,
                        require_both_thresholds,
                    )

                existing_df["Selected Model Prediction"] = existing_df["Prediction"]
                existing_df["Selected Model Correct"] = existing_df.apply(
                    lambda r: _correct_state(r.get("Selected Model Prediction"), r.get("Ground Truth")),
                    axis=1,
                )
                existing_df["Ensemble Raw Prediction"] = (
                    existing_df["Filename"].map(ensemble_raw_prediction_by_filename).fillna("Unknown")
                )
                existing_df["Ensemble Raw Vote %"] = existing_df["Filename"].map(ensemble_vote_pct_by_filename)
                existing_df["Ensemble Prediction"] = existing_df["Filename"].map(ensemble_prediction_by_filename).fillna("Unknown")
                existing_df["Ensemble Correct"] = existing_df.apply(
                    lambda r: _correct_state(r.get("Ensemble Prediction"), r.get("Ground Truth")),
                    axis=1,
                )
                existing_df["Decision"] = existing_df.apply(_selected_threshold_decision, axis=1)
                existing_df["Decision Source"] = existing_df.apply(
                    lambda r: "Selected model"
                    if str(r.get("Decision")) == str(r.get("Prediction"))
                    and pd.to_numeric(pd.Series([r.get("Confidence")]), errors="coerce").iloc[0] >= float(selected_confidence_threshold)
                    else "Top-N ensemble",
                    axis=1,
                )
                existing_df["Decision Vote %"] = existing_df["Filename"].map(ensemble_vote_pct_by_filename)
                existing_df["Decision Correct"] = existing_df.apply(
                    lambda r: _correct_state(r.get("Decision"), r.get("Ground Truth")),
                    axis=1,
                )

                selected_metrics = compute_inference_metrics(existing_df)
                metric_cols = st.columns(4)
                metric_cols[0].metric("ACC", fmt_metric(selected_metrics["ACC"]))
                metric_cols[1].metric("MCC", fmt_metric(selected_metrics["MCC"]))
                metric_cols[2].metric("AUC", fmt_metric(selected_metrics["AUC"]))
                metric_cols[3].metric("N analyzed", selected_metrics["N"])

                per_class_metric_rows = []
                for metric_name, metric_value in selected_metrics.items():
                    if not metric_name.startswith(("F1 ", "Recall ", "Precision ", "Support ")):
                        continue
                    metric_type, class_label = metric_name.split(" ", 1)
                    per_class_metric_rows.append(
                        {
                            "Class": class_label,
                            "Metric": metric_type,
                            "Score": metric_value if metric_type == "Support" else fmt_metric(metric_value),
                        }
                    )
                if per_class_metric_rows:
                    per_class_metrics_df = pd.DataFrame(per_class_metric_rows)
                    per_class_metrics_df = per_class_metrics_df.pivot(
                        index="Class",
                        columns="Metric",
                        values="Score",
                    ).reset_index()
                    st.dataframe(_arrow_safe_dataframe(per_class_metrics_df), use_container_width=True)

                _show_decision_metrics(
                    "Selected model with threshold/fallback metrics",
                    existing_df,
                    "Decision",
                )
                _plot_one_vs_rest_roc(existing_df, "Selected model inference ROC curves")
                _render_threshold_heatmap(existing_df, require_both_thresholds)

                existing_df['Confidence'] = existing_df['Confidence'].apply(fmt_confidence)
                existing_df["Decision Vote %"] = existing_df["Decision Vote %"].apply(
                    lambda v: "" if pd.isna(v) else f"{float(v):.1f}"
                )
                existing_df["Ensemble Raw Vote %"] = existing_df["Ensemble Raw Vote %"].apply(
                    lambda v: "" if pd.isna(v) else f"{float(v):.1f}"
                )
                existing_display_df = existing_df[
                    [
                        'Filename', 'Head', 'Head Config', 'Ground Truth',
                        'Selected Model Prediction', 'Selected Model Correct',
                        'Ensemble Prediction', 'Ensemble Correct',
                        'Ensemble Raw Prediction', 'Ensemble Raw Vote %',
                        *score_cols, 'Confidence',
                        'Decision', 'Decision Correct', 'Decision Source',
                        'Decision Vote %', 'Top-N Correct %', 'Timestamp', 'Log Path',
                    ]
                ]

                st.divider()
                st.subheader("🖼️ Grad-CAM Generation")
                force_gradcam_all = st.checkbox(
                    "Regenerate Grad-CAM even when files already exist",
                    value=False,
                    key="inference_force_gradcam_all",
                )
                if st.button("Compute Grad-CAM for all existing results", key="inference_gradcam_all"):
                    log_path_by_filename = existing_df.set_index("Filename")["Log Path"].to_dict()
                    gradcam_result = _generate_gradcams_for_files(
                        existing_df["Filename"].tolist(),
                        log_path_by_filename=log_path_by_filename,
                        skip_existing=not force_gradcam_all,
                    )
                    st.success(
                        f"Grad-CAM generated for {gradcam_result['generated']} image(s); "
                        f"skipped {gradcam_result['skipped']} existing image(s)."
                    )
                    if gradcam_result["errors"]:
                        with st.expander("Grad-CAM errors"):
                            st.write("\n".join(gradcam_result["errors"]))

                st.dataframe(
                    existing_display_df.drop(columns=['Log Path'], errors='ignore'),
                    use_container_width=True,
                )

                st.divider()
                st.subheader("🔎 Observe One Inference Image")
                observe_options = existing_display_df['Filename'].tolist()
                selected_observe_filename = st.selectbox(
                    "Select an inference image to inspect",
                    options=observe_options,
                    key="inference_observe_image_select",
                )
                observe_row = existing_display_df[existing_display_df['Filename'] == selected_observe_filename].iloc[0]
                observe_img_path = os.path.join(inference_dir, selected_observe_filename)

                obs_cols = st.columns([1, 1])
                with obs_cols[0]:
                    if os.path.exists(observe_img_path):
                        st.image(observe_img_path, caption=selected_observe_filename, use_container_width=True)
                    else:
                        st.warning(f"Image file not found: {observe_img_path}")
                with obs_cols[1]:
                    st.markdown(f"**Decision:** {observe_row.get('Decision', 'Unknown')}")
                    st.markdown(f"**Selected model:** {observe_row.get('Selected Model Prediction', 'Unknown')}")
                    st.markdown(f"**Top-N ensemble:** {observe_row.get('Ensemble Prediction', 'Unknown')}")
                    st.markdown(f"**Confidence:** {observe_row['Confidence']}")
                    st.markdown(f"**Ground Truth:** {observe_row['Ground Truth']}")
                    observe_score_cols = [c for c in observe_row.index if str(c).startswith('Score ')]
                    if observe_score_cols:
                        observe_score_df = pd.DataFrame(
                            [{
                                'Class': str(c)[len('Score '):],
                                'Score': observe_row[c],
                            } for c in observe_score_cols]
                        )
                        st.dataframe(_arrow_safe_dataframe(observe_score_df), use_container_width=True, hide_index=True)
                    correct_value = str(observe_row.get('Decision Correct', '')).strip()
                    if correct_value == "✓":
                        st.success("Decision correct")
                    elif correct_value == "X":
                        st.info("Decision: Unknown / removed by threshold")
                    else:
                        st.error("Decision incorrect")

                st.markdown("**Grad-CAM composite explanations: last 4 layers**")
                force_gradcam_one = st.checkbox(
                    "Regenerate this image's Grad-CAM",
                    value=False,
                    key="inference_force_gradcam_one",
                )
                if st.button("Compute Grad-CAM for this image", key="inference_gradcam_one"):
                    gradcam_result = _generate_gradcams_for_files(
                        [selected_observe_filename],
                        log_path_by_filename={selected_observe_filename: observe_row.get('Log Path')},
                        skip_existing=not force_gradcam_one,
                    )
                    if gradcam_result["generated"]:
                        st.success("Grad-CAM generated for this image.")
                    elif gradcam_result["skipped"]:
                        st.info("Grad-CAM already exists for this image.")
                    if gradcam_result["errors"]:
                        st.warning("; ".join(gradcam_result["errors"]))

                gradcam_images = find_inference_gradcam_images(observe_row.get('Log Path'), selected_observe_filename, n_layers=4)
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
