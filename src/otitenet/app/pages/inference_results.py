"""Inference Results page."""

import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

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
                df["Ground Truth"] = df["Filename"].apply(lambda filename: inference_ground_truth(filename, inference_dir))
                df["Correct"] = df.apply(lambda r: labels_match(r["Prediction"], r["Ground Truth"]), axis=1)
                return df

            top_n_existing_df = _fetch_latest_inference_results_for_models(top_n_model_ids)
            performance_rows = []
            if not top_n_model_df.empty:
                for _, perf_model_row in top_n_model_df.iterrows():
                    perf_model_id = perf_model_row.get("Model ID")
                    model_results_df = top_n_existing_df[top_n_existing_df["Model ID"] == perf_model_id].copy() if not top_n_existing_df.empty else pd.DataFrame()
                    metrics = compute_inference_metrics(model_results_df)
                    rank_value = perf_model_row.get("#", model_number_map.get(perf_model_row.get("_selection_key"), "?"))
                    try:
                        _tmp_perf_args = args_from_inference_row(args, perf_model_row.to_dict())
                        best_head_entry = _best_head_entry_for_args_global(_tmp_perf_args)
                        head_config = str(best_head_entry.get("config", ""))
                        head_label = best_head_entry.get("label") or _head_config_label_global(head_config)
                        head_score = best_head_entry.get("mcc", np.nan)
                    except Exception:
                        head_config = ""
                        head_label = "—"
                        head_score = np.nan
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

                existing_df['Confidence'] = existing_df['Confidence'].apply(fmt_confidence)
                existing_df = existing_df[
                    ['Filename', 'Head', 'Head Config', 'Ground Truth', 'Prediction', *score_cols, 'Correct', 'Confidence', 'Top-N Correct %', 'Timestamp', 'Log Path']
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
                    observe_score_cols = [c for c in observe_row.index if str(c).startswith('Score ')]
                    if observe_score_cols:
                        observe_score_df = pd.DataFrame(
                            [{
                                'Class': str(c)[len('Score '):],
                                'Score': observe_row[c],
                            } for c in observe_score_cols]
                        )
                        st.dataframe(_arrow_safe_dataframe(observe_score_df), use_container_width=True, hide_index=True)
                    correct_value = observe_row['Correct']
                    if correct_value is True:
                        st.success("Correct: True")
                    elif correct_value is False:
                        st.error("Correct: False")
                    else:
                        st.info("Correct: Unknown")

                gradcam_images = find_inference_gradcam_images(observe_row.get('Log Path'), selected_observe_filename, n_layers=4)
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
