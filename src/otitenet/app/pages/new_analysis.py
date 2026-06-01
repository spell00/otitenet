
# /home/simon/otitenet/otitenet/app/pages/new_analysis.py

from __future__ import annotations

import copy
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from otitenet.app.analysis import run_analysis_on_file
from otitenet.app.display_metrics import _arrow_safe_dataframe
from otitenet.app.utils import (
    format_classifier_config,
    parse_classifier_config,
    resolve_best_classifier_config,
    strip_extension,
)


IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def _fmt_confidence(value) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return ""


def _normalize_analysis_result(result):
    """
    Accept old/new run_analysis_on_file return shapes.

    Known shapes:
    - (pred_label, confidence, log_path, existing)
    - (pred_label, confidence, log_path, existing, gradcam_path)
    - variants with extra values appended
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
    class_scores = result[5] if len(result) > 5 and isinstance(result[5], dict) else {}

    return pred_label, confidence, complete_log_path, existing, gradcam_path, class_scores


def _clone_args(args):
    try:
        return copy.copy(args)
    except Exception:
        return args


def _current_head_config(args):
    """
    Resolve the active learned-embedding classifier head.

    Priority:
    1. Explicit sidebar selection
    2. args.best_classifier_config
    3. best cached/optimized head
    4. args.n_neighbors fallback
    """
    explicit = st.session_state.get("sidebar_classification_head_config")
    explicit_model_key = st.session_state.get("sidebar_classification_head_model_key")
    selected_model_key = st.session_state.get("selected_model_selection_key")
    explicit_matches_model = (
        explicit_model_key is not None
        and selected_model_key is not None
        and explicit_model_key == selected_model_key
    )
    if explicit_matches_model and explicit is not None and str(explicit).strip() not in {"", "None", "nan"}:
        return str(explicit)

    existing = getattr(args, "best_classifier_config", None)
    if existing is not None and str(existing).strip() not in {"", "None", "nan"}:
        return str(existing)

    siamese_inference = str(getattr(args, "siamese_inference", "linearsvc")).strip().lower()
    if siamese_inference == "linearsvc":
        return "baseline_linear_svc"
    if siamese_inference == "logisticregression":
        return "baseline_logreg"

    try:
        return str(resolve_best_classifier_config(args, use_optimized=True))
    except Exception:
        return str(getattr(args, "n_neighbors", 1))


def _set_head_on_args(args, head_config):
    if head_config is None:
        return args

    args.best_classifier_config = str(head_config)
    args.learned_classifier_label = format_classifier_config(head_config)
    head_meta = parse_classifier_config(head_config)
    args.classification_head_family = head_meta.get("family")

    if head_meta.get("family") == "knn":
        args.n_neighbors = int(head_meta["k"])
    elif head_meta.get("family") == "prototype":
        args.prototypes_to_use = "class"
        args.prototype_strategy = str(head_meta.get("strategy", "mean"))
        args.prototype_components = int(head_meta.get("components", 1))
    elif head_meta.get("family") == "baseline":
        baseline_name = str(head_meta.get("name", ""))
        if baseline_name == "linear_svc":
            args.siamese_inference = "linearsvc"
        elif baseline_name in {"logreg", "logistic_regression"}:
            args.siamese_inference = "logisticregression"

    return args


def _prepare_analysis_args(ctx, infer_method: str, dist_metric: str):
    args = _clone_args(ctx.args)

    args.infer_method = infer_method
    args.dist_metric = dist_metric

    head_config = _current_head_config(args)
    _set_head_on_args(args, head_config)

    return args, head_config


def _delete_existing_results_for_files(ctx, filenames: List[str]) -> None:
    """
    Remove existing rows for the current person/model/files before a forced analysis.

    run_analysis_on_file usually handles force_reanalyze, but this prevents old
    duplicate rows if old schemas/logic allowed duplicates.
    """
    if not filenames:
        return

    person_id = st.session_state.get("person_id") or getattr(ctx, "selected_person_id", None)
    model_id = getattr(ctx.args, "model_id", None) or st.session_state.get("selected_model_params", {}).get("model_id")

    if person_id is None or model_id is None:
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


def _find_gradcam_images(log_path: str, filename: str, n_layers: int = 4) -> List[str]:
    """
    Find Grad-CAM composite images generated by analysis.py.

    Expected layout:
      <log_path>/<base>/<base>_grad_cam_all_classes_layerX.png
    """
    if not log_path or not filename:
        return []

    import glob
    import re

    base = strip_extension(os.path.basename(str(filename)))
    root = os.path.join(str(log_path), base)

    patterns = [
        os.path.join(root, f"{base}_grad_cam_all_classes_layer*.png"),
        os.path.join(root, f"*{base}*grad*cam*layer*.png"),
        os.path.join(str(log_path), "**", f"{base}_grad_cam_all_classes_layer*.png"),
    ]

    found = []
    for pattern in patterns:
        try:
            found.extend(glob.glob(pattern, recursive=True))
        except Exception:
            pass

    found = [p for p in found if os.path.isfile(p)]
    found = list(dict.fromkeys(found))

    def layer_key(path):
        match = re.search(r"layer(\d+)", os.path.basename(str(path)).lower())
        return int(match.group(1)) if match else -1

    found = sorted(found, key=layer_key, reverse=True)
    return found[: int(n_layers)]


def _sync_uploaded_image_store(uploaded_files) -> List[Dict[str, Any]]:
    """
    Persist uploaded image bytes independently of task/model selection.

    Streamlit widget values can be invalidated by reruns triggered from other
    controls. Keeping a normalized byte store lets the same imported images be
    analyzed against multiple tasks/models without another upload.
    """
    store_key = "new_analysis_imported_images"

    if not uploaded_files:
        return list(st.session_state.get(store_key, []))

    stored_by_name = {
        str(item.get("Filename")): item
        for item in st.session_state.get(store_key, [])
        if isinstance(item, dict) and item.get("Filename")
    }
    for file_obj in uploaded_files:
        try:
            file_bytes = file_obj.getvalue()
        except Exception:
            pos = None
            try:
                pos = file_obj.tell()
            except Exception:
                pass
            file_bytes = file_obj.read()
            if pos is not None:
                try:
                    file_obj.seek(pos)
                except Exception:
                    pass

        filename = os.path.basename(file_obj.name)
        stored_by_name[filename] = {
            "Filename": filename,
            "Bytes": file_bytes,
            "Size": len(file_bytes) if file_bytes is not None else 0,
            "Type": getattr(file_obj, "type", None),
        }

    st.session_state[store_key] = list(stored_by_name.values())

    return list(st.session_state.get(store_key, []))


def _select_images_for_current_run(uploaded_images: List[Dict[str, Any]], is_admin: bool):
    if not uploaded_images:
        return [], {}

    if not is_admin:
        return uploaded_images, {
            "mode": "all",
            "start": 0,
            "end": len(uploaded_images),
            "total": len(uploaded_images),
        }

    st.subheader("Batch run")
    mode = st.radio(
        "Files to process",
        options=["All uploaded files", "Next batch"],
        horizontal=True,
        key="new_analysis_batch_mode",
    )

    total = len(uploaded_images)
    if mode == "All uploaded files":
        st.caption(f"This run will process all {total} uploaded file(s).")
        return uploaded_images, {
            "mode": "all",
            "start": 0,
            "end": total,
            "total": total,
        }

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        batch_size = int(
            st.number_input(
                "Batch size",
                min_value=1,
                max_value=max(total, 1),
                value=min(10, total),
                step=1,
                key="new_analysis_batch_size",
            )
        )

    current_start = int(st.session_state.get("new_analysis_batch_start", 0))
    current_start = min(max(current_start, 0), total)

    if current_start >= total:
        with c2:
            st.caption("All uploaded files have been covered by next-batch runs.")
        with c3:
            if st.button("Reset batch position", key="new_analysis_reset_batch_position"):
                st.session_state["new_analysis_batch_start"] = 0
                st.rerun()
        return [], {
            "mode": "next",
            "start": current_start,
            "end": current_start,
            "total": total,
        }

    with c2:
        start_1_based = int(
            st.number_input(
                "Start at",
                min_value=1,
                max_value=max(total, 1),
                value=current_start + 1,
                step=1,
                key=f"new_analysis_batch_start_input_{current_start}",
            )
        )

    start = min(max(start_1_based - 1, 0), max(total - 1, 0))
    end = min(start + batch_size, total)
    st.session_state["new_analysis_batch_start"] = start

    with c3:
        st.caption(f"This run will process files {start + 1}-{end} of {total}.")
        if st.button("Reset batch position", key="new_analysis_reset_batch_position"):
            st.session_state["new_analysis_batch_start"] = 0
            st.rerun()

    return uploaded_images[start:end], {
        "mode": "next",
        "start": start,
        "end": end,
        "total": total,
    }


def _render_uploaded_previews(uploaded_images) -> None:
    if not uploaded_images:
        return

    with st.expander("Uploaded image preview", expanded=len(uploaded_images) <= 3):
        cols_per_row = 3
        for start in range(0, len(uploaded_images), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, image_info in zip(cols, uploaded_images[start : start + cols_per_row]):
                with col:
                    filename = image_info.get("Filename") if isinstance(image_info, dict) else getattr(image_info, "name", "")
                    image_data = image_info.get("Bytes") if isinstance(image_info, dict) else image_info
                    try:
                        st.image(image_data, caption=filename, use_container_width=True)
                    except Exception:
                        st.caption(filename)


def _render_results_table(results: List[Dict[str, Any]]) -> None:
    if not results:
        return

    df = pd.DataFrame(results)

    score_cols = sorted([c for c in df.columns if c.startswith("Score ")])
    display_cols = [
        "Filename",
        "Prediction",
        "Confidence",
        *score_cols,
        "Existing",
        "Grad-CAM",
        "Log Path",
        "Elapsed (s)",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    table = df[display_cols].copy()

    if "Confidence" in table.columns:
        table["Confidence"] = table["Confidence"].apply(_fmt_confidence)
    for col in score_cols:
        if col in table.columns:
            table[col] = table[col].apply(_fmt_confidence)
    if "Existing" in table.columns:
        table["Existing"] = table["Existing"].apply(bool)

    st.subheader("Analysis results")
    st.dataframe(_arrow_safe_dataframe(table.drop(columns=["Log Path"], errors="ignore")), use_container_width=True)


def _render_single_result_observer(results: List[Dict[str, Any]], n_layers: int = 4) -> None:
    if not results:
        return

    st.subheader("Observe result")

    options = list(range(len(results)))

    def option_label(i):
        r = results[i]
        return f"{r.get('Filename')} | pred={r.get('Prediction')} | conf={_fmt_confidence(r.get('Confidence'))}"

    selected = st.selectbox(
        "Select analyzed image",
        options=options,
        format_func=option_label,
        key="new_analysis_observe_result",
    )

    row = results[selected]

    c1, c2 = st.columns([1, 1])

    with c1:
        file_bytes = row.get("Bytes")
        if file_bytes:
            st.image(file_bytes, caption=f"Input: {row.get('Filename')}", use_container_width=True)

    with c2:
        st.markdown(f"**Prediction:** {row.get('Prediction')}")
        st.markdown(f"**Confidence:** {_fmt_confidence(row.get('Confidence'))}")
        score_items = [
            (str(k[len("Score "):]), row.get(k))
            for k in sorted(row.keys())
            if str(k).startswith("Score ")
        ]
        if score_items:
            scores_df = pd.DataFrame(
                [{"Class": label, "Score": _fmt_confidence(score)} for label, score in score_items]
            )
            st.dataframe(scores_df, hide_index=True, use_container_width=True)
        st.markdown(f"**Existing cached result:** {row.get('Existing')}")
        if row.get("Log Path"):
            with st.expander("Log path"):
                st.code(str(row.get("Log Path")))

    gradcam_images = _find_gradcam_images(row.get("Log Path"), row.get("Filename"), n_layers=n_layers)

    if gradcam_images:
        st.markdown("**Grad-CAM**")
        for path in gradcam_images:
            st.image(path, caption=os.path.basename(path), use_container_width=True)
    elif row.get("Grad-CAM Path"):
        st.info(f"Grad-CAM path returned: {row.get('Grad-CAM Path')}")
    else:
        st.info(
            "No Grad-CAM montage found yet. Fresh predictions should generate Grad-CAM automatically. "
            "Try Force New Analysis if this result was loaded from cache."
        )


def _render_current_model_summary(args, head_config, is_admin: bool) -> None:
    st.subheader("Current analysis model")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"**Model:** {getattr(args, 'model_name', '—')}")
        st.markdown(f"**Size:** {getattr(args, 'new_size', '—')}")
        st.markdown(f"**Normalize:** {getattr(args, 'normalize', '—')}")

    with c2:
        st.markdown(f"**Loss:** {getattr(args, 'classif_loss', '—')}")
        st.markdown(f"**DLoss:** {getattr(args, 'dloss', '—')}")
        st.markdown(f"**Distance:** {getattr(args, 'dist_fct', '—')}")

    with c3:
        st.markdown(f"**Head:** {format_classifier_config(head_config)}")
        st.markdown(f"**n_neighbors:** {getattr(args, 'n_neighbors', '—')}")
        st.markdown(f"**Mode:** {'admin-selected model' if is_admin else 'production model'}")


def render(ctx):
    st.header("🧪 New Analysis")

    selected_person_id = (
        st.session_state.get("person_id")
        or getattr(ctx, "selected_person_id", None)
    )

    if selected_person_id is None:
        st.warning("Please create or select a family member from the sidebar before running a new analysis.")
        return

    st.session_state["person_id"] = selected_person_id

    is_admin = bool(getattr(ctx, "is_admin", False))

    if not is_admin and not st.session_state.get("production_model"):
        st.warning("No production model is currently selected by the admin.")
        return

    upload_key_version = int(st.session_state.get("new_analysis_upload_key_version", 0))
    upload_key = f"new_analysis_uploaded_files_{upload_key_version}"

    uploaded_files = st.file_uploader(
        "Upload one or more otoscope images",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key=upload_key,
    )
    uploaded_images = _sync_uploaded_image_store(uploaded_files)

    if uploaded_images:
        if st.button("Clear uploaded images", key="new_analysis_clear_uploads"):
            st.session_state["new_analysis_imported_images"] = []
            st.session_state["new_analysis_last_results"] = []
            st.session_state["new_analysis_batch_start"] = 0
            st.session_state["new_analysis_upload_key_version"] = upload_key_version + 1
            st.rerun()
        _render_uploaded_previews(uploaded_images)

    st.divider()

    if is_admin:
        st.subheader("Inference settings")

        infer_method = st.selectbox(
            "Inference Method",
            options=["majority_vote", "prototypes", "prototype_distance"],
            index=0,
            help=(
                "majority_vote: learned classifier / KNN-style output. "
                "prototypes: direct prototype distance. "
                "prototype_distance: inverse distance ratio."
            ),
            key="new_analysis_infer_method",
        )

        if infer_method == "prototype_distance":
            dist_metric = st.selectbox(
                "Distance Metric",
                options=["euclidean", "cosine"],
                index=0,
                key="new_analysis_dist_metric",
            )
        else:
            dist_metric = "euclidean"

        c_speed1, c_speed2, c_speed3 = st.columns(3)

        with c_speed1:
            skip_validation = st.checkbox(
                "⚡ Skip validation metrics",
                value=False,
                help="Skip loading validation metrics to speed up inference.",
                key="new_analysis_skip_validation",
            )

        with c_speed2:
            fast_infer = st.checkbox(
                "⚡ Fast inference",
                value=False,
                help="Use faster inference path where supported.",
                key="new_analysis_fast_infer",
            )

        with c_speed3:
            n_gradcam_layers = st.number_input(
                "Grad-CAM layers to display",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                key="new_analysis_gradcam_layers",
            )

        force_reanalyze = st.checkbox(
            "🔄 Force New Analysis",
            value=False,
            key="new_analysis_force_reanalyze",
        )

    else:
        infer_method = "majority_vote"
        dist_metric = "euclidean"
        skip_validation = True
        fast_infer = False
        n_gradcam_layers = 4
        force_reanalyze = False

        st.caption(
            "Client mode uses the admin-selected production model. "
            "If the production model changed, a new result is computed; otherwise the saved result may be loaded."
        )

    analysis_args, head_config = _prepare_analysis_args(ctx, infer_method, dist_metric)
    _render_current_model_summary(analysis_args, head_config, is_admin=is_admin)

    images_for_run, run_batch = _select_images_for_current_run(uploaded_images, is_admin=is_admin)

    if uploaded_images and len(uploaded_images) > 1:
        st.info(
            f"{len(uploaded_images)} uploaded file(s) are stored. "
            f"{len(images_for_run)} file(s) are queued for the next run."
        )

    st.divider()

    run_clicked = st.button(
        "▶️ Run Analysis",
        key="new_analysis_run",
        disabled=not bool(images_for_run),
        type="primary",
    )

    if not run_clicked:
        previous_results = st.session_state.get("new_analysis_last_results", [])
        if previous_results:
            _render_results_table(previous_results)
            _render_single_result_observer(previous_results, n_layers=int(n_gradcam_layers))
        return

    filenames = [os.path.basename(f["Filename"]) for f in images_for_run]

    if force_reanalyze:
        _delete_existing_results_for_files(ctx, filenames)

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    results = []
    start_time = time.time()

    run_total = len(images_for_run)

    for idx, image_info in enumerate(images_for_run):
        file_start = time.time()
        filename = os.path.basename(image_info["Filename"])
        display_idx = int(run_batch.get("start", 0)) + idx + 1

        status_text.write(
            f"Processing run item {idx + 1}/{run_total} "
            f"(uploaded file {display_idx}/{run_batch.get('total', run_total)}): {filename}"
        )
        progress_bar.progress(idx / max(run_total, 1))

        try:
            file_bytes = image_info["Bytes"]

            result = run_analysis_on_file(
                filename,
                file_bytes,
                analysis_args,
                ctx.cursor,
                ctx.conn,
                force_reanalyze=force_reanalyze,
                show_validation_metrics=not skip_validation,
                fast_infer=fast_infer,
                quiet=True,
            )

            pred_label, confidence, complete_log_path, existing, gradcam_path, class_scores = _normalize_analysis_result(result)

            gradcam_images = _find_gradcam_images(
                complete_log_path,
                filename,
                n_layers=int(n_gradcam_layers),
            )

            row = {
                "Filename": filename,
                "Prediction": pred_label,
                "Confidence": confidence,
                "Existing": bool(existing),
                "Log Path": complete_log_path,
                "Grad-CAM Path": gradcam_path,
                "Grad-CAM": len(gradcam_images),
                "Elapsed (s)": round(time.time() - file_start, 2),
                "Bytes": file_bytes,
            }
            for label, score in (class_scores or {}).items():
                row[f"Score {label}"] = score
            results.append(row)

            st.session_state["last_uploaded_name"] = filename
            st.session_state["last_complete_log_path"] = complete_log_path
            st.session_state["last_prediction"] = pred_label
            st.session_state["last_confidence"] = confidence

        except Exception as e:
            results.append(
                {
                    "Filename": filename,
                    "Prediction": "ERROR",
                    "Confidence": None,
                    "Existing": False,
                    "Log Path": None,
                    "Grad-CAM": 0,
                    "Elapsed (s)": round(time.time() - file_start, 2),
                    "Error": str(e),
                    "Bytes": None,
                }
            )

            with st.expander(f"Error details for {filename}"):
                st.code(traceback.format_exc())

        progress_bar.progress((idx + 1) / max(run_total, 1))

    status_text.success(f"Finished {run_total} file(s) in {time.time() - start_time:.1f}s.")

    st.session_state["new_analysis_last_results"] = results

    if run_batch.get("mode") == "next":
        st.session_state["new_analysis_batch_start"] = min(
            int(run_batch.get("end", 0)),
            int(run_batch.get("total", 0)),
        )

    _render_results_table(results)

    error_rows = [r for r in results if r.get("Prediction") == "ERROR"]
    if error_rows:
        st.warning(f"{len(error_rows)} file(s) failed.")

    st.divider()
    _render_single_result_observer(results, n_layers=int(n_gradcam_layers))
