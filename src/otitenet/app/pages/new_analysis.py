
# /home/simon/otitenet/otitenet/app/pages/new_analysis.py

from __future__ import annotations

import copy
import os
import time
import traceback
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from otitenet.app.analysis import run_analysis_on_file
from otitenet.app.display_metrics import _arrow_safe_dataframe, _best_head_config_for_args_global
from otitenet.app.image_processing import infer_base_resize_size, preprocessing_trace
from otitenet.app.services.inference_results_service import args_from_inference_row
from otitenet.app.services.production_model_service import apply_production_model_to_args
from otitenet.app.utils import (
    _ensure_model_number_map,
    _make_model_selection_key,
    enumerate_classification_heads,
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


def _clean_model_value(value):
    try:
        if value is None or pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _same_model_identity(left: dict | None, right: dict | None) -> bool:
    left = left or {}
    right = right or {}

    def _first(row, *keys):
        for key in keys:
            value = _clean_model_value(row.get(key))
            if value:
                return value
        return ""

    left_id = _first(left, "Registry ID", "DB Model ID", "registry_id", "model_id", "Model ID", "id")
    right_id = _first(right, "Registry ID", "DB Model ID", "registry_id", "model_id", "Model ID", "id")
    if left_id and right_id:
        try:
            return int(float(left_id)) == int(float(right_id))
        except Exception:
            return left_id == right_id

    left_path = _first(left, "Log Path", "log_path", "Artifact Log Path", "Best Model Dir", "Source Run Path")
    right_path = _first(right, "Log Path", "log_path", "Artifact Log Path", "Best Model Dir", "Source Run Path")
    return bool(left_path and right_path and left_path.rstrip("/") == right_path.rstrip("/"))


def _selected_model_matches_production(production_model: dict | None) -> bool:
    selected = st.session_state.get("selected_model_params") or {}
    if not _same_model_identity(selected, production_model):
        return False

    selected_head = st.session_state.get("sidebar_classification_head_config")
    production_head = (
        (production_model or {}).get("Best Head Config")
        or (production_model or {}).get("Head Config")
        or (production_model or {}).get("best_classifier_config")
        or (production_model or {}).get("classification_head_config")
        or (production_model or {}).get("head_config")
    )
    if selected_head and production_head:
        return str(selected_head) == str(production_head)
    return True


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


def _set_head_on_args(args, head_config, selected_n_aug=None, use_sidebar_n_aug: bool = True):
    if head_config is None:
        return args

    args.best_classifier_config = str(head_config)
    args.learned_classifier_label = format_classifier_config(head_config)
    if selected_n_aug is not None:
        try:
            args.n_aug = int(float(str(selected_n_aug)))
        except Exception:
            args.n_aug = selected_n_aug
    else:
        explicit_n_aug = st.session_state.get("sidebar_classification_head_n_aug")
        explicit_model_key = st.session_state.get("sidebar_classification_head_model_key")
        selected_model_key = st.session_state.get("selected_model_selection_key")
        if (
            use_sidebar_n_aug
            and explicit_n_aug is not None
            and explicit_model_key is not None
            and selected_model_key is not None
            and explicit_model_key == selected_model_key
        ):
            try:
                args.n_aug = int(float(str(explicit_n_aug)))
            except Exception:
                args.n_aug = explicit_n_aug
        else:
            try:
                heads = enumerate_classification_heads(args)
                matching_head = next((h for h in heads if str(h.get("config")) == str(head_config)), None)
                if matching_head is not None and matching_head.get("n_aug") is not None:
                    args.n_aug = int(float(str(matching_head.get("n_aug"))))
            except Exception:
                pass
    head_meta = parse_classifier_config(head_config)
    args.classification_head_family = head_meta.get("family")

    if head_meta.get("family") == "knn":
        args.n_neighbors = int(head_meta["k"])
    elif head_meta.get("family") == "prototype":
        args.prototypes_to_use = "class"
        args.prototype_strategy = str(head_meta.get("strategy", "mean"))
        args.prototype_components = int(head_meta.get("components", 1))
        args.n_neighbors = 1
    elif head_meta.get("family") == "baseline":
        baseline_name = str(head_meta.get("name", ""))
        if baseline_name == "linear_svc":
            args.siamese_inference = "linearsvc"
        elif baseline_name in {"logreg", "logistic_regression"}:
            args.siamese_inference = "logisticregression"

    return args


def _prepare_analysis_args(
    ctx,
    is_admin: bool = False,
    use_production_model: bool = False,
):
    """Prepare the exact args object used by New Analysis.

    Important admin-mode rule:
    - "Selected model" must use the sidebar-selected args only.
    - "Production model" normally uses the published production metadata.
    - If the admin-selected sidebar model is the same model/head as production,
      production analysis reuses the sidebar runtime args so it gives the same
      prediction as "Selected model". This avoids stale production metadata
      such as an old/wrong distance function, n_aug, or classifier-head field
      changing the result for the same model.

    Client / non-admin mode still uses production metadata because there is no
    trusted sidebar-selected model for clients.
    """
    selected_args = _clone_args(ctx.args)
    args = _clone_args(ctx.args)
    production_model = getattr(ctx, "production_model", None) or st.session_state.get("production_model")

    selected_matches_production = bool(
        is_admin
        and production_model
        and _selected_model_matches_production(production_model)
    )

    use_production_metadata = False

    if not is_admin:
        # Client mode: always use the admin-published production model metadata.
        use_production_metadata = True
        args = apply_production_model_to_args(
            args,
            production_model,
            getattr(ctx, "data_dir", "./data"),
        )
        args._analysis_model_source_label = "production model"

    elif use_production_model and selected_matches_production:
        # Admin comparison mode: if production and selected are the same model,
        # make production use the same runtime settings as the selected path.
        # This makes the comparison deterministic and prevents stale production
        # metadata from changing the distance function / head behavior.
        args = selected_args
        args._analysis_model_source_label = "production model (same as selected; sidebar runtime settings)"
        args._analysis_production_matches_selected = True

    elif use_production_model:
        # Admin explicitly selected production, and the sidebar model is not the
        # same production model. Use the published production metadata.
        use_production_metadata = True
        args = apply_production_model_to_args(
            args,
            production_model,
            getattr(ctx, "data_dir", "./data"),
        )
        args._analysis_model_source_label = "production model"

    else:
        # Admin explicitly selected the sidebar model. Do not inject production
        # metadata, even if the model identity happens to match production.
        args._analysis_model_source_label = "admin-selected model"

    args._analysis_uses_production_metadata = bool(use_production_metadata)

    if use_production_metadata and getattr(args, "best_classifier_config", None) is not None:
        head_config = str(args.best_classifier_config)
    else:
        head_config = _current_head_config(args)

    _set_head_on_args(args, head_config)

    args.infer_method = "selected_model_head"

    # Keep both names synchronized. Some downstream code uses dist_fct and some
    # uses dist_metric. A mismatch here can produce different predictions for
    # the same model.
    dist_value = (
        getattr(args, "dist_fct", None)
        or getattr(args, "dist_metric", None)
        or "euclidean"
    )
    args.dist_fct = dist_value
    args.dist_metric = dist_value

    args.allow_inference_reencode = False

    return args, head_config

def _load_ranked_models(cursor, top_n: int) -> pd.DataFrame:
    _, best_models_table = _ensure_model_number_map(cursor)
    if best_models_table is None or best_models_table.empty:
        return pd.DataFrame()
    df = best_models_table.copy().reset_index(drop=True)
    if "#" not in df.columns:
        df["#"] = range(1, len(df) + 1)
    df["_selection_key"] = df.apply(lambda r: _make_model_selection_key(r.to_dict()), axis=1)
    return df.head(int(top_n)).copy()


def _prepare_model_args_from_row(ctx, row: Dict[str, Any]):
    args = args_from_inference_row(ctx.args, row)
    row_n_aug = row.get("Head N Aug", row.get("N Aug", row.get("N_Aug")))
    if row_n_aug is not None and str(row_n_aug).strip().lower() not in {"", "none", "nan", "null", "—"}:
        try:
            args.n_aug = int(float(str(row_n_aug)))
        except Exception:
            args.n_aug = row_n_aug
    head_config = row.get("Best Head Config") or row.get("Head Config") or _best_head_config_for_args_global(args)
    _set_head_on_args(args, str(head_config), selected_n_aug=row_n_aug, use_sidebar_n_aug=False)
    args.infer_method = "selected_model_head"
    args.dist_metric = getattr(args, "dist_fct", "euclidean")
    args.allow_inference_reencode = False
    return args, str(head_config)


def _vote_decision(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [r for r in rows if str(r.get("Prediction")) != "ERROR"]
    if not valid_rows:
        return {
            "Ensemble Prediction": "Unknown",
            "Ensemble Raw Prediction": "Unknown",
            "Ensemble Vote %": np.nan,
            "Models Voted": 0,
            "Votes": "",
        }
    counts = Counter(str(r.get("Prediction") or "Unknown") for r in valid_rows)
    consensus, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    vote_pct = 100.0 * float(count) / float(len(valid_rows)) if valid_rows else np.nan
    return {
        "Ensemble Prediction": consensus,
        "Ensemble Raw Prediction": consensus,
        "Ensemble Vote %": vote_pct,
        "Models Voted": len(valid_rows),
        "Votes": ", ".join(f"{label}:{n}" for label, n in counts.most_common()),
    }


def _threshold_decision(selected_prediction, selected_confidence, ensemble_prediction, ensemble_vote_pct):
    conf_threshold = float(st.session_state.get("production_selected_confidence_threshold", 0.0))
    vote_threshold = float(st.session_state.get("production_ensemble_vote_threshold_pct", 0.0))
    require_both = bool(st.session_state.get("production_require_both_thresholds", False))
    try:
        conf = float(selected_confidence)
    except Exception:
        conf = np.nan
    try:
        vote_pct = float(ensemble_vote_pct)
    except Exception:
        vote_pct = np.nan
    selected_passes = pd.notna(conf) and conf >= conf_threshold
    ensemble_passes = pd.notna(vote_pct) and vote_pct >= vote_threshold
    if require_both:
        return selected_prediction if selected_passes and ensemble_passes else "Unknown"
    if selected_passes:
        return selected_prediction
    return ensemble_prediction if ensemble_passes else "Unknown"


def _delete_existing_results_for_files(ctx, filenames: List[str], args=None) -> None:
    """
    Remove existing rows for the current person/model/files before a forced analysis.

    run_analysis_on_file usually handles force_reanalyze, but this prevents old
    duplicate rows if old schemas/logic allowed duplicates.
    """
    if not filenames:
        return

    person_id = st.session_state.get("person_id") or getattr(ctx, "selected_person_id", None)
    args = args or ctx.args
    model_id = getattr(args, "model_id", None) or st.session_state.get("selected_model_params", {}).get("model_id")
    head_config = (
        getattr(args, "best_classifier_config", None)
        or getattr(args, "classification_head_config", None)
        or getattr(args, "classifier_head_config", None)
        or getattr(args, "head_config", None)
        or ""
    )

    if person_id is None or model_id is None:
        return

    placeholders = ",".join(["%s"] * len(filenames))

    try:
        try:
            ctx.cursor.execute(
                f"""
                DELETE FROM results
                WHERE person_id=%s AND model_id=%s AND COALESCE(head_config, '')=%s AND filename IN ({placeholders})
                """,
                tuple([int(person_id), model_id, str(head_config)] + list(filenames)),
            )
        except Exception as exc:
            if "head_config" not in str(exc):
                raise
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


def _fmt_stats(stats: dict) -> str:
    return (
        f"shape={stats.get('shape')} | "
        f"min={float(stats.get('min', 0.0)):.4f} | "
        f"max={float(stats.get('max', 0.0)):.4f} | "
        f"mean={float(stats.get('mean', 0.0)):.4f} | "
        f"std={float(stats.get('std', 0.0)):.4f}"
    )


def _render_preprocessing_trace(uploaded_images, args) -> None:
    if not uploaded_images:
        return
    with st.expander("Preprocessing steps for selected image", expanded=False):
        labels = [str(item.get("Filename", f"image_{i+1}")) for i, item in enumerate(uploaded_images)]
        selected = st.selectbox(
            "Image",
            options=list(range(len(uploaded_images))),
            format_func=lambda i: labels[i],
            key="new_analysis_preprocess_trace_image",
        )
        image_info = uploaded_images[int(selected)]
        try:
            image = Image.open(BytesIO(image_info["Bytes"]))
            image.load()
            trace = preprocessing_trace(
                image,
                int(getattr(args, "new_size", 64)),
                normalize=getattr(args, "normalize", "yes"),
                base_size=infer_base_resize_size(getattr(args, "path", None)),
            )
        except Exception as exc:
            st.warning(f"Could not render preprocessing trace: {exc}")
            return

        st.caption(
            f"resize_mode={trace['resize_mode']} | normalize_mode={trace['normalize_mode']} | "
            f"base_size={trace.get('base_resize_size', '—')} | model_size={getattr(args, 'new_size', '—')}"
        )
        cols = st.columns(3)
        for col, key, title in [
            (cols[0], "raw", "Raw RGB"),
            (cols[1], "downsized", f"Downsized {trace.get('base_resize_size', '—')}x{trace.get('base_resize_size', '—')}"),
            (cols[2], "resized", "Final model resize"),
        ]:
            with col:
                st.image(trace[key]["image"], caption=f"{title} {trace[key]['size']}", use_container_width=True)
                st.caption(_fmt_stats(trace[key]["stats"]))
        st.markdown("**Tensor stats**")
        st.caption("Before normalize: " + _fmt_stats(trace["tensor_before_normalize"]))
        st.caption("After normalize: " + _fmt_stats(trace["tensor_after_normalize"]))


def _render_results_table(results: List[Dict[str, Any]]) -> None:
    if not results:
        return

    df = pd.DataFrame(results)

    score_cols = sorted([c for c in df.columns if c.startswith("Score ")])
    display_cols = [
        "Filename",
        "Decision",
        "Prediction",
        "Selected Model Prediction",
        "Selected Model Confidence",
        "Ensemble Prediction",
        "Ensemble Vote %",
        "Models Voted",
        "Votes",
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
    if "Selected Model Confidence" in table.columns:
        table["Selected Model Confidence"] = table["Selected Model Confidence"].apply(_fmt_confidence)
    if "Ensemble Vote %" in table.columns:
        table["Ensemble Vote %"] = table["Ensemble Vote %"].apply(
            lambda v: "" if pd.isna(v) else f"{float(v):.1f}"
        )
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
        return f"{r.get('Filename')} | decision={r.get('Decision', r.get('Prediction'))} | conf={_fmt_confidence(r.get('Confidence'))}"

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
        st.markdown(f"**Decision:** {row.get('Decision', row.get('Prediction'))}")
        if row.get("Selected Model Prediction") is not None:
            st.markdown(f"**Selected model:** {row.get('Selected Model Prediction')}")
        if row.get("Ensemble Prediction") is not None:
            st.markdown(f"**Top-N ensemble:** {row.get('Ensemble Prediction')}")
        st.markdown(f"**Confidence:** {_fmt_confidence(row.get('Confidence'))}")
        
        st.markdown("**Per-Class Confidence Scores:**")
        score_items = [
            (str(k[len("Score "):]), row.get(k))
            for k in sorted(row.keys())
            if str(k).startswith("Score ")
        ]
        if score_items:
            scores_df = pd.DataFrame(
                [{"Class": label, "Confidence": _fmt_confidence(score)} for label, score in score_items]
            )
            scores_df = scores_df.sort_values('Confidence', ascending=False)
            st.dataframe(scores_df, hide_index=True, use_container_width=True)
        else:
            st.info("No per-class confidence scores available. Recompute analysis to generate class scores.")
        
        st.markdown(f"**Existing cached result:** {row.get('Existing')}")
        if row.get("Log Path"):
            with st.expander("Log path"):
                st.code(str(row.get("Log Path")))
        per_model_rows = row.get("Per-Model Predictions")
        if per_model_rows:
            with st.expander("Per-model Top-N predictions", expanded=False):
                per_model_df = pd.DataFrame(per_model_rows)
                for conf_col in [c for c in ["Confidence"] if c in per_model_df.columns]:
                    per_model_df[conf_col] = per_model_df[conf_col].apply(_fmt_confidence)
                st.dataframe(
                    _arrow_safe_dataframe(per_model_df.drop(columns=["Log Path"], errors="ignore")),
                    hide_index=True,
                    use_container_width=True,
                )

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


def _render_current_model_summary(args, head_config, is_admin: bool, use_production_model: bool = False) -> None:
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
        st.markdown(f"**Head Config:** `{head_config}`")
        st.markdown(f"**N Aug:** {getattr(args, 'n_aug', '—')}")
        st.markdown(f"**n_neighbors:** {getattr(args, 'n_neighbors', '—')}")
        mode_label = getattr(args, "_analysis_model_source_label", None)
        if not mode_label:
            mode_label = "production model" if use_production_model or not is_admin else "admin-selected model"
        st.markdown(f"**Mode:** {mode_label}")


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
        has_production_model = bool(getattr(ctx, "production_model", None) or st.session_state.get("production_model"))
        model_source_options = ["Production model", "Selected model"] if has_production_model else ["Selected model"]
        default_model_source = "Production model" if has_production_model else "Selected model"
        model_source = st.radio(
            "Model source",
            options=model_source_options,
            index=model_source_options.index(default_model_source),
            horizontal=True,
            key="new_analysis_model_source",
            help="Production model uses the admin-published model/head. Selected model uses the current sidebar selection and its selected/best head.",
        )
        use_production_model = model_source == "Production model"
        fast_infer = False

        c_speed1, c_speed2 = st.columns(2)

        with c_speed1:
            skip_validation = st.checkbox(
                "⚡ Skip validation metrics",
                value=False,
                help="Skip loading validation metrics to speed up inference.",
                key="new_analysis_skip_validation",
            )

        with c_speed2:
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
        generate_gradcam = st.checkbox(
            "Generate Grad-CAM during run",
            value=False,
            key="new_analysis_generate_gradcam",
            help="Grad-CAM is slow. Leave off for faster predictions; compute explanations later from the result view/gallery.",
        )

    else:
        use_production_model = True
        skip_validation = True
        fast_infer = False
        n_gradcam_layers = 4
        force_reanalyze = False
        generate_gradcam = st.checkbox(
            "Generate Grad-CAM during run",
            value=False,
            key="new_analysis_client_generate_gradcam",
            help="Grad-CAM is slow. Leave off for faster predictions.",
        )

        st.caption(
            "Client mode uses the admin-selected production model. "
            "If the production model changed, a new result is computed; otherwise the saved result may be loaded."
        )

    use_topn_ensemble = bool(st.session_state.get("production_use_topn_ensemble", False))
    top_n_models = int(st.session_state.get("production_top_n_models", 5))
    if is_admin:
        with st.expander("Production decision thresholds", expanded=False):
            use_topn_ensemble = st.checkbox(
                "Use Top-N ensemble for this analysis",
                value=use_topn_ensemble,
                key="new_analysis_use_topn_ensemble",
            )
            top_n_models = st.number_input(
                "Top-N models to run",
                min_value=1,
                max_value=25,
                value=top_n_models,
                step=1,
                key="new_analysis_top_n_models",
            )
            st.session_state["production_use_topn_ensemble"] = bool(use_topn_ensemble)
            st.session_state["production_top_n_models"] = int(top_n_models)
            st.session_state["production_selected_confidence_threshold"] = st.slider(
                "Selected model confidence threshold",
                0.0,
                1.0,
                float(st.session_state.get("production_selected_confidence_threshold", 0.0)),
                0.01,
                key="new_analysis_selected_conf_threshold",
            )
            st.session_state["production_ensemble_vote_threshold_pct"] = st.slider(
                "Top-N ensemble vote threshold (%)",
                0.0,
                100.0,
                float(st.session_state.get("production_ensemble_vote_threshold_pct", 0.0)),
                1.0,
                key="new_analysis_ensemble_vote_threshold",
            )
            st.session_state["production_require_both_thresholds"] = st.checkbox(
                "Require both thresholds",
                value=bool(st.session_state.get("production_require_both_thresholds", False)),
                key="new_analysis_require_both_thresholds",
            )
    elif use_topn_ensemble:
        st.info(
            f"Top-{top_n_models} ensemble mode is active. "
            f"Confidence threshold={float(st.session_state.get('production_selected_confidence_threshold', 0.0)):.2f}; "
            f"vote threshold={float(st.session_state.get('production_ensemble_vote_threshold_pct', 0.0)):.0f}%."
        )

    analysis_args, head_config = _prepare_analysis_args(
        ctx,
        is_admin=is_admin,
        use_production_model=bool(use_production_model),
    )
    _render_current_model_summary(
        analysis_args,
        head_config,
        is_admin=is_admin,
        use_production_model=bool(use_production_model),
    )
    _render_preprocessing_trace(uploaded_images, analysis_args)

    topn_models_df = pd.DataFrame()
    if use_topn_ensemble:
        topn_models_df = _load_ranked_models(ctx.cursor, top_n_models)
        if topn_models_df.empty:
            st.warning("Top-N ensemble is enabled, but no ranked models were found. Falling back to the selected model.")
            use_topn_ensemble = False
        else:
            st.markdown("**Top-N models that will run**")
            topn_cols = [c for c in ["#", "Model ID", "Model Name", "Valid MCC", "Test MCC", "Valid AUC", "Test AUC", "Head", "Head Config"] if c in topn_models_df.columns]
            st.dataframe(_arrow_safe_dataframe(topn_models_df[topn_cols]), use_container_width=True)

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
        _delete_existing_results_for_files(ctx, filenames, args=analysis_args)

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    results = []
    start_time = time.time()

    run_total = len(images_for_run)
    model_total = int(len(topn_models_df)) if use_topn_ensemble and not topn_models_df.empty else 1
    total_jobs = max(1, run_total * model_total)
    completed_jobs = 0

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

            if use_topn_ensemble and not topn_models_df.empty:
                per_model_rows = []
                for model_i, (_, model_row) in enumerate(topn_models_df.iterrows(), start=1):
                    model_dict = model_row.drop(labels=["_selection_key"], errors="ignore").to_dict()
                    try:
                        model_args, model_head_config = _prepare_model_args_from_row(ctx, model_dict)
                        status_text.write(
                            f"Processing {filename}: model {model_i}/{model_total} "
                            f"(ID {model_dict.get('Model ID', '?')})"
                        )
                        result = run_analysis_on_file(
                            filename,
                            file_bytes,
                            model_args,
                            ctx.cursor,
                            ctx.conn,
                            force_reanalyze=force_reanalyze,
                            show_validation_metrics=not skip_validation,
                            fast_infer=fast_infer,
                            quiet=True,
                            generate_gradcam=bool(generate_gradcam and model_i == 1),
                        )
                        pred_label, confidence, complete_log_path, existing, gradcam_path, class_scores = _normalize_analysis_result(result)
                        per_model_row = {
                            "Filename": filename,
                            "Model ID": model_dict.get("Model ID"),
                            "Model": model_dict.get("Model Name"),
                            "Rank": model_dict.get("#"),
                            "Head": format_classifier_config(model_head_config),
                            "Head Config": model_head_config,
                            "Prediction": pred_label,
                            "Confidence": confidence,
                            "Existing": bool(existing),
                            "Log Path": complete_log_path,
                        }
                        for label, score in (class_scores or {}).items():
                            per_model_row[f"Score {label}"] = score
                    except Exception as model_exc:
                        per_model_row = {
                            "Filename": filename,
                            "Model ID": model_dict.get("Model ID"),
                            "Model": model_dict.get("Model Name"),
                            "Rank": model_dict.get("#"),
                            "Prediction": "ERROR",
                            "Confidence": np.nan,
                            "Existing": False,
                            "Error": str(model_exc),
                        }
                    per_model_rows.append(per_model_row)
                    completed_jobs += 1
                    progress_bar.progress(min(1.0, completed_jobs / total_jobs))

                selected_row = per_model_rows[0] if per_model_rows else {}
                vote_info = _vote_decision(per_model_rows)
                decision = _threshold_decision(
                    selected_row.get("Prediction", "Unknown"),
                    selected_row.get("Confidence"),
                    vote_info.get("Ensemble Raw Prediction", "Unknown"),
                    vote_info.get("Ensemble Vote %"),
                )
                complete_log_path = selected_row.get("Log Path")
                gradcam_images = _find_gradcam_images(
                    complete_log_path,
                    filename,
                    n_layers=int(n_gradcam_layers),
                )
                row = {
                    "Filename": filename,
                    "Decision": decision,
                    "Prediction": decision,
                    "Selected Model Prediction": selected_row.get("Prediction", "Unknown"),
                    "Selected Model Confidence": selected_row.get("Confidence"),
                    "Selected Model ID": selected_row.get("Model ID"),
                    "Ensemble Prediction": vote_info.get("Ensemble Prediction"),
                    "Ensemble Raw Prediction": vote_info.get("Ensemble Raw Prediction"),
                    "Ensemble Vote %": vote_info.get("Ensemble Vote %"),
                    "Models Voted": vote_info.get("Models Voted"),
                    "Votes": vote_info.get("Votes"),
                    "Confidence": selected_row.get("Confidence"),
                    "Existing": any(bool(r.get("Existing")) for r in per_model_rows),
                    "Log Path": complete_log_path,
                    "Grad-CAM Path": None,
                    "Grad-CAM": len(gradcam_images),
                    "Elapsed (s)": round(time.time() - file_start, 2),
                    "Bytes": file_bytes,
                    "Per-Model Predictions": per_model_rows,
                }
                for key, value in selected_row.items():
                    if str(key).startswith("Score "):
                        row[key] = value
                results.append(row)
            else:
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
                    generate_gradcam=bool(generate_gradcam),
                )

                pred_label, confidence, complete_log_path, existing, gradcam_path, class_scores = _normalize_analysis_result(result)

                gradcam_images = _find_gradcam_images(
                    complete_log_path,
                    filename,
                    n_layers=int(n_gradcam_layers),
                )

                row = {
                    "Filename": filename,
                    "Decision": pred_label,
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
                completed_jobs += 1

            st.session_state["last_uploaded_name"] = filename
            st.session_state["last_complete_log_path"] = results[-1].get("Log Path")
            st.session_state["last_prediction"] = results[-1].get("Decision", results[-1].get("Prediction"))
            st.session_state["last_confidence"] = results[-1].get("Confidence")

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

        progress_bar.progress(min(1.0, completed_jobs / total_jobs))

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