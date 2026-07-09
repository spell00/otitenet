from __future__ import annotations

import sys
import time
import os
from html import escape
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otitenet.offline.deployment import load_deployment
from otitenet.offline.history import (
    clear_history,
    create_user,
    get_user,
    history_csv,
    history_for_user,
    load_history,
    load_users,
    record_result,
)
from otitenet.offline.predictor import load_model, predict
from otitenet.app.image_processing import preprocessing_trace


st.set_page_config(layout="wide")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "TIFF"}
HISTORY_PREVIEW_MAX_SIZE = (1200, 1200)


@st.cache_resource
def _load_runtime():
    deployment = load_deployment()
    device = "cpu"
    model = load_model(deployment, device=device)
    return deployment, model, device


st.title("Ear Health Check")

try:
    deployment, model, device = _load_runtime()
except Exception as exc:
    st.error(f"Offline deployment is not ready: {exc}")
    st.stop()


def compute_and_save_grad_cam_offline(*args, **kwargs):
    model_type = str(deployment.model_type).lower()
    if model_type.startswith("onnx"):
        return (
            False,
            [],
            "Grad-CAM is disabled for the compact ONNX desktop runtime. "
            "Use the exact PyTorch desktop build for Grad-CAM.",
        )
    try:
        from otitenet.app.services.gradcam_service import (
            compute_and_save_grad_cam_offline as _compute,
        )
    except Exception as exc:
        return False, [], f"Grad-CAM is not available in this offline build: {exc}"
    return _compute(*args, **kwargs)


def _selected_user() -> dict:
    users = load_users()
    selected_id = st.session_state.get("offline_person_id")
    if selected_id not in {user["id"] for user in users}:
        selected_id = users[0]["id"]
        st.session_state["offline_person_id"] = selected_id
    return get_user(selected_id)


def _missingish(value) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "nan", "NaN", "—"}


def _load_history_preview_image(image_path: str | os.PathLike):
    path = Path(image_path)
    if not path.exists():
        return None, f"Original image not found: {path}"

    try:
        with Image.open(path) as image:
            image.thumbnail(HISTORY_PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            else:
                image = image.copy()
            image.load()
        return image, None
    except Exception as exc:
        return None, f"Could not load saved image: {exc}"


def _deployment_value(*keys, default="—"):
    """
    Read a deployment value from the places used by the online/offline exporters.

    Priority is production_params first because those are the values that should
    mirror the online production model. The manifest itself is then used as a
    fallback for older offline packages.
    """
    params = deployment.manifest.get("production_params", {}) or {}
    preprocessing = deployment.manifest.get("preprocessing", {}) or {}

    # Production params should win over generic manifest keys.
    for source in (params, deployment.manifest, preprocessing):
        for key in keys:
            value = source.get(key)
            if not _missingish(value):
                return value
    return default


def _deployment_distance_value():
    dist = _deployment_value(
        "Dist_Fct",
        "dist_fct",
        "dist_metric",
        "Distance",
        "distance",
        "metric",
        default="—",
    )
    if _missingish(dist):
        return "—"
    return str(dist)


def _deployment_model_id():
    return _deployment_value("model_id", "Model ID", "Registry ID", "DB Model ID", "id", default="?")


def _deployment_log_path():
    return _deployment_value("log_path", "Log Path", "Artifact Log Path", "Best Model Dir", "Source Run Path")


def _deployment_dataset():
    return _deployment_value(
        "path",
        "Dataset",
        "dataset",
        "Artifact Dataset",
        "Combo Dataset",
        "data_path",
    )


def _deployment_head_values():
    head_config = _deployment_value(
        "Best Head Config",
        "Head Config",
        "best_classifier_config",
        "classification_head_config",
        "head_config",
    )
    head_name = _deployment_value(
        "Best Classification Head",
        "head_name",
        "head_name_selected",
        "Head",
        "head",
        "learned_classifier_label",
    )
    head_family = _deployment_value(
        "classification_head_family",
        "head_family",
        "Head Family",
    )
    head_n_aug = _deployment_value(
        "Head N Aug",
        "Best Head N Aug",
        "best_head_n_aug",
        "head_n_aug",
        "n_aug",
        "N Aug",
    )
    return head_name, head_config, head_family, head_n_aug


def _render_current_deployment_summary(compact: bool = False):
    """
    Offline equivalent of the online 'Current analysis model' panel.

    This intentionally surfaces the production metadata that can change
    predictions: distance, head config, n_aug, KNN/prototype settings,
    preprocessing, labels, and training run.
    """
    head_name, head_config, head_family, head_n_aug = _deployment_head_values()

    normalize = _deployment_value("normalize", "Normalize")
    n_calibration = _deployment_value("n_calibration", "N_Calibration")
    n_positives = _deployment_value("n_positives", "NPos", "npos")
    n_negatives = _deployment_value("n_negatives", "NNeg", "nneg")
    new_size = _deployment_value("new_size", "NSize", "nsize")
    classif_loss = _deployment_value("classif_loss", "Classif_Loss")
    dloss = _deployment_value("dloss", "DLoss")
    fgsm = _deployment_value("fgsm", "FGSM")
    distance = _deployment_distance_value()

    n_neighbors = _deployment_value("n_neighbors", "N_Neighbors")
    prototypes_to_use = _deployment_value("prototypes_to_use", "Prototypes", "prototypes")
    prototype_strategy = _deployment_value("prototype_strategy", "Proto_Strat")
    prototype_components = _deployment_value("prototype_components", "Proto_Comp")

    model_id = _deployment_model_id()
    log_path = _deployment_log_path()
    dataset = _deployment_dataset()

    if compact:
        st.info(f"Production model: {deployment.model_name} #{model_id}")
        st.caption(f"Runtime: {deployment.model_type} on {device}")
        st.caption(
            "Head: "
            f"{head_name} | config: {head_config} | family: {head_family} | n_aug: {head_n_aug}"
        )
        st.caption(
            f"Distance: {distance} | n_neighbors: {n_neighbors} | "
            f"prototypes: {prototypes_to_use} / {prototype_strategy} / {prototype_components}"
        )
        st.caption(
            f"Preprocessing: normalize={normalize} | size={new_size} | n_calibration={n_calibration}"
        )
        if not _missingish(log_path):
            st.caption(f"Training run: {log_path}")
        if deployment.labels:
            st.caption("Labels: " + ", ".join(deployment.labels))
        return

    st.subheader("Current production model")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"**Model:** {deployment.model_name}")
        st.markdown(f"**Model ID:** {model_id}")
        st.markdown(f"**Runtime:** {deployment.model_type} on {device}")
        st.markdown(f"**Size:** {new_size}")

    with c2:
        st.markdown(f"**Normalize:** {normalize}")
        st.markdown(f"**Loss:** {classif_loss}")
        st.markdown(f"**DLoss:** {dloss}")
        st.markdown(f"**Distance:** {distance}")

    with c3:
        st.markdown(f"**Head:** {head_name}")
        st.markdown(f"**Head Config:** `{head_config}`")
        st.markdown(f"**Family:** {head_family}")
        st.markdown(f"**N Aug:** {head_n_aug}")
        st.markdown(f"**n_neighbors:** {n_neighbors}")

    with st.expander("More production metadata", expanded=False):
        extra_rows = [
            {"Field": "Training run", "Value": log_path},
            {"Field": "Dataset", "Value": dataset},
            {"Field": "FGSM", "Value": fgsm},
            {"Field": "n_calibration", "Value": n_calibration},
            {"Field": "n_positives", "Value": n_positives},
            {"Field": "n_negatives", "Value": n_negatives},
            {"Field": "prototypes_to_use", "Value": prototypes_to_use},
            {"Field": "prototype_strategy", "Value": prototype_strategy},
            {"Field": "prototype_components", "Value": prototype_components},
            {"Field": "Labels", "Value": ", ".join(deployment.labels or [])},
        ]
        st.dataframe(extra_rows, hide_index=True, use_container_width=True)

    if distance.lower() == "cosine":
        st.warning(
            "This offline package says Distance=cosine. If the online production model "
            "is euclidean, rebuild/export the offline deployment after setting production "
            "to the euclidean model. The offline app cannot infer the online distance unless "
            "it is present in the deployment manifest."
        )


def _fmt_stats(stats: dict) -> str:
    return (
        f"shape={stats.get('shape')} | "
        f"min={float(stats.get('min', 0.0)):.4f} | "
        f"max={float(stats.get('max', 0.0)):.4f} | "
        f"mean={float(stats.get('mean', 0.0)):.4f} | "
        f"std={float(stats.get('std', 0.0)):.4f}"
    )


def _deployment_normalize_mode() -> str:
    preprocessing = deployment.manifest.get("preprocessing", {}) or {}
    normalization = str(preprocessing.get("normalization", "")).lower()
    normalize = str(preprocessing.get("normalize", "")).lower()
    if normalization == "per_image":
        return "per_image"
    if normalization == "channel_mean_std":
        return "yes"
    if normalization in {"none", "no"} or normalize in {"no", "false", "0"}:
        return "no"
    if normalize in {"yes", "true", "1"}:
        return "yes"
    return "yes"


def _gradcam_available() -> bool:
    return not str(deployment.model_type).lower().startswith("onnx")


def _display_label(label: str) -> str:
    normalized = str(label or "Unknown").strip()
    if normalized.lower().replace("_", "") == "notnormal":
        return "Not Normal"
    return normalized


def _result_palette(label: str) -> dict[str, str]:
    key = _display_label(label).lower()
    if key == "normal":
        return {"color": "#15803d", "background": "#f0fdf4", "border": "#86efac"}
    if key == "not normal":
        return {"color": "#b91c1c", "background": "#fef2f2", "border": "#fca5a5"}
    if key == "wax":
        return {"color": "#ca8a04", "background": "#fefce8", "border": "#fde047"}
    if key == "tube":
        return {"color": "#111827", "background": "#f9fafb", "border": "#111827"}
    return {"color": "#374151", "background": "#f9fafb", "border": "#d1d5db"}


def _result_guidance(label: str, confidence: float | None) -> str:
    if confidence is not None and confidence < 0.50:
        return "Low confidence. Try another photo or consult a doctor."

    key = _display_label(label).lower()
    if key == "tube":
        return "A tube is present. The model is limited when a tube is visible."
    if key == "wax":
        return "Wax is visible. Clean the ear if appropriate and try again the next day, because cleaning can affect the result."
    if key == "not normal":
        return "The model found an abnormal result. Consider consulting a doctor."
    if key == "normal":
        return "The image looks normal according to the model."
    return "Review the result and consult a doctor if symptoms continue."


def _render_result_card(row: dict) -> None:
    label = row.get("Prediction", row.get("prediction", "Unknown"))
    confidence_value = row.get("Confidence", row.get("confidence"))
    try:
        confidence = float(confidence_value) if confidence_value is not None else None
    except (TypeError, ValueError):
        confidence = None

    palette = _result_palette(label)
    display_label = escape(_display_label(label))
    confidence_text = "Confidence unavailable" if confidence is None else f"Confidence: {confidence:.1%}"
    filename = row.get("Filename") or row.get("filename")
    filename_html = (
        f"<div style='font-size:0.9rem;color:#4b5563;margin-bottom:0.35rem;'>{escape(str(filename))}</div>"
        if filename
        else ""
    )
    guidance = escape(_result_guidance(label, confidence))

    st.markdown(
        f"""
        <div style="border:1px solid {palette['border']};background:{palette['background']};
                    border-radius:8px;padding:1rem 1.1rem;margin:0.75rem 0;">
            {filename_html}
            <div style="font-size:1.8rem;font-weight:700;color:{palette['color']};line-height:1.15;">
                {display_label}
            </div>
            <div style="font-size:1.15rem;font-weight:650;color:{palette['color']};margin-top:0.35rem;">
                {confidence_text}
            </div>
            <div style="font-size:1rem;color:#374151;margin-top:0.65rem;">
                {guidance}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _probability_rows(probabilities: list[dict]) -> list[dict]:
    rows = [
        {
            "Class": _display_label(item.get("label", "")),
            "Confidence": f"{float(item.get('probability') or 0):.1%}",
        }
        for item in probabilities or []
    ]
    return sorted(rows, key=lambda item: float(item["Confidence"].rstrip("%")), reverse=True)


def _render_preprocessing_trace(valid_uploads):
    if not valid_uploads:
        return
    with st.expander("Advanced details: preprocessing steps", expanded=False):
        selected = st.selectbox(
            "Image",
            options=list(range(len(valid_uploads))),
            format_func=lambda i: valid_uploads[i]["name"],
            key="offline_preprocess_trace_image",
        )
        upload = valid_uploads[int(selected)]
        height, width = deployment.input_size
        if height != width:
            st.warning(f"Preprocessing trace expects square model input, got {height}x{width}.")
            return
        preprocessing = deployment.manifest.get("preprocessing", {}) or {}
        base_resize_size = int(preprocessing.get("base_resize_size") or 64)
        try:
            trace = preprocessing_trace(
                upload["image"],
                height,
                normalize=_deployment_normalize_mode(),
                base_size=base_resize_size,
            )
        except Exception as exc:
            st.warning(f"Could not render preprocessing trace: {exc}")
            return

        st.caption(
            f"resize_mode={preprocessing.get('resize_mode', trace['resize_mode'])} | "
            f"normalize_mode={trace['normalize_mode']} | base_size={trace.get('base_resize_size', '—')} | model_size={height}"
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


with st.sidebar:
    st.markdown("### 👤 Family Member")
    st.info("Offline mode stores results locally on this device.")

    users = load_users()
    current_id = st.session_state.get("offline_person_id", users[0]["id"])
    user_ids = [user["id"] for user in users]
    if current_id not in user_ids:
        current_id = user_ids[0]

    selected_idx = st.selectbox(
        "Current person",
        options=list(range(len(users))),
        index=user_ids.index(current_id),
        format_func=lambda i: users[i]["name"],
        key="offline_person_select",
    )
    selected_user = users[selected_idx]
    st.session_state["offline_person_id"] = selected_user["id"]

    with st.form("offline_create_user_form", clear_on_submit=True):
        new_person_name = st.text_input("New person name")
        submitted = st.form_submit_button("Create user")
        if submitted:
            try:
                created = create_user(new_person_name)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["offline_person_id"] = created["id"]
                st.success(f"Created {created['name']}.")
                st.rerun()

    st.markdown("---")
    with st.expander("Advanced model information", expanded=False):
        _render_current_deployment_summary(compact=True)


def render_new_analysis():
    selected_user = _selected_user()
    st.header("New analysis")
    st.caption(f"Saving new analyses for {selected_user['name']}.")
    _render_current_deployment_summary(compact=False)
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload one or more otoscope images",
        accept_multiple_files=True,
        key="offline_new_analysis_upload",
    )
    st.caption("Accepted image files: JPG, JPEG, PNG, BMP, TIF, TIFF.")

    if uploaded_files:
        if st.button("🗑️ Clear selected images", key="offline_clear_upload"):
            st.session_state["offline_new_analysis_upload"] = None
            st.rerun()

    if not uploaded_files:
        st.info("Upload an ear image to get a result.")
        return

    valid_uploads = []
    invalid_uploads = []
    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.getvalue()
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            invalid_uploads.append((uploaded_file.name, "unsupported file type"))
            continue

        try:
            Image.open(BytesIO(image_bytes)).verify()
            image = Image.open(BytesIO(image_bytes))
            image.load()
        except Exception:
            invalid_uploads.append((uploaded_file.name, "could not be opened as an image"))
            continue

        if image.format not in IMAGE_FORMATS:
            invalid_uploads.append((uploaded_file.name, f"unsupported image format: {image.format}"))
            continue

        valid_uploads.append(
            {
                "name": uploaded_file.name,
                "bytes": image_bytes,
                "image": image,
            }
        )

    if invalid_uploads:
        for filename, reason in invalid_uploads:
            st.error(f"{filename}: {reason}.")

    if not valid_uploads:
        return

    st.info(f"{len(valid_uploads)} valid image(s) selected for analysis.")

    with st.expander("Uploaded image preview", expanded=len(valid_uploads) <= 3):
        cols_per_row = 3
        for start in range(0, len(valid_uploads), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, upload in zip(cols, valid_uploads[start : start + cols_per_row]):
                with col:
                    st.image(upload["image"], caption=upload["name"], use_container_width=True)

    _render_preprocessing_trace(valid_uploads)

    if st.button("▶️ Run Analysis", type="primary", key="offline_run_analysis"):
        progress = st.progress(0.0)
        status = st.empty()
        rows = []
        start_time = time.time()

        for idx, upload in enumerate(valid_uploads):
            status.write(f"Processing {idx + 1}/{len(valid_uploads)}: {upload['name']}")
            progress.progress(idx / max(len(valid_uploads), 1))

            try:
                result = predict(upload["image"], model, deployment, device=device)
                row = record_result(
                    person_id=selected_user["id"],
                    person_name=selected_user["name"],
                    filename=upload["name"],
                    image_bytes=upload["bytes"],
                    prediction=result,
                    deployment_manifest=deployment.manifest,
                )
                rows.append(
                    {
                        "Filename": upload["name"],
                        "Prediction": result["label"],
                        "Confidence": float(result["confidence"]),
                        "Probabilities": result.get("probabilities", []),
                        "Saved At": row.get("timestamp") or row.get("created_at") or "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "Filename": upload["name"],
                        "Prediction": "ERROR",
                        "Confidence": None,
                        "Error": str(exc),
                    }
                )

            progress.progress((idx + 1) / max(len(valid_uploads), 1))

        status.success(f"Finished {len(valid_uploads)} file(s) in {time.time() - start_time:.1f}s.")
        st.session_state["offline_last_results"] = rows

    rows = st.session_state.get("offline_last_results", [])
    if rows:
        st.subheader("Result")
        for row in rows:
            _render_result_card(row)

        display_rows = []
        for row in rows:
            display = dict(row)
            if display.get("Confidence") is not None:
                display["Confidence"] = f"{float(display['Confidence']):.1%}"
            # Add per-class confidence columns
            if display.get("Probabilities"):
                for prob_item in display["Probabilities"]:
                    label = prob_item.get("label", "")
                    prob = prob_item.get("probability", 0)
                    display[f"Score {_display_label(label)}"] = f"{prob:.1%}"
            # Remove the raw Probabilities list from display
            display.pop("Probabilities", None)
            display_rows.append(display)

        with st.expander("Advanced details", expanded=False):
            st.dataframe(display_rows, hide_index=True, use_container_width=True)

        if _gradcam_available():
            st.markdown("---")
            st.subheader("Grad-CAM Explanations")

            gradcam_dir = os.path.join("data", "offline_gradcam", selected_user["id"])
            filenames_with_images = [upload["name"] for upload in valid_uploads]

            if filenames_with_images:
                selected_for_gradcam = st.multiselect(
                    "Select images to compute Grad-CAM for",
                    options=filenames_with_images,
                    default=[],
                    key="offline_new_analysis_gradcam_select"
                )

                if selected_for_gradcam:
                    if st.button("Compute Grad-CAM for selected images", key="offline_new_analysis_gradcam_compute"):
                        with st.spinner("Computing Grad-CAM..."):
                            success_count = 0
                            error_count = 0
                            for filename in selected_for_gradcam:
                                try:
                                    upload = next((u for u in valid_uploads if u["name"] == filename), None)
                                    if upload:
                                        base_filename = os.path.splitext(filename)[0]

                                        success, saved_paths, error_msg = compute_and_save_grad_cam_offline(
                                            deployment=deployment,
                                            model=model,
                                            image=upload["image"],
                                            output_dir=gradcam_dir,
                                            filename=base_filename,
                                            layer=5,
                                            device=device,
                                            alpha=0.55,
                                        )

                                        if success:
                                            success_count += 1
                                            st.success(f"{filename}: generated {len(saved_paths)} image(s)")
                                            for gc_path in saved_paths:
                                                if os.path.exists(gc_path):
                                                    st.image(gc_path, caption=os.path.basename(gc_path), use_container_width=True)
                                        else:
                                            error_count += 1
                                            st.error(f"{filename}: {error_msg}")
                                except Exception as e:
                                    error_count += 1
                                    st.error(f"{filename}: error computing Grad-CAM: {e}")

                            st.info(f"Completed: {success_count} successful, {error_count} failed")
    else:
        st.info("Run analysis to see predictions.")


def _render_deployment_warnings():
    params = deployment.manifest.get("production_params", {}) or {}
    head_type = str(deployment.manifest.get("head_type", "")).lower()
    model_type = str(deployment.manifest.get("model_type", "")).lower()
    prototypes = str(params.get("prototypes_to_use") or params.get("prototypes") or "").lower()
    head = str(params.get("head") or deployment.manifest.get("head", "")).lower()

    if not params:
        st.warning(
            "This packaged deployment does not record the localhost production parameters. "
            "Recreate it with scripts/create_mobile_deployment.py so model id, preprocessing, and head settings can be verified."
        )
        return

    uses_web_head = prototypes not in {"", "none", "no", "nan"} or "prototype" in head or "knn" in head
    uses_direct_classifier = head_type == "linear_classifier" or model_type.endswith("_classifier")
    if uses_web_head and uses_direct_classifier:
        st.error(
            "This deployment was packaged as a direct classifier, but the localhost production model uses a prototype/KNN-style head. "
            "Scores will differ until the deployment is rebuilt with a matching head."
        )


def render_historics():
    selected_user = _selected_user()
    st.header("History")

    history_scope = st.radio(
        "History scope",
        options=["Current person", "Everyone"],
        horizontal=True,
        key="offline_history_scope",
    )
    showing_everyone = history_scope == "Everyone"
    rows = load_history() if showing_everyone else history_for_user(selected_user["id"])
    scope_label = "everyone" if showing_everyone else selected_user["name"]
    st.caption(f"Showing saved analyses for {scope_label}.")

    actions = st.columns([1, 1, 3])
    with actions[0]:
        export_name = "all_people" if showing_everyone else selected_user["name"].replace(" ", "_")
        st.download_button(
            "Download history",
            data=history_csv(rows),
            file_name=f"otitenet_offline_history_{export_name}.csv",
            mime="text/csv",
            disabled=not rows,
            key="offline_history_download",
        )
    with actions[1]:
        if st.button(
            "Clear history",
            disabled=not rows,
            key="offline_history_clear",
        ):
            clear_history(None if showing_everyone else selected_user["id"])
            st.session_state.pop("offline_last_result", None)
            st.session_state.pop("offline_last_history_row", None)
            st.success(f"History cleared for {scope_label}.")
            st.rerun()

    if not rows:
        st.info("No offline results have been saved for this view yet.")
        return

    display_rows = []
    for row in rows:
        display_row = {
            "Timestamp": row.get("timestamp"),
            "Filename": row.get("filename"),
            "Prediction": row.get("prediction"),
            "Confidence": f"{float(row.get('confidence') or 0):.1%}",
            "Model ID": row.get("model_id"),
            "Model": row.get("model_name"),
        }
        if showing_everyone:
            display_row = {"Person": row.get("person_name", "Offline user"), **display_row}
        display_rows.append(display_row)

    st.dataframe(display_rows, hide_index=True, use_container_width=True)

    selected_idx = st.selectbox(
        "Observe result",
        options=list(range(len(rows))),
        format_func=lambda i: (
            f"{rows[i].get('person_name', 'Offline user')} | "
            if showing_everyone
            else ""
        )
        + f"{rows[i].get('filename')} | {rows[i].get('prediction')} | {float(rows[i].get('confidence') or 0):.1%}",
        key="offline_history_selected",
    )
    row = rows[selected_idx]
    row_person_id = row.get("person_id") or selected_user["id"]

    c1, c2 = st.columns([1, 1])
    with c1:
        image_path = row.get("image_path")
        if image_path:
            preview_image, preview_error = _load_history_preview_image(image_path)
            if preview_image is not None:
                st.image(preview_image, caption=row.get("filename"), use_container_width=True)
            else:
                st.warning(preview_error)
    with c2:
        if showing_everyone:
            st.markdown(f"**Person:** {row.get('person_name', 'Offline user')}")
        _render_result_card(row)

        with st.expander("Advanced details", expanded=False):
            st.markdown(f"**Model:** {row.get('model_name')} #{row.get('model_id')}")
            st.markdown(f"**Timestamp:** {row.get('timestamp')}")

            probabilities = row.get("probabilities", [])
            if probabilities:
                st.markdown("**Per-class confidence scores:**")
                st.dataframe(_probability_rows(probabilities), hide_index=True, use_container_width=True)
            else:
                st.info("No per-class confidence scores available.")

    if _gradcam_available():
        st.markdown("---")
        st.markdown("**Grad-CAM Explanation**")

        gradcam_dir = os.path.join("data", "offline_gradcam", str(row_person_id))
        base_filename = os.path.splitext(row.get("filename", ""))[0]
        existing_gradcam = []
        if os.path.exists(gradcam_dir):
            for f in os.listdir(gradcam_dir):
                if f.startswith(base_filename) and f.endswith(".png"):
                    existing_gradcam.append(os.path.join(gradcam_dir, f))
        existing_gradcam.sort()

        if existing_gradcam:
            st.success(f"Found {len(existing_gradcam)} Grad-CAM image(s)")
            for gc_path in existing_gradcam[:4]:
                st.image(gc_path, caption=os.path.basename(gc_path), use_container_width=True)
        else:
            st.info("No Grad-CAM images computed for this result yet.")

            gradcam_key = f"offline_gradcam_{row_person_id}_{row.get('image_sha256') or selected_idx}"
            if st.button("Compute Grad-CAM on demand", key=gradcam_key):
                try:
                    with st.spinner("Computing Grad-CAM..."):
                        image_path = row.get("image_path")
                        if not image_path or not os.path.exists(image_path):
                            st.error("Original image not found")
                        else:
                            with Image.open(image_path) as original_image:
                                image = original_image.convert("RGB")
                                image.load()

                            success, saved_paths, error_msg = compute_and_save_grad_cam_offline(
                                deployment=deployment,
                                model=model,
                                image=image,
                                output_dir=gradcam_dir,
                                filename=base_filename,
                                layer=5,
                                device=device,
                                alpha=0.55,
                            )

                            if success:
                                st.success(f"Grad-CAM computed successfully. Generated {len(saved_paths)} image(s)")
                                for gc_path in saved_paths:
                                    if os.path.exists(gc_path):
                                        st.image(gc_path, caption=os.path.basename(gc_path), use_container_width=True)
                            else:
                                st.error(f"Grad-CAM computation failed: {error_msg}")
                except Exception as e:
                    st.error(f"Error computing Grad-CAM: {e}")


tab_new_analysis, tab_historics = st.tabs(["New analysis", "History"])

_render_deployment_warnings()

with tab_new_analysis:
    render_new_analysis()

with tab_historics:
    render_historics()
