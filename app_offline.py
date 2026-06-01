from __future__ import annotations

import sys
import time
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


st.set_page_config(layout="wide")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "TIFF"}


@st.cache_resource
def _load_runtime():
    deployment = load_deployment()
    device = "cpu"
    model = load_model(deployment, device=device)
    return deployment, model, device


st.title("Ear Health Classifier with SHAP 👂")

try:
    deployment, model, device = _load_runtime()
except Exception as exc:
    st.error(f"Offline deployment is not ready: {exc}")
    st.stop()


def _selected_user() -> dict:
    users = load_users()
    selected_id = st.session_state.get("offline_person_id")
    if selected_id not in {user["id"] for user in users}:
        selected_id = users[0]["id"]
        st.session_state["offline_person_id"] = selected_id
    return get_user(selected_id)


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
    st.markdown("**Current Optimization:**")
    st.info(f"Production model: {deployment.model_name} #{deployment.manifest.get('model_id', '?')}")
    st.caption(f"Runtime: {deployment.model_type} on {device}")
    if deployment.labels:
        st.caption("Labels: " + ", ".join(deployment.labels))


def render_new_analysis():
    selected_user = _selected_user()
    st.header("🧪 New Analysis")
    st.caption(f"Saving new analyses for {selected_user['name']}.")
    st.caption(
        "Offline mode uses the production model packaged in the desktop app. "
        "No admin controls, training logs, or model history are loaded."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more otoscope images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="offline_new_analysis_upload",
    )
    st.caption("Accepted image files: JPG, JPEG, PNG, BMP, TIF, TIFF.")

    if not uploaded_files:
        st.info("Upload ear images to run the packaged production model.")
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
        display_rows = []
        for row in rows:
            display = dict(row)
            if display.get("Confidence") is not None:
                display["Confidence"] = f"{float(display['Confidence']):.1%}"
            display_rows.append(display)
        st.subheader("Analysis results")
        st.dataframe(display_rows, hide_index=True, use_container_width=True)
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
    st.header("📊 Historics")

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

    c1, c2 = st.columns([1, 1])
    with c1:
        image_path = row.get("image_path")
        if image_path:
            st.image(image_path, caption=row.get("filename"), use_container_width=True)
    with c2:
        if showing_everyone:
            st.markdown(f"**Person:** {row.get('person_name', 'Offline user')}")
        st.markdown(f"**Prediction:** {row.get('prediction')}")
        st.markdown(f"**Confidence:** {float(row.get('confidence') or 0):.1%}")
        st.markdown(f"**Model:** {row.get('model_name')} #{row.get('model_id')}")
        st.markdown(f"**Timestamp:** {row.get('timestamp')}")
        st.dataframe(row.get("probabilities", []), hide_index=True, use_container_width=True)


tab_new_analysis, tab_historics = st.tabs(["🧪 New Analysis", "📊 Historics"])

_render_deployment_warnings()

with tab_new_analysis:
    render_new_analysis()

with tab_historics:
    render_historics()
