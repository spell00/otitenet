from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otitenet.offline.deployment import load_deployment
from otitenet.offline.history import load_history, record_result
from otitenet.offline.predictor import load_model, predict


st.set_page_config(layout="wide")


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

with st.sidebar:
    st.markdown("### 👤 Family Member")
    st.info("Offline mode stores results locally on this device.")
    person_name = st.text_input("Name", value=st.session_state.get("offline_person_name", "Offline user"))
    st.session_state["offline_person_name"] = person_name.strip() or "Offline user"

    st.markdown("---")
    st.markdown("**Current Optimization:**")
    st.info(f"Production model: {deployment.model_name} #{deployment.manifest.get('model_id', '?')}")
    st.caption(f"Runtime: {deployment.model_type} on {device}")
    if deployment.labels:
        st.caption("Labels: " + ", ".join(deployment.labels))


def render_new_analysis():
    st.header("🧪 New Analysis")
    st.caption(
        "Offline mode uses the production model packaged in the desktop app. "
        "No admin controls, training logs, or model history are loaded."
    )

    uploaded_file = st.file_uploader(
        "Upload one otoscope image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=False,
        key="offline_new_analysis_upload",
    )

    if uploaded_file is None:
        st.info("Upload an ear image to run the packaged production model.")
        return

    image_bytes = uploaded_file.getvalue()
    image = Image.open(uploaded_file)
    left, right = st.columns([1, 1])

    with left:
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with right:
        if st.button("▶️ Run Analysis", type="primary", key="offline_run_analysis"):
            try:
                result = predict(image, model, deployment, device=device)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

            row = record_result(
                filename=uploaded_file.name,
                image_bytes=image_bytes,
                prediction=result,
                deployment_manifest=deployment.manifest,
            )
            st.session_state["offline_last_result"] = result
            st.session_state["offline_last_history_row"] = row

        result = st.session_state.get("offline_last_result")
        if result:
            st.metric("Prediction", result["label"], f"{result['confidence']:.1%}")
            st.dataframe(result["probabilities"], hide_index=True, use_container_width=True)
        else:
            st.info("Run analysis to see the prediction.")


def render_historics():
    st.header("📊 Historics")
    rows = load_history()

    if not rows:
        st.info("No offline results have been saved on this device yet.")
        return

    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "Timestamp": row.get("timestamp"),
                "Filename": row.get("filename"),
                "Prediction": row.get("prediction"),
                "Confidence": f"{float(row.get('confidence') or 0):.1%}",
                "Model ID": row.get("model_id"),
                "Model": row.get("model_name"),
            }
        )

    st.dataframe(display_rows, hide_index=True, use_container_width=True)

    selected_idx = st.selectbox(
        "Observe result",
        options=list(range(len(rows))),
        format_func=lambda i: f"{rows[i].get('filename')} | {rows[i].get('prediction')} | {float(rows[i].get('confidence') or 0):.1%}",
        key="offline_history_selected",
    )
    row = rows[selected_idx]

    c1, c2 = st.columns([1, 1])
    with c1:
        image_path = row.get("image_path")
        if image_path:
            st.image(image_path, caption=row.get("filename"), use_container_width=True)
    with c2:
        st.markdown(f"**Prediction:** {row.get('prediction')}")
        st.markdown(f"**Confidence:** {float(row.get('confidence') or 0):.1%}")
        st.markdown(f"**Model:** {row.get('model_name')} #{row.get('model_id')}")
        st.markdown(f"**Timestamp:** {row.get('timestamp')}")
        st.dataframe(row.get("probabilities", []), hide_index=True, use_container_width=True)


tab_new_analysis, tab_historics = st.tabs(["🧪 New Analysis", "📊 Historics"])

with tab_new_analysis:
    render_new_analysis()

with tab_historics:
    render_historics()
