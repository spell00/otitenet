
# /home/simon/otitenet/otitenet/app/pages/past_results.py

from __future__ import annotations

import glob
import os
from typing import Any, Optional

import pandas as pd
import streamlit as st

from otitenet.app.database import list_image_results
from otitenet.app.display_metrics import _arrow_safe_dataframe
from otitenet.app.services.inference_results_service import (
    find_inference_gradcam_images,
    fmt_confidence,
)
from otitenet.app.utils import strip_extension


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"]


def _image_search_locations(filename: str):
    return [
        os.path.join("data", "queries", filename),
        os.path.join("data", "datasets", "inference", filename),
    ]


def _find_image(filename: str) -> Optional[str]:
    if not filename:
        return None

    for path in _image_search_locations(filename):
        if os.path.exists(path):
            return path

    for root in ["data/queries", "data/datasets/inference", "data"]:
        try:
            matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
            matches = [m for m in matches if os.path.isfile(m)]
            if matches:
                return matches[0]
        except Exception:
            pass

    return None


def _fetch_person_results(ctx, person_id: int, limit: int = 1000) -> pd.DataFrame:
    cursor = ctx.cursor

    cursor.execute(
        """
        SELECT
            filename,
            model_name,
            task,
            pred_label,
            confidence,
            log_path,
            timestamp,
            nsize,
            fgsm,
            normalize,
            n_calibration,
            classif_loss,
            dloss,
            dist_fct,
            prototypes,
            npos,
            nneg,
            n_neighbors,
            model_id
        FROM results
        WHERE person_id=%s
        ORDER BY timestamp DESC
        LIMIT %s
        """,
        (int(person_id), int(limit)),
    )

    rows = cursor.fetchall() or []

    cols = [
        "Filename",
        "Model",
        "Task",
        "Prediction",
        "Confidence",
        "Log Path",
        "Timestamp",
        "Size",
        "FGSM",
        "Normalize",
        "N_Cal",
        "Loss",
        "DLoss",
        "Dist",
        "Prototypes",
        "NPos",
        "NNeg",
        "N_Neighbors",
        "Model ID",
    ]

    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        return df

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    return df


def _clear_person_results(ctx, person_id: int) -> int:
    cursor = ctx.cursor
    conn = ctx.conn
    cursor.execute("DELETE FROM results WHERE person_id=%s", (int(person_id),))
    deleted = cursor.rowcount if cursor.rowcount is not None else 0
    conn.commit()
    return deleted


def _history_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def _render_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Images", df["Filename"].nunique() if "Filename" in df.columns else 0)
    c3.metric("Models", df["Model ID"].nunique() if "Model ID" in df.columns else 0)

    try:
        latest = pd.to_datetime(df["Timestamp"], errors="coerce").max()
        c4.metric("Latest", "—" if pd.isna(latest) else latest.strftime("%Y-%m-%d %H:%M"))
    except Exception:
        c4.metric("Latest", "—")


def _render_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    st.subheader("Filters")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        filenames = sorted(df["Filename"].dropna().astype(str).unique().tolist()) if "Filename" in df.columns else []
        selected_files = st.multiselect(
            "Images",
            options=filenames,
            default=[],
            key="past_results_filter_files",
            help="Empty means all images.",
        )

    with c2:
        preds = sorted(df["Prediction"].dropna().astype(str).unique().tolist()) if "Prediction" in df.columns else []
        selected_preds = st.multiselect(
            "Predictions",
            options=preds,
            default=[],
            key="past_results_filter_predictions",
            help="Empty means all predictions.",
        )

    with c3:
        model_ids = sorted(df["Model ID"].dropna().astype(str).unique().tolist()) if "Model ID" in df.columns else []
        selected_models = st.multiselect(
            "Model IDs",
            options=model_ids,
            default=[],
            key="past_results_filter_models",
            help="Empty means all models.",
        )

    with c4:
        search = st.text_input(
            "Search filename/model/log path",
            value="",
            key="past_results_search",
        )

    out = df.copy()

    if selected_files:
        out = out[out["Filename"].astype(str).isin(selected_files)]

    if selected_preds:
        out = out[out["Prediction"].astype(str).isin(selected_preds)]

    if selected_models:
        out = out[out["Model ID"].astype(str).isin(selected_models)]

    if search.strip():
        s = search.strip().lower()
        mask = pd.Series(False, index=out.index)
        for col in ["Filename", "Model", "Prediction", "Log Path"]:
            if col in out.columns:
                mask = mask | out[col].astype(str).str.lower().str.contains(s, na=False)
        out = out[mask]

    return out


def _render_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No stored results match the filters.")
        return

    st.subheader("Stored analyses")

    cols = [
        "Filename",
        "Prediction",
        "Confidence",
        "Timestamp",
        "Model ID",
        "Model",
        "Size",
        "Loss",
        "DLoss",
        "Dist",
        "Normalize",
        "N_Neighbors",
        "Prototypes",
        "Log Path",
    ]
    cols = [c for c in cols if c in df.columns]

    display = df[cols].copy()
    if "Confidence" in display.columns:
        display["Confidence"] = display["Confidence"].apply(fmt_confidence)

    st.dataframe(
        _arrow_safe_dataframe(display.drop(columns=["Log Path"], errors="ignore")),
        use_container_width=True,
    )


def _render_observer(df: pd.DataFrame) -> None:
    if df.empty:
        return

    st.subheader("Observe one result")

    options = df.index.tolist()

    def label(idx):
        row = df.loc[idx]
        return f"{row.get('Filename')} | pred={row.get('Prediction')} | model_id={row.get('Model ID')} | {row.get('Timestamp')}"

    idx = st.selectbox(
        "Select result",
        options=options,
        format_func=label,
        key="past_results_observer_select",
    )

    row = df.loc[idx]

    c1, c2 = st.columns([1, 1])

    with c1:
        img_path = _find_image(row.get("Filename"))
        if img_path:
            st.image(img_path, caption=row.get("Filename"), use_container_width=True)
        else:
            st.warning(f"Could not find original image: {row.get('Filename')}")

    with c2:
        st.markdown(f"**Prediction:** {row.get('Prediction')}")
        st.markdown(f"**Confidence:** {fmt_confidence(row.get('Confidence'))}")
        st.markdown(f"**Model ID:** {row.get('Model ID')}")
        st.markdown(f"**Model:** {row.get('Model')}")
        st.markdown(f"**Timestamp:** {row.get('Timestamp')}")
        with st.expander("Model/log details"):
            st.write(row.drop(labels=[]).to_dict())
            if row.get("Log Path"):
                st.code(str(row.get("Log Path")))

    gradcams = []
    try:
        gradcams = find_inference_gradcam_images(row.get("Log Path"), row.get("Filename"), n_layers=4)
    except Exception:
        gradcams = []

    if gradcams:
        st.markdown("**Grad-CAM**")
        for p in gradcams:
            st.image(p, caption=os.path.basename(p), use_container_width=True)


def render(ctx: Any) -> None:
    st.header("📊 Past Results")

    person_id = (
        st.session_state.get("person_id")
        or getattr(ctx, "selected_person_id", None)
    )

    if person_id is None:
        st.warning("Please select a family member from the sidebar before viewing past results.")
        return

    st.session_state["person_id"] = person_id

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        limit = st.number_input(
            "Max rows to load",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100,
            key="past_results_limit",
        )

    with c2:
        reload_clicked = st.button("Reload past results", key="past_results_reload")

    with c3:
        download_placeholder = st.empty()

    with c4:
        clear_clicked = st.button("Clear history", key="past_results_clear")

    cache_key = f"past_results_person_{person_id}_limit_{int(limit)}"
    if reload_clicked:
        st.session_state.pop(cache_key, None)

    if cache_key not in st.session_state:
        with st.spinner("Loading past results..."):
            st.session_state[cache_key] = _fetch_person_results(ctx, int(person_id), int(limit))

    df = st.session_state.get(cache_key, pd.DataFrame())

    with download_placeholder:
        st.download_button(
            "Download history",
            data=_history_csv_bytes(df),
            file_name=f"otitenet_history_person_{person_id}.csv",
            mime="text/csv",
            disabled=df.empty,
            key="past_results_download",
        )

    if df.empty:
        st.info("No past results found for this person yet.")
        return

    if clear_clicked:
        try:
            deleted = _clear_person_results(ctx, int(person_id))
        except Exception as exc:
            st.error(f"Could not clear history: {exc}")
        else:
            st.session_state.pop(cache_key, None)
            st.success(f"Cleared {deleted} stored result{'s' if deleted != 1 else ''}.")
            st.rerun()

    _render_summary(df)

    filtered = _render_filters(df)

    st.divider()
    _render_table(filtered)

    st.divider()
    _render_observer(filtered)
