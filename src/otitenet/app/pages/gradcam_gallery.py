
# /home/simon/otitenet/otitenet/app/pages/gradcam_gallery.py

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from otitenet.app.display_metrics import (
    _arrow_safe_dataframe,
    _head_config_label_global,
)
from otitenet.app.services.gradcam_service import (
    compute_and_save_grad_cam_online,
    create_temp_grad_cam_dir,
    pil_image_to_tensor,
)
from otitenet.app.services.inference_results_service import (
    find_inference_gradcam_images,
    fmt_confidence,
    inference_ground_truth,
    labels_match,
)
from otitenet.app.utils import strip_extension


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"]


def _is_image_file(path: str) -> bool:
    return str(path).endswith(tuple(IMAGE_EXTENSIONS))


def _existing_image_path(filename: str, inference_dir: str = "data/datasets/inference") -> Optional[str]:
    """Find the original image from common query/inference locations."""
    candidates = [
        os.path.join(inference_dir, filename),
        os.path.join("data", "queries", filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Fallback: search common data folders by basename.
    for root in ["data/queries", "data/datasets/inference", "data"]:
        try:
            matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
            matches = [m for m in matches if os.path.isfile(m)]
            if matches:
                return matches[0]
        except Exception:
            pass

    return None


def _manual_find_gradcam_images(log_path: str, filename: str, n_layers: int = 4) -> List[str]:
    """
    Fallback Grad-CAM discovery when service helper is unavailable or misses legacy layouts.

    New predictions write:
      <log_path>/<base_name>/<base_name>_grad_cam_all_classes_layerX.png

    Legacy layouts may put files deeper under queries or use similar filenames.
    """
    if not log_path or not filename:
        return []

    base = strip_extension(os.path.basename(str(filename)))
    roots = []

    lp = str(log_path).rstrip("/")
    roots.append(lp)
    roots.append(os.path.join(lp, base))

    if not lp.endswith("queries"):
        roots.append(os.path.join(lp, "queries"))
        roots.append(os.path.join(lp, "queries", base))

    patterns = []
    for root in roots:
        patterns.extend(
            [
                os.path.join(root, f"{base}_grad_cam_all_classes_layer*.png"),
                os.path.join(root, f"*{base}*grad*cam*layer*.png"),
                os.path.join(root, f"*{base}*Grad*CAM*.png"),
                os.path.join(root, "**", f"{base}_grad_cam_all_classes_layer*.png"),
                os.path.join(root, "**", f"*{base}*grad*cam*.png"),
            ]
        )

    found = []
    for pat in patterns:
        try:
            found.extend(glob.glob(pat, recursive=True))
        except Exception:
            pass

    found = [p for p in found if os.path.isfile(p) and _is_image_file(p)]

    # Deduplicate while preserving order.
    found = list(dict.fromkeys(found))

    def _layer_key(path):
        name = os.path.basename(path).lower()
        # Try to sort layer0, layer1, layer2...
        import re
        m = re.search(r"layer(\d+)", name)
        if m:
            return int(m.group(1))
        return 9999

    found = sorted(found, key=_layer_key)

    if n_layers and len(found) > n_layers:
        found = found[-int(n_layers):]

    return found


def _find_gradcam_images(log_path: str, filename: str, n_layers: int = 4) -> List[str]:
    try:
        imgs = find_inference_gradcam_images(log_path, filename, n_layers=n_layers)
        if imgs:
            return imgs
    except Exception:
        pass

    return _manual_find_gradcam_images(log_path, filename, n_layers=n_layers)


def _fetch_result_rows(ctx, limit: int = 500, person_id: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch recent result rows with enough metadata to build a Grad-CAM gallery.
    """
    cursor = ctx.cursor

    where = []
    params = []

    if person_id is not None:
        where.append("r.person_id=%s")
        params.append(int(person_id))

    where_sql = "WHERE " + " AND ".join(where) if where else ""

    # Keep this query mostly based on results so it works even if registry joins fail.
    query = f"""
        SELECT
            r.filename,
            r.pred_label,
            r.confidence,
            r.timestamp,
            r.log_path,
            r.model_id,
            r.model_name,
            r.task,
            r.nsize,
            r.fgsm,
            r.normalize,
            r.n_calibration,
            r.classif_loss,
            r.dloss,
            r.dist_fct,
            r.prototypes,
            r.npos,
            r.nneg,
            r.n_neighbors,
            r.person_id
        FROM results r
        {where_sql}
        ORDER BY r.timestamp DESC
        LIMIT %s
    """

    params.append(int(limit))

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall() or []

    columns = [
        "Filename",
        "Prediction",
        "Confidence",
        "Timestamp",
        "Log Path",
        "Model ID",
        "Model Name",
        "Task",
        "NSize",
        "FGSM",
        "Normalize",
        "N_Calibration",
        "Classif_Loss",
        "DLoss",
        "Dist_Fct",
        "Prototypes",
        "NPos",
        "NNeg",
        "N_Neighbors",
        "Person ID",
    ]

    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        return df

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    return df


def _attach_gradcam_status(df: pd.DataFrame, n_layers: int = 4) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    counts = []
    first_paths = []

    for _, row in out.iterrows():
        imgs = _find_gradcam_images(row.get("Log Path"), row.get("Filename"), n_layers=n_layers)
        counts.append(len(imgs))
        first_paths.append(imgs[0] if imgs else "")

    out["Grad-CAM Count"] = counts
    out["Grad-CAM Available"] = out["Grad-CAM Count"] > 0
    out["First Grad-CAM Path"] = first_paths

    return out


def _format_model_label(row: Dict[str, Any]) -> str:
    bits = [
        f"ID={row.get('Model ID')}",
        f"name={row.get('Model Name')}",
        f"size={row.get('NSize')}",
        f"loss={row.get('Classif_Loss')}",
        f"dloss={row.get('DLoss')}",
        f"dist={row.get('Dist_Fct')}",
        f"norm={row.get('Normalize')}",
        f"nn={row.get('N_Neighbors')}",
    ]
    return " | ".join([b for b in bits if not b.endswith("=None")])


def _render_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    n_rows = len(df)
    n_files = df["Filename"].nunique() if "Filename" in df.columns else 0
    n_models = df["Model ID"].nunique() if "Model ID" in df.columns else 0
    n_gc = int(df.get("Grad-CAM Available", pd.Series(dtype=bool)).sum()) if "Grad-CAM Available" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Result rows", n_rows)
    c2.metric("Images", n_files)
    c3.metric("Models", n_models)
    c4.metric("Rows with Grad-CAM", n_gc)


def _render_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    st.subheader("Filters")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        model_ids = sorted([str(x) for x in df["Model ID"].dropna().unique()]) if "Model ID" in df.columns else []
        selected_models = st.multiselect(
            "Model IDs",
            options=model_ids,
            default=model_ids[: min(len(model_ids), 10)],
            key="gradcam_filter_model_ids",
        )

    with c2:
        preds = sorted([str(x) for x in df["Prediction"].dropna().unique()]) if "Prediction" in df.columns else []
        selected_preds = st.multiselect(
            "Predictions",
            options=preds,
            default=preds,
            key="gradcam_filter_predictions",
        )

    with c3:
        only_available = st.checkbox(
            "Only rows with Grad-CAM",
            value=True,
            key="gradcam_only_available",
        )

    with c4:
        search = st.text_input(
            "Search filename/model/log path",
            value="",
            key="gradcam_search",
        )

    out = df.copy()

    if selected_models and "Model ID" in out.columns:
        out = out[out["Model ID"].astype(str).isin(selected_models)]

    if selected_preds and "Prediction" in out.columns:
        out = out[out["Prediction"].astype(str).isin(selected_preds)]

    if only_available and "Grad-CAM Available" in out.columns:
        out = out[out["Grad-CAM Available"] == True]

    if search.strip():
        s = search.strip().lower()
        mask = pd.Series(False, index=out.index)
        for col in ["Filename", "Model Name", "Prediction", "Log Path"]:
            if col in out.columns:
                mask = mask | out[col].astype(str).str.lower().str.contains(s, na=False)
        out = out[mask]

    return out


def _display_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No rows match the current filters.")
        return

    display_cols = [
        "Filename",
        "Prediction",
        "Confidence",
        "Grad-CAM Count",
        "Timestamp",
        "Model ID",
        "Model Name",
        "NSize",
        "Classif_Loss",
        "DLoss",
        "Dist_Fct",
        "Normalize",
        "N_Neighbors",
        "Log Path",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    table = df[display_cols].copy()
    if "Confidence" in table.columns:
        table["Confidence"] = table["Confidence"].apply(fmt_confidence)

    st.dataframe(
        _arrow_safe_dataframe(table.drop(columns=["Log Path"], errors="ignore")),
        use_container_width=True,
    )


def _compute_grad_cam_on_demand(row: pd.Series, ctx: Any, n_layers: int) -> Tuple[bool, str]:
    """Compute Grad-CAM on demand for a selected result row.

    Args:
        row: Selected result row with model and image information
        ctx: App context with args and device
        n_layers: Number of layers to compute Grad-CAM for

    Returns:
        Tuple of (success, message)
    """
    try:
        from otitenet.app.model_loading import load_model_and_prototypes
        from PIL import Image

        from otitenet.app.image_processing import infer_base_resize_size, preprocess_image
        from otitenet.app.services.inference_results_service import args_from_inference_row

        filename = row.get("Filename")
        log_path = row.get("Log Path")

        if not log_path or not filename:
            return False, "Missing log_path or filename"

        # Build args from the result row
        model_args = args_from_inference_row(ctx.args, row.to_dict())

        # Load model and prototypes
        with st.spinner("Loading model and prototypes..."):
            model, _, prototypes, image_size, device_str, _, unique_labels, _, _ = load_model_and_prototypes(model_args)

        # Load and preprocess image
        img_path = _existing_image_path(filename, inference_dir="data/datasets/inference")
        if not img_path or not os.path.exists(img_path):
            return False, f"Image file not found: {filename}"

        image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess_image(
            image,
            image_size,
            model_args.normalize,
            base_size=infer_base_resize_size(getattr(model_args, "path", None)),
        )

        # Prepare prototypes dict
        class_prototypes = {}
        if hasattr(prototypes, 'class_prototypes'):
            proto_dict = prototypes.class_prototypes.get('train', {})
            for label, proto_array in proto_dict.items():
                if proto_array is not None:
                    class_prototypes[label] = proto_array

        if not class_prototypes:
            return False, "No prototypes available for Grad-CAM computation"

        # Compute Grad-CAM for each layer
        output_dir = log_path
        base_filename = strip_extension(filename)

        with st.spinner("Computing Grad-CAM heatmaps..."):
            for layer in range(max(1, n_layers - 3), n_layers + 1):
                try:
                    saved_paths = compute_and_save_grad_cam_online(
                        model=model,
                        image_tensor=image_tensor,
                        class_prototypes=class_prototypes,
                        output_dir=output_dir,
                        filename=base_filename,
                        layer=layer,
                        device=device_str,
                        alpha=0.55,
                    )
                except Exception as e:
                    st.warning(f"Failed to compute Grad-CAM for layer {layer}: {e}")
                    continue

        return True, f"Grad-CAM computed successfully for {filename}"

    except Exception as e:
        return False, f"Grad-CAM computation failed: {str(e)}"


def _render_selected_observation(row: pd.Series, inference_dir: str, n_layers: int, ctx: Any) -> None:
    filename = row.get("Filename")
    log_path = row.get("Log Path")

    st.subheader("Observe selected image")

    left, right = st.columns([1, 1])

    with left:
        img_path = _existing_image_path(filename, inference_dir=inference_dir)
        if img_path and os.path.exists(img_path):
            st.image(img_path, caption=f"Original: {filename}", use_container_width=True)
        else:
            st.warning(f"Original image file not found for: {filename}")

    with right:
        st.markdown(f"**Prediction:** {row.get('Prediction')}")
        st.markdown(f"**Confidence:** {fmt_confidence(row.get('Confidence'))}")
        st.markdown(f"**Model:** {_format_model_label(row.to_dict())}")
        if row.get("Timestamp") is not None:
            st.markdown(f"**Timestamp:** {row.get('Timestamp')}")
        if log_path:
            with st.expander("Log path"):
                st.code(str(log_path))

        gt = inference_ground_truth(filename, inference_dir)
        if gt not in (None, "", "unknown", "Unknown"):
            st.markdown(f"**Ground truth:** {gt}")
            correct = labels_match(row.get("Prediction"), gt)
            if correct is True:
                st.success("Correct: True")
            elif correct is False:
                st.error("Correct: False")
            else:
                st.info("Correct: Unknown")

    gradcam_images = _find_gradcam_images(log_path, filename, n_layers=n_layers)

    st.markdown("**Grad-CAM composite explanations**")
    if not gradcam_images:
        st.info(
            "No Grad-CAM image found for this row. Fresh predictions generate Grad-CAM automatically. "
            "To regenerate, rerun inference/new analysis for this image with force re-analyze enabled."
        )
        
        # Add on-demand computation button
        if st.button("🔄 Compute Grad-CAM on demand", key=f"compute_gradcam_{filename}_{log_path}"):
            success, message = _compute_grad_cam_on_demand(row, ctx, n_layers)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        return

    for gc_path in gradcam_images:
        try:
            st.image(gc_path, caption=os.path.basename(gc_path), use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not display {gc_path}: {exc}")


def _render_grid(df: pd.DataFrame, inference_dir: str, n_layers: int, max_items: int) -> None:
    if df.empty:
        return

    st.subheader("Gallery grid")

    rows = df.head(int(max_items)).copy()
    cols_per_row = st.slider(
        "Images per row",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        key="gradcam_cols_per_row",
    )

    for start in range(0, len(rows), cols_per_row):
        cols = st.columns(cols_per_row)
        chunk = rows.iloc[start : start + cols_per_row]

        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                filename = row.get("Filename")
                imgs = _find_gradcam_images(row.get("Log Path"), filename, n_layers=n_layers)
                if imgs:
                    # Show the deepest/latest layer first in grid.
                    st.image(imgs[-1], caption=f"{filename} | pred={row.get('Prediction')}", use_container_width=True)
                else:
                    img_path = _existing_image_path(filename, inference_dir=inference_dir)
                    if img_path:
                        st.image(img_path, caption=f"{filename} | no Grad-CAM", use_container_width=True)
                    else:
                        st.info(f"{filename}: no Grad-CAM")


def render(ctx: Any) -> None:
    st.header("📊 Grad-CAM Gallery")
    st.caption(
        "Browse Grad-CAM montages generated during inference/new analysis. "
        "Fresh predictions generate last-four-layer Grad-CAM images automatically."
    )

    cursor = ctx.cursor

    if "person_id" not in st.session_state:
        st.session_state.person_id = getattr(ctx, "selected_person_id", None)

    selected_person_id = st.session_state.get("person_id") or getattr(ctx, "selected_person_id", None)

    controls = st.columns([1, 1, 1, 1])

    with controls[0]:
        limit = st.number_input(
            "Max result rows to scan",
            min_value=10,
            max_value=5000,
            value=500,
            step=50,
            key="gradcam_scan_limit",
        )

    with controls[1]:
        n_layers = st.number_input(
            "Last N layers",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            key="gradcam_n_layers",
        )

    with controls[2]:
        inference_dir = st.text_input(
            "Inference image folder",
            value="data/datasets/inference",
            key="gradcam_inference_dir",
        )

    with controls[3]:
        use_person_filter = st.checkbox(
            "Filter current person",
            value=selected_person_id is not None,
            key="gradcam_filter_current_person",
        )

    person_filter = int(selected_person_id) if use_person_filter and selected_person_id is not None else None

    reload_clicked = st.button("Reload Grad-CAM gallery", key="gradcam_reload")

    cache_key = f"gradcam_rows_limit_{int(limit)}_person_{person_filter}_layers_{int(n_layers)}"
    if reload_clicked:
        st.session_state.pop(cache_key, None)

    if cache_key not in st.session_state:
        with st.spinner("Loading result rows and checking Grad-CAM files..."):
            rows_df = _fetch_result_rows(ctx, limit=int(limit), person_id=person_filter)
            rows_df = _attach_gradcam_status(rows_df, n_layers=int(n_layers))
            st.session_state[cache_key] = rows_df

    df = st.session_state.get(cache_key, pd.DataFrame())

    if df.empty:
        st.warning("No result rows found. Run inference/new analysis first.")
        return

    _render_summary(df)

    filtered_df = _render_filters(df)

    st.divider()
    _display_table(filtered_df)

    if filtered_df.empty:
        return

    st.divider()
    view_mode = st.radio(
        "View mode",
        ["Observe one image", "Gallery grid"],
        horizontal=True,
        key="gradcam_view_mode",
    )

    if view_mode == "Observe one image":
        options = filtered_df.index.tolist()

        def _option_label(idx):
            row = filtered_df.loc[idx]
            return f"{row.get('Filename')} | pred={row.get('Prediction')} | model_id={row.get('Model ID')} | gradcam={row.get('Grad-CAM Count')}"

        selected_idx = st.selectbox(
            "Select image/result",
            options=options,
            format_func=_option_label,
            key="gradcam_selected_result_idx",
        )

        selected_row = filtered_df.loc[selected_idx]
        _render_selected_observation(selected_row, inference_dir=inference_dir, n_layers=int(n_layers), ctx=ctx)

    else:
        max_items = st.number_input(
            "Max gallery items",
            min_value=1,
            max_value=200,
            value=min(20, len(filtered_df)),
            step=1,
            key="gradcam_max_grid_items",
        )
        _render_grid(filtered_df, inference_dir=inference_dir, n_layers=int(n_layers), max_items=int(max_items))
