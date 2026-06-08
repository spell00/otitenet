
# /home/simon/otitenet/otitenet/app/pages/raw_pixel_classification.py

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from otitenet.app.display_metrics import _arrow_safe_dataframe
from otitenet.app.model_loading import load_model_and_prototypes
from otitenet.data.data_getters import get_images_loaders


def _ensure_args(args):
    if not hasattr(args, "bs"):
        args.bs = 32
    if not hasattr(args, "groupkfold"):
        args.groupkfold = 1
    if not hasattr(args, "random_recs"):
        args.random_recs = 0
    if not hasattr(args, "prototypes_to_use"):
        args.prototypes_to_use = "class"
    if not hasattr(args, "normalize"):
        args.normalize = "no"
    return args


def _extract_batch_xy(batch):
    """
    Try to extract image tensor and labels from common loader formats.
    """
    x = None
    y = None

    if isinstance(batch, dict):
        for key in ["inputs", "input", "x", "image", "images", "data"]:
            if key in batch:
                x = batch[key]
                break
        for key in ["cats", "labels", "label", "y", "targets", "target"]:
            if key in batch:
                y = batch[key]
                break
    elif isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            x, y = batch[0], batch[1]

    if x is None or y is None:
        return None, None

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim == 4:
        x = x.reshape(x.shape[0], -1)
    else:
        x = x.reshape(x.shape[0], -1)

    return x, y.reshape(-1)


def _load_raw_pixels(args, max_per_split: int = 2000):
    args = _ensure_args(args)

    with st.spinner("Loading data through existing OtiteNet data loader..."):
        _model, _shap, prototypes, _image_size, _device, data, unique_labels, unique_batches, _data_getter = load_model_and_prototypes(args)
        batch_encoder = LabelEncoder().fit(np.asarray(unique_batches))

        loaders = get_images_loaders(
            data=data,
            batch_encoder=batch_encoder,
            random_recs=getattr(args, "random_recs", 0),
            weighted_sampler=0,
            is_transform=0,
            samples_weights=None,
            epoch=1,
            unique_labels=unique_labels,
            triplet_dloss=getattr(args, "dloss", "triplet"),
            bs=getattr(args, "bs", 32),
            prototypes_to_use=getattr(args, "prototypes_to_use", "class"),
            prototypes=prototypes,
            size=getattr(args, "new_size", 64),
            normalize=getattr(args, "normalize", "no"),
        )

    out = {}

    for split in ["train", "valid", "test"]:
        if split not in loaders:
            continue

        xs = []
        ys = []
        seen = 0

        with st.spinner(f"Collecting raw pixels for {split}..."):
            for batch in loaders[split]:
                x, y = _extract_batch_xy(batch)
                if x is None:
                    continue

                xs.append(x)
                ys.append(y)
                seen += len(y)

                if seen >= int(max_per_split):
                    break

        if xs:
            X = np.concatenate(xs, axis=0)[: int(max_per_split)]
            y = np.concatenate(ys, axis=0)[: int(max_per_split)]
            out[split] = {"X": X, "y": y}

    if "train" not in out:
        raise RuntimeError("Could not collect raw train pixels from loaders.")

    return out


def _make_model(name: str):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    if name == "ridge":
        return make_pipeline(StandardScaler(with_mean=False), RidgeClassifier())
    if name == "linear_svc":
        return make_pipeline(StandardScaler(with_mean=False), LinearSVC(max_iter=5000))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1, max_depth=None)
    if name == "knn":
        return make_pipeline(StandardScaler(with_mean=False), KNeighborsClassifier(n_neighbors=5))
    raise ValueError(f"Unknown raw classifier: {name}")


def _fit_eval_raw_classifier(data, classifier_name: str):
    clf = _make_model(classifier_name)

    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    clf.fit(X_train, y_train)

    rows = []
    predictions = {}

    for split in ["train", "valid", "test"]:
        if split not in data:
            continue

        X = data[split]["X"]
        y = data[split]["y"]
        pred = clf.predict(X)

        rows.append(
            {
                "Split": split,
                "N": len(y),
                "Accuracy": float(accuracy_score(y, pred)),
                "MCC": float(matthews_corrcoef(y, pred)),
            }
        )

        predictions[split] = pd.DataFrame({"label": y, "prediction": pred})

    return clf, pd.DataFrame(rows), predictions


def _preprocess_uploaded_image(file_obj, size: int):
    img = Image.open(file_obj).convert("RGB")
    img = img.resize((int(size), int(size)))
    arr = np.asarray(img).astype("float32") / 255.0
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    return img, arr.reshape(1, -1)


def _render_pca(data):
    if "train" not in data:
        return

    st.subheader("Raw pixel PCA")

    X = data["train"]["X"]
    y = data["train"]["y"]

    max_samples = min(1000, len(X))
    X_sub = X[:max_samples]
    y_sub = y[:max_samples]

    with st.spinner("Computing PCA on raw pixels..."):
        pca = PCA(n_components=2, random_state=1)
        coords = pca.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y_sub, alpha=0.7, s=20)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Raw pixel PCA")
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Class")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render(ctx: Any) -> None:
    st.header("🔬 Raw Pixel Classification")
    st.caption(
        "Train simple baseline classifiers directly on flattened image pixels. "
        "This is useful as a sanity check against the learned embedding models."
    )

    args = ctx.args

    c1, c2, c3 = st.columns(3)

    with c1:
        classifier_name = st.selectbox(
            "Raw classifier",
            ["logistic_regression", "ridge", "linear_svc", "random_forest", "knn"],
            index=0,
            key="raw_pixel_classifier",
        )

    with c2:
        max_per_split = st.number_input(
            "Max samples per split",
            min_value=50,
            max_value=10000,
            value=2000,
            step=50,
            key="raw_pixel_max_per_split",
        )

    with c3:
        show_pca = st.checkbox(
            "Show raw PCA",
            value=True,
            key="raw_pixel_show_pca",
        )

    st.markdown(
        f"**Dataset:** `{getattr(args, 'path', '—')}` | "
        f"**Size:** `{getattr(args, 'new_size', '—')}` | "
        f"**Normalize:** `{getattr(args, 'normalize', '—')}`"
    )

    if st.button("🚀 Train raw-pixel baseline", key="raw_pixel_train", type="primary"):
        try:
            data = _load_raw_pixels(args, max_per_split=int(max_per_split))
            clf, metrics_df, predictions = _fit_eval_raw_classifier(data, classifier_name)

            st.session_state["raw_pixel_data"] = data
            st.session_state["raw_pixel_model"] = clf
            st.session_state["raw_pixel_metrics"] = metrics_df
            st.session_state["raw_pixel_predictions"] = predictions

            st.success("Raw-pixel baseline trained.")
        except Exception as exc:
            st.error(f"Raw-pixel training failed: {exc}")
            st.code(traceback.format_exc())

    metrics_df = st.session_state.get("raw_pixel_metrics")
    data = st.session_state.get("raw_pixel_data")
    clf = st.session_state.get("raw_pixel_model")

    if metrics_df is not None:
        st.subheader("Metrics")
        st.dataframe(_arrow_safe_dataframe(metrics_df), use_container_width=True)

    if show_pca and data is not None:
        _render_pca(data)

    st.divider()
    st.subheader("Predict uploaded image with raw-pixel baseline")

    if clf is None:
        st.info("Train a raw-pixel baseline first.")
        return

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key="raw_pixel_upload_predict",
    )

    if uploaded is not None:
        img, X = _preprocess_uploaded_image(uploaded, size=int(getattr(args, "new_size", 64)))
        pred = clf.predict(X)[0]

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, caption=uploaded.name, use_container_width=True)
        with c2:
            st.metric("Raw-pixel prediction", str(pred))
