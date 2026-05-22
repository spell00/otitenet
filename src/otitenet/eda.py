import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

from otitenet.data.data_getters import PerImageNormalize


def load_infos(dataset_name: str, data_root: str = "data") -> pd.DataFrame:
    info_path = Path(data_root) / dataset_name / "infos.csv"
    if not info_path.exists():
        raise FileNotFoundError(f"infos.csv not found at {info_path}")
    return pd.read_csv(info_path)


def sample_raw_for_pca(
    dataset_name: str,
    data_root: str = "data",
    sample_size: int = 200,
    resize: int = 64,
    label_col: str = "label",
    shape_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load a sample of raw images from data/<dataset_name> and return PCA-ready DF.

    label_col chooses which column from infos.csv to use for coloring (falls back to 'label').
    shape_col (optional) chooses which column to use for marker shapes; stored in 'shape'.
    """
    df = load_infos(dataset_name, data_root=data_root)
    if df.empty:
        raise ValueError("infos.csv is empty; cannot run PCA")

    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    rows = []
    arrays = []
    base_dir = Path(data_root) / dataset_name
    for _, row in sample_df.iterrows():
        fname = str(row.get("name"))
        label_color = row.get(label_col, row.get("label", "unknown"))
        # Preserve the original label column (if it exists) so shapes can use true labels
        label_true = row.get("label", label_color)
        shape_val = row.get(shape_col, label_true) if shape_col else None
        img_path = base_dir / fname
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        if resize and resize > 0:
            img = img.resize((resize, resize))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arrays.append(arr.flatten())
        rows.append({
            "file": fname,
            "label": label_true,
            "color": label_color,
            "shape": shape_val,
            "label_col": label_col,
            "shape_col": shape_col,
        })

    if not arrays:
        raise ValueError("No images found for PCA sample")

    stack = np.stack(arrays)
    n_components = min(3, stack.shape[0], stack.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(stack)
    comp_cols = [f"pc{i+1}" for i in range(comps.shape[1])]
    comp_df = pd.DataFrame(comps, columns=comp_cols)
    meta_df = pd.DataFrame(rows)
    out_df = pd.concat([meta_df.reset_index(drop=True), comp_df], axis=1)
    return out_df, pca.explained_variance_ratio_


def _flatten_proto(proto_arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(proto_arr)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr.astype(np.float32)


def prototype_pca(
    model_rows: List[dict],
    n_components: int = 2,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run PCA across class prototypes from multiple models.

    model_rows entries must provide 'Model Name' and 'Log Path'.
    """
    vectors = []
    meta = []
    for row in model_rows:
        log_path = Path(str(row.get("Log Path", "")))
        model_name = row.get("Model Name", "model")
        if not log_path.exists():
            continue
        proto_file = log_path / "prototypes.pkl"
        if not proto_file.exists():
            continue
        try:
            import pickle

            with open(proto_file, "rb") as f:
                proto_obj = pickle.load(f)
            class_protos = getattr(proto_obj, "class_prototypes", {}) or {}
            for lbl, proto_arr in class_protos.items():
                vec = _flatten_proto(proto_arr)
                vectors.append(vec)
                meta.append({"model": model_name, "class": str(lbl)})
        except Exception:
            continue

    if not vectors:
        raise ValueError("No prototypes available for PCA")

    mat = np.vstack(vectors)
    n_comp = min(n_components, mat.shape[0], mat.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    comps = pca.fit_transform(mat)
    comp_cols = [f"pc{i+1}" for i in range(comps.shape[1])]
    comp_df = pd.DataFrame(comps, columns=comp_cols)
    meta_df = pd.DataFrame(meta)
    out_df = pd.concat([meta_df.reset_index(drop=True), comp_df], axis=1)
    return out_df, pca.explained_variance_ratio_


def model_representation_pca(
    model: torch.nn.Module,
    image_paths: List[Path],
    labels: List[str],
    size: int,
    normalize: str = "no",
    device: str = "cpu",
    batch_size: int = 32,
    n_components: int = 2,
    label_values: Optional[List[str]] = None,
    color_values: Optional[List[str]] = None,
    shape_values: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run PCA on embeddings produced by a model for given images.

    label_values/color_values/shape_values provide metadata for plotting; lengths must match image_paths.
    """
    if len(image_paths) == 0:
        raise ValueError("No images provided for model PCA")

    ops = [transforms.Resize((size, size)), transforms.ToTensor()]
    if str(normalize).lower() in ["yes", "true", "1"]:
        ops.append(PerImageNormalize())
    transform = transforms.Compose(ops)

    tensors = []
    kept_labels = []
    kept_colors = []
    kept_shapes = []
    kept_files = []

    label_values = label_values or labels
    color_values = color_values or label_values
    # If shapes are not provided, keep an empty list so we can zip safely
    if shape_values is None:
        shape_values = [None] * len(image_paths)

    for pth, lbl, color_val, shape_val in zip(image_paths, label_values, color_values, shape_values):
        try:
            img = Image.open(pth).convert("RGB")
            tensors.append(transform(img))
            kept_labels.append(lbl)
            kept_colors.append(color_val)
            kept_shapes.append(shape_val)
            kept_files.append(pth.name)
        except Exception:
            continue

    if not tensors:
        raise ValueError("Failed to load any images for PCA")

    stack = torch.stack(tensors).to(device)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, stack.size(0), batch_size):
            batch = stack[start:start + batch_size]
            out = model(batch)
            if isinstance(out, tuple) or isinstance(out, list):
                emb = out[0]
            else:
                emb = out
            if emb.ndim > 2:
                emb = torch.flatten(emb, start_dim=1)
            embeddings.append(emb.cpu())
    emb_mat = torch.cat(embeddings, dim=0).numpy()

    n_comp = min(n_components, emb_mat.shape[0], emb_mat.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    comps = pca.fit_transform(emb_mat)
    comp_cols = [f"pc{i+1}" for i in range(comps.shape[1])]
    comp_df = pd.DataFrame(comps, columns=comp_cols)
    meta_df = pd.DataFrame({
        "file": kept_files,
        "label": kept_labels,
        "color": kept_colors,
        "shape": kept_shapes,
    })
    out_df = pd.concat([meta_df.reset_index(drop=True), comp_df], axis=1)
    return out_df, pca.explained_variance_ratio_
