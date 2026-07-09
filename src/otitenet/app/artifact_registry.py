"""Stable CSV-backed registries for data and best-model artifacts."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st


DATASET_REGISTRY_PATH = Path("configs/datasets.csv")
BEST_MODELS_REGISTRY_PATH = Path("configs/best_models.csv")


def _repo_rel(path: str | Path) -> str:
    return os.path.normpath(str(path)).replace("\\", "/").lstrip("./")


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


@st.cache_data(ttl=300)
def scan_dataset_registry(data_dir: str | Path = "data") -> pd.DataFrame:
    """Scan raw and processed datasets into a stable tabular registry."""
    data_root = Path(data_dir)
    rows: list[dict[str, object]] = []

    raw_root = data_root / "datasets"
    ignored_raw = {"Banque_Comert_Turquie_2020", "Banque_Calaman_USA_2020"}
    if raw_root.is_dir():
        for entry in sorted(raw_root.iterdir()):
            if not entry.is_dir():
                continue
            rel_path = _repo_rel(entry)
            rows.append(
                {
                    "dataset_id": f"raw:{entry.name}",
                    "kind": "raw",
                    "path": rel_path,
                    "infos_csv": "",
                    "pixel_size": "",
                    "sources": entry.name,
                    "has_inference": _bool_text(entry.name == "inference"),
                    "n_samples": "",
                    "ignored": _bool_text(entry.name in ignored_raw),
                    "dvc_target": rel_path,
                    "description": "",
                }
            )

    if data_root.is_dir():
        for infos_path in sorted(data_root.rglob("infos.csv")):
            dataset_dir = infos_path.parent
            rel_path = _repo_rel(dataset_dir)
            match = re.search(r"(?:^|/)otite_ds_(\d+)(?:/|$)", rel_path)
            pixel_size = match.group(1) if match else ""
            sources = dataset_dir.name if dataset_dir.name != f"otite_ds_{pixel_size}" else ""
            n_samples = ""
            try:
                # Header-aware count without loading image data.
                n_samples = max(0, sum(1 for _ in infos_path.open("r", encoding="utf-8")) - 1)
            except Exception:
                pass
            rows.append(
                {
                    "dataset_id": f"processed:{rel_path}",
                    "kind": "processed",
                    "path": rel_path,
                    "infos_csv": _repo_rel(infos_path),
                    "pixel_size": pixel_size,
                    "sources": sources,
                    "has_inference": _bool_text("inference" in rel_path.lower()),
                    "n_samples": n_samples,
                    "ignored": "no",
                    "dvc_target": rel_path,
                    "description": "",
                }
            )

    return pd.DataFrame(rows).sort_values(["kind", "path"]).reset_index(drop=True)


@st.cache_data(ttl=300)
def load_dataset_registry(data_dir: str | Path = "data") -> pd.DataFrame:
    if DATASET_REGISTRY_PATH.exists():
        return pd.read_csv(DATASET_REGISTRY_PATH, dtype=str).fillna("")
    return scan_dataset_registry(data_dir)


def available_dataset_paths_from_registry(data_dir: str | Path = "data") -> list[str]:
    df = load_dataset_registry(data_dir)
    if df.empty or "path" not in df.columns:
        return []
    rows = df[(df.get("kind", "") == "processed") & (df.get("ignored", "no") != "yes")]
    paths = []
    for value in rows["path"].astype(str):
        value = value.replace("\\", "/")
        for prefix in ("./data/", "data/"):
            if value.startswith(prefix):
                value = value[len(prefix) :]
                break
        if value:
            paths.append(value)
    return sorted(dict.fromkeys(paths))


def _parse_model_artifact_path(model_path: Path) -> dict[str, object] | None:
    rel_model = _repo_rel(model_path)
    parts = rel_model.split("/")
    try:
        base_idx = parts.index("best_models")
        nsize_idx = next(i for i in range(base_idx + 3, len(parts)) if parts[i].startswith("nsize"))
    except (ValueError, StopIteration):
        return None

    task = parts[base_idx + 1] if len(parts) > base_idx + 1 else ""
    model_name = parts[base_idx + 2] if len(parts) > base_idx + 2 else ""
    dataset_path = "/".join(parts[base_idx + 3 : nsize_idx])
    tail = parts[nsize_idx:]

    def tail_value(prefix: str) -> str:
        for segment in tail:
            if segment.startswith(prefix):
                return segment[len(prefix) :]
        return ""

    protoagg = tail_value("protoagg_")
    prototype_strategy = ""
    prototype_components = ""
    if protoagg:
        proto_parts = protoagg.rsplit("_", 1)
        prototype_strategy = proto_parts[0]
        prototype_components = proto_parts[1] if len(proto_parts) == 2 else ""

    model_dir = model_path.parent
    artifact_id = hashlib.sha1(_repo_rel(model_dir).encode("utf-8")).hexdigest()[:12]
    return {
        "artifact_id": artifact_id,
        "task": task,
        "model_name": model_name,
        "dataset_path": dataset_path,
        "nsize": tail_value("nsize"),
        "fgsm": tail_value("fgsm"),
        "n_calibration": tail_value("ncal"),
        "classif_loss": tail[3] if len(tail) > 3 else "",
        "dloss": tail[4] if len(tail) > 4 else "",
        "prototypes": tail_value("prototypes_"),
        "npos": tail_value("npos"),
        "nneg": tail_value("nneg"),
        "prototype_strategy": prototype_strategy,
        "prototype_components": prototype_components,
        "normalize": tail_value("norm"),
        "dist_fct": tail_value("dist_"),
        "n_neighbors": tail_value("knn"),
        "model_dir": _repo_rel(model_dir),
        "model_path": rel_model,
        "prototypes_path": _repo_rel(model_dir / "prototypes.pkl"),
        "train_encodings_path": _repo_rel(model_dir / "train_encodings.npz"),
        "valid_encodings_path": _repo_rel(model_dir / "valid_encodings.npz"),
        "test_encodings_path": _repo_rel(model_dir / "test_encodings.npz"),
        "run_log_path": "",
        "is_best": "yes",
        "notes": "",
    }


def scan_best_models_registry(log_root: str | Path = "logs/best_models") -> pd.DataFrame:
    rows = []
    root = Path(log_root)
    if root.is_dir():
        for model_path in sorted(root.rglob("model.pth")):
            parsed = _parse_model_artifact_path(model_path)
            if parsed:
                rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = _attach_source_run_paths(df, root)
    return df.sort_values(["task", "model_name", "dataset_path", "model_dir"]).reset_index(drop=True)


def _strip_token_prefix(value: object, prefix: str) -> str:
    text = str(value or "").strip()
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _model_csv_key(row: pd.Series | dict[str, object]) -> tuple[str, ...]:
    get = row.get
    fgsm = str(get("fgsm", "") or "").strip()
    if fgsm.startswith("fsgm"):
        fgsm = "fgsm" + fgsm[len("fsgm") :]
    return (
        str(get("task", "") or "").strip(),
        str(get("model_name", get("model", "")) or "").strip(),
        str(get("dataset_path", get("path", "")) or "").strip().replace("\\", "/"),
        _strip_token_prefix(get("nsize", ""), "nsize"),
        _strip_token_prefix(fgsm, "fgsm"),
        _strip_token_prefix(get("n_calibration", get("n_calibration", "")), "ncal"),
        str(get("classif_loss", get("loss", "")) or "").strip(),
        str(get("dloss", "") or "").strip(),
        _strip_token_prefix(get("dist_fct", ""), "dist_"),
        _strip_token_prefix(get("prototypes", get("prototype", "")), "prototypes_"),
        _strip_token_prefix(get("npos", get("n_positives", "")), "npos"),
        _strip_token_prefix(get("nneg", get("n_negatives", "")), "nneg"),
        str(get("normalize", "") or "").strip(),
        str(get("n_neighbors", "") or "").strip(),
    )


def _attach_source_run_paths(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Attach original logs/<task>/<run-id> paths from task-level models.csv files."""
    source_by_key: dict[tuple[str, ...], str] = {}
    for models_csv in sorted(root.glob("*/models.csv")):
        try:
            models_df = pd.read_csv(models_csv, dtype=str).fillna("")
        except Exception:
            continue
        if "complete_log_path" not in models_df.columns:
            continue
        for _, row in models_df.iterrows():
            key = _model_csv_key(row)
            source_by_key.setdefault(key, str(row.get("complete_log_path", "")).strip())

    if not source_by_key:
        return df

    df = df.copy()
    for idx, row in df.iterrows():
        source = source_by_key.get(_model_csv_key(row))
        if source:
            df.at[idx, "run_log_path"] = source
    return df


@st.cache_data(ttl=300)
def load_best_models_registry(log_root: str | Path = "logs/best_models") -> pd.DataFrame:
    scanned = scan_best_models_registry(log_root)
    if not BEST_MODELS_REGISTRY_PATH.exists():
        return scanned

    try:
        stored = pd.read_csv(BEST_MODELS_REGISTRY_PATH, dtype=str).fillna("")
    except Exception:
        return scanned

    if stored.empty:
        return scanned
    if scanned.empty:
        return stored

    merged = pd.concat([stored, scanned], ignore_index=True, sort=False).fillna("")
    if "model_dir" in merged.columns:
        merged = merged.drop_duplicates(subset=["model_dir"], keep="last")
    elif "model_path" in merged.columns:
        merged = merged.drop_duplicates(subset=["model_path"], keep="last")
    else:
        merged = merged.drop_duplicates(keep="last")
    sort_cols = [c for c in ["task", "model_name", "dataset_path", "model_dir"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols)
    return merged.reset_index(drop=True)


def has_model_artifacts(path: str | Path) -> bool:
    """Return True when a directory contains the model files needed for inference."""
    if not path:
        return False
    root = Path(path)
    return (root / "model.pth").is_file() and (root / "prototypes.pkl").is_file()


def is_source_run_artifact_dir(path: str | Path) -> bool:
    """Return True for primary run artifact dirs: logs/<task>/<uuid>."""
    if not path:
        return False
    text = str(path).strip().replace("\\", "/").rstrip("/")
    parts = text.split("/")
    if "logs" in parts:
        parts = parts[parts.index("logs") :]
    return (
        len(parts) >= 3
        and parts[0] == "logs"
        and parts[1] != "best_models"
        and has_model_artifacts(text)
    )


def allow_legacy_best_models_artifacts() -> bool:
    return str(os.environ.get("OTITENET_ALLOW_LEGACY_BEST_MODELS", "")).strip().lower() in {"1", "true", "yes"}


def preferred_model_artifact_dir(row: dict[str, object]) -> str:
    """Prefer original source-run artifacts; best_models is legacy opt-in only."""
    for key in ("source_run_log_path", "Source Run Path", "run_log_path", "Run Log Path"):
        value = str(row.get(key, "") or "").strip()
        if value and is_source_run_artifact_dir(value):
            return value
    for key in ("log_path", "Log Path"):
        value = str(row.get(key, "") or "").strip()
        if value and is_source_run_artifact_dir(value):
            return value
    if not allow_legacy_best_models_artifacts():
        return ""
    for key in ("best_model_dir", "Best Model Dir", "model_dir", "Model Dir", "log_path", "Log Path"):
        value = str(row.get(key, "") or "").strip()
        if value and has_model_artifacts(value):
            return value
    return ""
