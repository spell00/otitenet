# /home/simon/otitenet/otitenet/api/mobile_deployment.py

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any


DEPLOYMENT_ROOT = Path("data/mobile_deployments")
CURRENT_DIR = DEPLOYMENT_ROOT / "current"
CURRENT_MANIFEST = CURRENT_DIR / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def get_current_deployment_manifest() -> Optional[Dict[str, Any]]:
    if not CURRENT_MANIFEST.exists():
        return None

    with CURRENT_MANIFEST.open("r") as f:
        return json.load(f)


def get_deployment_file_path(filename: str) -> Optional[Path]:
    # Prevent path traversal
    safe_name = Path(filename).name
    path = CURRENT_DIR / safe_name

    if not path.exists():
        return None

    return path


def write_current_manifest(manifest: Dict[str, Any]) -> None:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)

    with CURRENT_MANIFEST.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)


def create_simple_torch_classifier_manifest(
    model_id: int,
    model_name: str,
    model_file: str,
    labels: list[str],
    input_size: tuple[int, int] = (224, 224),
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Dict[str, Any]:
    """
    First deployment type:
    - Android downloads one lightweight Torch/ExecuTorch/TorchScript model
    - Android preprocesses image
    - Android runs classifier directly
    """
    model_path = CURRENT_DIR / Path(model_file).name

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in deployment dir: {model_path}")

    manifest = {
        "deployment_id": "current",
        "model_id": int(model_id),
        "model_name": model_name,
        "model_type": "torch_classifier",
        "head_type": "linear_classifier",
        "requires_reference_arrays": False,
        "input": {
            "image_size": list(input_size),
            "channels": 3,
        },
        "preprocessing": {
            "resize": list(input_size),
            "normalize_mean": list(normalize_mean),
            "normalize_std": list(normalize_std),
        },
        "labels": labels,
        "files": {
            "model": model_path.name,
            "manifest": "manifest.json",
        },
        "sha256": {
            "model": sha256_file(model_path),
        },
    }

    write_current_manifest(manifest)
    return manifest


def create_knn_embedding_manifest(
    model_id: int,
    model_name: str,
    embedding_model_file: str,
    reference_embeddings_file: str,
    reference_labels_file: str,
    labels: list[str],
    k: int,
    distance: str = "cosine",
    input_size: tuple[int, int] = (224, 224),
) -> Dict[str, Any]:
    """
    Future deployment type:
    - Android runs embedding model
    - Android loads reference embeddings
    - Android performs KNN locally
    """
    embedding_path = CURRENT_DIR / Path(embedding_model_file).name
    ref_emb_path = CURRENT_DIR / Path(reference_embeddings_file).name
    ref_lab_path = CURRENT_DIR / Path(reference_labels_file).name

    for p in [embedding_path, ref_emb_path, ref_lab_path]:
        if not p.exists():
            raise FileNotFoundError(f"Deployment file not found: {p}")

    manifest = {
        "deployment_id": "current",
        "model_id": int(model_id),
        "model_name": model_name,
        "model_type": "torch_embedding",
        "head_type": "knn",
        "requires_reference_arrays": True,
        "k": int(k),
        "distance": distance,
        "input": {
            "image_size": list(input_size),
            "channels": 3,
        },
        "preprocessing": {
            "resize": list(input_size),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "labels": labels,
        "files": {
            "embedding_model": embedding_path.name,
            "reference_embeddings": ref_emb_path.name,
            "reference_labels": ref_lab_path.name,
            "manifest": "manifest.json",
        },
        "sha256": {
            "embedding_model": sha256_file(embedding_path),
            "reference_embeddings": sha256_file(ref_emb_path),
            "reference_labels": sha256_file(ref_lab_path),
        },
    }

    write_current_manifest(manifest)
    return manifest
