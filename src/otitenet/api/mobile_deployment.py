# /home/simon/otitenet/otitenet/api/mobile_deployment.py

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, Any

from otitenet.data.transforms_manifest import image_preprocessing_manifest


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


def write_current_transforms(preprocessing: Dict[str, Any]) -> str:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    path = CURRENT_DIR / "transforms.json"
    payload = {
        "schema_version": 1,
        "image": preprocessing,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path.name


def _manifest_model_id(model_id) -> str:
    return str(model_id)


def _head_metadata(production_params: Dict[str, Any] | None) -> Dict[str, Any]:
    params = production_params or {}
    return {
        "head_name": params.get("head_name") or params.get("head") or params.get("learned_classifier_label"),
        "head_config": params.get("head_config") or params.get("classification_head_config") or params.get("best_classifier_config"),
        "head_family": params.get("head_family"),
        "head_n_aug": params.get("head_n_aug") or params.get("n_aug") or params.get("N Aug"),
    }


_DISTANCE_ALIAS_KEYS = ("Dist_Fct", "dist_fct", "dist_metric", "Distance")


def _is_present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "None", "none", "nan", "NaN", "—", "null", "NULL"}


def normalized_distance_metadata(
    production_params: Dict[str, Any] | None,
    distance: str | None = None,
    default: str = "euclidean",
) -> tuple[str, Dict[str, Any]]:
    """Return one canonical distance and params with all distance aliases synchronized."""
    params = dict(production_params or {})
    chosen = distance
    if not _is_present(chosen):
        for key in (*_DISTANCE_ALIAS_KEYS, "distance"):
            if _is_present(params.get(key)):
                chosen = params[key]
                break
    if not _is_present(chosen):
        chosen = default

    canonical = str(chosen).strip().lower()
    for key in _DISTANCE_ALIAS_KEYS:
        params[key] = canonical
    return canonical, params


def _base_resize_size_from_production(production_params: Dict[str, Any] | None, default: int = 64) -> int:
    params = production_params or {}
    for key in ("Dataset", "dataset", "path", "Artifact Dataset", "Combo Dataset"):
        value = str(params.get(key) or "").replace("\\", "/")
        match = re.search(r"(?:^|/)otite_ds_(\d+)(?:/|$)", value)
        if match:
            return int(match.group(1))
    return int(default)


def create_simple_torch_classifier_manifest(
    model_id: int,
    model_name: str,
    model_file: str,
    labels: list[str],
    input_size: tuple[int, int] = (224, 224),
    normalize: str = "yes",
    normalize_mean: tuple[float, float, float] | None = None,
    normalize_std: tuple[float, float, float] | None = None,
    production_params: Dict[str, Any] | None = None,
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

    is_onnx = model_path.suffix.lower() == ".onnx"
    _distance, production_params = normalized_distance_metadata(production_params)
    preprocessing = image_preprocessing_manifest(
        input_size,
        normalize=normalize,
        base_resize_size=_base_resize_size_from_production(production_params),
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    transforms_file = write_current_transforms(preprocessing)

    manifest = {
        "deployment_id": "current",
        "model_id": _manifest_model_id(model_id),
        "model_name": model_name,
        "model_type": "onnx_classifier" if is_onnx else "torch_classifier",
        "runtime": "onnxruntime" if is_onnx else "torch",
        "head_type": "linear_classifier",
        **_head_metadata(production_params),
        "requires_reference_arrays": False,
        "input": {
            "image_size": list(input_size),
            "channels": 3,
        },
        "preprocessing": preprocessing,
        "production_params": production_params or {},
        "labels": labels,
        "files": {
            "model": model_path.name,
            "transforms": transforms_file,
            "manifest": "manifest.json",
        },
        "sha256": {
            "model": sha256_file(model_path),
            "transforms": sha256_file(CURRENT_DIR / transforms_file),
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
    normalize: str = "yes",
    production_params: Dict[str, Any] | None = None,
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

    distance, production_params = normalized_distance_metadata(production_params, distance)
    preprocessing = image_preprocessing_manifest(
        input_size,
        normalize=normalize,
        base_resize_size=_base_resize_size_from_production(production_params),
    )
    transforms_file = write_current_transforms(preprocessing)

    manifest = {
        "deployment_id": "current",
        "model_id": _manifest_model_id(model_id),
        "model_name": model_name,
        "model_type": "torch_embedding",
        "head_type": "knn",
        **_head_metadata(production_params),
        "requires_reference_arrays": True,
        "k": int(k),
        "distance": distance,
        "input": {
            "image_size": list(input_size),
            "channels": 3,
        },
        "preprocessing": preprocessing,
        "production_params": production_params or {},
        "labels": labels,
        "files": {
            "embedding_model": embedding_path.name,
            "reference_embeddings": ref_emb_path.name,
            "reference_labels": ref_lab_path.name,
            "transforms": transforms_file,
            "manifest": "manifest.json",
        },
        "sha256": {
            "embedding_model": sha256_file(embedding_path),
            "reference_embeddings": sha256_file(ref_emb_path),
            "reference_labels": sha256_file(ref_lab_path),
            "transforms": sha256_file(CURRENT_DIR / transforms_file),
        },
    }

    write_current_manifest(manifest)
    return manifest


def create_sklearn_embedding_manifest(
    model_id: int,
    model_name: str,
    embedding_model_file: str,
    classifier_file: str,
    labels: list[str],
    input_size: tuple[int, int] = (224, 224),
    normalize: str = "yes",
    production_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create an embedding deployment with a saved sklearn baseline head."""
    embedding_path = CURRENT_DIR / Path(embedding_model_file).name
    classifier_path = CURRENT_DIR / Path(classifier_file).name

    for p in [embedding_path, classifier_path]:
        if not p.exists():
            raise FileNotFoundError(f"Deployment file not found: {p}")

    _distance, production_params = normalized_distance_metadata(production_params)
    preprocessing = image_preprocessing_manifest(
        input_size,
        normalize=normalize,
        base_resize_size=_base_resize_size_from_production(production_params),
    )
    transforms_file = write_current_transforms(preprocessing)

    manifest = {
        "deployment_id": "current",
        "model_id": _manifest_model_id(model_id),
        "model_name": model_name,
        "model_type": "torch_embedding_baseline",
        "runtime": "torch",
        "head_type": "baseline",
        **_head_metadata(production_params),
        "requires_reference_arrays": True,
        "input": {
            "image_size": list(input_size),
            "channels": 3,
        },
        "preprocessing": preprocessing,
        "production_params": production_params or {},
        "labels": labels,
        "files": {
            "embedding_model": embedding_path.name,
            "classifier_head": classifier_path.name,
            "transforms": transforms_file,
            "manifest": "manifest.json",
        },
        "sha256": {
            "embedding_model": sha256_file(embedding_path),
            "classifier_head": sha256_file(classifier_path),
            "transforms": sha256_file(CURRENT_DIR / transforms_file),
        },
    }

    write_current_manifest(manifest)
    return manifest


def create_torch_prototype_manifest(
    model_id: int,
    model_name: str,
    model_file: str,
    prototypes_file: str,
    labels: list[str],
    input_size: tuple[int, int],
    normalize: str,
    distance: str,
    head_config: str,
    production_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create an exact web-compatible deployment for prototype-head inference."""
    model_path = CURRENT_DIR / Path(model_file).name
    prototypes_path = CURRENT_DIR / Path(prototypes_file).name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in deployment dir: {model_path}")
    if not prototypes_path.exists():
        raise FileNotFoundError(f"Prototype file not found in deployment dir: {prototypes_path}")

    distance, production_params = normalized_distance_metadata(production_params, distance)
    preprocessing = image_preprocessing_manifest(
        input_size,
        normalize=normalize,
        base_resize_size=_base_resize_size_from_production(production_params),
    )
    transforms_file = write_current_transforms(preprocessing)

    manifest = {
        "deployment_id": "current",
        "model_id": str(model_id),
        "model_name": model_name,
        "model_type": "torch_prototype",
        "runtime": "torch",
        "head_type": "prototype",
        **_head_metadata(production_params),
        "head_config": str(head_config),
        "distance": distance,
        "requires_reference_arrays": True,
        "input": {
            "image_size": list(input_size),
            "channels": 3,
        },
        "preprocessing": preprocessing,
        "production_params": production_params or {},
        "labels": labels,
        "files": {
            "model": model_path.name,
            "prototypes": prototypes_path.name,
            "transforms": transforms_file,
            "manifest": "manifest.json",
        },
        "sha256": {
            "model": sha256_file(model_path),
            "prototypes": sha256_file(prototypes_path),
            "transforms": sha256_file(CURRENT_DIR / transforms_file),
        },
    }
    write_current_manifest(manifest)
    return manifest


def create_onnx_prototype_manifest(
    model_id: int,
    model_name: str,
    model_file: str,
    prototypes_file: str,
    labels: list[str],
    input_size: tuple[int, int],
    normalize: str,
    distance: str,
    head_config: str,
    quantization: str = "none",
    production_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a compact deployment using ONNX embeddings plus the saved prototype head."""
    model_path = CURRENT_DIR / Path(model_file).name
    prototypes_path = CURRENT_DIR / Path(prototypes_file).name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in deployment dir: {model_path}")
    if not prototypes_path.exists():
        raise FileNotFoundError(f"Prototype file not found in deployment dir: {prototypes_path}")

    distance, production_params = normalized_distance_metadata(production_params, distance)
    preprocessing = image_preprocessing_manifest(
        input_size,
        normalize=normalize,
        base_resize_size=_base_resize_size_from_production(production_params),
    )
    transforms_file = write_current_transforms(preprocessing)

    manifest = {
        "deployment_id": "current",
        "model_id": _manifest_model_id(model_id),
        "model_name": model_name,
        "model_type": "onnx_embedding_prototype",
        "runtime": "onnxruntime",
        "head_type": "prototype",
        **_head_metadata(production_params),
        "head_config": str(head_config),
        "distance": distance,
        "quantization": str(quantization or "none"),
        "requires_reference_arrays": True,
        "input": {
            "image_size": list(input_size),
            "channels": 3,
        },
        "preprocessing": preprocessing,
        "production_params": production_params or {},
        "labels": labels,
        "files": {
            "model": model_path.name,
            "prototypes": prototypes_path.name,
            "transforms": transforms_file,
            "manifest": "manifest.json",
        },
        "sha256": {
            "model": sha256_file(model_path),
            "prototypes": sha256_file(prototypes_path),
            "transforms": sha256_file(CURRENT_DIR / transforms_file),
        },
    }
    write_current_manifest(manifest)
    return manifest
