from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from otitenet.app.image_processing import (
    image_to_chw_array,
    preprocess_image_array as app_preprocess_image_array,
)
from otitenet.offline.deployment import OfflineDeployment


class OnnxClassifier:
    def __init__(self, path: Path):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict_logits(self, array: np.ndarray) -> np.ndarray:
        output = self.session.run(None, {self.input_name: array})[0]
        return np.asarray(output, dtype=np.float32)


class OnnxEmbeddingModel:
    def __init__(self, path: Path):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict_embedding(self, array: np.ndarray) -> np.ndarray:
        output = self.session.run(None, {self.input_name: array})[0]
        embedding = np.asarray(output, dtype=np.float32)
        if embedding.ndim > 2:
            embedding = embedding.reshape(embedding.shape[0], -1)
        return embedding


class PrototypeHeadModel:
    def __init__(self, embedding_model: Any, prototypes: dict[str, np.ndarray]):
        self.embedding_model = embedding_model
        self.prototypes = prototypes

    def predict_embedding(self, array: np.ndarray) -> np.ndarray:
        return self.embedding_model.predict_embedding(array)


class SklearnHeadModel:
    def __init__(self, embedding_model: Any, classifier: Any, head_labels: list[str]):
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.head_labels = [str(label) for label in head_labels]

    def predict_embedding(self, array: np.ndarray) -> np.ndarray:
        return self.embedding_model.predict_embedding(array)


def load_model(deployment: OfflineDeployment, device: str = "cpu"):
    path = deployment.model_file
    if deployment.model_type in {"onnx_embedding_baseline", "onnx_baseline"}:
        classifier, head_labels = _load_classifier_head(deployment)
        return SklearnHeadModel(OnnxEmbeddingModel(path), classifier, head_labels)

    if deployment.model_type in {"torch_embedding_baseline", "torch_baseline"}:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "This baseline-head deployment requires PyTorch unless it is exported "
                "to an ONNX embedding deployment."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "PyTorch could not be imported from this exact desktop runtime. "
                "Rebuild and reinstall the exact Tauri package."
            ) from exc

        torch_runtime = importlib.import_module("otitenet.offline.torch_runtime")
        torch_model = torch_runtime.load_state_dict_model(path, deployment, device)
        classifier, head_labels = _load_classifier_head(deployment)
        return SklearnHeadModel(TorchEmbeddingModel(torch_model, device), classifier, head_labels)

    if deployment.model_type in {"onnx_prototype", "onnx_embedding_prototype"}:
        return PrototypeHeadModel(OnnxEmbeddingModel(path), _load_prototypes(deployment))

    if deployment.model_type in {"torch_prototype", "torch_embedding_prototype"}:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "This exact production deployment requires PyTorch because the web app "
                "uses a PyTorch embedding model plus a prototype/KNN head."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "PyTorch could not be imported from this exact desktop runtime. "
                "The bundled torch shared libraries are likely stale or mismatched. "
                "Rebuild and reinstall the exact Tauri package so libtorch_python.so, "
                "libtorch_cpu.so, and the torch Python package come from the same build."
            ) from exc

        torch_runtime = importlib.import_module("otitenet.offline.torch_runtime")
        torch_model = torch_runtime.load_state_dict_model(path, deployment, device)
        return PrototypeHeadModel(TorchEmbeddingModel(torch_model, device), _load_prototypes(deployment))

    if deployment.model_type == "onnx_classifier" or path.suffix.lower() == ".onnx":
        return OnnxClassifier(path)

    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "This offline deployment still points to a PyTorch model. "
                "Run scripts/export_offline_onnx_model.py before building the "
                "desktop app so the packaged runtime can use ONNX Runtime."
            ) from exc
        raise
    except Exception as exc:
        raise RuntimeError(
            "PyTorch could not be imported from this exact desktop runtime. "
            "The bundled torch shared libraries are likely stale or mismatched. "
            "Rebuild and reinstall the exact Tauri package."
        ) from exc

    try:
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        return model
    except Exception:
        torch_runtime = importlib.import_module("otitenet.offline.torch_runtime")
        return torch_runtime.load_state_dict_model(path, deployment, device)


class TorchEmbeddingModel:
    def __init__(self, model: Any, device: str = "cpu"):
        self.model = model
        self.device = device

    def predict_embedding(self, array: np.ndarray) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(array).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        if output.ndim > 2:
            output = output.reshape(output.shape[0], -1)
        return output.detach().cpu().numpy().astype(np.float32)


def _load_prototypes(deployment: OfflineDeployment) -> dict[str, np.ndarray]:
    files = deployment.manifest.get("files", {})
    prototype_name = files.get("prototypes") or files.get("prototype_file")
    if not prototype_name:
        raise ValueError("Prototype deployment manifest does not define a prototype file.")

    prototype_path = deployment.root / Path(prototype_name).name
    if not prototype_path.exists():
        raise FileNotFoundError(f"Prototype file not found: {prototype_path}")

    payload = np.load(prototype_path, allow_pickle=True)
    labels = payload["labels"]
    vectors = payload["prototypes"]
    prototypes: dict[str, np.ndarray] = {}
    for label, vector in zip(labels, vectors):
        label_text = str(label.item() if hasattr(label, "item") else label)
        vector_array = np.asarray(vector, dtype=np.float32)
        if vector_array.ndim == 1:
            vector_array = vector_array[None, :]
        prototypes[label_text] = vector_array
    return prototypes


def _load_classifier_head(deployment: OfflineDeployment) -> tuple[Any, list[str]]:
    files = deployment.manifest.get("files", {})
    classifier_name = files.get("classifier_head") or files.get("head") or files.get("classifier")
    if not classifier_name:
        raise ValueError("Baseline deployment manifest does not define a classifier head file.")

    classifier_path = deployment.root / Path(classifier_name).name
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier head file not found: {classifier_path}")

    try:
        import joblib

        payload = joblib.load(classifier_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading a baseline classifier head requires joblib.") from exc

    if isinstance(payload, dict) and "classifier" in payload:
        classifier = payload["classifier"]
        labels = payload.get("labels")
    else:
        classifier = payload
        labels = deployment.labels

    if labels is None or len(labels) == 0:
        labels = deployment.labels
    return classifier, [str(label) for label in labels]


def preprocess_image_array(image: Image.Image, deployment: OfflineDeployment) -> np.ndarray:
    preprocessing = deployment.manifest.get("preprocessing", {})
    normalization = str(preprocessing.get("normalization", "")).lower()
    normalize = preprocessing.get("normalize")
    resize_mode = str(preprocessing.get("resize_mode", "app_64_then_size")).lower()
    base_match = re.match(r"app_(\d+)_then_size", resize_mode)
    base_resize_size = int(preprocessing.get("base_resize_size") or (base_match.group(1) if base_match else 64))
    if normalization == "per_image" or (
        not normalization and str(normalize).lower() in {"yes", "true", "1", "per_image"}
    ):
        height, width = deployment.input_size
        if height != width:
            raise ValueError(f"App-compatible preprocessing expects square input, got {height}x{width}.")
        return app_preprocess_image_array(image, height, normalize="per_image", base_size=base_resize_size)

    mean = preprocessing.get("normalize_mean")
    std = preprocessing.get("normalize_std")
    if normalization in {"none", "no"} and not (mean and std):
        height, width = deployment.input_size
        if height != width:
            raise ValueError(f"App-compatible preprocessing expects square input, got {height}x{width}.")
        return app_preprocess_image_array(image, height, normalize="no", base_size=base_resize_size)

    height, width = deployment.input_size
    if resize_mode.startswith("app_") and resize_mode.endswith("_then_size"):
        if height != width:
            raise ValueError(f"App-compatible preprocessing expects square input, got {height}x{width}.")
        array = image_to_chw_array(image, height, normalize="no", base_size=base_resize_size)
    else:
        image = image.convert("RGB").resize((width, height))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = array.transpose(2, 0, 1)


    if mean and std:
        mean_array = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        std_array = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        array = (array - mean_array) / std_array

    return array[None, ...].astype(np.float32)


def preprocess_image_tensor(image: Image.Image, deployment: OfflineDeployment, device: str = "cpu"):
    import torch

    return torch.from_numpy(preprocess_image_array(image, deployment)).to(device)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def _prototype_probabilities(
    embedding: np.ndarray,
    prototypes: dict[str, np.ndarray],
    labels: list[str],
    dist_fct_name: str,
) -> np.ndarray:
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    distances = []
    for label in labels:
        proto = prototypes.get(str(label))
        if proto is None:
            distances.append(np.inf)
            continue
        proto = np.asarray(proto, dtype=np.float32)
        if proto.ndim == 1:
            proto = proto[None, :]
        if str(dist_fct_name).lower() == "cosine":
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            proto_norm = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
            distances.append(float(np.min(1.0 - np.matmul(emb_norm, proto_norm.T))))
        else:
            diff = emb[:, None, :] - proto[None, :, :]
            distances.append(float(np.min(np.sqrt(np.sum(diff * diff, axis=2)))))
    distances_array = np.asarray(distances, dtype=np.float64)

    finite_mask = np.isfinite(distances_array)
    if not np.any(finite_mask):
        return np.zeros(len(labels), dtype=np.float64)

    finite_distances = distances_array[finite_mask]
    temperature = float(np.std(finite_distances))
    if temperature <= 1e-8:
        temperature = 1.0

    logits = np.full(len(labels), -np.inf, dtype=np.float64)
    logits[finite_mask] = -distances_array[finite_mask] / temperature
    logits = logits - np.max(logits[finite_mask])
    exp = np.zeros(len(labels), dtype=np.float64)
    exp[finite_mask] = np.exp(logits[finite_mask])
    total = float(exp.sum())
    if total <= 0:
        return np.zeros(len(labels), dtype=np.float64)
    return exp / total


def _classifier_probabilities(classifier: Any, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(classifier, "predict_proba"):
        probs = np.asarray(classifier.predict_proba(embedding), dtype=np.float64)
        classes = np.asarray(getattr(classifier, "classes_", np.arange(probs.shape[1])))
        return probs[0], classes

    if hasattr(classifier, "decision_function"):
        scores = np.asarray(classifier.decision_function(embedding), dtype=np.float64)
        classes = np.asarray(getattr(classifier, "classes_", np.arange(scores.shape[-1] if scores.ndim > 1 else 2)))
        if scores.ndim == 1:
            pos = 1.0 / (1.0 + np.exp(-scores))
            return np.asarray([1.0 - pos[0], pos[0]], dtype=np.float64), classes
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(scores)
        return (exp / np.sum(exp, axis=1, keepdims=True))[0], classes

    pred = classifier.predict(embedding)[0]
    classes = np.asarray(getattr(classifier, "classes_", [pred]))
    probs = np.zeros(len(classes), dtype=np.float64)
    matches = np.where(classes == pred)[0]
    probs[int(matches[0]) if len(matches) else 0] = 1.0
    return probs, classes


def _align_classifier_probabilities(
    probs: np.ndarray,
    classes: np.ndarray,
    head_labels: list[str],
    manifest_labels: list[str],
) -> np.ndarray:
    class_to_label: dict[str, str] = {}
    for cls in classes:
        class_key = str(cls.item() if hasattr(cls, "item") else cls)
        label = class_key
        try:
            idx = int(float(class_key))
            if 0 <= idx < len(head_labels):
                label = head_labels[idx]
        except (TypeError, ValueError):
            pass
        class_to_label[class_key] = str(label)

    by_label = {}
    for cls, prob in zip(classes, probs):
        class_key = str(cls.item() if hasattr(cls, "item") else cls)
        by_label[class_to_label.get(class_key, class_key)] = float(prob)

    return np.asarray([by_label.get(str(label), 0.0) for label in manifest_labels], dtype=np.float64)


def predict(image: Image.Image, model: Any, deployment: OfflineDeployment, device: str = "cpu") -> dict[str, Any]:
    labels = deployment.labels

    if isinstance(model, PrototypeHeadModel):
        embedding = model.predict_embedding(preprocess_image_array(image, deployment))[0]
        params = deployment.manifest.get("production_params", {}) or {}
        dist_fct = params.get("dist_fct") or deployment.manifest.get("distance") or "euclidean"
        probs = _prototype_probabilities(embedding, model.prototypes, labels, str(dist_fct))
    elif isinstance(model, SklearnHeadModel):
        embedding = model.predict_embedding(preprocess_image_array(image, deployment))
        head_probs, classes = _classifier_probabilities(model.classifier, embedding)
        probs = _align_classifier_probabilities(head_probs, classes, model.head_labels, labels)
    elif isinstance(model, OnnxClassifier):
        logits = model.predict_logits(preprocess_image_array(image, deployment))
        probs = _softmax(logits)[0]
    else:
        import torch

        tensor = preprocess_image_tensor(image, deployment, device=device)
        torch_runtime = importlib.import_module("otitenet.offline.torch_runtime")
        with torch.no_grad():
            logits = torch_runtime.torch_logits_from_output(model(tensor), model)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    if labels and len(labels) != len(probs):
        raise ValueError(
            f"Model produced {len(probs)} outputs but manifest defines {len(labels)} labels."
        )

    if not labels:
        labels = [str(i) for i in range(len(probs))]

    best_idx = int(np.argmax(probs))
    return {
        "label": labels[best_idx],
        "confidence": float(probs[best_idx]),
        "probabilities": [
            {"label": label, "probability": float(prob)}
            for label, prob in zip(labels, probs)
        ],
    }
