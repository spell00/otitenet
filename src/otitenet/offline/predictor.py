from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from otitenet.app.image_processing import preprocess_image_array as app_preprocess_image_array
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


def load_model(deployment: OfflineDeployment, device: str = "cpu"):
    path = deployment.model_file
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


def preprocess_image_array(image: Image.Image, deployment: OfflineDeployment) -> np.ndarray:
    preprocessing = deployment.manifest.get("preprocessing", {})
    normalization = str(preprocessing.get("normalization", "")).lower()
    normalize = preprocessing.get("normalize")
    if normalization == "per_image" or str(normalize).lower() in {"yes", "true", "1", "per_image"}:
        height, width = deployment.input_size
        if height != width:
            raise ValueError(f"App-compatible preprocessing expects square input, got {height}x{width}.")
        return app_preprocess_image_array(image, height, normalize="yes")

    mean = preprocessing.get("normalize_mean")
    std = preprocessing.get("normalize_std")
    if normalization in {"none", "no"} and not (mean and std):
        height, width = deployment.input_size
        if height != width:
            raise ValueError(f"App-compatible preprocessing expects square input, got {height}x{width}.")
        return app_preprocess_image_array(image, height, normalize="no")

    image = image.convert("RGB")
    height, width = deployment.input_size
    image = image.resize((width, height))

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
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        emb = torch.as_tensor(embedding, dtype=torch.float32)
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        distances = []
        for label in labels:
            proto = prototypes.get(str(label))
            if proto is None:
                distances.append(float("inf"))
                continue
            proto_t = torch.as_tensor(proto, dtype=emb.dtype)
            if proto_t.ndim == 1:
                proto_t = proto_t.unsqueeze(0)
            if str(dist_fct_name).lower() == "cosine":
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
                proto_norm = torch.nn.functional.normalize(proto_t, p=2, dim=1)
                distances.append(1.0 - torch.mm(emb_norm, proto_norm.T).mean().item())
            else:
                distances.append(torch.cdist(emb, proto_t, p=2).mean().item())
        distances_array = np.asarray(distances, dtype=np.float64)
    else:
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
                distances.append(1.0 - float(np.matmul(emb_norm, proto_norm.T).mean()))
            else:
                diff = emb[:, None, :] - proto[None, :, :]
                distances.append(float(np.sqrt(np.sum(diff * diff, axis=2)).mean()))
        distances_array = np.asarray(distances, dtype=np.float64)

    inv = 1.0 / (distances_array + 1e-8)
    inv[~np.isfinite(inv)] = 0.0
    total = inv.sum()
    if total <= 0:
        return np.zeros(len(labels), dtype=np.float64)
    return inv / total


def predict(image: Image.Image, model: Any, deployment: OfflineDeployment, device: str = "cpu") -> dict[str, Any]:
    labels = deployment.labels

    if isinstance(model, PrototypeHeadModel):
        embedding = model.predict_embedding(preprocess_image_array(image, deployment))[0]
        params = deployment.manifest.get("production_params", {}) or {}
        dist_fct = params.get("dist_fct") or deployment.manifest.get("distance") or "euclidean"
        probs = _prototype_probabilities(embedding, model.prototypes, labels, str(dist_fct))
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
