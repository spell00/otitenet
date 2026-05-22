from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


def load_model(deployment: OfflineDeployment, device: str = "cpu"):
    path = deployment.model_file
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


def preprocess_image_array(image: Image.Image, deployment: OfflineDeployment) -> np.ndarray:
    image = image.convert("RGB")
    height, width = deployment.input_size
    image = image.resize((width, height))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)

    preprocessing = deployment.manifest.get("preprocessing", {})
    mean = preprocessing.get("normalize_mean")
    std = preprocessing.get("normalize_std")
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


def predict(image: Image.Image, model: Any, deployment: OfflineDeployment, device: str = "cpu") -> dict[str, Any]:
    labels = deployment.labels

    if isinstance(model, OnnxClassifier):
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
