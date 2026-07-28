import json
from pathlib import Path

import numpy as np

from scripts.export_offline_onnx_model import update_manifest
from otitenet.offline import deployment
from otitenet.offline.predictor import _knn_probabilities


def _write_deployment(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "embedding_model.onnx").write_bytes(b"fake")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "model_type": "onnx_classifier",
                "labels": ["Normal", "NotNormal"],
                "files": {"model": "embedding_model.onnx"},
            }
        ),
        encoding="utf-8",
    )


def test_load_deployment_uses_env_override(tmp_path, monkeypatch):
    deployment_root = tmp_path / "deployment"
    _write_deployment(deployment_root)
    monkeypatch.setenv("OTITENET_DEPLOYMENT_DIR", str(deployment_root))

    loaded = deployment.load_deployment()

    assert loaded.root == deployment_root
    assert loaded.model_file == deployment_root / "embedding_model.onnx"


def test_load_deployment_uses_pyinstaller_bundle_root(tmp_path, monkeypatch):
    bundled_root = tmp_path / "bundle"
    deployment_root = bundled_root / deployment.DEFAULT_DEPLOYMENT_DIR
    _write_deployment(deployment_root)
    cwd = tmp_path / "other"
    cwd.mkdir()
    monkeypatch.setattr(deployment.sys, "_MEIPASS", str(bundled_root), raising=False)
    monkeypatch.chdir(cwd)

    loaded = deployment.load_deployment()

    assert loaded.root == deployment_root
    assert loaded.model_file == deployment_root / "embedding_model.onnx"


def test_knn_probabilities_vote_against_manifest_labels():
    reference_embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        dtype=np.float32,
    )
    reference_labels = np.asarray([1, 1, 2])

    probs = _knn_probabilities(
        np.asarray([1.0, 0.0], dtype=np.float32),
        reference_embeddings,
        reference_labels,
        ["Normal", "NotNormal", "Wax"],
        k=3,
        distance="cosine",
    )

    assert np.allclose(probs, [0.0, 2.0 / 3.0, 1.0 / 3.0])


def test_onnx_manifest_preserves_knn_embedding_head(tmp_path):
    deployment_root = tmp_path / "deployment"
    deployment_root.mkdir()
    (deployment_root / "model.pth").write_bytes(b"torch")
    (deployment_root / "embedding_model.onnx").write_bytes(b"onnx")
    (deployment_root / "reference_embeddings.npy").write_bytes(b"emb")
    (deployment_root / "reference_labels.npy").write_bytes(b"labels")
    (deployment_root / "manifest.json").write_text(
        json.dumps(
            {
                "model_type": "torch_embedding_knn",
                "head_type": "knn",
                "files": {
                    "model": "model.pth",
                    "embedding_model": "model.pth",
                    "reference_embeddings": "reference_embeddings.npy",
                    "reference_labels": "reference_labels.npy",
                },
            }
        ),
        encoding="utf-8",
    )

    updated = update_manifest(
        deployment_root,
        deployment_root / "embedding_model.onnx",
        keep_pytorch=True,
        quantization="none",
        embedding_output=True,
    )

    assert updated["model_type"] == "onnx_embedding_knn"
    assert updated["runtime"] == "onnxruntime"
    assert updated["head_type"] == "knn"
    assert updated["files"]["model"] == "embedding_model.onnx"
    assert updated["files"]["embedding_model"] == "embedding_model.onnx"
    assert updated["files"]["reference_embeddings"] == "reference_embeddings.npy"
