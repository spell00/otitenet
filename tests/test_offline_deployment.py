import json
from pathlib import Path

from otitenet.offline import deployment


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
