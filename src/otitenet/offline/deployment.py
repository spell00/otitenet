from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DEPLOYMENT_DIR = Path("data/mobile_deployments/current")


@dataclass(frozen=True)
class OfflineDeployment:
    root: Path
    manifest: dict[str, Any]

    @property
    def labels(self) -> list[str]:
        return [str(label) for label in self.manifest.get("labels", [])]

    @property
    def model_type(self) -> str:
        return str(self.manifest.get("model_type", "torch_classifier"))

    @property
    def model_name(self) -> str:
        return str(self.manifest.get("model_name", "production_model"))

    @property
    def input_size(self) -> tuple[int, int]:
        raw_size = (
            self.manifest.get("input", {}).get("image_size")
            or self.manifest.get("preprocessing", {}).get("resize")
            or [224, 224]
        )
        return int(raw_size[0]), int(raw_size[1])

    @property
    def model_file(self) -> Path:
        files = self.manifest.get("files", {})
        model_name = (
            files.get("model")
            or files.get("classifier_model")
            or files.get("embedding_model")
        )
        if not model_name:
            raise ValueError("Deployment manifest does not define a model file.")
        return self.root / Path(model_name).name


def _default_deployment_candidates() -> list[Path]:
    candidates: list[Path] = []

    env_dir = os.environ.get("OTITENET_DEPLOYMENT_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        candidates.append(Path(bundled_root) / DEFAULT_DEPLOYMENT_DIR)

    candidates.append(DEFAULT_DEPLOYMENT_DIR)

    source_root = Path(__file__).resolve().parents[3]
    candidates.append(source_root / DEFAULT_DEPLOYMENT_DIR)

    return candidates


def _resolve_deployment_root(root: str | Path) -> Path:
    root_path = Path(root)
    if root_path != DEFAULT_DEPLOYMENT_DIR or root_path.is_absolute():
        return root_path

    for candidate in _default_deployment_candidates():
        if (candidate / "manifest.json").exists():
            return candidate

    return root_path


def load_deployment(root: str | Path = DEFAULT_DEPLOYMENT_DIR) -> OfflineDeployment:
    root = _resolve_deployment_root(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Offline deployment manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    deployment = OfflineDeployment(root=root, manifest=manifest)
    if not deployment.model_file.exists():
        raise FileNotFoundError(f"Offline deployment model not found: {deployment.model_file}")

    return deployment
