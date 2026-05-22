from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_DATA_DIR = Path.home() / ".otitenet" / "offline"
HISTORY_FILE = APP_DATA_DIR / "history.json"
IMAGE_DIR = APP_DATA_DIR / "images"


def _ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def image_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
    except Exception:
        return []
    return []


def save_history(rows: list[dict[str, Any]]) -> None:
    _ensure_dirs()
    with HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def record_result(
    *,
    filename: str,
    image_bytes: bytes,
    prediction: dict[str, Any],
    deployment_manifest: dict[str, Any],
) -> dict[str, Any]:
    _ensure_dirs()
    digest = image_digest(image_bytes)
    suffix = Path(filename).suffix or ".jpg"
    image_path = IMAGE_DIR / f"{digest}{suffix}"
    if not image_path.exists():
        image_path.write_bytes(image_bytes)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "image_sha256": digest,
        "image_path": str(image_path),
        "prediction": prediction.get("label"),
        "confidence": prediction.get("confidence"),
        "probabilities": prediction.get("probabilities", []),
        "model_id": deployment_manifest.get("model_id"),
        "model_name": deployment_manifest.get("model_name"),
        "model_type": deployment_manifest.get("model_type"),
    }

    rows = load_history()
    rows.insert(0, row)
    save_history(rows)
    return row
