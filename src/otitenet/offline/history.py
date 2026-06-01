from __future__ import annotations

import hashlib
import json
import csv
import io
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_DATA_DIR = Path.home() / ".otitenet" / "offline"
HISTORY_FILE = APP_DATA_DIR / "history.json"
USERS_FILE = APP_DATA_DIR / "users.json"
IMAGE_DIR = APP_DATA_DIR / "images"
DEFAULT_USER_ID = "offline-default"
DEFAULT_USER_NAME = "Offline user"


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


def load_users() -> list[dict[str, Any]]:
    """Return offline users, creating a default user if none exist."""
    if USERS_FILE.exists():
        try:
            with USERS_FILE.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                users = [u for u in payload if isinstance(u, dict) and u.get("id") and u.get("name")]
                if users:
                    return users
        except Exception:
            pass
    users = [
        {
            "id": DEFAULT_USER_ID,
            "name": DEFAULT_USER_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    save_users(users)
    return users


def save_users(users: list[dict[str, Any]]) -> None:
    _ensure_dirs()
    with USERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def create_user(name: str) -> dict[str, Any]:
    clean_name = " ".join(name.strip().split())
    if not clean_name:
        raise ValueError("User name cannot be empty.")

    users = load_users()
    for user in users:
        if str(user.get("name", "")).casefold() == clean_name.casefold():
            return user

    user = {
        "id": uuid.uuid4().hex,
        "name": clean_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    users.append(user)
    save_users(users)
    return user


def get_user(user_id: str | None) -> dict[str, Any]:
    users = load_users()
    for user in users:
        if user.get("id") == user_id:
            return user
    return users[0]


def history_for_user(person_id: str | None) -> list[dict[str, Any]]:
    """Return history for one user. Legacy rows belong to the default user."""
    user_id = person_id or DEFAULT_USER_ID
    return [
        row
        for row in load_history()
        if row.get("person_id", DEFAULT_USER_ID) == user_id
    ]


def clear_history(person_id: str | None = None, *, remove_images: bool = True) -> None:
    """Remove all history, or only one user's rows when person_id is supplied."""
    if person_id is None:
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        if remove_images and IMAGE_DIR.exists():
            shutil.rmtree(IMAGE_DIR)
        return

    rows = load_history()
    removed = [
        row
        for row in rows
        if row.get("person_id", DEFAULT_USER_ID) == person_id
    ]
    remaining = [
        row
        for row in rows
        if row.get("person_id", DEFAULT_USER_ID) != person_id
    ]
    save_history(remaining)

    if not remove_images:
        return

    remaining_paths = {str(row.get("image_path")) for row in remaining if row.get("image_path")}
    for row in removed:
        image_path = row.get("image_path")
        if not image_path or str(image_path) in remaining_paths:
            continue
        try:
            Path(image_path).unlink(missing_ok=True)
        except Exception:
            pass


def history_csv(rows: list[dict[str, Any]]) -> str:
    """Serialize offline history rows to CSV for local export."""
    columns = [
        "timestamp",
        "person_id",
        "person_name",
        "filename",
        "prediction",
        "confidence",
        "model_id",
        "model_name",
        "model_type",
        "image_sha256",
        "image_path",
        "probabilities",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        out = dict(row)
        if isinstance(out.get("probabilities"), (list, dict)):
            out["probabilities"] = json.dumps(out["probabilities"], ensure_ascii=True)
        writer.writerow(out)
    return buffer.getvalue()


def record_result(
    *,
    person_id: str,
    person_name: str,
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
        "person_id": person_id,
        "person_name": person_name,
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
