from __future__ import annotations

import os
from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _display_label(label: str) -> str:
    normalized = str(label or "Unknown").strip()
    if normalized.lower().replace("_", "") == "notnormal":
        return "Not Normal"
    return normalized


def result_palette(label: str) -> dict[str, str]:
    key = _display_label(label).lower()
    if key == "normal":
        return {"color": "#15803d", "background": "#f0fdf4", "border": "#86efac"}
    if key == "not normal":
        return {"color": "#b91c1c", "background": "#fef2f2", "border": "#fca5a5"}
    if key == "wax":
        return {"color": "#ca8a04", "background": "#fefce8", "border": "#fde047"}
    if key == "tube":
        return {"color": "#111827", "background": "#f9fafb", "border": "#111827"}
    return {"color": "#374151", "background": "#f9fafb", "border": "#d1d5db"}


def result_guidance(label: str, confidence: float | None) -> str:
    if confidence is not None and confidence < 0.50:
        return "Next step: low confidence. Try another photo or consult a doctor."

    key = _display_label(label).lower()
    if key == "tube":
        return "Next step: a tube is present. The model is limited when a tube is visible."
    if key == "wax":
        return "Next step: wax is visible. Clean the ear if appropriate and try again the next day, because cleaning can affect the result."
    if key == "not normal":
        return "Next step: the model found an abnormal result. Consider consulting a doctor."
    if key == "normal":
        return "Next step: the image looks normal according to the model."
    return "Next step: review the result and consult a doctor if symptoms continue."


def display_label(label: str) -> str:
    return _display_label(label)


def find_gradcam_image_paths(gradcam_dir: str | os.PathLike | None, filename: str | None, max_count: int = 4) -> List[str]:
    if not gradcam_dir or not filename:
        return []

    gradcam_path = Path(gradcam_dir)
    if not gradcam_path.exists():
        return []

    stem = Path(str(filename)).stem
    if not stem:
        return []

    matches = []
    for candidate in gradcam_path.iterdir():
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if candidate.name.startswith(stem):
            matches.append(str(candidate))

    matches.sort()
    return matches[:max_count]
