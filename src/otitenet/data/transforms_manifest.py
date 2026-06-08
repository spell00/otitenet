from __future__ import annotations

from typing import Any


def normalize_yes_no(value: Any) -> str:
    text = str(value or "no").strip().lower()
    if text in {"yes", "true", "1", "per_image"}:
        return "yes"
    return "no"


def image_preprocessing_manifest(
    input_size: tuple[int, int],
    normalize: Any = "no",
    normalize_mean: tuple[float, float, float] | None = None,
    normalize_std: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    normalize_value = normalize_yes_no(normalize)
    manifest: dict[str, Any] = {
        "resize": [int(input_size[0]), int(input_size[1])],
        "resize_mode": "app_64_then_size",
        "color_mode": "RGB",
        "normalize": normalize_value,
    }
    if normalize_value == "yes":
        manifest["normalization"] = "per_image"
        manifest["normalization_axes"] = ["height", "width"]
        manifest["std_correction"] = 1
        manifest["epsilon"] = 1e-6
    elif normalize_mean is not None and normalize_std is not None:
        manifest["normalization"] = "channel_mean_std"
        manifest["normalize_mean"] = list(normalize_mean)
        manifest["normalize_std"] = list(normalize_std)
    else:
        manifest["normalization"] = "none"
    return manifest
