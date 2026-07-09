from __future__ import annotations

from typing import Any

TORCHVISION_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
TORCHVISION_NORMALIZE_STD = (0.229, 0.224, 0.225)


def normalize_yes_no(value: Any) -> str:
    text = str(value or "no").strip().lower()
    if text in {"yes", "true", "1", "imagenet", "torchvision", "channel", "channel_mean_std", "per_image"}:
        return "yes"
    return "no"


def normalize_mode(value: Any) -> str:
    text = str(value or "no").strip().lower()
    if text in {"yes", "true", "1", "imagenet", "torchvision", "channel", "channel_mean_std"}:
        return "imagenet"
    if text in {"per_image", "per-image", "legacy", "sklearn"}:
        return "per_image"
    return "no"


def image_preprocessing_manifest(
    input_size: tuple[int, int],
    normalize: Any = "yes",
    base_resize_size: int = 64,
    normalize_mean: tuple[float, float, float] | None = None,
    normalize_std: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    mode = normalize_mode(normalize)
    normalize_value = "yes" if mode != "no" else "no"
    base_resize_size = int(base_resize_size or 64)
    manifest: dict[str, Any] = {
        "resize": [int(input_size[0]), int(input_size[1])],
        "resize_mode": f"app_{base_resize_size}_then_size",
        "base_resize": [base_resize_size, base_resize_size],
        "base_resize_size": base_resize_size,
        "color_mode": "RGB",
        "normalize": normalize_value,
    }
    if mode == "per_image":
        manifest["normalization"] = "per_image"
        manifest["normalization_axes"] = ["height", "width"]
        manifest["std_correction"] = 1
        manifest["epsilon"] = 1e-6
    elif mode == "imagenet":
        manifest["normalization"] = "channel_mean_std"
        manifest["normalize_mean"] = list(TORCHVISION_NORMALIZE_MEAN)
        manifest["normalize_std"] = list(TORCHVISION_NORMALIZE_STD)
    elif normalize_mean is not None and normalize_std is not None:
        manifest["normalization"] = "channel_mean_std"
        manifest["normalize_mean"] = list(normalize_mean)
        manifest["normalize_std"] = list(normalize_std)
    else:
        manifest["normalization"] = "none"
    return manifest
