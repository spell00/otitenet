"""
Image preprocessing and transformation utilities.

Handles image loading, resizing, normalization, and tensor conversions
for both inference and visualization.
"""

import numpy as np
from PIL import Image
import re

from otitenet.data.transforms_manifest import (
    TORCHVISION_NORMALIZE_MEAN,
    TORCHVISION_NORMALIZE_STD,
    normalize_mode,
)

IMAGENET_NORMALIZE_MEAN = np.asarray(TORCHVISION_NORMALIZE_MEAN, dtype=np.float32).reshape(3, 1, 1)
IMAGENET_NORMALIZE_STD = np.asarray(TORCHVISION_NORMALIZE_STD, dtype=np.float32).reshape(3, 1, 1)
BASE_RESIZE_SIZE = 64


def infer_base_resize_size(dataset_path, default: int = BASE_RESIZE_SIZE) -> int:
    """Infer the app's first resize size from a dataset path like otite_ds_64."""
    match = re.search(r"(?:^|/)otite_ds_(\d+)(?:/|$)", str(dataset_path or "").replace("\\", "/"))
    if not match:
        return int(default)
    try:
        parsed = int(match.group(1))
    except Exception:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _resize_for_model(img: Image.Image, size: int, base_size: int = BASE_RESIZE_SIZE) -> Image.Image:
    img = img.convert("RGB")
    base_size = int(base_size or BASE_RESIZE_SIZE)
    img = img.resize((base_size, base_size))
    size = int(size)
    if size != -1:
        img = img.resize((size, size))
    return img


def _per_image_normalize_array(array: np.ndarray) -> np.ndarray:
    mean = array.mean(axis=(1, 2), keepdims=True)
    std = array.std(axis=(1, 2), keepdims=True, ddof=1)
    return (array - mean) / (std + 1e-6)


def image_to_chw_array(img: Image.Image, size: int, normalize="yes", base_size: int = BASE_RESIZE_SIZE) -> np.ndarray:
    """Return a CHW float32 image array usable by ONNX/NumPy runtimes."""
    img = _resize_for_model(img, size, base_size=base_size)
    array = np.asarray(img, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = array.transpose(2, 0, 1)
    mode = normalize_mode(normalize)
    if mode == "imagenet":
        array = (array - IMAGENET_NORMALIZE_MEAN) / IMAGENET_NORMALIZE_STD
    elif mode == "per_image":
        array = _per_image_normalize_array(array)
    return array.astype(np.float32)


def _array_stats(array: np.ndarray) -> dict:
    arr = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def preprocessing_trace(img: Image.Image, size: int, normalize="yes", base_size: int = BASE_RESIZE_SIZE) -> dict:
    """Return visual preprocessing stages and numeric tensor stats for debugging."""
    raw = img.convert("RGB")
    base_size = int(base_size or BASE_RESIZE_SIZE)
    down = raw.resize((base_size, base_size))
    if int(size) == -1:
        final = down
    else:
        final = down.resize((int(size), int(size)))

    unnormalized = image_to_chw_array(raw, size, normalize="no", base_size=base_size)
    normalized = image_to_chw_array(raw, size, normalize=normalize, base_size=base_size)
    mode = normalize_mode(normalize)
    return {
        "resize_mode": f"app_{base_size}_then_size",
        "base_resize_size": base_size,
        "normalize_mode": mode,
        "raw": {
            "image": raw,
            "size": list(raw.size),
            "stats": _array_stats(np.asarray(raw, dtype=np.float32) / 255.0),
        },
        "downsized": {
            "image": down,
            "size": list(down.size),
            "stats": _array_stats(np.asarray(down, dtype=np.float32) / 255.0),
        },
        "resized": {
            "image": final,
            "size": list(final.size),
            "stats": _array_stats(np.asarray(final, dtype=np.float32) / 255.0),
        },
        "tensor_before_normalize": _array_stats(unnormalized),
        "tensor_after_normalize": _array_stats(normalized),
    }


def preprocess_image_array(img: Image.Image, size: int, normalize="yes", base_size: int = BASE_RESIZE_SIZE) -> np.ndarray:
    """Preprocess a PIL image as NCHW float32 without importing Torch."""
    return image_to_chw_array(img, size, normalize=normalize, base_size=base_size)[None, ...]


def get_image_arrays(path, size=-1, normalize="yes", base_size: int = BASE_RESIZE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and return display/model NCHW arrays for non-Torch runtimes."""
    original = Image.open(path).convert("RGB")
    display_array = image_to_chw_array(original, size, normalize="no", base_size=base_size)[None, ...]
    model_array = image_to_chw_array(original, size, normalize=normalize, base_size=base_size)[None, ...]
    return display_array, model_array


def get_image(path, size=-1, normalize='yes', base_size: int = BASE_RESIZE_SIZE):
    """Load and preprocess an image from disk.
    
    Args:
        path: Path to image file
        size: Target size for resizing (-1 to keep original)
        normalize: Whether to apply per-image normalization ('yes'/'no')
        
    Returns:
        Tuple of (transformed_tensor, display_tensor)
    """
    import torch

    display_array, model_array = get_image_arrays(path, size=size, normalize=normalize, base_size=base_size)
    return torch.from_numpy(display_array), torch.from_numpy(model_array)


def preprocess_image(img: Image.Image, size: int, normalize='yes', device='cpu', base_size: int = BASE_RESIZE_SIZE):
    """Preprocess a PIL image for model inference.
    
    Args:
        img: PIL Image object
        size: Target size for resizing
        normalize: Whether to apply per-image normalization
        device: Target device for tensor
        
    Returns:
        Preprocessed tensor ready for model input
    """
    import torch

    tensor = torch.from_numpy(image_to_chw_array(img, size, normalize=normalize, base_size=base_size))
    return tensor.unsqueeze(0).to(device)
