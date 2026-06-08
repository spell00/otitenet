"""
Image preprocessing and transformation utilities.

Handles image loading, resizing, normalization, and tensor conversions
for both inference and visualization.
"""

import numpy as np
from PIL import Image


def _normalize_flag(normalize) -> bool:
    return str(normalize).lower() in ["yes", "true", "1"]


def _resize_for_model(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    img = img.resize((64, 64))
    if size != -1:
        img = img.resize((size, size))
    return img


def _per_image_normalize_array(array: np.ndarray) -> np.ndarray:
    mean = array.mean(axis=(1, 2), keepdims=True)
    std = array.std(axis=(1, 2), keepdims=True, ddof=1)
    return (array - mean) / (std + 1e-6)


def image_to_chw_array(img: Image.Image, size: int, normalize="no") -> np.ndarray:
    """Return a CHW float32 image array usable by ONNX/NumPy runtimes."""
    img = _resize_for_model(img, size)
    array = np.asarray(img, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = array.transpose(2, 0, 1)
    if _normalize_flag(normalize):
        array = _per_image_normalize_array(array)
    return array.astype(np.float32)


def preprocess_image_array(img: Image.Image, size: int, normalize="no") -> np.ndarray:
    """Preprocess a PIL image as NCHW float32 without importing Torch."""
    return image_to_chw_array(img, size, normalize=normalize)[None, ...]


def get_image_arrays(path, size=-1, normalize="no") -> tuple[np.ndarray, np.ndarray]:
    """Load an image and return display/model NCHW arrays for non-Torch runtimes."""
    original = Image.open(path).convert("RGB")
    display_array = image_to_chw_array(original, size, normalize="no")[None, ...]
    model_array = image_to_chw_array(original, size, normalize=normalize)[None, ...]
    return display_array, model_array


def get_image(path, size=-1, normalize='no'):
    """Load and preprocess an image from disk.
    
    Args:
        path: Path to image file
        size: Target size for resizing (-1 to keep original)
        normalize: Whether to apply per-image normalization ('yes'/'no')
        
    Returns:
        Tuple of (transformed_tensor, display_tensor)
    """
    import torch

    display_array, model_array = get_image_arrays(path, size=size, normalize=normalize)
    return torch.from_numpy(display_array), torch.from_numpy(model_array)


def preprocess_image(img: Image.Image, size: int, normalize='no', device='cpu'):
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

    tensor = torch.from_numpy(image_to_chw_array(img, size, normalize=normalize))
    return tensor.unsqueeze(0).to(device)
