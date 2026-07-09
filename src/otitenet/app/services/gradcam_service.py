"""Grad-CAM computation service for on-demand explanation generation.

This module provides utilities to compute Grad-CAM heatmaps on demand
for both online (PyTorch) and offline (deployment-based) applications.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from otitenet.logging.grad_cam import (
        _compute_grad_cam_heatmap,
        _create_grad_cam_montage,
        _save_overlay,
    )
except Exception as _gradcam_dependency_error:
    _compute_grad_cam_heatmap = None
    _create_grad_cam_montage = None
    _save_overlay = None


def compute_grad_cam_for_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    prototype_embedding: torch.Tensor,
    layer: int = 5,
    device: str = "cpu",
) -> np.ndarray:
    """Compute a single Grad-CAM heatmap for an image against a prototype.

    Args:
        model: PyTorch model with .model attribute (feature extractor)
        image_tensor: Input image tensor (1, C, H, W)
        prototype_embedding: Reference embedding to compute similarity against
        layer: Layer index in the feature extractor to target
        device: Device to run computation on

    Returns:
        Normalized heatmap as numpy array (H, W)
    """
    if _compute_grad_cam_heatmap is None:
        raise RuntimeError(f"Grad-CAM dependencies are unavailable: {_gradcam_dependency_error}")

    model = model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)

    image_tensor = image_tensor.to(device).requires_grad_(True)

    with torch.enable_grad():
        heatmap = _compute_grad_cam_heatmap(
            model, image_tensor, prototype_embedding, layer
        )

    return heatmap


def compute_grad_cam_all_classes(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_prototypes: Dict[Any, np.ndarray],
    layer: int = 5,
    device: str = "cpu",
    alpha: float = 0.55,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute Grad-CAM heatmaps for all class prototypes.

    Args:
        model: PyTorch model with .model attribute (feature extractor)
        image_tensor: Input image tensor (1, C, H, W)
        class_prototypes: Dictionary mapping class labels to prototype embeddings
        layer: Layer index in the feature extractor to target
        device: Device to run computation on
        alpha: Transparency for overlay visualization

    Returns:
        Tuple of (class_overlays, montage_heatmap) where:
        - class_overlays: Dict mapping class labels to heatmaps
        - montage_heatmap: Combined montage (or None if montage fails)
    """
    if _compute_grad_cam_heatmap is None:
        raise RuntimeError(f"Grad-CAM dependencies are unavailable: {_gradcam_dependency_error}")

    model = model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)

    image_tensor = image_tensor.to(device).requires_grad_(True)

    class_overlays = {}
    sorted_labels = sorted(class_prototypes.keys())

    for label in sorted_labels:
        proto = class_prototypes[label]
        if proto is None:
            continue

        proto_tensor = torch.as_tensor(
            proto, device=device, dtype=image_tensor.dtype
        )
        if proto_tensor.ndim > 1:
            proto_tensor = proto_tensor[0]
        proto_tensor = torch.as_tensor(
            proto_tensor, device=device, dtype=image_tensor.dtype
        )

        with torch.enable_grad():
            heatmap = _compute_grad_cam_heatmap(
                model, image_tensor, proto_tensor, layer
            )

        class_overlays[label] = heatmap
        model.zero_grad(set_to_none=True)

    return class_overlays, None


def save_grad_cam_overlay(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.55,
) -> None:
    """Save a Grad-CAM overlay image.

    Args:
        image_tensor: Original image tensor (1, C, H, W)
        heatmap: Computed heatmap (H, W)
        output_path: Path to save the overlay image
        alpha: Transparency for the heatmap overlay
    """
    if _save_overlay is None:
        raise RuntimeError(f"Grad-CAM dependencies are unavailable: {_gradcam_dependency_error}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _save_overlay(image_tensor, heatmap, output_path, alpha=alpha)


def save_grad_cam_montage(
    image_tensor: torch.Tensor,
    class_overlays: Dict[str, np.ndarray],
    output_dir: str,
    filename: str,
    layer: int,
    alpha: float = 0.55,
) -> str:
    """Save a montage with original image and all class overlays.

    Args:
        image_tensor: Original image tensor (1, C, H, W)
        class_overlays: Dictionary mapping class labels to heatmaps
        output_dir: Directory to save the montage
        filename: Base filename for the montage
        layer: Layer number for filename suffix
        alpha: Transparency for heatmap overlays

    Returns:
        Path to the saved montage image
    """
    if _create_grad_cam_montage is None:
        raise RuntimeError(f"Grad-CAM dependencies are unavailable: {_gradcam_dependency_error}")

    _create_grad_cam_montage(
        image_tensor, class_overlays, output_dir, filename, layer, alpha
    )
    montage_path = os.path.join(
        output_dir, f"{filename}_grad_cam_all_classes_layer{layer}.png"
    )
    return montage_path


def compute_and_save_grad_cam_online(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_prototypes: Dict[Any, np.ndarray],
    output_dir: str,
    filename: str,
    layer: int = 5,
    device: str = "cpu",
    alpha: float = 0.55,
) -> List[str]:
    """Compute and save Grad-CAM overlays for all classes (online app).

    This is the main entry point for the online app where we have:
    - Full PyTorch model with gradient support
    - Access to class prototypes
    - File system access to save results

    Args:
        model: PyTorch model with .model attribute
        image_tensor: Input image tensor (1, C, H, W)
        class_prototypes: Dictionary mapping class labels to prototype embeddings
        output_dir: Directory to save Grad-CAM outputs
        filename: Base filename for outputs
        layer: Layer index to target
        device: Device to run computation on
        alpha: Transparency for overlays

    Returns:
        List of paths to generated Grad-CAM images
    """
    os.makedirs(output_dir, exist_ok=True)

    class_overlays, _ = compute_grad_cam_all_classes(
        model=model,
        image_tensor=image_tensor,
        class_prototypes=class_prototypes,
        layer=layer,
        device=device,
        alpha=alpha,
    )

    saved_paths = []

    # Save individual class overlays
    for label, heatmap in class_overlays.items():
        overlay_path = os.path.join(output_dir, f"{filename}_class{label}_layer{layer}.png")
        save_grad_cam_overlay(image_tensor, heatmap, overlay_path, alpha=alpha)
        saved_paths.append(overlay_path)

    # Save montage
    if class_overlays:
        montage_path = save_grad_cam_montage(
            image_tensor=image_tensor,
            class_overlays=class_overlays,
            output_dir=output_dir,
            filename=filename,
            layer=layer,
            alpha=alpha,
        )
        saved_paths.append(montage_path)

    return saved_paths


def compute_and_save_grad_cam_offline(
    deployment: Any,
    model: Any,
    image: Image.Image,
    output_dir: str,
    filename: str,
    layer: int = 5,
    device: str = "cpu",
    alpha: float = 0.55,
) -> Tuple[bool, List[str], str]:
    """Compute and save Grad-CAM overlays for offline app.

    The offline app uses deployment-based models which may be:
    - ONNX models (no gradient support - Grad-CAM not possible)
    - PyTorch models with gradient support

    Args:
        deployment: OfflineDeployment object
        model: Loaded model (may be ONNX or PyTorch)
        image: PIL Image to compute Grad-CAM for
        output_dir: Directory to save outputs
        filename: Base filename for outputs
        layer: Layer index to target (for PyTorch models)
        device: Device to run computation on
        alpha: Transparency for overlays

    Returns:
        Tuple of (success, saved_paths, error_message)
        - success: Whether Grad-CAM computation succeeded
        - saved_paths: List of paths to generated images (empty if failed)
        - error_message: Error message if failed, empty otherwise
    """
    try:
        # Check if this is a PyTorch model with gradient support
        has_torch = False
        try:
            import torch
            has_torch = True
        except ImportError:
            return False, [], "PyTorch not available - Grad-CAM requires PyTorch"

        if not has_torch:
            return False, [], "PyTorch not available - Grad-CAM requires PyTorch"

        # Check model type
        model_type = str(deployment.model_type).lower()

        if "onnx" in model_type:
            return (
                False,
                [],
                "ONNX models do not support gradient computation - Grad-CAM not available",
            )

        # For PyTorch models, check if we can access the feature extractor
        if not hasattr(model, "model"):
            return (
                False,
                [],
                "Model does not expose .model attribute - Grad-CAM structure not compatible",
            )

        # Load prototypes if available
        from otitenet.offline.predictor import _load_prototypes

        try:
            prototypes = _load_prototypes(deployment)
        except Exception as e:
            return False, [], f"Failed to load prototypes: {e}"

        # Preprocess image
        from otitenet.offline.predictor import preprocess_image_tensor

        image_tensor = preprocess_image_tensor(image, deployment, device=device)

        # Compute Grad-CAM
        saved_paths = compute_and_save_grad_cam_online(
            model=model,
            image_tensor=image_tensor,
            class_prototypes=prototypes,
            output_dir=output_dir,
            filename=filename,
            layer=layer,
            device=device,
            alpha=alpha,
        )

        return True, saved_paths, ""

    except Exception as e:
        return False, [], f"Grad-CAM computation failed: {e}"


def pil_image_to_tensor(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Convert PIL Image to PyTorch tensor with preprocessing.

    Args:
        image: PIL Image
        device: Device to place tensor on

    Returns:
        Tensor (1, C, H, W)
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to standard size
    image = image.resize((224, 224))

    # Convert to numpy and normalize
    array = np.array(image, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)

    # Convert to tensor
    tensor = torch.from_numpy(array).unsqueeze(0).to(device)

    return tensor


def create_temp_grad_cam_dir(base_dir: str = "data/temp_gradcam") -> str:
    """Create a temporary directory for Grad-CAM outputs.

    Args:
        base_dir: Base directory for temp Grad-CAM files

    Returns:
        Path to created temporary directory
    """
    os.makedirs(base_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=base_dir)
    return temp_dir
