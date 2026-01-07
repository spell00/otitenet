import os
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class _FeatureHook:
    """Utility to capture activations and gradients for Grad-CAM."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            module.register_forward_hook(self._forward_hook),
            module.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _resolve_layer(model: torch.nn.Module, layer: Optional[int]) -> torch.nn.Module:
    """Return the target layer inside the feature extractor."""

    if layer is None:
        layer = -1
    if not hasattr(model, "model"):
        raise AttributeError("Expected model to expose a 'model' attribute containing the feature extractor.")

    extractor = model.model
    if isinstance(layer, int):
        return extractor[layer]

    raise TypeError("layer must be an integer index into the feature extractor")


def _compute_grad_cam_heatmap(
    model: torch.nn.Module,
    image: torch.Tensor,
    reference_embedding: torch.Tensor,
    layer: Optional[int],
) -> np.ndarray:
    """Compute a Grad-CAM heatmap using a reference embedding as similarity target."""

    target_module = _resolve_layer(model, layer)
    hook = _FeatureHook(target_module)
    try:
        output = model(image)
        # Handle models that return tuples (e.g., with auxiliary outputs)
        if isinstance(output, tuple):
            embedding = output[0]
        else:
            embedding = output
        
        distance = F.pairwise_distance(embedding, reference_embedding.unsqueeze(0), p=2)
        # Negative distance focuses the heatmap on similarity regions
        score = -distance
        score.backward()

        if hook.gradients is None or hook.activations is None:
            raise RuntimeError("Failed to capture activations/gradients for Grad-CAM computation")

        gradients = hook.gradients
        activations = hook.activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam
    finally:
        hook.close()


def _save_overlay(image: torch.Tensor, heatmap: np.ndarray, output_path: str, alpha: float = 0.5) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image_np = image.squeeze().detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    # Validate image dimensions
    if image_np.shape[0] <= 0 or image_np.shape[1] <= 0:
        raise ValueError(f"Invalid image dimensions: {image_np.shape}. Height and width must be > 0")
    
    # Validate heatmap dimensions
    if heatmap.shape[0] <= 0 or heatmap.shape[1] <= 0:
        raise ValueError(f"Invalid heatmap dimensions: {heatmap.shape}. Height and width must be > 0")
    
    image_np -= image_np.min()
    max_val = image_np.max()
    if max_val > 0:
        image_np /= max_val
    else:
        image_np = np.zeros_like(image_np) + 0.5  # Fallback to neutral gray

    plt.figure(figsize=(4, 4))
    plt.imshow(image_np)
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay_from_heatmap(
    image: torch.Tensor,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.5,
) -> None:
    """Re-render an overlay from a cached heatmap without recomputing Grad-CAM."""

    _save_overlay(image, heatmap, output_path, alpha=alpha)


def log_grad_cam_similarity(
    model: torch.nn.Module,
    index: int,
    inputs: dict,
    group: str,
    output_dir: str,
    filename: str,
    prototype_embedding: np.ndarray,
    device: str = "cuda",
    layer: Optional[int] = 7,
    alpha: float = 0.55,
    fallback_to_cpu: bool = True,
) -> None:
    """Log Grad-CAM overlay comparing an image with its class prototype embedding."""

    if "inputs" not in inputs[group] or len(inputs[group]["inputs"]) == 0:
        return

    try:
        images = torch.cat([torch.as_tensor(x) for x in inputs[group]["inputs"]]).to(device)
        image = images[index : index + 1].clone().requires_grad_(True)

        prototype_tensor = torch.as_tensor(prototype_embedding, device=device, dtype=image.dtype)

        model = model.to(device)
        model.eval()
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            heatmap = _compute_grad_cam_heatmap(model, image, prototype_tensor, layer)

        overlay_path = os.path.join(output_dir, f"{filename}_grad_cam.png")
        heatmap_path = overlay_path.replace(".png", "_heatmap.npy")
        np.save(heatmap_path, heatmap)
        _save_overlay(image, heatmap, overlay_path, alpha=alpha)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if fallback_to_cpu and device != "cpu":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_grad_cam_similarity(
                model,
                index,
                inputs,
                group,
                output_dir,
                filename,
                prototype_embedding,
                device="cpu",
                layer=layer,
                alpha=alpha,
                fallback_to_cpu=False,
            )
        else:
            raise e


def _create_grad_cam_montage(
    image: torch.Tensor,
    class_overlays: Dict[Any, np.ndarray],
    output_dir: str,
    filename: str,
    layer: int,
    alpha: float = 0.55,
) -> None:
    """Create a montage combining original image with all class Grad-CAM overlays."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare image
    image_np = image.squeeze().detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np -= image_np.min()
    max_val = image_np.max()
    if max_val > 0:
        image_np /= max_val
    else:
        image_np = np.zeros_like(image_np) + 0.5
    
    # Create montage grid
    num_classes = len(class_overlays)
    cols = min(3, num_classes + 1)  # Original + classes, max 3 cols
    rows = (num_classes + 1 + cols - 1) // cols  # +1 for original, ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    # Plot original image
    ax = axes.flat[0]
    ax.imshow(image_np)
    ax.set_title("Original")
    ax.axis("off")
    
    # Plot each class overlay
    sorted_labels = sorted(class_overlays.keys())
    for idx, label in enumerate(sorted_labels):
        ax = axes.flat[idx + 1]
        heatmap = class_overlays[label]
        ax.imshow(image_np)
        ax.imshow(heatmap, cmap="jet", alpha=alpha)
        ax.set_title(f"Class {label}")
        ax.axis("off")
    
    # Hide unused subplots
    for idx in range(num_classes + 1, len(axes.flat)):
        axes.flat[idx].axis("off")
    
    plt.tight_layout()
    montage_path = os.path.join(output_dir, f"{filename}_grad_cam_all_classes_layer{layer}.png")
    plt.savefig(montage_path, dpi=100, bbox_inches="tight")
    plt.close()


def log_grad_cam_all_classes(
    model: torch.nn.Module,
    index: int,
    inputs: Dict[str, Any],
    group: str,
    output_dir: str,
    filename: str,
    class_protos: Dict[Any, Any],
    device: str = "cuda",
    layer: Optional[int] = 5,
    alpha: float = 0.55,
    fallback_to_cpu: bool = True,
) -> None:
    """Generate Grad-CAM overlays for each class prototype.

    Saves one image per class in ``output_dir`` named ``{filename}_class{label}_layer{layer}.png``.
    Also saves a montage with all classes in ``{filename}_grad_cam_all_classes_layer{layer}.png``.
    Uses the first prototype for each class if multiple are provided.
    """

    if group not in inputs or "inputs" not in inputs[group] or len(inputs[group]["inputs"]) == 0:
        return

    try:
        images = torch.cat([torch.as_tensor(x) for x in inputs[group]["inputs"]]).to(device)
        image = images[index : index + 1].clone().requires_grad_(True)

        model = model.to(device)
        model.eval()
        model.zero_grad(set_to_none=True)

        # Store overlays for montage creation
        class_overlays = {}
        sorted_labels = sorted(class_protos.keys())

        # Iterate over classes and generate one overlay per class
        for label in sorted_labels:
            proto = class_protos[label]
            if proto is None:
                continue
            proto_tensor = torch.as_tensor(proto, device=device, dtype=image.dtype)
            # If multiple prototypes, take the first
            if proto_tensor.ndim > 1:
                proto_tensor = proto_tensor[0]
            # Ensure it's still a proper tensor after indexing
            proto_tensor = torch.as_tensor(proto_tensor, device=device, dtype=image.dtype)

            with torch.enable_grad():
                heatmap = _compute_grad_cam_heatmap(model, image, proto_tensor, layer)

            overlay_path = os.path.join(output_dir, f"{filename}_class{label}_layer{layer}.png")
            heatmap_path = overlay_path.replace(".png", "_heatmap.npy")
            np.save(heatmap_path, heatmap)
            _save_overlay(image, heatmap, overlay_path, alpha=alpha)
            
            # Store heatmap for montage
            class_overlays[label] = heatmap

        # Create a montage with original + all class overlays
        if class_overlays:
            _create_grad_cam_montage(image, class_overlays, output_dir, filename, layer, alpha)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if fallback_to_cpu and device != "cpu":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_grad_cam_all_classes(
                model,
                index,
                inputs,
                group,
                output_dir,
                filename,
                class_protos,
                device="cpu",
                layer=layer,
                alpha=alpha,
                fallback_to_cpu=False,
            )
        else:
            raise e
