import os
from typing import Optional

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
        embedding = model(image)
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
    image_np -= image_np.min()
    image_np /= image_np.max() + 1e-8

    plt.figure(figsize=(4, 4))
    plt.imshow(image_np)
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def log_grad_cam_similarity(
    model: torch.nn.Module,
    index: int,
    inputs: dict,
    group: str,
    output_dir: str,
    filename: str,
    prototype_embedding: np.ndarray,
    device: str = "cuda",
    layer: Optional[int] = 5,
    alpha: float = 0.55,
) -> None:
    """Log Grad-CAM overlay comparing an image with its class prototype embedding."""

    if "inputs" not in inputs[group] or len(inputs[group]["inputs"]) == 0:
        return

    images = torch.cat([torch.as_tensor(x) for x in inputs[group]["inputs"]]).to(device)
    image = images[index : index + 1].clone().requires_grad_(True)

    prototype_tensor = torch.as_tensor(prototype_embedding, device=device, dtype=image.dtype)

    model = model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        heatmap = _compute_grad_cam_heatmap(model, image, prototype_tensor, layer)

    overlay_path = os.path.join(output_dir, f"{filename}_grad_cam.png")
    _save_overlay(image, heatmap, overlay_path, alpha=alpha)
