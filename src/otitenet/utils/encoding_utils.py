"""Shared utilities for encoding data and computing prototypes with augmentation support."""

import numpy as np
import torch
from torchvision.transforms import v2 as transforms
from torch import nn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from otitenet.data.transforms_manifest import (
    TORCHVISION_NORMALIZE_MEAN,
    TORCHVISION_NORMALIZE_STD,
    normalize_mode,
)


class PerImageNormalizeTransform(nn.Module):
    """Per-image CHW tensor normalization retained for explicit legacy runs."""

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        mean = img.mean(dim=(-2, -1), keepdim=True)
        std = img.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (img - mean) / std


def normalize_transform(normalize):
    mode = normalize_mode(normalize)
    if mode == "imagenet":
        return transforms.Normalize(
            mean=TORCHVISION_NORMALIZE_MEAN,
            std=TORCHVISION_NORMALIZE_STD,
        )
    if mode == "per_image":
        return PerImageNormalizeTransform()
    return None


def _append_normalize_transform(ops: list, normalize) -> list:
    norm = normalize_transform(normalize)
    if norm is not None:
        ops.append(norm)
    return ops


def get_base_transform(normalize='yes'):
    """Get base transform (no augmentation) for inference and validation/test encoding."""
    ops = [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
    return transforms.Compose(_append_normalize_transform(ops, normalize))


def get_knn_augmentation_transform(image_size: int, normalize='yes', translate: float = 0.10):
    """Get augmentation transform for KNN expansion (train set only).
    
    Args:
        image_size: Target image size for RandomResizedCrop
        normalize: no|yes/imagenet|per_image normalization mode
        translate: max RandomAffine translation fraction per axis
        
    Returns:
        Composed transform with random flips, rotations, translation, crops, and optional normalization
    """
    ops = [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomAffine(
            degrees=0,
            translate=(float(translate), float(translate)),
        ),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.8, 1.2),
                )
            ]),
            p=0.5,
        ),
    ]
    return transforms.Compose(_append_normalize_transform(ops, normalize))


def encode_split_with_augmentation(
    inputs, labels, batches,
    model, device, batch_size,
    split_name: str = 'train',
    n_aug: int = 0,
    image_size: int = 224,
    normalize='yes',
):
    """Encode a data split with optional augmentation for train only.
    
    Semantics:
    - n_aug = 0: Only original images (1 sample per image)
    - n_aug >= 1: Original + n_aug augmented copies (1+n_aug samples per image)
    - Augmentation only applied to 'train' split; valid/test always use base transform
    
    Args:
        inputs: List of PIL images
        labels: Array of labels
        batches: Array of batch/domain IDs
        model: Trained neural network model
        device: Device to run model on
        batch_size: Batch size for encoding
        split_name: 'train', 'valid', or 'test'
        n_aug: Number of extra augmented copies per image (0 = originals only)
        image_size: Image size for transforms
        
    Returns:
        encodings: (N, D) array of encoded vectors
        labels_out: (N,) array of labels (repeated per augmentation)
    """
    base_transform = get_base_transform(normalize=normalize)
    knn_aug_transform = get_knn_augmentation_transform(image_size, normalize=normalize)
    
    encs = []
    labs = []
    
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            batch_batches = batches[i:i + batch_size]
            
            transformed = []
            labs_batch = []
            
            for j in range(len(batch_inputs)):
                # Determine number of repeats (original + extras)
                repeats = 1 + max(0, n_aug) if split_name == 'train' else 1
                
                for r in range(repeats):
                    # r == 0 -> original; r > 0 -> augmented (when n_aug > 0)
                    if split_name == 'train' and r > 0:
                        sample = knn_aug_transform(batch_inputs[j])
                    else:
                        sample = base_transform(batch_inputs[j])
                    
                    transformed.append(sample)
                    labs_batch.append(batch_labels[j])
            
            if not transformed:
                continue
            
            tensor_batch = torch.stack(transformed).to(device)
            encoded, _ = model(tensor_batch)
            encs.append(encoded.detach().cpu().numpy())
            labs.extend(labs_batch)
    
    return np.concatenate(encs), np.array(labs)


def compute_prototypes_by_strategy(
    encodings: np.ndarray,
    class_ids: np.ndarray,
    strategy: str = 'mean',
    n_components: int = 1,
    random_state: int = 1
) -> dict:
    """Compute prototypes per class using specified strategy.
    
    Args:
        encodings: (N, D) array of encoded vectors
        class_ids: (N,) array of class indices
        strategy: 'mean', 'kmeans', or 'gmm'
        n_components: Number of clusters/components per class
        random_state: Random seed for clustering
        
    Returns:
        dict {class_id: [(proto_vector, component_idx), ...]} with n_components entries per class
    """
    prototypes_by_class = {}
    unique_classes = np.unique(class_ids)
    
    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_encs = encodings[cls_mask]
        
        if len(cls_encs) == 0:
            continue
        
        protos_for_class = []
        
        if strategy == 'mean' or len(cls_encs) <= 1:
            # Single mean prototype per class
            protos_for_class.append((np.mean(cls_encs, axis=0), 0))
        elif strategy == 'kmeans':
            n_clust = min(n_components, len(cls_encs))
            km = KMeans(n_clusters=n_clust, n_init=5, random_state=random_state)
            km.fit(cls_encs)
            for i in range(n_clust):
                protos_for_class.append((km.cluster_centers_[i], i))
        elif strategy in ['gmm', 'em', 'expectation_maximization']:
            n_comp = min(n_components, len(cls_encs))
            gm = GaussianMixture(n_components=n_comp, random_state=random_state)
            gm.fit(cls_encs)
            for i in range(n_comp):
                protos_for_class.append((gm.means_[i], i))
        
        prototypes_by_class[cls_id] = protos_for_class
    
    return prototypes_by_class


def flatten_prototype_dict(proto_dict: dict) -> tuple:
    """Flatten prototype dict to arrays of vectors and labels.
    
    Args:
        proto_dict: {class_id: [(proto_vec, comp_idx), ...], ...}
        
    Returns:
        (proto_vecs, proto_labels): Arrays of prototype vectors and their class IDs
    """
    proto_vecs = []
    proto_labels = []
    
    for cls_id in sorted(proto_dict.keys()):
        for proto_vec, comp_idx in proto_dict[cls_id]:
            proto_vecs.append(proto_vec)
            proto_labels.append(cls_id)
    
    if not proto_vecs:
        return np.array([]), np.array([])
    
    return np.stack(proto_vecs), np.array(proto_labels)
