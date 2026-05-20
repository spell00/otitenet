"""
Image preprocessing and transformation utilities.

Handles image loading, resizing, normalization, and tensor conversions
for both inference and visualization.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from otitenet.data.data_getters import PerImageNormalize


def get_image(path, size=-1, normalize='no'):
    """Load and preprocess an image from disk.
    
    Args:
        path: Path to image file
        size: Target size for resizing (-1 to keep original)
        normalize: Whether to apply per-image normalization ('yes'/'no')
        
    Returns:
        Tuple of (transformed_tensor, display_tensor)
    """
    original = Image.open(path).convert('RGB')

    # Match training/inference dataset preprocessing in data_getters.get_images:
    # always downsample to 64 first, then resize to target size.
    downsample_size = 64
    img = original.resize((downsample_size, downsample_size))
    if size != -1:
        img = img.resize((size, size))

    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    # Display tensor should stay unnormalized for faithful preview.
    display_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    model_tensor = display_tensor.clone().squeeze(0)
    if str(normalize).lower() in ['yes', 'true', '1']:
        model_tensor = PerImageNormalize()(model_tensor)

    return display_tensor, model_tensor.unsqueeze(0)


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
    # Match data_getters.get_images preprocessing.
    img = img.resize((64, 64))
    img = img.resize((size, size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC → CHW
    tensor = torch.tensor(img_array)
    
    if str(normalize).lower() in ['yes', 'true', '1']:
        per_img_norm = PerImageNormalize()
        tensor = per_img_norm(tensor)
    
    tensor = tensor.unsqueeze(0).to(device)
    return tensor
