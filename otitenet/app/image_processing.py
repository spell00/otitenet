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
    ops = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
    if str(normalize).lower() in ['yes', 'true', '1']:
        ops.append(PerImageNormalize())
    
    transform = transforms.Compose(ops)
    original = Image.open(path).convert('RGB')
    if size != -1:
        png = transforms.Resize((size, size))(original)
    else:
        png = original
    print(size, png)
    
    return transform(original).unsqueeze(0), transform(png).unsqueeze(0)


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
    img = img.resize((size, size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC → CHW
    tensor = torch.tensor(img_array)
    
    if str(normalize).lower() in ['yes', 'true', '1']:
        per_img_norm = PerImageNormalize()
        tensor = per_img_norm(tensor)
    
    tensor = tensor.unsqueeze(0).to(device)
    return tensor
