import torch


def safe_torch_load(path, device=None):
    """
    Safely load a PyTorch model with automatic device mapping.
    
    Args:
        path (str): Path to the model file
        device (str, optional): Target device. If None, uses CPU if CUDA unavailable.
    
    Returns:
        Loaded model state dict
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        return torch.load(path, map_location=device)
    except Exception as e:
        print(f"Warning: Failed to load {path} on {device}, trying CPU...")
        return torch.load(path, map_location='cpu')
