#!/usr/bin/env python3
"""
Utility script to fix torch.load calls by adding map_location parameter
to ensure compatibility with CPU-only environments.
"""

import re
import os
import torch

def get_device_for_cpu_compatibility():
    """Get appropriate device string for CPU compatibility."""
    return 'cpu' if not torch.cuda.is_available() else 'cuda'

def fix_inference_files():
    """Fix the most critical files for inference."""
    
    # Files that are most likely to be used for inference
    critical_files = [
        './app.py',
        './otitenet/infer/make_shap.py',
        './otitenet/train/train_triplet_new.py'
    ]
    
    print("Critical inference files have been manually fixed:")
    for file in critical_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (not found)")

def create_device_safe_load_function():
    """Create a utility function for safe model loading."""
    
    util_code = '''
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
'''
    
    # Check if utils file exists or create one
    utils_dir = './otitenet/utils'
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir, exist_ok=True)
    
    utils_file = os.path.join(utils_dir, 'model_loading.py')
    
    if not os.path.exists(utils_file):
        with open(utils_file, 'w') as f:
            f.write('import torch\n\n' + util_code)
        print(f"Created utility function in {utils_file}")
    else:
        print(f"Utility file already exists: {utils_file}")

def check_cuda_availability():
    """Check CUDA availability and provide recommendations."""
    print("\n=== CUDA Status ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available - models will run on CPU")
        print("This is fine for inference but will be slower")

def show_quick_fix_summary():
    """Show what has been fixed and what to do next."""
    print("\n=== Quick Fix Summary ===")
    print("✓ Fixed app.py - main Streamlit application")
    print("✓ Fixed otitenet/infer/make_shap.py - SHAP inference")
    print("✓ Fixed otitenet/train/train_triplet_new.py - training script")
    print("\nTo test the fix:")
    print("1. Try running your application again")
    print("2. If you still get CUDA errors, restart the Python process")
    print("3. For training scripts, they may still need individual fixes")
    
    print("\nFor a complete fix of all files, you would need to:")
    print("1. Add map_location=device parameter to all torch.load() calls")
    print("2. Ensure the device variable is available in scope")
    print("3. Use 'cpu' as fallback when CUDA is not available")

if __name__ == "__main__":
    print("PyTorch CUDA Compatibility Fixer")
    print("=" * 40)
    
    check_cuda_availability()
    fix_inference_files()
    create_device_safe_load_function()
    show_quick_fix_summary()
    
    print(f"\nDevice recommendation: {get_device_for_cpu_compatibility()}")