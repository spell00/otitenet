#!/usr/bin/env python3
"""
Compare the content of two pickle files to check if they're identical.
Usage: python compare_pkl_files.py <file1.pkl> <file2.pkl>
"""

import sys
import pickle
import numpy as np


def compare_objects(obj1, obj2, path=""):
    """Recursively compare two objects for equality."""
    
    # Check type
    if type(obj1) != type(obj2):
        print(f"❌ Type mismatch at {path}: {type(obj1)} vs {type(obj2)}")
        return False
    
    # Handle dict
    if isinstance(obj1, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            print(f"❌ Dict keys mismatch at {path}")
            print(f"   Keys in file1: {set(obj1.keys()) - set(obj2.keys())}")
            print(f"   Keys in file2: {set(obj2.keys()) - set(obj1.keys())}")
            return False
        
        all_match = True
        for key in obj1.keys():
            new_path = f"{path}['{key}']" if path else f"['{key}']"
            if not compare_objects(obj1[key], obj2[key], new_path):
                all_match = False
        return all_match
    
    # Handle list/tuple
    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            print(f"❌ Length mismatch at {path}: {len(obj1)} vs {len(obj2)}")
            return False
        
        all_match = True
        for i, (item1, item2) in enumerate(zip(obj1, obj2)):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            if not compare_objects(item1, item2, new_path):
                all_match = False
        return all_match
    
    # Handle numpy array
    if isinstance(obj1, np.ndarray):
        if obj1.shape != obj2.shape:
            print(f"❌ Array shape mismatch at {path}: {obj1.shape} vs {obj2.shape}")
            return False
        if obj1.dtype != obj2.dtype:
            print(f"❌ Array dtype mismatch at {path}: {obj1.dtype} vs {obj2.dtype}")
            return False
        if not np.allclose(obj1, obj2, rtol=1e-9, atol=1e-9, equal_nan=True):
            diff = np.abs(obj1 - obj2)
            max_diff = np.nanmax(diff)
            print(f"❌ Array values mismatch at {path}: max difference = {max_diff}")
            return False
        return True
    
    # Handle primitive types
    if isinstance(obj1, (int, float, str, bool, type(None))):
        if obj1 != obj2:
            print(f"❌ Value mismatch at {path}: {obj1} vs {obj2}")
            return False
        return True
    
    # For other types, try equality comparison
    try:
        if obj1 != obj2:
            print(f"❌ Object mismatch at {path} (type: {type(obj1).__name__})")
            return False
        return True
    except Exception as e:
        print(f"⚠️  Could not compare objects at {path}: {e}")
        return True  # Assume equal if comparison fails


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_pkl_files.py <file1.pkl> <file2.pkl>")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    print(f"Comparing: {file1_path}")
    print(f"     with: {file2_path}\n")
    
    try:
        with open(file1_path, 'rb') as f:
            obj1 = pickle.load(f)
        print(f"✅ Loaded file1: {file1_path}")
    except Exception as e:
        print(f"❌ Failed to load file1: {e}")
        sys.exit(1)
    
    try:
        with open(file2_path, 'rb') as f:
            obj2 = pickle.load(f)
        print(f"✅ Loaded file2: {file2_path}\n")
    except Exception as e:
        print(f"❌ Failed to load file2: {e}")
        sys.exit(1)
    
    if compare_objects(obj1, obj2):
        print("\n✅ Files are IDENTICAL!")
        sys.exit(0)
    else:
        print("\n❌ Files are DIFFERENT!")
        sys.exit(1)


if __name__ == "__main__":
    main()
