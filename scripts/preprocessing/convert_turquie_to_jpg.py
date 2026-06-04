#!/usr/bin/env python3
"""
Convert Banque_Comert_Turquie_2020 dataset to Banque_Comert_Turquie_2020_jpg:
- Converts all images (tiff, jpeg, png, etc.) to .jpg
- Preserves label subfolders
- Skips duplicate base names (keeps first found)
- Overwrites target directory
"""
import os
import shutil
from PIL import Image

SRC = "data/datasets/Banque_Comert_Turquie_2020"
DST = "data/datasets/Banque_Comert_Turquie_2020_jpg"

# Remove old target dir if exists
if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(DST, exist_ok=True)

for label in os.listdir(SRC):
    src_label_dir = os.path.join(SRC, label)
    dst_label_dir = os.path.join(DST, label)
    if not os.path.isdir(src_label_dir):
        continue
    os.makedirs(dst_label_dir, exist_ok=True)
    seen_basenames = set()
    for fname in os.listdir(src_label_dir):
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        if base in seen_basenames:
            continue
        if ext not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            continue
        src_path = os.path.join(src_label_dir, fname)
        dst_path = os.path.join(dst_label_dir, base + ".jpg")
        try:
            img = Image.open(src_path).convert("RGB")
            img.save(dst_path, "JPEG", quality=95)
            seen_basenames.add(base)
            print(f"Converted {src_path} -> {dst_path}")
        except Exception as e:
            print(f"[ERROR] Could not convert {src_path}: {e}")
print("Done.")
