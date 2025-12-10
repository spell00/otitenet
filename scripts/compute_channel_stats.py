#!/usr/bin/env python3
"""Utility to recompute per-channel image normalisation statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch
from PIL import Image
from torchvision import transforms

RGB_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.suffix.lower() in RGB_EXTENSIONS:
            yield path


def compute_channel_stats(image_paths: Iterable[Path], target_size: int | None) -> Tuple[torch.Tensor, torch.Tensor]:
    pixel_sum = torch.zeros(3, dtype=torch.double)
    pixel_sq_sum = torch.zeros(3, dtype=torch.double)
    pixel_count = 0

    if target_size is None:
        transformer = transforms.ToTensor()
    else:
        transformer = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])

    for path in image_paths:
        with Image.open(path) as img:
            tensor = transformer(img.convert("RGB"))
        c, h, w = tensor.shape
        reshaped = tensor.view(c, -1)
        pixel_sum += reshaped.sum(dim=1).to(pixel_sum.dtype)
        pixel_sq_sum += (reshaped ** 2).sum(dim=1).to(pixel_sq_sum.dtype)
        pixel_count += h * w

    if pixel_count == 0:
        raise RuntimeError("No images found under the given directory.")

    mean = pixel_sum / pixel_count
    std = torch.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    return mean.float(), std.float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-channel mean/std for dataset normalisation.")
    parser.add_argument("image_dir", type=Path, help="Directory containing RGB images (searched recursively).")
    parser.add_argument("--size", type=int, default=224, help="Resize square edge before stats; use -1 to disable.")
    args = parser.parse_args()

    image_dir: Path = args.image_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    size = None if args.size == -1 else args.size
    mean, std = compute_channel_stats(list_images(image_dir), size)

    print("mean =", [round(v, 8) for v in mean.tolist()])
    print("std  =", [round(v, 8) for v in std.tolist()])


if __name__ == "__main__":
    main()
