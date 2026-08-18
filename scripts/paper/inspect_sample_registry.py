#!/usr/bin/env python3
"""
Quick debugging tool to inspect the global sample registry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def list_manifests(registry_dir: Path) -> None:
    """List all registered manifests with summary stats."""
    if not registry_dir.exists():
        print(f"Registry not found: {registry_dir}")
        return
    
    manifests = sorted(registry_dir.glob("*.csv"))
    if not manifests:
        print(f"No manifests in {registry_dir}")
        return
    
    print(f"\n{Run ID:<40} {Scenario:<10} {Train:<6} {Valid:<6} {Test:<6} {Total:<6}")
    print("-" * 85)
    
    for manifest in manifests:
        try:
            df = pd.read_csv(manifest, comment="#")
            run_id = manifest.stem.rsplit("_", 1)[0].split("_")[0]
            scenario = manifest.stem.split("_")[-3] if "_" in manifest.stem else "?"
            
            train = len(df[df["split"] == "train"])
            valid = len(df[df["split"] == "valid"])
            test = len(df[df["split"] == "test"])
            total = len(df)
            
            print(f"{run_id:<40} {scenario:<10} {train:<6} {valid:<6} {test:<6} {total:<6}")
        except Exception as e:
            print(f"Error reading {manifest.name}: {e}")


def inspect_manifest(manifest_path: Path) -> None:
    """Detailed inspection of a single manifest."""
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return
    
    # Read header comments
    print("\n[Metadata]")
    with manifest_path.open() as f:
        for line in f:
            if line.startswith("#"):
                print(line.rstrip())
            else:
                break
    
    # Read data
    df = pd.read_csv(manifest_path, comment="#")
    
    print(f"\n[Data Summary]")
    print(f"Total samples: {len(df)}")
    print(f"\nSplit breakdown:")
    for split, count in df["split"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {split:<8} {count:6d} ({pct:5.1f}%)")
    
    if "class_label" in df.columns:
        print(f"\nClass breakdown:")
        for cls, count in df["class_label"].value_counts().items():
            pct = 100 * count / len(df)
            print(f"  {cls:<20} {count:6d} ({pct:5.1f}%)")
    
    print(f"\n[First 10 samples]")
    display_cols = ["sample_id", "split", "class_label"] if "class_label" in df.columns else ["sample_id", "split"]
    print(df[display_cols].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the global sample registry."
    )
    parser.add_argument(
        "--registry-dir",
        default="paper_outputs/SAMPLE_MANIFESTS",
        help="Registry directory",
    )
    parser.add_argument(
        "--inspect",
        help="Inspect a specific manifest file",
    )
    
    args = parser.parse_args()
    
    registry_dir = Path(args.registry_dir)
    
    if args.inspect:
        manifest = Path(args.inspect)
        if not manifest.is_absolute():
            manifest = registry_dir / manifest
        inspect_manifest(manifest)
    else:
        list_manifests(registry_dir)


if __name__ == "__main__":
    main()
