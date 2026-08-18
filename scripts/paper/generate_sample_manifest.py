#!/usr/bin/env python3
"""
Generate a complete sample-level manifest for inference-fraction experiments.
Tracks which samples went into train/valid/test and their metadata.
Writes to both run directory and a global registry for debugging.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def generate_sample_manifest(
    infos_csv: Path,
    output_dir: Path,
    run_id: str,
    scenario_label: str,
    n_calibration: int,
) -> Path:
    """
    Read infos.csv and output sample assignments with metadata.
    
    Args:
        infos_csv: Path to infos.csv from dataset
        output_dir: Where to write the manifest
        run_id: Experiment run tag
        scenario_label: Fraction label (0p5, 0p25, etc.)
        n_calibration: Calibration set size for this run
    
    Returns:
        Path to generated manifest file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read infos.csv
    df = pd.read_csv(infos_csv)
    
    # Ensure required columns exist
    required = {"path", "set", "class"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in infos.csv: {missing}")
    
    # Extract metadata
    df["sample_id"] = df["path"].apply(lambda x: Path(x).stem)
    df["split"] = df["set"]  # set column = train/valid/test assignment
    df["class_label"] = df["class"]
    
    # Output columns in sensible order
    output_cols = ["sample_id", "split", "class_label", "path"]
    extra_cols = [c for c in df.columns if c not in output_cols + ["set", "class"]]
    output_cols.extend(extra_cols)
    
    manifest_df = df[output_cols].copy()
    
    # Write to run directory
    run_manifest = output_dir / "sample_assignments.csv"
    manifest_df.to_csv(run_manifest, index=False)
    
    # Write with metadata header
    with run_manifest.open("r") as f:
        content = f.read()
    
    header = f"""# Sample Assignment Manifest
# Run ID: {run_id}
# Scenario: {scenario_label}
# Calibration Count: {n_calibration}
# Total Samples: {len(manifest_df)}
# Breakdown: train={len(manifest_df[manifest_df['split']=='train'])}, valid={len(manifest_df[manifest_df['split']=='valid'])}, test={len(manifest_df[manifest_df['split']=='test'])}
#
"""
    
    with run_manifest.open("w") as f:
        f.write(header)
        f.write(content)
    
    return run_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample-level manifest for inference-fraction runs."
    )
    parser.add_argument("--infos-csv", required=True, help="Path to infos.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for manifest")
    parser.add_argument("--run-id", required=True, help="Run tag/ID")
    parser.add_argument("--scenario-label", required=True, help="Scenario label (0p5, etc.)")
    parser.add_argument("--n-calibration", type=int, required=True, help="Calibration set size")
    
    args = parser.parse_args()
    
    manifest_path = generate_sample_manifest(
        Path(args.infos_csv),
        Path(args.output_dir),
        args.run_id,
        args.scenario_label,
        args.n_calibration,
    )
    
    print(f"Generated: {manifest_path}")


if __name__ == "__main__":
    main()
