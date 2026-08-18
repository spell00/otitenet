#!/usr/bin/env python3
"""
Register a run-specific sample manifest in the global debugging registry.
Call after training completes to ensure all runs are tracked.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def register_run_manifest(
    run_manifest: Path,
    run_id: str,
    scenario_label: str,
    n_calibration: int,
    global_registry: Path = None,
) -> Path:
    """
    Copy run manifest to global registry with metadata.
    
    Args:
        run_manifest: Path to sample_assignments.csv in run directory
        run_id: Experiment run tag
        scenario_label: Scenario label (0p5, etc.)
        n_calibration: Calibration set size
        global_registry: Global registry directory (default: paper_outputs/SAMPLE_MANIFESTS)
    
    Returns:
        Path to registered manifest
    """
    if global_registry is None:
        global_registry = Path(__file__).resolve().parents[2] / "paper_outputs" / "SAMPLE_MANIFESTS"
    
    global_registry.mkdir(parents=True, exist_ok=True)
    
    # Read original manifest
    df = pd.read_csv(run_manifest, comment="#")
    
    # Generate registry filename
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "").replace("-", "")
    registry_file = global_registry / f"{run_id}_{scenario_label}_{timestamp}.csv"
    
    # Write with enhanced metadata
    with registry_file.open("w") as f:
        f.write(f"# Global Sample Registry\n")
        f.write(f"# Run ID: {run_id}\n")
        f.write(f"# Scenario: {scenario_label}\n")
        f.write(f"# Calibration Count: {n_calibration}\n")
        f.write(f"# Registered: {datetime.now().isoformat()}\n")
        f.write(f"# Source: {run_manifest.absolute()}\n")
        f.write(f"# Total Samples: {len(df)}\n")
        split_breakdown = {split: len(df[df['split']==split]) for split in df['split'].unique()}
        for split, count in sorted(split_breakdown.items()):
            f.write(f"# {split.upper()}: {count}\n")
        f.write("#\n")
    
    # Append data
    df.to_csv(registry_file, mode="a", index=False)
    
    return registry_file


def main():
    parser = argparse.ArgumentParser(
        description="Register a run manifest in the global sample registry."
    )
    parser.add_argument("--run-manifest", required=True, help="Path to sample_assignments.csv")
    parser.add_argument("--run-id", required=True, help="Run tag/ID")
    parser.add_argument("--scenario-label", required=True, help="Scenario label (0p5, etc.)")
    parser.add_argument("--n-calibration", type=int, required=True, help="Calibration set size")
    parser.add_argument("--global-registry", help="Global registry directory (optional)")
    
    args = parser.parse_args()
    
    registry_path = register_run_manifest(
        Path(args.run_manifest),
        args.run_id,
        args.scenario_label,
        args.n_calibration,
        Path(args.global_registry) if args.global_registry else None,
    )
    
    print(f"Registered: {registry_path}")


if __name__ == "__main__":
    main()
