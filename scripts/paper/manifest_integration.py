#!/usr/bin/env python3
"""
Integration helpers for sample manifests in training pipelines.
Call after training to automatically generate and register manifests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def generate_and_register_manifest(
    dataset_path: str,
    run_output_dir: Path,
    run_id: str,
    scenario_label: str,
    n_calibration: int,
    root: Path = None,
) -> None:
    """
    Generate sample manifest and register it in global registry.
    
    Args:
        dataset_path: Path to dataset (containing infos.csv)
        run_output_dir: Output directory for this run (where logs go)
        run_id: Run tag
        scenario_label: Scenario label (0p5, etc.)
        n_calibration: Calibration size
        root: Root project directory (auto-detected if None)
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]
    
    dataset_dir = root / dataset_path if not Path(dataset_path).is_absolute() else Path(dataset_path)
    infos_csv = dataset_dir / "infos.csv"
    
    if not infos_csv.exists():
        print(f"⚠ infos.csv not found: {infos_csv}")
        return
    
    scripts_dir = root / "scripts" / "paper"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate manifest in run directory
    print(f"Generating sample manifest for {run_id}...")
    cmd_gen = [
        str(root / ".conda" / "bin" / "python"),
        str(scripts_dir / "generate_sample_manifest.py"),
        "--infos-csv", str(infos_csv),
        "--output-dir", str(run_output_dir),
        "--run-id", run_id,
        "--scenario-label", scenario_label,
        "--n-calibration", str(n_calibration),
    ]
    
    try:
        subprocess.run(cmd_gen, check=True, cwd=root)
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate manifest: {e}")
        return
    
    # Register in global registry
    run_manifest = run_output_dir / "sample_assignments.csv"
    if run_manifest.exists():
        print(f"Registering manifest in global registry...")
        cmd_reg = [
            str(root / ".conda" / "bin" / "python"),
            str(scripts_dir / "register_sample_manifest.py"),
            "--run-manifest", str(run_manifest),
            "--run-id", run_id,
            "--scenario-label", scenario_label,
            "--n-calibration", str(n_calibration),
        ]
        
        try:
            subprocess.run(cmd_reg, check=True, cwd=root)
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to register manifest: {e}")
    else:
        print(f"✗ Manifest not found: {run_manifest}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and register sample manifests.")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset")
    parser.add_argument("--run-output-dir", required=True, help="Run output directory")
    parser.add_argument("--run-id", required=True, help="Run tag")
    parser.add_argument("--scenario-label", required=True, help="Scenario label")
    parser.add_argument("--n-calibration", type=int, required=True, help="Calibration size")
    
    args = parser.parse_args()
    
    generate_and_register_manifest(
        args.dataset_path,
        Path(args.run_output_dir),
        args.run_id,
        args.scenario_label,
        args.n_calibration,
    )
