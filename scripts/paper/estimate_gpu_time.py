#!/usr/bin/env python3
"""Estimate OtiteNet paper experiment runtimes from existing run summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        return out or "unknown GPU"
    except Exception:
        return "GPU not detected by nvidia-smi"


def collect() -> pd.DataFrame:
    rows = []
    for summary_path in (ROOT / "logs" / "otitis_four_class").glob("*/run_summary.json"):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        meta_path = summary_path.parent / "run_metadata.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        args = meta.get("args", {}) if isinstance(meta, dict) else {}
        duration = summary.get("duration_seconds")
        if duration is None:
            continue
        rows.append(
            {
                "run_dir": str(summary_path.parent),
                "duration_minutes": float(duration) / 60.0,
                "run_tag": args.get("run_tag", ""),
                "model_name": args.get("model_name", ""),
                "task": args.get("task", ""),
                "n_calibration": args.get("n_calibration", ""),
                "fgsm": args.get("fgsm", ""),
                "classif_loss": args.get("classif_loss", ""),
                "dloss": args.get("dloss", ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate GPU runtimes for paper experiments.")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_outputs" / "tables" / "gpu_time_estimates.csv")
    args = parser.parse_args()

    df = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"GPU: {_gpu_name()}")
    if df.empty:
        print("No run_summary.json durations found.")
        return 0
    top_like = df[(df["model_name"] == "resnet18") & (df["task"] == "otite_four_class")]
    use = top_like if not top_like.empty else df
    print("Observed run duration, minutes:")
    print(use["duration_minutes"].describe(percentiles=[0.25, 0.5, 0.75]).to_string())
    print(f"Wrote per-run estimates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
