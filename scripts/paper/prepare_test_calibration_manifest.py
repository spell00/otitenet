#!/usr/bin/env python3
"""Create a stratified calibration manifest from held-out evaluation splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _label_column(df: pd.DataFrame) -> str:
    for col in ("label", "labels", "old_label", "class"):
        if col in df.columns:
            return col
    raise ValueError(f"No label column found. Columns: {list(df.columns)}")


def _name_column(df: pd.DataFrame) -> str:
    for col in ("name", "names", "image", "path", "filename"):
        if col in df.columns:
            return col
    raise ValueError(f"No image-name column found. Columns: {list(df.columns)}")


def _proportional_label_counts(labels: list[str], counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(int(counts[label]) for label in labels)
    if n > total:
        raise ValueError(f"n={n} exceeds the available split size ({total}).")
    if n < len(labels):
        raise ValueError(f"n={n} is smaller than the number of labels in the test split ({len(labels)}).")

    quotas = {label: (n * int(counts[label])) / total for label in labels}
    requested = {label: int(np.floor(quotas[label])) for label in labels}
    remaining = n - sum(requested.values())
    by_remainder = sorted(
        labels,
        key=lambda label: (quotas[label] - requested[label], int(counts[label]), label),
        reverse=True,
    )
    for label in by_remainder[:remaining]:
        requested[label] += 1

    # Keep every class represented when the requested support size allows it.
    for label in labels:
        if requested[label] > 0:
            continue
        requested[label] = 1
        donor = max(
            (candidate for candidate in labels if requested[candidate] > 1),
            key=lambda candidate: (requested[candidate] - quotas[candidate], requested[candidate], candidate),
        )
        requested[donor] -= 1
    return requested


def _sample_split(split_csv: Path, n: int, seed: int, source_group: str) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    if df.empty:
        raise ValueError(f"Split CSV is empty: {split_csv}")

    label_col = _label_column(df)
    name_col = _name_column(df)
    labels = sorted(df[label_col].dropna().astype(str).unique().tolist())
    label_values = df[label_col].astype(str)
    counts = {label: int((label_values == label).sum()) for label in labels}
    requested_by_label = _proportional_label_counts(labels, counts, n)

    rng = np.random.default_rng(seed)
    rows = []
    for label in labels:
        take = requested_by_label[label]
        pool = df[label_values == label]
        if len(pool) < take:
            raise ValueError(
                f"Not enough {source_group} images for label {label}: need {take}, found {len(pool)}"
            )
        chosen = pool.iloc[rng.choice(len(pool), size=take, replace=False)].copy()
        rows.append(chosen)

    manifest = pd.concat(rows, ignore_index=True)
    manifest = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out = pd.DataFrame(
        {
            "name": manifest[name_col].astype(str),
            "label": manifest[label_col].astype(str),
            "source_group": source_group,
        }
    )
    if "batch" in manifest.columns:
        out["batch"] = manifest["batch"].astype(str)
    return out


def build_manifest(
    split_csv: Path,
    n: int,
    seed: int,
    output: Path,
    extra_splits: list[tuple[str, Path]] | None = None,
) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be > 0 for a calibration manifest")

    split_specs = [("test", split_csv), *(extra_splits or [])]
    parts = [
        _sample_split(path, n, seed + offset, source_group)
        for offset, (source_group, path) in enumerate(split_specs)
    ]
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a proportional stratified calibration manifest.")
    parser.add_argument("--split-csv", required=True, type=Path, help="Path to an E01 run splits/test.csv file.")
    parser.add_argument("--valid-split-csv", type=Path, help="Optional E01 run splits/valid.csv file; samples n rows from valid too.")
    parser.add_argument("--n", required=True, type=int, help="Number of images per evaluation split to move into calibration/train.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    extra_splits = [("valid", args.valid_split_csv)] if args.valid_split_csv else None
    out = build_manifest(args.split_csv, args.n, args.seed, args.output, extra_splits=extra_splits)
    counts = out.groupby("source_group")["label"].value_counts().sort_index().to_dict()
    print(f"Wrote {len(out)} calibration rows to {args.output}")
    print(f"Label counts by source: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
