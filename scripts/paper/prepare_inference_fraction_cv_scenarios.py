#!/usr/bin/env python3
"""Prepare rotating inference-only validation/test folds and calibration fractions.

Historical samples are always assigned to training.  The inference samples are
partitioned once with StratifiedGroupKFold.  For CV run i, fold i is validation,
fold i+1 (cyclically) is test, and calibration samples are drawn from the other
folds.  Calibration subsets are nested within each CV run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_FRACTIONS = "0.5,0.25,0.1,0.05,0.02,0"


def label_for(fraction: float) -> str:
    return (f"{fraction:.3f}".rstrip("0").rstrip(".") or "0").replace(".", "p")


def allocation(labels: pd.Series, n_samples: int) -> dict[str, int]:
    counts = labels.astype(str).value_counts().sort_index().to_dict()
    total = sum(counts.values())
    if not 0 <= n_samples <= total:
        raise ValueError(f"Requested {n_samples} samples from a pool of {total}")
    if n_samples == 0:
        return {label: 0 for label in counts}

    quotas = {label: n_samples * count / total for label, count in counts.items()}
    selected = {label: int(np.floor(quota)) for label, quota in quotas.items()}
    remainder = n_samples - sum(selected.values())
    order = sorted(
        counts,
        key=lambda label: (quotas[label] - selected[label], counts[label], label),
        reverse=True,
    )
    for label in order[:remainder]:
        selected[label] += 1

    if n_samples >= len(counts):
        for label in counts:
            if selected[label] > 0:
                continue
            donors = [candidate for candidate in counts if selected[candidate] > 1]
            if not donors:
                raise ValueError("Cannot keep every class represented in calibration")
            donor = max(donors, key=lambda candidate: (selected[candidate], counts[candidate], candidate))
            selected[label] = 1
            selected[donor] -= 1
    return selected


def make_fold_assignments(
    inference: pd.DataFrame,
    n_splits: int,
    seed: int,
    group_column: str,
) -> pd.Series:
    if group_column not in inference.columns:
        raise ValueError(
            f"Group column {group_column!r} is absent; available columns: "
            f"{inference.columns.tolist()}"
        )
    if inference[group_column].isna().any():
        raise ValueError(f"Group column {group_column!r} contains missing values")

    labels = inference["label"].astype(str).to_numpy()
    groups = inference[group_column].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments = np.zeros(len(inference), dtype=np.int64)
    dummy = np.zeros(len(inference), dtype=np.int8)
    for fold, (_, held_out) in enumerate(splitter.split(dummy, labels, groups), start=1):
        if np.any(assignments[held_out] != 0):
            raise AssertionError("An inference sample was assigned to multiple folds")
        assignments[held_out] = fold
    if np.any(assignments == 0):
        raise AssertionError("At least one inference sample was not assigned to a fold")

    check = pd.DataFrame({"group": groups, "fold": assignments}).groupby("group")["fold"].nunique()
    if (check != 1).any():
        raise AssertionError("A StratifiedGroupKFold group crosses fold boundaries")
    return pd.Series(assignments, index=inference.index, name="cv_fold")


def stratified_queues(pool: pd.DataFrame, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    return {
        label: rng.permutation(
            pool.index[pool["label"].astype(str).eq(label)].to_numpy()
        ).astype(int).tolist()
        for label in sorted(pool["label"].astype(str).unique())
    }


def select_from_queues(pool: pd.DataFrame, queues: dict[str, list[int]], n_samples: int) -> set[int]:
    requested = allocation(pool["label"], n_samples)
    return {
        index
        for label, count in requested.items()
        for index in queues[label][:count]
    }


def link_images(base: Path, target: Path, names: pd.Series, mode: str) -> None:
    for name in sorted(set(names.astype(str))):
        source = base / name
        destination = target / name
        if destination.exists() or destination.is_symlink():
            continue
        if not source.exists():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            os.link(source, destination)
        elif mode == "symlink":
            os.symlink(os.path.relpath(source, destination.parent), destination)
        else:
            shutil.copy2(source, destination)


def validate_manifest(rows: list[dict[str, object]], n_splits: int, fractions: list[float]) -> None:
    manifest = pd.DataFrame(rows)
    expected = n_splits * len(fractions)
    if len(manifest) != expected:
        raise AssertionError(f"Expected {expected} scenarios, found {len(manifest)}")
    for fold in range(1, n_splits + 1):
        subset = manifest[manifest["cv_run"].eq(fold)]
        if subset["scenario_fraction"].tolist() != fractions:
            raise AssertionError(f"Unexpected fraction order for CV run {fold}")
        if subset["valid_fold"].nunique() != 1 or int(subset["valid_fold"].iloc[0]) != fold:
            raise AssertionError(f"Unexpected validation fold for CV run {fold}")
        expected_test = fold % n_splits + 1
        if subset["test_fold"].nunique() != 1 or int(subset["test_fold"].iloc[0]) != expected_test:
            raise AssertionError(f"Unexpected test fold for CV run {fold}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-base", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--fractions", default=DEFAULT_FRACTIONS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-column",
        default="name",
        help="Entity ID kept intact by StratifiedGroupKFold. Default name treats each image as one group.",
    )
    parser.add_argument("--image-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    args = parser.parse_args()

    fractions = [float(value) for value in args.fractions.split(",") if value.strip()]
    if fractions != sorted(fractions, reverse=True):
        raise ValueError("Fractions must be supplied in descending order for nested calibration sets")
    if fractions[-1] != 0.0:
        raise ValueError("The final fraction must be 0 (the historical-only baseline)")

    infos_path = args.base_dir / "infos.csv"
    base = pd.read_csv(infos_path)
    required = {"dataset", "name", "label"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"Missing required infos.csv columns: {sorted(missing)}")
    inference = base[base["dataset"].eq("inference")].copy()
    historical = base[~base["dataset"].eq("inference")].copy()
    if inference.empty or historical.empty:
        raise ValueError("Both inference and historical rows are required")

    fold_assignments = make_fold_assignments(
        inference,
        n_splits=args.n_splits,
        seed=args.seed,
        group_column=args.group_column,
    )
    inference["cv_fold"] = fold_assignments
    args.out_base.mkdir(parents=True, exist_ok=True)
    fold_manifest_path = args.out_base / f"{args.prefix}_seed{args.seed}_folds.csv"
    fold_manifest = inference[["name", "label", args.group_column]].copy()
    if args.group_column == "name":
        fold_manifest = fold_manifest.loc[:, ~fold_manifest.columns.duplicated()]
    fold_manifest["cv_fold"] = inference["cv_fold"].to_numpy()
    fold_manifest.to_csv(fold_manifest_path, index=False)
    fold_sha = hashlib.sha256(fold_manifest_path.read_bytes()).hexdigest()

    rows: list[dict[str, object]] = []
    total_inference = len(inference)
    for cv_run in range(1, args.n_splits + 1):
        valid_fold = cv_run
        test_fold = cv_run % args.n_splits + 1
        valid_idx = set(inference.index[inference["cv_fold"].eq(valid_fold)].astype(int))
        test_idx = set(inference.index[inference["cv_fold"].eq(test_fold)].astype(int))
        candidate = inference[~inference["cv_fold"].isin([valid_fold, test_fold])]
        queues = stratified_queues(candidate, args.seed + cv_run * 10_000)
        previous_selection: set[int] | None = None

        for fraction in fractions:
            n_calibration = int(round(total_inference * fraction))
            selected = select_from_queues(candidate, queues, n_calibration)
            if previous_selection is not None and not selected.issubset(previous_selection):
                raise AssertionError(
                    f"Calibration sets are not nested for CV run {cv_run}, fraction {fraction}"
                )
            previous_selection = selected
            if selected & valid_idx or selected & test_idx or valid_idx & test_idx:
                raise AssertionError("Calibration, validation, and test overlap")

            scenario_label = label_for(fraction)
            output_dir = args.out_base / (
                f"{args.prefix}_cv{cv_run}_train{scenario_label}_seed{args.seed}"
            )
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"Refusing to overwrite existing scenario {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

            output = base.copy()
            output["group"] = "unused"
            output.loc[historical.index, "group"] = "train"
            output.loc[list(selected), "group"] = "train"
            output.loc[list(valid_idx), "group"] = "valid"
            output.loc[list(test_idx), "group"] = "test"
            output.to_csv(output_dir / "infos.csv", index=False)
            link_images(args.base_dir, output_dir, output["name"], args.image_mode)

            inference_groups = output.loc[inference.index, "group"]
            summary = {
                "cv_run": cv_run,
                "valid_fold": valid_fold,
                "test_fold": test_fold,
                "scenario_fraction": fraction,
                "scenario_label": scenario_label,
                "seed": args.seed,
                "group_column": args.group_column,
                "dataset_path": str(output_dir),
                "total_inference": total_inference,
                "inference_train": int(inference_groups.eq("train").sum()),
                "inference_valid": int(inference_groups.eq("valid").sum()),
                "inference_test": int(inference_groups.eq("test").sum()),
                "inference_unused": int(inference_groups.eq("unused").sum()),
                "historical_train": int(output.loc[historical.index, "group"].eq("train").sum()),
                "fold_manifest_sha256": fold_sha,
                "infos_sha256": hashlib.sha256((output_dir / "infos.csv").read_bytes()).hexdigest(),
            }
            (output_dir / "scenario_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            rows.append(summary)

    validate_manifest(rows, args.n_splits, fractions)
    scenario_manifest = args.out_base / f"{args.prefix}_seed{args.seed}_scenarios.csv"
    pd.DataFrame(rows).to_csv(scenario_manifest, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote fold manifest: {fold_manifest_path}")
    print(f"Wrote {len(rows)} scenarios: {scenario_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
