"""Build stable CSV registries for datasets and best-model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from otitenet.app.artifact_registry import (
    BEST_MODELS_REGISTRY_PATH,
    DATASET_REGISTRY_PATH,
    scan_best_models_registry,
    scan_dataset_registry,
)


def _write_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--best-models-dir", default="logs/best_models")
    parser.add_argument("--datasets-out", default=str(DATASET_REGISTRY_PATH))
    parser.add_argument("--models-out", default=str(BEST_MODELS_REGISTRY_PATH))
    parser.add_argument(
        "--print-dvc-commands",
        action="store_true",
        help="Print the DVC commands to run after reviewing the generated registries.",
    )
    args = parser.parse_args()

    datasets = scan_dataset_registry(args.data_dir)
    models = scan_best_models_registry(args.best_models_dir)

    _write_csv(datasets, Path(args.datasets_out))
    _write_csv(models, Path(args.models_out))

    if args.print_dvc_commands:
        dvc_targets = (
            datasets.loc[datasets["ignored"].astype(str) != "yes", "dvc_target"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        print()
        print("Suggested DVC commands:")
        for target in dvc_targets:
            print(f"  dvc add {target}")
        print("  dvc add logs/best_models")
        print("  dvc push")
        print("  git add configs/datasets.csv configs/best_models.csv data/**/*.dvc logs/best_models.dvc .gitignore")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
