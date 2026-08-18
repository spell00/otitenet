#!/usr/bin/env python3
"""Build complete fresh inference-fraction tables.

For CNN/MLP, prefer the newer INF_FRAC_FRESH_OPTUNA_CNN_MLP file for a
fraction and fall back to INF_FRAC_FRESH_CNN_MLP when the newer run is not
ready.  For Siamese, merge INF_FRAC_FRESH_OPTUNA_SIAMESE and
INF_FRAC_FRESH_SIAMESE, then add any available retrained-head results.

Two files are written:
  * all candidates (audit/provenance table)
  * one validation-selected winner per fraction

Test metrics never participate in selection.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/simon/otitenet")
PROGRESS_ROOT = PROJECT_ROOT / "logs" / "progresses" / "four_classes_220726"
FRACTIONS = ["0p5", "0p25", "0p1", "0p05", "0p02", "0p0"]
N_CALIBRATION = {"0p5": 131, "0p25": 66, "0p1": 26, "0p05": 13, "0p02": 5, "0p0": 0}

DEFAULT_OUTPUT = PROJECT_ROOT / "inference_fraction_fresh_wide_6_rows.csv"
DEFAULT_ALL_OUTPUT = PROJECT_ROOT / "inference_fraction_fresh_all_candidates.csv"
DEFAULT_SUPPLEMENTAL = (
    PROJECT_ROOT / "paper_outputs" / "siamese_head_optuna" / "app_retrained_head_results.csv"
)


def fraction_dir(fraction: str) -> Path:
    directory_fraction = "0" if fraction == "0p0" else fraction
    return PROGRESS_ROOT / (
        "home_simon_otitenet_data_otite_ds_64_USA_Turquie_Chili_GMFUNL_"
        f"inference_fraction_hist_v2_train{directory_fraction}_seed42"
    )


def first_match(patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def cnn_source(fraction: str) -> tuple[Path | None, str]:
    """Prefer newer Optuna CNN/MLP; use older fresh CNN/MLP as fallback."""
    directory = fraction_dir(fraction)
    if fraction == "0p0":
        zero = first_match(
            [str(directory / "INF_FRAC_E03_PREVBEST_P0_R01*completed_runs_metrics.csv")]
        )
        return zero, "cnn_mlp_zero_partial" if zero else "cnn_mlp_missing"
    preferred = first_match(
        [
            str(directory / f"INF_FRAC_FRESH_OPTUNA_CNN_MLP_P{fraction}_S42*completed_runs_metrics.csv"),
            str(PROGRESS_ROOT / "**" / f"INF_FRAC_FRESH_OPTUNA_CNN_MLP_P{fraction}_S42*completed_runs_metrics.csv"),
        ]
    )
    if preferred:
        return preferred, "cnn_mlp_fresh_optuna"

    fallback = first_match(
        [
            str(directory / f"INF_FRAC_FRESH_CNN_MLP_P{fraction}_S42*completed_runs_metrics.csv"),
            str(PROGRESS_ROOT / f"INF_FRAC_FRESH_CNN_MLP_P{fraction}_S42*completed_runs_metrics.csv"),
            str(PROGRESS_ROOT / "**" / f"INF_FRAC_FRESH_CNN_MLP_P{fraction}_S42*completed_runs_metrics.csv"),
        ]
    )
    return fallback, "cnn_mlp_fresh_fallback" if fallback else "cnn_mlp_missing"


def siamese_sources(fraction: str) -> list[tuple[Path, str]]:
    if fraction == "0p0":
        directory = fraction_dir(fraction)
        specs = [
            (
                "INF_FRAC_FRESH_OPTUNA_SIAMESE_P0_S42*completed_runs_metrics.csv",
                "siamese_fresh_optuna_zero",
            ),
            (
                "INF_FRAC_FRESH_SIAMESE_P0_S42*completed_runs_metrics.csv",
                "siamese_fresh_zero",
            ),
            (
                "INF_FRAC_E03_PREVBEST_P0_R01*completed_runs_metrics.csv",
                "siamese_prevbest_zero",
            ),
        ]
        found: list[tuple[Path, str]] = []
        for filename, source_kind in specs:
            path = first_match([str(directory / filename), str(PROGRESS_ROOT / "**" / filename)])
            if path and all(path != existing for existing, _ in found):
                found.append((path, source_kind))
        return found
    directory = fraction_dir(fraction)
    specs = [
        (f"INF_FRAC_FRESH_OPTUNA_SIAMESE_P{fraction}_S42*completed_runs_metrics.csv", "siamese_fresh_optuna"),
        (f"INF_FRAC_FRESH_SIAMESE_P{fraction}_S42*completed_runs_metrics.csv", "siamese_fresh"),
    ]
    found: list[tuple[Path, str]] = []
    for filename, source_kind in specs:
        path = first_match([str(directory / filename), str(PROGRESS_ROOT / "**" / filename)])
        if path and all(path != existing for existing, _ in found):
            found.append((path, source_kind))
    return found


def clean_metrics(path: Path, expected_kind: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "status" in df and df["status"].notna().any():
        completed = df["status"].astype(str).str.lower().eq("completed")
        if completed.any():
            df = df[completed]
    if expected_kind and "kind" in df and df["kind"].notna().any():
        matches = df["kind"].astype(str).str.lower().eq(expected_kind)
        if matches.any():
            df = df[matches]
    for column in ["valid_mcc", "test_mcc", "test_accuracy", "valid_accuracy"]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["valid_mcc"])


def value(row: pd.Series, *columns: str):
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return np.nan


def best_cnn_candidate(fraction: str) -> dict | None:
    path, source_kind = cnn_source(fraction)
    if path is None:
        return recover_cnn_from_optuna_db(fraction)
    df = clean_metrics(path, "cnn_mlp")
    if source_kind == "cnn_mlp_fresh_fallback" and "variant" in df and df["variant"].notna().any():
        transfer = df[df["variant"].astype(str).str.lower().eq("cnn_transfer")]
        if not transfer.empty:
            df = transfer
    if df.empty:
        return None
    row = df.loc[df["valid_mcc"].idxmax()]
    architecture = value(row, "variant", "model_name", "model")
    if pd.isna(architecture):
        architecture = "cnn_mlp"
    return {
        "fraction": fraction,
        "n_calibration": int(value(row, "n_calibration") or N_CALIBRATION[fraction]),
        "model_family": "CNN/MLP",
        "model_head": str(architecture),
        "valid_mcc": float(row["valid_mcc"]),
        "test_mcc": value(row, "test_mcc"),
        "test_accuracy": value(row, "test_accuracy", "test_acc"),
        "uuid": value(row, "uuid"),
        "n_aug": np.nan,
        "source_kind": source_kind,
        "source": str(path),
        "selection_note": (
            "preferred newer file"
            if source_kind.endswith("optuna")
            else "zero-calibration partial result"
            if source_kind == "cnn_mlp_zero_partial"
            else "fallback: newer file missing"
        ),
    }


def recover_cnn_from_optuna_db(fraction: str) -> dict | None:
    """Recover an older fresh CNN/MLP winner when its metrics CSV is absent.

    The Optuna database identifies the validation-selected trial and stores its
    run_root.  The matching comparison_summary.csv supplies that same trial's
    test metrics, avoiding the historical best-trial/last-trial mismatch.
    """
    db_pattern = (
        PROGRESS_ROOT
        / "tmp"
        / "db"
        / f"INF_FRAC_FRESH_CNN_MLP_P{fraction}_S42*.db"
    )
    databases = sorted(Path(p) for p in glob.glob(str(db_pattern)))
    if not databases:
        return None
    try:
        import optuna
    except ImportError:
        print(f"  WARN cannot recover CNN/MLP {fraction}: optuna is not installed")
        return None

    best_result = None
    for database in databases:
        storage = f"sqlite:///{database.resolve()}"
        for summary in optuna.get_all_study_summaries(storage=storage):
            if summary.best_trial is None:
                continue
            study = optuna.load_study(study_name=summary.study_name, storage=storage)
            trial = study.best_trial
            run_root = trial.user_attrs.get("run_root")
            if not run_root:
                continue
            run_root = Path(run_root)
            if not run_root.is_absolute():
                run_root = PROJECT_ROOT / run_root
            comparison = run_root / "comparison_summary.csv"
            if not comparison.exists():
                continue
            metrics = pd.read_csv(comparison)
            metrics["valid_mcc"] = pd.to_numeric(metrics["valid_mcc"], errors="coerce")
            metrics = metrics.dropna(subset=["valid_mcc"])
            if metrics.empty:
                continue
            # The objective is the best variant within this trial. Match it
            # numerically; fall back to the highest validation MCC if needed.
            delta = (metrics["valid_mcc"] - float(trial.value)).abs()
            row = metrics.loc[delta.idxmin() if delta.min() < 1e-6 else metrics["valid_mcc"].idxmax()]
            candidate = {
                "fraction": fraction,
                "n_calibration": int(value(row, "n_calibration") or N_CALIBRATION[fraction]),
                "model_family": "CNN/MLP",
                "model_head": str(value(row, "variant", "model_name") or "cnn_mlp"),
                "valid_mcc": float(row["valid_mcc"]),
                "test_mcc": value(row, "test_mcc"),
                "test_accuracy": value(row, "test_acc", "test_accuracy"),
                "uuid": run_root.name,
                "n_aug": trial.params.get("n_aug", np.nan),
                "source_kind": "cnn_mlp_fresh_recovered_from_optuna_db",
                "source": f"{database}; {comparison}",
                "selection_note": (
                    f"fallback recovered from Optuna best trial {trial.number}; "
                    "newer FRESH_OPTUNA CSV missing"
                ),
            }
            if best_result is None or candidate["valid_mcc"] > best_result["valid_mcc"]:
                best_result = candidate
    return best_result


def best_original_siamese_candidate(fraction: str) -> dict | None:
    frames = []
    for path, source_kind in siamese_sources(fraction):
        frame = clean_metrics(path, "siamese")
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_source_kind"] = source_kind
        frame["_source"] = str(path)
        frames.append(frame)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    if "uuid" in merged:
        merged = merged.sort_values("valid_mcc", ascending=False).drop_duplicates("uuid", keep="first")
    row = merged.loc[merged["valid_mcc"].idxmax()]
    head = value(row, "siamese_inference", "variant")
    if pd.isna(head):
        head = "original head"
    return {
        "fraction": fraction,
        "n_calibration": int(value(row, "n_calibration") or N_CALIBRATION[fraction]),
        "model_family": "Siamese",
        "model_head": str(head),
        "valid_mcc": float(row["valid_mcc"]),
        "test_mcc": value(row, "test_mcc"),
        "test_accuracy": value(row, "test_accuracy", "test_acc"),
        "uuid": value(row, "uuid"),
        "n_aug": value(row, "n_aug"),
        "source_kind": str(row["_source_kind"]),
        "source": str(row["_source"]),
        "selection_note": "best original Siamese run across both CSV families",
    }


def load_batch_retrained() -> list[dict]:
    path = PROJECT_ROOT / "siamese_heads_real_comparison.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df["new_valid_mcc"] = pd.to_numeric(df.get("new_valid_mcc"), errors="coerce")
    df = df.dropna(subset=["fraction", "new_valid_mcc"])
    rows = []
    for fraction, group in df.groupby("fraction"):
        row = group.loc[group["new_valid_mcc"].idxmax()]
        rows.append(
            {
                "fraction": str(fraction),
                "n_calibration": N_CALIBRATION.get(str(fraction), np.nan),
                "model_family": "Siamese",
                "model_head": str(row.get("head", "retrained head")),
                "valid_mcc": row["new_valid_mcc"],
                "test_mcc": row.get("new_test_mcc", np.nan),
                "test_accuracy": row.get("new_test_acc", np.nan),
                "uuid": row.get("uuid", np.nan),
                "n_aug": row.get("n_aug", 0),
                "source_kind": "siamese_retrained_fixed_grid",
                "source": str(path),
                "selection_note": "best fixed-grid retrained head by validation MCC",
            }
        )
    return rows


def load_optuna_retrained() -> list[dict]:
    path = PROJECT_ROOT / "paper_outputs" / "siamese_head_optuna" / "best_heads.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df["valid_mcc"] = pd.to_numeric(df.get("valid_mcc"), errors="coerce")
    df = df.dropna(subset=["fraction", "valid_mcc"])
    rows = []
    for fraction, group in df.groupby("fraction"):
        row = group.loc[group["valid_mcc"].idxmax()]
        rows.append(
            {
                "fraction": str(fraction),
                "n_calibration": N_CALIBRATION.get(str(fraction), np.nan),
                "model_family": "Siamese",
                "model_head": str(row.get("head", "Optuna head")),
                "valid_mcc": row["valid_mcc"],
                "test_mcc": row.get("test_mcc", np.nan),
                "test_accuracy": row.get("test_accuracy", np.nan),
                "uuid": row.get("uuid", np.nan),
                "n_aug": row.get("n_aug", np.nan),
                "source_kind": "siamese_retrained_optuna",
                "source": str(path),
                "selection_note": "best Optuna-retrained head by validation MCC",
            }
        )
    return rows


def load_supplemental(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"fraction", "model_family", "model_head", "valid_mcc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Supplemental results missing columns: {sorted(missing)}")
    rows = []
    for _, raw in df.iterrows():
        fraction = str(raw["fraction"])
        rows.append(
            {
                "fraction": fraction,
                "n_calibration": raw.get("n_calibration", N_CALIBRATION.get(fraction, np.nan)),
                "model_family": raw["model_family"],
                "model_head": raw["model_head"],
                "valid_mcc": raw["valid_mcc"],
                "test_mcc": raw.get("test_mcc", np.nan),
                "test_accuracy": raw.get("test_accuracy", np.nan),
                "uuid": raw.get("uuid", np.nan),
                "n_aug": raw.get("n_aug", np.nan),
                "source_kind": raw.get("source_kind", "supplemental"),
                "source": raw.get("source", str(path)),
                "selection_note": raw.get("selection_note", "supplemental result"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", nargs="+", default=FRACTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all-output", type=Path, default=DEFAULT_ALL_OUTPUT)
    parser.add_argument("--supplemental-results", type=Path, default=DEFAULT_SUPPLEMENTAL)
    args = parser.parse_args()

    candidates: list[dict] = []
    for fraction in args.fractions:
        print(f"\nFRACTION {fraction}")
        cnn = best_cnn_candidate(fraction)
        original = best_original_siamese_candidate(fraction)
        for candidate in [cnn, original]:
            if candidate:
                candidates.append(candidate)
                print(
                    f"  FOUND {candidate['source_kind']:<32} "
                    f"valid_mcc={candidate['valid_mcc']:.6f} {candidate['model_family']}/{candidate['model_head']}"
                )
            else:
                print("  MISSING candidate source")

    candidates.extend(load_batch_retrained())
    candidates.extend(load_optuna_retrained())
    candidates.extend(load_supplemental(args.supplemental_results))

    all_df = pd.DataFrame(candidates)
    if all_df.empty:
        print("No results found", file=sys.stderr)
        return 1
    all_df["valid_mcc"] = pd.to_numeric(all_df["valid_mcc"], errors="coerce")
    all_df = all_df[all_df["fraction"].isin(args.fractions)].dropna(subset=["valid_mcc"])
    order = {fraction: index for index, fraction in enumerate(args.fractions)}
    all_df["_order"] = all_df["fraction"].map(order)
    all_df = all_df.sort_values(["_order", "valid_mcc"], ascending=[True, False]).drop(columns="_order")

    wide_rows = []
    for fraction in args.fractions:
        fraction_candidates = all_df[all_df["fraction"] == fraction]
        row = {"fraction": fraction, "n_calibration": N_CALIBRATION[fraction]}
        for family, prefix in [("Siamese", "siamese"), ("CNN/MLP", "cnn_mlp")]:
            family_candidates = fraction_candidates[fraction_candidates["model_family"] == family]
            if family_candidates.empty:
                row.update(
                    {
                        f"{prefix}_head": np.nan,
                        f"{prefix}_valid_mcc": np.nan,
                        f"{prefix}_test_mcc": np.nan,
                        f"{prefix}_test_accuracy": np.nan,
                        f"{prefix}_uuid": np.nan,
                        f"{prefix}_n_aug": np.nan,
                        f"{prefix}_source_kind": "missing_not_ready",
                        f"{prefix}_source": np.nan,
                    }
                )
                continue
            selected = family_candidates.loc[family_candidates["valid_mcc"].idxmax()]
            row.update(
                {
                    f"{prefix}_head": selected["model_head"],
                    f"{prefix}_valid_mcc": selected["valid_mcc"],
                    f"{prefix}_test_mcc": selected["test_mcc"],
                    f"{prefix}_test_accuracy": selected["test_accuracy"],
                    f"{prefix}_uuid": selected["uuid"],
                    f"{prefix}_n_aug": selected["n_aug"],
                    f"{prefix}_source_kind": selected["source_kind"],
                    f"{prefix}_source": selected["source"],
                }
            )
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)

    args.all_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.all_output, index=False)
    wide.to_csv(args.output, index=False)

    display = [
        "fraction",
        "n_calibration",
        "siamese_head",
        "siamese_valid_mcc",
        "siamese_test_mcc",
        "cnn_mlp_head",
        "cnn_mlp_valid_mcc",
        "cnn_mlp_test_mcc",
    ]
    print("\nSIX ROWS — BEST SIAMESE AND CNN/MLP SELECTED SEPARATELY BY VALIDATION MCC")
    print(wide[display].to_string(index=False))
    print(f"\nAll candidates: {args.all_output}")
    print(f"Wide six rows:  {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
