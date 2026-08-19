#!/usr/bin/env python3
"""Build complete fresh inference-fraction tables.

For CNN/MLP, prefer the newer INF_FRAC_FRESH_OPTUNA_CNN_MLP file for a
fraction and fall back to INF_FRAC_FRESH_CNN_MLP when the newer run is not
ready.  For Siamese, merge INF_FRAC_FRESH_OPTUNA_SIAMESE and
INF_FRAC_FRESH_SIAMESE, then add retrained-head results from both batch
exports and the app's durable, split-aware learned-head caches.  Cached heads
on CE/dist=none models are reported separately as CNN/new-head results.

Three result files are written:
  * all candidates (audit/provenance table)
  * compact wide table with one row per fraction
  * detailed selected-model table with one row per model family/fraction

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
N_CALIBRATION = {"0p5": 122, "0p25": 61, "0p1": 24, "0p05": 12, "0p02": 5, "0p0": 0}

DEFAULT_OUTPUT = PROJECT_ROOT / "inference_fraction_fresh_wide_6_rows.csv"
DEFAULT_ALL_OUTPUT = PROJECT_ROOT / "inference_fraction_fresh_all_candidates.csv"
DEFAULT_DETAILED_OUTPUT = PROJECT_ROOT / "inference_fraction_fresh_detailed_models.csv"
DEFAULT_SUPPLEMENTAL = None


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
        preferred_zero = first_match(
            [
                str(directory / "INF_FRAC_FRESH_OPTUNA_CNN_MLP_P0_S42*completed_runs_metrics.csv"),
                str(PROGRESS_ROOT / "**" / "INF_FRAC_FRESH_OPTUNA_CNN_MLP_P0_S42*completed_runs_metrics.csv"),
            ]
        )
        if preferred_zero:
            return preferred_zero, "cnn_mlp_fresh_optuna_zero"
        fresh_zero = first_match(
            [
                str(directory / "INF_FRAC_FRESH_CNN_MLP_P0_S42*completed_runs_metrics.csv"),
                str(PROGRESS_ROOT / "**" / "INF_FRAC_FRESH_CNN_MLP_P0_S42*completed_runs_metrics.csv"),
            ]
        )
        return fresh_zero, "cnn_mlp_fresh_zero" if fresh_zero else "cnn_mlp_missing"
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


def clean_metrics(
    path: Path,
    expected_kind: str | None = None,
    require_expected_kind: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "status" in df and df["status"].notna().any():
        completed = df["status"].astype(str).str.lower().eq("completed")
        if completed.any():
            df = df[completed]
    if expected_kind and "kind" in df and df["kind"].notna().any():
        matches = df["kind"].astype(str).str.lower().eq(expected_kind)
        if matches.any():
            df = df[matches]
        elif require_expected_kind:
            return df.iloc[0:0]
    elif expected_kind and require_expected_kind:
        return df.iloc[0:0]
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
    variant = value(row, "variant")
    if pd.isna(variant):
        variant = "cnn_mlp"
    distance_function = value(row, "dist_fct")
    if pd.isna(distance_function):
        distance_function = "none"
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
        "backbone": str(value(row, "model_name", "model") or architecture),
        "variant": variant,
        "classif_loss": value(row, "classif_loss", "loss"),
        "dloss": value(row, "dloss"),
        "distance_function": distance_function,
        "fgsm": value(row, "fgsm"),
        "normalize": value(row, "normalize"),
        "prototype": value(row, "prototype", "prototypes"),
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
                "backbone": str(value(row, "model_name") or trial.params.get("model_name", "unknown")),
                "variant": value(row, "variant"),
                "classif_loss": value(row, "classif_loss") if pd.notna(value(row, "classif_loss")) else trial.params.get("classif_loss", np.nan),
                "dloss": value(row, "dloss") if pd.notna(value(row, "dloss")) else trial.params.get("dloss", np.nan),
                "distance_function": trial.params.get("dist_fct", "none"),
                "fgsm": value(row, "fgsm") if pd.notna(value(row, "fgsm")) else trial.params.get("fgsm", np.nan),
                "normalize": trial.params.get("normalize", np.nan),
                "prototype": trial.params.get("prototype", np.nan),
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
        frame = clean_metrics(
            path,
            "siamese",
            require_expected_kind=(fraction == "0p0"),
        )
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
        "backbone": value(row, "model_name", "model"),
        "variant": value(row, "variant"),
        "classif_loss": value(row, "classif_loss", "loss"),
        "dloss": value(row, "dloss"),
        "distance_function": value(row, "dist_fct"),
        "fgsm": value(row, "fgsm"),
        "normalize": value(row, "normalize"),
        "prototype": value(row, "prototype", "prototypes"),
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


def load_app_cached_heads(fractions: Iterable[str]) -> list[dict]:
    """Load learned heads using the same durable-cache lookup as the app.

    The CSV supplemental export is intentionally not treated as authoritative:
    it can omit fractions even when their split-aware
    ``knn_optimization_cache.pkl`` files exist.  Registry rows provide the
    exact model/dataset identity, while ``best_cached_head_metrics_for_model_row``
    resolves the corresponding cache and selects its best validation MCC.
    """
    source_root = PROJECT_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    try:
        # App modules import Streamlit, whose bare-mode context warnings are
        # irrelevant for this CLI and otherwise overwhelm the actual table.
        import logging

        logging.getLogger("streamlit").setLevel(logging.ERROR)
        import mysql.connector

        from otitenet.app.database import _db_config
        from otitenet.app.pages.leaderboard import (
            _models_dataframe_from_rows,
            _query_best_models,
        )
        from otitenet.app.utils import best_cached_head_metrics_for_model_row
    except Exception as exc:
        print(f"  WARN cannot load app learned-head caches: {exc}")
        return []

    try:
        connection = mysql.connector.connect(**_db_config())
        cursor = connection.cursor(buffered=True)
        registry_rows, use_db_rank = _query_best_models(cursor)
        models = _models_dataframe_from_rows(registry_rows, use_db_rank)
    except Exception as exc:
        print(f"  WARN cannot query app model registry: {exc}")
        return []
    finally:
        try:
            cursor.close()
            connection.close()
        except Exception:
            pass

    results: list[dict] = []
    path_columns = [
        column
        for column in ["Dataset", "Log Path", "Artifact Log Path", "Best Model Dir", "Source Run Path"]
        if column in models
    ]

    for fraction in fractions:
        directory_fraction = "0" if fraction == "0p0" else fraction
        dataset = (
            "otite_ds_64/USA_Turquie_Chili_GMFUNL_"
            f"inference_fraction_hist_v2_train{directory_fraction}_seed42"
        )
        matches = pd.Series(False, index=models.index)
        for column in path_columns:
            matches |= models[column].astype(str).str.contains(dataset, regex=False, na=False)
        fraction_models = models[matches].copy()

        if "N_Calibration" in fraction_models:
            n_calibration = pd.to_numeric(fraction_models["N_Calibration"], errors="coerce")
            fraction_models = fraction_models[n_calibration.eq(N_CALIBRATION[fraction])]

        for _, model_row in fraction_models.iterrows():
            cached = best_cached_head_metrics_for_model_row(model_row.to_dict())
            if not cached:
                continue
            valid_mcc = pd.to_numeric(pd.Series([cached.get("Valid MCC")]), errors="coerce").iloc[0]
            if pd.isna(valid_mcc):
                continue

            classif_loss = str(model_row.get("Classif_Loss", "") or "").strip().lower()
            dist_fct = str(model_row.get("Dist_Fct", "") or "").strip().lower()
            is_cnn = classif_loss == "ce" and dist_fct == "none"
            family = "CNN/new head" if is_cnn else "Siamese"
            model_name = str(model_row.get("Model Name", "") or "unknown")
            head_config = cached.get("Head Config") or cached.get("Config")
            head_name = cached.get("Head") or head_config or "cached head"
            model_dir = model_row.get("Best Model Dir") or model_row.get("Log Path")

            results.append(
                {
                    "fraction": fraction,
                    "n_calibration": N_CALIBRATION[fraction],
                    "model_family": family,
                    "model_head": str(head_name),
                    "valid_mcc": float(valid_mcc),
                    "test_mcc": cached.get("Test MCC", np.nan),
                    "test_accuracy": np.nan,
                    "uuid": model_row.get("Artifact ID", np.nan),
                    "n_aug": cached.get("Head N Aug", np.nan),
                    "backbone": model_name,
                    "variant": "cached learned head",
                    "classif_loss": classif_loss,
                    "dloss": model_row.get("DLoss", np.nan),
                    "distance_function": dist_fct,
                    "fgsm": model_row.get("FGSM", np.nan),
                    "normalize": model_row.get("Normalize", np.nan),
                    "prototype": model_row.get("Prototypes", np.nan),
                    "source_kind": (
                        "cnn_app_cached_head" if is_cnn else "siamese_app_cached_head"
                    ),
                    "source": f"app registry + learned-head cache: {model_dir}",
                    "selection_note": (
                        f"app cache; model={model_name}; classif_loss={classif_loss}; "
                        f"dist_fct={dist_fct}; config={head_config}"
                    ),
                }
            )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", nargs="+", default=FRACTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all-output", type=Path, default=DEFAULT_ALL_OUTPUT)
    parser.add_argument("--detailed-output", type=Path, default=DEFAULT_DETAILED_OUTPUT)
    parser.add_argument(
        "--supplemental-results",
        type=Path,
        default=DEFAULT_SUPPLEMENTAL,
        help="Optional extra candidate CSV; app caches are loaded dynamically by default",
    )
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
    app_cached = load_app_cached_heads(args.fractions)
    candidates.extend(app_cached)
    print(f"\nFOUND {len(app_cached)} app learned-head cache candidates")

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
    detailed_rows = []
    family_specs = [
        ("Siamese", "siamese"),
        ("CNN/MLP", "cnn_mlp"),
        ("CNN/new head", "cnn_head"),
    ]
    for fraction in args.fractions:
        fraction_candidates = all_df[all_df["fraction"] == fraction]
        row = {"fraction": fraction, "n_calibration": N_CALIBRATION[fraction]}
        for family, prefix in family_specs:
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
            detailed_rows.append(
                {
                    "fraction": fraction,
                    "n_calibration": N_CALIBRATION[fraction],
                    "model_family": family,
                    "backbone": selected.get("backbone", np.nan),
                    "variant": selected.get("variant", np.nan),
                    "head": selected.get("model_head", np.nan),
                    "classif_loss": selected.get("classif_loss", np.nan),
                    "dloss": selected.get("dloss", np.nan),
                    "distance_function": selected.get("distance_function", np.nan),
                    "prototype": selected.get("prototype", np.nan),
                    "fgsm": selected.get("fgsm", np.nan),
                    "normalize": selected.get("normalize", np.nan),
                    "n_aug": selected.get("n_aug", np.nan),
                    "valid_mcc": selected.get("valid_mcc", np.nan),
                    "test_mcc": selected.get("test_mcc", np.nan),
                    "test_accuracy": selected.get("test_accuracy", np.nan),
                    "artifact_or_run_id": selected.get("uuid", np.nan),
                    "source_kind": selected.get("source_kind", np.nan),
                    "source": selected.get("source", np.nan),
                    "selection_note": selected.get("selection_note", np.nan),
                }
            )
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
    detailed = pd.DataFrame(detailed_rows)

    args.all_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.detailed_output.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.all_output, index=False)
    wide.to_csv(args.output, index=False)
    detailed.to_csv(args.detailed_output, index=False)

    display = [
        "fraction",
        "n_calibration",
        "siamese_head",
        "siamese_valid_mcc",
        "siamese_test_mcc",
        "cnn_mlp_head",
        "cnn_mlp_valid_mcc",
        "cnn_mlp_test_mcc",
        "cnn_head_head",
        "cnn_head_valid_mcc",
        "cnn_head_test_mcc",
    ]
    print("\nSIX ROWS — BEST SIAMESE, CNN/MLP, AND CNN/NEW HEAD SELECTED SEPARATELY BY VALIDATION MCC")
    print(wide[display].to_string(index=False))
    detailed_display = [
        "fraction", "model_family", "backbone", "variant", "head",
        "classif_loss", "dloss", "distance_function", "n_aug",
        "valid_mcc", "test_mcc",
    ]
    print("\nDETAILED SELECTED MODELS — ONE MODEL FAMILY PER LINE")
    print(detailed[detailed_display].to_string(index=False))
    print(f"\nAll candidates: {args.all_output}")
    print(f"Wide six rows:  {args.output}")
    print(f"Detailed models: {args.detailed_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
