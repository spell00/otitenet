#!/usr/bin/env python3
"""Run fresh Siamese and CNN/MLP Optuna studies for every CV/fraction scenario."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".conda/bin/python"
TASK = "four_classes_220726"
EXPECTED_LABELS = ["0p5", "0p25", "0p1", "0p05", "0p02", "0"]
EXPECTED_CALIBRATION = [122, 61, 26, 13, 5, 0]


def quoted(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def append_catalog(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def common(scenario: pd.Series, args: argparse.Namespace, tag: str) -> list[str]:
    dataset_path = Path(str(scenario.dataset_path))
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    return [
        "--path", str(dataset_path),
        "--task", TASK,
        "--train_datasets", "from_infos_csv",
        "--valid_dataset", "from_infos_csv",
        "--test_dataset", "from_infos_csv",
        "--groupkfold", "1",
        "--n_calibration", str(int(scenario.inference_train)),
        "--seed", str(args.seed),
        "--n_trials", str(args.n_trials),
        "--n_epochs", str(args.n_epochs),
        "--early_stop", str(args.early_stop),
        "--num_workers", str(args.num_workers),
        "--run_tag", tag,
        "--reset_opt_state", "1",
    ]


def siamese_command(scenario: pd.Series, args: argparse.Namespace) -> tuple[str, list[str]]:
    tag = (
        f"INF_FRAC_CV_FRESH_SIAMESE_CV{int(scenario.cv_run)}_"
        f"P{scenario.scenario_label}_S{args.seed}"
    )
    command = [
        str(PYTHON), "-m", "otitenet.train.train_triplet_new",
        *common(scenario, args, tag),
        "--exp_id", "inference_fraction_cv_fresh_siamese",
        "--calibration_preassigned_train", "1",
        "--kind", "siamese",
        "--model_name", "resnet18",
        "--new_size", "64",
        "--bs", str(args.siamese_batch_size),
        "--prototypes_to_use", "None",
        "--normalize", "None",
        "--siamese_inference", "linearsvc",
        "--log_dvclive", "1",
        "--dvclive_save_dvc_exp", "1",
        "--dvclive_monitor_system", "0",
        "--log_comet", "0",
        "--log_mlflow", "0",
        "--heavy_best_analysis", "0",
        "--run_explainability", "0",
        "--epoch_progress", "1",
        "--amp", "1",
        "--amp_dtype", "bf16",
    ]
    return tag, command


def cnn_command(scenario: pd.Series, args: argparse.Namespace) -> tuple[str, list[str]]:
    tag = (
        f"INF_FRAC_CV_FRESH_CNN_MLP_CV{int(scenario.cv_run)}_"
        f"P{scenario.scenario_label}_S{args.seed}"
    )
    command = [
        str(PYTHON), "-m", "otitenet.train.train_cnn_mlp_compare",
        *common(scenario, args, tag),
        "--exp_id", "inference_fraction_cv_fresh_cnn_mlp",
        "--new_size", "64",
        "--bs", str(args.cnn_batch_size),
        "--compare_all", "1",
        "--amp", "1",
        "--amp_dtype", "bf16",
        "--log_mlflow", "0",
        "--verbose", "1",
    ]
    return tag, command


def validate_scenarios(scenarios: pd.DataFrame, n_splits: int) -> None:
    required = {
        "cv_run", "valid_fold", "test_fold", "scenario_label",
        "scenario_fraction", "inference_train", "dataset_path",
    }
    missing = required.difference(scenarios.columns)
    if missing:
        raise ValueError(f"Scenario manifest is missing columns: {sorted(missing)}")
    if len(scenarios) != n_splits * len(EXPECTED_LABELS):
        raise ValueError(
            f"Expected {n_splits * len(EXPECTED_LABELS)} scenarios, got {len(scenarios)}"
        )
    for fold in range(1, n_splits + 1):
        subset = scenarios[scenarios["cv_run"].eq(fold)]
        if subset["scenario_label"].astype(str).tolist() != EXPECTED_LABELS:
            raise ValueError(f"Unexpected fraction order for CV run {fold}")
        if subset["inference_train"].astype(int).tolist() != EXPECTED_CALIBRATION:
            raise ValueError(f"Unexpected calibration counts for CV run {fold}")
        expected_test = fold % n_splits + 1
        if not subset["valid_fold"].eq(fold).all() or not subset["test_fold"].eq(expected_test).all():
            raise ValueError(f"Invalid validation/test rotation for CV run {fold}")
        for path in subset["dataset_path"]:
            dataset_path = Path(str(path))
            if not dataset_path.is_absolute():
                dataset_path = ROOT / dataset_path
            if not (dataset_path / "infos.csv").exists():
                raise FileNotFoundError(dataset_path / "infos.csv")


def run_one(
    phase: str,
    scenario: pd.Series,
    command: list[str],
    tag: str,
    output: Path,
    catalog: Path,
    env: dict[str, str],
    dry_run: bool,
) -> int:
    key = f"{phase}_cv{int(scenario.cv_run)}_{scenario.scenario_label}"
    log = output / "logs" / f"{key}.log"
    base = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "cv_run": int(scenario.cv_run),
        "valid_fold": int(scenario.valid_fold),
        "test_fold": int(scenario.test_fold),
        "scenario_fraction": float(scenario.scenario_fraction),
        "scenario_label": str(scenario.scenario_label),
        "n_calibration": int(scenario.inference_train),
        "run_tag": tag,
        "status": "planned",
        "log": str(log.relative_to(output)),
        "command": quoted(command),
    }
    append_catalog(catalog, base)
    print(f"\n=== {key}: {tag} ===\n$ {quoted(command)}", flush=True)
    if dry_run:
        return 0
    with log.open("w") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    append_catalog(
        catalog,
        {
            **base,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed" if return_code == 0 else f"failed:{return_code}",
        },
    )
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "paper_outputs/inference_fraction_cv_fresh_optuna",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--siamese-batch-size", type=int, default=64)
    parser.add_argument("--cnn-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = pd.read_csv(args.scenario_manifest, dtype={"scenario_label": str})
    validate_scenarios(scenarios, args.n_splits)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / f"run_{stamp}"
    (output / "logs").mkdir(parents=True, exist_ok=False)
    (output / "scenario_manifest.csv").write_bytes(args.scenario_manifest.read_bytes())
    total_trials = len(scenarios) * args.n_trials * 2
    (output / "run_context.json").write_text(
        json.dumps(
            {
                "created_utc": stamp,
                "task": TASK,
                "trial_source": "fresh Optuna; no previous-best parameters",
                "split_design": "historical always train; rotating inference validation/test folds",
                "phase_order": ["siamese", "cnn_mlp"],
                "n_splits": args.n_splits,
                "fractions": EXPECTED_LABELS,
                "n_calibration": EXPECTED_CALIBRATION,
                "scenarios_per_phase": len(scenarios),
                "n_trials_per_scenario": args.n_trials,
                "total_optuna_trials": total_trials,
            },
            indent=2,
        )
        + "\n"
    )
    catalog = output / "run_catalog.csv"
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    print(f"Experiment root: {output}", flush=True)
    print(f"Total fresh Optuna trials: {total_trials}", flush=True)
    for phase, builder in (("siamese", siamese_command), ("cnn_mlp", cnn_command)):
        print(f"\n=== Phase: {phase} ===", flush=True)
        for _, scenario in scenarios.iterrows():
            tag, command = builder(scenario, args)
            return_code = run_one(
                phase, scenario, command, tag, output, catalog, env, args.dry_run
            )
            if return_code:
                return return_code
    print("All fresh CV fraction studies completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
