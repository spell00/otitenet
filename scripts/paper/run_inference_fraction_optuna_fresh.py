#!/usr/bin/env python3
"""Fresh five-fraction Optuna studies: Siamese first, then CNN/MLP."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".conda/bin/python"
TASK = "four_classes_220818"
EXPECTED = ["0p5", "0p25", "0p1", "0p05", "0p02", "0"]
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
    return [
        "--path", str(ROOT / str(scenario.dataset_path)),
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
    tag = f"INF_FRAC_FRESH_SIAMESE_P{scenario.scenario_label}_S{args.seed}"
    command = [str(PYTHON), "-m", "otitenet.train.train_triplet_new", *common(scenario, args, tag),
        "--exp_id", "inference_fraction_fresh_siamese",
        "--calibration_preassigned_train", "1",
        # The loss is deliberately Optuna-selected, so do not rely on the
        # trainer's loss-based kind heuristic for this Siamese-only phase.
        "--kind", "siamese",
        "--model_name", "resnet18", "--new_size", "64", "--bs", str(args.siamese_batch_size),
        "--prototypes_to_use", "None",
        "--normalize", "None", "--siamese_inference", "linearsvc",
        "--log_dvclive", "1", "--dvclive_save_dvc_exp", "1", "--dvclive_monitor_system", "0",
        "--log_comet", "0", "--log_mlflow", "0", "--heavy_best_analysis", "0",
        "--run_explainability", "0", "--epoch_progress", "1", "--amp", "1", "--amp_dtype", "bf16"]
    return tag, command


def cnn_command(scenario: pd.Series, args: argparse.Namespace) -> tuple[str, list[str]]:
    tag = f"INF_FRAC_FRESH_CNN_MLP_P{scenario.scenario_label}_S{args.seed}"
    command = [str(PYTHON), "-m", "otitenet.train.train_cnn_mlp_compare", *common(scenario, args, tag),
        "--exp_id", "inference_fraction_fresh_cnn_mlp",
        "--new_size", "64", "--bs", str(args.cnn_batch_size),
        "--compare_all", "1", "--amp", "1", "--amp_dtype", "bf16",
        "--log_mlflow", "0", "--verbose", "1"]
    return tag, command


def run_one(phase: str, scenario: pd.Series, command: list[str], tag: str,
            output: Path, catalog: Path, env: dict[str, str], dry_run: bool) -> int:
    log = output / "logs" / f"{phase}_{scenario.scenario_label}.log"
    base = {"timestamp": datetime.now(timezone.utc).isoformat(), "phase": phase,
            "scenario_fraction": scenario.scenario_fraction, "scenario_label": scenario.scenario_label,
            "n_calibration": int(scenario.inference_train), "run_tag": tag,
            "status": "planned", "log": str(log.relative_to(output)), "command": quoted(command)}
    append_catalog(catalog, base)
    print(f"\n=== {phase}: {tag} ===\n$ {quoted(command)}", flush=True)
    if dry_run:
        return 0
    with log.open("w") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    append_catalog(catalog, {**base, "timestamp": datetime.now(timezone.utc).isoformat(),
                              "status": "completed" if return_code == 0 else f"failed:{return_code}"})
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "paper_outputs/inference_fraction_fresh_optuna")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--siamese-batch-size", type=int, default=64)
    parser.add_argument("--cnn-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = pd.read_csv(args.scenario_manifest, dtype={"scenario_label": str})
    if scenarios.scenario_label.tolist() != EXPECTED:
        raise ValueError(f"Expected fraction order {EXPECTED}, got {scenarios.scenario_label.tolist()}")
    calibration_counts = scenarios.inference_train.astype(int).tolist()
    if calibration_counts != EXPECTED_CALIBRATION:
        raise ValueError(
            f"Expected n_calibration order {EXPECTED_CALIBRATION}, got {calibration_counts}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / f"run_{stamp}"
    (output / "logs").mkdir(parents=True, exist_ok=False)
    (output / "scenario_manifest.csv").write_bytes(args.scenario_manifest.read_bytes())
    (output / "run_context.json").write_text(json.dumps({
        "created_utc": stamp, "task": TASK, "trial_source": "fresh Optuna; no previous-best parameters",
        "phase_order": ["siamese", "cnn_mlp"], "fractions": scenarios.scenario_fraction.tolist(),
        "n_calibration": scenarios.inference_train.astype(int).tolist(), "n_trials_per_fraction_per_phase": args.n_trials,
        "total_optuna_trials": len(scenarios) * args.n_trials * 2,
        "resources": {"gpu_jobs": 1, "num_workers": args.num_workers,
                      "siamese_batch_size": args.siamese_batch_size, "cnn_batch_size": args.cnn_batch_size},
    }, indent=2) + "\n")
    catalog = output / "run_catalog.csv"
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    print(f"Experiment root: {output}", flush=True)
    for phase, builder in (("siamese", siamese_command), ("cnn_mlp", cnn_command)):
        print(f"\n=== Phase: {phase} ===", flush=True)
        for _, scenario in scenarios.iterrows():
            tag, command = builder(scenario, args)
            return_code = run_one(phase, scenario, command, tag, output, catalog, env, args.dry_run)
            if return_code:
                return return_code
    print("All fresh fraction studies completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
