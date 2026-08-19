#!/usr/bin/env python3
"""Run E03 support curves on inference-fraction data using historically best fixed model configs."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PYTHON = ROOT / ".conda" / "bin" / "python"
TASK = "four_classes_220726"
DEFAULT_SCENARIO_MANIFEST = "data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v2_seed42_scenarios.csv"
DEFAULT_OUTPUT_DIR = ROOT / "paper_outputs/inference_fraction_e03_prevbest"

CONFIG_COLUMNS = [
    "model_name",
    "fgsm",
    "n_calibration",
    "dist_fct",
    "knn",
    "n_negatives",
    "dloss",
    "classif_loss",
    "normalize",
    "prototypes",
    "n_positives",
]


def _clean(value, default="") -> str:
    text = str(value if value is not None else default).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return str(default)
    return text


def _clean_int(value, default="1") -> str:
    text = _clean(value, default)
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return str(default)


def _quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _cmd_arg(cmd: list[str], flag: str, default: str = "") -> str:
    try:
        return str(cmd[cmd.index(flag) + 1])
    except ValueError:
        return default


def _int_or_default(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _previous_best_configs(top_k: int, source_tasks: list[str]) -> pd.DataFrame:
    path = ROOT / "completed_runs_metrics.csv"
    df = pd.read_csv(path)
    df = df[df["status"].astype(str).str.lower().eq("completed")].copy()
    df = df[df["task"].astype(str).isin(source_tasks)]
    df["valid_mcc_num"] = pd.to_numeric(df["valid_mcc"], errors="coerce")
    df = df.dropna(subset=["valid_mcc_num"])
    for col in CONFIG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df.sort_values("valid_mcc_num", ascending=False)
    df = df.drop_duplicates(subset=CONFIG_COLUMNS, keep="first")
    return df.head(int(top_k)).reset_index(drop=True)


def _build_cmd(args, cfg: pd.Series, scenario: pd.Series, rank: int) -> list[str]:
    version = _clean(getattr(args, "version_label", ""), "")
    version_part = f"_{version}" if version else ""
    n_calib = int(scenario.get("inference_train", 122))
    run_tag = f"INF_FRAC_E03_PREVBEST{version_part}_P{scenario.scenario_label}_R{rank:02d}"
    cmd = [
        str(PYTHON), "-m", "otitenet.train.train_triplet_new",
        "--path", str(ROOT / scenario.get("dataset_path")),
        "--groupkfold", "1",
        "--task", TASK,
        "--train_datasets", "from_infos_csv",
        "--valid_dataset", "from_infos_csv",
        "--test_dataset", "from_infos_csv",
        "--n_calibration", str(n_calib),
        "--seed", str(args.seed),
        "--n_trials", str(args.n_trials),
        "--n_epochs", str(args.n_epochs),
        "--early_stop", str(args.early_stop),
        "--num_workers", str(args.num_workers),
        "--run_tag", run_tag,
        "--exp_id", "inference_fraction_e03_prevbest",
        "--calibration_preassigned_train", "1",
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
        "--model_name", _clean(cfg.get("model_name"), "resnet18"),
        "--new_size", "64",
        "--bs", str(args.batch_size),
        "--fgsm", _clean_int(cfg.get("fgsm"), "0"),
        "--prototypes_to_use", _clean(cfg.get("prototypes"), "no"),
        "--n_positives", _clean_int(cfg.get("n_positives"), "1"),
        "--n_negatives", _clean_int(cfg.get("n_negatives"), "1"),
        "--dloss", _clean(cfg.get("dloss"), "inverseTriplet"),
        "--dist_fct", _clean(cfg.get("dist_fct"), "cosine"),
        "--classif_loss", _clean(cfg.get("classif_loss"), "arcface"),
        "--normalize", _clean(cfg.get("normalize"), "yes"),
        "--siamese_inference", "linearsvc",
    ]
    knn = _clean(cfg.get("knn"), "")
    if knn:
        cmd.extend(["--n_neighbors", str(int(float(knn)))])
    return cmd


def _run(cmd: list[str], dataset_path: str, run_id: str, scenario_label: str, n_calibration: int) -> int:
    print("$", _quote(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    
    # Generate and register sample manifest after training completes
    if rc == 0:
        print(f"[Manifest] Generating sample assignments for {run_id}...", flush=True)
        manifest_cmd = [
            str(ROOT / ".conda" / "bin" / "python"),
            str(ROOT / "scripts" / "paper" / "manifest_integration.py"),
            "--dataset-path", dataset_path,
            "--run-output-dir", str(ROOT / "logs" / TASK / run_id),
            "--run-id", run_id,
            "--scenario-label", scenario_label,
            "--n-calibration", str(n_calibration),
        ]
        try:
            subprocess.run(manifest_cmd, check=True, cwd=ROOT)
        except Exception as e:
            print(f"[Manifest] Warning: {e}", file=sys.stderr)
    
    return rc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E03 best configs on inference-fraction data.")
    parser.add_argument("--top-configs", type=int, default=5)
    parser.add_argument("--source-tasks", default="four_classes_220726")
    parser.add_argument("--scenario-manifest", default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--version-label", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = _previous_best_configs(
        args.top_configs,
        [task.strip() for task in args.source_tasks.split(",") if task.strip()],
    )
    if configs.empty:
        print("No previous best configs found.", file=sys.stderr)
        return 2
    
    print("Selected E03 best configs:", flush=True)
    print(configs[["task", "run_tag", "valid_mcc", *CONFIG_COLUMNS]].to_string(index=False), flush=True)

    scenarios = pd.read_csv(ROOT / args.scenario_manifest)
    print(f"\nScenarios from {args.scenario_manifest}:", flush=True)
    print(scenarios[["scenario_fraction", "scenario_label", "dataset_path", "inference_train"]].to_string(index=False), flush=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    for idx, (_, scenario) in enumerate(scenarios.iterrows()):
        scenario_label = scenario.get("scenario_label", "unknown")
        print(f"\n=== Scenario {scenario_label} ({idx+1}/{len(scenarios)}) ===", flush=True)
        for rank, (_, cfg) in enumerate(configs.iterrows(), 1):
            cmd = _build_cmd(args, cfg, scenario, rank)
            if args.dry_run:
                print("[DRY RUN]", _quote(cmd), flush=True)
            else:
                run_tag = _cmd_arg(cmd, "--run_tag")
                rc = _run(
                    cmd,
                    dataset_path=str(scenario.get("dataset_path")),
                    run_id=run_tag,
                    scenario_label=scenario_label,
                    n_calibration=int(scenario.get("inference_train", 122)),
                )
                if rc != 0:
                    print(f"Warning: run exited with code {rc}", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
