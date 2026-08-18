#!/usr/bin/env python3
"""Run E03v2 Pareto experiments over new-sample train fractions.

Assumes scenario datasets were prepared with
prepare_exp3v2_scenario_datasets.py. Each scenario dataset has explicit
infos.csv groups, so train/valid/test are passed as from_infos_csv and the four
non-inference datasets stay in train.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_prev_best_e03_experiments import (  # noqa: E402
    CONFIG_COLUMNS,
    MANIFEST_COLUMNS,
    PYTHON,
    TASK,
    _clean,
    _clean_int,
    _dataset_key_from_path,
    _previous_best_configs,
    _task_run_dirs,
)

DEFAULT_SCENARIO_MANIFEST = ROOT / "data/otite_ds_224/USA_Turquie_Chili_GMFUNL_inference_exp3v2_seed42_scenarios.csv"


def _quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _cmd_arg(cmd: list[str], flag: str, default: str = "") -> str:
    try:
        return str(cmd[cmd.index(flag) + 1])
    except ValueError:
        return default


def _register_run(cmd: list[str], run_uuid: str, scenario_label: str) -> None:
    task = _cmd_arg(cmd, "--task", TASK)
    dataset_path = _cmd_arg(cmd, "--path")
    dataset_key = _dataset_key_from_path(dataset_path) if dataset_path else ""
    manifest_dir = ROOT / "logs" / "progresses" / task / "exp3v2" / "csv"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"PROD_{task}_job_manifest.csv"
    row = {
        "job_id": _cmd_arg(cmd, "--run_tag"),
        "job_state": "done",
        "model": _cmd_arg(cmd, "--model_name", "resnet18"),
        "uuid": run_uuid,
        "task": task,
        "dataset_name": dataset_key,
        "dataset_key": dataset_key,
        "train_datasets": _cmd_arg(cmd, "--train_datasets"),
        "valid_dataset": _cmd_arg(cmd, "--valid_dataset"),
        "test_dataset": _cmd_arg(cmd, "--test_dataset"),
        "n_calibration": _cmd_arg(cmd, "--n_calibration"),
        "classif_loss": _cmd_arg(cmd, "--classif_loss"),
        "dloss": _cmd_arg(cmd, "--dloss"),
        "dist_fct": _cmd_arg(cmd, "--dist_fct"),
        "new_size": _cmd_arg(cmd, "--new_size"),
        "fgsm": _cmd_arg(cmd, "--fgsm"),
        "prototype": _cmd_arg(cmd, "--prototypes_to_use"),
        "prototypes": _cmd_arg(cmd, "--prototypes_to_use"),
        "n_positives": _cmd_arg(cmd, "--n_positives"),
        "n_negatives": _cmd_arg(cmd, "--n_negatives"),
        "normalize": _cmd_arg(cmd, "--normalize"),
        "knn": _cmd_arg(cmd, "--n_neighbors"),
        "n_neighbors": _cmd_arg(cmd, "--n_neighbors"),
        "prototype_strategy": "",
        "prototype_components": "",
        "exp_id": _cmd_arg(cmd, "--run_tag"),
    }
    # Add scenario metadata as extra columns at the end; app readers ignore unknown columns.
    columns = list(MANIFEST_COLUMNS) + ["scenario_label"]
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        out = {k: row.get(k, "") for k in columns}
        out["scenario_label"] = scenario_label
        writer.writerow(out)
    print(f"Registered run {run_uuid} in {manifest_path.relative_to(ROOT)}", flush=True)


def _build_cmd(args: argparse.Namespace, cfg: pd.Series, scenario: pd.Series, rank: int, seed: int) -> list[str]:
    version = _clean(getattr(args, "version_label", ""), "")
    if not version:
        version = "NEWONLY" if "newonly" in str(args.scenario_manifest).lower() else "HIST"
    scenario_label = str(scenario["scenario_label"])
    run_tag = f"PAPER_E03V2_{version}_R{rank:02d}_P{scenario_label}_S{seed}"
    dataset_path = str(ROOT / str(scenario["dataset_path"]))
    cmd = [
        str(PYTHON), "-m", "otitenet.train.train_triplet_new",
        "--path", dataset_path,
        "--groupkfold", "1",
        "--task", TASK,
        "--train_datasets", "from_infos_csv",
        "--valid_dataset", "from_infos_csv",
        "--test_dataset", "from_infos_csv",
        "--n_calibration", "0",
        "--seed", str(seed),
        "--n_trials", str(args.n_trials),
        "--n_epochs", str(args.n_epochs),
        "--early_stop", str(args.early_stop),
        "--num_workers", str(args.num_workers),
        "--run_tag", run_tag,
        "--exp_id", "paper_exp3v2",
        "--log_dvclive", "1",
        "--dvclive_save_dvc_exp", "1",
        "--dvclive_monitor_system", "1",
        "--log_comet", "0",
        "--log_mlflow", "0",
        "--heavy_best_analysis", "0",
        "--run_explainability", "0",
        "--epoch_progress", "1",
        "--reset_opt_state", "1",
        "--model_name", _clean(cfg.get("model_name"), "resnet18"),
        "--new_size", "224",
        "--fgsm", _clean_int(cfg.get("fgsm"), "1"),
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


def _run(cmd: list[str], scenario_label: str) -> int:
    print("\n$", _quote(cmd), flush=True)
    before = _task_run_dirs(TASK)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc == 0:
        new_dirs = sorted(_task_run_dirs(TASK) - before)
        if new_dirs:
            _register_run(cmd, new_dirs[-1], scenario_label)
        else:
            print(f"Warning: could not detect a new run dir for {_cmd_arg(cmd, --run_tag)}", file=sys.stderr)
    return rc


def _validate_scenarios(scenarios: pd.DataFrame) -> None:
    for _, row in scenarios.iterrows():
        path = ROOT / str(row["dataset_path"])
        infos = path / "infos.csv"
        if not infos.exists():
            raise FileNotFoundError(infos)
        df = pd.read_csv(infos)
        inf = df[df["dataset"].eq("inference")]
        counts = inf.groupby(["group", "label"]).size().to_dict()
        print(f"Scenario {row['scenario_label']} {path.relative_to(ROOT)} inference counts: {counts}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E03v2 Pareto curve across explicit scenario datasets.")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--top-configs", type=int, default=8)
    parser.add_argument("--source-tasks", default="four_classes_220726")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--version-label", default="", help="Optional tag component; defaults to HIST or NEWONLY based on scenario manifest name.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = pd.read_csv(args.scenario_manifest)
    if scenarios.empty:
        print(f"No scenarios found in {args.scenario_manifest}", file=sys.stderr)
        return 2
    _validate_scenarios(scenarios)
    configs = _previous_best_configs(
        args.top_configs,
        [task.strip() for task in args.source_tasks.split(",") if task.strip()],
    )
    if configs.empty:
        print("No previous-best configs found.", file=sys.stderr)
        return 2
    print("Selected previous-best configs:", flush=True)
    print(configs[["task", "run_tag", "valid_mcc", *CONFIG_COLUMNS]].to_string(index=False), flush=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    for seed in seeds:
        for _, scenario in scenarios.iterrows():
            for idx, cfg in configs.iterrows():
                cmd = _build_cmd(args, cfg, scenario, rank=idx + 1, seed=seed)
                if args.dry_run:
                    print(_quote(cmd))
                    continue
                rc = _run(cmd, str(scenario["scenario_label"]))
                if rc != 0:
                    return rc
        if args.dry_run:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
