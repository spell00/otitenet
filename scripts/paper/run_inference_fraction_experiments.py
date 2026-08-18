#!/usr/bin/env python3
"""Run traceable inference-only train-fraction experiments sequentially."""
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
from run_prev_best_e03_experiments import CONFIG_COLUMNS, PYTHON, TASK, _clean, _clean_int, _previous_best_configs, _task_run_dirs

DEFAULT_MANIFEST = ROOT / "data/otite_ds_224/USA_Turquie_Chili_GMFUNL_inference_fraction_v1_seed42_scenarios.csv"


def quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def arg(cmd: list[str], flag: str, default: str = "") -> str:
    try: return str(cmd[cmd.index(flag) + 1])
    except ValueError: return default


def build_cmd(cfg: pd.Series, scenario: pd.Series, rank: int, seed: int, args: argparse.Namespace) -> list[str]:
    tag = f"{args.experiment_label.upper()}_R{rank:02d}_P{scenario.scenario_label}_S{seed}"
    cmd = [str(PYTHON), "-m", "otitenet.train.train_triplet_new", "--path", str(ROOT / scenario.dataset_path),
           "--groupkfold", "1", "--task", TASK, "--train_datasets", "from_infos_csv", "--valid_dataset", "from_infos_csv", "--test_dataset", "from_infos_csv",
           "--n_calibration", str(int(scenario.inference_train)), "--calibration_preassigned_train", "1",
           "--seed", str(seed), "--n_trials", str(args.n_trials), "--n_epochs", str(args.n_epochs), "--early_stop", str(args.early_stop),
           "--num_workers", str(args.num_workers), "--run_tag", tag, "--exp_id", args.experiment_label, "--log_dvclive", "1", "--dvclive_save_dvc_exp", "1",
           "--dvclive_monitor_system", "0", "--log_comet", "0", "--log_mlflow", "0", "--heavy_best_analysis", "0", "--run_explainability", "0",
           "--epoch_progress", "1", "--reset_opt_state", "1", "--model_name", _clean(cfg.get("model_name"), "resnet18"), "--new_size", "224",
           "--fgsm", _clean_int(cfg.get("fgsm"), "1"), "--prototypes_to_use", _clean(cfg.get("prototypes"), "no"),
           "--n_positives", _clean_int(cfg.get("n_positives"), "1"), "--n_negatives", _clean_int(cfg.get("n_negatives"), "1"),
           "--dloss", _clean(cfg.get("dloss"), "inverseTriplet"), "--dist_fct", _clean(cfg.get("dist_fct"), "cosine"),
           "--classif_loss", _clean(cfg.get("classif_loss"), "arcface"), "--normalize", _clean(cfg.get("normalize"), "yes"), "--siamese_inference", "linearsvc"]
    knn = _clean(cfg.get("knn"), "")
    if knn: cmd += ["--n_neighbors", str(int(float(knn)))]
    return cmd


def append_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if not exists: w.writeheader()
        w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario-manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--top-configs", type=int, default=8)
    p.add_argument("--selected-configs", type=Path, default=None, help="Optional frozen selected_configs.csv from a prior launch")
    p.add_argument("--source-tasks", default=TASK)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--early-stop", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-root", type=Path, default=ROOT / "paper_outputs/inference_fraction_v1")
    p.add_argument("--experiment-label", default="inference_fraction_v1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    scenarios = pd.read_csv(args.scenario_manifest)
    expected = ["0p5", "0p25", "0p1", "0p05", "0p02"]
    if scenarios.scenario_label.tolist() != expected:
        raise ValueError(f"Scenario order must be {expected}, got {scenarios.scenario_label.tolist()}")
    if (scenarios[["inference_valid", "inference_test"]].nunique() != 1).any() or (scenarios.historical_valid_test != 0).any():
        raise ValueError("Scenario manifest does not have fixed inference-only validation/test groups")
    if args.selected_configs is not None:
        configs = pd.read_csv(args.selected_configs).head(args.top_configs).copy()
    else:
        configs = _previous_best_configs(args.top_configs, [x.strip() for x in args.source_tasks.split(",") if x.strip()])
    if configs.empty: raise RuntimeError("No completed source configurations found")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_root / f"run_{stamp}"
    logs = out / "logs"; logs.mkdir(parents=True, exist_ok=False)
    (out / "scenario_manifest.csv").write_bytes(args.scenario_manifest.read_bytes())
    (out / "selected_configs.csv").write_text(configs.to_csv(index=False))
    context = {"created_utc": stamp, "scenario_manifest": str(args.scenario_manifest), "top_configs": args.top_configs, "seed": args.seed,
               "selected_configs_source": str(args.selected_configs) if args.selected_configs else "computed",
               "scenario_order": expected, "notes": "n_calibration equals inference_train; samples are preassigned to train in infos.csv; fixed inference-only validation/test remain unchanged."}
    (out / "run_context.json").write_text(json.dumps(context, indent=2) + "\n")
    catalog = out / "run_catalog.csv"
    print(f"Experiment root: {out}", flush=True)
    print(configs[["run_tag", "valid_mcc", *CONFIG_COLUMNS]].to_string(index=False), flush=True)
    # Complete the full five-fraction curve for each frozen configuration before
    # moving to the next one. This yields an interpretable result after every
    # 5 * n_trials models instead of waiting for every config at one fraction.
    for rank, (_, cfg) in enumerate(configs.iterrows(), start=1):
        for _, scenario in scenarios.iterrows():
            cmd = build_cmd(cfg, scenario, rank, args.seed, args)
            log = logs / f"{scenario.scenario_label}_rank{rank:02d}.log"
            row = {"scenario_fraction": scenario.scenario_fraction, "scenario_label": scenario.scenario_label, "rank": rank,
                   "source_run_tag": cfg.get("run_tag", ""), "run_tag": arg(cmd, "--run_tag"), "command": quote(cmd), "log": str(log.relative_to(out)), "status": "planned", "uuid": ""}
            append_row(catalog, row)
            print(f"\n=== {row['run_tag']} ===\n$ {row['command']}", flush=True)
            if args.dry_run: continue
            before = _task_run_dirs(TASK)
            with log.open("w") as fh:
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True); fh.write(line); fh.flush()
                rc = process.wait()
            after = _task_run_dirs(TASK)
            uuids = sorted(after - before)
            row.update(status="completed" if rc == 0 else f"failed:{rc}", uuid=uuids[-1] if uuids else "")
            append_row(catalog, row)
            if rc: return rc
    print(f"Completed. Generate figures with: {PYTHON} scripts/paper/plot_inference_fraction_performance.py --experiment-root {out}", flush=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
