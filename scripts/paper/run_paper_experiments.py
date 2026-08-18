#!/usr/bin/env python3
"""Numbered experiment runner for the OtiteAI/OtiteNet paper.

The default commands intentionally target the current top four-class model family
from data/production_model_four_class.json. They do not launch a broad grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_test_calibration_manifest import build_manifest  # noqa: E402

PYTHON = ROOT / ".conda" / "bin" / "python"
TASK = "four_classes_220726"
DEFAULT_DATASET_PATH = "./data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference"
DEFAULT_PUBLIC_TRAIN = "Banque_Comert_Turquie_2020_jpg,Banque_Calaman_USA_2020_trie_CM,GMFUNL_jan2023"
DEFAULT_VALID = "Banque_Viscaino_Chili_2020"
DEFAULT_TARGET_TEST = "inference"
DEFAULT_OUTPUT_DIR = ROOT / "paper_outputs"

# Columns read by otitenet.app.args._progress_manifest_model_rows() when it scans
# logs/progresses/<task>/**/PROD_<task>_job_manifest.csv for "done" jobs to populate
# Quick Model Selection - the same discovery path launch.sh-produced jobs use.
JOB_MANIFEST_COLUMNS = [
    "job_id", "job_state", "model", "uuid", "task", "dataset_name", "dataset_key",
    "train_datasets", "valid_dataset", "test_dataset", "n_calibration",
    "classif_loss", "loss", "dloss", "dist_fct", "new_size", "fgsm",
    "prototype", "prototypes", "n_positives", "n_negatives", "normalize",
    "knn", "n_neighbors", "prototype_strategy", "prototype_components", "exp_id",
]


@dataclass(frozen=True)
class Experiment:
    number: str
    name: str
    purpose: str
    command: list[str]
    gpu_time: str
    required_before: str = ""


def _load_top_model() -> dict:
    path = ROOT / "data" / "production_model_four_class.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _clean(value, default="") -> str:
    text = str(value if value is not None else default).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return str(default)
    return text


def _top_model_args(top: dict) -> list[str]:
    """Map production registry fields to train_triplet_new CLI args."""
    return [
        "--model_name", _clean(top.get("Model Name"), "resnet18"),
        "--new_size", _clean(top.get("NSize"), "224"),
        "--fgsm", _clean(top.get("FGSM"), "1"),
        "--prototypes_to_use", _clean(top.get("Prototypes"), "no"),
        "--n_positives", _clean(top.get("NPos"), "1"),
        "--n_negatives", _clean(top.get("NNeg"), "1"),
        "--dloss", _clean(top.get("DLoss"), "inverseTriplet"),
        "--dist_fct", _clean(top.get("Dist_Fct"), "euclidean"),
        "--classif_loss", _clean(top.get("Classif_Loss"), "arcface"),
        "--normalize", _clean(top.get("Normalize"), "yes"),
        "--siamese_inference", "linearsvc",
    ]


def _train_base(
    args: argparse.Namespace,
    *,
    n_calibration: int,
    run_tag: str,
    seed: int,
    manifest: Path | None = None,
) -> list[str]:
    top = _load_top_model()
    cmd = [
        str(PYTHON), "-m", "otitenet.train.train_triplet_new",
        "--path", args.dataset_path,
        "--groupkfold", "1",
        "--task", TASK,
        "--train_datasets", args.train_datasets,
        "--valid_dataset", args.valid_dataset,
        "--test_dataset", args.test_dataset,
        "--n_calibration", str(n_calibration),
        "--seed", str(seed),
        "--n_trials", str(args.n_trials),
        "--n_epochs", str(args.n_epochs),
        "--early_stop", str(args.early_stop),
        "--num_workers", str(args.num_workers),
        "--run_tag", run_tag,
        "--exp_id", "paper",
        "--log_dvclive", "1",
        "--dvclive_save_dvc_exp", "1",
        "--dvclive_monitor_system", "1",
        "--log_comet", "0",
        "--log_mlflow", "0",
        "--heavy_best_analysis", "0",
        "--run_explainability", "0",
        "--epoch_progress", "1",
    ]
    if getattr(args, "force_rerun", False):
        cmd.extend(["--reset_opt_state", "1"])
    cmd.extend(_top_model_args(top))
    if manifest is not None:
        cmd.extend(["--calibration_manifest_path", str(manifest)])
    return cmd


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    out = Path(args.output_dir)
    manifest4 = out / "manifests" / f"{args.test_dataset}_valid_test_n4_seed{args.seed}.csv"
    return [
        Experiment(
            "E01",
            "target_zero_shot_top_model",
            "Train/evaluate the current top model family with the target source held out and no target-image calibration.",
            _train_base(args, n_calibration=0, run_tag="PAPER_E01_ZERO_SHOT", seed=args.seed),
            "typically 6-13 min on the current A100 from historical runs; budget 15-25 min for fresh runs",
        ),
        Experiment(
            "E02",
            "target_4shot_valid_test_calibration",
            "Repeat E01 but add four proportional stratified valid images and four proportional stratified test images to the calibration/train support set.",
            _train_base(args, n_calibration=4, run_tag="PAPER_E02_4SHOT", seed=args.seed, manifest=manifest4),
            "typically 6-13 min on the current A100; compare directly with E01",
            required_before=(
                "Run E01, then create the manifest with: "
                f"{PYTHON} scripts/paper/prepare_test_calibration_manifest.py "
                "--split-csv <E01_RUN_DIR>/splits/test.csv "
                "--valid-split-csv <E01_RUN_DIR>/splits/valid.csv "
                f"--n 4 --seed {args.seed} --output {manifest4}"
            ),
        ),
        Experiment(
            "E03",
            "target_support_curve_top_model",
            "Optional curve for how much target support helps. Uses the same top model and repeats support sizes/seeds.",
            [
                str(PYTHON), "scripts/paper/run_paper_experiments.py",
                "--run-support-curve",
                "--support-sizes", args.support_sizes,
                "--seeds", args.seeds,
                "--n-trials", str(args.n_trials),
                "--n-epochs", str(args.n_epochs),
                "--early-stop", str(args.early_stop),
            ],
            "roughly one E01/E02 runtime per support-size/seed combination",
            required_before="Create one valid+test calibration manifest per support size/seed from the E01 valid/test splits.",
        ),
        Experiment(
            "E04",
            "paper_tables_and_figures",
            "Build paper tables, confusion matrices, missed-case tables, and support-curve plots from completed run folders.",
            [str(PYTHON), "scripts/paper/make_paper_outputs.py", "--discover", "--output-dir", str(out)],
            "CPU only; usually under 2 min",
        ),
        Experiment(
            "E05",
            "existing_broad_analysis_refresh",
            "Optional legacy broad analysis from all logs. Use for appendix/exploration, not as the main paper claim.",
            [
                str(PYTHON), "scripts/analysis/generate_paper_analysis.py",
                "--task", TASK,
                "--dataset", "otite_ds_64_USA_Turquie_Chili_GMFUNL_inference",
            ],
            "CPU mostly; depends on number of logs and plots",
        ),
    ]


def _quote(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _cmd_arg(cmd: list[str], flag: str, default: str = "") -> str:
    try:
        return str(cmd[cmd.index(flag) + 1])
    except ValueError:
        return default


def _dataset_key_from_path(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}_{p.name}"


def _task_run_dirs(task: str) -> set[str]:
    root = ROOT / "logs" / task
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def _register_paper_run(cmd: list[str], run_uuid: str) -> None:
    """Append a launch.sh-style job-manifest row so Quick Model Selection picks this run up."""
    task = _cmd_arg(cmd, "--task", TASK)
    dataset_path = _cmd_arg(cmd, "--path")
    dataset_key = _dataset_key_from_path(dataset_path) if dataset_path else ""
    manifest_dir = ROOT / "logs" / "progresses" / task / "paper" / "csv"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"PROD_{task}_job_manifest.csv"
    run_tag = _cmd_arg(cmd, "--run_tag")
    row = {
        "job_id": run_tag,
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
        "exp_id": run_tag,
    }
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOB_MANIFEST_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in JOB_MANIFEST_COLUMNS})
    print(f"Registered run {run_uuid} in {manifest_path}", flush=True)


def _run(cmd: list[str]) -> int:
    print("\n$", _quote(cmd), flush=True)
    is_training = "otitenet.train.train_triplet_new" in cmd
    task = _cmd_arg(cmd, "--task", TASK) if is_training else ""
    before = _task_run_dirs(task) if is_training else set()
    rc = subprocess.call(cmd, cwd=ROOT)
    if is_training and rc == 0:
        new_dirs = sorted(_task_run_dirs(task) - before)
        if new_dirs:
            _register_paper_run(cmd, new_dirs[-1])
        else:
            print(f"Warning: could not detect a new run directory under logs/{task} to register.", file=sys.stderr)
    return rc


def _print_plan(experiments: list[Experiment]) -> None:
    for exp in experiments:
        print(f"\n{exp.number}. {exp.name}")
        print(f"Purpose: {exp.purpose}")
        print(f"GPU time: {exp.gpu_time}")
        if exp.required_before:
            print(f"Before: {exp.required_before}")
        print("Command:")
        print(_quote(exp.command))


def _find_existing_run(task: str, run_tag: str) -> Path | None:
    """Reuse a prior completed run for this exact run_tag instead of retraining."""
    root = ROOT / "logs" / task
    if not root.is_dir():
        return None
    for run_dir in root.iterdir():
        meta = run_dir / "run_metadata.json"
        if not meta.exists() or not (run_dir / "model.pth").exists():
            continue
        try:
            data = json.loads(meta.read_text())
        except Exception:
            continue
        if str(data.get("run_tag", "")) == run_tag:
            return run_dir
    return None


def _run_support_curve(args: argparse.Namespace) -> int:
    # 0 (zero-shot) must run first per seed: it needs no calibration manifest, and its
    # splits/test.csv is what we sample from to auto-build the n>0 manifests below.
    sizes = sorted(dict.fromkeys(int(x.strip()) for x in args.support_sizes.split(",") if x.strip()))
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out = Path(args.output_dir)
    for seed in seeds:
        zero_shot_run: Path | None = None
        for n in sizes:
            run_tag = f"PAPER_E03_N{n}_S{seed}"
            manifest = None
            if n > 0:
                manifest = out / "manifests" / f"{args.test_dataset}_valid_test_n{n}_seed{seed}.csv"
                if not manifest.exists():
                    if zero_shot_run is None:
                        print(
                            f"No zero-shot (n=0) run available for seed {seed}; "
                            f"cannot auto-build the n={n} calibration manifest. "
                            "Include 0 in --support-sizes so it runs first."
                        )
                        return 2
                    test_split = zero_shot_run / "splits" / "test.csv"
                    valid_split = zero_shot_run / "splits" / "valid.csv"
                    if not test_split.exists() or not valid_split.exists():
                        print(f"Cannot auto-build manifest: missing {valid_split} or {test_split}.")
                        return 2
                    build_manifest(test_split, n, seed, manifest, extra_splits=[("valid", valid_split)])
                    print(f"Auto-built calibration manifest: {manifest}", flush=True)

            cmd = _train_base(args, n_calibration=n, run_tag=run_tag, seed=seed, manifest=manifest)
            existing = None if getattr(args, "force_rerun", False) else _find_existing_run(TASK, run_tag)
            if existing is not None:
                print(f"Reusing existing run for {run_tag}: {existing}", flush=True)
                _register_paper_run(cmd, existing.name)
                run_dir = existing
            else:
                before = _task_run_dirs(TASK)
                rc = _run(cmd)
                if rc != 0:
                    return rc
                new_dirs = sorted(_task_run_dirs(TASK) - before)
                run_dir = (ROOT / "logs" / TASK / new_dirs[-1]) if new_dirs else None

            if n == 0:
                zero_shot_run = run_dir
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print or run the numbered OtiteNet paper experiments.")
    parser.add_argument("--run", choices=["E01", "E02", "E03", "E04", "E05", "all"], help="Execute one numbered experiment. Default only prints commands.")
    parser.add_argument("--run-support-curve", action="store_true", help="Execute E03 support curve from existing manifests.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--train-datasets", default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--valid-dataset", default=DEFAULT_VALID)
    parser.add_argument("--test-dataset", default=DEFAULT_TARGET_TEST)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--support-sizes", default="8,12,16,20,0,4")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing paper runs and reset Optuna/cache state for new training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = build_experiments(args)
    if args.run_support_curve:
        return _run_support_curve(args)
    if not args.run:
        _print_plan(experiments)
        return 0
    selected = experiments if args.run == "all" else [e for e in experiments if e.number == args.run]
    for exp in selected:
        if exp.required_before and exp.number in {"E02"}:
            manifest = Path(args.output_dir) / "manifests" / f"{args.test_dataset}_valid_test_n4_seed{args.seed}.csv"
            if not manifest.exists():
                print(f"Missing required manifest: {manifest}")
                print(exp.required_before)
                return 2
        rc = _run(exp.command)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
