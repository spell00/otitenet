#!/usr/bin/env python3
"""Run E03 support curves using historically best fixed model configs."""

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
from prepare_test_calibration_manifest import build_manifest  # noqa: E402

PYTHON = ROOT / ".conda" / "bin" / "python"
TASK = "four_classes_220726"
DEFAULT_DATASET_PATH = "./data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference"
DEFAULT_PUBLIC_TRAIN = "Banque_Comert_Turquie_2020_jpg,Banque_Calaman_USA_2020_trie_CM,GMFUNL_jan2023"
DEFAULT_VALID = "Banque_Viscaino_Chili_2020"
DEFAULT_TARGET_TEST = "inference"
DEFAULT_OUTPUT_DIR = ROOT / "paper_outputs"

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

MANIFEST_COLUMNS = [
    "job_id", "job_state", "model", "uuid", "task", "dataset_name", "dataset_key",
    "train_datasets", "valid_dataset", "test_dataset", "n_calibration",
    "classif_loss", "loss", "dloss", "dist_fct", "new_size", "fgsm",
    "prototype", "prototypes", "n_positives", "n_negatives", "normalize",
    "knn", "n_neighbors", "prototype_strategy", "prototype_components", "exp_id",
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


def _dataset_key_from_path(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}_{p.name}"


def _task_run_dirs(task: str) -> set[str]:
    root = ROOT / "logs" / task
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def _support_sizes_for_execution(sizes: list[int]) -> list[int]:
    return list(dict.fromkeys(sizes))


def _int_or_default(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _find_existing_zero_shot_split(args: argparse.Namespace, seed: int) -> Path | None:
    root = ROOT / "logs" / TASK
    if not root.is_dir():
        return None

    matches: list[Path] = []
    for run_dir in root.iterdir():
        split = run_dir / "splits" / "test.csv"
        meta = run_dir / "run_metadata.json"
        if not split.exists() or not meta.exists():
            continue
        try:
            data = json.loads(meta.read_text())
        except Exception:
            continue
        run_args = data.get("args", {})
        if _int_or_default(run_args.get("n_calibration"), -1) != 0:
            continue
        if _int_or_default(run_args.get("seed"), -1) != int(seed):
            continue
        if str(run_args.get("task", "")) != TASK:
            continue
        if str(run_args.get("test_dataset", "")) != str(args.test_dataset):
            continue
        if str(run_args.get("valid_dataset", "")) != str(args.valid_dataset):
            continue
        if str(run_args.get("train_datasets", "")) != str(args.train_datasets):
            continue
        matches.append(run_dir)

    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _register_run(cmd: list[str], run_uuid: str) -> None:
    task = _cmd_arg(cmd, "--task", TASK)
    dataset_path = _cmd_arg(cmd, "--path")
    dataset_key = _dataset_key_from_path(dataset_path) if dataset_path else ""
    manifest_dir = ROOT / "logs" / "progresses" / task / "prev_best" / "csv"
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
        "exp_id": _cmd_arg(cmd, "--run_tag"),
    }
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})


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


def _build_cmd(args, cfg: pd.Series, n: int, seed: int, rank: int, manifest: Path | None) -> list[str]:
    version = _clean(getattr(args, "version_label", ""), "")
    version_part = f"_{version}" if version else ""
    run_tag = f"PAPER_E03_PREVBEST{version_part}_R{rank:02d}_N{n}_S{seed}"
    cmd = [
        str(PYTHON), "-m", "otitenet.train.train_triplet_new",
        "--path", args.dataset_path,
        "--groupkfold", "1",
        "--task", TASK,
        "--train_datasets", args.train_datasets,
        "--valid_dataset", args.valid_dataset,
        "--test_dataset", args.test_dataset,
        "--n_calibration", str(n),
        "--seed", str(seed),
        "--n_trials", str(args.n_trials),
        "--n_epochs", str(args.n_epochs),
        "--early_stop", str(args.early_stop),
        "--num_workers", str(args.num_workers),
        "--run_tag", run_tag,
        "--exp_id", "paper_prevbest",
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
    if manifest is not None:
        cmd.extend(["--calibration_manifest_path", str(manifest)])
    return cmd


def _run(cmd: list[str]) -> int:
    print("\n$", _quote(cmd), flush=True)
    before = _task_run_dirs(TASK)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc == 0:
        new_dirs = sorted(_task_run_dirs(TASK) - before)
        if new_dirs:
            _register_run(cmd, new_dirs[-1])
        else:
            print(f"Warning: could not detect a new run dir for {_cmd_arg(cmd, '--run_tag')}", file=sys.stderr)
    return rc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E03 with historical best configs on four_classes_220726.")
    parser.add_argument("--top-configs", type=int, default=8)
    parser.add_argument("--source-tasks", default="four_classes_220726")
    parser.add_argument("--support-sizes", default="0")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--train-datasets", default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--valid-dataset", default=DEFAULT_VALID)
    parser.add_argument("--test-dataset", default=DEFAULT_TARGET_TEST)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
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
    print("Selected previous-best configs:", flush=True)
    print(configs[["task", "run_tag", "valid_mcc", *CONFIG_COLUMNS]].to_string(index=False), flush=True)

    sizes = _support_sizes_for_execution([int(x.strip()) for x in args.support_sizes.split(",") if x.strip()])
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out = Path(args.output_dir)
    for seed in seeds:
        zero_shot_run: Path | None = _find_existing_zero_shot_split(args, seed)
        if zero_shot_run is not None:
            print(
                f"Using zero-shot splits for seed {seed}: "
                f"{zero_shot_run / 'splits' / 'valid.csv'} and {zero_shot_run / 'splits' / 'test.csv'}",
                flush=True,
            )
        for n in sizes:
            manifest = None
            if n > 0:
                manifest = out / "manifests" / f"{args.test_dataset}_prevbest_valid_test_n{n}_seed{seed}.csv"
                if not manifest.exists():
                    if args.dry_run:
                        print(
                            f"# Would auto-build calibration manifest from the zero-shot valid/test splits: {manifest}",
                            flush=True,
                        )
                    elif zero_shot_run is None:
                        print(
                            f"No existing zero-shot split for seed {seed}; cannot build n={n} manifest "
                            "without first running n_calibration=0. Run once with --support-sizes 0, "
                            "or provide an existing manifest.",
                            file=sys.stderr,
                        )
                        return 2
                    else:
                        valid_split = zero_shot_run / "splits" / "valid.csv"
                        test_split = zero_shot_run / "splits" / "test.csv"
                        if not valid_split.exists() or not test_split.exists():
                            print(
                                f"Cannot auto-build manifest: missing {valid_split} or {test_split}.",
                                file=sys.stderr,
                            )
                            return 2
                        build_manifest(test_split, n, seed, manifest, extra_splits=[("valid", valid_split)])
                        print(f"Auto-built calibration manifest: {manifest}", flush=True)
            for idx, cfg in configs.iterrows():
                cmd = _build_cmd(args, cfg, n=n, seed=seed, rank=idx + 1, manifest=manifest)
                if args.dry_run:
                    print(_quote(cmd))
                    continue
                before = _task_run_dirs(TASK)
                rc = _run(cmd)
                if rc != 0:
                    return rc
                if n == 0 and zero_shot_run is None:
                    new_dirs = sorted(_task_run_dirs(TASK) - before)
                    if new_dirs:
                        zero_shot_run = ROOT / "logs" / TASK / new_dirs[-1]
        if args.dry_run:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
