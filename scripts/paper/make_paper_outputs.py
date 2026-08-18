#!/usr/bin/env python3
"""Generate paper tables and figures from completed OtiteNet run folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _discover_runs() -> list[Path]:
    root = ROOT / "logs" / "otitis_four_class"
    if not root.exists():
        return []
    runs = []
    for summary in root.glob("*/run_summary.json"):
        meta = _read_json(summary.parent / "run_metadata.json")
        args = meta.get("args", {}) if isinstance(meta, dict) else {}
        if str(args.get("run_tag", "")).startswith("PAPER_"):
            runs.append(summary.parent)
    return sorted(runs)


def _normal_abnormal(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: "Normal" if x == "Normal" else "Abnormal")


def _load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    path = run_dir / f"{split}_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "label" not in df.columns or "pred" not in df.columns:
        return pd.DataFrame()
    df["split"] = split
    df["run_dir"] = str(run_dir)
    df["binary_label"] = _normal_abnormal(df["label"])
    df["binary_pred"] = _normal_abnormal(df["pred"])
    df["binary_error_type"] = np.where(
        (df["binary_label"] == "Abnormal") & (df["binary_pred"] == "Normal"),
        "false_negative_abnormal_as_normal",
        np.where(
            (df["binary_label"] == "Normal") & (df["binary_pred"] == "Abnormal"),
            "false_positive_normal_as_abnormal",
            "correct_or_multiclass_with_same_binary_status",
        ),
    )
    df["missed"] = df["label"].astype(str) != df["pred"].astype(str)
    return df


def _run_record(run_dir: Path) -> dict:
    summary = _read_json(run_dir / "run_summary.json")
    meta = _read_json(run_dir / "run_metadata.json")
    args = meta.get("args", {}) if isinstance(meta, dict) else {}
    split_config = summary.get("split_config", {}) if isinstance(summary, dict) else {}
    best_values = summary.get("best_values", {}) if isinstance(summary, dict) else {}
    params = summary.get("params", {}) if isinstance(summary, dict) else {}

    record = {
        "run_dir": str(run_dir),
        "run_tag": args.get("run_tag", ""),
        "task": args.get("task", ""),
        "model_name": args.get("model_name", ""),
        "n_calibration": args.get("n_calibration", ""),
        "seed": args.get("seed", ""),
        "train_datasets": split_config.get("train_datasets", args.get("train_datasets", "")),
        "valid_dataset": split_config.get("valid_dataset", args.get("valid_dataset", "")),
        "test_dataset": split_config.get("test_dataset", args.get("test_dataset", "")),
        "duration_seconds": summary.get("duration_seconds", np.nan),
        "best_mcc": summary.get("best_mcc", np.nan),
        "best_acc": summary.get("best_acc", np.nan),
        "classif_loss": args.get("classif_loss", ""),
        "dloss": args.get("dloss", ""),
        "fgsm": args.get("fgsm", ""),
        "normalize": args.get("normalize", ""),
        "dist_fct_arg": args.get("dist_fct", ""),
        "dist_fct_selected": params.get("dist_fct", ""),
    }

    for split in ("valid", "test"):
        values = best_values.get(split, {}) if isinstance(best_values, dict) else {}
        record[f"{split}_mcc"] = values.get("mcc", np.nan)
        record[f"{split}_acc"] = values.get("acc", np.nan)
        record[f"{split}_tpr"] = values.get("tpr", np.nan)
        record[f"{split}_tnr"] = values.get("tnr", np.nan)
        preds = _load_predictions(run_dir, split)
        if not preds.empty:
            record[f"{split}_mcc_from_predictions"] = matthews_corrcoef(preds["label"], preds["pred"])
            record[f"{split}_accuracy_from_predictions"] = accuracy_score(preds["label"], preds["pred"])
            binary = confusion_matrix(preds["binary_label"], preds["binary_pred"], labels=["Normal", "Abnormal"])
            tn, fp, fn, tp = binary.ravel()
            record[f"{split}_binary_fp"] = int(fp)
            record[f"{split}_binary_fn"] = int(fn)
            record[f"{split}_binary_fp_rate"] = fp / (fp + tn) if (fp + tn) else np.nan
            record[f"{split}_binary_fn_rate"] = fn / (fn + tp) if (fn + tp) else np.nan
            record[f"{split}_n"] = int(len(preds))
    return record


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n")


def _plot_confusion(df: pd.DataFrame, out_path: Path, title: str) -> None:
    labels = sorted(set(df["label"].astype(str)) | set(df["pred"].astype(str)))
    cm = confusion_matrix(df["label"].astype(str), df["pred"].astype(str), labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_support_curve(results: pd.DataFrame, out_path: Path) -> None:
    df = results.copy()
    df["n_calibration"] = pd.to_numeric(df["n_calibration"], errors="coerce")
    df["test_mcc"] = pd.to_numeric(df["test_mcc"], errors="coerce")
    df = df.dropna(subset=["n_calibration", "test_mcc"])
    if df.empty:
        return
    grouped = df.groupby("n_calibration")["test_mcc"].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(grouped["n_calibration"], grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", capsize=4)
    ax.set_xlabel("Target-domain calibration images")
    ax.set_ylabel("Test MCC")
    ax.set_title("Target support-set calibration curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_outputs(run_dirs: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    records = [_run_record(run_dir) for run_dir in run_dirs]
    results = pd.DataFrame(records)
    results.to_csv(table_dir / "paper_run_summary.csv", index=False)
    if not results.empty:
        cols = [
            c
            for c in [
                "run_tag",
                "model_name",
                "n_calibration",
                "seed",
                "train_datasets",
                "valid_dataset",
                "test_dataset",
                "valid_mcc",
                "test_mcc",
                "valid_acc",
                "test_acc",
                "test_binary_fp",
                "test_binary_fn",
                "test_binary_fp_rate",
                "test_binary_fn_rate",
                "duration_seconds",
                "run_dir",
            ]
            if c in results.columns
        ]
        _write_markdown_table(results[cols], table_dir / "paper_run_summary.md")
        _plot_support_curve(results, figure_dir / "support_curve_test_mcc.png")

    missed_frames = []
    for run_dir in run_dirs:
        tag = _run_record(run_dir).get("run_tag", run_dir.name)
        for split in ("valid", "test"):
            preds = _load_predictions(run_dir, split)
            if preds.empty:
                continue
            missed = preds[preds["missed"]].copy()
            if not missed.empty:
                missed["run_tag"] = tag
                keep = [
                    c
                    for c in ["run_tag", "split", "name", "label", "pred", "binary_error_type", "run_dir"]
                    if c in missed.columns
                ]
                missed_frames.append(missed[keep])
            _plot_confusion(preds, figure_dir / f"confusion_{tag}_{split}.png", f"{tag} {split}")

    if missed_frames:
        missed_all = pd.concat(missed_frames, ignore_index=True)
    else:
        missed_all = pd.DataFrame(
            columns=["run_tag", "split", "name", "label", "pred", "binary_error_type", "run_dir"]
        )
    missed_all.to_csv(table_dir / "missed_cases_for_physician_review.csv", index=False)
    _write_markdown_table(missed_all.head(200), table_dir / "missed_cases_for_physician_review.md")
    print(f"Wrote paper outputs to {output_dir}")
    print(f"Run folders analysed: {len(run_dirs)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures from run folders.")
    parser.add_argument("--run-dir", action="append", type=Path, default=[], help="Completed run directory. May be repeated.")
    parser.add_argument("--discover", action="store_true", help="Discover logs/otitis_four_class/* runs whose run_tag starts with PAPER_.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_outputs")
    args = parser.parse_args()

    run_dirs = list(args.run_dir)
    if args.discover:
        run_dirs.extend(_discover_runs())
    run_dirs = sorted(set(path.resolve() for path in run_dirs if (path / "run_summary.json").exists()))
    if not run_dirs:
        raise SystemExit("No completed run directories found. Pass --run-dir or use --discover after running PAPER_* experiments.")
    make_outputs(run_dirs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
