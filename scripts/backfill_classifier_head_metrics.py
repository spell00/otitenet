"""Backfill best classifier-head metrics from optimization caches into registries."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import mysql.connector
import pandas as pd

from otitenet.app.pages.learned_embedding import _persist_best_head_metrics_to_registry
from otitenet.app.services.embedding_optimization_service import args_from_model_row
from otitenet.app.utils import enumerate_classification_heads


def _row_dict_from_models_csv(row: pd.Series) -> dict:
    return {
        "Task": row.get("task", ""),
        "Model Name": row.get("model", ""),
        "Dataset": row.get("path", ""),
        "Log Path": row.get("complete_log_path", ""),
        "train_datasets": str(row.get("train_datasets", "")).replace(";", ","),
        "valid_dataset": row.get("valid_dataset", ""),
        "test_dataset": row.get("test_dataset", ""),
        "split_config_key": str(row.get("split_config_key", "")).replace(";", ","),
        "NSize": str(row.get("nsize", "")).replace("nsize", ""),
        "FGSM": str(row.get("fgsm", "")).replace("fsgm", "").replace("fgsm", ""),
        "N_Calibration": str(row.get("n_calibration", "")).replace("ncal", ""),
        "Classif_Loss": row.get("loss", ""),
        "DLoss": row.get("dloss", ""),
        "Dist_Fct": str(row.get("dist_fct", "")).replace("dist_", ""),
        "Prototypes": str(row.get("prototype", "")).replace("prototypes_", ""),
        "NPos": str(row.get("n_positives", "")).replace("npos", ""),
        "NNeg": str(row.get("n_negatives", "")).replace("nneg", ""),
        "Normalize": row.get("normalize", ""),
        "N_Neighbors": str(row.get("n_neighbors", "")).replace("knn", ""),
    }


def _row_dict_from_best_models_csv(row: pd.Series) -> dict:
    dataset = row.get("dataset_path", "")
    return {
        "Task": row.get("task", ""),
        "Model Name": row.get("model_name", ""),
        "Dataset": dataset,
        "Log Path": row.get("run_log_path", "") or row.get("model_dir", ""),
        "Best Model Dir": row.get("model_dir", ""),
        "Source Run Path": row.get("run_log_path", ""),
        "NSize": row.get("nsize", ""),
        "FGSM": row.get("fgsm", ""),
        "N_Calibration": row.get("n_calibration", ""),
        "Classif_Loss": row.get("classif_loss", ""),
        "DLoss": row.get("dloss", ""),
        "Dist_Fct": row.get("dist_fct", ""),
        "Prototypes": row.get("prototypes", ""),
        "NPos": row.get("npos", ""),
        "NNeg": row.get("nneg", ""),
        "Normalize": row.get("normalize", ""),
        "N_Neighbors": row.get("n_neighbors", ""),
    }


def _iter_registry_rows(task: str | None):
    seen = set()
    for models_csv in sorted(Path("logs/best_models").glob("*/models.csv")):
        try:
            df = pd.read_csv(models_csv, dtype=str).fillna("")
        except Exception:
            continue
        for _, row in df.iterrows():
            row_dict = _row_dict_from_models_csv(row)
            if task and row_dict.get("Task") != task:
                continue
            key = row_dict.get("Log Path") or tuple(row_dict.items())
            if key in seen:
                continue
            seen.add(key)
            yield row_dict

    best_models = Path("configs/best_models.csv")
    if best_models.exists():
        try:
            df = pd.read_csv(best_models, dtype=str).fillna("")
        except Exception:
            df = pd.DataFrame()
        for _, row in df.iterrows():
            row_dict = _row_dict_from_best_models_csv(row)
            if task and row_dict.get("Task") != task:
                continue
            key = row_dict.get("Best Model Dir") or row_dict.get("Log Path") or tuple(row_dict.items())
            if key in seen:
                continue
            seen.add(key)
            yield row_dict


def _result_from_head(head: dict) -> dict:
    config = head.get("config")
    metric_keys = {
        "train_mcc",
        "valid_mcc",
        "test_mcc",
        "all_mcc",
        "train_accuracy",
        "valid_accuracy",
        "test_accuracy",
        "train_auc",
        "valid_auc",
        "test_auc",
        "all_auc",
        "ece",
        "brier",
        "valid_ece",
        "valid_brier",
    }
    metrics = {
        key: value
        for key, value in head.items()
        if key in metric_keys
        or " F1 " in str(key)
        or " Recall " in str(key)
        or " Precision " in str(key)
        or " Support " in str(key)
    }
    metrics.setdefault("valid_mcc", head.get("valid_mcc", head.get("mcc")))
    return {
        "best_config": config,
        "best_k": config,
        "best_mcc": head.get("valid_mcc", head.get("mcc")),
        "best_valid_mcc": head.get("valid_mcc", head.get("mcc")),
        "best_head_metrics": metrics,
        "train_datasets": head.get("train_datasets"),
        "valid_dataset": head.get("valid_dataset"),
        "test_dataset": head.get("test_dataset"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None)
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--user", default="y_user")
    parser.add_argument("--password", default="password")
    parser.add_argument("--database", default="results_db")
    args = parser.parse_args()

    conn = None
    cursor = None
    if not args.no_db:
        conn = mysql.connector.connect(
            host=args.host,
            user=args.user,
            password=args.password,
            database=args.database,
        )
        cursor = conn.cursor()

    ctx = SimpleNamespace(conn=conn, cursor=cursor)
    updated = 0
    skipped = 0
    try:
        for row_dict in _iter_registry_rows(args.task):
            try:
                model_args = args_from_model_row(SimpleNamespace(), row_dict)
                heads = enumerate_classification_heads(model_args, include_all_n_aug=True)
                if not heads:
                    skipped += 1
                    continue
                best_head = max(
                    heads,
                    key=lambda h: float(h.get("valid_mcc", h.get("mcc", float("-inf")))),
                )
                result = _result_from_head(best_head)
                _persist_best_head_metrics_to_registry(ctx, row_dict, model_args, result)
                updated += 1
            except Exception as exc:
                skipped += 1
                print(f"[skip] {row_dict.get('Log Path') or row_dict.get('Best Model Dir')}: {exc}")
    finally:
        if conn is not None:
            conn.close()

    print(f"updated={updated} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
