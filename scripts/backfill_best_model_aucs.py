#!/usr/bin/env python3

import os
import json
import math
import argparse
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import mysql.connector
from sklearn.metrics import roc_auc_score


DEFAULT_POS_LABEL = "NotNormal"


def connect_db(args):
    return mysql.connector.connect(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database,
        buffered=True,
        autocommit=False,
    )


def ensure_auc_columns(cursor, conn, table_name: str = "best_models_registry"):
    """Add AUC columns to best_models_registry if they do not exist."""
    for col in ["train_auc", "valid_auc", "test_auc"]:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
              AND COLUMN_NAME = %s
            """,
            (table_name, col),
        )
        exists = cursor.fetchone()[0] > 0

        if not exists:
            print(f"[schema] Adding column {table_name}.{col}")
            cursor.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` DOUBLE NULL")

    conn.commit()


def read_json(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Could not read JSON {path}: {e}")
        return None


def write_json(path: str, payload: dict) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True
    except Exception as e:
        print(f"[warn] Could not write JSON {path}: {e}")
        return False


def resolve_original_log_path(log_path: str) -> Optional[str]:
    """Some copied best-model folders point back to original training folder in run_metadata.json."""
    metadata_path = os.path.join(log_path, "run_metadata.json")
    metadata = read_json(metadata_path)

    if not metadata:
        return None

    original_log_path = metadata.get("complete_log_path")
    if original_log_path and original_log_path != log_path:
        return original_log_path

    return None


def resolve_prediction_csv(log_path: str, split: str) -> Optional[str]:
    """Find train/valid/test predictions CSV for a model folder."""
    if not log_path:
        return None

    candidates = [
        os.path.join(log_path, f"{split}_predictions.csv"),
        os.path.join(log_path, "app_predictions", f"fast__use_training_setting__eval_npz__{split}_predictions.csv"),
        os.path.join(log_path, "app_predictions", f"fast__knn__eval_npz__{split}_predictions.csv"),
        os.path.join(log_path, "app_predictions", f"fast__mlp_head__eval_npz__{split}_predictions.csv"),
        os.path.join(log_path, "app_predictions", f"exact__use_training_setting__predict__{split}_predictions.csv"),
        os.path.join(log_path, "app_predictions", f"exact__knn__predict__{split}_predictions.csv"),
        os.path.join(log_path, "app_predictions", f"exact__mlp_head__predict__{split}_predictions.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    summary = read_json(os.path.join(log_path, "run_summary.json"))
    if summary:
        artifacts = summary.get("artifacts") or {}
        artifact_path = artifacts.get(f"{split}_predictions_csv")

        if artifact_path:
            if not os.path.isabs(artifact_path):
                artifact_path = os.path.join(log_path, artifact_path)

            if os.path.exists(artifact_path):
                return artifact_path

    original_log_path = resolve_original_log_path(log_path)
    if original_log_path:
        return resolve_prediction_csv(original_log_path, split)

    return None


def find_probability_column(df: pd.DataFrame, pos_label: str) -> Optional[str]:
    """Find the score/probability column to use for binary AUC."""
    exact = f"probs_{pos_label}"

    if exact in df.columns:
        return exact

    lower_map = {str(c).lower(): c for c in df.columns}
    if exact.lower() in lower_map:
        return lower_map[exact.lower()]

    prob_cols = [c for c in df.columns if str(c).startswith("probs_")]

    # Prefer a probability column whose suffix matches pos_label case-insensitively
    for c in prob_cols:
        suffix = str(c).replace("probs_", "", 1)
        if suffix.lower() == str(pos_label).lower():
            return c

    # If binary and only one probability column exists, use it
    if len(prob_cols) == 1:
        return prob_cols[0]

    # Common fallback names
    for c in ["proba", "prob", "probability", "score", "y_score"]:
        if c in df.columns:
            return c

    return None


def compute_binary_auc_from_csv(csv_path: Optional[str], pos_label: str = DEFAULT_POS_LABEL) -> Optional[float]:
    if not csv_path or not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[warn] Could not read {csv_path}: {e}")
        return None

    if "label" not in df.columns:
        print(f"[warn] No label column in {csv_path}")
        return None

    prob_col = find_probability_column(df, pos_label)
    if prob_col is None:
        print(f"[warn] No probability column found in {csv_path}")
        return None

    y_true = (df["label"].astype(str) == str(pos_label)).astype(int).to_numpy()
    y_score = pd.to_numeric(df[prob_col], errors="coerce").to_numpy()

    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0:
        print(f"[warn] AUC undefined for {csv_path}: no valid scores")
        return None

    if len(np.unique(y_true)) < 2:
        print(f"[warn] AUC undefined for {csv_path}: only one class present")
        return None

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception as e:
        print(f"[warn] AUC failed for {csv_path}: {e}")
        return None


def compute_aucs_for_log_path(log_path: str, pos_label: str = DEFAULT_POS_LABEL) -> Dict[str, Any]:
    aucs = {}

    for split in ["train", "valid", "test"]:
        csv_path = resolve_prediction_csv(log_path, split)
        auc = compute_binary_auc_from_csv(csv_path, pos_label=pos_label)

        aucs[f"{split}_auc"] = auc
        aucs[f"{split}_csv"] = csv_path

    return aucs


def update_run_summary_json(log_path: str, aucs: Dict[str, Any], dry_run: bool = False) -> bool:
    """Write AUC values into run_summary.json under best_values.<split>.auc."""
    summary_path = os.path.join(log_path, "run_summary.json")

    if not os.path.exists(summary_path):
        original_log_path = resolve_original_log_path(log_path)
        if original_log_path:
            summary_path = os.path.join(original_log_path, "run_summary.json")

    if not os.path.exists(summary_path):
        return False

    payload = read_json(summary_path)
    if payload is None:
        return False

    best_values = payload.setdefault("best_values", {})

    changed = False

    for split in ["train", "valid", "test"]:
        auc_val = aucs.get(f"{split}_auc")

        if auc_val is None:
            continue

        split_dict = best_values.setdefault(split, {})
        old_val = split_dict.get("auc")

        try:
            new_val = float(auc_val)
        except Exception:
            continue

        if old_val != new_val:
            split_dict["auc"] = new_val
            changed = True

    if not changed:
        return True

    if dry_run:
        return True

    return write_json(summary_path, payload)


def clean_float_for_db(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None

    try:
        x = float(x)
    except Exception:
        return None

    if math.isnan(x) or math.isinf(x):
        return None

    return x


def print_db_counts(cursor, table_name: str):
    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
    total = cursor.fetchone()[0]

    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM `{table_name}`
        WHERE log_path IS NOT NULL AND log_path <> ''
        """
    )
    with_log_path = cursor.fetchone()[0]

    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM `{table_name}`
        WHERE train_auc IS NOT NULL
           OR valid_auc IS NOT NULL
           OR test_auc IS NOT NULL
        """
    )
    with_auc = cursor.fetchone()[0]

    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM `{table_name}`
        WHERE log_path IS NOT NULL
          AND log_path <> ''
          AND (train_auc IS NULL OR valid_auc IS NULL OR test_auc IS NULL)
        """
    )
    missing_any_auc = cursor.fetchone()[0]

    print(f"[info] Total rows in {table_name}: {total}")
    print(f"[info] Rows with log_path: {with_log_path}")
    print(f"[info] Rows with at least one AUC: {with_auc}")
    print(f"[info] Rows with log_path and at least one missing AUC: {missing_any_auc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--user", default="y_user")
    parser.add_argument("--password", default="password")
    parser.add_argument("--database", default="results_db")
    parser.add_argument("--table", default="best_models_registry")
    parser.add_argument("--pos-label", default=DEFAULT_POS_LABEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    # If enabled, only process rows where one of train/valid/test AUC is missing
    parser.add_argument("--only-missing", action="store_true")

    # If enabled, DB values are not overwritten when already present
    parser.add_argument("--preserve-existing-db", action="store_true")

    # If enabled, do not patch run_summary.json
    parser.add_argument("--skip-summary-json", action="store_true")

    args = parser.parse_args()

    conn = connect_db(args)
    cursor = conn.cursor(buffered=True)

    ensure_auc_columns(cursor, conn, table_name=args.table)
    print_db_counts(cursor, args.table)

    where = "WHERE log_path IS NOT NULL AND log_path <> ''"

    if args.only_missing:
        where += " AND (train_auc IS NULL OR valid_auc IS NULL OR test_auc IS NULL)"

    limit_sql = f" LIMIT {int(args.limit)}" if args.limit else ""

    cursor.execute(
        f"""
        SELECT id, log_path, train_auc, valid_auc, test_auc
        FROM `{args.table}`
        {where}
        ORDER BY id ASC
        {limit_sql}
        """
    )

    rows = cursor.fetchall() or []
    print(f"[info] Found {len(rows)} model rows to process")

    updated_db = 0
    updated_json = 0
    skipped_no_auc = 0

    for model_id, log_path, old_train_auc, old_valid_auc, old_test_auc in rows:
        aucs = compute_aucs_for_log_path(log_path, pos_label=args.pos_label)

        train_auc = clean_float_for_db(aucs.get("train_auc"))
        valid_auc = clean_float_for_db(aucs.get("valid_auc"))
        test_auc = clean_float_for_db(aucs.get("test_auc"))

        if args.preserve_existing_db:
            if old_train_auc is not None:
                train_auc = old_train_auc
            if old_valid_auc is not None:
                valid_auc = old_valid_auc
            if old_test_auc is not None:
                test_auc = old_test_auc

        has_any_auc = any(v is not None for v in [train_auc, valid_auc, test_auc])

        summary_updated = False
        if not args.skip_summary_json:
            summary_updated = update_run_summary_json(
                log_path,
                {
                    "train_auc": train_auc,
                    "valid_auc": valid_auc,
                    "test_auc": test_auc,
                    "train_csv": aucs.get("train_csv"),
                    "valid_csv": aucs.get("valid_csv"),
                    "test_csv": aucs.get("test_csv"),
                },
                dry_run=args.dry_run,
            )

        print(
            f"[model {model_id}] "
            f"train_auc={train_auc} valid_auc={valid_auc} test_auc={test_auc} "
            f"summary_updated={summary_updated} "
            f"log_path={log_path}"
        )

        if not has_any_auc:
            skipped_no_auc += 1
            continue

        if not args.dry_run:
            cursor.execute(
                f"""
                UPDATE `{args.table}`
                SET train_auc = %s,
                    valid_auc = %s,
                    test_auc = %s
                WHERE id = %s
                """,
                (train_auc, valid_auc, test_auc, model_id),
            )
            conn.commit()

        updated_db += 1
        if summary_updated:
            updated_json += 1

    cursor.close()
    conn.close()

    print(
        f"[done] updated_db={updated_db}, "
        f"updated_json={updated_json}, "
        f"skipped_no_auc={skipped_no_auc}, "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()