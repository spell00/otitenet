"""Sync configs/best_models.csv artifact pointers into best_models_registry."""

from __future__ import annotations

import argparse

import mysql.connector
import pandas as pd


def _ensure_columns(cursor) -> None:
    columns = {
        "artifact_id": "VARCHAR(32) NULL",
        "best_model_dir": "TEXT NULL",
        "source_run_log_path": "TEXT NULL",
        "valid_mcc": "FLOAT NULL",
        "test_mcc": "FLOAT NULL",
        "all_mcc": "FLOAT NULL",
        "train_mcc": "FLOAT NULL",
        "train_auc": "FLOAT NULL",
        "valid_auc": "FLOAT NULL",
        "test_auc": "FLOAT NULL",
        "all_auc": "FLOAT NULL",
        "valid_accuracy": "FLOAT NULL",
        "ece": "FLOAT NULL",
        "brier": "FLOAT NULL",
        "best_head_config": "VARCHAR(255) NULL",
        "best_head_name": "VARCHAR(255) NULL",
        "best_head_family": "VARCHAR(64) NULL",
        "best_head_n_aug": "INT NULL",
    }
    for name, column_type in columns.items():
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'best_models_registry'
              AND COLUMN_NAME = %s
            """,
            (name,),
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(f"ALTER TABLE best_models_registry ADD COLUMN {name} {column_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/best_models.csv")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--user", default="y_user")
    parser.add_argument("--password", default="password")
    parser.add_argument("--database", default="results_db")
    args = parser.parse_args()

    registry = pd.read_csv(args.registry, dtype=str).fillna("")
    conn = mysql.connector.connect(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database,
    )
    try:
        cursor = conn.cursor()
        _ensure_columns(cursor)
        updated = 0
        for _, row in registry.iterrows():
            optional_assignments = []
            optional_values = []
            for csv_col, db_col in [
                ("valid_mcc", "valid_mcc"),
                ("test_mcc", "test_mcc"),
                ("all_mcc", "all_mcc"),
                ("train_mcc", "train_mcc"),
                ("train_auc", "train_auc"),
                ("valid_auc", "valid_auc"),
                ("test_auc", "test_auc"),
                ("all_auc", "all_auc"),
                ("valid_accuracy", "valid_accuracy"),
                ("ece", "ece"),
                ("brier", "brier"),
                ("best_head_config", "best_head_config"),
                ("best_head_name", "best_head_name"),
                ("best_head_family", "best_head_family"),
                ("best_head_n_aug", "best_head_n_aug"),
            ]:
                value = row.get(csv_col, "")
                if value != "":
                    optional_assignments.append(f"{db_col}=%s")
                    optional_values.append(value)
            if row.get("valid_mcc", "") != "":
                optional_assignments.append("mcc=%s")
                optional_values.append(row.get("valid_mcc", ""))

            metric_sql = ""
            if optional_assignments:
                metric_sql = ", " + ", ".join(optional_assignments)

            cursor.execute(
                f"""
                UPDATE best_models_registry
                SET artifact_id=%s,
                    best_model_dir=%s,
                    source_run_log_path=%s
                    {metric_sql}
                WHERE log_path=%s
                   OR log_path=%s
                   OR best_model_dir=%s
                   OR source_run_log_path=%s
                """,
                (
                    row.get("artifact_id", ""),
                    row.get("model_dir", ""),
                    row.get("run_log_path", ""),
                    *optional_values,
                    row.get("model_dir", ""),
                    row.get("run_log_path", ""),
                    row.get("model_dir", ""),
                    row.get("run_log_path", ""),
                ),
            )
            updated += cursor.rowcount
        conn.commit()
        print(f"Updated {updated} best_models_registry rows from {args.registry}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
