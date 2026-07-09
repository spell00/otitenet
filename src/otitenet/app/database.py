"""
Database operations for the Otite application.

This module handles all MySQL database interactions including:
- Connection management
- Schema migrations
- Results storage and retrieval
- Model registry queries
- User and people management
"""

import streamlit as st
import mysql.connector
from mysql.connector import Error
import os
import json
from pathlib import Path
from otitenet.app.utils import (
    LEGACY_TRAIN_DATASETS,
    LEGACY_VALID_DATASET,
    extract_params_from_log_path,
    split_config_key,
    _unique_preserve_order,
)
import numpy as np

def _mysql_value(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return str(x.tolist())
    return x


def _mysql_values(values):
    return tuple(_mysql_value(v) for v in values)


def is_mysql_connection_lost_error(exc) -> bool:
    """Return True for MySQL errors that mean the cursor/connection is unusable."""
    errno = getattr(exc, "errno", None)
    if errno in {2006, 2013, 2055}:
        return True
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "mysql server has gone away",
            "lost connection",
            "bad file descriptor",
            "eof occurred in violation of protocol",
            "decryption_failed_or_bad_record_mac",
            "bad record mac",
        )
    )


def _safe_rollback(conn):
    try:
        conn.rollback()
    except Exception:
        pass


def _table_row_count(cursor, table_name: str) -> int | None:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row = cursor.fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def _dedupe_max_rows() -> int:
    try:
        return int(os.environ.get("OTITENET_DEDUPE_MAX_ROWS", "10000"))
    except Exception:
        return 10000


def _db_config() -> dict:
    """Return MySQL connection settings with env overrides for local installs."""
    return {
        "host": os.environ.get("OTITENET_DB_HOST", "localhost"),
        "user": os.environ.get("OTITENET_DB_USER", "y_user"),
        "password": os.environ.get("OTITENET_DB_PASSWORD", "password"),
        "database": os.environ.get("OTITENET_DB_NAME", "results_db"),
        "buffered": True,
        "autocommit": True,
        # Avoid sporadic C-extension parsing issues seen as
        # "bytearray index out of range" on some environments.
        "use_pure": True,
        "ssl_disabled": True,
    }


@st.cache_resource
def get_db_connection():
    """Create and cache a MySQL database connection."""
    try:
        conn = mysql.connector.connect(**_db_config())
        return conn
    except Error as e:
        st.error(f"❌ Database connection error: {e}")
        st.stop()


def create_db():
    """Initialize database connection with retry logic."""
    try:
        conn = get_db_connection()
        if not conn.is_connected():
            conn.reconnect(attempts=3, delay=1)
        
        # Ping the server to ensure connection is alive
        conn.ping(reconnect=True, attempts=3, delay=1)
        
        cursor = conn.cursor(buffered=True)
        # Minimal check to ensure it works
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return conn, cursor
    except Exception:
        # If the cached connection is broken, clear cache and try one more time
        try:
            get_db_connection.clear()
            conn = get_db_connection()
            if not conn.is_connected():
                conn.reconnect(attempts=3, delay=1)
            conn.ping(reconnect=True, attempts=3, delay=1)
            cursor = conn.cursor(buffered=True)
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return conn, cursor
        except Exception as e2:
            # Final fallback: bypass cache and open a direct uncached connection.
            # This recovers from rare cache/connector corruption states.
            try:
                conn = mysql.connector.connect(**_db_config())
                if not conn.is_connected():
                    conn.reconnect(attempts=3, delay=1)
                conn.ping(reconnect=True, attempts=3, delay=1)
                cursor = conn.cursor(buffered=True)
                cursor.execute("SELECT 1")
                cursor.fetchone()
                return conn, cursor
            except Exception as e3:
                if "Too many connections" in str(e3):
                    st.error("❌ MySQL has too many connections. Please restart MySQL service or wait for connections to timeout.")
                else:
                    st.error(f"❌ Database error: {e3}")
                st.stop()


def cleanup_duplicate_results_rows(conn, cursor):
    """Keep one latest inference result per exact image/person/model tuple."""
    row_count = _table_row_count(cursor, "results")
    max_rows = _dedupe_max_rows()
    if row_count is not None and row_count > max_rows:
        print(
            f"[database] Skipping results duplicate cleanup: {row_count} rows exceeds "
            f"OTITENET_DEDUPE_MAX_ROWS={max_rows}"
        )
        return

    try:
        cursor.execute(
            """
            DELETE r1 FROM results r1
            JOIN results r2
              ON COALESCE(r1.filename, '') = COALESCE(r2.filename, '')
             AND COALESCE(r1.person_id, -1) = COALESCE(r2.person_id, -1)
             AND COALESCE(r1.model_id, -1) = COALESCE(r2.model_id, -1)
             AND (
                    COALESCE(r1.timestamp, '1970-01-01') < COALESCE(r2.timestamp, '1970-01-01')
                 OR (
                        COALESCE(r1.timestamp, '1970-01-01') = COALESCE(r2.timestamp, '1970-01-01')
                    AND r1.id < r2.id
                    )
                 )
            WHERE r1.id <> r2.id
              AND r1.model_id IS NOT NULL
            """
        )
        conn.commit()
    except Exception as e:
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise

    # Legacy fallback for rows without model_id: exact same parameter tuple.
    try:
        cursor.execute(
            """
            DELETE r1 FROM results r1
            JOIN results r2
              ON COALESCE(r1.filename, '') = COALESCE(r2.filename, '')
             AND COALESCE(r1.person_id, -1) = COALESCE(r2.person_id, -1)
             AND COALESCE(r1.model_name, '') = COALESCE(r2.model_name, '')
             AND COALESCE(r1.task, '') = COALESCE(r2.task, '')
             AND COALESCE(r1.path, '') = COALESCE(r2.path, '')
             AND COALESCE(r1.n_neighbors, '') = COALESCE(r2.n_neighbors, '')
             AND COALESCE(r1.nsize, '') = COALESCE(r2.nsize, '')
             AND COALESCE(r1.fgsm, '') = COALESCE(r2.fgsm, '')
             AND COALESCE(r1.normalize, '') = COALESCE(r2.normalize, '')
             AND COALESCE(r1.n_calibration, '') = COALESCE(r2.n_calibration, '')
             AND COALESCE(r1.classif_loss, '') = COALESCE(r2.classif_loss, '')
             AND COALESCE(r1.dloss, '') = COALESCE(r2.dloss, '')
             AND COALESCE(r1.dist_fct, '') = COALESCE(r2.dist_fct, '')
             AND COALESCE(r1.prototypes, '') = COALESCE(r2.prototypes, '')
             AND COALESCE(r1.npos, '') = COALESCE(r2.npos, '')
             AND COALESCE(r1.nneg, '') = COALESCE(r2.nneg, '')
             AND (
                    COALESCE(r1.timestamp, '1970-01-01') < COALESCE(r2.timestamp, '1970-01-01')
                 OR (
                        COALESCE(r1.timestamp, '1970-01-01') = COALESCE(r2.timestamp, '1970-01-01')
                    AND r1.id < r2.id
                    )
                 )
            WHERE r1.id <> r2.id
              AND r1.model_id IS NULL
              AND r2.model_id IS NULL
            """
        )
        conn.commit()
    except Exception as e:
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def cleanup_duplicate_best_models_registry(conn, cursor):
    """Keep one best row for each exact best-model configuration."""
    row_count = _table_row_count(cursor, "best_models_registry")
    max_rows = _dedupe_max_rows()
    if row_count is not None and row_count > max_rows:
        print(
            f"[database] Skipping best_models_registry duplicate cleanup: {row_count} rows exceeds "
            f"OTITENET_DEDUPE_MAX_ROWS={max_rows}"
        )
        return

    columns = [
        "task", "model_name", "nsize", "fgsm", "prototypes", "npos", "nneg",
        "dloss", "dist_fct", "classif_loss", "n_calibration", "normalize",
        "n_neighbors", "prototype_strategy", "prototype_components",
        "train_datasets", "valid_dataset", "test_dataset", "split_config_key",
    ]
    join_clause = " AND ".join([f"b1.{col} <=> b2.{col}" for col in columns])
    try:
        cursor.execute(
            f"""
            DELETE b1 FROM best_models_registry b1
            JOIN best_models_registry b2
              ON {join_clause}
             AND b1.id <> b2.id
             AND (
                    COALESCE(b2.valid_mcc, b2.mcc, -9999) > COALESCE(b1.valid_mcc, b1.mcc, -9999)
                 OR (
                        COALESCE(b2.valid_mcc, b2.mcc, -9999) = COALESCE(b1.valid_mcc, b1.mcc, -9999)
                    AND COALESCE(b2.valid_auc, -9999) > COALESCE(b1.valid_auc, -9999)
                    )
                 OR (
                        COALESCE(b2.valid_mcc, b2.mcc, -9999) = COALESCE(b1.valid_mcc, b1.mcc, -9999)
                    AND COALESCE(b2.valid_auc, -9999) = COALESCE(b1.valid_auc, -9999)
                    AND b2.id > b1.id
                    )
                 )
            """
        )
        conn.commit()
    except Exception as e:
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def cleanup_incompatible_best_models_registry(conn, cursor):
    """Remove app-incompatible registry rows that do not have primary run artifacts."""
    from otitenet.app.artifact_registry import is_source_run_artifact_dir

    try:
        cursor.execute(
            """
            SELECT id, log_path, source_run_log_path
            FROM best_models_registry
            """
        )
        rows = cursor.fetchall() or []
    except Exception as e:
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise
        return

    delete_ids = []
    for row in rows:
        row_id = row[0]
        log_path = row[1] if len(row) > 1 else ""
        source_run_log_path = row[2] if len(row) > 2 else ""
        if is_source_run_artifact_dir(source_run_log_path) or is_source_run_artifact_dir(log_path):
            continue
        delete_ids.append(row_id)

    if not delete_ids:
        return

    try:
        placeholders = ",".join(["%s"] * len(delete_ids))
        cursor.execute(
            f"DELETE FROM best_models_registry WHERE id IN ({placeholders})",
            tuple(delete_ids),
        )
        conn.commit()
        print(f"[database] Removed {len(delete_ids)} incompatible best_models_registry row(s).")
    except Exception as e:
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def cleanup_duplicate_registry_csvs():
    """Deduplicate local registry CSVs by exact model/config keys."""
    try:
        import pandas as pd
    except Exception:
        return

    paths = [Path("configs/best_models.csv")]
    paths.extend(Path("logs/best_models").glob("*/models.csv"))
    key_candidates = [
        "task", "model_name", "dataset_path", "nsize", "fgsm", "n_calibration",
        "classif_loss", "dloss", "prototypes", "npos", "nneg",
        "prototype_strategy", "prototype_components", "normalize", "dist_fct",
        "n_neighbors", "train_datasets", "valid_dataset", "test_dataset",
        "split_config_key",
    ]

    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
            if df.empty:
                continue
            key_cols = [col for col in key_candidates if col in df.columns]
            if not key_cols:
                key_cols = [col for col in ["model_dir", "log_path", "run_log_path"] if col in df.columns]
            if not key_cols:
                continue
            metric_cols = [col for col in ["valid_mcc", "mcc", "valid_auc"] if col in df.columns]
            if metric_cols:
                sort_df = df.copy()
                for col in metric_cols:
                    sort_df[col] = pd.to_numeric(sort_df[col], errors="coerce")
                sort_df["_original_order"] = range(len(sort_df))
                sort_df = sort_df.sort_values(
                    metric_cols + ["_original_order"],
                    ascending=[False] * len(metric_cols) + [False],
                    na_position="last",
                )
                deduped = sort_df.drop_duplicates(subset=key_cols, keep="first")
                deduped = deduped.sort_values("_original_order").drop(columns=["_original_order"])
                deduped = deduped[df.columns]
            else:
                deduped = df.drop_duplicates(subset=key_cols, keep="last")
            if len(deduped) != len(df):
                deduped.to_csv(path, index=False)
        except Exception:
            continue


def ensure_results_model_id(conn, cursor):
    """Ensure `results.model_id` exists (migration fallback)."""
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'results'
              AND COLUMN_NAME = 'model_id'
            """
        )
        has_col = cursor.fetchone()[0] > 0
        if not has_col:
            cursor.execute("ALTER TABLE results ADD COLUMN model_id INT NULL AFTER person_id")
            try:
                cursor.execute(
                    """
                    ALTER TABLE results
                    ADD CONSTRAINT fk_results_model_id
                    FOREIGN KEY (model_id) REFERENCES best_models_registry(id)
                    ON DELETE SET NULL
                    """
                )
            except Exception as e:
                # If FK already exists or add fails, continue without blocking runtime
                if is_mysql_connection_lost_error(e):
                    raise
                pass
            conn.commit()

        cleanup_duplicate_results_rows(conn, cursor)

        # Older databases made inference results unique by a coarse parameter
        # tuple. Several ranked registry rows can share those parameters, which
        # makes one model reuse or overwrite another model's stored prediction.
        # Prefer one latest result per person/model/image instead.
        try:
            cursor.execute("SHOW INDEX FROM results WHERE Key_name='unique_result_model_image'")
            has_model_unique = bool(cursor.fetchall())
            if not has_model_unique:
                cursor.execute(
                    """
                    ALTER TABLE results
                    ADD UNIQUE KEY unique_result_model_image (filename(100), person_id, model_id)
                    """
                )
                conn.commit()
            cursor.execute("SHOW INDEX FROM results WHERE Key_name='unique_analysis'")
            has_old_unique = bool(cursor.fetchall())
            if has_old_unique:
                cursor.execute("ALTER TABLE results DROP INDEX unique_analysis")
                conn.commit()
        except Exception as e:
            # Duplicate historical rows or limited DB privileges should not stop
            # the app; the model_id-first lookup below still fixes reads where
            # distinct rows already exist.
            _safe_rollback(conn)
            if is_mysql_connection_lost_error(e):
                raise
    except Exception as e:
        # Soft-fail: do not crash app if schema check fails
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def ensure_results_class_scores(conn, cursor):
    """Ensure `results.class_scores` exists for per-class inference probabilities."""
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'results'
              AND COLUMN_NAME = 'class_scores'
            """
        )
        has_col = cursor.fetchone()[0] > 0
        if not has_col:
            cursor.execute("ALTER TABLE results ADD COLUMN class_scores TEXT NULL AFTER confidence")
            conn.commit()
    except Exception as e:
        # Soft-fail: preserve app availability if migration cannot run.
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def ensure_results_head_config(conn, cursor):
    """Ensure result caching distinguishes learned classifier heads."""
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'results'
              AND COLUMN_NAME = 'head_config'
            """
        )
        has_col = cursor.fetchone()[0] > 0
        if not has_col:
            cursor.execute("ALTER TABLE results ADD COLUMN head_config VARCHAR(255) NOT NULL DEFAULT '' AFTER model_id")
            conn.commit()

        try:
            cursor.execute("UPDATE results SET head_config='' WHERE head_config IS NULL")
            conn.commit()
        except Exception:
            _safe_rollback(conn)

        try:
            cursor.execute("SHOW INDEX FROM results WHERE Key_name='unique_result_model_image'")
            has_old_unique = bool(cursor.fetchall())
            if has_old_unique:
                cursor.execute("ALTER TABLE results DROP INDEX unique_result_model_image")
                conn.commit()

            cursor.execute("SHOW INDEX FROM results WHERE Key_name='unique_result_model_head_image'")
            has_head_unique = bool(cursor.fetchall())
            if not has_head_unique:
                cursor.execute(
                    """
                    ALTER TABLE results
                    ADD UNIQUE KEY unique_result_model_head_image (filename(100), person_id, model_id, head_config)
                    """
                )
                conn.commit()
        except Exception as e:
            _safe_rollback(conn)
            if is_mysql_connection_lost_error(e):
                raise
    except Exception as e:
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def ensure_registry_metrics_columns(conn, cursor):
    """Ensure `best_models_registry` has optional metrics, split, and artifact pointer columns."""
    columns = {
        'valid_mcc': 'FLOAT NULL',
        'test_mcc': 'FLOAT NULL',
        'all_mcc': 'FLOAT NULL',
        'train_mcc': 'FLOAT NULL',
        'train_auc': 'FLOAT NULL',
        'valid_auc': 'FLOAT NULL',
        'test_auc': 'FLOAT NULL',
        'all_auc': 'FLOAT NULL',
        'valid_accuracy': 'FLOAT NULL',
        'ece': 'FLOAT NULL',
        'brier': 'FLOAT NULL',
        'best_head_config': 'VARCHAR(255) NULL',
        'best_head_name': 'VARCHAR(255) NULL',
        'best_head_family': 'VARCHAR(64) NULL',
        'best_head_n_aug': 'INT NULL',
        'train_datasets': 'TEXT NULL',
        'valid_dataset': 'VARCHAR(255) NULL',
        'test_dataset': 'VARCHAR(255) NULL',
        'split_config_key': 'VARCHAR(1024) NULL',
        'artifact_id': 'VARCHAR(32) NULL',
        'best_model_dir': 'TEXT NULL',
        'source_run_log_path': 'TEXT NULL',
    }
    for col, col_type in columns.items():
        try:
            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'best_models_registry'
                  AND COLUMN_NAME = '{col}'
                """
            )
            has_col = cursor.fetchone()[0] > 0
            if not has_col:
                cursor.execute(f"ALTER TABLE best_models_registry ADD COLUMN {col} {col_type}")
                conn.commit()
        except Exception as e:
            print(f"Migration error for {col}: {e}")
            _safe_rollback(conn)
            if is_mysql_connection_lost_error(e):
                raise

    try:
        legacy_train = LEGACY_TRAIN_DATASETS
        legacy_valid = LEGACY_VALID_DATASET
        legacy_key = split_config_key(legacy_train, legacy_valid, legacy_valid)
        cursor.execute(
            """
            UPDATE best_models_registry
            SET
                train_datasets = COALESCE(NULLIF(train_datasets, ''), %s),
                valid_dataset = COALESCE(NULLIF(valid_dataset, ''), %s),
                test_dataset = COALESCE(NULLIF(test_dataset, ''), COALESCE(NULLIF(valid_dataset, ''), %s)),
                split_config_key = COALESCE(NULLIF(split_config_key, ''), %s)
            WHERE train_datasets IS NULL OR train_datasets = ''
               OR valid_dataset IS NULL OR valid_dataset = ''
               OR test_dataset IS NULL OR test_dataset = ''
               OR split_config_key IS NULL OR split_config_key = ''
            """,
            (legacy_train, legacy_valid, legacy_valid, legacy_key),
        )
        conn.commit()
    except Exception as e:
        print(f"Migration error while backfilling legacy split config: {e}")
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def ensure_best_models_registry_nsize(conn, cursor):
    """Ensure `best_models_registry.nsize` exists and backfill from log_path if missing.
    
    Adds column as INT NULL and parses size from the standard best_models path.
    """
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'best_models_registry'
              AND COLUMN_NAME = 'nsize'
            """
        )
        has_col = cursor.fetchone()[0] > 0
    except Exception as e:
        # If schema introspection fails, do not block the app
        if is_mysql_connection_lost_error(e):
            raise
        return

    if not has_col:
        try:
            cursor.execute("ALTER TABLE best_models_registry ADD COLUMN nsize INT NULL")
            conn.commit()
        except Exception as e:
            _safe_rollback(conn)
            if is_mysql_connection_lost_error(e):
                raise
            # Could not add column; bail out quietly
            return

    # Backfill missing values from log_path
    try:
        cursor.execute(
            """
            SELECT id, log_path FROM best_models_registry
            WHERE (nsize IS NULL) AND log_path IS NOT NULL AND log_path <> ''
            """
        )
        rows = cursor.fetchall() or []
        for rid, lp in rows:
            try:
                params = extract_params_from_log_path(lp)
                ns = params.get('new_size')
                if ns is None:
                    continue
                val = int(float(ns))
                cursor.execute("UPDATE best_models_registry SET nsize=%s WHERE id=%s", (val, rid))
            except Exception:
                # Skip unparseable rows
                continue
        if rows:
            conn.commit()
    except Exception as e:
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def check_ds_exists(cursor, filename, args):
    """Check if a result already exists for given file and parameters.
    
    Returns:
        Row tuple (pred_label, confidence, log_path) if exists, None otherwise
    """
    head_config = _result_head_config(args)
    model_id = None
    try:
        raw_model_id = getattr(args, "model_id", None)
        if raw_model_id not in (None, "", "None"):
            model_id = int(raw_model_id)
    except Exception:
        model_id = None

    if model_id is not None:
        try:
            cursor.execute(
                '''
                SELECT pred_label, confidence, log_path FROM results
                WHERE filename=%s AND person_id=%s AND model_id=%s AND COALESCE(head_config, '')=%s
                ORDER BY timestamp DESC
                LIMIT 1
                ''',
                (filename, st.session_state.person_id, model_id, head_config),
            )
            row = cursor.fetchone()
            if row:
                return row
        except mysql.connector.errors.ProgrammingError as e:
            if "head_config" not in str(e):
                raise
        except Exception:
            pass

    fallback_query = ''' 
        SELECT pred_label, confidence, log_path FROM results
        WHERE filename=%s AND person_id=%s AND model_name=%s AND task=%s AND path=%s
              AND n_neighbors=%s AND nsize=%s AND fgsm=%s AND normalize=%s AND
              n_calibration=%s AND classif_loss=%s AND dloss=%s AND dist_fct=%s AND prototypes=%s
              AND npos=%s AND nneg=%s AND COALESCE(head_config, '')=%s
        ORDER BY timestamp DESC
        LIMIT 1
    '''
    fallback_values = (
        filename,
        st.session_state.person_id,
        args.model_name,
        args.task,
        args.path,
        str(args.n_neighbors),
        str(args.new_size),
        str(args.fgsm),
        args.normalize,
        str(args.n_calibration),
        args.classif_loss,
        args.dloss,
        args.dist_fct,
        args.prototypes_to_use,
        str(args.n_positives),
        str(args.n_negatives),
        head_config,
    )
    try:
        cursor.execute(fallback_query, fallback_values)
    except mysql.connector.errors.ProgrammingError as e:
        if "head_config" not in str(e):
            raise
        cursor.execute(
            fallback_query.replace(" AND COALESCE(head_config, '')=%s", ""),
            fallback_values[:-1],
        )

    row = cursor.fetchone()
    return row


def list_image_results(cursor, person_id, filename):
    """Return all stored analyses for this image and person."""
    cursor.execute('''
        SELECT model_name, task, pred_label, confidence, log_path, timestamp,
               nsize, fgsm, normalize, n_calibration, classif_loss, dloss, dist_fct,
               prototypes, npos, nneg, n_neighbors, model_id
        FROM results
        WHERE filename=%s AND person_id=%s
        ORDER BY timestamp DESC
    ''', (filename, person_id))
    return cursor.fetchall()


def fetch_model_by_log_path(cursor, log_path: str):
    """Fetch model registry entry by log path."""
    cursor.execute(
        """
        SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
               n_calibration, accuracy, mcc, normalize, n_neighbors, log_path
        FROM best_models_registry
        WHERE log_path=%s
        LIMIT 1
        """,
        (log_path,),
    )
    return cursor.fetchone()


def _result_head_config(args) -> str:
    for attr in ("best_classifier_config", "classification_head_config", "classifier_head_config", "head_config"):
        value = getattr(args, attr, None)
        if value is not None and str(value).strip() not in {"", "None", "none", "nan", "NaN", "—"}:
            return str(value).strip()
    return ""


def resolve_model_id(cursor, args, log_path: str):
    """Best-effort lookup of best_models_registry.id for the current run.
    
    Tries in order:
    1. Explicit model_id from args
    2. Lookup by log_path (with /queries suffix handling)
    3. Lookup by all parameter values
    
    Returns:
        int: Model ID if found, None otherwise
    """
    try:
        explicit = getattr(args, "model_id", None)
        if explicit not in (None, "", "None"):
            return int(explicit)
    except Exception:
        pass

    if log_path:
        candidates = [log_path]
        lp_trim = str(log_path).rstrip("/")
        if lp_trim.endswith("/queries"):
            candidates.append(lp_trim[: -len("/queries")])
        for lp in _unique_preserve_order(candidates):
            try:
                cursor.execute("SELECT id FROM best_models_registry WHERE log_path=%s LIMIT 1", (lp,))
                row = cursor.fetchone()
                if row:
                    return int(row[0])
            except Exception:
                pass

    try:
        cursor.execute(
            """
            SELECT id FROM best_models_registry
            WHERE model_name=%s AND nsize=%s AND fgsm=%s AND prototypes=%s AND npos=%s AND nneg=%s AND dloss=%s
                  AND dist_fct=%s AND classif_loss=%s AND n_calibration=%s AND normalize=%s AND n_neighbors=%s
            ORDER BY id ASC
            LIMIT 1
            """,
            (
                getattr(args, "model_name", None),
                str(getattr(args, "new_size", None)),
                str(getattr(args, "fgsm", None)),
                getattr(args, "prototypes_to_use", None),
                str(getattr(args, "n_positives", None)),
                str(getattr(args, "n_negatives", None)),
                getattr(args, "dloss", None),
                getattr(args, "dist_fct", None),
                getattr(args, "classif_loss", None),
                str(getattr(args, "n_calibration", None)),
                getattr(args, "normalize", None),
                str(getattr(args, "n_neighbors", None)),
            ),
        )
        row = cursor.fetchone()
        if row:
            return int(row[0])
    except Exception:
        return None

    return None


def insert_score(cursor, conn, filename, args, pred_label, confidence, log_path, class_scores=None):
    """Upsert the result row (respecting unique constraint) and upsert usage summary.
    
    Args:
        cursor: Database cursor
        conn: Database connection
        filename: Image filename
        args: Model arguments namespace
        pred_label: Predicted label
        confidence: Prediction confidence
        log_path: Model log path
    """
    model_id = resolve_model_id(cursor, args, log_path)
    filename = str(filename)
    pred_label = str(pred_label)
    confidence = float(confidence)
    log_path = str(log_path) if log_path is not None else None
    model_id = int(model_id) if model_id is not None else None
    head_config = _result_head_config(args)
    class_scores_json = None
    if isinstance(class_scores, dict):
        try:
            normalized_scores = {str(k): float(v) for k, v in class_scores.items()}
            class_scores_json = json.dumps(normalized_scores, ensure_ascii=True)
        except Exception:
            class_scores_json = None

    query = '''
        INSERT INTO results (
            filename, model_name, task, path, n_neighbors, nsize, fgsm, normalize, n_calibration, classif_loss,
            dloss, dist_fct, prototypes, npos, nneg, pred_label, confidence, class_scores, log_path, person_id, model_id, head_config
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            pred_label = VALUES(pred_label),
            confidence = VALUES(confidence),
            class_scores = VALUES(class_scores),
            log_path = VALUES(log_path),
            person_id = VALUES(person_id),
            model_id = VALUES(model_id),
            head_config = VALUES(head_config),
            timestamp = CURRENT_TIMESTAMP
    '''
    values = (
        filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
        args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
        str(args.n_positives), str(args.n_negatives), pred_label, confidence, class_scores_json, log_path,
        st.session_state.person_id, model_id, head_config
    )
    try:
        cursor.execute(query, _mysql_values(values))
    except mysql.connector.errors.ProgrammingError as e:
        if "head_config" in str(e):
            query_no_head = '''
                INSERT INTO results (
                    filename, model_name, task, path, n_neighbors, nsize, fgsm, normalize, n_calibration, classif_loss,
                    dloss, dist_fct, prototypes, npos, nneg, pred_label, confidence, class_scores, log_path, person_id, model_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    pred_label = VALUES(pred_label),
                    confidence = VALUES(confidence),
                    class_scores = VALUES(class_scores),
                    log_path = VALUES(log_path),
                    person_id = VALUES(person_id),
                    model_id = VALUES(model_id),
                    timestamp = CURRENT_TIMESTAMP
            '''
            values_no_head = values[:-1]
            cursor.execute(query_no_head, _mysql_values(values_no_head))
            conn.commit()
            return
        if "class_scores" not in str(e):
            raise
        query_legacy = '''
            INSERT INTO results (
                filename, model_name, task, path, n_neighbors, nsize, fgsm, normalize, n_calibration, classif_loss,
                dloss, dist_fct, prototypes, npos, nneg, pred_label, confidence, log_path, person_id, model_id, head_config
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                pred_label = VALUES(pred_label),
                confidence = VALUES(confidence),
                log_path = VALUES(log_path),
                person_id = VALUES(person_id),
                model_id = VALUES(model_id),
                head_config = VALUES(head_config),
                timestamp = CURRENT_TIMESTAMP
        '''
        legacy_values = (
            filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
            args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
            str(args.n_positives), str(args.n_negatives), pred_label, confidence, log_path,
            st.session_state.person_id, model_id, head_config
        )
        cursor.execute(query_legacy, _mysql_values(legacy_values))
    except mysql.connector.errors.IntegrityError:
        # If the unique key blocks insertion, perform an explicit update to refresh timestamp and values
        update_q = '''
            UPDATE results
            SET pred_label=%s, confidence=%s, class_scores=%s, log_path=%s, person_id=%s, model_id=%s, head_config=%s, timestamp=CURRENT_TIMESTAMP
            WHERE filename=%s AND model_name=%s AND task=%s AND path=%s AND n_neighbors=%s AND nsize=%s AND fgsm=%s
                  AND normalize=%s AND n_calibration=%s AND classif_loss=%s AND dloss=%s AND dist_fct=%s AND prototypes=%s
                  AND npos=%s AND nneg=%s AND COALESCE(head_config, '')=%s
        '''
        update_vals = (
            pred_label, confidence, class_scores_json, log_path, st.session_state.person_id, model_id, head_config,
            filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
            args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
            str(args.n_positives), str(args.n_negatives), head_config
        )
        try:
            cursor.execute(update_q, _mysql_values(update_vals))
        except mysql.connector.errors.ProgrammingError as e:
            if "head_config" in str(e):
                update_q_no_head = '''
                    UPDATE results
                    SET pred_label=%s, confidence=%s, class_scores=%s, log_path=%s, person_id=%s, model_id=%s, timestamp=CURRENT_TIMESTAMP
                    WHERE filename=%s AND model_name=%s AND task=%s AND path=%s AND n_neighbors=%s AND nsize=%s AND fgsm=%s
                          AND normalize=%s AND n_calibration=%s AND classif_loss=%s AND dloss=%s AND dist_fct=%s AND prototypes=%s
                          AND npos=%s AND nneg=%s
                '''
                update_vals_no_head = (
                    pred_label, confidence, class_scores_json, log_path, st.session_state.person_id, model_id,
                    filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
                    args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
                    str(args.n_positives), str(args.n_negatives)
                )
                cursor.execute(update_q_no_head, _mysql_values(update_vals_no_head))
                return
            if "class_scores" not in str(e):
                raise
            update_q_legacy = '''
                UPDATE results
                SET pred_label=%s, confidence=%s, log_path=%s, person_id=%s, model_id=%s, head_config=%s, timestamp=CURRENT_TIMESTAMP
                WHERE filename=%s AND model_name=%s AND task=%s AND path=%s AND n_neighbors=%s AND nsize=%s AND fgsm=%s
                      AND normalize=%s AND n_calibration=%s AND classif_loss=%s AND dloss=%s AND dist_fct=%s AND prototypes=%s
                      AND npos=%s AND nneg=%s AND COALESCE(head_config, '')=%s
            '''
            update_vals_legacy = (
                pred_label, confidence, log_path, st.session_state.person_id, model_id, head_config,
                filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
                args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
                str(args.n_positives), str(args.n_negatives), head_config
            )
            cursor.execute(update_q_legacy, _mysql_values(update_vals_legacy))

    summary_query = '''
        INSERT INTO model_usage_summary (
            model_name, task, path, nsize, fgsm, normalize, n_calibration, classif_loss,
            dloss, dist_fct, prototypes, npos, nneg, n_neighbors, num_samples_analyzed
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            num_samples_analyzed = num_samples_analyzed + 1,
            last_used = CURRENT_TIMESTAMP
    '''
    summary_values = (
        args.model_name, args.task, args.path, str(args.new_size), str(args.fgsm),
        args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
        str(args.n_positives), str(args.n_negatives), str(args.n_neighbors), 1
    )
    cursor.execute(summary_query, _mysql_values(summary_values))
    conn.commit()

def ensure_production_model_table(conn, cursor):
    """Ensure production_model table exists for storing the global production model."""
    def _ensure_column(column_name: str, ddl: str):
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'production_model'
              AND COLUMN_NAME = %s
            """,
            (column_name,),
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(f"ALTER TABLE production_model ADD COLUMN {ddl}")
            conn.commit()

    def _drop_model_id_foreign_keys():
        """Drop any FK constraints attached to production_model.model_id."""
        try:
            cursor.execute(
                """
                SELECT CONSTRAINT_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'production_model'
                  AND COLUMN_NAME = 'model_id'
                  AND REFERENCED_TABLE_NAME IS NOT NULL
                """
            )
            rows = cursor.fetchall() or []
            for row in rows:
                constraint_name = row[0]
                if constraint_name:
                    cursor.execute(f"ALTER TABLE production_model DROP FOREIGN KEY `{constraint_name}`")
                    conn.commit()
        except Exception as e:
            _safe_rollback(conn)
            if is_mysql_connection_lost_error(e):
                raise
            print(f"[database] Warning: could not drop production_model.model_id FK: {e}")

    def _ensure_model_id_is_text():
        """
        Older databases created production_model.model_id as INT with a FK to
        best_models_registry.id. New production IDs can be strings like
        PROD_otitis_four_class_..., so the column must be text.
        """
        cursor.execute(
            """
            SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'production_model'
              AND COLUMN_NAME = 'model_id'
            """
        )
        row = cursor.fetchone()
        if not row:
            cursor.execute("ALTER TABLE production_model ADD COLUMN model_id VARCHAR(512) NOT NULL AFTER label_scheme")
            conn.commit()
            return

        dtype = str(row[0] or "").lower()
        max_len = row[1]
        is_nullable = str(row[2] or "").upper()
        needs_text = dtype not in {"varchar", "char", "text", "mediumtext", "longtext"}
        needs_longer = dtype in {"varchar", "char"} and (max_len is None or int(max_len) < 512)
        needs_not_null = is_nullable != "NO"

        if needs_text or needs_longer or needs_not_null:
            _drop_model_id_foreign_keys()
            cursor.execute(
                """
                ALTER TABLE production_model
                MODIFY COLUMN model_id VARCHAR(512) NOT NULL
                """
            )
            conn.commit()
            print("[database] Ensured production_model.model_id is VARCHAR(512)")

    try:
        # First check if table exists
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'production_model'
            """
        )
        table_exists = cursor.fetchone()[0] > 0

        if not table_exists:
            cursor.execute(
                """
                CREATE TABLE production_model (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    label_scheme VARCHAR(50) NOT NULL DEFAULT 'four_class',
                    model_id VARCHAR(512) NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    model_number VARCHAR(50),
                    log_path TEXT,
                    head_config VARCHAR(255),
                    head_name VARCHAR(255),
                    head_family VARCHAR(64),
                    head_n_aug INT,
                    prototype_strategy VARCHAR(64),
                    prototype_components INT,
                    nsize INT NULL,
                    fgsm VARCHAR(32) NULL,
                    prototypes VARCHAR(64) NULL,
                    npos VARCHAR(32) NULL,
                    nneg VARCHAR(32) NULL,
                    dloss VARCHAR(128) NULL,
                    dist_fct VARCHAR(64) NULL,
                    classif_loss VARCHAR(128) NULL,
                    n_calibration VARCHAR(32) NULL,
                    normalize VARCHAR(64) NULL,
                    n_neighbors VARCHAR(32) NULL,
                    dataset TEXT NULL,
                    train_datasets TEXT NULL,
                    valid_dataset TEXT NULL,
                    test_dataset TEXT NULL,
                    split_config_key TEXT NULL,
                    set_by VARCHAR(255),
                    set_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
            print("Created production_model table")
        else:
            _ensure_column("label_scheme", "label_scheme VARCHAR(50) NOT NULL DEFAULT 'binary' AFTER id")
            _ensure_column("head_config", "head_config VARCHAR(255) NULL AFTER log_path")
            _ensure_column("head_name", "head_name VARCHAR(255) NULL AFTER head_config")
            _ensure_column("head_family", "head_family VARCHAR(64) NULL AFTER head_name")
            _ensure_column("head_n_aug", "head_n_aug INT NULL AFTER head_family")
            _ensure_column("prototype_strategy", "prototype_strategy VARCHAR(64) NULL AFTER head_n_aug")
            _ensure_column("prototype_components", "prototype_components INT NULL AFTER prototype_strategy")
            _ensure_column("nsize", "nsize INT NULL AFTER prototype_components")
            _ensure_column("fgsm", "fgsm VARCHAR(32) NULL AFTER nsize")
            _ensure_column("prototypes", "prototypes VARCHAR(64) NULL AFTER fgsm")
            _ensure_column("npos", "npos VARCHAR(32) NULL AFTER prototypes")
            _ensure_column("nneg", "nneg VARCHAR(32) NULL AFTER npos")
            _ensure_column("dloss", "dloss VARCHAR(128) NULL AFTER nneg")
            _ensure_column("dist_fct", "dist_fct VARCHAR(64) NULL AFTER dloss")
            _ensure_column("classif_loss", "classif_loss VARCHAR(128) NULL AFTER dist_fct")
            _ensure_column("n_calibration", "n_calibration VARCHAR(32) NULL AFTER classif_loss")
            _ensure_column("normalize", "normalize VARCHAR(64) NULL AFTER n_calibration")
            _ensure_column("n_neighbors", "n_neighbors VARCHAR(32) NULL AFTER normalize")
            _ensure_column("dataset", "dataset TEXT NULL AFTER n_neighbors")
            _ensure_column("train_datasets", "train_datasets TEXT NULL AFTER dataset")
            _ensure_column("valid_dataset", "valid_dataset TEXT NULL AFTER train_datasets")
            _ensure_column("test_dataset", "test_dataset TEXT NULL AFTER valid_dataset")
            _ensure_column("split_config_key", "split_config_key TEXT NULL AFTER test_dataset")
            _ensure_model_id_is_text()
    except Exception as e:
        print(f"Error creating production_model table: {e}")
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def ensure_learned_embedding_heads_table(conn, cursor):
    """Ensure learned_embedding_heads table exists for storing classification head results."""
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'learned_embedding_heads'
            """
        )
        table_exists = cursor.fetchone()[0] > 0

        if not table_exists:
            cursor.execute(
                """
                CREATE TABLE learned_embedding_heads (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_id INT NULL,
                    log_path TEXT NULL,
                    config VARCHAR(255) NOT NULL,
                    head_family VARCHAR(64) NOT NULL,
                    n_aug INT DEFAULT 0,
                    head_cache_version INT DEFAULT 2,
                    train_mcc FLOAT NULL,
                    valid_mcc FLOAT NULL,
                    test_mcc FLOAT NULL,
                    all_mcc FLOAT NULL,
                    train_auc FLOAT NULL,
                    valid_auc FLOAT NULL,
                    test_auc FLOAT NULL,
                    all_auc FLOAT NULL,
                    valid_accuracy FLOAT NULL,
                    train_datasets TEXT NULL,
                    valid_dataset VARCHAR(255) NULL,
                    test_dataset VARCHAR(255) NULL,
                    split_config_key VARCHAR(255) NULL,
                    details TEXT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_model_id (model_id),
                    INDEX idx_config (config),
                    INDEX idx_n_aug (n_aug),
                    INDEX idx_split_config (split_config_key)
                )
                """
            )
            conn.commit()
            print("Created learned_embedding_heads table")
        else:
            # Ensure all columns exist (for future schema updates)
            columns_to_ensure = {
                'model_id': 'INT NULL',
                'log_path': 'TEXT NULL',
                'config': 'VARCHAR(255) NOT NULL',
                'head_family': 'VARCHAR(64) NOT NULL',
                'n_aug': 'INT DEFAULT 0',
                'head_cache_version': 'INT DEFAULT 2',
                'train_mcc': 'FLOAT NULL',
                'valid_mcc': 'FLOAT NULL',
                'test_mcc': 'FLOAT NULL',
                'all_mcc': 'FLOAT NULL',
                'train_auc': 'FLOAT NULL',
                'valid_auc': 'FLOAT NULL',
                'test_auc': 'FLOAT NULL',
                'all_auc': 'FLOAT NULL',
                'valid_accuracy': 'FLOAT NULL',
                'train_datasets': 'TEXT NULL',
                'valid_dataset': 'VARCHAR(255) NULL',
                'test_dataset': 'VARCHAR(255) NULL',
                'split_config_key': 'VARCHAR(255) NULL',
                'details': 'TEXT NULL',
            }
            for col_name, col_type in columns_to_ensure.items():
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'learned_embedding_heads'
                      AND COLUMN_NAME = %s
                    """,
                    (col_name,),
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(f"ALTER TABLE learned_embedding_heads ADD COLUMN {col_name} {col_type}")
                    conn.commit()
                else:
                    # Check if column needs to be altered (e.g., split_config_key length)
                    cursor.execute(
                        """
                        SELECT COLUMN_TYPE, CHARACTER_MAXIMUM_LENGTH
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = DATABASE()
                          AND TABLE_NAME = 'learned_embedding_heads'
                          AND COLUMN_NAME = %s
                        """,
                        (col_name,),
                    )
                    row = cursor.fetchone()
                    if row:
                        current_type = row[0]
                        max_len = row[1]
                        # Check if split_config_key is too long
                        if col_name == 'split_config_key' and max_len and max_len > 255:
                            try:
                                # Drop index first if it exists
                                cursor.execute("SHOW INDEX FROM learned_embedding_heads WHERE Key_name = 'idx_split_config'")
                                if cursor.fetchone():
                                    cursor.execute("ALTER TABLE learned_embedding_heads DROP INDEX idx_split_config")
                                # Alter column
                                cursor.execute("ALTER TABLE learned_embedding_heads MODIFY COLUMN split_config_key VARCHAR(255) NULL")
                                # Re-create index
                                cursor.execute("ALTER TABLE learned_embedding_heads ADD INDEX idx_split_config (split_config_key)")
                                conn.commit()
                                print(f"[database] Altered split_config_key from VARCHAR({max_len}) to VARCHAR(255)")
                            except Exception as alter_error:
                                print(f"[database] Warning: could not alter split_config_key: {alter_error}")
    except Exception as e:
        print(f"Error ensuring learned_embedding_heads table: {e}")
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise


def save_learned_embedding_heads(cursor, conn, model_id: int | None, log_path: str | None, heads: list[dict]):
    """Save learned embedding classification heads to database.
    
    Args:
        cursor: Database cursor
        conn: Database connection
        model_id: ID from best_models_registry (if available)
        log_path: Model log path
        heads: List of head dictionaries from enumerate_classification_heads
    """
    if not heads:
        return
    
    # Find the best head overall
    best_head = None
    best_mcc = -float('inf')
    for head in heads:
        valid_mcc = head.get('valid_mcc') or head.get('mcc')
        if valid_mcc is not None and float(valid_mcc) > best_mcc:
            best_mcc = float(valid_mcc)
            best_head = head
    
    for head in heads:
        try:
            config = head.get('config', '')
            head_family = head.get('family', _infer_head_family_from_config(config))
            n_aug = head.get('n_aug', 0)
            
            # Build split_config_key from datasets if available
            train_datasets = head.get('train_datasets')
            valid_dataset = head.get('valid_dataset')
            test_dataset = head.get('test_dataset')
            split_config_key = None
            if train_datasets or valid_dataset or test_dataset:
                from otitenet.app.utils import split_config_key
                split_config_key = split_config_key(train_datasets or '', valid_dataset or '', test_dataset or '')
            
            query = """
                INSERT INTO learned_embedding_heads (
                    model_id, log_path, config, head_family, n_aug, head_cache_version,
                    train_mcc, valid_mcc, test_mcc, all_mcc,
                    train_auc, valid_auc, test_auc, all_auc,
                    valid_accuracy, train_datasets, valid_dataset, test_dataset,
                    split_config_key, details
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    valid_mcc = VALUES(valid_mcc),
                    test_mcc = VALUES(test_mcc),
                    all_mcc = VALUES(all_mcc),
                    train_auc = VALUES(train_auc),
                    valid_auc = VALUES(valid_auc),
                    test_auc = VALUES(test_auc),
                    all_auc = VALUES(all_auc),
                    valid_accuracy = VALUES(valid_accuracy),
                    details = VALUES(details),
                    updated_at = CURRENT_TIMESTAMP
            """
            values = (
                model_id,
                log_path,
                config,
                head_family,
                n_aug,
                head.get('head_cache_version', 2),
                head.get('train_mcc'),
                head.get('valid_mcc'),
                head.get('test_mcc'),
                head.get('all_mcc'),
                head.get('train_auc'),
                head.get('valid_auc'),
                head.get('test_auc'),
                head.get('all_auc'),
                head.get('valid_accuracy'),
                train_datasets,
                valid_dataset,
                test_dataset,
                split_config_key,
                head.get('details', ''),
            )
            cursor.execute(query, _mysql_values(values))
        except Exception as e:
            print(f"Error saving learned embedding head {head.get('config')}: {e}")
            continue
    
    conn.commit()
    
    # Update best_models_registry with best head information
    if best_head and (model_id or log_path):
        try:
            from otitenet.app.utils import format_classifier_config
            
            update_query = """
                UPDATE best_models_registry
                SET best_head_config = %s,
                    best_head_name = %s,
                    best_head_family = %s,
                    best_head_n_aug = %s
            """
            params = [
                best_head.get('config'),
                format_classifier_config(best_head.get('config')),
                best_head.get('family', _infer_head_family_from_config(best_head.get('config', ''))),
                best_head.get('n_aug', 0),
            ]
            
            if model_id:
                update_query += " WHERE id = %s"
                params.append(model_id)
            elif log_path:
                update_query += " WHERE log_path = %s"
                params.append(log_path)
            
            cursor.execute(update_query, _mysql_values(tuple(params)))
            conn.commit()
            print(f"[database] Updated best_models_registry with best head: {best_head.get('config')}")
        except Exception as e:
            print(f"[database] Warning: could not update best_models_registry with best head: {e}")


def load_learned_embedding_heads(cursor, model_id: int | None = None, log_path: str | None = None) -> list[dict]:
    """Load learned embedding classification heads from database.
    
    Args:
        cursor: Database cursor
        model_id: Optional model ID to filter by
        log_path: Optional log path to filter by
    
    Returns:
        List of head dictionaries compatible with enumerate_classification_heads format
    """
    try:
        query = """
            SELECT 
                config, head_family, n_aug, head_cache_version,
                train_mcc, valid_mcc, test_mcc, all_mcc,
                train_auc, valid_auc, test_auc, all_auc,
                valid_accuracy, train_datasets, valid_dataset, test_dataset,
                split_config_key, details, model_id, log_path
            FROM learned_embedding_heads
        """
        params = []
        
        if model_id is not None:
            query += " WHERE model_id = %s"
            params.append(model_id)
        elif log_path is not None:
            query += " WHERE log_path = %s"
            params.append(log_path)
        
        query += " ORDER BY valid_mcc DESC, created_at DESC"
        
        if params:
            cursor.execute(query, tuple(params))
        else:
            cursor.execute(query)
        
        rows = cursor.fetchall()
        
        heads = []
        for row in rows:
            head = {
                'config': row[0],
                'family': row[1],
                'n_aug': row[2],
                'head_cache_version': row[3],
                'train_mcc': row[4],
                'valid_mcc': row[5],
                'test_mcc': row[6],
                'all_mcc': row[7],
                'train_auc': row[8],
                'valid_auc': row[9],
                'test_auc': row[10],
                'all_auc': row[11],
                'valid_accuracy': row[12],
                'train_datasets': row[13],
                'valid_dataset': row[14],
                'test_dataset': row[15],
                'split_config_key': row[16],
                'details': row[17],
                'model_id': row[18],
                'log_path': row[19],
            }
            # Add label for compatibility
            from otitenet.app.utils import format_classifier_config
            head['label'] = format_classifier_config(head['config'])
            head['mcc'] = head['valid_mcc']  # For backward compatibility
            heads.append(head)
        
        return heads
    except Exception as e:
        print(f"Error loading learned embedding heads: {e}")
        return []


def _infer_head_family_from_config(config: str) -> str:
    """Infer head family from config string."""
    if not config:
        return 'unknown'
    
    config_lower = str(config).lower()
    
    if config_lower.isdigit():
        return 'knn'
    if 'knn' in config_lower or 'neighbor' in config_lower:
        return 'knn'
    if 'protot' in config_lower:
        if 'gmm' in config_lower:
            return 'gmm'
        if 'kmeans' in config_lower or 'k-mean' in config_lower:
            return 'kmeans'
        if 'mean' in config_lower:
            return 'mean'
        return 'prototypes'
    if 'baseline' in config_lower:
        # Extract baseline type
        for baseline in ['logreg', 'ridge', 'naive_bayes', 'svc', 'random_forest', 'gradient_boosting', 'decision_tree', 'lda', 'qda']:
            if baseline in config_lower:
                return baseline
        return 'baseline'
    
    return 'unknown'


def get_production_model(cursor, label_scheme: str | None = None):
    """Get the current production model from database."""
    label_scheme = label_scheme or "four_class"
    try:
        cursor.execute(
            """
            SELECT pm.model_id, pm.model_name, pm.model_number, pm.log_path, pm.set_by, pm.set_at,
                   pm.label_scheme,
                   pm.head_config, pm.head_name, pm.head_family, pm.head_n_aug,
                   pm.prototype_strategy, pm.prototype_components,
                   bmr.model_name, bmr.nsize, bmr.fgsm, bmr.prototypes, bmr.npos, bmr.nneg,
                   bmr.dloss, bmr.dist_fct, bmr.classif_loss, bmr.n_calibration, bmr.accuracy,
                   bmr.mcc, bmr.normalize, bmr.n_neighbors, bmr.prototype_strategy, bmr.prototype_components,
                   bmr.train_datasets, bmr.valid_dataset, bmr.test_dataset, bmr.split_config_key,
                   bmr.best_model_dir, bmr.id,
                   pm.nsize, pm.fgsm, pm.prototypes, pm.npos, pm.nneg, pm.dloss, pm.dist_fct,
                   pm.classif_loss, pm.n_calibration, pm.normalize, pm.n_neighbors, pm.dataset,
                   pm.train_datasets, pm.valid_dataset, pm.test_dataset, pm.split_config_key
            FROM production_model pm
            LEFT JOIN best_models_registry bmr
              ON pm.log_path = bmr.log_path
              OR pm.log_path = bmr.source_run_log_path
              OR CAST(pm.model_id AS CHAR) = CAST(bmr.id AS CHAR)
            WHERE pm.label_scheme=%s
            ORDER BY
              CASE
                WHEN pm.log_path = bmr.log_path THEN 0
                WHEN pm.log_path = bmr.source_run_log_path THEN 1
                WHEN CAST(pm.model_id AS CHAR) = CAST(bmr.id AS CHAR) THEN 2
                ELSE 3
              END,
              pm.set_at DESC
            LIMIT 1
            """,
            (label_scheme,),
        )
        row = cursor.fetchone()
        if row:
            if row[3] and not os.path.exists(os.path.join(str(row[3]), "model.pth")):
                print(f"Production model artifact missing, ignoring stale production row: {row[3]}")
                return None
            artifact_params = extract_params_from_log_path(row[33] or row[3] or "")
            def _prefer(saved_idx, registry_idx=None, fallback=None):
                value = row[saved_idx] if saved_idx is not None else None
                if value is not None and str(value).strip() not in {"", "None", "none", "nan", "NaN", "—"}:
                    return value
                if registry_idx is not None:
                    value = row[registry_idx]
                    if value is not None and str(value).strip() not in {"", "None", "none", "nan", "NaN", "—"}:
                        return value
                return fallback
            return {
                'model_id': row[0],
                'model_name': row[1],
                'model_number': row[2],
                'log_path': row[3],
                'set_by': row[4],
                'set_at': row[5],
                'label_scheme': row[6],
                'Head Config': row[7],
                'head_config': row[7],
                'classification_head_config': row[7],
                'best_classifier_config': row[7],
                'Head': row[8],
                'head_name': row[8],
                'learned_classifier_label': row[8],
                'head_family': row[9],
                'head_n_aug': row[10],
                'n_aug': row[10],
                'prototype_strategy': row[11] if row[11] is not None else row[27],
                'prototype_components': row[12] if row[12] is not None else row[28],
                'Model Name': row[13] if row[13] is not None else row[1],
                'NSize': _prefer(35, 14),
                'FGSM': _prefer(36, 15),
                'Prototypes': _prefer(37, 16),
                'NPos': _prefer(38, 17),
                'NNeg': _prefer(39, 18),
                'DLoss': _prefer(40, 19),
                'Dist_Fct': _prefer(41, 20),
                'Classif_Loss': _prefer(42, 21),
                'N_Calibration': _prefer(43, 22),
                'Accuracy': row[23],
                'MCC': row[24],
                'Normalize': _prefer(44, 25),
                'N_Neighbors': _prefer(45, 26),
                'train_datasets': _prefer(47, 29),
                'valid_dataset': _prefer(48, 30),
                'test_dataset': _prefer(49, 31),
                'split_config_key': _prefer(50, 32),
                'Best Model Dir': row[33],
                'Registry ID': row[34],
                'DB Model ID': row[34],
                'Dataset': _prefer(46, None, artifact_params.get('Dataset')),
            }
        return None
    except Exception as e:
        print(f"Error getting production model: {e}")
        return None


def set_production_model(cursor, conn, model_dict, set_by_email):
    """Set the production model in database."""
    def _is_blank(value) -> bool:
        if value is None:
            return True
        try:
            if isinstance(value, float) and np.isnan(value):
                return True
        except Exception:
            pass
        return str(value).strip().lower() in {"", "none", "nan", "<na>", "nat"}

    def _production_model_id(model_info) -> str:
        for key in ("model_id", "Registry ID", "Model ID", "Artifact ID", "exp ID", "id"):
            value = model_info.get(key)
            if not _is_blank(value):
                return str(value).strip()
        log_path = str(model_info.get("Log Path") or model_info.get("log_path") or "").strip()
        if log_path:
            return f"log:{log_path}"
        model_name = str(model_info.get("Model Name") or model_info.get("model_name") or "model").strip()
        model_number = str(model_info.get("model_number") or model_info.get("#") or "").strip()
        return f"model:{model_name}:{model_number or 'unranked'}"

    def _first_present(*keys):
        for key in keys:
            value = _mysql_value(model_dict.get(key))
            if not _is_blank(value):
                return value
        return None

    def _optional_int(*keys):
        value = _first_present(*keys)
        if _is_blank(value):
            return None
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None

    try:
        label_scheme = model_dict.get("label_scheme") or "four_class"
        production_model_id = _production_model_id(model_dict)
        cursor.execute("DELETE FROM production_model WHERE label_scheme=%s", (label_scheme,))

        # Insert new production model
        cursor.execute(
            """
            INSERT INTO production_model (
                label_scheme, model_id, model_name, model_number, log_path,
                head_config, head_name, head_family, head_n_aug,
                prototype_strategy, prototype_components,
                nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
                n_calibration, normalize, n_neighbors, dataset, train_datasets, valid_dataset,
                test_dataset, split_config_key, set_by
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                label_scheme,
                production_model_id,
                _first_present('Model Name', 'model_name'),
                _first_present('model_number', '#'),
                _first_present('Log Path', 'log_path'),
                _first_present('Best Head Config', 'Head Config', 'head_config', 'classification_head_config', 'best_classifier_config'),
                _first_present('Head', 'head_name', 'learned_classifier_label', 'Best Classification Head'),
                _first_present('head_family', 'classification_head_family'),
                _optional_int('head_n_aug', 'n_aug', 'N Aug', 'Head N Aug', 'Best Head N Aug'),
                _first_present('prototype_strategy', 'Proto_Strat'),
                _optional_int('prototype_components', 'Proto_Comp'),
                _first_present('NSize', 'new_size'),
                _first_present('FGSM', 'fgsm'),
                _first_present('Prototypes', 'prototypes_to_use', 'prototypes'),
                _first_present('NPos', 'n_positives'),
                _first_present('NNeg', 'n_negatives'),
                _first_present('DLoss', 'dloss'),
                _first_present('Dist_Fct', 'dist_fct', 'dist_metric', 'Distance'),
                _first_present('Classif_Loss', 'classif_loss'),
                _first_present('N_Calibration', 'n_calibration'),
                _first_present('Normalize', 'normalize'),
                _first_present('N_Neighbors', 'n_neighbors'),
                _first_present('Dataset', 'path', 'Artifact Dataset', 'Combo Dataset'),
                _first_present('train_datasets', 'Train Datasets'),
                _first_present('valid_dataset', 'Valid Dataset'),
                _first_present('test_dataset', 'Test Dataset'),
                _first_present('split_config_key'),
                set_by_email
            )
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error setting production model: {e}")
        _safe_rollback(conn)
        return False

def ensure_people_user_email(conn, cursor):
    """Ensure people.user_email exists so people can be scoped per logged-in user."""
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'people'
              AND COLUMN_NAME = 'user_email'
            """
        )
        has_col = cursor.fetchone()[0] > 0

        if not has_col:
            cursor.execute(
                """
                ALTER TABLE people
                ADD COLUMN user_email VARCHAR(255) NULL AFTER id
                """
            )
            conn.commit()

        try:
            cursor.execute(
                """
                CREATE INDEX idx_people_user_email
                ON people(user_email)
                """
            )
            conn.commit()
        except Exception as e:
            # index probably already exists
            _safe_rollback(conn)
            if is_mysql_connection_lost_error(e):
                raise

    except Exception as e:
        print(f"Could not ensure people.user_email column: {e}")
        _safe_rollback(conn)
        if is_mysql_connection_lost_error(e):
            raise
