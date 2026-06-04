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

@st.cache_resource
def get_db_connection():
    """Create and cache a MySQL database connection."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="y_user",
            password="password",
            database="results_db",
            buffered=True,
            autocommit=True,
            # Avoid sporadic C-extension parsing issues seen as
            # "bytearray index out of range" on some environments.
            use_pure=True,
        )
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
                conn = mysql.connector.connect(
                    host="localhost",
                    user="y_user",
                    password="password",
                    database="results_db",
                    buffered=True,
                    autocommit=True,
                    use_pure=True,
                )
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
            except Exception:
                # If FK already exists or add fails, continue without blocking runtime
                pass
            conn.commit()

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
        except Exception:
            # Duplicate historical rows or limited DB privileges should not stop
            # the app; the model_id-first lookup below still fixes reads where
            # distinct rows already exist.
            try:
                conn.rollback()
            except Exception:
                pass
    except Exception:
        # Soft-fail: do not crash app if schema check fails
        conn.rollback()


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
    except Exception:
        # Soft-fail: preserve app availability if migration cannot run.
        try:
            conn.rollback()
        except Exception:
            pass


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
    except Exception:
        # If schema introspection fails, do not block the app
        return

    if not has_col:
        try:
            cursor.execute("ALTER TABLE best_models_registry ADD COLUMN nsize INT NULL")
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
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
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def check_ds_exists(cursor, filename, args):
    """Check if a result already exists for given file and parameters.
    
    Returns:
        Row tuple (pred_label, confidence, log_path) if exists, None otherwise
    """
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
                WHERE filename=%s AND person_id=%s AND model_id=%s
                ORDER BY timestamp DESC
                LIMIT 1
                ''',
                (filename, st.session_state.person_id, model_id),
            )
            row = cursor.fetchone()
            if row:
                return row
        except Exception:
            pass

    cursor.execute(''' 
        SELECT pred_label, confidence, log_path FROM results
        WHERE filename=%s AND person_id=%s AND model_name=%s AND task=%s AND path=%s
              AND n_neighbors=%s AND nsize=%s AND fgsm=%s AND normalize=%s AND
              n_calibration=%s AND classif_loss=%s AND dloss=%s AND dist_fct=%s AND prototypes=%s
              AND npos=%s AND nneg=%s
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (
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
    ))

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
    values = (
        filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
        args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
        str(args.n_positives), str(args.n_negatives), pred_label, confidence, class_scores_json, log_path,
        st.session_state.person_id, model_id
    )
    try:
        cursor.execute(query, _mysql_values(values))
    except mysql.connector.errors.ProgrammingError as e:
        # Backward-compatible fallback when class_scores column is not yet present.
        if "class_scores" not in str(e):
            raise
        query_legacy = '''
            INSERT INTO results (
                filename, model_name, task, path, n_neighbors, nsize, fgsm, normalize, n_calibration, classif_loss,
                dloss, dist_fct, prototypes, npos, nneg, pred_label, confidence, log_path, person_id, model_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                pred_label = VALUES(pred_label),
                confidence = VALUES(confidence),
                log_path = VALUES(log_path),
                person_id = VALUES(person_id),
                model_id = VALUES(model_id),
                timestamp = CURRENT_TIMESTAMP
        '''
        legacy_values = (
            filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
            args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
            str(args.n_positives), str(args.n_negatives), pred_label, confidence, log_path,
            st.session_state.person_id, model_id
        )
        cursor.execute(query_legacy, _mysql_values(legacy_values))
    except mysql.connector.errors.IntegrityError:
        # If the unique key blocks insertion, perform an explicit update to refresh timestamp and values
        update_q = '''
            UPDATE results
            SET pred_label=%s, confidence=%s, class_scores=%s, log_path=%s, person_id=%s, model_id=%s, timestamp=CURRENT_TIMESTAMP
            WHERE filename=%s AND model_name=%s AND task=%s AND path=%s AND n_neighbors=%s AND nsize=%s AND fgsm=%s
                  AND normalize=%s AND n_calibration=%s AND classif_loss=%s AND dloss=%s AND dist_fct=%s AND prototypes=%s
                  AND npos=%s AND nneg=%s
        '''
        update_vals = (
            pred_label, confidence, class_scores_json, log_path, st.session_state.person_id, model_id,
            filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
            args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
            str(args.n_positives), str(args.n_negatives)
        )
        try:
            cursor.execute(update_q, _mysql_values(update_vals))
        except mysql.connector.errors.ProgrammingError as e:
            if "class_scores" not in str(e):
                raise
            update_q_legacy = '''
                UPDATE results
                SET pred_label=%s, confidence=%s, log_path=%s, person_id=%s, model_id=%s, timestamp=CURRENT_TIMESTAMP
                WHERE filename=%s AND model_name=%s AND task=%s AND path=%s AND n_neighbors=%s AND nsize=%s AND fgsm=%s
                      AND normalize=%s AND n_calibration=%s AND classif_loss=%s AND dloss=%s AND dist_fct=%s AND prototypes=%s
                      AND npos=%s AND nneg=%s
            '''
            update_vals_legacy = (
                pred_label, confidence, log_path, st.session_state.person_id, model_id,
                filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
                args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.dist_fct, args.prototypes_to_use,
                str(args.n_positives), str(args.n_negatives)
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
                    model_id INT NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    model_number VARCHAR(50),
                    log_path TEXT,
                    head_config VARCHAR(255),
                    head_name VARCHAR(255),
                    head_family VARCHAR(64),
                    head_n_aug INT,
                    prototype_strategy VARCHAR(64),
                    prototype_components INT,
                    set_by VARCHAR(255),
                    set_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES best_models_registry(id) ON DELETE CASCADE
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
    except Exception as e:
        print(f"Error creating production_model table: {e}")
        try:
            conn.rollback()
        except:
            pass


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
                   bmr.train_datasets, bmr.valid_dataset, bmr.test_dataset, bmr.split_config_key
            FROM production_model pm
            LEFT JOIN best_models_registry bmr ON pm.model_id = bmr.id
            WHERE pm.label_scheme=%s
            ORDER BY pm.set_at DESC
            LIMIT 1
            """,
            (label_scheme,),
        )
        row = cursor.fetchone()
        if row:
            if row[3] and not os.path.exists(os.path.join(str(row[3]), "model.pth")):
                print(f"Production model artifact missing, ignoring stale production row: {row[3]}")
                return None
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
                'Model Name': row[13],
                'NSize': row[14],
                'FGSM': row[15],
                'Prototypes': row[16],
                'NPos': row[17],
                'NNeg': row[18],
                'DLoss': row[19],
                'Dist_Fct': row[20],
                'Classif_Loss': row[21],
                'N_Calibration': row[22],
                'Accuracy': row[23],
                'MCC': row[24],
                'Normalize': row[25],
                'N_Neighbors': row[26],
                'train_datasets': row[29],
                'valid_dataset': row[30],
                'test_dataset': row[31],
                'split_config_key': row[32],
            }
        return None
    except Exception as e:
        print(f"Error getting production model: {e}")
        return None


def set_production_model(cursor, conn, model_dict, set_by_email):
    """Set the production model in database."""
    try:
        label_scheme = model_dict.get("label_scheme") or "four_class"
        cursor.execute("DELETE FROM production_model WHERE label_scheme=%s", (label_scheme,))
        
        # Insert new production model
        cursor.execute(
            """
            INSERT INTO production_model (
                label_scheme, model_id, model_name, model_number, log_path,
                head_config, head_name, head_family, head_n_aug,
                prototype_strategy, prototype_components, set_by
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                label_scheme,
                model_dict.get('model_id'),
                model_dict.get('Model Name'),
                model_dict.get('model_number'),
                model_dict.get('Log Path'),
                model_dict.get('Head Config') or model_dict.get('head_config') or model_dict.get('classification_head_config') or model_dict.get('best_classifier_config'),
                model_dict.get('Head') or model_dict.get('head_name') or model_dict.get('learned_classifier_label'),
                model_dict.get('head_family'),
                model_dict.get('head_n_aug') or model_dict.get('n_aug') or model_dict.get('N Aug'),
                model_dict.get('prototype_strategy') or model_dict.get('Proto_Strat'),
                model_dict.get('prototype_components') or model_dict.get('Proto_Comp'),
                set_by_email
            )
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error setting production model: {e}")
        try:
            conn.rollback()
        except:
            pass
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
        except Exception:
            # index probably already exists
            try:
                conn.rollback()
            except Exception:
                pass

    except Exception as e:
        print(f"Could not ensure people.user_email column: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
