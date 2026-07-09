"""
Bootstrap utilities for the Otitenet application.

The original code expected a module `otitenet.bootstrap` that provided
the following functions:
    - initialize_database
    - initialize_model_ranks_once
    - initialize_user_state
    - is_current_user_admin
    - load_production_model

Those functions live in the existing `otitenet/app/database.py` and
`otitenet/app/utils.py`.  This file simply imports and re‑exports them so
that `from otitenet.bootstrap import …` works correctly.
"""

import streamlit as st

# Database‑related bootstrapping
from .database import (
    create_db,
    get_db_connection,
    ensure_results_model_id,
    ensure_results_class_scores,
    ensure_results_head_config,
    ensure_best_models_registry_nsize,
    check_ds_exists,
    list_image_results,
    fetch_model_by_log_path,
    resolve_model_id,
    insert_score,
)

@st.cache_resource
def initialize_database():
    from .database import (
        create_db,
        get_db_connection,
        ensure_results_model_id,
        ensure_results_class_scores,
        ensure_results_head_config,
        ensure_best_models_registry_nsize,
        ensure_registry_metrics_columns,
        cleanup_duplicate_best_models_registry,
        cleanup_incompatible_best_models_registry,
        cleanup_duplicate_registry_csvs,
        is_mysql_connection_lost_error,
    )

    def _refresh_connection():
        try:
            get_db_connection.clear()
        except Exception:
            pass
        return create_db()

    def _run_db_step(name, fn, conn, cursor):
        try:
            try:
                conn.ping(reconnect=True, attempts=2, delay=1)
            except Exception:
                conn, cursor = _refresh_connection()
            fn(conn, cursor)
        except Exception as e:
            print(f"[bootstrap] Warning: {name} failed: {e}")
            if is_mysql_connection_lost_error(e):
                try:
                    conn, cursor = _refresh_connection()
                except Exception as refresh_error:
                    print(f"[bootstrap] Warning: database reconnect after {name} failed: {refresh_error}")
        return conn, cursor

    conn, cursor = create_db()

    conn, cursor = _run_db_step("ensure_results_model_id", ensure_results_model_id, conn, cursor)
    conn, cursor = _run_db_step("ensure_results_class_scores", ensure_results_class_scores, conn, cursor)
    conn, cursor = _run_db_step("ensure_results_head_config", ensure_results_head_config, conn, cursor)
    conn, cursor = _run_db_step("ensure_best_models_registry_nsize", ensure_best_models_registry_nsize, conn, cursor)
    conn, cursor = _run_db_step("ensure_registry_metrics_columns", ensure_registry_metrics_columns, conn, cursor)
    conn, cursor = _run_db_step("cleanup_incompatible_best_models_registry", cleanup_incompatible_best_models_registry, conn, cursor)
    conn, cursor = _run_db_step("cleanup_duplicate_best_models_registry", cleanup_duplicate_best_models_registry, conn, cursor)

    try:
        cleanup_duplicate_registry_csvs()
    except Exception as e:
        print(f"[bootstrap] Warning: cleanup_duplicate_registry_csvs failed: {e}")

    try:
        from .database import ensure_production_model_table
        conn, cursor = _run_db_step("ensure_production_model_table", ensure_production_model_table, conn, cursor)
    except ImportError:
        print("[bootstrap] Warning: ensure_production_model_table not found")

    try:
        from .database import ensure_people_user_email
        conn, cursor = _run_db_step("ensure_people_user_email", ensure_people_user_email, conn, cursor)
    except ImportError:
        print("[bootstrap] Warning: ensure_people_user_email not found")

    try:
        from .database import ensure_learned_embedding_heads_table
        conn, cursor = _run_db_step("ensure_learned_embedding_heads_table", ensure_learned_embedding_heads_table, conn, cursor)
    except ImportError:
        print("[bootstrap] Warning: ensure_learned_embedding_heads_table not found")

    try:
        conn.ping(reconnect=True, attempts=2, delay=1)
        cursor.execute("SELECT 1")
        cursor.fetchone()
    except Exception:
        conn, cursor = _refresh_connection()
    return conn, cursor

# Model‑rank initialization (placeholder – actual logic lives elsewhere)
def initialize_model_ranks_once():
    """Placeholder for model‑rank initialization."""
    # In the original code this performed a one‑time DB migration.
    # If additional logic is needed, import and call it here.
    pass

# User‑state initialization
def initialize_user_state():
    """Initialize keys used across the app."""
    import streamlit as st

    defaults = {
        "is_admin": False,
        "user_email": None,
        "user_id": None,
        "person_id": None,
        "person_name": None,
        "selected_person": None,
        "production_model": None,
        "production_task": "notNormal",
        "label_scheme": "binary",
        "current_model_id": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Admin check – uses Streamlit session_state set by the sidebar component
def is_current_user_admin():
    """Return True if the logged‑in user is an admin."""
    import streamlit as st
    return bool(st.session_state.get("is_admin", False))

# Production model loader (delegates to the service layer)
def load_production_model(_cursor):
    """Load the current production model into session_state."""
    from .services.production_model_service import get_production_model, set_production_model
    import time
    import streamlit as st

    task = st.session_state.get("production_task", "notNormal")
    now = time.monotonic()
    model_info = get_production_model(_cursor, task=task)
    if model_info:
        st.session_state["production_model"] = model_info
        st.session_state["production_model_cache_task"] = task
        st.session_state["production_model_cache_ts"] = now
    return model_info
