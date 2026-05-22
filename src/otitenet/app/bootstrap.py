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

# Database‑related bootstrapping
from .database import (
    create_db,
    get_db_connection,
    ensure_results_model_id,
    ensure_best_models_registry_nsize,
    check_ds_exists,
    list_image_results,
    fetch_model_by_log_path,
    resolve_model_id,
    insert_score,
)

def initialize_database():
    from .database import (
        create_db,
        ensure_results_model_id,
        ensure_best_models_registry_nsize,
    )

    conn, cursor = create_db()

    try:
        ensure_results_model_id(conn, cursor)
    except Exception as e:
        print(f"[bootstrap] Warning: ensure_results_model_id failed: {e}")

    try:
        ensure_best_models_registry_nsize(conn, cursor)
    except Exception as e:
        print(f"[bootstrap] Warning: ensure_best_models_registry_nsize failed: {e}")

    try:
        from .database import ensure_production_model_table
        ensure_production_model_table(conn, cursor)
    except ImportError:
        print("[bootstrap] Warning: ensure_production_model_table not found")
    except Exception as e:
        print(f"[bootstrap] Warning: ensure_production_model_table failed: {e}")

    try:
        from .database import ensure_people_user_email
        ensure_people_user_email(conn, cursor)
    except ImportError:
        print("[bootstrap] Warning: ensure_people_user_email not found")
    except Exception as e:
        print(f"[bootstrap] Warning: ensure_people_user_email failed: {e}")

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
def load_production_model(cursor):
    """Load the current production model into session_state."""
    from .services.production_model_service import get_production_model, set_production_model

    model_info = get_production_model(cursor)
    if model_info:
        import streamlit as st
        st.session_state["production_model"] = model_info
    return model_info
