# /home/simon/otitenet/otitenet/app/components/account_sidebar.py

import os

import streamlit as st
from mysql.connector import Error

from otitenet.data.labels import (
    DEFAULT_LABEL_TASK,
    TASK_LABEL_SCHEMES,
    label_scheme_for_task,
    task_display_name,
)


ADMIN_EMAIL = "simonjpelletier@gmail.com"


def _rerun():
    """Compatibility wrapper for Streamlit rerun."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _get_first_value(row, key_or_index):
    """
    Helper for cursors that may return tuples or dicts.
    """
    if row is None:
        return None

    if isinstance(row, dict):
        return row.get(key_or_index)

    if isinstance(key_or_index, int):
        return row[key_or_index]

    raise ValueError("Tuple rows require integer indexes.")


def require_login(conn, cursor):
    """
    Login/signup logic.

    This reproduces the original app.py behavior:
    - user enters email
    - if user exists, load users.id
    - if not, create the user
    - store user_email and user_id in st.session_state
    """
    if "user_email" not in st.session_state:
        st.session_state.user_email = None

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    if "person_id" not in st.session_state:
        st.session_state.person_id = None

    if "person_name" not in st.session_state:
        st.session_state.person_name = None

    if (
        st.session_state.user_email is not None
        and st.session_state.user_id is not None
    ):
        return st.session_state.user_id

    st.title("Login Required")

    email = st.text_input(
        "Enter your email to log in or sign up:",
        key="login_email",
    )

    if st.button("Continue", key="login_continue_button") and email.strip():
        clean_email = email.strip()

        try:
            cursor.execute(
                "SELECT id FROM users WHERE email=%s",
                (clean_email,),
            )
            row = cursor.fetchone()

            if row:
                user_id = _get_first_value(row, 0)
            else:
                cursor.execute(
                    "INSERT INTO users (email) VALUES (%s)",
                    (clean_email,),
                )
                conn.commit()
                user_id = cursor.lastrowid

            st.session_state.user_email = clean_email
            st.session_state.user_id = user_id
            st.session_state.is_admin = clean_email == ADMIN_EMAIL

            _rerun()

        except Error as e:
            st.error(f"❌ Database error during login: {e}")
            st.stop()

    st.stop()


def _fetch_people(cursor, user_id):
    """
    Fetch people/family members for the logged-in user.
    """
    cursor.execute(
        """
        SELECT id, name
        FROM people
        WHERE user_id=%s
        ORDER BY name
        """,
        (user_id,),
    )
    return cursor.fetchall()


def _normalize_people_rows(rows):
    """
    Convert cursor rows to a clean list of (person_id, person_name).
    Supports tuple cursors and dict cursors.
    """
    people = []

    for row in rows:
        if isinstance(row, dict):
            people.append((row["id"], row["name"]))
        else:
            people.append((row[0], row[1]))

    return people


def render_person_sidebar(conn, cursor):
    """
    Render person/family-member selector.

    This function must run before inference pages because those pages expect:

        st.session_state.person_id

    to exist and be selected.
    """
    st.sidebar.markdown("### 👤 Family Member")

    if "person_id" not in st.session_state:
        st.session_state.person_id = None

    if "person_name" not in st.session_state:
        st.session_state.person_name = None

    if st.session_state.get("user_id") is None:
        st.sidebar.warning("No user_id found. Login did not complete.")
        st.stop()

    user_id = st.session_state.user_id

    try:
        people_rows = _fetch_people(cursor, user_id)
        people = _normalize_people_rows(people_rows)

    except Error as e:
        st.sidebar.error(f"❌ Could not load people: {e}")
        st.stop()

    person_options = [name for _, name in people]
    person_ids = {name: person_id for person_id, name in people}

    if person_options:
        current_index = 0

        if st.session_state.person_id is not None:
            for i, (person_id, _) in enumerate(people):
                if person_id == st.session_state.person_id:
                    current_index = i
                    break

        selected_person = st.sidebar.selectbox(
            "Choose a person",
            person_options,
            index=current_index,
            key="person_selector",
        )

        st.session_state.person_id = person_ids[selected_person]
        st.session_state.person_name = selected_person

    with st.sidebar.expander("➕ Add a new person", expanded=not bool(person_options)):
        new_name = st.text_input(
            "Person's name",
            key="new_person_name",
        )

        if st.button("Add Person", key="add_person_button"):
            clean_name = new_name.strip()

            if not clean_name:
                st.warning("Please enter a name.")
            else:
                try:
                    cursor.execute(
                        """
                        INSERT INTO people (user_id, name)
                        VALUES (%s, %s)
                        """,
                        (user_id, clean_name),
                    )
                    conn.commit()

                    st.session_state.person_id = None
                    st.session_state.person_name = None

                    _rerun()

                except Error as e:
                    st.error(f"❌ Could not add person: {e}")

    with st.sidebar.expander("❌ Remove person"):
        if person_options:
            person_to_remove = st.selectbox(
                "Select person to delete",
                person_options,
                key="remove_person_selector",
            )

            if st.button("Delete Person", key="delete_person_button"):
                try:
                    person_id = person_ids[person_to_remove]

                    cursor.execute(
                        "DELETE FROM results WHERE person_id=%s",
                        (person_id,),
                    )
                    cursor.execute(
                        "DELETE FROM people WHERE id=%s",
                        (person_id,),
                    )
                    conn.commit()

                    if st.session_state.person_id == person_id:
                        st.session_state.person_id = None
                        st.session_state.person_name = None

                    _rerun()

                except Error as e:
                    st.error(f"❌ Could not delete person: {e}")
        else:
            st.caption("No person to remove.")

    with st.sidebar.expander("🚪 Account"):
        st.write(f"Logged in as: `{st.session_state.user_email}`")

        if st.button("Log out", key="logout_button"):
            st.session_state.user_email = None
            st.session_state.user_id = None
            st.session_state.person_id = None
            st.session_state.person_name = None
            st.session_state.is_admin = False
            _rerun()

        st.markdown("---")

        if st.button("Delete my account", key="delete_account_button"):
            try:
                cursor.execute(
                    """
                    DELETE FROM results
                    WHERE person_id IN (
                        SELECT id FROM people WHERE user_id=%s
                    )
                    """,
                    (user_id,),
                )
                cursor.execute(
                    "DELETE FROM people WHERE user_id=%s",
                    (user_id,),
                )
                cursor.execute(
                    "DELETE FROM users WHERE id=%s",
                    (user_id,),
                )
                conn.commit()

                st.session_state.clear()
                _rerun()

            except Error as e:
                st.error(f"❌ Could not delete account: {e}")

    if st.session_state.person_id is None:
        st.warning("Please create or select a family member to proceed.")
        return None

    return st.session_state.person_id


def render_labeling_task_sidebar():
    """Select which task/labeling scenario is active for production inference."""
    st.sidebar.markdown("### 🏷️ Labeling")

    task_root = os.path.join("logs", "best_models")
    if os.path.isdir(task_root):
        tasks = [
            name
            for name in sorted(os.listdir(task_root))
            if os.path.isdir(os.path.join(task_root, name)) and name in TASK_LABEL_SCHEMES
        ]
    else:
        tasks = []
    if not tasks:
        tasks = list(TASK_LABEL_SCHEMES.keys())
    current = st.session_state.get("production_task", DEFAULT_LABEL_TASK)
    if current not in tasks:
        current = DEFAULT_LABEL_TASK

    selected = st.sidebar.selectbox(
        "Task",
        options=tasks,
        index=tasks.index(current),
        format_func=task_display_name,
        key="production_task_selector",
    )

    if selected != st.session_state.get("production_task"):
        st.session_state["production_task"] = selected
        st.session_state["label_scheme"] = label_scheme_for_task(selected)
        st.session_state["production_model"] = None
        for key in [
            "selected_model_params",
            "selected_params_version",
            "selected_params_last_sync",
            "selected_model_log_path",
            "selected_model_selection_key",
            "selected_model_version",
            "sidebar_best_model_key",
            "sidebar_best_model_last_sync",
            "sidebar_classification_head_config",
            "sidebar_classification_head_model_key",
            "optimized_k_value",
            "k_opt_current_selection",
            "model_number_map",
            "best_models_table",
        ]:
            st.session_state.pop(key, None)
        _rerun()

    st.session_state["production_task"] = selected
    st.session_state["label_scheme"] = label_scheme_for_task(selected)
    return selected


def render_current_optimization_sidebar(is_admin=False):
    """
    Render current optimization / selected model status.

    This keeps the old app behavior but avoids crashing if the relevant
    session state keys are missing.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Optimization:**")

    def _is_present(value):
        return value is not None and str(value).strip() not in {"", "None", "nan", "?"}

    def _selected_model_status():
        selected = st.session_state.get("selected_model_params")
        if not isinstance(selected, dict):
            return None

        model_name = (
            selected.get("Model Name")
            or selected.get("model_name")
            or selected.get("model")
        )
        model_id = (
            selected.get("Model ID")
            or selected.get("model_id")
            or selected.get("id")
        )
        model_number = (
            selected.get("#")
            or selected.get("model_number")
        )
        if not _is_present(model_number):
            selection_key = st.session_state.get("selected_model_selection_key")
            model_number = st.session_state.get("model_number_map", {}).get(selection_key)

        if not (model_name or model_id or _is_present(model_number)):
            return None

        label = f"Selected model: {model_name or 'model'}"
        if _is_present(model_number):
            label += f" #{model_number}"
        if _is_present(model_id):
            label += f" (ID: {model_id})"
        return label

    current_selection = st.session_state.get("k_opt_current_selection")

    if current_selection:
        if is_admin:
            selected_status = _selected_model_status()
            if selected_status:
                st.sidebar.success(selected_status)
            else:
                st.sidebar.success("Optimization selected")

        else:
            if current_selection.get("type") == "knn":
                k = current_selection.get("k", "?")
                mcc = current_selection.get("mcc", 0)
                st.sidebar.success(f"✅ KNN with k={k}\n\nMCC: {mcc:.4f}")
            else:
                strategy = str(current_selection.get("strategy", "?")).upper()
                n_comp = current_selection.get("n_comp", "?")
                mcc = current_selection.get("mcc", 0)

                st.sidebar.success(
                    f"✅ {strategy} Strategy\n"
                    f"n_comp={n_comp}\n\n"
                    f"MCC: {mcc:.4f}"
                )

    else:
        selected_status = _selected_model_status()
        if selected_status:
            st.sidebar.success(selected_status)
            return

        production_model = st.session_state.get("production_model")

        if production_model:
            if isinstance(production_model, dict):
                model_id = (
                    production_model.get("model_id")
                    or production_model.get("Model ID")
                    or production_model.get("id")
                )
                model_name = (
                    production_model.get("model_name")
                    or production_model.get("Model Name")
                    or production_model.get("model")
                )
                model_number = production_model.get("model_number") or production_model.get("#")

                if model_name or model_id:
                    task = production_model.get("label_task", st.session_state.get("production_task", DEFAULT_LABEL_TASK))
                    label = f"Production model: {model_name or 'model'}"
                    if _is_present(model_number):
                        label += f" #{model_number}"
                    if _is_present(model_id):
                        label += f" (ID: {model_id})"
                    st.sidebar.info(
                        f"{label}\n\n"
                        f"Task: {task_display_name(task)}"
                    )
                else:
                    st.sidebar.info("Production model loaded.")
            else:
                st.sidebar.info("Production model loaded.")
        else:
            st.sidebar.info("No optimization applied yet. Run 'Optimize k' in Tab 1.")


def is_current_user_admin():
    """
    Convenience function if another module imports it from here.
    """
    return st.session_state.get("user_email") == ADMIN_EMAIL
