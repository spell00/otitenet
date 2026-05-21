import json
from pathlib import Path
import streamlit as st


PRODUCTION_MODEL_STATE_KEY = "production_model"
PRODUCTION_MODEL_FILE = Path("data/production_model.json")


def get_production_model(cursor=None):
    """
    Load the current production model.

    Priority:
    1. st.session_state["production_model"]
    2. data/production_model.json
    3. None
    """
    if PRODUCTION_MODEL_STATE_KEY in st.session_state:
        return st.session_state[PRODUCTION_MODEL_STATE_KEY]

    if PRODUCTION_MODEL_FILE.exists():
        try:
            with open(PRODUCTION_MODEL_FILE, "r") as f:
                model_info = json.load(f)

            st.session_state[PRODUCTION_MODEL_STATE_KEY] = model_info
            return model_info

        except Exception as e:
            st.warning(f"Could not load production model file: {e}")
            return None

    return None


def set_production_model(model_info, cursor=None, conn=None):
    """
    Save the selected production model.

    model_info should usually be a dict containing things like:
    {
        "model_id": 1,
        "model_name": "resnet18",
        "head_name": "...",
        "log_path": "...",
        ...
    }
    """
    if model_info is None:
        return None

    PRODUCTION_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(PRODUCTION_MODEL_FILE, "w") as f:
        json.dump(model_info, f, indent=2, default=str)

    st.session_state[PRODUCTION_MODEL_STATE_KEY] = model_info
    return model_info


def apply_production_model_to_args(args, production_model, data_dir="./data"):
    """
    Apply production model settings to args for non-admin users.

    This is intentionally defensive because production_model may have
    different keys depending on what your leaderboard/admin page saved.
    """
    if not production_model:
        return args

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
        log_path = (
            production_model.get("log_path")
            or production_model.get("Log Path")
            or production_model.get("path")
        )
        head_name = (
            production_model.get("head_name")
            or production_model.get("head_name_selected")
            or production_model.get("Head")
        )

        if model_id is not None:
            setattr(args, "model_id", model_id)

        if model_name is not None:
            setattr(args, "model_name", model_name)

        if log_path is not None:
            setattr(args, "log_path", log_path)

        if head_name is not None:
            setattr(args, "head_name_selected", head_name)

        setattr(args, "production_model", production_model)

    return args