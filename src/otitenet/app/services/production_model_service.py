import json
from pathlib import Path
import streamlit as st

from otitenet.app.utils import parse_classifier_config
from otitenet.data.labels import DEFAULT_LABEL_TASK, label_scheme_for_task


PRODUCTION_MODEL_STATE_KEY = "production_model"
PRODUCTION_MODEL_FILE = Path("data/production_model.json")


def _production_model_file(label_scheme: str) -> Path:
    return Path(f"data/production_model_{label_scheme}.json")


def get_production_model(cursor=None, task: str | None = None, label_scheme: str | None = None):
    """
    Load the current production model.

    Priority:
    1. st.session_state["production_model"]
    2. data/production_model.json
    3. None
    """
    task = task or st.session_state.get("production_task", DEFAULT_LABEL_TASK)
    label_scheme = label_scheme or label_scheme_for_task(task)

    if PRODUCTION_MODEL_STATE_KEY in st.session_state and st.session_state[PRODUCTION_MODEL_STATE_KEY] is not None:
        model = st.session_state[PRODUCTION_MODEL_STATE_KEY]
        if not isinstance(model, dict) or model.get("label_task", task) == task:
            return model

    if cursor is not None:
        try:
            from otitenet.app.database import get_production_model as get_db_production_model

            model_info = get_db_production_model(cursor, label_scheme=label_scheme)
            if model_info:
                model_info.setdefault("label_task", task)
                st.session_state[PRODUCTION_MODEL_STATE_KEY] = model_info
                return model_info
        except Exception as e:
            st.warning(f"Could not load production model from database: {e}")

    model_file = _production_model_file(label_scheme)
    fallback_files = [model_file]
    if label_scheme == "four_class":
        fallback_files.append(PRODUCTION_MODEL_FILE)

    for candidate in fallback_files:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r") as f:
                model_info = json.load(f)

            model_info.setdefault("label_scheme", label_scheme)
            model_info.setdefault("label_task", task)
            st.session_state[PRODUCTION_MODEL_STATE_KEY] = model_info
            return model_info

        except Exception as e:
            st.warning(f"Could not load production model file {candidate}: {e}")
            return None

    return None


def set_production_model(model_info, cursor=None, conn=None, task: str | None = None, label_scheme: str | None = None):
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

    task = task or st.session_state.get("production_task", DEFAULT_LABEL_TASK)
    label_scheme = label_scheme or label_scheme_for_task(task)
    model_info = dict(model_info)
    model_info["label_scheme"] = label_scheme
    model_info["label_task"] = task

    model_file = _production_model_file(label_scheme)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    with open(model_file, "w") as f:
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
        head_config = (
            production_model.get("best_classifier_config")
            or production_model.get("classification_head_config")
            or production_model.get("head_config")
            or production_model.get("Head Config")
        )

        if model_id is not None:
            setattr(args, "model_id", model_id)

        if model_name is not None:
            setattr(args, "model_name", model_name)

        if log_path is not None:
            setattr(args, "log_path", log_path)

        if head_name is not None:
            setattr(args, "head_name_selected", head_name)

        if head_config is not None:
            setattr(args, "best_classifier_config", str(head_config))
            head_meta = parse_classifier_config(head_config)
            setattr(args, "classification_head_family", head_meta.get("family"))
            if head_meta.get("family") == "prototype":
                setattr(args, "prototypes_to_use", "class")
                setattr(args, "prototype_strategy", str(head_meta.get("strategy", "mean")))
                setattr(args, "prototype_components", int(head_meta.get("components", 1)))
            elif head_meta.get("family") == "knn":
                setattr(args, "n_neighbors", int(head_meta.get("k", getattr(args, "n_neighbors", 1))))

        if production_model.get("label_scheme") is not None:
            setattr(args, "label_scheme", production_model.get("label_scheme"))
        if production_model.get("label_task") is not None:
            setattr(args, "task", production_model.get("label_task"))
        for key in ["train_datasets", "valid_dataset", "test_dataset"]:
            if production_model.get(key) is not None:
                setattr(args, key, production_model.get(key))
        if production_model.get("split_config_key") or (
            log_path is not None and "/split_" in str(log_path).replace("\\", "/")
        ):
            setattr(args, "split_config_in_path", True)
            setattr(args, "_split_config_in_path", True)

        setattr(args, "production_model", production_model)

    return args
