import inspect
import json
from pathlib import Path
from typing import Any

import streamlit as st

from otitenet.app.utils import parse_classifier_config
from otitenet.data.labels import DEFAULT_LABEL_TASK, label_scheme_for_task


PRODUCTION_MODEL_STATE_KEY = "production_model"
PRODUCTION_MODEL_FILE = Path("data/production_model.json")


_EMPTY_VALUES = {"", "None", "none", "nan", "NaN", "—", "null", "NULL"}


def _production_model_file(label_scheme: str) -> Path:
    return Path(f"data/production_model_{label_scheme}.json")


def _is_present(value: Any) -> bool:
    return value is not None and str(value).strip() not in _EMPTY_VALUES


def _first(mapping: dict, *keys, default=None):
    for key in keys:
        value = mapping.get(key)
        if _is_present(value):
            return value
    return default


def _coerce_db_value(column: str, value: Any, column_types: dict[str, str] | None = None):
    if not _is_present(value):
        return None

    column_type = str((column_types or {}).get(column, "")).lower()
    if any(token in column_type for token in ("int", "decimal", "numeric", "float", "double", "real")):
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if "int" in column_type:
            return int(number)
        return number

    return value


def _set_all(mapping: dict, keys: list[str], value):
    if not _is_present(value):
        return
    for key in keys:
        mapping[key] = value


def _production_model_id_for_db(model_info: dict) -> str:
    for key in ("model_id", "Registry ID", "DB Model ID", "Model ID", "Artifact ID", "exp ID", "id"):
        value = model_info.get(key)
        if _is_present(value):
            return str(value).strip()
    log_path = str(model_info.get("Log Path") or model_info.get("log_path") or "").strip()
    if log_path:
        return f"log:{log_path}"
    model_name = str(model_info.get("Model Name") or model_info.get("model_name") or "model").strip()
    model_number = str(model_info.get("model_number") or model_info.get("#") or "").strip()
    return f"model:{model_name}:{model_number or 'unranked'}"


def _normalize_production_model(model_info: dict | None, args=None) -> dict | None:
    """
    Normalize aliases used by the app/database/leaderboard.

    This is the important part for your bug:
    production may contain Dist_Fct=cosine while the selected runtime args have
    dist_fct=euclidean. When args are provided, args win.
    """
    if model_info is None:
        return None

    model_info = dict(model_info)

    # Prefer the selected/sidebar args when the caller provides them.
    dist = None
    if args is not None:
        dist = (
            getattr(args, "dist_fct", None)
            or getattr(args, "dist_metric", None)
            or getattr(args, "distance", None)
        )
    if not _is_present(dist):
        dist = _first(model_info, "Dist_Fct", "dist_fct", "dist_metric", "Distance", "distance")

    if _is_present(dist):
        dist = str(dist).strip()
        _set_all(model_info, ["Dist_Fct", "dist_fct", "dist_metric", "Distance"], dist)

    # Head config aliases.
    head_config = None
    if args is not None:
        head_config = (
            getattr(args, "best_classifier_config", None)
            or getattr(args, "classification_head_config", None)
            or getattr(args, "classifier_head_config", None)
            or getattr(args, "head_config", None)
        )
    if not _is_present(head_config):
        head_config = _first(
            model_info,
            "Best Head Config",
            "best_head_config",
            "Head Config",
            "best_classifier_config",
            "classification_head_config",
            "classifier_head_config",
            "head_config",
        )

    if _is_present(head_config):
        head_config = str(head_config).strip()
        _set_all(
            model_info,
            [
                "Best Head Config",
                "best_head_config",
                "Head Config",
                "best_classifier_config",
                "classification_head_config",
                "classifier_head_config",
                "head_config",
            ],
            head_config,
        )

    # Head name aliases.
    head_name = None
    if args is not None:
        head_name = getattr(args, "learned_classifier_label", None) or getattr(args, "head_name_selected", None)
    if not _is_present(head_name):
        head_name = _first(model_info, "Best Classification Head", "head_name", "head_name_selected", "Head")
    if _is_present(head_name):
        _set_all(model_info, ["Best Classification Head", "head_name", "head_name_selected", "Head"], str(head_name))

    # Runtime classifier settings. Args win.
    runtime_aliases = {
        "n_aug": ["Head N Aug", "Best Head N Aug", "best_head_n_aug", "head_n_aug", "N Aug", "n_aug"],
        "n_neighbors": ["N_Neighbors", "n_neighbors"],
        "prototypes_to_use": ["Prototypes", "prototypes", "prototypes_to_use"],
        "prototype_strategy": ["Proto_Strat", "prototype_strategy"],
        "prototype_components": ["Proto_Comp", "prototype_components"],
        "siamese_inference": ["siamese_inference"],
        "classification_head_family": ["classification_head_family"],
    }
    if args is not None:
        for attr, keys in runtime_aliases.items():
            value = getattr(args, attr, None)
            if _is_present(value):
                _set_all(model_info, keys, value)

    return model_info


def _load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else None
    except Exception as e:
        st.warning(f"Could not load production model file {path}: {e}")
        return None


def _save_json_file(path: Path, model_info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(model_info, f, indent=2, default=str)


def _try_get_db_model(cursor, label_scheme: str):
    try:
        from otitenet.app.database import get_production_model as get_db_production_model

        try:
            return get_db_production_model(cursor, label_scheme=label_scheme)
        except TypeError:
            return get_db_production_model(cursor)
    except Exception as e:
        st.warning(f"Could not load production model from database: {e}")
        return None


def _try_save_db_model(cursor, conn, model_info: dict, label_scheme: str) -> bool:
    """
    Best-effort DB persistence. This is intentionally permissive because the app
    has had multiple production_model schemas/functions during development.
    """
    if cursor is None or conn is None:
        return False

    # 1) Prefer project database helpers if available.
    try:
        import otitenet.app.database as db

        for fn_name in [
            "set_production_model",
            "save_production_model",
            "upsert_production_model",
            "update_production_model",
        ]:
            fn = getattr(db, fn_name, None)
            if fn is None:
                continue

            try:
                sig = inspect.signature(fn)
                kwargs = {}
                if "label_scheme" in sig.parameters:
                    kwargs["label_scheme"] = label_scheme
                if "model_info" in sig.parameters:
                    kwargs["model_info"] = model_info
                    result = fn(cursor, conn, **kwargs)
                else:
                    # Common positional possibilities.
                    try:
                        result = fn(cursor, conn, model_info, **kwargs)
                    except TypeError:
                        try:
                            result = fn(cursor, model_info, **kwargs)
                        except TypeError:
                            result = fn(model_info, **kwargs)
                if result is False:
                    continue
                try:
                    conn.commit()
                except Exception:
                    pass
                return True
            except Exception:
                continue
    except Exception:
        pass

    # 2) Generic direct SQL fallbacks.
    try:
        payload = json.dumps(model_info, default=str)

        # MySQL/MariaDB.
        cursor.execute("SHOW TABLES")
        tables_raw = cursor.fetchall()
        tables = set()
        for row in tables_raw:
            if isinstance(row, dict):
                tables.update(str(v) for v in row.values())
            elif isinstance(row, (list, tuple)) and row:
                tables.add(str(row[0]))
            else:
                tables.add(str(row))

        candidate_tables = [
            "production_models",
            "production_model",
            "app_production_model",
        ]

        for table in candidate_tables:
            if table not in tables:
                continue

            cursor.execute(f"DESCRIBE {table}")
            desc = cursor.fetchall()
            columns = []
            column_types = {}
            for row in desc:
                if isinstance(row, dict):
                    column = str(row.get("Field") or row.get("field") or row.get("COLUMN_NAME"))
                    columns.append(column)
                    column_types[column] = str(row.get("Type") or row.get("type") or row.get("DATA_TYPE") or "")
                elif isinstance(row, (list, tuple)) and row:
                    column = str(row[0])
                    columns.append(column)
                    if len(row) > 1:
                        column_types[column] = str(row[1])

            # JSON payload schema.
            json_col = next((c for c in ["model_info", "model_json", "payload", "json", "data"] if c in columns), None)
            label_col = next((c for c in ["label_scheme", "scheme"] if c in columns), None)
            active_col = next((c for c in ["is_active", "active"] if c in columns), None)

            if json_col:
                if label_col:
                    cursor.execute(f"DELETE FROM {table} WHERE {label_col}=%s", (label_scheme,))
                    if active_col:
                        cursor.execute(
                            f"INSERT INTO {table} ({label_col}, {json_col}, {active_col}) VALUES (%s, %s, %s)",
                            (label_scheme, payload, 1),
                        )
                    else:
                        cursor.execute(
                            f"INSERT INTO {table} ({label_col}, {json_col}) VALUES (%s, %s)",
                            (label_scheme, payload),
                        )
                else:
                    cursor.execute(f"DELETE FROM {table}")
                    if active_col:
                        cursor.execute(f"INSERT INTO {table} ({json_col}, {active_col}) VALUES (%s, %s)", (payload, 1))
                    else:
                        cursor.execute(f"INSERT INTO {table} ({json_col}) VALUES (%s)", (payload,))
                conn.commit()
                return True

            # Wide-table schema: insert any matching columns present. Keep this
            # as a dict so label/distance aliases cannot create duplicate cols.
            assignments = {}
            for key, value in model_info.items():
                if key in columns and key != label_col:
                    assignments[key] = value
            if "Dist_Fct" in columns:
                assignments["Dist_Fct"] = model_info.get("Dist_Fct")
            if "dist_fct" in columns:
                assignments["dist_fct"] = model_info.get("dist_fct")
            if "dist_metric" in columns:
                assignments["dist_metric"] = model_info.get("dist_metric")
            if "model_id" in columns:
                assignments["model_id"] = _production_model_id_for_db(model_info)
            if "model_name" in columns and not _is_present(assignments.get("model_name")):
                assignments["model_name"] = _first(model_info, "model_name", "Model Name", "model", default="production_model")
            if "log_path" in columns and not _is_present(assignments.get("log_path")):
                assignments["log_path"] = _first(model_info, "log_path", "Log Path", "Best Model Dir", "Source Run Path", default="")

            if assignments:
                assignments = {
                    column: _coerce_db_value(column, value, column_types)
                    for column, value in assignments.items()
                }
                if label_col:
                    cursor.execute(f"DELETE FROM {table} WHERE {label_col}=%s", (label_scheme,))
                    cols = [label_col] + list(assignments.keys())
                    vals = [label_scheme] + list(assignments.values())
                    placeholders = ",".join(["%s"] * len(cols))
                    quoted_cols = ",".join(f"`{c}`" for c in cols)
                    cursor.execute(f"INSERT INTO {table} ({quoted_cols}) VALUES ({placeholders})", tuple(vals))
                else:
                    cursor.execute(f"DELETE FROM {table}")
                    cols = list(assignments.keys())
                    vals = list(assignments.values())
                    placeholders = ",".join(["%s"] * len(cols))
                    quoted_cols = ",".join(f"`{c}`" for c in cols)
                    cursor.execute(f"INSERT INTO {table} ({quoted_cols}) VALUES ({placeholders})", tuple(vals))
                conn.commit()
                return True

    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        st.warning(f"Could not save production model to database: {e}")

    return False


def get_production_model(cursor=None, task: str | None = None, label_scheme: str | None = None):
    """
    Load the current production model.

    IMPORTANT:
    Session and JSON are checked before DB, because the DB row may be stale
    after pressing "Set as Production model" during development.
    """
    task = task or st.session_state.get("production_task", DEFAULT_LABEL_TASK)
    label_scheme = label_scheme or label_scheme_for_task(task)

    # 1) Current Streamlit session wins.
    model = st.session_state.get(PRODUCTION_MODEL_STATE_KEY)
    if isinstance(model, dict) and model.get("label_task", task) == task:
        model = _normalize_production_model(model)
        model.setdefault("label_scheme", label_scheme)
        model.setdefault("label_task", task)
        st.session_state[PRODUCTION_MODEL_STATE_KEY] = model
        return model

    # 2) JSON file wins over DB because it is what set_production_model writes reliably.
    model_file = _production_model_file(label_scheme)
    fallback_files = [model_file]
    if label_scheme == "four_class":
        fallback_files.append(PRODUCTION_MODEL_FILE)

    for candidate in fallback_files:
        model = _load_json_file(candidate)
        if model:
            model = _normalize_production_model(model)
            model.setdefault("label_scheme", label_scheme)
            model.setdefault("label_task", task)
            st.session_state[PRODUCTION_MODEL_STATE_KEY] = model
            return model

    # 3) DB fallback only.
    if cursor is not None:
        model = _try_get_db_model(cursor, label_scheme)
        if model:
            model = _normalize_production_model(model)
            model.setdefault("label_scheme", label_scheme)
            model.setdefault("label_task", task)
            st.session_state[PRODUCTION_MODEL_STATE_KEY] = model
            # Mirror DB fallback to JSON so later reruns are stable.
            try:
                _save_json_file(model_file, model)
            except Exception:
                pass
            return model

    return None


def set_production_model(model_info, cursor=None, conn=None, task: str | None = None, label_scheme: str | None = None, args=None):
    """
    Save the selected production model.

    Pass args=ctx.args or current sidebar args when possible. If args is passed,
    args.dist_fct wins over stale model_info["Dist_Fct"].
    """
    if model_info is None:
        return None

    task = task or st.session_state.get("production_task", DEFAULT_LABEL_TASK)
    label_scheme = label_scheme or label_scheme_for_task(task)

    model_info = _normalize_production_model(dict(model_info), args=args)
    model_info["label_scheme"] = label_scheme
    model_info["label_task"] = task

    model_file = _production_model_file(label_scheme)
    _save_json_file(model_file, model_info)

    # Also write legacy file for four_class because parts of the app may still read it.
    if label_scheme == "four_class":
        try:
            _save_json_file(PRODUCTION_MODEL_FILE, model_info)
        except Exception:
            pass

    st.session_state[PRODUCTION_MODEL_STATE_KEY] = model_info

    saved_to_db = _try_save_db_model(cursor, conn, model_info, label_scheme)
    if not saved_to_db and cursor is not None and conn is not None:
        st.warning("Production model was saved to JSON/session, but not to the database. If another startup loader reads DB directly, it may still show stale values.")

    return model_info


def apply_production_model_to_args(args, production_model, data_dir="./data"):
    """
    Apply production model settings to args for production inference.

    Defensive alias normalization is used so dist_fct/dist_metric cannot diverge.
    """
    if not production_model:
        return args

    if isinstance(production_model, dict):
        production_model = _normalize_production_model(production_model)

        def _first_local(*keys):
            return _first(production_model, *keys)

        def _set_if_present(attr, *keys, cast=None):
            value = _first_local(*keys)
            if value is None:
                return
            if cast is not None:
                try:
                    value = cast(value)
                except Exception:
                    pass
            setattr(args, attr, value)

        model_id = _first_local("Registry ID", "DB Model ID", "registry_id", "model_id", "Model ID", "id")
        model_name = _first_local("model_name", "Model Name", "model")
        log_path = _first_local("log_path", "Log Path", "path")
        dataset_path = _first_local(
            "Dataset",
            "dataset",
            "Artifact Dataset",
            "Combo Dataset",
            "data_path",
        )
        head_name = _first_local("Best Classification Head", "head_name", "head_name_selected", "Head")
        head_config = _first_local(
            "Best Head Config",
            "best_head_config",
            "Head Config",
            "best_classifier_config",
            "classification_head_config",
            "classifier_head_config",
            "head_config",
        )
        head_n_aug = _first_local(
            "Head N Aug",
            "Best Head N Aug",
            "best_head_n_aug",
            "head_n_aug",
            "N Aug",
            "n_aug",
        )

        if model_id is not None:
            setattr(args, "model_id", model_id)

        if model_name is not None:
            setattr(args, "model_name", model_name)

        if log_path is not None:
            setattr(args, "log_path", log_path)

        if dataset_path is not None:
            dataset_path = str(dataset_path)
            if dataset_path and not dataset_path.startswith("data/") and not dataset_path.startswith("./data/"):
                dataset_path = f"{data_dir.rstrip('/')}/{dataset_path}"
            setattr(args, "path", dataset_path)

        _set_if_present("new_size", "NSize", "nsize", "new_size", cast=lambda v: int(float(str(v).replace("nsize", ""))))
        _set_if_present("normalize", "Normalize", "normalize", cast=str)
        _set_if_present("fgsm", "FGSM", "fgsm", cast=str)

        # Critical: synchronize distance aliases.
        _set_if_present("dist_fct", "Dist_Fct", "dist_fct", "dist_metric", "Distance", cast=str)
        if _is_present(getattr(args, "dist_fct", None)):
            setattr(args, "dist_metric", str(getattr(args, "dist_fct")))

        _set_if_present("classif_loss", "Classif_Loss", "classif_loss", cast=str)
        _set_if_present("dloss", "DLoss", "dloss", cast=str)
        _set_if_present("n_calibration", "N_Calibration", "n_calibration", cast=str)
        _set_if_present("n_positives", "NPos", "npos", "n_positives", cast=str)
        _set_if_present("n_negatives", "NNeg", "nneg", "n_negatives", cast=str)
        _set_if_present("n_neighbors", "N_Neighbors", "n_neighbors", cast=lambda v: int(float(str(v))))
        _set_if_present("prototypes_to_use", "Prototypes", "prototypes", "prototypes_to_use", cast=str)
        _set_if_present("prototype_strategy", "Proto_Strat", "prototype_strategy", cast=str)
        _set_if_present("prototype_components", "Proto_Comp", "prototype_components", cast=lambda v: int(float(str(v))))

        if head_name is not None:
            setattr(args, "head_name_selected", head_name)
            setattr(args, "learned_classifier_label", head_name)

        if head_config is not None:
            setattr(args, "best_classifier_config", str(head_config))
            setattr(args, "classification_head_config", str(head_config))
            setattr(args, "classifier_head_config", str(head_config))
            setattr(args, "head_config", str(head_config))
            head_meta = parse_classifier_config(head_config)
            setattr(args, "classification_head_family", head_meta.get("family"))

            if head_meta.get("family") == "prototype":
                setattr(args, "prototypes_to_use", "class")
                setattr(args, "prototype_strategy", str(head_meta.get("strategy", "mean")))
                setattr(args, "prototype_components", int(head_meta.get("components", 1)))
                setattr(args, "n_neighbors", 1)
            elif head_meta.get("family") == "knn":
                setattr(args, "n_neighbors", int(head_meta.get("k", getattr(args, "n_neighbors", 1))))
            elif head_meta.get("family") == "baseline":
                baseline_name = str(head_meta.get("name", ""))
                if baseline_name == "linear_svc":
                    setattr(args, "siamese_inference", "linearsvc")
                elif baseline_name in {"logreg", "logistic_regression"}:
                    setattr(args, "siamese_inference", "logisticregression")

        if head_n_aug is not None:
            try:
                setattr(args, "n_aug", int(float(str(head_n_aug).strip())))
            except Exception:
                setattr(args, "n_aug", head_n_aug)

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
