"""
Argument and configuration building for the Streamlit app.

This module handles all UI widget setup and argument namespace construction,
keeping app.py focused on core logic.
"""

import os
import argparse
import re
from typing import Optional

import streamlit as st
import pandas as pd

from otitenet.app.utils import (
    extract_params_from_log_path,
    _make_model_selection_key,
    _unique_preserve_order,
    attach_task_column,
    filter_models_df_by_task,
    get_split_mcc_metrics,
    get_model_split_config,
    is_done_manifest_model_row,
    format_classifier_config,
    parse_classifier_config,
    resolve_best_classifier_config,
    enumerate_classification_heads,
    ensure_int,
    normalize_train_datasets,
    split_config_key,
    split_config_segment,
)

from otitenet.app.database import set_production_model
from otitenet.app.utils_dataset_names import get_short_dataset_name, get_short_dataset_names
from otitenet.data.labels import label_scheme_for_task


def _available_dataset_paths(data_dir: str) -> list[str]:
    """Return dataset folders under data_dir, including nested folders with infos.csv."""
    out = []
    if not os.path.isdir(data_dir):
        return out
    for root, _dirs, files in os.walk(data_dir):
        if "infos.csv" not in files:
            continue
        rel = os.path.relpath(root, data_dir).replace("\\", "/")
        if rel != ".":
            out.append(rel)
    return sorted(out)


def _infer_new_size_from_dataset_path(dataset_path: str, default: int = 224) -> int:
    match = re.search(r"(?:^|/)otite_ds_(\d+)(?:/|$)", str(dataset_path or ""))
    if match:
        return int(match.group(1))
    try:
        return int(str(dataset_path).split("_")[-1])
    except Exception:
        return default


def _validate_and_resolve_dataset(params_dict: dict, data_dir: str = './data') -> dict:
    """Validate the Dataset key inside the params_dict, replacing with a valid folder if needed."""
    task = params_dict.get("Task") or params_dict.get("task") or "otite"
    model_name = params_dict.get("Model Name") or params_dict.get("model_name") or "default"
    dataset = params_dict.get("Dataset")
    
    # Strip any data/ prefix
    if isinstance(dataset, str) and (dataset.startswith("data/") or dataset.startswith("./data/")):
        dataset = dataset.replace("./data/", "").replace("data/", "")
        
    if dataset == "otite_ds_-1":
        dataset = None
        
    model_log_dir = os.path.join('logs/best_models', task, model_name)
    if os.path.isdir(model_log_dir):
        log_subdirs = sorted([d for d in os.listdir(model_log_dir) if os.path.isdir(os.path.join(model_log_dir, d)) and d != 'otite_ds_-1'])
        if log_subdirs:
            if not dataset or dataset not in log_subdirs:
                # Prefer otite_ds_224 if available as a robust fallback, else first available
                if 'otite_ds_224' in log_subdirs:
                    dataset = 'otite_ds_224'
                else:
                    dataset = log_subdirs[0]
                    
    available_datasets = _available_dataset_paths(data_dir)
    if not dataset or dataset not in available_datasets:
        valid_datasets = [d for d in available_datasets if d != 'otite_ds_-1']
        if valid_datasets:
            if 'otite_ds_224' in valid_datasets:
                dataset = 'otite_ds_224'
            else:
                dataset = valid_datasets[0]
        else:
            dataset = 'otite_ds_64'
            
    params_dict["Dataset"] = dataset
    return params_dict


def _args_namespace_from_model_row(model_row: dict) -> argparse.Namespace:
    """Build an args-like namespace from a Quick Model Selection / registry row."""
    row = dict(model_row)
    row.update(extract_params_from_log_path(row.get("Log Path") or row.get("log_path")))
    row = _validate_and_resolve_dataset(row)
    dataset = row.get("Dataset") or row.get("path") or "otite_ds_64"
    if isinstance(dataset, str) and not dataset.startswith("data/"):
        dataset = f"data/{dataset}"
    proto_comp = row.get("Proto_Comp") or row.get("prototype_components") or 1
    try:
        proto_comp = int(proto_comp)
    except (TypeError, ValueError):
        proto_comp = 1
    return argparse.Namespace(
        task=row.get("Task") or row.get("task") or "otite",
        model_name=row.get("Model Name") or row.get("model_name"),
        path=dataset,
        new_size=ensure_int(row.get("NSize") or row.get("new_size") or 64),
        fgsm=str(row.get("FGSM") or row.get("fgsm") or "0"),
        n_calibration=str(row.get("N_Calibration") or row.get("n_calibration") or "0"),
        classif_loss=str(row.get("Classif_Loss") or row.get("classif_loss") or "triplet"),
        dloss=str(row.get("DLoss") or row.get("dloss") or "triplet"),
        prototypes_to_use=str(row.get("Prototypes") or row.get("prototypes_to_use") or "class"),
        n_positives=str(row.get("NPos") or row.get("n_positives") or "1"),
        n_negatives=str(row.get("NNeg") or row.get("n_negatives") or "1"),
        normalize=str(row.get("Normalize") or row.get("normalize") or "no"),
        dist_fct=str(row.get("Dist_Fct") or row.get("dist_fct") or "euclidean"),
        n_neighbors=int(row.get("N_Neighbors") or row.get("n_neighbors") or 1),
        prototype_strategy=str(row.get("Proto_Strat") or row.get("prototype_strategy") or "mean"),
        prototype_components=proto_comp,
        train_datasets=str(row.get("train_datasets") or row.get("Train Datasets") or ""),
        valid_dataset=str(row.get("valid_dataset") or row.get("Valid Dataset") or ""),
        test_dataset=str(row.get("test_dataset") or row.get("Test Dataset") or ""),
        split_config_in_path=bool(row.get("_split_config_in_path") or row.get("Split Segment") or row.get("split_config_key")),
        _split_config_in_path=bool(row.get("_split_config_in_path") or row.get("Split Segment") or row.get("split_config_key")),
    )


def _render_classification_head_selector(model_row: dict, model_selection_key: str) -> Optional[str]:
    """Show classification-head selectbox for the current Quick Model Selection row."""
    tmp_args = _args_namespace_from_model_row(model_row)
    heads = enumerate_classification_heads(tmp_args)
    if not heads:
        st.caption("No learned-embedding heads cached. Run optimization in Tab 1.")
        return None

    # Always select the head with the highest Valid MCC as default
    best_head = max(heads, key=lambda h: h.get("valid_mcc", h.get("mcc", float('-inf'))))
    head_configs = [h["config"] for h in heads]
    head_labels = {
        h["config"]: f"{h['label']} (Valid MCC {h.get('valid_mcc', h.get('mcc', float('nan'))):.4f})" + (f" — {h['details']}" if h.get("details") else "")
        for h in heads
    }

    widget_key = f"sidebar_classification_head_{model_selection_key}"
    prev_model_key = st.session_state.get("sidebar_classification_head_model_key")
    if prev_model_key != model_selection_key:
        st.session_state["sidebar_classification_head_model_key"] = model_selection_key
        st.session_state[widget_key] = best_head["config"]

    current = st.session_state.get(widget_key)
    if current not in head_configs:
        st.session_state[widget_key] = best_head["config"]

    st.selectbox(
        "Classification head",
        options=head_configs,
        format_func=lambda c: head_labels.get(c, format_classifier_config(c)),
        key=widget_key,
        help="Learned-embedding classifier used for inference across tabs (when optimized inference is enabled).",
    )
    selected_cfg = st.session_state.get(widget_key, best_head["config"])
    selected_head = next((h for h in heads if h.get("config") == selected_cfg), best_head)
    st.session_state["sidebar_classification_head_config"] = selected_cfg
    st.session_state["sidebar_classification_head_n_aug"] = selected_head.get("n_aug")
    st.session_state["optimized_k_value"] = selected_cfg
    st.session_state["learned_classifier_label"] = format_classifier_config(selected_cfg)
    # Always show the best Valid MCC for the selected model in the sidebar
    st.markdown(f"**Best Valid MCC for this model:** {best_head.get('valid_mcc', best_head.get('mcc', float('nan'))):.4f}")
    return selected_cfg


from otitenet.app.ui_helpers import choose_dataset
from otitenet.utils.update_model_ranks import update_model_ranks


DEFAULT_APP_TRAIN_DATASETS = "Banque_Calaman_USA_2020_trie_CM,Banque_Comert_Turquie_2020_jpg"


def _split_combo_key_from_values(train_datasets, valid_dataset, test_dataset) -> str:
    return split_config_key(train_datasets, valid_dataset, test_dataset)


def _split_combo_key_from_row(row) -> str:
    if row is None:
        return ""
    getter = row.get if hasattr(row, "get") else lambda _key, _default=None: _default
    key = getter("split_config_key")
    if key and str(key).strip() not in {"", "None", "nan"}:
        return str(key).strip().replace(";", ",")
    return _split_combo_key_from_values(
        getter("train_datasets") or getter("Train Datasets") or "",
        getter("valid_dataset") or getter("Valid Dataset") or "",
        getter("test_dataset") or getter("Test Dataset") or "",
    )


def _split_combo_label(combo_key: str) -> str:
    train_datasets, valid_dataset, test_dataset = (str(combo_key or "").split("|") + ["", "", ""])[:3]
    train_label = get_short_dataset_names(train_datasets) or "unknown train"
    valid_label = get_short_dataset_name(valid_dataset) or "unknown valid"
    test_label = get_short_dataset_name(test_dataset) or "unknown test"
    return f"train: {train_label} | valid: {valid_label} | test: {test_label}"


def _split_combo_options(df: pd.DataFrame):
    if df is None or df.empty:
        return [], {}

    labels = {}
    options = []
    for _, row in df.iterrows():
        combo_key = _split_combo_key_from_row(row)
        if not combo_key.replace("|", "").strip():
            continue
        if combo_key not in labels:
            labels[combo_key] = _split_combo_label(combo_key)
            options.append(combo_key)
    return options, labels


def _safe_selectbox_sync(key, options, default_value=None):
    """Ensure a selectbox key in session state is valid for the current options."""
    current = st.session_state.get(key)
    if current not in options:
        if default_value in options:
            st.session_state[key] = default_value
        elif options:
            st.session_state[key] = options[0]
        else:
            st.session_state[key] = None


def _split_segment_from_params(params: dict) -> str:
    """Return the split path segment implied by selected model metadata, if any."""
    if not params:
        return ""
    explicit = params.get("Split Segment")
    if explicit and str(explicit).strip() not in {"", "None", "nan"}:
        return str(explicit).strip()
    split_key = params.get("split_config_key")
    if split_key and str(split_key).strip() not in {"", "None", "nan"}:
        train_datasets, valid_dataset, test_dataset = (str(split_key).replace(";", ",").split("|") + ["", "", ""])[:3]
        return split_config_segment(train_datasets, valid_dataset, test_dataset)
    if params.get("_split_config_in_path") or params.get("split_config_in_path"):
        return split_config_segment(
            params.get("train_datasets") or params.get("Train Datasets") or "",
            params.get("valid_dataset") or params.get("Valid Dataset") or "",
            params.get("test_dataset") or params.get("Test Dataset") or "",
        )
    return ""


def build_args_from_sidebar(cursor, conn, is_admin, data_dir='./data'):
    """Build args namespace by constructing sidebar UI and gathering selections.
    
    This function handles all UI widget logic for parameter selection, model ranking,
    and state synchronization.
    
    Args:
        cursor: Database cursor for querying best_models_registry
        is_admin: Whether the current user is an admin
        data_dir: Root data directory path (default: './data')
        
    Returns:
        argparse.Namespace with model and inference parameters
    """
    # For non-admin users, hide all sidebar controls below Current Optimization
    if not is_admin:
        # Return minimal args for non-admin
        return argparse.Namespace(
            model_name='default',
            fgsm='0',
            n_calibration='0',
            n_neighbors=1,
            dloss='triplet',
            prototypes_to_use='class',
            n_positives='1',
            n_negatives='1',
            device='cpu',
            task='otite',
            classif_loss='triplet',
            path=os.path.join(data_dir, 'otite_ds_64'),
            valid_dataset='Banque_Viscaino_Chili_2020',
            dist_fct='euclidean',
            grad_cam_layer=7,
            grad_cam_alpha=0.55,
            shap_layer=4,
            proto_strategies=['mean'],
            new_size=64,
            bs=32,
            groupkfold=1,
            random_recs=0,
            seed=42,
            normalize='no',
            train_datasets=DEFAULT_APP_TRAIN_DATASETS,
            test_dataset='Banque_Viscaino_Chili_2020',
            model_id=None,
            use_pretrained_encodings=True,
            use_trained_encoder=True,
            prototype_strategy='mean',
            prototype_components=1,
            best_classifier_config='1',
            learned_classifier_label='KNN k=1',
        )
    
    active_task = st.session_state.get("production_task")
    selected_params = st.session_state.get('selected_model_params') or {}
    selected_task = selected_params.get("Task") or selected_params.get("task")
    if active_task and selected_task and str(selected_task) != str(active_task):
        selected_params = {}
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
        ]:
            st.session_state.pop(key, None)
    selected_params_version = st.session_state.get('selected_params_version')
    last_synced_version = st.session_state.get('selected_params_last_sync')
    should_sync = bool(selected_params) and selected_params_version != last_synced_version
    if should_sync:
        st.session_state['selected_params_last_sync'] = selected_params_version

    def sync_value(key, value):
        if should_sync and value is not None:
            st.session_state[key] = value
    
    # ---- User selects paths ---- #
    with st.sidebar:
        st.header("Model Parameters")

        # ---- Predefine model_dir, model_name, dataset_segment for use in Quick Model Selection ---- #

        # Only create the task selectbox ONCE at the top of the sidebar
        task_root = 'logs/best_models'
        task_list = sorted(os.listdir(task_root)) if os.path.isdir(task_root) else []
        if not task_list:
            st.error("No tasks found under logs/best_models.")
            st.stop()
        task_default = selected_params.get("Task") or active_task
        sync_value('task_selectbox', task_default)
        _safe_selectbox_sync('task_selectbox', task_list, task_default)
        task = st.selectbox("task", task_list, key="task_selectbox")
        if task and task != st.session_state.get("production_task"):
            st.session_state["production_task"] = task
            st.session_state["label_scheme"] = label_scheme_for_task(task)
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
            st.rerun()

        model_dir = os.path.join(task_root, task)
        model_name_list = [name for name in sorted(os.listdir(model_dir)) if not name.endswith('.csv')] if os.path.isdir(model_dir) else []
        if not model_name_list:
            st.error(f"No models found for task {task}.")
            st.stop()
        model_name_default = selected_params.get("Model Name")
        # remove files ending in .json or any extension really
        model_name_list = [name for name in model_name_list if not any(name.endswith(ext) for ext in ['.json', '.csv', '.txt'])]
        sync_value('model_name_selectbox', model_name_default)
        _safe_selectbox_sync('model_name_selectbox', model_name_list, model_name_default)
        model_name = st.selectbox("Model Name", model_name_list, key="model_name_selectbox")

        # Determine available dataset folders for this model
        model_dataset_base = os.path.join(model_dir, model_name)
        available_datasets = [d for d in os.listdir(model_dataset_base)
                             if os.path.isdir(os.path.join(model_dataset_base, d)) and d.startswith('otite_ds_')]
        # Default dataset from metadata or fallback
        default_ds = selected_params.get('Dataset')
        if default_ds not in available_datasets:
            default_ds = available_datasets[0] if available_datasets else None
        selected_path = choose_dataset("Select --path dataset", available_datasets,
                                      default=default_ds, key="dataset_selectbox")
        dataset_segment = selected_path if selected_path else 'otite_ds_64'

        # ---- Model Selection from Leaderboard ---- #
        st.subheader("📋 Quick Model Selection")
        try:
            model_rows = []
            use_db_rank = False
            # Try to use model_rank if available
            try:
                update_model_ranks()
                cursor.execute(
                    """
                    SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
                           n_calibration, accuracy, mcc, normalize, n_neighbors, log_path, model_rank,
                           prototype_strategy, prototype_components,
                           train_datasets, valid_dataset, test_dataset, split_config_key
                    FROM best_models_registry
                    WHERE model_rank IS NOT NULL
                    ORDER BY model_rank ASC
                    """
                )
                model_rows = cursor.fetchall()
                use_db_rank = True
            except Exception:
                # Fallback if model_rank column doesn't exist yet
                try:
                    cursor.execute(
                        """
                        SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
                               n_calibration, accuracy, mcc, normalize, n_neighbors, log_path,
                               prototype_strategy, prototype_components,
                               train_datasets, valid_dataset, test_dataset, split_config_key
                        FROM best_models_registry
                        ORDER BY mcc DESC
                        """
                    )
                    model_rows = cursor.fetchall()
                except Exception:
                    cursor.execute(
                        """
                        SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
                               n_calibration, accuracy, mcc, normalize, n_neighbors, log_path,
                               prototype_strategy, prototype_components
                        FROM best_models_registry
                        ORDER BY mcc DESC
                        """
                    )
                    model_rows = [tuple(list(row) + [None, None, None, None]) for row in (cursor.fetchall() or [])]
                use_db_rank = False

            if model_rows:
                # Align columns with the database query (19 columns)
                if use_db_rank:
                    _cols = [
                        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                        "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", 
                        "#", "Proto_Strat", "Proto_Comp", "train_datasets", "valid_dataset", "test_dataset", "split_config_key"
                    ]
                else:
                    _cols = [
                        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                        "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path",
                        "Proto_Strat", "Proto_Comp", "train_datasets", "valid_dataset", "test_dataset", "split_config_key"
                    ]
                _df = pd.DataFrame(model_rows, columns=_cols)
                for idx, row in _df.iterrows():
                    split_config = get_model_split_config(row.get("Log Path"))
                    for col in ["train_datasets", "valid_dataset", "test_dataset", "split_config_key"]:
                        if col in _df.columns and (pd.isna(row.get(col)) or str(row.get(col)).strip() in {"", "None", "nan"}):
                            _df.at[idx, col] = split_config.get(col)
                _df = attach_task_column(_df)
                _df = filter_models_df_by_task(_df, active_task)
                _df = _df[_df.apply(is_done_manifest_model_row, axis=1)].copy()
                
                if not _df.empty:
                    # Instead of using only registry splits, list all subfolders under otite_ds_64
                    dataset_base = os.path.join(model_dir, model_name, dataset_segment)
                    if os.path.isdir(dataset_base):
                        all_split_folders = [d for d in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, d))]
                    else:
                        all_split_folders = []
                    split_options = all_split_folders
                    split_labels = {d: d for d in all_split_folders}
                    selected_split_key = None
                    if split_options:
                        current_split_key = st.session_state.get("sidebar_split_combo_key")
                        default_split_key = current_split_key if current_split_key in split_options else split_options[0]
                        if st.session_state.get("sidebar_split_combo_key") not in split_options:
                            st.session_state["sidebar_split_combo_key"] = default_split_key

                        selected_split_key = st.selectbox(
                            "Datasets",
                            options=split_options,
                            format_func=lambda key: split_labels.get(key, key),
                            key="sidebar_split_combo_key",
                            help="All available split subfolders under otite_ds_64.",
                        )

                        previous_split_key = st.session_state.get("sidebar_split_combo_last_key")
                        if previous_split_key is not None and previous_split_key != selected_split_key:
                            selected_params = {}
                            should_sync = False
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
                                "sidebar_classification_head_n_aug",
                            ]:
                                st.session_state.pop(key, None)
                        st.session_state["sidebar_split_combo_last_key"] = selected_split_key

                        _df["_split_combo_key"] = _df.apply(_split_combo_key_from_row, axis=1)
                        _df = _df[_df["_split_combo_key"] == selected_split_key].drop(columns=["_split_combo_key"])
                        if _df.empty:
                            st.info("No models found for the selected train/valid/test dataset combination.")
                    else:
                        st.warning(
                            "No train_datasets metadata found for these registry rows yet. "
                            "New training runs will record it; older runs need run_metadata.json/run_summary.json with split_config."
                        )

                if not _df.empty:
                    _group_cols = [
                        "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
                        "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize", "N_Neighbors",
                        "train_datasets", "valid_dataset", "test_dataset",
                    ]
                    _dedupe_frame = _df[_group_cols].copy().fillna("").astype(str)
                    # Use row-wise join to ensure a Series (avoids pandas agg returning a DataFrame)
                    _df["_dedupe_key"] = _dedupe_frame.apply(lambda r: "|".join(r.values.tolist()), axis=1)
                    _df = _df.sort_values("MCC", ascending=False)
                    _df = _df.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"])
                    _df = _df.dropna(subset=["Log Path"])
                    _df = _df[_df["Log Path"].astype(str) != ""]
                    _df = _df.reset_index(drop=True)
                    
                    # Attach split metrics so ranking can use validation MCC
                    _df["Valid MCC"] = pd.NA
                    for idx, row in _df.iterrows():
                        split_metrics = get_split_mcc_metrics(row.get("Log Path"))
                        if split_metrics:
                            _df.at[idx, "Valid MCC"] = split_metrics.get('valid_mcc', pd.NA)

                    # Sort by validation MCC first (fallback to MCC), then re-number rows
                    valid_sort = pd.to_numeric(_df.get("Valid MCC"), errors="coerce")
                    mcc_sort = pd.to_numeric(_df.get("MCC"), errors="coerce")
                    _df = _df.assign(
                        _valid_sort=valid_sort.fillna(float('-inf')),
                        _mcc_sort=mcc_sort.fillna(float('-inf')),
                    )
                    _df = _df.sort_values(["_valid_sort", "_mcc_sort"], ascending=[False, False]).reset_index(drop=True)
                    _df = _df.drop(columns=["_valid_sort", "_mcc_sort"])

                    _df["#"] = range(1, len(_df) + 1)
                    cols = ["#"] + [col for col in _df.columns if col != "#"]
                    _df = _df[cols]

                    key_to_row = {}
                    key_to_label = {}
                    model_number_map = {}
                    for _, r in _df.iterrows():
                        rd = r.to_dict()
                        selection_key = _make_model_selection_key(rd)
                        if selection_key in key_to_row:
                            continue
                        key_to_row[selection_key] = rd
                        model_num = rd.get("#", "?")
                        model_number_map[selection_key] = model_num
                        key_to_label[selection_key] = (
                            f"#{model_num}: {rd.get('Model ID')}"
                        )
                    st.session_state['model_number_map'] = model_number_map
                else:
                    st.info("No models in Quick Model Selection match a manifest row with job_state='done'.")
                    st.session_state['model_number_map'] = {}
                    key_to_row = {}
                    key_to_label = {}

                # Auto-apply the top-ranked model so defaults come from Quick Model Selection
                if not selected_params and len(_df) > 0:
                    try:
                        best_row = _df.iloc[0].to_dict()
                        best_row.update(extract_params_from_log_path(best_row.get("Log Path")))
                        best_row = _validate_and_resolve_dataset(best_row, data_dir)
                        best_row["Task"] = best_row.get("Task") or active_task
                        if "N_Neighbors" in best_row:
                            best_row["n_neighbors"] = best_row["N_Neighbors"]
                        if "NSize" in best_row:
                            best_row["new_size"] = best_row["NSize"]
                        if "Dist_Fct" in best_row:
                            best_row["dist_fct"] = best_row["Dist_Fct"]
                        if "Classif_Loss" in best_row:
                            best_row["classif_loss"] = best_row["Classif_Loss"]
                        best_row["model_id"] = best_row.get("Model ID")

                        selected_params = best_row
                        st.session_state.selected_model_params = best_row
                        st.session_state.selected_params_version = st.session_state.get('selected_params_version', 0) + 1
                        st.session_state.selected_model_log_path = best_row.get('Log Path')
                        best_key = _make_model_selection_key(best_row)
                        st.session_state.selected_model_selection_key = best_key
                        st.session_state.selected_model_version = st.session_state.get('selected_model_version', 0) + 1
                        st.session_state.sidebar_best_model_key = best_key
                        should_sync = True
                        # Classification head widget renders after model selectbox
                    except Exception as auto_exc:
                        st.warning(f"Could not auto-apply best model: {auto_exc}")

                sidebar_options = _unique_preserve_order(list(key_to_row.keys()))

                # Sync this widget to the canonical selection (log_path)
                canonical_key = st.session_state.get('selected_model_selection_key')
                if canonical_key and canonical_key in sidebar_options:
                    last = st.session_state.get('sidebar_best_model_last_sync')
                    ver = st.session_state.get('selected_model_version')
                    if ver is not None and ver != last:
                        st.session_state['sidebar_best_model_key'] = canonical_key
                        st.session_state['sidebar_best_model_last_sync'] = ver

                # Ensure the selectbox value is valid before rendering to avoid "not in iterable" errors.
                current_sidebar_key = st.session_state.get('sidebar_best_model_key')
                if current_sidebar_key not in sidebar_options:
                    st.session_state['sidebar_best_model_key'] = sidebar_options[0] if sidebar_options else None

                selected_key = st.selectbox(
                    "Select from best models:",
                    options=sidebar_options,
                    format_func=lambda k: key_to_label.get(k, str(k)),
                    index=0,
                    key="sidebar_best_model_key",
                )

                if selected_key and selected_key in key_to_row:
                    try:
                        _render_classification_head_selector(key_to_row[selected_key], selected_key)
                    except Exception as clf_exc:
                        st.warning(f"Could not load classification heads: {clf_exc}")

                if selected_key and st.button("✅ Apply Selected Model", key="apply_sidebar_model"):
                    rd = key_to_row.get(selected_key)
                    if rd:
                        model_dict = {
                            'Model ID': rd.get('Model ID'),
                            'model_id': rd.get('Model ID'),
                            'Model Name': rd.get('Model Name'),
                            'NSize': rd.get('NSize'),
                            'FGSM': rd.get('FGSM'),
                            'Prototypes': rd.get('Prototypes'),
                            'NPos': rd.get('NPos'),
                            'NNeg': rd.get('NNeg'),
                            'DLoss': rd.get('DLoss'),
                            'Dist_Fct': rd.get('Dist_Fct'),
                            'Classif_Loss': rd.get('Classif_Loss'),
                            'N_Calibration': rd.get('N_Calibration'),
                            'Accuracy': rd.get('Accuracy'),
                            'MCC': rd.get('MCC'),
                            'Normalize': rd.get('Normalize'),
                            'N_Neighbors': rd.get('N_Neighbors'),
                            'prototype_strategy': rd.get('Proto_Strat'),
                            'prototype_components': rd.get('Proto_Comp'),
                            'train_datasets': rd.get('train_datasets'),
                            'valid_dataset': rd.get('valid_dataset'),
                            'test_dataset': rd.get('test_dataset'),
                            'split_config_key': rd.get('split_config_key'),
                            'Log Path': rd.get('Log Path'),
                        }
                        model_dict.update(extract_params_from_log_path(model_dict.get("Log Path")))
                        model_dict = _validate_and_resolve_dataset(model_dict, data_dir)
                        model_dict["Task"] = model_dict.get("Task") or active_task
                        if "N_Neighbors" in model_dict:
                            model_dict["n_neighbors"] = model_dict["N_Neighbors"]
                        if "NSize" in model_dict:
                            model_dict["new_size"] = model_dict["NSize"]
                        if "Dist_Fct" in model_dict:
                            model_dict["dist_fct"] = model_dict["Dist_Fct"]
                        if "Classif_Loss" in model_dict:
                            model_dict["classif_loss"] = model_dict["Classif_Loss"]

                        st.session_state.selected_model_params = model_dict
                        st.session_state.selected_params_version = st.session_state.get('selected_params_version', 0) + 1
                        st.session_state.selected_model_log_path = model_dict.get('Log Path')
                        st.session_state.selected_model_selection_key = selected_key
                        st.session_state.selected_model_version = st.session_state.get('selected_model_version', 0) + 1

                        # Eagerly sync key UI widgets so the next run reflects selection immediately
                        try:
                            _ds_val = model_dict.get('Dataset')
                            _all_ds = [d for d in _available_dataset_paths(data_dir) if d != 'otite_ds_-1']
                            if _ds_val and _ds_val in _all_ds:
                                st.session_state['dataset_selectbox'] = _ds_val
                            elif _all_ds:
                                # Resolved dataset not in session yet — wipe stale value so choose_dataset picks correctly
                                st.session_state.pop('dataset_selectbox', None)
                        except Exception:
                            pass
                        try:
                            if model_dict.get('Task'):
                                st.session_state['task_selectbox'] = model_dict['Task']
                        except Exception:
                            pass
                        try:
                            if model_dict.get('new_size') is not None:
                                st.session_state['new_size_input'] = int(float(model_dict['new_size']))
                        except Exception:
                            pass
                        try:
                            if model_dict.get('n_neighbors') is not None:
                                st.session_state['n_neighbors_input'] = int(float(model_dict['n_neighbors']))
                        except Exception:
                            pass
                        try:
                            if model_dict.get('Dist_Fct'):
                                st.session_state['dist_fct_selectbox'] = str(model_dict.get('Dist_Fct')).strip().lower()
                        except Exception:
                            pass
                        # Other selectors (best-effort; choose_dataset etc. will validate on rerun)
                        for key_name, dict_key in [
                            ('model_name_selectbox', 'Model Name'),
                            ('fgsm_selectbox', 'FGSM'),
                            ('n_calibration_selectbox', 'N_Calibration'),
                            ('classif_loss_selectbox', 'Classif_Loss'),
                            ('dloss_selectbox', 'DLoss'),
                            ('prototypes_selectbox', 'Prototypes'),
                            ('npos_selectbox', 'NPos'),
                            ('nneg_selectbox', 'NNeg'),
                        ]:
                            try:
                                v = model_dict.get(dict_key)
                                if v is not None:
                                    st.session_state[key_name] = v
                            except Exception:
                                pass
                        try:
                            if model_dict.get('Normalize') in ['yes', 'no']:
                                st.session_state['normalize_input'] = model_dict.get('Normalize')
                        except Exception:
                            pass
                        st.success(f"✅ Applied: {model_dict.get('Model Name')}")
                        st.rerun()
                
                # Admin: Production model control (outside the Apply button so it persists)
                if is_admin and selected_key:
                    st.divider()
                    rd = key_to_row.get(selected_key)
                    if rd:
                        if st.button("🚀 Set as Production Model", key="set_production_model"):
                            production_task = active_task or rd.get("Task") or "notNormal"
                            model_dict = {
                                'label_task': production_task,
                                'label_scheme': label_scheme_for_task(production_task),
                                'Model ID': rd.get('Model ID'),
                                'model_id': rd.get('Model ID'),
                                'Model Name': rd.get('Model Name'),
                                'NSize': rd.get('NSize'),
                                'FGSM': rd.get('FGSM'),
                                'Prototypes': rd.get('Prototypes'),
                                'NPos': rd.get('NPos'),
                                'NNeg': rd.get('NNeg'),
                                'DLoss': rd.get('DLoss'),
                                'Dist_Fct': rd.get('Dist_Fct'),
                                'Classif_Loss': rd.get('Classif_Loss'),
                                'N_Calibration': rd.get('N_Calibration'),
                                'Accuracy': rd.get('Accuracy'),
                                'MCC': rd.get('MCC'),
                                'Normalize': rd.get('Normalize'),
                                'N_Neighbors': rd.get('N_Neighbors'),
                                'prototype_strategy': rd.get('Proto_Strat'),
                                'prototype_components': rd.get('Proto_Comp'),
                                'train_datasets': rd.get('train_datasets'),
                                'valid_dataset': rd.get('valid_dataset'),
                                'test_dataset': rd.get('test_dataset'),
                                'split_config_key': rd.get('split_config_key'),
                                'Log Path': rd.get('Log Path'),
                                'model_number': st.session_state.model_number_map.get(selected_key, '?'),
                            }
                            selected_head_model_key = st.session_state.get('sidebar_classification_head_model_key')
                            selected_head_config = None
                            if selected_head_model_key == selected_key:
                                selected_head_config = st.session_state.get('sidebar_classification_head_config')
                            if not selected_head_config:
                                selected_head_config = "baseline_linear_svc"

                            selected_head_label = format_classifier_config(selected_head_config)
                            selected_head_meta = parse_classifier_config(selected_head_config)
                            model_dict['Head Config'] = str(selected_head_config)
                            model_dict['head_config'] = str(selected_head_config)
                            model_dict['classification_head_config'] = str(selected_head_config)
                            model_dict['best_classifier_config'] = str(selected_head_config)
                            model_dict['Head'] = selected_head_label
                            model_dict['head_name'] = selected_head_label
                            model_dict['learned_classifier_label'] = selected_head_label
                            model_dict['head_family'] = selected_head_meta.get('family')
                            selected_head_n_aug = st.session_state.get('sidebar_classification_head_n_aug')
                            if selected_head_n_aug is not None:
                                model_dict['N Aug'] = selected_head_n_aug
                                model_dict['n_aug'] = selected_head_n_aug
                                model_dict['head_n_aug'] = selected_head_n_aug

                            # Copy all relevant fields from selected_head_meta
                            for key, value in selected_head_meta.items():
                                if value is not None:
                                    model_dict[key] = value

                            # Allow any head to be set as production model; deployment script enforces restriction
                            set_by_email = st.session_state.get('user_email', 'unknown')
                            if set_production_model(cursor, conn, model_dict, set_by_email):
                                st.session_state['production_model'] = model_dict
                                st.success("✅ Model set as production model for this labeling scenario")
                            else:
                                st.error("❌ Failed to save production model to database")
                        
                        # Show current production model
                        if 'production_model' in st.session_state:
                            prod = st.session_state['production_model']
                            prod_head = (
                                prod.get('Head')
                                or prod.get('head_name')
                                or prod.get('learned_classifier_label')
                                or format_classifier_config(
                                    prod.get('Head Config')
                                    or prod.get('head_config')
                                    or prod.get('classification_head_config')
                                    or prod.get('best_classifier_config')
                                )
                            )
                            if not prod_head or str(prod_head) in {'None', 'nan', '—'}:
                                prod_head = 'head not stored yet'
                            st.info(f"🎯 **Current Production Model:** {prod.get('Model Name')} (#{prod.get('model_number', '?')}) - {prod_head}")
                        else:
                            st.warning("⚠️ No production model set")
        except Exception as e:
            st.warning(f"Could not load models: {e}")


        st.divider()

        # nsize selectbox is rendered after dataset + split selection (see below)

        # ---- Build model_dataset_dir: task/model/dataset/[split]/nsize{N} ---- #
        # Always use the selected split subfolder (from sidebar_split_combo_key) after dataset_segment
        sidebar_split_key = st.session_state.get("sidebar_split_combo_key")
        if sidebar_split_key:
            model_dataset_dir = os.path.join(model_dir, model_name, dataset_segment, sidebar_split_key)
            split_segment = sidebar_split_key
        else:
            # Fallback: use first available split subfolder if present
            dataset_base = os.path.join(model_dir, model_name, dataset_segment)
            split_folders = [d for d in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, d))]
            if split_folders:
                model_dataset_dir = os.path.join(dataset_base, split_folders[0])
                split_segment = split_folders[0]
            else:
                model_dataset_dir = dataset_base
                split_segment = None

        # ---- nsize selectbox: scan filesystem for available nsize dirs ---- #
        available_nsize_dirs = sorted([
            d for d in os.listdir(model_dataset_dir)
            if os.path.isdir(os.path.join(model_dataset_dir, d)) and d.startswith('nsize')
        ]) if os.path.isdir(model_dataset_dir) else []
        available_sizes = []
        for d in available_nsize_dirs:
            try:
                available_sizes.append(int(d[len('nsize'):]))
            except ValueError:
                pass
        available_sizes = sorted(available_sizes)

        if available_sizes:
            # Default from DB / selected params; fallback to first available
            nsize_default = selected_params.get('new_size') or selected_params.get('NSize')
            try:
                nsize_default = int(nsize_default) if nsize_default is not None else None
            except (TypeError, ValueError):
                nsize_default = None
            if nsize_default not in available_sizes:
                nsize_default = available_sizes[0]
            sync_value('nsize_selectbox', nsize_default)
            _safe_selectbox_sync('nsize_selectbox', available_sizes, nsize_default)
            new_size = st.selectbox("new_size (nsize)", available_sizes, key="nsize_selectbox")
        else:
            # No nsize dirs found – fall back to inferring from dataset name
            new_size = _infer_new_size_from_dataset_path(selected_path)
            st.caption(f"No nsize directories found; inferred new_size={new_size}")

        # Navigate into nsize dir
        nsize_subdir = os.path.join(model_dataset_dir, f"nsize{new_size}")
        if os.path.isdir(nsize_subdir):
            model_dataset_dir = nsize_subdir

        # ---- FGSM selection ---- #
        # Look for fgsm* directories under model_dataset_dir (now inside nsize dir)
        possible_fgsm = sorted([
            d for d in os.listdir(model_dataset_dir)
            if os.path.isdir(os.path.join(model_dataset_dir, d)) and d.startswith('fgsm')
        ]) if os.path.isdir(model_dataset_dir) else []
        if possible_fgsm:
            fgsm_list = possible_fgsm
        else:
            st.warning(f"No FGSM entries found for model {model_name} in {model_dataset_dir}.")
            st.stop()

        fgsm_default = selected_params.get("FGSM")
        sync_value('fgsm_selectbox', fgsm_default)
        _safe_selectbox_sync('fgsm_selectbox', fgsm_list, fgsm_default)
        fgsm = st.selectbox("fgsm", fgsm_list, key="fgsm_selectbox")

        n_cal_dir = os.path.join(model_dataset_dir, fgsm)
        n_calibration_list = sorted(os.listdir(n_cal_dir)) if os.path.isdir(n_cal_dir) else []
        if not n_calibration_list:
            st.error(f"No calibration folders found in {n_cal_dir}.")
            st.stop()
        n_calibration_default = selected_params.get("N_Calibration")
        sync_value('n_calibration_selectbox', n_calibration_default)
        _safe_selectbox_sync('n_calibration_selectbox', n_calibration_list, n_calibration_default)
        n_calibration = st.selectbox("n_calibration", n_calibration_list, key="n_calibration_selectbox")

        classif_dir = os.path.join(n_cal_dir, n_calibration)
        classif_loss_list = sorted(os.listdir(classif_dir)) if os.path.isdir(classif_dir) else []
        if not classif_loss_list:
            st.error(f"No classif_loss folders found in {classif_dir}.")
            st.stop()
        classif_loss_default = selected_params.get("classif_loss") or selected_params.get("Classif_Loss")
        sync_value('classif_loss_selectbox', classif_loss_default)
        _safe_selectbox_sync('classif_loss_selectbox', classif_loss_list, classif_loss_default)
        classif_loss = st.selectbox("classif_loss", classif_loss_list, key="classif_loss_selectbox")

        dloss_dir = os.path.join(classif_dir, classif_loss)
        dloss_list = sorted(os.listdir(dloss_dir)) if os.path.isdir(dloss_dir) else []
        if not dloss_list:
            st.error(f"No dloss folders found in {dloss_dir}.")
            st.stop()
        dloss_default = selected_params.get("DLoss")
        sync_value('dloss_selectbox', dloss_default)
        _safe_selectbox_sync('dloss_selectbox', dloss_list, dloss_default)
        dloss = st.selectbox("dloss", dloss_list, key="dloss_selectbox")

        proto_dir = os.path.join(dloss_dir, dloss)
        prototypes_list = sorted(os.listdir(proto_dir)) if os.path.isdir(proto_dir) else []
        if not prototypes_list:
            st.error(f"No prototype folders found in {proto_dir}.")
            st.stop()

        prototypes_default = selected_params.get("Prototypes")

        if prototypes_default is not None:
            prototypes_default = str(prototypes_default)

            if prototypes_default not in prototypes_list:
                prefixed = f"prototypes_{prototypes_default}"
                if prefixed in prototypes_list:
                    prototypes_default = prefixed

        sync_value('prototypes_selectbox', prototypes_default)
        _safe_selectbox_sync('prototypes_selectbox', prototypes_list, prototypes_default)
        prototypes_to_use = st.selectbox("prototypes_to_use", prototypes_list, key="prototypes_selectbox")
        # Prototype classification strategies
        st.markdown("**Prototype Strategies**")
        proto_strategies = st.multiselect(
            "Use prototype classification with strategies:",
            options=['mean', 'kmeans', 'gmm'],
            default=['mean', 'kmeans', 'gmm'],
            key='proto_strategies_checkbox',
            help="Select strategies to compute prototype-based predictions. Empty = KNN only."
        )

        npos_dir = os.path.join(proto_dir, prototypes_to_use)
        n_positives_list = sorted(os.listdir(npos_dir)) if os.path.isdir(npos_dir) else []
        if not n_positives_list:
            st.error(f"No npos folders found in {npos_dir}.")
            st.stop()
        npos_default = str(selected_params.get("NPos")) if selected_params.get("NPos") is not None else None
        sync_value('npos_selectbox', npos_default)
        _safe_selectbox_sync('npos_selectbox', n_positives_list, npos_default)
        n_positives = st.selectbox('npos', n_positives_list, key="npos_selectbox")

        nneg_dir = os.path.join(npos_dir, n_positives)
        n_negatives_list = sorted(os.listdir(nneg_dir)) if os.path.isdir(nneg_dir) else []
        if not n_negatives_list:
            st.error(f"No nneg folders found in {nneg_dir}.")
            st.stop()
        nneg_default = str(selected_params.get("NNeg")) if selected_params.get("NNeg") is not None else None
        sync_value('nneg_selectbox', nneg_default)
        _safe_selectbox_sync('nneg_selectbox', n_negatives_list, nneg_default)
        n_negatives = st.selectbox('nneg', n_negatives_list, key="nneg_selectbox")

        n_neighbors_default = selected_params.get("n_neighbors")
        if should_sync and n_neighbors_default is not None:
            try:
                st.session_state['n_neighbors_input'] = int(n_neighbors_default)
            except (TypeError, ValueError):
                st.session_state['n_neighbors_input'] = 1
        n_neighbors_default_value = st.session_state.get('n_neighbors_input', 1)
        n_neighbors = st.number_input("n_neighbors", value=int(n_neighbors_default_value), step=1, key="n_neighbors_input")

        normalize_default = selected_params.get("Normalize", "no")
        if should_sync and normalize_default is not None:
            st.session_state['normalize_input'] = normalize_default
        _safe_selectbox_sync('normalize_input', ['yes', 'no'], normalize_default)
        normalize = st.selectbox("normalize", ['yes', 'no'], key="normalize_input")

        device = st.selectbox("device", ['cpu', 'cuda'], index=1, key="device_selectbox")
        valid_dataset_default = selected_params.get('valid_dataset', 'Banque_Viscaino_Chili_2020')
        if should_sync and valid_dataset_default is not None:
            st.session_state['valid_dataset_input'] = valid_dataset_default
        valid_dataset = st.text_input("valid_dataset", value=st.session_state.get('valid_dataset_input', valid_dataset_default), key="valid_dataset_input")
        train_datasets_default = selected_params.get('train_datasets', DEFAULT_APP_TRAIN_DATASETS)
        if should_sync and train_datasets_default is not None:
            st.session_state['train_datasets_input'] = str(train_datasets_default)
        train_datasets = st.text_input(
            "train_datasets",
            value=st.session_state.get('train_datasets_input', str(train_datasets_default or "")),
            key="train_datasets_input",
            disabled=True,
            help="Chosen from existing saved train/valid/test combinations above.",
        )
        test_dataset_default = selected_params.get('test_dataset', valid_dataset)
        if should_sync and test_dataset_default is not None:
            st.session_state['test_dataset_input'] = str(test_dataset_default)
        test_dataset = st.text_input(
            "test_dataset",
            value=st.session_state.get('test_dataset_input', str(test_dataset_default or "")),
            key="test_dataset_input",
            disabled=True,
            help="Chosen from existing saved train/valid/test combinations above.",
        )
        dist_fct_options = ['euclidean', 'cosine']
        dist_fct_default = str(selected_params.get('dist_fct') or selected_params.get('Dist_Fct') or 'euclidean').strip().lower()
        if dist_fct_default not in dist_fct_options:
            dist_fct_default = 'euclidean'

        # If Streamlit has a stale value for this widget, it can crash with
        # "<value> is not in iterable" during serialization. Repair it before rendering.
        current_dist_fct = st.session_state.get('dist_fct_selectbox')
        if current_dist_fct is not None:
            current_dist_fct = str(current_dist_fct).strip().lower()
        if current_dist_fct not in dist_fct_options:
            st.session_state['dist_fct_selectbox'] = dist_fct_default

        if should_sync and dist_fct_default is not None:
            st.session_state['dist_fct_selectbox'] = dist_fct_default

        dist_fct = st.selectbox(
            "dist_fct",
            dist_fct_options,
            index=dist_fct_options.index(st.session_state['dist_fct_selectbox']),
            key="dist_fct_selectbox",
        )

        # Explanation parameters are fixed here to avoid cluttering the sidebar UI.
        grad_cam_layer = int(st.session_state.get('grad_cam_layer', 7))
        grad_cam_alpha = float(st.session_state.get('grad_cam_alpha', 0.55))
        shap_layer = int(st.session_state.get('shap_layer', 4))

        st.session_state['grad_cam_layer'] = grad_cam_layer
        st.session_state['grad_cam_alpha'] = grad_cam_alpha
        st.session_state['shap_layer'] = shap_layer

    # ---- Build args Namespace ---- #
    # Build the correct path including the split subfolder if present
    sidebar_split_key = st.session_state.get("sidebar_split_combo_key")
    if selected_path:
        if sidebar_split_key:
            path_with_split = os.path.join(data_dir, selected_path, sidebar_split_key)
        else:
            path_with_split = os.path.join(data_dir, selected_path)
    else:
        path_with_split = None

    args = argparse.Namespace(
        model_name=model_name,
        fgsm=fgsm,
        n_calibration=n_calibration,
        n_neighbors=n_neighbors,
        dloss=dloss,
        prototypes_to_use=prototypes_to_use,
        n_positives=n_positives,
        n_negatives=n_negatives,
        device=device,
        task=task,
        classif_loss=classif_loss,
        path=path_with_split,
        valid_dataset=valid_dataset,
        dist_fct=dist_fct,
        grad_cam_layer=grad_cam_layer,
        grad_cam_alpha=grad_cam_alpha,
        shap_layer=shap_layer,
        proto_strategies=proto_strategies
    )

    # Fixed parameters (TODO: Remove these later)
    args.new_size = new_size
    args.bs = 32
    args.groupkfold = 1
    args.random_recs = 0
    args.seed = 42
    args.normalize = normalize
    args.train_datasets = str(train_datasets or selected_params.get('train_datasets', DEFAULT_APP_TRAIN_DATASETS) or DEFAULT_APP_TRAIN_DATASETS)
    args.test_dataset = str(test_dataset or selected_params.get('test_dataset', valid_dataset) or valid_dataset)
    args.split_config_in_path = bool(
        selected_params.get('_split_config_in_path')
        or selected_params.get('Split Segment')
        or selected_params.get('split_config_key')
    )
    args._split_config_in_path = args.split_config_in_path
    args.model_id = selected_params.get('model_id') or selected_params.get('Model ID')
    args.log_path = (
        selected_params.get('Log Path')
        or selected_params.get('log_path')
        or st.session_state.get('selected_model_log_path')
        or ''
    )
    args.use_pretrained_encodings = bool(st.session_state.get('use_saved_encodings_tab1', True))
    args.use_trained_encoder = args.use_pretrained_encodings
    args.siamese_inference = str(selected_params.get("siamese_inference") or "linearsvc")

    args.prototype_strategy = str(
        selected_params.get('Proto_Strat')
        or selected_params.get('prototype_strategy')
        or 'mean'
    )
    try:
        args.prototype_components = int(
            selected_params.get('Proto_Comp')
            or selected_params.get('prototype_components')
            or 1
        )
    except (TypeError, ValueError):
        args.prototype_components = 1

    selected_head_model_key = st.session_state.get("sidebar_classification_head_model_key")
    current_model_key = st.session_state.get("selected_model_selection_key")
    selected_head = None
    if selected_head_model_key and current_model_key and selected_head_model_key == current_model_key:
        selected_head = st.session_state.get("sidebar_classification_head_config")
    if selected_head:
        args.best_classifier_config = str(selected_head)
    else:
        if args.siamese_inference == "linearsvc":
            args.best_classifier_config = "baseline_linear_svc"
        elif args.siamese_inference == "logisticregression":
            args.best_classifier_config = "baseline_logreg"
        else:
            args.best_classifier_config = resolve_best_classifier_config(args, use_optimized=True)
    args.learned_classifier_label = format_classifier_config(args.best_classifier_config)
    head_meta = parse_classifier_config(args.best_classifier_config)
    args.classification_head_family = head_meta.get("family")
    if head_meta.get("family") == "prototype":
        args.prototypes_to_use = "class"
        args.prototype_strategy = str(head_meta.get("strategy", args.prototype_strategy))
        args.prototype_components = int(head_meta.get("components", args.prototype_components))
    elif head_meta.get("family") == "knn":
        args.n_neighbors = int(head_meta.get("k", args.n_neighbors))
        args.siamese_inference = "knn"
    elif head_meta.get("family") == "baseline":
        baseline_name = str(head_meta.get("name", ""))
        if baseline_name == "linear_svc":
            args.siamese_inference = "linearsvc"
        elif baseline_name in {"logreg", "logistic_regression"}:
            args.siamese_inference = "logisticregression"
    st.session_state["optimized_k_value"] = args.best_classifier_config
    st.session_state["learned_classifier_label"] = args.learned_classifier_label
    if not str(args.best_classifier_config).isdigit():
        st.session_state["selected_prototype_approach"] = args.best_classifier_config

    return args


def get_args(cursor, is_admin, data_dir='./data'):
    """Wrapper for backward compatibility. Call build_args_from_sidebar directly."""
    return build_args_from_sidebar(cursor, is_admin, data_dir)
