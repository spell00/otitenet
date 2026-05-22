"""
Argument and configuration building for the Streamlit app.

This module handles all UI widget setup and argument namespace construction,
keeping app.py focused on core logic.
"""

import os
import argparse
from typing import Optional

import streamlit as st
import pandas as pd

from otitenet.app.utils import (
    extract_params_from_log_path,
    _make_model_selection_key,
    _unique_preserve_order,
    get_split_mcc_metrics,
    is_done_manifest_model_row,
    format_classifier_config,
    resolve_best_classifier_config,
    enumerate_classification_heads,
    ensure_int,
)

from otitenet.app.database import set_production_model


def _args_namespace_from_model_row(model_row: dict) -> argparse.Namespace:
    """Build an args-like namespace from a Quick Model Selection / registry row."""
    row = dict(model_row)
    row.update(extract_params_from_log_path(row.get("Log Path") or row.get("log_path")))
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
    )


def _render_classification_head_selector(model_row: dict, model_selection_key: str) -> Optional[str]:
    """Show classification-head selectbox for the current Quick Model Selection row."""
    tmp_args = _args_namespace_from_model_row(model_row)
    heads = enumerate_classification_heads(tmp_args)
    if not heads:
        st.caption("No learned-embedding heads cached. Run optimization in Tab 1.")
        return None

    head_configs = [h["config"] for h in heads]
    head_labels = {
        h["config"]: f"{h['label']} (MCC {h['mcc']:.4f})" + (f" — {h['details']}" if h.get("details") else "")
        for h in heads
    }

    widget_key = f"sidebar_classification_head_{model_selection_key}"
    prev_model_key = st.session_state.get("sidebar_classification_head_model_key")
    if prev_model_key != model_selection_key:
        st.session_state["sidebar_classification_head_model_key"] = model_selection_key
        st.session_state[widget_key] = head_configs[0]

    current = st.session_state.get(widget_key)
    if current not in head_configs:
        st.session_state[widget_key] = head_configs[0]

    st.selectbox(
        "Classification head",
        options=head_configs,
        format_func=lambda c: head_labels.get(c, format_classifier_config(c)),
        key=widget_key,
        help="Learned-embedding classifier used for inference across tabs (when optimized inference is enabled).",
    )
    selected_cfg = st.session_state.get(widget_key, head_configs[0])
    st.session_state["sidebar_classification_head_config"] = selected_cfg
    st.session_state["optimized_k_value"] = selected_cfg
    st.session_state["learned_classifier_label"] = format_classifier_config(selected_cfg)
    return selected_cfg


from otitenet.app.ui_helpers import choose_dataset
from otitenet.utils.update_model_ranks import update_model_ranks


DEFAULT_APP_TRAIN_DATASETS = "Banque_Calaman_USA_2020_trie_CM,Banque_Comert_Turquie_2020_jpg"


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
    
    selected_params = st.session_state.get('selected_model_params', {})
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
                           prototype_strategy, prototype_components
                    FROM best_models_registry
                    WHERE model_rank IS NOT NULL
                    ORDER BY model_rank ASC
                    """
                )
                model_rows = cursor.fetchall()
                use_db_rank = True
            except Exception:
                # Fallback if model_rank column doesn't exist yet
                cursor.execute(
                    """
                    SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss,
                           n_calibration, accuracy, mcc, normalize, n_neighbors, log_path,
                           prototype_strategy, prototype_components
                    FROM best_models_registry
                    ORDER BY mcc DESC
                    """
                )
                model_rows = cursor.fetchall()
                use_db_rank = False

            if model_rows:
                # Align columns with the database query (19 columns)
                if use_db_rank:
                    _cols = [
                        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                        "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", 
                        "#", "Proto_Strat", "Proto_Comp"
                    ]
                else:
                    _cols = [
                        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                        "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path",
                        "Proto_Strat", "Proto_Comp"
                    ]
                _df = pd.DataFrame(model_rows, columns=_cols)
                _df = _df[_df.apply(is_done_manifest_model_row, axis=1)].copy()
                
                if not _df.empty:
                    _group_cols = [
                        "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
                        "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize", "N_Neighbors",
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
                        try:
                            key_to_label[selection_key] = (
                                f"#{model_num} - {rd.get('Model Name')} (Size:{rd.get('NSize')}, MCC:{float(rd.get('MCC')):.3f}, Dist:{rd.get('Dist_Fct')}, Norm:{rd.get('Normalize')})"
                            )
                        except Exception:
                            key_to_label[selection_key] = (
                                f"#{model_num} - {rd.get('Model Name')} (Size:{rd.get('NSize')}, MCC:{rd.get('MCC')}, Dist:{rd.get('Dist_Fct')}, Norm:{rd.get('Normalize')})"
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
                            'Log Path': rd.get('Log Path'),
                        }
                        model_dict.update(extract_params_from_log_path(model_dict.get("Log Path")))
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
                            if model_dict.get('Dataset'):
                                st.session_state['dataset_selectbox'] = model_dict['Dataset']
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
                                'Log Path': rd.get('Log Path'),
                                'model_number': st.session_state.model_number_map.get(selected_key, '?'),
                            }
                            selected_head_config = st.session_state.get('sidebar_classification_head_config')
                            if not selected_head_config:
                                try:
                                    selected_head_config = resolve_best_classifier_config(_args_namespace_from_model_row(rd), use_optimized=True)
                                except Exception:
                                    selected_head_config = str(rd.get('N_Neighbors') or 1)
                            selected_head_label = format_classifier_config(selected_head_config)
                            model_dict['Head Config'] = str(selected_head_config)
                            model_dict['head_config'] = str(selected_head_config)
                            model_dict['classification_head_config'] = str(selected_head_config)
                            model_dict['best_classifier_config'] = str(selected_head_config)
                            model_dict['Head'] = selected_head_label
                            model_dict['head_name'] = selected_head_label
                            model_dict['learned_classifier_label'] = selected_head_label
                            # Save to database for persistence across sessions
                            set_by_email = st.session_state.get('user_email', 'unknown')
                            if set_production_model(cursor, conn, model_dict, set_by_email):
                                st.session_state['production_model'] = model_dict
                                st.success("✅ Model set as production model (saved to database)")
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
        
        available_datasets = sorted([
            name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))
        ])
        dataset_default = selected_params.get('Dataset')
        if dataset_default not in available_datasets:
            dataset_default = 'otite_ds_64' if 'otite_ds_64' in available_datasets else (available_datasets[0] if available_datasets else None)
        selected_path = choose_dataset(
            "Select --path dataset", available_datasets, default=dataset_default, key="dataset_selectbox"
        )

        # ---- Other args with defaults ---- #
        # Prefer explicit new_size from selection; else infer from dataset suffix; fallback to 224
        new_size_raw = selected_params.get('new_size')
        if new_size_raw is None and selected_path:
            try:
                new_size = int(str(selected_path).split('_')[-1])
            except Exception:
                new_size = 224
        else:
            try:
                new_size = int(new_size_raw) if new_size_raw is not None else 224
            except Exception:
                new_size = 224
        # Expose new_size control like other parameters
        if should_sync and new_size is not None:
            st.session_state['new_size_input'] = int(new_size)
        new_size = st.number_input("new_size", value=int(st.session_state.get('new_size_input', new_size)), step=1, key="new_size_input")

        task_root = 'logs/best_models'
        task_list = sorted(os.listdir(task_root)) if os.path.isdir(task_root) else []
        if not task_list:
            st.error("No tasks found under logs/best_models.")
            st.stop()
        task_default = selected_params.get("Task")
        sync_value('task_selectbox', task_default)
        _safe_selectbox_sync('task_selectbox', task_list, task_default)
        task = st.selectbox("task", task_list, key="task_selectbox")

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

        fgsm_dir = os.path.join(model_dir, model_name, selected_path if selected_path else 'otite_ds_64', f'nsize{new_size}')
        fgsm_list = sorted(os.listdir(fgsm_dir)) if os.path.isdir(fgsm_dir) else []
        if not fgsm_list:
            st.error(f"No FGSM entries found in {fgsm_dir}.")
            st.stop()
        fgsm_default = selected_params.get("FGSM")
        sync_value('fgsm_selectbox', fgsm_default)
        _safe_selectbox_sync('fgsm_selectbox', fgsm_list, fgsm_default)
        fgsm = st.selectbox("fgsm", fgsm_list, key="fgsm_selectbox")

        n_cal_dir = os.path.join(fgsm_dir, fgsm)
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
        classif_loss_default = selected_params.get("classif_loss")
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
        dist_fct_options = ['euclidean', 'cosine']
        dist_fct_default = str(selected_params.get('dist_fct', 'euclidean')).strip().lower()
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
        path=os.path.join(data_dir, selected_path) if selected_path else None,
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
    args.train_datasets = str(selected_params.get('train_datasets', DEFAULT_APP_TRAIN_DATASETS) or DEFAULT_APP_TRAIN_DATASETS)
    args.test_dataset = str(selected_params.get('test_dataset', valid_dataset) or valid_dataset)
    args.model_id = selected_params.get('model_id') or selected_params.get('Model ID')
    args.use_pretrained_encodings = bool(st.session_state.get('use_saved_encodings_tab1', True))
    args.use_trained_encoder = args.use_pretrained_encodings

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

    selected_head = st.session_state.get("sidebar_classification_head_config")
    if selected_head:
        args.best_classifier_config = str(selected_head)
    else:
        args.best_classifier_config = resolve_best_classifier_config(args, use_optimized=True)
    args.learned_classifier_label = format_classifier_config(args.best_classifier_config)
    st.session_state["optimized_k_value"] = args.best_classifier_config
    st.session_state["learned_classifier_label"] = args.learned_classifier_label
    if not str(args.best_classifier_config).isdigit():
        st.session_state["selected_prototype_approach"] = args.best_classifier_config

    return args


def get_args(cursor, is_admin, data_dir='./data'):
    """Wrapper for backward compatibility. Call build_args_from_sidebar directly."""
    return build_args_from_sidebar(cursor, is_admin, data_dir)