"""
Argument and configuration building for the Streamlit app.

This module handles all UI widget setup and argument namespace construction,
keeping app.py focused on core logic.
"""

import os
import argparse
import streamlit as st
import pandas as pd

from otitenet.app.utils import (
    extract_params_from_log_path,
    _make_model_selection_key,
    _unique_preserve_order,
)
from otitenet.app.ui_helpers import choose_dataset
from otitenet.utils.update_model_ranks import update_model_ranks


def build_args_from_sidebar(cursor, data_dir='./data'):
    """Build args namespace by constructing sidebar UI and gathering selections.
    
    This function handles all UI widget logic for parameter selection, model ranking,
    and state synchronization.
    
    Args:
        cursor: Database cursor for querying best_models_registry
        data_dir: Root data directory path (default: './data')
        
    Returns:
        argparse.Namespace with model and inference parameters
    """
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
                           n_calibration, accuracy, mcc, normalize, n_neighbors, log_path, model_rank
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
                           n_calibration, accuracy, mcc, normalize, n_neighbors, log_path
                    FROM best_models_registry
                    ORDER BY mcc DESC
                    """
                )
                model_rows = cursor.fetchall()
                use_db_rank = False

            if model_rows:
                # Build the same deduped view as Tab1, then take the top unique rows.
                if use_db_rank:
                    _cols = [
                        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                        "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path", "#"
                    ]
                else:
                    _cols = [
                        "Model ID", "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "Dist_Fct",
                        "Classif_Loss", "N_Calibration", "Accuracy", "MCC", "Normalize", "N_Neighbors", "Log Path"
                    ]
                _df = pd.DataFrame(model_rows, columns=_cols)
                
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
                
                # model_rank is already in the dataframe from the database query (if available)
                # Otherwise add dynamic numbering
                if "#" in _df.columns:
                    # Move it to the first column
                    cols = ["#"] + [col for col in _df.columns if col != "#"]
                    _df = _df[cols]
                else:
                    # Add dynamic numbering as fallback
                    _df.insert(0, "#", range(1, len(_df) + 1))

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
        if task_default not in task_list:
            task_default = task_list[0]
        sync_value('task_selectbox', task_default)
        task = st.selectbox("task", task_list, key="task_selectbox")

        model_dir = os.path.join(task_root, task)
        model_name_list = [name for name in sorted(os.listdir(model_dir)) if not name.endswith('.csv')] if os.path.isdir(model_dir) else []
        if not model_name_list:
            st.error(f"No models found for task {task}.")
            st.stop()
        model_name_default = selected_params.get("Model Name")
        # remove files ending in .json or any extension really
        model_name_list = [name for name in model_name_list if not any(name.endswith(ext) for ext in ['.json', '.csv', '.txt'])]
        if model_name_default not in model_name_list:
            model_name_default = model_name_list[0]
        sync_value('model_name_selectbox', model_name_default)
        model_name = st.selectbox("Model Name", model_name_list, key="model_name_selectbox")

        fgsm_dir = os.path.join(model_dir, model_name, selected_path if selected_path else 'otite_ds_64', f'nsize{new_size}')
        fgsm_list = sorted(os.listdir(fgsm_dir)) if os.path.isdir(fgsm_dir) else []
        if not fgsm_list:
            st.error(f"No FGSM entries found in {fgsm_dir}.")
            st.stop()
        fgsm_default = selected_params.get("FGSM")
        if fgsm_default not in fgsm_list:
            fgsm_default = fgsm_list[0]
        sync_value('fgsm_selectbox', fgsm_default)
        fgsm = st.selectbox("fgsm", fgsm_list, key="fgsm_selectbox")

        n_cal_dir = os.path.join(fgsm_dir, fgsm)
        n_calibration_list = sorted(os.listdir(n_cal_dir)) if os.path.isdir(n_cal_dir) else []
        if not n_calibration_list:
            st.error(f"No calibration folders found in {n_cal_dir}.")
            st.stop()
        n_calibration_default = selected_params.get("N_Calibration")
        if n_calibration_default not in n_calibration_list:
            n_calibration_default = n_calibration_list[0]
        sync_value('n_calibration_selectbox', n_calibration_default)
        n_calibration = st.selectbox("n_calibration", n_calibration_list, key="n_calibration_selectbox")

        classif_dir = os.path.join(n_cal_dir, n_calibration)
        classif_loss_list = sorted(os.listdir(classif_dir)) if os.path.isdir(classif_dir) else []
        if not classif_loss_list:
            st.error(f"No classif_loss folders found in {classif_dir}.")
            st.stop()
        classif_loss_default = selected_params.get("classif_loss")
        if classif_loss_default not in classif_loss_list:
            classif_loss_default = classif_loss_list[0]
        sync_value('classif_loss_selectbox', classif_loss_default)
        classif_loss = st.selectbox("classif_loss", classif_loss_list, key="classif_loss_selectbox")

        dloss_dir = os.path.join(classif_dir, classif_loss)
        dloss_list = sorted(os.listdir(dloss_dir)) if os.path.isdir(dloss_dir) else []
        if not dloss_list:
            st.error(f"No dloss folders found in {dloss_dir}.")
            st.stop()
        dloss_default = selected_params.get("DLoss")
        if dloss_default not in dloss_list:
            dloss_default = dloss_list[0]
        sync_value('dloss_selectbox', dloss_default)
        dloss = st.selectbox("dloss", dloss_list, key="dloss_selectbox")

        proto_dir = os.path.join(dloss_dir, dloss)
        prototypes_list = sorted(os.listdir(proto_dir)) if os.path.isdir(proto_dir) else []
        if not prototypes_list:
            st.error(f"No prototype folders found in {proto_dir}.")
            st.stop()
        prototypes_default = selected_params.get("Prototypes")
        if prototypes_default not in prototypes_list:
            prototypes_default = prototypes_list[0]
        sync_value('prototypes_selectbox', prototypes_default)
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
        if npos_default not in n_positives_list:
            npos_default = n_positives_list[0]
        sync_value('npos_selectbox', npos_default)
        n_positives = st.selectbox('npos', n_positives_list, key="npos_selectbox")

        nneg_dir = os.path.join(npos_dir, n_positives)
        n_negatives_list = sorted(os.listdir(nneg_dir)) if os.path.isdir(nneg_dir) else []
        if not n_negatives_list:
            st.error(f"No nneg folders found in {nneg_dir}.")
            st.stop()
        nneg_default = str(selected_params.get("NNeg")) if selected_params.get("NNeg") is not None else None
        if nneg_default not in n_negatives_list:
            nneg_default = n_negatives_list[0]
        sync_value('nneg_selectbox', nneg_default)
        n_negatives = st.selectbox('nneg', n_negatives_list, key="nneg_selectbox")

        # Apply optimized k from Tab 1 KNN Optimization if available
        if 'optimized_k_value' in st.session_state:
            opt_val = st.session_state.pop('optimized_k_value')
            # Only apply if it's an integer KNN k (not a prototype strategy string)
            if isinstance(opt_val, int) or (isinstance(opt_val, str) and opt_val.isdigit()):
                n_neighbors_default = int(opt_val)
                should_sync = True  # Force sync so the widget picks up the new value
            else:
                # Prototype strategy approach; skip sidebar sync, but store in session for Tab 2/3
                st.session_state['selected_prototype_approach'] = opt_val

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
        normalize = st.selectbox("normalize", ['yes', 'no'], index=1 if normalize_default == "no" else 0, key="normalize_input")

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
    args.model_id = selected_params.get('model_id') or selected_params.get('Model ID')

    return args


def get_args(cursor, data_dir='./data'):
    """Wrapper for backward compatibility. Call build_args_from_sidebar directly."""
    return build_args_from_sidebar(cursor, data_dir)
