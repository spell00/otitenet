"""
Analysis and inference functions for the Streamlit app.

This module handles model inference, KNN caching, and file analysis,
keeping app.py focused on UI orchestration.
"""

import os
import json
import pickle
import numpy as np
import torch
import streamlit as st
from datetime import datetime

from otitenet.train.train_triplet_new import TrainAE
from otitenet.data.data_getters import PerImageNormalize, get_images_loaders
from otitenet.utils.utils import get_empty_traces
from otitenet.app.model_loading import (
    load_model_and_prototypes,
    load_model_for_log_path
)
from otitenet.app.database import check_ds_exists, insert_score
from otitenet.app.image_processing import get_image, preprocess_image
from otitenet.app.utils import (
    get_model_params_path,
    _make_model_selection_key,
    _ensure_model_number_map,
)
from otitenet.app.inference import (
    predict_label_from_prototypes as _predict_label_from_prototypes,
    predict_with_prototype_distance_ratio as _predict_with_prototype_distance_ratio,
    predict_with_kde as _predict_with_kde,
)
from otitenet.ml import fit_knn_classifier
from otitenet.app.utils import get_model_cache_key


def get_or_build_knn(_args, data, unique_labels, unique_batches, prototypes):
    """Return a cached KNN for the selected model; build once if missing.

    This avoids re-running predict loops for 'valid'/'test' during inference.
    Encodes training data with model and fits KNN on encoded embeddings.
    
    NOTE: If prototypes are enabled, returns None since classification uses prototypes/KDE instead.
    
    Args:
        _args: Model/inference arguments
        data: Dataset object
        unique_labels: Array of unique labels
        unique_batches: Array of unique batch identifiers
        prototypes: Prototype configuration dict
        
    Returns:
        Tuple of (fitted_knn, unique_labels)
    """
    # Skip KNN if prototypes are enabled
    use_prototypes = (_args.prototypes_to_use in ['combined', 'class'])
    if use_prototypes:
        print("⏭️  Skipping KNN: using prototype-based classification instead")
        return None, unique_labels
    
    # Check cache
    cache = st.session_state.setdefault('knn_cache', {})
    key = get_model_cache_key(_args)
    if key in cache:
        return cache[key]['knn'], cache[key]['unique_labels']

    # ---- Build KNN by encoding training data ---- 
    from otitenet.train.train_triplet_new import TrainAE
    from otitenet.data.data_getters import get_images_loaders, get_empty_traces
    from otitenet.app.model_loading import load_model_and_prototypes
    
    # Initialize training wrapper for encoding
    train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                    log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
                    log_mlflow=False, groupkfold=_args.groupkfold)
    train.n_batches = len(unique_batches)
    train.n_cats = len(unique_labels)
    train.unique_batches = unique_batches
    train.unique_labels = unique_labels
    train.epoch = 1
    train.params = {
        'n_neighbors': _args.n_neighbors,
        'lr': 0,
        'wd': 0,
        'smoothing': 0,
        'is_transform': 0,
        'valid_dataset': _args.valid_dataset
    }
    train.set_arcloss()

    # Build data loaders
    lists, traces = get_empty_traces()
    loaders = get_images_loaders(
        data=data,
        random_recs=_args.random_recs,
        weighted_sampler=0,
        is_transform=0,
        samples_weights=None,
        epoch=1,
        unique_labels=unique_labels,
        triplet_dloss=_args.dloss, bs=_args.bs,
        prototypes_to_use=_args.prototypes_to_use,
        prototypes=prototypes,
        size=_args.new_size,
        normalize=_args.normalize,
    )

    # Encode training data with model
    with torch.no_grad():
        try:
            model, _, _, _, _, _, _, _, _ = load_model_and_prototypes(_args)
            train.model = model
            _, lists, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
        except Exception as e:
            st.error(f"❌ Could not encode training set for KNN: {e}")
            raise

    # Extract encoded data and fit KNN
    train_encs = np.concatenate(lists['train']['encoded_values'])
    train_cats = np.concatenate(lists['train']['cats'])
    
    # Fit KNN (n_neighbors is adjusted by classif_loss strategy)
    if _args.classif_loss not in ['ce', 'hinge']:
        nn_count = int(_args.n_neighbors)
        knn = fit_knn_classifier(train_encs, train_cats, n_neighbors=nn_count, metric='minkowski')
    else:
        # If classification is CE/hinge, fallback to 1-NN
        knn = fit_knn_classifier(train_encs, train_cats, n_neighbors=1, metric='minkowski')

    # Cache result and training data for KDE if needed
    cache[key] = {'knn': knn, 'unique_labels': unique_labels}
    st.session_state['train_embeddings'] = train_encs
    st.session_state['train_labels'] = train_cats
    
    return knn, unique_labels


def run_analysis_on_file(filename, file_bytes, _args, cursor, conn, force_reanalyze=False, show_validation_metrics=True, fast_infer=False):
    """Run complete analysis on a file and return results."""
    # Make sure model numbering is available even if user didn't open Tab 1/2 first
    model_number_map, best_models_table = _ensure_model_number_map(cursor)
    params = get_model_params_path(_args)
    complete_log_path = f"logs/best_models/{_args.task}/{_args.model_name}/{params}/queries"
    
    # Check if already analyzed
    exists = None if force_reanalyze else check_ds_exists(cursor, filename, _args)
    
    if exists is not None and not force_reanalyze:
        # Load previous results (indices match SELECT pred_label, confidence, log_path)
        pred_label = exists[0]
        pred_confidence = exists[1]
        log_path = exists[2]
        st.info(
            f"✅ Already analyzed with this exact model & params → Pred: {pred_label} (conf {pred_confidence:.2f})."
        )
        return pred_label, pred_confidence, log_path, exists
    
    # Run fresh analysis
    st.info("🔄 Running analysis...")
    with st.spinner("Loading model and processing image..."):
        # Ensure reproducible split: reset seeds before data loading
        import random
        random.seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)
        
        model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
            load_model_and_prototypes(_args)
        
        save_path = os.path.join('data/queries', filename)
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
        
        # Load saved hyperparameters
        params_path = os.path.join(f'logs/best_models/{_args.task}/{_args.model_name}',
                                   f'{_args.path.split("/")[-1]}/nsize{_args.new_size}/{_args.fgsm}/{_args.n_calibration}/'
                                   f'{_args.classif_loss}/{_args.dloss}/{_args.prototypes_to_use}/'
                                   f'{_args.n_positives}/{_args.n_negatives}',
                                   'params.json')
        saved_search = {}
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                    saved_search = payload.get('search_params', {}) or {}
            except Exception:
                saved_search = {}
        
        # Do not override sidebar-selected n_neighbors from saved search params.
        # The user's current selection should take precedence over training-time defaults.
        
        # Prepare inference resources
        params = get_model_params_path(_args)
        train_complete_log_path = f'logs/best_models/{_args.task}/{_args.model_name}/{params}'

        if not fast_infer:
            loaders = get_images_loaders(data=data,
                                         random_recs=_args.random_recs,
                                         weighted_sampler=0,
                                         is_transform=0,
                                         samples_weights=None,
                                         epoch=1,
                                         unique_labels=unique_labels,
                                         triplet_dloss=_args.dloss, bs=_args.bs,
                                         prototypes_to_use=_args.prototypes_to_use,
                                         prototypes=prototypes,
                                         size=_args.new_size,
                                         normalize=_args.normalize)
            # Build or reuse cached KNN once per model selection
            knn, unique_labels_knn = get_or_build_knn(_args, data, unique_labels, unique_batches, prototypes)
            nets = {'cnn': shap_model, 'knn': knn}
        else:
            loaders = None
            unique_labels_knn = unique_labels
            nets = {'cnn': shap_model, 'knn': None}
        
        original, image = get_image(f'data/queries/{filename}', size=image_size, normalize=_args.normalize)
        
        # Optionally fetch cached validation metrics; bulk analysis can skip this entirely
        valid_acc = None
        valid_mcc = None
        valid_metrics_source = None
        if show_validation_metrics:
            # Resolve the correct model_id based on log_path or model parameters
            from otitenet.app.model_loading import resolve_model_id
            resolved_model_id = resolve_model_id(_args, train_complete_log_path)
            
            # Prefer metrics that match the exact model ID we are using right now
            current_model_row = None
            try:
                if best_models_table is not None and not best_models_table.empty and resolved_model_id is not None:
                    match = best_models_table[best_models_table["Model ID"] == resolved_model_id]
                    if not match.empty:
                        current_model_row = match.iloc[0].to_dict()
            except Exception:
                current_model_row = None

            if current_model_row:
                try:
                    reg_acc = current_model_row.get('Accuracy')
                    reg_mcc = current_model_row.get('MCC')
                    if reg_acc is not None and reg_mcc is not None:
                        valid_acc = float(reg_acc)
                        valid_mcc = float(reg_mcc)
                        valid_metrics_source = 'registry'
                except Exception:
                    pass

            params_path_metrics = os.path.join(train_complete_log_path, "params.json")
            if valid_metrics_source is None and os.path.exists(params_path_metrics):
                try:
                    with open(params_path_metrics, "r", encoding="utf-8") as f:
                        params_payload = json.load(f)
                    best_metrics = params_payload.get("best_metrics", {})
                    valid_metrics = best_metrics.get("valid", {})
                    acc_list = valid_metrics.get("acc") or []
                    mcc_list = valid_metrics.get("mcc") or []
                    if isinstance(acc_list, list) and acc_list:
                        valid_acc = float(acc_list[-1])
                    if isinstance(mcc_list, list) and mcc_list:
                        valid_mcc = float(mcc_list[-1])
                    if valid_acc is not None and valid_mcc is not None:
                        valid_metrics_source = 'params.json'
                except Exception as e:
                    st.warning(f"Could not read saved validation metrics: {e}")
            if valid_acc is None or valid_mcc is None:
                valid_metrics_source = None
        
        # Get prediction (fast prototype-based, KDE, or KNN)
        with torch.no_grad():
            emb_tensor = nets['cnn'](image.to(device_str))
            if fast_infer:
                class_protos = prototypes.get('class', {}).get('train', {}) if isinstance(prototypes, dict) else {}
                pred_lbl_fast = _predict_with_prototype_distance_ratio(emb_tensor, class_protos, dist_fct_name=str(_args.dist_fct).lower())
                if pred_lbl_fast is None:
                    pred_label = unique_labels[0]
                    pred_confidence = 0.0
                else:
                    pred_label = pred_lbl_fast
                    pred_confidence = 1.0
            else:
                # Check which classification method to use
                use_prototypes = (_args.prototypes_to_use in ['combined', 'class'] and 
                                prototypes.get('class', {}).get('train'))
                use_kde = (use_prototypes and 
                          getattr(_args, 'prototype_kind', 'distance').lower() == 'kde')
                use_knn = not use_prototypes and not use_kde
                
                embedding = emb_tensor.detach().cpu().numpy()
                
                if use_kde:
                    # Use KDE classifier
                    if 'train_embeddings' not in st.session_state or 'train_labels' not in st.session_state:
                        # Fallback to prototype distance if KDE data not available
                        class_protos = prototypes.get('class', {}).get('train', {})
                        pred_lbl = _predict_with_prototype_distance_ratio(emb_tensor, class_protos, 
                                                                         dist_fct_name=str(_args.dist_fct).lower())
                        pred_label = pred_lbl if pred_lbl is not None else unique_labels[0]
                        pred_confidence = 1.0
                    else:
                        pred_label, pred_confidence = _predict_with_kde(
                            emb_tensor, 
                            st.session_state['train_embeddings'],
                            st.session_state['train_labels'],
                            unique_labels,
                            kde_kernel=getattr(_args, 'kde_kernel', 'gaussian'),
                            kde_bandwidth=getattr(_args, 'kde_bandwidth', 'scott')
                        )
                        if pred_label is None:
                            pred_label = unique_labels[0]
                            pred_confidence = 0.0
                
                elif use_prototypes:
                    # Use prototype distance classification
                    class_protos = prototypes.get('class', {}).get('train', {})
                    pred_lbl = _predict_with_prototype_distance_ratio(emb_tensor, class_protos, 
                                                                     dist_fct_name=str(_args.dist_fct).lower())
                    if pred_lbl is None:
                        pred_label = unique_labels[0]
                        pred_confidence = 0.0
                    else:
                        pred_label = pred_lbl
                        pred_confidence = 1.0
                
                else:
                    # Use KNN classifier (original behavior)
                    pred_probs = nets['knn'].predict_proba(embedding)
                    pred_class = int(np.argmax(pred_probs, axis=1)[0])
                    pred_confidence = float(pred_probs[0, pred_class]) if pred_probs.ndim == 2 else float(np.max(pred_probs))
                    pred_label = unique_labels[pred_class]
        
        if show_validation_metrics:
            if valid_metrics_source == 'registry':
                st.write(f"**Validation Accuracy (from registry):** {valid_acc:.3f}")
                st.write(f"**Validation MCC (from registry):** {valid_mcc:.3f}")
            elif valid_metrics_source == 'params.json':
                st.write(f"**Validation Accuracy (from params.json):** {valid_acc:.3f}")
                st.write(f"**Validation MCC (from params.json):** {valid_mcc:.3f}")
            else:
                st.info("Validation metrics unavailable (skipped recomputation to speed up inference).")
        
        # Display prediction with appropriate method label
        use_prototypes = (_args.prototypes_to_use in ['combined', 'class'] and 
                        prototypes.get('class', {}).get('train'))
        use_kde = (use_prototypes and 
                  getattr(_args, 'prototype_kind', 'distance').lower() == 'kde')
        
        if use_kde:
            method_label = f"KDE ({getattr(_args, 'kde_kernel', 'gaussian')} kernel)"
        elif use_prototypes:
            method_label = f"Prototype-based ({getattr(_args, 'prototype_kind', 'distance')})"
        else:
            method_label = f"KNN (k={_args.n_neighbors})"
        
        st.write(f"**Predicted Label ({method_label}):** {pred_label} ({pred_confidence:.2f} confidence)")
        st.caption(
            f"Model run → id: {_args.model_id}, name: {_args.model_name}, size: {_args.new_size}, fgsm: {_args.fgsm}, dist: {_args.dist_fct}, protos: {_args.prototypes_to_use}, normalize: {_args.normalize}"
        )
        
        # Insert into database
        insert_score(cursor, conn, filename, _args, pred_label, pred_confidence, complete_log_path)
        st.success("✅ Results saved to database.")

        # Track the model number used for this run so the UI can show it immediately
        selection_key = _make_model_selection_key({
            "Model Name": _args.model_name,
            "NSize": _args.new_size,
            "FGSM": _args.fgsm,
            "Prototypes": _args.prototypes_to_use,
            "NPos": _args.n_positives,
            "NNeg": _args.n_negatives,
            "DLoss": _args.dloss,
            "Dist_Fct": _args.dist_fct,
            "Classif_Loss": _args.classif_loss,
            "N_Calibration": _args.n_calibration,
            "Normalize": _args.normalize,
            "N_Neighbors": getattr(_args, "n_neighbors", None),
        })
        model_number = model_number_map.get(selection_key, "?")
        if model_number == "?" and best_models_table is not None and not best_models_table.empty:
            try:
                match = best_models_table[best_models_table["Log Path"] == complete_log_path]
                if not match.empty:
                    model_number = match.iloc[0].get("#", model_number)
            except Exception:
                pass
        st.session_state['last_model_number'] = model_number
    
    return pred_label, pred_confidence, complete_log_path, None
