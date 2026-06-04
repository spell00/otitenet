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
from sklearn.preprocessing import LabelEncoder

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
    dataset_path_segment,
    _make_model_selection_key,
    _ensure_model_number_map,
)
from otitenet.app.inference import (
    predict_label_from_prototypes as _predict_label_from_prototypes,
    predict_with_prototype_distance_ratio as _predict_with_prototype_distance_ratio,
    predict_with_prototype_distance_ratio_proba as _predict_with_prototype_distance_ratio_proba,
    predict_with_kde as _predict_with_kde,
)
from otitenet.ml import fit_knn_classifier, fit_linearsvc_classifier, fit_logreg_classifier
from otitenet.app.utils import get_model_cache_key


from contextlib import contextmanager, nullcontext


def _normalize_probability_map(probas, labels):
    """Convert model probabilities to a {class_label: probability} mapping."""
    if probas is None:
        return {}

    if isinstance(probas, dict):
        out = {}
        for key, value in probas.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                continue
        return out

    try:
        arr = np.asarray(probas, dtype=float)
        if arr.ndim == 2:
            arr = arr[0]
        arr = arr.reshape(-1)
    except Exception:
        return {}

    out = {}
    for idx, value in enumerate(arr):
        if idx >= len(labels):
            break
        try:
            out[str(labels[idx])] = float(value)
        except Exception:
            continue
    return out


def _classifier_classes_to_labels(classes, unique_labels):
    labels = []
    for raw_label in classes:
        try:
            raw_idx = int(raw_label)
            if 0 <= raw_idx < len(unique_labels):
                labels.append(unique_labels[raw_idx])
                continue
        except Exception:
            pass
        labels.append(raw_label)
    return labels


def _decision_scores_to_proba(scores):
    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 1:
        probs_pos = 1.0 / (1.0 + np.exp(-arr))
        return np.vstack([1.0 - probs_pos, probs_pos]).T
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    denom = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def _head_config_from_args(_args):
    config = (
        getattr(_args, "best_classifier_config", None)
        or getattr(_args, "classification_head_config", None)
        or getattr(_args, "head_config", None)
    )
    if config is not None and str(config).strip() not in {"", "None", "nan", "—"}:
        return str(config).strip()

    mode = str(getattr(_args, "siamese_inference", "linearsvc")).strip().lower()
    if mode == "linearsvc":
        return "baseline_linear_svc"
    if mode == "logisticregression":
        return "baseline_logreg"
    return str(getattr(_args, "n_neighbors", 1))


def _classifier_kind_from_config(config):
    config = str(config or "").strip().lower()
    if config in {"linearsvc", "linear_svc", "baseline_linearsvc", "baseline_linear_svc"}:
        return "linearsvc"
    if config in {"logisticregression", "logistic_regression", "logreg", "baseline_logreg", "baseline_logistic_regression"}:
        return "logisticregression"
    if config.startswith("baseline_linear_svc"):
        return "linearsvc"
    if config.startswith("baseline_logreg") or config.startswith("baseline_logistic_regression"):
        return "logisticregression"
    return "knn"


@contextmanager
def _quiet_streamlit_messages(enabled: bool = False):
    """Temporarily silence Streamlit status calls during bulk/background inference."""
    if not enabled:
        yield
        return

    names = [
        "info", "success", "warning", "error", "write", "caption",
        "markdown", "toast", "metric", "dataframe", "table",
    ]
    originals = {name: getattr(st, name, None) for name in names}
    original_spinner = getattr(st, "spinner", None)

    def _noop(*args, **kwargs):
        return None

    try:
        for name in names:
            if hasattr(st, name):
                setattr(st, name, _noop)
        if original_spinner is not None:
            setattr(st, "spinner", lambda *args, **kwargs: nullcontext())
        yield
    finally:
        for name, value in originals.items():
            if value is not None:
                setattr(st, name, value)
        if original_spinner is not None:
            setattr(st, "spinner", original_spinner)


def _fit_batch_encoder(unique_batches):
    """Fit a batch encoder once so loaders and training-loop bookkeeping agree."""
    encoder = LabelEncoder()
    encoder.fit(np.asarray(unique_batches))
    return encoder


def get_or_build_embedding_classifier(_args, data, unique_labels, unique_batches, prototypes):
    """Return a cached embedding classifier for the selected model/head."""
    use_prototypes = (_args.prototypes_to_use in ['combined', 'class'])
    if use_prototypes:
        print("⏭️  Skipping embedding classifier: using prototype-based classification instead")
        return None, unique_labels

    head_config = _head_config_from_args(_args)
    classifier_kind = _classifier_kind_from_config(head_config)

    cache = st.session_state.setdefault('embedding_classifier_cache', {})
    key = f"{get_model_cache_key(_args)}|head={head_config}"
    if key in cache:
        print(f"♻️  Using cached {cache[key].get('classifier_kind', 'embedding')} classifier")
        return cache[key]['classifier'], cache[key]['unique_labels']

    def _candidate_train_encoding_paths():
        candidates = []
        for attr in ["train_encodings_path", "Train Encodings Path"]:
            value = getattr(_args, attr, None)
            if value:
                candidates.append(str(value))

        for attr in ["log_path", "best_model_dir", "Artifact Log Path", "Best Model Dir"]:
            value = getattr(_args, attr, None)
            if not value:
                continue
            path = str(value)
            if os.path.isfile(path):
                path = os.path.dirname(path)
            candidates.append(os.path.join(path, "train_encodings.npz"))

        model_params = get_model_params_path(_args)
        candidates.append(
            os.path.join(
                f'logs/best_models/{_args.task}/{_args.model_name}',
                model_params,
                'train_encodings.npz',
            )
        )

        if bool(getattr(_args, "split_config_in_path", False) or getattr(_args, "_split_config_in_path", False)):
            try:
                old_split = getattr(_args, "split_config_in_path", False)
                old_private = getattr(_args, "_split_config_in_path", False)
                _args.split_config_in_path = False
                _args._split_config_in_path = False
                legacy_params = get_model_params_path(_args)
                candidates.append(
                    os.path.join(
                        f'logs/best_models/{_args.task}/{_args.model_name}',
                        legacy_params,
                        'train_encodings.npz',
                    )
                )
            finally:
                _args.split_config_in_path = old_split
                _args._split_config_in_path = old_private

        out = []
        for path in candidates:
            if not path:
                continue
            normalized = os.path.normpath(path)
            if normalized not in out:
                out.append(normalized)
        return out

    # ---- Fast path: load saved train_encodings.npz if available ----
    train_encs = None
    train_cats = None
    use_pretrained = getattr(_args, 'use_pretrained_encodings', True)
    use_trained_encoder = getattr(_args, 'use_trained_encoder', use_pretrained)
    allow_reencode = bool(getattr(_args, "allow_inference_reencode", True))
    
    if use_pretrained:
        checked_paths = _candidate_train_encoding_paths()
        print(f"[Encodings] use_pretrained={use_pretrained}, checking paths: {checked_paths}")

        for npz_path in checked_paths:
            if not os.path.exists(npz_path):
                continue
            try:
                print(f"[Encodings] Loading saved encodings from {npz_path}")
                with st.spinner("⚡ Loading saved encodings…"):
                    npz = np.load(npz_path)
                    train_encs = npz['embeddings']
                    train_cats = npz['cats']
                    st.success(f"✅ Loaded {len(train_encs)} saved encodings ({train_encs.nbytes / 1024 / 1024:.1f} MB)")
                    print(f"[Encodings] Successfully loaded {len(train_encs)} encodings")
                break
            except Exception as e:
                print(f"[Encodings] Error loading {npz_path}: {e}")
                st.warning(f"⚠️ Could not load saved encodings ({npz_path}), falling back to re-encoding: {e}")
                train_encs = None
        if train_encs is None:
            print(f"[Encodings] No saved encodings found in checked paths: {checked_paths}")
            st.info(f"ℹ️ No saved encodings found. Checked {len(checked_paths)} path(s).")
    else:
        print(f"[Encodings] use_pretrained_encodings is False, re-encoding from model")
        st.info("ℹ️ Using model re-encoding (not loading saved encodings)")

    if train_encs is None:
        if not allow_reencode:
            raise RuntimeError(
                "Saved train encodings are required for bulk inference, but none were found. "
                "Rebuild/sync the best-model artifacts so train_encodings.npz is present; "
                "the app will not re-encode the training split during bulk inference."
            )
        if use_trained_encoder:
            st.info("ℹ️ Encoding with trained model weights from the selected checkpoint")
        # ---- Slow path: encode training data with model ----
        from otitenet.train.train_triplet_new import TrainAE
        # from otitenet.data.data_getters import get_images_loaders, get_empty_traces
        from otitenet.app.model_loading import load_model_and_prototypes

        # Initialize training wrapper for encoding
        train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                        log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
                        log_mlflow=False, groupkfold=_args.groupkfold)
        train.n_batches = len(unique_batches)
        train.n_cats = len(unique_labels)
        train.unique_batches = unique_batches
        train.unique_labels = unique_labels
        train._batch_encoder = _fit_batch_encoder(unique_batches)
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
            batch_encoder=train._batch_encoder
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

        train_encs = np.concatenate(lists['train']['encoded_values'])
        train_cats = np.concatenate(lists['train']['cats'])
    
    if classifier_kind == "linearsvc":
        classifier = fit_linearsvc_classifier(train_encs, train_cats)
        print(f"Using LinearSVC classifier on embeddings (train_n={train_encs.shape[0]})")
    elif classifier_kind == "logisticregression":
        classifier = fit_logreg_classifier(train_encs, train_cats)
        print(f"Using LogisticRegression classifier on embeddings (train_n={train_encs.shape[0]})")
    elif _args.classif_loss not in ['ce', 'hinge']:
        nn_count = int(_args.n_neighbors)
        classifier = fit_knn_classifier(train_encs, train_cats, n_neighbors=nn_count, metric='minkowski')
        print(f"Using KNN classifier on embeddings (k={nn_count}, train_n={train_encs.shape[0]})")
    else:
        classifier = fit_knn_classifier(train_encs, train_cats, n_neighbors=1, metric='minkowski')
        print(f"Using KNN classifier on embeddings (k=1, train_n={train_encs.shape[0]})")

    cache[key] = {'classifier': classifier, 'unique_labels': unique_labels, 'classifier_kind': classifier_kind}
    st.session_state['train_embeddings'] = train_encs
    st.session_state['train_labels'] = train_cats

    return classifier, unique_labels


def get_or_build_knn(_args, data, unique_labels, unique_batches, prototypes):
    """Backward-compatible wrapper for older callers."""
    return get_or_build_embedding_classifier(_args, data, unique_labels, unique_batches, prototypes)


def _safe_generate_gradcam_for_prediction(filename, _args, model, image, prototypes, device_str, complete_log_path):
    """Generate Grad-CAM montages during prediction, but never fail inference if Grad-CAM fails.

    From now on fresh predictions compute the last four layers by default, so the
    observer page can show four wide composite panels. The legacy one-layer buttons
    elsewhere in the app are left unchanged.
    """
    try:
        from otitenet.logging.grad_cam import log_grad_cam_all_classes
        from otitenet.app.utils import strip_extension

        current_layer = int(getattr(_args, "grad_cam_layer", 7))
        current_alpha = float(getattr(_args, "grad_cam_alpha", 0.55))
        first_layer = max(0, current_layer - 3)
        layers_to_compute = list(range(first_layer, current_layer + 1))

        base_name = strip_extension(os.path.basename(str(filename)))
        image_output_dir = os.path.join(str(complete_log_path), base_name)
        os.makedirs(image_output_dir, exist_ok=True)

        class_prototypes = {}
        if isinstance(prototypes, dict):
            class_prototypes = prototypes.get("class", {}).get("train", {}) or {}

        inputs = {"queries": {"inputs": [image]}}
        generated_paths = []
        for layer in layers_to_compute:
            log_grad_cam_all_classes(
                model,
                0,
                inputs,
                "queries",
                image_output_dir,
                base_name,
                class_prototypes,
                device=device_str,
                layer=int(layer),
                alpha=current_alpha,
            )
            montage_path = os.path.join(image_output_dir, f"{base_name}_grad_cam_all_classes_layer{int(layer)}.png")
            if os.path.exists(montage_path):
                generated_paths.append(montage_path)

        return generated_paths[-1] if generated_paths else image_output_dir
    except Exception as gradcam_exc:
        print(f"[Grad-CAM] skipped for {filename}: {gradcam_exc}")
        return None


def run_analysis_on_file(
    filename,
    file_bytes,
    _args,
    cursor,
    conn,
    force_reanalyze=False,
    show_validation_metrics=True,
    fast_infer=False,
    quiet=False,
    generate_gradcam=True,
):
    """Run complete analysis on a file and return results.

    Set quiet=True for bulk/background inference so per-image Streamlit
    traces do not flood the UI.
    """
    with _quiet_streamlit_messages(quiet):
        return _run_analysis_on_file_impl(
            filename,
            file_bytes,
            _args,
            cursor,
            conn,
            force_reanalyze=force_reanalyze,
            show_validation_metrics=show_validation_metrics,
            fast_infer=fast_infer,
            generate_gradcam=generate_gradcam,
        )


def _run_analysis_on_file_impl(
    filename,
    file_bytes,
    _args,
    cursor,
    conn,
    force_reanalyze=False,
    show_validation_metrics=True,
    fast_infer=False,
    generate_gradcam=True,
):
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
        
        try:
            model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
                load_model_and_prototypes(_args)
            batch_encoder = _fit_batch_encoder(unique_batches)
        except FileNotFoundError as exc:
            selected_log_path = getattr(_args, "log_path", None)
            st.error(
                "Selected model artifacts are missing. Choose another model or rerun training for this row. "
                f"Log Path: {selected_log_path or 'not set'}"
            )
            raise exc
        
        save_path = os.path.join('data/queries', filename)
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
        
        # Load saved hyperparameters
        params_path = os.path.join(f'logs/best_models/{_args.task}/{_args.model_name}',
                                   f'{dataset_path_segment(_args.path)}/nsize{_args.new_size}/{_args.fgsm}/{_args.n_calibration}/'
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
        if not hasattr(_args, "n_epochs"):
            _args.n_epochs = 1

        if not hasattr(_args, "epoch"):
            _args.epoch = 0
        if not fast_infer:
            # Build or reuse cached embedding classifier once per model selection.
            embedding_classifier, unique_labels_knn = get_or_build_embedding_classifier(_args, data, unique_labels, unique_batches, prototypes)
            nets = {'cnn': shap_model, 'embedding_classifier': embedding_classifier}
        else:
            loaders = None
            unique_labels_knn = unique_labels
            nets = {'cnn': shap_model, 'embedding_classifier': None}
        
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
        pred_probas = {}
        with torch.no_grad():
            emb_tensor = nets['cnn'](image.to(device_str))
            if fast_infer:
                class_protos = prototypes.get('class', {}).get('train', {}) if isinstance(prototypes, dict) else {}
                pred_lbl_fast, pred_probas_fast = _predict_with_prototype_distance_ratio_proba(
                    emb_tensor, class_protos, dist_fct_name=str(_args.dist_fct).lower()
                )
                if pred_lbl_fast is None:
                    pred_label = unique_labels[0]
                    pred_confidence = 0.0
                else:
                    pred_label = pred_lbl_fast
                    pred_confidence = float(pred_probas_fast.get(pred_label, 0.0))
                pred_probas = _normalize_probability_map(pred_probas_fast, unique_labels)
            else:
                # Check which classification method to use
                use_prototypes = (_args.prototypes_to_use in ['combined', 'class'] and 
                                prototypes.get('class', {}).get('train'))
                use_kde = (use_prototypes and 
                          getattr(_args, 'prototype_kind', 'distance').lower() == 'kde')
                classifier_kind = _classifier_kind_from_config(_head_config_from_args(_args))
                
                embedding = emb_tensor.detach().cpu().numpy()
                
                if use_kde:
                    # Use KDE classifier
                    if 'train_embeddings' not in st.session_state or 'train_labels' not in st.session_state:
                        # Fallback to prototype distance if KDE data not available
                        class_protos = prototypes.get('class', {}).get('train', {})
                        pred_lbl, pred_probas = _predict_with_prototype_distance_ratio_proba(
                            emb_tensor,
                            class_protos,
                            dist_fct_name=str(_args.dist_fct).lower(),
                        )
                        pred_label = pred_lbl if pred_lbl is not None else unique_labels[0]
                        pred_confidence = float(pred_probas.get(pred_label, 0.0)) if pred_lbl is not None else 0.0
                        pred_probas = _normalize_probability_map(pred_probas, unique_labels)
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
                        pred_probas = {str(pred_label): float(pred_confidence)}
                
                elif use_prototypes:
                    # Use prototype distance classification
                    class_protos = prototypes.get('class', {}).get('train', {})
                    pred_lbl, pred_probas = _predict_with_prototype_distance_ratio_proba(
                        emb_tensor,
                        class_protos,
                        dist_fct_name=str(_args.dist_fct).lower(),
                    )
                    if pred_lbl is None:
                        pred_label = unique_labels[0]
                        pred_confidence = 0.0
                    else:
                        pred_label = pred_lbl
                        pred_confidence = float(pred_probas.get(pred_label, 0.0))
                    pred_probas = _normalize_probability_map(pred_probas, unique_labels)
                
                else:
                    # Use the selected embedding classifier head.
                    embedding_classifier = nets['embedding_classifier']
                    if hasattr(embedding_classifier, "predict_proba"):
                        pred_probs = embedding_classifier.predict_proba(embedding)
                    elif hasattr(embedding_classifier, "decision_function"):
                        pred_probs = _decision_scores_to_proba(embedding_classifier.decision_function(embedding))
                    else:
                        pred_raw = embedding_classifier.predict(embedding)
                        pred_probs = np.zeros((1, len(unique_labels)), dtype=float)
                        try:
                            pred_idx = int(pred_raw[0])
                            pred_probs[0, pred_idx] = 1.0
                        except Exception:
                            pred_probs[0, 0] = 1.0
                    pred_class = int(np.argmax(pred_probs, axis=1)[0])
                    pred_confidence = float(pred_probs[0, pred_class]) if pred_probs.ndim == 2 else float(np.max(pred_probs))
                    raw_class_labels = getattr(embedding_classifier, "classes_", unique_labels)
                    class_labels = _classifier_classes_to_labels(raw_class_labels, unique_labels)
                    pred_label = class_labels[pred_class] if pred_class < len(class_labels) else unique_labels[pred_class]
                    pred_probas = _normalize_probability_map(pred_probs, class_labels)
        
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
        elif _classifier_kind_from_config(_head_config_from_args(_args)) == "linearsvc":
            method_label = "LinearSVC"
        elif _classifier_kind_from_config(_head_config_from_args(_args)) == "logisticregression":
            method_label = "LogisticRegression"
        else:
            method_label = f"KNN (k={_args.n_neighbors})"
        
        st.write(f"**Predicted Label ({method_label}):** {pred_label} ({pred_confidence:.2f} confidence)")
        st.caption(
            f"Model run → id: {_args.model_id}, name: {_args.model_name}, size: {_args.new_size}, fgsm: {_args.fgsm}, dist: {_args.dist_fct}, protos: {_args.prototypes_to_use}, normalize: {_args.normalize}"
        )
        
        # Insert into database
        insert_score(
            cursor,
            conn,
            filename,
            _args,
            pred_label,
            pred_confidence,
            complete_log_path,
            class_scores=pred_probas,
        )
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

        gradcam_path = None
        if generate_gradcam:
            # This is intentionally best-effort so inference still succeeds if Grad-CAM fails.
            gradcam_path = _safe_generate_gradcam_for_prediction(
                filename,
                _args,
                model,
                image,
                prototypes,
                device_str,
                complete_log_path,
            )
    
    return pred_label, pred_confidence, complete_log_path, None, gradcam_path, pred_probas
