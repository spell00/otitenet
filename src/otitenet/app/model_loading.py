"""
Model loading and caching utilities.

Handles loading trained models, prototypes, and associated metadata
with proper caching and fallback logic.
"""

import os
import json
import pickle
import random
import torch
import numpy as np
import streamlit as st
from otitenet.data.data_getters import GetData
from otitenet.models.cnn import Net, Net_shap
from otitenet.app.utils import get_model_params_path


@st.cache_resource
def resolve_model_paths(base_dir: str, normalize_val: str, dist_fct_val: str, requested_k: int):
    """Find model/prototype paths, falling back across dist_fct and knn folders if needed.
    
    Args:
        base_dir: Base directory containing model files
        normalize_val: Normalization setting ('yes'/'no')
        dist_fct_val: Distance function ('euclidean'/'cosine')
        requested_k: Requested number of neighbors
        
    Returns:
        Tuple of (model_path, prototype_path, resolved_k)
    """
    norm_base_dir = os.path.join(base_dir, f'norm{normalize_val}')
    
    # Try exact dist_fct first
    all_candidates = []
    dist_fct_options = [dist_fct_val, 'euclidean', 'cosine', 'none']  # fallback order
    
    for dist_fct in dist_fct_options:
        norm_dir = os.path.join(norm_base_dir, f'dist_{dist_fct}')
        if not os.path.exists(norm_dir):
            continue
            
        # Try exact k first
        requested_dir = os.path.join(norm_dir, f'knn{requested_k}')
        model_path = os.path.join(requested_dir, 'model.pth')
        proto_path = os.path.join(requested_dir, 'prototypes.pkl')
        if os.path.exists(model_path) and os.path.exists(proto_path):
            return model_path, proto_path, requested_k
        
        # Collect all available k values in this dist folder
        try:
            for entry in os.listdir(norm_dir):
                if not entry.startswith('knn'):
                    continue
                k_str = entry[3:]
                try:
                    k_val = int(k_str)
                except ValueError:
                    continue
                cand_dir = os.path.join(norm_dir, entry)
                cand_model = os.path.join(cand_dir, 'model.pth')
                cand_proto = os.path.join(cand_dir, 'prototypes.pkl')
                if os.path.exists(cand_model) and os.path.exists(cand_proto):
                    diff = abs(k_val - requested_k)
                    all_candidates.append((diff, k_val, cand_model, cand_proto))
        except Exception:
            continue
    
    # Return best candidate if any found
    if all_candidates:
        all_candidates.sort()
        _, best_k, best_model, best_proto = all_candidates[0]
        return best_model, best_proto, best_k
    
    # Fallback: return requested paths (will trigger FileNotFoundError downstream)
    norm_dir = os.path.join(norm_base_dir, f'dist_{dist_fct_val}')
    requested_dir = os.path.join(norm_dir, f'knn{requested_k}')
    model_path = os.path.join(requested_dir, 'model.pth')
    proto_path = os.path.join(requested_dir, 'prototypes.pkl')
    return model_path, proto_path, requested_k


def load_saved_search_params(model_log_dir: str) -> dict:
    """Load hyperparameter search params (if any) from params.json in a log dir.
    
    Args:
        model_log_dir: Directory containing params.json
        
    Returns:
        Dictionary of search parameters, or empty dict if not found
    """
    params_path = os.path.join(model_log_dir, "params.json")
    if not os.path.exists(params_path):
        return {}
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        search_params = payload.get('search_params') or {}
        if isinstance(search_params, dict):
            return search_params
    except Exception:
        return {}
    return {}


def load_model_parameters(model_dir: str) -> dict:
    """Load complete model parameters from parameters.json saved during training.
    
    This includes all optimized and fixed hyperparameters.
    
    Args:
        model_dir: Directory containing the saved model and parameters.json
        
    Returns:
        Dictionary with all parameters, or empty dict if file doesn't exist
    """
    params_path = os.path.join(model_dir, 'parameters.json')
    if not os.path.exists(params_path):
        print(f"⚠️  No parameters.json found at {params_path}")
        return {}
    
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        print(f"✅ Loaded parameters from {params_path}")
        return params
    except Exception as e:
        print(f"❌ Error loading parameters from {params_path}: {e}")
        return {}


@st.cache_resource
def _load_model_and_prototypes_cached(
    model_name, task, new_size, fgsm, n_calibration, classif_loss, dloss,
    prototypes_to_use, n_positives, n_negatives, n_neighbors, normalize,
    dist_fct, device, path, valid_dataset
):
    # Reconstruct namespace locally
    import argparse
    _args = argparse.Namespace(
        model_name=model_name, task=task, new_size=int(new_size), fgsm=str(fgsm),
        n_calibration=str(n_calibration), classif_loss=str(classif_loss), dloss=str(dloss),
        prototypes_to_use=str(prototypes_to_use), n_positives=str(n_positives),
        n_negatives=str(n_negatives), n_neighbors=int(n_neighbors), normalize=str(normalize),
        dist_fct=str(dist_fct), device=str(device), path=str(path), valid_dataset=str(valid_dataset)
    )

    # Ensure seeds are set for reproducible split
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    
    params = get_model_params_path(_args)
    parts = params.split('/')
    # remove the last three components: norm..., dist_..., knn...
    base_params = '/'.join(parts[:-3])
    base_dir = f'logs/best_models/{_args.task}/{_args.model_name}/{base_params}'
    model_path, proto_path, resolved_k = resolve_model_paths(
        base_dir, _args.normalize, _args.dist_fct, int(_args.n_neighbors)
    )

    model_dir = os.path.dirname(model_path)

    data_getter = GetData(_args.path, _args.valid_dataset, _args, manifest_dir=model_dir)
    data, unique_labels, unique_batches = data_getter.get_variables()
    n_cats = len(unique_labels)
    n_batches = len(unique_batches)

    if not (os.path.exists(model_path) and os.path.exists(proto_path)):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # If we had to fall back, keep args in sync so downstream cache keys are correct
    try:
        _args.n_neighbors = resolved_k
    except Exception:
        pass

    try:
        model = Net(_args.device, n_cats=n_cats, n_batches=n_batches,
                    model_name=_args.model_name, is_stn=0,
                    n_subcenters=n_batches)
        model.load_state_dict(torch.load(model_path, map_location=_args.device))
        model.to(_args.device)
        model.eval()

        shap_model = Net_shap(_args.device, n_cats=n_cats, n_batches=n_batches,
                              model_name=_args.model_name, is_stn=0,
                              n_subcenters=n_batches)
        shap_model.load_state_dict(torch.load(model_path, map_location=_args.device))
        shap_model.to(_args.device)
        shap_model.eval()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
            print("Warning: GPU OOM during model loading. Falling back to CPU.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            _args.device = 'cpu'
            
            model = Net('cpu', n_cats=n_cats, n_batches=n_batches,
                        model_name=_args.model_name, is_stn=0,
                        n_subcenters=n_batches)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.to('cpu')
            model.eval()

            shap_model = Net_shap('cpu', n_cats=n_cats, n_batches=n_batches,
                                  model_name=_args.model_name, is_stn=0,
                                  n_subcenters=n_batches)
            shap_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            shap_model.to('cpu')
            shap_model.eval()
        else:
            raise e

    with open(proto_path, 'rb') as f:
        proto_obj = pickle.load(f)
    prototypes = {
        'combined': proto_obj.prototypes,
        'class': proto_obj.class_prototypes,
        'batch': proto_obj.batch_prototypes
    }

    return model, shap_model, prototypes, _args.new_size, _args.device, data, unique_labels, unique_batches, data_getter, resolved_k


def load_model_and_prototypes(_args):
    """Load trained model, SHAP model, and prototypes with data setup."""
    res = _load_model_and_prototypes_cached(
        model_name=_args.model_name,
        task=_args.task,
        new_size=int(_args.new_size),
        fgsm=str(_args.fgsm),
        n_calibration=str(_args.n_calibration),
        classif_loss=str(_args.classif_loss),
        dloss=str(_args.dloss),
        prototypes_to_use=str(_args.prototypes_to_use),
        n_positives=str(_args.n_positives),
        n_negatives=str(_args.n_negatives),
        n_neighbors=int(_args.n_neighbors),
        normalize=str(_args.normalize),
        dist_fct=str(_args.dist_fct),
        device=str(_args.device),
        path=str(_args.path),
        valid_dataset=str(_args.valid_dataset)
    )
    
    # Sync resolved_k back to _args
    try:
        _args.n_neighbors = res[9]
    except Exception:
        pass
        
    return res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]


#@st.cache_resource
def load_model_for_log_path(log_path: str, model_name: str, device: str = 'cpu'):
    """Cached loader for single model by log_path (used by Ensemble tab).
    
    Args:
        log_path: Full path to model logs
        model_name: Model architecture name
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        Loaded model in eval mode, or None on error
    """
    try:
        model_path = os.path.join(log_path, 'model.pth')
        if not os.path.exists(model_path):
            return None

        state_dict = torch.load(model_path, map_location=device)
        # Infer n_batches and n_subcenters from checkpoint weights
        n_batches = None
        n_subcenters = None
        n_cats = None
        if 'subcenters' in state_dict:
            subcenters_shape = state_dict['subcenters'].shape
            n_cats = subcenters_shape[0]
            n_subcenters = subcenters_shape[1]
        if 'dann.weight' in state_dict:
            n_batches = state_dict['dann.weight'].shape[0]
        # Fallbacks if not found
        if n_cats is None:
            n_cats = 2
        if n_batches is None:
            n_batches = 4
        if n_subcenters is None:
            n_subcenters = 4

        model = Net(device, n_cats=n_cats, n_batches=n_batches, model_name=model_name, is_stn=0, n_subcenters=n_subcenters)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model from {log_path}: {e}")
        return None


def clear_cached_model():
    """Clear cached model resources."""
    try:
        load_model_and_prototypes.clear()
        resolve_model_paths.clear()
        load_model_for_log_path.clear()
    except Exception:
        pass

def resolve_model_id(model_name=None, complete_log_path=None, model_id=None):
    """
    Resolve a stable model identifier from the available model fields.
    Used by analysis.py to know which model/result entry is being used.
    """
    if model_id is not None and str(model_id).strip():
        return str(model_id)

    if model_name is not None and str(model_name).strip():
        return str(model_name)

    if complete_log_path is not None and str(complete_log_path).strip():
        import os
        return os.path.basename(str(complete_log_path).rstrip("/"))

    return "unknown_model"