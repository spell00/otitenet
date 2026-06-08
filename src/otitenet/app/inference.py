"""
Inference and prediction utilities.

Handles different prediction strategies including:
- KNN-based classification
- Prototype distance methods
- KDE (Kernel Density Estimation)
- Distance ratio methods
"""

import torch
import numpy as np
from otitenet.utils.kde import make_kde_classifier


def _prototype_min_distances(
    embedding_tensor: torch.Tensor,
    class_prototypes: dict,
    dist_fct_name: str = 'euclidean',
) -> dict:
    """Return each class distance to its closest prototype."""
    emb = embedding_tensor.detach().cpu()
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)

    distances = {}
    for label, proto in (class_prototypes or {}).items():
        if proto is None:
            continue
        proto_t = torch.as_tensor(proto, dtype=emb.dtype)
        if proto_t.ndim == 1:
            proto_t = proto_t.unsqueeze(0)
        elif proto_t.ndim > 2:
            proto_t = proto_t.reshape(proto_t.shape[0], -1)
        if proto_t.numel() == 0:
            continue

        if str(dist_fct_name).lower() == 'cosine':
            emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
            proto_norm = torch.nn.functional.normalize(proto_t, p=2, dim=1)
            distances[label] = float((1.0 - torch.mm(emb_norm, proto_norm.T)).min().item())
        else:
            distances[label] = float(torch.cdist(emb, proto_t, p=2).min().item())
    return distances


def prototype_distance_probabilities_from_distances(
    distances: dict,
    temperature: float | None = None,
) -> dict:
    """Convert closest-prototype distances into probabilities.

    Uses softmax(-distance / temperature), so nearer prototypes get higher
    probability while preserving a calibrated confidence-like ranking.
    """
    finite = {
        label: float(dist)
        for label, dist in (distances or {}).items()
        if np.isfinite(float(dist))
    }
    if not finite:
        return {label: 0.0 for label in (distances or {})}

    values = np.array(list(finite.values()), dtype=float)
    if temperature is None:
        spread = float(np.std(values))
        temperature = spread if spread > 1e-8 else 1.0
    temperature = max(float(temperature), 1e-8)
    logits = -values / temperature
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    out = {label: 0.0 for label in (distances or {})}
    out.update({label: float(prob) for label, prob in zip(finite.keys(), probs)})
    return out


def predict_label_from_prototypes(
    embedding_tensor: torch.Tensor,
    class_prototypes: dict,
    dist_fct_name: str = 'euclidean'
):
    """Return the nearest class label to the embedding using provided prototypes.
    
    Works with either Euclidean or cosine distance, averaging over multiple
    prototypes per class.
    
    Args:
        embedding_tensor: Input embedding tensor
        class_prototypes: Dictionary mapping class labels to prototype tensors
        dist_fct_name: Distance function ('euclidean' or 'cosine')
        
    Returns:
        Predicted class label, or None on error
    """
    try:
        distances = _prototype_min_distances(embedding_tensor, class_prototypes, dist_fct_name)
        probas = prototype_distance_probabilities_from_distances(distances)
        return max(probas, key=probas.get) if probas else None
    except Exception:
        return None


def predict_with_prototype_distance_ratio(
    embedding_tensor: torch.Tensor,
    class_prototypes: dict,
    dist_fct_name: str = 'euclidean'
):
    """Predict using distance ratio to class prototypes.
    
    Probability is calculated as softmax over negative closest-prototype
    distances, so the nearest prototype gets highest priority.
    
    Args:
        embedding_tensor: Input embedding tensor
        class_prototypes: Dictionary mapping class labels to prototype tensors
        dist_fct_name: Distance function ('euclidean' or 'cosine')
        
    Returns:
        Class with highest probability, or None on error
    """
    try:
        distances = _prototype_min_distances(embedding_tensor, class_prototypes, dist_fct_name)
        probas = prototype_distance_probabilities_from_distances(distances)
        return max(probas, key=probas.get)
    except Exception as e:
        print(f"Error in prototype distance ratio prediction: {e}")
        return None


def predict_with_prototype_distance_ratio_proba(
    embedding_tensor: torch.Tensor,
    class_prototypes: dict,
    dist_fct_name: str = 'euclidean'
):
    """Return both predicted label and softmax closest-prototype probabilities."""
    try:
        distances = _prototype_min_distances(embedding_tensor, class_prototypes, dist_fct_name)
        probas = prototype_distance_probabilities_from_distances(distances)
        if not probas:
            return None, {}
        return max(probas, key=probas.get), probas
    except Exception as e:
        print(f"Error in prototype distance ratio prediction: {e}")
        return None, {}


def predict_with_kde(
    embedding_tensor: torch.Tensor,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    unique_labels: np.ndarray,
    kde_kernel: str = 'gaussian',
    kde_bandwidth: str = 'scott'
):
    """Predict using Kernel Density Estimation.
    
    Args:
        embedding_tensor: Input embedding (torch tensor)
        train_embeddings: Training embeddings (n_samples, n_features)
        train_labels: Training labels (n_samples,)
        unique_labels: Unique label values
        kde_kernel: KDE kernel type
        kde_bandwidth: KDE bandwidth parameter
        
    Returns:
        Tuple of (predicted_label, confidence), or (None, None) on error
    """
    try:
        emb_np = embedding_tensor.detach().cpu().numpy()
        if emb_np.ndim == 1:
            emb_np = emb_np.reshape(1, -1)
        
        # Fit KDE classifier on training data
        kde = make_kde_classifier(
            kernel=kde_kernel,
            bandwidth=kde_bandwidth,
            learnable=False,
            soft=True
        )
        kde.fit(train_embeddings, train_labels)
        
        # Predict
        preds = kde.predict(emb_np)
        probas = kde.predict_proba(emb_np)
        
        # Get label and confidence
        pred_label = unique_labels[preds[0]] if preds[0] < len(unique_labels) else unique_labels[0]
        confidence = float(np.max(probas[0])) if probas.ndim == 2 else float(probas[0])
        
        return pred_label, confidence
    except Exception as e:
        print(f"Error in KDE prediction: {e}")
        return None, None


def predict_with_baseline(
    embedding_tensor: torch.Tensor,
    classifier_obj,
    unique_labels: np.ndarray,
):
    """Predict using a pre-fitted baseline classifier.
    
    Args:
        embedding_tensor: Input embedding (torch tensor)
        classifier_obj: Fitted sklearn-style classifier
        unique_labels: Unique label values (ordered)
        
    Returns:
        Predicted class label, or None on error
    """
    try:
        emb_np = embedding_tensor.detach().cpu().numpy()
        if emb_np.ndim == 1:
            emb_np = emb_np.reshape(1, -1)
            
        preds = classifier_obj.predict(emb_np)
        # If preds are integers, map to unique_labels
        if isinstance(preds[0], (int, np.integer)):
            return unique_labels[preds[0]]
        return preds[0]
    except Exception as e:
        print(f"Error in baseline prediction: {e}")
        return None


def predict_label(
    embedding_tensor: torch.Tensor,
    _args,
    unique_labels: np.ndarray,
    train_embeddings: np.ndarray = None,
    train_labels: np.ndarray = None,
    class_prototypes: dict = None,
    best_config: str = None,
    classifier_obj = None
):
    """Unified prediction entry point.
    
    Args:
        embedding_tensor: Input embedding (torch tensor)
        _args: Model/inference arguments
        unique_labels: Array of unique labels
        train_embeddings: Optional training embeddings (for KNN/KDE)
        train_labels: Optional training labels (for KNN/KDE)
        class_prototypes: Optional prototypes (for Prototype methods)
        best_config: Optional specific best configuration (e.g. 'baseline_xgboost')
        classifier_obj: Optional pre-fitted classifier object
        
    Returns:
        Predicted label
    """
    config = best_config if best_config else str(_args.n_neighbors)
    
    # 1. Prototype methods
    if config.startswith('protot_'):
        return predict_label_from_prototypes(
            embedding_tensor, class_prototypes, 
            getattr(_args, 'dist_fct', 'euclidean')
        )
        
    # 2. KDE methods
    if config.startswith('kde'):
        label, _ = predict_with_kde(
            embedding_tensor, train_embeddings, train_labels, unique_labels
        )
        return label
        
    # 3. Baseline methods
    if config.startswith('baseline_') and classifier_obj is not None:
        return predict_with_baseline(embedding_tensor, classifier_obj, unique_labels)
        
    # 4. Default: KNN
    from otitenet.ml import fit_knn_classifier
    try:
        k_val = int(config)
    except:
        k_val = 5
        
    knn = fit_knn_classifier(train_embeddings, train_labels, n_neighbors=k_val)
    return predict_with_baseline(embedding_tensor, knn, unique_labels)
