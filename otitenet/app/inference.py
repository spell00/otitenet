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
        emb = embedding_tensor.detach().cpu()
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)

        # For cosine, we maximize similarity; for euclidean, we minimize distance
        best_label = None
        best_score = -float('inf') if dist_fct_name == 'cosine' else float('inf')

        if str(dist_fct_name).lower() == 'cosine':
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        for label, proto in class_prototypes.items():
            if proto is None:
                continue
            proto_t = torch.as_tensor(proto, dtype=emb.dtype)
            if proto_t.ndim == 1:
                proto_t = proto_t.unsqueeze(0)

            if str(dist_fct_name).lower() == 'cosine':
                proto_t = torch.nn.functional.normalize(proto_t, p=2, dim=1)
                sim = torch.mm(emb, proto_t.T).mean().item()
                if sim > best_score:
                    best_score = sim
                    best_label = label
            else:
                dist = torch.cdist(emb, proto_t, p=2).mean().item()
                if dist < best_score:
                    best_score = dist
                    best_label = label

        return best_label
    except Exception:
        return None


def predict_with_prototype_distance_ratio(
    embedding_tensor: torch.Tensor,
    class_prototypes: dict,
    dist_fct_name: str = 'euclidean'
):
    """Predict using distance ratio to class prototypes.
    
    Probability is calculated as inverse distance ratio:
    prob(class) = (1/dist_to_proto) / sum(1/dist_to_all_protos)
    
    Args:
        embedding_tensor: Input embedding tensor
        class_prototypes: Dictionary mapping class labels to prototype tensors
        dist_fct_name: Distance function ('euclidean' or 'cosine')
        
    Returns:
        Class with highest probability, or None on error
    """
    try:
        emb = embedding_tensor.detach().cpu()
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)

        # Compute distances to each class prototype
        distances = {}
        for label, proto in class_prototypes.items():
            if proto is None:
                continue
            proto_t = torch.as_tensor(proto, dtype=emb.dtype)
            if proto_t.ndim == 1:
                proto_t = proto_t.unsqueeze(0)

            if str(dist_fct_name).lower() == 'cosine':
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
                proto_norm = torch.nn.functional.normalize(proto_t, p=2, dim=1)
                sim = torch.mm(emb_norm, proto_norm.T).mean().item()
                distances[label] = 1.0 - sim  # Convert similarity to distance
            else:
                dist = torch.cdist(emb, proto_t, p=2).mean().item()
                distances[label] = dist

        # Compute inverse distance probabilities
        inv_distances = {label: 1.0 / (dist + 1e-8) for label, dist in distances.items()}
        total_inv = sum(inv_distances.values())
        probas = {label: inv_dist / total_inv for label, inv_dist in inv_distances.items()}
        
        # Return class with highest probability
        best_label = max(probas, key=probas.get)
        return best_label
    except Exception as e:
        print(f"Error in prototype distance ratio prediction: {e}")
        return None


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
