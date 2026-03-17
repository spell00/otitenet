"""
Classifier hyperparameter optimization.

Functions for optimizing classifier hyperparameters.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from otitenet.logging.metrics import MCC
from otitenet.utils.encoding_utils import compute_prototypes_by_strategy, flatten_prototype_dict
from .evaluation import evaluate_knn_with_k_search, evaluate_baseline_classifiers
from .classifiers import predict_with_prototypes


def optimize_k_neighbors(train_encs: np.ndarray, train_cats: np.ndarray,
                        valid_encs: np.ndarray, valid_cats: np.ndarray,
                        min_k: int = 1, max_k: int = 20) -> Tuple[int, float, List[Dict]]:
    """Optimize number of neighbors for KNN.
    
    Args:
        train_encs: Training embeddings
        train_cats: Training labels
        valid_encs: Validation embeddings
        valid_cats: Validation labels
        min_k: Minimum k to try
        max_k: Maximum k to try
        
    Returns:
        Tuple of (best_k, best_mcc, mcc_per_k_list)
    """
    return evaluate_knn_with_k_search(
        train_encs, train_cats, valid_encs, valid_cats, min_k, max_k
    )


def optimize_prototype_components(train_encs: np.ndarray, train_cats: np.ndarray,
                                  valid_encs: np.ndarray, valid_cats: np.ndarray,
                                  strategies: Optional[List[str]] = None,
                                  max_components: int = 5,
                                  metric: str = 'euclidean') -> Dict:
    """Optimize number of components for prototype-based classification.
    
    Args:
        train_encs: Training embeddings
        train_cats: Training labels
        valid_encs: Validation embeddings
        valid_cats: Validation labels
        strategies: List of prototype strategies to try
        max_components: Maximum number of components to try
        metric: Distance metric for prototype matching
        
    Returns:
        Dictionary with results for each strategy
    """
    if strategies is None:
        strategies = ['mean', 'kmeans', 'gmm']
    
    results = {}
    
    for strategy in strategies:
        try:
            strategy_results = []
            best_mcc = -1
            best_n_components = None
            
            for n_components in range(1, max_components + 1):
                # Compute prototypes
                proto_dict = compute_prototypes_by_strategy(
                    train_encs, train_cats, strategy, n_components, random_state=1
                )
                
                # Flatten prototypes
                proto_vecs, proto_labels = flatten_prototype_dict(proto_dict)
                
                if len(proto_vecs) == 0:
                    continue
                
                # Predict on validation set
                preds = predict_with_prototypes(
                    valid_encs, proto_vecs, proto_labels, metric=metric
                )
                mcc_val = float(MCC(valid_cats, preds))
                
                strategy_results.append({
                    'n_components': n_components,
                    'mcc': mcc_val,
                    'n_prototypes': len(proto_vecs)
                })
                
                if mcc_val > best_mcc:
                    best_mcc = mcc_val
                    best_n_components = n_components
            
            results[strategy] = {
                'best_mcc': best_mcc,
                'best_n_components': best_n_components,
                'per_components': strategy_results
            }
        except Exception as e:
            results[strategy] = {'best_mcc': None, 'error': str(e)}
    
    return results


def find_best_classifier(train_encs: np.ndarray, train_cats: np.ndarray,
                        valid_encs: np.ndarray, valid_cats: np.ndarray,
                        min_k: int = 1, max_k: int = 20,
                        include_knn: bool = True,
                        include_baselines: bool = True,
                        include_prototypes: bool = True,
                        prototype_strategies: Optional[List[str]] = None,
                        max_components: int = 5) -> Tuple[str, float, Dict]:
    """Find the best classifier among all available options.
    
    This is the main entry point for comprehensive classifier comparison.
    
    Args:
        train_encs: Training embeddings
        train_cats: Training labels
        valid_encs: Validation embeddings
        valid_cats: Validation labels
        min_k: Minimum k for KNN
        max_k: Maximum k for KNN
        include_knn: Whether to include KNN optimization
        include_baselines: Whether to include baseline classifiers
        include_prototypes: Whether to include prototype-based methods
        prototype_strategies: List of prototype strategies
        max_components: Maximum components for prototypes
        
    Returns:
        Tuple of (best_method_name, best_mcc, detailed_results)
    """
    all_results = {}
    
    best_method = None
    best_mcc = -1.0
    best_config = None

    # 1. KNN optimization
    if include_knn:
        best_k_knn, best_mcc_knn, mcc_per_k = optimize_k_neighbors(
            train_encs, train_cats, valid_encs, valid_cats, min_k, max_k
        )
        all_results['knn'] = {
            'best_k': best_k_knn,
            'best_mcc': best_mcc_knn,
            'mcc_per_k': mcc_per_k
        }
        if best_mcc_knn > best_mcc:
            best_method = 'knn'
            best_mcc = best_mcc_knn
            best_config = str(best_k_knn)
    else:
        all_results['knn'] = {}
    
    # 2. Baseline classifiers
    if include_baselines:
        baseline_results = evaluate_baseline_classifiers(
            train_encs, train_cats, valid_encs, valid_cats
        )
        all_results['baselines'] = baseline_results
        
        for clf_name, clf_data in baseline_results.items():
            clf_mcc = clf_data.get('mcc')
            if clf_mcc is not None and clf_mcc > best_mcc:
                best_method = f'baseline_{clf_name}'
                best_mcc = clf_mcc
                best_config = f'baseline_{clf_name}'
    
    # 3. Prototype-based methods
    if include_prototypes:
        proto_results = optimize_prototype_components(
            train_encs, train_cats, valid_encs, valid_cats,
            strategies=prototype_strategies,
            max_components=max_components
        )
        all_results['prototypes'] = proto_results
        
        for strategy, strategy_data in proto_results.items():
            strategy_mcc = strategy_data.get('best_mcc')
            if strategy_mcc is not None and strategy_mcc > best_mcc:
                best_method = 'prototype'
                best_mcc = strategy_mcc
                n_comp = strategy_data.get('best_n_components')
                best_config = f'protot_{strategy}_{n_comp}'
    
    return best_config, float(best_mcc), all_results
