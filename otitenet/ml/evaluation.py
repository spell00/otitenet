"""
Classifier evaluation utilities.

Functions for evaluating classifiers and computing metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from otitenet.logging.metrics import MCC
from .classifiers import (
    fit_knn_classifier,
    fit_baseline_classifiers,
    fit_kde_classifier,
    predict_with_prototypes,
)


def evaluate_knn_with_k_search(train_encs: np.ndarray, train_cats: np.ndarray,
                               valid_encs: np.ndarray, valid_cats: np.ndarray,
                               min_k: int = 1, max_k: int = 20,
                               metric: str = 'minkowski') -> Tuple[int, float, List[Dict]]:
    """Evaluate KNN with different k values and return best k.
    
    Args:
        train_encs: Training embeddings
        train_cats: Training labels
        valid_encs: Validation embeddings
        valid_cats: Validation labels
        min_k: Minimum k to try
        max_k: Maximum k to try
        metric: Distance metric
        
    Returns:
        Tuple of (best_k, best_mcc, mcc_per_k_list)
    """
    best_k = 1
    best_mcc = -1.0
    max_k = min(max_k + 1, train_encs.shape[0] + 1)
    mcc_per_k = []
    
    for k in range(min_k, max_k):
        knn = fit_knn_classifier(train_encs, train_cats, n_neighbors=k, metric=metric)
        preds = knn.predict(valid_encs)
        mcc_val = MCC(valid_cats, preds)
        mcc_train = MCC(train_cats, knn.predict(train_encs))
        mcc_per_k.append({'k': k, 'valid_mcc': float(mcc_val), 'train_mcc': float(mcc_train)})
        
        if mcc_val > best_mcc:
            best_k = k
            best_mcc = mcc_val
    
    return best_k, float(best_mcc), mcc_per_k


def evaluate_baseline_classifiers(train_encs: np.ndarray, train_cats: np.ndarray,
                                  valid_encs: np.ndarray, valid_cats: np.ndarray,
                                  progress_placeholder=None, classifier_params: dict = None) -> Dict:
    """Evaluate multiple baseline classifiers.
    
    Args:
        train_encs: Training embeddings
        train_cats: Training labels
        valid_encs: Validation embeddings
        valid_cats: Validation labels
        progress_placeholder: Optional Streamlit placeholder for progress updates
        classifier_params: Dictionary of classifier parameters. Example:
            {
                'logreg_max_iter': 10,
                'linearsvc_max_iter': 2,
                'svc_kernel': 'rbf',
                'svc_max_iter': 2,
                'rfc_n_estimators': 1,
                'gbc_n_estimators': 1
            }
    Returns:
        Dictionary with results for each classifier
    """
    results = {}
    classifiers = fit_baseline_classifiers(train_encs, train_cats, valid_encs, valid_cats, progress_placeholder, classifier_params)
    classifier_names = list(classifiers.keys())
    total = len(classifier_names)
    for idx, (name, clf) in enumerate(classifiers.items()):
        # Update progress if placeholder is provided
        if progress_placeholder is not None:
            percent = int((idx / total) * 100)
            progress_placeholder.text(f"Evaluating {name}... ({percent}%)")
        if clf is None:
            results[name] = {'mcc': None, 'train_mcc': None, 'error': 'Failed to fit'}
            continue
        try:
            mcc_val = classifiers[name]['valid_mcc']
            train_mcc = classifiers[name]['train_mcc']
            results[name] = {
                'mcc': mcc_val, 'train_mcc': train_mcc, 'classifier': clf
            }
        except Exception as e:
            results[name] = {'mcc': None, 'train_mcc': None, 'error': str(e)}
    # Final progress update
    if progress_placeholder is not None:
        progress_placeholder.text("Baseline classifier evaluation complete. (100%)")
    return results


def evaluate_kde_classifiers(train_encs: np.ndarray, train_cats: np.ndarray,
                            valid_encs: np.ndarray, valid_cats: np.ndarray,
                            kernels: Optional[List[str]] = None,
                            bandwidths: Optional[List[str]] = None) -> Dict:
    """Evaluate KDE classifiers with different configurations.
    
    Args:
        train_encs: Training embeddings
        train_cats: Training labels
        valid_encs: Validation embeddings
        valid_cats: Validation labels
        kernels: List of kernel types to try
        bandwidths: List of bandwidth values to try
        
    Returns:
        Dictionary with best KDE configuration
    """
    if kernels is None:
        kernels = ['gaussian', 'exponential']
    if bandwidths is None:
        bandwidths = ['scott', 'silverman']
    
    best_result = {'mcc': -1, 'kernel': None, 'bandwidth': None}
    
    for kernel in kernels:
        for bandwidth in bandwidths:
            try:
                kde = fit_kde_classifier(
                    train_encs, train_cats,
                    kernel=kernel,
                    bandwidth=bandwidth,
                    learnable=False,
                    soft=True
                )
                valid_preds = kde.predict(valid_encs)
                mcc_val = float(MCC(valid_cats, valid_preds))
                mcc_train = float(MCC(train_cats, kde.predict(train_encs)))
                
                if mcc_val > best_result['mcc']:
                    best_result = {
                        'valid_mcc': mcc_val,
                        'train_mcc': mcc_train,
                        'kernel': kernel,
                        'bandwidth': bandwidth,
                        'classifier': kde
                    }
            except Exception as e:
                print(f"KDE ({kernel}, {bandwidth}) failed: {e}")
                continue
    
    return best_result


def evaluate_all_classifiers(train_encs: np.ndarray, train_cats: np.ndarray,
                            valid_encs: np.ndarray, valid_cats: np.ndarray,
                            min_k: int = 1, max_k: int = 20,
                            include_kde: bool = True,
                            include_baselines: bool = True) -> Dict:
    """Evaluate all classifier types and return comprehensive results.
    
    Args:
        train_encs: Training embeddings
        train_cats: Training labels
        valid_encs: Validation embeddings
        valid_cats: Validation labels
        min_k: Minimum k for KNN
        max_k: Maximum k for KNN
        include_kde: Whether to evaluate KDE classifiers
        include_baselines: Whether to evaluate baseline classifiers
        
    Returns:
        Dictionary with results for all classifier types
    """
    results = {}
    
    # KNN evaluation
    best_k, best_mcc_knn, mcc_per_k = evaluate_knn_with_k_search(
        train_encs, train_cats, valid_encs, valid_cats, min_k, max_k
    )
    results['knn'] = {
        'best_k': best_k,
        'best_mcc': best_mcc_knn,
        'mcc_per_k': mcc_per_k
    }
    
    # Baseline classifiers
    if include_baselines:
        baseline_results = evaluate_baseline_classifiers(
            train_encs, train_cats, valid_encs, valid_cats
        )
        results['baselines'] = baseline_results
    
    # KDE classifiers
    if include_kde:
        kde_result = evaluate_kde_classifiers(
            train_encs, train_cats, valid_encs, valid_cats
        )
        if kde_result.get('mcc', -1) > -1:
            results['kde'] = kde_result
    
    return results


def compare_classifiers(results: Dict) -> Tuple[str, float, Dict]:
    """Compare all classifier results and return the best one.
    
    Args:
        results: Dictionary from evaluate_all_classifiers
        
    Returns:
        Tuple of (best_method, best_mcc, best_params)
    """
    best_method = 'knn'
    best_mcc = results.get('knn', {}).get('best_mcc', -1)
    best_params = {'k': results.get('knn', {}).get('best_k', 1)}
    
    # Check baselines
    for baseline_name, baseline_data in results.get('baselines', {}).items():
        baseline_mcc = baseline_data.get('mcc')
        if baseline_mcc is not None and baseline_mcc > best_mcc:
            best_method = f'baseline_{baseline_name}'
            best_mcc = baseline_mcc
            best_params = {'method': baseline_name}
    
    # Check KDE
    kde_data = results.get('kde', {})
    kde_mcc = kde_data.get('mcc', -1)
    if kde_mcc > best_mcc:
        best_method = 'kde'
        best_mcc = kde_mcc
        best_params = {
            'kernel': kde_data.get('kernel'),
            'bandwidth': kde_data.get('bandwidth')
        }
    
    return best_method, best_mcc, best_params
