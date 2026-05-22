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
        from otitenet.ml.classifiers import _get_clf_metrics
        valid_mcc, valid_auc = _get_clf_metrics(knn, valid_encs, valid_cats)
        train_mcc, train_auc = _get_clf_metrics(knn, train_encs, train_cats)
        
        mcc_per_k.append({
            'k': k, 
            'valid_mcc': float(valid_mcc), 
            'valid_auc': float(valid_auc),
            'train_mcc': float(train_mcc),
            'train_auc': float(train_auc)
        })
        
        if valid_mcc > best_mcc:
            best_k = k
            best_mcc = valid_mcc
    
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
    label_map = classifiers.pop('label_map', {})
    label_encoder = classifiers.pop('label_encoder', None)
    results['label_map'] = label_map
    results['label_encoder'] = label_encoder
    
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
            mcc_val = clf['valid_mcc']
            train_mcc = clf['train_mcc']
            results[name] = {
                'mcc': mcc_val, 'train_mcc': train_mcc, 'classifier': clf['classifier'],
                'valid_auc': clf.get('valid_auc'), 'test_auc': clf.get('test_auc'),
                'train_auc': clf.get('train_auc')
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
                        'mcc': mcc_val,
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
                            include_baselines: bool = True,
                            test_encs: Optional[np.ndarray] = None,
                            test_cats: Optional[np.ndarray] = None) -> Dict:
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
        'valid_mcc': best_mcc_knn,
        'mcc_per_k': mcc_per_k
    }
    # Add AUC and test metrics for best k
    for item in mcc_per_k:
        if item['k'] == best_k:
            results['knn']['valid_auc'] = item.get('valid_auc')
            break
            
    if test_encs is not None and test_cats is not None:
        from otitenet.ml.classifiers import fit_knn_classifier, _get_clf_metrics
        best_knn = fit_knn_classifier(train_encs, train_cats, n_neighbors=best_k)
        test_mcc, test_auc = _get_clf_metrics(best_knn, test_encs, test_cats)
        results['knn']['test_mcc'] = test_mcc
        results['knn']['test_auc'] = test_auc
    
    # Baseline classifiers
    if include_baselines:
        from otitenet.ml.classifiers import _get_clf_metrics
        baseline_results = evaluate_baseline_classifiers(
            train_encs, train_cats, valid_encs, valid_cats
        )
        if test_encs is not None and test_cats is not None:
            # Use LabelEncoder if present for consistent label mapping
            label_encoder = baseline_results.get('label_encoder')
            label_map = baseline_results.get('label_map', {})
            test_cats_int = None
            
            if label_encoder:
                try:
                    test_cats_int = label_encoder.transform(test_cats)
                except Exception:
                    test_cats_int = None
            
            if test_cats_int is None and label_map:
                try:
                    # Fallback to manual string-safe lookup
                    test_cats_int = np.array([label_map.get(str(x)) for x in test_cats])
                    if None in test_cats_int:
                        test_cats_int = None
                except Exception:
                    test_cats_int = None
                    
            for clf_name, clf_data in baseline_results.items():
                if clf_name in ['label_map', 'label_encoder']:
                    continue
                clf_obj = clf_data.get('classifier')
                if clf_obj:
                    # Defensive check: if double-wrapped as dict, extract object
                    if isinstance(clf_obj, dict) and 'classifier' in clf_obj:
                        clf_obj = clf_obj['classifier']
                        
                    if clf_obj and hasattr(clf_obj, 'predict'):
                        test_mcc, test_auc = _get_clf_metrics(clf_obj, test_encs, test_cats, y_int=test_cats_int)
                        clf_data['test_mcc'] = test_mcc
                        clf_data['test_auc'] = test_auc
        results['baselines'] = baseline_results
    
    # KDE classifiers
    if include_kde:
        kde_result = evaluate_kde_classifiers(
            train_encs, train_cats, valid_encs, valid_cats
        )
        if kde_result.get('mcc', -1) > -1:
            results['kde'] = kde_result
            
    # Find overall best and populate top-level metrics
    best_method, best_mcc, best_params = compare_classifiers(results)
    results['best_valid_mcc'] = best_mcc
    
    if best_method == 'knn':
        results['best_valid_auc'] = results['knn'].get('valid_auc')
        results['best_test_mcc'] = results['knn'].get('test_mcc')
        results['best_test_auc'] = results['knn'].get('test_auc')
    elif best_method.startswith('baseline_'):
        bl_name = best_method.replace('baseline_', '')
        bl_data = results.get('baselines', {}).get(bl_name, {})
        results['best_valid_auc'] = bl_data.get('valid_auc')
        results['best_test_mcc'] = bl_data.get('test_mcc')
        results['best_test_auc'] = bl_data.get('test_auc')
    elif best_method == 'kde':
        results['best_valid_auc'] = results.get('kde', {}).get('valid_auc')
        results['best_test_mcc'] = results.get('kde', {}).get('test_mcc')
        results['best_test_auc'] = results.get('kde', {}).get('test_auc')
        
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
    baselines_block = results.get('baselines', {})
    if not isinstance(baselines_block, dict):
        baselines_block = {}
    for baseline_name, baseline_data in baselines_block.items():
        if baseline_name in ('label_map', 'label_encoder') or not isinstance(baseline_data, dict):
            continue
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
