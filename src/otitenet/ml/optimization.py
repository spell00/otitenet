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
                                  metric: str = 'euclidean',
                                  test_encs: Optional[np.ndarray] = None,
                                  test_cats: Optional[np.ndarray] = None) -> Dict:
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
                
                # Predict and compute metrics
                from otitenet.ml.classifiers import _get_clf_metrics
                from sklearn.neighbors import KNeighborsClassifier
                
                # Use KNN with k=1 on prototypes for prediction
                proto_clf = KNeighborsClassifier(n_neighbors=1, metric=metric)
                proto_clf.fit(proto_vecs, proto_labels)
                
                valid_mcc, valid_auc = _get_clf_metrics(proto_clf, valid_encs, valid_cats)
                
                test_metrics = {}
                if test_encs is not None and test_cats is not None:
                    test_mcc, test_auc = _get_clf_metrics(proto_clf, test_encs, test_cats)
                    test_metrics = {'test_mcc': test_mcc, 'test_auc': test_auc}
                
                res = {
                    'n_components': n_components,
                    'mcc': valid_mcc,
                    'valid_mcc': valid_mcc,
                    'valid_auc': valid_auc,
                    'n_prototypes': len(proto_vecs),
                    **test_metrics
                }
                strategy_results.append(res)
                
                if valid_mcc > best_mcc:
                    best_mcc = valid_mcc
                    best_n_components = n_components
                    best_valid_auc = valid_auc
                    best_test_mcc = test_metrics.get('test_mcc')
                    best_test_auc = test_metrics.get('test_auc')
            
            results[strategy] = {
                'best_mcc': best_mcc,
                'best_valid_mcc': best_mcc,
                'best_valid_auc': best_valid_auc,
                'best_test_mcc': best_test_mcc,
                'best_test_auc': best_test_auc,
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
                        max_components: int = 5,
                        test_encs: Optional[np.ndarray] = None,
                        test_cats: Optional[np.ndarray] = None) -> Tuple[str, float, Dict]:
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
    best_valid_auc = None
    best_test_mcc = None
    best_test_auc = None

    # 1. KNN optimization
    if include_knn:
        best_k_knn, best_mcc_knn, mcc_per_k = optimize_k_neighbors(
            train_encs, train_cats, valid_encs, valid_cats, min_k, max_k
        )
        all_results['knn'] = {
            'best_k': best_k_knn,
            'best_mcc': best_mcc_knn,
            'valid_mcc': best_mcc_knn,
            'mcc_per_k': mcc_per_k
        }
        # Add AUC and test metrics for best k
        for item in mcc_per_k:
            if item['k'] == best_k_knn:
                all_results['knn']['valid_auc'] = item.get('valid_auc')
                break
        
        if test_encs is not None and test_cats is not None:
            from otitenet.ml.classifiers import fit_knn_classifier, _get_clf_metrics
            best_knn = fit_knn_classifier(train_encs, train_cats, n_neighbors=best_k_knn)
            test_mcc, test_auc = _get_clf_metrics(best_knn, test_encs, test_cats)
            all_results['knn']['test_mcc'] = test_mcc
            all_results['knn']['test_auc'] = test_auc
        if best_mcc_knn > best_mcc:
            best_method = 'knn'
            best_mcc = best_mcc_knn
            best_config = str(best_k_knn)
            best_valid_auc = all_results['knn'].get('valid_auc')
            best_test_mcc = all_results['knn'].get('test_mcc')
            best_test_auc = all_results['knn'].get('test_auc')
    else:
        all_results['knn'] = {}
    
    # 2. Baseline classifiers
    if include_baselines:
        from .evaluation import evaluate_baseline_classifiers
        from .classifiers import _get_clf_metrics
        # Update default baseline params to include XGBoost if requested
        clf_params = {
            'logreg_params': {}, 'linear_svc_params': {}, 'xgboost_params': {},
            'naive_bayes_params': {}, 'rbf_svc_params': {}, 'random_forest_params': {},
            'gradient_boosting_params': {}, 'decision_tree_params': {}, 'lda_params': {},
            'qda_params': {}, 'mlp_head_params': {}
        }
        baseline_results = evaluate_baseline_classifiers(
            train_encs, train_cats, valid_encs, valid_cats, classifier_params=clf_params
        )
        
        # Add test metrics for baselines
        if test_encs is not None and test_cats is not None:
            # Use LabelEncoder if present
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
                    test_mcc, test_auc = _get_clf_metrics(clf_obj, test_encs, test_cats, y_int=test_cats_int)
                    clf_data['test_mcc'] = test_mcc
                    clf_data['test_auc'] = test_auc
        
        all_results['baselines'] = baseline_results
        
        for clf_name, clf_data in baseline_results.items():
            if clf_name in ('label_map', 'label_encoder') or not isinstance(clf_data, dict):
                continue
            clf_mcc = clf_data.get('mcc')
            if clf_mcc is not None and clf_mcc > best_mcc:
                best_method = f'baseline_{clf_name}'
                best_mcc = clf_mcc
                best_config = f'baseline_{clf_name}'
                best_valid_auc = clf_data.get('valid_auc')
                best_test_mcc = clf_data.get('test_mcc')
                best_test_auc = clf_data.get('test_auc')
    
    # 3. Prototype-based methods
    if include_prototypes:
        proto_results = optimize_prototype_components(
            train_encs, train_cats, valid_encs, valid_cats,
            strategies=prototype_strategies,
            max_components=max_components,
            test_encs=test_encs,
            test_cats=test_cats
        )
        all_results['prototypes'] = proto_results
        
        for strategy, strategy_data in proto_results.items():
            strategy_mcc = strategy_data.get('best_mcc')
            if strategy_mcc is not None and strategy_mcc > best_mcc:
                best_method = 'prototype'
                best_mcc = strategy_mcc
                n_comp = strategy_data.get('best_n_components')
                best_config = f'protot_{strategy}_{n_comp}'
                best_valid_auc = strategy_data.get('best_valid_auc')
                best_test_mcc = strategy_data.get('best_test_mcc')
                best_test_auc = strategy_data.get('best_test_auc')
    
    all_results['best_valid_auc'] = best_valid_auc if 'best_valid_auc' in locals() else None
    all_results['best_test_mcc'] = best_test_mcc if 'best_test_mcc' in locals() else None
    all_results['best_test_auc'] = best_test_auc if 'best_test_auc' in locals() else None
    
    return best_config, float(best_mcc), all_results
