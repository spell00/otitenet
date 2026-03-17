"""
Machine Learning module for classical ML algorithms and evaluation.

This module contains all non-deep-learning classification code including:
- KNN classifiers
- Baseline classifiers (LogReg, NB, SVC)
- KDE classifiers
- Prototype-based classification
- Classifier evaluation and comparison
- Hyperparameter optimization for classifiers
"""

from .classifiers import (
    fit_knn_classifier,
    fit_baseline_classifiers,
    fit_kde_classifier,
    predict_with_prototypes,
    build_knn_from_model,
)

from .evaluation import (
    evaluate_knn_with_k_search,
    evaluate_all_classifiers,
    compare_classifiers,
)

from .optimization import (
    optimize_k_neighbors,
    optimize_prototype_components,
    find_best_classifier,
)

__all__ = [
    # Classifier fitting
    'fit_knn_classifier',
    'fit_baseline_classifiers',
    'fit_kde_classifier',
    'predict_with_prototypes',
    'build_knn_from_model',
    # Evaluation
    'evaluate_knn_with_k_search',
    'evaluate_all_classifiers',
    'compare_classifiers',
    # Optimization
    'optimize_k_neighbors',
    'optimize_prototype_components',
    'find_best_classifier',
]

