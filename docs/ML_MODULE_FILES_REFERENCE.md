# ML Module Files Reference

## Location
All files located in: `/home/simon/otitenet/otitenet/ml/`

## File Listing

### 1. `__init__.py` (47 lines)
**Purpose**: Package initialization and clean API exports

**Imports & Exports**:
```python
from .classifiers import (
    fit_knn_classifier,
    fit_baseline_classifiers,
    fit_kde_classifier,
    predict_with_prototypes,
)
from .evaluation import (
    evaluate_knn_with_k_search,
    evaluate_baseline_classifiers,
    evaluate_kde_classifiers,
    evaluate_all_classifiers,
    compare_classifiers,
)
from .optimization import (
    optimize_k_neighbors,
    optimize_prototype_components,
    find_best_classifier,
)

__all__ = [
    'fit_knn_classifier',
    'fit_baseline_classifiers',
    'fit_kde_classifier',
    'predict_with_prototypes',
    'evaluate_knn_with_k_search',
    'evaluate_baseline_classifiers',
    'evaluate_kde_classifiers',
    'evaluate_all_classifiers',
    'compare_classifiers',
    'optimize_k_neighbors',
    'optimize_prototype_components',
    'find_best_classifier',
]
```

---

### 2. `classifiers.py` (140 lines)
**Purpose**: Core functions for fitting different classifier types

**Key Functions**:

#### `fit_knn_classifier(train_encs, train_cats, n_neighbors=5, metric='minkowski')`
- Fits KNeighborsClassifier with auto k-adjustment
- Prevents k > n_samples
- Returns: sklearn KNeighborsClassifier instance

#### `fit_baseline_classifiers(train_encs, train_cats)`
- Fits LogisticRegression, GaussianNB, LinearSVC
- Returns: dict with all 3 classifiers

#### `fit_kde_classifier(train_encs, train_cats, kernel='gaussian', bandwidth='scott')`
- Fits KDE classifier with configurable parameters
- Returns: fitted classifier

#### `predict_with_prototypes(embeddings, proto_vecs, proto_labels, metric='euclidean')`
- Classify using nearest prototype matching
- Supports euclidean and cosine distance
- Returns: prediction array

**Dependencies**:
- numpy
- sklearn.neighbors.KNeighborsClassifier
- sklearn.linear_model.LogisticRegression
- sklearn.naive_bayes.GaussianNB
- sklearn.svm.LinearSVC
- sklearn.neighbors.KernelDensity

---

### 3. `evaluation.py` (200 lines)
**Purpose**: Grid search and classifier comparison utilities

**Key Functions**:

#### `evaluate_knn_with_k_search(train_encs, train_cats, valid_encs, valid_cats, min_k=1, max_k=10, metric='minkowski')`
- Tests KNN with k from min_k to max_k
- Returns: dict with best_k, best_mcc, mcc_per_k

#### `evaluate_baseline_classifiers(train_encs, train_cats, valid_encs, valid_cats)`
- Evaluates LogReg, NB, LinearSVC
- Returns: dict with mcc for each classifier

#### `evaluate_kde_classifiers(train_encs, train_cats, valid_encs, valid_cats, kernels=None, bandwidths=None)`
- Grid search over KDE parameters
- Returns: dict with best kernel, bandwidth, and mcc

#### `evaluate_all_classifiers(train_encs, train_cats, valid_encs, valid_cats, min_k=1, max_k=10, include_kde=True, include_baselines=True)`
- Comprehensive evaluation of all methods
- Returns: unified results dict

#### `compare_classifiers(results)`
- Finds best classifier from results
- Returns: (method_name, best_mcc, best_params)

**Metrics**:
- Matthews Correlation Coefficient (MCC) from sklearn.metrics
- Applied to classification results

**Return Format** (consistent):
```python
{
    'knn': {
        'best_k': int,
        'best_mcc': float,
        'mcc_per_k': {k: mcc, ...}
    },
    'baselines': {
        'logistic_regression': {'mcc': float},
        'naive_bayes': {'mcc': float},
        'linear_svc': {'mcc': float}
    },
    'kde': {
        'kernel': str,
        'bandwidth': str,
        'mcc': float
    }
}
```

---

### 4. `optimization.py` (170 lines)
**Purpose**: Hyperparameter optimization entry points

**Key Functions**:

#### `optimize_k_neighbors(train_encs, train_cats, valid_encs, valid_cats, min_k=1, max_k=10)`
- Wrapper for KNN k optimization
- Returns: (best_k, best_mcc)

#### `optimize_prototype_components(train_encs, train_cats, valid_encs, valid_cats, strategies=None, max_components=5)`
- Searches prototype strategies and component counts
- Strategies: 'mean', 'kmeans', 'gmm'
- Returns: (best_strategy, best_components, best_mcc)

#### `find_best_classifier(train_encs, train_cats, valid_encs, valid_cats, min_k=1, max_k=10, include_kde=True, include_baselines=True, include_prototypes=False, prototype_strategies=None, max_components=3)`
- Main comprehensive optimization function
- Searches all classifier types
- Returns: (best_k, best_mcc, all_results)

**Configuration Options**:
- `min_k`, `max_k`: KNN k search range
- `include_kde`: Whether to evaluate KDE (True/False)
- `include_baselines`: Whether to evaluate LogReg/NB/LinearSVC (True/False)
- `include_prototypes`: Whether to evaluate prototype strategies (True/False)
- `prototype_strategies`: List of strategies to try
- `max_components`: Maximum components for prototype strategies

---

### 5. `README.md` (160 lines)
**Purpose**: Comprehensive documentation

**Sections**:
1. **Overview** - Module purpose and design
2. **Installation** - How to use the module
3. **API Reference** - Detailed function documentation
4. **Usage Examples**:
   - Simple KNN fitting
   - Comprehensive classifier comparison
   - Evaluation only (no fitting)
   - Grid search over hyperparameters
5. **Design Principles**:
   - Numpy-based for framework independence
   - Sklearn-compatible classifiers
   - Consistent evaluation metrics
   - Flexible configuration
6. **Integration Guide**:
   - How to migrate from old code
   - Backward compatibility notes
   - Migration examples
7. **Benefits**:
   - Code reusability
   - Consistency across codebase
   - Easier testing and maintenance
   - Single source of truth

---

## Complete File Statistics

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 47 | Package initialization |
| `classifiers.py` | 140 | Classifier fitting |
| `evaluation.py` | 200 | Grid search & comparison |
| `optimization.py` | 170 | Hyperparameter optimization |
| `README.md` | 160 | Documentation |
| **TOTAL** | **717** | Complete ML module |

---

## Module Imports Summary

### External Dependencies
```python
# Required imports in each module
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KernelDensity
from sklearn.metrics import matthews_corrcoef as MCC
```

### Internal Dependencies
```python
# classifiers.py - No internal dependencies
# evaluation.py - Uses functions from classifiers.py
# optimization.py - Uses functions from evaluation.py
```

---

## Quick Import Examples

### Option 1: Import Individual Functions
```python
from otitenet.ml.classifiers import fit_knn_classifier
from otitenet.ml.evaluation import evaluate_all_classifiers
from otitenet.ml.optimization import find_best_classifier
```

### Option 2: Import from Package (Recommended)
```python
from otitenet.ml import (
    fit_knn_classifier,
    evaluate_all_classifiers,
    find_best_classifier
)
```

### Option 3: Import All
```python
from otitenet.ml import *
```

---

## Function Dependency Graph

```
classifiers.py (Fitting Functions)
├── fit_knn_classifier()
├── fit_baseline_classifiers()
├── fit_kde_classifier()
└── predict_with_prototypes()

evaluation.py (Evaluation Functions)
├── evaluate_knn_with_k_search()
│   └── uses fit_knn_classifier()
├── evaluate_baseline_classifiers()
│   └── uses fit_baseline_classifiers()
├── evaluate_kde_classifiers()
│   └── uses fit_kde_classifier()
├── evaluate_all_classifiers()
│   ├── uses evaluate_knn_with_k_search()
│   ├── uses evaluate_baseline_classifiers()
│   └── uses evaluate_kde_classifiers()
└── compare_classifiers()
    └── takes evaluation results

optimization.py (Optimization Functions)
├── optimize_k_neighbors()
│   └── uses evaluate_knn_with_k_search()
├── optimize_prototype_components()
│   └── custom prototype strategy loop
└── find_best_classifier()
    ├── uses evaluate_all_classifiers()
    └── optionally uses optimize_prototype_components()
```

---

## Testing Strategy

### Unit Tests Needed

**For classifiers.py**:
- `test_fit_knn_classifier()` - Various k values
- `test_fit_baseline_classifiers()` - All 3 classifiers
- `test_fit_kde_classifier()` - Different kernels/bandwidths
- `test_predict_with_prototypes()` - Distance metrics

**For evaluation.py**:
- `test_evaluate_knn_with_k_search()` - k range
- `test_evaluate_baseline_classifiers()` - All methods
- `test_evaluate_kde_classifiers()` - Grid search
- `test_evaluate_all_classifiers()` - Comprehensive
- `test_compare_classifiers()` - Best selection

**For optimization.py**:
- `test_optimize_k_neighbors()` - k range
- `test_optimize_prototype_components()` - Strategies
- `test_find_best_classifier()` - Full pipeline

### Integration Tests Needed
- Backward compatibility with old code
- Metric consistency before/after refactoring
- Performance benchmarks

---

## Version Control Notes

**New Files** (Not in git yet):
- `otitenet/ml/__init__.py`
- `otitenet/ml/classifiers.py`
- `otitenet/ml/evaluation.py`
- `otitenet/ml/optimization.py`
- `otitenet/ml/README.md`

**Modified Files**:
- `app.py` - Lines 51-56, 719-733, 1551-1660
- `otitenet/train/train_triplet_new.py` - Lines 33-40, 1762-1880, 1943-1957, 2098-2112, 2159-2173

---

## Deployment Checklist

Before deploying to production:

- [ ] Run all tests
- [ ] Verify backward compatibility
- [ ] Check performance (no regressions)
- [ ] Code review
- [ ] Update CI/CD pipelines
- [ ] Document in project README
- [ ] Create migration guide for team
- [ ] Update API documentation
- [ ] Tag version in git

---

## Maintenance Notes

### Code Style
- PEP 8 compliant
- Descriptive variable names
- Clear function docstrings
- Type hints in docstrings

### Performance
- No unnecessary copies
- Vectorized operations
- Efficient sklearn usage
- Minimal overhead

### Extensibility
- Easy to add new classifiers
- Consistent interfaces
- Pluggable strategies
- Configurable parameters

---

## Future Enhancements

1. **Add XGBoost classifier** - Fit and evaluate
2. **Add RandomForest classifier** - Fit and evaluate
3. **Add SVM with different kernels** - Fit and evaluate
4. **Parallel grid search** - Speed up evaluation
5. **Cached evaluation results** - Avoid recomputation
6. **Cross-validation support** - k-fold CV
7. **Hyperparameter ranges** - More flexible config
8. **Custom loss functions** - Beyond MCC

---

## Support & Questions

For questions about the ML module:
1. Check `otitenet/ml/README.md` for usage examples
2. Review `REFACTORING_DETAILED_CHANGES.md` for before/after
3. Check docstrings in source files
4. Review test files for usage patterns

