# Machine Learning Module

This module contains all classical machine learning (non-deep-learning) classification code.

## Module Structure

### `classifiers.py` - Classifier Fitting
Core functions for fitting various classifier types:
- `fit_knn_classifier()` - Fit KNN on embeddings
- `fit_baseline_classifiers()` - Fit LogReg, NaiveBayes, LinearSVC
- `fit_kde_classifier()` - Fit Kernel Density Estimation classifier
- `predict_with_prototypes()` - Predict using nearest prototype matching

### `evaluation.py` - Classifier Evaluation
Evaluation utilities with grid search:
- `evaluate_knn_with_k_search()` - Try multiple k values
- `evaluate_baseline_classifiers()` - Evaluate all baselines
- `evaluate_kde_classifiers()` - Try different KDE configurations
- `evaluate_all_classifiers()` - Comprehensive evaluation
- `compare_classifiers()` - Find best among all methods

### `optimization.py` - Hyperparameter Optimization
Unified API for classifier optimization:
- `optimize_k_neighbors()` - Find best k for KNN
- `optimize_prototype_components()` - Optimize prototype strategies
- `find_best_classifier()` - Main entry point for full search

## Usage

### Quick Start - Find Best Classifier

```python
from otitenet.ml import find_best_classifier

# Find best classifier among all options
best_method, best_mcc, results = find_best_classifier(
    train_encs, train_cats,
    valid_encs, valid_cats,
    min_k=1,
    max_k=20,
    include_baselines=True,
    include_prototypes=True,
    max_components=5
)

print(f"Best method: {best_method} with MCC: {best_mcc:.3f}")
```

### Fit Specific Classifiers

```python
from otitenet.ml import fit_knn_classifier, fit_baseline_classifiers

# Fit KNN
knn = fit_knn_classifier(train_encs, train_cats, n_neighbors=5)
preds = knn.predict(test_encs)

# Fit all baseline classifiers
classifiers = fit_baseline_classifiers(train_encs, train_cats)
logreg = classifiers['logreg']
```

### Evaluate with Grid Search

```python
from otitenet.ml import evaluate_all_classifiers, compare_classifiers

# Evaluate all classifier types
results = evaluate_all_classifiers(
    train_encs, train_cats,
    valid_encs, valid_cats,
    min_k=1,
    max_k=20,
    include_kde=True,
    include_baselines=True
)

# Find best
best_method, best_mcc, best_params = compare_classifiers(results)
```

### Prototype-Based Classification

```python
from otitenet.ml import optimize_prototype_components
from otitenet.utils.encoding_utils import compute_prototypes_by_strategy

# Optimize prototype configurations
proto_results = optimize_prototype_components(
    train_encs, train_cats,
    valid_encs, valid_cats,
    strategies=['mean', 'kmeans', 'gmm'],
    max_components=5
)

# Get best strategy
for strategy, data in proto_results.items():
    print(f"{strategy}: MCC={data['best_mcc']:.3f}, "
          f"n_components={data['best_n_components']}")
```

## Integration with Other Modules

This module integrates with:
- `otitenet.utils.encoding_utils` - Prototype computation
- `otitenet.utils.kde` - KDE classifier implementation
- `otitenet.logging.metrics` - MCC and other metrics

## Design Principles

1. **Separation from Deep Learning**: All classical ML separate from torch models
2. **Numpy-based**: Works with numpy arrays (embeddings already extracted)
3. **Consistent API**: All functions follow similar input/output patterns
4. **Composable**: Small functions that can be combined
5. **No Side Effects**: Pure functions that don't modify state

## Migration Benefits

Before this module:
- KNN fitting duplicated in `app.py` and `train_triplet_new.py`
- Baseline classifiers duplicated in both files
- Prototype evaluation code repeated
- ~500 lines of duplicated ML code

After:
- Single source of truth for all ML operations
- ~200 lines of well-organized, reusable code
- Easy to add new classifiers or evaluation strategies
- Consistent behavior across training and inference

## Adding New Classifiers

To add a new classifier:

1. Add fitting function to `classifiers.py`:
```python
def fit_my_classifier(train_encs, train_cats, **params):
    clf = MyClassifier(**params)
    clf.fit(train_encs, train_cats)
    return clf
```

2. Add evaluation to `evaluation.py`:
```python
def evaluate_my_classifier(train_encs, train_cats, valid_encs, valid_cats):
    clf = fit_my_classifier(train_encs, train_cats)
    preds = clf.predict(valid_encs)
    mcc = MCC(valid_cats, preds)
    return {'mcc': mcc, 'classifier': clf}
```

3. Integrate into `optimization.py` if grid search needed

4. Update `__init__.py` exports
