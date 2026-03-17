# Quick Start: Multi-Classifier Validation

## How to Use the Multi-Classifier Evaluation

### In Your Validation Loop

```python
# During predict() method when group == 'valid'
if group == 'valid' and train_encs is not None and train_cats is not None:
    # Evaluate all classifiers and pick the best
    classifier_info = self.evaluate_multi_classifiers(
        train_encs, train_cats, valid_encs, valid_cats
    )
    
    best_classifier_method = classifier_info['method']
    best_classifier = classifier_info['classifier']
    best_mcc = classifier_info['mcc']
    eval_time = classifier_info['time']
    
    # Log to Tracking
    if self.log_tracking and run is not None:
        run[f'validation/classifier_method'] = best_classifier_method
        run[f'validation/classifier_mcc'] = best_mcc
        run[f'validation/classifier_eval_time'] = eval_time
        
        # Log all methods for comparison
        for method, info in classifier_info['all_results'].items():
            run[f'validation/methods/{method}/mcc'] = info['mcc']
            run[f'validation/methods/{method}/time'] = info['time']
```

### Expected Output

```
============================================================
Multi-Classifier Validation Results:
  knn                MCC: 0.8342  Time: 0.512s
  kde                MCC: 0.7891  Time: 0.389s
  prototypes         MCC: 0.8156  Time: 0.085s
  linear             MCC: 0.7234  Time: 0.021s
============================================================
Best Method: KNN (MCC: 0.8342)
============================================================
```

## What Each Classifier Does

### 1. **KNN** (k-Nearest Neighbors)
- **Best For:** Triplet loss, softmax_contrastive
- **Speed:** ~50ms per k value
- **What We Try:** k = 1 to 10, picks best
- **Output:** Single k value that maximizes validation MCC

### 2. **KDE** (Kernel Density Estimation)
- **Best For:** Prototype-based classification
- **Speed:** ~100ms per configuration
- **What We Try:** 
  - Kernels: gaussian, exponential
  - Bandwidth: scott, silverman
- **Output:** Best (kernel, bandwidth) pair

### 3. **Prototypes**
- **Best For:** When using prototypes explicitly
- **Speed:** ~50ms per strategy
- **What We Try:**
  - Kinds: distance, distance_weighted
- **Output:** Best prototype classification method
- **Requires:** `prototypes_to_use != 'no'`

### 4. **Linear Classifier** (Logistic Regression)
- **Best For:** Fallback, always available
- **Speed:** ~20ms
- **What We Try:** Standard LogisticRegression(max_iter=1000)
- **Output:** Simple linear classifier

## Speed Comparison

| Approach | Time per Validation | Trials Needed | Total Time |
|----------|-------------------|---------------|-----------|
| Ax Optimization | 15-30s per trial | 50+ trials | **12-25 hours** |
| Multi-Classifier | 1-2s per epoch | 0 (automatic) | **30-60 minutes** |

**You save:** 20x faster classifier selection ⚡

## When to Use Each Approach

### Use Ax Optimization For:
- Learning rate tuning
- Weight decay
- Loss margins (triplet margin)
- Domain loss weight (gamma)
- FGSM epsilon

### Use Multi-Classifier For:
- KNN k selection
- Prototype classification type
- KDE configuration
- Classifier type selection

## Enabling Multi-Classifier in Your Training

Simply call during validation:

```python
def predict(self, group, loader, lists, traces):
    """..."""
    
    train_encs = _safe_concat(lists['train']['encoded_values'])
    train_cats = _safe_concat(lists['train']['cats'])
    
    # NEW: Multi-classifier evaluation
    if group == 'valid' and train_encs is not None:
        classifier_info = self.evaluate_multi_classifiers(
            train_encs, train_cats, valid_encs, valid_cats
        )
        # Use best_classifier for rest of predictions
        best_classifier = classifier_info['classifier']
    
    # ... rest of validation code
```

## What Gets Logged to Tracking

Under `run['validation/']`:
- `classifier_method` → which method won (knn, kde, prototypes, linear)
- `classifier_mcc` → the MCC score of winner
- `classifier_eval_time` → time spent evaluating all classifiers
- `methods/{knn,kde,prototypes,linear}/mcc` → scores for each method
- `methods/{knn,kde,prototypes,linear}/time` → time for each method

This gives you complete visibility into what classifiers are working best as training progresses!

## Troubleshooting

**Issue:** "All classifiers failed to evaluate"
- Solution: Check that `train_encs` and `train_cats` are not empty
- Check that `valid_encs` and `valid_cats` are valid

**Issue:** KDE consistently fails
- Solution: KDE requires sufficient samples (~50+), may fail with small datasets
- Use `--prototypes_to_use='no'` to skip KDE evaluation

**Issue:** Prototypes classifier fails
- Solution: Need `--prototypes_to_use='batch'` or `'class'` for prototypes evaluation
- Linear fallback will still work

