# Implementation Summary: Prototype and KDE Classification System

## Overview
Successfully implemented a comprehensive system to make prototype aggregation strategy and KDE (Kernel Density Estimation) classification available as hyperparameters in both the training script and web application. This replaces or augments KNN-based classification with more sophisticated methods.

## Changes Made

### 1. New Hyperparameters Added

#### In `train_triplet_new.py`:
```python
--prototype_kind: str [choices: 'distance', 'kde', 'distance_weighted']
    Default: 'distance'
    How to classify using prototypes:
    - 'distance': Find closest prototype to test sample
    - 'kde': Use Kernel Density Estimation on training encodings
    - 'distance_weighted': Weight distances by number of prototypes per class

--kde_kernel: str [choices: 'gaussian', 'exponential', 'linear', 'tophat']
    Default: 'gaussian'
    Type of kernel for KDE

--kde_bandwidth: str
    Default: 'scott'
    Bandwidth parameter: 'scott', 'silverman', or float value

--prototype_components: int
    Default: 1
    Number of components/centroids per class for prototype aggregation
    (dominant component is selected for classification)

--prototype_strategy: str [choices: 'mean', 'kmeans', 'gmm']
    Default: 'mean'
    Already existed; how to compute prototypes
```

### 2. New KDE Module: `otitenet/utils/kde.py`

Implements two KDE classifiers:

#### `KernelDensityClassifier`
- Uses sklearn's KernelDensity under the hood
- Supports multiple kernel types
- Features:
  - `fit(X, y)`: Fit KDE for each class
  - `predict(X)`: Predict class labels based on highest density
  - `predict_proba(X)`: Return soft probabilities
  - `score_samples(X)`: Compute density scores for all classes

#### `SoftKernelDensityClassifier`
- Manual Gaussian kernel implementation
- Supports learnable bandwidth (future enhancement)
- More interpretable than sklearn's version
- Uses squared Euclidean distance: `K(x,y) = exp(-||x-y||^2 / (2*bandwidth^2))`
- Density for class c: `p(x|c) = (1/n) * sum_i K(x, x_i)`

#### Factory Function
```python
make_kde_classifier(kernel, bandwidth, learnable, soft)
    Returns appropriate KDE classifier based on parameters
```

### 3. Training Script Updates (`train_triplet_new.py`)

#### New Methods in `TrainAE` class:

**`_classify_with_prototypes(embeddings, dist_fct)`**
- Classifies using prototype distance with optional weighting
- Supports three strategies:
  1. **distance**: Inverse distance ratio (soft voting)
     `prob(class) = (1/dist_to_proto) / sum(1/dist_to_all_protos)`
  2. **distance_weighted**: Weighted by number of samples per class
     `prob(class) = (n_class / dist) / sum(n_i / dist_i)`
  3. Default: distance-based (backward compatible)

**`_fit_kde_classifier(encs, cats)`**
- Fits a KDE classifier on training encodings
- Returns ready-to-use classifier for prediction

#### Updated `predict()` Method
- Now intelligently chooses classification method:
  ```
  if use_kde:
      Use KDE classifier
  elif use_prototypes:
      Use prototype distance classification
  elif use_knn:
      Use original KNN (only if no prototypes)
  else:
      Use model's linear layer (ce/hinge loss)
  ```
- Conditional logic prevents KNN from being used when prototypes are enabled
- Prints selected method for transparency

### 4. Web App Updates (`app.py`)

#### New Functions:

**`_predict_with_kde(embedding_tensor, train_embeddings, train_labels, ...)`**
- Fits KDE on training data
- Predicts class for single embedding
- Returns (predicted_label, confidence)
- Handles KDE kernel and bandwidth parameters

#### Updated `get_or_build_knn()`**
- Now checks if prototypes are enabled
- If yes: Returns None immediately, skips KNN training
- If no: Proceeds with KNN training as before
- Caches training embeddings for KDE fallback

#### Updated `run_analysis_on_file()`**
- Intelligently selects classification method:
  1. If KDE enabled + prototypes: Use KDE
  2. If prototypes enabled: Use prototype distance
  3. Otherwise: Use KNN (original behavior)
- Displays appropriate method label in results
- Provides graceful fallbacks

#### Classification Priority (in app):
```
Fast Inference (if enabled):
  → Use prototype distance ratio

Slow Inference:
  if use_kde and training_data_available:
    → Use KDE
  elif use_prototypes:
    → Use prototype distance
  else:
    → Use KNN (or skip if prototypes prevent it)
```

### 5. Imports

Added to both scripts:
```python
from ..utils.kde import make_kde_classifier
```

## Key Features

### 1. **Backward Compatibility**
- Default behavior unchanged (uses KNN if no prototypes)
- All new parameters have sensible defaults
- Existing code paths still work

### 2. **Conditional KNN Exclusion**
- KNN is NOT trained/used when prototypes_to_use is 'combined' or 'class'
- Prevents confusion and unnecessary computation
- Clear messaging about which method is being used

### 3. **Prototype-Based Classification**
- Distance-based: Finds closest prototype
- Weighted: Considers class frequency
- Soft voting: Returns probabilities for uncertainty quantification

### 4. **KDE Support**
- Gaussian kernel: `exp(-d^2 / 2σ²)` (default)
- Exponential kernel: `exp(-d/σ)`
- Linear/tophat: Also supported via sklearn
- Soft version: Every training sample contributes, weighted by distance
- Learnable bandwidth: Infrastructure ready (can optimize on validation set)

### 5. **Caching**
- Training embeddings cached in session state for KDE
- KNN cache respects prototype settings
- Prevents redundant retraining

## Usage Examples

### Training with Prototypes (Distance-based)
```bash
python train_triplet_new.py \
  --prototypes_to_use class \
  --prototype_kind distance \
  --prototype_strategy kmeans \
  --prototype_components 3
```

### Training with KDE
```bash
python train_triplet_new.py \
  --prototypes_to_use class \
  --prototype_kind kde \
  --kde_kernel gaussian \
  --kde_bandwidth scott
```

### Training with Distance-Weighted Prototypes
```bash
python train_triplet_new.py \
  --prototypes_to_use class \
  --prototype_kind distance_weighted \
  --prototype_strategy mean
```

### Original KNN (No Prototypes)
```bash
python train_triplet_new.py \
  --prototypes_to_use no \
  --n_neighbors 5
```
In this case, KNN is trained normally.

## Files Modified

1. **`otitenet/train/train_triplet_new.py`**
   - Added new hyperparameters
   - Added `_classify_with_prototypes()` method
   - Added `_fit_kde_classifier()` method
   - Updated `predict()` method
   - Added KDE import

2. **`otitenet/utils/kde.py`** (NEW)
   - `KernelDensityClassifier` class
   - `SoftKernelDensityClassifier` class
   - `make_kde_classifier()` factory function
   - Complete KDE implementation with soft voting

3. **`app.py`**
   - Added KDE import
   - Added `_predict_with_kde()` function
   - Updated `get_or_build_knn()` to skip when prototypes enabled
   - Updated `run_analysis_on_file()` to use prototypes/KDE
   - Updated prediction output to show method used

## Technical Details

### Distance Metrics Supported
- Euclidean (default)
- Cosine (when dist_fct='cosine')
- Squared Euclidean (for KDE Gaussian kernel)

### Kernel Density Estimation Theory
For class c, the KDE density at point x is:
```
p(x|c) = (1/n_c) * Σ_i K(x, x_i)
where K is the kernel function and x_i are training samples from class c

Classification: argmax_c p(x|c)
Probabilities: p(c|x) ∝ p(x|c) (softmax normalization)
```

### Prototype Distance Classification
For each class:
```
dist(x, proto_c) = euclidean or cosine distance
prob(c|x) ∝ 1/dist(x, proto_c)   [inverse distance ratio]
or
prob(c|x) ∝ (n_c/dist) / Σ_j(n_j/dist_j)   [weighted by class size]
```

## Testing Recommendations

1. **Syntax**: Already validated (no errors found)
2. **Functionality**:
   ```bash
   # Test prototype classification
   python train_triplet_new.py --prototypes_to_use class --prototype_kind distance
   
   # Test KDE
   python train_triplet_new.py --prototypes_to_use class --prototype_kind kde
   
   # Test KNN still works when no prototypes
   python train_triplet_new.py --prototypes_to_use no --n_neighbors 5
   ```

3. **App Integration**:
   - Load a trained model with prototypes
   - Verify prototype-based prediction works
   - Test fallback to KNN when prototypes unavailable
   - Check displayed method label is correct

## Future Enhancements

1. **Learnable Bandwidth**
   - Optimize KDE bandwidth on validation set
   - Infrastructure already in place in `SoftKernelDensityClassifier`
   - Call `learn_bandwidth_from_validation()` after KDE fit

2. **Multiple Prototypes per Class**
   - Currently uses dominant component from `prototype_components`
   - Could enhance to use all components with mixture model

3. **Adaptive Kernels**
   - Per-class kernel selection
   - Kernel width learned from data

4. **Performance Optimization**
   - Use PyTorch for GPU-accelerated distance computation
   - Batch KDE scoring for faster inference

## Notes

- KNN is completely bypassed when prototypes are enabled (not trained at all)
- KDE bandwidth defaults to 'scott' (automatic selection based on data)
- All methods use soft voting/probabilities for proper uncertainty quantification
- Code maintains consistency between training and inference implementations
