# Changes Summary

## 1. Gallery Per-Image Model Display (app.py)

### What was added:
- Added helper function `_get_model_meta()` to lookup model rank/MCC from `best_models_table`
- For each image in the gallery, now displays which models have Grad-CAMs available
- Models are shown with rank (#), name, and MCC value in decreasing MCC order
- Display format: `#<rank> <ModelName> (MCC <value>)` separated by pipes

### Example:
```
Models available: #1 ResNet18 (MCC 0.845) | #2 ViT (MCC 0.823) | #4 ResNet50 (MCC 0.801)
```

### Files modified:
- `/home/simon/otitenet/app.py` (lines 3953-3976)

---

## 2. Prototype Aggregation Strategies (New Feature)

### What was added:
Added support for multiple prototype computation methods beyond simple mean:
- **mean** (default): Existing behavior - average of all encodings
- **kmeans**: K-means clustering, returns dominant cluster center
- **gmm**: Gaussian Mixture Model - returns dominant component mean
- **em** / **expectation_maximization**: Alias for GMM

### Implementation Details:

#### Prototypes Class Changes (`otitenet/utils/prototypes.py`):
- New constructor parameters:
  - `strategy` (str): Aggregation method (default: 'mean')
  - `components` (int): Number of components/centroids per class (default: 1)
  - `random_state` (int): Seed for clustering algorithms
- New helper method `_compute_prototype()` that applies selected strategy
- All prototype setters (both/class/batch) now use the unified method

#### CLI Arguments (`otitenet/train/train_triplet_new.py`):
```bash
--prototype_strategy {mean,kmeans,gmm,em,expectation_maximization}  # Default: 'mean'
--prototype_components <int>                                         # Default: 1 (components per class)
```

#### Model Path Integration:
- Best model directory now includes prototype aggregation in path:
  - Format: `.../<protoagg_STRATEGY_COMPONENTS>/...`
  - Example: `...protoagg_kmeans_3/...` for K-means with 3 components

### Example Usage:
```bash
# Use 3-component GMM for prototype aggregation
python train_triplet_new.py \
  --prototype_strategy gmm \
  --prototype_components 3 \
  --seed 42
```

### Performance Considerations:
- K-means and GMM are computed once per group during training (not per epoch)
- The dominant component is selected (highest weight/most samples)
- Single sample per class: falls back to that sample (no clustering)
- Backward compatible: default 'mean' produces identical results to old code

### Files modified:
- `/home/simon/otitenet/otitenet/utils/prototypes.py` (complete refactor)
- `/home/simon/otitenet/otitenet/train/train_triplet_new.py` (initialization + CLI)

---

## 3. Model Selection Sorting (app.py)

### What was added:
- Gallery model selection now sorted by MCC value (descending)
- Highest performing models appear first in the dropdown

### Files modified:
- `/home/simon/otitenet/app.py` (lines 3842-3851)

---

## How to Use These Features

### 1. Test Gallery Per-Image Models Display:
1. Go to Tab 5 "Grad-CAM Gallery"
2. Select multiple models
3. Each image card now shows "Models available: ..." with ranks and MCC scores
4. Models are sorted by MCC (best first)

### 2. Experiment with Prototype Strategies:
```bash
# Default (mean)
python launch.sh

# K-means with 3 clusters
python launch.sh --prototype_strategy kmeans --prototype_components 3

# GMM with 2 components  
python launch.sh --prototype_strategy gmm --prototype_components 2
```

### 3. Hyperparameter Optimization:
Since `prototype_strategy` and `prototype_components` are now exposed as CLI args, they can be included in hyperparameter search grids:

```python
# In hyperparameter tuning config
{
    'prototype_strategy': ['mean', 'kmeans', 'gmm'],
    'prototype_components': [1, 2, 3],
    # ... other params
}
```

---

## Next Steps (Optional Enhancements)

1. **Add to app.py UI**: Expose prototype strategy/components as toggles in sidebar for manual selection
2. **Visualization**: Show which prototypes are being used for each model in Tab 2
3. **Comparison**: Add comparison view showing same image with different prototype strategies
4. **Logging**: Log which strategy/components were used to MLflow/Tracking for better tracking
5. **SOTA Methods**: Consider adding:
   - Medoid selection (center point from data)
   - Variational Autoencoder prototype learning
   - Contrastive learning-based prototypes
