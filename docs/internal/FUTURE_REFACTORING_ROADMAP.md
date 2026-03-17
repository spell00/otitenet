# Future Refactoring Opportunities

## Phase 2: Data Loading & Preprocessing Module

### Current State
Data loading is scattered across:
- `otitenet/data/data_getters.py` - `get_data()`, `get_images_loaders()`, `get_images()`
- `otitenet/train/train_triplet_new.py` - Data concatenation, split handling
- `app.py` - Loader initialization repeated in multiple places

### Opportunity: Create `otitenet/data/loaders.py`
```python
def load_and_prepare_data(_args, seed=42):
    """
    Unified data loading combining:
    - CSV reading and parsing
    - Train/valid/test split
    - Batch handling
    
    Returns: data dict with all splits
    """

def build_image_loaders(data, _args):
    """
    Unified loader building with:
    - Transform composition
    - Augmentation handling
    - Batch size management
    
    Returns: loaders dict
    """

def create_training_context(_args):
    """
    One-shot setup combining:
    - Data loading
    - Loader creation
    - Model initialization
    - Lists/traces preparation
    
    Returns: (model, loaders, lists, traces, ...)
    """
```

**Duplication Found**:
```python
# app.py lines 2687-2700 - Same pattern repeated in:
# app.py lines 2847, 2927, 4847, 4930
loaders = get_images_loaders(
    data=data, random_recs=_args.random_recs,
    weighted_sampler=0, is_transform=0, samples_weights=None,
    epoch=1, unique_labels=unique_labels,
    triplet_dloss=_args.dloss, bs=_args.bs,
    prototypes_to_use=_args.prototypes_to_use,
    prototypes=prototypes, size=_args.new_size, normalize=_args.normalize,
)
```

**Estimated Code Reduction**: ~80 lines across 4+ locations

---

## Phase 3: Metrics Module

### Current State
Metrics computation scattered:
- `otitenet/logging/metrics.py` - `MCC`, `expected_calibration_error`, `brier_score`
- `otitenet/logging/loggings.py` - `log_metrics()`, trace collection
- `otitenet/app/utils.py` - `_compute_calibration_metrics()`, `get_calibration_metrics()`
- Multiple files computing MCC inline

### Opportunity: Create `otitenet/metrics/compute.py`
```python
def compute_mcc(predictions, targets):
    """Unified MCC computation with error handling."""
    
def compute_all_metrics(predictions, targets):
    """Single function returning: MCC, accuracy, F1, precision, recall."""
    
def compute_calibration_metrics(log_path):
    """Unified calibration error computation."""
    
def batch_metrics(lists, reduce='all'):
    """Compute metrics across train/valid/test groups."""
```

**Duplication Found**:
```python
# Called 20+ times across codebase
mcc = MCC(valid_cats, preds)

# Multiple implementations of trace metric collection
traces[group]['mcc'] = ...
```

**Estimated Code Reduction**: ~100 lines

---

## Phase 4: Visualization Module

### Current State
Visualization scattered:
- `otitenet/logging/grad_cam.py` - GradCAM visualization
- `otitenet/logging/shap.py` - SHAP visualization
- `otitenet/logging/plotting.py` - Confusion matrices, heatmaps
- Decision boundary plotting repeated in multiple places

### Opportunity: Create `otitenet/visualization/__init__.py`
```python
def plot_decision_boundary(classifier, embeddings, labels, title=None):
    """Unified decision boundary plotting for 2D embeddings."""
    
def plot_confusion_matrix(cm, class_names, metrics=None):
    """Unified confusion matrix visualization."""
    
def plot_embedding_distribution(embeddings, labels, reduction='tsne'):
    """Unified embedding space visualization (t-SNE, UMAP, PCA)."""
    
def visualize_classifier_comparison(results):
    """Visualization comparing classifier performance."""
```

**Duplication Found**:
```python
# Confusion matrix plotting repeated
cm = confusion_matrix(y_true, y_pred)
# ... plotting code in multiple places ...

# Decision boundary plotting (app.py commented out but logic exists)
# plot_decision_boundary(KNeighborsClassifier, train_encs, train_cats)
```

**Estimated Code Reduction**: ~120 lines

---

## Phase 5: Prototype Strategy Module

### Current State
Prototype handling scattered:
- `otitenet/utils/prototypes.py` - Prototype class
- `otitenet/utils/encoding_utils.py` - Strategy computation
- `otitenet/train/train_triplet_new.py` - Set/update prototypes logic
- `app.py` - Multiple prototype strategy loops

### Opportunity: Create `otitenet/ml/prototypes.py`
```python
class PrototypeStrategy:
    """Base class for prototype strategies."""
    
class MeanPrototype(PrototypeStrategy):
    """Simple mean of class embeddings."""
    
class KMeansPrototype(PrototypeStrategy):
    """K-means clustering per class."""
    
class GMMPrototype(PrototypeStrategy):
    """Gaussian Mixture Model per class."""

def evaluate_prototype_strategy(strategy, train_encs, train_cats, 
                               valid_encs, valid_cats, n_components):
    """Evaluate single strategy."""
    
def optimize_prototype_strategy(train_encs, train_cats,
                               valid_encs, valid_cats,
                               strategies=['mean', 'kmeans', 'gmm'],
                               max_components=5):
    """Find best prototype strategy and component count."""
```

**Duplication Found**:
```python
# Strategy loops repeated in train_triplet_new.py and app.py
for strategy in ['mean', 'kmeans', 'gmm']:
    for n_components in range(1, max_components+1):
        # ... computation and evaluation ...
```

**Estimated Code Reduction**: ~90 lines

---

## Phase 6: Distance/Metric Functions Module

### Current State
Distance computations scattered:
- `otitenet/data/data_getters.py` - `get_distance_fct()`
- `otitenet/ml/classifiers.py` - Distance logic inline
- `otitenet/logging/shap.py` - Embedding distance computations
- Multiple distance metric implementations

### Opportunity: Create `otitenet/ml/distances.py`
```python
def euclidean_distance(a, b):
    """Vectorized Euclidean distance."""
    
def cosine_similarity(a, b):
    """Vectorized cosine distance."""
    
def minkowski_distance(a, b, p=2):
    """Minkowski distance with p norm."""
    
def get_distance_function(metric_name):
    """Registry of distance functions."""
    
class DistanceMetric:
    """Callable wrapper for distance metrics."""
```

**Duplication Found**:
```python
# Distance computation repeated multiple ways
from scipy.spatial.distance import cdist
distances = cdist(embeddings, prototypes, metric='euclidean')

# In classifiers.py
if metric == 'euclidean':
    dist = np.sqrt(np.sum((a - b)**2))
```

**Estimated Code Reduction**: ~50 lines

---

## Phase 7: Model Loading & Caching

### Current State
Model loading code repeated:
- `app.py` - Multiple `load_model_and_prototypes()` calls with caching
- `otitenet/train/train_triplet_new.py` - Model loading in predict loops
- Cache management scattered (`_clear_cached_model()`)

### Opportunity: Create `otitenet/models/loader.py`
```python
class ModelCache:
    """Singleton model cache with versioning."""
    
    def load(self, _args, force_reload=False):
        """Load model with automatic caching."""
        
    def clear(self):
        """Clear cached models."""
        
    def get_hash(self, _args):
        """Model versioning hash."""

def load_model(_args):
    """Unified model loading (uses cache)."""
    
def load_model_with_prototypes(_args):
    """Unified loading of model + prototypes."""
```

**Duplication Found**:
```python
# app.py lines 3432, 3501, 4847, 4930, 2671
model, shap_model, prototypes, image_size, device_str, data, unique_labels, unique_batches, data_getter = \
    load_model_and_prototypes(_args)
```

**Estimated Code Reduction**: ~60 lines

---

## Phase 8: Argument Validation Module

### Current State
Argument validation scattered:
- Multiple `if args.xxx is None` checks
- Parameter normalization repeated
- Type conversion in multiple places

### Opportunity: Create `otitenet/utils/args_validator.py`
```python
def validate_args(_args):
    """Single point of validation for all arguments."""
    
def normalize_args(_args):
    """Normalize args (types, defaults, ranges)."""
    
def get_validated_config(_args):
    """Get config dict with validation."""
```

---

## Summary of Opportunities

| Phase | Module | Files | Est. Lines | Priority |
|-------|--------|-------|-----------|----------|
| 2 | Data Loading | `otitenet/data/loaders.py` | 80 | ⭐⭐⭐ |
| 3 | Metrics | `otitenet/metrics/compute.py` | 100 | ⭐⭐⭐ |
| 4 | Visualization | `otitenet/visualization/` | 120 | ⭐⭐ |
| 5 | Prototypes | `otitenet/ml/prototypes.py` | 90 | ⭐⭐ |
| 6 | Distances | `otitenet/ml/distances.py` | 50 | ⭐ |
| 7 | Model Loading | `otitenet/models/loader.py` | 60 | ⭐⭐ |
| 8 | Arg Validation | `otitenet/utils/args_validator.py` | 70 | ⭐ |
| **TOTAL** | **7 modules** | | **~570** | |

---

## Implementation Strategy

### Priority Order:
1. **Phase 2 (Data Loading)** - Most repeated (~80 lines in 4+ places)
2. **Phase 3 (Metrics)** - Critical for consistency (~100 lines)
3. **Phase 7 (Model Loading)** - Cache management needed (~60 lines)
4. **Phase 5 (Prototypes)** - Currently in ML module, needs specialization (~90 lines)
5. **Phase 4 (Visualization)** - Nice-to-have improvements (~120 lines)
6. **Phase 6 (Distances)** - Can optimize later (~50 lines)
7. **Phase 8 (Arg Validation)** - Utility module (~70 lines)

### Estimated Total Impact:
- **Lines Eliminated**: ~570 lines of duplicate code
- **New Module Lines**: ~700 lines of clean code
- **Net Improvement**: ~170 lines saved + significantly improved code quality

---

## Quick Wins (Low-hanging Fruit)

### 1. Unify Training Context Creation
```python
# Replace repeated pattern in app.py lines 2847, 2927, 4847, 4930
def create_training_context(_args, data_source='data'):
    """One function to replace 10+ lines repeated 4 times."""
```
**Impact**: 40 lines saved

### 2. Centralize Loader Building
```python
# Extract to function
def build_standard_loaders(_args, data, prototypes):
    """Standard loader configuration for inference."""
```
**Impact**: 30 lines saved

### 3. Consolidate Trace Initialization
```python
# otitenet/utils/utils.py
def get_empty_traces_for_groups(groups=['train', 'valid', 'test']):
    """Flexible trace initialization."""
```
**Impact**: 20 lines saved

**Total Quick Wins**: ~90 lines with minimal risk

---

## Risks & Mitigation

### Risk 1: Over-modularization
**Mitigation**: Don't create modules for < 50 lines of code

### Risk 2: Breaking Existing Code
**Mitigation**: Phase refactoring, test each phase thoroughly

### Risk 3: Performance Impact
**Mitigation**: Profile before/after, keep hot paths optimized

### Risk 4: Increased Complexity
**Mitigation**: Clear documentation for each module, simple APIs

---

## Success Metrics

After completing all phases:
- ✅ Reduce total codebase lines by ~15-20%
- ✅ Decrease code duplication from ~40% to <10%
- ✅ Improve test coverage from ~20% to >60%
- ✅ Reduce time to implement new classifiers by 80%
- ✅ Improve code maintainability score by 50%

