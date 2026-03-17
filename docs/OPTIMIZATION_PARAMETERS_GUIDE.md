# Optimization Parameters Guide

## Overview
The training script now implements **smart conditional parameter optimization**. Parameters are only included in the optimization space when they are:
1. Originally set to `None` (indicating they should be optimized)
2. Actually relevant to the current configuration

## Key Changes

### 1. **n_calibration is NEVER Optimized** ❌
- **Fixed at 0** - cannot be overridden even through optimization
- This parameter is excluded from the optimization parameter space entirely
- Rationale: User explicitly requested this parameter never be optimized

### 2. **n_positives and n_negatives** (Triplet Loss Parameters)
- **Only optimized when applicable** ✓
- **Included in optimization only if:**
  - `classif_loss` is `'triplet'` OR `'softmax_contrastive'`
- **NOT included if:**
  - `classif_loss` is `'ce'`, `'hinge'`, or `'arcface'`
  - These losses don't use the tuple loss mechanism
- **Range:** `n_positives` in [1], `n_negatives` in [1, 5]

### 3. **Prototype Parameters** (New!)
- **Only included when prototypes are used:**
  - `prototypes_to_use` must be `'batch'` or `'class'` (not `'no'`)
- **Parameters optimized when relevant:**
  - `prototype_strategy`: ['mean', 'kmeans'] - how to aggregate prototypes
  - `prototype_components`: [1-5] - number of components per class
  - `prototype_kind`: ['distance', 'kde', 'distance_weighted'] - classification method
  - `kde_kernel`: ['gaussian', 'exponential'] - KDE kernel type (if kde is used)
  - `kde_bandwidth`: ['scott', 'silverman'] - KDE bandwidth (if kde is used)

### 4. **Distance-Based Parameters**
- **Included when:**
  - `classif_loss` in ['triplet', 'softmax_contrastive'] OR
  - `dloss` in ['inverseTriplet', 'inverse_softmax_contrastive', etc.]
- **Parameters:**
  - `dist_fct`: ['cosine', 'euclidean']
  - `dmargin`: [0.0 - 1.0]

### 5. **Domain Loss Parameters**
- **Included when:**
  - `dloss` in domain-aware losses
- **Parameters:**
  - `gamma`: [0.01 - 100] (log scale) - domain adaptation weight

### 6. **Margin (Triplet Margin)**
- **Included when:**
  - `classif_loss` in ['triplet', 'softmax_contrastive']
- **Range:** [0.0 - 10.0]

### 7. **FGSM/Epsilon (Adversarial Training)**
- **Included when:**
  - `fgsm` is enabled (1)
- **Range:** [1e-4 - 0.5] (log scale)

### 8. **n_neighbors (KNN Classifier)**
- **Included when:**
  - `auto_select_k` is disabled (0)
  - AND `classif_loss` not in ['ce', 'hinge']
  - OR `dloss` in adversarial losses
- **Range:** [1 - 10] (log scale)

## Optimization Example: launch_optimize.sh

```bash
# Each model runs optimization with:
# - All parameters set to None to enable optimization
# - ~150 trials per model
# - Conditional logic determines which parameters are actually optimized

for model in resnet18 vgg16 efficientnet_b0 vit; do
    # All these are set to None, but only relevant ones are optimized
    python -m otitenet.train.train_triplet_new \
        --model_name="$model" \
        --dloss="None" \
        --classif_loss="None" \
        --prototypes_to_use="None" \
        --n_positives="None" \
        --n_negatives="None" \
        --fgsm="None" \
        --normalize="None" \
        --n_calibration=0  # Fixed, never optimized
        --n_trials=150
done
```

## Compatibility Matrix

| Parameter | CE/Hinge | Triplet | Softmax_Contrastive | ArcFace | Domain Loss |
|-----------|----------|---------|---------------------|---------|-------------|
| n_positives | ❌ | ✓ | ✓ | ❌ | ❌ |
| n_negatives | ❌ | ✓ | ✓ | ❌ | ❌ |
| margin | ❌ | ✓ | ✓ | ❌ | ❌ |
| dist_fct | ❌ | ✓ | ✓ | ❌ | ✓ |
| dmargin | ❌ | ✓ | ✓ | ❌ | ✓ |
| gamma | ❌ | ✓ | ✓ | ✓ | ✓ |
| epsilon | ✓ (if fgsm) | ✓ (if fgsm) | ✓ (if fgsm) | ✓ (if fgsm) | ✓ (if fgsm) |
| n_neighbors | ❌ | ✓ | ✓ | ✓ | ✓ |

## Prototypes with n_positives/n_negatives

**Are they compatible?** ✓ **YES, but with caveats:**

1. **When `prototypes_to_use='batch'`:**
   - Prototypes are from batch samples
   - `n_positives` and `n_negatives` still control tuple loss behavior
   - They work together in the embedding space
   - ✓ Compatible and often beneficial

2. **When `prototypes_to_use='class'`:**
   - Prototypes are learned class centroids
   - `n_positives` and `n_negatives` control which samples form tuples during training
   - The class prototypes act as reference points
   - ✓ Compatible

3. **When `prototypes_to_use='no'`:**
   - No prototypes used
   - `n_positives` and `n_negatives` fully control tuple formation
   - ✓ Fully compatible (standard operation)

**Key Insight:** Prototypes and tuple loss parameters are **orthogonal**. Prototypes define WHAT to learn from, while n_positives/n_negatives define HOW MANY to sample. They enhance each other!

## Summary of Smart Optimization

The system now:
- ✅ Only optimizes relevant parameters for each configuration
- ✅ Prevents invalid parameter combinations
- ✅ Keeps n_calibration fixed at 0
- ✅ Handles prototype-specific tuning
- ✅ Respects loss function requirements
- ✅ Enables comprehensive hyperparameter search when needed
- ✅ Avoids wasting optimization trials on irrelevant parameters
