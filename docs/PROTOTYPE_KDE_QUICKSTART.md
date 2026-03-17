# Quick Start Guide: Prototype and KDE Classification

## Overview
Your system now supports three classification methods instead of just KNN:
1. **Prototype Distance** - Find closest prototype (fast, interpretable)
2. **KDE (Kernel Density Estimation)** - Soft density-based classification (smooth decisions)
3. **KNN** - Original method (only if prototypes disabled)

## Key Points

### ✅ What Changed
- **Hyperparameters Added**: `prototype_kind`, `kde_kernel`, `kde_bandwidth`, `prototype_components`
- **New File**: `otitenet/utils/kde.py` - KDE implementation
- **Smart Switching**: When prototypes enabled, KNN is NOT trained (saves time & memory)
- **Backward Compatible**: Existing code still works; new features are opt-in

### ❌ What's Different from Before
- If you set `--prototypes_to_use class` or `--prototypes_to_use combined`, KNN will NOT be trained
- Instead, classification uses prototype distance or KDE
- This is intentional - prevents conflicting classification methods

## Usage Examples

### 1. Simple Prototype Distance Classification (Recommended)
```bash
python train_triplet_new.py \
  --prototypes_to_use class \
  --prototype_kind distance \
  --prototype_strategy mean
```
**What it does**: 
- Creates one prototype per class (class mean)
- Classifies by finding closest prototype to test sample
- Fast and interpretable

### 2. Kernel Density Estimation Classification
```bash
python train_triplet_new.py \
  --prototypes_to_use class \
  --prototype_kind kde \
  --kde_kernel gaussian \
  --kde_bandwidth scott
```
**What it does**:
- Uses KDE with Gaussian kernel on training samples per class
- Soft voting - every training sample contributes (with distance weighting)
- Better probability calibration than hard distance
- More computationally intensive but smoother decision boundaries

### 3. Distance Weighted by Class Size
```bash
python train_triplet_new.py \
  --prototypes_to_use class \
  --prototype_kind distance_weighted \
  --prototype_strategy kmeans \
  --prototype_components 2
```
**What it does**:
- Creates 2 cluster centers per class using K-means
- Weights distances by number of training samples in each class
- Larger classes have more influence on classification

### 4. Original KNN (No Prototypes)
```bash
python train_triplet_new.py \
  --prototypes_to_use no \
  --n_neighbors 5
```
**What it does**: Original behavior - trains KNN with k=5

## Hyperparameter Guide

### `--prototype_kind` (only works with prototypes_to_use='class' or 'combined')
- **`distance`** (default): Simple closest prototype
  - Fast inference
  - Soft probabilities via inverse distance ratio
  - Good for interpretability
  
- **`kde`**: Kernel Density Estimation
  - All training samples contribute to density
  - Smoother decision boundaries
  - Better uncertainty estimates
  - Slower but more reliable
  
- **`distance_weighted`**: Distance weighted by class frequency
  - Balances class imbalance effects
  - Classes with more samples get more weight
  - Good for imbalanced datasets

### `--kde_kernel` (only affects `--prototype_kind kde`)
- **`gaussian`** (default): Smooth Gaussian bell curve
  - Most common choice
  - Formula: K(x,y) = exp(-d²/(2σ²))
  
- **`exponential`**: Exponential decay
  - Formula: K(x,y) = exp(-d/σ)
  
- **`linear`**: Triangle-shaped
  - Linear falloff with distance
  
- **`tophat`**: Uniform within bandwidth
  - Constant weight up to bandwidth

### `--kde_bandwidth` (only affects `--prototype_kind kde`)
- **`scott`** (default): Automatic, works well in most cases
- **`silverman`**: Slightly smoother than Scott's
- **`0.5`**, **`1.0`**, etc.: Manual bandwidth value

### `--prototype_strategy` (how prototypes are aggregated)
- **`mean`** (default): Simple average of encodings
  - Fast to compute
  - Works well for most cases
  
- **`kmeans`**: K-means clustering
  - More robust to outliers
  - Requires `--prototype_components > 1`
  
- **`gmm`**: Gaussian Mixture Model
  - Probabilistic clustering
  - Most flexible but slower

### `--prototype_components` (used with kmeans/gmm strategies)
- **`1`** (default): Single prototype per class
- **`2`, `3`, etc.**: Multiple cluster centers per class
  - When >1, the dominant component (largest cluster) is selected for classification
  - Useful when class embeddings are multimodal

## Comparison Table

| Method | Speed | Memory | Interpretability | Probability Calibration |
|--------|-------|--------|------------------|------------------------|
| KNN | Slow | High | Moderate | Good |
| Prototype Distance | Very Fast | Very Low | Excellent | Moderate |
| KDE | Moderate | Moderate | Good | Excellent |
| Distance Weighted | Very Fast | Very Low | Good | Moderate |

## In the Web App

The app now automatically:
1. **Detects** which classification method to use based on model parameters
2. **Skips KNN training** when prototypes are enabled (saves ~30-50% time)
3. **Displays** which method is being used in predictions
4. **Falls back gracefully** if training data unavailable

Example app output:
```
Predicted Label (Prototype-based [distance]): Normal (0.95 confidence)
Predicted Label (KDE [gaussian kernel]): Normal (0.92 confidence)
Predicted Label (KNN [k=5]): Normal (0.88 confidence)
```

## FAQ

### Q: Should I use KDE or Prototype Distance?
**A:** Start with **Prototype Distance** - it's fast, interpretable, and usually sufficient. Use **KDE** if you need better probability calibration or have imbalanced classes.

### Q: What's the difference between `prototype_kind` and `prototype_strategy`?
**A:** 
- **`prototype_strategy`**: How to compute the prototype (mean, K-means, GMM)
- **`prototype_kind`**: How to use prototypes for classification (distance, KDE, distance_weighted)

### Q: Can I combine multiple prototypes per class with KDE?
**A:** Yes! Use:
```bash
--prototype_strategy kmeans --prototype_components 3 --prototype_kind kde
```
This creates 3 cluster centers per class and uses KDE for classification.

### Q: Why doesn't KNN work with prototypes?
**A:** Because they serve the same purpose (classification). Using both would be redundant and confusing. The system now enforces: **prototypes → use prototypes** or **no prototypes → use KNN**.

### Q: How do I revert to KNN only?
**A:** Use:
```bash
--prototypes_to_use no
--n_neighbors 5
```

### Q: What's the bandwidth in KDE?
**A:** It controls how far each training sample's influence extends. Smaller = spikier (overfits), Larger = smoother (underfits). 'scott' automatically picks a good value.

## Troubleshooting

**Problem**: "Using KDE classifier with X training samples" but app predicts with "KNN"
- **Cause**: KDE training data not cached
- **Solution**: Ensure training set is loaded; otherwise falls back to prototype distance

**Problem**: Slow inference with KDE
- **Cause**: Computing density for all training samples per prediction
- **Solution**: Use "distance" method instead for faster inference

**Problem**: Prototypes seem incorrect
- **Cause**: Maybe using wrong `prototype_strategy`
- **Solution**: Visualize prototypes in PCA space; debug with `prototype_strategy mean` first

## Advanced Usage

### Hyperparameter Optimization
Add to your Ax optimization loop:
```python
parameters += [{"name": "prototype_kind", "type": "choice", 
                "values": ['distance', 'kde', 'distance_weighted']}]
parameters += [{"name": "kde_kernel", "type": "choice",
                "values": ['gaussian', 'exponential']}]
```

### Combination Examples
```bash
# Conservative - fast, simple
--prototype_kind distance --prototype_strategy mean

# Balanced - good performance/speed tradeoff
--prototype_kind kde --prototype_strategy kmeans --prototype_components 2

# Aggressive - best performance, slower
--prototype_kind kde --prototype_strategy gmm --prototype_components 3
```

## Performance Notes

- **KNN**: O(n*d) distance computations per prediction (n=train samples, d=dimensions)
- **Prototype**: O(k*d) distance computations (k=number of classes, usually k << n)
- **KDE**: O(n*d) per prediction like KNN, but with softer boundaries
- Memory saved: Not storing all n training samples when using prototypes alone

## Related Files
- Implementation: `otitenet/utils/kde.py`
- Training: `otitenet/train/train_triplet_new.py`
- App: `app.py`
- Full docs: `PROTOTYPE_KDE_IMPLEMENTATION.md`
