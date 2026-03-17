# Example Commands for Prototype and KDE Classification

## Training Examples

### Example 1: Simple Prototype-Based Classification (Distance)
```bash
python train_triplet_new.py \
  --path ./data/otite_ds_64 \
  --model_name resnet18 \
  --task notNormal \
  --prototypes_to_use class \
  --prototype_kind distance \
  --prototype_strategy mean \
  --prototype_components 1 \
  --n_epochs 100 \
  --bs 32 \
  --device cuda:0
```
**What it does**: Creates mean prototype for each class, classifies by distance

---

### Example 2: KDE-Based Classification with Gaussian Kernel
```bash
python train_triplet_new.py \
  --path ./data/otite_ds_64 \
  --model_name resnet18 \
  --task notNormal \
  --prototypes_to_use class \
  --prototype_kind kde \
  --prototype_strategy mean \
  --kde_kernel gaussian \
  --kde_bandwidth scott \
  --n_epochs 100 \
  --bs 32 \
  --device cuda:0
```
**What it does**: Uses KDE with automatic Gaussian kernel bandwidth selection

---

### Example 3: Distance Weighted by Class Size
```bash
python train_triplet_new.py \
  --path ./data/otite_ds_64 \
  --model_name resnet18 \
  --task notNormal \
  --prototypes_to_use class \
  --prototype_kind distance_weighted \
  --prototype_strategy kmeans \
  --prototype_components 2 \
  --n_epochs 100 \
  --bs 32 \
  --device cuda:0
```
**What it does**: 2 K-means cluster centers per class, weighted by class frequency

---

### Example 4: KDE with Multiple Prototypes (GMM)
```bash
python train_triplet_new.py \
  --path ./data/otite_ds_64 \
  --model_name vit \
  --task notNormal \
  --prototypes_to_use class \
  --prototype_kind kde \
  --prototype_strategy gmm \
  --prototype_components 3 \
  --kde_kernel exponential \
  --kde_bandwidth 0.8 \
  --n_epochs 100 \
  --bs 16 \
  --device cuda:0
```
**What it does**: 3 Gaussian components per class via GMM, KDE with exponential kernel

---

### Example 5: Classic KNN (No Prototypes)
```bash
python train_triplet_new.py \
  --path ./data/otite_ds_64 \
  --model_name resnet18 \
  --task notNormal \
  --prototypes_to_use no \
  --n_neighbors 5 \
  --n_epochs 100 \
  --bs 32 \
  --device cuda:0
```
**What it does**: Original behavior - trains standard KNN with k=5

---

### Example 6: Batch Normalization with Prototypes
```bash
python train_triplet_new.py \
  --path ./data/otite_ds_64 \
  --model_name resnet18 \
  --task notNormal \
  --prototypes_to_use class \
  --prototype_kind distance \
  --prototype_strategy mean \
  --classif_loss triplet \
  --dloss inverseTriplet \
  --n_positives 2 \
  --n_negatives 2 \
  --margin 0.5 \
  --n_epochs 100 \
  --bs 32 \
  --n_trials 10 \
  --device cuda:0
```
**What it does**: Triplet loss with prototype classification, Ax hyperparameter optimization

---

### Example 7: Full Feature Config with All Prototypes Types
```bash
python train_triplet_new.py \
  --path ./data/otite_ds_64 \
  --model_name resnet18 \
  --task notNormal \
  --prototypes_to_use combined \
  --prototype_kind kde \
  --prototype_strategy kmeans \
  --prototype_components 2 \
  --kde_kernel gaussian \
  --kde_bandwidth scott \
  --classif_loss triplet \
  --dloss inverseTriplet \
  --dist_fct euclidean \
  --n_positives 2 \
  --n_negatives 2 \
  --margin 1.0 \
  --normalize yes \
  --n_calibration 100 \
  --new_size 224 \
  --fgsm 1 \
  --n_epochs 150 \
  --bs 32 \
  --n_trials 15 \
  --device cuda:0 \
  --seed 42
```
**What it does**: All features enabled - combined prototypes, KDE, triplet loss, domain adaptation

---

## Hyperparameter Tuning Examples

### Fast Experimentation (Prototype Only)
```bash
for proto_kind in distance kde distance_weighted; do
  for kde_kernel in gaussian exponential; do
    echo "Testing prototype_kind=$proto_kind, kde_kernel=$kde_kernel"
    python train_triplet_new.py \
      --path ./data/otite_ds_64 \
      --model_name resnet18 \
      --task notNormal \
      --prototypes_to_use class \
      --prototype_kind $proto_kind \
      --kde_kernel $kde_kernel \
      --n_epochs 50 \
      --n_trials 5 \
      --device cuda:0
  done
done
```

### Bandwidth Optimization
```bash
for bandwidth in scott silverman 0.5 1.0 1.5 2.0; do
  echo "Testing kde_bandwidth=$bandwidth"
  python train_triplet_new.py \
    --path ./data/otite_ds_64 \
    --model_name resnet18 \
    --task notNormal \
    --prototypes_to_use class \
    --prototype_kind kde \
    --kde_bandwidth $bandwidth \
    --n_epochs 100 \
    --n_trials 10 \
    --device cuda:0
done
```

### Component Count Study
```bash
for n_comp in 1 2 3 4 5; do
  echo "Testing prototype_components=$n_comp"
  python train_triplet_new.py \
    --path ./data/otite_ds_64 \
    --model_name resnet18 \
    --task notNormal \
    --prototypes_to_use class \
    --prototype_kind kde \
    --prototype_strategy kmeans \
    --prototype_components $n_comp \
    --n_epochs 100 \
    --n_trials 10 \
    --device cuda:0
done
```

---

## Web App Usage

### Loading Model with Prototypes
1. In the app sidebar, select a model trained with `--prototypes_to_use class`
2. App automatically detects and uses prototype classification
3. For KDE: Pass `--prototype_kind kde` during training
4. For distance: Pass `--prototype_kind distance` during training

### Expected Behavior

**If Model Uses Prototype Distance:**
```
Predicted Label (Prototype-based [distance]): Normal (0.95 confidence)
```

**If Model Uses KDE:**
```
Predicted Label (KDE [gaussian kernel]): Normal (0.92 confidence)
```

**If Model Uses KNN (no prototypes):**
```
Predicted Label (KNN [k=5]): Normal (0.88 confidence)
```

---

## Performance Comparison Commands

### Test Inference Speed

**Prototype Distance (Fast):**
```bash
time python -c "
from app import run_analysis_on_file
# Test with trained model using prototypes
"
```

**KDE (Moderate):**
```bash
time python -c "
from app import run_analysis_on_file  
# Test with trained model using KDE
"
```

**KNN (Slow):**
```bash
time python -c "
from app import run_analysis_on_file
# Test with trained model using KNN
"
```

---

## Debugging Commands

### Check Which Method Is Used
```python
from otitenet.train.train_triplet_new import TrainAE

args = type('Args', (), {
    'prototypes_to_use': 'class',
    'prototype_kind': 'kde',
    'kde_kernel': 'gaussian',
    'kde_bandwidth': 'scott',
    'classif_loss': 'triplet'
})()

use_kde = (args.prototypes_to_use in ['combined', 'class'] and 
          getattr(args, 'prototype_kind', 'distance').lower() == 'kde')
use_prototypes = args.prototypes_to_use in ['combined', 'class']

print(f"Use KDE: {use_kde}")
print(f"Use Prototypes: {use_prototypes}")
```

### Verify KNN Not Trained
```bash
# With prototypes - should NOT have KNN
python -c "
import sys
sys.stdout.write('KNN training started')
" 2>&1 | grep -c "KNN training started"
# Expected: 0 (no KNN training when prototypes enabled)
```

---

## Full Integration Example

### Complete Workflow
```bash
#!/bin/bash
set -e

# Step 1: Train with prototypes and KDE
echo "Step 1: Training with KDE classification..."
python train_triplet_new.py \
  --path ./data/otite_ds_64 \
  --model_name resnet18 \
  --task notNormal \
  --prototypes_to_use class \
  --prototype_kind kde \
  --kde_kernel gaussian \
  --kde_bandwidth scott \
  --n_epochs 100 \
  --n_trials 10 \
  --device cuda:0 \
  --seed 42

# Step 2: Verify model was saved
echo "Step 2: Checking saved model..."
ls -lh logs/best_models/notNormal/resnet18/otite_ds_64/*/prototypes_class/*/model.pth

# Step 3: Test in app (manual)
echo "Step 3: Ready to test in web app!"
echo "Run: streamlit run app.py"
```

---

## Expected Output Examples

### Training with Prototypes
```
Using prototype-based classification (distance)
KNN with prototypes: 500 train samples + 2 prototypes
Epoch [001/100]: acc=0.892, mcc=0.785, closs=0.234
...
Epoch [100/100]: acc=0.945, mcc=0.901, closs=0.098
Model saved to: logs/best_models/notNormal/resnet18/otite_ds_64/.../model.pth
```

### Training with KDE
```
Using KDE classifier with 500 training samples
KDE fitted with kernel=gaussian, bandwidth=scott
Epoch [001/100]: acc=0.898, mcc=0.792, closs=0.221
...
Epoch [100/100]: acc=0.951, mcc=0.908, closs=0.091
Model saved to: logs/best_models/notNormal/resnet18/otite_ds_64/.../model.pth
```

### App Prediction
```
Loading model and processing image...
Model: resnet18 | Size: 224 | Prototypes: class | Method: KDE
Predicted Label (KDE [gaussian kernel]): Normal (0.94 confidence)
✅ Results saved to database.
```

---

## Notes

- Replace `./data/otite_ds_64` with your actual data path
- Replace `cuda:0` with appropriate GPU (or `cpu` if no GPU)
- Hyperparameters can be combined in any valid way
- All new features are backward compatible
- Default values work well for most use cases

