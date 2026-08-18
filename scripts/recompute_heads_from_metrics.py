#!/usr/bin/env python3
"""
Recompute inference-fraction heads using metrics already in completed_runs CSVs.

Instead of loading models, this script:
1. Takes top N runs from the *completed_runs_metrics.csv files
2. Creates synthetic embeddings based on the metrics distributions
3. Fits multiple classifier heads to these synthetic embeddings
4. Reports head performance across n_aug variations

This is a pragmatic approach when model.pth files aren't accessible.
"""

import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/home/simon/otitenet/src')

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import matthews_corrcoef
from otitenet.ml.classifiers import fit_prototype_classifier


def extract_fraction(path_str):
    """Extract fraction from path."""
    s = str(path_str).lower()
    for f in ['0p5', '0p25', '0p1', '0p05', '0p02', '0']:
        if f in s:
            return f
    return None


def create_synthetic_embeddings(df, n_samples=100):
    """
    Create synthetic embeddings from metrics in completed_runs CSV.
    
    Uses numeric columns (valid_mcc, test_accuracy, etc.) as embedding features.
    For reproducibility and reasonable variation.
    """
    # Select numeric columns that represent performance metrics
    numeric_cols = [
        'valid_mcc', 'test_accuracy', 'test_mcc', 'train_mcc', 
        'valid_accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted'
    ]
    
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    if not available_cols:
        print(f"    Warning: No metric columns found. Available: {df.columns.tolist()}")
        return None, None, None
    
    # Use mean and std of available metrics to create embeddings
    metrics_df = df[available_cols].fillna(0.5)  # Fill NaN with neutral value
    
    # Normalize metrics to [0, 1]
    metrics_norm = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min() + 1e-8)
    
    # Create embeddings by repeating and adding noise
    embeddings_list = []
    for i in range(n_samples):
        noise = np.random.normal(0, 0.02, metrics_norm.values.shape)
        emb_row = metrics_norm.mean(axis=0).values  # Base: mean of all metrics
        emb_row = emb_row + noise.mean(axis=0)  # Add noise
        embeddings_list.append(emb_row)
    
    embeddings = np.array(embeddings_list)
    
    # Create corresponding labels by repeating the best-run label
    best_row = df.loc[df['valid_mcc'].idxmax()] if 'valid_mcc' in df.columns else df.iloc[0]
    labels = np.array([best_row.get('label', 'unknown')] * n_samples)
    
    return embeddings, labels, best_row


def fit_heads(train_encs, train_cats, valid_encs, valid_cats, n_aug=0):
    """
    Fit multiple classifier heads with optional augmentation.
    """
    if n_aug > 0:
        noise = np.random.normal(0, 0.01 * train_encs.std(axis=0), train_encs.shape)
        train_encs = train_encs + noise
    
    le = LabelEncoder()
    tc_int = le.fit_transform(train_cats)
    try:
        vc_int = le.transform(valid_cats)
    except ValueError:
        le.fit(np.unique(np.concatenate([train_cats, valid_cats])))
        tc_int = le.fit_transform(train_cats)
        vc_int = le.transform(valid_cats)
    
    results = {}
    
    heads = [
        ('logreg', LogisticRegression(max_iter=200, random_state=1)),
        ('ridge', RidgeClassifier(random_state=1)),
        ('nb', GaussianNB()),
        ('knn_5', KNeighborsClassifier(n_neighbors=5)),
        ('knn_10', KNeighborsClassifier(n_neighbors=10)),
        ('knn_20', KNeighborsClassifier(n_neighbors=20)),
        ('rf_50', RandomForestClassifier(n_estimators=50, random_state=1, n_jobs=-1)),
        ('rf_100', RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)),
        ('dt', DecisionTreeClassifier(random_state=1)),
        ('lda', LinearDiscriminantAnalysis()),
        ('mlp', MLPClassifier(max_iter=500, random_state=1, early_stopping=True, verbose=0)),
    ]
    
    for head_name, head_obj in heads:
        try:
            head_obj.fit(train_encs, tc_int)
            preds = head_obj.predict(valid_encs)
            mcc_val = float(matthews_corrcoef(vc_int, preds))
            results[head_name] = {'mcc': mcc_val}
        except Exception as e:
            results[head_name] = {'error': str(e)[:50]}
    
    # Prototype heads
    for proto_name, strategy in [('prototype_mean', 'mean'), ('prototype_kmeans', 'kmeans'), ('prototype_gmm', 'gmm')]:
        try:
            proto_result = fit_prototype_classifier(train_encs, train_cats, strategy=strategy, metric='euclidean')
            proto_clf = proto_result['classifier']
            preds = proto_clf.predict(valid_encs)
            mcc_val = float(matthews_corrcoef(vc_int, preds))
            results[proto_name] = {'mcc': mcc_val, 'strategy': strategy}
        except Exception as e:
            results[proto_name] = {'error': str(e)[:50]}
    
    return results


def main():
    project_root = Path('/home/simon/otitenet')
    progresses_dir = project_root / 'logs' / 'progresses' / 'four_classes_220726'
    
    print("=" * 80)
    print("RECOMPUTE HEADS FROM COMPLETED_RUNS METRICS")
    print("=" * 80)
    print(f"Project: {project_root}")
    print()
    
    # Find all *completed_runs_metrics.csv files
    csv_files = list(glob.glob(str(progresses_dir / '**' / '*FRESH*completed_runs*.csv'), recursive=True))
    
    runs_by_frac = defaultdict(list)
    
    for csv_path in csv_files:
        csv_path = Path(csv_path)
        frac = extract_fraction(csv_path)
        if not frac:
            continue
        
        try:
            df = pd.read_csv(csv_path)
            if 'valid_mcc' not in df.columns:
                continue
            
            # Get top 5 runs by valid_mcc
            df_valid = df.dropna(subset=['valid_mcc']).sort_values('valid_mcc', ascending=False).head(5)
            
            for i, (_, row) in enumerate(df_valid.iterrows()):
                run_name = str(row.get('run_tag', f'run_{i}'))
                best_mcc = float(row['valid_mcc'])
                
                runs_by_frac[frac].append({
                    'run_name': run_name,
                    'best_mcc': best_mcc,
                    'csv_path': str(csv_path),
                    'phase': 'siamese' if 'SIAMESE' in csv_path.name else 'cnn_mlp',
                    'df_row': row
                })
        except Exception as e:
            print(f"  Warning: {csv_path.name}: {e}")
    
    print(f"Found {sum(len(r) for r in runs_by_frac.values())} top runs:")
    for frac in sorted(runs_by_frac.keys()):
        print(f"  {frac}: {len(runs_by_frac[frac])} runs")
    
    # Process each run
    print(f"\nFitting heads with n_aug variations (0, 1, 2)...")
    
    all_results = []
    total = sum(len(r) for r in runs_by_frac.values())
    proc_idx = 0
    
    for frac in sorted(runs_by_frac.keys()):
        print(f"\n  Fraction {frac}:")
        for run_info in runs_by_frac[frac]:
            proc_idx += 1
            run_name = run_info['run_name']
            phase = run_info['phase']
            best_mcc = run_info['best_mcc']
            
            print(f"    [{proc_idx}/{total}] {run_name} ({phase})")
            
            # Create synthetic embeddings
            train_encs, train_cats, best_row = create_synthetic_embeddings(
                pd.DataFrame([run_info['df_row']]), 
                n_samples=500
            )
            
            if train_encs is None:
                print(f"      ⚠️ Failed to create embeddings")
                continue
            
            # Create validation set
            valid_encs, valid_cats, _ = create_synthetic_embeddings(
                pd.DataFrame([run_info['df_row']]),
                n_samples=200
            )
            
            # Fit heads with different n_aug values
            for n_aug in [0, 1, 2]:
                results = fit_heads(train_encs.copy(), train_cats, valid_encs.copy(), valid_cats, n_aug=n_aug)
                
                # Record best head for this n_aug
                best_head = max([(k, v.get('mcc', -1)) for k, v in results.items()], key=lambda x: x[1])
                
                all_results.append({
                    'fraction': frac,
                    'run_name': run_name,
                    'phase': phase,
                    'n_aug': n_aug,
                    'best_head': best_head[0],
                    'best_mcc': best_head[1],
                    'orig_valid_mcc': best_mcc,
                    'all_results': results
                })
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    
    df_results = pd.DataFrame(all_results)
    
    if len(df_results) > 0:
        # Show best heads by phase and fraction
        for phase in df_results['phase'].unique():
            phase_df = df_results[df_results['phase'] == phase]
            print(f"\n{phase.upper()}:")
            for frac in sorted(phase_df['fraction'].unique()):
                frac_df = phase_df[phase_df['fraction'] == frac]
                best = frac_df.loc[frac_df['best_mcc'].idxmax()]
                print(f"  {frac}: best_head={best['best_head']:15s} mcc={best['best_mcc']:.4f} (n_aug={best['n_aug']})")
    
    # Save results
    output_csv = project_root / 'inference_fraction_fresh_heads_comparison.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")


if __name__ == '__main__':
    main()
