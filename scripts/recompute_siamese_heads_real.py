#!/usr/bin/env python3
"""
Recompute Siamese classifier heads using REAL cached embeddings
(train_encodings.npz / valid_encodings.npz / test_encodings.npz) saved
during fresh inference-fraction Optuna runs.

For each fraction, takes the top-N runs by valid_mcc from the
completed_runs_metrics.csv, locates the run's UUID directory under
logs/four_classes_220726/<uuid>/, loads its train/valid/test encodings,
and fits multiple classifier heads (LogReg, RF, KNN-k, LDA, MLP, and
Prototype variants: mean/kmeans/gmm).

Usage:
    python3 scripts/recompute_siamese_heads_real.py --top-n 5
    python3 scripts/recompute_siamese_heads_real.py --fractions 0p5 0p25 --top-n 3
"""

import sys
import glob
import argparse
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
from sklearn.svm import LinearSVC
from sklearn.metrics import matthews_corrcoef, accuracy_score
from otitenet.ml.classifiers import fit_prototype_classifier

PROJECT_ROOT = Path('/home/simon/otitenet')
LOGS_DIR = PROJECT_ROOT / 'logs' / 'four_classes_220726'
PROGRESSES_DIR = PROJECT_ROOT / 'logs' / 'progresses' / 'four_classes_220726'

FRACTIONS = ['0p5', '0p25', '0p1', '0p05', '0p02']


def find_siamese_csv(fraction):
    """Find the fresh siamese completed_runs_metrics.csv for a fraction."""
    pattern = str(PROGRESSES_DIR / f'*train{fraction}_seed42' / f'INF_FRAC_FRESH_SIAMESE_P{fraction}_S42*completed_runs_metrics.csv')
    files = glob.glob(pattern)
    return files[0] if files else None


def find_run_dir(uuid):
    """Find the run directory for a UUID (siamese runs save directly under logs/four_classes_220726/<uuid>/)."""
    candidate = LOGS_DIR / uuid
    if candidate.exists() and (candidate / 'train_encodings.npz').exists():
        return candidate
    return None


def load_encodings(run_dir):
    """Load train/valid/test encodings for a run."""
    result = {}
    for split in ['train', 'valid', 'test']:
        npz_path = run_dir / f'{split}_encodings.npz'
        if npz_path.exists():
            d = np.load(npz_path, allow_pickle=True)
            result[split] = {
                'embeddings': d['embeddings'],
                'labels': d['labels'],
            }
        else:
            result[split] = None
    return result


def get_heads():
    return [
        ('logreg', LogisticRegression(max_iter=200, random_state=1)),
        ('linearsvc', LinearSVC(max_iter=2000, random_state=1)),
        ('ridge', RidgeClassifier(random_state=1)),
        ('nb', GaussianNB()),
        ('knn_5', KNeighborsClassifier(n_neighbors=5)),
        ('knn_10', KNeighborsClassifier(n_neighbors=10)),
        ('knn_20', KNeighborsClassifier(n_neighbors=20)),
        ('rf_50', RandomForestClassifier(n_estimators=50, random_state=1, n_jobs=-1)),
        ('rf_100', RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)),
        ('rf_200', RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1)),
        ('dt', DecisionTreeClassifier(random_state=1)),
        ('lda', LinearDiscriminantAnalysis()),
        ('mlp', MLPClassifier(max_iter=500, random_state=1, early_stopping=True, verbose=0)),
    ]


def fit_and_eval_heads(train_encs, train_labels, valid_encs, valid_labels, test_encs=None, test_labels=None):
    """Fit all heads on train, evaluate on valid (and test if given)."""
    le = LabelEncoder()
    le.fit(np.unique(np.concatenate([train_labels, valid_labels])))
    tc_int = le.transform(train_labels)
    vc_int = le.transform(valid_labels)
    test_int = le.transform(test_labels) if test_labels is not None else None

    results = {}

    for head_name, head_obj in get_heads():
        try:
            head_obj.fit(train_encs, tc_int)
            valid_preds = head_obj.predict(valid_encs)
            valid_mcc = float(matthews_corrcoef(vc_int, valid_preds))
            valid_acc = float(accuracy_score(vc_int, valid_preds))
            entry = {'valid_mcc': valid_mcc, 'valid_acc': valid_acc}
            if test_encs is not None:
                test_preds = head_obj.predict(test_encs)
                entry['test_mcc'] = float(matthews_corrcoef(test_int, test_preds))
                entry['test_acc'] = float(accuracy_score(test_int, test_preds))
            results[head_name] = entry
        except Exception as e:
            results[head_name] = {'error': str(e)[:80]}

    # Prototype heads
    for proto_name, strategy in [('prototype_mean', 'mean'), ('prototype_kmeans', 'kmeans'), ('prototype_gmm', 'gmm')]:
        try:
            proto_result = fit_prototype_classifier(train_encs, tc_int, strategy=strategy, metric='euclidean')
            clf = proto_result['classifier']
            valid_preds = clf.predict(valid_encs)
            valid_mcc = float(matthews_corrcoef(vc_int, valid_preds))
            valid_acc = float(accuracy_score(vc_int, valid_preds))
            entry = {'valid_mcc': valid_mcc, 'valid_acc': valid_acc}
            if test_encs is not None:
                test_preds = clf.predict(test_encs)
                entry['test_mcc'] = float(matthews_corrcoef(test_int, test_preds))
                entry['test_acc'] = float(accuracy_score(test_int, test_preds))
            results[proto_name] = entry
        except Exception as e:
            results[proto_name] = {'error': str(e)[:80]}

    return results


def main():
    parser = argparse.ArgumentParser(description='Recompute Siamese heads with real cached embeddings.')
    parser.add_argument('--fractions', nargs='+', default=FRACTIONS)
    parser.add_argument('--top-n', type=int, default=5, help='Top N runs per fraction to evaluate')
    parser.add_argument('--output', type=str, default=str(PROJECT_ROOT / 'siamese_heads_real_comparison.csv'))
    args = parser.parse_args()

    all_rows = []

    for frac in args.fractions:
        print(f'\n{"=" * 70}')
        print(f'Fraction: {frac}')
        print(f'{"=" * 70}')

        csv_path = find_siamese_csv(frac)
        if not csv_path:
            print(f'  ⚠️ No CSV found for fraction {frac}')
            continue

        df = pd.read_csv(csv_path)
        df_siamese = df[df['kind'] == 'siamese'] if 'kind' in df.columns else df
        df_top = df_siamese.dropna(subset=['valid_mcc', 'uuid']).sort_values('valid_mcc', ascending=False).head(args.top_n)

        print(f'  CSV: {csv_path}')
        print(f'  Top {len(df_top)} runs by valid_mcc:')

        for rank, (_, row) in enumerate(df_top.iterrows(), start=1):
            uuid = str(row['uuid'])
            orig_valid_mcc = float(row['valid_mcc'])
            orig_test_mcc = row.get('test_mcc', None)

            run_dir = find_run_dir(uuid)
            if run_dir is None:
                print(f'    [{rank}] {uuid[:8]}... ⚠️ Run dir/encodings not found')
                continue

            encodings = load_encodings(run_dir)
            if encodings['train'] is None or encodings['valid'] is None:
                print(f'    [{rank}] {uuid[:8]}... ⚠️ Missing train/valid encodings')
                continue

            train_encs = encodings['train']['embeddings']
            train_labels = encodings['train']['labels']
            valid_encs = encodings['valid']['embeddings']
            valid_labels = encodings['valid']['labels']
            test_encs = encodings['test']['embeddings'] if encodings['test'] else None
            test_labels = encodings['test']['labels'] if encodings['test'] else None

            print(f'    [{rank}] {uuid[:8]}... train_n={len(train_encs)}, valid_n={len(valid_encs)}, orig_valid_mcc={orig_valid_mcc:.4f}')

            head_results = fit_and_eval_heads(train_encs, train_labels, valid_encs, valid_labels, test_encs, test_labels)

            for head_name, metrics in head_results.items():
                if 'error' in metrics:
                    continue
                all_rows.append({
                    'fraction': frac,
                    'rank': rank,
                    'uuid': uuid,
                    'head': head_name,
                    'orig_valid_mcc': orig_valid_mcc,
                    'orig_test_mcc': orig_test_mcc,
                    'new_valid_mcc': metrics.get('valid_mcc'),
                    'new_valid_acc': metrics.get('valid_acc'),
                    'new_test_mcc': metrics.get('test_mcc'),
                    'new_test_acc': metrics.get('test_acc'),
                })

            # Print best head for this run
            valid_results = {k: v for k, v in head_results.items() if 'valid_mcc' in v}
            if valid_results:
                best_head = max(valid_results.items(), key=lambda kv: kv[1]['valid_mcc'])
                print(f'         Best head: {best_head[0]:20s} valid_mcc={best_head[1]["valid_mcc"]:.4f} test_mcc={best_head[1].get("test_mcc", float("nan")):.4f}')

    # Save results
    df_out = pd.DataFrame(all_rows)
    if len(df_out) > 0:
        df_out.to_csv(args.output, index=False)
        print(f'\n{"=" * 70}')
        print(f'Saved {len(df_out)} rows to {args.output}')
        print(f'{"=" * 70}')

        # Print summary: best head per fraction (by best valid_mcc across all runs/heads)
        print('\nBEST HEAD PER FRACTION (by valid_mcc):')
        for frac in df_out['fraction'].unique():
            frac_df = df_out[df_out['fraction'] == frac]
            best_row = frac_df.loc[frac_df['new_valid_mcc'].idxmax()]
            print(f'  {frac}: head={best_row["head"]:20s} valid_mcc={best_row["new_valid_mcc"]:.4f} '
                  f'test_mcc={best_row["new_test_mcc"]:.4f} test_acc={best_row["new_test_acc"]:.4f} (uuid={best_row["uuid"][:8]})')

        # Average by head across all fractions
        print('\nAVERAGE PERFORMANCE BY HEAD (across all fractions/runs):')
        head_avg = df_out.groupby('head')[['new_valid_mcc', 'new_test_mcc']].mean().sort_values('new_valid_mcc', ascending=False)
        print(head_avg.to_string())
    else:
        print('\n⚠️ No results collected. Check that run directories and encodings exist.')


if __name__ == '__main__':
    main()
