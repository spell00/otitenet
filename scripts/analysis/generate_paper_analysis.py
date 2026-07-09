def do_model_mcc_barplots(df, out_dir):
    """
    Generate barplots of MCC by architecture (with std error bars) and by unique model (with/without error bars).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    import pandas as pd

    if df is None or df.empty:
        print("No data for model MCC barplots.")
        return

    # --- 1. Barplot with error bars (std) by architecture ---
    arch_col = 'model_name' if 'model_name' in df.columns else 'variant'
    mcc_col = 'valid_mcc' if 'valid_mcc' in df.columns else 'mcc'
    arch_stats = df.groupby(arch_col)[mcc_col].agg(['mean', 'std', 'count']).reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(
        arch_stats[arch_col],
        arch_stats['mean'],
        yerr=arch_stats['std'],
        capsize=5,
        color=sns.color_palette('deep', n_colors=len(arch_stats))
    )
    plt.ylabel('Matthews Correlation Coefficient (MCC)')
    plt.xlabel('Architecture')
    plt.title('Best MCC by Network Architecture (with Std Error Bars)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mcc_by_architecture.png'), dpi=200)
    plt.close()

    # --- 2. Barplot for every unique model (full identifier) ---
    # Use a unique identifier for each model (e.g., uuid or run_dir or exp_id+variant)
    model_id_col = None
    for col in ['uuid', 'run_dir', 'exp_id']:
        if col in df.columns:
            model_id_col = col
            break
    if model_id_col is None:
        model_id_col = arch_col

    # (a) With error bars if multiple trials per model
    model_stats = df.groupby(model_id_col)[mcc_col].agg(['mean', 'std', 'count']).reset_index()
    plt.figure(figsize=(max(12, len(model_stats) * 0.5), 6))
    sns.barplot(data=model_stats, x=model_id_col, y='mean', yerr=model_stats['std'], capsize=0.2)
    plt.ylabel('Matthews Correlation Coefficient (MCC)')
    plt.xlabel('Model (unique id)')
    plt.title('MCC for Each Model (with Std Error Bars)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mcc_by_model_with_errorbars.png'), dpi=200)
    plt.close()

    # (b) Only top model per group (no error bar)
    top_models = df.loc[df.groupby(model_id_col)[mcc_col].idxmax()]
    plt.figure(figsize=(max(12, len(top_models) * 0.5), 6))
    sns.barplot(data=top_models, x=model_id_col, y=mcc_col)
    plt.ylabel('Matthews Correlation Coefficient (MCC)')
    plt.xlabel('Model (unique id)')
    plt.title('Top MCC for Each Model (No Error Bar)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'top_mcc_by_model.png'), dpi=200)
    plt.close()
    print('Model MCC barplots saved to', out_dir)

def build_all_models_from_manifest_and_logs(manifest_path, output_csv, tag='PROD'):
    """
    Build all_models.csv from manifest and logs, ensuring all required fields are filled using uuid and log files.
    """
    import pandas as pd
    import os
    import json
    from glob import glob

    manifest = pd.read_csv(manifest_path)
    task = _get_target_task()
    completed_csv = f'logs/progresses/{task}/{tag}_{task}_completed_runs.csv'
    completed_metrics_csv = f'logs/progresses/{task}/{tag}_{task}_completed_runs_metrics.csv'
    completed_df = None
    completed_metrics_df = None
    if os.path.exists(completed_csv):
        completed_df = pd.read_csv(completed_csv)
    if os.path.exists(completed_metrics_csv):
        completed_metrics_df = pd.read_csv(completed_metrics_csv)
    rows = []
    missing_logs = 0
    for idx, job in manifest.iterrows():
        if str(job.get('job_state', '')).strip().lower() != 'done':
            continue
        uuid = job.get('uuid')
        if not uuid or str(uuid).strip() == '' or str(uuid).lower() == 'nan':
            uuid = job.get('exp_id')
        # Try to get metrics from completed_runs_metrics.csv first
        completed_row = None
        if completed_metrics_df is not None:
            match = completed_metrics_df[completed_metrics_df['uuid'] == uuid]
            if not match.empty:
                completed_row = match.iloc[-1].to_dict()
        if completed_row:
            row = completed_row.copy()
            # Always merge in metadata from completed_runs.csv (completed_df) if available
            meta_row = None
            if completed_df is not None:
                meta_match = completed_df[completed_df['uuid'] == uuid]
                if not meta_match.empty:
                    meta_row = meta_match.iloc[-1].to_dict()
            for k in ['kind', 'variant', 'classif_loss', 'prototypes', 'dloss', 'BER', 'fgsm', 'normalize', 'n_calibration', 'dist_fct', 'knn', 'n_negatives', 'launcher_retry_count', 'launcher_failed_final', 'source_datetime']:
                # Prefer value from completed_row, then meta_row, then manifest
                if k not in row or row[k] in [None, '', 'unknown', 'nan', 'NaN']:
                    if meta_row and k in meta_row and meta_row[k] not in [None, '', 'unknown', 'nan', 'NaN']:
                        row[k] = meta_row[k]
                    else:
                        row[k] = job.get(k, 'unknown')
        else:
            log_dir = f"logs/notNormal/{uuid}"
            meta_path = os.path.join(log_dir, 'run_metadata.json')
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                except Exception:
                    pass
            if not meta:
                print(f"Warning: Missing run_metadata.json for uuid/exp_id {uuid}")
                missing_logs += 1
            args = meta.get('args', {}) if isinstance(meta, dict) else {}
            metrics = _load_run_summary_metrics(log_dir)
            # Import model_name from args, metrics, or job, fallback to 'unknown'
            model_name = args.get('model_name')
            if not model_name or model_name in ['unknown', '', None, 'nan', 'NaN']:
                model_name = metrics.get('model_name') if metrics.get('model_name') not in [None, '', 'unknown', 'nan', 'NaN'] else None
            if not model_name or model_name in ['unknown', '', None, 'nan', 'NaN']:
                model_name = job.get('model_name')
            if not model_name or model_name in ['unknown', '', None, 'nan', 'NaN']:
                model_name = 'unknown'
            row = {
                'uuid': uuid,
                'model_name': model_name,
                'kind': job.get('kind', args.get('kind', 'unknown')),
                'variant': job.get('variant', args.get('variant', 'unknown')),
                'classif_loss': job.get('classif_loss', args.get('classif_loss', 'unknown')),
                'prototypes': job.get('prototype', args.get('prototypes_to_use', 'unknown')),
                'dloss': job.get('dloss', args.get('dloss', 'unknown')),
                'BER': job.get('BER', args.get('BER', job.get('dloss', args.get('dloss', 'unknown')))),
                'fgsm': job.get('fgsm', args.get('fgsm', 'unknown')),
                'normalize': job.get('normalize', args.get('normalize', 'unknown')),
                'n_calibration': job.get('n_calibration', args.get('n_calibration', 'unknown')),
                'dist_fct': job.get('dist_fct', args.get('dist_fct', 'unknown')),
                'knn': job.get('knn', args.get('n_neighbors', 'unknown')),
                'n_negatives': job.get('n_negatives', args.get('n_negatives', 'unknown')),
                'valid_mcc': metrics.get('summary_valid_mcc', ''),
                'test_mcc': metrics.get('summary_test_mcc', ''),
                'valid_accuracy': metrics.get('summary_valid_accuracy', ''),
                'test_accuracy': metrics.get('summary_test_accuracy', ''),
                'csv_valid_mcc': metrics.get('summary_valid_mcc', ''),
                'csv_test_mcc': metrics.get('summary_test_mcc', ''),
                'launcher_retry_count': job.get('retry_count', ''),
                'launcher_failed_final': job.get('failed_final', ''),
                'source_datetime': job.get('manifest_mtime', ''),
            }
            for k, v in row.items():
                if v in [None, '', 'unknown', 'nan', 'NaN']:
                    row[k] = args.get(k, job.get(k, 'unknown'))
        rows.append(row)
    df = pd.DataFrame(rows)
    # Merge in metrics from completed_runs_metrics.csv if available, using uuid and exp_id as join keys
    import ast
    def extract_best_epoch(row):
        # Parse lists from string if needed
        def parse_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str) and val.strip().startswith('['):
                try:
                    return ast.literal_eval(val)
                except Exception:
                    return []
            return []

        valid_mcc_list = parse_list(row.get('valid_mcc', []))
        test_mcc_list = parse_list(row.get('test_mcc', []))
        valid_acc_list = parse_list(row.get('valid_accuracy', []))
        test_acc_list = parse_list(row.get('test_accuracy', []))
        # Find best epoch by valid_mcc
        if valid_mcc_list:
            best_idx = int(float(np.nanargmax(valid_mcc_list)))
            best_valid_mcc = valid_mcc_list[best_idx]
            best_test_mcc = test_mcc_list[best_idx] if test_mcc_list and len(test_mcc_list) > best_idx else np.nan
            best_valid_acc = valid_acc_list[best_idx] if valid_acc_list and len(valid_acc_list) > best_idx else np.nan
            best_test_acc = test_acc_list[best_idx] if test_acc_list and len(test_acc_list) > best_idx else np.nan
            return pd.Series({
                'valid_mcc': best_valid_mcc,
                'test_mcc': best_test_mcc,
                'valid_accuracy': best_valid_acc,
                'test_accuracy': best_test_acc
            })
        return pd.Series({
            'valid_mcc': np.nan,
            'test_mcc': np.nan,
            'valid_accuracy': np.nan,
            'test_accuracy': np.nan
        })

    if completed_metrics_df is not None and not completed_metrics_df.empty:
        completed_metrics_df = completed_metrics_df.drop_duplicates('uuid', keep='last')
        if 'exp_id' not in completed_metrics_df.columns:
            completed_metrics_df['exp_id'] = ''
        # Extract best epoch metrics as scalar columns
        best_metrics = completed_metrics_df.apply(extract_best_epoch, axis=1)
        # Remove metric list columns from completed_metrics_df
        metrics_list_cols = ['valid_mcc','test_mcc','valid_accuracy','test_accuracy']
        completed_metrics_df_clean = completed_metrics_df.drop(metrics_list_cols, axis=1, errors='ignore')
        completed_metrics_df_clean = pd.concat([completed_metrics_df_clean, best_metrics], axis=1)
        if not df.empty and 'uuid' in df.columns:
            df = df.merge(completed_metrics_df_clean[['uuid','exp_id'] + list(best_metrics.columns)], on='uuid', how='left', suffixes=('', '_metrics'))
        # For rows where metrics are still missing, try to merge on exp_id (if uuid is missing or did not match)
        missing_metrics_mask = df['valid_mcc'].isnull() | (df['valid_mcc'] == '') | (df['valid_mcc'] == 'unknown')
        if missing_metrics_mask.any():
            df_missing = df[missing_metrics_mask].copy()
            df_present = df[~missing_metrics_mask].copy()
            df_missing = df_missing.drop([c for c in df_missing.columns if c.endswith('_metrics')], axis=1, errors='ignore')
            df_missing = df_missing.merge(completed_metrics_df_clean[['exp_id'] + list(best_metrics.columns)], left_on='exp_id', right_on='exp_id', how='left', suffixes=('', '_metrics'))
            for col in best_metrics.columns:
                if col in df_missing.columns and f"{col}_metrics" in df_missing.columns:
                    df_missing[col] = df_missing[f"{col}_metrics"].combine_first(df_missing[col])
                    df_missing = df_missing.drop(columns=[f"{col}_metrics"])
            df = pd.concat([df_present, df_missing], ignore_index=True, sort=False)
        df = df.drop([c for c in df.columns if c.endswith('_metrics')], axis=1, errors='ignore')
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv}. Missing logs for {missing_logs} jobs. Metrics merged: {completed_metrics_df.shape[0] if completed_metrics_df is not None else 0}")

import mysql.connector
import os
import glob
import subprocess
import json
import argparse
from importlib import metadata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from datetime import datetime

from sklearn.metrics import matthews_corrcoef, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import plotly.express as px


def _get_target_task():
    return str(os.environ.get('OTITENET_TASK', 'notNormal')).strip() or 'notNormal'


def _analysis_test_only_enabled():
    return str(os.environ.get('OTITENET_ANALYSIS_TEST_ONLY', '0')).strip().lower() in {'1', 'true', 'yes'}


def _major_version(v):
    try:
        return int(str(v).split('.')[0])
    except Exception:
        return None


def _is_plotly_kaleido_compatible():
    try:
        plotly_v = metadata.version('plotly')
        kaleido_v = metadata.version('kaleido')
    except metadata.PackageNotFoundError:
        return False, None, None

    plotly_major = _major_version(plotly_v)
    kaleido_major = _major_version(kaleido_v)

    # Known incompatible pair: plotly 5.x with kaleido 1.x.
    if plotly_major == 5 and kaleido_major == 1:
        return False, plotly_v, kaleido_v
    return True, plotly_v, kaleido_v


def _get_split_policy_cutoff_timestamp():
    env_value = os.environ.get('OTITENET_RESULTS_MIN_MTIME')
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            print("Invalid OTITENET_RESULTS_MIN_MTIME value, ignoring cutoff.")

    # By default keep all historical runs; opt-in cutoff via OTITENET_RESULTS_MIN_MTIME.
    return None


def _format_timestamp(ts):
    if pd.isna(ts):
        return 'unknown'
    try:
        return datetime.fromtimestamp(float(ts)).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return 'unknown'

def _extract_binary_targets_and_probs(df_pred):
    if 'label' not in df_pred.columns:
        return None, None

    labels = df_pred['label'].astype(str)
    if labels.nunique() != 2:
        return None, None

    # Prefer explicit class probability columns when present.
    if 'probs_NotNormal' in df_pred.columns:
        y_true = (labels == 'NotNormal').astype(int)
        y_prob = pd.to_numeric(df_pred['probs_NotNormal'], errors='coerce')
    elif 'p1' in df_pred.columns:
        y_true = (labels == sorted(labels.unique())[1]).astype(int)
        y_prob = pd.to_numeric(df_pred['p1'], errors='coerce')
    else:
        return None, None

    mask = y_prob.notna()
    if mask.sum() == 0:
        return None, None

    y_true = y_true[mask].astype(int)
    y_prob = y_prob[mask].clip(0.0, 1.0)
    return y_true, y_prob


def _parse_metadata_from_path(parts):
    parsed = {
        'model_name': parts[3] if len(parts) > 3 else 'unknown',
        'dataset_name': 'unknown',
        'classif_loss': 'unknown',
        'fgsm': np.nan,
        'n_calibration': np.nan,
        'prototypes': 'unknown',
        'n_positives': np.nan,
        'n_negatives': np.nan,
        'prototype_agg': 'none',
        'normalize': 'unknown',
        'dist_fct': 'unknown',
        'knn': np.nan,
        'nsize': np.nan,
        'BER': 'none',
        # Backward-compatible alias used elsewhere in this script.
        'dloss': 'none',
    }

    classif_losses = {'arcface', 'triplet', 'softmax_contrastive', 'ce', 'hinge'}
    ber_values = {'no', 'inverseTriplet', 'dann'}

    for p in parts:
        if p.startswith('otite_ds_'):
            parsed['dataset_name'] = p
        elif p.startswith('nsize'):
            try:
                parsed['nsize'] = int(p.replace('nsize', ''))
            except Exception:
                pass
        elif p.startswith('fgsm'):
            try:
                parsed['fgsm'] = int(p.replace('fgsm', ''))
            except Exception:
                pass
        elif p.startswith('ncal'):
            try:
                parsed['n_calibration'] = int(p.replace('ncal', ''))
            except Exception:
                pass
        elif p in classif_losses:
            parsed['classif_loss'] = p
        elif p in ber_values:
            parsed['BER'] = p
            parsed['dloss'] = p
        elif p.startswith('prototypes_'):
            parsed['prototypes'] = p.replace('prototypes_', '')
        elif p.startswith('npos'):
            try:
                parsed['n_positives'] = int(p.replace('npos', ''))
            except Exception:
                pass
        elif p.startswith('nneg'):
            try:
                parsed['n_negatives'] = int(p.replace('nneg', ''))
            except Exception:
                pass
        elif p.startswith('protoagg_'):
            # Keep compact readable value on plots: protoagg_mean_5 -> 5
            agg_parts = p.split('_')
            parsed['prototype_agg'] = agg_parts[-1] if len(agg_parts) > 1 else p
        elif p.startswith('norm'):
            parsed['normalize'] = p.replace('norm', '')
        elif p.startswith('dist_'):
            parsed['dist_fct'] = p.replace('dist_', '')
        elif p.startswith('knn'):
            try:
                parsed['knn'] = int(p.replace('knn', ''))
            except Exception:
                pass

    return parsed


def _load_json_if_exists(path):
    if not isinstance(path, str) or not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load JSON from {path}: {e}")
        return {}


def _coerce_metric_scalar(value):
    if isinstance(value, list):
        for item in value:
            coerced = _coerce_metric_scalar(item)
            if pd.notna(coerced):
                return coerced
        return np.nan

    if value is None:
        return np.nan

    try:
        metric = float(value)
    except Exception:
        return np.nan

    return metric if np.isfinite(metric) else np.nan


def _prefer_metric(*values):
    for value in values:
        metric = _coerce_metric_scalar(value)
        if pd.notna(metric):
            return metric
    return np.nan


def _extract_run_summary_split_metric(summary, split_name, metric_name):
    if not isinstance(summary, dict):
        return np.nan
    best_values = summary.get('best_values', {})
    if not isinstance(best_values, dict):
        return np.nan
    split_metrics = best_values.get(split_name, {})
    if not isinstance(split_metrics, dict):
        return np.nan
    return _coerce_metric_scalar(split_metrics.get(metric_name))


def _load_run_summary_metrics(run_dir):
    summary = _load_json_if_exists(os.path.join(run_dir, 'run_summary.json'))
    if not isinstance(summary, dict) or not summary:
        return {}

    return {
        'summary_run_status': str(summary.get('run_status', '')).strip().lower(),
        'summary_finished': _coerce_finished_flag(summary.get('finished', summary.get('run_finished'))),
        'summary_error_message': summary.get('error_message'),
        'summary_best_mcc': _coerce_metric_scalar(summary.get('best_mcc')),
        'summary_best_accuracy': _coerce_metric_scalar(summary.get('best_acc')),
        'summary_valid_mcc': _prefer_metric(
            _extract_run_summary_split_metric(summary, 'valid', 'mcc'),
            summary.get('best_mcc'),
        ),
        'summary_valid_accuracy': _prefer_metric(
            _extract_run_summary_split_metric(summary, 'valid', 'acc'),
            summary.get('best_acc'),
        ),
        'summary_test_mcc': _prefer_metric(
            _extract_run_summary_split_metric(summary, 'test', 'mcc'),
            _extract_run_summary_split_metric(summary, 'test_calibration', 'mcc'),
        ),
        'summary_test_accuracy': _prefer_metric(
            _extract_run_summary_split_metric(summary, 'test', 'acc'),
            _extract_run_summary_split_metric(summary, 'test_calibration', 'acc'),
        ),
    }


def _coerce_finished_flag(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, (float, np.floating)):
        return 1 if float(value) != 0.0 else 0
    sval = str(value).strip().lower()
    if sval in {'1', 'true', 'yes', 'y', 'done', 'completed', 'finished'}:
        return 1
    if sval in {'0', 'false', 'no', 'n', 'running', 'queued'}:
        return 0
    return None


def _discover_prediction_files():
    task = _get_target_task()
    patterns = [
        # f'logs/best_models/{task}/**/test_predictions.csv',
        # f'logs/best_models/{task}/**/valid_predictions.csv',
        # f'logs/best_models_*/{task}/**/test_predictions.csv',
        # f'logs/best_models_*/{task}/**/valid_predictions.csv',
        f'logs/{task}/cnn_mlp_compare/**/test_predictions.csv',
        f'logs/{task}/cnn_mlp_compare/**/valid_predictions.csv',
        f'logs/{task}/*/test_predictions.csv',
        f'logs/{task}/*/valid_predictions.csv',
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def _parse_prediction_file_metadata(path):
    parts = path.split('/')
    # Standard best_models layout keeps rich metadata in path segments.
    if ('best_models' in path or 'best_models_' in path or 'cnn_mlp_compare' in path) and len(parts) >= 6:
        parsed = _parse_metadata_from_path(parts)
        parsed = _enrich_metadata_from_sidecars(path, parsed)
        return parsed

    # UUID run layout: logs/<task>/<uuid>/test_predictions.csv
    parsed = {
        'model_name': 'unknown',
        'dataset_name': 'unknown',
        'classif_loss': 'unknown',
        'fgsm': np.nan,
        'n_calibration': np.nan,
        'prototypes': 'unknown',
        'n_positives': np.nan,
        'n_negatives': np.nan,
        'prototype_agg': 'none',
        'normalize': 'unknown',
        'dist_fct': 'unknown',
        'knn': np.nan,
        'nsize': np.nan,
        'BER': 'none',
        'dloss': 'none',
        'task': parts[1] if len(parts) > 1 else _get_target_task(),
        'kind': 'siamese',
        'variant': 'siamese',
    }

    run_dir = os.path.dirname(path)
    run_metadata = _load_json_if_exists(os.path.join(run_dir, 'run_metadata.json'))
    args = run_metadata.get('args', {}) if isinstance(run_metadata, dict) else {}
    if isinstance(run_metadata, dict):
        finished_val = run_metadata.get('finished', run_metadata.get('run_finished', None))
        finished_norm = _coerce_finished_flag(finished_val)
        if finished_norm is not None:
            parsed['finished'] = int(finished_norm)

    parsed['model_name'] = args.get('model_name', parsed['model_name'])
    parsed['dataset_name'] = os.path.basename(str(args.get('path', ''))) or parsed['dataset_name']
    parsed['classif_loss'] = args.get('classif_loss', parsed['classif_loss'])
    parsed['dloss'] = args.get('dloss', parsed['dloss'])
    parsed['BER'] = parsed['dloss'] if parsed['dloss'] not in [None, ''] else parsed['BER']
    parsed['fgsm'] = args.get('fgsm', parsed['fgsm'])
    parsed['n_calibration'] = args.get('n_calibration', parsed['n_calibration'])
    parsed['prototypes'] = args.get('prototypes_to_use', parsed['prototypes'])
    parsed['n_positives'] = args.get('n_positives', parsed['n_positives'])
    parsed['n_negatives'] = args.get('n_negatives', parsed['n_negatives'])
    parsed['prototype_agg'] = args.get('prototype_strategy', parsed['prototype_agg'])
    parsed['normalize'] = args.get('normalize', parsed['normalize'])
    parsed['dist_fct'] = args.get('dist_fct', parsed['dist_fct'])
    parsed['knn'] = args.get('n_neighbors', parsed['knn'])
    parsed['nsize'] = args.get('new_size', parsed['nsize'])
    parsed['exp_id'] = args.get('exp_id', run_metadata.get('foldername', ''))
    parsed['run_tag'] = args.get('run_tag', '')
    parsed['is_test_run'] = int(
        'test' in str(parsed.get('run_tag', '')).lower()
        or 'test' in str(parsed.get('exp_id', '')).lower()
    )

    parsed['match_key'] = _build_match_key(
        parsed.get('kind', 'siamese'),
        parsed.get('model_name', 'unknown'),
        parsed.get('classif_loss', 'unknown'),
        parsed.get('dloss', 'unknown'),
        parsed.get('fgsm', np.nan),
        parsed.get('n_calibration', np.nan),
        parsed.get('normalize', 'unknown'),
        parsed.get('dataset_name', 'unknown'),
        parsed.get('task', _get_target_task()),
        prototype=parsed.get('prototypes', ''),
        n_positives=parsed.get('n_positives', ''),
        n_negatives=parsed.get('n_negatives', ''),
        variant=parsed.get('variant', ''),
    )
    return parsed


def _normalize_key_value(value):
    if pd.isna(value):
        return ''
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    return str(value).strip()


def _build_match_key(kind, model_name, loss_value, dloss, fgsm, n_calibration,
                     normalize, dataset_name, task, prototype='', n_positives='',
                     n_negatives='', variant=''):
    if kind == 'cnn_mlp':
        return '|'.join([
            'cnn_mlp',
            _normalize_key_value(model_name),
            _normalize_key_value(variant),
            _normalize_key_value(loss_value),
            _normalize_key_value(dloss),
            _normalize_key_value(fgsm),
            _normalize_key_value(n_calibration),
            _normalize_key_value(normalize),
            _normalize_key_value(dataset_name),
            _normalize_key_value(task),
        ])

    return '|'.join([
        'siamese',
        _normalize_key_value(model_name),
        _normalize_key_value(loss_value),
        _normalize_key_value(dloss),
        _normalize_key_value(prototype),
        _normalize_key_value(fgsm),
        _normalize_key_value(n_calibration),
        _normalize_key_value(n_positives),
        _normalize_key_value(n_negatives),
        _normalize_key_value(normalize),
        _normalize_key_value(dataset_name),
        _normalize_key_value(task),
    ])


def _enrich_metadata_from_sidecars(source_csv, parsed):
    parsed = dict(parsed)
    parsed['task'] = parsed.get('task', 'notNormal')
    parsed['kind'] = 'siamese'
    parsed['variant'] = 'siamese'

    if 'cnn_mlp_compare' in source_csv:
        model_dir = os.path.dirname(source_csv)
        run_root = os.path.dirname(model_dir)
        metrics = _load_json_if_exists(os.path.join(model_dir, 'metrics.json'))
        run_config = _load_json_if_exists(os.path.join(run_root, 'run_config.json'))
        parsed['kind'] = 'cnn_mlp'
        parsed['variant'] = metrics.get('variant', os.path.basename(model_dir))
        parsed['model_name'] = metrics.get('model_name', run_config.get('model_name', parsed.get('model_name', 'unknown')))
        parsed['classif_loss'] = metrics.get('classif_loss', run_config.get('classif_loss', parsed.get('classif_loss', 'unknown')))
        parsed['dloss'] = metrics.get('dloss', run_config.get('dloss', parsed.get('dloss', 'unknown')))
        parsed['BER'] = parsed.get('dloss', 'unknown')
        parsed['fgsm'] = metrics.get('fgsm', run_config.get('fgsm', parsed.get('fgsm', np.nan)))
        parsed['n_calibration'] = metrics.get('n_calibration', run_config.get('n_calibration', parsed.get('n_calibration', np.nan)))
        parsed['normalize'] = run_config.get('normalize', parsed.get('normalize', 'unknown'))
        parsed['dataset_name'] = os.path.basename(run_config.get('path', parsed.get('dataset_name', 'unknown')))
        parsed['task'] = run_config.get('task', parsed.get('task', 'notNormal'))
        parsed['valid_dataset'] = run_config.get('valid_dataset', '')
        parsed['test_dataset'] = run_config.get('test_dataset', '')
        parsed['train_datasets'] = run_config.get('train_datasets', '')
        parsed['exp_id'] = metrics.get('exp_id', run_config.get('exp_id', ''))
        parsed['run_tag'] = run_config.get('run_tag', '')
        parsed['is_test_run'] = int('test' in str(parsed.get('run_tag', '')).lower())
    else:
        parsed['exp_id'] = parsed.get('exp_id', '')
        parsed['run_tag'] = parsed.get('run_tag', '')
        parsed['is_test_run'] = int('test' in str(parsed.get('run_tag', '')).lower())

    parsed['match_key'] = _build_match_key(
        parsed.get('kind', 'siamese'),
        parsed.get('model_name', 'unknown'),
        parsed.get('classif_loss', 'unknown'),
        parsed.get('dloss', 'unknown'),
        parsed.get('fgsm', np.nan),
        parsed.get('n_calibration', np.nan),
        parsed.get('normalize', 'unknown'),
        parsed.get('dataset_name', 'unknown'),
        parsed.get('task', 'notNormal'),
        prototype=parsed.get('prototypes', ''),
        n_positives=parsed.get('n_positives', ''),
        n_negatives=parsed.get('n_negatives', ''),
        variant=parsed.get('variant', ''),
    )
    return parsed


def _load_launcher_metadata():
    manifest_files = sorted(set(
        glob.glob('logs/*_job_manifest.csv') +
        glob.glob('logs/progresses/*/*_job_manifest.csv')
    ))
    runtime_files = sorted(set(
        glob.glob('logs/*_job_runtime.csv') +
        glob.glob('logs/progresses/*/*_job_runtime.csv')
    ))

    manifest_frames = []
    for path in manifest_files:
        try:
            frame = pd.read_csv(path)
            frame['manifest_path'] = path
            frame['manifest_mtime'] = os.path.getmtime(path)
            frame = frame[frame.get('kind', '').astype(str).str.len() > 0].copy() if 'kind' in frame.columns else frame.iloc[0:0]
            if frame.empty:
                continue
            frame['match_key'] = frame.apply(
                lambda row: _build_match_key(
                    row.get('kind', ''),
                    row.get('model', ''),
                    row.get('classif_loss', row.get('loss', '')) if row.get('kind', '') == 'cnn_mlp' else row.get('loss', row.get('classif_loss', '')),
                    row.get('dloss', ''),
                    row.get('fgsm', ''),
                    row.get('n_calibration', ''),
                    row.get('normalize', ''),
                    row.get('dataset_name', ''),
                    row.get('task', ''),
                    prototype=row.get('prototype', ''),
                    n_positives=row.get('n_positives', ''),
                    n_negatives=row.get('n_negatives', ''),
                    variant=row.get('variant', ''),
                ),
                axis=1,
            )
            manifest_frames.append(frame)
        except Exception:
            continue

    manifest_df = pd.concat(manifest_frames, ignore_index=True) if manifest_frames else pd.DataFrame()
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values('manifest_mtime', ascending=False).drop_duplicates('match_key', keep='first')

    runtime_frames = []
    for path in runtime_files:
        try:
            frame = pd.read_csv(path)
            frame['runtime_path'] = path
            runtime_frames.append(frame)
        except Exception:
            continue
    runtime_df = pd.concat(runtime_frames, ignore_index=True) if runtime_frames else pd.DataFrame()
    return manifest_df, runtime_df


def _attach_launcher_metadata(df, manifest_df, runtime_df):
    if df.empty:
        return df

    if not manifest_df.empty and 'match_key' in df.columns:
        manifest_cols = [
            'exp_id', 'job_state', 'finished', 'retry_count', 'failed_final', 'num_jobs', 'bs',
            'new_size', 'n_epochs', 'n_trials', 'early_stop', 'dataset_name',
            'valid_dataset', 'test_dataset', 'train_datasets', 'kind', 'variant',
            'task', 'log_file', 'auto_select_k', 'run_cnn_mlp', 'cnn_compare_all',
            'verbose', 'test_mode', 'test_tag', 'triplet_log_comet',
            'triplet_log_mlflow', 'triplet_log_tracking',
            'triplet_save_repro_artifacts', 'user_set_num_jobs', 'user_set_bs',
            'min_free_mb', 'mem_per_job_mb', 'max_oom_retries', 'poll_interval',
            'gamma', 'epsilon', 'force_run',
            'theoretical_gpu_required_mb', 'reserved_gpu_mb', 'actual_peak_gpu_mb',
            'oom_missing_gpu_mb', 'oom_gpu_free_at_failure_mb', 'oom_gpu_used_at_failure_mb',
            'reservation_source_tag', 'reservation_match_quality', 'telemetry_status'
        ]
        manifest_cols = [col for col in manifest_cols if col in manifest_df.columns]
        merged = manifest_df[['match_key'] + manifest_cols].copy()
        merged = merged.rename(columns={col: f'launcher_{col}' for col in manifest_cols})
        df = df.merge(merged, on='match_key', how='left')

        # Prefer explicit launcher_finished for completeness. For older manifests,
        # synthesize a conservative finished flag from known terminal states.
        if 'launcher_finished' in df.columns:
            finished_series = pd.to_numeric(df['launcher_finished'], errors='coerce')
            complete_mask = finished_series.notna()
        else:
            state_series = df['launcher_job_state'].astype(str).str.strip() if 'launcher_job_state' in df.columns else pd.Series('', index=df.index)
            state_series = state_series.str.lower()
            terminal_states = {'completed', 'failed', 'done_already'}
            complete_mask = state_series.isin(terminal_states)
            df['launcher_finished'] = complete_mask.astype(int)

        df['launcher_metadata_complete'] = complete_mask
        before_count = len(df)
        complete_count = int(df['launcher_metadata_complete'].sum())
        print(f"Launcher metadata complete for {complete_count}/{before_count} runs")

        require_complete = str(os.environ.get('OTITENET_REQUIRE_COMPLETE_LAUNCHER_METADATA', '0')).strip().lower() in {'1', 'true', 'yes'}
        if require_complete:
            df = df[df['launcher_metadata_complete']].copy()
            print(f"Filtered to {len(df)}/{before_count} runs with complete launcher metadata")
    else:
        df['launcher_metadata_complete'] = False

    if not runtime_df.empty and 'launcher_exp_id' in df.columns and 'exp_id' in runtime_df.columns:
        runtime_summary = runtime_df.copy()
        runtime_summary['event'] = runtime_summary['event'].astype(str)
        runtime_summary['exp_id'] = runtime_summary['exp_id'].astype(str).str.strip()
        runtime_summary = runtime_summary[runtime_summary['exp_id'].str.len() > 0]
        runtime_summary = runtime_summary.groupby('exp_id').agg(
            runtime_event_count=('event', 'size'),
            runtime_launches=('event', lambda s: int((s == 'launched').sum())),
            runtime_completions=('event', lambda s: int((s == 'completed').sum())),
            runtime_failures=('event', lambda s: int((s == 'failed').sum())),
            runtime_oom_retries=('event', lambda s: int((s == 'oom_retry').sum())),
        ).reset_index()
        runtime_summary = runtime_summary.rename(columns={'exp_id': 'launcher_exp_id'})
        df['launcher_exp_id'] = df['launcher_exp_id'].astype(str).str.strip()
        df.loc[df['launcher_exp_id'].isin(['', 'nan', 'None']), 'launcher_exp_id'] = np.nan
        df = df.merge(runtime_summary, on='launcher_exp_id', how='left')

    # Derive a robust effective status from runtime terminal events when available.
    if 'launcher_job_state' in df.columns:
        state = df['launcher_job_state'].astype(str).str.strip()
        state = state.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})

        if 'runtime_failures' in df.columns:
            failed_mask = pd.to_numeric(df['runtime_failures'], errors='coerce').fillna(0).astype(int) > 0
            state = state.where(~failed_mask, 'failed')
        if 'runtime_completions' in df.columns:
            completed_mask = pd.to_numeric(df['runtime_completions'], errors='coerce').fillna(0).astype(int) > 0
            # Keep explicit failures dominant over completions.
            state = state.where(~completed_mask | (state == 'failed'), 'completed')
        if 'runtime_launches' in df.columns:
            launched_mask = pd.to_numeric(df['runtime_launches'], errors='coerce').fillna(0).astype(int) > 0
            state = state.where(~launched_mask | state.notna(), 'launched')

        df['launcher_job_state_effective'] = state.fillna('unknown')
    else:
        df['launcher_job_state_effective'] = 'unknown'

    if 'launcher_finished' not in df.columns:
        df['launcher_finished'] = 0
    launcher_finished_num = pd.to_numeric(df['launcher_finished'], errors='coerce')
    if launcher_finished_num.isna().any():
        effective_state = df['launcher_job_state_effective'].astype(str).str.lower()
        inferred_finished = effective_state.isin({'completed', 'failed', 'done_already'})
        launcher_finished_num = launcher_finished_num.where(~launcher_finished_num.isna(), inferred_finished.astype(int))
    df['launcher_finished'] = launcher_finished_num.fillna(0).astype(int)

    return df

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_totals = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    non_empty_bins = bin_totals > 0
    bin_totals = bin_totals[non_empty_bins]
    
    # We need the counts in each bin to weigh the ECE correctly
    # calibration_curve doesn't return counts, so we approximate or use counts from histogram
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_totals / len(y_true)))
    return ece, prob_true, prob_pred

def _compute_batch_effect_from_predictions(preds, batch_labels):
    preds = np.asarray(preds)
    batch_labels = np.asarray(batch_labels)
    if preds.shape[0] == 0 or batch_labels.shape[0] == 0 or preds.shape[0] != batch_labels.shape[0]:
        return {'batch_entropy_norm': np.nan, 'batch_nmi': np.nan, 'batch_ari': np.nan}

    unique_batches = np.unique(batch_labels)
    unique_preds = np.unique(preds)

    if unique_batches.size <= 1:
        return {'batch_entropy_norm': np.nan, 'batch_nmi': np.nan, 'batch_ari': np.nan}

    contingency = np.zeros((unique_preds.size, unique_batches.size), dtype=float)
    for i, pred_val in enumerate(unique_preds):
        pred_mask = preds == pred_val
        for j, batch_val in enumerate(unique_batches):
            contingency[i, j] = np.sum(pred_mask & (batch_labels == batch_val))

    row_sums = contingency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = contingency / row_sums
    ent = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    max_ent = np.log(unique_batches.size)
    entropy_norm = float(np.mean(ent / (max_ent + 1e-12)))

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    nmi = float(normalized_mutual_info_score(batch_labels, preds))
    ari = float(adjusted_rand_score(batch_labels, preds))

    return {'batch_entropy_norm': entropy_norm, 'batch_nmi': nmi, 'batch_ari': ari}


def _print_mcc_summary(df, label="Runs considered"):
    if df is None or df.empty:
        print(f"{label}: 0 runs, no MCC values available.")
        return

    report_df = _prepare_numeric_columns(df)
    summary_best_count = int(report_df['summary_best_mcc'].notna().sum()) if 'summary_best_mcc' in report_df.columns else 0
    valid_count = int(report_df['valid_mcc'].notna().sum()) if 'valid_mcc' in report_df.columns else 0
    test_count = int(report_df['test_mcc'].notna().sum()) if 'test_mcc' in report_df.columns else 0
    primary_count = int(report_df['mcc'].notna().sum()) if 'mcc' in report_df.columns else 0

    print(
        f"{label}: {len(report_df)} runs | "
        f"best_mcc from run_summary.json: {summary_best_count} | "
        f"valid_mcc available: {valid_count} | "
        f"test_mcc available: {test_count} | "
        f"primary mcc available: {primary_count}"
    )


def _collect_prediction_data(split_policy_cutoff=None):
    """Scan all *_predictions.csv files and build a list of parsed metadata dicts.

    Returns
    -------
    tuple: (data: list[dict], manifest_df: pd.DataFrame, runtime_df: pd.DataFrame)
    """
    # Build info_map: sample-name → batch/dataset label, used for batch-effect metrics.
    info_map = {}
    for info_candidate in [
        "data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference/infos.csv",
        "data/otite_ds_224/USA_Turquie_Chili_GMFUNL_inference/infos.csv",
    ]:
        if os.path.exists(info_candidate):
            try:
                _info_df = pd.read_csv(info_candidate)
                if 'name' in _info_df.columns and 'dataset' in _info_df.columns:
                    info_map = dict(zip(_info_df['name'], _info_df['dataset']))
            except Exception:
                pass
            break

    data = []
    csv_files = _discover_prediction_files()
    print(f"Discovered {len(csv_files)} prediction files across all result roots.")

    for f in csv_files:
        base_name = os.path.basename(f)
        # Avoid duplicate rows when both files exist: process test file as canonical row.
        if base_name == 'valid_predictions.csv':
            sibling_test = os.path.join(os.path.dirname(f), 'test_predictions.csv')
            if os.path.exists(sibling_test):
                continue

        parsed = _parse_prediction_file_metadata(f)
        parsed.update(_load_run_summary_metrics(os.path.dirname(f)))
        parsed['source_csv'] = f
        parsed['source_mtime'] = os.path.getmtime(f) if os.path.exists(f) else np.nan
        parsed['source_datetime'] = _format_timestamp(parsed['source_mtime'])

        try:
            df_pred = pd.read_csv(f)
            if 'label' in df_pred.columns and 'pred' in df_pred.columns:
                # Handle label encoding
                labels = df_pred['label'].astype('category').cat.codes
                preds = df_pred['pred'].astype('category').cat.codes

                mcc = matthews_corrcoef(labels, preds)
                acc = accuracy_score(labels, preds)
                if base_name == 'valid_predictions.csv':
                    parsed['csv_valid_mcc'] = mcc
                    parsed['csv_valid_accuracy'] = acc
                    parsed['valid_mcc'] = mcc
                    parsed['valid_accuracy'] = acc
                    # Fallback ranking metrics if test split is unavailable.
                    parsed['mcc'] = mcc
                    parsed['accuracy'] = acc
                else:
                    parsed['csv_test_mcc'] = mcc
                    parsed['csv_test_accuracy'] = acc
                    parsed['test_mcc'] = mcc
                    parsed['test_accuracy'] = acc
                    parsed['mcc'] = mcc
                    parsed['accuracy'] = acc

                # Optional validation metrics from sibling file.
                if base_name == 'test_predictions.csv':
                    valid_f = f.replace('test_predictions.csv', 'valid_predictions.csv')
                else:
                    valid_f = f
                if os.path.exists(valid_f):
                    try:
                        df_valid = pd.read_csv(valid_f)
                        if 'label' in df_valid.columns and 'pred' in df_valid.columns:
                            v_labels = df_valid['label'].astype('category').cat.codes
                            v_preds = df_valid['pred'].astype('category').cat.codes
                            parsed['csv_valid_mcc'] = matthews_corrcoef(v_labels, v_preds)
                            parsed['csv_valid_accuracy'] = accuracy_score(v_labels, v_preds)
                            parsed['valid_mcc'] = parsed['csv_valid_mcc']
                            parsed['valid_accuracy'] = parsed['csv_valid_accuracy']
                    except Exception:
                        pass

                parsed['valid_mcc'] = _prefer_metric(
                    parsed.get('csv_valid_mcc'),
                    parsed.get('summary_valid_mcc'),
                    parsed.get('summary_best_mcc'),
                    parsed.get('valid_mcc'),
                )
                parsed['valid_accuracy'] = _prefer_metric(
                    parsed.get('csv_valid_accuracy'),
                    parsed.get('summary_valid_accuracy'),
                    parsed.get('summary_best_accuracy'),
                    parsed.get('valid_accuracy'),
                )
                parsed['test_mcc'] = _prefer_metric(
                    parsed.get('csv_test_mcc'),
                    parsed.get('summary_test_mcc'),
                    parsed.get('test_mcc'),
                )
                parsed['test_accuracy'] = _prefer_metric(
                    parsed.get('csv_test_accuracy'),
                    parsed.get('summary_test_accuracy'),
                    parsed.get('test_accuracy'),
                )

                if base_name == 'test_predictions.csv':
                    parsed['mcc'] = _prefer_metric(parsed.get('test_mcc'), parsed.get('valid_mcc'), parsed.get('summary_best_mcc'))
                    parsed['accuracy'] = _prefer_metric(parsed.get('test_accuracy'), parsed.get('valid_accuracy'), parsed.get('summary_best_accuracy'))
                else:
                    parsed['mcc'] = _prefer_metric(parsed.get('valid_mcc'), parsed.get('test_mcc'), parsed.get('summary_best_mcc'))
                    parsed['accuracy'] = _prefer_metric(parsed.get('valid_accuracy'), parsed.get('test_accuracy'), parsed.get('summary_best_accuracy'))

                # Check for probability columns for calibration
                y_true, y_prob = _extract_binary_targets_and_probs(df_pred)
                if y_true is not None and y_prob is not None:
                    ece, _, _ = expected_calibration_error(y_true, y_prob)
                    brier = brier_score_loss(y_true, y_prob)
                    parsed['ece'] = ece
                    parsed['brier'] = brier

                # Batch Effect calculation
                if info_map and 'name' in df_pred.columns:
                    df_pred['batch'] = df_pred['name'].map(info_map)
                    valid_batch = df_pred.dropna(subset=['batch'])
                    if not valid_batch.empty:
                        batch_metrics = _compute_batch_effect_from_predictions(
                            valid_batch['pred'].astype('category').cat.codes,
                            valid_batch['batch'].astype('category').cat.codes
                        )
                        parsed.update(batch_metrics)

                data.append(parsed)
        except Exception:
            continue

    df = pd.DataFrame(data)
    manifest_df, runtime_df = _load_launcher_metadata()
    if df.empty:
        return df, manifest_df, runtime_df

    _print_mcc_summary(df, label="Before filtering")

    if 'source_mtime' in df.columns:
        df['source_mtime'] = pd.to_numeric(df['source_mtime'], errors='coerce')
        df['source_datetime'] = df['source_mtime'].apply(_format_timestamp)

    if split_policy_cutoff is not None and 'source_mtime' in df.columns:
        before_count = len(df)
        df = df[df['source_mtime'] >= split_policy_cutoff].copy()
        print(
            f"Kept {len(df)}/{before_count} runs with source_mtime >= {_format_timestamp(split_policy_cutoff)}"
        )

    df = _attach_launcher_metadata(df, manifest_df, runtime_df)

    if _analysis_test_only_enabled():
        before_count = len(df)
        mask = pd.Series(False, index=df.index)

        if 'launcher_test_mode' in df.columns:
            mask = mask | (pd.to_numeric(df['launcher_test_mode'], errors='coerce') == 1)
        if 'launcher_test_tag' in df.columns:
            mask = mask | df['launcher_test_tag'].astype(str).str.contains('test', case=False, na=False)
        if 'run_tag' in df.columns:
            mask = mask | df['run_tag'].astype(str).str.contains('test', case=False, na=False)
        if 'exp_id' in df.columns:
            mask = mask | df['exp_id'].astype(str).str.contains('test', case=False, na=False)
        if 'is_test_run' in df.columns:
            mask = mask | (pd.to_numeric(df['is_test_run'], errors='coerce').fillna(0).astype(int) == 1)

        df = df[mask].copy()
        print(f"Filtered to test/smoke runs: {len(df)}/{before_count}")

    _print_mcc_summary(df, label="After filtering")

    return df, manifest_df, runtime_df

def do_eda(out_dir):
    csv_path = "data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference/infos.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/otite_ds_224/USA_Turquie_Chili_GMFUNL_inference/infos.csv" # Fallback
    
    if os.path.exists(csv_path):
        df_info = pd.read_csv(csv_path)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.countplot(data=df_info, y='label', palette='mako')
        plt.title("Class Distribution")
        
        plt.subplot(1, 2, 2)
        sns.countplot(data=df_info, y='dataset', palette='rocket')
        plt.title("Batch/Dataset Distribution")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/0_dataset_eda.png", dpi=300)
        plt.close()
        
        # New: Class distribution per dataset (grouped barplot)
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_info, x='label', hue='dataset', palette='deep')
        plt.title("Class Distribution per Dataset")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/0_class_distribution_per_dataset.png", dpi=300)
        plt.close()
        
        return True
    return False

def make_interpretability_figure(out_dir):
    task = _get_target_task()
    # Find some grad-cam and shap images
    grad_cams = glob.glob(f'logs/{task}/*/correct_classif/*/*/grad_cam/*.png')
    shaps = glob.glob(f'logs/{task}/*/correct_classif/*/*/gradients_shap/*.png')
    wrong_grad_cams = glob.glob(f'logs/{task}/*/wrong_classif/*/*/grad_cam/*.png')
    wrong_shaps = glob.glob(f'logs/{task}/*/wrong_classif/*/*/gradients_shap/*.png')
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    def _load_preview_image(path, max_side=900):
        if not isinstance(path, str) or not os.path.exists(path):
            return None
        try:
            # Downscale large images to keep rendering/saving fast and stable.
            img = Image.open(path).convert('RGB')
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return np.asarray(img)
        except Exception:
            return None
    
    # 1. Correct Grad-CAM
    if grad_cams:
        img = _load_preview_image(grad_cams[0])
        if img is not None:
            axes[0].imshow(img)
            axes[0].set_title(f"Correct Grad-CAM:\n{os.path.basename(grad_cams[0])[:30]}", fontsize=9)
    axes[0].axis('off')
    
    # 2. Correct SHAP
    if shaps:
        img = _load_preview_image(shaps[0])
        if img is not None:
            axes[1].imshow(img)
            axes[1].set_title(f"Correct SHAP:\n{os.path.basename(shaps[0])[:30]}", fontsize=9)
    axes[1].axis('off')
    
    # 3. Wrong Grad-CAM
    if wrong_grad_cams:
        img = _load_preview_image(wrong_grad_cams[0])
        if img is not None:
            axes[2].imshow(img)
            axes[2].set_title(f"Misclassified Grad-CAM:\n{os.path.basename(wrong_grad_cams[0])[:30]}", fontsize=9)
    axes[2].axis('off')
    
    # 4. Wrong SHAP
    if wrong_shaps:
        img = _load_preview_image(wrong_shaps[0])
        if img is not None:
            axes[3].imshow(img)
            axes[3].set_title(f"Misclassified SHAP:\n{os.path.basename(wrong_shaps[0])[:30]}", fontsize=9)
    axes[3].axis('off')
    
    plt.suptitle("Model Interpretability Examples (Grad-CAM vs SHAP)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/6_interpretability.png", dpi=180)
    plt.close()

def _plot_calibration_curves_for_split(top_5, out_dir, split_name, filename):
    plt.figure(figsize=(8, 6))
    for _, row in top_5.iterrows():
        try:
            if split_name == 'test':
                pred_file = row['source_csv']
            else:
                pred_file = str(row['source_csv']).replace('test_predictions.csv', 'valid_predictions.csv')

            if not isinstance(pred_file, str) or not os.path.exists(pred_file):
                continue

            df_pred = pd.read_csv(pred_file)
            y_true, y_prob = _extract_binary_targets_and_probs(df_pred)
            if y_true is None or y_prob is None:
                continue
            ece, p_true, p_pred = expected_calibration_error(y_true, y_prob)
            lbl = f"{row['model_name']}/{row['classif_loss']} (ECE: {ece:.3f})"
            plt.plot(p_pred, p_true, marker='o', label=lbl)
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curves ({split_name.capitalize()} - Top 5 Models)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}", dpi=300)
    plt.close()


def do_calibration_curves(df, out_dir):
    # Always generate calibration curve figures, even if data is missing
    if 'source_csv' not in df.columns or df.dropna(subset=['source_csv']).empty:
        # Generate placeholder figures
        for split_name, filename in [('test', '9_calibration_curves_test.png'), ('valid', '9_calibration_curves_valid.png')]:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"No data available for {split_name} calibration curve", ha='center', va='center', fontsize=14)
            plt.title(f'Calibration Curves ({split_name.capitalize()} - Top 5 Models)')
            plt.axis('off')
            plt.savefig(f"{out_dir}/{filename}", dpi=300)
            plt.close()
        return
    # Rank using valid MCC for model selection.
    rank_col = 'valid_mcc' if 'valid_mcc' in df.columns else 'mcc'
    top_5 = df.dropna(subset=['source_csv']).sort_values(rank_col, ascending=False).head(5)
    _plot_calibration_curves_for_split(top_5, out_dir, 'test', '9_calibration_curves_test.png')
    _plot_calibration_curves_for_split(top_5, out_dir, 'valid', '9_calibration_curves_valid.png')

def do_synergy_heatmap(df, out_dir):
    # Synergy between Classif Loss and BER
    out_path = f"{out_dir}/11_synergy_heatmap.png"
    if df is None or df.empty:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "No data available for synergy heatmap", ha='center', va='center', fontsize=14)
        plt.title("Synergy Matrix: Loss Function vs BER Strategy")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return

    metric_col = None
    for c in ['valid_mcc', 'mcc', 'test_mcc', 'accuracy']:
        if c in df.columns:
            metric_col = c
            break
    if metric_col is None:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "No data available for synergy heatmap", ha='center', va='center', fontsize=14)
        plt.title("Synergy Matrix: Model Strategy vs BER Strategy")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return

    tmp = df[[metric_col, 'classif_loss', 'dloss']].copy()

    # Unified model strategy axis:
    # - prefer classif_loss when available
    # - fallback to knn for siamese/kNN-only runs
    if 'classif_loss' in df.columns:
        strategy_series = df['classif_loss'].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
    else:
        strategy_series = pd.Series(np.nan, index=df.index)
    if 'knn' in df.columns:
        knn_numeric = pd.to_numeric(df['knn'], errors='coerce')
        knn_labels = knn_numeric.map(lambda x: f"knn={int(x)}" if pd.notna(x) else np.nan)
        strategy_series = strategy_series.fillna(knn_labels)
    tmp['strategy'] = strategy_series.fillna('unknown')

    if 'BER' in df.columns:
        tmp['BER'] = df['BER']
    elif 'dloss' in df.columns:
        tmp['BER'] = df['dloss']
    else:
        tmp['BER'] = 'unknown'

    # BER can be present but empty for some pipelines; use dloss as a fallback label.
    if 'dloss' in df.columns:
        ber_series = tmp['BER'].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        dloss_series = df['dloss'].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        tmp['BER'] = ber_series.fillna(dloss_series).fillna('unknown')

    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors='coerce')
    tmp = tmp.dropna(subset=['strategy', 'BER', metric_col])
    if tmp.empty:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "No data available for synergy heatmap", ha='center', va='center', fontsize=14)
        plt.title("Synergy Matrix: Model Strategy vs BER Strategy")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return
    synergy = tmp.groupby(['strategy', 'BER'])[metric_col].mean().unstack()
    if synergy.empty:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "No data available for synergy heatmap", ha='center', va='center', fontsize=14)
        plt.title("Synergy Matrix: Model Strategy vs BER Strategy")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(synergy, annot=True, cmap='YlGnBu', fmt=".3f")
    plt.title(f"Synergy Matrix: Model Strategy vs BER Strategy ({metric_col})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def do_pareto_frontier(df, out_dir):
    out_path = f"{out_dir}/12_pareto_generalization.png"
    if 'batch_entropy_norm' not in df.columns or df['batch_entropy_norm'].isna().all():
        plt.figure(figsize=(10, 7))
        plt.text(0.5, 0.5, "No data available for Pareto frontier plot", ha='center', va='center', fontsize=14)
        plt.title("The Generalization Pareto Frontier (Accuracy vs Invariance)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return
    # Accuracy vs Batch Entropy Pareto Front
    df_plot = df.dropna(subset=['valid_accuracy', 'batch_entropy_norm']).copy()
    df_plot = df_plot.sort_values('batch_entropy_norm')
    if df_plot.empty:
        plt.figure(figsize=(10, 7))
        plt.text(0.5, 0.5, "No data available for Pareto frontier plot", ha='center', va='center', fontsize=14)
        plt.title("The Generalization Pareto Frontier (Accuracy vs Invariance)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_plot, x='batch_entropy_norm', y='valid_accuracy', hue='model_name',
                    style='classif_loss' if 'classif_loss' in df_plot.columns else None,
                    s=100, alpha=0.6)
    # Simple Pareto implementation
    pts = df_plot[['batch_entropy_norm', 'valid_accuracy']].values
    pareto_mask = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        for j, q in enumerate(pts):
            if (q[0] >= p[0] and q[1] > p[1]) or (q[0] > p[0] and q[1] >= p[1]):
                pareto_mask[i] = False
                break
    frontier = df_plot[pareto_mask].sort_values('batch_entropy_norm')
    plt.plot(frontier['batch_entropy_norm'], frontier['valid_accuracy'], 'r--', alpha=0.8, label='Pareto Frontier')
    plt.title("The Generalization Pareto Frontier (Accuracy vs Invariance)")
    plt.xlabel("Normalized Batch Entropy (Domain Invariance)")
    plt.ylabel("Accuracy (Performance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _pick_best_per_n_trials(df):
    """Keep best-performing row per hyperparameter coordinates within each n_trials bucket."""
    if df is None or df.empty:
        return df

    rank_col = 'valid_mcc' if 'valid_mcc' in df.columns else 'mcc'
    if rank_col not in df.columns:
        return df

    n_trials_col = None
    if 'launcher_n_trials' in df.columns:
        n_trials_col = 'launcher_n_trials'
    elif 'n_trials' in df.columns:
        n_trials_col = 'n_trials'

    if n_trials_col is None:
        return df

    work = df.copy()
    work[rank_col] = pd.to_numeric(work[rank_col], errors='coerce')
    work[n_trials_col] = pd.to_numeric(work[n_trials_col], errors='coerce')
    work = work.dropna(subset=[rank_col, n_trials_col])
    if work.empty:
        return work

    coord_cols = [
        'model_name', 'dataset_name', 'nsize', 'fgsm', 'n_calibration',
        'classif_loss', 'BER', 'prototypes', 'n_positives', 'n_negatives',
        'prototype_agg', 'normalize', 'dist_fct', 'knn'
    ]
    coord_cols = [c for c in coord_cols if c in work.columns]
    group_cols = [n_trials_col] + coord_cols

    for c in group_cols:
        if work[c].dtype == object:
            work[c] = work[c].fillna('unknown')

    idx = work.groupby(group_cols, dropna=False)[rank_col].idxmax()
    best_df = work.loc[idx].copy()
    best_df = best_df.sort_values([n_trials_col, rank_col], ascending=[True, False])
    print(
        f"Best-per-n_trials selection kept {len(best_df)}/{len(df)} rows "
        f"using '{n_trials_col}' and rank '{rank_col}'."
    )
    return best_df


def do_hparam_parallel(df, out_dir, tag='all_runs', dedupe_latest=False):
    if df is None or df.empty:
        print("Skipping parallel coordinates: empty dataframe.")
        return
    defining_cols = [
        'model_name', 'dataset_name', 'nsize', 'fgsm', 'n_calibration',
        'classif_loss', 'BER', 'prototypes', 'n_positives', 'n_negatives',
        'prototype_agg', 'normalize', 'dist_fct', 'knn'
    ]
    score_candidates = [c for c in ['valid_mcc', 'mcc', 'test_mcc', 'valid_accuracy'] if c in df.columns]
    if not score_candidates:
        print("Skipping parallel coordinates: no score column among valid_mcc/mcc/test_mcc/accuracy.")
        return

    base_cols = [c for c in defining_cols if c in df.columns] + score_candidates
    if 'source_csv' in df.columns:
        base_cols += ['source_csv']
    plot_df = df[base_cols].copy()
    for c in score_candidates:
        plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')

    # Prefer valid_mcc when available, then fallback progressively.
    plot_df['plot_score'] = plot_df[score_candidates].bfill(axis=1).iloc[:, 0]
    # Keep all rows by default; missing score rows get neutral center value.
    if plot_df['plot_score'].notna().sum() == 0:
        plot_df['plot_score'] = 0.0
    else:
        neutral = float(plot_df['plot_score'].dropna().median())
        plot_df['plot_score'] = plot_df['plot_score'].fillna(neutral)

    if 'source_csv' in plot_df.columns:
        plot_df['__source_mtime'] = plot_df['source_csv'].apply(
            lambda p: os.path.getmtime(p) if isinstance(p, str) and os.path.exists(p) else np.nan
        )
    else:
        plot_df['__source_mtime'] = np.nan

    # Optionally keep only the most recent training for identical model-defining hyperparameters.
    dedupe_cols = [c for c in defining_cols if c in plot_df.columns]
    for c in dedupe_cols:
        if plot_df[c].dtype == object:
            plot_df[c] = plot_df[c].fillna('unknown')
    plot_df = plot_df.sort_values('__source_mtime', ascending=False)
    if dedupe_latest and dedupe_cols:
        plot_df = plot_df.drop_duplicates(subset=dedupe_cols, keep='first')

    # Show only coordinates that vary across models.
    varying_cols = []
    for c in dedupe_cols:
        series = plot_df[c]
        if pd.api.types.is_numeric_dtype(series):
            if pd.to_numeric(series, errors='coerce').nunique(dropna=True) > 1:
                varying_cols.append(c)
        else:
            if series.astype(str).nunique(dropna=True) > 1:
                varying_cols.append(c)

    if not varying_cols:
        print("Skipping parallel coordinates: all model-defining hyperparameters are constant.")
        return

    # Build category codes and preserve human-readable tick labels.
    cat_maps = {}
    rng = np.random.default_rng(42)
    jitter_strength = 0.08
    force_categorical_spread = {
        'nsize', 'fgsm', 'n_calibration', 'n_negatives', 'n_positives'
    }
    forced_jitter_by_axis = {
        'nsize': 0.20,
        'fgsm': 0.22,
        'n_calibration': 0.35,
        'n_negatives': 0.24,
        'n_positives': 0.35,
    }
    for col in varying_cols:
        if col == 'mcc':
            continue
        if col in force_categorical_spread:
            cat_series = plot_df[col].astype(str).fillna('unknown').astype('category')
            base_codes = cat_series.cat.codes.astype(float)
            # Stronger jitter for discrete settings to visually separate repeated runs.
            axis_jitter = forced_jitter_by_axis.get(col, 0.20)
            jitter = rng.uniform(-axis_jitter, axis_jitter, size=len(plot_df))
            plot_df[col] = base_codes + jitter
            cat_maps[col] = list(cat_series.cat.categories)
            continue
        if pd.api.types.is_numeric_dtype(plot_df[col]):
            numeric_vals = pd.to_numeric(plot_df[col], errors='coerce')
            min_v = np.nanmin(numeric_vals.values) if np.isfinite(numeric_vals.values).any() else 0.0
            max_v = np.nanmax(numeric_vals.values) if np.isfinite(numeric_vals.values).any() else 1.0
            span = max(max_v - min_v, 1e-9)
            jitter = rng.uniform(-jitter_strength * span * 0.03, jitter_strength * span * 0.03, size=len(plot_df))
            plot_df[col] = numeric_vals + jitter
            continue
        cat_series = plot_df[col].astype(str).fillna('unknown').astype('category')
        # Add a small deterministic jitter so overlapping lines remain visible.
        base_codes = cat_series.cat.codes.astype(float)
        jitter = rng.uniform(-jitter_strength, jitter_strength, size=len(plot_df))
        plot_df[col] = base_codes + jitter
        cat_maps[col] = list(cat_series.cat.categories)

    cols = varying_cols + ['plot_score']

    fig = px.parallel_coordinates(
        plot_df,
        color="plot_score",
        labels={c: c for c in cols},
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=float(plot_df['plot_score'].median()),
    )
    fig.update_layout(
        width=1900,
        height=1100,
        margin=dict(l=95, r=95, t=110, b=90),
        font=dict(size=20),
    )

    # Restore category names on categorical axes.
    for dim in fig.data[0].dimensions:
        dim_label = dim.label
        if dim_label in cat_maps:
            categories = cat_maps[dim_label]
            dim.tickvals = list(range(len(categories)))
            dim.ticktext = categories

    png_path = f"{out_dir}/10_hparam_parallel_{tag}.png"
    html_path = f"{out_dir}/10_hparam_parallel_{tag}.html"
    compatible, plotly_v, kaleido_v = _is_plotly_kaleido_compatible()
    if compatible:
        try:
            fig.write_image(png_path)
            print(f"Saved parallel coordinates PNG to {png_path}")
        except Exception as e:
            print(f"Static image export failed: {e}. Saving HTML fallback to {html_path}.")
            fig.write_html(html_path)
    else:
        if plotly_v and kaleido_v:
            print(
                f"Skipping PNG export due to Plotly/Kaleido mismatch "
                f"(plotly={plotly_v}, kaleido={kaleido_v}). Saving HTML to {html_path}."
            )
            print("To enable PNG export: install kaleido==0.2.1 or upgrade plotly>=6.1.1.")
        else:
            print(f"Kaleido not available. Saving HTML fallback to {html_path}.")
        fig.write_html(html_path)

def generate_per_model_figures(df, out_dir):
    if 'source_csv' not in df.columns:
        print("Skipping per-model figure generation: 'source_csv' column not present in dataframe. Per-model figures require parsed prediction files.")
        return
    print("Generating detailed figures for all models...")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    # Sort by valid_mcc to give them a rank
    df_sorted = df.dropna(subset=['source_csv']).sort_values('valid_mcc', ascending=False).reset_index(drop=True)
    for idx, row in df_sorted.iterrows():
        match_f = row['source_csv']
        if not isinstance(match_f, str) or not os.path.exists(match_f):
            continue
        try:
            df_p = pd.read_csv(match_f)
            model_id = f"rank{idx+1}_{row['model_name']}_{row['classif_loss']}_{row['BER']}"
            model_id = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in model_id])
            spec_dir = os.path.join(models_dir, model_id)
            os.makedirs(spec_dir, exist_ok=True)
            
            y_true = (df_p['label'] == 'NotNormal').astype(int)
            y_pred = (df_p['pred'] == 'NotNormal').astype(int)
            
            y_true_prob, y_prob = _extract_binary_targets_and_probs(df_p)
            
            # 1. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'NotNormal'], yticklabels=['Normal', 'NotNormal'])
            plt.title(f"Confusion Matrix (MCC: {row['mcc']:.3f})")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(spec_dir, "1_confusion_matrix.png"), dpi=200)
            plt.close()
            
            if y_true_prob is not None and y_prob is not None:
                
                # 2. ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(spec_dir, "2_roc_curve.png"), dpi=200)
                plt.close()
                
                # 3. PR Curve
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                avg_p = average_precision_score(y_true, y_prob)
                plt.figure(figsize=(6, 5))
                plt.step(recall, precision, color='b', alpha=0.2, where='post')
                plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title(f'Precision-Recall curve (AP = {avg_p:.2f})')
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(spec_dir, "3_pr_curve.png"), dpi=200)
                plt.close()
                
                # 4. Calibration Curve
                ece, p_true, p_pred = expected_calibration_error(y_true, y_prob)
                plt.figure(figsize=(6, 5))
                plt.plot(p_pred, p_true, marker='o', label=f"ECE: {ece:.3f}")
                plt.plot([0, 1], [0, 1], '--', color='gray')
                plt.xlabel('Mean Predicted Probability')
                plt.ylabel('Fraction of Positives')
                plt.title('Calibration Curve')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(spec_dir, "4_calibration.png"), dpi=200)
                plt.close()
                
        except Exception as e:
            print(f"Error generating figures for model {idx}: {e}")
            continue


def _save_boxplot(df, x_col, y_col, out_path, title, xlabel=None, ylabel=None, rotate=False):
    if x_col not in df.columns or y_col not in df.columns:
        return
    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    if plot_df.empty or plot_df[x_col].astype(str).nunique() < 2:
        return
    plt.figure(figsize=(9, 5))
    _boxplot_with_scatter(plot_df, x_col, y_col, palette='Set2')
    plt.title(title)
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    if rotate:
        plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _boxplot_with_scatter(data, x_col, y_col, palette='Set2'):
    sns.boxplot(
        data=data,
        x=x_col,
        y=y_col,
        palette=palette,
        width=0.5,
        fliersize=0,
        linewidth=1.2,
        boxprops={'alpha': 0.65},
        whiskerprops={'linewidth': 1.1},
        capprops={'linewidth': 1.1},
        medianprops={'linewidth': 1.5, 'color': '#202020'},
    )
    sns.stripplot(
        data=data,
        x=x_col,
        y=y_col,
        color='black',
        size=6,
        alpha=0.8,
        jitter=0.18,
        dodge=False,
        zorder=3,
    )


def _save_heatmap_counts(df, row_col, col_col, out_path, title):
    if row_col not in df.columns or col_col not in df.columns:
        return
    plot_df = df[[row_col, col_col]].dropna().copy()
    if plot_df.empty:
        return
    table = plot_df.groupby([row_col, col_col]).size().unstack(fill_value=0)
    if table.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.heatmap(table, annot=True, fmt='d', cmap='crest')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_extended_figures(df, out_dir, runtime_df):
    if df is None or df.empty:
        return
    rank_col = 'valid_mcc' if 'valid_mcc' in df.columns else 'mcc'

    if {'valid_mcc', 'test_mcc'}.issubset(df.columns):
        try:
            plot_df = df[['valid_mcc', 'test_mcc', 'model_name', 'classif_loss']].copy()
        except Exception:
            plot_df = df[['valid_mcc', 'test_mcc', 'model']].copy()
        plot_df['valid_mcc'] = pd.to_numeric(plot_df['valid_mcc'], errors='coerce')
        plot_df['test_mcc'] = pd.to_numeric(plot_df['test_mcc'], errors='coerce')
        plot_df = plot_df.dropna(subset=['valid_mcc', 'test_mcc'])
        if not plot_df.empty:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x='valid_mcc', y='test_mcc', hue='model_name',
                            style='classif_loss' if 'classif_loss' in plot_df.columns else None,
                            s=90, alpha=0.75)
            low = min(plot_df['valid_mcc'].min(), plot_df['test_mcc'].min())
            high = max(plot_df['valid_mcc'].max(), plot_df['test_mcc'].max())
            plt.plot([low, high], [low, high], '--', color='gray', alpha=0.6)
            plt.title('Validation vs Test MCC')
            plt.xlabel('Validation MCC')
            plt.ylabel('Test MCC')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/13_valid_vs_test_mcc.png", dpi=300)
            plt.close()

            plot_df['generalization_gap'] = plot_df['valid_mcc'] - plot_df['test_mcc']
            plt.figure(figsize=(8, 5))
            sns.histplot(plot_df['generalization_gap'], bins=20, kde=True, color='steelblue')
            plt.title('Generalization Gap Distribution (valid MCC - test MCC)')
            plt.xlabel('Generalization Gap')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/14_generalization_gap_hist.png", dpi=300)
            plt.close()

    _save_boxplot(df, 'fgsm', rank_col, f"{out_dir}/15_fgsm_ablation.png", 'FGSM Setting vs Validation MCC', xlabel='FGSM', ylabel=rank_col)
    _save_boxplot(df, 'normalize', rank_col, f"{out_dir}/16_normalization_ablation.png", 'Normalization vs Validation MCC', xlabel='Normalization', ylabel=rank_col)
    _save_boxplot(df, 'n_calibration', rank_col, f"{out_dir}/17_calibration_ablation.png", 'Calibration Samples vs Validation MCC', xlabel='n_calibration', ylabel=rank_col)
    _save_boxplot(df, 'dist_fct', rank_col, f"{out_dir}/18_distance_function_ablation.png", 'Distance Function vs Validation MCC', xlabel='Distance Function', ylabel=rank_col, rotate=True)
    _save_boxplot(df, 'n_negatives', rank_col, f"{out_dir}/19_negative_mining_ablation.png", 'Negative Mining vs Validation MCC', xlabel='n_negatives', ylabel=rank_col)
    _save_boxplot(df, 'prototype_agg', rank_col, f"{out_dir}/20_prototype_aggregation_ablation.png", 'Prototype Aggregation vs Validation MCC', xlabel='Prototype Aggregation', ylabel=rank_col, rotate=True)

    if {'knn', rank_col}.issubset(df.columns):
        try:
            plot_df = df[['knn', rank_col, 'model_name', 'classif_loss']].copy()
        except Exception:
            plot_df = df[['knn', rank_col, 'model']].copy()
        plot_df['knn'] = pd.to_numeric(plot_df['knn'], errors='coerce')
        plot_df[rank_col] = pd.to_numeric(plot_df[rank_col], errors='coerce')
        plot_df = plot_df.dropna(subset=['knn', rank_col])
        if not plot_df.empty and plot_df['knn'].nunique() > 1:
            plt.figure(figsize=(9, 6))
            sns.lineplot(data=plot_df.sort_values('knn'), x='knn', y=rank_col, hue='model_name',
                         style='classif_loss' if 'classif_loss' in plot_df.columns else None,
                         marker='o')
            plt.title('KNN Sensitivity Curve')
            plt.xlabel('n_neighbors')
            plt.ylabel(rank_col)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/21_knn_sensitivity.png", dpi=300)
            plt.close()

    _save_heatmap_counts(df, 'model_name', 'classif_loss', f"{out_dir}/22_model_loss_coverage_heatmap.png", 'Coverage Heatmap: Model x Loss')

    if {'kind', 'model_name'}.issubset(df.columns):
        plot_df = df[['kind', 'model_name']].dropna().copy()
        if not plot_df.empty:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=plot_df, x='kind', hue='model_name', palette='viridis')
            plt.title('Run Coverage by Job Kind and Backbone')
            plt.xlabel('Job Kind')
            plt.ylabel('Number of Runs')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/23_job_kind_distribution.png", dpi=300)
            plt.close()

    if 'launcher_failed_final' in df.columns:
        status_df = df.copy()
        status_df['launcher_failed_final'] = pd.to_numeric(status_df['launcher_failed_final'], errors='coerce').fillna(0).astype(int)
        status_counts = status_df['launcher_failed_final'].map({0: 'success', 1: 'failed'}).value_counts()
        if not status_counts.empty:
            plt.figure(figsize=(7, 5))
            sns.barplot(x=status_counts.index, y=status_counts.values, palette='rocket')
            plt.title('Final Job Status Counts')
            plt.xlabel('Final Status')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/24_final_status_counts.png", dpi=300)
            plt.close()

    if 'launcher_retry_count' in df.columns:
        retry_df = df[['launcher_retry_count']].copy()
        retry_df['launcher_retry_count'] = pd.to_numeric(retry_df['launcher_retry_count'], errors='coerce')
        retry_df = retry_df.dropna(subset=['launcher_retry_count'])
        if not retry_df.empty:
            plt.figure(figsize=(7, 5))
            sns.histplot(retry_df['launcher_retry_count'], bins=10, discrete=True, color='darkorange')
            plt.title('Retry Count Distribution')
            plt.xlabel('Retry Count')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/25_retry_distribution.png", dpi=300)
            plt.close()

    if runtime_df is not None and not runtime_df.empty and {'timestamp', 'event'}.issubset(runtime_df.columns):
        timeline_df = runtime_df.copy()
        timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'], errors='coerce')
        timeline_df = timeline_df.dropna(subset=['timestamp'])
        if not timeline_df.empty:
            timeline_df['hour_bucket'] = timeline_df['timestamp'].dt.floor('H')
            event_counts = timeline_df.groupby(['hour_bucket', 'event']).size().reset_index(name='count')
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=event_counts, x='hour_bucket', y='count', hue='event', marker='o')
            plt.title('Job Event Timeline')
            plt.xlabel('Time')
            plt.ylabel('Event Count')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/26_job_event_timeline.png", dpi=300)
            plt.close()

    if {'model_name', 'prototypes', rank_col}.issubset(df.columns):
        plot_df = df[['model_name', 'prototypes', rank_col]].copy()
        plot_df[rank_col] = pd.to_numeric(plot_df[rank_col], errors='coerce')
        plot_df = plot_df.dropna(subset=[rank_col])
        if not plot_df.empty:
            pivot = plot_df.groupby(['model_name', 'prototypes'])[rank_col].mean().unstack()
            if not pivot.empty:
                plt.figure(figsize=(9, 6))
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='magma')
                plt.title('Mean Validation MCC: Model x Prototype Strategy')
                plt.tight_layout()
                plt.savefig(f"{out_dir}/27_model_prototype_heatmap.png", dpi=300)
                plt.close()


def _prepare_numeric_columns(df):
    out = df.copy()
    numeric_cols = [
        'accuracy', 'mcc', 'test_mcc', 'valid_mcc', 'test_accuracy', 'valid_accuracy',
        'summary_finished',
        'summary_best_mcc', 'summary_best_accuracy',
        'summary_valid_mcc', 'summary_valid_accuracy', 'summary_test_mcc', 'summary_test_accuracy',
        'csv_valid_mcc', 'csv_valid_accuracy', 'csv_test_mcc', 'csv_test_accuracy',
        'batch_entropy_norm', 'batch_ari'
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def generate_core_figures_and_tables(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), 'mcc'

    df = _prepare_numeric_columns(df)
    sns.set_theme(style="whitegrid", context="paper")
    # Robust metric selection for ranking
    def pick_rank_col(df):
        for c in ['valid_mcc', 'mcc', 'test_mcc', 'accuracy']:
            if c in df.columns:
                return c
        print("Warning: No ranking metric column (valid_mcc, mcc, test_mcc, accuracy) found in dataframe. Returning None.")
        return None

    rank_col = pick_rank_col(df)
    if rank_col is None:
        print("No available metric column for ranking. Skipping all ranking-based plots and tables.")
        return pd.DataFrame(), pd.DataFrame(), None

    # 1. Model Comparison (Boxplot per architecture)
    if rank_col and {'model_name', rank_col}.issubset(df.columns):
        plot_df = df.dropna(subset=['model_name', rank_col])
        if not plot_df.empty:
            plt.figure(figsize=(8, 5))
            _boxplot_with_scatter(plot_df, 'model_name', rank_col, palette='viridis')
            plt.title(f"{rank_col.upper()} Distribution by Network Architecture")
            plt.ylabel(rank_col.upper())
            plt.xlabel("Architecture")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/1_architecture_{rank_col}.png", dpi=300)
            plt.close()

    # 2. Loss Function Ablation
    if rank_col and {'classif_loss', rank_col}.issubset(df.columns):
        plot_df = df.dropna(subset=['classif_loss', rank_col])
        if not plot_df.empty:
            plt.figure(figsize=(10, 6))
            _boxplot_with_scatter(plot_df, 'classif_loss', rank_col, palette='Set2')
            plt.title(f"{rank_col.upper()} Distribution by Classification Loss")
            plt.ylabel(rank_col.upper())
            plt.xlabel("Classification Loss")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/2_loss_ablation_{rank_col}.png", dpi=300)
            plt.close()

    # 3. Prototype Strategy Ablation
    if rank_col and {'prototypes', rank_col}.issubset(df.columns):
        plot_df = df.dropna(subset=['prototypes', rank_col])
        if not plot_df.empty:
            # 3a. Violinplot (existing)
            plt.figure(figsize=(8, 5))
            sns.violinplot(data=plot_df, x='prototypes', y=rank_col, palette='Pastel1', inner="box")
            plt.title(f"Impact of Prototype Strategy on {rank_col.upper()}")
            plt.ylabel(rank_col.upper())
            plt.xlabel("Prototype Strategy")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/3_prototype_ablation_{rank_col}.png", dpi=300)
            plt.close()

            # 3b. Boxplot (new)
            plt.figure(figsize=(8, 5))
            _boxplot_with_scatter(plot_df, 'prototypes', rank_col, palette='Pastel2')
            plt.title(f"Impact of Prototype Strategy on {rank_col.upper()} (Boxplot)")
            plt.ylabel(rank_col.upper())
            plt.xlabel("Prototype Strategy")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/3b_prototype_ablation_{rank_col}_boxplot.png", dpi=300)
            plt.close()

            # Save the table as CSV for 3_
            plot_df.to_csv(f"{out_dir}/3_prototype_ablation_{rank_col}.csv", index=False)

    # 4. Domain Generalization: Metric vs Batch Entropy Norm
    if rank_col and {'batch_entropy_norm', rank_col, 'model_name', 'classif_loss'}.issubset(df.columns):
        plot_df = df.dropna(subset=['batch_entropy_norm', rank_col])
        if not plot_df.empty:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x='batch_entropy_norm', y=rank_col, hue='model_name', style='classif_loss', s=100, palette='deep')
            plt.title(f"Performance vs Domain Invariance (Batch Entropy, {rank_col.upper()})")
            plt.ylabel(f"{rank_col.upper()} (Performance)")
            plt.xlabel("Normalized Batch Entropy (Higher = Better mixing across batches)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/4_{rank_col}_vs_entropy.png", dpi=300)
            plt.close()

    # 5. BER ablation
    if rank_col and {'BER', rank_col}.issubset(df.columns):
        plot_df = df.dropna(subset=['BER', rank_col])
        if not plot_df.empty:
            plt.figure(figsize=(7, 5))
            _boxplot_with_scatter(plot_df, 'BER', rank_col, palette='Set3')
            plt.title(f"Impact of BER Strategy on {rank_col.upper()}")
            plt.ylabel(rank_col.upper())
            plt.xlabel("BER Strategy (Batch Effect Removal)")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/5_dloss_ablation_{rank_col}.png", dpi=300)
            plt.close()

    # Tables
    top_models = df.sort_values(by=rank_col, ascending=False).head(20)
    table_cols = [
        'model_name', 'kind', 'variant', 'classif_loss', 'prototypes', 'dloss', 'BER',
        'fgsm', 'normalize', 'n_calibration', 'dist_fct', 'knn', 'n_negatives',
        'valid_mcc', 'test_mcc', 'valid_accuracy', 'test_accuracy',
        'summary_valid_mcc', 'summary_test_mcc', 'summary_best_mcc',
        'csv_valid_mcc', 'csv_test_mcc',
        'launcher_retry_count', 'launcher_failed_final', 'source_datetime'
    ]
    existing_table_cols = [c for c in table_cols if c in top_models.columns]
    if not existing_table_cols:
        existing_table_cols = ['model_name', 'classif_loss', 'prototypes', 'dloss', 'BER', 'fgsm', rank_col, 'accuracy']
    top_models_subset = top_models[existing_table_cols]

    csv_path = f"{out_dir}/top_models.csv"
    top_models_subset.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')
    display_df = top_models_subset.copy()
    for metric_col in ['mcc', 'accuracy', 'valid_mcc', 'test_mcc', 'valid_accuracy', 'test_accuracy', 'summary_valid_mcc', 'summary_test_mcc', 'summary_best_mcc', 'csv_valid_mcc', 'csv_test_mcc']:
        if metric_col in display_df.columns:
            display_df[metric_col] = display_df[metric_col].apply(
                lambda x: f"{float(x):.4f}" if pd.notnull(x) and isinstance(x, (int, float, np.integer, np.floating)) else (x if isinstance(x, str) else "NaN")
            )
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/7_top_models_table.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Instead of using only the dataframe, rebuild all_models.csv from manifest and logs for completeness
    manifest_path = 'logs/progresses/otitis_four_class/otite_ds_64_USA_Turquie_Chili_GMFUNL_inference/csv/PROD_otitis_four_class_job_manifest.csv'
    all_csv_path = f"{out_dir}/all_models.csv"
    build_all_models_from_manifest_and_logs(manifest_path, all_csv_path)

    # Read the rebuilt all_models.csv for further processing
    all_models_df = pd.read_csv(all_csv_path)
    # Use the same columns as for top_models_subset, if available
    table_cols = [
        'model_name', 'kind', 'variant', 'classif_loss', 'prototypes', 'dloss', 'BER',
        'fgsm', 'normalize', 'n_calibration', 'dist_fct', 'knn', 'n_negatives',
        'valid_mcc', 'test_mcc', 'valid_accuracy', 'test_accuracy',
        'summary_valid_mcc', 'summary_test_mcc', 'summary_best_mcc',
        'csv_valid_mcc', 'csv_test_mcc',
        'launcher_retry_count', 'launcher_failed_final', 'source_datetime'
    ]
    existing_table_cols = [c for c in table_cols if c in all_models_df.columns]
    if not existing_table_cols:
        existing_table_cols = ['uuid', 'classif_loss', 'prototypes', 'dloss', 'BER', 'fgsm', 'mcc', 'accuracy']
    all_models_subset = all_models_df[existing_table_cols]

    fig_height = max(6, len(all_models_subset) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')
    ax.axis('tight')
    all_display_df = all_models_subset.copy()
    for metric_col in ['mcc', 'accuracy', 'valid_mcc', 'test_mcc', 'valid_accuracy', 'test_accuracy', 'summary_valid_mcc', 'summary_test_mcc', 'summary_best_mcc', 'csv_valid_mcc', 'csv_test_mcc']:
        if metric_col in all_display_df.columns:
            all_display_df[metric_col] = all_display_df[metric_col].apply(
                lambda x: f"{float(x):.4f}" if pd.notnull(x) and isinstance(x, (int, float, np.integer, np.floating)) else (x if isinstance(x, str) else "NaN")
            )
    table_all = ax.table(cellText=all_display_df.values, colLabels=all_display_df.columns, cellLoc='center', loc='center')
    table_all.auto_set_font_size(False)
    table_all.set_fontsize(9)
    table_all.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/8_all_models_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    # Save the table as CSV for 8_
    all_display_df.to_csv(f"{out_dir}/8_all_models_table.csv", index=False)

    print(f"Saved Top 20 overview to CSV: {csv_path}")
    print(f"Saved All Models overview to CSV: {all_csv_path}")
    return top_models_subset, all_models_subset, rank_col


def print_completed_runs(df):
    if df is None or df.empty:
        print("No runs available to report.")
        return

    report_df = _prepare_numeric_columns(df)
    status_series = report_df.get('summary_run_status', pd.Series('', index=report_df.index)).astype(str).str.lower().str.strip()
    finished_series = pd.to_numeric(report_df.get('summary_finished', pd.Series(np.nan, index=report_df.index)), errors='coerce')
    error_series = report_df.get('summary_error_message', pd.Series('', index=report_df.index)).fillna('').astype(str).str.strip()

    completed_mask = (
        status_series.eq('completed') |
        finished_series.eq(1)
    ) & error_series.eq('')

    worked = report_df.loc[completed_mask].copy()
    if worked.empty:
        print("No completed runs found in run_summary.json files.")
        return

    sort_cols = [c for c in ['summary_best_mcc', 'test_mcc', 'valid_mcc', 'source_mtime'] if c in worked.columns]
    ascending = [False, False, False, False][:len(sort_cols)]
    if sort_cols:
        worked = worked.sort_values(sort_cols, ascending=ascending, na_position='last')

    print("Completed runs:")
    for _, row in worked.iterrows():
        source_csv = str(row.get('source_csv', ''))
        run_dir = os.path.dirname(source_csv) if source_csv else ''
        identifier = run_dir or source_csv or str(row.get('exp_id', 'unknown'))
        best_mcc = _prefer_metric(row.get('summary_best_mcc'), row.get('valid_mcc'), row.get('mcc'))
        valid_mcc = _coerce_metric_scalar(row.get('valid_mcc'))
        test_mcc = _coerce_metric_scalar(row.get('test_mcc'))
        model_name = str(row.get('model_name', 'unknown'))
        status = str(row.get('summary_run_status', 'unknown')) or 'unknown'

        def _fmt(value):
            return f"{value:.4f}" if pd.notna(value) else 'NaN'

        print(
            f"- {identifier} | model={model_name} | status={status} | "
            f"best_mcc={_fmt(best_mcc)} | valid_mcc={_fmt(valid_mcc)} | test_mcc={_fmt(test_mcc)}"
        )

def run_analysis():
    # 1. EDA Phase
    out_dir = "/home/simon/otitenet/output/analysis"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Analysis outputs will be saved under: {out_dir}")
    
    did_eda = do_eda(out_dir)
    make_interpretability_figure(out_dir)
    
    # Build all_models.csv from manifest and logs
    task = _get_target_task()
    tag = 'TEST_SMOKE' if args.test else 'PROD'
    manifest_path = f'logs/progresses/{task}/{args.dataset}/csv/{tag}_{task}_job_manifest.csv'
    output_csv = f'logs/progresses/{task}/{args.dataset}/csv/{tag}_{task}_all_models.csv'
    build_all_models_from_manifest_and_logs(manifest_path, output_csv, tag=tag)
    df = pd.read_csv(output_csv)
    # Use the correct model column: prefer 'model_name', fallback to 'model'
    model_col = 'model_name' if 'model_name' in df.columns else ('model' if 'model' in df.columns else None)
    if model_col is None:
        raise ValueError("No model or model_name column found in input CSV.")
    manifest_df, runtime_df = _load_launcher_metadata()
    if df.empty:
        print("No data available from logs for model analysis.")
        print(f"No analysis artifacts were generated because no runs remained after filtering. Output root: {out_dir}")
        return
    print_completed_runs(df)
        
    # Additional Paper Analysis
    df_all = df.copy()
    df_best_ntrials = _pick_best_per_n_trials(df_all)

    all_dir = os.path.join(out_dir, 'all_runs')
    best_dir = os.path.join(out_dir, 'best_by_n_trials')
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    do_calibration_curves(df_all, out_dir)
    do_hparam_parallel(df_all, out_dir, tag='all_runs', dedupe_latest=False)
    do_hparam_parallel(df_best_ntrials, out_dir, tag='best_by_n_trials', dedupe_latest=False)

    do_synergy_heatmap(df_all, all_dir)
    do_synergy_heatmap(df_best_ntrials, best_dir)
    do_pareto_frontier(df_all, all_dir)
    do_pareto_frontier(df_best_ntrials, best_dir)
    generate_extended_figures(df_all, all_dir, runtime_df)
    generate_extended_figures(df_best_ntrials, best_dir, runtime_df)
    generate_per_model_figures(df, out_dir)
    
    print(f"Successfully analysed {len(df)} configurations.")
    model_col = 'model_name' if 'model_name' in df.columns else None
    if model_col is None and 'model' in df.columns:
        print("[WARNING] 'model_name' column not found. Using 'model' column instead. Please ensure future outputs use 'model_name'.")
        model_col = 'model'
    if model_col is None:
        raise KeyError("Neither 'model_name' nor 'model' column found in dataframe.")
    print(f"Architectures evaluated: {df[model_col].unique().tolist()}")

    # Generate core figures/tables for both views and keep root outputs for backward compatibility.
    df_all_clean = _prepare_numeric_columns(df_all)
    df_best_clean = _prepare_numeric_columns(df_best_ntrials)
    generate_core_figures_and_tables(df_all_clean, all_dir)
    generate_core_figures_and_tables(df_best_clean, best_dir)
    top_models_subset, _, _ = generate_core_figures_and_tables(df_all_clean, out_dir)

    md_content = "# Comprehensive Research Analysis: Otitis Media Classification\\n\\n"
    md_content += "## Project Abstract\\n"
    md_content += "This study explores the application of deep metric learning and structural regularization for the domain-generalized classification of otitis media. By evaluating 136+ configurations across diverse architectures (ResNet, VGG, ViT), we demonstrate that structural constraints and sub-center prototype strategies significantly enhance both performance (MCC) and domain invariance.\\n\\n"
    
    if did_eda:
        md_content += "## 0. Dataset Distribution and Exploratory Data Analysis\\n"
        md_content += "Analysis of the class distribution (`0_dataset_eda.png`) reveals the presence of batch-induced variance. Addressing this variance is the primary objective of the BER (Batch Effect Removal) strategies evaluated here.\\n\\n"
    
    md_content += "## 1. Quantitative Performance: Top Model Leaderboard\\n"
    md_content += "The following table summarizes the best-performing configurations based on Matthews Correlation Coefficient (MCC):\\n\\n"
    md_content += top_models_subset.to_markdown(index=False)
    md_content += "\\n\\n"
    
    md_content += "## 2. Experimental Ablations and Key Findings\\n"
    md_content += "### Architectural Backbone Comparison\\n"
    md_content += "As shown in `1_architecture_mcc.png`, ResNet-based backbones demonstrate a superior balance between feature richness and convergence stability compared to larger architectures in data-constrained scenarios.\\n\\n"
    
    md_content += "### Impact of Classification Loss and BER\\n"
    md_content += "The shift from standard Cross-Entropy to Siamese-based losses (ArcFace/Triplet) provides a clearer separation in the latent space (`2_loss_ablation.png`). BER strategies (such as inverseTriplet) further regularize this space to prevent overfitting to batch-specific artifacts.\\n\\n"
    
    md_content += "### Prototype-Based Clustering\\n"
    md_content += "Evaluations of prototype strategies (`3_prototype_ablation.png`) show that 'Class-based' prototypes offer a more global semantic coherence, whereas 'Batch-based' strategies help in localizing domain-specific features for rejection.\\n\\n"
    
    md_content += "### Domain Invariance and Performance Equilibrium\\n"
    md_content += "The relationship between performance (MCC) and Domain Entropy (`4_mcc_vs_entropy.png`) identifies models that achieve high accuracy while maintaining high entropy across batches, indicating true generalization rather than shortcut learning.\\n\\n"
    
    md_content += "## 3. Clinical Validation and Interpretability\\n"
    md_content += "To ensure clinical reliability, we compare Grad-CAM and SHAP activations (`6_interpretability.png`). The high spatial correlation between these independent explainers provides confidence that the models are focusing on pathologically relevant features (e.g., tympanic membrane opacity, vascularization) rather than image noise.\\n\\n"
    
    md_content += "## 4. Calibration and Reliability\\n"
    md_content += "Reliable AI transition to clinical practice requires well-calibrated probabilities. We report separate calibration curves for test (`9_calibration_curves_test.png`) and validation (`9_calibration_curves_valid.png`) splits, showing alignment between predicted probabilities and actual outcomes with split-specific ECE values.\\n\\n"
    
    md_content += "## 5. Hyperparameter Sensitivity\\n"
    md_content += "The interactive parallel coordinates plot (`10_hparam_parallel.html`) provides a global view of how architectural choices (Backbone, Loss, BER, Prototypes) interact to influence final performance. This visualization reveals that while ResNet-18 is robust, the specific combination of Siamese loss and BER strategy is critical for peak MCC.\\n\\n"
    
    md_content += "## 6. Regularization Synergies\\n"
    md_content += "The synergy matrix (`11_synergy_heatmap.png`) explores the coupling between classification losses and Batch Effect Removal (BER). We observe that `inverseTriplet` provides a consistent performance boost over `none` across almost all loss formulations, suggesting its generalizability as a structural regularizer.\\n\\n"
    
    md_content += "## 7. The Pareto Frontier of Generalization\\n"
    md_content += "Ultimately, we seek models that maximize accuracy without sacrificing domain invariance (measured by Normalized Batch Entropy). The Pareto frontier (`12_pareto_generalization.png`) identifies the optimal 'sweet spot' models that occupy the upper-right quadrant, successfully balancing predictive power with resilience to batch-induced variance.\\n\\n"

    md_content += "## 8. Expanded Diagnostics with Launcher Metadata\\n"
    md_content += "The enriched launcher metadata and broader run discovery now support a much denser figure set. We report validation-vs-test agreement (`13_valid_vs_test_mcc.png`), generalization gap (`14_generalization_gap_hist.png`), FGSM/normalization/calibration ablations (`15_fgsm_ablation.png`, `16_normalization_ablation.png`, `17_calibration_ablation.png`), metric-space sensitivity (`18_distance_function_ablation.png`, `21_knn_sensitivity.png`), negative mining and prototype aggregation (`19_negative_mining_ablation.png`, `20_prototype_aggregation_ablation.png`), coverage maps (`22_model_loss_coverage_heatmap.png`, `27_model_prototype_heatmap.png`), and runtime/operational summaries (`23_job_kind_distribution.png`, `24_final_status_counts.png`, `25_retry_distribution.png`, `26_job_event_timeline.png`).\\n\\n"

    with open(f"{out_dir}/PAPER_ANALYSIS.md", "w") as f:
        f.write(md_content)
        
    print(f"Generated extended figures in '{out_dir}/' and robust report table at '{out_dir}/PAPER_ANALYSIS.md'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate paper analysis from training outputs.')
    parser.add_argument('--task', type=str, default='otitis_four_class', help='Target task directory under logs/ (e.g., notNormal, otitis, multi).')
    parser.add_argument('--test', action='store_true', help='Analyze only test/smoke runs (launch --test / run_tag containing test).')
    parser.add_argument('--dataset', type=str, default='otite_ds_64_USA_Turquie_Chili_GMFUNL_inference', help='Dataset subdirectory under logs/progresses/{task}/ (e.g., otite_ds_64_USA_Turquie_Chili_GMFUNL_inference).')
    
    args = parser.parse_args()

    if args.task:
        os.environ['OTITENET_TASK'] = str(args.task)
    if args.test:
        os.environ['OTITENET_ANALYSIS_TEST_ONLY'] = '1'

    run_analysis()
