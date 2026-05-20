import os

import pandas as pd


COMPLETED_RUNS_METRICS_HEADER = [
    'timestamp', 'exp_id',
    'trial_index',
    'uuid', 'status', 'error',
    'model', 'model_name',
    'kind', 'variant',
    'loss', 'classif_loss',
    'prototype', 'prototypes',
    'dloss', 'BER',
    'fgsm', 'normalize', 'n_calibration', 'dist_fct', 'knn', 'n_negatives',
    'retry_count', 'launcher_retry_count', 'launcher_failed_final',
    'valid_mcc', 'test_mcc', 'valid_accuracy', 'test_accuracy',
    'source_datetime',
    'batch_entropy_norm', 'batch_nmi', 'batch_ari',
]


def append_completed_run_metrics(csv_path, row, header=None):
    header = header or COMPLETED_RUNS_METRICS_HEADER
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df_row = pd.DataFrame([row]).reindex(columns=header)
    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False, columns=header)
        return

    try:
        existing_header = list(pd.read_csv(csv_path, nrows=0).columns)
    except Exception:
        existing_header = []

    if existing_header != header:
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception:
            existing_df = pd.DataFrame()

        # Backfill aliases from older schemas.
        if 'model' not in existing_df.columns and 'model_name' in existing_df.columns:
            existing_df['model'] = existing_df['model_name']
        if 'loss' not in existing_df.columns and 'classif_loss' in existing_df.columns:
            existing_df['loss'] = existing_df['classif_loss']
        if 'prototype' not in existing_df.columns and 'prototypes' in existing_df.columns:
            existing_df['prototype'] = existing_df['prototypes']
        if 'retry_count' not in existing_df.columns and 'launcher_retry_count' in existing_df.columns:
            existing_df['retry_count'] = existing_df['launcher_retry_count']
        if 'trial_index' not in existing_df.columns:
            existing_df['trial_index'] = ''

        existing_df = existing_df.reindex(columns=header)
        merged_df = pd.concat([existing_df, df_row], ignore_index=True)
        merged_df.to_csv(csv_path, index=False, columns=header)
        return

    df_row.to_csv(csv_path, mode='a', header=False, index=False, columns=header)
