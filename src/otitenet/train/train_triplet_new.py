from otitenet.logging.metrics import compute_batch_effect_metrics
import logging
import os
import sys
import traceback
import matplotlib

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

import mlflow
import warnings
import torchvision
import torch.nn.functional as F
import uuid
import copy
import shutil
import pickle
import hashlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
import torch
from torch import nn
from contextlib import nullcontext
from PIL import Image
from datetime import datetime
import time
from otitenet.data.transforms_manifest import image_preprocessing_manifest
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
try:
    from umap.umap_ import UMAP
except ModuleNotFoundError:
    UMAP = None
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.neural_network import MLPClassifier
from otitenet.ml import (
    find_best_classifier,
    evaluate_knn_with_k_search,
    fit_knn_classifier,
    fit_linearsvc_classifier,
    fit_logreg_classifier,
    evaluate_all_classifiers,
    fit_baseline_classifiers,
    fit_kde_classifier,
)
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef as MCC
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import mysql.connector

# ...rest of user-supplied code (full content pasted by user)...
from otitenet.logging.metrics import compute_batch_effect_metrics

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

from ..logging.loggings import LogConfusionMatrix, add_to_tracking, add_to_mlflow, create_tracking_run, log_mlflow, \
    TRACKING_API_TOKEN, TRACKING_PROJECT_NAME, TRACKING_MODEL_NAME, add_to_logger
from ..logging.dvclive_tracking import DVCLiveTracker
from ..data.data_getters import GetData, get_distance_fct, get_images_loaders, get_n_features
from ..data.dataset_paths import resolve_processed_dataset_path
from ..utils.utils import get_optimizer, to_categorical, get_empty_dicts, get_empty_traces, \
    log_traces, save_tensor, get_best_params, get_best_params_comet
from ..utils.encoding_utils import encode_split_with_augmentation, get_base_transform, get_knn_augmentation_transform
from ..utils.memory_telemetry import (
    emit_gpu_telemetry,
    estimate_theoretical_gpu_required_mb,
    gpu_memory_stats,
    reset_gpu_peak,
    record_gpu_peak,
)
from ..utils.training_config import disable_stn_when_unsupported, validate_n_calibration
from ..models.cnn import Net, Net_deep_shap, Net_shap
from ..logging.losses import TupletLoss,ArcFaceLoss, ArcFaceLossWithHSM, \
        ArcFaceLossWithSubcenters, ArcFaceLossWithSubcentersHSM, SoftmaxContrastiveLoss
from ..utils.kde import make_kde_classifier
from ..logging.plotting import save_roc_curve, plot_pca
from .batch_effects import get_batch_metrics
from .completed_runs_metrics import append_completed_run_metrics

warnings.filterwarnings("ignore")


class _NullSummaryWriter:
    def __init__(self, *args, **kwargs):
        self.disabled = True

    def add_scalar(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None

    def close(self):
        return None


def _make_summary_writer(path, enabled=True):
    if not enabled:
        return _NullSummaryWriter(path)
    try:
        from tensorboardX import SummaryWriter
    except ModuleNotFoundError:
        return _NullSummaryWriter(path)
    return SummaryWriter(path)


def _normalize_train_datasets(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [str(x).strip() for x in value]
    else:
        parts = [x.strip() for x in str(value).replace(";", ",").split(",")]
    return ",".join(list(dict.fromkeys([x for x in parts if x and x.lower() not in {"none", "nan", "null"}])))


def _normalize_single_dataset(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return _normalize_train_datasets(value)
    return str(value or "").strip()


def _dataset_path_segment(path):
    text = str(path or "").strip().replace("\\", "/").strip("/")
    for prefix in ("./data/", "data/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text or "otite_ds_64"


def _split_config_from_args(args):
    train_datasets = _normalize_train_datasets(
        getattr(args, "effective_train_datasets", None)
        or getattr(args, "train_datasets", "")
    )
    valid_dataset = _normalize_single_dataset(
        getattr(args, "effective_valid_dataset", None)
        or getattr(args, "valid_dataset", "")
    )
    test_dataset = _normalize_single_dataset(
        getattr(args, "effective_test_dataset", None)
        or getattr(args, "test_dataset", "")
    )
    key = "|".join([train_datasets, valid_dataset, test_dataset])
    
    # Generate dataset subdirectory string instead of split hash
    from otitenet.data.dataset_paths import dataset_output_subdir, DATASET_SHORT_NAMES
    
    # Collect all unique datasets
    all_datasets = set()
    if train_datasets and train_datasets != 'from_infos_csv':
        all_datasets.update([d.strip() for d in train_datasets.split(',')])
    if valid_dataset and valid_dataset != 'from_infos_csv':
        all_datasets.add(valid_dataset)
    if test_dataset and test_dataset != 'from_infos_csv':
        all_datasets.add(test_dataset)
    
    # Generate subdirectory string using short names, sorted and deduplicated
    # This ensures that the same datasets in any order yield the same segment, and no duplicates
    if all_datasets:
        # Use DATASET_SHORT_NAMES if available, else just the name
        short_names = []
        for d in sorted(all_datasets):
            short = DATASET_SHORT_NAMES.get(d, d) if 'DATASET_SHORT_NAMES' in locals() or 'DATASET_SHORT_NAMES' in globals() else d
            short_names.append(short)
        # Remove duplicates while preserving order (shouldn't be any after set, but just in case)
        seen = set()
        unique_short_names = [x for x in short_names if not (x in seen or seen.add(x))]
        dataset_subdir = "-".join(unique_short_names)
    else:
        dataset_subdir = ""

    return {
        "train_datasets": train_datasets,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "split_config_key": key,
        "split_segment": dataset_subdir,
    }


def _ensure_registry_split_columns(cursor, conn):
    columns = {
        "task": "VARCHAR(128) NULL",
        "dataset_path": "VARCHAR(512) NULL",
        "siamese_inference": "VARCHAR(64) NULL",
        "run_tag": "VARCHAR(128) NULL",
        "config_key": "VARCHAR(64) NULL",
        "train_datasets": "TEXT NULL",
        "valid_dataset": "VARCHAR(255) NULL",
        "test_dataset": "VARCHAR(255) NULL",
        "split_config_key": "VARCHAR(1024) NULL",
        "artifact_id": "VARCHAR(32) NULL",
        "best_model_dir": "TEXT NULL",
        "source_run_log_path": "TEXT NULL",
        "batch_entropy_norm": "DOUBLE NULL",
        "batch_nmi": "DOUBLE NULL",
        "batch_ari": "DOUBLE NULL",
    }
    for col, col_type in columns.items():
        try:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'best_models_registry'
                  AND COLUMN_NAME = %s
                """,
                (col,),
            )
            has_col = cursor.fetchone()[0] > 0
            if not has_col:
                cursor.execute(f"ALTER TABLE best_models_registry ADD COLUMN {col} {col_type}")
                conn.commit()
        except Exception as e:
            print(f"Warning: could not ensure best_models_registry.{col}: {e}")

    # New uniqueness model: one best row per full configuration key.
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'best_models_registry'
              AND INDEX_NAME = 'uniq_config_key'
            """
        )
        has_idx = cursor.fetchone()[0] > 0
        if not has_idx:
            cursor.execute("CREATE UNIQUE INDEX uniq_config_key ON best_models_registry(config_key)")
            conn.commit()
    except Exception as e:
        print(f"Warning: could not ensure best_models_registry.uniq_config_key: {e}")

    # Legacy unique_combo is too coarse (collides across task/dataset/split/head method).
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'best_models_registry'
              AND INDEX_NAME = 'unique_combo'
            """
        )
        has_legacy_idx = cursor.fetchone()[0] > 0
        if has_legacy_idx:
            cursor.execute("DROP INDEX unique_combo ON best_models_registry")
            conn.commit()
    except Exception as e:
        print(f"Warning: could not drop legacy best_models_registry.unique_combo index: {e}")


def _best_registry_config_key(params):
    """Stable key for best-model grouping across full run-defining config."""
    key_fields = [
        str(params.get('task', '') or '').strip(),
        str(params.get('model_name', '') or '').strip(),
        str(params.get('dataset_path', '') or '').strip().replace('\\', '/'),
        str(params.get('train_datasets', '') or '').strip(),
        str(params.get('valid_dataset', '') or '').strip(),
        str(params.get('test_dataset', '') or '').strip(),
        str(params.get('split_config_key', '') or '').strip(),
        str(params.get('siamese_inference', '') or '').strip(),
        str(params.get('classif_loss', '') or '').strip(),
        str(params.get('dloss', '') or '').strip(),
        str(params.get('prototypes', '') or '').strip(),
        str(params.get('prototype_strategy', '') or '').strip(),
        str(params.get('prototype_components', '') or '').strip(),
        str(params.get('fgsm', '') or '').strip(),
        str(params.get('n_calibration', '') or '').strip(),
        str(params.get('normalize', '') or '').strip(),
        str(params.get('nsize', '') or '').strip(),
        str(params.get('dist_fct', '') or '').strip(),
        str(params.get('n_neighbors', '') or '').strip(),
        str(params.get('npos', '') or '').strip(),
        str(params.get('nneg', '') or '').strip(),
    ]
    return hashlib.sha1("|".join(key_fields).encode("utf-8")).hexdigest()

# Ax emits noisy kwargs type-check warnings in this environment even for valid runtime values.
# Keep actual training errors visible while silencing that specific logger.
logging.getLogger("ax.utils.common.kwargs").setLevel(logging.ERROR)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from otitenet.utils.utils import set_random_seeds as _set_global_seeds  # Use shared seed function


# Default seed before args are parsed; overridden later via args.seed
_set_global_seeds(1, deterministic=True)

from ..logging.shap import log_shap_images_gradients
from ..logging.grad_cam import log_grad_cam_similarity
from ..utils.prototypes import Prototypes


def _make_comet_experiment(api_key, project_name):
    from ..logging.loggings import prepare_comet_lazy_import

    prepare_comet_lazy_import()
    try:
        import comet_ml
    except ModuleNotFoundError:
        print("[Comet] comet_ml is not installed: skipping Comet logging.")
        return None
    return comet_ml.Experiment(api_key=api_key, project_name=project_name)

def update_best_model_registry(params, accuracy, mcc, log_path, batch_metrics=None, source_run_log_path=None, best_model_dir=None):
    """Update best_models_registry table with model performance metrics.
    
    Args:
        params: Dict with model parameters
        accuracy: Model accuracy
        mcc: Matthews Correlation Coefficient
        log_path: Path to model logs
        batch_metrics: Optional dict with batch effect metrics (batch_entropy_norm, batch_nmi, batch_ari)
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="y_user",
            password="password",
            database="results_db"
        )
        cursor = conn.cursor()
        _ensure_registry_split_columns(cursor, conn)
        
        # New columns for prototype aggregation
        proto_strat = params.get('prototype_strategy', 'mean')
        proto_comp = params.get('prototype_components', 1)
        train_datasets = params.get('train_datasets', '')
        valid_dataset = params.get('valid_dataset', '')
        test_dataset = params.get('test_dataset', '')
        split_key = params.get('split_config_key', '|'.join([str(train_datasets or ''), str(valid_dataset or ''), str(test_dataset or '')]))
        config_key = params.get('config_key') or _best_registry_config_key(params)
        best_model_dir = best_model_dir or log_path
        artifact_id = hashlib.sha1(str(log_path).replace("\\", "/").strip("/").encode("utf-8")).hexdigest()[:12]

        try:
            cursor.execute("""
                SELECT mcc FROM best_models_registry
                WHERE config_key=%s
            """, (
                config_key,
            ))
            row = cursor.fetchone()
        except mysql.connector.Error:
            row = None
        if row is None or row[0] is None or float(mcc) >= float(row[0]):
            # Extract batch metrics if provided
            batch_entropy = None
            batch_nmi = None
            batch_ari = None
            if batch_metrics:
                batch_entropy = batch_metrics.get('batch_entropy_norm', batch_metrics.get('batch_entropy'))
                batch_nmi = batch_metrics.get('batch_nmi')
                batch_ari = batch_metrics.get('batch_ari')
                # Convert numpy types to Python floats if needed
                if batch_entropy is not None:
                    try:
                        batch_entropy = float(batch_entropy) if not (isinstance(batch_entropy, float) and batch_entropy != batch_entropy) else None  # None if NaN
                    except (TypeError, ValueError):
                        batch_entropy = None
                if batch_nmi is not None:
                    try:
                        batch_nmi = float(batch_nmi) if not (isinstance(batch_nmi, float) and batch_nmi != batch_nmi) else None
                    except (TypeError, ValueError):
                        batch_nmi = None
                if batch_ari is not None:
                    try:
                        batch_ari = float(batch_ari) if not (isinstance(batch_ari, float) and batch_ari != batch_ari) else None
                    except (TypeError, ValueError):
                        batch_ari = None
            
            try:
                cursor.execute("""
                    INSERT INTO best_models_registry
                    (config_key, task, model_name, dataset_path, siamese_inference, run_tag,
                     nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss, n_calibration,
                     accuracy, mcc, batch_entropy_norm, batch_nmi, batch_ari, normalize, n_neighbors,
                     prototype_strategy, prototype_components, train_datasets, valid_dataset, test_dataset,
                     split_config_key, log_path, artifact_id, best_model_dir, source_run_log_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        accuracy = IF(VALUES(mcc) >= COALESCE(mcc, -9999), VALUES(accuracy), accuracy),
                        mcc = IF(VALUES(mcc) >= COALESCE(mcc, -9999), VALUES(mcc), mcc),
                        batch_entropy_norm = IF(VALUES(mcc) >= COALESCE(mcc, -9999), VALUES(batch_entropy_norm), batch_entropy_norm),
                        batch_nmi = IF(VALUES(mcc) >= COALESCE(mcc, -9999), VALUES(batch_nmi), batch_nmi),
                        batch_ari = IF(VALUES(mcc) >= COALESCE(mcc, -9999), VALUES(batch_ari), batch_ari),
                        log_path = IF(VALUES(mcc) >= COALESCE(mcc, -9999), VALUES(log_path), log_path),
                        artifact_id = IF(VALUES(mcc) >= COALESCE(mcc, -9999), VALUES(artifact_id), artifact_id),
                        best_model_dir = IF(VALUES(mcc) >= COALESCE(mcc, -9999), VALUES(best_model_dir), best_model_dir),
                        source_run_log_path = IF(VALUES(mcc) >= COALESCE(mcc, -9999), VALUES(source_run_log_path), source_run_log_path),
                        task = VALUES(task),
                        model_name = VALUES(model_name),
                        dataset_path = VALUES(dataset_path),
                        siamese_inference = VALUES(siamese_inference),
                        run_tag = VALUES(run_tag),
                        nsize = VALUES(nsize),
                        fgsm = VALUES(fgsm),
                        prototypes = VALUES(prototypes),
                        npos = VALUES(npos),
                        nneg = VALUES(nneg),
                        dloss = VALUES(dloss),
                        dist_fct = VALUES(dist_fct),
                        classif_loss = VALUES(classif_loss),
                        n_calibration = VALUES(n_calibration),
                        normalize = VALUES(normalize),
                        n_neighbors = VALUES(n_neighbors),
                        prototype_strategy = VALUES(prototype_strategy),
                        prototype_components = VALUES(prototype_components),
                        train_datasets = VALUES(train_datasets),
                        valid_dataset = VALUES(valid_dataset),
                        test_dataset = VALUES(test_dataset),
                        split_config_key = VALUES(split_config_key)
                """, (
                    config_key,
                    params.get('task', ''),
                    params['model_name'],
                    params.get('dataset_path', ''),
                    params.get('siamese_inference', ''),
                    params.get('run_tag', ''),
                    str(params.get('nsize', 224)),
                    params['fgsm'],
                    params['prototypes'],
                    params['npos'],
                    params['nneg'],
                    params['dloss'],
                    params.get('dist_fct', 'euclidean'),
                    params.get('classif_loss', 'triplet'),
                    params['n_calibration'],
                    float(accuracy),
                    float(mcc),
                    batch_entropy,
                    batch_nmi,
                    batch_ari,
                    params['normalize'],
                    str(params.get('n_neighbors', 1)),
                    proto_strat,
                    proto_comp,
                    train_datasets,
                    valid_dataset,
                    test_dataset,
                    split_key,
                    log_path,
                    artifact_id,
                    best_model_dir,
                    source_run_log_path,
                ))
            except mysql.connector.Error as e:
                # Backward-compatibility for legacy DB schemas.
                cursor.execute("""
                    REPLACE INTO best_models_registry
                    (model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss, n_calibration,
                     accuracy, mcc, normalize, n_neighbors, log_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    params['model_name'],
                    str(params.get('nsize', 224)),
                    params['fgsm'],
                    params['prototypes'],
                    params['npos'],
                    params['nneg'],
                    params['dloss'],
                    params.get('dist_fct', 'euclidean'),
                    params.get('classif_loss', 'triplet'),
                    params['n_calibration'],
                    float(accuracy),
                    float(mcc),
                    params['normalize'],
                    str(params.get('n_neighbors', 1)),
                    log_path
                ))
                print(f"Registry DB schema mismatch ({e}); wrote legacy registry row.")
            conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        print(f"❌ Registry DB error: {e}")


def append_run_metrics_db(row, run_log_path=""):
    """Append every run into DB history table (never overwrite)."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="y_user",
            password="password",
            database="results_db"
        )
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_runs_registry (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exp_id VARCHAR(255) NULL,
                trial_index VARCHAR(64) NULL,
                uuid VARCHAR(128) NULL,
                status VARCHAR(64) NULL,
                error TEXT NULL,
                task VARCHAR(128) NULL,
                run_tag VARCHAR(128) NULL,
                model_name VARCHAR(128) NULL,
                kind VARCHAR(64) NULL,
                variant VARCHAR(64) NULL,
                classif_loss VARCHAR(64) NULL,
                dloss VARCHAR(64) NULL,
                prototypes VARCHAR(64) NULL,
                fgsm VARCHAR(32) NULL,
                normalize VARCHAR(32) NULL,
                n_calibration VARCHAR(32) NULL,
                dist_fct VARCHAR(32) NULL,
                n_neighbors VARCHAR(32) NULL,
                n_negatives VARCHAR(32) NULL,
                train_datasets TEXT NULL,
                valid_dataset VARCHAR(255) NULL,
                test_dataset VARCHAR(255) NULL,
                split_config_key VARCHAR(1024) NULL,
                valid_mcc DOUBLE NULL,
                test_mcc DOUBLE NULL,
                valid_accuracy DOUBLE NULL,
                test_accuracy DOUBLE NULL,
                batch_entropy_norm DOUBLE NULL,
                batch_nmi DOUBLE NULL,
                batch_ari DOUBLE NULL,
                run_log_path TEXT NULL,
                INDEX idx_model_runs_lookup (task, model_name, classif_loss, dloss, fgsm, n_calibration)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            INSERT INTO model_runs_registry
            (exp_id, trial_index, uuid, status, error, task, run_tag, model_name, kind, variant,
             classif_loss, dloss, prototypes, fgsm, normalize, n_calibration, dist_fct, n_neighbors,
             n_negatives, train_datasets, valid_dataset, test_dataset, split_config_key,
             valid_mcc, test_mcc, valid_accuracy, test_accuracy, batch_entropy_norm, batch_nmi, batch_ari,
             run_log_path)
            VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row.get('exp_id', ''),
                str(row.get('trial_index', '') or ''),
                row.get('uuid', ''),
                row.get('status', ''),
                row.get('error', ''),
                row.get('task', ''),
                row.get('run_tag', ''),
                row.get('model_name', ''),
                row.get('kind', ''),
                row.get('variant', ''),
                row.get('classif_loss', ''),
                row.get('dloss', ''),
                row.get('prototypes', ''),
                str(row.get('fgsm', '') or ''),
                row.get('normalize', ''),
                str(row.get('n_calibration', '') or ''),
                row.get('dist_fct', ''),
                str(row.get('knn', '') or ''),
                str(row.get('n_negatives', '') or ''),
                row.get('train_datasets', ''),
                row.get('valid_dataset', ''),
                row.get('test_dataset', ''),
                row.get('split_config_key', ''),
                row.get('valid_mcc', None),
                row.get('test_mcc', None),
                row.get('valid_accuracy', None),
                row.get('test_accuracy', None),
                row.get('batch_entropy_norm', None),
                row.get('batch_nmi', None),
                row.get('batch_ari', None),
                run_log_path,
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        print(f"[Warning] Could not append run into model_runs_registry: {e}")

def register_all_params_to_tracking(run, args, params):
    """Register all hyperparameters (both fixed and optimized) to Tracking.
    
    Args:
        run: Tracking run object
        args: Command line arguments (fixed parameters)
        params: Optimized parameters dict
    """
    if run is None:
        return
    
    # Register fixed arguments
    run['args/model_name'] = args.model_name
    run['args/task'] = args.task
    run['args/device'] = args.device
    run['args/new_size'] = args.new_size
    run['args/n_epochs'] = args.n_epochs
    run['args/n_trials'] = args.n_trials
    run['args/early_stop'] = args.early_stop
    run['args/groupkfold'] = args.groupkfold
    run['args/is_stn'] = args.is_stn
    run['args/weighted_sampler'] = args.weighted_sampler
    run['args/bs'] = args.bs
    run['args/seed'] = args.seed
    
    # Fixed hyperparameters
    run['fixed/n_calibration'] = args.n_calibration
    run['fixed/normalize'] = args.normalize
    run['fixed/dloss'] = args.dloss
    run['fixed/classif_loss'] = args.classif_loss
    run['fixed/prototypes_to_use'] = args.prototypes_to_use
    run['fixed/fgsm'] = args.fgsm
    run['fixed/n_positives'] = args.n_positives
    run['fixed/n_negatives'] = args.n_negatives
    
    # Prototype parameters
    run['fixed/prototype_strategy'] = getattr(args, 'prototype_strategy', 'mean')
    run['fixed/prototype_components'] = getattr(args, 'prototype_components', 1)
    run['fixed/prototype_kind'] = getattr(args, 'prototype_kind', 'distance')
    run['fixed/kde_kernel'] = getattr(args, 'kde_kernel', 'gaussian')
    run['fixed/kde_bandwidth'] = getattr(args, 'kde_bandwidth', 'scott')
    run['fixed/siamese_inference'] = getattr(args, 'siamese_inference', 'knn')
    
    # Register optimized parameters
    for key, value in params.items():
        run[f'optimized/{key}'] = value
    
    # Register augmentation parameter
    run['fixed/n_aug'] = getattr(args, 'n_aug', 1)


def register_all_params_to_comet(experiment, args, params):
    """Register fixed and optimized parameters to Comet."""
    if experiment is None:
        return
    experiment.log_parameter('model_name', args.model_name)
    experiment.log_parameter('task', args.task)
    experiment.log_parameter('device', args.device)
    experiment.log_parameter('new_size', args.new_size)
    experiment.log_parameter('n_epochs', args.n_epochs)
    experiment.log_parameter('n_trials', args.n_trials)
    experiment.log_parameter('early_stop', args.early_stop)
    experiment.log_parameter('groupkfold', args.groupkfold)
    experiment.log_parameter('is_stn', args.is_stn)
    experiment.log_parameter('weighted_sampler', args.weighted_sampler)
    experiment.log_parameter('bs', args.bs)
    experiment.log_parameter('seed', args.seed)
    experiment.log_parameter('siamese_inference', getattr(args, 'siamese_inference', 'knn'))
    run_tag = getattr(args, 'run_tag', 'prod')
    experiment.log_parameter('run_tag', run_tag)
    experiment.log_parameter('is_test', int('test' in str(run_tag).lower()))
    for key, value in params.items():
        experiment.log_parameter(f'optimized/{key}', value)


def _tensor_to_numpy(tensor):
    detached = tensor.detach()
    if torch.is_floating_point(detached) and detached.dtype in (torch.float16, torch.bfloat16):
        detached = detached.float()
    return detached.cpu().numpy()


def _update_confmat_gpu(confmat, y_true, y_pred, n_classes):
    """Accumulate confusion-matrix counts directly on the active device."""
    if confmat is None:
        confmat = torch.zeros((n_classes, n_classes), device=y_true.device, dtype=torch.float32)
    y_true = y_true.long().view(-1)
    y_pred = y_pred.long().view(-1)
    valid = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    if valid.any():
        idx = y_true[valid] * n_classes + y_pred[valid]
        hist = torch.bincount(idx, minlength=n_classes * n_classes).view(n_classes, n_classes).float()
        confmat += hist
    return confmat


def _acc_mcc_from_confmat_gpu(confmat):
    """Compute multiclass accuracy and MCC from a confusion matrix on-device."""
    if confmat is None:
        return 0.0, 0.0
    s = confmat.sum()
    if float(s.item()) <= 0.0:
        return 0.0, 0.0
    c = torch.trace(confmat)
    acc = c / s

    row_sum = confmat.sum(dim=1)
    col_sum = confmat.sum(dim=0)
    cov_ytyp = c * s - torch.dot(row_sum, col_sum)
    cov_ypyp = s * s - torch.dot(col_sum, col_sum)
    cov_ytyt = s * s - torch.dot(row_sum, row_sum)

    denom = torch.sqrt(torch.clamp(cov_ypyp * cov_ytyt, min=0.0))
    if float(denom.item()) <= 0.0:
        mcc = torch.tensor(0.0, device=confmat.device)
    else:
        mcc = cov_ytyp / denom
    return float(acc.item()), float(mcc.item())


def _extract_best_values_as_scalars(best_vals):
    """Convert the epoch-list values dict into per-split scalar metrics.

    `best_vals` is a snapshot of the `values` dict taken at the epoch that
    achieved the best validation MCC.  Each entry is structured as:
        values[split][metric] = [epoch0_value, epoch1_value, ...]

    We store the **last** value in each list, which corresponds to the epoch
    that triggered the snapshot (i.e. the best-valid-MCC epoch).  This lets
    the analysis script read ``best_values['valid']['mcc']`` as a plain float.
    """
    if not isinstance(best_vals, dict):
        return {}

    def _last_scalar(seq):
        if isinstance(seq, (list, tuple)) and seq:
            val = seq[-1]
        else:
            val = seq
        try:
            f = float(val)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    result = {}
    for split in ('train', 'valid', 'test', 'all',
                  'valid_calibration', 'test_calibration', 'train_calibration'):
        split_dict = best_vals.get(split)
        if not isinstance(split_dict, dict):
            continue
        result[split] = {
            metric: _last_scalar(values)
            for metric, values in split_dict.items()
        }
    return result


def _resolve_strategy_label(classif_loss, knn):
    loss_txt = str(classif_loss).strip()
    if loss_txt and loss_txt.lower() not in {"nan", "none", "unknown"}:
        return loss_txt

    knn_txt = str(knn).strip()
    if knn_txt and knn_txt.lower() not in {"nan", "none", "unknown"}:
        try:
            return f"knn={int(float(knn_txt))}"
        except Exception:
            return f"knn={knn_txt}"

    return "unknown"


class TrainAE:
    def _append_completed_run_csv(self, params, best_vals, batch_metrics, run_uuid, final_status, final_error):
        """
        Append a row to the completed_runs CSV with all key metadata and metrics for this run.
        Writes to both the per-task CSV and the global completed_runs_metrics.csv in the root logs directory.
        Uses pandas for robust, consistent CSV writing.
        """
        from datetime import datetime

        run_tag = getattr(self.args, 'run_tag', 'PROD')
        task = getattr(self.args, 'task', 'notNormal')
        dataset = _dataset_path_segment(self.path).replace("/", "_")
        # Per-task CSV (legacy behavior)
        completed_csv = f'logs/progresses/{task}/{dataset}/{run_tag}_{task}_completed_runs_metrics.csv'
        os.makedirs(os.path.dirname(completed_csv), exist_ok=True)
        # Global CSV for all runs
        global_csv = 'completed_runs_metrics.csv'
        run_id = {
            'model_name': getattr(self.args, 'model_name', ''),
            'kind': getattr(self.args, 'kind', ''),
            'variant': getattr(self.args, 'variant', ''),
            'classif_loss': getattr(self.args, 'classif_loss', ''),
            'prototypes': getattr(self.args, 'prototypes_to_use', ''),
            'dloss': getattr(self.args, 'dloss', ''),
            'BER': getattr(self.args, 'BER', ''),
            'fgsm': getattr(self.args, 'fgsm', ''),
            'normalize': getattr(self.args, 'normalize', ''),
            'n_calibration': getattr(self.args, 'n_calibration', ''),
            'dist_fct': getattr(self.args, 'dist_fct', ''),
            'knn': params.get('n_neighbors', getattr(self.args, 'n_neighbors', '')),
            'n_negatives': getattr(self.args, 'n_negatives', ''),
        }
        strategy_label = _resolve_strategy_label(run_id['classif_loss'], run_id['knn'])
        split_config = _split_config_from_args(self.args)
        # Resolve lists to scalars
        best_vals_scalars = _extract_best_values_as_scalars(best_vals)
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'exp_id': getattr(self.args, 'exp_id', ''),
            'trial_index': getattr(self.args, 'trial_index', ''),
            'uuid': run_uuid,
            'status': final_status,
            'error': final_error,
            'task': task,
            'run_tag': run_tag,
            'model': run_id['model_name'],
            'model_name': run_id['model_name'],
            'kind': run_id['kind'],
            'variant': run_id['variant'],
            'loss': strategy_label,
            'classif_loss': strategy_label,
            'prototype': run_id['prototypes'],
            'prototypes': run_id['prototypes'],
            'dloss': run_id['dloss'],
            'BER': run_id['BER'],
            'fgsm': run_id['fgsm'],
            'normalize': run_id['normalize'],
            'n_calibration': run_id['n_calibration'],
            'dist_fct': run_id['dist_fct'],
            'knn': run_id['knn'],
            'n_negatives': run_id['n_negatives'],
            'train_datasets': split_config['train_datasets'],
            'valid_dataset': split_config['valid_dataset'],
            'test_dataset': split_config['test_dataset'],
            'split_config_key': split_config['split_config_key'],
            'retry_count': getattr(self.args, 'retry_count', ''),
            'train_mcc': best_vals_scalars.get('train', {}).get('mcc', best_vals.get('train_mcc', '')),
            'valid_mcc': best_vals_scalars.get('valid', {}).get('mcc', best_vals.get('valid_mcc', '')),
            'test_mcc': best_vals_scalars.get('test', {}).get('mcc', best_vals.get('test_mcc', '')),
            'train_auc': best_vals_scalars.get('train', {}).get('auc', best_vals.get('train_auc', '')),
            'valid_auc': best_vals_scalars.get('valid', {}).get('auc', best_vals.get('valid_auc', '')),
            'test_auc': best_vals_scalars.get('test', {}).get('auc', best_vals.get('test_auc', '')),
            'valid_accuracy': best_vals_scalars.get('valid', {}).get('acc', best_vals.get('valid_accuracy', '')),
            'test_accuracy': best_vals_scalars.get('test', {}).get('acc', best_vals.get('test_accuracy', '')),
            'launcher_retry_count': getattr(self.args, 'retry_count', ''),
            'launcher_failed_final': getattr(self.args, 'failed_final', ''),
            'source_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'batch_entropy_norm': (
                batch_metrics.get('batch_entropy_norm', batch_metrics.get('batch_entropy', ''))
                if batch_metrics else ''
            ),
            'batch_nmi': batch_metrics.get('batch_nmi', '') if batch_metrics else '',
            'batch_ari': batch_metrics.get('batch_ari', '') if batch_metrics else '',
        }
        # Write to per-task CSV (legacy)
        try:
            append_completed_run_metrics(completed_csv, row)
        except Exception as e:
            print(f"[Warning] Could not write to per-task completed_runs_metrics.csv: {e}")
        # Write to global CSV (new, unified manifest)
        try:
            append_completed_run_metrics(global_csv, row)
        except Exception as e:
            print(f"[Warning] Could not write to global completed_runs_metrics.csv: {e}")

        # Persist every run in DB history (append-only).
        append_run_metrics_db(row, run_log_path=str(getattr(self, 'complete_log_path', '') or ''))

    def __init__(self, args, path, load_tb=False, log_metrics=False, keep_models=True, log_inputs=True,
                 log_plots=False, log_tb=False, log_tracking=False, log_comet=False,
                 log_mlflow=True, log_dvclive=True, groupkfold=True):
        """

        Args:
            args: contains multiple arguments passed in the command line
            log_path: Path where the tensorboard logs are saved
            path: Path to the data (in .csv format)
            load_tb: If True, loads previous runs already saved
        """
        self.hparams_names = None
        self.best_acc = 0
        self.best_best_mcc = -1
        self.best_closs_final = np.inf
        self.complete_log_path = None
        # Always assign a UUID for every run (including siamese)
        if hasattr(args, 'uuid') and args.uuid:
            self.foldername = str(args.uuid)
        else:
            self.foldername = str(uuid.uuid4())
            args.uuid = self.foldername
        # Always set kind for all runs (cnn, siamese, etc.)
        if not hasattr(args, 'kind') or not args.kind:
            # Heuristic: if "triplet" or "arcface" or "cosine" in classif_loss, it's siamese; else cnn_mlp
            closs = getattr(args, 'classif_loss', '')
            if any(x in str(closs).lower() for x in ['triplet', 'arcface', 'cosine']):
                args.kind = 'siamese'
            else:
                args.kind = 'cnn_mlp'
        self.logged_inputs = False
        self.log_tb = log_tb
        self.log_tracking = log_tracking
        self.log_comet = log_comet
        self.log_mlflow = log_mlflow
        self.log_dvclive = log_dvclive
        self.dvclive_tracker = None
        self.args = args
        self.path = resolve_processed_dataset_path(path, 
                                                   getattr(args, 'train_datasets', None),
                                                   getattr(args, 'valid_dataset', None),
                                                   getattr(args, 'test_dataset', None))
        args.path = self.path

        self.amp_enabled = bool(int(getattr(self.args, 'amp', 1))) and str(getattr(self.args, 'device', '')).startswith('cuda')
        amp_dtype_name = str(getattr(self.args, 'amp_dtype', 'bf16')).lower()
        if amp_dtype_name not in ['bf16', 'fp16']:
            amp_dtype_name = 'bf16'
        self.amp_dtype = torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16
        self.use_grad_scaler = self.amp_enabled and self.amp_dtype == torch.float16
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

        # Toggle saving reproducibility artefacts (manifests + sanity samples)
        self.save_repro_artifacts = bool(int(getattr(self.args, 'save_repro_artifacts', 1)))
        # Keep optimization runs lean by default: explainability is opt-in.
        self.run_explainability = bool(int(getattr(self.args, 'run_explainability', 0)))
        self.shap_model = None

        # Ensure reproducible data splits and training behavior
        _set_global_seeds(getattr(self.args, 'seed', 1), deterministic=True)
        self.log_metrics = log_metrics
        self.log_plots = log_plots
        self.log_inputs = log_inputs
        self.load_tb = load_tb
        self.groupkfold = groupkfold
        # NOTE: self.foldername is intentionally NOT reset here; it was assigned
        # the run UUID near the top of __init__ and must be preserved.

        self.verbose = 1
        self.epoch = 0

        self.n_cats = None
        self.data = None
        self.unique_labels = None
        self.unique_batches = None
        self._batch_encoder = None
        self.triplet_loss = None
        self.triplet_dloss = None
        
        self.arcloss = None

        # Store all samples (including those previously removed)
        self.all_samples = {
            'inputs': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                       'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'labels': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                       'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'old_labels': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                           'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'names': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                      'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'cats': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                     'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'batches': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                        'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'encoded_values': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                               'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'subcenters': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                           'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
        }
        self.initial_nsamples = None
        self.prototypes = {'train': [], 'valid': [], 'test': [], 'calibration': [],
                           'train_calibration': [], 'valid_calibration': [], 'test_calibration': [], 'all': []}
        self.class_prototypes = {'train': [], 'valid': [], 'test': [], 'calibration': [],
                                 'train_calibration': [], 'valid_calibration': [], 'test_calibration': [], 'all': []}
        self.batch_prototypes = {'train': [], 'valid': [], 'test': [], 'calibration': [],
                                 'train_calibration': [], 'valid_calibration': [], 'test_calibration': [], 'all': []}
        self.combined_prototypes = {'train': [], 'valid': [], 'test': [], 'calibration': [],
                                    'train_calibration': [], 'valid_calibration': [], 'test_calibration': [], 'all': []}
        self.n_prototypes = {'train': None, 'valid': None, 'test': None, 'calibration': None,
                             'train_calibration': None, 'valid_calibration': None, 'test_calibration': None, 'all': None}
        self.best_params = {}
        self._sanity_saved = False
        self._file_events_csv = None
        self._epoch_timing_csv = None
        self._epoch_metrics_csv = None

    def _autocast_context(self):
        if self.amp_enabled:
            return torch.autocast(device_type='cuda', dtype=self.amp_dtype)
        return nullcontext()

    def _batch_progress_enabled(self):
        return bool(int(getattr(self.args, 'epoch_progress', 1))) and self.verbose > 0

    def _print_batch_progress(self, group, batch_idx, total_batches):
        if not self._batch_progress_enabled() or total_batches <= 0:
            return
        current = batch_idx + 1
        progress = current / total_batches
        remaining = max(total_batches - current, 0)
        bar_len = 24
        filled = int(bar_len * progress)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"\rEpoch {self.epoch + 1}/{getattr(self.args, 'n_epochs', 1)} [{group}] |{bar}| "
            f"{current}/{total_batches} ({progress * 100:5.1f}%) remaining:{remaining}",
            end="",
            flush=True,
        )

    def _finish_batch_progress(self, total_batches):
        if self._batch_progress_enabled() and total_batches > 0:
            print("")

    def _ensure_timing_files(self):
        if not getattr(self, 'complete_log_path', None):
            return
        os.makedirs(self.complete_log_path, exist_ok=True)

        if self._file_events_csv is None:
            self._file_events_csv = os.path.join(self.complete_log_path, 'file_events.csv')
            if not os.path.exists(self._file_events_csv):
                with open(self._file_events_csv, 'w') as f:
                    f.write('timestamp,label,path,size_bytes\n')

        if self._epoch_timing_csv is None:
            self._epoch_timing_csv = os.path.join(self.complete_log_path, 'epoch_timing.csv')
            if not os.path.exists(self._epoch_timing_csv):
                with open(self._epoch_timing_csv, 'w') as f:
                    f.write('epoch,gap_prev_s,deep_train_s,classifier_fit_s,eval_s,prototypes_s,loader_refresh_s,logging_s,best_update_s,total_s\n')

        if self._epoch_metrics_csv is None:
            self._epoch_metrics_csv = os.path.join(self.complete_log_path, 'epoch_metrics.csv')
            if not os.path.exists(self._epoch_metrics_csv):
                with open(self._epoch_metrics_csv, 'w') as f:
                    f.write('epoch,train_closs,train_dloss,valid_mcc,test_mcc,valid_acc,test_acc,lr,best_mcc\n')

    def _json_safe(self, value):
        import numpy as np
        # Recursively convert all values to JSON-serializable types
        if isinstance(value, dict):
            return {self._json_safe(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if isinstance(value, (datetime,)):
            return value.isoformat()
        # Handle other numpy scalar types (e.g., np.float32, np.float64)
        if hasattr(value, 'item') and callable(value.item):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _save_run_metadata(self, params):
        self._ensure_timing_files()
        if not getattr(self, 'complete_log_path', None):
            return
        metadata_path = os.path.join(self.complete_log_path, 'run_metadata.json')
        try:
            args_dict = {
                k: self._json_safe(v)
                for k, v in vars(self.args).items()
            }
            params_dict = {
                k: self._json_safe(v)
                for k, v in (params or {}).items()
            }
            args_dict = {
                k: self._json_safe(v)
                for k, v in vars(self.args).items()
            }
            params_dict = {
                k: self._json_safe(v)
                for k, v in (params or {}).items()
            }
            payload = {
                'created_at': datetime.now().isoformat(timespec='seconds'),
                'finished_at': None,
                'finished': 0,
                'run_finished': 0,
                'run_status': 'running',
                'error_message': None,
                'complete_log_path': self.complete_log_path,
                'foldername': self.foldername,
                'args': args_dict,
                'optimized_params': params_dict,
                'train_datasets': args_dict.get('train_datasets'),
                'valid_dataset': args_dict.get('valid_dataset'),
                'test_dataset': args_dict.get('test_dataset'),
                'preprocessing': self._json_safe(
                    image_preprocessing_manifest(
                        (
                            int(getattr(self.args, 'new_size', 224)),
                            int(getattr(self.args, 'new_size', 224)),
                        ),
                        normalize=getattr(self.args, 'normalize', 'no'),
                    )
                ),
            }
            with open(metadata_path, 'w') as f:
                json.dump(payload, f, indent=2)
            self._log_file_event('run_metadata_json', metadata_path)
        except Exception as e:
            print(f"Warning: could not save run metadata: {e}")

    def _update_run_metadata_status(self, run_status, error_message=None):
        if not getattr(self, 'complete_log_path', None):
            return
        metadata_path = os.path.join(self.complete_log_path, 'run_metadata.json')
        try:
            payload = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    payload = json.load(f)
            payload['run_status'] = str(run_status)
            payload['finished'] = 0 if str(run_status) == 'running' else 1
            payload['run_finished'] = 0 if str(run_status) == 'running' else 1
            payload['finished_at'] = datetime.now().isoformat(timespec='seconds') if payload['run_finished'] == 1 else None
            payload['error_message'] = str(error_message) if error_message not in [None, ''] else None
            with open(metadata_path, 'w') as f:
                json.dump(payload, f, indent=2)
            self._log_file_event('run_metadata_json', metadata_path)
        except Exception as e:
            print(f"Warning: could not update run metadata status: {e}")

    def _last_or_nan(self, values, group, key):
        try:
            seq = values.get(group, {}).get(key, [])
            if len(seq) == 0:
                return np.nan
            return seq[-1]
        except Exception:
            return np.nan

    def _log_epoch_metrics(self, epoch, values):
        self._ensure_timing_files()
        if self._epoch_metrics_csv is None:
            return
        train_closs = self._last_or_nan(values, 'train', 'closs')
        train_dloss = self._last_or_nan(values, 'train', 'dloss')
        valid_mcc = self._last_or_nan(values, 'valid', 'mcc')
        test_mcc = self._last_or_nan(values, 'test', 'mcc')
        valid_acc = self._last_or_nan(values, 'valid', 'acc')
        test_acc = self._last_or_nan(values, 'test', 'acc')
        try:
            lr_val = self.scheduler.get_last_lr()[0]
        except Exception:
            lr_val = np.nan
        with open(self._epoch_metrics_csv, 'a') as f:
            f.write(
                f"{epoch + 1},{train_closs},{train_dloss},{valid_mcc},{test_mcc},"
                f"{valid_acc},{test_acc},{lr_val},{self.best_mcc}\n"
            )

    def _save_run_summary(self, params, best_vals, start_time, run_status='completed', error_message=None):
        self._ensure_timing_files()
        if not getattr(self, 'complete_log_path', None):
            return
        import tempfile, os
        summary_path = os.path.join(self.complete_log_path, 'run_summary.json')
        tmp_path = summary_path + '.tmp'
        try:
            payload = {
                'start_time': self._json_safe(start_time),
                'end_time': datetime.now().isoformat(timespec='seconds'),
                'duration_seconds': float((datetime.now() - start_time).total_seconds()),
                'run_status': str(run_status),
                'finished': 0 if str(run_status) == 'running' else 1,
                'run_finished': 0 if str(run_status) == 'running' else 1,
                'error_message': self._json_safe(error_message) if error_message else None,
                'best_mcc': self._json_safe(self.best_mcc),
                'best_acc': self._json_safe(self.best_acc),
                'best_closs': self._json_safe(self.best_closs),
                # Store per-split scalar metrics so the analysis script can read
                # best_values['valid']['mcc'] and best_values['test']['mcc'] as floats.
                # best_vals is the values-dict snapshot at the best epoch, where each
                # split['metric'] is a list of epoch values; we take the last entry (which
                # corresponds to the epoch that triggered the best-valid-MCC update).
                'best_values': self._json_safe(
                    _extract_best_values_as_scalars(best_vals)
                ),
                'split_config': self._json_safe(_split_config_from_args(self.args)),
                'preprocessing': self._json_safe(
                    image_preprocessing_manifest(
                        (
                            int(getattr(self.args, 'new_size', 224)),
                            int(getattr(self.args, 'new_size', 224)),
                        ),
                        normalize=getattr(self.args, 'normalize', 'no'),
                    )
                ),
                'calibration_manifest_path': self._json_safe(getattr(self.args, 'calibration_manifest_path', '')),
                'batch_metrics': self._json_safe(getattr(self, 'batch_metrics', {})),
                'params': {k: self._json_safe(v) for k, v in (params or {}).items()},
                'artifacts': {
                    'valid_predictions_csv': os.path.join(self.complete_log_path, 'valid_predictions.csv'),
                    'test_predictions_csv': os.path.join(self.complete_log_path, 'test_predictions.csv'),
                    'epoch_timing_csv': os.path.join(self.complete_log_path, 'epoch_timing.csv'),
                    'epoch_metrics_csv': os.path.join(self.complete_log_path, 'epoch_metrics.csv'),
                    'file_events_csv': os.path.join(self.complete_log_path, 'file_events.csv'),
                },
            }
            with open(tmp_path, 'w') as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, summary_path)
            self._log_file_event('run_summary_json', summary_path)
        except Exception as e:
            print(f"Warning: could not save run summary: {e}")

    def _log_file_event(self, label, path):
        self._ensure_timing_files()
        if self._file_events_csv is None:
            return
        ts = datetime.now().isoformat(timespec='seconds')
        size = -1
        try:
            size = os.path.getsize(path)
        except Exception:
            pass
        with open(self._file_events_csv, 'a') as f:
            f.write(f'{ts},{label},{path},{size}\n')
        print(f'FileEvent|time={ts}|label={label}|path={path}|size={size}')

    def _log_epoch_timing(self, epoch, timing):
        self._ensure_timing_files()
        if self._epoch_timing_csv is None:
            return
        with open(self._epoch_timing_csv, 'a') as f:
            f.write(
                f"{epoch + 1},{timing.get('gap_prev_s', 0.0):.3f},{timing.get('train_s', 0.0):.3f},"
                f"{timing.get('classifier_fit_s', 0.0):.3f},{timing.get('eval_s', 0.0):.3f},{timing.get('prototypes_s', 0.0):.3f},"
                f"{timing.get('loader_refresh_s', 0.0):.3f},{timing.get('logging_s', 0.0):.3f},"
                f"{timing.get('best_update_s', 0.0):.3f},{timing.get('total_s', 0.0):.3f}\n"
            )
        print(
            f"EpochTiming|epoch={epoch + 1}|gap_prev={timing.get('gap_prev_s', 0.0):.2f}s|"
            f"deep_train={timing.get('train_s', 0.0):.2f}s|classifier_fit={timing.get('classifier_fit_s', 0.0):.2f}s|"
            f"eval={timing.get('eval_s', 0.0):.2f}s|"
            f"prototypes={timing.get('prototypes_s', 0.0):.2f}s|loader={timing.get('loader_refresh_s', 0.0):.2f}s|"
            f"logging={timing.get('logging_s', 0.0):.2f}s|best_update={timing.get('best_update_s', 0.0):.2f}s|"
            f"total={timing.get('total_s', 0.0):.2f}s"
        )
        print(
            f"DeepVsClassifierTiming|epoch={epoch + 1}|"
            f"deep_layers_train_s={timing.get('train_s', 0.0):.2f}|"
            f"embedding_classifier_fit_s={timing.get('classifier_fit_s', 0.0):.2f}"
        )

    def _save_raw_inputs_manifest(self, data_getter, path):
        """Persist raw input metadata (name, label, batch, full path) for all splits."""
        train_exception = None
        try:
            os.makedirs(path, exist_ok=True)
            rows = []
            for group in ['all', 'train', 'valid', 'test']:
                names = data_getter.data['names'].get(group, [])
                labels = data_getter.data['labels'].get(group, [])
                batches = data_getter.data['batches'].get(group, [])
                for n, l, b in zip(names, labels, batches):
                    rows.append({
                        'group': group,
                        'name': n,
                        'label': l,
                        'batch': b,
                        'path': os.path.join(self.path, str(n))
                    })
            if rows:
                pd.DataFrame(rows).to_csv(os.path.join(path, 'raw_inputs_manifest.csv'), index=False)
        except Exception as e:
            print(f"Warning: could not save raw inputs manifest: {e}")

    def _save_split_sanity_samples(self, path):
        """Save the first few train/valid samples (arrays + PNG) and record basic pixel stats."""
        # if self._sanity_saved:
        #     return
        snap_dir = os.path.join(path, 'sanity_samples')
        os.makedirs(snap_dir, exist_ok=True)

        meta_rows = []
        for group in ['train', 'valid']:
            names = self.data['names'].get(group, [])
            labels = self.data['labels'].get(group, [])
            batches = self.data['batches'].get(group, [])
            inputs = self.data['inputs'].get(group, [])
            count = min(5, len(names))
            for i in range(count):
                arr = inputs[i]
                # Capture stats before any clipping/scaling
                n_pixels = int(arr.size)
                pixel_min = float(np.min(arr)) if arr.size else 0.0
                pixel_max = float(np.max(arr)) if arr.size else 0.0
                name_i = str(names[i])
                label_i = labels[i]
                batch_i = batches[i]
                safe_name = name_i.replace('/', '_').replace('\\', '_')
                np.save(os.path.join(snap_dir, f"{group}_{i}_{safe_name}.npy"), arr)
                try:
                    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
                    img.save(os.path.join(snap_dir, f"{group}_{i}_{safe_name}.png"))
                except Exception as e:
                    print(f"Warning: could not save sanity PNG for {group}/{name_i}: {e}")
                meta_rows.append({
                    'group': group,
                    'index': i,
                    'name': name_i,
                    'label': label_i,
                    'batch': batch_i,
                    'path': os.path.join(self.path, name_i),
                    'array_file': f"sanity_samples/{group}_{i}_{safe_name}.npy",
                    'image_file': f"sanity_samples/{group}_{i}_{safe_name}.png",
                    'n_pixels': n_pixels,
                    'pixel_min': pixel_min,
                    'pixel_max': pixel_max,
                })

        if meta_rows:
            try:
                pd.DataFrame(meta_rows).to_csv(os.path.join(snap_dir, 'samples_metadata.csv'), index=False)
            except Exception as e:
                print(f"Warning: could not save sanity samples metadata: {e}")

        self._sanity_saved = True

    def _ensure_raw_manifest_from_current(self, path):
        """If raw_inputs_manifest.csv is missing, rebuild it from in-memory self.data."""
        manifest_path = os.path.join(path, 'raw_inputs_manifest.csv')
        if os.path.exists(manifest_path):
            return
        try:
            rows = []
            for group in ['all', 'train', 'valid', 'test']:
                names = self.data['names'].get(group, []) if self.data else []
                labels = self.data['labels'].get(group, []) if self.data else []
                batches = self.data['batches'].get(group, []) if self.data else []
                for n, l, b in zip(names, labels, batches):
                    rows.append({
                        'group': group,
                        'name': n,
                        'label': l,
                        'batch': b,
                        'path': os.path.join(self.path, str(n))
                    })
            if rows:
                os.makedirs(path, exist_ok=True)
                pd.DataFrame(rows).to_csv(manifest_path, index=False)
        except Exception as e:
            print(f"Warning: could not rebuild raw input manifest: {e}")

    def save_wrong_classif_imgs(self, run, models, lists, preds, names, group):
        # Save Grad-CAM / SHAP artefacts for valid and test predictions
        for label in self.unique_labels:
            base_wrong = f'{self.complete_log_path}/wrong_classif/{label}/{group}'
            base_correct = f'{self.complete_log_path}/correct_classif/{label}/{group}'
            os.makedirs(f'{base_wrong}/gradients_shap', exist_ok=True)
            os.makedirs(f'{base_wrong}/imgs', exist_ok=True)
            os.makedirs(f'{base_wrong}/knn_shap', exist_ok=True)
            os.makedirs(f'{base_wrong}/grad_cam', exist_ok=True)
            os.makedirs(f'{base_correct}/gradients_shap', exist_ok=True)
            os.makedirs(f'{base_correct}/knn_shap', exist_ok=True)
            os.makedirs(f'{base_correct}/grad_cam', exist_ok=True)

        cats = np.concatenate(lists[group]['cats'])
        preds = np.concatenate(lists[group]['preds'])
        names = np.concatenate(lists[group]['names'])
        wrong = 0
        correct = 0
        prototype_store = self.class_prototypes.get('train', {})
        if not isinstance(prototype_store, dict):
            prototype_store = {}

        for i, (cat, pred_idx, name) in enumerate(zip(cats, np.argmax(preds, 1), names)):
            pred_label = self.unique_labels[pred_idx]
            proto = prototype_store.get(pred_label)
            if cat != pred_idx:
                wrong += 1
                try:
                    img = Image.open(f'{self.args.path_original}/{name}')
                    img.save(f'{self.complete_log_path}/wrong_classif/{self.unique_labels[cat]}/{group}/imgs/{wrong}_{name}.png')
                except Exception as e:
                    print(f'Failed to save image {name}: {e}')
                log_shap_images_gradients(
                    models,
                    i,
                    lists,
                    group,
                    f'{self.complete_log_path}/wrong_classif/{self.unique_labels[cat]}/{group}/',
                    f'{wrong}_{name}',
                )
                if proto is not None:
                    log_grad_cam_similarity(
                        models['cnn'],
                        i,
                        lists,
                        group,
                        f'{self.complete_log_path}/wrong_classif/{self.unique_labels[cat]}/{group}/grad_cam',
                        f'{wrong}_{name}',
                        proto,
                        device=self.args.device,
                    )
            else:
                log_shap_images_gradients(
                    models,
                    i,
                    lists,
                    group,
                    f'{self.complete_log_path}/correct_classif/{self.unique_labels[cat]}/{group}/',
                    f'{correct}_{name}',
                )
                if proto is not None:
                    log_grad_cam_similarity(
                        models['cnn'],
                        i,
                        lists,
                        group,
                        f'{self.complete_log_path}/correct_classif/{self.unique_labels[cat]}/{group}/grad_cam',
                        f'{correct}_{name}',
                        proto,
                        device=self.args.device,
                    )
                correct += 1

        for label in self.unique_labels:
            base_wrong = f'{self.complete_log_path}/wrong_classif/{label}/{group}'
            base_correct = f'{self.complete_log_path}/correct_classif/{label}/{group}'
            run[f'wrong_classif/{label}/{group}'].upload_files(f'{base_wrong}/imgs')
            run[f'wrong_classif/{label}/{group}'].upload_files(f'{base_wrong}/gradients_shap')
            run[f'wrong_classif/{label}/{group}'].upload_files(f'{base_wrong}/knn_shap')
            run[f'wrong_classif/{label}/{group}'].upload_files(f'{base_wrong}/grad_cam')
            run[f'correct_classif/{label}/{group}'].upload_files(f'{base_correct}/gradients_shap')
            run[f'correct_classif/{label}/{group}'].upload_files(f'{base_correct}/knn_shap')
            run[f'correct_classif/{label}/{group}'].upload_files(f'{base_correct}/grad_cam')

    def set_prototypes(self, group, list1):
        self.prototypes.set_prototypes(group, list1)
        self.combined_prototypes = self.prototypes.prototypes
        self.class_prototypes = self.prototypes.class_prototypes
        self.batch_prototypes = self.prototypes.batch_prototypes
        self.n_prototypes = self.prototypes.n_prototypes
        self.means = self.prototypes.means

    def _append_batch_outputs(self, lists, group, domain_np, label_np, enc_np, names, old_labels, dpreds_np=None, add_preds=False):
        enc_np = np.asarray(enc_np)
        label_np = np.asarray(label_np).reshape(-1)
        domain_idx = np.asarray([np.argmax(d) for d in np.asarray(domain_np)]).reshape(-1)
        names_np = np.asarray(list(names) if isinstance(names, (list, tuple)) else names, dtype=object).reshape(-1)
        old_labels_np = np.asarray(_tensor_to_numpy(old_labels) if torch.is_tensor(old_labels) else old_labels, dtype=object).reshape(-1)

        n = min(len(enc_np), len(label_np), len(domain_idx), len(names_np), len(old_labels_np))
        if len(enc_np) != len(label_np) or len(enc_np) != len(domain_idx):
            print(
                f"[BatchAlign] group={group} encodings={len(enc_np)} labels={len(label_np)} "
                f"domains={len(domain_idx)} names={len(names_np)} old_labels={len(old_labels_np)}; using {n}"
            )
        if n <= 0:
            return

        enc_np = enc_np[:n]
        label_np = label_np[:n]
        domain_idx = domain_idx[:n].astype(int)
        names_np = names_np[:n]
        old_labels_np = old_labels_np[:n]

        lists[group]['domains'] += [self._batch_encoder.inverse_transform(domain_idx)]
        lists[group]['classes'] += [label_np]
        if add_preds and dpreds_np is not None:
            lists[group]['preds'] += [np.asarray(dpreds_np)[:n]]
        lists[group]['encoded_values'] += [enc_np]
        lists[group]['names'] += [names_np]
        lists[group]['cats'] += [label_np]
        lists[group]['labels'] += [np.array([self.unique_labels[int(x)] for x in label_np])]
        lists[group]['old_labels'] += [old_labels_np]

    def make_samples_weights(self):
        self.n_batches = len(
            set(np.concatenate((
            self.data['batches']['train'],
            self.data['batches']['valid'],
            self.data['batches']['test']
            )))
        )
        self.class_weights = {
            label: 1 / (len(np.where(label == self.data['labels']['train'])[0]) /
                        self.data['labels']['train'].shape[0])
            for label in self.unique_labels if
            label in self.data['labels']['train']}
        self.unique_unique_labels = list(self.class_weights.keys())

        self.samples_weights = {
            group: [self.class_weights[label] for name, label in
                    zip(self.data['names'][group],
                        self.data['labels'][group])] if group == 'train' else [
                1 for name, label in
                zip(self.data['names'][group], self.data['labels'][group])] for group in
            ['train', 'valid', 'test']}
        self.n_cats = len(self.unique_labels)
        self.scaler = None
        if self.args.classif_loss == 'ce':
            self.celoss = nn.CrossEntropyLoss()
        elif self.args.classif_loss == 'hinge':
            self.celoss = nn.MultiMarginLoss(p=2, margin=self.params['margin'])
        if self.args.dloss == 'DANN':
            self.dceloss = nn.CrossEntropyLoss()

    def make_triplet_loss(self, dist_fct, params):
        if self.args.n_positives + self.args.n_negatives > 2:
            self.triplet_loss = TupletLoss(margin=params['margin'])
            self.triplet_dloss = TupletLoss(margin=params['dmargin'])
        elif self.args.classif_loss == 'softmax_contrastive':
            self.triplet_loss = SoftmaxContrastiveLoss(
                temperature=params['smoothing'], distance_metric=params['dist_fct']
            )
        elif self.args.classif_loss in ['triplet']:
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(
                distance_function=dist_fct, margin=params['margin'], swap=False
            )
        if self.args.dloss in ['inverseTriplet']:
            self.triplet_dloss = nn.TripletMarginWithDistanceLoss(
                distance_function=dist_fct, margin=params['dmargin'], swap=False
            )
        elif self.args.dloss in ['inverse_softmax_contrastive']:
            self.triplet_dloss = SoftmaxContrastiveLoss(
                temperature=params['smoothing'], distance_metric=params['dist_fct']
            )
        elif self.args.dloss in ['none']:
            self.triplet_dloss = None

    def _save_inference_classifier(self, model_dir=None):
        """Persist siamese embedding classifier artifact when configured.

        Currently only saves the MLP head classifier. KNN remains on-demand.
        """
        try:
            mode = str(getattr(self.args, 'siamese_inference', 'knn')).strip().lower()
            clf = getattr(self, 'inference_classifier', None)
            clf_kind = str(getattr(self, 'inference_classifier_kind', '')).strip().lower()
            if mode != 'mlp_head' or clf is None or clf_kind != 'mlp_head':
                return
            target_dir = model_dir if model_dir else self.complete_log_path
            if not target_dir:
                return
            os.makedirs(target_dir, exist_ok=True)
            mlp_path = os.path.join(target_dir, 'mlp_head.pkl')
            with open(mlp_path, 'wb') as f:
                pickle.dump(clf, f)
            self._log_file_event('mlp_head_pkl', mlp_path)
        except Exception as e:
            print(f"Warning: failed to save inference classifier artifact: {e}")


    def train(self, params):
        """

        Args:
            params: Contains the hyperparameters to be optimized

        Returns:
            best_closs: The best classification loss on the valid set

        """
        start_time = datetime.now()
        train_exception = None
        self.params = params
        self.best_mcc = -1.0
        self.best_acc = 0.0
        self.best_closs = np.inf
        self.best_loss = np.inf
        self.complete_log_path = None
        self.foldername = None
        try:
            theoretical_gpu_required_mb = None
            values, best_values, _, best_traces = get_empty_dicts()
            best_vals = values
            # Setup necessary objects for training (data, model, losses)
            data_getter, loaders = self.setup_training_objects(params)
            loggers = {'cm_logger': LogConfusionMatrix(self.complete_log_path)}
            print(f'See results using: tensorboard --logdir={self.complete_log_path} --port=6006')

            # Persist run configuration and metadata early
            self._save_run_metadata(params)
            self.dvclive_tracker = DVCLiveTracker(
                self.args,
                params,
                self.complete_log_path,
                self.foldername,
                enabled=self.log_dvclive,
            )

            if self.log_tracking:
                run = create_tracking_run(self.args, params, self.complete_log_path, self.foldername)
            else:
                run = None
            if self.log_comet:
                from ..logging.loggings import make_comet_logger

                comet_logger = make_comet_logger(self.args, self.params, self.complete_log_path, self.foldername)
            else:
                comet_logger = None

            loggers['logger_cm'] = _make_summary_writer(f'{self.complete_log_path}/cm', enabled=self.log_tb)
            loggers['logger'] = _make_summary_writer(f'{self.complete_log_path}/traces', enabled=self.log_tb)
            # sceloss = nn.CrossEntropyLoss(label_smoothing=params['smoothing'])
            optimizer_model = get_optimizer(self.model, params['lr'], params['wd'], params['optimizer_type'])
            self.scheduler = ReduceLROnPlateau(optimizer_model, mode='max', factor=0.1, patience=10)  # Reduce on plateau
            if torch.cuda.is_available() and str(self.args.device).startswith('cuda'):
                reset_gpu_peak(self.args.device)
                theoretical_gpu_required_mb = estimate_theoretical_gpu_required_mb(
                    self.model,
                    int(getattr(self.args, 'bs', 32)),
                    int(getattr(self.args, 'new_size', 64)),
                )
                emit_gpu_telemetry(
                    'start',
                    theoretical_gpu_required_mb=f'{theoretical_gpu_required_mb:.2f}',
                )

            early_stop_counter = 0
            warmup = 0
            prev_epoch_end = None
            best_lists = {}
            for epoch in range(0, self.args.n_epochs):
                self.epoch = epoch
                epoch_start = time.perf_counter()
                gap_prev = 0.0 if prev_epoch_end is None else max(0.0, epoch_start - prev_epoch_end)
                best_update_s = 0.0
                if early_stop_counter >= self.args.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.', epoch)
                    break
                lists, traces = get_empty_traces()
                self.model.train()
                if self.args.dloss == 'no':
                    train_groups = ['train']
                else:
                    train_groups = ['all', 'train']
                t_train_start = time.perf_counter()
                for group in train_groups:
                    if group == 'train' and epoch < warmup:
                        continue
                    _, best_lists[group], _ = self.loop(group, optimizer_model, params['gamma'], loaders[group], lists, traces)
                train_s = time.perf_counter() - t_train_start
                if epoch < warmup:
                    values['all']['dloss'] += [np.mean(traces['all']['dloss'])]
                    add_to_tracking(values, run)
                    prev_epoch_end = time.perf_counter()
                    continue
                t_log_start = time.perf_counter()
                media_every = int(getattr(self.args, 'epoch_media_every', 0))
                should_log_media = media_every > 0 and ((epoch + 1) % media_every == 0)
                data_img = f'{self.complete_log_path}/{group}_data.png'
                adv_img = f'{self.complete_log_path}/{group}_adv_data.png'
                # Save images in run only on configured epochs to avoid expensive I/O/network.
                if should_log_media and self.log_tracking and run is not None and os.path.exists(data_img):
                    run['example_data'].upload(data_img)
                    if self.args.fgsm and os.path.exists(adv_img):
                        run['example_adv_data'].upload(adv_img)
                if should_log_media and self.log_comet and comet_logger is not None and os.path.exists(data_img):
                    comet_logger.log_image(data_img, name=f'{group}_data_epoch_{epoch}')
                    if self.args.fgsm and os.path.exists(adv_img):
                        comet_logger.log_image(adv_img, name=f'{group}_adv_data_epoch_{epoch}')
                logging_s = time.perf_counter() - t_log_start
                self.model.eval()
                t_eval_start = time.perf_counter()
                classifier_fit_s = 0.0
                self._embedding_classifier_cache = {}
                self.set_prototypes('train', best_lists['train'])
                # Run predict to accumulate encodings first
                _, best_lists['valid'], _, knn = self.predict('valid', loaders['valid'], lists, traces)
                classifier_fit_s += float(getattr(self, '_last_classifier_fit_s', 0.0) or 0.0)

                _, best_lists['test'], _, knn = self.predict('test', loaders['test'], lists, traces)
                classifier_fit_s += float(getattr(self, '_last_classifier_fit_s', 0.0) or 0.0)

                eval_s = time.perf_counter() - t_eval_start
                # put the best lists together. keys are all only in one dict
                best_lists = {**best_lists['train'], **best_lists['valid'], **best_lists['test']}
                # After encodings exist, update prototypes
                t_proto_start = time.perf_counter()
                self.set_prototypes('valid', best_lists)
                self.set_prototypes('test', best_lists)
                if 'all' in train_groups:
                    self.set_prototypes('all', best_lists)
                self.set_means_classes(best_lists)
                prototypes_s = time.perf_counter() - t_proto_start
                prototypes = {
                    'combined': self.combined_prototypes,
                    'class': self.class_prototypes,
                    'batch': self.batch_prototypes
                }
                t_loader_start = time.perf_counter()
                loaders = get_images_loaders(data=self.data,
                                             batch_encoder=self._batch_encoder,
                                            random_recs=self.args.random_recs,
                                            weighted_sampler=self.args.weighted_sampler,
                                            is_transform=params['is_transform'],
                                            samples_weights=self.samples_weights,
                                            epoch=epoch+1,  # loaders for Next epoch
                                            unique_labels=self.unique_labels,
                                            triplet_dloss=self.args.dloss, bs=self.args.bs,
                                            prototypes_to_use=self.args.prototypes_to_use,
                                            prototypes=prototypes,
                                            size=self.args.new_size,
                                            normalize=self.args.normalize,
                                            n_aug=params.get('n_aug', 1),
                                            num_workers=getattr(self.args, 'num_workers', 0),
                                            )
                loader_refresh_s = time.perf_counter() - t_loader_start
                values = log_traces(traces, values)
                self.scheduler.step(values['valid']['mcc'][-1])
                
                # Report intermediate value to Optuna for pruning
                if hasattr(self, 'trial') and self.trial is not None:
                    try:
                        valid_mcc = float(values['valid']['mcc'][-1])
                        self.trial.report(valid_mcc, step=epoch)
                        if self.trial.should_prune():
                            print(f"Trial pruned at epoch {epoch} due to low validation MCC: {valid_mcc:.4f}")
                            raise optuna.TrialPruned(f"Validation MCC {valid_mcc:.4f} fell below median at epoch {epoch}")
                    except optuna.TrialPruned:
                        raise
                    except Exception:
                        pass  # Don't fail training if trial.report() has issues
                
                print('Current LR:', self.scheduler.get_last_lr())
                if self.log_tb:
                    try:
                        add_to_logger(values, loggers['logger'], epoch)
                    except:
                        print("Problem with add_to_logger!")
                if self.log_tracking:
                    add_to_tracking(values, run)
                if self.log_mlflow:
                    add_to_mlflow(values, epoch)
                if self.log_comet:
                    from ..logging.loggings import add_to_comet

                    add_to_comet(values, comet_logger, epoch)
                if self.log_dvclive and self.dvclive_tracker is not None:
                    self.dvclive_tracker.log_epoch(
                        values,
                        epoch,
                        {
                            "timing/deep_train_s": train_s,
                            "timing/classifier_fit_s": classifier_fit_s,
                            "timing/eval_s": eval_s,
                            "timing/prototypes_s": prototypes_s,
                            "timing/loader_refresh_s": loader_refresh_s,
                            "timing/logging_s": logging_s,
                        },
                    )
                if values['valid']['mcc'][-1] > self.best_mcc:
                    t_best_start = time.perf_counter()
                    print(f"Best Classification Mcc Epoch {epoch}, "
                            f"Acc: test: {values['test']['acc'][-1]}, valid: {values['valid']['acc'][-1]}"
                            f"Mcc: test: {values['test']['mcc'][-1]}, valid: {values['valid']['mcc'][-1]}"
                            f"Classification "
                            f" valid loss: {values['valid']['closs'][-1]},"
                            f" test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                    self.best_mcc = values['valid']['mcc'][-1]
                    model_path = f'{self.complete_log_path}/model.pth'
                    torch.save(self.model.state_dict(), model_path)
                    self._log_file_event('model_checkpoint', model_path)
                    self.save_prototypes(self.complete_log_path)
                    self.save_samples_weights(self.complete_log_path)
                    self.log_predictions(best_lists, run, comet_logger, 0)

                    loggers['cm_logger'].add(best_lists)
                    best_vals = values.copy()
                    best_vals['rec_loss'] = self.best_loss
                    # best_vals['dom_loss'] = self.best_dom_loss
                    # best_vals['dom_acc'] = self.best_dom_acc
                    early_stop_counter = 0
                    print(values['valid']['mcc'][-1], self.best_mcc)
                    if int(getattr(self.args, 'heavy_best_analysis', 0)):
                        self.make_encoded_values()
                        self.cluster_and_visualize(run, best_lists, ['train', 'valid', 'test'],
                                                    n_clusters=self.n_cats * self.n_batches)
                    # try:
                    #     assert len(data_getter.data['labels']['all']) == self.initial_nsamples
                    # except Exception as e:
                    #     print("Problem with data integrity before removing noisy samples")
                        # Write error to error file in run directory
                    #     error_file = os.path.join(self.complete_log_path, 'data_integrity_error_before_removal.txt')
                    #     with open(error_file, 'w') as ef:
                    #         ef.write(f"Exception in data integrity check before removing noisy samples:\n{str(e)}\n")
                    #         import traceback
                    #         traceback.print_exc(file=ef)
                    #     exit(1)
                    if self.args.remove_noisy_samples:
                        try:
                            data_getter.remove_noisy_samples(self.complete_log_path)
                            assert len(data_getter.data['labels']['all']) == self.initial_nsamples
                        except Exception as e:
                            print("Problem with removing noisy samples")
                            # Write error to error file in run directory
                            error_file = os.path.join(self.complete_log_path, 'remove_noisy_samples_error.txt')
                            with open(error_file, 'w') as ef:
                                ef.write(f"Exception in remove_noisy_samples:\n{str(e)}\n")
                                import traceback
                                traceback.print_exc(file=ef)
                            exit(1)
                    best_update_s = time.perf_counter() - t_best_start
                else:
                    # if epoch > self.warmup:
                    early_stop_counter += 1

                if values['valid']['acc'][-1] > self.best_acc:
                    print(f"Best Classification Acc Epoch {epoch}, "
                            f"Acc: {values['test']['acc'][-1]}"
                            f"Mcc: {values['test']['mcc'][-1]}"
                            f"Classification "
                            f" valid loss: {values['valid']['closs'][-1]},"
                            f" test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")

                    self.best_acc = values['valid']['acc'][-1]

                if values['train']['closs'][-1] < self.best_closs:
                    print(f"Best Classification Loss Epoch {epoch}, "
                            f"Acc: {values['test']['acc'][-1]} "
                            f"Mcc: {values['test']['mcc'][-1]} "
                            f"Classification "
                            f"valid loss: {values['valid']['closs'][-1]}, "
                            f"test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                    self.best_closs = values['train']['closs'][-1]

                self._log_epoch_metrics(epoch, values)

                epoch_total = time.perf_counter() - epoch_start
                self._log_epoch_timing(epoch, {
                    'gap_prev_s': gap_prev,
                    'train_s': train_s,
                    'classifier_fit_s': classifier_fit_s,
                    'eval_s': eval_s,
                    'prototypes_s': prototypes_s,
                    'loader_refresh_s': loader_refresh_s,
                    'logging_s': logging_s,
                    'best_update_s': best_update_s,
                    'total_s': epoch_total,
                })
                prev_epoch_end = time.perf_counter()

            early_stop_counter = 0
            if self.args.n_calibration > 0:
                print("\n========== ENTERING CALIBRATION PHASE ==========")
                print(f"Calibration epochs: {self.args.n_epochs} | n_calibration={self.args.n_calibration}")
                for epoch in range(0, self.args.n_epochs):
                    values, _, _, _ = get_empty_dicts()  # Pas élégant
                    self.epoch = epoch
                    if early_stop_counter >= self.args.early_stop:
                        if self.verbose > 0:
                            print('EARLY STOPPING.', epoch)
                        break
                    lists, traces = get_empty_traces()
                    self.model.train()
                    self.model.to(self.args.device)
                    
                    best_lists = {}
                    for group in ['all', 'train', 'calibration']:
                        _, best_lists[group], _ = self.loop_calibration(group, optimizer_model, params['gamma'], loaders[group], lists, traces)
                        self.set_prototypes(group, best_lists[group])
                    self.model.eval()
                    for group in ["valid", "test"]:
                        _, best_lists[group], _, _ = self.predict(group, loaders[group], lists, traces)
                        self.set_prototypes(group, best_lists[group])
                    # put the best lists together. keys are all only in one dict
                    best_lists = {**best_lists['all'], **best_lists['train'], **best_lists['valid'], **best_lists['test'], **best_lists['calibration']}

                    # self.set_prototypes('train', best_lists)
                    # self.set_prototypes('valid', best_lists)
                    # self.set_prototypes('test', best_lists)
                    # self.set_prototypes('all', best_lists)
                    # self.set_prototypes('calibration', best_lists)

                    # self.model.eval()
                    # for group in ["valid", "test"]:
                    #     _, best_lists, _, _ = self.predict(group, loaders[group], lists, traces)
                    prototypes = {
                        'combined': self.combined_prototypes,
                        'class': self.class_prototypes,
                        'batch': self.batch_prototypes
                    }
                    loaders = get_images_loaders(data=self.data,
                                                batch_encoder=self._batch_encoder,
                                                random_recs=self.args.random_recs,
                                                weighted_sampler=self.args.weighted_sampler,
                                                is_transform=params['is_transform'],
                                                samples_weights=self.samples_weights,
                                                epoch=epoch+1,
                                                unique_labels=self.unique_labels,
                                                triplet_dloss=self.args.dloss, bs=self.args.bs,
                                                prototypes_to_use=self.args.prototypes_to_use,
                                                prototypes=prototypes,
                                                size=self.args.new_size,
                                                normalize=self.args.normalize,
                                                n_aug=params.get('n_aug', 1),
                                                num_workers=getattr(self.args, 'num_workers', 0),
                                                )

                    values = log_traces(traces, values)

                    # Add calib to the keys of values that are not calibration
                    keys = list(values.keys())
                    for key in keys:
                        if key in ['train', 'valid', 'test']:
                            values[key + '_calibration'] = values[key]
                            values.pop(key, None)
                            best_lists[key + '_calibration'] = best_lists[key]
                            best_lists.pop(key, None)
                      
                    if self.log_tb:
                        try:
                            add_to_logger(values, loggers['logger'], epoch)
                        except:
                            print("Problem with add_to_logger!")
                    if self.log_tracking:
                        add_to_tracking(values, run)
                    if self.log_mlflow:
                        add_to_mlflow(values, epoch)
                    if self.log_comet:
                        from ..logging.loggings import add_to_comet

                        add_to_comet(values, comet_logger, epoch)
                    if values['valid_calibration']['mcc'][-1] > self.best_mcc:
                        print(f"Best Classification Mcc Epoch {epoch}, "
                                f"Acc: test: {values['test_calibration']['acc'][-1]}, valid: {values['valid_calibration']['acc'][-1]}"
                                f"Mcc: test: {values['test_calibration']['mcc'][-1]}, valid: {values['valid_calibration']['mcc'][-1]}"
                                f"Classification "
                                f" valid loss: {values['valid_calibration']['closs'][-1]},"
                                f" test loss: {values['test_calibration']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                        self.best_mcc = values['valid_calibration']['mcc'][-1]
                        torch.save(self.model.state_dict(), f'{self.complete_log_path}/model.pth')
                        self.save_prototypes(self.complete_log_path)
                        self.save_samples_weights(self.complete_log_path)
                        best_vals = values.copy()
                        best_vals['rec_loss'] = self.best_loss
                        # After training, perform clustering and visualize
                        # Store all samples before clustering
                        if int(getattr(self.args, 'heavy_best_analysis', 0)):
                            self.make_encoded_values()
                            self.cluster_and_visualize(run, best_lists, 
                                                       ['train', 'valid', 'test'], 
                                                       n_clusters=self.n_cats * self.n_batches, update_lists=False)
                        early_stop_counter = 0
                    else:
                        print('missed', values['valid_calibration']['mcc'][-1], self.best_mcc)
                        early_stop_counter += 1
                    if values['valid_calibration']['acc'][-1] > self.best_acc:
                        print(f"Best Classification Acc Epoch {epoch}, "
                                f"Acc: {values['test_calibration']['acc'][-1]}"
                                f"Mcc: {values['test_calibration']['mcc'][-1]}"
                                f"Classification "
                                f" valid loss: {values['valid_calibration']['closs'][-1]},"
                                f" test loss: {values['test_calibration']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                        self.best_acc = values['valid_calibration']['acc'][-1]
                    if values['train_calibration']['closs'][-1] < self.best_closs:
                        print(f"Best Classification Loss Epoch {epoch}, "
                                f"Acc: {values['test_calibration']['acc'][-1]} "
                                f"Mcc: {values['test_calibration']['mcc'][-1]} "
                                f"Classification "
                                f"valid loss: {values['valid_calibration']['closs'][-1]}, "
                                f"test loss: {values['test_calibration']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                        self.best_closs = values['valid_calibration']['closs'][-1]

        except optuna.TrialPruned as e:
            train_exception = e
            print(f"Training pruned: {e}")
            if self.log_dvclive and self.dvclive_tracker is not None:
                self.dvclive_tracker.log_final({"run/pruned": 1})
                self.dvclive_tracker.end()
            if self.log_tracking and run is not None:
                try:
                    run.stop()
                except Exception as stop_exc:
                    print(f"Warning: failed to stop Tracking run after pruning: {stop_exc}")
            self._update_run_metadata_status('pruned', error_message=str(e))
            raise
        except Exception as e:
            train_exception = e
            print(f"Training failed: {e}")
            if self.log_dvclive and self.dvclive_tracker is not None:
                self.dvclive_tracker.log_final({"run/failed": 1})
                self.dvclive_tracker.end()
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available() and str(self.args.device).startswith('cuda'):
                stats = gpu_memory_stats(self.args.device)
                missing_mb = None
                if theoretical_gpu_required_mb is not None and stats.get('free_mb') is not None:
                    missing_mb = max(0.0, float(theoretical_gpu_required_mb) - float(stats['free_mb']))
                oom_error = 'out of memory' in str(e).lower()
                emit_gpu_telemetry(
                    'oom' if oom_error else 'failure',
                    theoretical_gpu_required_mb=(f'{theoretical_gpu_required_mb:.2f}' if theoretical_gpu_required_mb is not None else None),
                    actual_peak_gpu_mb=(f"{stats.get('actual_peak_gpu_mb', 0.0):.2f}" if stats else None),
                    oom_missing_gpu_mb=(f'{missing_mb:.2f}' if missing_mb is not None else None),
                    oom_gpu_free_at_failure_mb=(f"{stats.get('free_mb', 0.0):.2f}" if stats else None),
                    oom_gpu_used_at_failure_mb=(f"{stats.get('used_mb', 0.0):.2f}" if stats else None),
                )
            self._update_run_metadata_status('failed', error_message=str(e))

        lists, traces = get_empty_traces()
        # values, _, _, _ = get_empty_dicts()  # Pas élégant
        # Loading best model that was saved during training
        model_path = f'{self.complete_log_path}/model.pth'
        if not os.path.exists(model_path):
            if train_exception is not None:
                if isinstance(train_exception, optuna.TrialPruned):
                    raise train_exception
                self._update_run_metadata_status('failed', error_message=str(train_exception))
                raise RuntimeError(
                    f"Training aborted before checkpoint creation: {model_path}"
                ) from train_exception
            self._update_run_metadata_status('failed', error_message=f"Checkpoint missing: {model_path}")
            raise FileNotFoundError(f"Checkpoint missing: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.args.device))
        self.model.eval()
        if self.run_explainability and self.shap_model is not None:
            # Optional second model for explainability methods.
            self.shap_model.load_state_dict(torch.load(model_path, map_location=self.args.device))
            self.shap_model.eval()
        print("\n========== FINAL EVALUATION PASS (BEST CHECKPOINT) ==========")
        print("Running train loop snapshot + train/valid/test predictions...")
        best_lists = {}
        with torch.no_grad():
            _, _, _ = self.loop('train', optimizer_model, params['gamma'], loaders['train'], lists, traces)
            for group in ["train", "valid", "test"]:
                _, best_lists[group], traces, knn = self.predict(group, loaders[group], lists, traces)
            batch_metrics = get_batch_metrics(lists)
            for key, value in batch_metrics.items():
                traces[key] = value
            # Store batch metrics for database saving
            self.batch_metrics = batch_metrics
            
            print("\n========== BATCH EFFECTS FINAL REPORT (train+valid+test) ==========")
            print(f"Batch Entropy (mixing): {batch_metrics.get('batch_entropy', 0.0):.4f} (ideal: 1.0)")
            print(f"Batch Entropy Loss: {batch_metrics.get('batch_entropy_loss', 0.0):.4f}")
            print(f"Batch KNN Accuracy: {batch_metrics.get('batch_acc_loss', 0.0):.4f} (ideal: chance level)")
            print(f"Batch KNN MCC: {batch_metrics.get('batch_mcc_loss', 0.0):.4f} (ideal: 0.0)")
            print(f"Batch NMI: {batch_metrics.get('batch_nmi', 0.0):.4f}")
            print(f"Batch ARI: {batch_metrics.get('batch_ari', 0.0):.4f}")
            print(f"Batch Silhouette: {batch_metrics.get('batch_silhouette', 0.0):.4f}")
            print("==================================================================\n")

        # Persist fitted embedding classifier artifact (when using mlp_head)
        self._save_inference_classifier(self.complete_log_path)

        best_lists = {**best_lists['train'], **best_lists['valid'], **best_lists['test']}

        # Logging every model is taking too much resources and it makes it quite complicated to get information when
        # Too many runs have been made. This will make the notebook so much easier to work with
        if self.best_mcc > self.best_best_mcc:
            self.save_best_run(run, comet_logger, params, best_vals, best_lists)
            # Save best model parameters and hyperparamters into a csv, including task

        if self.log_tracking and run is not None:
            try:
                run.stop()
            except Exception as e:
                print(f"Warning: failed to stop Tracking run: {e}")

        final_status = 'completed' if train_exception is None else 'completed_with_recovered_error'
        final_error = None if train_exception is None else str(train_exception)
        self._update_run_metadata_status(final_status, error_message=final_error)
        if torch.cuda.is_available() and str(self.args.device).startswith('cuda'):
            stats = gpu_memory_stats(self.args.device)
            emit_gpu_telemetry(
                'end',
                theoretical_gpu_required_mb=(f'{theoretical_gpu_required_mb:.2f}' if theoretical_gpu_required_mb is not None else None),
                actual_peak_gpu_mb=(f"{stats.get('actual_peak_gpu_mb', 0.0):.2f}" if stats else None),
            )
        run_uuid = getattr(self.args, 'uuid', self.foldername)
        print(f"uuid: {run_uuid} Final status: {final_status}")
        self._save_run_summary(params, best_vals, start_time, run_status=final_status, error_message=final_error)
        # Append to completed_runs CSV for robust manifest/metrics tracking
        self._append_completed_run_csv(params, best_vals, getattr(self, 'batch_metrics', None), run_uuid, final_status, final_error)
        if self.log_dvclive and self.dvclive_tracker is not None:
            final_metrics = {
                "best_mcc": self.best_mcc,
                "best_acc": self.best_acc,
                "best_closs": self.best_closs,
                "duration_seconds": float((datetime.now() - start_time).total_seconds()),
            }
            for key, value in (getattr(self, "batch_metrics", {}) or {}).items():
                final_metrics[f"batch/{key}"] = value
            self.dvclive_tracker.log_final(
                final_metrics,
                [
                    os.path.join(self.complete_log_path, "model.pth"),
                    os.path.join(self.complete_log_path, "run_metadata.json"),
                    os.path.join(self.complete_log_path, "run_summary.json"),
                    os.path.join(self.complete_log_path, "epoch_metrics.csv"),
                ],
            )
            self.dvclive_tracker.end()
        return self.best_mcc

    def save_best_run(self, run, comet_logger, params, best_vals, best_lists=None):
        # Log the final K value used to Tracking if available
        if self.log_tracking and run is not None:
            final_k = params.get('n_neighbors', getattr(self.args, 'n_neighbors', 5))
            run["final_n_neighbors"] = final_k
        if self.log_comet and comet_logger is not None:
            comet_logger.log_parameter("final_n_neighbors", params.get('n_neighbors', getattr(self.args, 'n_neighbors', 5)))
        
        # Create csv file if not exists
        models_csv_path = f'logs/best_models/{self.args.task}/models.csv'
        log_dir = f'logs/best_models/{self.args.task}'

        # Ensure reproducibility artifacts are present before copying to best_models
        if self.save_repro_artifacts:
            self._ensure_raw_manifest_from_current(f'logs/best_models/{self.args.task}/')
            self._save_split_sanity_samples(f'logs/best_models/{self.args.task}/')

        # Prefer per-run normalize flag from params when available; fallback to args
        # normalize_val = str(params.get("normalize", getattr(self.args, "normalize", "no")))
        # Keep args in sync so downstream paths/registry use the same value
        # self.args.normalize = normalize_val
        dist_fct_val = str(params.get("dist_fct", getattr(self.args, "dist_fct", "none")))
        self.args.dist_fct = dist_fct_val
        task_val = str(self.args.task) if getattr(self.args, "task", None) else "notNormal"
        # Ensure prototypes_to_use doesn't have "prototypes_" prefix for consistent formatting
        proto_val = str(self.args.prototypes_to_use)
        if proto_val.startswith("prototypes_"):
            proto_val = proto_val[len("prototypes_"):]
        split_config = _split_config_from_args(self.args)
        split_csv = {
            **split_config,
            "train_datasets": split_config["train_datasets"].replace(",", ";"),
            "split_config_key": split_config["split_config_key"].replace(",", ";"),
        }
        dataset_segment = _dataset_path_segment(self.path)
        # Use deduplicated split_segment for all path constructions
        split_config = _split_config_from_args(self.args)
        split_segment = split_config["split_segment"]
        # Only append split_segment if it is not already part of dataset_segment
        # if split_segment and split_segment not in dataset_segment:
        #     dataset_path_with_split = f"{dataset_segment}/{split_segment}"
        # else:
        dataset_path_with_split = dataset_segment


        line_found = False
        if not os.path.exists(models_csv_path):
            os.makedirs(log_dir, exist_ok=True)
            with open(models_csv_path, 'w') as f:
                f.write('valid_mcc,model,path,n_neighbors,nsize,fgsm,n_calibration,loss,dloss,dist_fct,prototype,n_positives,n_negatives,normalize,task,train_datasets,valid_dataset,test_dataset,split_config_key,complete_log_path\n')
                f.write(f'{self.best_mcc},{self.args.model_name},{dataset_segment},{params["n_neighbors"]},nsize{self.args.new_size},fsgm{self.args.fgsm},ncal{self.args.n_calibration},' \
                    f'{self.args.classif_loss},{self.args.dloss},{dist_fct_val},prototypes_{proto_val},' \
                    f'npos{self.args.n_positives},nneg{self.args.n_negatives},{self.args.normalize},{task_val},' \
                    f'{split_csv["train_datasets"]},{split_csv["valid_dataset"]},{split_csv["test_dataset"]},{split_csv["split_config_key"]},{self.complete_log_path}\n')

        else:
            with open(models_csv_path, 'r') as f:
                lines = f.readlines()
            # lines = pd.read_csv(models_csv_path).to_csv(index=False).splitlines(keepends=True)  # Reformat lines to ensure consistent formatting
            for i, line in enumerate(lines[1:]):  # skip header
                line_parts = line.strip().split(',')
                if len(line_parts) < 16:
                    raise RuntimeError(
                        f"Malformed row in {models_csv_path} at data row {i + 2}: "
                        f"expected >=16 comma-separated fields, got {len(line_parts)}. "
                        f"row={line!r}"
                    )
                try:
                    line_mcc = float(line_parts[0])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid valid_mcc in {models_csv_path} at data row {i + 2}: "
                        f"value={line_parts[0]!r}. row={line!r}"
                    ) from exc
                compare_parts = line_parts[1:-1]
                expected_parts = [self.args.model_name, dataset_segment, str(params["n_neighbors"]), f'nsize{self.args.new_size}', f'fsgm{self.args.fgsm}', f'ncal{self.args.n_calibration}', 
                                       str(self.args.classif_loss), str(self.args.dloss), dist_fct_val,
                                       f'prototypes_{proto_val}',
                                       f'npos{self.args.n_positives}', f'nneg{self.args.n_negatives}',
                                       self.args.normalize,
                                       task_val]
                if len(compare_parts) >= 18:
                    expected_parts += [
                        split_csv["train_datasets"],
                        split_csv["valid_dataset"],
                        split_csv["test_dataset"],
                        split_csv["split_config_key"],
                    ]
                if compare_parts == expected_parts:
                    if self.best_mcc >= line_mcc:
                        lines[i + 1] = f'{self.best_mcc},{self.args.model_name},{dataset_segment},{params["n_neighbors"]},nsize{self.args.new_size},fgsm{self.args.fgsm},' \
                                        f'ncal{self.args.n_calibration},{self.args.classif_loss},{self.args.dloss},{dist_fct_val},prototypes_{proto_val},' \
                                        f'npos{self.args.n_positives},nneg{self.args.n_negatives},{self.args.normalize},{task_val},' \
                                        f'{split_csv["train_datasets"]},{split_csv["valid_dataset"]},{split_csv["test_dataset"]},{split_csv["split_config_key"]},{self.complete_log_path}\n'
                    else:
                        lines[i + 1] = line
                    line_found = True
                    break

            if not line_found:
                lines.append(f'{self.best_mcc},{self.args.model_name},{dataset_segment},{params["n_neighbors"]},'\
                             f'nsize{self.args.new_size},fgsm{self.args.fgsm},'\
                             f'ncal{self.args.n_calibration},{self.args.classif_loss},' \
                             f'{self.args.dloss},{dist_fct_val},prototypes_{proto_val},' \
                             f'npos{self.args.n_positives},nneg{self.args.n_negatives},' \
                             f'{self.args.normalize},{task_val},' \
                             f'{split_csv["train_datasets"]},{split_csv["valid_dataset"]},{split_csv["test_dataset"]},{split_csv["split_config_key"]},{self.complete_log_path}\n')

            # Update CSV with consistent formatting
            for i, line in enumerate(lines[1:], start=1):
                line_parts = line.strip().split(',')
                if len(line_parts) > 5:
                    # Rebuild line with consistent prototypes naming
                    if line_parts[10].startswith('prototypes_'):
                        line_parts[10] = 'prototypes_' + line_parts[10][len('prototypes_'):]
                    lines[i] = ','.join(line_parts) + '\n' if line_parts[-1] != '\n' else ','.join(line_parts)

            with open(models_csv_path, 'w') as f:
                f.writelines(lines)

        # Standardize parameter names for CSV and paths
        proto_val = str(self.args.prototypes_to_use)
        if proto_val.startswith("prototypes_"):
            proto_val = proto_val[len("prototypes_"):]
        
        # Ensure prototypes_to_use doesn't have "prototypes_" prefix already
        proto_val = str(self.args.prototypes_to_use)
        if proto_val.startswith("prototypes_"):
            proto_val = proto_val[len("prototypes_"):]

        # Get split config for path construction
        split_config = _split_config_from_args(self.args)
        n_neighbors_val = params.get('n_neighbors', getattr(self.args, 'n_neighbors', 5))
        params_str = f'{dataset_path_with_split}/nsize{self.args.new_size}/fgsm{self.args.fgsm}/ncal{self.args.n_calibration}/{self.args.classif_loss}/' \
            f'{self.args.dloss}/prototypes_{proto_val}/' \
            f'npos{self.args.n_positives}/nneg{self.args.n_negatives}/' \
            f'protoagg_{getattr(self.args, "prototype_strategy", "mean")}_{getattr(self.args, "prototype_components", 1)}/' \
            f'norm{self.args.normalize}/' \
            f'dist_{dist_fct_val}/' \
            f'knn{n_neighbors_val}'
        model_dir = f'logs/best_models/{self.args.task}/{self.args.model_name}/{params_str}'

        try:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
            copy_start = time.perf_counter()
            shutil.copytree(self.complete_log_path, model_dir)
            copy_sec = time.perf_counter() - copy_start
            print(f"BestModelSync|seconds={copy_sec:.2f}|src={self.complete_log_path}|dst={model_dir}")
        except Exception as e:
            print(f"Error copying model directory: {e}")

        # Persist encoded sample vectors for downstream reuse (e.g., KNN)
        self.save_encoded_vectors(self.complete_log_path, best_lists)
        self.save_encoded_vectors(model_dir, best_lists)

        self.best_best_mcc = self.best_mcc
        self.keep_best_run(run, comet_logger, best_vals)
        self.save_prototypes(model_dir)
        self.save_samples_weights(model_dir)
        self._save_inference_classifier(model_dir)

        # --- Update best model registry in MySQL ---
        registry_params = {
            'task': self.args.task,
            'model_name': self.args.model_name,
            'dataset_path': _dataset_path_segment(self.path),
            'siamese_inference': getattr(self.args, 'siamese_inference', ''),
            'run_tag': getattr(self.args, 'run_tag', ''),
            'nsize': self.args.new_size,
            'fgsm': self.args.fgsm,
            'prototypes': self.args.prototypes_to_use,
            'npos': self.args.n_positives,
            'nneg': self.args.n_negatives,
            'dloss': self.args.dloss,
            'dist_fct': params.get('dist_fct', getattr(self.args, 'dist_fct', 'euclidean')),
            'classif_loss': self.args.classif_loss,
            'n_calibration': self.args.n_calibration,
            'normalize': self.args.normalize,
            'n_neighbors': params.get('n_neighbors', getattr(self.args, 'n_neighbors', 5)),
            'prototype_strategy': params.get('prototype_strategy', getattr(self.args, 'prototype_strategy', 'mean')),
            'prototype_components': params.get('prototype_components', getattr(self.args, 'prototype_components', 1)),
            'train_datasets': split_config['train_datasets'],
            'valid_dataset': split_config['valid_dataset'],
            'test_dataset': split_config['test_dataset'],
            'split_config_key': split_config['split_config_key'],
        }
        
        # Get batch metrics if available
        batch_metrics_dict = None
        if hasattr(self, 'batch_metrics') and self.batch_metrics:
            # Extract validation batch metrics
            batch_metrics_dict = {
                'batch_entropy_norm': self.batch_metrics.get('valid_batch_entropy_norm'),
                'batch_nmi': self.batch_metrics.get('valid_batch_nmi'),
                'batch_ari': self.batch_metrics.get('valid_batch_ari'),
            }
        
        update_best_model_registry(
            registry_params,
            accuracy=best_vals.get('valid_acc', self.best_acc),
            mcc=self.best_mcc,
            log_path=self.complete_log_path,
            batch_metrics=batch_metrics_dict,
            source_run_log_path=self.complete_log_path,
            best_model_dir=model_dir,
        )

    # TODO Decide if valuable class method
    def save_prototypes(self, model_dir):
        path = f'{model_dir}/prototypes.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.prototypes, f)
        self._log_file_event('prototypes_pkl', path)

    # TODO Decide if valuable class method
    def save_samples_weights(self, model_dir):
        path = f'{model_dir}/sample_weights.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.samples_weights, f)
        self._log_file_event('sample_weights_pkl', path)

    def save_encoded_vectors(self, model_dir, best_lists):
        """Save encoded vectors (embeddings) for each group (train, valid, test) as .npz files.
        
        This enables downstream KNN or other ML algorithms to work with saved encodings
        without needing to recompute them through the model.
        
        Args:
            model_dir: Directory path where to save the encodings
            best_lists: Dictionary containing 'train', 'valid', 'test' keys with encoded_values, 
                       names, labels, and cats arrays
        """
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            for group in ['train', 'valid', 'test']:
                if group not in best_lists:
                    continue
                
                group_data = best_lists[group]
                
                # Concatenate list of arrays into single arrays
                embeddings = np.concatenate(group_data.get('encoded_values', []))
                names = np.concatenate(group_data.get('names', []))
                labels = np.concatenate(group_data.get('labels', []))
                cats = np.concatenate(group_data.get('cats', []))
                
                # Save as .npz file
                npz_path = os.path.join(model_dir, f'{group}_encodings.npz')
                np.savez(
                    npz_path,
                    embeddings=embeddings,
                    names=names,
                    labels=labels,
                    cats=cats
                )
                print(f"Saved {group} encodings to {npz_path}")
                self._log_file_event(f'{group}_encodings_npz', npz_path)
        except Exception as e:
            print(f"Error saving encoded vectors: {e}")

    def keep_best_run(self, run, comet_logger, best_vals):
        # Tracking/Comet can be disabled; do not fail an otherwise successful trial.
        self.best_params_tracking = {}
        self.best_params = {}
        if self.log_tracking and run is not None:
            try:
                self.best_params_tracking = get_best_params(run, best_vals)
            except Exception as e:
                print(f"Warning: failed to fetch Tracking best params: {e}")
        if self.log_comet and comet_logger is not None:
            try:
                self.best_params = get_best_params_comet(comet_logger, best_vals)
            except Exception as e:
                print(f"Warning: failed to fetch Comet best params: {e}")

    def set_arcloss(self):
        if self.args.classif_loss == 'arcface':
            self.arcloss = ArcFaceLoss(
                get_n_features(self.args.model_name),
                len(self.unique_labels),
                s=30,
                m=0.5,
                device=self.args.device
            )
        elif self.args.classif_loss == 'arcfacewithsubcenters':
            self.arcloss = ArcFaceLossWithSubcenters(
                get_n_features(self.args.model_name),
                len(self.unique_labels),
                s=30,
                m=0.5,
                num_subcenters=self.n_batches * 2,
                device=self.args.device
            )
        elif self.args.classif_loss == 'ArcFaceLossWithHSM':
            self.arcloss = ArcFaceLossWithHSM(
                get_n_features(self.args.model_name),
                len(self.unique_labels),
                s=30,
                m=0.5,
                num_subcenters=self.n_batches * 2,
                device=self.args.device
            )
        elif self.args.classif_loss == 'arcfacewithsubcentersHSM':
            self.arcloss = ArcFaceLossWithSubcentersHSM(
                get_n_features(self.args.model_name),
                len(self.unique_labels),
                s=30,
                m=0.5,
                num_subcenters=self.n_batches * 2,
                device=self.args.device
            )


    def make_encoded_values(self, groups=['train', 'valid', 'test']):
        """Encode all samples with optional augmentation for train set.
        
        Semantics:
        - n_aug = 0: Only original images per sample
        - n_aug >= 1: Original + n_aug augmented copies per image (train only)
        - valid/test always use base transform without augmentation
        """
        # Get n_aug from params; defaults to 0 (originals only)
        n_aug = max(0, int(self.params.get('n_aug', 0))) if hasattr(self, 'params') else 0
        
        base_transform = get_base_transform()
        knn_aug_transform = get_knn_augmentation_transform(self.args.new_size)

        for group in groups:
            # Collect encoded samples and mirrored metadata (labels/names/etc.)
            all_samples = []
            all_subcenters = []
            group_labels, group_old_labels = [], []
            group_names, group_cats, group_batches = [], [], []

            for i in range(0, len(self.all_samples['inputs'][group]), self.args.bs):
                transformed_samples = []
                labels_batch, old_labels_batch = [], []
                names_batch, cats_batch, batches_batch = [], [], []

                batch = self.all_samples['inputs'][group][i:i + self.args.bs]

                for j in range(len(batch)):
                    # Determine number of repeats: 1 original + n_aug extra for train; 1 for others
                    repeats = 1 + max(0, n_aug) if group == 'train' else 1
                    
                    for r in range(repeats):
                        # r == 0 -> original; r > 0 -> augmented (when n_aug > 0)
                        if group == 'train' and r > 0:
                            sample = knn_aug_transform(batch[j])
                        else:
                            sample = base_transform(batch[j])
                        
                        transformed_samples.append(sample)

                        labels_batch.append(self.all_samples['labels'][group][i + j])
                        old_labels_batch.append(self.all_samples['old_labels'][group][i + j])
                        names_batch.append(self.all_samples['names'][group][i + j])
                        cats_batch.append(self.all_samples['cats'][group][i + j])
                        batches_batch.append(self.all_samples['batches'][group][i + j])

                if not transformed_samples:
                    continue

                transformed_tensor = torch.stack(transformed_samples).to(self.args.device)

                # Use transformed_samples for encoding
                try:
                    encoded = self.model(transformed_tensor)[0]
                except Exception as e:
                    print(f"Error occurred: {e}")
                    exit()

                if 'arcfacewithsubcenters' in self.args.classif_loss:
                    _, subcenters = self.arcloss(
                        encoded,
                        torch.Tensor(cats_batch).long().to(self.args.device)
                    )
                    all_subcenters.append(_tensor_to_numpy(subcenters))

                all_samples.append(_tensor_to_numpy(encoded))

                # Mirror metadata for each augmented sample
                group_labels.extend(labels_batch)
                group_old_labels.extend(old_labels_batch)
                group_names.extend(names_batch)
                group_cats.extend(cats_batch)
                group_batches.extend(batches_batch)

            if all_samples:
                self.all_samples['encoded_values'][group] = np.concatenate(all_samples)
                self.all_samples['labels'][group] = np.array(group_labels)
                self.all_samples['old_labels'][group] = np.array(group_old_labels)
                self.all_samples['names'][group] = np.array(group_names)
                self.all_samples['cats'][group] = np.array(group_cats)
                self.all_samples['batches'][group] = np.array(group_batches)
            else:
                self.all_samples['encoded_values'][group] = np.array([])

            if 'arcfacewithsubcenters' in self.args.classif_loss:
                self.all_samples['subcenters'][group] = np.concatenate(all_subcenters) if all_subcenters else np.array([])

    def log_predictions(self, best_lists, run=None, comet_logger=None, step=0):
        cats, labels, preds, scores, names = [{'train': [], 'valid': [], 'test': []} for _ in range(5)]
        for group in ['valid', 'test']:
            if len(best_lists[group]['preds']) == 0:
                raise ValueError(
                    f"Cannot log {group} predictions: best_lists[{group!r}]['preds'] is empty. "
                    f"This usually means the split was evaluated with loop() instead of predict(), "
                    f"or the {group} loader produced no batches. "
                    f"Split size={len(self.data['labels'].get(group, []))}, "
                    f"loader_batches={len(self.loaders.get(group, [])) if hasattr(self, 'loaders') else 'unknown'}."
                )
            cats[group] = np.concatenate(best_lists[group]['cats'])
            labels[group] = np.concatenate(best_lists[group]['labels'])
            # Use KNN outputs directly (assume best_lists[group]['preds'] contains KNN probabilities or one-hot)
            scores[group] = np.concatenate(best_lists[group]['preds'])
            preds[group] = scores[group].argmax(1)
            names[group] = np.concatenate(best_lists[group]['names'])
            observed_counts = self._cat_counts(cats[group])
            if len(observed_counts) < 2:
                raise ValueError(
                    f"Cannot log {group} ROC AUC because only one class is present in observed predictions. "
                    f"Observed counts from best_lists[{group!r}]['cats']: {observed_counts}. "
                    f"Data split counts: {self._label_counts(self.data['labels'][group])}"
                )
            auc_value = self._roc_auc_present_classes(cats[group], scores[group], group)
            score_colnames = [f'probs_{l}' for l in self.unique_labels]
            colnames = ['label'] + score_colnames + ['pred', 'name']
            df = pd.DataFrame(
                np.concatenate((labels[group].reshape(-1, 1),
                               scores[group],
                               np.array([self.unique_labels[x] for x in preds[group]]).reshape(-1, 1),
                               names[group].reshape(-1, 1)), 1),
                columns=colnames
            )
            df.to_csv(f'{self.complete_log_path}/{group}_predictions.csv', index=False)
            if self.log_tracking and run is not None:
                run[f"{group}_predictions"].track_files(f'{self.complete_log_path}/{group}_predictions.csv')
                run[f'{group}_AUC'] = auc_value
                try:
                    save_roc_curve(
                        scores[group],
                        to_categorical(cats[group], len(self.unique_labels)),
                        self.unique_labels,
                        f'{self.complete_log_path}/{group}_roc_curve.png',
                        binary=False,
                        acc=metrics.accuracy_score(cats[group], np.array(scores[group].argmax(1))),
                        mlops='mlflow',
                        epoch=step,
                        logger=None
                    )
                    run[f'{group}_ROC'].upload(f'{self.complete_log_path}/{group}_roc_curve.png')
                except ValueError as exc:
                    print(f"Warning: skipped {group} ROC curve plot: {exc}")
            if self.log_comet and comet_logger is not None:
                comet_logger.log_metric(f'{group}_AUC', auc_value, step=step)

                # TODO ADD OTHER FINAL METRICS HERE
                comet_logger.log_metric(f'{group}_acc',
                            metrics.accuracy_score(cats[group], scores[group].argmax(1)),
                            step=step)
                comet_logger.log_metric(f'{group}_mcc',
                            MCC(cats[group], scores[group].argmax(1)),
                            step=step)

                try:
                    save_roc_curve(
                        scores[group],
                        to_categorical(cats[group], len(self.unique_labels)),
                        self.unique_labels,
                        f'{self.complete_log_path}/{group}_roc_curve.png',
                        binary=False,
                        acc=metrics.accuracy_score(cats[group], np.array(scores[group].argmax(1))),
                        mlops='comet',
                        epoch=step,
                        logger=comet_logger
                    )
                except ValueError as exc:
                    print(f"Warning: skipped {group} ROC curve plot: {exc}")
            if self.log_mlflow:
                mlflow.log_metric(f'{group}_AUC', auc_value, step=step)

                # TODO ADD OTHER FINAL METRICS HERE
                mlflow.log_metric(f'{group}_acc',
                            metrics.accuracy_score(cats[group], scores[group].argmax(1)),
                            step=step)
                mlflow.log_metric(f'{group}_mcc',
                            MCC(cats[group], scores[group].argmax(1)),
                            step=step)

                try:
                    save_roc_curve(
                        scores[group],
                        to_categorical(cats[group], len(self.unique_labels)),
                        self.unique_labels,
                        f'{self.complete_log_path}/{group}_roc_curve.png',
                        binary=False,
                        acc=metrics.accuracy_score(cats[group], np.array(scores[group].argmax(1))),
                        mlops='mlflow',
                        epoch=step,
                        logger=None
                    )
                except ValueError as exc:
                    print(f"Warning: skipped {group} ROC curve plot: {exc}")
    
    def _ensure_embedding(self, tensor):
        """Return a 2D embedding tensor from either encoded prototypes or image tensors."""
        if tensor.dim() > 2:
            embedding, _ = self.model(tensor)
            return embedding
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def get_losses(self, enc, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample, dpreds, domains, labels, group):
        if self.args.classif_loss == 'cosine':
            classif_loss = 1 - torch.nn.functional.cosine_similarity(enc, self.model(to_rec)).mean()
        elif self.args.classif_loss == 'triplet' or self.args.classif_loss == 'softmax_contrastive':
            not_to_rec = not_to_rec.to(self.args.device).float()
            not_to_rec = not_to_rec.requires_grad_(True)
            if self.args.prototypes_to_use in ['no', 'batch'] or self.epoch == 0: # On the first epoch, we don't use prototypes but regular triplet loss	
                try:
                    pos_enc, _ = self.model(to_rec)
                except:
                    pass
                neg_enc, _ = self.model(not_to_rec)
            else:
                pos_enc = self._ensure_embedding(to_rec)
                neg_enc = self._ensure_embedding(not_to_rec)
            classif_loss = self.triplet_loss(enc, pos_enc, neg_enc)
        elif self.args.classif_loss in ['ce', 'hinge']:
            if enc.dim() != 2:
                raise RuntimeError(
                    f"Expected 2D encoder output before classification head, got shape={tuple(enc.shape)}"
                )
            expected_in = int(self.model.linear.in_features)
            got_in = int(enc.shape[1])
            if got_in != expected_in:
                raise RuntimeError(
                    f"Classifier head input mismatch: model_name={getattr(self.args, 'model_name', 'unknown')} "
                    f"enc_dim={got_in} linear_in={expected_in} linear_out={int(self.model.linear.out_features)}"
                )
            preds = self.model.linear(enc)
            labels = labels.to(self.args.device).long()
            classif_loss = self.celoss(preds, labels)

        elif 'arcfacewithsubcenters' in self.args.classif_loss:
            classif_loss, subcenters = self.arcloss(enc, labels)
            if 'subcenters' in lists[group]:
                lists[group]['subcenters'] += [_tensor_to_numpy(subcenters)]
            else:
                lists[group]['subcenters'] = [_tensor_to_numpy(subcenters)]

        elif 'arcface' in self.args.classif_loss:
            classif_loss = self.arcloss(enc, labels)
        else:
            exit(f"Classif loss {self.args.classif_loss} not implemented")

        if self.args.dloss == 'DANN':
            dloss = self.dceloss(dpreds, domains)

        elif self.args.dloss in ['inverseTriplet', 'inverse_softmax_contrastive'] and self.triplet_dloss is not None:
            pos_batch_sample, neg_batch_sample = pos_batch_sample.to(
                self.args.device).float(), neg_batch_sample.to(self.args.device).float()
            pos_batch_sample, neg_batch_sample = pos_batch_sample.requires_grad_(True), neg_batch_sample.requires_grad_(True)
            pos_enc = self._ensure_embedding(pos_batch_sample)
            neg_enc = self._ensure_embedding(neg_batch_sample)
            dloss = self.triplet_dloss(enc, neg_enc, pos_enc)
            # domain = domain.argmax(1)
        else:
            dloss = torch.Tensor([0]).to(self.args.device)[0]
        return classif_loss, dloss

    def loop(self, group, optimizer_model, gamma, loader, lists, traces):
        """

        Args:
            group: Which set? Train, valid or test
            optimizer_model: Object that contains the optimizer for the autoencoder
            ae: AutoEncoder (pytorch model, inherits nn.Module)
            sceloss: torch.nn.CrossEntropyLoss instance
            triplet_loss: torch.nn.TripletMarginLoss instance
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # triplet_loss = nn.TripletMarginLoss(margin, p=2)
        classif_loss = torch.tensor([0]).to(self.args.device)[0]
        cls_confmat = None
        total_batches = len(loader)
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            self._print_batch_progress(group, i, total_batches)
            # if group in ['train']:
            if self.model.training:
                optimizer_model.zero_grad()
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            data = data.requires_grad_(True)
            to_rec = to_rec.to(self.args.device).float()
            to_rec = to_rec.requires_grad_(True)
            not_to_rec = not_to_rec.to(self.args.device).float()
            not_to_rec = not_to_rec.requires_grad_(True)
            pos_batch_sample = pos_batch_sample.to(self.args.device).float()
            pos_batch_sample = pos_batch_sample.requires_grad_(True)
            neg_batch_sample = neg_batch_sample.to(self.args.device).float()
            neg_batch_sample = neg_batch_sample.requires_grad_(True)

            domains = to_categorical(domain.long(), self.n_batches).float().to(self.args.device)

            with self._autocast_context():
                enc, dpreds = self.model(data)

            domain_np = _tensor_to_numpy(domains)
            label_np = _tensor_to_numpy(labels)
            dpreds_np = _tensor_to_numpy(dpreds)
            enc_np = _tensor_to_numpy(enc.view(enc.shape[0], -1))

            self._append_batch_outputs(
                lists,
                group,
                domain_np,
                label_np,
                enc_np,
                names,
                old_labels,
                dpreds_np=dpreds_np,
                add_preds=(group == 'train'),
            )
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds_np.argmax(1),
                                                    _tensor_to_numpy(domains.argmax(1)))])]
            if self.model.training:
                with self._autocast_context():
                    classif_loss, dloss = self.get_losses(enc, to_rec, not_to_rec, pos_batch_sample,
                                                        neg_batch_sample, dpreds, domains, labels, group)
                traces[group]['closs'] += [classif_loss.item()]
                traces[group]['dloss'] += [dloss.item()]
                if group in ['train']:
                    if i == 0 and int(getattr(self.args, 'save_debug_images', 0)):
                        save_tensor(data[0], f'{self.complete_log_path}/{group}_data.png')
                    # total_loss = classif_loss + gamma * dloss
                    if self.use_grad_scaler:
                        self.grad_scaler.scale(classif_loss).backward()
                    else:
                        classif_loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                elif group in ['all']:
                    if gamma > 0:
                        dloss = gamma * dloss
                        if self.use_grad_scaler:
                            self.grad_scaler.scale(dloss).backward()
                        else:
                            dloss.backward()
                if self.args.fgsm:
                    adv_data = data + self.params['epsilon'] * data.grad.sign()
                    adv_data = torch.clamp(adv_data, -1, 1)
                    with self._autocast_context():
                        enc, dpreds = self.model(adv_data)
                    enc = enc.detach()
                    if group in ['train']:
                        if i == 0 and int(getattr(self.args, 'save_debug_images', 0)):
                            save_tensor(adv_data[0], f'{self.complete_log_path}/{group}_adv_data.png')
                        if self.args.classif_loss not in ['ce', 'hinge', 'arcface', 'arcfacewithsubcenters']:
                            adv_to_rec = to_rec + self.params['epsilon'] * to_rec.grad.sign()
                            adv_to_rec = torch.clamp(adv_to_rec, -1, 1)
                            adv_not_to_rec = not_to_rec + self.params['epsilon'] * not_to_rec.grad.sign()
                            adv_not_to_rec = torch.clamp(adv_not_to_rec, -1, 1)
                            adv_to_rec = adv_to_rec.detach()
                            adv_not_to_rec = adv_not_to_rec.detach()
                        else:
                            adv_to_rec = to_rec
                            adv_not_to_rec = not_to_rec
                        with self._autocast_context():
                            adv_classif_loss, _ = self.get_losses(enc, adv_to_rec, adv_not_to_rec, pos_batch_sample, neg_batch_sample, dpreds, domains, labels, group)
                        if self.use_grad_scaler:
                            self.grad_scaler.scale(0.1 * adv_classif_loss).backward()
                        else:
                            (0.1 * adv_classif_loss).backward()
                        del adv_data, adv_to_rec, adv_not_to_rec, adv_classif_loss
                    elif group in ['all']:
                        if self.args.classif_loss in ['inverseTriplet', 'inverse_softmax_contrastive']:
                            adv_pos_batch_sample = pos_batch_sample + self.params['epsilon'] * pos_batch_sample.grad.sign()
                            adv_pos_batch_sample = torch.clamp(adv_pos_batch_sample, -1, 1)
                            adv_neg_batch_sample = neg_batch_sample + self.params['epsilon'] * neg_batch_sample.grad.sign()
                            adv_neg_batch_sample = torch.clamp(adv_neg_batch_sample, -1, 1)
                            adv_pos_batch_sample = adv_pos_batch_sample.detach()
                            adv_neg_batch_sample = adv_neg_batch_sample.detach()
                        else:
                            adv_pos_batch_sample = pos_batch_sample
                            adv_neg_batch_sample = neg_batch_sample
                        with self._autocast_context():
                            _, adv_dloss = self.get_losses(enc, to_rec, not_to_rec, adv_pos_batch_sample, adv_neg_batch_sample, dpreds, domains, labels, group)
                        if self.use_grad_scaler:
                            self.grad_scaler.scale(0.1 * adv_dloss).backward()
                        else:
                            (0.1 * adv_dloss).backward()
                        del adv_data, adv_pos_batch_sample, adv_neg_batch_sample, adv_dloss
                del dloss

                if self.use_grad_scaler:
                    self.grad_scaler.step(optimizer_model)
                    self.grad_scaler.update()
                else:
                    optimizer_model.step()
                    record_gpu_peak(self.args.device)
            del data, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample, enc, dpreds, domains
        self._finish_batch_progress(total_batches)
        return classif_loss.detach(), lists, traces

    def loop_calibration(self, group, optimizer_model, gamma, loader, lists, traces):
        """

        Args:
            group: Which set? Train, valid or test
            optimizer_model: Object that contains the optimizer for the autoencoder
            ae: AutoEncoder (pytorch model, inherits nn.Module)
            sceloss: torch.nn.CrossEntropyLoss instance
            triplet_loss: torch.nn.TripletMarginLoss instance
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # triplet_loss = nn.TripletMarginLoss(margin, p=2)
        if group not in lists:
            lists[group] = {
                'set': [], 'preds': [], 'domain_preds': [], 'probs': [], 'lows': [],
                'cats': [], 'times': [], 'classes': [], 'domains': [], 'dloss': [],
                'dom_acc': [], 'labels': [], 'old_labels': [], 'encoded_values': [],
                'enc_values': [], 'age': [], 'gender': [], 'atn': [], 'names': [],
                'inputs': [], 'rec_values': [], 'subcenters': [],
            }
        if group not in traces:
            traces[group] = {
                'dloss': [], 'closs': [], 'dom_acc': [], 'acc': [], 'top3': [],
                'mcc': [], 'tpr': [], 'tnr': [], 'ppv': [], 'npv': [],
            }
        classif_loss = None
        total_batches = len(loader)
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            self._print_batch_progress(group, i, total_batches)
            # if group in ['train']:
            if self.model.training:
                optimizer_model.zero_grad()
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            domains = to_categorical(domain.long(), self.n_batches).float().to(self.args.device)
            with self._autocast_context():
                enc, dpreds = self.model(data)

            # if self.args.classif_loss == 'cosine':
            #     classif_loss = 1 - torch.nn.functional.cosine_similarity(enc, self.model(to_rec)).mean()
            # elif self.args.classif_loss == 'triplet' or self.args.classif_loss == 'softmax_contrastive':
            #     not_to_rec = not_to_rec.to(self.args.device).float()
            #     if not self.args.prototypes_to_use:  # or self.epoch == 0:
            #         pos_enc, _ = self.model(to_rec)
            #         neg_enc, _ = self.model(not_to_rec)
            #     else:
            #         # Both are already encoded when prototypes are registered
            #         pos_enc = to_rec
            #         neg_enc = not_to_rec
            #     classif_loss = self.triplet_loss(enc, pos_enc, neg_enc)
            # if self.args.dloss == 'DANN':
            #     dloss = self.celoss(dpreds, domains)
            # elif self.args.dloss == 'inverseTriplet':
            #     pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
            #         self.args.device).float(), pos_batch_sample.to(self.args.device).float()
            #     pos_enc, _ = self.model(pos_batch_sample)
            #     neg_enc, _ = self.model(neg_batch_sample)
            #     dloss = self.triplet_dloss(enc, neg_enc, pos_enc)
            #     # domain = domain.argmax(1)
            # else:
            #     dloss = torch.Tensor([0]).to(self.args.device)[0]
            with self._autocast_context():
                classif_loss, dloss = self.get_losses(enc, to_rec, not_to_rec, pos_batch_sample,
                                                      neg_batch_sample, dpreds, domains, labels, group)

            domain_np = _tensor_to_numpy(domains)
            label_np = _tensor_to_numpy(labels)
            dpreds_np = _tensor_to_numpy(dpreds)
            enc_np = _tensor_to_numpy(enc.view(enc.shape[0], -1))

            self._append_batch_outputs(
                lists,
                group,
                domain_np,
                label_np,
                enc_np,
                names,
                old_labels,
            )
            traces[group]['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds_np.argmax(1),
                                                    _tensor_to_numpy(domains.argmax(1)))])]
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds_np.argmax(1),
                                                    _tensor_to_numpy(domains.argmax(1)))])]
            traces[group]['closs'] += [0]
            traces[group]['dloss'] += [dloss.item()]
            if self.model.training:
                if group in ['calibration', 'train']:
                    # total_loss = classif_loss + gamma * dloss
                    classif_loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                elif group in ['all']:
                    if gamma > 0:
                        dloss = gamma * dloss
                        dloss.backward()
                optimizer_model.step()
            # pbar.update(1)
        self._finish_batch_progress(total_batches)
        return classif_loss, lists, traces


    def _classify_with_prototypes(self, embeddings: np.ndarray, dist_fct: str = 'euclidean'):
        """
        Classify using prototype distance with optional weighting by class size.
        
        Args:
            embeddings: (n_samples, n_features)
            dist_fct: 'euclidean' or 'cosine'
            
        Returns:
            predictions: (n_samples,)
            probas: (n_samples, n_classes)
        """
        if not self.class_prototypes.get('train'):
            return None, None
        
        from scipy.spatial.distance import cdist
        
        predictions = []
        probas_list = []

        train_cats_raw = self.all_samples.get('cats', {}).get('train')
        if isinstance(train_cats_raw, np.ndarray):
            train_cats = np.asarray(train_cats_raw).reshape(-1)
        elif train_cats_raw is not None:
            train_cat_chunks = [
                np.atleast_1d(chunk)
                for chunk in train_cats_raw
                if chunk is not None and np.size(chunk) > 0
            ]
            train_cats = np.concatenate(train_cat_chunks) if train_cat_chunks else np.array([])
        else:
            train_cats = np.array([])

        class_counts = {}
        if getattr(self.args, 'prototype_kind', 'distance').lower() == 'distance_weighted':
            for clas in self.unique_labels:
                class_counts[clas] = int(np.sum(train_cats == clas)) if train_cats.size else 0
        
        for emb in embeddings:
            emb = emb.reshape(1, -1)
            distances = {}
            
            # Compute distance to each class prototype
            for label, proto in self.class_prototypes['train'].items():
                if proto is None or not isinstance(proto, np.ndarray) or proto.size == 0:
                    distances[label] = np.inf
                    continue
                
                proto = np.asarray(proto)
                if proto.ndim == 1:
                    proto = proto.reshape(1, -1)
                else:
                    proto = proto.reshape(proto.shape[0], -1)
                
                if dist_fct.lower() == 'cosine':
                    # Cosine distance
                    from sklearn.metrics.pairwise import cosine_distances
                    dist = float(np.min(cosine_distances(emb, proto)[0]))
                else:
                    # Euclidean distance
                    dist = float(np.min(cdist(emb, proto, metric='euclidean')[0]))
                
                distances[label] = dist
            
            # Apply distance-based classification from closest prototype per class.
            prototype_kind = getattr(self.args, 'prototype_kind', 'distance').lower()
            finite_distances = {
                label: float(dist)
                for label, dist in distances.items()
                if np.isfinite(float(dist))
            }
            if prototype_kind == 'distance_weighted':
                class_priors = {
                    label: max(float(class_counts.get(label, 0)), 1.0)
                    for label in finite_distances
                }
            else:
                class_priors = {label: 1.0 for label in finite_distances}

            if finite_distances:
                labels = list(finite_distances.keys())
                dist_values = np.array([finite_distances[label] for label in labels], dtype=np.float64)
                temperature = float(np.std(dist_values))
                if not np.isfinite(temperature) or temperature <= 1e-8:
                    temperature = 1.0
                logits = -dist_values / temperature
                logits += np.log(np.array([class_priors[label] for label in labels], dtype=np.float64))
                logits -= np.max(logits)
                probs_arr = np.exp(logits)
                probs_arr /= max(float(np.sum(probs_arr)), 1e-12)
                probas = {label: 0.0 for label in distances}
                probas.update({label: float(prob) for label, prob in zip(labels, probs_arr)})
            else:
                probas = {label: 0.0 for label in distances}
            
            best_label = max(probas, key=probas.get)
            predictions.append(best_label)
            probas_list.append(probas)
        
        # Convert to class indices and probability matrix
        preds = np.array([np.argwhere(p == self.unique_labels).flatten()[0] if p in self.unique_labels else 0 
                         for p in predictions])
        proba_matrix = np.array([[probas_list[i].get(label, 0.0) for label in self.unique_labels] 
                                 for i in range(len(predictions))])
        
        return preds, proba_matrix
    
    def _fit_kde_classifier(self, encs: np.ndarray, cats: np.ndarray):
        """
        Fit a KDE classifier on training encodings.
        
        Args:
            encs: Training encodings (n_samples, n_features)
            cats: Training labels (n_samples,)
            
        Returns:
            KDE classifier instance
        """
        kde = make_kde_classifier(
            kernel=getattr(self.args, 'kde_kernel', 'gaussian'),
            bandwidth=getattr(self.args, 'kde_bandwidth', 'scott'),
            learnable=False,
            soft=True
        )
        kde.fit(encs, cats)
        return kde

    def evaluate_multi_classifiers(self, train_encs, train_cats, valid_encs, valid_cats):
        """
        Evaluate multiple classification strategies in parallel and select the best one.
        This is much faster than sequential Ax optimization and works well for classifier selection.
        
        Args:
            train_encs: Training encodings (n_samples, n_features)
            train_cats: Training labels (n_samples,)
            valid_encs: Validation encodings (n_samples, n_features)
            valid_cats: Validation labels (n_samples,)
            
        Returns:
            best_classifier_info: dict with keys:
                - 'method': str name of best method
                - 'classifier': fitted classifier instance (if applicable)
                - 'mcc': float MCC score
                - 'time': float execution time in seconds
                - 'params': dict of parameters used
        """
        import time
        
        # Use ML module for comprehensive evaluation
        start_time = time.time()
        
        all_results = evaluate_all_classifiers(
            train_encs, train_cats,
            valid_encs, valid_cats,
            min_k=1,
            max_k=min(10, train_encs.shape[0]),
            include_kde=(self.args.prototypes_to_use != 'no'),
            include_baselines=True
        )
        
        total_time = time.time() - start_time
        
        # Find best method
        best_method = 'knn'
        best_mcc = all_results.get('knn', {}).get('best_mcc', -1)
        best_params = {'k': all_results.get('knn', {}).get('best_k', 1)}
        
        # Check baselines
        for name, data in all_results.get('baselines', {}).items():
            if data.get('mcc', -1) > best_mcc:
                best_method = name
                best_mcc = data['mcc']
                best_params = {'method': name}
        
        # Check KDE
        kde_data = all_results.get('kde', {})
        if kde_data.get('mcc', -1) > best_mcc:
            best_method = 'kde'
            best_mcc = kde_data['mcc']
            best_params = {
                'kernel': kde_data.get('kernel'),
                'bandwidth': kde_data.get('bandwidth')
            }
        
        # Add prototype-based classification if enabled
        if self.args.prototypes_to_use != 'no' and self.class_prototypes.get('train'):
            try:
                preds, _ = self._classify_with_prototypes(
                    valid_encs,
                    dist_fct=self.params.get('dist_fct', 'euclidean')
                )
                if preds is not None:
                    proto_mcc = MCC(valid_cats, preds)
                    if proto_mcc > best_mcc:
                        best_method = 'prototypes'
                        best_mcc = proto_mcc
                        best_params = {'method': 'prototypes'}
            except Exception as e:
                print(f"Prototype evaluation failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"Multi-Classifier Validation Results:")
        print(f"  KNN:        MCC={all_results.get('knn', {}).get('best_mcc', 'N/A'):.4f} (k={all_results.get('knn', {}).get('best_k', 'N/A')})")
        
        for name, data in all_results.get('baselines', {}).items():
            print(f"  {name.upper():11} MCC={data.get('mcc', -1):.4f}")
        
        kde_data = all_results.get('kde', {})
        if kde_data:
            print(f"  KDE:        MCC={kde_data.get('mcc', -1):.4f} (kernel={kde_data.get('kernel', 'N/A')}, bw={kde_data.get('bandwidth', 'N/A')})")
        
        print(f"  Prototypes: MCC={best_mcc if best_method == 'prototypes' else 'N/A'}")
        print(f"  Best method: {best_method.upper()} (MCC={best_mcc:.4f})")
        print(f"  Total time: {total_time:.2f}s")
        print(f"{'='*60}\n")
        
        # Fit and return the best classifier
        best_classifier = None
        if best_method == 'knn':
            best_classifier = fit_knn_classifier(
                train_encs, train_cats,
                n_neighbors=best_params['k'],
                metric='minkowski'
            )
        elif best_method in ['logistic_regression', 'naive_bayes', 'linear_svc']:
            from otitenet.ml import fit_baseline_classifiers
            classifiers = fit_baseline_classifiers(train_encs, train_cats)
            best_classifier = classifiers.get(best_method)
        elif best_method == 'kde':
            from otitenet.ml import fit_kde_classifier
            best_classifier = fit_kde_classifier(
                train_encs, train_cats,
                kernel=best_params.get('kernel', 'gaussian'),
                bandwidth=best_params.get('bandwidth', 'scott')
            )
        # Note: prototypes don't need a sklearn-style classifier
        
        return {
            'method': best_method,
            'classifier': best_classifier,
            'mcc': best_mcc,
            'time': total_time,
            'params': best_params
        }

    def predict(self, group, loader, lists, traces):

        """

        Args:
            group: Which set? Train, valid or test
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # import nearest_neighbors from sklearn
        classif_loss = None
        cls_confmat = None
        classifier_fit_s = 0.0

        def _safe_concat(chunks):
            if chunks and len(chunks) > 0:
                try:
                    return np.concatenate(chunks)
                except Exception:
                    return None
            return None

        train_encs = _safe_concat(lists['train']['encoded_values'])
        train_cats = _safe_concat(lists['train']['cats'])
                
        # Decide which classifier to use
        use_prototypes = (self.args.prototypes_to_use in ['combined', 'class'] and 
                         self.class_prototypes.get('train') and 
                         self.args.classif_loss not in ['ce', 'hinge'])
        use_kde = (use_prototypes and 
                  getattr(self.args, 'prototype_kind', 'distance').lower() == 'kde')
        use_embedding_classifier = (self.args.classif_loss not in ['ce', 'hinge'] and not use_prototypes and not use_kde)
        siamese_inference = str(getattr(self.args, 'siamese_inference', 'linearsvc')).strip().lower()
        if siamese_inference not in ['knn', 'mlp_head', 'linearsvc', 'logisticregression']:
            siamese_inference = 'linearsvc'
        
        # Initialize classifiers
        KNeighborsClassifier = None
        kde_classifier = None
        embedding_classifier = None
        embedding_classifier_kind = None
        
        if use_kde and train_encs is not None and train_cats is not None:
            kde_classifier = self._fit_kde_classifier(train_encs, train_cats)
            print(f"Using KDE classifier with {len(train_encs)} training samples")
        
        elif use_prototypes and train_encs is not None and train_cats is not None:
            print(f"Using prototype-based classification ({getattr(self.args, 'prototype_kind', 'distance')})") 
        
        elif use_embedding_classifier:
            if train_encs is None or train_cats is None or train_encs.size == 0 or train_cats.size == 0:
                embedding_classifier = None
            else:
                cache = getattr(self, '_embedding_classifier_cache', {})
                cache_key = (
                    siamese_inference,
                    int(self.params.get('n_neighbors', 0) or 0),
                    tuple(train_encs.shape),
                    int(np.asarray(train_cats).shape[0]),
                )
                cached = cache.get(cache_key)
                if cached is not None:
                    embedding_classifier, embedding_classifier_kind = cached
                    print(f"Reusing {embedding_classifier_kind} classifier on embeddings (train_n={train_encs.shape[0]}, fit_s=0.000)")

                if embedding_classifier is None and siamese_inference == 'mlp_head':
                    try:
                        fit_start = time.perf_counter()
                        embedding_classifier = MLPClassifier(
                            hidden_layer_sizes=(128,),
                            activation='relu',
                            solver='adam',
                            alpha=1e-4,
                            batch_size='auto',
                            learning_rate_init=1e-3,
                            max_iter=300,
                            random_state=1,
                            early_stopping=True,
                            n_iter_no_change=15,
                        )
                        embedding_classifier.fit(train_encs, train_cats)
                        classifier_fit_s += time.perf_counter() - fit_start
                        embedding_classifier_kind = 'mlp_head'
                        print(f"Using MLP head on embeddings (train_n={train_encs.shape[0]}, fit_s={classifier_fit_s:.3f})")
                    except Exception as e:
                        print(f"MLP head fitting failed, falling back to KNN: {e}")
                        siamese_inference = 'knn'

                if embedding_classifier is None and siamese_inference == 'knn':
                    requested_k = self.params.get('n_neighbors', 0)
                    if requested_k is None or requested_k <= 0:
                        requested_k = min(25, max(1, train_encs.shape[0]))
                        self.params['n_neighbors'] = requested_k
                    k_val = min(requested_k, max(1, train_encs.shape[0]))
                    fit_start = time.perf_counter()
                    embedding_classifier = fit_knn_classifier(
                        train_encs, train_cats,
                        n_neighbors=k_val,
                        metric='minkowski'
                    )
                    classifier_fit_s += time.perf_counter() - fit_start
                    embedding_classifier_kind = 'knn'
                    print(f"Using KNN classifier with k={k_val} (fit_s={classifier_fit_s:.3f})")
                elif embedding_classifier is None and siamese_inference == 'linearsvc':
                    fit_start = time.perf_counter()
                    embedding_classifier = fit_linearsvc_classifier(train_encs, train_cats)
                    classifier_fit_s += time.perf_counter() - fit_start
                    embedding_classifier_kind = 'linearsvc'
                    print(f"Using LinearSVC classifier on embeddings (train_n={train_encs.shape[0]}, fit_s={classifier_fit_s:.3f})")
                elif embedding_classifier is None and siamese_inference == 'logisticregression':
                    fit_start = time.perf_counter()
                    embedding_classifier = fit_logreg_classifier(train_encs, train_cats)
                    classifier_fit_s += time.perf_counter() - fit_start
                    embedding_classifier_kind = 'logisticregression'
                    print(f"Using LogisticRegression classifier on embeddings (train_n={train_encs.shape[0]}, fit_s={classifier_fit_s:.3f})")
                if embedding_classifier is not None and cached is None:
                    cache[cache_key] = (embedding_classifier, embedding_classifier_kind)
                    self._embedding_classifier_cache = cache
        else:
            KNeighborsClassifier = self.model
        # plot_decision_boundary(KNeighborsClassifier, train_encs, train_cats)
        # plt.savefig(f'{self.complete_log_path}/decision_boundary.png')
        total_batches = len(loader)
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            self._print_batch_progress(group, i, total_batches)
            # if group in ['train']:
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            domains = to_categorical(domain.long(), self.n_batches).float().to(self.args.device)
            with self._autocast_context():
                enc, dpreds = self.model(data)
            enc_np = _tensor_to_numpy(enc.view(enc.shape[0], -1))
            
            # Predict based on chosen method
            if use_kde and kde_classifier is not None:
                preds = kde_classifier.predict(enc_np)
                proba = kde_classifier.predict_proba(enc_np)
            elif use_prototypes and self.class_prototypes.get('train'):
                preds, proba = self._classify_with_prototypes(enc_np, dist_fct=getattr(self.args, 'dist_fct', 'euclidean'))
                if preds is None:
                    preds = np.zeros(enc_np.shape[0], dtype=int)
                    proba = np.full((enc_np.shape[0], len(self.unique_labels)), 1.0 / max(len(self.unique_labels), 1))
            elif self.args.classif_loss not in ['ce', 'hinge']:
                if embedding_classifier is not None:
                    preds = embedding_classifier.predict(enc_np)
                    if hasattr(embedding_classifier, 'predict_proba'):
                        proba = embedding_classifier.predict_proba(enc_np)
                    else:
                        pred_idx = np.asarray(preds, dtype=int)
                        n_cls = max(len(self.unique_labels), 1)
                        proba = np.zeros((pred_idx.shape[0], n_cls), dtype=np.float32)
                        for row_i, cls_i in enumerate(pred_idx):
                            cls_i_int = int(cls_i)
                            if 0 <= cls_i_int < n_cls:
                                proba[row_i, cls_i_int] = 1.0
                else:
                    preds = np.zeros(enc_np.shape[0], dtype=int)
                    proba = np.full((enc_np.shape[0], len(self.unique_labels)), 1.0 / max(len(self.unique_labels), 1))
            else:
                with self._autocast_context():
                    out = self.model.linear(enc)
                preds = _tensor_to_numpy(out.argmax(1))
                proba = _tensor_to_numpy(torch.softmax(out, 1))
            if self.args.dloss == 'DANN':
                dloss = self.dceloss(dpreds, domains)

            elif self.args.dloss in ['inverseTriplet', 'inverse_softmax_contrastive'] and self.args.prototypes_to_use not in ['both', 'batch'] and self.triplet_dloss is not None:
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
                    self.args.device).float(), pos_batch_sample.to(self.args.device).float()
                pos_enc, _ = self.model(pos_batch_sample)
                neg_enc, _ = self.model(neg_batch_sample)
                dloss = self.triplet_dloss(enc, neg_enc, pos_enc)
                # domain = domain.argmax(1)
            else:
                dloss = torch.Tensor([0]).to(self.args.device)[0]

            domain_np = _tensor_to_numpy(domains)
            label_np = _tensor_to_numpy(labels)
            dpreds_np = _tensor_to_numpy(dpreds)
            enc_np = _tensor_to_numpy(enc.view(enc.shape[0], -1))

            self._append_batch_outputs(
                lists,
                group,
                domain_np,
                label_np,
                enc_np,
                names,
                old_labels,
            )
            labels_t = labels.to(self.args.device).long().view(-1)
            preds_t = torch.as_tensor(preds, device=self.args.device).long().view(-1)
            if preds_t.numel() != labels_t.numel():
                raise ValueError(
                    f"Prediction/label length mismatch in group={group}: "
                    f"preds={preds_t.numel()} labels={labels_t.numel()} "
                    f"exp_id={getattr(self.args, 'exp_id', 'unknown')}"
                )
            cls_confmat = _update_confmat_gpu(cls_confmat, labels_t, preds_t, len(self.unique_labels))
            batch_acc = (preds_t == labels_t).float().mean().item() if labels_t.numel() > 0 else 0.0
            dom_pred_t = dpreds.argmax(1).long()
            # Handle domains tensor dimension mismatch - if 1D, use directly; if 2D, use argmax
            if domains.dim() == 1:
                dom_true_t = domains.long()
            else:
                dom_true_t = domains.argmax(1).long()
            dom_acc = (dom_pred_t == dom_true_t).float().mean().item() if dom_true_t.numel() > 0 else 0.0
            traces[group]['acc'] += [batch_acc]
            traces[group]['dom_acc'] += [dom_acc]
            traces[group]['closs'] += [0]
            traces[group]['dloss'] += [dloss.item()]
            lists[group]['preds'] += [proba]
            lists[group]['inputs'] += [data.detach().cpu()]
        self._finish_batch_progress(total_batches)

        try:
            _, mcc_val = _acc_mcc_from_confmat_gpu(cls_confmat)
            traces[group]['mcc'] += [np.round(mcc_val, 3)]
        except:
            pass
        # Add other metrics here
        if len(np.unique(np.concatenate(lists[group]['cats']))) > 1:
            try:
                traces[group]['tpr'] += [np.round(
                    metrics.recall_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=1)
                    , 3)
                ]
                traces[group]['tnr'] += [np.round(
                    metrics.recall_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=0)
                    , 3)
                ]
                traces[group]['ppv'] += [np.round(
                    metrics.precision_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=1)
                    , 3)
                ]
                traces[group]['npv'] += [np.round(
                    metrics.precision_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=0)
                    , 3)
                ]
            except:
                pass

        # Add AUC calculation
        try:
            from sklearn.metrics import roc_auc_score
            y_true = np.concatenate(lists[group]['cats'])
            y_prob = np.concatenate(lists[group]['preds'])
            if y_prob.shape[1] > 2:
                auc_val = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            elif y_prob.shape[1] == 2:
                auc_val = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auc_val = 0.0
            traces[group]['auc'] += [np.round(auc_val, 3)]
        except:
            pass

        pca_every = int(getattr(self.args, 'test_pca_every', 0))
        if group == 'test' and pca_every > 0 and ((self.epoch + 1) % pca_every == 0):
            # plot the PCA of the encoded values
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            valid_encs = np.concatenate(lists['valid']['encoded_values'])
            valid_cats = np.concatenate(lists['valid']['cats'])
            try:
                test_encs = np.concatenate(lists['test']['encoded_values'])
            except:
                pass
            test_cats = np.concatenate(lists['test']['cats'])
            train_encs = np.concatenate(lists['train']['encoded_values'])
            train_cats = np.concatenate(lists['train']['cats'])
            train_encs_pca = pca.fit_transform(train_encs)
            valid_encs_pca = pca.transform(valid_encs)
            test_encs_pca = pca.transform(test_encs)
            # plot the values and save img
            plt.figure(figsize=(10, 10))
            plt.scatter(train_encs_pca[:, 0], train_encs_pca[:, 1], c=train_cats, marker='o')
            plt.scatter(valid_encs_pca[:, 0], valid_encs_pca[:, 1], c=valid_cats, marker='x')
            plt.scatter(test_encs_pca[:, 0], test_encs_pca[:, 1], c=test_cats, marker='^')
            plt.colorbar()
            plt.title('PCA of encoded values')
            plt.savefig(f'{self.complete_log_path}/pca_batches.png')
            self._log_file_event('pca_batches_png', f'{self.complete_log_path}/pca_batches.png')

        # Auto-select best k if enabled
        if self.args.auto_select_k and self.args.classif_loss not in ['ce', 'hinge'] and group == 'valid' and siamese_inference == 'knn':
            # Only attempt auto-k if we already accumulated valid encodings
            valid_encs = _safe_concat(lists['valid']['encoded_values']) if lists['valid']['encoded_values'] else None
            valid_cats = _safe_concat(lists['valid']['cats']) if lists['valid']['cats'] else None
            
            best_k, best_mcc = 1, -1
            max_k = min(11, train_encs.shape[0] + 1)  # k from 1 to 10, or max available samples
            
            # Try with class prototypes if available
            use_prototypes = self.args.prototypes_to_use in ['combined', 'class'] and self.class_prototypes.get('train')
            
            if use_prototypes:
                # Build training data from prototypes + encoded values
                proto_encs = []
                proto_cats = []
                for clas, proto in self.class_prototypes['train'].items():
                    if isinstance(proto, np.ndarray) and proto.size > 0:
                        proto_encs.append(proto.reshape(1, -1) if proto.ndim == 1 else proto)
                        proto_cats.append(np.array([np.argwhere(clas == self.unique_labels).flatten()[0]]))
                
                if proto_encs:
                    proto_encs = np.concatenate(proto_encs)
                    proto_cats = np.concatenate(proto_cats)
                    train_encs_aug = np.concatenate([train_encs, proto_encs])
                    train_cats_aug = np.concatenate([train_cats, proto_cats])
                else:
                    train_encs_aug, train_cats_aug = train_encs, train_cats
            else:
                train_encs_aug, train_cats_aug = train_encs, train_cats
            
            if valid_encs is not None and valid_encs.size > 0 and train_encs is not None:
                # Use ML module for k optimization
                fit_start = time.perf_counter()
                best_k, best_mcc, mcc_per_k = evaluate_knn_with_k_search(
                    train_encs_aug, train_cats_aug,
                    valid_encs, valid_cats,
                    min_k=1,
                    max_k=max_k
                )
                classifier_fit_s += time.perf_counter() - fit_start
                #best_k = best_k_result['best_k']
                #best_mcc = best_k_result['best_mcc']
                
                self.params['n_neighbors'] = best_k
                print(f"Auto-selected k={best_k} with validation MCC={best_mcc:.4f}")
            else:
                print("Auto-select k skipped due to lack of validation encodings or training encodings.")
        elif self.args.auto_select_k and group == 'valid' and siamese_inference != 'knn':
            print(f"Auto-select k skipped because siamese_inference={siamese_inference}; KNN k-search only applies to siamese_inference=knn.")

        self._last_classifier_fit_s = classifier_fit_s
        return None, lists, traces, embedding_classifier if embedding_classifier is not None else KNeighborsClassifier

    def predict_prototypes(self, group, loader, lists, traces):
        """

        Args:
            group: Which set? Train, valid or test
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # import nearest_neighbors from sklearn
        classif_loss = None
        train_prototypes = np.stack([self.class_prototypes['train'][x] for x in self.class_prototypes['train']])
        train_labels = list(self.class_prototypes['train'].keys())
        train_cats = [np.argwhere(x == self.unique_labels).flatten() for x in train_labels]
        
        # Use ML module for KNN fitting
        KNeighborsClassifier = fit_knn_classifier(
            train_prototypes, train_cats,
            n_neighbors=1,
            metric='minkowski'
        )
        # plot_decision_boundary(KNeighborsClassifier, train_encs, train_cats)
        # plt.savefig(f'{self.complete_log_path}/decision_boundary.png')
        total_batches = len(loader)
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            self._print_batch_progress(group, i, total_batches)
            # if group in ['train']:
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            domains = to_categorical(domain.long(), self.n_batches).float().to(self.args.device)
            with self._autocast_context():
                enc, dpreds = self.model(data)
            if self.args.classif_loss not in ['ce', 'hinge']:
                enc_np = _tensor_to_numpy(enc.view(enc.shape[0], -1))
                preds = KNeighborsClassifier.predict(enc_np)
                proba = KNeighborsClassifier.predict_proba(enc_np)
            else:
                with self._autocast_context():
                    out = self.model.linear(enc)
                preds = _tensor_to_numpy(out.argmax(1))
                proba = _tensor_to_numpy(torch.softmax(out, 1))
            if self.args.dloss == 'DANN':
                dloss = self.dceloss(dpreds, domains)

            elif self.args.dloss in ['inverseTriplet', 'inverse_softmax_contrastive'] and self.args.prototypes_to_use not in ['both', 'batch'] and self.triplet_dloss is not None:
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
                    self.args.device).float(), pos_batch_sample.to(self.args.device).float()
                pos_enc, _ = self.model(pos_batch_sample)
                neg_enc, _ = self.model(neg_batch_sample)
                dloss = self.triplet_dloss(enc, neg_enc, pos_enc)
                # domain = domain.argmax(1)
            else:
                dloss = torch.Tensor([0]).to(self.args.device)[0]

            domain_np = _tensor_to_numpy(domains)
            label_np = _tensor_to_numpy(labels)
            dpreds_np = _tensor_to_numpy(dpreds)
            enc_np = _tensor_to_numpy(enc.view(enc.shape[0], -1))

            self._append_batch_outputs(
                lists,
                group,
                domain_np,
                label_np,
                enc_np,
                names,
                old_labels,
            )
            traces[group]['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(preds,
                                                    label_np)])]
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds_np.argmax(1),
                                                    _tensor_to_numpy(domains.argmax(1)))])]
            traces[group]['closs'] += [0]
            traces[group]['dloss'] += [dloss.item()]
            lists[group]['preds'] += [proba]
            lists[group]['inputs'] += [data.detach().cpu()]
        self._finish_batch_progress(total_batches)

        try:
            traces[group]['mcc'] += [np.round(
                MCC(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1))
                , 3)
            ]
        except:
            pass
        # Add other metrics here
        if len(np.unique(np.concatenate(lists[group]['cats']))) > 1:
            try:
                traces[group]['tpr'] += [np.round(
                    metrics.recall_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=1)
                    , 3)
                ]
                traces[group]['tnr'] += [np.round(
                    metrics.recall_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=0)
                    , 3)
                ]
                traces[group]['ppv'] += [np.round(
                    metrics.precision_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=1)
                    , 3)
                ]
                traces[group]['npv'] += [np.round(
                    metrics.precision_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=0)
                    , 3)
                ]
            except:
                pass

        if group == 'test':
            # plot the PCA of the encoded values
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            valid_encs = np.concatenate(lists['valid']['encoded_values'])
            valid_cats = np.concatenate(lists['valid']['cats'])
            try:
                test_encs = np.concatenate(lists['test']['encoded_values'])
            except:
                pass
            test_cats = np.concatenate(lists['test']['cats'])
            train_encs = np.concatenate(lists['train']['encoded_values'])
            train_cats = np.concatenate(lists['train']['cats'])
            train_encs_pca = pca.fit_transform(train_encs)
            valid_encs_pca = pca.transform(valid_encs)
            test_encs_pca = pca.transform(test_encs)
            # plot the values and save img
            plt.figure(figsize=(10, 10))
            plt.scatter(train_encs_pca[:, 0], train_encs_pca[:, 1], c=train_cats, marker='o')
            plt.scatter(valid_encs_pca[:, 0], valid_encs_pca[:, 1], c=valid_cats, marker='x')
            plt.scatter(test_encs_pca[:, 0], test_encs_pca[:, 1], c=test_cats, marker='^')
            plt.colorbar()
            plt.title('PCA of encoded values')
            plt.savefig(f'{self.complete_log_path}/pca_batches.png')

        return None, lists, traces, KNeighborsClassifier

    def cluster_and_visualize(self, run, lists, groups, n_clusters=5, min_cluster_size=10, update_lists=True):
        """
        Perform clustering on all samples (including previously removed ones), visualize the clusters,
        and dynamically filter samples that belong to small clusters.
        
        Args:
            lists: Dictionary containing the encoded values, labels, etc.
            n_clusters: Number of clusters to form.
            min_cluster_size: Minimum size of a cluster to be considered valid.
        
        Returns:
            lists: Updated lists with filtered samples.
        """
        
        # Concatenate all samples (including previously removed ones)
        all_encoded_values = np.concatenate([
            self.all_samples['encoded_values'][group] for group in groups
        ])
        # all_encoded_values = all_encoded_values.reshape(all_encoded_values.shape[0], -1)
        all_labels = np.concatenate([self.all_samples['labels'][group] for group in groups])
        all_cats = np.concatenate([self.all_samples['cats'][group] for group in groups])
        all_names = np.concatenate([self.all_samples['names'][group] for group in groups])
        all_subcenters = np.concatenate([self.all_samples['subcenters'][group] for group in groups])
        all_batches = np.concatenate([self.all_samples['batches'][group] for group in groups])
        all_batches_cat = self._batch_encoder.transform(all_batches)
        
        # Perform K-Means clustering with sub-centers as initial centroids
        subcenters = self.model.get_subcenters().detach().cpu().numpy()
        subcenters = subcenters.reshape(-1, subcenters.shape[-1])
        kmeans = KMeans(n_clusters=n_clusters, init=subcenters, random_state=42)
        try:
            cluster_labels = kmeans.fit_predict(all_encoded_values)
        except:
            pass
        for ordin in ["PCA", "UMAP"]:
            # Reduce dimensionality for visualization
            if ordin == "PCA":
                pca = PCA(n_components=3)
            elif ordin == "Isomap":
                pca = Isomap(n_components=3)
            elif ordin == "MDS":
                pca = MDS(n_components=3)
            elif ordin == "UMAP":
                if UMAP is None:
                    print("[INFO] umap is not installed: skipping UMAP ordination.")
                    continue
                pca = UMAP(n_components=3)
            else:
                raise ValueError("Invalid ordination method")
            reduced_data = pca.fit_transform(all_encoded_values)
            class_prototypes = copy.deepcopy(self.class_prototypes)
            batch_prototypes = copy.deepcopy(self.batch_prototypes)
            for group in groups:
                for clas in np.unique(all_labels):
                    class_prototypes[group][clas] = pca.transform(class_prototypes[group][clas].reshape(1, -1))
                for batch in np.unique(all_batches):
                    try:
                        batch_prototypes[group][batch] = pca.transform(batch_prototypes[group][batch].reshape(1, -1))
                    except:
                        # Remove batch from batch_prototypes if it is not group
                        batch_prototypes[group].pop(batch, None)
            batch_prototypes.pop('all', None)
            class_prototypes.pop('all', None)
            # Get percentage of variance explained by the two components
            if ordin == "PCA":
                explained_variance = np.round(pca.explained_variance_ratio_ * 100, 2)
            else:
                explained_variance = None

            if 'subcenter' in self.args.classif_loss:
                try:
                    subcenters_reduced = pca.transform(subcenters)
                except Exception:
                    subcenters_reduced = None
                plot_pca(
                    reduced_data,
                    all_cats,               # color by class ids (numeric)
                    all_batches,
                    subcenters_reduced,      # overlay subcenters as prototypes
                    explained_variance,
                    self.complete_log_path,
                    'subcenters',
                    ordin,
                )
            
            # Transform prototypes for visualization
            class_prototypes_reduced = {}
            batch_prototypes_reduced = {}
            
            for group in groups:
                class_prototypes_reduced[group] = {}
                batch_prototypes_reduced[group] = {}
                
                # Transform class prototypes
                for clas, proto in self.class_prototypes.get(group, {}).items():
                    try:
                        if isinstance(proto, np.ndarray) and proto.size > 0:
                            class_prototypes_reduced[group][clas] = pca.transform(proto.reshape(1, -1))[0]
                    except Exception:
                        pass
                
                # Transform batch prototypes
                for batch, proto in self.batch_prototypes.get(group, {}).items():
                    try:
                        if isinstance(proto, np.ndarray) and proto.size > 0:
                            batch_prototypes_reduced[group][batch] = pca.transform(proto.reshape(1, -1))[0]
                    except Exception:
                        pass
            
            # Consolidate prototype dicts into arrays for plotting overlay
            def _dict_of_points_to_array(proto_dict_by_group):
                points = []
                try:
                    for _grp, d in proto_dict_by_group.items():
                        for _k, v in d.items():
                            if isinstance(v, np.ndarray) and v.size > 0:
                                points.append(v.reshape(-1))
                except Exception:
                    pass
                return np.vstack(points) if len(points) > 0 else None

            class_proto_points = _dict_of_points_to_array(class_prototypes_reduced)
            batch_proto_points = _dict_of_points_to_array(batch_prototypes_reduced)

            # Plot with prototypes overlaid
            plot_pca(
                reduced_data,
                all_cats,            # numeric class ids for color mapping
                all_batches,
                class_proto_points,
                explained_variance,
                self.complete_log_path,
                'labels',
                ordin,
            )
            plot_pca(
                reduced_data,
                cluster_labels,
                all_batches,
                None,
                explained_variance,
                self.complete_log_path,
                'clusters',
                ordin,
            )
            plot_pca(
                reduced_data,
                all_batches_cat,
                all_batches,
                batch_proto_points,
                explained_variance,
                self.complete_log_path,
                'batches',
                ordin,
            )
            if self.log_tracking:
                run[f"clusters_{ordin}"].upload(f'{self.complete_log_path}/clusters_{ordin}.png')
                run[f"batches_{ordin}"].upload(f'{self.complete_log_path}/batches_{ordin}.png')  # TODO make the batches pca in this function
                run[f"labels_{ordin}"].upload(f'{self.complete_log_path}/labels_{ordin}.png')
                if 'subcenter' in self.args.classif_loss:
                    run[f"subcenters_{ordin}"].upload(f'{self.complete_log_path}/subcenters_{ordin}.png')
            
            if self.log_mlflow:
                mlflow.log_artifact(f'{self.complete_log_path}/clusters_{ordin}.png')
                mlflow.log_artifact(f'{self.complete_log_path}/pca_batches_{ordin}.png')

        if 'arcfacewithsubcenters' not in self.args.classif_loss:
            # Identify small clusters
            cluster_sizes = np.bincount(cluster_labels)
            small_clusters = np.where(cluster_sizes < min_cluster_size)[0]  # Clusters with fewer than min_cluster_size samples
            noisy_samples = np.isin(cluster_labels, small_clusters)
        else:
            cluster_sizes = np.bincount(all_subcenters)
            small_clusters = np.where(cluster_sizes < min_cluster_size)[0]  # Clusters with fewer than min_cluster_size samples
            noisy_samples = np.isin(all_subcenters, small_clusters)
            
        # Get indices of noisy samples
        noisy_indices = np.where(noisy_samples)[0]
        
        # Log the noisy samplesminimi
        noisy_names = all_names[noisy_indices]
        noisy_labels = all_labels[noisy_indices]
        
        # Save noisy samples to a CSV file
        noisy_df = pd.DataFrame({
            'names': noisy_names,
            'labels': noisy_labels,
            'cluster': cluster_labels[noisy_indices]
        })
        noisy_df.to_csv(f'{self.complete_log_path}/noisy_samples.csv', index=False)
        
        if self.log_tracking:
            run["noisy_samples"].track_files(f'{self.complete_log_path}/noisy_samples.csv')
        
        if self.log_mlflow:
            mlflow.log_artifact(f'{self.complete_log_path}/noisy_samples.csv')
        
        if update_lists:
            # Update the lists to exclude noisy samples
            for group in ['train', 'valid', 'test']:
                # Compute mask based on current group's names array (accounts for augmentation)
                group_names = self.all_samples.get('names', {}).get(group, np.array([]))
                if len(group_names) == 0:
                    continue
                
                mask = ~np.isin(group_names, noisy_names)

                # Apply mask to each field that exists and matches dimension
                for key in ['labels', 'old_labels', 'names', 'cats', 'batches', 'encoded_values', 'inputs']:
                    try:
                        arr = self.all_samples.get(key, {}).get(group, None)
                        if arr is not None and hasattr(arr, '__len__') and len(arr) == len(mask):
                            lists[group][key] = arr[mask]
                    except Exception:
                        pass
                
                if 'arcfacewithsubcenters' in self.args.classif_loss:
                    try:
                        arr = self.all_samples.get('subcenters', {}).get(group, None)
                        if arr is not None and hasattr(arr, '__len__') and len(arr) == len(mask):
                            lists[group]['subcenters'] = arr[mask]
                    except Exception:
                        pass

        return lists

    def set_means_classes(self, list1):
        encodings = np.concatenate(list1['train']['encoded_values'])
        labels = np.concatenate(list1['train']['labels'])
        means = {}
        for label in self.unique_labels:
            inds = np.where(labels == label)[0]
            means[label] = np.mean(encodings[inds], axis=0)
        self.means = means

    def _append_to_group(self, target_group, values_by_key):
        for key, values in values_by_key.items():
            if len(values) == 0:
                continue
            selected = np.concatenate(values, axis=0)
            current = self.data[key][target_group]
            if len(current) == 0:
                self.data[key][target_group] = selected
            else:
                self.data[key][target_group] = np.concatenate((current, selected), axis=0)

    def _label_counts(self, labels):
        labels_arr = np.asarray(labels)
        return {
            str(label): int(np.sum(labels_arr == label))
            for label in np.unique(labels_arr)
        }

    def _cat_counts(self, cats):
        cats_arr = np.asarray(cats).reshape(-1)
        return {
            str(self.unique_labels[int(cat)] if 0 <= int(cat) < len(self.unique_labels) else cat): int(np.sum(cats_arr == cat))
            for cat in np.unique(cats_arr)
        }

    def _roc_auc_present_classes(self, cats, scores, group):
        cats_arr = np.asarray(cats).reshape(-1).astype(int)
        scores_arr = np.asarray(scores)
        present = np.unique(cats_arr)
        if len(present) < 2:
            raise ValueError(
                f"Cannot compute {group} ROC AUC because only one class is present. "
                f"Observed counts: {self._cat_counts(cats_arr)}"
            )
        if len(present) == 2:
            positive_class = int(present[1])
            return metrics.roc_auc_score(
                (cats_arr == positive_class).astype(int),
                scores_arr[:, positive_class],
            )

        present_scores = scores_arr[:, present]
        row_sums = present_scores.sum(axis=1, keepdims=True)
        present_scores = np.divide(
            present_scores,
            row_sums,
            out=np.full_like(present_scores, 1.0 / len(present), dtype=float),
            where=row_sums != 0,
        )
        remap = {int(label): idx for idx, label in enumerate(present)}
        present_y = np.array([remap[int(label)] for label in cats_arr])
        return metrics.roc_auc_score(
            to_categorical(present_y, len(present)),
            present_scores,
            multi_class='ovr',
        )

    def _validate_split_class_counts(self, context):
        problems = {}
        counts_by_group = {}
        for group in ['train', 'valid', 'test']:
            counts = self._label_counts(self.data['labels'][group])
            counts_by_group[group] = counts
            if len(counts) < 2:
                problems[group] = counts
        if problems:
            raise ValueError(
                f"Invalid split after {context}: train/valid/test must each contain at least two classes. "
                f"Problem groups: {problems}. All counts: {counts_by_group}"
            )

    def _validate_eval_labels_seen_in_train(self):
        train_labels = set(np.asarray(self.data['labels']['train']).tolist())
        problems = {}
        for group in ['valid', 'test']:
            group_labels = set(np.asarray(self.data['labels'][group]).tolist())
            missing = sorted(str(label) for label in group_labels if label not in train_labels)
            if missing:
                problems[group] = missing
        if problems:
            raise ValueError(
                "Invalid split: every valid/test label must be present in train. "
                f"Missing train labels by split: {problems}"
            )

    def _calibration_manifest_path(self, data_getter):
        explicit = str(getattr(self.args, 'calibration_manifest_path', '') or '').strip()
        if explicit:
            return explicit
        if data_getter is not None and getattr(data_getter, 'manifest_dir', None) is not None:
            path = os.path.join(data_getter.manifest_dir, 'splits', 'calibration.csv')
            if os.path.exists(path):
                return path
        return None

    def _write_calibration_manifest(self, data_getter, selected_records=None):
        if data_getter is None or getattr(data_getter, 'manifest_dir', None) is None:
            return
        calibration_manifest = os.path.join(data_getter.manifest_dir, 'splits', 'calibration.csv')
        os.makedirs(os.path.dirname(calibration_manifest), exist_ok=True)
        df = pd.DataFrame({
            'name': self.data['names']['calibration'],
            'label': self.data['labels']['calibration'],
            'batch': self.data['batches']['calibration'],
        })
        if selected_records:
            by_name = {str(item.get('name')): item for item in selected_records}
            df['source_group'] = [by_name.get(str(name), {}).get('source_group', '') for name in df['name']]
            df['source_index'] = [by_name.get(str(name), {}).get('source_index', '') for name in df['name']]
        df.to_csv(calibration_manifest, index=False)
        try:
            self.args.calibration_manifest_path = calibration_manifest
        except Exception:
            pass

    def _apply_calibration_manifest(self, manifest_path, data_getter, n_calibration):
        if not manifest_path or not os.path.exists(manifest_path):
            return False
        try:
            manifest = pd.read_csv(manifest_path)
        except Exception as exc:
            print(f"[CalibrationSplit] could not read calibration manifest {manifest_path}: {exc}")
            return False
        if 'name' not in manifest.columns or manifest.empty:
            return False

        requested_names = [str(name) for name in manifest['name'].dropna().tolist()]
        if n_calibration and len(requested_names) != int(n_calibration):
            print(
                f"[CalibrationSplit] manifest count {len(requested_names)} does not match "
                f"n_calibration={n_calibration}; drawing a new split."
            )
            return False

        source_groups = ['valid', 'test', 'train']
        name_to_source = {}
        for group in source_groups:
            for idx, name in enumerate(np.asarray(self.data['names'][group]).tolist()):
                name_to_source.setdefault(str(name), (group, int(idx)))

        missing = [name for name in requested_names if name not in name_to_source]
        if missing:
            print(
                f"[CalibrationSplit] manifest {manifest_path} could not be reused; "
                f"{len(missing)} sample(s) are missing from current train/valid/test splits."
            )
            return False

        selected_by_group = {group: [] for group in source_groups}
        for name in requested_names:
            group, idx = name_to_source[name]
            selected_by_group[group].append(idx)

        calibration_values = {key: [] for key in ['inputs', 'labels', 'old_labels', 'names', 'cats', 'batches']}
        train_append_values = {key: [] for key in ['inputs', 'labels', 'old_labels', 'names', 'cats', 'batches']}
        selected_records = []
        for group in source_groups:
            selected = np.array(sorted(set(selected_by_group[group])), dtype=int)
            if selected.size == 0:
                continue
            for key in calibration_values:
                selected_values = self.data[key][group][selected]
                calibration_values[key].append(selected_values)
                if group != 'train':
                    train_append_values[key].append(selected_values)
            for idx in selected:
                selected_records.append({
                    'name': str(self.data['names'][group][idx]),
                    'source_group': group,
                    'source_index': int(idx),
                })
            if group in ['valid', 'test']:
                keep_mask = np.ones(len(self.data['labels'][group]), dtype=bool)
                keep_mask[selected] = False
                for key in calibration_values:
                    self.data[key][group] = self.data[key][group][keep_mask]

        self._append_to_group('calibration', calibration_values)
        self._append_to_group('train', train_append_values)

        actual = len(self.data['labels']['calibration'])
        if actual != len(requested_names):
            raise RuntimeError(
                f"Calibration manifest reuse mismatch: requested {len(requested_names)}, built {actual}."
            )

        if data_getter is not None and getattr(data_getter, 'manifest_dir', None) is not None:
            data_getter.data = self.data
            data_getter.save_split_manifests(data_getter.manifest_dir)
            self._write_calibration_manifest(data_getter, selected_records=selected_records)

        counts = {
            str(label): int(np.sum(np.asarray(self.data['labels']['calibration']) == label))
            for label in np.unique(self.data['labels']['calibration'])
        }
        sources = {group: int(len(set(selected_by_group[group]))) for group in source_groups}
        print(
            f"[CalibrationSplit] reused manifest={manifest_path} "
            f"n_calibration={actual} counts={counts} sources={sources}"
        )
        return True

    def _apply_calibration_split(self, data_getter):
        n_calibration = int(getattr(self.args, 'n_calibration', 0) or 0)
        if n_calibration == 0:
            return

        if self._apply_calibration_manifest(self._calibration_manifest_path(data_getter), data_getter, n_calibration):
            return

        source_groups = ['valid', 'test']
        labels = [
            label for label in list(self.unique_labels)
            if any(np.any(np.asarray(self.data['labels'][group]) == label) for group in source_groups)
        ]
        if not labels:
            raise ValueError("Cannot build calibration split because valid/test contain no labels.")

        if n_calibration < len(labels):
            raise ValueError(
                f"n_calibration={n_calibration} is too small for {len(labels)} classes. "
                "Use at least one calibration sample per valid/test class, or set n_calibration=0."
            )

        available = {label: {group: [] for group in source_groups} for label in labels}
        for group in source_groups:
            group_labels = np.asarray(self.data['labels'][group])
            for label in labels:
                for idx in np.where(group_labels == label)[0]:
                    available[label][group].append(int(idx))

        base = n_calibration // len(labels)
        remainder = n_calibration % len(labels)
        requested_by_label = {
            label: base + (1 if i < remainder else 0)
            for i, label in enumerate(labels)
        }
        shortages = {
            str(label): {
                'requested': requested_by_label[label],
                'available': sum(len(available[label][group]) for group in source_groups),
                'available_by_split': {group: len(available[label][group]) for group in source_groups},
            }
            for label in labels
            if sum(len(available[label][group]) for group in source_groups) < requested_by_label[label]
        }
        if shortages:
            raise ValueError(
                "Cannot build calibration split from valid/test with the requested n_calibration. "
                f"Per-class availability: {shortages}"
            )

        rng = np.random.default_rng(int(getattr(self.args, 'seed', 42)))
        selected_by_group = {group: [] for group in source_groups}
        for label in labels:
            label_choices = []
            for group in source_groups:
                label_choices.extend((group, idx) for idx in available[label][group])
            chosen_positions = rng.choice(
                len(label_choices),
                size=requested_by_label[label],
                replace=False,
            )
            for pos in chosen_positions:
                group, idx = label_choices[int(pos)]
                selected_by_group[group].append(idx)

        calibration_values = {key: [] for key in ['inputs', 'labels', 'old_labels', 'names', 'cats', 'batches']}
        train_append_values = {key: [] for key in ['inputs', 'labels', 'old_labels', 'names', 'cats', 'batches']}
        selected_records = []
        for group in source_groups:
            selected = np.array(sorted(set(selected_by_group[group])), dtype=int)
            if selected.size == 0:
                continue
            for key in calibration_values:
                selected_values = self.data[key][group][selected]
                calibration_values[key].append(selected_values)
                if group != 'train':
                    train_append_values[key].append(selected_values)
            for idx in selected:
                selected_records.append({
                    'name': str(self.data['names'][group][idx]),
                    'source_group': group,
                    'source_index': int(idx),
                })
            if group != 'train':
                keep_mask = np.ones(len(self.data['labels'][group]), dtype=bool)
                keep_mask[selected] = False
                for key in calibration_values:
                    self.data[key][group] = self.data[key][group][keep_mask]

        self._append_to_group('calibration', calibration_values)
        self._append_to_group('train', train_append_values)

        actual = len(self.data['labels']['calibration'])
        if actual != n_calibration:
            raise RuntimeError(
                f"Calibration split construction mismatch: requested {n_calibration}, built {actual}."
            )

        if data_getter is not None and getattr(data_getter, 'manifest_dir', None) is not None:
            data_getter.data = self.data
            data_getter.save_split_manifests(data_getter.manifest_dir)
            self._write_calibration_manifest(data_getter, selected_records=selected_records)

        counts = {
            str(label): int(np.sum(np.asarray(self.data['labels']['calibration']) == label))
            for label in labels
        }
        sources = {group: int(len(set(selected_by_group[group]))) for group in source_groups}
        print(f"[CalibrationSplit] n_calibration={n_calibration} labels={list(map(str, labels))} counts={counts} sources={sources}")

    def setup_training_objects(self, params):
        # Assign hyperparameters
        self.params = params
        if 'dloss' in params:
            self.args.dloss = params['dloss']
        if 'classif_loss' in params:
            self.args.classif_loss = params['classif_loss']
        if 'prototypes_to_use' in params:
            self.args.prototypes_to_use = params['prototypes_to_use']
        if 'n_positives' in params:
            self.args.n_positives = int(params['n_positives'])
        if 'n_negatives' in params:
            self.args.n_negatives = int(params['n_negatives'])
        if 'fgsm' in params:
            self.args.fgsm = int(params['fgsm'])
        if 'normalize' in params:
            self.args.normalize = params['normalize']
        if 'dist_fct' in params:
            self.args.dist_fct = params['dist_fct']
        if 'distance_fct' in params:
            self.args.distance_fct = params['distance_fct']
            
        if not self.args.fgsm:
            params['epsilon'] = 0
        if 'margin' not in params:
            params['margin'] = 0
        if 'dmargin' not in params:
            params['dmargin'] = 0
        if 'dist_fct' not in params:
            params['dist_fct'] = 'none'
        if 'gamma' not in params:
            params['gamma'] = 0
        if 'n_neighbors' not in params:
            params['n_neighbors'] = 0
        if 'is_transform' not in params:
            params['is_transform'] = 1
        if 'normalize' not in params:
            params['normalize'] = 'no'
        if 'dropout' not in params:
            params['dropout'] = 0.0
        if 'optimizer_type' not in params:
            params['optimizer_type'] = 'adam'
        if 'smoothing' not in params:
            params['smoothing'] = 0.0
        if 'lr' not in params:
            params['lr'] = 0.001
        if 'wd' not in params:
            params['wd'] = 0.0
            
        dist_fct = get_distance_fct(params['dist_fct'])
        self.make_triplet_loss(dist_fct, params)

        if not hasattr(self, 'complete_log_path') or not self.complete_log_path:
            self.foldername = str(uuid.uuid4())
            self.complete_log_path = f'logs/{self.args.task}/{self.foldername}'
            os.makedirs(self.complete_log_path, exist_ok=True)

        data_getter = GetData(self.path, self.args.valid_dataset, self.args, manifest_dir=self.complete_log_path)
        split_debug = getattr(data_getter, 'split_debug', {}) or {}
        if split_debug.get('effective_train_datasets') is not None:
            self.args.effective_train_datasets = _normalize_train_datasets(split_debug.get('effective_train_datasets'))
        if split_debug.get('effective_valid_dataset') is not None:
            self.args.effective_valid_dataset = _normalize_single_dataset(split_debug.get('effective_valid_dataset'))
        if split_debug.get('effective_test_dataset') is not None:
            self.args.effective_test_dataset = _normalize_single_dataset(split_debug.get('effective_test_dataset'))
        self.unique_labels = data_getter.unique_labels
        self.unique_batches = data_getter.unique_batches
        # self._batch_encoder should be defined only once, in __init__
        if not hasattr(self, '_batch_encoder') or self._batch_encoder is None:
            from sklearn.preprocessing import LabelEncoder
            self._batch_encoder = LabelEncoder()
            self._batch_encoder.fit(self.unique_batches)
        
        proto_strategy = getattr(self.args, 'prototype_strategy', 'mean')
        proto_components = getattr(self.args, 'prototype_components', 1)
        self.prototypes = Prototypes(
            self.unique_labels,
            self.unique_batches,
            strategy=proto_strategy,
            components=proto_components,
            random_state=getattr(self.args, 'seed', 1),
        )
        
        self.data, self.unique_labels, self.unique_batches = data_getter.get_variables()
        self._validate_split_class_counts('initial split')
        self._validate_eval_labels_seen_in_train()
        self._apply_calibration_split(data_getter)
        self._validate_split_class_counts('calibration split')
        self.make_samples_weights()
        self.set_arcloss()

        # Initialize model
        self.model = Net(self.args.device, self.n_cats, self.n_batches,
                    model_name=self.args.model_name, is_stn=self.args.is_stn,
                    n_subcenters=self.n_batches)
        if self.run_explainability:
            if self.args.classif_loss in ['ce', 'hinge']:
                self.shap_model = Net_deep_shap(self.args.device, self.n_cats, self.n_batches,
                                    model_name=self.args.model_name, is_stn=self.args.is_stn,
                                    n_subcenters=self.n_batches, shape=(3, self.args.new_size, self.args.new_size))
            else:
                self.shap_model = Net_shap(self.args.device, self.n_cats, self.n_batches,
                                    model_name=self.args.model_name, is_stn=self.args.is_stn,
                                    n_subcenters=self.n_batches)
        prototypes = {
            'combined': self.combined_prototypes,
            'class': self.class_prototypes,
            'batch': self.batch_prototypes
        }
        loaders = get_images_loaders(data=self.data,
                                     batch_encoder=self._batch_encoder,
                                        random_recs=self.args.random_recs,
                                        weighted_sampler=self.args.weighted_sampler,
                                        is_transform=params['is_transform'],
                                        samples_weights=self.samples_weights,
                                        epoch=0,
                                        unique_labels=self.unique_labels,
                                        triplet_dloss=self.args.dloss, bs=self.args.bs,
                                        prototypes_to_use=self.args.prototypes_to_use,
                                        prototypes=prototypes,
                                        size=self.args.new_size,
                                        normalize=self.args.normalize,
                                        n_aug=params.get('n_aug', 1),
                                        num_workers=getattr(self.args, 'num_workers', 0),
                                        )

        return data_getter, loaders


def set_run(run, params):
    for k, v in params.items():
        if hasattr(run, 'log_parameter'):
            run.log_parameter(k, v)
        else:
            run[k] = v
    if hasattr(run, 'log_parameter'):
        run.log_parameter('SHAP', 1)
    else:
        run['SHAP'] = 1
    return run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--optuna_pruner', type=str, default='median', choices=['median', 'none'],
                        help='Optuna pruner strategy: median (default) or none (disable pruning)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--groupkfold', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--path', type=str, default='./data/otite_ds_64')
    parser.add_argument('--path_original', type=str, default='./data/otite_ds_-1')
    parser.add_argument('--exp_id', type=str, default='otite')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (-1 uses all CPU cores)')
    parser.add_argument('--dloss', type=str, default=None, help='domain loss - if None, will optimize')
    parser.add_argument('--classif_loss', type=str, default=None, help='triplet or cosine - if None, will optimize')
    parser.add_argument('--task', type=str, default='notNormal', help='Binary classification?')
    parser.add_argument('--is_stn', type=int, default=1, help='Transform train data?')
    parser.add_argument('--weighted_sampler', type=int, default=1, help='Weighted sampler?')
    parser.add_argument('--n_calibration', type=int, default=0, help='Number of balanced calibration samples to draw from valid/test (0 disables calibration)')
    parser.add_argument('--calibration_manifest_path', type=str, default='', help='Optional splits/calibration.csv to reuse exact calibration samples during retraining')
    parser.add_argument('--remove_noisy_samples', type=int, default=0, help='Remove noisy samples?')
    parser.add_argument('--noisy_cluster_limit', type=int, default=10, help='Noisy cluster limit?')
    parser.add_argument('--prototypes_to_use', type=str, default=None, help='Which prototypes - if None, will optimize')
    parser.add_argument('--prototype_strategy', type=str, default='mean', choices=['mean', 'kmeans', 'gmm'], help='TRAINING: How to aggregate/learn prototypes during training (mean/kmeans/gmm). Determines shape of prototype set.')
    parser.add_argument('--prototype_components', type=int, default=1, help='Components/centroids per class for prototype aggregation (used with kmeans/gmm during training)')
    parser.add_argument('--prototype_kind', type=str, default='distance', choices=['distance', 'kde', 'distance_weighted'], help='VALIDATION: How to classify using learned prototypes at test time (distance/kde/distance_weighted)')
    parser.add_argument('--siamese_inference', type=str, default='linearsvc', choices=['knn', 'mlp_head', 'linearsvc', 'logisticregression'], help='Siamese inference classifier over embeddings: knn, mlp_head, linearsvc, or logisticregression')
    parser.add_argument('--kde_kernel', type=str, default='gaussian', choices=['gaussian', 'exponential', 'linear', 'tophat'], help='KDE kernel type')
    parser.add_argument('--kde_bandwidth', type=str, default='scott', help='KDE bandwidth: scott, silverman, or float value')
    parser.add_argument('--n_positives', type=int, default=None, help='Number of positive samples per anchor - if None, will optimize (only for triplet-based losses)')
    parser.add_argument('--n_negatives', type=int, default=None, help='Number of negative samples per anchor - if None, will optimize (only for triplet-based losses)')
    parser.add_argument('--fgsm', type=int, default=None, help='Use Fast Gradient Sign Method - if None, will optimize')
    parser.add_argument('--dist_fct', type=str, default='euclidean', choices=['euclidean', 'cosine'], help='Distance function for prototypes/triplets')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--dmargin', type=float, default=1.0, help='Margin for domain triplet loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma weight for domain loss')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for FGSM')
    parser.add_argument('--n_neighbors', type=int, default=1, help='Number of neighbors for KNN evaluation')
    parser.add_argument('--valid_dataset', type=str, default='Banque_Viscaino_Chili_2020', help='Validation dataset')
    parser.add_argument('--train_datasets', type=str, default='Banque_Comert_Turquie_2020_jpg,Banque_Calaman_USA_2020_trie_CM,GMFUNL_jan2023', help='Optional comma-separated training datasets. If empty, all datasets except valid/test are used.')
    parser.add_argument('--test_dataset', type=str, default='inference', help='Optional dedicated test dataset. If omitted or unavailable, half of the validation set is used as test.')
    parser.add_argument('--new_size', type=int, default=64, help='New size for images')
    parser.add_argument('--normalize', type=str, default=None, help='Normalize images - if None, will optimize')
    parser.add_argument('--n_aug', type=int, default=1, help='Number of augmentations per image (1=original only)')
    parser.add_argument('--auto_select_k', type=int, default=1, help='Automatically select best k (1-10) each epoch based on validation MCC')
    parser.add_argument('--save_repro_artifacts', type=int, default=1, help='Save manifests + sanity samples for reproducibility')
    parser.add_argument('--run_tag', type=str, default='prod', help='Tag to identify run type in logs/MLOps (e.g., TEST_SMOKE)')
    parser.add_argument('--trial_index', type=int, default=0, help='Current trial index when launched by an external scheduler')
    parser.add_argument('--reset_opt_state', type=int, default=0, help='Reset Bayesian optimization state and start from scratch (0/1)')
    parser.add_argument('--force_retrain', type=int, default=0, help='Force fresh training from best parameters in resume final-pass, skipping checkpoint reuse (0/1)')
    parser.add_argument('--log_tracking', type=int, default=0, help='Enable legacy Tracking logging (0/1)')
    parser.add_argument('--log_dvclive', type=int, default=1, help='Enable DVCLive logging (0/1)')
    parser.add_argument('--dvclive_save_dvc_exp', type=int, default=1, help='Ask DVCLive to save a DVC experiment snapshot (0/1)')
    parser.add_argument('--dvclive_branch_exp', type=int, default=1, help='Promote saved DVC experiment to dvc-exp/<current-branch>/<experiment> (0/1)')
    parser.add_argument('--dvclive_monitor_system', type=int, default=1, help='Enable DVCLive system metrics monitoring (0/1)')
    parser.add_argument('--dvclive_dvcyaml', type=str, default='auto', help='DVCLive dvc.yaml mode: none|auto|<custom_path>')
    parser.add_argument('--log_comet', type=int, default=0, help='Enable legacy Comet logging (0/1). Default off; comet_ml is imported only when enabled.')
    parser.add_argument('--log_mlflow', type=int, default=0, help='Enable MLflow logging (0/1)')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--epoch_progress', type=int, default=1, help='Show per-epoch batch progress bar (0/1)')
    parser.add_argument('--test_pca_every', type=int, default=0, help='Generate test PCA plot every N epochs (0 disables)')
    parser.add_argument('--save_debug_images', type=int, default=0, help='Save sample train/adv images per epoch (0/1)')
    parser.add_argument('--epoch_media_every', type=int, default=0, help='Upload sample images to trackers every N epochs (0 disables)')
    parser.add_argument('--heavy_best_analysis', type=int, default=0, help='Run expensive encoded-values clustering on each new best (0/1)')
    parser.add_argument('--run_explainability', type=int, default=0, help='Run SHAP/Grad-CAM artifact generation (0/1). Default 0 keeps only optimization + eval outputs.')
    parser.add_argument('--amp', type=int, default=1, help='Enable CUDA automatic mixed precision (0/1)')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'], help='AMP dtype to use on CUDA')


    args = parser.parse_args()

    # Allow environment variable override for num_workers
    env_num_workers = os.environ.get("NUM_WORKERS")
    if env_num_workers is not None:
        try:
            args.num_workers = int(env_num_workers)
            print(f"[INFO] Overriding num_workers from environment: {args.num_workers}")
        except Exception:
            print(f"[WARN] Invalid NUM_WORKERS env var: {env_num_workers}")

    _set_global_seeds(getattr(args, 'seed', 1), deterministic=True)
    args.exp_id = f'{args.exp_id}_{args.task}_{args.run_tag}'
    if str(getattr(args, "device", "")).startswith("cuda") and not torch.cuda.is_available():
        print(f"[Config] CUDA device requested ({args.device}) but torch.cuda.is_available() is false; using CPU.")
        args.device = "cpu"
    
    # Track which parameters were originally None (to optimize them)
    optimize_params = {}
    
    # Convert string "None" to actual None for optional parameters
    if args.dloss == "None":
        args.dloss = None
    if args.classif_loss == "None":
        args.classif_loss = None
    if args.prototypes_to_use == "None":
        args.prototypes_to_use = None
    if args.n_positives == "None":
        args.n_positives = None
    if args.n_negatives == "None":
        args.n_negatives = None
    if args.fgsm == "None":
        args.fgsm = None
    if args.normalize == "None":
        args.normalize = None
    
    # Mark which ones are None before applying defaults
    # n_calibration is fixed by the CLI and is never optimized.
    if args.dloss is None:
        optimize_params['dloss'] = True
        args.dloss = "inverseTriplet"
    if args.classif_loss is None:
        optimize_params['classif_loss'] = True
        args.classif_loss = "softmax_contrastive"
    if args.prototypes_to_use is None:
        optimize_params['prototypes_to_use'] = True
        args.prototypes_to_use = "no"
    if args.n_positives is None:
        optimize_params['n_positives'] = True
        args.n_positives = 1
    if args.n_negatives is None:
        optimize_params['n_negatives'] = True
        args.n_negatives = 1
    if args.fgsm is None:
        optimize_params['fgsm'] = True
        args.fgsm = 1
    if args.normalize is None:
        optimize_params['normalize'] = True
        args.normalize = "no"
    args = validate_n_calibration(args)
    args = disable_stn_when_unsupported(args)
    if int(getattr(args, 'auto_select_k', 0) or 0) and str(getattr(args, 'siamese_inference', 'linearsvc')).strip().lower() != 'knn':
        print(
            f"[Config] Disabling auto_select_k because siamese_inference={args.siamese_inference}; "
            "KNN k-search only applies when siamese_inference=knn."
        )
        args.auto_select_k = 0
    
    try:
        mlflow.create_experiment(
            args.dloss,
            # artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            # tags={"version": "v1", "priority": "P1"},
        )
    except Exception as e:
        print(f"\n\nExperiment {args.exp_id} already exists: {e}\n\n")

    train = TrainAE(args, args.path, load_tb=False, log_metrics=True, keep_models=True,
                    log_inputs=False, log_plots=True, log_tb=False,
                    log_tracking=bool(int(args.log_tracking)),
                    log_comet=bool(int(args.log_comet)),
                    log_mlflow=bool(int(args.log_mlflow)),
                    log_dvclive=bool(int(args.log_dvclive)),
                    groupkfold=args.groupkfold)
    train.verbose = int(getattr(args, 'verbose', 1))

    trial_counter = {'counter': 0}
    total_trials = int(getattr(args, 'n_trials', 0) or 0)

    def _csv_escape(value):
        s = "" if value is None else str(value)
        return '"' + s.replace('"', '""') + '"'

    def _append_trial_runtime_event(trial_idx, event, status, score=None, error_message=None):
        progress_root = os.path.join('logs', 'progresses', str(getattr(args, 'task', 'unknown')))
        csv_dir = os.path.join(progress_root, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        trial_runtime_path = os.path.join(
            csv_dir,
            f"{getattr(args, 'run_tag', 'RUN')}_{getattr(args, 'task', 'unknown')}_trial_runtime.csv"
        )
        if not os.path.exists(trial_runtime_path):
            with open(trial_runtime_path, 'w') as f:
                f.write(
                    'timestamp,exp_id,run_tag,task,trial_index,total_trials,event,status,score,error_message,run_dir\\n'
                )

        run_dir = getattr(train, 'complete_log_path', '')
        ts = datetime.now().isoformat(timespec='seconds')
        row = [
            ts,
            getattr(args, 'exp_id', ''),
            getattr(args, 'run_tag', ''),
            getattr(args, 'task', ''),
            int(trial_idx),
            int(total_trials),
            event,
            status,
            '' if score is None else score,
            '' if error_message in [None, ''] else error_message,
            run_dir,
        ]
        with open(trial_runtime_path, 'a') as f:
            f.write(','.join(_csv_escape(x) for x in row) + '\\n')

    def _finalize_trial_metadata(train_obj, trial_idx, status, error_message=None):
        """Force terminal metadata for each Ax trial, even if training flow exits unexpectedly."""
        trial_dir = getattr(train_obj, 'complete_log_path', None)
        if not trial_dir:
            return
        metadata_path = os.path.join(trial_dir, 'run_metadata.json')
        payload = {}
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    payload = json.load(f)
        except Exception:
            payload = {}

        payload['trial_index'] = int(trial_idx)
        payload['run_status'] = str(status)
        payload['finished'] = 1
        payload['run_finished'] = 1
        payload['finished_at'] = datetime.now().isoformat(timespec='seconds')
        payload['error_message'] = str(error_message) if error_message not in [None, ''] else None

        try:
            with open(metadata_path, 'w') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"Warning: could not finalize trial metadata for trial {trial_idx}: {e}")

    def _params_cache_key(params):
        normalized = {}
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, float):
                normalized[k] = f"{v:.12g}"
            else:
                normalized[k] = str(v)
        return json.dumps(normalized, sort_keys=True)

    trial_cache_by_key = {}
    trial_cache_rows = []
    
    progress_root = os.path.join('logs', 'progresses', str(getattr(args, 'task', 'unknown')))
    cache_dir = os.path.join(progress_root, "tmp", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    trial_cache_path = os.path.join(
        cache_dir,
        f"{getattr(args, 'run_tag', 'RUN')}_{getattr(args, 'task', 'unknown')}_{getattr(args, 'exp_id', 'exp')}_optuna_trial_cache.json"
    )

    def _persist_trial_cache():
        payload = {
            "schema_version": 2,
            "exp_id": getattr(args, 'exp_id', ''),
            "run_tag": getattr(args, 'run_tag', ''),
            "task": getattr(args, 'task', ''),
            "seed": int(getattr(args, 'seed', 41)),
            "trials": trial_cache_rows,
            "updated_at": datetime.now().isoformat(timespec='seconds'),
        }
        with open(trial_cache_path, 'w') as f:
            json.dump(payload, f, indent=2)

    def safe_eval(trial):
        trial_counter['counter'] += 1
        trial_idx = trial_counter['counter']
        start_time = datetime.now().isoformat(timespec='seconds')
        print(f"[{start_time}] Optuna Trial {trial_idx}/{getattr(args, 'n_trials', 0)} | start")
        _append_trial_runtime_event(trial_idx, event='started', status='running')
        
        # Build params dictionary using Optuna suggestions
        params = {}
        params['lr'] = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        params['wd'] = trial.suggest_float("wd", 1e-8, 1e-2, log=True)
        params['smoothing'] = trial.suggest_float("smoothing", 0.0, 0.2)
        
        if optimize_params.get('dloss'):
            params['dloss'] = trial.suggest_categorical("dloss", ['no', 'inverseTriplet'])
        if optimize_params.get('classif_loss'):
            params['classif_loss'] = trial.suggest_categorical("classif_loss", ['arcface', 'triplet', 'softmax_contrastive', 'ce', 'hinge'])
        if optimize_params.get('prototypes_to_use'):
            params['prototypes_to_use'] = trial.suggest_categorical("prototypes_to_use", ['batch', 'no', 'class'])
            
        triplet_losses = ['triplet', 'softmax_contrastive']
        dloss_val = params.get('dloss', getattr(args, 'dloss', ''))
        classif_loss_val = params.get('classif_loss', getattr(args, 'classif_loss', ''))
        
        if optimize_params.get('n_positives') and classif_loss_val in triplet_losses:
            params['n_positives'] = trial.suggest_categorical("n_positives", [1])
        if optimize_params.get('n_negatives') and classif_loss_val in triplet_losses:
            params['n_negatives'] = trial.suggest_categorical("n_negatives", [1, 5])
        if optimize_params.get('fgsm'):
            params['fgsm'] = trial.suggest_categorical("fgsm", [0, 1])
        if optimize_params.get('normalize'):
            params['normalize'] = trial.suggest_categorical("normalize", ['yes', 'no'])
            
        proto_to_use = params.get('prototypes_to_use', getattr(args, 'prototypes_to_use', ''))
        if proto_to_use != 'no':
            params['prototype_strategy'] = trial.suggest_categorical("prototype_strategy", ['mean', 'kmeans'])
            params['prototype_components'] = trial.suggest_int("prototype_components", 1, 5)
            params['prototype_kind'] = trial.suggest_categorical("prototype_kind", ['distance', 'kde', 'distance_weighted'])
            params['kde_kernel'] = trial.suggest_categorical("kde_kernel", ['gaussian', 'exponential'])
            params['kde_bandwidth'] = trial.suggest_categorical("kde_bandwidth", ['scott', 'silverman'])
            
        if classif_loss_val in ['triplet', 'softmax_contrastive'] or dloss_val in ['revTriplet', 'inverseTriplet', 'normae', 'inverse_softmax_contrastive']:
            params['dist_fct'] = trial.suggest_categorical("dist_fct", ['cosine', 'euclidean'])
            params['dmargin'] = trial.suggest_float("dmargin", 0.0, 1.0)
            
        if dloss_val in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae', 'inverse_softmax_contrastive']:
            params['gamma'] = trial.suggest_float("gamma", 1e-2, 1e2, log=True)
            
        if classif_loss_val in ['triplet', 'softmax_contrastive']:
            params['margin'] = trial.suggest_float("margin", 0.0, 10.0)
            
        fgsm_val = params.get('fgsm', getattr(args, 'fgsm', 0))
        if fgsm_val:
            params['epsilon'] = trial.suggest_float("epsilon", 1e-4, 5e-1, log=True)
            
        if not args.auto_select_k and (classif_loss_val not in ['ce', 'hinge'] or dloss_val in ['inverse_softmax_contrastive', 'inverseTriplet']):
            params['n_neighbors'] = trial.suggest_int("n_neighbors", 1, 10, log=True)
            
        params['n_aug'] = trial.suggest_int("n_aug", 1, 5)
        # Check if the param combination is in the cache first to skip retraining
        cache_key = _params_cache_key(params)
        if cache_key in trial_cache_by_key:
            cached = trial_cache_by_key[cache_key]
            if cached.get('status') == 'completed':
                cached_score = float(cached.get('score', -1e9))
                cached_acc = float(cached.get('acc', 0.0))
                # Persist path to Optuna even if reused from cache
                log_path = cached.get('complete_log_path') or f"logs/{getattr(args, 'task', 'notNormal')}/{cached.get('foldername', 'unknown')}"
                trial.set_user_attr("complete_log_path", log_path)
                print(f"[{start_time}] Optuna Trial {trial_idx}/{getattr(args, 'n_trials', 0)} | reused | mcc={cached_score} acc={cached_acc}")
                _append_trial_runtime_event(trial_idx, event='reused', status='completed', score=cached_score)
                return cached_score
            elif cached.get('status') == 'failed':
                print(f"[{start_time}] Optuna Trial {trial_idx}/{getattr(args, 'n_trials', 0)} | requeue | previous attempt failed")
                _append_trial_runtime_event(trial_idx, event='requeue', status='running')
        
        try:
            import traceback
            train.args.trial_index = int(trial_idx)
            train.trial = trial  # Pass trial to train so it can report intermediate values for pruning
            score = train.train(params)
            end_time = datetime.now().isoformat(timespec='seconds')
            _finalize_trial_metadata(train, trial_idx, status='completed', error_message=None)
            trial.set_user_attr("complete_log_path", train.complete_log_path)
            _append_trial_runtime_event(trial_idx, event='completed', status='completed', score=score)
            
            row = {
                'trial_index': int(trial_idx),
                'status': 'completed',
                'score': float(score),
                'mcc': float(getattr(train, 'best_mcc', score)),
                'acc': float(getattr(train, 'best_acc', 0.0)),
            }
            # Spread all batch metrics into the row for easy CSV/Optuna analysis
            batch_metrics = getattr(train, 'batch_metrics', {})
            if isinstance(batch_metrics, dict):
                for k, v in batch_metrics.items():
                    try:
                        row[k] = float(v)
                    except (ValueError, TypeError):
                        row[k] = v
            row.update({
                'params': params,
                'cache_key': _params_cache_key(params),
                'complete_log_path': train.complete_log_path,
                'foldername': train.foldername,
                'start_time': start_time,
                'end_time': end_time
            })
            trial_cache_by_key[row['cache_key']] = row
            trial_cache_rows.append(row)
            _persist_trial_cache()
            
            print(f"[{end_time}] Optuna Trial {trial_idx}/{getattr(args, 'n_trials', 0)} | done | mcc={score} acc={row['acc']}")
            return score
        except optuna.TrialPruned as pruned_exc:
            # Trial was pruned due to low intermediate performance - this is expected and not a failure
            end_time = datetime.now().isoformat(timespec='seconds')
            _append_trial_runtime_event(trial_idx, event='pruned', status='pruned', score=-1.0)
            print(f"[{end_time}] Optuna Trial {trial_idx}/{getattr(args, 'n_trials', 0)} | pruned: {pruned_exc}")
            raise  # Re-raise so Optuna knows this was a pruned trial
        except Exception as exc:
            end_time = datetime.now().isoformat(timespec='seconds')
            _finalize_trial_metadata(train, trial_idx, status='failed', error_message=str(exc))
            _append_trial_runtime_event(trial_idx, event='failed', status='failed', score=-1e9, error_message=str(exc))
            
            # Per user request: let's not keep failures in cache, they stay in error logs
            print(f"[{end_time}] Optuna Trial {trial_idx}/{getattr(args, 'n_trials', 0)} | failed (not cached): {exc}")
            traceback.print_exc()
            return -1e9

    db_dir = os.path.join(progress_root, "tmp", "db")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{getattr(args, 'run_tag', 'RUN')}_{getattr(args, 'task', 'unknown')}_{getattr(args, 'exp_id', 'exp')}_optuna_{getattr(args, 'model_name', 'model')}.db")
    storage_name = f"sqlite:///{db_path}"

    if getattr(args, 'reset_opt_state', 0) and os.path.exists(db_path):
        os.remove(db_path)
        print(f"[Optuna Resume] reset_opt_state=1, removed {db_path}.")

    study_name = f"otitenet_study_{getattr(args, 'model_name', 'model')}"
    # TPESampler with warm_up_steps: explore more initially, then exploit
    n_trials_target = max(1, getattr(args, 'n_trials', 1))
    warmup_steps = max(3, n_trials_target // 4)  # 25% of trials for warmup exploration
    sampler = TPESampler(n_startup_trials=warmup_steps, seed=int(getattr(args, 'seed', 42)))
    # Optional Optuna pruner (default: MedianPruner, can be disabled with --optuna_pruner none)
    pruner_name = str(getattr(args, 'optuna_pruner', 'median')).strip().lower()
    if n_trials_target <= 10 or pruner_name == 'none':
        pruner = None
    else:
        pruner = MedianPruner(
            n_startup_trials=max(5, n_trials_target // 4),
            n_warmup_steps=max(8, int(getattr(args, 'n_epochs', 80) * 0.15)),
        )
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True, 
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    # Reload existing JSON cache (including the old Ax cache)
    ax_trial_cache_path = trial_cache_path.replace('_optuna_trial_cache.json', '_ax_trial_cache.json')
    for p in [ax_trial_cache_path, trial_cache_path]:
        if os.path.exists(p) and not getattr(args, 'reset_opt_state', 0):
            try:
                with open(p, 'r') as f:
                    payload = json.load(f)
                    for r in payload.get("trials", []):
                        ckey = r.get("cache_key")
                        if not ckey: continue
                        existing = trial_cache_by_key.get(ckey)
                        # Preference: completed > failed; otherwise latest wins.
                        if not existing or (r.get("status") == "completed" and existing.get("status") != "completed") or (r.get("status") == existing.get("status")):
                            trial_cache_by_key[ckey] = r
            except Exception:
                pass
    
    # Rebuild trial_cache_rows from the deduplicated map
    trial_cache_rows = list(trial_cache_by_key.values())

    # Figure out which cache params are NOT in Optuna yet
    optuna_known_params = set()
    already_evaluated = 0
    failures_to_redo = []
    
    for t in study.trials:
        if t.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED):
            continue
        
        ckey = _params_cache_key(t.params)
        optuna_known_params.add(ckey)
        
        # Success if value is valid, or if pruned
        if t.value is not None and t.value > -1e8:
            already_evaluated += 1
        elif t.state == optuna.trial.TrialState.PRUNED:
            already_evaluated += 1
        else:
            # Failure in Optuna: mark for redo
            failures_to_redo.append(t.params)

    # Count successful/pruned ones in cache that might not be in Optuna yet
    for r in trial_cache_rows:
        ckey = r.get("cache_key")
        if r.get("status") in ["completed", "pruned"] and ckey not in optuna_known_params:
            already_evaluated += 1

    trials_to_run = max(0, getattr(args, 'n_trials', 0) - already_evaluated)
    trial_counter['counter'] = already_evaluated

    print(f"[Optuna Resume] Scheduling {trials_to_run} additional trials (successful={already_evaluated}, target={getattr(args, 'n_trials', 0)}).")
    if trials_to_run == 0:
        print(
            "[Optuna Resume] No new trials scheduled. Existing DB/cache state already satisfies --n_trials. "
            "No fresh training loop will run, so DVCLive may not create a new run for this launch. "
            "To force a new run, use one of: --reset_opt_state=1, a different --run_tag/--exp_id, or a larger --n_trials."
        )
        print(f"[Optuna Resume] Reused DB: {db_path}")
        print(f"[Optuna Resume] Reused cache: {trial_cache_path}")

    # Enqueue trials from local cache to Optuna if missing
    enqueued_count = 0
    for r in trial_cache_rows:
        ckey = r.get("cache_key")
        if not ckey: continue
        
        if r.get("status") == "completed" and ckey not in optuna_known_params:
            try:
                study.enqueue_trial(r["params"])
                optuna_known_params.add(ckey)
                enqueued_count += 1
            except Exception:
                pass
        elif r.get("status") == "failed":
            # Old failures in cache: redo them
            try:
                study.enqueue_trial(r["params"])
                enqueued_count += 1
            except Exception:
                pass
                
    # Also redo Optuna-only failures
    for f_params in failures_to_redo:
        try:
            study.enqueue_trial(f_params)
            enqueued_count += 1
        except Exception:
            pass

    print(f"[Optuna Queue] Enqueued {enqueued_count} trials from local cache/failures to Optuna.")

    if trials_to_run > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Disable overly verbose output
        study.optimize(safe_eval, n_trials=trials_to_run)
        
    try:
        best_parameters = study.best_params
        best_trial = study.best_trial
    except ValueError:
        best_parameters = {}
        best_trial = None

    lists, traces = get_empty_traces()
    # values, _, _, _ = get_empty_dicts()  # Pas élégant
    # Loading best model that was saved during training
    # Build params path consistent with save_best_run()
    try:
        ds_part = _dataset_path_segment(train.path)
    except Exception:
        ds_part = "otite_ds_64"
    # Use best parameters if available to build the search path accurately
    best_dist = str(best_parameters.get("dist_fct", getattr(train.args, "dist_fct", "none"))) if isinstance(best_parameters, dict) else str(getattr(train.args, "dist_fct", "none"))
    best_knn = int(best_parameters.get("n_neighbors", getattr(train.args, "n_neighbors", 0))) if isinstance(best_parameters, dict) else int(getattr(train.args, "n_neighbors", 0))
    best_classif = str(best_parameters.get("classif_loss", train.args.classif_loss))
    best_dloss = str(best_parameters.get("dloss", train.args.dloss))
    best_proto = str(best_parameters.get("prototypes_to_use", train.args.prototypes_to_use))
    best_npos = int(best_parameters.get("n_positives", train.args.n_positives))
    best_nneg = int(best_parameters.get("n_negatives", train.args.n_negatives))
    
    base_dir = (
        f"logs/best_models/{train.args.task}/{train.args.model_name}/"
        f"{ds_part}/nsize{train.args.new_size}/fgsm{train.args.fgsm}/ncal{train.args.n_calibration}/"
        f"{best_classif}/{best_dloss}/prototypes_{best_proto}/"
        f"npos{best_npos}/nneg{best_nneg}"
    )

    # Prefer fully specified path with norm/dist/knn; else find any model.pth under base_dir; else fallback to current run.
    # Include cached completed trial paths so resume-with-0-trials can still run final evaluation.
    candidate_paths = []
    full_dir = os.path.join(
        base_dir,
        f"norm{train.args.normalize}",
        f"dist_{best_dist}",
        f"knn{best_knn}"
    )
    # Prefer Optuna-recorded best trial path if available.
    if best_trial:
        best_log_path = best_trial.user_attrs.get("complete_log_path")
        if best_log_path and str(best_log_path).strip().lower() not in {"none", "null", "nan", ""}:
            candidate_paths.append(os.path.join(best_log_path, "model.pth"))

    # Also consider all completed Optuna trials (highest score first), because older studies
    # may have a best trial without complete_log_path in user attrs.
    try:
        complete_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        complete_trials.sort(key=lambda t: float(t.value), reverse=True)
        for t in complete_trials:
            t_log_path = t.user_attrs.get("complete_log_path")
            if t_log_path and str(t_log_path).strip().lower() not in {"none", "null", "nan", ""}:
                candidate_paths.append(os.path.join(str(t_log_path), "model.pth"))
    except Exception:
        pass

    # Include local JSON cache entries from completed trials.
    try:
        cached_completed = [r for r in trial_cache_rows if r.get('status') == 'completed']
        cached_completed.sort(key=lambda r: float(r.get('score', -1e9)), reverse=True)
        for r in cached_completed:
            cached_log_path = r.get('complete_log_path')
            if cached_log_path and str(cached_log_path).strip().lower() not in {"none", "null", "nan", ""}:
                candidate_paths.append(os.path.join(str(cached_log_path), "model.pth"))
    except Exception:
        pass

    candidate_paths.append(os.path.join(full_dir, "model.pth"))


    def _datasets_match(meta, args):
        def _as_dataset_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple, set, np.ndarray)):
                raw_parts = [str(x).strip() for x in value]
            else:
                raw_parts = [x.strip() for x in str(value).replace(';', ',').split(',')]
            return [x.lower() for x in raw_parts if x and x.lower() not in {'none', 'nan', 'null'}]

        def _single_dataset(value):
            items = _as_dataset_list(value)
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            return ",".join(items)

        meta_args = meta.get('args') if isinstance(meta.get('args'), dict) else {}

        # Prefer effective split fields when present; fall back to explicit split fields.
        meta_train = _as_dataset_list(meta_args.get('effective_train_datasets', meta.get('train_datasets')))
        meta_valid = _single_dataset(meta_args.get('effective_valid_dataset', meta.get('valid_dataset')))
        meta_test = _single_dataset(meta_args.get('effective_test_dataset', meta.get('test_dataset')))

        arg_train = _as_dataset_list(
            getattr(args, 'effective_train_datasets', None) or getattr(args, 'train_datasets', '')
        )
        arg_valid = _single_dataset(
            getattr(args, 'effective_valid_dataset', None) or getattr(args, 'valid_dataset', '')
        )
        arg_test = _single_dataset(
            getattr(args, 'effective_test_dataset', None) or getattr(args, 'test_dataset', '')
        )

        checks = []
        if meta_train and arg_train:
            checks.append(sorted(meta_train) == sorted(arg_train))
        if meta_valid and arg_valid:
            checks.append(meta_valid == arg_valid)
        if meta_test and arg_test:
            checks.append(meta_test == arg_test)

        # If metadata does not include split fields, do not block checkpoint reuse.
        if not checks:
            return True
        return all(checks)

    def _path_matches_checkpoint_filters(model_path):
        return True

    if int(getattr(train.args, 'force_retrain', 0) or 0):
        print(
            "[CheckpointLoad] force_retrain=1: skipping checkpoint reuse and "
            "training a fresh model from best parameters."
        )
        retrain_params = dict(best_parameters) if isinstance(best_parameters, dict) else {}
        train.train(retrain_params)
        print("[CheckpointLoad] Forced retraining completed. Exiting resume final-pass path.")
        sys.exit(0)

    # De-duplicate candidate paths while preserving priority.
    dedup_candidate_paths = []
    seen_candidate_paths = set()
    for p in candidate_paths:
        if not p:
            continue
        key = os.path.abspath(str(p))
        if key in seen_candidate_paths:
            continue
        seen_candidate_paths.add(key)
        dedup_candidate_paths.append(str(p))
    candidate_paths = dedup_candidate_paths

    found_model_path = None
    weak_model_path = None
    for p in candidate_paths:
        if not _path_matches_checkpoint_filters(p):
            continue
        meta_path = os.path.join(os.path.dirname(p), 'run_metadata.json')
        if not os.path.exists(p):
            continue
        if not os.path.exists(meta_path):
            if weak_model_path is None:
                weak_model_path = p
            continue
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if _datasets_match(meta, train.args):
                found_model_path = p
                break
            if weak_model_path is None:
                weak_model_path = p
        except Exception as e:
            print(f"Warning: Could not read metadata {meta_path}: {e}")
            if weak_model_path is None:
                weak_model_path = p
    # Fallback: search all model.pth under base_dir and check metadata
    if found_model_path is None and os.path.isdir(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if "model.pth" in files:
                meta_path = os.path.join(root, 'run_metadata.json')
                candidate = os.path.join(root, "model.pth")
                if not _path_matches_checkpoint_filters(candidate):
                    continue
                if not os.path.exists(meta_path):
                    if weak_model_path is None:
                        weak_model_path = candidate
                    continue
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    if _datasets_match(meta, train.args):
                        found_model_path = candidate
                        break
                    if weak_model_path is None:
                        weak_model_path = candidate
                except Exception as e:
                    print(f"Warning: Could not read metadata {meta_path}: {e}")
                    if weak_model_path is None:
                        weak_model_path = candidate
    if found_model_path is None and weak_model_path is not None:
        found_model_path = weak_model_path
    # Last resort: use current training run's complete path, if a trial ran in this process.
    if found_model_path is None and getattr(train, "complete_log_path", None):
        found_model_path = f"{train.complete_log_path}/model.pth"

    if found_model_path is None or not os.path.exists(found_model_path):
        print("[CheckpointLoad] No compatible trained checkpoint was found for the requested dataset split.")
        print(f"[CheckpointLoad] Requested path: {getattr(train.args, 'path', '')}")
        print(f"[CheckpointLoad] Requested train_datasets: {getattr(train.args, 'train_datasets', '')}")
        print(f"[CheckpointLoad] Requested valid_dataset: {getattr(train.args, 'valid_dataset', '')}")
        print(f"[CheckpointLoad] Requested test_dataset: {getattr(train.args, 'test_dataset', '')}")
        print(f"[CheckpointLoad] Candidate count checked: {len(candidate_paths)}")
        if isinstance(best_parameters, dict) and best_parameters:
            print("[CheckpointLoad] Retraining once from Optuna best parameters to create a compatible checkpoint.")
            train.train(dict(best_parameters))
            print("[CheckpointLoad] Retraining completed. Exiting resume final-pass path.")
            sys.exit(0)
        print("[CheckpointLoad] No best parameters are available; run with --reset_opt_state=1 or check Optuna/cache state.")
        sys.exit(1)

    if not hasattr(train, "model") or not hasattr(train, "shap_model"):
        print("[Optuna Resume] Model attributes missing (all trials reused). Initializing minimal state for final evaluation...")
        data_getter, loaders = train.setup_training_objects(best_parameters)

    checkpoint_state = torch.load(
        found_model_path,
        map_location=train.args.device
    )

    def _strict_load_or_retrain_from_scratch(model, state_dict, model_name):
        try:
            model.load_state_dict(state_dict)
            return
        except RuntimeError as strict_exc:
            print(
                f"[CheckpointLoad] {model_name}: strict load failed due to shape mismatch. "
                "Training a new model from scratch with the selected best parameters."
            )
            print(f"[CheckpointLoad] Source checkpoint: {found_model_path}")
            print(f"[CheckpointLoad] Runtime error: {strict_exc}")

            retrain_params = dict(best_parameters) if isinstance(best_parameters, dict) else {}
            try:
                train.train(retrain_params)
            except Exception as retrain_exc:
                raise RuntimeError(
                    "Checkpoint shape mismatch detected, and fallback retraining from scratch failed."
                ) from retrain_exc

            print(
                "[CheckpointLoad] Retraining from scratch completed successfully. "
                "Exiting resume final-pass path to avoid mixing old and new checkpoints."
            )
            sys.exit(0)

    _strict_load_or_retrain_from_scratch(train.model, checkpoint_state, "train.model")
    # Need another model because the other can't be used to get shap values
    train.model.eval()
    if train.run_explainability and train.shap_model is not None:
        _strict_load_or_retrain_from_scratch(train.shap_model, checkpoint_state, "train.shap_model")
        train.shap_model.eval()
    prototypes = {
        'combined': train.combined_prototypes,
        'class': train.class_prototypes,
        'batch': train.batch_prototypes,
    }
    
    loaders = get_images_loaders(data=train.data,
                                 batch_encoder=train._batch_encoder,
                                    random_recs=train.args.random_recs,
                                    weighted_sampler=train.args.weighted_sampler,
                                    is_transform=1,
                                    samples_weights=train.samples_weights,
                                    epoch=1,
                                    unique_labels=train.unique_labels,
                                    triplet_dloss=train.args.dloss, bs=train.args.bs,
                                    prototypes_to_use=train.args.prototypes_to_use,
                                    prototypes=prototypes,
                                    size=train.args.new_size,
                                    normalize=train.args.normalize,
                                    n_aug=getattr(train.args, 'n_aug', 1),
                                    num_workers=getattr(train.args, 'num_workers', 0),
                                    )
    print("\n========== POST-OPTIMIZATION FINAL EVALUATION ==========")
    print("Evaluating best checkpoint with train loop snapshot + train/valid/test predictions...")
    best_lists = {}
    with torch.no_grad():
        # Keep one loop snapshot for training dynamics, then compute inference probabilities via predict().
        _, _, _ = train.loop('train', None, train.params['gamma'], loaders['train'], lists, traces)
        for group in ["train", "valid", "test"]:
            _, best_lists[group], traces, _ = train.predict(group, loaders[group], lists, traces)
        
    best_lists = {**best_lists['train'], **best_lists['valid'], **best_lists['test']}
    if train.log_comet:
        COMET_API_KEY = os.environ.get("COMET_API_KEY", "")
        COMET_PROJECT_NAME = os.environ.get("COMET_PROJECT_NAME", "otitenet")
        experiment = _make_comet_experiment(COMET_API_KEY, COMET_PROJECT_NAME)
        if experiment is not None:
            experiment = set_run(experiment, train.best_params)
            # Register all parameters (fixed and optimized) to Comet
            register_all_params_to_comet(experiment, train.args, train.best_params)
            train.log_predictions(best_lists, None, experiment, 0)
            # Ensure classifier is defined by running predict for 'test' group
            _, _, _, classifier = train.predict('test', loaders['test'], lists, traces)
            if train.run_explainability and train.shap_model is not None:
                train.save_wrong_classif_imgs(experiment, {'cnn': train.shap_model, 'knn': classifier}, best_lists, best_lists['test']['preds'], 
                                            best_lists['test']['names'], 'test')
            else:
                print('[INFO] Explainability disabled (run_explainability=0): skipping SHAP/Grad-CAM artifact generation.')
            
            experiment.end()
