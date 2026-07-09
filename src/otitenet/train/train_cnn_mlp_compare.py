import argparse
import hashlib
import json
import os
import traceback
import uuid
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from sklearn.metrics import matthews_corrcoef as MCC
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    VGG16_Weights,
    densenet121,
    efficientnet_b0,
    resnet18,
    resnet50,
    vit_b_16,
    vgg16,
)

from ..data.data_getters import GetData, get_images_loaders
from ..utils.memory_telemetry import (
    emit_gpu_telemetry,
    estimate_theoretical_gpu_required_mb,
    gpu_memory_stats,
    reset_gpu_peak,
    record_gpu_peak,
)
from otitenet.utils.utils import set_random_seeds as _set_global_seeds
from .completed_runs_metrics import append_completed_run_metrics

try:
    import mysql.connector
except Exception:
    mysql = None

try:
    import mlflow
except Exception:  # Optional dependency in some environments.
    mlflow = None

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except Exception:  # Optional dependency in some environments.
    optuna = None
    TPESampler = None
    MedianPruner = None


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _csv_escape(value):
    text = "" if value in [None, ""] else str(value)
    text = text.replace('"', '""')
    return f'"{text}"'


def _normalize_train_datasets(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [str(x).strip() for x in value]
    else:
        parts = [x.strip() for x in str(value).replace(";", ",").split(",")]
    clean = [x for x in parts if x and x.lower() not in {"none", "nan", "null"}]
    return ",".join(list(dict.fromkeys(clean)))


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
    train_datasets = _normalize_train_datasets(getattr(args, "train_datasets", ""))
    valid_dataset = _normalize_single_dataset(getattr(args, "valid_dataset", ""))
    test_dataset = _normalize_single_dataset(getattr(args, "test_dataset", ""))
    split_key = "|".join([train_datasets, valid_dataset, test_dataset])
    return {
        "train_datasets": train_datasets,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "split_config_key": split_key,
    }


def _ensure_best_registry_columns(cursor, conn):
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
    except Exception:
        pass


def _best_registry_config_key(params):
    key_fields = [
        str(params.get('task', '') or '').strip(),
        str(params.get('model_name', '') or '').strip(),
        str(params.get('dataset_path', '') or '').strip().replace('\\', '/'),
        str(params.get('train_datasets', '') or '').strip(),
        str(params.get('valid_dataset', '') or '').strip(),
        str(params.get('test_dataset', '') or '').strip(),
        str(params.get('split_config_key', '') or '').strip(),
        str(params.get('variant', '') or '').strip(),
        str(params.get('classif_loss', '') or '').strip(),
        str(params.get('dloss', '') or '').strip(),
        str(params.get('fgsm', '') or '').strip(),
        str(params.get('n_calibration', '') or '').strip(),
        str(params.get('normalize', '') or '').strip(),
        str(params.get('nsize', '') or '').strip(),
    ]
    return hashlib.sha1("|".join(key_fields).encode("utf-8")).hexdigest()


def append_run_metrics_db(row, run_log_path=""):
    if mysql is None:
        return
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
    except Exception as e:
        print(f"[Warning] Could not append run into model_runs_registry: {e}")


def update_best_model_registry(params, accuracy, mcc, log_path, batch_metrics=None, source_run_log_path=None):
    if mysql is None:
        return
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="y_user",
            password="password",
            database="results_db"
        )
        cursor = conn.cursor()
        _ensure_best_registry_columns(cursor, conn)

        split_key = params.get('split_config_key', '')
        config_key = params.get('config_key') or _best_registry_config_key(params)
        artifact_id = hashlib.sha1(str(log_path).replace("\\", "/").strip("/").encode("utf-8")).hexdigest()[:12]
        batch_entropy = batch_metrics.get('batch_entropy_norm') if batch_metrics else None
        batch_nmi = batch_metrics.get('batch_nmi') if batch_metrics else None
        batch_ari = batch_metrics.get('batch_ari') if batch_metrics else None

        cursor.execute(
            """
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
                dloss = VALUES(dloss),
                classif_loss = VALUES(classif_loss),
                n_calibration = VALUES(n_calibration),
                normalize = VALUES(normalize),
                train_datasets = VALUES(train_datasets),
                valid_dataset = VALUES(valid_dataset),
                test_dataset = VALUES(test_dataset),
                split_config_key = VALUES(split_config_key)
            """,
            (
                config_key,
                params.get('task', ''),
                params.get('model_name', ''),
                params.get('dataset_path', ''),
                params.get('siamese_inference', ''),
                params.get('run_tag', ''),
                str(params.get('nsize', '')),
                str(params.get('fgsm', '')),
                str(params.get('prototypes', 'no')),
                str(params.get('npos', 1)),
                str(params.get('nneg', 1)),
                str(params.get('dloss', '')),
                str(params.get('dist_fct', 'none')),
                str(params.get('classif_loss', '')),
                str(params.get('n_calibration', '')),
                float(accuracy),
                float(mcc),
                batch_entropy,
                batch_nmi,
                batch_ari,
                str(params.get('normalize', '')),
                str(params.get('n_neighbors', 0)),
                str(params.get('prototype_strategy', 'mean')),
                int(params.get('prototype_components', 1)),
                str(params.get('train_datasets', '')),
                str(params.get('valid_dataset', '')),
                str(params.get('test_dataset', '')),
                split_key,
                log_path,
                artifact_id,
                log_path,
                source_run_log_path,
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[Warning] Could not update best_models_registry from cnn/mlp run: {e}")



def _append_trial_runtime_event(args, trial_idx, total_trials, event, status, score=None, error_message=None, run_dir=None):
    progress_root = os.path.join('logs', 'progresses', str(getattr(args, 'task', 'unknown')))
    csv_dir = os.path.join(progress_root, "csv")
    _ensure_dir(csv_dir)
    csv_path = os.path.join(csv_dir, f"{getattr(args, 'run_tag', 'RUN')}_{getattr(args, 'task', 'unknown')}_trial_runtime.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('timestamp,exp_id,run_tag,task,trial_index,total_trials,event,status,score,error_message,run_dir\n')

    ts = datetime.now().isoformat(timespec='seconds')
    row = [
        _csv_escape(ts),
        _csv_escape(getattr(args, 'exp_id', '')),
        _csv_escape(getattr(args, 'run_tag', '')),
        _csv_escape(getattr(args, 'task', '')),
        _csv_escape(trial_idx),
        _csv_escape(total_trials),
        _csv_escape(event),
        _csv_escape(status),
        _csv_escape(score if score is not None else ''),
        _csv_escape(error_message if error_message is not None else ''),
        _csv_escape(run_dir if run_dir is not None else ''),
    ]
    with open(csv_path, 'a') as f:
        f.write(','.join(row) + '\n')


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


def _build_sample_weights(cats):
    cats = np.asarray(cats)
    cls_ids, counts = np.unique(cats, return_counts=True)
    inv = {int(c): 1.0 / max(int(n), 1) for c, n in zip(cls_ids, counts)}
    weights = np.array([inv[int(c)] for c in cats], dtype=np.float32)
    return weights


class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        feats = self.features(x).flatten(1)
        logits = self.head(feats)
        return logits, feats


class ImageMLP(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        in_features = int(np.prod(in_shape))
        self.feature_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        feats = self.feature_net(x)
        logits = self.classifier(feats)
        return logits, feats


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class TransferNet(nn.Module):
    def __init__(self, model_name, n_classes, transfer_learning):
        super().__init__()
        model_name = model_name.lower()
        self.model_name = model_name
        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if transfer_learning else None
            base = resnet18(weights=weights)
            self.features = nn.Sequential(*list(base.children())[:-1])
            self.feat_dim = base.fc.in_features
            self.classifier = nn.Linear(self.feat_dim, n_classes)
        elif model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if transfer_learning else None
            base = resnet50(weights=weights)
            self.features = nn.Sequential(*list(base.children())[:-1])
            self.feat_dim = base.fc.in_features
            self.classifier = nn.Linear(self.feat_dim, n_classes)
        elif model_name == "vgg16":
            weights = VGG16_Weights.DEFAULT if transfer_learning else None
            base = vgg16(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            self.feat_mlp = nn.Sequential(*list(base.classifier.children())[:-1])
            self.feat_dim = base.classifier[-1].in_features
            self.classifier = nn.Linear(self.feat_dim, n_classes)
        elif model_name == "densenet121":
            weights = DenseNet121_Weights.DEFAULT if transfer_learning else None
            base = densenet121(weights=weights)
            self.features = base.features
            self.feat_dim = base.classifier.in_features
            self.classifier = nn.Linear(self.feat_dim, n_classes)
        elif model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if transfer_learning else None
            base = efficientnet_b0(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            self.feat_dim = base.classifier[1].in_features
            self.classifier = nn.Linear(self.feat_dim, n_classes)
        elif model_name == "vit":
            weights = ViT_B_16_Weights.DEFAULT if transfer_learning else None
            self.vit = vit_b_16(weights=weights)
            self.vit_image_size = int(getattr(self.vit, "image_size", 224))
            self.feat_dim = self.vit.heads.head.in_features
            self.vit.heads = nn.Identity()
            self.classifier = nn.Linear(self.feat_dim, n_classes)
        else:
            raise ValueError(f"Unsupported backbone: {model_name}")

    def forward(self, x):
        if self.model_name in ["resnet18", "resnet50"]:
            feats = self.features(x).flatten(1)
        elif self.model_name == "vgg16":
            feats = self.features(x)
            feats = self.avgpool(feats)
            feats = torch.flatten(feats, 1)
            feats = self.feat_mlp(feats)
        elif self.model_name == "efficientnet_b0":
            feats = self.features(x)
            feats = self.avgpool(feats)
            feats = torch.flatten(feats, 1)
        elif self.model_name == "vit":
            if x.shape[-2] != self.vit_image_size or x.shape[-1] != self.vit_image_size:
                x = nn.functional.interpolate(
                    x,
                    size=(self.vit_image_size, self.vit_image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            feats = self.vit(x)
        else:
            feats = self.features(x)
            feats = nn.functional.relu(feats, inplace=False)
            feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        logits = self.classifier(feats)
        return logits, feats


def _batch_to_xy(batch, device):
    data, names, labels, _, old_labels, domain, *_ = batch
    data = data.to(device).float()
    labels = labels.to(device).long().reshape(-1)
    domain = domain.to(device).long().reshape(-1)
    return data, labels, domain, names, old_labels


def _build_optimizer(parameters, lr, wd, optimizer_type):
    optimizer_type = str(optimizer_type).lower()
    if optimizer_type == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
    return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)


def _build_classification_loss(loss_name, label_smoothing):
    lname = str(loss_name).lower()
    if lname == "hinge":
        return nn.MultiMarginLoss()
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _use_domain_loss(dloss_name):
    return str(dloss_name).lower() in ["dann", "inversetriplet"]


def _resolve_strategy_label(classif_loss, knn):
    loss_txt = str(classif_loss).strip()
    if loss_txt and loss_txt.lower() not in {"nan", "none", "unknown"}:
        return loss_txt

    knn_num = pd.to_numeric(knn, errors='coerce')
    if pd.notna(knn_num):
        return f"knn={int(knn_num)}"

    knn_txt = str(knn).strip()
    if knn_txt and knn_txt.lower() not in {"nan", "none", "unknown"}:
        return f"knn={knn_txt}"

    return "unknown"


@dataclass
class EpochStats:
    loss: float
    acc: float
    mcc: float


class CNNSupervisedCompare:
    def _extract_features_and_batches(self, model, loader):
        model.eval()
        features = []
        batches = []
        with torch.no_grad():
            for batch in loader:
                x, _, d, _, _ = _batch_to_xy(batch, self.device)
                _, feats = model(x)
                features.append(feats.cpu().numpy())
                batches.append(d.cpu().numpy())
        features = np.concatenate(features, axis=0)
        batches = np.concatenate(batches, axis=0)
        return features, batches

    def _compute_batch_metrics(self, model, loaders):
        # Import batch metrics function
        from otitenet.train.batch_effects import get_batch_metrics
        lists = {}
        for split in ['train', 'valid', 'test']:
            feats, batches = self._extract_features_and_batches(model, loaders[split])
            lists[split] = {'encoded_values': [feats], 'domains': [batches]}
        metrics = get_batch_metrics(lists)
        return metrics

    def _append_completed_run_csv(self, params, best_vals, batch_metrics, run_uuid, final_status, final_error, trial_index_override=None):
        """
        Append a row to the completed_runs CSV with all key metadata and metrics for this run (CNN/MLP version),
        using pandas to ensure correct column alignment by header names.
        """
        from datetime import datetime
        run_tag = getattr(self.args, 'run_tag', 'PROD')
        task = getattr(self.args, 'task', 'notNormal')
        split_config = _split_config_from_args(self.args)
        completed_csv = f'logs/progresses/{task}/{run_tag}_{task}_completed_runs_metrics.csv'
        global_csv = 'completed_runs_metrics.csv'
        source_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_name = getattr(self.args, 'model_name', '')
        classif_loss = getattr(self.args, 'classif_loss', '')
        knn_value = ''
        strategy_label = _resolve_strategy_label(classif_loss, knn_value)
        row = {
            'timestamp': source_ts,
            'exp_id': getattr(self.args, 'exp_id', ''),
            'trial_index': trial_index_override if trial_index_override is not None else getattr(self.args, 'trial_index', ''),
            'uuid': run_uuid,
            'status': final_status,
            'error': final_error,
            'task': task,
            'run_tag': run_tag,
            'model': model_name,
            'model_name': getattr(self.args, 'model_name', ''),
            'kind': 'cnn_mlp',
            'variant': params.get('variant', ''),
            'loss': strategy_label,
            'classif_loss': strategy_label,
            'prototype': '',
            'prototypes': '',
            'dloss': getattr(self.args, 'dloss', ''),
            'BER': '',
            'fgsm': getattr(self.args, 'fgsm', ''),
            'normalize': getattr(self.args, 'normalize', ''),
            'n_calibration': getattr(self.args, 'n_calibration', ''),
            'dist_fct': '',
            'knn': knn_value,
            'n_negatives': '',
            'train_datasets': split_config['train_datasets'],
            'valid_dataset': split_config['valid_dataset'],
            'test_dataset': split_config['test_dataset'],
            'split_config_key': split_config['split_config_key'],
            'retry_count': getattr(self.args, 'retry_count', ''),
            'valid_mcc': best_vals.get('valid_mcc', ''),
            'test_mcc': best_vals.get('test_mcc', ''),
            'valid_accuracy': best_vals.get('valid_acc', ''),
            'test_accuracy': best_vals.get('test_acc', ''),
            'launcher_retry_count': getattr(self.args, 'retry_count', ''),
            'launcher_failed_final': getattr(self.args, 'failed_final', ''),
            'source_datetime': source_ts,
            'batch_entropy_norm': (
                batch_metrics.get('batch_entropy_norm', batch_metrics.get('batch_entropy', ''))
                if batch_metrics else ''
            ),
            'batch_nmi': batch_metrics.get('batch_nmi', '') if batch_metrics else '',
            'batch_ari': batch_metrics.get('batch_ari', '') if batch_metrics else '',
        }

        try:
            append_completed_run_metrics(completed_csv, row)
        except Exception as e:
            print(f"[Warning] Could not write to per-task completed_runs_metrics.csv: {e}")

        try:
            append_completed_run_metrics(global_csv, row)
        except Exception as e:
            print(f"[Warning] Could not write to global completed_runs_metrics.csv: {e}")

        append_run_metrics_db(row, run_log_path=str(best_vals.get('run_dir', '') or ''))

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.amp_enabled = bool(int(getattr(self.args, 'amp', 1))) and str(getattr(self.args, 'device', '')).startswith('cuda')
        amp_dtype_name = str(getattr(self.args, 'amp_dtype', 'bf16')).lower()
        if amp_dtype_name not in ['bf16', 'fp16']:
            amp_dtype_name = 'bf16'
        self.amp_dtype = torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16
        self.use_grad_scaler = self.amp_enabled and self.amp_dtype == torch.float16
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)
        self.run_id = str(uuid.uuid4())
        self.run_root = os.path.join("logs", args.task, "cnn_mlp_compare", args.run_tag, self.run_id)
        _ensure_dir(self.run_root)
        self._write_run_metadata(finished=False, status="running", error_message=None)

        _set_global_seeds(getattr(args, "seed", 42), deterministic=True)

        data_getter = GetData(args.path, args.valid_dataset, args, manifest_dir=self.run_root)
        self.data, self.unique_labels, self.unique_batches = data_getter.get_variables()
        self.n_classes = len(self.unique_labels)
        self.n_domains = len(self.unique_batches)

        self.samples_weights = {
            "train": _build_sample_weights(self.data["cats"]["train"]),
            "valid": np.ones(len(self.data["cats"]["valid"]), dtype=np.float32),
            "test": np.ones(len(self.data["cats"]["test"]), dtype=np.float32),
        }

    def _autocast_context(self):
        if self.amp_enabled:
            return torch.autocast(device_type='cuda', dtype=self.amp_dtype)
        return nullcontext()

    def _print_epoch_progress(self, epoch_idx, total_epochs, phase, step_idx, total_steps):
        if total_steps <= 0:
            return
        width = 24
        progress = float(step_idx) / float(total_steps)
        filled = int(progress * width)
        bar = "#" * filled + "-" * (width - filled)
        pct = progress * 100.0
        remaining = max(int(total_steps) - int(step_idx), 0)
        print(
            f"Epoch {epoch_idx + 1}/{total_epochs} [{phase}] "
            f"|{bar}| {step_idx}/{total_steps} ({pct:5.1f}%) remaining:{remaining}"
        )

    def _write_run_metadata(self, finished, status, error_message=None):
        metadata_path = os.path.join(self.run_root, "run_metadata.json")
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "finished": bool(finished),
            "run_status": str(status),
            "error_message": None if error_message in [None, ""] else str(error_message),
            "run_root": self.run_root,
            "run_id": self.run_id,
            "args": vars(self.args),
        }
        if bool(finished):
            payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _make_loaders(self):
        prototypes = {"combined": {}, "class": {}, "batch": {}}
        batch_encoder = LabelEncoder().fit(np.array(self.unique_batches))
        return get_images_loaders(
            data=self.data,
            batch_encoder=batch_encoder,
            random_recs=self.args.random_recs,
            weighted_sampler=self.args.weighted_sampler,
            is_transform=1,
            samples_weights=self.samples_weights,
            epoch=0,
            unique_labels=self.unique_labels,
            triplet_dloss="no",
            prototypes_to_use="no",
            n_positives=1,
            n_negatives=1,
            prototypes=prototypes,
            bs=self.args.bs,
            size=self.args.new_size,
            normalize=self.args.normalize,
            n_aug=self.args.n_aug,
            num_workers=getattr(self.args, "num_workers", 0),
        )

    def _train_epoch(self, model, domain_head, loader, classif_criterion, domain_criterion, optimizer, epoch_idx, total_epochs):
        model.train()
        if domain_head is not None:
            domain_head.train()
        confmat = None
        losses = []
        total_steps = len(loader)
        for step_idx, batch in enumerate(loader, start=1):
            x, y, d, _, _ = _batch_to_xy(batch, self.device)
            if int(self.args.fgsm) == 1:
                x = x.detach().requires_grad_(True)
            optimizer.zero_grad()
            with self._autocast_context():
                logits, feats = model(x)
                classif_loss = classif_criterion(logits, y)
                domain_loss = torch.tensor(0.0, device=self.device)
                if _use_domain_loss(self.args.dloss) and domain_head is not None:
                    rev_feats = ReverseLayerF.apply(feats, 1.0)
                    domain_logits = domain_head(rev_feats)
                    domain_loss = domain_criterion(domain_logits, d)
                total_loss = classif_loss + float(self.args.gamma) * domain_loss
            if self.use_grad_scaler:
                self.grad_scaler.scale(total_loss).backward(retain_graph=int(self.args.fgsm) == 1)
            else:
                total_loss.backward(retain_graph=int(self.args.fgsm) == 1)

            if int(self.args.fgsm) == 1 and x.grad is not None:
                adv_x = torch.clamp(x + float(self.args.epsilon) * x.grad.sign(), -1.0, 1.0).detach()
                with self._autocast_context():
                    adv_logits, adv_feats = model(adv_x)
                    adv_classif = classif_criterion(adv_logits, y)
                    adv_total = adv_classif
                    if _use_domain_loss(self.args.dloss) and domain_head is not None:
                        adv_domain = domain_head(ReverseLayerF.apply(adv_feats, 1.0))
                        adv_total = adv_total + float(self.args.gamma) * domain_criterion(adv_domain, d)
                if self.use_grad_scaler:
                    self.grad_scaler.scale(0.1 * adv_total).backward()
                else:
                    (0.1 * adv_total).backward()

            if self.use_grad_scaler:
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
            else:
                optimizer.step()
            record_gpu_peak(self.device)

            losses.append(float(total_loss.item()))
            pred = logits.argmax(1)
            confmat = _update_confmat_gpu(confmat, y, pred, self.n_classes)
            self._print_epoch_progress(epoch_idx, total_epochs, "train", step_idx, total_steps)

        acc, mcc = _acc_mcc_from_confmat_gpu(confmat)
        return EpochStats(loss=float(np.mean(losses) if losses else 0.0), acc=acc, mcc=mcc)

    def _eval_epoch(self, model, domain_head, loader, classif_criterion, domain_criterion, epoch_idx, total_epochs, phase):
        model.eval()
        if domain_head is not None:
            domain_head.eval()
        all_true, all_pred, all_names, all_old_labels = [], [], [], []
        confmat = None
        losses = []
        total_steps = len(loader)
        with torch.no_grad():
            for step_idx, batch in enumerate(loader, start=1):
                x, y, d, names, old_labels = _batch_to_xy(batch, self.device)
                with self._autocast_context():
                    logits, feats = model(x)
                    classif_loss = classif_criterion(logits, y)
                    domain_loss = torch.tensor(0.0, device=self.device)
                    if _use_domain_loss(self.args.dloss) and domain_head is not None:
                        domain_logits = domain_head(feats)
                        domain_loss = domain_criterion(domain_logits, d)
                    loss = classif_loss + float(self.args.gamma) * domain_loss
                pred = logits.argmax(1)

                losses.append(float(loss.item()))
                all_true.append(y.detach().cpu().numpy())
                all_pred.append(pred.detach().cpu().numpy())
                confmat = _update_confmat_gpu(confmat, y, pred, self.n_classes)
                all_names.extend(list(names))
                all_old_labels.extend(list(old_labels))
                self._print_epoch_progress(epoch_idx, total_epochs, phase, step_idx, total_steps)

        y_true = np.concatenate(all_true) if all_true else np.array([])
        y_pred = np.concatenate(all_pred) if all_pred else np.array([])
        acc, mcc = _acc_mcc_from_confmat_gpu(confmat)
        pred_df = pd.DataFrame(
            {
                "name": all_names,
                "true_cat": y_true,
                "pred_cat": y_pred,
                "true_label": [self.unique_labels[t] for t in y_true] if y_true.size else [],
                "pred_label": [self.unique_labels[p] for p in y_pred] if y_pred.size else [],
                "old_label": all_old_labels,
            }
        )
        return EpochStats(loss=float(np.mean(losses) if losses else 0.0), acc=acc, mcc=mcc), pred_df

    def _build_model(self, variant):
        if variant == "cnn_transfer":
            model = TransferNet(self.args.model_name, self.n_classes, transfer_learning=True)
            return model, model.feat_dim
        if variant == "cnn_scratch":
            model = SimpleCNN(self.n_classes)
            return model, 128
        if variant == "mlp":
            # Use hidden representation size proxy to size domain head.
            model = ImageMLP((3, self.args.new_size, self.args.new_size), self.n_classes)
            return model, 256
        raise ValueError(f"Unknown variant: {variant}")

    def run_variant(self, variant):
        model_dir = os.path.join(self.run_root, variant)
        _ensure_dir(model_dir)
        theoretical_gpu_required_mb = None

        loaders = self._make_loaders()
        print("Repeat: 0")
        print(f"Train Batches: {np.unique(self.data['batches']['train'])}")
        print(f"Valid Batches: {np.unique(self.data['batches']['valid'])}")
        print(f"Test Batches: {np.unique(self.data['batches']['test'])}")
        print("")
        model, feat_dim = self._build_model(variant)
        model = model.to(self.device)
        domain_head = None
        if _use_domain_loss(self.args.dloss):
            domain_head = nn.Linear(int(feat_dim), self.n_domains).to(self.device)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            reset_gpu_peak(self.device)
            theoretical_gpu_required_mb = estimate_theoretical_gpu_required_mb(
                model,
                int(getattr(self.args, "bs", 32)),
                int(getattr(self.args, "new_size", 64)),
            )
            emit_gpu_telemetry(
                "start",
                theoretical_gpu_required_mb=f"{theoretical_gpu_required_mb:.2f}",
                variant=variant,
            )

        classif_criterion = _build_classification_loss(self.args.classif_loss, self.args.label_smoothing)
        domain_criterion = nn.CrossEntropyLoss()

        params = list(model.parameters())
        if domain_head is not None:
            params += list(domain_head.parameters())
        optimizer = _build_optimizer(params, self.args.lr, self.args.wd, self.args.optimizer_type)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

        best_valid_mcc = -1.0
        best_valid_acc = 0.0
        best_epoch = -1
        best_test_stats = None
        patience_counter = 0

        history_rows = []

        total_epochs = int(self.args.n_epochs)
        for epoch in range(total_epochs):
            # Mirror Siamese logging by showing an "all" split pass each epoch.
            self._eval_epoch(
                model, domain_head, loaders["all"], classif_criterion, domain_criterion, epoch, total_epochs, "all"
            )
            tr = self._train_epoch(
                model, domain_head, loaders["train"], classif_criterion, domain_criterion, optimizer, epoch, total_epochs
            )
            va, va_df = self._eval_epoch(
                model, domain_head, loaders["valid"], classif_criterion, domain_criterion, epoch, total_epochs, "valid"
            )
            te, te_df = self._eval_epoch(
                model, domain_head, loaders["test"], classif_criterion, domain_criterion, epoch, total_epochs, "test"
            )
            scheduler.step(va.mcc)

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": tr.loss,
                    "train_acc": tr.acc,
                    "train_mcc": tr.mcc,
                    "valid_loss": va.loss,
                    "valid_acc": va.acc,
                    "valid_mcc": va.mcc,
                    "test_loss": te.loss,
                    "test_acc": te.acc,
                    "test_mcc": te.mcc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            if va.mcc > best_valid_mcc:
                best_valid_mcc = va.mcc
                best_valid_acc = va.acc
                best_epoch = epoch
                best_test_stats = te
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                if domain_head is not None:
                    torch.save(domain_head.state_dict(), os.path.join(model_dir, "best_domain_head.pth"))
                va_df.to_csv(os.path.join(model_dir, "valid_predictions.csv"), index=False)
                te_df.to_csv(os.path.join(model_dir, "test_predictions.csv"), index=False)
            else:
                patience_counter += 1

            # Report intermediate value to Optuna for pruning
            if hasattr(self, 'trial') and self.trial is not None:
                try:
                    self.trial.report(float(va.mcc), step=epoch)
                    if self.trial.should_prune():
                        print(f"[{variant}] Trial pruned at epoch {epoch} due to low validation MCC: {va.mcc:.4f}")
                        raise optuna.TrialPruned(f"Validation MCC {va.mcc:.4f} fell below median at epoch {epoch}")
                except optuna.TrialPruned:
                    raise
                except Exception:
                    pass  # Don't fail training if trial.report() has issues

            if patience_counter >= self.args.early_stop:
                break

            if int(self.args.verbose) == 1:
                print(
                    f"[{variant}] epoch={epoch} train_mcc={tr.mcc:.4f} "
                    f"valid_mcc={va.mcc:.4f} test_mcc={te.mcc:.4f}"
                )

        pd.DataFrame(history_rows).to_csv(os.path.join(model_dir, "history.csv"), index=False)

        # Compute batch effect metrics using last hidden layer (features)
        batch_metrics = self._compute_batch_metrics(model, loaders)
        result = {
            "exp_id": self.args.exp_id,
            "run_tag": self.args.run_tag,
            "finished": True,
            "variant": variant,
            "model_name": self.args.model_name,
            "dloss": self.args.dloss,
            "classif_loss": self.args.classif_loss,
            "fgsm": int(self.args.fgsm),
            "n_calibration": int(self.args.n_calibration),
            "best_epoch": int(best_epoch),
            "valid_mcc": float(best_valid_mcc),
            "valid_acc": float(best_valid_acc),
            "test_mcc": float(best_test_stats.mcc if best_test_stats else 0.0),
            "test_acc": float(best_test_stats.acc if best_test_stats else 0.0),
            "n_classes": int(self.n_classes),
            "run_dir": model_dir,
            "batch_metrics": batch_metrics,
        }
        with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            stats = gpu_memory_stats(self.device)
            emit_gpu_telemetry(
                "end",
                theoretical_gpu_required_mb=(f"{theoretical_gpu_required_mb:.2f}" if theoretical_gpu_required_mb is not None else None),
                actual_peak_gpu_mb=(f"{stats.get('actual_peak_gpu_mb', 0.0):.2f}" if stats else None),
                variant=variant,
            )

        if int(self.args.log_mlflow) == 1 and mlflow is not None:
            try:
                mlflow.set_experiment(f"cnn_mlp_compare_{self.args.task}")
                with mlflow.start_run(run_name=f"{self.args.exp_id}_{variant}_{self.args.run_tag}"):
                    mlflow.log_params(
                        {
                            "exp_id": self.args.exp_id,
                            "run_tag": self.args.run_tag,
                            "variant": variant,
                            "model_name": self.args.model_name,
                            "dloss": self.args.dloss,
                            "classif_loss": self.args.classif_loss,
                            "fgsm": int(self.args.fgsm),
                            "n_calibration": int(self.args.n_calibration),
                            "test_mode": 1 if "test" in self.args.run_tag.lower() else 0,
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "valid_mcc": float(best_valid_mcc),
                            "test_mcc": float(best_test_stats.mcc if best_test_stats else 0.0),
                            "test_acc": float(best_test_stats.acc if best_test_stats else 0.0),
                        }
                    )
            except Exception:
                pass
        return result

    def run(self):
        variants = [self.args.variant]
        if self.args.compare_all:
            variants = ["cnn_transfer", "cnn_scratch", "mlp"]

        base_trial_index = int(getattr(self.args, 'trial_index', 0) or 0)
        n_variants = max(len(variants), 1)

        all_results = []
        for variant_pos, variant in enumerate(variants):
            if base_trial_index > 0:
                row_trial_index = (base_trial_index - 1) * n_variants + variant_pos + 1
            else:
                row_trial_index = variant_pos + 1
            try:
                if int(self.args.verbose) == 1:
                    print(f"\n=== Training {variant} ===")
                result = self.run_variant(variant)
                all_results.append(result)
                split_config = _split_config_from_args(self.args)
                best_registry_params = {
                    'task': getattr(self.args, 'task', ''),
                    'model_name': getattr(self.args, 'model_name', ''),
                    'dataset_path': _dataset_path_segment(getattr(self.args, 'path', '')),
                    'siamese_inference': '',
                    'run_tag': getattr(self.args, 'run_tag', ''),
                    'variant': variant,
                    'classif_loss': getattr(self.args, 'classif_loss', ''),
                    'dloss': getattr(self.args, 'dloss', ''),
                    'fgsm': getattr(self.args, 'fgsm', ''),
                    'n_calibration': getattr(self.args, 'n_calibration', ''),
                    'normalize': getattr(self.args, 'normalize', ''),
                    'nsize': getattr(self.args, 'new_size', ''),
                    'prototypes': 'no',
                    'npos': 1,
                    'nneg': 1,
                    'dist_fct': 'none',
                    'n_neighbors': 0,
                    'prototype_strategy': 'mean',
                    'prototype_components': 1,
                    'train_datasets': split_config['train_datasets'],
                    'valid_dataset': split_config['valid_dataset'],
                    'test_dataset': split_config['test_dataset'],
                    'split_config_key': split_config['split_config_key'],
                }
                update_best_model_registry(
                    best_registry_params,
                    accuracy=result.get('valid_acc', 0.0),
                    mcc=result.get('valid_mcc', -1.0),
                    log_path=result.get('run_dir', ''),
                    batch_metrics=result.get('batch_metrics') or {},
                    source_run_log_path=self.run_root,
                )
                # Write to completed_runs CSV for each successful variant
                self._append_completed_run_csv(
                    params={"variant": variant},
                    best_vals={
                        "valid_mcc": result.get("valid_mcc", ""),
                        "test_mcc": result.get("test_mcc", ""),
                        "valid_acc": result.get("valid_acc", ""),  # fixed: was incorrectly reading test_acc
                        "test_acc": result.get("test_acc", ""),
                        "n_classes": result.get("n_classes", ""),
                        "run_dir": result.get("run_dir", ""),
                    },
                    batch_metrics=result.get("batch_metrics"),
                    run_uuid=self.run_id,
                    final_status="completed",
                    final_error=None,
                    trial_index_override=row_trial_index,
                )
                if int(self.args.verbose) == 1:
                    print(
                        f"[{variant}] best valid MCC={result['valid_mcc']:.4f} | "
                        f"test ACC={result['test_acc']:.4f} | test MCC={result['test_mcc']:.4f}"
                    )
            except Exception as exc:
                print(f"Variant {variant} failed: {exc}")
                traceback.print_exc()
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    stats = gpu_memory_stats(self.device)
                    oom_error = "out of memory" in str(exc).lower()
                    emit_gpu_telemetry(
                        "oom" if oom_error else "failure",
                        actual_peak_gpu_mb=(f"{stats.get('actual_peak_gpu_mb', 0.0):.2f}" if stats else None),
                        oom_gpu_free_at_failure_mb=(f"{stats.get('free_mb', 0.0):.2f}" if stats else None),
                        oom_gpu_used_at_failure_mb=(f"{stats.get('used_mb', 0.0):.2f}" if stats else None),
                        variant=variant,
                    )
                all_results.append(
                    {
                        "exp_id": self.args.exp_id,
                        "run_tag": self.args.run_tag,
                        "finished": False,
                        "variant": variant,
                        "model_name": self.args.model_name,
                        "dloss": self.args.dloss,
                        "classif_loss": self.args.classif_loss,
                        "fgsm": int(self.args.fgsm),
                        "n_calibration": int(self.args.n_calibration),
                        "best_epoch": -1,
                        "valid_mcc": np.nan,
                        "test_mcc": np.nan,
                        "test_acc": np.nan,
                        "n_classes": int(self.n_classes),
                        "run_dir": os.path.join(self.run_root, variant),
                        "error_message": str(exc),
                    }
                )
                # Also log failed run to completed_runs CSV
                self._append_completed_run_csv(
                    params={"variant": variant},
                    best_vals={
                        "valid_mcc": np.nan,
                        "test_mcc": np.nan,
                        "valid_acc": np.nan,
                        "test_acc": np.nan,
                        "n_classes": int(self.n_classes),
                        "run_dir": os.path.join(self.run_root, variant),
                    },
                    batch_metrics=None,
                    run_uuid=self.run_id,
                    final_status="failed",
                    final_error=str(exc),
                    trial_index_override=row_trial_index,
                )

        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(self.run_root, "comparison_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        with open(os.path.join(self.run_root, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=2)

        has_failure = bool(summary_df.get("finished", pd.Series([], dtype=bool)).eq(False).any()) if not summary_df.empty else False
        final_status = "completed_with_failures" if has_failure else "completed"
        self._write_run_metadata(finished=True, status=final_status, error_message=None)

        if int(self.args.verbose) == 1:
            print("\n=== Comparison Summary ===")
            if not summary_df.empty:
                print(summary_df[["variant", "valid_mcc", "test_acc", "test_mcc"]].to_string(index=False))
            print(f"Saved results to: {self.run_root}")
        return summary_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./data/otite_ds_64")
    parser.add_argument("--task", type=str, default="notNormal")
    parser.add_argument("--exp_id", type=str, default="cnn_mlp_compare")
    parser.add_argument("--run_tag", type=str, default="prod")
    parser.add_argument("--trial_index", type=int, default=0, help="Current trial index when launched by an external scheduler")
    parser.add_argument("--valid_dataset", type=str, default="Banque_Viscaino_Chili_2020")
    parser.add_argument("--train_datasets", type=str, default="")
    parser.add_argument("--test_dataset", type=str, default="")
    parser.add_argument("--groupkfold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--new_size", type=int, default=64)
    parser.add_argument("--normalize", type=str, default="yes", choices=["yes", "no", "per_image", "imagenet"])
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (-1 uses all CPU cores)")
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials for hyperparameter optimization")
    parser.add_argument("--optuna_pruner", type=str, default="median", choices=["median", "none"],
                        help="Optuna pruner strategy: median (default) or none (disable pruning)")
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--n_aug", type=int, default=1)

    parser.add_argument("--weighted_sampler", type=int, default=1)
    parser.add_argument("--random_recs", type=int, default=0)
    parser.add_argument("--n_calibration", type=int, default=0)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--optimizer_type", type=str, default="adam")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--classif_loss", type=str, default="ce", choices=["ce", "hinge"])
    parser.add_argument("--dloss", type=str, default="no", choices=["no", "DANN", "inverseTriplet"])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--fgsm", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--log_mlflow", type=int, default=0)

    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "resnet50", "vgg16", "densenet121", "efficientnet_b0", "vit"])
    parser.add_argument("--variant", type=str, default="cnn_transfer", choices=["cnn_transfer", "cnn_scratch", "mlp"])
    parser.add_argument("--compare_all", type=int, default=1, help="Train cnn_transfer + cnn_scratch + mlp in one run")
    parser.add_argument("--reset_opt_state", type=int, default=0, help="Reset Optuna study state and start from scratch")
    parser.add_argument("--amp", type=int, default=1, help="Enable CUDA automatic mixed precision (0/1)")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype to use on CUDA")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.compare_all = bool(int(args.compare_all))
    _set_global_seeds(getattr(args, "seed", 42), deterministic=True)
    print(f"Starting non-siamese comparison at {datetime.now().isoformat()}")

    n_trials = int(getattr(args, "n_trials", 1) or 1)
    if n_trials <= 1 or optuna is None:
        if n_trials > 1 and optuna is None:
            print("[Optuna] optuna is not installed, falling back to a single run.")
        trainer = CNNSupervisedCompare(args)
        trainer.run()
    else:
        progress_root = os.path.join('logs', 'progresses', str(getattr(args, 'task', 'unknown')))
        db_dir = os.path.join(progress_root, "tmp", "db")
        _ensure_dir(db_dir)
        db_path = os.path.join(
            db_dir,
            f"{getattr(args, 'run_tag', 'RUN')}_{getattr(args, 'task', 'unknown')}_{getattr(args, 'exp_id', 'exp')}_optuna_{getattr(args, 'model_name', 'model')}_cnn_mlp_compare.db",
        )
        if int(getattr(args, 'reset_opt_state', 0)) == 1 and os.path.exists(db_path):
            os.remove(db_path)
            print(f"[Optuna] reset_opt_state=1, removed {db_path}.")

        storage_name = f"sqlite:///{db_path}"
        study_name = f"cnn_mlp_compare_{getattr(args, 'model_name', 'model')}"
        
        # TPESampler with warm_up_steps: explore more initially, then exploit
        n_trials_target = max(1, n_trials)
        warmup_steps = max(3, n_trials_target // 4)  # 25% of trials for warmup exploration
        sampler = TPESampler(
            n_startup_trials=warmup_steps, seed=int(getattr(args, 'seed', 42))
        ) if TPESampler else None
        # Optional Optuna pruner (default: MedianPruner, can be disabled with --optuna_pruner none)
        pruner_name = str(getattr(args, 'optuna_pruner', 'median')).strip().lower()
        if n_trials_target <= 10 or pruner_name == 'none':
            pruner = None
        else:
            pruner = MedianPruner(
                n_startup_trials=max(5, n_trials_target // 4),
                n_warmup_steps=max(8, int(getattr(args, 'n_epochs', 80) * 0.15)),
            ) if MedianPruner else None
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        already_done = sum(1 for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED))
        trials_to_run = max(0, n_trials - already_done)
        trial_counter = {'counter': already_done}

        def objective(trial):
            trial_counter['counter'] += 1
            trial_idx = trial_counter['counter']

            trial_args = argparse.Namespace(**deepcopy(vars(args)))
            trial_args.trial_index = int(trial_idx)
            trial_args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
            trial_args.wd = trial.suggest_float("wd", 1e-8, 1e-2, log=True)
            trial_args.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
            trial_args.n_aug = trial.suggest_int("n_aug", 1, 5)
            trial_args.optimizer_type = trial.suggest_categorical("optimizer_type", ["adam", "adamw"])
            if str(getattr(trial_args, 'dloss', 'no')).lower() in ["dann", "inversetriplet"]:
                trial_args.gamma = trial.suggest_float("gamma", 1e-2, 1e2, log=True)
            if int(getattr(trial_args, 'fgsm', 0)) == 1:
                trial_args.epsilon = trial.suggest_float("epsilon", 1e-4, 5e-1, log=True)

            print(f"[Optuna] Trial {trial_idx}/{n_trials} start")
            _append_trial_runtime_event(trial_args, trial_idx, n_trials, event='started', status='running')
            try:
                trainer = CNNSupervisedCompare(trial_args)
                trainer.trial = trial  # Pass trial to trainer for optional pruning/reporting
                summary_df = trainer.run()
                if summary_df is None or summary_df.empty or 'valid_mcc' not in summary_df.columns:
                    score = -1e9
                else:
                    score = float(pd.to_numeric(summary_df['valid_mcc'], errors='coerce').max())
                    if not np.isfinite(score):
                        score = -1e9
                trial.set_user_attr("run_root", getattr(trainer, 'run_root', ''))
                _append_trial_runtime_event(trial_args, trial_idx, n_trials, event='completed', status='completed', score=score, run_dir=getattr(trainer, 'run_root', None))
                print(f"[Optuna] Trial {trial_idx}/{n_trials} done | score={score}")
                return score
            except optuna.TrialPruned as pruned_exc:
                # Trial was pruned due to low intermediate performance - this is expected and not a failure
                _append_trial_runtime_event(trial_args, trial_idx, n_trials, event='pruned', status='pruned', score=-1.0)
                print(f"[Optuna] Trial {trial_idx}/{n_trials} pruned: {pruned_exc}")
                raise  # Re-raise so Optuna knows this was a pruned trial
            except Exception as exc:
                _append_trial_runtime_event(trial_args, trial_idx, n_trials, event='failed', status='failed', score=-1e9, error_message=str(exc))
                print(f"[Optuna] Trial {trial_idx}/{n_trials} failed: {exc}")
                traceback.print_exc()
                return -1e9

        if trials_to_run > 0:
            study.optimize(objective, n_trials=trials_to_run)
        try:
            print(f"[Optuna] Best value={study.best_value} best_params={study.best_params}")
        except ValueError:
            pass
