import argparse
import json
import os
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from sklearn.metrics import accuracy_score
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
from otitenet.utils.utils import set_random_seeds as _set_global_seeds

try:
    import mlflow
except Exception:  # Optional dependency in some environments.
    mlflow = None


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _safe_mcc(y_true, y_pred):
    try:
        return float(MCC(y_true, y_pred))
    except Exception:
        return 0.0


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
    labels = labels.to(device).long()
    domain = domain.to(device).long()
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


@dataclass
class EpochStats:
    loss: float
    acc: float
    mcc: float


class CNNSupervisedCompare:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.run_id = str(uuid.uuid4())
        self.run_root = os.path.join("logs", args.task, "cnn_mlp_compare", args.run_tag, self.run_id)
        _ensure_dir(self.run_root)

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

    def _make_loaders(self):
        prototypes = {"combined": {}, "class": {}, "batch": {}}
        return get_images_loaders(
            data=self.data,
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
        )

    def _train_epoch(self, model, domain_head, loader, classif_criterion, domain_criterion, optimizer):
        model.train()
        if domain_head is not None:
            domain_head.train()
        all_true, all_pred = [], []
        losses = []
        for batch in loader:
            x, y, d, _, _ = _batch_to_xy(batch, self.device)
            if int(self.args.fgsm) == 1:
                x = x.detach().requires_grad_(True)
            optimizer.zero_grad()
            logits, feats = model(x)
            classif_loss = classif_criterion(logits, y)
            domain_loss = torch.tensor(0.0, device=self.device)
            if _use_domain_loss(self.args.dloss) and domain_head is not None:
                rev_feats = ReverseLayerF.apply(feats, 1.0)
                domain_logits = domain_head(rev_feats)
                domain_loss = domain_criterion(domain_logits, d)
            total_loss = classif_loss + float(self.args.gamma) * domain_loss
            total_loss.backward(retain_graph=int(self.args.fgsm) == 1)

            if int(self.args.fgsm) == 1 and x.grad is not None:
                adv_x = torch.clamp(x + float(self.args.epsilon) * x.grad.sign(), -1.0, 1.0).detach()
                adv_logits, adv_feats = model(adv_x)
                adv_classif = classif_criterion(adv_logits, y)
                adv_total = adv_classif
                if _use_domain_loss(self.args.dloss) and domain_head is not None:
                    adv_domain = domain_head(ReverseLayerF.apply(adv_feats, 1.0))
                    adv_total = adv_total + float(self.args.gamma) * domain_criterion(adv_domain, d)
                (0.1 * adv_total).backward()

            optimizer.step()

            losses.append(float(total_loss.item()))
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(logits.argmax(1).detach().cpu().numpy())

        y_true = np.concatenate(all_true) if all_true else np.array([])
        y_pred = np.concatenate(all_pred) if all_pred else np.array([])
        acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
        mcc = _safe_mcc(y_true, y_pred) if y_true.size else 0.0
        return EpochStats(loss=float(np.mean(losses) if losses else 0.0), acc=acc, mcc=mcc)

    def _eval_epoch(self, model, domain_head, loader, classif_criterion, domain_criterion):
        model.eval()
        if domain_head is not None:
            domain_head.eval()
        all_true, all_pred, all_names, all_old_labels = [], [], [], []
        losses = []
        with torch.no_grad():
            for batch in loader:
                x, y, d, names, old_labels = _batch_to_xy(batch, self.device)
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
                all_names.extend(list(names))
                all_old_labels.extend(list(old_labels))

        y_true = np.concatenate(all_true) if all_true else np.array([])
        y_pred = np.concatenate(all_pred) if all_pred else np.array([])
        acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
        mcc = _safe_mcc(y_true, y_pred) if y_true.size else 0.0
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

        loaders = self._make_loaders()
        model, feat_dim = self._build_model(variant)
        model = model.to(self.device)
        domain_head = None
        if _use_domain_loss(self.args.dloss):
            domain_head = nn.Linear(int(feat_dim), self.n_domains).to(self.device)

        classif_criterion = _build_classification_loss(self.args.classif_loss, self.args.label_smoothing)
        domain_criterion = nn.CrossEntropyLoss()

        params = list(model.parameters())
        if domain_head is not None:
            params += list(domain_head.parameters())
        optimizer = _build_optimizer(params, self.args.lr, self.args.wd, self.args.optimizer_type)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

        best_valid_mcc = -1.0
        best_epoch = -1
        best_test_stats = None
        patience_counter = 0

        history_rows = []

        for epoch in range(self.args.n_epochs):
            tr = self._train_epoch(model, domain_head, loaders["train"], classif_criterion, domain_criterion, optimizer)
            va, va_df = self._eval_epoch(model, domain_head, loaders["valid"], classif_criterion, domain_criterion)
            te, te_df = self._eval_epoch(model, domain_head, loaders["test"], classif_criterion, domain_criterion)
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

            if patience_counter >= self.args.early_stop:
                break

            if int(self.args.verbose) == 1:
                print(
                    f"[{variant}] epoch={epoch} train_mcc={tr.mcc:.4f} "
                    f"valid_mcc={va.mcc:.4f} test_mcc={te.mcc:.4f}"
                )

        pd.DataFrame(history_rows).to_csv(os.path.join(model_dir, "history.csv"), index=False)

        result = {
            "exp_id": self.args.exp_id,
            "run_tag": self.args.run_tag,
            "variant": variant,
            "model_name": self.args.model_name,
            "dloss": self.args.dloss,
            "classif_loss": self.args.classif_loss,
            "fgsm": int(self.args.fgsm),
            "n_calibration": int(self.args.n_calibration),
            "best_epoch": int(best_epoch),
            "valid_mcc": float(best_valid_mcc),
            "test_mcc": float(best_test_stats.mcc if best_test_stats else 0.0),
            "test_acc": float(best_test_stats.acc if best_test_stats else 0.0),
            "n_classes": int(self.n_classes),
            "run_dir": model_dir,
        }
        with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

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

        all_results = []
        for variant in variants:
            try:
                if int(self.args.verbose) == 1:
                    print(f"\n=== Training {variant} ===")
                result = self.run_variant(variant)
                all_results.append(result)
                if int(self.args.verbose) == 1:
                    print(
                        f"[{variant}] best valid MCC={result['valid_mcc']:.4f} | "
                        f"test ACC={result['test_acc']:.4f} | test MCC={result['test_mcc']:.4f}"
                    )
            except Exception as exc:
                print(f"Variant {variant} failed: {exc}")
                traceback.print_exc()

        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(self.run_root, "comparison_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        with open(os.path.join(self.run_root, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=2)

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
    parser.add_argument("--valid_dataset", type=str, default="Banque_Viscaino_Chili_2020")
    parser.add_argument("--groupkfold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--new_size", type=int, default=64)
    parser.add_argument("--normalize", type=str, default="no")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=80)
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.compare_all = bool(int(args.compare_all))
    _set_global_seeds(getattr(args, "seed", 42), deterministic=True)
    print(f"Starting non-siamese comparison at {datetime.now().isoformat()}")

    trainer = CNNSupervisedCompare(args)
    trainer.run()