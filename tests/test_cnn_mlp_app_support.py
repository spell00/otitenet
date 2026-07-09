import argparse
from pathlib import Path

import numpy as np
import torch

from otitenet.app import analysis
from otitenet.app.args import _split_combo_key_from_row
from otitenet.app import model_loading
from otitenet.train.train_cnn_mlp_compare import build_supervised_model


class _FakeGetData:
    def __init__(self, path, valid_dataset, args, manifest_dir=None):
        self.path = path
        self.valid_dataset = valid_dataset
        self.args = args
        self.manifest_dir = manifest_dir

    def get_variables(self):
        data = {
            "cats": {
                "train": np.asarray([0, 1]),
                "valid": np.asarray([0, 1]),
                "test": np.asarray([0, 1]),
            }
        }
        return data, np.asarray(["normal", "wax"]), np.asarray(["batch_a"])


def _cnn_args(model_dir: Path, **overrides):
    values = {
        "kind": "cnn_mlp",
        "variant": "mlp",
        "task": "otitis_four_class",
        "model_name": "resnet18",
        "new_size": 8,
        "classif_loss": "ce",
        "dloss": "no",
        "normalize": "yes",
        "device": "cpu",
        "path": "data/otite_ds_64/USA",
        "valid_dataset": "valid_a",
        "train_datasets": "train_a",
        "test_dataset": "inference",
        "log_path": model_dir.as_posix(),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_cnn_mlp_loader_loads_nested_best_model_without_prototypes(tmp_path, monkeypatch):
    run_root = tmp_path / "logs" / "otitis_four_class" / "cnn_mlp_compare" / "PROD" / "run-1"
    model_dir = run_root / "mlp"
    model_dir.mkdir(parents=True)
    model, _ = build_supervised_model("mlp", "resnet18", 2, new_size=8, transfer_learning=False)
    torch.save(model.state_dict(), model_dir / "best_model.pth")
    args = _cnn_args(run_root)

    monkeypatch.setattr(model_loading, "GetData", _FakeGetData)
    model_loading.clear_cached_model()

    loaded, image_size, device, data, labels, batches, _getter, resolved_dir, variant = (
        model_loading.load_cnn_mlp_model_and_data(args)
    )

    assert image_size == 8
    assert device == "cpu"
    assert labels.tolist() == ["normal", "wax"]
    assert batches.tolist() == ["batch_a"]
    assert resolved_dir == model_dir.as_posix()
    assert variant == "mlp"
    logits, features = loaded(torch.zeros(1, 3, 8, 8))
    assert logits.shape == (1, 2)
    assert features.shape[0] == 1


def test_cnn_mlp_direct_prediction_uses_softmax_labels():
    class ToyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0, 3.0]], dtype=torch.float32), torch.zeros(1, 2)

    label, confidence, probas = analysis._predict_cnn_mlp_direct(
        ToyModel(),
        torch.zeros(1, 3, 8, 8),
        "cpu",
        np.asarray(["normal", "wax"]),
    )

    assert label == "wax"
    assert confidence > 0.9
    assert set(probas) == {"normal", "wax"}
    assert abs(sum(probas.values()) - 1.0) < 1e-6


def test_cnn_mlp_args_detects_variant_or_artifact_path(tmp_path):
    model_dir = tmp_path / "mlp"
    model_dir.mkdir()
    (model_dir / "best_model.pth").write_bytes(b"model")

    assert analysis._is_cnn_mlp_args(argparse.Namespace(variant="mlp", log_path=""))
    assert analysis._is_cnn_mlp_args(argparse.Namespace(variant="", log_path=model_dir.as_posix()))
    assert not analysis._is_cnn_mlp_args(argparse.Namespace(variant="", log_path="logs/otitis_four_class/run"))


def test_generic_log_path_loader_supports_cnn_mlp_best_model(tmp_path):
    model_dir = tmp_path / "run" / "mlp"
    model_dir.mkdir(parents=True)
    model, _ = build_supervised_model("mlp", "resnet18", 2, new_size=8, transfer_learning=False)
    torch.save(model.state_dict(), model_dir / "best_model.pth")
    model_loading.clear_cached_model()

    loaded = model_loading.load_model_for_log_path(model_dir.as_posix(), "resnet18", device="cpu")

    assert loaded is not None
    logits, features = loaded(torch.zeros(1, 3, 8, 8))
    assert logits.shape == (1, 2)
    assert features.shape[0] == 1


def test_sidebar_dedupe_key_preserves_cnn_and_mlp_variants():
    rows = [
        {
            "Model Name": "resnet18",
            "NSize": "224",
            "FGSM": "0",
            "Prototypes": "",
            "NPos": "",
            "NNeg": "",
            "DLoss": "no",
            "Dist_Fct": "",
            "Classif_Loss": "ce",
            "N_Calibration": "4",
            "Normalize": "yes",
            "N_Neighbors": "",
            "train_datasets": "train",
            "valid_dataset": "valid",
            "test_dataset": "inference",
            "Kind": "cnn_mlp",
            "Variant": "cnn_transfer",
            "Artifact ID": "cnn-run",
            "Log Path": "logs/task/cnn-run/cnn_transfer",
        },
        {
            "Model Name": "resnet18",
            "NSize": "224",
            "FGSM": "0",
            "Prototypes": "",
            "NPos": "",
            "NNeg": "",
            "DLoss": "no",
            "Dist_Fct": "",
            "Classif_Loss": "ce",
            "N_Calibration": "4",
            "Normalize": "yes",
            "N_Neighbors": "",
            "train_datasets": "train",
            "valid_dataset": "valid",
            "test_dataset": "inference",
            "Kind": "cnn_mlp",
            "Variant": "mlp",
            "Artifact ID": "mlp-run",
            "Log Path": "logs/task/mlp-run/mlp",
        },
    ]
    group_cols = [
        "Model Name", "NSize", "FGSM", "Prototypes", "NPos", "NNeg",
        "DLoss", "Dist_Fct", "Classif_Loss", "N_Calibration", "Normalize", "N_Neighbors",
        "train_datasets", "valid_dataset", "test_dataset", "Kind", "Variant", "Artifact ID", "Log Path",
    ]

    keys = ["|".join(str(row.get(col, "")) for col in group_cols) for row in rows]

    assert len(set(keys)) == 2
    assert _split_combo_key_from_row(rows[0]) == _split_combo_key_from_row(rows[1])
