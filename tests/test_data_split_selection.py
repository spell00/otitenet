from argparse import Namespace

import numpy as np
import torch
from torch import nn
from PIL import Image

from otitenet.data.data_getters import GetData
from otitenet.app.pages import admin_analytics
from otitenet.app.pages.admin_analytics import _collect_raw_split_arrays, _collect_raw_split_datasets
from otitenet.app.utils_dataset_names import get_short_dataset_name


def _write_split_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        ("train_a", "train_a_normal.png", "Normal", "valid"),
        ("train_a", "train_a_effusion.png", "Effusion", "valid"),
        ("train_b", "train_b_normal.png", "Normal", "test"),
        ("train_b", "train_b_effusion.png", "Effusion", "test"),
        ("valid_a", "valid_a_normal.png", "Normal", "train"),
        ("valid_a", "valid_a_effusion.png", "Effusion", "train"),
        ("test_a", "test_a_normal.png", "Normal", "train"),
        ("test_a", "test_a_effusion.png", "Effusion", "train"),
    ]
    with (root / "infos.csv").open("w", encoding="utf-8") as handle:
        handle.write("dataset,name,label,group\n")
        for dataset, name, label, group in rows:
            Image.new("RGB", (8, 8), color=(120, 80, 40)).save(root / name)
            handle.write(f"{dataset},{name},{label},{group}\n")


def _args(**overrides):
    values = {
        "normalize": "no",
        "new_size": 8,
        "task": "notNormal",
        "groupkfold": 1,
        "seed": 42,
        "train_datasets": "train_a,train_b",
        "valid_dataset": "valid_a",
        "test_dataset": "test_a",
    }
    values.update(overrides)
    return Namespace(**values)


def test_requested_dataset_split_overrides_infos_groups(tmp_path):
    _write_split_fixture(tmp_path)

    data_getter = GetData(tmp_path.as_posix(), "valid_a", _args())

    assert set(data_getter.data["batches"]["train"]) == {"train_a", "train_b"}
    assert set(data_getter.data["batches"]["valid"]) == {"valid_a"}
    assert set(data_getter.data["batches"]["test"]) == {"test_a"}
    assert data_getter.split_debug["mode"] == "deterministic"


def test_from_infos_csv_keeps_precomputed_infos_groups(tmp_path):
    _write_split_fixture(tmp_path)

    data_getter = GetData(
        tmp_path.as_posix(),
        "from_infos_csv",
        _args(
            train_datasets="from_infos_csv",
            valid_dataset="from_infos_csv",
            test_dataset="from_infos_csv",
        ),
    )

    assert set(data_getter.data["batches"]["train"]) == {"valid_a", "test_a"}
    assert set(data_getter.data["batches"]["valid"]) == {"train_a"}
    assert set(data_getter.data["batches"]["test"]) == {"train_b"}
    assert data_getter.split_debug["mode"] == "infos_groups"


def test_eda_raw_split_arrays_preserve_test_sample_count():
    data = {
        "inputs": {
            "train": [[[1]], [[2]]],
            "valid": [[[3]]],
            "test": [[[4]], [[5]], [[6]]],
        },
        "cats": {
            "train": [0, 1],
            "valid": [0],
            "test": [1, 1, 0],
        },
        "batches": {
            "train": ["train_a", "train_b"],
            "valid": ["valid_a"],
            "test": ["inference", "inference", "inference"],
        },
    }

    _raw, _cats, groups = _collect_raw_split_arrays(data)
    datasets = _collect_raw_split_datasets(data)

    assert groups.tolist().count("test") == 3
    assert groups.tolist() == ["train", "train", "valid", "test", "test", "test"]
    assert datasets.tolist()[-3:] == ["inference", "inference", "inference"]


def test_eda_cnn_mlp_features_use_penultimate_model_output(monkeypatch):
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            flat = x.reshape(x.shape[0], -1)
            feats = torch.stack([flat.mean(dim=1), flat.sum(dim=1)], dim=1)
            logits = torch.zeros((x.shape[0], 2), device=x.device)
            return logits, feats

    class FakeLoader:
        def __iter__(self):
            x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
            names = ["a", "b"]
            labels = torch.tensor([0, 1])
            old_labels = ["Normal", "NotNormal"]
            domain = torch.tensor([0, 0])
            yield (x, names, labels, None, old_labels, domain)

    monkeypatch.setattr(
        "otitenet.data.data_getters.get_images_loaders",
        lambda **_kwargs: {"train": FakeLoader(), "valid": None, "test": None},
    )

    features, cats, groups, datasets = admin_analytics._collect_cnn_mlp_features(
        FakeModel(),
        data={},
        unique_labels=np.asarray(["Normal", "NotNormal"]),
        unique_batches=np.asarray(["batch"]),
        _args=_args(),
    )

    expected_flat = torch.arange(24, dtype=torch.float32).reshape(2, -1)
    expected = torch.stack([expected_flat.mean(dim=1), expected_flat.sum(dim=1)], dim=1).numpy()
    assert np.array_equal(features, expected)
    assert cats.tolist() == [0, 1]
    assert groups.tolist() == ["train", "train"]
    assert datasets.tolist() == ["batch", "batch"]


def test_eda_projection_uses_categorical_legend_not_colorbar():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = admin_analytics._plot_2d_projection(
        ax,
        np.asarray([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2]]),
        np.asarray(["Normal", "NotNormal", "Wax"]),
        "classes",
        groups=np.asarray(["train", "valid", "test"]),
        legend_title="Class",
    )

    assert result is None
    assert len(fig.axes) == 1
    assert len(ax.get_legend().texts) > 0
    plt.close(fig)


def test_gmfunl_dataset_aliases_display_as_quebec():
    assert get_short_dataset_name("GMFUNL") == "Quebec"
    assert get_short_dataset_name("GMFUNL_jan2023") == "Quebec"
    assert get_short_dataset_name("Quebec") == "Quebec"
    assert admin_analytics._dataset_display_names(["GMFUNL", "GMFUNL_jan2023", "Quebec"]).tolist() == [
        "Quebec",
        "Quebec",
        "Quebec",
    ]
