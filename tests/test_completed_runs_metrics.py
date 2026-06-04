import pandas as pd

from otitenet.train.completed_runs_metrics import (
    COMPLETED_RUNS_METRICS_HEADER,
    append_completed_run_metrics,
)


def test_append_completed_run_metrics_keeps_new_fields_aligned(tmp_path):
    path = tmp_path / "completed_runs_metrics.csv"

    append_completed_run_metrics(
        path,
        {
            "uuid": "run-1",
            "task": "notNormal",
            "run_tag": "prod",
            "model_name": "resnet18",
            "classif_loss": "arcface",
            "train_datasets": "A,B",
            "valid_dataset": "C",
            "test_dataset": "D",
            "split_config_key": "A,B|C|D",
            "train_mcc": 0.4,
            "valid_mcc": 0.5,
            "test_mcc": 0.6,
            "valid_auc": 0.7,
            "test_auc": 0.8,
        },
    )

    df = pd.read_csv(path)

    assert list(df.columns) == COMPLETED_RUNS_METRICS_HEADER
    assert df.loc[0, "task"] == "notNormal"
    assert df.loc[0, "train_datasets"] == "A,B"
    assert df.loc[0, "valid_auc"] == 0.7


def test_append_completed_run_metrics_rewrites_legacy_header(tmp_path):
    path = tmp_path / "completed_runs_metrics.csv"
    pd.DataFrame(
        [
            {
                "uuid": "legacy-run",
                "model_name": "resnet18",
                "classif_loss": "triplet",
                "valid_mcc": 0.1,
            }
        ]
    ).to_csv(path, index=False)

    append_completed_run_metrics(
        path,
        {
            "uuid": "new-run",
            "task": "notNormal",
            "run_tag": "prod",
            "model_name": "resnet50",
            "classif_loss": "arcface",
            "valid_mcc": 0.2,
        },
    )

    df = pd.read_csv(path)

    assert list(df.columns) == COMPLETED_RUNS_METRICS_HEADER
    assert len(df) == 2
    assert df.loc[0, "uuid"] == "legacy-run"
    assert df.loc[1, "task"] == "notNormal"
