from types import SimpleNamespace

import pandas as pd

from otitenet.app.utils import (
    INFERENCE_TRAIN_SAMPLES_COLUMN,
    attach_inference_train_sample_count,
    canonical_dataset_path,
    inference_train_sample_count_from_path,
)
from otitenet.app.args import (
    _dataset_with_split_fallback,
    _filter_models_df_by_dataset,
)
from otitenet.app.pages import leaderboard


def test_counts_only_unique_inference_rows_used_for_training(tmp_path):
    dataset = tmp_path / "scenario"
    dataset.mkdir()
    pd.DataFrame([
        {"dataset": "inference", "name": "a.jpg", "group": "train"},
        {"dataset": "inference", "name": "a.jpg", "group": "train"},
        {"dataset": "inference", "name": "b.jpg", "group": "train"},
        {"dataset": "inference", "name": "c.jpg", "group": "valid"},
        {"dataset": "inference", "name": "d.jpg", "group": "test"},
        {"dataset": "historical", "name": "h.jpg", "group": "train"},
    ]).to_csv(dataset / "infos.csv", index=False)

    assert inference_train_sample_count_from_path(str(dataset)) == 2

    enriched = attach_inference_train_sample_count(pd.DataFrame([{"Dataset": str(dataset)}]))
    assert enriched.iloc[0][INFERENCE_TRAIN_SAMPLES_COLUMN] == 2


def test_missing_manifest_leaves_count_missing(tmp_path):
    value = inference_train_sample_count_from_path(str(tmp_path / "missing"))
    assert pd.isna(value)


def test_fraction_experiment_uses_inference_train_count_as_n_calibration(tmp_path):
    dataset = tmp_path / "inference_fraction_hist_v1_train0p5_seed42"
    dataset.mkdir()
    pd.DataFrame([
        {"dataset": "inference", "name": "a.jpg", "group": "train"},
        {"dataset": "inference", "name": "b.jpg", "group": "train"},
        {"dataset": "inference", "name": "c.jpg", "group": "valid"},
    ]).to_csv(dataset / "infos.csv", index=False)

    enriched = attach_inference_train_sample_count(pd.DataFrame([{
        "Dataset": str(dataset),
        "exp ID": "inference_fraction_hist_v1",
        "N_Calibration": 0,
        "n_calibration": 0,
        "n_cal": 0,
    }]))

    row = enriched.iloc[0]
    assert row[INFERENCE_TRAIN_SAMPLES_COLUMN] == 2
    assert row["N_Calibration"] == 2
    assert row["n_calibration"] == 2
    assert row["n_cal"] == 2


def test_non_fraction_experiment_preserves_real_calibration_value(tmp_path):
    dataset = tmp_path / "ordinary_dataset"
    dataset.mkdir()
    pd.DataFrame([
        {"dataset": "inference", "name": "a.jpg", "group": "train"},
    ]).to_csv(dataset / "infos.csv", index=False)

    enriched = attach_inference_train_sample_count(pd.DataFrame([{
        "Dataset": str(dataset),
        "N_Calibration": 4,
        "n_calibration": 4,
    }]))

    assert enriched.iloc[0]["N_Calibration"] == 4
    assert enriched.iloc[0]["n_calibration"] == 4


def test_canonicalizes_host_encoded_dataset_paths():
    expected = "otite_ds_64/scenario"
    assert canonical_dataset_path(
        "home/simon/otitenet/data/otite_ds_64/scenario"
    ) == expected
    assert canonical_dataset_path(
        "/home/simon/otitenet/data/otite_ds_64/scenario"
    ) == expected
    assert canonical_dataset_path("data/otite_ds_64/scenario") == expected


def test_counts_samples_from_host_encoded_relative_path(tmp_path, monkeypatch):
    dataset = tmp_path / "data" / "otite_ds_64" / "scenario"
    dataset.mkdir(parents=True)
    pd.DataFrame([
        {"dataset": "inference", "name": "a.jpg", "group": "train"},
        {"dataset": "inference", "name": "b.jpg", "group": "train"},
        {"dataset": "inference", "name": "c.jpg", "group": "valid"},
    ]).to_csv(dataset / "infos.csv", index=False)
    monkeypatch.chdir(tmp_path)

    malformed = "home/simon/otitenet/data/otite_ds_64/scenario"
    assert inference_train_sample_count_from_path(malformed) == 2


def test_dataset_filter_prevents_fraction_cross_contamination():
    rows = pd.DataFrame([
        {
            "Dataset": "home/simon/otitenet/data/otite_ds_64/scenario_train0p25",
            "N_Calibration": 61,
        },
        {
            "Dataset": "home/simon/otitenet/data/otite_ds_64/scenario_train0p02",
            "N_Calibration": 5,
        },
    ])

    filtered = _filter_models_df_by_dataset(
        rows, "otite_ds_64/scenario_train0p25"
    )
    assert filtered["N_Calibration"].tolist() == [61]


def test_explicit_fraction_dataset_is_not_overwritten_by_generic_split_dataset():
    explicit = (
        "/home/simon/otitenet/data/otite_ds_64/"
        "USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v2_train0p25_seed42"
    )
    inferred = "otite_ds_64/USA_Turquie_Chili_GMFUNL_inference"

    assert _dataset_with_split_fallback(explicit, inferred) == (
        "otite_ds_64/"
        "USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v2_train0p25_seed42"
    )


def test_split_dataset_is_used_when_artifact_dataset_is_missing():
    inferred = "otite_ds_64/USA_Turquie_Chili_GMFUNL_inference"
    assert _dataset_with_split_fallback("", inferred) == inferred


def test_leaderboard_dataset_filter_matches_absolute_fresh_run_path(monkeypatch):
    selected = (
        "otite_ds_64/"
        "USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v2_train0p25_seed42"
    )
    models = pd.DataFrame([
        {
            "Dataset": f"/home/simon/otitenet/data/{selected}",
            "N_Calibration": 61,
            "Source": "progress metrics",
        },
        {
            "Dataset": (
                "/home/simon/otitenet/data/otite_ds_64/"
                "USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v2_train0p02_seed42"
            ),
            "N_Calibration": 5,
            "Source": "progress metrics",
        },
    ])
    monkeypatch.setattr(
        leaderboard,
        "st",
        SimpleNamespace(session_state={
            "sidebar_dataset_last_key": selected,
            "sidebar_data_source_toggle": "Database + Manifest (done jobs only)",
        }),
    )

    filtered = leaderboard._filter_models_df_by_sidebar_split(models)
    assert filtered["N_Calibration"].tolist() == [61]
