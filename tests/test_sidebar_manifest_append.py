import pandas as pd

from otitenet.app.args import _append_manifest_model_rows


def test_append_manifest_model_rows_ignores_empty_manifest_with_db_columns():
    db_df = pd.DataFrame(
        [
            {
                "Model ID": 1,
                "Model Name": "resnet18",
                "Log Path": "logs/notNormal/run-1",
                "Registry ID": 1,
                "exp ID": "",
            }
        ]
    )
    manifest_df = pd.DataFrame()

    out = _append_manifest_model_rows(db_df, manifest_df)

    assert list(out.columns) == list(db_df.columns)
    assert len(out) == 1
    assert out.loc[0, "Model ID"] == 1


def test_append_manifest_model_rows_aligns_missing_and_extra_columns():
    db_df = pd.DataFrame(
        [
            {
                "Model ID": 1,
                "Model Name": "resnet18",
                "Log Path": "logs/notNormal/run-1",
                "Registry ID": 1,
            }
        ]
    )
    manifest_df = pd.DataFrame(
        [
            {
                "Model Name": "resnet50",
                "Log Path": "logs/notNormal/run-2",
                "Artifact Dataset": "otite_ds_64/USA",
            }
        ]
    )

    out = _append_manifest_model_rows(db_df, manifest_df)

    assert len(out) == 2
    assert "Artifact Dataset" in out.columns
    assert pd.isna(out.loc[1, "Model ID"])
    assert out.loc[1, "Artifact Dataset"] == "otite_ds_64/USA"


def test_append_manifest_model_rows_preserves_calibration_aliases():
    db_df = pd.DataFrame(
        [
            {
                "Model ID": 1,
                "Model Name": "resnet18",
                "N_Calibration": "4",
            }
        ]
    )
    manifest_df = pd.DataFrame(
        [
            {
                "Model Name": "resnet50",
                "n_cal": "8",
            }
        ]
    )

    out = _append_manifest_model_rows(db_df, manifest_df)

    assert out.loc[0, "N_Calibration"] == "4"
    assert out.loc[0, "n_cal"] == "4"
    assert out.loc[1, "N_Calibration"] == "8"
    assert out.loc[1, "n_cal"] == "8"
