from pathlib import Path

import pandas as pd

from scripts.paper.prepare_test_calibration_manifest import build_manifest


def test_build_manifest_uses_total_n_with_proportional_label_counts(tmp_path: Path):
    rows = [
        {"name": f"normal_{i}.jpg", "label": "Normal", "batch": "inference"}
        for i in range(27)
    ]
    rows.extend(
        {"name": f"notnormal_{i}.jpg", "label": "NotNormal", "batch": "inference"}
        for i in range(6)
    )
    split_csv = tmp_path / "test.csv"
    output_csv = tmp_path / "calibration.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)

    manifest = build_manifest(split_csv, n=16, seed=42, output=output_csv)

    assert len(manifest) == 16
    assert manifest["label"].value_counts().to_dict() == {"Normal": 13, "NotNormal": 3}
    assert len(pd.read_csv(output_csv)) == 16


def test_build_manifest_samples_n_per_source_split(tmp_path: Path):
    test_rows = [
        {"name": f"test_normal_{i}.jpg", "label": "Normal", "batch": "test"}
        for i in range(27)
    ]
    test_rows.extend(
        {"name": f"test_notnormal_{i}.jpg", "label": "NotNormal", "batch": "test"}
        for i in range(6)
    )
    valid_rows = [
        {"name": f"valid_normal_{i}.jpg", "label": "Normal", "batch": "valid"}
        for i in range(10)
    ]
    valid_rows.extend(
        {"name": f"valid_notnormal_{i}.jpg", "label": "NotNormal", "batch": "valid"}
        for i in range(6)
    )
    test_csv = tmp_path / "test.csv"
    valid_csv = tmp_path / "valid.csv"
    output_csv = tmp_path / "calibration.csv"
    pd.DataFrame(test_rows).to_csv(test_csv, index=False)
    pd.DataFrame(valid_rows).to_csv(valid_csv, index=False)

    manifest = build_manifest(test_csv, n=8, seed=42, output=output_csv, extra_splits=[("valid", valid_csv)])

    assert len(manifest) == 16
    assert manifest.groupby("source_group")["label"].value_counts().to_dict() == {
        ("test", "Normal"): 7,
        ("test", "NotNormal"): 1,
        ("valid", "Normal"): 5,
        ("valid", "NotNormal"): 3,
    }
    assert len(pd.read_csv(output_csv)) == 16
