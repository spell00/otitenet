from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/paper/prepare_inference_fraction_cv_scenarios.py"


def test_rotating_inference_cross_test_and_nested_calibration(tmp_path):
    base_dir = tmp_path / "base"
    out_base = tmp_path / "out"
    base_dir.mkdir()

    rows = []
    labels = ["Normal", "NotNormal", "Wax", "Tube"]
    for index in range(100):
        name = f"inference_{index}.jpg"
        rows.append(
            {
                "dataset": "inference",
                "name": name,
                "raw_label": labels[index % 4],
                "label": labels[index % 4],
                "group": "unused",
            }
        )
        (base_dir / name).touch()
    for index in range(20):
        name = f"historical_{index}.jpg"
        rows.append(
            {
                "dataset": "historical",
                "name": name,
                "raw_label": labels[index % 4],
                "label": labels[index % 4],
                "group": "train",
            }
        )
        (base_dir / name).touch()
    pd.DataFrame(rows).to_csv(base_dir / "infos.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-dir", str(base_dir),
            "--out-base", str(out_base),
            "--prefix", "test_cv",
            "--fractions", "0.5,0.25,0",
            "--n-splits", "5",
            "--group-column", "name",
            "--seed", "42",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = pd.read_csv(out_base / "test_cv_seed42_scenarios.csv", dtype={"scenario_label": str})
    assert len(manifest) == 15
    assert manifest.groupby("cv_run")["valid_fold"].first().to_dict() == {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5
    }
    assert manifest.groupby("cv_run")["test_fold"].first().to_dict() == {
        1: 2, 2: 3, 3: 4, 4: 5, 5: 1
    }

    validation_appearances = {}
    test_appearances = {}
    for _, scenario in manifest[manifest["scenario_label"].eq("0")].iterrows():
        infos = pd.read_csv(Path(scenario.dataset_path) / "infos.csv")
        historical = infos[~infos["dataset"].eq("inference")]
        inference = infos[infos["dataset"].eq("inference")]
        assert historical["group"].eq("train").all()
        assert inference["group"].eq("train").sum() == 0
        for name in inference.loc[inference["group"].eq("valid"), "name"]:
            validation_appearances[name] = validation_appearances.get(name, 0) + 1
        for name in inference.loc[inference["group"].eq("test"), "name"]:
            test_appearances[name] = test_appearances.get(name, 0) + 1
    assert set(validation_appearances.values()) == {1}
    assert set(test_appearances.values()) == {1}
    assert len(validation_appearances) == 100
    assert len(test_appearances) == 100

    for cv_run in range(1, 6):
        fold_rows = manifest[manifest["cv_run"].eq(cv_run)].set_index("scenario_label")
        selected_50 = set(
            pd.read_csv(Path(fold_rows.loc["0p5", "dataset_path"]) / "infos.csv")
            .query("dataset == 'inference' and group == 'train'")["name"]
        )
        selected_25 = set(
            pd.read_csv(Path(fold_rows.loc["0p25", "dataset_path"]) / "infos.csv")
            .query("dataset == 'inference' and group == 'train'")["name"]
        )
        assert len(selected_50) == 50
        assert len(selected_25) == 25
        assert selected_25 < selected_50
