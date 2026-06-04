from pathlib import Path

import pandas as pd
import pytest

from otitenet.app.artifact_registry import (
    available_dataset_paths_from_registry,
    preferred_model_artifact_dir,
    scan_best_models_registry,
    scan_dataset_registry,
)


def _touch_model(root: Path, rel_dir: str) -> Path:
    model_dir = root / rel_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pth").write_bytes(b"model")
    (model_dir / "prototypes.pkl").write_bytes(b"proto")
    return model_dir


def test_best_models_registry_has_one_row_per_model_and_source_run(tmp_path):
    root = tmp_path / "logs" / "best_models"
    model_dir = _touch_model(
        root,
        "otitis_four_class/resnet18/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference/"
        "nsize224/fgsm0/ncal0/arcface/inverseTriplet/prototypes_class/npos1/nneg1/"
        "protoagg_mean_1/normyes/dist_cosine/knn2",
    )
    models_csv = root / "otitis_four_class" / "models.csv"
    models_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "valid_mcc": "0.88",
                "model": "resnet18",
                "path": "otite_ds_64/USA_Turquie_Chili_GMFUNL_inference",
                "n_neighbors": "2",
                "nsize": "nsize224",
                "fgsm": "fsgm0",
                "n_calibration": "ncal0",
                "loss": "arcface",
                "dloss": "inverseTriplet",
                "dist_fct": "cosine",
                "prototype": "prototypes_class",
                "n_positives": "npos1",
                "n_negatives": "nneg1",
                "normalize": "yes",
                "task": "otitis_four_class",
                "complete_log_path": "logs/otitis_four_class/source-run",
            }
        ]
    ).to_csv(models_csv, index=False)

    df = scan_best_models_registry(root)

    assert len(df) == 1
    row = df.iloc[0].to_dict()
    assert row["model_dir"].endswith(model_dir.as_posix().lstrip("/"))
    assert row["model_path"].endswith("model.pth")
    assert row["run_log_path"] == "logs/otitis_four_class/source-run"
    assert row["task"] == "otitis_four_class"
    assert row["dataset_path"] == "otite_ds_64/USA_Turquie_Chili_GMFUNL_inference"
    assert row["nsize"] == "224"
    assert row["fgsm"] == "0"
    assert row["classif_loss"] == "arcface"
    assert row["dloss"] == "inverseTriplet"
    assert row["prototypes"] == "class"
    assert row["n_neighbors"] == "2"


def test_dataset_registry_marks_ignored_raw_folders_and_processed_infos(tmp_path):
    data_root = tmp_path / "data"
    for name in [
        "Banque_Calaman_USA_2020",
        "Banque_Comert_Turquie_2020",
        "Banque_Calaman_USA_2020_trie_CM",
        "inference",
    ]:
        (data_root / "datasets" / name).mkdir(parents=True, exist_ok=True)
    processed = data_root / "otite_ds_64" / "USA_inference"
    processed.mkdir(parents=True)
    (processed / "infos.csv").write_text("name,dataset,label\none.jpg,inference,0\n", encoding="utf-8")

    df = scan_dataset_registry(data_root)
    ignored = set(df.loc[df["ignored"] == "yes", "sources"])
    processed_row = df.loc[df["kind"] == "processed"].iloc[0]

    assert {"Banque_Calaman_USA_2020", "Banque_Comert_Turquie_2020"} <= ignored
    assert processed_row["pixel_size"] == "64"
    assert processed_row["has_inference"] == "yes"
    assert processed_row["n_samples"] == 1


def test_current_best_models_registry_covers_all_model_files():
    root = Path("logs/best_models")
    if not root.is_dir():
        pytest.skip("local logs/best_models is not available")

    expected = len(list(root.rglob("model.pth")))
    df = scan_best_models_registry(root)

    assert len(df) == expected
    assert df["model_path"].is_unique


def test_available_dataset_paths_uses_csv_registry(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    pd.DataFrame(
        [
            {"kind": "processed", "path": "data/otite_ds_64/USA", "ignored": "no"},
            {"kind": "processed", "path": "data/otite_ds_64/old", "ignored": "yes"},
            {"kind": "raw", "path": "data/datasets/inference", "ignored": "no"},
        ]
    ).to_csv(configs / "datasets.csv", index=False)

    monkeypatch.chdir(tmp_path)

    assert available_dataset_paths_from_registry("data") == ["otite_ds_64/USA"]


def test_preferred_model_artifact_dir_uses_source_run_before_best_model_mirror(tmp_path):
    source = _touch_model(tmp_path, "logs/otitis_four_class/source-run")
    mirror = _touch_model(
        tmp_path,
        "logs/best_models/otitis_four_class/resnet18/otite_ds_64/USA/"
        "nsize224/fgsm0/ncal0/arcface/inverseTriplet/prototypes_class/npos1/nneg1/"
        "protoagg_mean_1/normyes/dist_cosine/knn2",
    )

    row = {
        "Source Run Path": source.as_posix(),
        "Best Model Dir": mirror.as_posix(),
        "Log Path": mirror.as_posix(),
    }

    assert preferred_model_artifact_dir(row) == source.as_posix()
