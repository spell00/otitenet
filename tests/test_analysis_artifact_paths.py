import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from otitenet.app import analysis
from otitenet.app import model_loading
from otitenet.app.pages.learned_embedding import _save_trained_head_artifacts
from otitenet.app.pages.inference_results import (
    _enable_missing_train_encoding_repair,
    _result_matches_model_row_df,
)


def _source_run(tmp_path: Path, rel: str = "logs/otitis_four_class/source-run") -> Path:
    root = tmp_path / rel
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.pth").write_bytes(b"model")
    (root / "prototypes.pkl").write_bytes(b"proto")
    return root


def _base_args(source: Path, **overrides):
    values = {
        "task": "otitis_four_class",
        "model_name": "resnet18",
        "source_run_log_path": source.as_posix(),
        "log_path": source.as_posix(),
        "best_model_dir": "",
        "path": "data/otite_ds_64/USA",
        "valid_dataset": "USA",
        "new_size": 224,
        "fgsm": 0,
        "n_calibration": 4,
        "classif_loss": "ce",
        "dloss": "no",
        "prototypes_to_use": "no",
        "n_positives": 1,
        "n_negatives": 1,
        "prototype_strategy": "mean",
        "prototype_components": 1,
        "normalize": "yes",
        "dist_fct": "none",
        "n_neighbors": 3,
        "best_classifier_config": "baseline_naive_bayes",
        "siamese_inference": "linearsvc",
        "use_pretrained_encodings": True,
        "use_trained_encoder": True,
        "allow_inference_reencode": False,
        "random_recs": 0,
        "bs": 2,
        "groupkfold": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_candidate_train_encoding_paths_use_source_run_only_by_default(tmp_path, monkeypatch):
    source = _source_run(tmp_path)
    mirror = tmp_path / "logs/best_models/otitis_four_class/resnet18/mirror"
    mirror.mkdir(parents=True)
    (mirror / "model.pth").write_bytes(b"model")
    (mirror / "prototypes.pkl").write_bytes(b"proto")
    args = _base_args(
        source,
        train_encodings_path=(mirror / "train_encodings.npz").as_posix(),
        best_model_dir=mirror.as_posix(),
    )
    monkeypatch.delenv("OTITENET_ALLOW_LEGACY_BEST_MODELS", raising=False)

    paths = analysis.candidate_train_encoding_paths(args)

    assert paths == [os.path.normpath((source / "train_encodings.npz").as_posix())]
    assert all("logs/best_models" not in path.replace("\\", "/") for path in paths)


def test_candidate_train_encoding_paths_include_legacy_mirror_only_when_enabled(tmp_path, monkeypatch):
    source = _source_run(tmp_path)
    mirror = tmp_path / "logs/best_models/otitis_four_class/resnet18/mirror"
    mirror.mkdir(parents=True)
    (mirror / "model.pth").write_bytes(b"model")
    (mirror / "prototypes.pkl").write_bytes(b"proto")
    args = _base_args(source, best_model_dir=mirror.as_posix())
    monkeypatch.setenv("OTITENET_ALLOW_LEGACY_BEST_MODELS", "1")

    paths = analysis.candidate_train_encoding_paths(args)

    assert os.path.normpath((source / "train_encodings.npz").as_posix()) in paths
    assert os.path.normpath((mirror / "train_encodings.npz").as_posix()) in paths


def test_query_output_log_path_uses_source_run_queries(tmp_path):
    source = _source_run(tmp_path)
    args = _base_args(source)

    assert analysis._query_output_log_path(args) == os.path.normpath((source / "queries").as_posix())


def test_model_loading_candidates_prefer_source_run_and_ignore_nan_and_legacy_by_default(tmp_path, monkeypatch):
    source = _source_run(tmp_path)
    mirror = tmp_path / "logs/best_models/otitis_four_class/resnet18/mirror"
    mirror.mkdir(parents=True)
    (mirror / "model.pth").write_bytes(b"model")
    (mirror / "prototypes.pkl").write_bytes(b"proto")
    args = _base_args(source, log_path="nan", best_model_dir=mirror.as_posix())
    monkeypatch.delenv("OTITENET_ALLOW_LEGACY_BEST_MODELS", raising=False)

    candidates = model_loading._model_artifact_candidates(
        args,
        log_path=args.log_path,
        source_run_log_path=args.source_run_log_path,
        best_model_dir=args.best_model_dir,
    )

    assert candidates == [
        (
            os.path.join(source.as_posix(), "model.pth"),
            os.path.join(source.as_posix(), "prototypes.pkl"),
            3,
        )
    ]
    assert all("nan/model.pth" not in candidate[0] for candidate in candidates)
    assert all("logs/best_models" not in candidate[0].replace("\\", "/") for candidate in candidates)


def test_model_loading_candidates_include_best_models_only_when_legacy_enabled(tmp_path, monkeypatch):
    source = _source_run(tmp_path)
    mirror = tmp_path / "logs/best_models/otitis_four_class/resnet18/mirror"
    mirror.mkdir(parents=True)
    (mirror / "model.pth").write_bytes(b"model")
    (mirror / "prototypes.pkl").write_bytes(b"proto")
    args = _base_args(source, best_model_dir=mirror.as_posix())
    monkeypatch.setenv("OTITENET_ALLOW_LEGACY_BEST_MODELS", "1")

    candidates = model_loading._model_artifact_candidates(
        args,
        source_run_log_path=args.source_run_log_path,
        best_model_dir=args.best_model_dir,
    )

    assert (
        os.path.join(mirror.as_posix(), "model.pth"),
        os.path.join(mirror.as_posix(), "prototypes.pkl"),
        3,
    ) in candidates


def test_missing_train_encodings_raise_without_reencoding(tmp_path, monkeypatch):
    source = _source_run(tmp_path)
    args = _base_args(source, allow_inference_reencode=False)
    monkeypatch.delenv("OTITENET_ALLOW_LEGACY_BEST_MODELS", raising=False)
    monkeypatch.setattr(analysis.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis.st, "warning", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="will not re-encode or train"):
        analysis.get_or_build_embedding_classifier(
            args,
            data=None,
            unique_labels=np.asarray(["normal", "wax"]),
            unique_batches=np.asarray(["batch"]),
            prototypes={},
        )


def test_explicit_reencode_saves_train_encodings_to_source_run(tmp_path, monkeypatch):
    source = _source_run(tmp_path)
    args = _base_args(source, allow_inference_reencode=True)
    expected_embeddings = np.asarray([[0.0, 0.1], [1.0, 1.1], [0.2, 0.0], [1.2, 1.0]], dtype=float)
    expected_cats = np.asarray([0, 1, 0, 1], dtype=int)

    class FakeTrainAE:
        def __init__(self, *args, **kwargs):
            self.params = {}

        def set_arcloss(self):
            return None

        def loop(self, split, *args):
            assert split == "train"
            lists = args[3]
            lists["train"]["encoded_values"].append(expected_embeddings)
            lists["train"]["cats"].append(expected_cats)
            return None, lists, None

    monkeypatch.setattr(analysis, "_trainae_class", lambda: FakeTrainAE)
    monkeypatch.setattr(
        "otitenet.app.model_loading.load_model_and_prototypes",
        lambda _args: (object(), None, None, None, None, None, None, None, None),
    )
    monkeypatch.setattr(
        analysis,
        "get_empty_traces",
        lambda: (
            {"train": {"encoded_values": [], "cats": []}},
            {},
        ),
    )
    monkeypatch.setattr(analysis, "get_images_loaders", lambda **kwargs: {"train": object()})
    monkeypatch.setattr(analysis.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis.st, "spinner", lambda *args, **kwargs: analysis.nullcontext())

    classifier, labels = analysis.get_or_build_embedding_classifier(
        args,
        data=None,
        unique_labels=np.asarray(["normal", "wax"]),
        unique_batches=np.asarray(["batch"]),
        prototypes={},
    )

    saved = np.load(source / "train_encodings.npz")
    assert np.array_equal(saved["embeddings"], expected_embeddings)
    assert np.array_equal(saved["cats"], expected_cats)
    head_files = list((source / "classifier_heads").glob("*.pkl"))
    assert len(head_files) == 1
    assert head_files[0].name == "resnet18__baseline_naive_bayes.pkl"
    assert classifier is not None
    assert labels.tolist() == ["normal", "wax"]


def test_saved_embedding_classifier_head_can_be_loaded_from_source_run(tmp_path):
    source = _source_run(tmp_path)
    args = _base_args(source, model_id=12, artifact_id="source-run")

    classifier = GaussianNB().fit(np.asarray([[0.0, 0.0], [1.0, 1.0]]), np.asarray([0, 1]))
    path = analysis._save_embedding_classifier_artifact(
        args,
        "baseline_naive_bayes",
        classifier,
        np.asarray(["normal", "wax"]),
        "naive_bayes",
    )

    loaded = analysis._load_embedding_classifier_artifact(args, "baseline_naive_bayes")

    assert path.endswith(".pkl")
    assert Path(path).is_file()
    assert loaded is not None
    loaded_classifier, loaded_labels = loaded
    assert loaded_classifier.predict(np.zeros((2, 2))).tolist() == [0, 0]
    assert loaded_labels.tolist() == ["normal", "wax"]


def test_saving_embedding_classifier_head_replaces_legacy_hashed_duplicate(tmp_path):
    source = _source_run(tmp_path)
    args = _base_args(source, model_id=12, artifact_id="source-run")
    legacy = source / "classifier_heads" / "resnet18__baseline_naive_bayes__abcdef123456.pkl"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"old")
    classifier = GaussianNB().fit(np.asarray([[0.0, 0.0], [1.0, 1.0]]), np.asarray([0, 1]))

    path = analysis._save_embedding_classifier_artifact(
        args,
        "baseline_naive_bayes",
        classifier,
        np.asarray(["normal", "wax"]),
        "naive_bayes",
    )

    assert Path(path).name == "resnet18__baseline_naive_bayes.pkl"
    assert Path(path).is_file()
    assert not legacy.exists()


def test_inference_page_enables_one_time_missing_encoding_repair():
    args = argparse.Namespace(allow_inference_reencode=False)

    returned = _enable_missing_train_encoding_repair(args)

    assert returned is args
    assert args.allow_inference_reencode is True


def test_top_n_result_matching_requires_same_head_config():
    results = pd.DataFrame(
        [
            {"Filename": "a.jpg", "Model ID": 7, "Head Config": "baseline_naive_bayes", "Log Path": "logs/task/run/queries"},
            {"Filename": "b.jpg", "Model ID": 7, "Head Config": "20", "Log Path": "logs/task/run/queries"},
        ]
    )
    row = {"Model ID": 7, "Best Head Config": "20", "Log Path": "logs/task/run"}

    mask = _result_matches_model_row_df(results, row)

    assert mask.tolist() == [False, True]


def test_tab2_head_training_saves_fitted_classifier_artifacts(tmp_path):
    source = _source_run(tmp_path)
    args = _base_args(source, model_id=15, artifact_id="source-run")
    X = np.asarray([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]])
    y = np.asarray([0, 0, 1, 1])
    knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    knn_weaker = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    nb = GaussianNB().fit(X, y)
    result = {
        "knn": {
            "mcc_per_k": [
                {"k": 1, "mcc": 1.0, "classifier": knn},
                {"k": 3, "mcc": 0.1, "classifier": knn_weaker},
            ]
        },
        "baselines": {"naive_bayes": {"mcc": 1.0, "classifier": nb}},
    }

    paths = _save_trained_head_artifacts(args, result, ["normal", "wax"])

    assert set(paths) == {"1", "baseline_naive_bayes"}
    assert Path(paths["1"]).is_file()
    assert Path(paths["baseline_naive_bayes"]).is_file()
    assert "classifier" not in result["knn"]["mcc_per_k"][0]
    assert "classifier" in result["knn"]["mcc_per_k"][1]
    assert "classifier" not in result["baselines"]["naive_bayes"]
    loaded = analysis._load_embedding_classifier_artifact(args, "baseline_naive_bayes")
    assert loaded is not None
    loaded_classifier, loaded_labels = loaded
    assert loaded_classifier.predict(np.asarray([[1.0, 1.0]])).tolist() == [1]
    assert loaded_labels.tolist() == ["normal", "wax"]
