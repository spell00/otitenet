from argparse import Namespace

import pandas as pd

from otitenet.app.services.inference_results_service import (
    filter_paths_to_unseen_evaluation,
    unseen_evaluation_splits_for_args,
)


def test_inference_filter_keeps_valid_and_test_but_excludes_train(tmp_path):
    dataset = tmp_path / "scenario"
    dataset.mkdir()
    pd.DataFrame(
        [
            {"name": "train.jpg", "group": "train"},
            {"name": "valid.JPG", "group": "valid"},
            {"name": "test.JPEG", "group": "test"},
            {"name": "other.png", "group": "test"},
        ]
    ).to_csv(dataset / "infos.csv", index=False)
    args = Namespace(path=str(dataset))
    candidates = [
        "/images/train.jpg",
        "/images/valid.JPG",
        "/images/TEST.jpeg",
        "/images/other.png",
        "/images/unlisted.jpg",
    ]

    filtered, split_map, manifest, error = filter_paths_to_unseen_evaluation(candidates, args)

    assert error is None
    assert manifest == str(dataset / "infos.csv")
    assert filtered == ["/images/valid.JPG", "/images/TEST.jpeg", "/images/other.png"]
    assert split_map["valid.jpg"] == "valid"
    assert split_map["test.jpeg"] == "test"


def test_inference_filter_fails_closed_without_split_manifest(tmp_path):
    args = Namespace(path=str(tmp_path / "missing"))

    allowed, manifest, error = unseen_evaluation_splits_for_args(args)

    assert allowed == {}
    assert manifest is None
    assert "no infos.csv" in error
