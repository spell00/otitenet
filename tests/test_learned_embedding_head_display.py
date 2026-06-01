import math
import pickle
from argparse import Namespace

from otitenet.app import utils
from otitenet.app.services.embedding_optimization_service import args_from_model_row


def test_best_display_classifier_heads_keeps_families_but_collapses_knn_neighbors():
    rows = [
        {"Model ID": 1, "N Aug": 0, "Family": "knn", "Classifier": "KNN", "Config": "1", "Valid MCC": 0.50},
        {"Model ID": 1, "N Aug": 0, "Family": "knn", "Classifier": "KNN", "Config": "5", "Valid MCC": 0.70},
        {"Model ID": 1, "N Aug": 0, "Family": "gmm", "Classifier": "GMM prototypes", "Config": "protot_gmm_2", "Valid MCC": 0.65},
        {"Model ID": 1, "N Aug": 0, "Family": "logreg", "Classifier": "Logistic Regression", "Config": "baseline_logreg", "Valid MCC": 0.60},
        {"Model ID": 1, "N Aug": 1, "Family": "knn", "Classifier": "KNN", "Config": "3", "Valid MCC": 0.55},
    ]

    out = utils.best_display_classifier_heads(rows)

    assert len(out) == 4
    by_key = {(row["N Aug"], row["Family"]): row for row in out}
    assert by_key[(0, "knn")]["Config"] == "5"
    assert by_key[(0, "gmm")]["Config"] == "protot_gmm_2"
    assert by_key[(0, "logreg")]["Config"] == "baseline_logreg"
    assert by_key[(1, "knn")]["Config"] == "3"


def test_enumerate_classification_heads_exposes_knn_prototype_and_baseline(monkeypatch):
    cache = {
        0: {
            "head_cache_version": 2,
            "knn": {
                "mcc_per_k": [
                    {"k": 1, "valid_mcc": 0.40, "test_mcc": 0.41, "valid_auc": 0.70, "test_auc": 0.71},
                    {"k": 3, "valid_mcc": 0.62, "test_mcc": 0.60, "valid_auc": 0.78, "test_auc": 0.77},
                ],
            },
            "prototypes": {
                "gmm": {
                    "best_mcc": 0.66,
                    "best_n_components": 2,
                    "train_mcc": 0.70,
                    "test_mcc": 0.64,
                    "valid_auc": 0.80,
                    "test_auc": 0.79,
                },
            },
            "baselines": {
                "logreg": {
                    "mcc": 0.58,
                    "train_mcc": 0.59,
                    "test_mcc": 0.57,
                    "valid_auc": 0.75,
                    "test_auc": 0.74,
                },
            },
        }
    }
    monkeypatch.setattr(utils, "load_optimization_cache_dict", lambda _args: cache)

    heads = utils.enumerate_classification_heads(object(), include_all_n_aug=True)
    by_config = {head["config"]: head for head in heads}

    assert {"1", "3", "protot_gmm_2", "baseline_logreg"}.issubset(by_config)
    assert by_config["3"]["n_aug"] == 0
    assert by_config["3"]["test_mcc"] == 0.60
    assert by_config["protot_gmm_2"]["head_cache_version"] == 2
    assert math.isclose(by_config["baseline_logreg"]["valid_auc"], 0.75)


def test_split_aware_head_enumeration_falls_back_to_legacy_cache_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = Namespace(
        task="notNormal",
        model_name="resnet18",
        path="data/otite_ds_64",
        new_size=224,
        fgsm="0",
        n_calibration="0",
        classif_loss="triplet",
        dloss="triplet",
        prototypes_to_use="class",
        n_positives="1",
        n_negatives="1",
        prototype_strategy="mean",
        prototype_components=1,
        normalize="yes",
        dist_fct="cosine",
        n_neighbors=3,
        train_datasets="train_a,train_b",
        valid_dataset="valid_a",
        test_dataset="test_a",
        split_config_in_path=True,
        _split_config_in_path=True,
    )
    cache_path = (
        tmp_path
        / "logs/best_models/notNormal/resnet18/otite_ds_64"
        / "nsize224/fgsm0/ncal0/triplet/triplet/prototypes_class"
        / "npos1/nneg1/protoagg_mean_1/normyes/knn_optimization_cache.pkl"
    )
    cache_path.parent.mkdir(parents=True)
    with cache_path.open("wb") as f:
        pickle.dump(
            {
                0: {
                    "knn": {"mcc_per_k": [{"k": 3, "valid_mcc": 0.61}]},
                    "best_k": 3,
                    "best_mcc": 0.61,
                }
            },
            f,
        )

    heads = utils.enumerate_classification_heads(args)

    assert [head["config"] for head in heads] == ["3"]


def test_args_from_model_row_preserves_protoagg_cache_path_params():
    base_args = Namespace(
        task="otitis_four_class",
        model_name="resnet18",
        new_size=224,
        fgsm="0",
        n_calibration="0",
        classif_loss="arcface",
        dloss="inverseTriplet",
        prototypes_to_use="class",
        n_positives=1,
        n_negatives=1,
        n_neighbors=2,
        normalize="yes",
        dist_fct="cosine",
        prototype_strategy="mean",
        prototype_components=1,
        path="./data/otite_ds_64",
    )
    row = {
        "Log Path": (
            "logs/best_models/otitis_four_class/resnet18/otite_ds_64/"
            "nsize224/fgsm0/ncal0/arcface/inverseTriplet/prototypes_class/"
            "npos1/nneg1/protoagg_gmm_2/normyes/dist_cosine/knn2"
        )
    }

    args = args_from_model_row(base_args, row)

    assert args.task == "otitis_four_class"
    assert args.model_name == "resnet18"
    assert args.path == "data/otite_ds_64"
    assert args.classif_loss == "arcface"
    assert args.dloss == "inverseTriplet"
    assert args.prototypes_to_use == "class"
    assert args.prototype_strategy == "gmm"
    assert int(args.prototype_components) == 2
    assert args.dist_fct == "cosine"
    assert int(args.n_neighbors) == 2


def test_multiclass_auc_uses_classes_present_in_split():
    from otitenet.app.pages.learned_embedding import _evaluate_auc_from_scores

    y_true = [0, 1, 2, 0, 1, 2]
    scores = [
        [0.80, 0.10, 0.08, 0.02],
        [0.10, 0.75, 0.10, 0.05],
        [0.05, 0.15, 0.75, 0.05],
        [0.70, 0.15, 0.10, 0.05],
        [0.15, 0.70, 0.10, 0.05],
        [0.05, 0.10, 0.80, 0.05],
    ]

    auc = _evaluate_auc_from_scores(y_true, scores, classes=[0, 1, 2, 3])

    assert not math.isnan(auc)
    assert auc > 0.95


def test_class_count_mismatch_errors_are_detected_without_traceback():
    from otitenet.app.pages.learned_embedding import (
        _extract_model_path_from_error,
        _is_class_count_mismatch_error,
    )

    exc = RuntimeError(
        "Model/task class-count mismatch while loading checkpoint. "
        "checkpoint_n_cats=2, dataset_n_cats=4, "
        "model_path=logs/best_models/otitis_four_class/resnet50/model.pth."
    )

    assert _is_class_count_mismatch_error(exc)
    assert _extract_model_path_from_error(exc) == "logs/best_models/otitis_four_class/resnet50/model.pth"


def test_training_label_omits_missing_model_number():
    from otitenet.app.pages.learned_embedding import _model_training_label

    args = Namespace(model_name="resnet18")
    label = _model_training_label({"Model ID": 757, "Model Name": "resnet18", "#": None}, args, 0)

    assert "#None" not in label
    assert "model ID 757" in label
    assert "n_aug=0" in label


def test_knn_head_progress_reports_each_head():
    from otitenet.app.pages.learned_embedding import _compute_knn_heads

    seen = []
    X_train = [[0.0], [1.0], [2.0], [3.0]]
    y_train = [0, 0, 1, 1]
    X_valid = [[0.1], [2.9]]
    y_valid = [0, 1]

    _compute_knn_heads(
        X_train,
        y_train,
        X_valid,
        y_valid,
        [1, 2, 3],
        on_head_start=seen.append,
    )

    assert seen == ["KNN k=1", "KNN k=2", "KNN k=3"]
