from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms

from otitenet.app.image_processing import infer_base_resize_size, preprocess_image_array
from otitenet.app.image_processing import preprocessing_trace
from otitenet.data.transforms_manifest import image_preprocessing_manifest
from otitenet.offline.deployment import OfflineDeployment
from otitenet.app.inference import predict_with_prototype_distance_ratio_proba
from otitenet.offline.predictor import preprocess_image_array as offline_preprocess_image_array
from otitenet.offline.predictor import _prototype_probabilities
from otitenet.offline.predictor import _align_classifier_probabilities
from otitenet.utils.encoding_utils import (
    PerImageNormalizeTransform,
    get_base_transform,
    get_knn_augmentation_transform,
)


def _sample_image(size: int = 64) -> Image.Image:
    values = (np.arange(size * size * 3, dtype=np.uint32) % 256).astype(np.uint8)
    return Image.fromarray(values.reshape(size, size, 3), mode="RGB")


def _training_preprocess(img: Image.Image, normalize: str) -> np.ndarray:
    tensor = get_base_transform(normalize=normalize)(img)
    return torch.as_tensor(tensor).detach().cpu().numpy()[None, ...]


def test_training_and_app_preprocessing_match_for_all_normalization_modes():
    img = _sample_image()

    for mode in ["yes", "imagenet", "no", "per_image"]:
        training_array = _training_preprocess(img, normalize=mode)
        app_array = preprocess_image_array(img, 64, normalize=mode)

        np.testing.assert_allclose(app_array, training_array, rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(
        preprocess_image_array(img, 64),
        _training_preprocess(img, normalize="yes"),
        rtol=1e-5,
        atol=1e-5,
    )


def test_manifest_and_offline_preprocessing_match_app_preprocessing():
    img = _sample_image()

    for mode in ["yes", "imagenet", "no", "per_image"]:
        manifest = {
            "input": {"image_size": [64, 64], "channels": 3},
            "preprocessing": image_preprocessing_manifest((64, 64), normalize=mode),
            "files": {"model": "model.pth"},
        }
        deployment = OfflineDeployment(root=Path("."), manifest=manifest)

        offline_array = offline_preprocess_image_array(img, deployment)
        app_array = preprocess_image_array(img, 64, normalize=mode)

        np.testing.assert_allclose(offline_array, app_array, rtol=1e-6, atol=1e-6)


def test_offline_channel_mean_std_uses_same_resize_path_as_app_for_224():
    img = _sample_image(96)
    manifest = {
        "input": {"image_size": [224, 224], "channels": 3},
        "preprocessing": image_preprocessing_manifest((224, 224), normalize="yes"),
        "files": {"model": "model.pth"},
    }
    deployment = OfflineDeployment(root=Path("."), manifest=manifest)

    offline_array = offline_preprocess_image_array(img, deployment)
    app_array = preprocess_image_array(img, 224, normalize="yes")

    np.testing.assert_allclose(offline_array, app_array, rtol=1e-6, atol=1e-6)


def test_preprocessing_trace_reports_visible_steps_and_tensor_stats():
    trace = preprocessing_trace(_sample_image(96), 224, normalize="yes")

    assert trace["resize_mode"] == "app_64_then_size"
    assert trace["normalize_mode"] == "imagenet"
    assert trace["raw"]["size"] == [96, 96]
    assert trace["downsized"]["size"] == [64, 64]
    assert trace["resized"]["size"] == [224, 224]
    assert trace["tensor_before_normalize"]["shape"] == [3, 224, 224]
    assert trace["tensor_after_normalize"]["shape"] == [3, 224, 224]


def test_model_dataset_path_controls_first_resize_before_model_size():
    img = _sample_image(96)
    base_size = infer_base_resize_size("data/otite_ds_224/USA_Turquie_Chili")

    trace = preprocessing_trace(img, 64, normalize="no", base_size=base_size)
    app_array = preprocess_image_array(img, 64, normalize="no", base_size=base_size)
    manifest = {
        "input": {"image_size": [64, 64], "channels": 3},
        "preprocessing": image_preprocessing_manifest((64, 64), normalize="no", base_resize_size=base_size),
        "files": {"model": "model.pth"},
    }
    deployment = OfflineDeployment(root=Path("."), manifest=manifest)

    assert base_size == 224
    assert trace["resize_mode"] == "app_224_then_size"
    assert trace["downsized"]["size"] == [224, 224]
    assert manifest["preprocessing"]["base_resize_size"] == 224
    assert manifest["preprocessing"]["resize_mode"] == "app_224_then_size"
    np.testing.assert_allclose(
        offline_preprocess_image_array(img, deployment),
        app_array,
        rtol=1e-6,
        atol=1e-6,
    )


def test_manifest_records_distinct_supported_normalization_modes():
    default_manifest = image_preprocessing_manifest((64, 64))
    imagenet_manifest = image_preprocessing_manifest((64, 64), normalize="yes")
    per_image_manifest = image_preprocessing_manifest((64, 64), normalize="per_image")
    no_manifest = image_preprocessing_manifest((64, 64), normalize="no")

    assert default_manifest == imagenet_manifest

    assert imagenet_manifest["normalize"] == "yes"
    assert imagenet_manifest["normalization"] == "channel_mean_std"
    assert imagenet_manifest["normalize_mean"] == [0.485, 0.456, 0.406]
    assert imagenet_manifest["normalize_std"] == [0.229, 0.224, 0.225]

    assert per_image_manifest["normalize"] == "yes"
    assert per_image_manifest["normalization"] == "per_image"

    assert no_manifest["normalize"] == "no"
    assert no_manifest["normalization"] == "none"


def test_knn_augmentation_has_translation_and_same_normalization_modes_as_base():
    img = _sample_image()

    for mode in ["yes", "imagenet", "no", "per_image"]:
        aug_transform = get_knn_augmentation_transform(64, normalize=mode)
        assert any(isinstance(op, transforms.RandomAffine) for op in aug_transform.transforms)

        norm_ops = [
            op for op in aug_transform.transforms
            if isinstance(op, (transforms.Normalize, PerImageNormalizeTransform))
        ]
        if mode == "no":
            assert norm_ops == []
        elif mode == "per_image":
            assert isinstance(norm_ops[-1], PerImageNormalizeTransform)
        else:
            assert isinstance(norm_ops[-1], transforms.Normalize)

        output = aug_transform(img)
        assert tuple(torch.as_tensor(output).shape) == (3, 64, 64)


def test_offline_prototype_probabilities_match_online_distance_softmax():
    embedding = np.asarray([[0.0, 0.0]], dtype=np.float32)
    prototypes = {
        "A": np.asarray([[0.1, 0.0], [10.0, 0.0]], dtype=np.float32),
        "B": np.asarray([[0.2, 0.0], [0.3, 0.0]], dtype=np.float32),
    }
    labels = ["A", "B"]

    pred_label, online_probs = predict_with_prototype_distance_ratio_proba(
        torch.as_tensor(embedding),
        prototypes,
        dist_fct_name="euclidean",
    )
    offline_probs = _prototype_probabilities(embedding, prototypes, labels, "euclidean")

    assert pred_label == "A"
    np.testing.assert_allclose(
        offline_probs,
        np.asarray([online_probs[label] for label in labels], dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_sklearn_head_probabilities_align_by_label_not_position():
    probs = np.asarray([0.10, 0.20, 0.30, 0.40])
    classes = np.asarray([0, 1, 2, 3])
    head_labels = ["Normal", "Tube", "NotNormal", "Wax"]
    manifest_labels = ["Normal", "NotNormal", "Wax", "Tube"]

    aligned = _align_classifier_probabilities(probs, classes, head_labels, manifest_labels)

    np.testing.assert_allclose(aligned, [0.10, 0.30, 0.40, 0.20])
