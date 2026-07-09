import argparse

import numpy as np
import torch

from otitenet.api import mobile_deployment
from otitenet.api.mobile_deployment import _head_metadata, create_torch_prototype_manifest, normalized_distance_metadata
from otitenet.app.args import _resolve_production_head_config, _resolve_production_head_n_aug, _set_distance_aliases
from otitenet.app.services.production_model_service import (
    _coerce_db_value,
    _production_model_id_for_db,
    apply_production_model_to_args,
)
from scripts import create_mobile_deployment
from scripts.create_mobile_deployment import (
    align_prototype_deployment_metadata,
    resolve_deployment_distance,
    resolve_head_config,
)


def test_apply_production_model_to_args_uses_best_head_config_and_n_aug():
    args = argparse.Namespace(n_neighbors=1, prototypes_to_use="no", new_size=64, normalize="no")
    production_model = {
        "Model ID": "946887804",
        "Model Name": "resnet18",
        "NSize": "64",
        "Normalize": "yes",
        "FGSM": "0",
        "Dist_Fct": "cosine",
        "Classif_Loss": "softmax_contrastive",
        "DLoss": "no",
        "N_Calibration": "4",
        "NPos": "1",
        "NNeg": "1",
        "Prototypes": "class",
        "Best Head Config": "protot_kmeans_2",
        "Best Classification Head": "Prototype: KMEANS (n_comp=2)",
        "Head N Aug": 2,
    }

    out = apply_production_model_to_args(args, production_model)

    assert out.model_id == "946887804"
    assert out.model_name == "resnet18"
    assert out.new_size == 64
    assert out.normalize == "yes"
    assert out.fgsm == "0"
    assert out.dist_fct == "cosine"
    assert out.classif_loss == "softmax_contrastive"
    assert out.dloss == "no"
    assert out.n_calibration == "4"
    assert out.n_positives == "1"
    assert out.n_negatives == "1"
    assert out.learned_classifier_label == "Prototype: KMEANS (n_comp=2)"
    assert out.head_name_selected == "Prototype: KMEANS (n_comp=2)"
    assert out.best_classifier_config == "protot_kmeans_2"
    assert out.classification_head_config == "protot_kmeans_2"
    assert out.classifier_head_config == "protot_kmeans_2"
    assert out.head_config == "protot_kmeans_2"
    assert out.classification_head_family == "prototype"
    assert out.prototypes_to_use == "class"
    assert out.prototype_strategy == "kmeans"
    assert out.prototype_components == 2
    assert out.n_aug == 2


def test_set_production_head_fallback_uses_row_best_head_before_default():
    row = {
        "Best Head Config": "protot_kmeans_2",
        "N Aug": 2,
    }

    assert _resolve_production_head_config(row) == "protot_kmeans_2"
    assert _resolve_production_head_n_aug(row) == 2
    assert _resolve_production_head_config(row, selected_head_config="baseline_logreg") == "baseline_logreg"
    assert _resolve_production_head_config({}) == "baseline_linear_svc"


def test_deployment_manifest_head_metadata_is_client_visible():
    params = {
        "head": "Prototype: KMEANS (n_comp=2)",
        "head_config": "protot_kmeans_2",
        "head_family": "prototype",
        "head_n_aug": 2,
    }

    metadata = _head_metadata(params)

    assert metadata == {
        "head_name": "Prototype: KMEANS (n_comp=2)",
        "head_config": "protot_kmeans_2",
        "head_family": "prototype",
        "head_n_aug": 2,
    }


def test_explicit_export_distance_overrides_stale_production_aliases():
    production_model = {
        "Dist_Fct": "cosine",
        "dist_fct": "cosine",
        "dist_metric": "cosine",
        "Distance": "cosine",
    }
    production_params = {"dist_fct": "cosine"}

    distance, params = resolve_deployment_distance(
        production_model,
        production_params,
        explicit_dist_fct="euclidean",
        fallback="cosine",
    )

    assert distance == "euclidean"
    assert params["Dist_Fct"] == "euclidean"
    assert params["dist_fct"] == "euclidean"
    assert params["dist_metric"] == "euclidean"
    assert params["Distance"] == "euclidean"


def test_distance_metadata_normalizes_all_aliases():
    distance, params = normalized_distance_metadata(
        {"Dist_Fct": "cosine", "dist_metric": "euclidean"},
        distance="euclidean",
    )

    assert distance == "euclidean"
    assert params["Dist_Fct"] == "euclidean"
    assert params["dist_fct"] == "euclidean"
    assert params["dist_metric"] == "euclidean"
    assert params["Distance"] == "euclidean"


def test_sidebar_production_distance_aliases_are_synchronized():
    model_dict = {
        "Dist_Fct": "cosine",
        "dist_fct": "cosine",
        "dist_metric": "cosine",
        "Distance": "cosine",
    }

    out = _set_distance_aliases(model_dict, "euclidean")

    assert out["Dist_Fct"] == "euclidean"
    assert out["dist_fct"] == "euclidean"
    assert out["dist_metric"] == "euclidean"
    assert out["Distance"] == "euclidean"


def test_production_model_db_id_falls_back_to_model_id_alias():
    model_info = {
        "model_id": None,
        "Model ID": "946887804",
        "Registry ID": None,
    }

    assert _production_model_id_for_db(model_info) == "946887804"


def test_production_model_db_coercion_turns_blank_int_into_null():
    assert _coerce_db_value("prototype_components", "", {"prototype_components": "int"}) is None
    assert _coerce_db_value("prototype_components", "2", {"prototype_components": "int"}) == 2


def test_legacy_production_model_insert_uses_null_for_blank_prototype_components():
    from otitenet.app import database

    class FakeCursor:
        def __init__(self):
            self.insert_params = None

        def execute(self, query, params=None):
            if "INSERT INTO production_model" in query:
                self.insert_params = params

    class FakeConn:
        def __init__(self):
            self.committed = False

        def commit(self):
            self.committed = True

        def rollback(self):
            pass

    cursor = FakeCursor()
    conn = FakeConn()

    saved = database.set_production_model(
        cursor,
        conn,
        {
            "label_scheme": "four_class",
            "Model ID": "991812442",
            "Model Name": "resnet18",
            "Best Head Config": "baseline_rbf_svc",
            "prototype_components": "",
            "Proto_Comp": "",
        },
        set_by_email="admin@example.com",
    )

    assert saved is True
    assert conn.committed is True
    assert cursor.insert_params[10] is None


def test_prototype_manifest_keeps_distance_aliases_in_sync(tmp_path, monkeypatch):
    monkeypatch.setattr(mobile_deployment, "CURRENT_DIR", tmp_path)
    monkeypatch.setattr(mobile_deployment, "CURRENT_MANIFEST", tmp_path / "manifest.json")

    torch.save({"linear.weight": torch.zeros((2, 512))}, tmp_path / "model.pth")
    np.savez(
        tmp_path / "prototypes.npz",
        labels=np.asarray(["Normal", "NotNormal"]),
        prototypes=np.zeros((2, 1, 512), dtype=np.float32),
    )

    manifest = create_torch_prototype_manifest(
        model_id="946887804",
        model_name="resnet18",
        model_file="model.pth",
        prototypes_file="prototypes.npz",
        labels=["Normal", "NotNormal"],
        input_size=(224, 224),
        normalize="yes",
        distance="euclidean",
        head_config="protot_mean_1",
        production_params={"dist_fct": "cosine"},
    )

    assert manifest["distance"] == "euclidean"
    assert manifest["production_params"]["Dist_Fct"] == "euclidean"
    assert manifest["production_params"]["dist_fct"] == "euclidean"
    assert manifest["production_params"]["dist_metric"] == "euclidean"
    assert manifest["production_params"]["Distance"] == "euclidean"


def test_mobile_deployment_resolves_best_head_config_key():
    production_model = {
        "Best Head Config": "protot_kmeans_2",
        "Head Config": "baseline_linear_svc",
    }

    assert resolve_head_config(production_model) == "protot_kmeans_2"
    assert resolve_head_config(production_model, explicit_head_config="3") == "3"


def test_prototype_deployment_metadata_aligns_to_four_class_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(create_mobile_deployment, "CURRENT_DIR", tmp_path)

    torch.save(
        {
            "linear.weight": torch.zeros((4, 512)),
            "subcenters": torch.zeros((4, 2, 512)),
        },
        tmp_path / "model.pth",
    )
    np.savez(
        tmp_path / "prototypes.npz",
        labels=np.asarray(["Normal", "NotNormal", "Wax", "Tube"]),
        prototypes=np.zeros((4, 2, 512), dtype=np.float32),
        strategy=np.asarray("kmeans"),
        components=np.asarray(2),
    )

    labels, head_config, params = align_prototype_deployment_metadata(
        labels=["Normal", "NotNormal"],
        model_file="model.pth",
        prototypes_file="prototypes.npz",
        head_config="protot_kmeans_3",
        production_params={"label_task": "notNormal", "label_scheme": "binary"},
        explicit_labels=False,
        explicit_head_config=False,
    )

    assert labels == ["Normal", "NotNormal", "Wax", "Tube"]
    assert head_config == "protot_kmeans_2"
    assert params["label_task"] == "otite_four_class"
    assert params["label_scheme"] == "four_class"
    assert params["prototype_strategy"] == "kmeans"
    assert params["prototype_components"] == 2
