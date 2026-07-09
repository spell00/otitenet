from argparse import Namespace

from otitenet.app.services.inference_results_service import args_from_inference_row


def test_args_from_inference_row_does_not_inherit_sidebar_prototype_head():
    base_args = Namespace(
        task="otitis_four_class",
        model_name="resnet18",
        path="data/otite_ds_64",
        new_size=224,
        fgsm="0",
        n_calibration="4",
        classif_loss="ce",
        dloss="no",
        dist_fct="cosine",
        normalize="yes",
        n_neighbors=19,
        prototypes_to_use="class",
        prototype_strategy="kmeans",
        prototype_components=2,
        n_aug=2,
        best_classifier_config="protot_kmeans_2",
        classification_head_config="protot_kmeans_2",
        siamese_inference="knn",
    )
    row = {
        "Model ID": 123,
        "Model Name": "resnet18",
        "Task": "otitis_four_class",
        "Dataset": "otite_ds_64",
        "NSize": 224,
        "FGSM": 0,
        "N_Calibration": 4,
        "Classif_Loss": "ce",
        "DLoss": "no",
        "Dist_Fct": "cosine",
        "Normalize": "yes",
        "Prototypes": "no",
        "NPos": 1,
        "NNeg": 1,
        "N_Neighbors": 1,
        "Proto_Strat": "mean",
        "Proto_Comp": 1,
        "Log Path": (
            "logs/best_models/otitis_four_class/resnet18/otite_ds_64/"
            "nsize224/fgsm0/ncal4/ce/no/prototypes_no/npos1/nneg1/"
            "protoagg_mean_1/normyes"
        ),
    }

    args = args_from_inference_row(base_args, row)

    assert not hasattr(args, "best_classifier_config")
    assert not hasattr(args, "classification_head_config")
    assert not hasattr(args, "n_aug")
    assert args.prototypes_to_use == "no"
    assert args.prototype_strategy == "mean"
    assert args.prototype_components == 1
    assert args.n_neighbors == 1
