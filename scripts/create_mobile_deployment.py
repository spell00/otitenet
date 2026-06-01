# /home/simon/otitenet/scripts/create_mobile_deployment.py

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from otitenet.api.mobile_deployment import (
    CURRENT_DIR,
    create_simple_torch_classifier_manifest,
    create_knn_embedding_manifest,
    create_torch_prototype_manifest,
)
from otitenet.app.utils import extract_params_from_log_path, parse_classifier_config
from otitenet.data.labels import DEFAULT_LABEL_TASK, label_scheme_for_task, labels_for_task
DEFAULT_LABELS = None


def parse_labels(labels_text: str) -> list[str]:
    labels = [x.strip() for x in labels_text.split(",") if x.strip()]
    if not labels:
        raise ValueError("At least one label is required.")
    return labels


def parse_input_size(text: str) -> tuple[int, int]:
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--input-size must be H,W, for example 224,224")
    return int(parts[0]), int(parts[1])


def _normalize_yes_no(value) -> str:
    text = str(value or "no").strip().lower()
    if text in {"yes", "true", "1", "per_image"}:
        return "yes"
    return "no"


def _production_value(production_model: dict | None, *keys, default=None):
    if not production_model:
        return default
    for key in keys:
        value = production_model.get(key)
        if value not in (None, ""):
            return value
    return default


def infer_input_size_from_production(production_model: dict | None, fallback: tuple[int, int]) -> tuple[int, int]:
    raw_size = _production_value(production_model, "NSize", "nsize", "new_size", default=None)
    if raw_size is None:
        return fallback
    text = str(raw_size).strip().lower()
    if text.startswith("nsize"):
        text = text[len("nsize"):]
    try:
        size = int(float(text))
    except Exception:
        return fallback
    return size, size


def build_production_params(production_model: dict | None) -> dict:
    if not production_model:
        return {}
    key_map = {
        "model_id": ("model_id", "Model ID", "id"),
        "model_name": ("model_name", "Model Name", "model"),
        "log_path": ("log_path", "Log Path", "logPath"),
        "new_size": ("NSize", "nsize", "new_size"),
        "normalize": ("Normalize", "normalize"),
        "prototypes_to_use": ("Prototypes", "prototypes"),
        "dist_fct": ("Dist_Fct", "dist_fct"),
        "classif_loss": ("Classif_Loss", "classif_loss"),
        "dloss": ("DLoss", "dloss"),
        "n_neighbors": ("N_Neighbors", "n_neighbors"),
        "n_positives": ("NPos", "npos", "n_positives"),
        "n_negatives": ("NNeg", "nneg", "n_negatives"),
        "n_calibration": ("N_Calibration", "n_calibration"),
        "head": ("Head", "head_name", "learned_classifier_label"),
        "head_config": ("Head Config", "head_config", "classification_head_config", "best_classifier_config"),
        "head_family": ("head_family",),
        "head_n_aug": ("head_n_aug", "n_aug", "N Aug"),
        "prototype_strategy": ("prototype_strategy", "Proto_Strat"),
        "prototype_components": ("prototype_components", "Proto_Comp"),
    }
    return {
        out_key: _production_value(production_model, *keys)
        for out_key, keys in key_map.items()
        if _production_value(production_model, *keys) is not None
    }


def resolve_head_config(production_model: dict | None, explicit_head_config: str | None = None) -> str | None:
    return (
        explicit_head_config
        or _production_value(
            production_model,
            "best_classifier_config",
            "classification_head_config",
            "head_config",
            "Head Config",
            default=None,
        )
    )


def resolve_deployment_type(requested_type: str, production_model: dict | None, head_config: str | None) -> str:
    if requested_type != "auto":
        return requested_type

    head_meta = parse_classifier_config(head_config)
    if head_meta.get("family") == "prototype":
        return "torch_prototype"
    if head_meta.get("family") == "knn":
        return "knn_embedding"
    return "torch_classifier"


# def ensure_supported_direct_classifier(production_model: dict | None, allow_head_mismatch: bool) -> None:
    # No restrictions: allow any head to be deployed
#     return


def copy_to_current(source: str | Path, target_name: str | None = None) -> str:
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(f"Missing file: {source}")

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)

    target = CURRENT_DIR / (target_name or source.name)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)

    return target.name


def resolve_prototype_file_from_production_model(production_model: dict) -> Path:
    for key in ("prototypes_file", "prototype_file", "prototypes_path"):
        value = production_model.get(key)
        if value:
            path = Path(value)
            if path.exists() and path.is_file():
                return path

    log_path = (
        production_model.get("log_path")
        or production_model.get("Log Path")
        or production_model.get("logPath")
    )
    if log_path:
        path = Path(log_path)
        candidates = []
        if path.is_file() and path.name == "prototypes.pkl":
            return path
        if path.is_dir():
            candidates.append(path / "prototypes.pkl")
            candidates.extend(path.glob("**/prototypes.pkl"))
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "Could not resolve prototypes.pkl from the production model. "
        "Exact prototype-head deployment requires the same prototype artifact used by the web app."
    )


def write_prototypes_npz(source: str | Path, production_model: dict, labels: list[str]) -> str:
    import pickle
    import numpy as np

    source = Path(source)
    head_config = (
        production_model.get("best_classifier_config")
        or production_model.get("classification_head_config")
        or production_model.get("head_config")
        or production_model.get("Head Config")
    )
    head_meta = parse_classifier_config(head_config)
    strategy = str(head_meta.get("strategy") or production_model.get("prototype_strategy") or "mean")
    components = int(head_meta.get("components") or production_model.get("prototype_components") or 1)

    with source.open("rb") as f:
        proto_obj = pickle.load(f)

    class_prototypes = getattr(proto_obj, "class_prototypes", {}) or {}
    train_prototypes = class_prototypes.get("train") or class_prototypes
    if not isinstance(train_prototypes, dict) or not train_prototypes:
        raise ValueError(f"No train class prototypes found in {source}")

    ordered_labels = [str(label) for label in labels]
    vectors = []
    for label in ordered_labels:
        proto = train_prototypes.get(label)
        if proto is None:
            try:
                proto = train_prototypes.get(int(label))
            except Exception:
                proto = None
        if proto is None:
            raise KeyError(f"Prototype for label {label!r} not found in {source}")
        proto_array = np.asarray(proto, dtype=np.float32)
        if proto_array.ndim == 1:
            proto_array = proto_array[None, :]
        vectors.append(proto_array)

    target = CURRENT_DIR / "prototypes.npz"
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        labels=np.asarray(ordered_labels),
        prototypes=np.asarray(vectors, dtype=object),
        strategy=np.asarray(strategy),
        components=np.asarray(components),
    )
    return target.name


def write_learned_prototypes_npz(production_model: dict, labels: list[str], head_config: str, task: str) -> str:
    import argparse as _argparse
    import numpy as np

    from otitenet.app.pages.learned_embedding import _encode_splits_for_args
    from otitenet.app.services.embedding_optimization_service import args_from_model_row
    from otitenet.utils.encoding_utils import compute_prototypes_by_strategy, flatten_prototype_dict

    head_meta = parse_classifier_config(head_config)
    if head_meta.get("family") != "prototype":
        raise ValueError(f"Cannot export learned prototypes for non-prototype head: {head_config}")

    log_path = _production_value(production_model, "log_path", "Log Path", "logPath", default="")
    parsed = extract_params_from_log_path(str(log_path))
    dataset = parsed.get("Dataset") or production_model.get("Dataset") or "otite_ds_64"
    dataset_path = str(dataset) if str(dataset).startswith("data/") else f"data/{dataset}"

    base_args = _argparse.Namespace(
        task=task,
        model_name=_production_value(production_model, "model_name", "Model Name", default=parsed.get("Model Name")),
        path=dataset_path,
        new_size=int(_production_value(production_model, "NSize", "new_size", default=parsed.get("new_size") or 224)),
        fgsm=str(_production_value(production_model, "FGSM", "fgsm", default=parsed.get("FGSM") or 0)),
        n_calibration=str(_production_value(production_model, "N_Calibration", "n_calibration", default=parsed.get("N_Calibration") or 0)),
        classif_loss=str(_production_value(production_model, "Classif_Loss", "classif_loss", default=parsed.get("classif_loss") or "triplet")),
        dloss=str(_production_value(production_model, "DLoss", "dloss", default=parsed.get("DLoss") or "triplet")),
        prototypes_to_use=str(_production_value(production_model, "Prototypes", "prototypes_to_use", default=parsed.get("Prototypes") or "class")),
        n_positives=int(_production_value(production_model, "NPos", "n_positives", default=parsed.get("NPos") or 1)),
        n_negatives=int(_production_value(production_model, "NNeg", "n_negatives", default=parsed.get("NNeg") or 1)),
        normalize=str(_production_value(production_model, "Normalize", "normalize", default=parsed.get("Normalize") or "no")),
        dist_fct=str(_production_value(production_model, "Dist_Fct", "dist_fct", default=parsed.get("Dist_Fct") or "cosine")),
        n_neighbors=int(_production_value(production_model, "N_Neighbors", "n_neighbors", default=parsed.get("N_Neighbors") or 1)),
        prototype_strategy=str(head_meta.get("strategy") or "mean"),
        prototype_components=int(head_meta.get("components") or 1),
        model_id=_production_value(production_model, "model_id", "Model ID", "id", default=None),
        log_path=log_path,
        bs=32,
        groupkfold=1,
        random_recs=0,
    )
    row = dict(production_model or {})
    row["Log Path"] = log_path
    row["Task"] = task
    model_args = args_from_model_row(base_args, row)
    model_args.prototype_strategy = str(head_meta.get("strategy") or "mean")
    model_args.prototype_components = int(head_meta.get("components") or 1)
    model_args.n_aug = int(_production_value(production_model, "head_n_aug", "n_aug", "N Aug", default=0) or 0)

    encoded = _encode_splits_for_args(model_args)
    X_train = encoded["train"]["X"]
    y_train = encoded["train"]["y"]

    proto_dict = compute_prototypes_by_strategy(
        X_train,
        y_train,
        str(head_meta.get("strategy") or "mean"),
        int(head_meta.get("components") or 1),
        random_state=1,
    )
    proto_vecs, proto_labels = flatten_prototype_dict(proto_dict)
    if len(proto_vecs) == 0:
        raise ValueError(f"No learned prototypes could be generated for {head_config}")

    ordered_labels = [str(label) for label in labels]
    prototypes_by_label = []
    for class_idx, _label in enumerate(ordered_labels):
        vectors = proto_vecs[proto_labels == class_idx]
        if len(vectors) == 0:
            raise KeyError(f"No learned prototype vectors generated for class index {class_idx} ({_label})")
        prototypes_by_label.append(np.asarray(vectors, dtype=np.float32))

    target = CURRENT_DIR / "prototypes.npz"
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        labels=np.asarray(ordered_labels),
        prototypes=np.asarray(prototypes_by_label, dtype=object),
        prototype_vectors=np.asarray(proto_vecs, dtype=np.float32),
        prototype_labels=np.asarray(proto_labels, dtype=np.int64),
        strategy=np.asarray(str(head_meta.get("strategy") or "mean")),
        components=np.asarray(int(head_meta.get("components") or 1)),
        n_aug=np.asarray(int(model_args.n_aug)),
    )
    return target.name


def _row_get(row, key, index=None):
    """
    Works with tuple cursors and dict cursors.
    """
    if row is None:
        return None

    if isinstance(row, dict):
        return row.get(key)

    if index is not None:
        return row[index]

    return None


def get_current_production_model_from_db(task: str = DEFAULT_LABEL_TASK):
    """
    Read the model selected by the web admin.

    Expected logic:
    - Admin clicks "Set as Production Model" in the web/sidebar/admin UI.
    - DB stores current production model.
    - This script reads that active production model.

    This function tries the clean database helper first. If unavailable,
    it falls back to querying common production_model table shapes.
    """
    from otitenet.app.database import create_db, ensure_production_model_table

    conn, cursor = create_db()
    ensure_production_model_table(conn, cursor)

    # Preferred path if your database.py already has get_production_model()
    try:
        from otitenet.app.database import get_production_model

        model = get_production_model(cursor, label_scheme=label_scheme_for_task(task))

        if model:
            return model

    except Exception as e:
        print(f"[mobile-deploy] database.get_production_model failed or unavailable: {e}")

    # Fallback 1: production_model table with best_models_registry join
    try:
        cursor.execute(
            """
            SELECT
                b.id AS model_id,
                b.model_name,
                b.log_path,
                b.nsize,
                b.mcc,
                b.accuracy
            FROM production_model p
            JOIN best_models_registry b
                ON b.id = p.model_id
            ORDER BY p.id DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()

        if row:
            if isinstance(row, dict):
                return row

            return {
                "model_id": row[0],
                "model_name": row[1],
                "log_path": row[2],
                "nsize": row[3],
                "mcc": row[4],
                "accuracy": row[5],
            }

    except Exception as e:
        print(f"[mobile-deploy] production_model join query failed: {e}")

    # Fallback 2: production_model stores log_path directly
    try:
        cursor.execute(
            """
            SELECT
                model_id,
                model_name,
                log_path
            FROM production_model
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()

        if row:
            if isinstance(row, dict):
                return row

            return {
                "model_id": row[0],
                "model_name": row[1],
                "log_path": row[2],
            }

    except Exception as e:
        print(f"[mobile-deploy] production_model direct query failed: {e}")

    raise RuntimeError(
        "No production model found. Set a production model in the web app first, "
        "or pass --model-file manually."
    )


def resolve_model_file_from_production_model(production_model: dict) -> Path:
    """
    Try to find the physical model file from the production model metadata.

    This is intentionally defensive because your app stores different paths
    depending on branch/version.

    It checks:
    - mobile_model_path
    - model_file
    - exported_model_path
    - log_path if it points to a file
    - common model files inside log_path directory
    """
    candidate_keys = [
        "mobile_model_path",
        "model_file",
        "exported_model_path",
        "model_path",
        "path",
    ]

    for key in candidate_keys:
        value = production_model.get(key)
        if value:
            p = Path(value)
            if p.exists() and p.is_file():
                return p

    log_path = (
        production_model.get("log_path")
        or production_model.get("Log Path")
        or production_model.get("logPath")
    )

    if log_path:
        p = Path(log_path)

        if p.exists() and p.is_file():
            return p

        if p.exists() and p.is_dir():
            common_names = [
                "model.ptl",
                "model.pt",
                "model.pth",
                "best_model.ptl",
                "best_model.pt",
                "best_model.pth",
                "checkpoint.pt",
                "checkpoint.pth",
                "model.onnx",
                "model.tflite",
            ]

            for name in common_names:
                candidate = p / name
                if candidate.exists() and candidate.is_file():
                    return candidate

            matches = []
            for pattern in ["*.ptl", "*.pt", "*.pth", "*.onnx", "*.tflite"]:
                matches.extend(p.glob(pattern))

            if matches:
                return sorted(matches)[0]

    model_id = production_model.get("model_id") or production_model.get("Model ID")
    model_name = production_model.get("model_name") or production_model.get("Model Name")

    raise FileNotFoundError(
        "Could not resolve a mobile model file from the production model.\n"
        f"production model id: {model_id}\n"
        f"production model name: {model_name}\n"
        f"production model keys: {list(production_model.keys())}\n\n"
        "Either export/copy a mobile model into the model's log_path directory, "
        "or pass --model-file explicitly."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create the current mobile deployment manifest for OtiteNet."
    )

    parser.add_argument(
        "--deployment-type",
        choices=["auto", "torch_classifier", "knn_embedding", "torch_prototype"],
        default="auto",
    )

    parser.add_argument(
        "--model-file",
        default=None,
        help=(
            "Optional path to exported mobile classifier model. "
            "If omitted, the script uses the current production model selected in the web app."
        ),
    )

    parser.add_argument(
        "--model-id",
        type=int,
        default=None,
        help="Optional model id. Defaults to production model id.",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name. Defaults to production model name.",
    )

    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="Comma-separated class labels in model output order. Defaults to the selected label scheme.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_LABEL_TASK,
        help="Production task/labeling scenario to export, for example notNormal or otite_four_class.",
    )

    parser.add_argument(
        "--input-size",
        default=None,
        help="Input size as H,W. Defaults to the selected production model NSize.",
    )
    parser.add_argument(
        "--allow-head-mismatch",
        action="store_true",
        help="Allow exporting a direct classifier even when the web production model uses a prototype/KNN head.",
    )
    parser.add_argument("--head-config", default=None)
    parser.add_argument("--normalize", choices=["yes", "no"], default=None)
    parser.add_argument("--dist-fct", default=None)

    # KNN embedding deployment
    parser.add_argument("--embedding-model-file", default=None)
    parser.add_argument("--reference-embeddings-file", default=None)
    parser.add_argument("--reference-labels-file", default=None)
    parser.add_argument("--prototypes-file", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--distance", default="cosine")

    args = parser.parse_args()

    labels = parse_labels(args.labels) if args.labels else list(labels_for_task(args.task))
    input_size = parse_input_size(args.input_size) if args.input_size else (224, 224)

    production_model = None

    if args.model_file is None or args.model_id is None or args.model_name is None:
        production_model = get_current_production_model_from_db(task=args.task)
        print("[mobile-deploy] Using production model from web app:")
        print(production_model)

    input_size = infer_input_size_from_production(production_model, input_size)
    normalize = _normalize_yes_no(_production_value(production_model, "Normalize", "normalize", default="no"))
    if args.normalize is not None:
        normalize = args.normalize
    production_params = build_production_params(production_model)
    production_params["label_task"] = args.task
    production_params["label_scheme"] = label_scheme_for_task(args.task)
    head_config = resolve_head_config(production_model, args.head_config)
    head_meta = parse_classifier_config(head_config)
    deployment_type = resolve_deployment_type(args.deployment_type, production_model, head_config)

    if head_config:
        production_params["head"] = head_config
        production_params["head_config"] = head_config
        production_params["head_family"] = head_meta.get("family")
        if head_meta.get("family") == "prototype":
            production_params["prototypes_to_use"] = "class"
            production_params["prototype_strategy"] = head_meta.get("strategy")
            production_params["prototype_components"] = head_meta.get("components")
        elif head_meta.get("family") == "knn":
            production_params["n_neighbors"] = head_meta.get("k")
    if args.dist_fct:
        production_params["dist_fct"] = args.dist_fct

    model_id = args.model_id
    if model_id is None:
        model_id = (
            production_model.get("model_id")
            or production_model.get("Model ID")
            or production_model.get("id")
        )

    model_name = args.model_name
    if model_name is None:
        model_name = (
            production_model.get("model_name")
            or production_model.get("Model Name")
            or "production_model"
        )

    if model_id is None:
        raise RuntimeError("Could not determine model_id. Pass --model-id manually.")

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)

    if deployment_type != args.deployment_type:
        print(f"[mobile-deploy] Auto-selected deployment type: {deployment_type} from head_config={head_config!r}")

    if deployment_type == "torch_classifier":
        # ensure_supported_direct_classifier(production_model, args.allow_head_mismatch)
        if args.model_file:
            source_model = Path(args.model_file)
        else:
            source_model = resolve_model_file_from_production_model(production_model)

        copied_model_name = copy_to_current(source_model)

        manifest = create_simple_torch_classifier_manifest(
            model_id=int(model_id),
            model_name=str(model_name),
            model_file=copied_model_name,
            labels=labels,
            input_size=input_size,
            normalize=normalize,
            production_params=production_params,
        )

    elif deployment_type == "knn_embedding":
        if args.embedding_model_file is None:
            if production_model is None:
                production_model = get_current_production_model_from_db(task=args.task)
            source_embedding = resolve_model_file_from_production_model(production_model)
        else:
            source_embedding = Path(args.embedding_model_file)

        if args.reference_embeddings_file is None:
            raise ValueError(
                "--reference-embeddings-file is required for knn_embedding deployment."
            )

        if args.reference_labels_file is None:
            raise ValueError(
                "--reference-labels-file is required for knn_embedding deployment."
            )

        copied_embedding_model_name = copy_to_current(source_embedding)
        copied_reference_embeddings_name = copy_to_current(args.reference_embeddings_file)
        copied_reference_labels_name = copy_to_current(args.reference_labels_file)

        manifest = create_knn_embedding_manifest(
            model_id=int(model_id),
            model_name=str(model_name),
            embedding_model_file=copied_embedding_model_name,
            reference_embeddings_file=copied_reference_embeddings_name,
            reference_labels_file=copied_reference_labels_name,
            labels=labels,
            k=int(head_meta.get("k") or args.k),
            distance=args.distance,
            input_size=input_size,
        )

    elif deployment_type == "torch_prototype":
        if args.model_file:
            source_model = Path(args.model_file)
        else:
            source_model = resolve_model_file_from_production_model(production_model)
        copied_model_name = copy_to_current(source_model)
        head_config = head_config or "protot_mean_1"
        if args.prototypes_file:
            copied_prototypes_name = write_prototypes_npz(args.prototypes_file, production_model or {}, labels)
        else:
            try:
                copied_prototypes_name = write_learned_prototypes_npz(production_model or {}, labels, str(head_config), args.task)
            except Exception as e:
                print(f"[mobile-deploy] Could not regenerate learned prototypes ({e}); falling back to prototypes.pkl")
                if production_model is None:
                    production_model = get_current_production_model_from_db(task=args.task)
                source_prototypes = resolve_prototype_file_from_production_model(production_model)
                copied_prototypes_name = write_prototypes_npz(source_prototypes, production_model or {}, labels)

        manifest = create_torch_prototype_manifest(
            model_id=int(model_id),
            model_name=str(model_name),
            model_file=copied_model_name,
            prototypes_file=copied_prototypes_name,
            labels=labels,
            input_size=input_size,
            normalize=normalize,
            distance=str(args.dist_fct or _production_value(production_model, "Dist_Fct", "dist_fct", default=args.distance)),
            head_config=str(head_config),
            production_params=production_params,
        )

    else:
        raise ValueError(f"Unsupported deployment type: {deployment_type}")

    print()
    print("Created current mobile deployment:")
    print(f"  directory: {CURRENT_DIR}")
    print(f"  manifest:  {CURRENT_DIR / 'manifest.json'}")
    print()
    print(manifest)


if __name__ == "__main__":
    main()
