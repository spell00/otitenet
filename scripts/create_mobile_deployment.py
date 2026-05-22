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
)
DEFAULT_LABELS = "normal,notNormal"


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


def copy_to_current(source: str | Path, target_name: str | None = None) -> str:
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(f"Missing file: {source}")

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)

    target = CURRENT_DIR / (target_name or source.name)
    shutil.copy2(source, target)

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


def get_current_production_model_from_db():
    """
    Read the model selected by the web admin.

    Expected logic:
    - Admin clicks "Set as Production Model" in the web/sidebar/admin UI.
    - DB stores current production model.
    - This script reads that active production model.

    This function tries the clean database helper first. If unavailable,
    it falls back to querying common production_model table shapes.
    """
    from otitenet.app.database import create_db

    conn, cursor = create_db()

    # Preferred path if your database.py already has get_production_model()
    try:
        from otitenet.app.database import get_production_model

        model = get_production_model(cursor)

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
        choices=["torch_classifier", "knn_embedding"],
        default="torch_classifier",
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
        help="Comma-separated class labels in model output order. Default: normal,notNormal",
    )

    parser.add_argument(
        "--input-size",
        default="224,224",
        help="Input size as H,W. Default: 224,224",
    )

    # KNN embedding deployment
    parser.add_argument("--embedding-model-file", default=None)
    parser.add_argument("--reference-embeddings-file", default=None)
    parser.add_argument("--reference-labels-file", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--distance", default="cosine")

    args = parser.parse_args()

    labels = parse_labels(args.labels)
    input_size = parse_input_size(args.input_size)

    production_model = None

    if args.model_file is None or args.model_id is None or args.model_name is None:
        production_model = get_current_production_model_from_db()
        print("[mobile-deploy] Using production model from web app:")
        print(production_model)

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

    if args.deployment_type == "torch_classifier":
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
        )

    elif args.deployment_type == "knn_embedding":
        if args.embedding_model_file is None:
            if production_model is None:
                production_model = get_current_production_model_from_db()
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
            k=args.k,
            distance=args.distance,
            input_size=input_size,
        )

    else:
        raise ValueError(f"Unsupported deployment type: {args.deployment_type}")

    print()
    print("Created current mobile deployment:")
    print(f"  directory: {CURRENT_DIR}")
    print(f"  manifest:  {CURRENT_DIR / 'manifest.json'}")
    print()
    print(manifest)


if __name__ == "__main__":
    main()
