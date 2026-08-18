#!/usr/bin/env python3
"""Optuna-tune classifier heads on cached fresh Siamese embeddings.

For every requested inference fraction this script inventories the three
canonical fresh result files:

* INF_FRAC_FRESH_OPTUNA_CNN_MLP_...
* INF_FRAC_FRESH_OPTUNA_SIAMESE_...
* INF_FRAC_FRESH_SIAMESE_...

The two Siamese CSVs are merged and deduplicated by UUID.  The CNN/MLP CSV is
kept as the comparison baseline; CNN/MLP checkpoints do not expose the cached
train/valid/test embeddings required to refit sklearn heads.

Each (UUID, head family, n_aug) gets an independent Optuna study.  Selection is
made only on validation MCC.  Test metrics are computed once using the best
validation parameters.  Here n_aug is embedding-space augmentation: 0 keeps
the original training set, 1/2 append one/two deterministic noisy copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path("/home/simon/otitenet")
TASK = "four_classes_220726"
STUDY_SCHEMA_VERSION = "v2_sample_aligned_targets"
PROGRESSES_DIR = PROJECT_ROOT / "logs" / "progresses" / TASK
RUNS_DIR = PROJECT_ROOT / "logs" / TASK
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper_outputs" / "siamese_head_optuna"
FRACTIONS = ["0p5", "0p25", "0p1", "0p05", "0p02"]
HEADS = [
    "random_forest",
    "extra_trees",
    "logreg",
    "linear_svc",
    "ridge",
    "knn",
    "gaussian_nb",
    "prototype",
]


def canonical_csv_paths(fraction: str) -> dict[str, Path]:
    """Return the three exact canonical paths requested for a fraction."""
    parent = PROGRESSES_DIR / (
        "home_simon_otitenet_data_otite_ds_64_USA_Turquie_Chili_GMFUNL_"
        f"inference_fraction_hist_v2_train{fraction}_seed42"
    )
    suffix = f"P{fraction}_S42_{TASK}_completed_runs_metrics.csv"
    return {
        "cnn_mlp_optuna": parent / f"INF_FRAC_FRESH_OPTUNA_CNN_MLP_{suffix}",
        "siamese_optuna": parent / f"INF_FRAC_FRESH_OPTUNA_SIAMESE_{suffix}",
        "siamese_fresh": parent / f"INF_FRAC_FRESH_SIAMESE_{suffix}",
    }


def read_source(path: Path, source_kind: str) -> pd.DataFrame:
    if not path.exists():
        print(f"  MISSING [{source_kind}] {path}", flush=True)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["source_kind"] = source_kind
    frame["source_csv"] = str(path)
    print(f"  FOUND   [{source_kind}] rows={len(frame):4d} {path}", flush=True)
    return frame


def load_fraction_sources(fraction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CNN baseline plus the union of both Siamese result families."""
    paths = canonical_csv_paths(fraction)
    cnn = read_source(paths["cnn_mlp_optuna"], "cnn_mlp_optuna")
    siamese_parts = [
        read_source(paths["siamese_optuna"], "siamese_optuna"),
        read_source(paths["siamese_fresh"], "siamese_fresh"),
    ]
    siamese_parts = [part for part in siamese_parts if not part.empty]
    if not siamese_parts:
        return cnn, pd.DataFrame()

    siamese = pd.concat(siamese_parts, ignore_index=True)
    if "kind" in siamese:
        siamese = siamese[siamese["kind"].fillna("siamese") == "siamese"]
    siamese = siamese.dropna(subset=["uuid", "valid_mcc"]).copy()
    siamese["valid_mcc"] = pd.to_numeric(siamese["valid_mcc"], errors="coerce")
    siamese = siamese.dropna(subset=["valid_mcc"])
    # If a UUID appears in both files, retain the row with the strongest
    # recorded validation MCC while preserving its source provenance.
    siamese = (
        siamese.sort_values("valid_mcc", ascending=False)
        .drop_duplicates(subset=["uuid"], keep="first")
        .reset_index(drop=True)
    )
    return cnn, siamese


def load_encodings(uuid: str) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    run_dir = RUNS_DIR / uuid
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "valid", "test"):
        path = run_dir / f"{split}_encodings.npz"
        if not path.exists():
            print(f"    SKIP uuid={uuid}: missing {path}", flush=True)
            return None
        with np.load(path, allow_pickle=True) as data:
            embeddings = np.asarray(data["embeddings"])
            targets = None
            target_key = None
            # Artifact schemas differ between runs.  `labels` can either be a
            # sample-aligned label vector or only the four class names; `cats`
            # is consistently the encoded per-sample target when present.
            for key in ("cats", "labels"):
                if key in data and len(data[key]) == len(embeddings):
                    targets = np.asarray(data[key])
                    target_key = key
                    break
            if targets is None:
                shapes = {key: np.asarray(data[key]).shape for key in data.files}
                print(
                    f"    SKIP uuid={uuid}: no sample-aligned target in {path}; shapes={shapes}",
                    flush=True,
                )
                return None
            print(
                f"    LOAD uuid={uuid[:8]} split={split} n={len(embeddings)} target={target_key}",
                flush=True,
            )
            loaded[split] = (embeddings, targets)
    return loaded


def stable_seed(*parts: object, base_seed: int) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return (int.from_bytes(digest[:4], "little") + base_seed) % (2**32 - 1)


def augment_training(
    x: np.ndarray,
    y: np.ndarray,
    n_aug: int,
    noise_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Append n_aug deterministic noisy copies; never alter valid/test data."""
    if n_aug == 0:
        return x, y
    rng = np.random.default_rng(seed)
    scale = np.std(x, axis=0, dtype=np.float64)
    scale = np.where(scale > 0, scale, 1.0)
    copies = [x]
    labels = [y]
    for _ in range(n_aug):
        noise = rng.normal(0.0, noise_scale, size=x.shape) * scale
        copies.append((x + noise).astype(x.dtype, copy=False))
        labels.append(y)
    return np.concatenate(copies), np.concatenate(labels)


class PrototypeClassifier:
    """Nearest prototype head supporting multiple prototypes per class."""

    def __init__(self, strategy: str, components: int, metric: str, seed: int):
        self.strategy = strategy
        self.components = components
        self.metric = metric
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PrototypeClassifier":
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture

        vectors: list[np.ndarray] = []
        labels: list[int] = []
        for label in np.unique(y):
            class_x = x[y == label]
            count = min(self.components, len(class_x))
            if self.strategy == "mean" or count == 1:
                class_vectors = np.mean(class_x, axis=0, keepdims=True)
            elif self.strategy == "kmeans":
                class_vectors = KMeans(
                    n_clusters=count, n_init=5, random_state=self.seed
                ).fit(class_x).cluster_centers_
            else:
                class_vectors = GaussianMixture(
                    n_components=count, covariance_type="diag", random_state=self.seed
                ).fit(class_x).means_
            vectors.extend(class_vectors)
            labels.extend([int(label)] * len(class_vectors))
        self.vectors_ = np.asarray(vectors)
        self.labels_ = np.asarray(labels)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
            p_norm = self.vectors_ / (
                np.linalg.norm(self.vectors_, axis=1, keepdims=True) + 1e-12
            )
            nearest = np.argmax(x_norm @ p_norm.T, axis=1)
        else:
            nearest = np.argmin(
                np.linalg.norm(x[:, None, :] - self.vectors_[None, :, :], axis=2),
                axis=1,
            )
        return self.labels_[nearest]


def suggest_model(trial: optuna.Trial, head: str, seed: int, n_train: int) -> Any:
    if head == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
            max_depth=trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight=trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
            n_jobs=-1,
            random_state=seed,
        )
    if head == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
            max_depth=trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            n_jobs=-1,
            random_state=seed,
        )
    if head == "logreg":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-4, 1e3, log=True),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            tol=trial.suggest_float("tol", 1e-6, 1e-2, log=True),
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
    if head == "linear_svc":
        return LinearSVC(
            C=trial.suggest_float("C", 1e-4, 1e3, log=True),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            tol=trial.suggest_float("tol", 1e-6, 1e-2, log=True),
            max_iter=10000,
            random_state=seed,
        )
    if head == "ridge":
        return RidgeClassifier(
            alpha=trial.suggest_float("alpha", 1e-4, 1e3, log=True),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            tol=trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        )
    if head == "knn":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 1, min(50, n_train)),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"]),
            n_jobs=-1,
        )
    if head == "gaussian_nb":
        return GaussianNB(var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-5, log=True))
    if head == "prototype":
        strategy = trial.suggest_categorical("strategy", ["mean", "kmeans", "gmm"])
        components = 1 if strategy == "mean" else trial.suggest_int("components", 1, 5)
        return PrototypeClassifier(
            strategy=strategy,
            components=components,
            metric=trial.suggest_categorical("metric", ["euclidean", "cosine"]),
            seed=seed,
        )
    raise ValueError(f"Unsupported head: {head}")


def prepare_arrays(encodings: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    train_x, train_y_raw = encodings["train"]
    valid_x, valid_y_raw = encodings["valid"]
    test_x, test_y_raw = encodings["test"]
    encoder = LabelEncoder().fit(np.concatenate([train_y_raw, valid_y_raw, test_y_raw]))
    # Scaling is fitted on train only and reused for valid/test.  It materially
    # helps distance/linear heads and does not leak validation/test statistics.
    scaler = StandardScaler().fit(train_x)
    return {
        "train_x": scaler.transform(train_x).astype(np.float32),
        "valid_x": scaler.transform(valid_x).astype(np.float32),
        "test_x": scaler.transform(test_x).astype(np.float32),
        "train_y": encoder.transform(train_y_raw),
        "valid_y": encoder.transform(valid_y_raw),
        "test_y": encoder.transform(test_y_raw),
    }


def tune_study(
    *,
    fraction: str,
    uuid: str,
    head: str,
    n_aug: int,
    arrays: dict[str, np.ndarray],
    n_trials: int,
    noise_scale: float,
    seed: int,
    storage: str | None,
    timeout: float | None,
    models_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aug_seed = stable_seed(uuid, n_aug, "augmentation", base_seed=seed)
    train_x, train_y = augment_training(
        arrays["train_x"], arrays["train_y"], n_aug, noise_scale, aug_seed
    )
    # The schema token prevents resuming studies produced by older loaders
    # which incorrectly treated the four class names as per-sample targets.
    study_name = (
        f"heads_{STUDY_SCHEMA_VERSION}_{TASK}_{fraction}_{uuid}_{head}_naug{n_aug}"
    )
    sampler = optuna.samplers.TPESampler(seed=stable_seed(study_name, base_seed=seed))
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    newly_run_trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        trial_seed = stable_seed(uuid, head, n_aug, trial.number, base_seed=seed)
        started = time.perf_counter()
        print(
            f"      FIT fraction={fraction} uuid={uuid[:8]} head={head} "
            f"n_aug={n_aug} trial={trial.number + 1}/{n_trials}",
            flush=True,
        )
        try:
            model = suggest_model(trial, head, trial_seed, len(train_x))
            model.fit(train_x, train_y)
            pred = model.predict(arrays["valid_x"])
            score = float(matthews_corrcoef(arrays["valid_y"], pred))
            status = "complete"
            error = None
        except Exception as exc:
            score = None
            status = "failed"
            error = repr(exc)
            print(f"        ERROR {error}", flush=True)
        elapsed = time.perf_counter() - started
        params = dict(trial.params)
        newly_run_trial_rows.append(
            {
                "fraction": fraction,
                "uuid": uuid,
                "head": head,
                "n_aug": n_aug,
                "trial_number": trial.number,
                "valid_mcc": score,
                "duration_seconds": elapsed,
                "status": status,
                "error": error,
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
        print(
            f"        DONE valid_mcc={score if score is not None else 'FAILED'} "
            f"seconds={elapsed:.2f} params={params}",
            flush=True,
        )
        if error is not None:
            raise RuntimeError(error)
        return score

    remaining = max(0, n_trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=timeout, catch=(Exception,))
    else:
        print(f"      RESUME study already has {len(study.trials)} trials", flush=True)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"Study {study_name} produced no completed trials")
    best = study.best_trial
    # Recreate the exact best model without asking Optuna for new values.
    final_trial = optuna.trial.FixedTrial(best.params, number=best.number)
    final_seed = stable_seed(uuid, head, n_aug, best.number, base_seed=seed)
    final_model = suggest_model(final_trial, head, final_seed, len(train_x))
    final_model.fit(train_x, train_y)
    valid_pred = final_model.predict(arrays["valid_x"])
    test_pred = final_model.predict(arrays["test_x"])
    model_path = models_dir / fraction / uuid / f"{head}_naug{n_aug}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(final_model, model_path)
        saved_model = str(model_path)
    except Exception as exc:
        # Locally defined/custom prototype estimators may not be serializable in
        # every joblib version; metrics and parameters remain fully recorded.
        saved_model = None
        print(f"      WARN could not save model: {exc!r}", flush=True)
    result = {
        "fraction": fraction,
        "uuid": uuid,
        "head": head,
        "n_aug": n_aug,
        "n_trials_total": len(study.trials),
        "best_trial_number": best.number,
        "best_params_json": json.dumps(best.params, sort_keys=True),
        "valid_mcc": float(matthews_corrcoef(arrays["valid_y"], valid_pred)),
        "valid_accuracy": float(accuracy_score(arrays["valid_y"], valid_pred)),
        "test_mcc": float(matthews_corrcoef(arrays["test_y"], test_pred)),
        "test_accuracy": float(accuracy_score(arrays["test_y"], test_pred)),
        "model_path": saved_model,
        "study_name": study_name,
    }
    print(
        f"      BEST head={head} n_aug={n_aug} valid_mcc={result['valid_mcc']:.6f} "
        f"test_mcc={result['test_mcc']:.6f}",
        flush=True,
    )
    # Export the complete persisted study, including trials from earlier runs.
    trial_rows = []
    for trial in study.trials:
        trial_rows.append(
            {
                "fraction": fraction,
                "uuid": uuid,
                "head": head,
                "n_aug": n_aug,
                "trial_number": trial.number,
                "valid_mcc": trial.value,
                "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
                "status": trial.state.name.lower(),
                "error": None,
                "params_json": json.dumps(trial.params, sort_keys=True),
                "ran_in_current_process": any(
                    row["trial_number"] == trial.number for row in newly_run_trial_rows
                ),
            }
        )
    return result, trial_rows


def best_cnn_baseline(fraction: str, cnn: pd.DataFrame) -> dict[str, Any] | None:
    if cnn.empty:
        return None
    if "kind" in cnn:
        cnn = cnn[cnn["kind"] == "cnn_mlp"]
    cnn = cnn.dropna(subset=["valid_mcc"])
    if cnn.empty:
        return None
    row = cnn.loc[pd.to_numeric(cnn["valid_mcc"]).idxmax()]
    return {
        "fraction": fraction,
        "model_type": "cnn_mlp_baseline",
        "uuid": row.get("uuid"),
        "head": row.get("model_name", row.get("model", "cnn_mlp")),
        "n_aug": None,
        "valid_mcc": row.get("valid_mcc"),
        "valid_accuracy": row.get("valid_accuracy"),
        "test_mcc": row.get("test_mcc"),
        "test_accuracy": row.get("test_accuracy"),
        "source_csv": row.get("source_csv"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", nargs="+", default=FRACTIONS)
    parser.add_argument("--top-n", type=int, default=5, help="Merged top UUIDs per fraction; 0 means all")
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=HEADS)
    parser.add_argument("--n-aug", nargs="+", type=int, choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument("--n-trials", type=int, default=20, help="Trials per UUID/head/n_aug study")
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=None, help="Optional seconds per study")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--storage", default=None, help="Optuna storage URL; default is output-dir/optuna.db")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{args.output_dir / 'optuna.db'}"
    print("=" * 88)
    print("REAL OPTUNA HPO FOR FRESH SIAMESE HEADS")
    print(f"fractions={args.fractions}")
    print(f"heads={args.heads}")
    print(f"n_aug={args.n_aug}; n_trials={args.n_trials}; top_n={args.top_n}")
    print(f"storage={storage}")
    print("Selection metric: validation MCC. Test is evaluated after selection.")
    print("=" * 88, flush=True)

    best_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for fraction in args.fractions:
        print(f"\nFRACTION {fraction}: canonical source inventory", flush=True)
        paths = canonical_csv_paths(fraction)
        for source_kind, path in paths.items():
            source_rows.append(
                {"fraction": fraction, "source_kind": source_kind, "path": str(path), "exists": path.exists()}
            )
        cnn, siamese = load_fraction_sources(fraction)
        baseline = best_cnn_baseline(fraction, cnn)
        if baseline:
            comparison_rows.append(baseline)
        if siamese.empty:
            print(f"  SKIP no usable Siamese rows for {fraction}", flush=True)
            continue
        selected = siamese.sort_values("valid_mcc", ascending=False)
        target = len(selected) if args.top_n == 0 else min(args.top_n, len(selected))
        print(
            f"  MERGED unique_siamese_runs={len(siamese)} target_available_runs={target}",
            flush=True,
        )

        processed = 0
        for row in selected.itertuples(index=False):
            if args.top_n > 0 and processed >= args.top_n:
                break
            uuid = str(row.uuid)
            encodings = load_encodings(uuid)
            if encodings is None:
                continue
            processed += 1
            rank = processed
            arrays = prepare_arrays(encodings)
            print(
                f"\n  RUN rank={rank} uuid={uuid} source={row.source_kind} "
                f"original_valid_mcc={float(row.valid_mcc):.6f} "
                f"n_train={len(arrays['train_y'])}",
                flush=True,
            )
            for head in args.heads:
                for n_aug in args.n_aug:
                    try:
                        result, trials = tune_study(
                            fraction=fraction,
                            uuid=uuid,
                            head=head,
                            n_aug=n_aug,
                            arrays=arrays,
                            n_trials=args.n_trials,
                            noise_scale=args.noise_scale,
                            seed=args.seed,
                            storage=storage,
                            timeout=args.timeout,
                            models_dir=args.output_dir / "models",
                        )
                    except Exception as exc:
                        print(
                            f"      STUDY FAILED uuid={uuid[:8]} head={head} n_aug={n_aug}: {exc!r}",
                            flush=True,
                        )
                        continue
                    result.update(
                        {
                            "rank": rank,
                            "source_kind": row.source_kind,
                            "source_csv": row.source_csv,
                            "original_valid_mcc": float(row.valid_mcc),
                            "original_test_mcc": getattr(row, "test_mcc", None),
                        }
                    )
                    best_rows.append(result)
                    comparison_rows.append({"model_type": "siamese_tuned_head", **result})
                    trial_rows.extend(trials)
                    pd.DataFrame(best_rows).to_csv(args.output_dir / "best_heads.csv", index=False)
                    pd.DataFrame(trial_rows).to_csv(args.output_dir / "trials.csv", index=False)

    pd.DataFrame(source_rows).to_csv(args.output_dir / "source_inventory.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "comparison_with_cnn_mlp.csv", index=False)
    if best_rows:
        best = pd.DataFrame(best_rows)
        winners = best.loc[best.groupby(["fraction", "uuid"])["valid_mcc"].idxmax()]
        winners.to_csv(args.output_dir / "winner_per_siamese_run.csv", index=False)
        print("\nWINNERS SELECTED BY VALIDATION MCC")
        print(
            winners[["fraction", "uuid", "head", "n_aug", "valid_mcc", "test_mcc"]].to_string(index=False)
        )
    print(f"\nOutputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
