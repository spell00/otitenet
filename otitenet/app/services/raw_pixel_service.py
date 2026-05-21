"""Raw-pixel classifier cache and progress helpers."""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Any, Callable

import joblib

from otitenet.app.utils import ensure_int


WarnFn = Callable[[str], None]


def _warn(warn: WarnFn | None, message: str) -> None:
    if warn is not None:
        warn(message)


def get_grid_search_param_combinations(params_dict: dict[str, str], type_map: dict[str, Callable] | None = None) -> dict[str, list[Any]]:
    """
    Convert comma-separated UI parameter strings into grid-search value lists.
    """
    grid_params = {}
    for key, value in params_dict.items():
        vals = [s.strip() for s in str(value).split(",") if s.strip()]
        if type_map and key in type_map:
            try:
                grid_params[key] = [type_map[key](x) for x in vals]
            except Exception:
                grid_params[key] = vals
        else:
            grid_params[key] = vals
    return grid_params


def raw_cache_dir(local_args: Any, root: str = "logs/raw_data") -> str:
    try:
        ds_name = os.path.basename(str(getattr(local_args, "path", "./data")).rstrip("/"))
        nsize = ensure_int(getattr(local_args, "new_size", None))
        base = os.path.join(root, ds_name)
        if nsize:
            base = os.path.join(base, f"size_{nsize}")
        os.makedirs(base, exist_ok=True)
        return base
    except Exception:
        return root


def save_baseline_models(cache_dir: str, baselines_dict: dict[str, Any], warn: WarnFn | None = None) -> None:
    """Save baseline classifiers as they become available."""
    for name, data_b in (baselines_dict or {}).items():
        try:
            clf = data_b.get("classifier") if isinstance(data_b, dict) else None
            if clf is not None:
                outp = os.path.join(cache_dir, f"baseline_{name}.joblib")
                joblib.dump(clf, outp)
        except Exception as exc:
            _warn(warn, f"Could not save baseline {name}: {exc}")


def save_knn_model(
    cache_dir: str,
    train_raw: Any,
    train_labels: Any,
    best_k_val: int | str | None,
    warn: WarnFn | None = None,
    fit_knn: Callable | None = None,
) -> str | None:
    """Fit and save the selected raw-pixel KNN model."""
    try:
        if best_k_val is not None:
            if fit_knn is None:
                from otitenet.ml import fit_knn_classifier

                fit_knn = fit_knn_classifier
            knn = fit_knn(train_raw, train_labels, n_neighbors=int(best_k_val))
            outp_knn = os.path.join(cache_dir, f"knn_k{int(best_k_val)}.joblib")
            joblib.dump(knn, outp_knn)
            return outp_knn
    except Exception as exc:
        _warn(warn, f"Could not save KNN: {exc}")
    return None


def save_raw_summary(
    cache_dir: str,
    n_aug: int | str,
    knn_results: dict[str, Any] | None,
    baseline_results: dict[str, Any] | None,
    proto_results: dict[str, Any] | None,
    batch_effects: dict[str, Any] | None = None,
    warn: WarnFn | None = None,
    now_fn: Callable[[], datetime] = datetime.now,
) -> None:
    """Save raw-pixel results under a given n_aug key."""
    try:
        pkl = os.path.join(cache_dir, "raw_results.pkl")

        if os.path.exists(pkl):
            with open(pkl, "rb") as fh:
                full_cache = pickle.load(fh)
        else:
            full_cache = {}

        # Backward compatibility: if old cache format is flat, start a new keyed cache.
        if not isinstance(full_cache, dict) or any(k in full_cache for k in ["knn", "baselines", "prototypes"]):
            full_cache = {}

        full_cache[int(n_aug)] = {
            "timestamp": now_fn().isoformat(),
            "n_aug": int(n_aug),
            "knn": knn_results or {},
            "baselines": {k: {"mcc": v.get("mcc")} for k, v in (baseline_results or {}).items()},
            "prototypes": proto_results or {},
            "batch_effects": batch_effects or {},
        }

        with open(pkl, "wb") as fh:
            pickle.dump(full_cache, fh)

    except Exception as exc:
        _warn(warn, f"Could not save raw results summary: {exc}")


def get_progress_file(cache_dir: str) -> str:
    """Path to raw-pixel progress checkpoint file."""
    return os.path.join(cache_dir, ".progress.pkl")


def save_progress(cache_dir: str, stage: str, data: dict[str, Any]) -> None:
    """Save progress checkpoint for resumable computation."""
    try:
        with open(get_progress_file(cache_dir), "wb") as fh:
            pickle.dump({"stage": stage, **data}, fh)
    except Exception:
        pass


def clear_progress(cache_dir: str) -> None:
    """Clear progress checkpoint when computation completes."""
    try:
        prog_file = get_progress_file(cache_dir)
        if os.path.exists(prog_file):
            os.remove(prog_file)
    except Exception:
        pass


def load_raw_results(cache_dir: str, warn: WarnFn | None = None) -> dict[Any, Any]:
    try:
        pkl = os.path.join(cache_dir, "raw_results.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as fh:
                cache = pickle.load(fh)

            # Convert legacy flat format to keyed n_aug cache.
            if isinstance(cache, dict) and any(k in cache for k in ["knn", "baselines", "prototypes"]):
                cache = {
                    0: {
                        "timestamp": "legacy",
                        "n_aug": 0,
                        "knn": cache.get("knn", {}),
                        "baselines": cache.get("baselines", {}),
                        "prototypes": cache.get("prototypes", {}),
                        "batch_effects": cache.get("batch_effects", {}),
                    }
                }
            return cache
    except Exception as exc:
        _warn(warn, f"Could not load raw results cache: {exc}")
    return {}


def raw_model_mcc(result: dict[str, Any], model_type: str, use_knn_best_fallback: bool = True) -> float | None:
    """Extract a comparable MCC value for one raw-pixel model result."""
    if not isinstance(result, dict):
        return None

    if model_type == "knn":
        knn_data = result.get("knn", {})
        mcc_list = knn_data.get("mcc_per_k", []) if isinstance(knn_data, dict) else []
        valid_mccs = []
        for item in mcc_list:
            try:
                if isinstance(item, dict) and item.get("valid_mcc") is not None:
                    valid_mccs.append(float(item.get("valid_mcc")))
                elif not isinstance(item, dict) and item is not None:
                    valid_mccs.append(float(item))
            except (TypeError, ValueError):
                continue
        if valid_mccs:
            return max(valid_mccs)
        if use_knn_best_fallback and isinstance(knn_data, dict) and knn_data.get("best_mcc") is not None:
            try:
                return float(knn_data.get("best_mcc"))
            except (TypeError, ValueError):
                return None
        return None

    if model_type in ["mean", "kmeans", "gmm"]:
        proto_data = result.get("prototypes", {})
        strat_data = proto_data.get(model_type, {}) if isinstance(proto_data, dict) else {}
        val = strat_data.get("best_mcc") if isinstance(strat_data, dict) else None
    else:
        baseline_data = result.get("baselines", {})
        b_data = baseline_data.get(model_type, {}) if isinstance(baseline_data, dict) else {}
        val = b_data.get("mcc") if isinstance(b_data, dict) else None

    if isinstance(val, (int, float)):
        return float(val)
    return None


def best_raw_mcc_by_model(cache: dict[Any, Any], model_types: list[str]) -> tuple[list[dict[str, Any]], bool]:
    """Return best MCC and n_aug per model type across the raw-pixel cache."""
    rows = []
    has_valid_mcc = False
    for model_type in model_types:
        best_mcc = -1.0
        best_n_aug = None
        for n_aug, result in (cache or {}).items():
            mcc = raw_model_mcc(result, model_type, use_knn_best_fallback=True)
            if mcc is not None:
                has_valid_mcc = True
            if mcc is not None and mcc > best_mcc:
                best_mcc = mcc
                best_n_aug = n_aug
        rows.append({
            "model_type": model_type,
            "mcc": best_mcc if best_mcc > -1 else 0,
            "n_aug": best_n_aug,
        })
    return rows, has_valid_mcc


def raw_mccs_for_n_aug(result: dict[str, Any], model_types: list[str]) -> tuple[list[float], bool]:
    """Return MCC values for all model types for a single n_aug result."""
    mccs = []
    has_valid_mcc = False
    for model_type in model_types:
        mcc = raw_model_mcc(result, model_type, use_knn_best_fallback=False)
        if mcc is not None:
            has_valid_mcc = True
        mccs.append(mcc if mcc is not None else 0)
    return mccs, has_valid_mcc
