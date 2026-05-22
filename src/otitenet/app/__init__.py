"""
Otite application package.

Submodules are intentionally not imported at package import time. Some admin
modules depend on Torch, sklearn, MySQL, or plotting libraries, while client and
offline code may only need a small utility module.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    # Utils
    "strip_extension": "otitenet.app.utils",
    "ensure_int": "otitenet.app.utils",
    "get_model_params_path": "otitenet.app.utils",
    "extract_params_from_log_path": "otitenet.app.utils",
    "build_params_from_args": "otitenet.app.utils",
    "get_calibration_metrics": "otitenet.app.utils",
    "_make_model_selection_key": "otitenet.app.utils",
    "_lookup_model_number": "otitenet.app.utils",
    "_ensure_model_number_map": "otitenet.app.utils",
    "_unique_preserve_order": "otitenet.app.utils",
    # Database
    "get_db_connection": "otitenet.app.database",
    "create_db": "otitenet.app.database",
    "ensure_results_model_id": "otitenet.app.database",
    "ensure_best_models_registry_nsize": "otitenet.app.database",
    "check_ds_exists": "otitenet.app.database",
    "list_image_results": "otitenet.app.database",
    "fetch_model_by_log_path": "otitenet.app.database",
    "resolve_model_id": "otitenet.app.database",
    "insert_score": "otitenet.app.database",
    # Image processing
    "get_image": "otitenet.app.image_processing",
    "preprocess_image": "otitenet.app.image_processing",
    "get_image_arrays": "otitenet.app.image_processing",
    "preprocess_image_array": "otitenet.app.image_processing",
    "image_to_chw_array": "otitenet.app.image_processing",
    # Model loading
    "resolve_model_paths": "otitenet.app.model_loading",
    "load_saved_search_params": "otitenet.app.model_loading",
    "load_model_parameters": "otitenet.app.model_loading",
    "load_model_and_prototypes": "otitenet.app.model_loading",
    "load_model_for_log_path": "otitenet.app.model_loading",
    "clear_cached_model": "otitenet.app.model_loading",
    # Inference
    "predict_label_from_prototypes": "otitenet.app.inference",
    "predict_with_prototype_distance_ratio": "otitenet.app.inference",
    "predict_with_kde": "otitenet.app.inference",
    # UI helpers
    "choose_dataset": "otitenet.app.ui_helpers",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
