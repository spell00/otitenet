"""
Otite application package.

Organized into domain-specific modules:
- utils: String/path utilities, calibration, model selection keys
- database: All MySQL database operations
- image_processing: Image loading and preprocessing
- model_loading: Model/prototype loading with caching
- inference: Prediction strategies (KNN, prototypes, KDE)
- ui_helpers: Streamlit UI components
"""

# Core utilities
from .utils import (
    strip_extension,
    ensure_int,
    get_model_params_path,
    extract_params_from_log_path,
    build_params_from_args,
    get_calibration_metrics,
    _make_model_selection_key,
    _lookup_model_number,
    _ensure_model_number_map,
    _unique_preserve_order,
)

# Database operations
from .database import (
    get_db_connection,
    create_db,
    ensure_results_model_id,
    ensure_best_models_registry_nsize,
    check_ds_exists,
    list_image_results,
    fetch_model_by_log_path,
    resolve_model_id,
    insert_score,
)

# Image processing
from .image_processing import (
    get_image,
    preprocess_image,
)

# Model loading
from .model_loading import (
    resolve_model_paths,
    load_saved_search_params,
    load_model_parameters,
    load_model_and_prototypes,
    load_model_for_log_path,
    clear_cached_model,
)

# Inference
from .inference import (
    predict_label_from_prototypes,
    predict_with_prototype_distance_ratio,
    predict_with_kde,
)

# UI helpers
from .ui_helpers import (
    choose_dataset,
)

__all__ = [
    # Utils
    'strip_extension',
    'ensure_int',
    'get_model_params_path',
    'extract_params_from_log_path',
    'build_params_from_args',
    'get_calibration_metrics',
    '_make_model_selection_key',
    '_lookup_model_number',
    '_ensure_model_number_map',
    '_unique_preserve_order',
    # Database
    'get_db_connection',
    'create_db',
    'ensure_results_model_id',
    'ensure_best_models_registry_nsize',
    'check_ds_exists',
    'list_image_results',
    'fetch_model_by_log_path',
    'resolve_model_id',
    'insert_score',
    # Image processing
    'get_image',
    'preprocess_image',
    # Model loading
    'resolve_model_paths',
    'load_saved_search_params',
    'load_model_parameters',
    'load_model_and_prototypes',
    'load_model_for_log_path',
    'clear_cached_model',
    # Inference
    'predict_label_from_prototypes',
    'predict_with_prototype_distance_ratio',
    'predict_with_kde',
    # UI helpers
    'choose_dataset',
]

