# Otite Application Module Structure

This package provides a well-organized, domain-specific module structure for the Otite ear health classifier Streamlit application.

## Module Organization

### `utils.py` - Core Utilities
String manipulation, path construction, and model selection helpers:
- `strip_extension()` - Remove file extensions
- `ensure_int()` - Safe integer conversion
- `get_model_params_path()` - Build model parameter paths
- `extract_params_from_log_path()` - Parse parameters from log paths
- `build_params_from_args()` - Convert args namespace to parameter dict
- `get_calibration_metrics()` - Cached calibration metric computation
- `_make_model_selection_key()` - Generate unique keys for model selection
- `_lookup_model_number()` - Resolve model numbers with fuzzy matching
- `_ensure_model_number_map()` - Initialize model number mapping from database
- `_unique_preserve_order()` - Deduplicate lists while preserving order

### `database.py` - MySQL Database Operations
All database interactions including connection management, queries, and migrations:
- `get_db_connection()` - Cached MySQL connection
- `create_db()` - Initialize database with retry logic
- `ensure_results_model_id()` - Schema migration for model_id column
- `ensure_best_models_registry_nsize()` - Schema migration for nsize column
- `check_ds_exists()` - Check if analysis result exists
- `list_image_results()` - Retrieve all analyses for an image
- `fetch_model_by_log_path()` - Lookup model by log path
- `resolve_model_id()` - Multi-strategy model ID resolution
- `insert_score()` - Upsert analysis results and usage summary

### `image_processing.py` - Image Utilities
Image loading, preprocessing, and tensor conversions:
- `get_image()` - Load and preprocess image from disk
- `preprocess_image()` - Prepare PIL image for model inference

### `model_loading.py` - Model Management
Model and prototype loading with caching and fallback logic:
- `resolve_model_paths()` - Find model files with dist_fct/k fallbacks
- `load_saved_search_params()` - Load hyperparameter search configuration
- `load_model_parameters()` - Load training parameters from JSON
- `load_model_and_prototypes()` - Main model loading function with OOM handling
- `load_model_for_log_path()` - Ensemble-specific model loading
- `clear_cached_model()` - Clear model resource caches

### `inference.py` - Prediction Strategies
Multiple classification approaches:
- `predict_label_from_prototypes()` - Nearest prototype classification (Euclidean/Cosine)
- `predict_with_prototype_distance_ratio()` - Inverse distance ratio prediction
- `predict_with_kde()` - Kernel Density Estimation classification

### `ui_helpers.py` - Streamlit UI Components
Reusable UI widgets and session state management:
- `choose_dataset()` - Dataset selector with state synchronization

### `context.py` - Runtime Context
Small dataclass used to pass shared Streamlit runtime objects into page modules:
- `AppContext` - connection, cursor, args, role, data directory, device, and production model

### `bootstrap.py` - Startup Flow
Database/session initialization that used to live directly in root `app.py`:
- `initialize_database()` - connect and run required schema checks
- `initialize_model_ranks_once()` - refresh model ranks once per session
- `initialize_user_state()` - initialize user-related session keys
- `load_production_model()` - load the admin-selected production model

### `navigation.py` - Tab Routing
Central tab labels and tab creation:
- `create_tabs()` - returns admin or client tabs from one shared definition

### `display_metrics.py` - Display and Inference Metric Helpers
Helpers that were previously defined in root `app.py`:
- `_arrow_safe_dataframe()` - Normalize dataframes for Streamlit/Arrow display
- `_inference_ground_truth_global()` - Resolve inference labels from the training log workbook
- `_metrics_from_binary_predictions_global()` - Compute inference ACC/MCC/AUC summaries
- `_compute_batch_effect_from_predictions()` - Batch-mixing diagnostics for predicted labels
- baseline display name/color constants

### `services/` - Application Workflow Services
Workflow logic that sits between Streamlit pages and lower-level modules:
- `production_model_service.py` - Apply the admin-selected production model to client inference args
- `optimization_service.py` - KNN/prototype/baseline optimization, prototype caching, and optimization cache management

### `pages/`, `components/`, `repositories/`
Scaffolded package boundaries for the next refactor steps:
- `pages/` - one Streamlit tab/page per module
- `components/` - reusable widgets and display blocks
- `repositories/` - SQL/database access split out of page code

Current extracted pages:
- `pages/gradcam_gallery.py` - admin Grad-CAM gallery tab

## Usage

Import functions directly from the package:

```python
# Database operations
from otitenet.app import create_db, insert_score, check_ds_exists

# Model loading
from otitenet.app import load_model_and_prototypes, resolve_model_paths

# Image processing
from otitenet.app import get_image, preprocess_image

# Inference
from otitenet.app import predict_label_from_prototypes, predict_with_kde

# Utilities
from otitenet.app import get_model_params_path, extract_params_from_log_path
```

## Benefits

1. **Domain Separation**: Each module handles a specific domain (database, models, inference, etc.)
2. **Code Reusability**: Functions can be imported and used across different parts of the application
3. **Easier Testing**: Isolated modules are easier to unit test
4. **Better Maintainability**: Related functions are grouped together logically
5. **Clear Dependencies**: Import structure makes dependencies explicit
6. **Reduced Coupling**: App.py is now ~500 lines lighter and focused on UI/orchestration

## Migration from app.py

The following functions were moved from the monolithic `app.py` to organized modules:

- **300+ lines** moved to `database.py` (10 functions)
- **70 lines** moved to `image_processing.py` (2 functions)
- **250+ lines** moved to `model_loading.py` (6 functions)
- **170 lines** moved to `inference.py` (3 functions)
- **50 lines** moved to `ui_helpers.py` (1 function)
- **400 lines** already in `utils.py` (10 functions)

**Total reduction**: ~1200+ lines removed from app.py and organized into domain-specific modules.

## Development Guidelines

### Adding New Database Functions
Place in `database.py` and ensure they:
- Accept `cursor` and optionally `conn` parameters
- Handle exceptions gracefully
- Use parameterized queries to prevent SQL injection
- Update `__init__.py` exports

### Adding New Model Functions
Place in `model_loading.py` and ensure they:
- Use `@st.cache_resource` for expensive operations
- Handle GPU OOM errors gracefully with CPU fallback
- Return consistent data structures
- Update `__init__.py` exports

### Adding New Inference Functions
Place in `inference.py` and ensure they:
- Accept torch tensors as input
- Return consistent tuple formats (label, confidence)
- Handle edge cases (None prototypes, empty arrays, etc.)
- Document expected input/output formats
- Update `__init__.py` exports

## File Structure

```
otitenet/app/
├── __init__.py           # Package exports
├── README.md             # This file
├── context.py            # Shared AppContext dataclass
├── bootstrap.py          # Startup/session/database initialization
├── navigation.py         # Streamlit tab labels and tab creation
├── display_metrics.py    # Display helpers and inference metric helpers
├── utils.py              # Core utilities and leaderboard/model-selection helpers
├── database.py           # Database operations
├── image_processing.py   # Image utilities (70 lines)
├── model_loading.py      # Model loading (270 lines)
├── inference.py          # Prediction strategies (170 lines)
├── ui_helpers.py         # UI components
├── services/
│   ├── optimization_service.py
│   └── production_model_service.py
├── pages/
│   └── gradcam_gallery.py
├── components/
│   └── account_sidebar.py
└── repositories/
```

The root `app.py` should continue shrinking toward a small shell that handles
bootstrap, auth, sidebar setup, and tab routing.
