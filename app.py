from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

st.set_page_config(layout="wide")

import torch

from otitenet.app.utils import set_random_seeds
from otitenet.app.args import build_args_from_sidebar
from otitenet.app.bootstrap import (
    initialize_database,
    initialize_model_ranks_once,
    initialize_user_state,
    is_current_user_admin,
    load_production_model,
)
from otitenet.app.components.account_sidebar import (
    render_labeling_task_sidebar,
    render_current_optimization_sidebar,
    render_person_sidebar,
    require_login,
)
from otitenet.app.context import AppContext
from otitenet.app.navigation import select_page
from otitenet.app.services.production_model_service import apply_production_model_to_args
from otitenet.app.pages import (
    admin_analytics,
    ensemble,
    gradcam_gallery,
    leaderboard,
    learned_embedding,
    new_analysis,
    past_results,
    raw_pixel_classification,
)


# -------------------------------------------------
# 1️⃣ App setup
# -------------------------------------------------
set_random_seeds(1)

conn, cursor = initialize_database()
initialize_model_ranks_once()
initialize_user_state()

require_login(conn, cursor)

is_admin = is_current_user_admin()
render_labeling_task_sidebar()
load_production_model(cursor)

st.title("Ear Health Classifier with SHAP 👂")


# -------------------------------------------------
# 2️⃣ Sidebar
# -------------------------------------------------
selected_person_id = render_person_sidebar(conn, cursor)
render_current_optimization_sidebar(is_admin)

with st.sidebar.expander("Production decision settings", expanded=False):
    st.session_state["production_use_topn_ensemble"] = st.checkbox(
        "Use Top-N ensemble for analysis",
        value=bool(st.session_state.get("production_use_topn_ensemble", False)),
        key="production_use_topn_ensemble_widget",
        help="When enabled, New Analysis runs the selected Top-N ranked models and applies the ensemble decision thresholds.",
    )
    st.session_state["production_top_n_models"] = st.number_input(
        "Top-N models",
        min_value=1,
        max_value=25,
        value=int(st.session_state.get("production_top_n_models", 5)),
        step=1,
        key="production_top_n_models_widget",
    )
    st.session_state["production_selected_confidence_threshold"] = st.slider(
        "Selected model confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("production_selected_confidence_threshold", 0.0)),
        step=0.01,
        key="production_selected_confidence_threshold_widget",
    )
    st.session_state["production_ensemble_vote_threshold_pct"] = st.slider(
        "Top-N ensemble vote threshold (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("production_ensemble_vote_threshold_pct", 0.0)),
        step=1.0,
        key="production_ensemble_vote_threshold_pct_widget",
    )
    st.session_state["production_require_both_thresholds"] = st.checkbox(
        "Require both thresholds",
        value=bool(st.session_state.get("production_require_both_thresholds", False)),
        key="production_require_both_thresholds_widget",
        help="Off: a confident selected-model result is kept even if ensemble consensus is low. On: both thresholds must pass.",
    )

data_dir = "./data"

args = build_args_from_sidebar(
    cursor=cursor,
    conn=conn,
    is_admin=is_admin,
    data_dir=data_dir,
)

if not is_admin:
    args = apply_production_model_to_args(
        args,
        st.session_state.get("production_model"),
        data_dir,
    )


# -------------------------------------------------
# 3️⃣ Shared app context
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ctx = AppContext(
    conn=conn,
    cursor=cursor,
    args=args,
    is_admin=is_admin,
    data_dir=data_dir,
    device=device,
    production_model=st.session_state.get("production_model"),
    current_model_key=st.session_state.get(
        "current_model_key",
        st.session_state.get("selected_model_version", "unknown"),
    ),
    selected_person_id=selected_person_id,
    user_id=st.session_state.get("user_id"),
    user_email=st.session_state.get("user_email"),
)


# -------------------------------------------------
# 4️⃣ Navigation
# -------------------------------------------------
active_page = select_page(is_admin)


# -------------------------------------------------
# 5️⃣ Render selected page
# -------------------------------------------------
if is_admin and active_page == "leaderboard":
    leaderboard.render(ctx)

elif is_admin and active_page == "learned_embedding":
    learned_embedding.render(ctx)

elif active_page == "past_results":
    past_results.render(ctx)

elif active_page == "new_analysis":
    new_analysis.render(ctx)

elif is_admin and active_page == "ensemble":
    ensemble.render(ctx)

elif is_admin and active_page == "gradcam_gallery":
    gradcam_gallery.render(ctx)

elif is_admin and active_page == "raw_pixel":
    raw_pixel_classification.render(ctx)

elif is_admin and active_page == "admin_analytics":
    admin_analytics.render(ctx)

elif is_admin and active_page == "inference_results":
    if selected_person_id is None:
        st.warning(
            "Please create or select a family member from the sidebar before viewing inference results."
        )
    else:
        from otitenet.app.pages import inference_results
        inference_results.render(ctx)
