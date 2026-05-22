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
    render_current_optimization_sidebar,
    render_person_sidebar,
    require_login,
)
from otitenet.app.context import AppContext
from otitenet.app.navigation import create_tabs
from otitenet.app.services.production_model_service import apply_production_model_to_args
from otitenet.app.pages import (
    admin_analytics,
    ensemble,
    gradcam_gallery,
    inference_results,
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
load_production_model(cursor)

st.title("Ear Health Classifier with SHAP 👂")


# -------------------------------------------------
# 2️⃣ Sidebar
# -------------------------------------------------
selected_person_id = render_person_sidebar(conn, cursor)
render_current_optimization_sidebar(is_admin)

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
tabs = create_tabs(is_admin)

if is_admin:
    (
        tab_leaderboard,
        tab_learned_embedding,
        tab_past_results,
        tab_new_analysis,
        tab_ensemble,
        tab_gradcam_gallery,
        tab_raw_pixel,
        tab_admin_analytics,
        tab_inference_results,
    ) = tabs
else:
    (
        tab_past_results,
        tab_new_analysis,
        tab_inference_results,
    ) = tabs


# -------------------------------------------------
# 5️⃣ Render pages
# -------------------------------------------------
if is_admin:
    with tab_leaderboard:
        leaderboard.render(ctx)

    with tab_learned_embedding:
        learned_embedding.render(ctx)

with tab_past_results:
    past_results.render(ctx)

with tab_new_analysis:
    new_analysis.render(ctx)

if is_admin:
    with tab_ensemble:
        ensemble.render(ctx)

    with tab_gradcam_gallery:
        gradcam_gallery.render(ctx)

    with tab_raw_pixel:
        raw_pixel_classification.render(ctx)

    with tab_admin_analytics:
        admin_analytics.render(ctx)

with tab_inference_results:
    if selected_person_id is None:
        st.warning(
            "Please create or select a family member from the sidebar before viewing inference results."
        )
    else:
        inference_results.render(ctx)
