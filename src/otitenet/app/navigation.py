import streamlit as st


ADMIN_PAGES = [
    ("leaderboard", "🏆 Best Models"),
    ("learned_embedding", "🧠 Embedding Classif"),
    ("gradcam_gallery", "🖼️ Grad-CAM"),
    ("new_analysis", "🧪 New Analysis"),
    ("ensemble", "👥 Ensemble"),
    ("past_results", "📊 Past Results"),
    ("raw_pixel", "🔬 Raw Pixel"),
    ("admin_analytics", "📈 Analytics"),
    ("inference_results", "🧾 Inference"),
]

CLIENT_PAGES = [
    ("past_results", "🖼️ Grad-CAM Gallery"),
    ("new_analysis", "🧪 New Analysis"),
    ("inference_results", "🧾 Inference Results"),
]


def select_page(is_admin: bool) -> str:
    """Render persistent top-level navigation and return the selected page id."""

    pages = ADMIN_PAGES if is_admin else CLIENT_PAGES
    page_ids = [page_id for page_id, _label in pages]
    page_labels = {page_id: label for page_id, label in pages}

    current_page = st.session_state.get("active_page_id")
    if current_page not in page_ids:
        current_page = page_ids[0]
        st.session_state["active_page_id"] = current_page

    selected_page = st.segmented_control(
        "Navigation",
        options=page_ids,
        format_func=lambda page_id: page_labels.get(page_id, str(page_id)),
        key="active_page_id",
        label_visibility="collapsed",
    )
    return selected_page or current_page


def create_tabs(is_admin: bool):
    """
    Create app tabs.

    Deprecated for top-level app navigation. Prefer ``select_page()``, which keeps
    the selected page in session state across Streamlit reruns.

    Admin gets the full app.
    Client/non-admin gets only the safe user-facing tabs.

    Returns
    -------
    tuple
        If admin:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9
        Else:
            tab3, tab4, tab9
    """

    if is_admin:
        return st.tabs(
            [
                "🏆 Best Models",
                "🧠 Embedding Classif",
                "🖼️ Grad-CAM",
                "🧪 New Analysis",
                "👥 Ensemble",
                "📊 Past Results",
                "🔬 Raw Pixel",
                "📈 Analytics",
                "🧾 Inference",
            ]
        )

    return st.tabs(
        [
            "🖼️ Grad-CAM Gallery",
            "🧪 New Analysis",
            "🧾 Inference Results",
        ]
    )
