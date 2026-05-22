# /home/simon/otitenet/otitenet/app/navigation.py

import streamlit as st


def create_tabs(is_admin: bool):
    """
    Create app tabs.

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