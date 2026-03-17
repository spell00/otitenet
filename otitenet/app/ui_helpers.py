"""
UI helper functions for Streamlit interface.

Contains reusable components for dataset selection, 
session state management, and other UI utilities.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def choose_dataset(label, datasets, default=None, key=None):
    """Return a dataset choice while keeping session_state and defaults in sync.
    
    Args:
        label: Label for the selection widget
        datasets: List of available datasets
        default: Default selection
        key: Session state key
        
    Returns:
        Selected dataset name
    """
    if len(datasets) == 0:
        st.warning(f"No datasets found.")
        return None

    # Clear stale session values that are no longer valid to avoid selectbox errors.
    if key and key in st.session_state and st.session_state[key] not in datasets:
        st.session_state.pop(key, None)

    # Prefer an existing, valid session value if present; otherwise fall back to default/first.
    current = None
    if key and key in st.session_state and st.session_state[key] in datasets:
        current = st.session_state[key]
    elif default in datasets:
        current = default
    elif datasets:
        current = datasets[0]

    index = datasets.index(current) if current in datasets else 0

    if len(datasets) <= 3:
        return st.radio(label, datasets, index=index, key=key)
    else:
        return st.selectbox(label, datasets, index=index, key=key)


def plot_knn_mcc_curves(knn_cache):
    """
    Plot MCC vs k for each n_aug value in the cache.
    Args:
        knn_cache: dict mapping n_aug values to result dicts with 'knn' key containing 'mcc_per_k'
    """
    plt.figure(figsize=(10, 6))
    has_data = False
    for n_aug_val, result in sorted(knn_cache.items()):
        knn_data = result.get('knn', {})
        mcc_per_k = knn_data.get('mcc_per_k', [])
        # Handle both dict and float formats
        k_vals = []
        mcc_vals = []
        for idx, item in enumerate(mcc_per_k):
            if isinstance(item, dict):
                k = item.get('k', idx + 1)
                mcc = item.get('valid_mcc', None)
            else:
                k = idx + 1
                mcc = item
            if mcc is not None:
                k_vals.append(k)
                mcc_vals.append(mcc)
        if k_vals and mcc_vals:
            plt.plot(k_vals, mcc_vals, marker='o', label=f'n_aug={n_aug_val}')
            has_data = True
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('MCC')
    plt.title('KNN MCC vs k for each n_aug')
    if has_data:
        plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()


def plot_prototype_mcc_curves(proto_cache, strategy_name, best_scores=None):
    """
    Plot MCC vs n_components for each n_aug value for a given prototype strategy.
    Args:
        proto_cache: dict mapping n_aug values to result dicts with 'prototypes' key
        strategy_name: 'kmeans', 'gmm', or 'mean'
        best_scores: dict mapping n_aug to (n_components, mcc) for best score (optional)
    """
    plt.figure(figsize=(10, 6))
    # Plot all curves
    all_points = []  # (n_aug, n_comp, mcc)
    has_data = False
    for n_aug_val, result in sorted(proto_cache.items()):
        proto_data = result.get('prototypes', {}).get(strategy_name, {})
        per_components = proto_data.get('per_components', [])
        n_comp_vals = []
        mcc_vals = []
        for comp_info in per_components:
            if isinstance(comp_info, dict):
                n_comp = comp_info.get('n_components', None)
                mcc = comp_info.get('mcc', None)
                if n_comp is not None and mcc is not None:
                    n_comp_vals.append(n_comp)
                    mcc_vals.append(mcc)
                    all_points.append((n_aug_val, n_comp, mcc))
        if n_comp_vals and mcc_vals:
            plt.plot(n_comp_vals, mcc_vals, marker='o', label=f'n_aug={n_aug_val}')
            has_data = True

    # Find the best among the points actually plotted (from all_points)
    if all_points:
        best_point = max(all_points, key=lambda x: x[2])
        n_aug_star, n_comp_star, mcc_star = best_point
        plt.plot(n_comp_star, mcc_star, marker='*', color='red', markersize=18, label=f'Best in plot: n_aug={n_aug_star}, n_comp={n_comp_star}')
        has_data = True

    plt.xlabel('n_components')
    plt.ylabel('MCC')
    plt.title(f'{strategy_name.upper()} Prototype MCC vs n_components for each n_aug')
    if has_data:
        plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()
