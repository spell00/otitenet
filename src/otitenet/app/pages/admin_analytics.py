"""Admin analytics / EDA page."""

import glob as glob_module
import traceback
from otitenet.app.utils_dataset_names import get_short_dataset_name, get_short_dataset_names

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from umap import UMAP
except ImportError:
    UMAP = None


def _run_comprehensive_eda(_args):
    """Zealous ML engineer's comprehensive EDA analysis."""
    import torch
    import seaborn as sns

    from otitenet.app.model_loading import load_model_and_prototypes
    from otitenet.data.data_getters import get_images_loaders
    from otitenet.train.train_triplet_new import TrainAE
    from otitenet.utils.utils import get_empty_traces

    with st.spinner("Loading model and data..."):
        model, _, prototypes, _, _, data, unique_labels, unique_batches, _ = load_model_and_prototypes(_args)
        train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                        log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
                        log_mlflow=False, groupkfold=_args.groupkfold)
        train.n_batches = len(unique_batches)
        train.n_cats = len(unique_labels)
        train.unique_batches = unique_batches
        train.unique_labels = unique_labels
        train.epoch = 1
        train.model = model
        train.params = {'n_neighbors': int(_args.n_neighbors)}
        train.set_arcloss()

    lists, traces = get_empty_traces()
    loaders = get_images_loaders(
        data=data, random_recs=_args.random_recs, weighted_sampler=0, is_transform=0,
        samples_weights=None, epoch=1, unique_labels=unique_labels,
        triplet_dloss=_args.dloss, bs=_args.bs, prototypes_to_use=_args.prototypes_to_use,
        prototypes=prototypes, size=_args.new_size, normalize=_args.normalize,
    )

    # Encode all sets
    with st.spinner("Encoding all samples..."):
        with torch.no_grad():
            for grp in ['train', 'valid', 'test']:
                if grp in loaders:
                    try:
                        _, lists, _ = train.loop(grp, None, 0, loaders[grp], lists, traces)
                    except:
                        pass

    # Collect data
    encs, cats, batches, raw_imgs = [], [], [], []
    grp_labels = []

    # Load raw images from data/queries if available
    import glob as glob_module
    query_imgs = sorted(glob_module.glob('data/queries/*'))
    if query_imgs:
        with st.spinner("Loading raw query images..."):
            for img_path in query_imgs[:500]:  # Limit to 500
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((_args.new_size, _args.new_size))
                    img_arr = np.array(img) / 255.0
                    raw_imgs.append(img_arr)
                except:
                    pass

    for grp in ['train', 'valid', 'test']:
        if lists[grp].get('encoded_values'):
            encs.append(np.concatenate(lists[grp]['encoded_values']))
            cats.append(np.concatenate(lists[grp]['cats']))
            grp_labels.append(np.array([grp] * len(lists[grp]['encoded_values'])))

    if not encs:
        raise RuntimeError("No encodings available")

    all_encs = np.concatenate(encs)
    all_cats = np.concatenate(cats)
    all_grps = np.concatenate(grp_labels) if grp_labels else np.zeros(len(all_cats), dtype=object)
    all_raw = np.concatenate(raw_imgs) if raw_imgs else None

    # Create tabs for different analyses
    eda_tabs = st.tabs(["📈 Distributions", "🔍 Raw PCA", "🌐 t-SNE", "🎨 UMAP", "📊 Statistics", "🔗 Correlations", "🎯 Prototypes"])

    # TAB 1: Distributions
    with eda_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Class Distribution**")
            class_counts = pd.Series(all_cats).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            class_counts.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Samples per Class')
            ax.set_ylabel('Count')
            ax.set_xlabel('Class ID')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with col2:
            st.write("**Group Distribution**")
            grp_counts = pd.Series(all_grps).value_counts()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            grp_counts.plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Samples per Split')
            ax.set_ylabel('Count')
            st.pyplot(fig, use_container_width=True)

    # TAB 2: Raw PCA (on pixel data)
    with eda_tabs[1]:
        if all_raw is not None and len(all_raw) > 0:
            with st.spinner("Computing PCA on raw pixels..."):
                max_samples = min(1000, len(all_raw))
                raw_flat = all_raw[:max_samples].reshape(max_samples, -1)  # Limit to available or 1000
                raw_cats_subset = all_cats[:max_samples]
                pca_raw = PCA(n_components=2)
                raw_pca_2d = pca_raw.fit_transform(raw_flat)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                scatter = ax.scatter(raw_pca_2d[:, 0], raw_pca_2d[:, 1], c=raw_cats_subset, cmap='tab20', alpha=0.7, s=20)
                ax.set_xlabel(f'PC1 ({pca_raw.explained_variance_ratio_[0]*100:.1f}%)')
                ax.set_ylabel(f'PC2 ({pca_raw.explained_variance_ratio_[1]*100:.1f}%)')
                ax.set_title(f'Raw Pixel PCA (first {max_samples} samples)')
                plt.colorbar(scatter, ax=ax, label='Class')
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("Raw pixel data not available")

    # TAB 3: t-SNE
    with eda_tabs[2]:
        with st.spinner("Computing t-SNE (this may take a moment)..."):
            try:
                sample_size = min(500, len(all_encs))
                sample_idx = np.random.choice(len(all_encs), sample_size, replace=False)
                encs_sample = all_encs[sample_idx]
                cats_sample = all_cats[sample_idx]
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                encs_tsne = tsne.fit_transform(encs_sample)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                scatter = ax.scatter(encs_tsne[:, 0], encs_tsne[:, 1], c=cats_sample, cmap='tab20', alpha=0.7, s=20)
                ax.set_title('t-SNE of Embeddings (sampled)')
                plt.colorbar(scatter, ax=ax, label='Class')
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"t-SNE failed: {e}")

    # TAB 4: UMAP
    with eda_tabs[3]:
        if UMAP is not None:
            with st.spinner("Computing UMAP..."):
                try:
                    sample_size = min(500, len(all_encs))
                    sample_idx = np.random.choice(len(all_encs), sample_size, replace=False)
                    encs_sample = all_encs[sample_idx]
                    cats_sample = all_cats[sample_idx]
                    umap = UMAP(n_components=2, random_state=42)
                    encs_umap = umap.fit_transform(encs_sample)
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    scatter = ax.scatter(encs_umap[:, 0], encs_umap[:, 1], c=cats_sample, cmap='tab20', alpha=0.7, s=20)
                    ax.set_title('UMAP of Embeddings (sampled)')
                    plt.colorbar(scatter, ax=ax, label='Class')
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"UMAP failed: {e}")
        else:
            st.info("UMAP not installed. Install with: pip install umap-learn")

    # TAB 5: Embedding Statistics
    with eda_tabs[4]:
        stats_data = []
        for cls_id in sorted(np.unique(all_cats)):
            mask = all_cats == cls_id
            enc_subset = all_encs[mask]
            stats_data.append({
                'Class': cls_id,
                'N Samples': len(enc_subset),
                'Mean Norm': np.linalg.norm(enc_subset.mean(axis=0)),
                'Std Norm': np.linalg.norm(enc_subset.std(axis=0)),
                'Min': enc_subset.min(),
                'Max': enc_subset.max(),
                'Median': np.median(enc_subset),
            })
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

        # Embedding norm distribution
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for cls_id in sorted(np.unique(all_cats)):
            mask = all_cats == cls_id
            norms = np.linalg.norm(all_encs[mask], axis=1)
            ax.hist(norms, alpha=0.6, label=f'Class {cls_id}', bins=30)
        ax.set_xlabel('Embedding L2 Norm')
        ax.set_ylabel('Frequency')
        ax.set_title('Embedding Norm Distribution by Class')
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    # TAB 6: Feature Correlations
    with eda_tabs[5]:
        st.write("**Embedding Dimension Correlations (first 20 dims)**")
        n_dims = min(20, all_encs.shape[1])
        corr_matrix = np.corrcoef(all_encs[:, :n_dims].T)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Embedding Dimension Correlations')
        st.pyplot(fig, use_container_width=True)

    # TAB 7: Prototype Analysis
    with eda_tabs[6]:
        proto_info = []
        if isinstance(prototypes, dict):
            class_train = prototypes.get('class', {}).get('train', {})
            for cls_id, proto in class_train.items():
                proto_arr = np.asarray(proto)
                if proto_arr.ndim > 1:
                    proto_arr = proto_arr[0]
                mask = all_cats == cls_id
                enc_cls = all_encs[mask]
                if len(enc_cls) == 0:  # Skip empty classes
                    continue
                distances = np.linalg.norm(enc_cls - proto_arr, axis=1)
                if len(distances) > 0:
                    proto_info.append({
                        'Class': cls_id,
                        'Proto Norm': np.linalg.norm(proto_arr),
                        'Mean Sample Dist': distances.mean(),
                        'Std Sample Dist': distances.std(),
                        'Min Dist': distances.min(),
                        'Max Dist': distances.max(),
                    })
        if proto_info:
            proto_df = pd.DataFrame(proto_info)
            st.dataframe(proto_df, use_container_width=True)
            st.write("**Sample-to-Prototype Distances**")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for i, info in enumerate(proto_info):
                cls_id = info['Class']
                mask = all_cats == cls_id
                enc_cls = all_encs[mask]
                if len(enc_cls) == 0:
                    continue
                proto_arr = np.asarray(class_train[cls_id])
                if proto_arr.ndim > 1:
                    proto_arr = proto_arr[0]
                distances = np.linalg.norm(enc_cls - proto_arr, axis=1)
                if len(distances) > 0:
                    ax.hist(distances, alpha=0.6, label=f'Class {cls_id}', bins=30)
            ax.set_xlabel('Distance to Class Prototype')
            ax.set_ylabel('Frequency')
            ax.set_title('Sample Distances to Class Prototypes')
            ax.legend()
            st.pyplot(fig, use_container_width=True)



def render(ctx):
    args = ctx.args
    current_model_key = ctx.current_model_key or "unknown"
    st.header("📊 Admin Analytics")
    st.caption("Advanced analysis: Raw PCA, t-SNE, UMAP, distributions, embeddings stats")

    # ---- Comprehensive EDA Suite ---- #
    st.divider()
    st.subheader("📊 Comprehensive EDA Suite")
    st.caption("Advanced analysis: Raw PCA, t-SNE, UMAP, distributions, embeddings stats")

    # Reset per-run render flag
    st.session_state['tab1_eda_rendered_this_run'] = False

    # EDA Suite button
    if st.button("🧪 Run Full EDA Suite", key="tab1_run_eda"):
        try:
            _run_comprehensive_eda(args)
            st.session_state['tab1_eda_model_key'] = current_model_key
            st.session_state['tab1_eda_keep'] = True
            st.session_state['tab1_eda_rendered_this_run'] = True
        except Exception as e:
            st.error(f"EDA failed: {e}")
            import traceback
            st.error(traceback.format_exc())

    # Auto-display EDA if previously run for this model and pinned
    if st.session_state.get('tab1_eda_model_key') == current_model_key:
        keep = st.checkbox(
            "📌 Pin EDA results",
            value=st.session_state.get('tab1_eda_keep', True),
            key="tab1_keep_eda_checkbox",
            help="Keep EDA visible across other interactions."
        )
        st.session_state['tab1_eda_keep'] = keep
        if keep:
            st.caption("💾 EDA will re-display after other actions")
            if not st.session_state.get('tab1_eda_rendered_this_run', False):
                try:
                    _run_comprehensive_eda(args)
                    st.session_state['tab1_eda_rendered_this_run'] = True
                except Exception as e:
                    st.error(f"EDA failed: {e}")
                    import traceback
                    st.error(traceback.format_exc())



