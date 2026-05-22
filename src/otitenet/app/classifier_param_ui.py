def mean_prototype_params_ui():
    st.markdown("**Mean Prototype Hyperparameters:**")
    n_prototypes = st.text_input(
        "Number of prototypes",
        value="1",
        key="mean_n_prototypes",
        help="Comma-separated values allowed for grid search."
    )
    return {'n_prototypes': n_prototypes}

def kmeans_prototype_params_ui():
    st.markdown("**KMeans Prototype Hyperparameters:**")
    n_clusters = st.text_input(
        "Number of clusters",
        value="3",
        key="kmeans_n_clusters",
        help="Comma-separated values allowed for grid search."
    )
    return {'n_clusters': n_clusters}

def gmm_prototype_params_ui():
    st.markdown("**GMM Prototype Hyperparameters:**")
    n_components = st.text_input(
        "Number of GMM components",
        value="2",
        key="gmm_n_components",
        help="Comma-separated values allowed for grid search."
    )
    return {'n_components': n_components}

def decision_tree_params_ui():
    st.markdown("**Decision Tree Hyperparameters:**")
    max_depth = st.text_input(
        "max_depth",
        value="10",
        key="dt_max_depth",
        help="Comma-separated values allowed for grid search."
    )
    min_samples_split = st.text_input(
        "min_samples_split",
        value="2",
        key="dt_min_samples_split",
        help="Comma-separated values allowed for grid search."
    )
    min_samples_leaf = st.text_input(
        "min_samples_leaf",
        value="1",
        key="dt_min_samples_leaf",
        help="Comma-separated values allowed for grid search."
    )
    return {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }

def gradient_boosting_params_ui():
    st.markdown("**Gradient Boosting Hyperparameters:**")
    n_estimators = st.text_input(
        "n_estimators",
        value="100",
        key="gb_n_estimators",
        help="Comma-separated values allowed for grid search."
    )
    learning_rate = st.text_input(
        "learning_rate",
        value="0.1",
        key="gb_learning_rate",
        help="Comma-separated values allowed for grid search."
    )
    return {'n_estimators': n_estimators, 'learning_rate': learning_rate}

def lda_params_ui():
    st.markdown("**LDA Hyperparameters:**")
    solver = st.selectbox(
        "Solver",
        ["svd", "lsqr", "eigen"],
        index=0,
        key="lda_solver",
        help="Solver to use in LDA."
    )
    return {'solver': solver}

def linear_svc_params_ui():
    st.markdown("**Linear SVC Hyperparameters:**")
    max_iter = st.text_input(
        "max_iter",
        value="100",
        key="linearsvc_max_iter",
        help="Comma-separated values allowed for grid search."
    )
    return {'max_iter': max_iter}

def svc_params_ui():
    st.markdown("**SVC Hyperparameters:**")
    kernel = st.selectbox(
        "Kernel",
        ["linear", "poly", "rbf", "sigmoid"],
        index=2,
        key="svc_kernel",
        help="Kernel type to be used in SVC."
    )
    max_iter = st.text_input(
        "max_iter",
        value="100",
        key="svc_max_iter",
        help="Comma-separated values allowed for grid search."
    )
    return {'kernel': kernel, 'max_iter': max_iter}

def qda_params_ui():
    st.markdown("**QDA Hyperparameters:**")
    reg_param = st.text_input(
        "reg_param",
        value="0.0",
        key="qda_reg_param",
        help="Comma-separated values allowed for grid search."
    )
    return {'reg_param': reg_param}
import streamlit as st

def random_forest_params_ui():
    st.markdown("**RandomForestClassifier Hyperparameters:**")
    rf_n_estimators = st.text_input(
        "n_estimators",
        value="100",
        key="rf_n_estimators",
        help="Comma-separated values allowed for grid search."
    )
    rf_max_depth = st.text_input(
        "max_depth",
        value="10",
        key="rf_max_depth",
        help="Comma-separated values allowed for grid search."
    )
    rf_min_samples_split = st.text_input(
        "min_samples_split",
        value="2",
        key="rf_min_samples_split",
        help="Comma-separated values allowed for grid search."
    )
    rf_min_samples_leaf = st.text_input(
        "min_samples_leaf",
        value="1",
        key="rf_min_samples_leaf",
        help="Comma-separated values allowed for grid search."
    )
    rf_max_features = st.selectbox(
        "max_features",
        ["auto", "sqrt", "log2", None],
        index=1,
        key="rf_max_features",
        help="Number of features to consider when looking for the best split."
    )
    rf_bootstrap = st.checkbox(
        "bootstrap",
        value=True,
        key="rf_bootstrap",
        help="Whether bootstrap samples are used when building trees."
    )
    rf_random_state = st.text_input(
        "random_state",
        value="1",
        key="rf_random_state",
        help="Comma-separated values allowed for grid search."
    )
    return {
        'n_estimators': rf_n_estimators,
        'max_depth': rf_max_depth,
        'min_samples_split': rf_min_samples_split,
        'min_samples_leaf': rf_min_samples_leaf,
        'max_features': rf_max_features,
        'bootstrap': rf_bootstrap,
        'random_state': rf_random_state
    }

def knn_params_ui():
    st.markdown("**KNN Hyperparameters:**")
    knn_n_neighbors = st.text_input(
        "n_neighbors",
        value="5",
        key="knn_n_neighbors",
        help="Comma-separated values allowed for grid search."
    )
    knn_metric = st.selectbox(
        "Distance metric",
        ["minkowski", "euclidean", "manhattan", "cosine"],
        index=0,
        key="knn_metric",
        help="Distance metric for KNN."
    )
    return {
        'n_neighbors': knn_n_neighbors,
        'metric': knn_metric
    }

def logreg_params_ui():
    st.markdown("**Logistic Regression Hyperparameters:**")
    logreg_max_iter = st.text_input(
        "max_iter",
        value="100",
        key="logreg_max_iter",
        help="Comma-separated values allowed for grid search."
    )
    logreg_c = st.text_input(
        "C (inverse regularization)",
        value="1.0",
        key="logreg_c",
        help="Comma-separated values allowed for grid search."
    )
    # Return with proper sklearn parameter names
    # max_iter is handled separately in classifiers.py, so we store it with prefix
    return {
        'logreg_max_iter': logreg_max_iter,
        'C': logreg_c  # Use sklearn's actual parameter name
    }

def ridge_params_ui():
    st.markdown("**Ridge Classifier Hyperparameters:**")
    ridge_alpha = st.text_input(
        "alpha",
        value="1.0",
        key="ridge_alpha",
        help="Comma-separated values allowed for grid search."
    )
    return {'alpha': ridge_alpha}  # Use sklearn's actual parameter name

def naive_bayes_params_ui():
    st.markdown("**Naive Bayes Hyperparameters:**")
    nb_var_smoothing = st.text_input(
        "var_smoothing",
        value="1e-9",
        key="nb_var_smoothing",
        help="Comma-separated values allowed for grid search."
    )
    return {'var_smoothing': nb_var_smoothing}
