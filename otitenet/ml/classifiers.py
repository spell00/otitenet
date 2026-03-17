"""
Classifier fitting and training functions.

Contains functions to fit various classifier types on embeddings.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from otitenet.utils.kde import make_kde_classifier


def fit_knn_classifier(train_encs: np.ndarray, train_cats: np.ndarray,
                       n_neighbors: int = 5, metric: str = 'minkowski', knn_params: dict = None):
    """Fit a KNN classifier on training embeddings.
    
    Args:
        train_encs: Training embeddings (n_samples, n_features)
        train_cats: Training labels (n_samples,)
        n_neighbors: Number of neighbors
        metric: Distance metric
        knn_params: Optional dict of KNN parameters
        
    Returns:
        Fitted KNN classifier
    """
    # Ensure k doesn't exceed number of samples
    n_neighbors = min(n_neighbors, train_encs.shape[0])
    params = {'n_neighbors': n_neighbors, 'metric': metric}
    if knn_params:
        params.update(knn_params)
    knn = KNN(**params)
    knn.fit(train_encs, train_cats)
    return knn


def fit_baseline_classifiers(
    train_encs: np.ndarray,
    train_cats: np.ndarray,
    valid_encs: np.ndarray,
    valid_cats: np.ndarray,
    progress_placeholder=None,
    classifier_params: dict = None
):
    """Fit multiple baseline classifiers on training embeddings.
    
    Args:
        train_encs: Training embeddings (n_samples, n_features)
        train_cats: Training labels (n_samples,)
        progress_placeholder: Optional Streamlit placeholder for progress updates
        classifier_params: Dictionary of classifier parameters. Example:
            {
                'logreg_max_iter': 10,
                'linearsvc_max_iter': 2,
                'svc_kernel': 'rbf',
                'svc_max_iter': 2,
                'rfc_n_estimators': 1,
                'gbc_n_estimators': 1
            }
        
    Returns:
        Dictionary mapping classifier names to dicts with classifier, MCC, and error if any
    """
    if classifier_params is None:
        classifier_params = {}
    # Convert string labels to integers for compatibility with sklearn
    label_map = {label: idx for idx, label in enumerate(np.unique(np.concatenate([train_cats, valid_cats])))}
    inv_label_map = {v: k for k, v in label_map.items()}
    train_cats_int = np.array([label_map[x] for x in train_cats])
    valid_cats_int = np.array([label_map[x] for x in valid_cats])
    logreg_max_iter = classifier_params.get('logreg_max_iter', 10)
    logreg_params = classifier_params.get('logreg_params', {})
    # Remove logreg_max_iter from logreg_params if present (handled separately)
    logreg_params = {k: v for k, v in logreg_params.items() if k != 'logreg_max_iter'}
    ridge_params = classifier_params.get('ridge_params', {})
    nb_params = classifier_params.get('naive_bayes_params', {})
    linearsvc_max_iter = classifier_params.get('linearsvc_max_iter', 2)
    linearsvc_params = classifier_params.get('linear_svc_params', {})
    svc_kernel = classifier_params.get('svc_kernel', 'rbf')
    svc_max_iter = classifier_params.get('svc_max_iter', 2)
    rbfsvc_params = classifier_params.get('rbf_svc_params', {})
    rfc_n_estimators = classifier_params.get('rfc_n_estimators', 1)
    rf_params = classifier_params.get('random_forest_params', {})
    gbc_n_estimators = classifier_params.get('gbc_n_estimators', 1)
    gb_params = classifier_params.get('gradient_boosting_params', {})
    dt_params = classifier_params.get('decision_tree_params', {})
    lda_params = classifier_params.get('lda_params', {})
    qda_params = classifier_params.get('qda_params', {})
    classifiers = {}
    from otitenet.logging.metrics import MCC
    results = {}
    # Only fit classifiers present in classifier_params (i.e., selected in UI)
    if 'logreg_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Logistic Regression (1/10).")
        try:
            logreg = LogisticRegression(max_iter=logreg_max_iter, random_state=1, **logreg_params)
            logreg.fit(train_encs, train_cats_int)
            preds = logreg.predict(valid_encs)
            valid_cats_str = np.array([inv_label_map[x] for x in valid_cats_int])
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = logreg.predict(train_encs)
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['logreg'] = {'classifier': logreg, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"LogReg fitting failed: {e}")
            results['logreg'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'ridge_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Ridge Classifier. (2/10).")
        try:
            ridge = RidgeClassifier(random_state=1, **ridge_params)
            ridge.fit(train_encs, train_cats_int)
            preds = ridge.predict(valid_encs)
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = ridge.predict(train_encs)
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['ridge'] = {'classifier': ridge, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"Ridge fitting failed: {e}")
            results['ridge'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'naive_bayes_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Naive Bayes. (3/10).")
        try:
            nb = GaussianNB(**nb_params)
            nb.fit(train_encs, train_cats_int)
            preds = nb.predict(valid_encs)
            # preds_str = np.array([inv_label_map[x] for x in preds])
            valid_cats_str = np.array([inv_label_map[x] for x in valid_cats_int])
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = nb.predict(train_encs)
            # train_preds_str = np.array([inv_label_map[x] for x in train_preds])
            train_cats_str = np.array([inv_label_map[x] for x in train_cats_int])
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['naive_bayes'] = {'classifier': nb, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"NaiveBayes fitting failed: {e}")
            results['naive_bayes'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'linear_svc_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Linear SVC. (4/10).")
        try:
            svc = LinearSVC(max_iter=linearsvc_max_iter, random_state=1, **linearsvc_params)
            svc.fit(train_encs, train_cats_int)
            preds = svc.predict(valid_encs)
            valid_cats_str = np.array([inv_label_map[x] for x in valid_cats_int])
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = svc.predict(train_encs)
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['linear_svc'] = {'classifier': svc, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"LinearSVC fitting failed: {e}")
            results['linear_svc'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'rbf_svc_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting RBF SVC. (5/10).")
        try:
            rbf_svc = SVC(kernel=svc_kernel, max_iter=svc_max_iter, random_state=1, **rbfsvc_params)
            rbf_svc.fit(train_encs, train_cats_int)
            preds = rbf_svc.predict(valid_encs)
            valid_cats_str = np.array([inv_label_map[x] for x in valid_cats_int])
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = rbf_svc.predict(train_encs)
            # train_preds_str = np.array([inv_label_map[x] for x in train_preds])
            train_cats_str = np.array([inv_label_map[x] for x in train_cats_int])
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['rbf_svc'] = {'classifier': rbf_svc, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"RBF SVC fitting failed: {e}")
            results['rbf_svc'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'random_forest_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Random Forest. (6/10).")
        try:
            rf = RandomForestClassifier(n_estimators=rfc_n_estimators, random_state=1, n_jobs=-1, **rf_params)
            rf.fit(train_encs, train_cats_int)
            preds = rf.predict(valid_encs)
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = rf.predict(train_encs)
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['random_forest'] = {'classifier': rf, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"RandomForest fitting failed: {e}")
            results['random_forest'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'gradient_boosting_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Gradient Boosting. (7/10).")
        try:
            gb = GradientBoostingClassifier(n_estimators=gbc_n_estimators, random_state=1, **gb_params)
            gb.fit(train_encs, train_cats_int)
            preds = gb.predict(valid_encs)
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = gb.predict(train_encs)
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['gradient_boosting'] = {'classifier': gb, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"GradientBoosting fitting failed: {e}")
            results['gradient_boosting'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'decision_tree_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Decision Tree. (8/10).")
        try:
            dt = DecisionTreeClassifier(random_state=1, **dt_params)
            dt.fit(train_encs, train_cats_int)
            preds = dt.predict(valid_encs)
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = dt.predict(train_encs)
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['decision_tree'] = {'classifier': dt, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"DecisionTree fitting failed: {e}")
            results['decision_tree'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'lda_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Linear Discriminant Analysis. (9/10).")
        try:
            lda = LinearDiscriminantAnalysis(**lda_params)
            lda.fit(train_encs, train_cats_int)
            preds = lda.predict(valid_encs)
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = lda.predict(train_encs)
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['lda'] = {'classifier': lda, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"LDA fitting failed: {e}")
            results['lda'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'qda_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Quadratic Discriminant Analysis. (10/10).")
        try:
            qda = QuadraticDiscriminantAnalysis(**qda_params)
            qda.fit(train_encs, train_cats_int)
            preds = qda.predict(valid_encs)
            valid_mcc = float(MCC(valid_cats_int, preds))
            train_preds = qda.predict(train_encs)
            train_mcc = float(MCC(train_cats_int, train_preds))
            results['qda'] = {'classifier': qda, 'valid_mcc': valid_mcc, 'train_mcc': train_mcc, 'error': None}
        except Exception as e:
            print(f"QDA fitting failed: {e}")
            results['qda'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    return results


def fit_kde_classifier(train_encs: np.ndarray, train_cats: np.ndarray,
                       kernel: str = 'gaussian', bandwidth: str = 'scott',
                       learnable: bool = False, soft: bool = True):
    """Fit a KDE classifier on training embeddings.
    
    Args:
        train_encs: Training embeddings (n_samples, n_features)
        train_cats: Training labels (n_samples,)
        kernel: KDE kernel type
        bandwidth: Bandwidth parameter
        learnable: Whether bandwidth is learnable
        soft: Whether to use soft predictions
        
    Returns:
        Fitted KDE classifier
    """
    kde = make_kde_classifier(
        kernel=kernel,
        bandwidth=bandwidth,
        learnable=learnable,
        soft=soft
    )
    kde.fit(train_encs, train_cats)
    return kde


def predict_with_prototypes(embeddings: np.ndarray, 
                           prototype_vectors: np.ndarray,
                           prototype_labels: np.ndarray,
                           metric: str = 'euclidean', proto_params: dict = None):
    """Predict labels using nearest prototype matching.
    
    Args:
        embeddings: Query embeddings (n_samples, n_features)
        prototype_vectors: Prototype vectors (n_prototypes, n_features)
        prototype_labels: Labels for each prototype (n_prototypes,)
        metric: Distance metric ('euclidean' or 'cosine')
        proto_params: Optional dict of parameters for prototype prediction
        
    Returns:
        Predicted labels (n_samples,)
    """
    # Allow override of metric and other params
    if proto_params:
        metric = proto_params.get('metric', metric)
    if metric == 'cosine':
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        prototypes_norm = prototype_vectors / (np.linalg.norm(prototype_vectors, axis=1, keepdims=True) + 1e-8)
        similarity = embeddings_norm @ prototypes_norm.T
        nearest_indices = np.argmax(similarity, axis=1)
    else:
        dists = np.linalg.norm(
            embeddings[:, None, :] - prototype_vectors[None, :, :],
            axis=2
        )
        nearest_indices = np.argmin(dists, axis=1)
    return prototype_labels[nearest_indices]


def build_knn_from_model(model, train_loader, unique_labels, n_neighbors: int = 5, 
                         metric: str = 'minkowski', device: str = 'cpu'):
    """Build a KNN classifier by encoding training data from a model.
    
    This function loads training data through a model for encoding, then fits a KNN classifier.
    Useful for inference pipelines where you need to fit KNN on the fly.
    
    Args:
        model: Trained deep learning model with inference capability
        train_loader: DataLoader for training data (should yield batches with embeddings or images)
        unique_labels: Array of unique label values
        n_neighbors: Number of neighbors for KNN
        metric: Distance metric
        device: Device to use for model ('cpu' or 'cuda')
        
    Returns:
        Tuple of (fitted_knn, train_encodings, train_labels)
        - fitted_knn: Fitted KNeighborsClassifier
        - train_encodings: (n_samples, n_features) array of encoded training data
        - train_labels: (n_samples,) array of training labels
        
    Example:
        knn, encs, labels = build_knn_from_model(
            model, train_loader, unique_labels, n_neighbors=5
        )
        predictions = knn.predict(test_encodings)
    """
    import torch
    
    train_encodings = []
    train_labels_list = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            # Handle different loader formats
            if isinstance(batch, (tuple, list)):
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    images = batch[0]
                    labels = None
            else:
                # Assume batch is image data only
                images = batch
                labels = None
            
            # Move to device
            if isinstance(images, torch.Tensor):
                images = images.to(device)
            
            # Encode with model
            encodings = model(images)
            if isinstance(encodings, torch.Tensor):
                encodings = encodings.detach().cpu().numpy()
            
            train_encodings.append(encodings)
            
            # Track labels if available
            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                train_labels_list.append(labels)
    
    # Concatenate all batches
    train_encodings = np.concatenate(train_encodings, axis=0)
    if train_labels_list:
        train_labels = np.concatenate(train_labels_list, axis=0)
    else:
        # If no labels in loader, create dummy labels (shouldn't happen in practice)
        train_labels = np.zeros(train_encodings.shape[0], dtype=int)
    
    # Fit KNN on encoded data
    knn = fit_knn_classifier(train_encodings, train_labels, n_neighbors=n_neighbors, metric=metric)
    
    return knn, train_encodings, train_labels
