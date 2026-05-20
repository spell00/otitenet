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
from sklearn.neural_network import MLPClassifier
from otitenet.utils.kde import make_kde_classifier
from otitenet.logging.metrics import MCC
from sklearn.metrics import roc_auc_score


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


def fit_linearsvc_classifier(train_encs: np.ndarray, train_cats: np.ndarray,
                             max_iter: int = 2000, **kwargs):
    """Fit a LinearSVC classifier on training embeddings.
    
    Args:
        train_encs: Training embeddings (n_samples, n_features)
        train_cats: Training labels (n_samples,)
        max_iter: Maximum number of iterations
        **kwargs: Additional parameters for LinearSVC
        
    Returns:
        Fitted LinearSVC classifier
    """
    svc = LinearSVC(max_iter=max_iter, random_state=1, **kwargs)
    svc.fit(train_encs, train_cats)
    return svc


def fit_logreg_classifier(train_encs: np.ndarray, train_cats: np.ndarray,
                          max_iter: int = 200, **kwargs):
    """Fit a LogisticRegression classifier on training embeddings.
    
    Args:
        train_encs: Training embeddings (n_samples, n_features)
        train_cats: Training labels (n_samples,)
        max_iter: Maximum number of iterations
        **kwargs: Additional parameters for LogisticRegression
        
    Returns:
        Fitted LogisticRegression classifier
    """
    lr = LogisticRegression(max_iter=max_iter, random_state=1, **kwargs)
    lr.fit(train_encs, train_cats)
    return lr


def fit_xgboost_classifier(train_encs: np.ndarray, train_cats: np.ndarray, **kwargs):
    """Fit an XGBoost classifier on training embeddings.
    
    Args:
        train_encs: Training embeddings (n_samples, n_features)
        train_cats: Training labels (n_samples,)
        **kwargs: Additional parameters for XGBClassifier
        
    Returns:
        Fitted XGBClassifier or None if not installed
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost not installed. Skipping.")
        return None
        
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 1,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    params.update(kwargs)
    
    xgb = XGBClassifier(**params)
    xgb.fit(train_encs, train_cats)
    return xgb


def _get_clf_metrics(clf, X, y, y_int=None):
    """Helper to compute MCC and AUC for a classifier."""
    # Defensive check: if passed a dict containing the classifier, extract it
    if isinstance(clf, dict) and 'classifier' in clf:
        clf = clf['classifier']
        
    if y_int is None:
        y_int = y
        
    if clf is None or not hasattr(clf, 'predict'):
        return 0.0, 0.5
        
    preds = clf.predict(X)
    mcc = float(MCC(y_int, preds))
    auc = 0.0
    try:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)
            if probs.shape[1] > 2:
                auc = float(roc_auc_score(y_int, probs, multi_class='ovr'))
            elif probs.shape[1] == 2:
                auc = float(roc_auc_score(y_int, probs[:, 1]))
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)
            if scores.ndim == 1 or scores.shape[1] == 2:
                # For binary classifiers like LinearSVC, scores is (N,) or (N, 2)
                auc = float(roc_auc_score(y_int, scores if scores.ndim == 1 else scores[:, 1]))
            else:
                # Multiclass decision_function
                auc = float(roc_auc_score(y_int, scores, multi_class='ovr'))
    except Exception:
        pass
    return mcc, auc


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
    if classifier_params is None or len(classifier_params) == 0:
        # Default comparison set used by app learned-embedding optimization.
        classifier_params = {
            'logreg_params': {},
            'ridge_params': {},
            'naive_bayes_params': {},
            'linear_svc_params': {},
            'rbf_svc_params': {},
            'random_forest_params': {},
            'gradient_boosting_params': {},
            'decision_tree_params': {},
            'lda_params': {},
            'qda_params': {},
            'xgboost_params': {},
            'mlp_head_params': {},
            'logreg_max_iter': 200,
            'linearsvc_max_iter': 2000,
            'svc_max_iter': 2000,
            'svc_kernel': 'rbf',
            'rfc_n_estimators': 100,
            'gbc_n_estimators': 100,
        }
    # Convert string labels to integers for compatibility with sklearn
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_cats_int = le.fit_transform(train_cats)
    # Ensure valid_cats can be transformed (handles cases where valid has classes not in train, though rare)
    try:
        valid_cats_int = le.transform(valid_cats)
    except ValueError:
        # Fallback: re-fit on everything if needed, though this might change indices
        le.fit(np.unique(np.concatenate([train_cats, valid_cats])))
        train_cats_int = le.transform(train_cats)
        valid_cats_int = le.transform(valid_cats)
    
    label_map = {str(label): idx for idx, label in enumerate(le.classes_)}
    inv_label_map = {idx: label for idx, label in enumerate(le.classes_)}
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
    mlp_head_params = classifier_params.get('mlp_head_params', {})
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
            valid_mcc, valid_auc = _get_clf_metrics(logreg, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(logreg, train_encs, train_cats_int)
            results['logreg'] = {
                'classifier': logreg, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"LogReg fitting failed: {e}")
            results['logreg'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'ridge_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Ridge Classifier. (2/10).")
        try:
            ridge = RidgeClassifier(random_state=1, **ridge_params)
            ridge.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(ridge, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(ridge, train_encs, train_cats_int)
            results['ridge'] = {
                'classifier': ridge, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"Ridge fitting failed: {e}")
            results['ridge'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'naive_bayes_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Naive Bayes. (3/10).")
        try:
            nb = GaussianNB(**nb_params)
            nb.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(nb, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(nb, train_encs, train_cats_int)
            results['naive_bayes'] = {
                'classifier': nb, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"NaiveBayes fitting failed: {e}")
            results['naive_bayes'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'linear_svc_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Linear SVC. (4/10).")
        try:
            svc = LinearSVC(max_iter=linearsvc_max_iter, random_state=1, **linearsvc_params)
            svc.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(svc, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(svc, train_encs, train_cats_int)
            results['linear_svc'] = {
                'classifier': svc, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"LinearSVC fitting failed: {e}")
            results['linear_svc'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'rbf_svc_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting RBF SVC. (5/10).")
        try:
            rbf_svc = SVC(kernel=svc_kernel, max_iter=svc_max_iter, random_state=1, **rbfsvc_params)
            rbf_svc.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(rbf_svc, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(rbf_svc, train_encs, train_cats_int)
            results['rbf_svc'] = {
                'classifier': rbf_svc, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"RBF SVC fitting failed: {e}")
            results['rbf_svc'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'random_forest_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Random Forest. (6/10).")
        try:
            rf = RandomForestClassifier(n_estimators=rfc_n_estimators, random_state=1, n_jobs=-1, **rf_params)
            rf.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(rf, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(rf, train_encs, train_cats_int)
            results['random_forest'] = {
                'classifier': rf, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"RandomForest fitting failed: {e}")
            results['random_forest'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'gradient_boosting_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Gradient Boosting. (7/10).")
        try:
            gb = GradientBoostingClassifier(n_estimators=gbc_n_estimators, random_state=1, **gb_params)
            gb.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(gb, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(gb, train_encs, train_cats_int)
            results['gradient_boosting'] = {
                'classifier': gb, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"GradientBoosting fitting failed: {e}")
            results['gradient_boosting'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'decision_tree_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Decision Tree. (8/10).")
        try:
            dt = DecisionTreeClassifier(random_state=1, **dt_params)
            dt.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(dt, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(dt, train_encs, train_cats_int)
            results['decision_tree'] = {
                'classifier': dt, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"DecisionTree fitting failed: {e}")
            results['decision_tree'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'lda_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Linear Discriminant Analysis. (9/10).")
        try:
            lda = LinearDiscriminantAnalysis(**lda_params)
            lda.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(lda, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(lda, train_encs, train_cats_int)
            results['lda'] = {
                'classifier': lda, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"LDA fitting failed: {e}")
            results['lda'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'qda_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting Quadratic Discriminant Analysis. (10/10).")
        try:
            qda = QuadraticDiscriminantAnalysis(**qda_params)
            qda.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(qda, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(qda, train_encs, train_cats_int)
            results['qda'] = {
                'classifier': qda, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"QDA fitting failed: {e}")
            results['qda'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'mlp_head_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting MLP Head. (11/11).")
        try:
            merged_params = {
                'hidden_layer_sizes': (128,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 1e-4,
                'learning_rate_init': 1e-3,
                'max_iter': 300,
                'random_state': 1,
                'early_stopping': True,
                'n_iter_no_change': 15,
            }
            merged_params.update(mlp_head_params or {})
            mlp_head = MLPClassifier(**merged_params)
            mlp_head.fit(train_encs, train_cats_int)
            valid_mcc, valid_auc = _get_clf_metrics(mlp_head, valid_encs, valid_cats_int)
            train_mcc, train_auc = _get_clf_metrics(mlp_head, train_encs, train_cats_int)
            results['mlp_head'] = {
                'classifier': mlp_head, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
            }
        except Exception as e:
            print(f"MLP head fitting failed: {e}")
            results['mlp_head'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    if 'xgboost_params' in classifier_params:
        if progress_placeholder is not None:
            progress_placeholder.info("⚙️ Fitting XGBoost. (12/12).")
        try:
            xgb = fit_xgboost_classifier(train_encs, train_cats_int, **classifier_params.get('xgboost_params', {}))
            if xgb is not None:
                valid_mcc, valid_auc = _get_clf_metrics(xgb, valid_encs, valid_cats_int)
                train_mcc, train_auc = _get_clf_metrics(xgb, train_encs, train_cats_int)
                results['xgboost'] = {
                    'classifier': xgb, 'valid_mcc': valid_mcc, 'valid_auc': valid_auc,
                    'train_mcc': train_mcc, 'train_auc': train_auc, 'error': None
                }
            else:
                results['xgboost'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': 'XGBoost not installed'}
        except Exception as e:
            print(f"XGBoost fitting failed: {e}")
            results['xgboost'] = {'classifier': None, 'valid_mcc': None, 'train_mcc': None, 'error': str(e)}
    results['label_map'] = label_map
    results['label_encoder'] = le
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
