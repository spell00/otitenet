import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
# import splitters and other utilities as needed
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score


def compute_entropy_from_proba(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute row-wise entropy from probability matrix.

    Parameters
    ----------
    proba : np.ndarray of shape (n_samples, n_classes)

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Entropy for each sample.
    """
    proba = np.clip(proba, eps, 1.0)
    return -np.sum(proba * np.log(proba), axis=1)

def compute_entropy_from_proba(proba, eps=1e-12):
    proba = np.clip(proba, eps, 1.0)
    return -np.sum(proba * np.log(proba), axis=1)


def _safe_concat(chunks): 
    if chunks and len(chunks) > 0: 
        try: 
            return np.concatenate(chunks) 
        except Exception: 
            return None 
    return None

def get_batch_metrics(lists, n_neighbors=5, random_state=42):
    """
    Compute batch effect metrics for all samples using KNN classifier.
    Concatenate all encoded values and batch/domain labels, then split for evaluation.
    Returns dict of metrics: batch_entropy, batch_entropy_loss, batch_acc_loss, batch_mcc_loss.
    """
    encs = []
    batches = []
    for group in ['train', 'valid', 'test']:
        enc_group = _safe_concat(lists[group]['encoded_values'])
        batch_group = _safe_concat(lists[group]['domains'])
        if enc_group is not None and batch_group is not None:
            if len(enc_group) != len(batch_group):
                raise ValueError(
                    f"Batch metric input mismatch in group={group}: "
                    f"encoded_values={len(enc_group)} domains={len(batch_group)}"
                )
            if len(enc_group) > 0:
                encs.append(enc_group)
                batches.append(batch_group)
    if len(encs) == 0:
        return {
            'batch_entropy': np.nan,
            'batch_entropy_loss': np.nan,
            'batch_acc_loss': np.nan,
            'batch_mcc_loss': np.nan,
        }
    encs = np.concatenate(encs, axis=0)
    batches = np.concatenate(batches, axis=0)
    unique_batches = np.unique(batches)
    n_batches = len(unique_batches)
    if len(encs) < 2 or n_batches <= 1:
        return {
            'batch_entropy': 1.0,
            'batch_entropy_loss': 0.0,
            'batch_acc_loss': 0.0,
            'batch_mcc_loss': 0.0,
        }
    batch_counts = np.unique(batches, return_counts=True)[1]
    if np.min(batch_counts) < 2:
        return {
            'batch_entropy': np.nan,
            'batch_entropy_loss': np.nan,
            'batch_acc_loss': np.nan,
            'batch_mcc_loss': np.nan,
        }
    idx = np.arange(len(encs))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.5,
        stratify=batches,
        random_state=random_state
    )
    x_train = encs[train_idx]
    y_train = batches[train_idx]
    x_test = encs[test_idx]
    y_test = batches[test_idx]
    k = min(max(1, n_neighbors), len(x_train))
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_test)
    preds = clf.predict(x_test)
    entropy = compute_entropy_from_proba(proba)
    max_entropy = np.log(n_batches)
    entropy_norm = entropy / (max_entropy + 1e-12)
    batch_entropy = float(np.mean(entropy_norm))
    batch_entropy_loss = 1.0 - batch_entropy
    batch_acc_loss = float(accuracy_score(y_test, preds))
    batch_mcc_loss = float(max(0.0, matthews_corrcoef(y_test, preds)))
    
    # Compute NMI, ARI, AMI, silhouette score and others
    
    nmi = float(normalized_mutual_info_score(y_test, preds))
    ari = float(adjusted_rand_score(y_test, preds))
    ami = float(adjusted_mutual_info_score(y_test, preds))
    silhouette = float(silhouette_score(x_test, preds)) if len(np.unique(preds)) > 1 else np.nan
    
    return {
        'batch_entropy': batch_entropy,
        'batch_entropy_loss': batch_entropy_loss,
        'batch_acc_loss': batch_acc_loss,
        'batch_mcc_loss': batch_mcc_loss,
        'batch_nmi': nmi,
        'batch_ari': ari,
        'batch_ami': ami,
        'batch_silhouette': silhouette
    }
