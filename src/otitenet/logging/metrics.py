# Libraries
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import brier_score_loss

smooth = 1

def get_mccs(lists, traces):
    """
    Function that gets the Matthews Correlation Coefficients. MCC is a statistical tool for model evaluation.
    It is a balanced measure which can be used even if the classes are of very different sizes.
    Args:
        lists:
        traces:

    Returns:
        traces: Same list as in the inputs arguments, except in now contains the MCC values
    """
    for group in ['train', 'valid', 'test']:
        try:
            preds, classes = np.concatenate(lists[group]['preds']).argmax(1), np.concatenate(
                lists[group]['classes'])
            traces[group]['mcc'] = MCC(preds, classes)
        except:
            pass
        

    return traces


# def jaccard_distance_loss(y_true, y_pred, smooth=100, backend=K):
#     y_true_f = backend.flatten(y_true)
#     y_pred_f = backend.flatten(y_pred)
#     intersection = backend.sum(backend.abs(y_true_f * y_pred_f))
#     sum_ = backend.sum(backend.abs(y_true_f) + backend.abs(y_pred_f))
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return (1 - jac) * smooth
# 
# 
# def mean_length_error(y_true, y_pred, backend=K):
#     y_true_f = backend.sum(backend.round(backend.flatten(y_true)))
#     y_pred_f = backend.sum(backend.round(backend.flatten(y_pred)))
#     delta = (y_pred_f - y_true_f)
#     return backend.mean(backend.tanh(delta))
# 
# 
# def dice_coef(y_true, y_pred, backend=K):
#     y_true_f = backend.flatten(y_true)
#     y_pred_f = backend.flatten(y_pred)
#     intersection = backend.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)
# 
# 
# def dice_coef_loss(y_true, y_pred, backend=K):
#     return -dice_coef(y_true, y_pred, backend=backend)
# 
# 
# def np_dice_coef(y_true, y_pred):
#     tr = y_true.flatten()
#     pr = y_pred.flatten()
#     return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)
# 
# 
# def batch_f1_score(batch_score, class_score):
#     return 2 * (1 - batch_score) * (class_score) / (1 - batch_score + class_score + 1e-4)
# 
# 
# # matthews_correlation
# def matthews_correlation(y_true, y_pred):
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     y_pred_neg = 1 - y_pred_pos
# 
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     y_neg = 1 - y_pos
# 
#     tp = K.sum(y_pos * y_pred_pos)
#     tn = K.sum(y_neg * y_pred_neg)
# 
#     fp = K.sum(y_neg * y_pred_pos)
#     fn = K.sum(y_pos * y_pred_neg)
# 
#     numerator = (tp * tn - fp * fn)
#     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
# 
#     return numerator / (denominator + K.epsilon())
# 
# # Compute LISI for: labels, batches, plates
# def rLISI(data, meta_data, perplexity=5):
#     import rpy2.robjects as robjects
#     from rpy2.robjects import pandas2ri
#     from rpy2.robjects.packages import importr
#     from rpy2.robjects.conversion import localconverter
#     lisi = importr('lisi')
#     # all_batches_r = robjects.IntVector(all_batches[all_ranks])
#     # all_data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
#     # labels = ['label1', 'label2']
#     labels = robjects.StrVector(['label1'])
#     new_meta_data = robjects.r.matrix(robjects.IntVector(meta_data), nrow=data.shape[0])
#     newdata = robjects.r.matrix(robjects.FloatVector(data.reshape(-1)), nrow=data.shape[0])
# 
#     new_meta_data.colnames = labels
#     results = lisi.compute_lisi(newdata, new_meta_data, labels, perplexity)
#     with localconverter(robjects.default_converter + pandas2ri.converter):
#         results = np.array(robjects.conversion.rpy2py(results))
#     mean = np.mean(results)
#     return mean  # , np.std(results), results
# 
# 
# # Compute LISI for: labels, batches, plates
# def rKBET(data, meta_data):
#     import rpy2.robjects as robjects
#     from rpy2.robjects import pandas2ri
#     from rpy2.robjects.packages import importr
#     from rpy2.robjects.conversion import localconverter
#     kbet = importr('kBET')
#     # all_batches_r = robjects.IntVector(all_batches[all_ranks])
#     # all_data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
#     # labels = ['label1', 'label2']
#     labels = robjects.StrVector(['label1'])
#     new_meta_data = robjects.IntVector(meta_data)
#     newdata = robjects.r.matrix(robjects.FloatVector(data.reshape(-1)), nrow=data.shape[0])
# 
#     new_meta_data.colnames = labels
#     results = kbet.kBET(newdata, new_meta_data, plot=False)
#     with localconverter(robjects.default_converter + pandas2ri.converter):
#         results = np.array(robjects.conversion.rpy2py(results))
#     # results[0]: summary - a rejection rate for the data, an expected rejection
#     # rate for random labeling and the significance for the observed result
#     try:
#         mean = results[0]['kBET.observed'][0]
#     except:
#         mean = 1
#     return mean  # , results[0]['kBET.signif'][0]

def batch_f1_score(batch_score, class_score):
    return 2 * (1 - batch_score) * (class_score) / (1 - batch_score + class_score + 1e-4)

def compute_batch_effect_metrics(embeddings, batch_labels, n_clusters=None):
    """
    Computes AMI, ARI, and batch entropy to assess batch effects in embeddings.

    Parameters:
    - embeddings (numpy array): Shape (n_samples, n_features), feature representation of samples.
    - batch_labels (numpy array or list): Shape (n_samples,), categorical batch labels.
    - n_clusters (int, optional): Number of clusters for KMeans. Defaults to the number of unique batches.

    Returns:
    - results (dict): Dictionary containing AMI, ARI, and batch entropy values.
    """
    unique_batches = np.unique(batch_labels)
    if n_clusters is None:
        n_clusters = len(unique_batches)

    # Clustering using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute AMI and ARI
    ami = adjusted_mutual_info_score(batch_labels, cluster_labels)
    ari = adjusted_rand_score(batch_labels, cluster_labels)

    # Compute Batch Entropy
    batch_counts_per_cluster = np.zeros((n_clusters, len(unique_batches)))

    for i, batch in enumerate(unique_batches):
        for j in range(n_clusters):
            batch_counts_per_cluster[j, i] = np.sum((batch_labels == batch) & (cluster_labels == j))

    # Normalize batch distribution within each cluster
    batch_proportions = batch_counts_per_cluster / batch_counts_per_cluster.sum(axis=1, keepdims=True)
    batch_proportions = np.nan_to_num(batch_proportions, nan=0.0)  # Handle clusters with no samples

    # Compute entropy for each cluster
    cluster_entropies = entropy(batch_proportions, axis=1, base=2)
    batch_entropy = np.mean(cluster_entropies)

    return {
        "AMI": ami,
        "ARI": ari,
        "Batch Entropy": batch_entropy
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        probs: predicted probabilities (N, C) or (N,) for binary.
        labels: true labels (N,), integer class ids.
        n_bins: number of equal-width bins over [0,1].
    Returns:
        Scalar ECE value in [0,1].
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim == 1:
        confs = probs
        preds = (probs >= 0.5).astype(int)
    else:
        preds = np.argmax(probs, axis=1)
        confs = probs[np.arange(len(probs)), preds]

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if not np.any(mask):
            continue
        acc_bin = np.mean(preds[mask] == labels[mask])
        conf_bin = np.mean(confs[mask])
        ece += np.abs(acc_bin - conf_bin) * (np.sum(mask) / len(confs))
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier score (lower is better).

    Supports binary or multiclass (one-vs-all average).
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim == 1:
        return float(brier_score_loss(labels, probs))
    n_classes = probs.shape[1]
    scores = []
    for c in range(n_classes):
        scores.append(brier_score_loss((labels == c).astype(int), probs[:, c]))
    return float(np.mean(scores))

