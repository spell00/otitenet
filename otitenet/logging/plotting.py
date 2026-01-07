from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.transforms as matplotlib_transforms
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import mlflow

def save_roc_curve(y_pred_proba, y_test, unique_labels, name, binary, acc, mlops, epoch=None, logger=None):
    """

    Args:
        model:
        x_test:
        y_test:
        unique_labels:
        name:
        binary: Is it a binary classification?
        acc:
        epoch:
        logger:

    Returns:

    """
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    os.makedirs(f'{dirs}', exist_ok=True)
    if binary:
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        roc_score = metrics.roc_auc_score(y_true=y_test, y_score=y_pred_proba)

        # create ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC curve (acc={np.round(acc, 3)})')
        plt.legend(loc="lower right")
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/{name}.png')
                stuck = False
            except:
                print('stuck...')
        plt.close()
    else:
        # Compute ROC curve and ROC area for each class
        from sklearn.preprocessing import label_binarize
        # enc = model(x_test)
        # y_pred_proba = model.classifier(enc)

        # y_preds = model.predict_proba(x_test)
        n_classes = len(unique_labels) - 1  # -1 to remove QC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        classes = np.arange(len(unique_labels))

        roc_score = metrics.roc_auc_score(y_true=y_test, y_score=y_pred_proba, multi_class='ovr')
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(label_binarize(y_test, classes=classes)[:, i], y_pred_proba[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        # roc for each class
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.title(f'ROC curve (AUC={np.round(roc_score, 3)}, acc={np.round(acc, 3)})')
        # ax.plot(fpr[0], tpr[0], label=f'AUC = {np.round(roc_score, 3)} (All)', color='k')
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'AUC = {np.round(roc_auc[i], 3)} ({unique_labels[i]})')
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        # sns.despine()
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/{name}')
                stuck = False
            except:
                print('stuck...')

        if logger is not None:
            if mlops == 'tensorboard':
                logger.add_figure(name, fig, epoch)
            if mlops == 'neptune':
                logger[name].log(fig)
        if mlops == 'mlflow':
            mlflow.log_figure(fig, name)
                
        plt.close(fig)

    return roc_score


def plot_confusion_matrix(cm, class_names, acc, mcc=None):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    cm_normal = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm_normal[np.isnan(cm_normal)] = 0
    plt.imshow(cm_normal, interpolation='nearest', cmap=plt.cm.Blues)
    title_suffix = f"acc: {acc:.3f}"
    if mcc is not None:
        title_suffix += f" | mcc: {mcc:.3f}"
    plt.title(f"Confusion matrix ({title_suffix})")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = 0.5

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm_normal[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', train_set=True, **kwargs):
    """
    FROM https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = matplotlib_transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ellipse1 = create_ellipse((mean_x, mean_y), (scale_x * 2 * ell_radius_x, scale_y * 2 * ell_radius_y),
                              ellipse.get_angle())
    verts1 = np.array(ellipse1.exterior.coords.xy)
    # patch1 = Polygon(verts1.T, color='blue', alpha=0.5)
    # print(ellipse1.area)
    if train_set:
        return ax.add_patch(ellipse), ellipse1
    else:
        return None, ellipse1


def rand_jitter(arr):
    return arr + np.random.randn(len(arr)) * 0.01

def plot_pca(data, labels, batches, prototypes, explained_variance, complete_log_path, name, ordin_name):
    """
    Plots the PCA of the data
    Args:
        data: data to be plotted
        labels: labels of the data
        batches: batches of the data
        explained_variance: explained variance of the PCA
        complete_log_path: path to save the plot
        name: name of the plot
        n: number of components to plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    markers = np.array(['o', 'x', '^', 's', 'P', 'D', 'v', '<', '>', 'p', 'h', 'H', '*', '+', 'X', '|', '_'])
    # Set a super-title for all subplots
    fig.suptitle(f'{ordin_name} of Encoded Representations: {name}', fontsize=16)
    for k in range(2):
        for j in range(2):
            # Leave quadrant (2,1) (row=1, col=0) completely blank.
            if k == 1 and j == 0:
                axs[k, j].axis('off')
                continue

            # scatter with batch info as shape
            for i, batch in enumerate(np.unique(batches)):
                mask = batches == batch
                axs[k, j].scatter(data[mask, k], data[mask, j+1], c=labels[mask],
                            cmap='viridis', alpha=0.6, marker=markers[i]
                            )
            if prototypes is not None:
                try:
                    axs[k, j].scatter(prototypes[:, k], prototypes[:, j+1], c='red', s=100, marker='x')
                except IndexError:
                    print("IndexError: prototypes shape mismatch.")
            # plt.colorbar()

            # plt.colorbar(scatter)
            if explained_variance is not None:
                axs[k, j].set_xlabel(f'Component {k+1}: {explained_variance[k]:.2f}%')
                axs[k, j].set_ylabel(f'Component {j+2}: {explained_variance[j+1]:.2f}%')
            else:
                axs[k, j].set_xlabel(f'Component {k+1}')
                axs[k, j].set_ylabel(f'Component {j+2}')
            plt.colorbar(axs[k, j].collections[0], ax=axs[k, j])
    plt.tight_layout()
    plt.savefig(f'{complete_log_path}/{name}_{ordin_name}.png')
    plt.close()
