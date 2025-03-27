import matplotlib.pyplot as plt
import numpy as np
import itertools

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.pipeline import Pipeline
import os
import math
import torch
import mlflow
import sklearn
import itertools
import numpy as np
import tensorflow as tf
from sklearn import metrics
from itertools import cycle
from matplotlib import pyplot as plt, cm
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay
# from src.ml.train.params_gp import *
# from src.ml.train.sklearn_train_nocv import Train
from skopt import gp_minimize
from sklearn.preprocessing import label_binarize, OneHotEncoder
from PIL import Image


def scale_data(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    scaler = None
    if scale == 'binarize':
        for group in list(data['inputs'].keys()):
            data['inputs'][group] = data['inputs'][group]
            data['inputs'][group][data['inputs'][group] > 0] = 1
            data['inputs'][group][data['inputs'][group] <= 0] = 0

    elif scale == 'robust_per_batch':
        scalers = {b: RobustScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = RobustScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'standard_per_batch':
        scalers = {b: StandardScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = StandardScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'minmax_per_batch':
        scalers = {b: MinMaxScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = MinMaxScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'robust':
        scaler = RobustScaler()
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'robust_minmax':
        scaler = Pipeline([('robust', RobustScaler()), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'standard':
        scaler = StandardScaler()
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'standard_minmax':
        scaler = Pipeline([('standard', StandardScaler()), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'minmax':
        scaler = MinMaxScaler()
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l1_minmax':
        scaler = Pipeline([('l1', Normalizer(norm='l1')), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l2_minmax':
        scaler = Pipeline([('l2', Normalizer(norm='l2')), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l1':
        scaler = Pipeline([('l1', Normalizer(norm='l1'))])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l2':
        scaler = Pipeline([('l2', Normalizer(norm='l2'))])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'none':
        return data, 'none'
    # Values put on [0, 1] interval to facilitate autoencoder reconstruction (enable use of sigmoid as final activation)
    # scaler = MinMaxScaler()
    # scaler.fit(data['inputs']['all'])
    # for group in list(data['inputs'].keys()):
    #     if data['inputs'][group].shape[0] > 0:
    #         columns = data['inputs'][group].columns
    #         indices = data['inputs'][group].index
    #         if data['inputs'][group].shape[0] > 0:
    #             data['inputs'][group] = scaler.transform(data['inputs'][group].to_numpy())
    #         else:
    #             data['inputs'][group] = data['inputs'][group]
    #         data['inputs'][group] = pd.DataFrame(data['inputs'][group], columns=columns, index=indices)

    return data, scaler


def scale_data_images(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    scaler = None
    if scale == 'binarize':
        for group in list(data['inputs'].keys()):
            data['inputs'][group] = data['inputs'][group]
            data['inputs'][group][data['inputs'][group] > 0] = 1
            data['inputs'][group][data['inputs'][group] <= 0] = 0

    elif scale == 'standard':
        # scaler = StandardScaler()
        for data_type in ['inputs', 'meta']:
            data[data_type]['all'] = data[data_type]['all'] - data[data_type]['all'].mean(axis=(1, 2), keepdims=True)
            data[data_type]['all'] = data[data_type]['all'] / data[data_type]['all'].std(axis=(1, 2), keepdims=True)
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    data[data_type][group] = data[data_type][group] - data[data_type][group].mean(axis=(1, 2),
                                                                                                  keepdims=True)
                    data[data_type][group] = data[data_type][group] / data[data_type][group].std(axis=(1, 2),
                                                                                                 keepdims=True)

    elif scale == 'none':
        return data, 'none'
    # Values put on [0, 1] interval to facilitate autoencoder reconstruction (enable use of sigmoid as final activation)
    # scaler = MinMaxScaler()
    # scaler.fit(data['inputs']['all'])
    # for group in list(data['inputs'].keys()):
    #     if data['inputs'][group].shape[0] > 0:
    #         columns = data['inputs'][group].columns
    #         indices = data['inputs'][group].index
    #         if data['inputs'][group].shape[0] > 0:
    #             data['inputs'][group] = scaler.transform(data['inputs'][group].to_numpy())
    #         else:
    #             data['inputs'][group] = data['inputs'][group]
    #         data['inputs'][group] = pd.DataFrame(data['inputs'][group], columns=columns, index=indices)

    return data, scaler


def scale_data_per_batch(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    if scale == 'robust':
        scalers = {b: RobustScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = RobustScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'standard':
        scalers = {b: StandardScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = StandardScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'minmax':
        scalers = {b: MinMaxScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = MinMaxScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'none':
        return data
    # Values put on [0, 1] interval to facilitate autoencoder reconstruction (enable use of sigmoid as final activation)
    # scaler = MinMaxScaler()
    # scaler.fit(data['inputs']['all'])
    # for group in list(data['inputs'].keys()):
    #     if data['inputs'][group].shape[0] > 0:
    #         columns = data['inputs'][group].columns
    #         indices = data['inputs'][group].index
    #         if data['inputs'][group].shape[0] > 0:
    #             data['inputs'][group] = scaler.transform(data['inputs'][group].to_numpy())
    #         else:
    #             data['inputs'][group] = data['inputs'][group]
    #         data['inputs'][group] = pd.DataFrame(data['inputs'][group], columns=columns, index=indices)

    return data, scaler


def plot_confusion_matrix(cm, class_names, acc):
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
    plt.title(f"Confusion matrix (acc: {acc})")
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


def get_unique_labels(labels):
    """
    Get unique labels for a set of labels
    :param labels:
    :return:
    """
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels += [label]
    return np.array(unique_labels)


def get_combinations(cm, acc_cutoff=0.6, prop_cutoff=0.8):
    # cm2 = np.zeros(shape=(2, 2))
    to_combine = []
    for i in range(len(list(cm.index)) - 1):
        for j in range(i + 1, len(list(cm.columns))):
            acc = (cm.iloc[i, i] + cm.iloc[j, j]) / (cm.iloc[i, j] + cm.iloc[j, i] + cm.iloc[i, i] + cm.iloc[j, j])
            prop = (cm.iloc[i, j] + cm.iloc[j, i] + cm.iloc[i, i] + cm.iloc[j, j]) / (
                    np.sum(cm.iloc[i, :]) + np.sum(cm.iloc[j, :]))
            if acc < acc_cutoff and prop > prop_cutoff:
                to_combine += [(i, j)]

    # Combine all tuple that have a class in common

    new = True
    while new:
        new = False
        for i in range(len(to_combine) - 1):
            for j in range(i + 1, len(to_combine)):
                if np.sum([1 if x in to_combine[j] else 0 for x in to_combine[i]]) > 0:
                    new_combination = tuple(set(to_combine[i] + to_combine[j]))
                    to_combine = list(
                        set([ele for x, ele in enumerate(to_combine) if x not in [i, j]] + [new_combination]))
                    new = True
                    break

    return to_combine


def to_csv(lists, complete_log_path, columns):
    encoded_data = {}
    encoded_batches = {}
    encoded_cats = {}
    encoded_names = {}
    for group in list(lists.keys()):
        if len(lists[group]['encoded_values']) == 0 and group == 'all':
            continue
        if len(lists[group]['encoded_values']) > 0:
            encoded_data[group] = pd.DataFrame(np.concatenate(lists[group]['encoded_values']),
                                               # index=np.concatenate(lists[group]['labels']),
                                               ).round(4)
            encoded_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['labels']), columns=['labels']), encoded_data[group]), 1)
            encoded_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['domains']), columns=['batches']), encoded_data[group]), 1)
            encoded_data[group].index = np.concatenate(lists[group]['names'])
            encoded_batches[group] = np.concatenate(lists[group]['domains'])
            encoded_cats[group] = np.concatenate(lists[group]['classes'])
            encoded_names[group] = np.concatenate(lists[group]['names'])
        else:
            encoded_data[group] = pd.DataFrame(
                np.empty(shape=(0, encoded_data['train'].shape[1]), dtype='float')).round(4)
            encoded_batches[group] = np.array([])
            encoded_cats[group] = np.array([])
            encoded_names[group] = np.array([])

    rec_data = {}
    rec_batches = {}
    rec_cats = {}
    rec_names = {}
    for group in list(lists.keys()):
        if len(lists[group]['rec_values']) == 0 and group == 'all':
            continue
        if len(lists[group]['rec_values']) > 0:
            rec_data[group] = pd.DataFrame(np.concatenate(lists[group]['rec_values']),
                                           # index=np.concatenate(lists[group]['names']),
                                           columns=list(columns)  # + ['gender', 'age']
                                           ).round(4)

            rec_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['labels']), columns=['labels']), rec_data[group]), 1)
            rec_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['domains']), columns=['batches']), rec_data[group]), 1)
            rec_data[group].index = np.concatenate(lists[group]['names'])
            rec_batches[group] = np.concatenate(lists[group]['domains'])
            rec_cats[group] = np.concatenate(lists[group]['classes'])
            rec_names[group] = np.concatenate(lists[group]['names'])
        else:
            rec_data[group] = pd.DataFrame(np.empty(shape=(0, rec_data['train'].shape[1]), dtype='float')).round(4)
            rec_batches[group] = np.array([])
            rec_cats[group] = np.array([])
            rec_names[group] = np.array([])

    rec_data = {
        "inputs": rec_data,
        "cats": rec_cats,
        "batches": rec_batches,
    }
    enc_data = {
        "inputs": encoded_data,
        "cats": encoded_cats,
        "batches": encoded_batches,
    }
    try:
        rec_data['inputs']['all'].to_csv(f'{complete_log_path}/recs.csv')
        enc_data['inputs']['all'].to_csv(f'{complete_log_path}/encs.csv')
    except:
        pd.concat((rec_data['inputs']['train'], rec_data['inputs']['valid'], rec_data['inputs']['test'])).to_csv(
            f'{complete_log_path}/recs.csv')
        pd.concat((enc_data['inputs']['train'], enc_data['inputs']['valid'], enc_data['inputs']['test'])).to_csv(
            f'{complete_log_path}/encs.csv')
    return rec_data, enc_data


def get_optimizer(model, learning_rate, weight_decay, optimizer_type, momentum=0.9):
    """
    This function takes a model with learning rate, weight decay and optionally momentum and returns an optimizer object
    Args:
        model: The PyTorch model to optimize
        learning_rate: The optimizer's learning rate
        weight_decay: The optimizer's weight decay
        optimizer_type: The optimizer's type [adam or sgd]
        momentum:

    Returns:
        an optimizer object
    """
    # TODO Add more optimizers
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay
                                     # betas=(0.5, 0.9)
                                     )
    elif optimizer_type == 'radam':
        optimizer = torch.optim.RAdam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay,
                                     )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.RAdam(params=model.parameters(),
                                      lr=learning_rate,
                                      weight_decay=weight_decay,
                                      )
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay,
                                        momentum=0.9
                                        )
    else:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum
                                    )
    return optimizer


def to_categorical(y, num_classes):
    """
    1-hot encodes a tensor
    Args:
        y: values to encode
        num_classes: Number of classes. Length of the 1-encoder

    Returns:
        Tensor corresponding to the one-hot encoded classes
    """
    return torch.eye(num_classes, dtype=torch.int)[y]


def plot_confusion_matrix(cm, class_names, acc):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes

    Returns:
        The figure of the confusion matrix
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix (Acc: {np.round(np.mean(acc), 2)})")
    plt.colorbar()
    plt.grid(b=None)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    # labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def log_confusion_matrix(logger, epoch, lists, unique_labels, traces, mlops):
    """
    Logs the confusion matrix with tensorboardX (tensorboard for pytorch)
    Args:
        logger: TensorboardX logger
        epoch: The epoch scores getting logged
        lists: Dict containing lists of information on the run getting logged
        unique_labels: list of strings containing the unique labels (classes)
        traces: Dict of Dict of lists of scores

    Returns:
        Nothing. The logger doesn't need to be returned, it saves things on disk
    """
    for values in list(lists.keys())[1:]:
        # Calculate the confusion matrix.
        if len(lists[values]['preds']) == 0:
            continue
        preds = np.concatenate(lists[values]['preds']).argmax(1)
        classes = np.concatenate(lists[values]['classes'])
        cm = sklearn.metrics.confusion_matrix(classes, preds)
        figure = plot_confusion_matrix(cm, class_names=unique_labels[:len(np.unique(classes))],
                                       acc=traces[values]['acc'])
        if mlops == "tensorboard":
            logger.add_figure(f"CM_{values}_all", figure, epoch)
        elif mlops == "neptune":
            logger[f"CM_{values}_all"].upload(figure)
        elif mlops == "mlflow":
            mlflow.log_figure(figure, f"CM_{values}_all")
        plt.close(figure)
        del cm, figure


def save_roc_curve(model, x_test, y_test, unique_labels, name, binary, acc, mlops, epoch=None, logger=None):
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
        y_pred_proba = model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        roc_score = roc_auc_score(y_true=y_test, y_score=y_pred_proba)

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
                plt.savefig(f'{dirs}/ROC.png')
                stuck = False
            except:
                print('stuck...')
        plt.close()
    else:
        # Compute ROC curve and ROC area for each class
        from sklearn.preprocessing import label_binarize
        try:
            y_pred_proba = model.predict_proba(x_test)
        except:
            y_pred_proba = model.classifier(x_test)

        y_preds = model.predict_proba(x_test)
        n_classes = len(unique_labels) - 1  # -1 to remove QC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        classes = np.arange(len(unique_labels))

        roc_score = roc_auc_score(y_true=OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray(),
                                  y_score=y_preds,
                                  multi_class='ovr')
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
                plt.savefig(f'{dirs}/ROC.png')
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


def save_precision_recall_curve(model, x_test, y_test, unique_labels, name, binary, acc, mlops, epoch=None,
                                logger=None):
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    os.makedirs(f'{dirs}', exist_ok=True)
    if binary:
        y_pred_proba = model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
        aps = metrics.average_precision_score(y_test, y_pred_proba)
        # aps = metrics.roc_auc_score(y_true=y_test, y_score=y_pred_proba)

        # create ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='Precision-Recall curve (average precision score = %0.2f)' % aps)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (True Positive Rate) (TP/P)')
        plt.ylabel('Precision (TP/PP)')
        plt.title(f'Precision-Recall curve (acc={np.round(acc, 3)})')
        plt.legend(loc="lower right")
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/Precision-Recall.png')
                stuck = False
            except:
                print('stuck...')
        plt.close()

    else:
        # Compute Precision-Recall curve and Precision-Recall area for each class
        y_pred_proba = model.predict_proba(x_test)
        y_preds = model.predict(x_test)
        n_classes = len(unique_labels) - 1
        precisions = dict()
        recalls = dict()
        average_precision = dict()
        classes = np.arange(n_classes)
        y_test_bin = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
        for i in range(n_classes):
            y_true = y_test_bin[:, i]
            precisions[i], recalls[i], _ = metrics.precision_recall_curve(y_true, y_pred_proba[:, i], pos_label=1)
            average_precision[i] = metrics.average_precision_score(y_true=y_true, y_score=y_pred_proba[:, i])
        
        # roc for each class
        fig, ax = plt.subplots()
        # A "micro-average": quantifying score on all classes jointly
        precisions["micro"], recalls["micro"], _ = metrics.precision_recall_curve(
            y_test_bin.ravel(), y_pred_proba.ravel()
        )
        average_precision["micro"] = metrics.average_precision_score(y_test_bin, y_pred_proba,
                                                                     average="micro")  # sns.despine()
        display = PrecisionRecallDisplay(
            recall=recalls["micro"],
            precision=precisions["micro"],
            average_precision=average_precision["micro"],
        )
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/PRC.png')
                stuck = False
            except:
                print('stuck...')
        display.plot()
        fig = display.figure_
        if logger is not None:
            if mlops == 'tensorboard':
                logger.add_figure(name, fig, epoch)
            if mlops == 'neptune':
                logger[name].log(fig)
            if mlops == 'mlops':
                mlflow.log_figure(fig, name)
        # setup plot details
        colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
        viridis = cm.get_cmap('viridis', 256)
        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recalls["micro"],
            precision=precisions["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

        for i, color in zip(range(n_classes), viridis.colors):
            display = PrecisionRecallDisplay(
                recall=recalls[i],
                precision=precisions[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Extension of Precision-Recall curve to multi-class")

        fig = display.figure_
        plt.savefig(f'{dirs}/{name}_multiclass.png')
        if logger is not None:
            if mlops == 'tensorboard':
                logger.add_figure(f'{name}_multiclass', fig, epoch)
            if mlops == 'neptune':
                logger[f'{name}_multiclass'].log(fig)
            if mlops == 'mlflow':
                mlflow.log_figure(fig, f'{name}_multiclass')

        plt.close(fig)

    # return pr_auc


def get_empty_dicts():
    values = {
        "rec_loss": [],
        "dom_loss": [],
        "dom_acc": [],
        "set": {
            "lisi": {'inputs': {'domains': [], 'labels': [], 'set': []},
                     'enc': {'domains': [], 'labels': [], 'set': []}, 'rec': {'domains': [], 'labels': [], 'set': []}},
            "silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []},
                           'enc': {'domains': [], 'labels': [], 'set': []},
                           'rec': {'domains': [], 'labels': [], 'set': []}},
            "kbet": {'inputs': {'domains': [], 'labels': [], 'set': []},
                     'enc': {'domains': [], 'labels': [], 'set': []}, 'rec': {'domains': [], 'labels': [], 'set': []}},
            "adjusted_rand_score": {'inputs': {'domains': [], 'labels': [], 'set': []},
                                    'enc': {'domains': [], 'labels': [], 'set': []},
                                    'rec': {'domains': [], 'labels': [], 'set': []}},
            "batch_entropy": {'inputs': {'domains': [], 'labels': [], 'set': []},
                                    'enc': {'domains': [], 'labels': [], 'set': []},
                                    'rec': {'domains': [], 'labels': [], 'set': []}},
            "adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': [], 'set': []},
                                           'enc': {'domains': [], 'labels': [], 'set': []},
                                           'rec': {'domains': [], 'labels': [], 'set': []}},
            "F1_lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                        'rec': {'domains': [], 'labels': []}},
            "F1_silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                              'rec': {'domains': [], 'labels': []}},
            "F1_kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                        'rec': {'domains': [], 'labels': []}},
            "F1_adjusted_rand_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []},
                                       'rec': {'domains': [], 'labels': []}},
            "F1_adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': []},
                                              'enc': {'domains': [], 'labels': []},
                                              'rec': {'domains': [], 'labels': []}},
        },
    }
    best_values = {
        'rec_loss': 100,
        'dom_loss': 100,
        'dom_acc': 0,
    }
    best_lists = {}
    best_traces = {
        'rec_loss': [1000],
        'dom_loss': [1000],
        'dom_acc': [0],
    }

    for g in ['all', 'train', 'valid', 'test', 'calibration']:
        values[g] = {
            "closs": [],
            "dloss": [],
            "bic": [],
            "aic": [],
            "lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                     'rec': {'domains': [], 'labels': []}},
            "silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                           'rec': {'domains': [], 'labels': []}},
            "kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                     'rec': {'domains': [], 'labels': []}},
            "adjusted_rand_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []},
                                    'rec': {'domains': [], 'labels': []}},
            "adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': []},
                                           'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "batch_entropy": {'inputs': {'domains': [], 'labels': []},
                                           'enc': {'domains': [], 'labels': []}, 'rec': {'domains': [], 'labels': []}},
            "F1_lisi": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                        'rec': {'domains': [], 'labels': []}},
            "F1_silhouette": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                              'rec': {'domains': [], 'labels': []}},
            "F1_kbet": {'inputs': {'domains': [], 'labels': [], 'set': []}, 'enc': {'domains': [], 'labels': []},
                        'rec': {'domains': [], 'labels': []}},
            "F1_adjusted_rand_score": {'inputs': {'domains': [], 'labels': []}, 'enc': {'domains': [], 'labels': []},
                                       'rec': {'domains': [], 'labels': []}},
            "F1_adjusted_mutual_info_score": {'inputs': {'domains': [], 'labels': []},
                                              'enc': {'domains': [], 'labels': []},
                                              'rec': {'domains': [], 'labels': []}},
            "F1_batch_entropy": {'inputs': {'domains': [], 'labels': []},
                                              'enc': {'domains': [], 'labels': []},
                                              'rec': {'domains': [], 'labels': []}},
            "kld": [],
            "acc": [],
            "dom_acc": [],
            "top3": [],
            "mcc": [],
            "tpr": [],
            "tnr": [],
            "ppv": [],
            "npv": [],
        }
        best_traces[g] = {
            'closs': [100],
            'dloss': [100],
            'dom_acc': [0],
            'acc': [0],
            'top3': [0],
            'mcc': [0],
            "tpr": [],
            "tnr": [],
            "ppv": [],
            "npv": [],
        }

        for m in ['acc', 'top3', 'mcc', 'dom_acc', 'tpr', 'tnr', 'ppv', 'npv']:
            best_values[f"{g}_{m}"] = 0
        best_values[f"{g}_loss"] = 100
        best_lists[g] = {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'times': [],
            'classes': [],
            'labels': [],
            'domains': [],
            'encoded_values': [],
            'enc_values': [],
            'age': [],
            'gender': [],
            'atn': [],
            'names': [],
            'rec_values': [],
            'inputs': [],
        }

    return values, best_values, best_lists, best_traces


def get_empty_traces():
    traces = {
        'rec_loss': [],
        'dom_loss': [],
        'dom_acc': [],
    }
    lists = {}
    for g in ['all', 'train', 'valid', 'test']:
        lists[g] = {
            'set': [],
            'preds': [],
            'domain_preds': [],
            'probs': [],
            'lows': [],
            'cats': [],
            'times': [],
            'classes': [],
            'domains': [],
            'dloss': [],
            'dom_acc': [],
            'labels': [],
            'old_labels': [],
            'encoded_values': [],
            'enc_values': [],
            'age': [],
            'gender': [],
            'atn': [],
            'names': [],
            'inputs': [],
            'rec_values': [],
            'subcenters': [],
        }
        traces[g] = {
            'dloss': [],
            'closs': [],
            'dom_acc': [],
            'acc': [],
            'top3': [],
            'mcc': [],
            'tpr': [],
            'tnr': [],
            'ppv': [],
            'npv': []
        }

    return lists, traces


def log_traces(traces, values):
    if len(traces['dom_loss']) > 0:
        values['dom_loss'] += [np.mean(traces['dom_loss'])]
    if len(traces['dom_acc']) > 0:
        values['dom_acc'] += [np.mean(traces['dom_acc'])]
    if len(traces['rec_loss']) > 0:
        values['rec_loss'] += [np.mean(traces['rec_loss'])]

    if len(traces['train']['closs']) > 0:
        for g in list(traces.keys())[3:]:
            values[g]['dloss'] += [np.mean(traces[g]['dloss'])]
            values[g]['closs'] += [np.mean(traces[g]['closs'])]
            values[g]['dom_acc'] += [np.mean(traces[g]['dom_acc'])]
            values[g]['acc'] += [np.mean(traces[g]['acc'])]
            values[g]['top3'] += [np.mean(traces[g]['top3'])]
            values[g]['mcc'] += traces[g]['mcc']
            values[g]['tpr'] += traces[g]['tpr']
            values[g]['tnr'] += traces[g]['tnr']
            values[g]['ppv'] += traces[g]['ppv']
            values[g]['npv'] += traces[g]['npv']

    return values


def get_best_loss_from_tb(event_acc):
    values = {}
    # plugin_data = event_acc.summary_metadata['_hparams_/session_start_info'].plugin_data.content
    # plugin_data = plugin_data_pb2.HParamsPluginData.FromString(plugin_data)
    # for name in event_acc.summary_metadata.keys():
    #     if name not in ['_hparams_/experiment', '_hparams_/session_start_info']:
    best_closs = event_acc.Tensors('valid/loss')
    best_closs = tf.make_ndarray(best_closs[0].tensor_proto).item()

    return best_closs


def get_best_acc_from_tb(event_acc):
    values = {}
    # plugin_data = event_acc.summary_metadata['_hparams_/session_start_info'].plugin_data.content
    # plugin_data = plugin_data_pb2.HParamsPluginData.FromString(plugin_data)
    # for name in event_acc.summary_metadata.keys():
    #     if name not in ['_hparams_/experiment', '_hparams_/session_start_info']:
    best_closs = event_acc.Tensors('valid/acc')
    best_closs = tf.make_ndarray(best_closs[0].tensor_proto).item()

    return best_closs


def get_best_values(values, ae_only=False, n_agg=10):
    """
    As the name implies - this function gets the best values
    Args:
        values: Dict containing the values to be logged
        ae_only: Boolean to return the appropriate output

    Returns:

    """
    best_values = {}
    if ae_only:
        best_values = {
            'rec_loss':  values['rec_loss'][-1],
            'dom_loss': values['dom_loss'][-1],
            'dom_acc': values['rec_loss'][-1],
        }
        for g in ['train', 'valid', 'test']:
            for k in ['acc', 'mcc', 'top3', 'tpr', 'tnr', 'ppv', 'npv']:
                best_values[f'{g}_{k}'] = math.nan
                best_values[f'{g}_loss'] = math.nan


    else:
        if len(values['dom_loss']) > 0:
            best_values['dom_loss'] = values['dom_loss'][-1]
        if len(values['dom_acc']) > 0:
            best_values['dom_acc'] = values['dom_acc'][-1]
        if len(values['rec_loss']) > 0:
            best_values['rec_loss'] = values['rec_loss'][-1]
        for g in ['train', 'valid', 'test']:
            for k in ['acc', 'mcc', 'top3', 'tpr', 'tnr', 'ppv', 'npv']:
                if g == 'test':
                    best_values[f'{g}_{k}'] = values[g][k][-1]
                    best_values[f'{g}_loss'] = values[g]['closs'][-1]
                else:
                    best_values[f'{g}_{k}'] = np.mean(values[g][k][-n_agg:])
                    best_values[f'{g}_loss'] = np.mean(values[g]['closs'][-n_agg:])

    return best_values



def count_labels(arr):
    """
    Counts elements in array

    :param arr:
    :return:
    """
    elements_count = {}
    for element in arr:
        if element in elements_count:
            elements_count[element] += 1
        else:
            elements_count[element] = 1
    to_remove = []
    for key, value in elements_count.items():
        print(f"{key}: {value}")
        if value <= 2:
            to_remove += [key]

    return to_remove



def save_tensor(tensor_image, name="saved_image.png"):
    """
    Save a tensor image to disk.
    """
    # Normalize to [0,1]
    tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())

    # Convert to [0,255] (uint8)
    tensor_image = (tensor_image * 255).byte()

    # Convert from (C, H, W) → (H, W, C) for saving
    tensor_image = tensor_image.permute(1, 2, 0).cpu().numpy()

    # Convert to PIL Image
    image_pil = Image.fromarray(tensor_image)

    # Save the image
    image_pil.save(name) 

def get_best_params(run, best_vals):
    """
    Placeholder function to get the best parameters.
    Returns:
        dict: Best parameters.
    """
    best_params = {}
    best_params['params/lr'] = run['lr'].fetch()
    best_params['params/wd'] = run['wd'].fetch()
    best_params['params/margin'] = run['margin'].fetch()
    best_params['params/dmargin'] = run['dmargin'].fetch()
    best_params['params/smoothing'] = run['smooth'].fetch()
    best_params['params/is_transform'] = run['is_transform'].fetch()
    best_params['params/gamma'] = run['gamma'].fetch()
    best_params['params/dropout'] = run['dropout'].fetch()
    best_params['params/dist_fct'] = run['distance_fct'].fetch()
    best_params['model_name'] = run['model_name'].fetch()
    best_params['exp_id'] = run['exp_id'].fetch()
    best_params['is_stn'] = run['is_stn'].fetch()
    best_params['n_calibration'] = run['n_calibration'].fetch()
    best_params['distance_fct'] = run['distance_fct'].fetch()
    best_params['groupkfold'] = run['groupkfold'].fetch()
    best_params['dloss'] = run['dloss'].fetch()
    best_params['is_transform'] = run['is_transform'].fetch()
    best_params['lr'] = run['lr'].fetch()
    best_params['wd'] = run['wd'].fetch()
    best_params['margin'] = run['margin'].fetch()
    best_params['dmargin'] = run['dmargin'].fetch()
    best_params['smooth'] = run['smooth'].fetch()
    best_params['gamma'] = run['gamma'].fetch()
    best_params['dropout'] = run['dropout'].fetch()
    best_params['optimizer_type'] = run['optimizer_type'].fetch()
    best_params['foldername'] = run['foldername'].fetch()
    best_params['seed'] = run['seed'].fetch()
    best_params['n_negatives'] = run['n_negatives'].fetch()
    best_params['n_positives'] = run['n_positives'].fetch()
    best_params['n_epochs'] = run['n_epochs'].fetch()
    best_params['bs'] = run['bs'].fetch()
    best_params['weighted_sampler'] = run['weighted_sampler'].fetch()
    best_params['random_recs'] = run['random_recs'].fetch()
    best_params['prototypes_to_use'] = run['prototypes_to_use'].fetch()
    best_params['classif_loss'] = run['classif_loss'].fetch()
    best_params['complete_log_path'] = run['complete_log_path'].fetch()

    best_params['mcc/valid'] = best_vals['valid']['mcc'][-1]
    best_params['acc/valid'] = best_vals['valid']['acc'][-1]
    best_params['mcc/test'] = best_vals['test']['mcc'][-1]
    best_params['acc/test'] = best_vals['test']['acc'][-1]

    return best_params