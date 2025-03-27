NEPTUNE_API_TOKEN = "YOUR-API-KEY"
NEPTUNE_PROJECT_NAME = "YOUR-PROJECT-NAME"
NEPTUNE_MODEL_NAME = "YOUR-MODEL-NAME"

import matplotlib

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

import uuid
import shutil
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
import torch
from torch import nn
import os
from PIL import Image

from sklearn import metrics
from tensorboardX import SummaryWriter
from ax.service.managed_loop import optimize
from sklearn.metrics import matthews_corrcoef as MCC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchvision.models import resnet18, ResNet18_Weights
from otitenet.dataset import get_images_loaders
from otitenet.utils import get_optimizer, to_categorical, get_empty_dicts, get_empty_traces, \
    log_traces, get_best_values, add_to_logger, add_to_neptune, \
    add_to_mlflow
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from otitenet.utils import get_unique_labels
from otitenet.loggings import LogConfusionMatrix
import neptune
import mlflow
import itertools
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# from torchvision import transforms
from torchvision.transforms import v2 as transforms
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.autograd import Variable

from torch.autograd import Function
# from sklearn.preprocessing import label_binarize, OneHotEncoder

import torch.nn.functional as F
# from otitenet.SVHModule import SpatialTransformer

warnings.filterwarnings("ignore")

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Net(nn.Module):
    def __init__(self, device, n_cats, n_batches, do=0.3, is_stn=1):
        super(Net, self).__init__()
        self.is_stn = is_stn
        self.model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Dropout2d(do),
                                          nn.Conv2d(16, 32, kernel_size=5),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Dropout2d(do),
                                          nn.Conv2d(32, 64, kernel_size=4),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Dropout(do),
                                          nn.Conv2d(64, 128, kernel_size=4),
                                          nn.ReLU(True),
                                          # nn.Dropout2d(0.5)
                                          ).to(device)
        self.classifier = nn.Linear(128, n_cats).to(device)
        self.dann = nn.Linear(128, n_batches).to(device)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(16, 32, kernel_size=5),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(32, 64, kernel_size=4),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          ).to(device)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        ).to(device)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # self.fc_loc[2].bias.data.copy_(
        #     torch.tensor([0.2, 0, 0], dtype=torch.float)) #set scaling to start at 0.2 (~28/128)

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, inp):
        # transform the input
        if self.is_stn:
            inp = self.stn(inp)

        # Perform the usual forward pass
        enc = self.model(inp).squeeze()
        out = self.classifier(enc)
        reverse = ReverseLayerF.apply(enc, 1)
        domout = self.dann(reverse)

        return enc, F.log_softmax(domout, dim=1)


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
        # enc = model(x_test)
        # y_pred_proba = model.classifier(enc)[::, 1]
        # y_pred_proba = model(x_test)[::, 1]
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
                plt.savefig(f'{dirs}/ROC.png')
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


class LogConfusionMatrix:
    def __init__(self, complete_log_path):
        self.complete_log_path = complete_log_path
        self.preds = {'train': [], 'valid': [], 'test': []}
        self.classes = {'train': [], 'valid': [], 'test': []}
        self.real_classes = {'train': [], 'valid': [], 'test': []}
        self.cats = {'train': [], 'valid': [], 'test': []}
        self.encs = {'train': [], 'valid': [], 'test': []}
        self.recs = {'train': [], 'valid': [], 'test': []}
        self.gender = {'train': [], 'valid': [], 'test': []}
        self.age = {'train': [], 'valid': [], 'test': []}
        self.batches = {'train': [], 'valid': [], 'test': []}

    def add(self, lists):
        # n_mini_batches = int(1000 / lists['train']['preds'][0].shape[0])
        for group in list(self.preds.keys()):
            # Calculate the confusion matrix.
            if len(lists[group]['preds']) == 0:
                continue
            self.preds[group] += [np.concatenate(lists[group]['preds']).argmax(1)]
            self.classes[group] += [np.concatenate(lists[group]['classes'])]
            self.real_classes[group] += [np.concatenate(lists[group]['old_labels'])]
            # self.encs[group] += [np.concatenate(lists[group]['encoded_values'])]
            # try:
            #     self.recs[group] += [np.concatenate(lists[group]['rec_values'])]
            # except:
            #     pass
            self.cats[group] += [np.concatenate(lists[group]['cats'])]
            self.batches[group] += [np.concatenate(lists[group]['domains'])]

    def plot(self, logger, epoch, unique_labels, mlops):
        for group in ['train', 'valid', 'test']:
            preds = np.concatenate(self.preds[group])
            classes = np.concatenate(self.classes[group])
            real_classes = np.concatenate(self.real_classes[group])
            unique_real_classes = np.unique(real_classes)

            corrects = [1 if pred==label else 0 for pred, label in zip(preds, classes)]
            real_cats = np.array([np.argwhere(x==unique_real_classes).flatten()[0] for x in real_classes])
            real_preds = []
            for i, pred in enumerate(preds):
                if corrects[i] == 1:
                    real_preds += [unique_real_classes[real_cats[i]]]
                else:
                    if pred == 1:
                        real_preds += [unique_real_classes[0]]
                    else:
                        real_preds += ['Normal']
            cm = metrics.confusion_matrix(real_classes, real_preds)

            acc = np.mean([0 if pred != c else 1 for pred, c in zip(preds, classes)])

            # try:
            #     figure = plot_confusion_matrix(cm, class_names=unique_real_classes[:len(np.unique(self.classes['train']))], acc=acc)
            # except:
            figure = plot_confusion_matrix(cm, class_names=unique_real_classes, acc=acc)
            if mlops == "tensorboard":
                logger.add_figure(f"CM_{group}_all", figure, epoch)
            elif mlops == "neptune":
                logger[f"CM_{group}_all"].upload(figure)
            elif mlops == "mlflow":
                mlflow.log_figure(figure, f"CM_{group}_all.png")
                # logger[f"CM_{group}_all"].log(figure)
            plt.savefig(f"{self.complete_log_path}/cm/{group}_all.png") # , dpi=300)
            plt.close()

            cm = metrics.confusion_matrix(classes, preds)

            acc = np.mean([0 if pred != c else 1 for pred, c in zip(preds, classes)])

            # try:
            #     figure = plot_confusion_matrix(cm, class_names=unique_real_classes[:len(np.unique(self.classes['train']))], acc=acc)
            # except:
            figure = plot_confusion_matrix(cm, class_names=unique_labels, acc=acc)
            if mlops == "tensorboard":
                logger.add_figure(f"CM_{group}", figure, epoch)
            elif mlops == "neptune":
                logger[f"CM_{group}"].upload(figure)
            elif mlops == "mlflow":
                mlflow.log_figure(figure, f"CM_{group}.png")
                # logger[f"CM_{group}_all"].log(figure)
            plt.savefig(f"{self.complete_log_path}/cm/{group}.png") # , dpi=300)
            plt.close()
        # del cm, figure

    def get_rf_results(self, run, args):
        hparams_names = [x.name for x in rfc_space]
        enc_data = {name: {x: None for x in ['train', 'valid', 'test']} for name in ['cats', 'inputs', 'batches']}
        rec_data = {name: {x: None for x in ['train', 'valid', 'test']} for name in ['cats', 'inputs', 'batches']}
        for group in list(self.preds.keys()):
            enc_data['inputs'][group] = self.encs[group]
            enc_data['cats'][group] = self.cats[group]
            enc_data['batches'][group] = self.batches[group]
            rec_data['inputs'][group] = self.encs[group]
            rec_data['cats'][group] = self.cats[group]
            rec_data['batches'][group] = self.batches[group]

        train = Train("Reconstruction", RandomForestClassifier, rec_data, hparams_names,
                      self.complete_log_path,
                      args, run, ovr=0, binary=False, mlops='neptune')
        _ = gp_minimize(train.train, rfc_space, n_calls=20, random_state=1)
        train = Train("Encoded", RandomForestClassifier, enc_data, hparams_names, self.complete_log_path,
                      args, run, ovr=0, binary=False, mlops='neptune')
        _ = gp_minimize(train.train, rfc_space, n_calls=20, random_state=1)

    def reset(self):
        self.preds = {'train': [], 'valid': [], 'test': []}
        self.classes = {'train': [], 'valid': [], 'test': []}
        self.real_classes = {'train': [], 'valid': [], 'test': []}
        self.cats = {'train': [], 'valid': [], 'test': []}
        self.encs = {'train': [], 'valid': [], 'test': []}
        self.recs = {'train': [], 'valid': [], 'test': []}
        self.gender = {'train': [], 'valid': [], 'test': []}
        self.age = {'train': [], 'valid': [], 'test': []}
        self.batches = {'train': [], 'valid': [], 'test': []}


class Dataset1(Dataset):
    def __init__(self, data, unique_batches, unique_labels, names=None, labels=None, old_labels=None, batches=None, sets=None, transform=None, crop_size=-1,
                 add_noise=False, random_recs=False, triplet_dloss=False, triplet_loss=True):
        """

        Args:
            data: Contains a dict of data
            meta: array of meta data
            names: array or list of names
            labels: array or list of labels
            batches: array or list of batches
            sets: array or list of sets
            transform: transform to apply to the data
            crop_size: crop size to apply to the data
            add_noise: Whether to add noise to the data
            random_recs: Whether to sample random reconstructions
            triplet_dloss: Whether to use triplet loss of the domain
        """
        self.random_recs = random_recs
        try:
            self.samples = data.to_numpy()
        except:
            self.samples = data

        self.add_noise = add_noise
        self.names = names
        self.sets = sets
        self.transform = transform
        self.crop_size = crop_size
        self.labels = labels
        self.cats = np.array([np.argwhere(label==unique_labels).flatten()[0] for label in labels])
        self.old_labels = old_labels
        self.unique_labels = unique_labels
        self.batches = batches
        self.unique_batches = np.unique(batches)
        self.labels_inds = {label: [i for i, x in enumerate(labels) if x == label] for label in self.unique_labels}
        self.batches_inds = {batch: [i for i, x in enumerate(batches) if x == batch] for batch in self.unique_batches}
        # self.labels_data = {label: data[labels_inds[label]] for label in self.unique_labels}
        # try:
        #     self.batches_data = {batch: data[batches_inds[batch]] for batch in batches}
        # except:
        #     pass
        self.n_labels = {label: len(self.labels_inds[label]) for label in labels}
        self.n_batches = {batch: len(self.batches_inds[batch]) for batch in batches}
        self.triplet_dloss = triplet_dloss
        self.triplet_loss = triplet_loss

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta_to_rec = None
        if self.labels is not None:
            cat = self.cats[idx]
            label = self.labels[idx]
            try:
                old_label = self.old_labels[idx]
            except:
                pass
            batch = self.batches[idx]
            # set = self.sets[idx]
            try:
                name = self.names[idx]
            except:
                name = str(self.names.iloc[idx])

        else:
            label = None
            cat = None
            old_label = None
            batch = None
            name = None
        if self.triplet_loss:
            # to_rec = self.labels_data[label][np.random.randint(0, self.n_labels[label])].copy()
            to_rec = self.samples[self.labels_inds[label][np.random.randint(0, self.n_labels[label])]].copy()
            not_label = None
            while not_label == label or not_label is None:
                not_label = self.unique_labels[np.random.randint(0, len(self.unique_labels))].copy()
            not_to_rec = self.samples[self.labels_inds[not_label][np.random.randint(0, self.n_labels[not_label])]].copy()
        else:
            to_rec = self.samples[idx].copy()
            not_to_rec = self.samples[idx].copy()
        if (self.triplet_dloss == 'revTriplet' or self.triplet_dloss == 'inverseTriplet') and len(self.unique_batches) > 1 and len(self.n_batches) > 1:
            not_batch_label = None
            while not_batch_label == batch or not_batch_label is None:
                not_batch_label = self.unique_batches[np.random.randint(0, len(self.unique_batches))].copy()
            pos_ind = np.random.randint(0, self.n_batches[batch])
            try:
                neg_ind = np.random.randint(0, self.n_batches[not_batch_label])
            except:
                pass
            pos_batch_sample = self.samples[self.batches_inds[batch][pos_ind]].copy()
            neg_batch_sample = self.samples[self.batches_inds[not_batch_label][neg_ind]].copy()
        else:
            pos_batch_sample = self.samples[idx].copy()
            neg_batch_sample = self.samples[idx].copy()
        x = self.samples[idx]
        if self.crop_size != -1:
            max_start_crop = x.shape[1] - self.crop_size
            ran = np.random.randint(0, max_start_crop)
            x = torch.Tensor(x)[:, ran:ran + self.crop_size]  # .to(device)
        if self.transform:
            x = self.transform(x).squeeze()
            to_rec = self.transform(to_rec).squeeze()
            not_to_rec = self.transform(not_to_rec).squeeze()
            pos_batch_sample = self.transform(pos_batch_sample).squeeze()
            neg_batch_sample = self.transform(neg_batch_sample).squeeze()

        if self.add_noise:
            if np.random.random() > 0.5:
                x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        batch = np.argwhere(batch == self.unique_batches).flatten()
        return x, name, cat, label, old_label, batch, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample


def get_best_values(values):
    """
    As the name implies - this function gets the best values
    Args:
        values: Dict containing the values to be logged
        ae_only: Boolean to return the appropriate output

    Returns:

    """
    best_values = {}
    if len(values['dom_loss']) > 0:
        best_values['dom_loss'] = values['dom_loss'][-1]
    if len(values['dom_acc']) > 0:
        best_values['dom_acc'] = values['dom_acc'][-1]
    if len(values['rec_loss']) > 0:
        best_values['rec_loss'] = values['rec_loss'][-1]
    for g in ['train', 'valid', 'test']:
        for k in ['acc', 'mcc', 'top3']:
            if g == 'test':
                best_values[f'{g}_{k}'] = values[g][k][-1]
                best_values[f'{g}_loss'] = values[g]['closs'][-1]
            else:
                best_values[f'{g}_{k}'] = np.mean(values[g][k])
                best_values[f'{g}_loss'] = np.mean(values[g]['closs'])

    return best_values

def get_images(path):
    pngs = []
    labels = []
    names = []
    datasets = []
    groups = []

    infos = pd.read_csv(f"{path}/infos.csv")
    for i, row in infos.iterrows():
        im = row['name']
        if im == ".DS_Store":
            continue

        png = Image.open(f"{path}/{im}").convert('RGB')
        png = transforms.Resize((64, 64))(png)
        try:
            pngs += [np.array(png)]
        except:
            continue
        labels += [row['label']]
        names += [row['name']]
        datasets += [row['dataset']]
        groups += [row['group']]

    return np.array(pngs), np.array(names), np.array(labels), np.array(datasets)


def get_images_loaders(data, random_recs, weighted_sampler, is_transform, samples_weights, unique_batches, 
                       unique_labels, triplet_dloss, bs=64, device='cuda'):
    """

    Args:
        data:
        ae:
        classifier:

    Returns:

    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomApply(
        #     nn.ModuleList([transforms.ColorJitter()])
        # , p=0.5),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-10, 10)),
        # transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(180, 180))])),
        # transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(270, 270))])),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        # transforms.RandomApply(nn.ModuleList([
        #     transforms.RandomAffine(
        #         degrees=10,
        #         translate=(.1, .1),
        #         # scale=(0.9, 1.1),
        #         shear=(.01, .01),
        #         interpolation=transforms.InterpolationMode.BILINEAR
        #     )
        # ]), p=0.5),
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 0.03))
        #     ]), p=0.5),
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.RandomResizedCrop(
        #             size=224,
        #             scale=(0.8, 1.),
        #             ratio=(0.8, 1.2)
        #         )
        #     ]), p=0.5),
        torchvision.transforms.Normalize(0.5, 0.5),
        # transforms.Lambda(lambda x: x.broadcast_to(3, x.shape[1], x.shape[2]))
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # transforms.Grayscale(),
        # transforms.Lambda(lambda x: x.broadcast_to(3, x.shape[1], x.shape[2]))
        # transforms.Resize(224),
        torchvision.transforms.Normalize(0.5, 0.5),
    ])

    if is_transform:
        transform_train = transform_train
    else:
        transform_train = transform

    train_set = Dataset1(data=data['inputs']['train'],
                         unique_batches=unique_batches,
                         unique_labels=unique_labels,
                         names=data['names']['train'],
                         labels=data['labels']['train'],
                         old_labels=data['old_labels']['train'],
                         batches=data['batches']['train'],
                         sets=None,
                         transform=transform_train, crop_size=-1, random_recs=random_recs,
                         triplet_dloss=triplet_dloss
                         )
    valid_set = Dataset1(data['inputs']['valid'],
                         unique_batches,
                         unique_labels,
                         data['names']['valid'],
                         data['labels']['valid'],
                         data['old_labels']['valid'],
                         data['batches']['valid'],
                         sets=None,
                         transform=transform, crop_size=-1, random_recs=False, 
                         triplet_dloss=triplet_dloss
                         )
    test_set = Dataset1(data['inputs']['test'],
                        unique_batches,
                         unique_labels,
                        data['names']['test'],
                        data['labels']['test'],
                        data['old_labels']['test'],
                        data['batches']['test'],
                        sets=None,
                        transform=transform, crop_size=-1, random_recs=False,
                        triplet_dloss=triplet_dloss
                        )
    if weighted_sampler:
        train_loader = DataLoader(train_set,
                            sampler=WeightedRandomSampler(samples_weights['train'], len(samples_weights['train']),
                                                          replacement=True),
                            batch_size=bs,
                            num_workers = 8,
                            prefetch_factor = 8,
                            pin_memory=True,
                            drop_last=True,
                            persistent_workers=True
                            )
    else:
        train_loader = DataLoader(train_set,
                            shuffle=True,
                            batch_size=bs,
                            num_workers = 8,
                            prefetch_factor = 8,
                            pin_memory=True,
                            drop_last=True,
                            persistent_workers=True
                            )
    loaders = {
        'train': train_loader,
        'test': DataLoader(test_set,
                           # sampler=WeightedRandomSampler(samples_weights['test'], sum(samples_weights['test']),
                           #                               replacement=False),
                           batch_size=bs,
                            num_workers = 8,
                            prefetch_factor = 8,
                           pin_memory=True,
                           drop_last=False,
                           persistent_workers=True
                           ),
        'valid': DataLoader(valid_set,
                            # sampler=WeightedRandomSampler(samples_weights['valid'], sum(samples_weights['valid']),
                            #                               replacement=False),
                            batch_size=bs,
                            num_workers = 8,
                            prefetch_factor = 8,
                            pin_memory=True,
                            drop_last=False,
                            persistent_workers=True

                            ),
    }
    all_set = Dataset1(data['inputs']['all'],
                       unique_batches,
                         unique_labels,
                       data['names']['all'],
                       data['labels']['all'],
                       data['old_labels']['all'],
                       data['batches']['all'],
                       sets=None,
                       transform=transform_train, crop_size=-1, random_recs=False,
                       triplet_dloss=triplet_dloss
                       )

    loaders['all'] = DataLoader(all_set,
                                # num_workers=0,
                                shuffle=True,
                                batch_size=bs,
                                num_workers = 8,
                                prefetch_factor = 8,
                                pin_memory=True,
                                drop_last=True,
                                persistent_workers=True
                                )

    return loaders




class GetData:
    def __init__(self, path, args) -> None:
        self.args = args
        self.data = {}
        self.unique_labels = np.array([])
        for info in ['inputs', 'names', 'old_labels', 'labels', 'cats', 'batches']:
            self.data[info] = {}
            for group in ['all', 'train', 'test', 'valid']:
                self.data[info][group] = np.array([])
        self.data['inputs']['all'], self.data['names']['all'], self.data['labels']['all'], self.data['batches']['all'] = get_images(path)
        self.data['old_labels']['all'] = self.data['labels']['all']
        new_labels = []
        for label in self.data['labels']['all']:
            if args.task == 'otitis':
                if label in ['Chronic otitis media', 'Chornic', 'Effusion', 'Aom', 'AOM', 'CSOM', 'OtitExterna', 'OtitisEksterna']:
                    new_labels += ['Otitis']
                else:
                    new_labels += ['notOtitis']
            elif args.task == 'multi':
                if label in ['Chronic otitis media', 'Chornic', 'Effusion', 'Aom', 'AOM', 'CSOM', 'OtitExterna', 'OtitisEksterna']:
                    new_labels += ['Otitis']
                elif label == 'Normal':
                    new_labels += ['Normal']
                else:
                    new_labels += ['NotNormal']
            elif args.task == 'notNormal':
                if label == "Normal":
                    new_labels += ['Normal']
                else:
                    new_labels += ['NotNormal']
            else:
                exit('WRONG TASK NAME: Otitis or notNormal')

        self.data['labels']['all'] = np.array(new_labels)
        # self.data['labels']['all'] = np.array([x if x == 'Normal' else 'NotNormal' for x in self.data['labels']['all']])
        # self.data['labels']['all'] = np.array(['otitis' if x in ['Chronic otitis media', 'Aom', 'Chornic', 'Effusion'] else 'NotOtitis' for x in self.data['labels']['all']])
        self.unique_labels = get_unique_labels(self.data['labels']['all'])
        self.unique_old_labels = get_unique_labels(self.data['old_labels']['all'])
        self.unique_batches = get_unique_labels(self.data['batches']['all'])
        self.data['cats']['all'] = np.array([np.argwhere(x == self.unique_labels).flatten() for x in self.data['labels']['all']]).flatten()
        self.data['cats']['train'] = np.array([np.argwhere(x == self.unique_labels).flatten() for x in self.data['labels']['all']]).flatten()
        self.data['inputs']['train'], self.data['names']['train'], self.data['labels']['train'], self.data['batches']['train'] =\
            self.data['inputs']['all'].copy(), self.data['names']['all'].copy(), self.data['labels']['all'].copy(), self.data['batches']['all'].copy()
        self.data['old_labels']['train'] = self.data['old_labels']['all'].copy()
        if not args.groupkfold:
            self.data, self.unique_labels, self.unique_batches = self.split('test', args.groupkfold, args.seed)
        else:
            self.split_deterministic()
    
    def get_variables(self):
        return self.data, self.unique_labels, self.unique_batches

    def split(self, group, groupkfold, seed):
        if groupkfold:
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            train_nums = np.arange(0, len(self.data['labels']['train']))
            # Remove samples from unwanted batches
            splitter = skf.split(train_nums, self.data['labels']['train'], self.data['batches']['train'])
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            train_nums = np.arange(0, len(self.data['labels']['train']))
            splitter = skf.split(train_nums, self.data['labels']['train'])

        train_inds, test_inds = splitter.__next__()
        # _, test_inds = splitter.__next__()
        # train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

        self.data['inputs']['train'], self.data['inputs'][group] = self.data['inputs']['train'][train_inds], \
            self.data['inputs']['train'][test_inds]
        self.data['labels']['train'], self.data['labels'][group] = self.data['labels']['train'][train_inds], \
            self.data['labels']['train'][test_inds]
        self.data['old_labels']['train'], self.data['old_labels'][group] = self.data['old_labels']['train'][train_inds], \
            self.data['old_labels']['train'][test_inds]
        self.data['names']['train'], self.data['names'][group] = self.data['names']['train'][train_inds], \
            self.data['names']['train'][test_inds]
        self.data['cats']['train'], self.data['cats'][group] = self.data['cats']['train'][train_inds], \
            self.data['cats']['train'][test_inds]
        self.data['batches']['train'], self.data['batches'][group] = self.data['batches']['train'][train_inds], \
            self.data['batches']['train'][test_inds]

        return self.data, self.unique_labels, self.unique_batches

    def split_deterministic(self):
        train_inds = np.argwhere(self.data['batches']['train'] == 'Banque_Comert_Turquie_2020_jpg').flatten()
        valid_inds = np.argwhere(self.data['batches']['train'] == 'Banque_Viscaino_Chili_2020').flatten()
        test_inds = np.argwhere(self.data['batches']['train'] == 'Banque_Calaman_USA_2020_trie_CM').flatten()

        self.data['inputs']['train'], self.data['inputs']['valid'], self.data['inputs']['test'] = self.data['inputs']['train'][train_inds], \
            self.data['inputs']['train'][valid_inds], self.data['inputs']['train'][test_inds]
        self.data['labels']['train'], self.data['labels']['valid'], self.data['labels']['test'] = self.data['labels']['train'][train_inds], \
            self.data['labels']['train'][valid_inds], self.data['labels']['train'][test_inds]
        self.data['old_labels']['train'], self.data['old_labels']['valid'], self.data['old_labels']['test'] = self.data['old_labels']['train'][train_inds], \
            self.data['old_labels']['train'][valid_inds], self.data['old_labels']['train'][test_inds]
        self.data['names']['train'], self.data['names']['valid'], self.data['names']['test'] = self.data['names']['train'][train_inds], \
            self.data['names']['train'][valid_inds], self.data['names']['train'][test_inds]
        self.data['cats']['train'], self.data['cats']['valid'], self.data['cats']['test'] = self.data['cats']['train'][train_inds], \
            self.data['cats']['train'][valid_inds], self.data['cats']['train'][test_inds]
        self.data['batches']['train'], self.data['batches']['valid'], self.data['batches']['test'] = self.data['batches']['train'][train_inds], \
            self.data['batches']['train'][valid_inds], self.data['batches']['train'][test_inds]

        return self.data, self.unique_labels, self.unique_batches

def get_distance_fct(distance_fct):
    if distance_fct == 'euclidean':
        dist_fct = nn.PairwiseDistance(p=2)
    elif distance_fct == 'cosine':
        dist_fct = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    elif distance_fct == 'manhattan':
        dist_fct = nn.PairwiseDistance(p=1)
    elif distance_fct == 'chebyshev':
        dist_fct = nn.PairwiseDistance(p=float('inf'))
    elif distance_fct == 'minkowski':
        dist_fct = nn.PairwiseDistance(p=3)
    elif distance_fct == 'SNR':
        dist_fct = lambda x, y: torch.var(x - y) / torch.var(x)
    return dist_fct


class TrainAE:

    def __init__(self, args, path, fix_thres=-1, load_tb=False, log_metrics=False, keep_models=True, log_inputs=True,
                 log_plots=False, log_tb=False, log_neptune=False, log_mlflow=True, groupkfold=True):
        """

        Args:
            args: contains multiple arguments passed in the command line
            log_path: Path where the tensorboard logs are saved
            path: Path to the data (in .csv format)
            fix_thres: If 1 > fix_thres >= 0 then the threshold is fixed to that value.
                       any other value means the threshold won't be fixed and will be
                       learned as an hyperparameter
            load_tb: If True, loads previous runs already saved
        """
        self.hparams_names = None
        self.best_acc = 0
        self.best_best_mcc = -1
        self.best_closs_final = np.inf
        self.logged_inputs = False
        self.log_tb = log_tb
        self.log_neptune = log_neptune
        self.log_mlflow = log_mlflow
        self.args = args
        self.path = path
        self.log_metrics = log_metrics
        self.log_plots = log_plots
        self.log_inputs = log_inputs
        self.keep_models = keep_models
        self.fix_thres = fix_thres
        self.load_tb = load_tb
        self.groupkfold = groupkfold
        self.foldername = None

        self.verbose = 1

        self.n_cats = None
        self.data = None
        self.unique_labels = None
        self.unique_batches = None
        self.triplet_loss = None

    def make_samples_weights(self):
        self.n_batches = len(
            set(np.concatenate((
            self.data['batches']['train'],
            self.data['batches']['valid'],
            self.data['batches']['test']
            )))
        )
        self.class_weights = {
            label: 1 / (len(np.where(label == self.data['labels']['train'])[0]) /
                        self.data['labels']['train'].shape[0])
            for label in self.unique_labels if
            label in self.data['labels']['train']}
        self.unique_unique_labels = list(self.class_weights.keys())

        self.samples_weights = {
            group: [self.class_weights[label] for name, label in
                    zip(self.data['names'][group],
                        self.data['labels'][group])] if group == 'train' else [
                1 for name, label in
                zip(self.data['names'][group], self.data['labels'][group])] for group in
            ['train', 'valid', 'test']}
        self.n_cats = len(self.unique_labels)
        self.scaler = None
        self.celoss = nn.CrossEntropyLoss()

    def train(self, params):
        """

        Args:
            params: Contains the hyperparameters to be optimized

        Returns:
            best_closs: The best classification loss on the valid set

        """
        start_time = datetime.now()
        # Fixing the hyperparameters that are not optimized

        print(params)

        # Assigns the hyperparameters getting optimized
        margin = params['margin']
        smooth = params['smoothing']
        # scale = params['scaler']
        wd = params['wd']
        lr = params['lr']
        if self.args.dloss not in ['revTriplet', 'revDANN', 'DANN',
                                    'inverseTriplet', 'normae'] or 'gamma' not in params:
            # gamma = 0 will ensure DANN is not learned
            params['gamma'] = 0
        gamma = params['gamma']
        params['dropout'] = 0
        dropout = params['dropout']

        optimizer_type = 'adam'
        dist_fct = get_distance_fct(self.args.distance_fct)
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=dist_fct, margin=margin, swap=True)
        # self.log_path is where tensorboard logs are saved
        self.foldername = str(uuid.uuid4())

        if self.log_mlflow:
            mlflow.set_experiment(
                self.args.exp_id,
            )
            try:
                mlflow.start_run()
            except:
                mlflow.end_run()
                mlflow.start_run()

            mlflow.log_param('margin', margin)
            mlflow.log_param('smooth', smooth)
            mlflow.log_param('wd', wd)
            mlflow.log_param('lr', lr)
            mlflow.log_param('gamma', gamma)
            mlflow.log_param('dropout', dropout)
            mlflow.log_param('optimizer_type', optimizer_type)
            mlflow.log_param('dloss', self.args.dloss)
            mlflow.log_param('foldername', self.foldername)
            mlflow.log_param('augmented', self.args.is_transform)
            mlflow.log_param('classif_loss', self.args.classif_loss)
            mlflow.log_param('task', self.args.task)
            mlflow.log_param('is_stn', self.args.is_stn)
            mlflow.log_param('n_calibration', self.args.n_calibration)
            mlflow.log_param('distance_fct', self.args.distance_fct)

        self.complete_log_path = f'logs/{self.args.task}/{self.foldername}'
        loggers = {'cm_logger': LogConfusionMatrix(self.complete_log_path)}
        print(f'See results using: tensorboard --logdir={self.complete_log_path} --port=6006')

        hparams_filepath = self.complete_log_path + '/hp'
        os.makedirs(hparams_filepath, exist_ok=True)
        self.args.model_name = 'ae_classifier'
        seed = 0
        epoch = 0
        best_closses = []
        best_mccs = []
        self.best_loss = np.inf
        self.best_closs = np.inf
        self.best_acc = 0
        data_getter = GetData(self.path, self.args)
        event_acc = EventAccumulator(hparams_filepath)
        event_acc.Reload()

        if self.log_neptune:
            # Create a Neptune run object
            run = neptune.init_run(
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
            )  # your credentials
            model = neptune.init_model_version(
                model=NEPTUNE_MODEL_NAME,
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
                # your credentials
            )
            run["dataset"].track_files(f"{self.path}/{self.args.csv_file}")
            run["metadata"].track_files(
                f"{self.path}/subjects_experiment_ATN_verified_diagnosis.csv"
            )
            # Track metadata and hyperparameters by assigning them to the run
            model["inputs_type"] = run["inputs_type"] = self.args.csv_file.split(".csv")[0]
            model["best_unique"] = run["best_unique"] = self.args.best_features_file.split(".tsv")[0]
            model["parameters"] = run["parameters"] = params
            model["csv_file"] = run["csv_file"] = self.args.csv_file
            model["model_name"] = run["model_name"] = 'resnet18'
            model["groupkfold"] = run["groupkfold"] = self.args.groupkfold
            model["foldername"] = run["foldername"] = self.foldername
            model["dataset_name"] = run["dataset_name"] = self.args.dataset
        else:
            model = None
            run = None


        for h in range(self.args.n_repeats):
            print("Repeat:", h)
            self.best_mcc = -1
            if not self.args.groupkfold:
                self.data, self.unique_labels, self.unique_batches = data_getter.split('valid', self.args.groupkfold, seed=1+h)
            else:
                self.data, self.unique_labels, self.unique_batches = data_getter.get_variables()
            # self.unique_batches = np.unique(self.data['batches']['all'])

            self.make_samples_weights()

            if self.args.n_calibration > 0:
                # take a few samples from train to be used for calibration. The indices
                for group in ['valid', 'test']:
                    indices = {label: np.argwhere(self.data['labels'][group] == label).flatten() for label in self.unique_labels}
                    calib_inds = np.concatenate([np.random.choice(indices[label], int(self.args.n_calibration/len(self.unique_labels)), replace=False) for label in self.unique_labels])

                    # add them to train
                    self.data['inputs']['train'] = np.concatenate((self.data['inputs']['train'], self.data['inputs'][group][calib_inds]))
                    self.data['labels']['train'] = np.concatenate((self.data['labels']['train'], self.data['labels'][group][calib_inds]))
                    self.data['old_labels']['train'] = np.concatenate((self.data['old_labels']['train'], self.data['old_labels'][group][calib_inds]))
                    self.data['names']['train'] = np.concatenate((self.data['names']['train'], self.data['names'][group][calib_inds]))
                    self.data['cats']['train'] = np.concatenate((self.data['cats']['train'], self.data['cats'][group][calib_inds]))
                    self.data['batches']['train'] = np.concatenate((self.data['batches']['train'], self.data['batches'][group][calib_inds]))
                    # remove them from test
                    self.data['inputs'][group] = np.delete(self.data['inputs'][group], calib_inds, axis=0)
                    self.data['labels'][group] = np.delete(self.data['labels'][group], calib_inds, axis=0)
                    self.data['old_labels'][group] = np.delete(self.data['old_labels'][group], calib_inds, axis=0)
                    self.data['names'][group] = np.delete(self.data['names'][group], calib_inds, axis=0)
                    self.data['cats'][group] = np.delete(self.data['cats'][group], calib_inds, axis=0)
                    self.data['batches'][group] = np.delete(self.data['batches'][group], calib_inds, axis=0)

            print("Train Batches:", np.unique(self.data['batches']['train']))
            print("Valid Batches:", np.unique(self.data['batches']['valid']))
            print("Test Batches:", np.unique(self.data['batches']['test']))
            # event_acc is used to verify if the hparams have already been tested. If they were,
            # the best classification loss is retrieved and we go to the next trial
            # If thres > 0, features that are 0 for a proportion of samples smaller than thres are removed
            # data = self.keep_good_features(thres)

            # Transform the data with the chosen scaler
            # data = copy.deepcopy(self.data)
            # data, self.scaler = scale_data(scale, data, self.args.device)
            loaders = get_images_loaders(data=self.data,
                                         random_recs=self.args.random_recs,
                                            weighted_sampler=self.args.weighted_sampler,
                                         is_transform=self.args.is_transform,
                                         samples_weights=self.samples_weights,
                                         unique_batches=self.unique_batches,
                                         unique_labels=self.unique_labels,
                                         triplet_dloss=self.args.dloss, bs=self.args.bs)
            # Initialize model
            model = Net(self.args.device, self.n_cats, self.n_batches, is_stn=self.args.is_stn)
            loggers['logger_cm'] = SummaryWriter(f'{self.complete_log_path}/cm')
            loggers['logger'] = SummaryWriter(f'{self.complete_log_path}/traces')
            sceloss = nn.CrossEntropyLoss(label_smoothing=smooth)
            optimizer_model = get_optimizer(model, lr, wd, optimizer_type)

            values, best_values, _, best_traces = get_empty_dicts()
            early_stop_counter = 0
            best_vals = values

            for epoch in range(0, self.args.n_epochs):
                if early_stop_counter >= self.args.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.', epoch)
                    break
                lists, traces = get_empty_traces()
                model.train()
                if args.dloss == 'no':
                    train_groups = ['train']
                else:
                    train_groups = ['all', 'train']
                for group in train_groups:
                    _, _, _ = self.loop(group, optimizer_model, model,  gamma, loaders[group], lists, traces)

                model.eval()
                for group in ["valid", "test"]:
                    _, _, _ = self.predict(group, model, loaders[group], lists, traces)

                # traces = self.get_mccs(lists, traces)
                # loggers['cm_logger'].add(lists)
                values = log_traces(traces, values)
                if self.log_tb:
                    try:
                        add_to_logger(values, loggers['logger'], epoch)
                    except:
                        print("Problem with add_to_logger!")
                if self.log_neptune:
                    add_to_neptune(values, run, epoch)
                if self.log_mlflow:
                    add_to_mlflow(values, epoch)
                if values['valid']['mcc'][-1] > self.best_mcc:
                    print(f"Best Classification Mcc Epoch {epoch}, "
                            f"Acc: test: {values['test']['acc'][-1]}, valid: {values['valid']['acc'][-1]}"
                            f"Mcc: test: {values['test']['mcc'][-1]}, valid: {values['valid']['mcc'][-1]}"
                            f"Classification "
                            f" valid loss: {values['valid']['closs'][-1]},"
                            f" test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                    self.best_mcc = values['valid']['mcc'][-1]
                    torch.save(model.state_dict(), f'{self.complete_log_path}/model.pth')
                    best_values = get_best_values(values.copy())
                    best_vals = values.copy()
                    best_vals['rec_loss'] = self.best_loss
                    # best_vals['dom_loss'] = self.best_dom_loss
                    # best_vals['dom_acc'] = self.best_dom_acc
                    early_stop_counter = 0
                else:
                    m = np.mean(values['valid']['mcc'])
                    print(m, self.best_mcc)

                if values['valid']['acc'][-1] > self.best_acc:
                    print(f"Best Classification Acc Epoch {epoch}, "
                            f"Acc: {values['test']['acc'][-1]}"
                            f"Mcc: {values['test']['mcc'][-1]}"
                            f"Classification "
                            f" valid loss: {values['valid']['closs'][-1]},"
                            f" test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")

                    self.best_acc = values['valid']['acc'][-1]
                    early_stop_counter = 0

                if values['valid']['closs'][-1] < self.best_closs:
                    print(f"Best Classification Loss Epoch {epoch}, "
                            f"Acc: {values['test']['acc'][-1]} "
                            f"Mcc: {values['test']['mcc'][-1]} "
                            f"Classification "
                            f"valid loss: {values['valid']['closs'][-1]}, "
                            f"test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                    self.best_closs = values['valid']['closs'][-1]
                    early_stop_counter = 0
                else:
                    # if epoch > self.warmup:
                    early_stop_counter += 1

            best_mccs += [self.best_mcc]

        best_lists, traces = get_empty_traces()
        # Loading best model that was saved during training
        model.load_state_dict(torch.load(f'{self.complete_log_path}/model.pth'))
        # Need another model because the other cant be use to get shap values
        # shap_model.load_state_dict(torch.load(f'{self.complete_log_path}/model.pth'))
        # model.load_state_dict(sd)
        model.eval()
        # shap_model.eval()
        with torch.no_grad():
            _, _, _ = self.loop('train', optimizer_model, model,  gamma, loaders[group], best_lists, traces)
            for group in ["valid", "test"]:
                closs, best_lists, traces = self.predict(group, model, loaders[group], best_lists, traces)
        if self.log_neptune:
            model["model"].upload(f'{self.complete_log_path}/model.pth')
            model["validation/closs"].log(self.best_closs)
        
        # Create a dataframe of best_lists valid and test. Have preds, classes, names and gaps between pred and class
        for group in ['valid', 'test']:
            df = pd.DataFrame(np.concatenate((np.concatenate(best_lists[group]['preds']),
                                                np.concatenate(best_lists[group]['labels']).reshape(-1, 1),
                                                np.concatenate(best_lists[group]['domains']).reshape(-1, 1),
                                                np.concatenate(best_lists[group]['names']).reshape(-1, 1),
                                                ), 1))
            df.columns = [f'probs_{l}' for l in self.unique_labels] + ['labels', 'domains', 'names']
            preds = df.iloc[:, :2].values
            preds = preds.astype(np.float).argmax(1).astype(np.int)
            df['correct'] = [1 if label == df.columns[pred].split('_')[1] else 0 for label, pred in zip(df['labels'], preds)]
            df['gaps'] = [1 - float(df[f'probs_{l}'].iloc[i]) for i, l in enumerate(df['labels'])]
            # reorder dataframe with gaps column
            df = df.sort_values(by=['gaps'], ascending=False)
            
            # save the valid dataframe
            df.to_csv(f'{self.complete_log_path}/{group}.csv')
            
        best_closses += [self.best_closs]
        shap_model = None
        # traces = self.get_mccs(lists, traces)
        loggers['cm_logger'].add(best_lists)
        values = log_traces(traces, values)
        self.log_rep(best_lists, run, 0)
        del model # , shap_model

        # Logging every model is taking too much resources and it makes it quite complicated to get information when
        # Too many runs have been made. This will make the notebook so much easier to work with
        if np.mean(best_mccs) > self.best_best_mcc:
            try:
                if os.path.exists(
                        f'logs/best_models/model_classifier/{self.args.dataset}'):
                    shutil.rmtree(
                        f'logs/best_models/model_classifier/{self.args.dataset}',
                        ignore_errors=True)
                shutil.copytree(f'{self.complete_log_path}',
                                f'logs/best_models/model_classifier/{self.args.dataset}')
                # print("File copied successfully.")

            # If source and destination are same
            except shutil.SameFileError:
                # print("Source and destination represents the same file.")
                pass
            self.best_best_mcc = np.mean(best_mccs)

        # Logs confusion matrices in the background. Also runs RandomForestClassifier on encoded and reconstructed
        # representations. This should be shorter than the actual calculation of the model above in the function,
        # otherwise the number of threads will keep increasing.
        # daemon = Thread(target=self.logging, daemon=True, name='Monitor', args=[run, cm_logger])
        # daemon.start()

        loggers['cm_logger'].plot(None, epoch, self.unique_labels, 'mlflow')
        if self.log_mlflow:
            mlflow.log_param('finished', 1)
        # self.logging(run, loggers['cm_logger'])

        if not self.keep_models:
            # shutil.rmtree(f'{self.complete_log_path}/traces', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/cm', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/hp', ignore_errors=True)
            shutil.rmtree(f'{self.complete_log_path}', ignore_errors=True)
        print('Duration: {}'.format(datetime.now() - start_time))
        best_closs = np.mean(best_closses)
        if best_closs < self.best_closs_final:
            self.best_closs_final = best_closs
            print("Best closs!")

        # It should not be necessary. To remove once certain the "Too many files open" error is no longer a problem
        # plt.close('all')

        return np.mean(best_mccs)

    def log_rep(self, best_lists, run, h):
        # if run is None:
        #     return None
        # best_traces = self.get_mccs(best_lists, traces)

        self.log_predictions(best_lists, run, h)

    def log_predictions(self, best_lists, run, step):
        cats, labels, preds, scores, names = [{'train': [], 'valid': [], 'test': []} for _ in range(5)]
        for group in ['valid', 'test']:
            cats[group] = np.concatenate(best_lists[group]['cats'])
            labels[group] = np.concatenate(best_lists[group]['labels'])
            scores[group] = torch.softmax(torch.Tensor(np.concatenate(best_lists[group]['preds'])), 1)
            preds[group] = scores[group].argmax(1)
            names[group] = np.concatenate(best_lists[group]['names'])
            pd.DataFrame(np.concatenate((labels[group].reshape(-1, 1), scores[group],
                                         np.array([self.unique_labels[x] for x in preds[group]]).reshape(-1, 1),
                                         names[group].reshape(-1, 1)), 1)).to_csv(
                f'{self.complete_log_path}/{group}_predictions.csv')
            if self.log_neptune:
                run[f"{group}_predictions"].track_files(f'{self.complete_log_path}/{group}_predictions.csv')
                run[f'{group}_AUC'] = metrics.roc_auc_score(y_true=cats[group], y_score=scores[group],
                                                            multi_class='ovr')
            if self.log_mlflow:
                mlflow.log_metric(f'{group}_AUC',
                            metrics.roc_auc_score(y_true=to_categorical(cats[group], len(self.unique_labels)), y_score=scores[group], multi_class='ovr'),
                            step=step)

                # TODO ADD OTHER FINAL METRICS HERE
                mlflow.log_metric(f'{group}_acc',
                            metrics.accuracy_score(cats[group], scores[group].argmax(1)),
                            step=step)
                mlflow.log_metric(f'{group}_mcc',
                            MCC(cats[group], scores[group].argmax(1)),
                            step=step)

            save_roc_curve(
                scores[group],
                to_categorical(cats[group], len(self.unique_labels)),
                self.unique_labels,
                f'{self.complete_log_path}/{group}_roc_curve.png',
                binary=False,
                acc=metrics.accuracy_score(cats[group], scores[group].argmax(1)),
                mlops='mlflow',
                epoch=step,
                logger=None
            )
    
    def loop(self, group, optimizer_model, model, gamma, loader, lists, traces):
        """

        Args:
            group: Which set? Train, valid or test
            optimizer_model: Object that contains the optimizer for the autoencoder
            ae: AutoEncoder (pytorch model, inherits nn.Module)
            sceloss: torch.nn.CrossEntropyLoss instance
            triplet_loss: torch.nn.TripletMarginLoss instance
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # triplet_loss = nn.TripletMarginLoss(margin, p=2)
        classif_loss = None
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            # if group in ['train']:
            if model.training:
                optimizer_model.zero_grad()
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            domains = to_categorical(domain.long(), self.n_batches).squeeze().float().to(self.args.device)
            enc, dpreds = model(data)

            if self.args.classif_loss == 'cosine':
                classif_loss = 1 - torch.nn.functional.cosine_similarity(enc, model(to_rec)).mean()
            elif self.args.classif_loss == 'triplet':
                not_to_rec = not_to_rec.to(self.args.device).float()
                pos_enc, _ = model(to_rec)
                neg_enc, _ = model(not_to_rec)
                classif_loss = self.triplet_loss(enc, model(to_rec)[0], model(not_to_rec)[0])

            if self.args.dloss == 'DANN':
                dloss = self.celoss(dpreds, domains)

            elif self.args.dloss == 'inverseTriplet':
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
                    self.args.device).float(), pos_batch_sample.to(self.args.device).float()
                pos_enc, _, _ = model(pos_batch_sample)
                neg_enc, _, _ = model(neg_batch_sample)
                dloss = self.triplet_loss(enc, pos_enc, neg_enc)
                # domain = domain.argmax(1)
            else:
                dloss = torch.Tensor([0]).to(self.args.device)[0]

            lists[group]['domains'] += [np.array([self.unique_batches[np.argmax(d)] for d in domains.detach().cpu().numpy()])]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            if group == 'train':
                lists[group]['preds'] += [dpreds.detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
            lists[group]['names'] += [names]
            # lists[group]['inputs'] += [data.view(data.shape[0], -1).detach().cpu().numpy()]
            lists[group]['labels'] += [np.array(
                [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            lists[group]['old_labels'] += [np.array(old_labels)]
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds.detach().cpu().numpy().argmax(1),
                                                    domains.argmax(1).detach().cpu().numpy())])]
            traces[group]['closs'] += [classif_loss.item()]
            traces[group]['dloss'] += [dloss.item()]
            lists[group]['cats'] += [labels.detach().cpu().numpy()]
            if model.training:
                if group in ['train']:
                    # total_loss = classif_loss + gamma * dloss
                    classif_loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                elif group in ['all']:
                    if gamma > 0:
                        dloss = gamma * dloss
                        dloss.backward()
                optimizer_model.step()
            # pbar.update(1)
        return classif_loss, lists, traces

    def predict(self, group, model, loader, lists, traces):
        """

        Args:
            group: Which set? Train, valid or test
            optimizer_model: Object that contains the optimizer for the autoencoder
            ae: AutoEncoder (pytorch model, inherits nn.Module)
            sceloss: torch.nn.CrossEntropyLoss instance
            triplet_loss: torch.nn.TripletMarginLoss instance
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # import nearest_neighbors from sklearn
        from sklearn.neighbors import KNeighborsClassifier as KNN
        KNeighborsClassifier = KNN(n_neighbors=10, metric='cosine')
        classif_loss = None
        train_encs = np.concatenate(lists['train']['encoded_values'])
        train_cats = np.concatenate(lists['train']['cats'])
        KNeighborsClassifier.fit(train_encs, train_cats)

        # plot_decision_boundary(KNeighborsClassifier, train_encs, train_cats)
        # plt.savefig(f'{self.complete_log_path}/decision_boundary.png')
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            # if group in ['train']:
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            domains = to_categorical(domain.long(), self.n_batches).squeeze().float().to(self.args.device)
            enc, dpreds = model(data)
            preds = KNeighborsClassifier.predict(enc.view(enc.shape[0], -1).detach().cpu().numpy())
            proba = KNeighborsClassifier.predict_proba(enc.view(enc.shape[0], -1).detach().cpu().numpy())
            if self.args.dloss == 'DANN':
                dloss = self.celoss(dpreds, domains)

            elif self.args.dloss == 'inverseTriplet':
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
                    self.args.device).float(), pos_batch_sample.to(self.args.device).float()
                pos_enc, _, _ = model(pos_batch_sample)
                neg_enc, _, _ = model(neg_batch_sample)
                dloss = self.triplet_loss(enc, pos_enc, neg_enc)
                # domain = domain.argmax(1)
            else:
                dloss = torch.Tensor([0]).to(self.args.device)[0]

            lists[group]['domains'] += [np.array([self.unique_batches[np.argmax(d)] for d in domains.detach().cpu().numpy()])]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]

            lists[group]['names'] += [names]
            lists[group]['cats'] += [labels.detach().cpu().numpy()]
            # lists[group]['inputs'] += [data.view(data.shape[0], -1).detach().cpu().numpy()]
            lists[group]['labels'] += [np.array(
                [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            lists[group]['old_labels'] += [np.array(old_labels)]
            traces[group]['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(preds,
                                                    labels.detach().cpu().numpy())])]
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds.detach().cpu().numpy().argmax(1),
                                                    domains.argmax(1).detach().cpu().numpy())])]
            traces[group]['closs'] += [0]
            traces[group]['dloss'] += [dloss.item()]
            lists[group]['preds'] += [proba]
        traces[group]['mcc'] += [np.round(
            MCC(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1)), 3)
        ]
            # pbar.update(1)

        if group == 'test':
            # plot the PCA of the encoded values
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            valid_encs = np.concatenate(lists['valid']['encoded_values'])
            valid_cats = np.concatenate(lists['valid']['cats'])
            test_encs = np.concatenate(lists['test']['encoded_values'])
            test_cats = np.concatenate(lists['test']['cats'])
            train_encs_pca = pca.fit_transform(train_encs)
            valid_encs_pca = pca.transform(valid_encs)
            test_encs_pca = pca.transform(test_encs)
            # plot the values and save img
            plt.figure(figsize=(10, 10))
            plt.scatter(train_encs_pca[:, 0], train_encs_pca[:, 1], c=train_cats, marker='o')
            plt.scatter(valid_encs_pca[:, 0], valid_encs_pca[:, 1], c=valid_cats, marker='x')
            plt.scatter(test_encs_pca[:, 0], test_encs_pca[:, 1], c=test_cats, marker='^')
            plt.colorbar()
            plt.title('PCA of encoded values')
            plt.savefig(f'pca.png')

        return None, lists, traces

    @staticmethod
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--predict_tests', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--variational', type=int, default=0)
    parser.add_argument('--zinb', type=int, default=0)
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--remove_zeros', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--groupkfold', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='prostate')
    parser.add_argument('--path', type=str, default='./data/otite_ds')
    parser.add_argument('--exp_id', type=str, default='otite')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--n_layers', type=int, default=1, help='N layers for classifier')
    parser.add_argument('--log1p', type=int, default=1, help='log1p the data? Should be 0 with zinb')
    parser.add_argument('--dloss', type=str, default="inverseTriplet", help='domain loss')
    parser.add_argument('--classif_loss', type=str, default='triplet', help='triplet or cosine')
    parser.add_argument('--distance_fct', type=str, default='euclidean', help='[euclidean, cosine, manhattan, chebyshev, minkowski, SNR]')
    parser.add_argument('--task', type=str, default='notNormal', help='Binary classification?')
    parser.add_argument('--is_transform', type=int, default=0, help='Transform train data?')
    parser.add_argument('--is_stn', type=int, default=1, help='Transform train data?')
    parser.add_argument('--weighted_sampler', type=int, default=1, help='Weighted sampler?')
    parser.add_argument('--n_calibration', type=int, default=0, help='n_calibration?')

    args = parser.parse_args()
    args.exp_id = f'{args.exp_id}_{args.task}'
    try:
        mlflow.create_experiment(
            args.dloss,
            # artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            # tags={"version": "v1", "priority": "P1"},
        )
    except:
        print(f"\n\nExperiment {args.exp_id} already exists\n\n")

    train = TrainAE(args, args.path, fix_thres=-1, load_tb=False, log_metrics=True, keep_models=True,
                    log_inputs=False, log_plots=True, log_tb=False, log_neptune=False,
                    log_mlflow=True, groupkfold=args.groupkfold)

    # train.train()
    # List of hyperparameters getting optimized
    parameters = [
        {"name": "lr", "type": "range", "bounds": [1e-6, 1e-2], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-8, 1e-2], "log_scale": True},
        # {"name": "l1", "type": "range", "bounds": [1e-8, 1e-2], "log_scale": True},
        {"name": "smoothing", "type": "range", "bounds": [0., 0.2]},
        {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
        {"name": "margin", "type": "range", "bounds": [0., 10.]},
    ]
    if args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
        # gamma = 0 will ensure DANN is not learned
        parameters += [{"name": "gamma", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]

    # Some hyperparameters are not always required. They are set to a default value in Train.train()

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=train.train,
        objective_name='closs',
        minimize=True,
        total_trials=args.n_trials,
        random_seed=41,

    )

    fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    print('Best Loss:', values[0]['loss'])
    print('Best Parameters:')
    print(json.dumps(best_parameters, indent=4))
