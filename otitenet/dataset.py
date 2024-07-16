#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import csv
from PIL import Image
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
from sklearn.preprocessing import minmax_scale as scale
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torch import nn

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class Images:
    def __init__(self, path, binarize=False, resize=True, test=False):
        self.path = path
        self.binarize = binarize
        self.resize = resize
        if not test:
            self.fnames = os.listdir(path)
        else:
            tmp = os.listdir(path)
            np.random.shuffle(tmp)
            self.fnames = tmp[:57]

    def process(self, i):
        fname = self.fnames[i]
        print(f"Processing sample #{i}: {fname}")
        png = Image.open(f"{self.path}/{fname}")
        if self.resize:
            png = transforms.Resize((299, 299))(png)
        png.load()  # required for png.split()

        new_png = Image.new("RGB", png.size, (255, 255, 255))
        new_png.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        label = fname.split('_')[0]
        mat_data = np.asarray(new_png)
        if self.binarize:
            mat_data[mat_data > 0] = 1

        return mat_data[:, :, 0], label, new_png

    def __len__(self):
        return len(self.fnames)


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



# This function is much faster than using pd.read_csv
def load_data(path):
    cols = csv.DictReader(open(path))
    data = []
    names = []
    for i, row in enumerate(cols):
        names += [list(row.values())[0]]
        data += [np.array(list(row.values())[1:], dtype=float)]
    data = np.stack(data)
    # data = pd.DataFrame(data, index=labels, columns=list(row.keys())[1:])
    labels = np.array([d.split('_')[1].split('-')[0] for d in names])
    batches = np.array([d.split('_')[0] for d in names])
    # data = get_normalized(torch.Tensor(np.array(data.values, dtype='float')))
    data[np.isnan(data)] = 0
    columns = list(row.keys())[1:]
    return data.T, names, labels, batches, columns


def load_checkpoint(checkpoint_path,
                    model,
                    optimizer,
                    name,
                    ):
    losses = {
        "train": [],
        "valid": [],
    }
    if name not in os.listdir(checkpoint_path):
        print("checkpoint not found...")
        return model, None, 1, losses, None, None, {'valid_loss': -1}
    checkpoint_dict = torch.load(checkpoint_path + '/' + name, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    best_values = checkpoint_dict['best_values']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    state_dict = checkpoint_dict['model']
    model.load_state_dict(state_dict)
    try:
        losses_recon = checkpoint_dict['losses_recon']
        kl_divs = checkpoint_dict['kl_divs']
    except:
        losses_recon = None
        kl_divs = None

    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, epoch, losses, kl_divs, losses_recon, best_values, checkpoint_dict['finished']


def final_save_checkpoint(checkpoint_path, model, optimizer, name):
    model, optimizer, epoch, losses, _, _, best_values, _ = load_checkpoint(checkpoint_path, model, optimizer, name)
    torch.save({'model': model.state_dict(),
                'losses': losses,
                'best_values': best_values,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                # 'learning_rate': learning_rate,
                'finished': True},
               checkpoint_path + '/' + name)


def save_checkpoint(model,
                    optimizer,
                    # learning_rate,
                    epoch,
                    checkpoint_path,
                    losses,
                    best_values,
                    name="cnn",
                    ):
    # model_for_saving = model_name(input_shape=input_shape, nb_classes=nb_classes, variant=variant,
    #                               activation=activation)
    # model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model.state_dict(),
                'losses': losses,
                'best_values': best_values,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                # 'learning_rate': learning_rate,
                'finished': False},
               checkpoint_path + '/' + name)


