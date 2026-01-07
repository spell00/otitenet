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
import numpy as np
from torch.utils.data import Dataset
import random
from torch.autograd import Variable

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class Dataset1(Dataset):
    def __init__(self, data, epoch, unique_labels, names=None,
                 labels=None, old_labels=None, 
                 batches=None, n_positives=1, n_negatives=1,
                 sets=None, transform=None, crop_size=-1,
                 add_noise=False, random_recs=False, triplet_dloss=False, 
                 triplet_loss=True, prototypes_to_use='no'):
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
        self.old_labels = old_labels
        self.unique_labels = unique_labels
        self.unique_batches = np.unique(batches)
        self.triplet_dloss = triplet_dloss
        self.triplet_loss = triplet_loss
        self.set_labels_and_batches(labels, batches)
            
        self.prototypes_to_use = prototypes_to_use
        self.means = {label: None for label in self.unique_labels}
        self.epoch = epoch
        self.n_positives = n_positives
        self.n_negatives = n_negatives

    def set_labels_and_batches(self, labels, batches):
        self.labels = labels
        self.batches = batches
        self.cats = np.array([
            np.argwhere(label==self.unique_labels).flatten()[0] for label in labels
        ])
        self.labels_inds = {label: 
            [i for i, x in enumerate(labels) if x == label]
            for label in self.unique_labels}
        self.batches_inds = {batch: 
            [i for i, x in enumerate(batches) if x == batch]
            for batch in self.unique_batches}
        self.n_labels = {label: len(self.labels_inds[label]) for label in labels}
        self.n_batches = {batch: len(self.batches_inds[batch]) for batch in batches}

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def set_prototypes(self, prototypes):
        self.prototypes = prototypes

    def set_class_prototypes(self, prototypes):
        self.class_prototypes = prototypes

    def set_batch_prototypes(self, prototypes):
        self.batch_prototypes = prototypes

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
            if self.prototypes_to_use == 'combined' and self.epoch > 0:
                # Use combined (class+batch) prototypes when available; otherwise fallback to real samples
                try:
                    rand_batch = list(self.n_batches.keys())[np.random.randint(len(self.n_batches))]
                    to_rec = self.prototypes[(label, rand_batch)].copy()
                except Exception:
                    to_rec = self.samples[self.labels_inds[label][np.random.randint(0, self.n_labels[label])]].copy()

                not_label = None
                while not_label == label or not_label is None:
                    not_label = self.unique_labels[np.random.randint(0, len(self.unique_labels))].copy()
                try:
                    rand_batch2 = list(self.n_batches.keys())[np.random.randint(len(self.n_batches))]
                    not_to_rec = self.prototypes[(not_label, rand_batch2)].copy()
                except Exception:
                    try:
                        not_to_rec = self.samples[self.labels_inds[not_label][np.random.randint(0, self.n_labels[not_label])]].copy()
                    except Exception:
                        not_to_rec = self.samples[idx].copy()

            elif (self.prototypes_to_use == 'class' or self.prototypes_to_use == 'both') and self.epoch > 0:
                # Use class prototypes when available; otherwise fallback to real samples
                try:
                    to_rec = self.class_prototypes[label].copy()
                except Exception:
                    to_rec = self.samples[self.labels_inds[label][np.random.randint(0, self.n_labels[label])]].copy()
                not_label = None
                while not_label == label or not_label is None:
                    not_label = self.unique_labels[np.random.randint(0, len(self.unique_labels))].copy()
                try:
                    not_to_rec = self.class_prototypes[not_label].copy()
                except Exception:
                    try:
                        not_to_rec = self.samples[self.labels_inds[not_label][np.random.randint(0, self.n_labels[not_label])]].copy()
                    except Exception:
                        not_to_rec = self.samples[idx].copy()
            else:
                to_rec = self.samples[self.labels_inds[label][np.random.randint(0, self.n_labels[label])]].copy()
                not_label = None
                while not_label == label or not_label is None:
                    not_label = self.unique_labels[np.random.randint(0, len(self.unique_labels))].copy()
                try:
                    not_to_rec = self.samples[self.labels_inds[not_label][np.random.randint(0, self.n_labels[not_label])]].copy()
                except:
                    exit()
        else:
            to_rec = self.samples[idx].copy()
            not_to_rec = self.samples[idx].copy()
        if (self.triplet_dloss == 'revTriplet' or self.triplet_dloss == 'inverseTriplet') and len(self.unique_batches) > 1 and len(self.n_batches) > 1:
            if (self.prototypes_to_use == 'batch' or self.prototypes_to_use == 'both') and self.epoch > 0:
                not_batch_label = None
                while not_batch_label == batch or not_batch_label is None:
                    not_batch_label = self.unique_batches[np.random.randint(0, len(self.unique_batches))].copy()
                # pos_ind = np.random.randint(0, self.n_batches[batch])
                # neg_ind = np.random.randint(0, self.n_batches[not_batch_label])
                pos_batch_sample = self.batch_prototypes[batch].copy()
                neg_batch_sample = self.batch_prototypes[not_batch_label].copy()
            else:
                not_batch_label = None
                while not_batch_label == batch or not_batch_label is None:
                    not_batch_label = self.unique_batches[np.random.randint(0, len(self.unique_batches))].copy()
                pos_ind = np.random.randint(0, self.n_batches[batch])
                neg_ind = np.random.randint(0, self.n_batches[not_batch_label])
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
            # Always transform input image
            x = self.transform(x).squeeze()

            # Only transform prototype/sample tensors that look like images (HWC/CHW); if encodings, convert to tensors directly
            def _maybe_transform_or_tensor(arr):
                try:
                    ndim = arr.ndim
                except Exception:
                    ndim = len(arr.shape) if hasattr(arr, 'shape') else 0
                if ndim >= 3:
                    return self.transform(arr).squeeze()
                else:
                    return torch.as_tensor(arr).float()

            to_rec = _maybe_transform_or_tensor(to_rec)
            not_to_rec = _maybe_transform_or_tensor(not_to_rec)
            pos_batch_sample = _maybe_transform_or_tensor(pos_batch_sample)
            neg_batch_sample = _maybe_transform_or_tensor(neg_batch_sample)

        if self.add_noise:
            if np.random.random() > 0.5:
                x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        batch = np.argwhere(batch == self.unique_batches).flatten()
        return x, name, cat, label, old_label, batch, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample


class Dataset2(Dataset):
    def __init__(self, data, epoch, unique_labels, names=None,
                 labels=None, old_labels=None, 
                 batches=None, sets=None, n_positives=1, n_negatives=1,
                 transform=None, crop_size=-1,
                 add_noise=False, random_recs=False, triplet_dloss=False,
                 triplet_loss=True, prototypes_to_use='no'):
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
            self.samples = {
                'train': data['train'].to_numpy(),
                'calibration': data['calibration'].to_numpy()
            }
        except:
            self.samples = data

        self.epoch = epoch
        self.add_noise = add_noise
        self.names = names
        self.sets = sets
        self.transform = transform
        self.crop_size = crop_size
        self.labels = labels
        self.cats = {g: np.array([
            np.argwhere(label==unique_labels).flatten()[0] for label in labels[g]
        ]) for g in ['train', 'calibration']}
        self.old_labels = old_labels
        self.unique_labels = unique_labels
        self.batches = batches
        self.unique_batches = np.unique(batches['train'])
        self.labels_inds = {g: 
            {label: [i for i, x in enumerate(labels[g]) if x == label] for label in self.unique_labels} 
            for g in ['train', 'calibration']}
        self.batches_inds = {g: 
            {batch: [i for i, x in enumerate(batches[g]) if x == batch] for batch in self.unique_batches}
            for g in ['train', 'calibration']}
        self.n_labels = {g: 
            {label: len(self.labels_inds[g][label]) for label in labels[g]}
            for g in ['train', 'calibration']}
        self.n_batches = {g: 
            {batch: len(self.batches_inds[g][batch]) for batch in batches[g]}
            for g in ['train', 'calibration']}
        self.triplet_dloss = triplet_dloss
        self.triplet_loss = triplet_loss
        labels = np.concatenate((labels['train'], labels['calibration']))
        batches = np.concatenate((batches['train'], batches['calibration']))
        self.set_labels_and_batches(labels, batches)
        self.prototypes_to_use = prototypes_to_use
        self.n_negatives = n_negatives
        self.n_positives = n_positives

    def __len__(self):
        return len(self.samples['calibration'])

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def set_prototypes(self, prototypes):
        self.prototypes = prototypes

    def set_class_prototypes(self, prototypes):
        self.class_prototypes = prototypes

    def set_batch_prototypes(self, prototypes):
        self.batch_prototypes = prototypes

    def set_labels_and_batches(self, labels, batches):
        self.labels = labels
        self.batches = batches
        self.cats = np.array([np.argwhere(label==self.unique_labels).flatten()[0] for label in labels])
        self.labels_inds = {label: [i for i, x in enumerate(labels) if x == label] for label in self.unique_labels}
        self.batches_inds = {batch: [i for i, x in enumerate(batches) if x == batch] for batch in self.unique_batches}
        self.n_labels = {label: len(self.labels_inds[label]) for label in labels}
        self.n_batches = {batch: len(self.batches_inds[batch]) for batch in batches}

    def set_encodings(self, encodings):
        """
        Sets the encodings for the samples
        """
        self.encodings = encodings

    def __getitem__(self, idx):
        meta_to_rec = None
        if self.labels is not None:
            cat = self.cats['calibration'][idx]
            label = self.labels['calibration'][idx]
            try:
                old_label = self.old_labels['calibration'][idx]
            except:
                pass
            batch = self.batches['calibration'][idx]
            # set = self.sets[idx]
            try:
                name = self.names['calibration'][idx]
            except:
                name = str(self.names['calibration'].iloc[idx])

        else:
            label = None
            cat = None
            old_label = None
            batch = None
            name = None
        if self.triplet_loss:
            if self.prototypes_to_use == 'combined' and self.epoch > 0:
                to_rec = self.prototypes[(label, 
                                          list(self.n_batches.keys())[np.random.randint(len(self.n_batches))]
                                          )].copy()
                not_label = None
                while not_label == label or not_label is None:
                    not_label = self.unique_labels[np.random.randint(0, len(self.unique_labels))].copy()
                not_to_rec = self.prototypes[(not_label, 
                                              list(self.n_batches.keys())[np.random.randint(len(self.n_batches))])].copy()
            elif (self.prototypes_to_use == 'class' or self.prototypes_to_use == 'both') and self.epoch > 0:
                to_rec = self.class_prototypes[label].copy()
                not_label = None
                while not_label == label or not_label is None:
                    not_label = self.unique_labels[np.random.randint(0, len(self.unique_labels))].copy()
                not_to_rec = self.class_prototypes[not_label].copy()
            else:
                to_rec = self.samples[self.labels_inds[label][np.random.randint(0, self.n_labels[label])]].copy()
                not_label = None
                while not_label == label or not_label is None:
                    not_label = self.unique_labels[np.random.randint(0, len(self.unique_labels))].copy()
                try:
                    not_to_rec = self.samples[self.labels_inds[not_label][np.random.randint(0, self.n_labels[not_label])]].copy()
                except:
                    exit()
        else:
            to_rec = self.samples['train'][idx].copy()
            not_to_rec = self.samples['train'][idx].copy()
        x = self.samples[idx].permute(1, 2, 0)
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
        return x, name, cat, label, old_label, batch, to_rec, not_to_rec


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


