import os
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from otitenet.utils import get_unique_labels
from otitenet.dataset import Dataset1
import torchvision
from torchvision.transforms import v2 as transforms
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn.functional as F



def get_images(path):
    pngs = []
    for dataset in os.listdir(path):
        for im in os.listdir(f"{path}/{dataset}"):
            png = Image.open(f"{path}/{dataset}/{im}")
            png = transforms.Resize((299, 299))(png)
            pngs += [np.array(png)]
    return pngs

def get_data(path, args, seed=42):
    """

    Args:
        path: Path where the csvs can be loaded. The folder designated by path needs to contain at least
                   one file named train_inputs.csv (when using --use_valid=0 and --use_test=0). When using
                   --use_valid=1 and --use_test=1, it must also contain valid_inputs.csv and test_inputs.csv.

    Returns:
        data
    """
    data = {}
    unique_labels = np.array([])
    for info in ['inputs', 'meta', 'names', 'labels', 'cats', 'batches', 'orders', 'sets']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'valid']:
        # print('GROUP:', group)
        if group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                splitter = skf.split(train_nums, data['labels']['train'], data['batches']['train'])

            else:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                splitter = skf.split(train_nums, data['labels']['train']).__next__()

            _, valid_inds = splitter.__next__()
            _, test_inds = splitter.__next__()
            train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

            data['inputs']['train'], data['inputs']['valid'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                data['inputs']['train'].iloc[valid_inds], data['inputs']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['valid'], data['labels']['test'] = data['labels']['train'][train_inds], \
                data['labels']['train'][valid_inds], data['labels']['train'][test_inds]
            data['names']['train'], data['names']['valid'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                data['names']['train'].iloc[valid_inds], data['names']['train'].iloc[test_inds]
            data['cats']['train'], data['cats']['valid'], data['cats']['test'] = data['cats']['train'][train_inds], data['cats']['train'][
                valid_inds], data['cats']['train'][test_inds]

        else:
            images = get_images(path)
            # matrix = pd.read_csv(f"{path}/{args.csv_file}", sep=",")
            names = matrix.iloc[:, 0]
            labels = matrix.iloc[:, 1]
            batches = matrix.iloc[:, 2]
            unique_batches = batches.unique()
            batches = np.stack([np.argwhere(x == unique_batches).squeeze() for x in batches])
            orders = np.array([0 for _ in batches])
            matrix = matrix.iloc[:, 3:].fillna(0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            if args.log1p:
                matrix.iloc[:] = np.log1p(matrix.values)
            pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' not in name]
            data['inputs'][group] = matrix.iloc[pos]
            data['names'][group] = names
            data['labels'][group] = labels.to_numpy()[pos]
            # This is juste to make the pipeline work. Meta should be 0 for the amide dataset

            # data['labels'][group] = np.array([x.split('-')[0] for i, x in enumerate(data['labels'][group])])
            unique_labels = get_unique_labels(data['labels'][group])
            data['cats'][group] = data['labels'][group]

    for key in list(data['names'].keys()):
        data['sets'][key] = np.array([key for _ in data['names'][key]])
        # print(key, data['sets'][key])
    for key in list(data.keys()):
        if key in ['inputs', 'meta']:
            data[key]['all'] = pd.concat((
                data[key]['train'], data[key]['valid'], data[key]['test']
            ), 0)
        else:
            data[key]['all'] = np.concatenate((
                data[key]['train'], data[key]['valid'], data[key]['test']
            ), 0)


    return data, unique_labels, unique_batches

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

def get_images(path, size=224):
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
        png = transforms.Resize((size, size))(png)
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
        #     nn.ModuleList([transforms.ColorJitter()])
        # , p=0.5),
        # transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-180, 180)),
        # transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(180, 180))])),
        # transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(270, 270))])),
        # transforms.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.3, 3.3)),
        transforms.RandomApply(nn.ModuleList([
            transforms.RandomAffine(
                degrees=10,
                translate=(.1, .1),
                # scale=(0.9, 1.1),
                shear=(.01, .01),
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        ]), p=0.5),
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 0.03))
        #     ]), p=0.5),
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.RandomResizedCrop(
        #             size=224,
        #             scale=(0.9, 1.),
        #             ratio=(0.9, 1.1)
        #         )
        #     ]), p=0.5),
        # transforms.Lambda(lambda x: x.broadcast_to(3, x.shape[1], x.shape[2])),
        torchvision.transforms.Normalize(0.5, 0.5),
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # transforms.Grayscale(),
        # transforms.Lambda(lambda x: x.broadcast_to(3, x.shape[1], x.shape[2])),
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
                            sampler=WeightedRandomSampler(list(samples_weights), len(samples_weights), replacement=True),
                            batch_size=bs,
                            num_workers = 0,
                            # prefetch_factor = 8,
                            pin_memory=True,
                            drop_last=False,
                            # persistent_workers=True
                            )
    else:
        train_loader = DataLoader(train_set,
                            shuffle=True,
                            batch_size=bs,
                            num_workers = 0,
                            # prefetch_factor = 8,
                            pin_memory=True,
                            drop_last=False,
                            # persistent_workers=True
                            )
    loaders = {
        'train': train_loader,
        'test': DataLoader(test_set,
                           # sampler=WeightedRandomSampler(samples_weights['test'], sum(samples_weights['test']),
                           #                               replacement=False),
                           batch_size=bs,
                            num_workers = 0,
                            # prefetch_factor = 8,
                           pin_memory=True,
                           drop_last=False,
                           # persistent_workers=True
                           ),
        'valid': DataLoader(valid_set,
                            # sampler=WeightedRandomSampler(samples_weights['valid'], sum(samples_weights['valid']),
                            #                               replacement=False),
                            batch_size=bs,
                            num_workers = 0,
                            # prefetch_factor = 8,
                            pin_memory=True,
                            drop_last=False,
                            # persistent_workers=True

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
                                num_workers = 0,
                                # prefetch_factor = 8,
                                pin_memory=True,
                                drop_last=False,
                                # persistent_workers=True
                                )

    return loaders


class GetData:
    def __init__(self, path, args) -> None:
        self.args = args
        self.data = {}
        self.unique_labels = np.array([])
        for info in ['inputs', 'names', 'old_labels', 'labels', 'cats', 'batches']:
            self.data[info] = {}
            for group in ['all', 'train', 'test', 'valid', 'calibration']:
                self.data[info][group] = np.array([])
        self.data['inputs']['all'], self.data['names']['all'], self.data['labels']['all'], self.data['batches']['all'] = get_images(path, args.new_shape)
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
