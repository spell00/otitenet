import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from PIL import Image
from ..utils.utils import get_unique_labels
from ..data.dataset import Dataset2, Dataset1
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# Per-image normalization transform (expects CHW tensor)
class PerImageNormalize:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        mean = img.mean(dim=(1, 2), keepdim=True)
        std = img.std(dim=(1, 2), keepdim=True) + 1e-6
        return (img - mean) / std

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
            images = get_images(path, normalize=args.normalize, size=args.new_size)
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


def get_images(path, normalize, size=224):
    pngs = []
    labels = []
    names = []
    datasets = []
    groups = []

    infos = pd.read_csv(f"{path}/infos.csv")
    per_image_normalize = PerImageNormalize()
    normalize_flag = str(normalize).lower() in ['yes', 'true', '1']
    
    for i, row in infos.iterrows():
        im = row['name']
        if im == ".DS_Store":
            continue

        png = Image.open(f"{path}/{im}").convert('RGB')
        # Always downsample first to a low-res anchor, then upscale to target size
        downsample_size = 64
        png = png.resize((downsample_size, downsample_size))
        if size != -1:
            png = png.resize((size, size))
        arr = np.array(png, dtype=np.float32) / 255.0  # Convert to [0,1] float
        if arr.ndim == 2:  # grayscale fallback
            arr = np.stack([arr] * 3, axis=-1)
        
        # Normalize per-image if requested
        if normalize_flag:
            # Convert HWC to CHW for normalization
            arr_chw = torch.from_numpy(arr.transpose(2, 0, 1))
            arr_chw = per_image_normalize(arr_chw)
            # Convert back to HWC numpy
            arr = arr_chw.permute(1, 2, 0).numpy()
        
        try:
            pngs += [arr]
        except Exception:
            continue
        labels += [row['label']]
        names += [row['name']]
        datasets += [row['dataset']]
        groups += [row['group']]

    return np.array(pngs), np.array(names), np.array(labels), np.array(datasets)


def get_images_loaders(data, random_recs, weighted_sampler, 
                       is_transform, samples_weights, epoch, 
                       unique_labels, triplet_dloss, prototypes_to_use,
                       n_positives=1, n_negatives=1,
                       prototypes=None, bs=64, size=224, normalize='no', n_aug=1):
    """

    Args:
        data:
        ae:
        classifier:
        n_aug: Number of augmentations to apply per image (default=1). 
               Always includes original image if n_aug > 0.

    Returns:

    """
    # Worker initialization function for reproducible data loading
    def worker_init_fn(worker_id):
        # Each worker gets a unique but deterministic seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)
    
    # Ensure n_aug is at least 1
    n_aug = max(1, int(n_aug))
    
    # Build transforms; only normalize when requested
    train_ops = [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.RandomResizedCrop(
                    size=size,
                    scale=(0.8, 1.),
                    ratio=(0.8, 1.2)
                )
            ]), p=0.5),
    ]
        
    eval_ops = [transforms.ToTensor()]
    # Normalization is handled in get_images if requested

    transform_train = transforms.Compose(train_ops)
    transform = transforms.Compose(eval_ops)

    train_set = Dataset1(data=data['inputs']['train'],
                         epoch=epoch,
                         unique_labels=unique_labels,
                         names=data['names']['train'],
                         labels=data['labels']['train'],
                         old_labels=data['old_labels']['train'],
                         batches=data['batches']['train'],
                         n_positives=n_positives,
                         n_negatives=n_negatives,
                         sets=None,
                         transform=transform_train, crop_size=-1,
                         random_recs=random_recs,
                         triplet_dloss=triplet_dloss,
                         prototypes_to_use=prototypes_to_use,
                         )
    valid_set = Dataset1(data['inputs']['valid'],
                         epoch,
                         unique_labels,
                         data['names']['valid'],
                         data['labels']['valid'],
                         data['old_labels']['valid'],
                         data['batches']['valid'],
                         n_positives=n_positives,
                         n_negatives=n_negatives,
                         sets=None,
                         transform=transform, crop_size=-1, random_recs=False, 
                         triplet_dloss=triplet_dloss,
                         prototypes_to_use=prototypes_to_use,
                         )
    test_set = Dataset1(data['inputs']['test'],
                        epoch,
                         unique_labels,
                        data['names']['test'],
                        data['labels']['test'],
                        data['old_labels']['test'],
                        data['batches']['test'],
                         n_positives=n_positives,
                         n_negatives=n_negatives,
                        sets=None,
                        transform=transform, crop_size=-1, random_recs=False,
                        triplet_dloss=triplet_dloss,
                         prototypes_to_use=prototypes_to_use,
                        )

    # Only keep prototypes for the train split; inference loaders operate without attached prototypes
    def _train_proto(proto_dict):
        try:
            return proto_dict.get('train', {}) if isinstance(proto_dict, dict) else {}
        except Exception:
            return {}

    combined_train_proto = _train_proto(prototypes.get('combined', {}))
    class_train_proto = _train_proto(prototypes.get('class', {}))
    batch_train_proto = _train_proto(prototypes.get('batch', {}))

    train_set.set_prototypes(combined_train_proto)
    valid_set.set_prototypes({})
    test_set.set_prototypes({})

    train_set.set_class_prototypes(class_train_proto)
    valid_set.set_class_prototypes({})
    test_set.set_class_prototypes({})

    train_set.set_batch_prototypes(batch_train_proto)
    valid_set.set_batch_prototypes({})
    test_set.set_batch_prototypes({})
    
    if weighted_sampler:
        train_loader = DataLoader(train_set,
                            sampler=WeightedRandomSampler(samples_weights['train'], len(samples_weights['train']),
                                                          replacement=True),
                            batch_size=bs,
                            num_workers=0,
                            # prefetch_factor=2,
                            pin_memory=True,
                            drop_last=True,
                            # persistent_workers=True,
                            worker_init_fn=worker_init_fn
                            )
    else:
        train_loader = DataLoader(train_set,
                            shuffle=True,
                            batch_size=bs,
                            num_workers=0,
                            # prefetch_factor=2,
                            pin_memory=True,
                            drop_last=True,
                            # persistent_workers=True,
                            worker_init_fn=worker_init_fn
                            )
    loaders = {
        'train': train_loader,
        'test': DataLoader(test_set,
                           # sampler=WeightedRandomSampler(samples_weights['test'], sum(samples_weights['test']),
                           #                               replacement=False),
                           batch_size=bs,
                            num_workers=0,
                            # prefetch_factor=2,
                           pin_memory=True,
                           drop_last=False,
                           # persistent_workers=True,
                           worker_init_fn=worker_init_fn
                           ),
        'valid': DataLoader(valid_set,
                            # sampler=WeightedRandomSampler(samples_weights['valid'], sum(samples_weights['valid']),
                            #                               replacement=False),
                            batch_size=bs,
                            num_workers=0,
                            # prefetch_factor=2,
                            pin_memory=True,
                            drop_last=False,
                            # persistent_workers=True,
                            worker_init_fn=worker_init_fn

                            ),
    }
    all_set = Dataset1(data['inputs']['all'],
                       epoch,
                         unique_labels,
                       data['names']['all'],
                       data['labels']['all'],
                       data['old_labels']['all'],
                       data['batches']['all'],
                         n_positives=n_positives,
                         n_negatives=n_negatives,
                       sets=None,
                       transform=transform_train, crop_size=-1, random_recs=False,
                       triplet_dloss=triplet_dloss,
                       prototypes_to_use=prototypes_to_use,
                       )

    if len(data['inputs']['calibration']) > 0:
        calibration_set = Dataset2(
            {'train': data['inputs']['train'], 'calibration': data['inputs']['calibration']},
            epoch, unique_labels,
            {'train': data['names']['all'], 'calibration': data['names']['calibration']},
            {'train': data['labels']['all'], 'calibration': data['labels']['calibration']},
            {'train': data['old_labels']['all'], 'calibration': data['old_labels']['calibration']},
            {'train': data['batches']['all'], 'calibration': data['batches']['calibration']},
            n_positives=n_positives,
            n_negatives=n_negatives,
            sets=None,
            transform=transform_train, crop_size=-1, random_recs=False,
            triplet_dloss=triplet_dloss,
            prototypes_to_use=prototypes_to_use,
        )
        calibration_set.set_prototypes({})
        calibration_set.set_class_prototypes({})
        calibration_set.set_batch_prototypes({})
        loaders['calibration'] = DataLoader(calibration_set,
                                    # num_workers=0,
                                    shuffle=True,
                                    batch_size=bs,
                                    num_workers=0,
                                    # prefetch_factor=8,
                                    pin_memory=True,
                                    drop_last=True,
                                    # persistent_workers=True,
                                    worker_init_fn=worker_init_fn
                                    )

    # Build all-split loader without attaching prototypes
    all_set.set_prototypes({})
    all_set.set_class_prototypes({})
    all_set.set_batch_prototypes({})

    loaders['all'] = DataLoader(all_set,
                                # num_workers=0,
                                shuffle=True,
                                batch_size=bs,
                                num_workers = 0,
                                # prefetch_factor = 8,
                                pin_memory=True,
                                drop_last=True,
                                # persistent_workers=True,
                                worker_init_fn=worker_init_fn
                                )        

    return loaders


class GetData:
    def __init__(self, path, valid, args, manifest_dir=None) -> None:
        self.args = args
        self.manifest_dir = manifest_dir
        self.data = {}
        self.unique_labels = np.array([])
        for info in ['inputs', 'names', 'old_labels', 'labels', 'cats', 'batches']:
            self.data[info] = {}
            for group in ['all', 'train', 'test', 'valid', 'calibration']:
                self.data[info][group] = np.array([])
        self.data['inputs']['all'], self.data['names']['all'], \
            self.data['labels']['all'], self.data['batches']['all'] = get_images(path, args.normalize, args.new_size)
        self.data['old_labels']['all'] = self.data['labels']['all']
        new_labels = []
        for label in self.data['labels']['all']:
            if args.task == 'otitis':
                if label in ['Chronic otitis media', 'Chornic', 'Effusion', 'Aom', 
                             'AOM', 'CSOM', 'OtitExterna', 'OtitisEksterna']:
                    new_labels += ['Otitis']
                else:
                    new_labels += ['notOtitis']
            elif args.task == 'multi':
                if label in ['Chronic otitis media', 'Chornic', 'Effusion', 'Aom', 
                             'AOM', 'CSOM', 'OtitExterna', 'OtitisEksterna']:
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
        self.unique_labels = get_unique_labels(self.data['labels']['all'])
        self.unique_old_labels = get_unique_labels(self.data['old_labels']['all'])
        self.unique_batches = get_unique_labels(self.data['batches']['all'])
        self.data['cats']['all'] = np.array([
            np.argwhere(x == self.unique_labels).flatten() for x in self.data['labels']['all']
        ]).flatten()
        self.data['cats']['train'] = np.array([
            np.argwhere(x == self.unique_labels).flatten() for x in self.data['labels']['all']
        ]).flatten()
        self.data['inputs']['train'], self.data['names']['train'] =\
            self.data['inputs']['all'].copy(), self.data['names']['all'].copy()
        self.data['labels']['train'], self.data['batches']['train'] =\
            self.data['labels']['all'].copy(), self.data['batches']['all'].copy()
        self.data['old_labels']['train'] = self.data['old_labels']['all'].copy()

        # If a manifest of splits exists (saved during training), enforce it to keep exact same images.
        manifest_applied = False
        if manifest_dir is not None:
            manifest_applied = self._apply_manifest_splits(manifest_dir)

        if not manifest_applied:
            if not args.groupkfold:
                self.data, self.unique_labels, self.unique_batches = self.split('test', args.groupkfold, args.seed)
            else:
                self.split_deterministic(valid=valid)
            if manifest_dir is not None:
                self.save_split_manifests(manifest_dir)
        self.prototypes = {
            'combined': {'train': None, 'valid': None, 'test': None, 'calibration': None, 'all': None},
            'class': {'train': None, 'valid': None, 'test': None, 'calibration': None, 'all': None},
            'batch': {'train': None, 'valid': None, 'test': None, 'calibration': None, 'all': None}
        }
        self.n_prototypes = {}

        self.all_samples = self.data.copy()

    def _manifest_paths(self, manifest_dir):
        split_dir = os.path.join(manifest_dir, 'splits')
        return {
            'root': split_dir,
            'train': os.path.join(split_dir, 'train.csv'),
            'valid': os.path.join(split_dir, 'valid.csv'),
            'test': os.path.join(split_dir, 'test.csv'),
        }

    def _apply_manifest_splits(self, manifest_dir):
        paths = self._manifest_paths(manifest_dir)
        if not all(os.path.exists(paths[k]) for k in ['train', 'valid', 'test']):
            return False

        try:
            manifests = {k: pd.read_csv(paths[k]) for k in ['train', 'valid', 'test']}
        except Exception:
            return False

        name_to_idx = {n: i for i, n in enumerate(self.data['names']['all'])}
        def _select(group):
            names = manifests[group]['name'].tolist()
            idxs = [name_to_idx[n] for n in names if n in name_to_idx]
            return idxs

        idx_train = _select('train')
        idx_valid = _select('valid')
        idx_test = _select('test')

        if not (idx_train and idx_valid and idx_test):
            return False

        for group, idxs in [('train', idx_train), ('valid', idx_valid), ('test', idx_test)]:
            self.data['inputs'][group] = self.data['inputs']['all'][idxs]
            self.data['labels'][group] = self.data['labels']['all'][idxs]
            self.data['old_labels'][group] = self.data['old_labels']['all'][idxs]
            self.data['names'][group] = self.data['names']['all'][idxs]
            self.data['cats'][group] = self.data['cats']['all'][idxs]
            self.data['batches'][group] = self.data['batches']['all'][idxs]

        return True

    def save_split_manifests(self, manifest_dir):
        paths = self._manifest_paths(manifest_dir)
        os.makedirs(paths['root'], exist_ok=True)
        for group in ['train', 'valid', 'test']:
            df = pd.DataFrame({
                'name': self.data['names'][group],
                'label': self.data['labels'][group],
                'batch': self.data['batches'][group],
            })
            df.to_csv(paths[group], index=False)
    
    def remove_noisy_samples(self, complete_log_path):
        # Load noisy samples if the file exists
        # Print number of samples in each group
        print(f'Number of samples: '
              # f'Train: {self.data["all"]}, '
              f'Train: {self.data["inputs"]["train"].shape[0]}, '
              f'Valid: {self.data["inputs"]["valid"].shape[0]}, '
              f'Test: {self.data["inputs"]["test"].shape[0]}, '
              f'Calibration: {self.data["inputs"]["calibration"].shape[0]}, '
        )
        noisy_samples_path = f'{complete_log_path}/noisy_samples.csv'
        if os.path.exists(noisy_samples_path):
            noisy_df = pd.read_csv(noisy_samples_path)
            noisy_names = noisy_df['names'].tolist()
            
            # Filter out noisy samples from the data
            for group in ['train', 'valid', 'test']:
                mask = ~np.isin(self.all_samples['names'][group], noisy_names)
                self.data['inputs'][group] = self.all_samples['inputs'][group][mask]
                self.data['labels'][group] = self.all_samples['labels'][group][mask]
                self.data['old_labels'][group] = self.all_samples['old_labels'][group][mask]
                self.data['names'][group] = self.all_samples['names'][group][mask]
                self.data['cats'][group] = self.all_samples['cats'][group][mask]
                self.data['batches'][group] = self.all_samples['batches'][group][mask]

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
        self.data['old_labels']['train'], self.data['old_labels'][group] =\
            self.data['old_labels']['train'][train_inds], self.data['old_labels']['train'][test_inds]
        self.data['names']['train'], self.data['names'][group] = self.data['names']['train'][train_inds], \
            self.data['names']['train'][test_inds]
        self.data['cats']['train'], self.data['cats'][group] = self.data['cats']['train'][train_inds], \
            self.data['cats']['train'][test_inds]
        self.data['batches']['train'], self.data['batches'][group] =\
            self.data['batches']['train'][train_inds], self.data['batches']['train'][test_inds]

        return self.data, self.unique_labels, self.unique_batches

    def split_deterministic(self, valid):
        datasets = ['Banque_Comert_Turquie_2020_jpg', 
                    'Banque_Calaman_USA_2020_trie_CM', 
                    'Banque_Viscaino_Chili_2020']
        train_inds = []
        for dataset in datasets:
            if valid != dataset:
                train_inds += [np.argwhere(self.data['batches']['train'] == dataset).flatten()]
        valid_inds = np.argwhere(self.data['batches']['train'] == valid).flatten()
        train_inds = np.concatenate(train_inds)
        test_inds = np.argwhere(self.data['batches']['train'] == 'GMFUNL_jan2023').flatten()

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
    else:
        dist_fct = None
    return dist_fct


def get_n_features(model_name):
    if model_name == 'resnet18':
        n_features = 512
    elif model_name == 'resnet50':
        n_features = 2048
    elif model_name == 'resnet101':
        n_features = 2048
    elif model_name == 'resnet152':
        n_features = 2048
    elif model_name == 'vgg16':
        n_features = 1000
    elif model_name == 'vgg11':
        n_features = 1000
    elif model_name == 'vit':
        n_features = 1000
    elif model_name == 'densenet121':
        n_features = 1000
    elif model_name == 'efficientnet_b0':
        n_features = 1000
    return n_features

