import os
import numpy as np
import pandas as pd
from src.utils import get_unique_labels
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


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
