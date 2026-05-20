import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from matplotlib import pyplot as plt


DEFAULT_DATASETS = [
    'Banque_Calaman_USA_2020_trie_CM',
    'Banque_Viscaino_Chili_2020',
    'Banque_Comert_Turquie_2020_jpg',
    'GMFUNL_jan2023',
]


def _resolve_datasets(path, include_datasets=None, exclude_datasets=None):
    include = [d for d in (include_datasets or DEFAULT_DATASETS) if d]
    exclude = set(exclude_datasets or [])
    selected = [d for d in include if d not in exclude]

    missing = [d for d in selected if not os.path.isdir(os.path.join(path, d))]
    if missing:
        print(f"[Preprocess] Skipping missing dataset folders: {missing}")

    selected = [d for d in selected if os.path.isdir(os.path.join(path, d))]
    print(f"[Preprocess] Selected datasets: {selected}")
    return selected



def make_dataset_folder(path, size, new_path='data/otite_ds', include_datasets=None, exclude_datasets=None, split_mode='by_dataset'):
    new_path = f"{new_path}_{size}"
    os.makedirs(new_path, exist_ok=True)

    pngs = np.array([])
    labels = np.array([])
    names = np.array([])
    datasets = np.array([])
    groups = np.array([])

    selected_datasets = _resolve_datasets(path, include_datasets, exclude_datasets)
    for dataset in selected_datasets:
        if dataset == 'Banque_Calaman_USA_2020_trie_CM':
            new_labels = np.array([])
            for label in os.listdir(f"{path}/{dataset}"):
                if label == ".DS_Store" or not os.path.isdir(f"{path}/{dataset}/{label}"):
                    continue
                for im in os.listdir(f"{path}/{dataset}/{label}"):
                    if not im.lower().endswith('.jpg') or 'Off' in im or 'off' in im:
                        continue
                    png = Image.open(f"{path}/{dataset}/{label}/{im}")
                    if size != -1:
                        png = transforms.Resize((size, size))(png)
                    png.save(f"{new_path}/{im}")
                    new_labels = np.concatenate((new_labels.reshape(1, -1), np.array([label]).reshape(1, -1)), 1)
                    names = np.concatenate((names.reshape(1, -1), np.array([im]).reshape(1, -1)), 1)
                    datasets = np.concatenate((datasets.reshape(1, -1), np.array([dataset]).reshape(1, -1)), 1)

            if split_mode == 'by_dataset':
                new_groups = np.array(['train'] * new_labels.shape[1])
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                train_nums = np.arange(0, new_labels.shape[1])
                splitter = skf.split(train_nums.flatten(), new_labels.flatten())
                train_inds, test_inds = splitter.__next__()
                new_groups = np.array(['train' if x in train_inds else 'test' for x in np.arange(new_labels.shape[1])])
            groups = np.concatenate((groups.reshape(1, -1), new_groups.reshape(1, -1)), 1)
            labels = np.concatenate((labels.reshape(1, -1), new_labels.reshape(1, -1)), 1)

        if dataset == 'eardrum':
            for label in os.listdir(f"{path}/{dataset}"):
                if label == ".DS_Store" or '.xlsx' in label:
                    continue
                for im in os.listdir(f"{path}/{dataset}/{label}"):
                    if 'Off' in im or 'off' in im or '.png' not in im:
                        continue
                    png = Image.open(f"{path}/{dataset}/{label}/{im}")
                    if size != -1:
                        png = transforms.Resize((size, size))(png)
                    png.save(f"{new_path}/{im}")

                    # pngs[group].append(np.array(png))
                    labels = np.concatenate((labels.reshape(1, -1), np.array([label]).reshape(1, -1)), 1)
                    names = np.concatenate((names.reshape(1, -1), np.array([im]).reshape(1, -1)), 1)
                    datasets = np.concatenate((datasets.reshape(1, -1), np.array([dataset]).reshape(1, -1)), 1)
                    # groups = np.concatenate((groups.reshape(1, -1), np.array([group]).reshape(1, -1)), 1)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_nums = np.arange(0, labels.shape[1])
            splitter = skf.split(train_nums, labels.flatten())
            train_inds, test_inds = splitter.__next__()
            groups = np.array(['train' if x in train_inds else 'test' for x in np.arange(labels.shape[1])])

        if dataset == 'Banque_Viscaino_Chili_2020':
            new_labels_list, new_names_list, kfold_groups_list = [], [], []
            for g in os.listdir(f"{path}/{dataset}"):
                if g == ".DS_Store":
                    continue
                kfold_group_tag = 'test' if g == 'Testing' else 'train'
                for label in os.listdir(f"{path}/{dataset}/{g}/"):
                    if label == ".DS_Store":
                        continue
                    for im in os.listdir(f"{path}/{dataset}/{g}/{label}"):
                        if '.jpg' not in im or 'Off' in im or 'off' in im:
                            continue
                        png = Image.open(f"{path}/{dataset}/{g}/{label}/{im}")
                        if size != -1:
                            png = transforms.Resize((size, size))(png)
                        png.save(f"{new_path}/{im}")
                        new_labels_list.append(label)
                        new_names_list.append(im)
                        kfold_groups_list.append(kfold_group_tag)
            new_labels_arr = np.array(new_labels_list)
            names = np.concatenate((names.reshape(1, -1), np.array(new_names_list).reshape(1, -1)), 1)
            datasets = np.concatenate((datasets.reshape(1, -1), np.full(len(new_names_list), dataset).reshape(1, -1)), 1)
            labels = np.concatenate((labels.reshape(1, -1), new_labels_arr.reshape(1, -1)), 1)
            if split_mode == 'by_dataset':
                all_rel = np.arange(len(new_labels_list))
                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                valid_rel, test_rel = next(skf.split(all_rel, new_labels_arr))
                valid_set_idx = set(valid_rel)
                new_groups = np.array(['valid' if i in valid_set_idx else 'test' for i in all_rel])
            else:
                new_groups = np.array(kfold_groups_list)
            groups = np.concatenate((groups.reshape(1, -1), new_groups.reshape(1, -1)), 1)
        if dataset == 'GMFUNL_jan2023':
            new_labels = np.array([])
            for label in os.listdir(f"{path}/{dataset}"):
                if label == ".DS_Store":
                    continue
                for im in os.listdir(f"{path}/{dataset}/{label}"):
                    if '.png' not in im or 'Off' in im or 'off' in im:
                        continue
                    png = Image.open(f"{path}/{dataset}/{label}/{im}")
                    if size != -1:
                        png = transforms.Resize((size, size))(png)
                    png.save(f"{new_path}/{im}")

                    new_labels = np.concatenate((new_labels.reshape(1, -1), np.array([label]).reshape(1, -1)), 1)
                    names = np.concatenate((names.reshape(1, -1), np.array([im]).reshape(1, -1)), 1)
                    datasets = np.concatenate((datasets.reshape(1, -1), np.array([dataset]).reshape(1, -1)), 1)
            # pngs[group] = np.stack(pngs[group])

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_nums = np.arange(0, labels.shape[1])
            splitter = skf.split(train_nums, labels.flatten())
            train_inds, test_inds = splitter.__next__()

            new_groups = np.array(['train' if x in train_inds else 'test' for x in np.arange(new_labels.shape[1])])
            groups = np.concatenate((groups.reshape(1, -1), new_groups.reshape(1, -1)), 1)
            labels = np.concatenate((labels.reshape(1, -1), new_labels.reshape(1, -1)), 1)


        if dataset == 'Banque_Comert_Turquie_2020_jpg':
            new_labels = np.array([])
            for label in os.listdir(f"{path}/{dataset}"):
                if label == ".DS_Store":
                    continue
                for im in os.listdir(f"{path}/{dataset}/{label}"):
                    if '.jpg' not in im or 'Off' in im or 'off' in im:
                        continue
                    png = Image.open(f"{path}/{dataset}/{label}/{im}")
                    if size != -1:
                        png = transforms.Resize((size, size))(png)
                    png.save(f"{new_path}/{im}")

                    # pngs[group].append(np.array(png))
                    new_labels = np.concatenate((new_labels.reshape(1, -1), np.array([label]).reshape(1, -1)), 1)
                    names = np.concatenate((names.reshape(1, -1), np.array([im]).reshape(1, -1)), 1)
                    datasets = np.concatenate((datasets.reshape(1, -1), np.array([dataset]).reshape(1, -1)), 1)
            if split_mode == 'by_dataset':
                new_groups = np.array(['train'] * new_labels.shape[1])
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                train_nums = np.arange(0, new_labels.shape[1])
                splitter = skf.split(train_nums.flatten(), new_labels.flatten())
                train_inds, test_inds = splitter.__next__()
                new_groups = np.array(['train' if x in train_inds else 'test' for x in np.arange(new_labels.shape[1])])
            groups = np.concatenate((groups.reshape(1, -1), new_groups.reshape(1, -1)), 1)
            labels = np.concatenate((labels.reshape(1, -1), new_labels.reshape(1, -1)), 1)

    df = pd.DataFrame(
        np.concatenate((
            datasets.reshape(-1, 1),
            names.reshape(-1, 1),
            labels.reshape(-1, 1),
            groups.reshape(-1, 1)
            ), 1), columns=['dataset', 'name', 'label', 'group']
        )
    df.to_csv(f"{new_path}/infos.csv", index=False)


def build_from_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    source_root = cfg.get('source_root', 'data/datasets')
    output_base = cfg.get('output_base', 'data/otite_ds')
    image_size = int(cfg.get('image_size', 64))
    include = cfg.get('include_datasets')
    exclude = cfg.get('exclude_datasets', [])

    split_mode = cfg.get('split_mode', 'by_dataset')
    print(
        f"[Preprocess] source_root={source_root} output_base={output_base} "
        f"image_size={image_size} split_mode={split_mode}"
    )
    make_dataset_folder(
        path=source_root,
        size=image_size,
        new_path=output_base,
        include_datasets=include,
        exclude_datasets=exclude,
        split_mode=split_mode,
    )

if __name__ == '__main__':
    default_cfg = 'configs/preprocessing_config.json'
    legacy_cfg = 'data/datasets/preprocessing_config.json'
    if os.path.exists(default_cfg):
        build_from_config(default_cfg)
    elif os.path.exists(legacy_cfg):
        build_from_config(legacy_cfg)
    else:
        make_dataset_folder(path='data/QC-CM', size=224, new_path='data/otite_ds')
