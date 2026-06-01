import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from matplotlib import pyplot as plt



def make_dataset_folder(path, new_path='data/otite_ds'):
    os.makedirs(new_path, exist_ok=True)

    pngs = np.array([])
    labels = np.array([])
    names = np.array([])
    datasets = np.array([])
    # groups = np.array([])

    for dataset in ['Banque_Calaman_USA_2020', 'Banque_Viscaino_Chili_2020', 'Banque_Comert_Turquie_2020_jpg']:
        if dataset == 'Banque_Calaman_USA_2020':

            df = pd.read_excel(f"{path}/{dataset}/Image Names and Labels.xlsx")
            names_tmp = np.array([x.split("'")[1] for x in df.loc[:, 'Image Name'].values])
            labels_tmp = df.loc[:, 'Label'].values
            datasets_tmp = np.array([dataset for _ in range(len(labels_tmp))])
            for im in os.listdir(f"{path}/{dataset}/"):
                if '.jpg' not in im:
                    continue
                png = Image.open(f"{path}/{dataset}/{im}")
                png = transforms.Resize((224, 224))(png)
                # pngs[g] += [np.array(png)]
                png.save(f"{new_path}/{im}")
                ind = np.where(names_tmp == im)[0][0]
                labels = np.concatenate((labels.reshape(1, -1), np.array([labels_tmp[ind]]).reshape(1, -1)), 1)
                names = np.concatenate((names.reshape(1, -1), np.array([im]).reshape(1, -1)), 1)
                datasets = np.concatenate((datasets.reshape(1, -1), np.array([datasets_tmp[ind]]).reshape(1, -1)), 1)
                # groups = np.concatenate((groups.reshape(1, -1), np.array(['train']).reshape(1, -1)), 1)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_nums = np.arange(0, labels.shape[1])
            splitter = skf.split(train_nums, labels.flatten())
            train_inds, test_inds = splitter.__next__()

            groups = np.array(['train' if x in train_inds else 'test' for x in np.arange(labels.shape[1])])

        if dataset == 'Banque_Viscaino_Chili_2020':
            for g in os.listdir(f"{path}/{dataset}"):
                if g == ".DS_Store":
                    continue
                if g == 'Testing':
                    group = 'test'
                else:
                    group = 'train'
                for label in os.listdir(f"{path}/{dataset}/{g}/"):
                    if label == ".DS_Store":
                        continue
                    for im in os.listdir(f"{path}/{dataset}/{g}/{label}"):
                        if '.jpg' not in im:
                            continue
                        png = Image.open(f"{path}/{dataset}/{g}/{label}/{im}")
                        png = transforms.Resize((224, 224))(png)
                        png.save(f"{new_path}/{im}")

                        # pngs[group].append(np.array(png))
                        labels = np.concatenate((labels.reshape(1, -1), np.array([label]).reshape(1, -1)), 1)
                        names = np.concatenate((names.reshape(1, -1), np.array([im]).reshape(1, -1)), 1)
                        datasets = np.concatenate((datasets.reshape(1, -1), np.array([dataset]).reshape(1, -1)), 1)
                        groups = np.concatenate((groups.reshape(1, -1), np.array([group]).reshape(1, -1)), 1)
            # pngs[group] = np.stack(pngs[group])

        if dataset == 'Banque_Comert_Turquie_2020_jpg':
            new_labels = np.array([])
            for label in os.listdir(f"{path}/{dataset}"):
                if label == ".DS_Store":
                    continue
                for im in os.listdir(f"{path}/{dataset}/{label}"):
                    if '.jpg' not in im and '.tiff' not in im:
                        continue
                    png = Image.open(f"{path}/{dataset}/{label}/{im}")
                    png = transforms.Resize((224, 224))(png)
                    png.save(f"{new_path}/{im}")

                    # pngs[group].append(np.array(png))
                    new_labels = np.concatenate((new_labels.reshape(1, -1), np.array([label]).reshape(1, -1)), 1)
                    names = np.concatenate((names.reshape(1, -1), np.array([im]).reshape(1, -1)), 1)
                    datasets = np.concatenate((datasets.reshape(1, -1), np.array([dataset]).reshape(1, -1)), 1)
                    # groups = np.concatenate((groups.reshape(1, -1), np.array([group]).reshape(1, -1)), 1)

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

if __name__ == '__main__':
    make_dataset_folder(path='data/datasets', new_path='data/otite_ds')
