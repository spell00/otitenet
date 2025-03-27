import numpy as np
import os
import pandas as pd
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_images(file_paths):
    images = []
    for path in file_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(img.flatten())
    return np.array(images)

def plot_gallery(images, h, w, titles=None, cmap='gray'):
    plt.figure(figsize=(2. * w, 2.26 * h))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(images.shape[0]):
        plt.subplot(h, w, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=cmap)
        if titles is not None:
            plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def main():
    # Specify the paths to the images
    # image_paths = os.listdir('data/otite_ds/')

    # import csv file infos.csv
    df = pd.read_csv('data/otite_ds/infos.csv')
    names = df['name'].values
    labels = df['label'].values
    datasets = df['dataset'].values
    image_paths = [f"data/otite_ds/{name}" for name in names]

    # Load images and flatten them
    images = load_images(image_paths)
    size = images.shape[1]
    # Perform PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(images)

    unique_dataset = np.unique(datasets)

    # Plot original images
    # plot_gallery(images, h=1, w=len(image_paths), titles=image_paths, cmap='gray')

    # Plot the PCA spectrum
    fig, ax = plt.subplots()
    # plt.figure(figsize=(8, 6))
    inds = np.argwhere(datasets==unique_dataset[0]).flatten()
    handle1 = ax.plot(projected[inds, 0], projected[inds, 1], 'o', markersize=5, color='blue', alpha=0.5, label=unique_dataset[0])
    inds = np.argwhere(datasets==unique_dataset[1]).flatten()
    handle2  = ax.plot(projected[inds, 0], projected[inds, 1], 'o', markersize=5, color='green', alpha=0.5, label=unique_dataset[1])
    inds = np.argwhere(datasets==unique_dataset[2]).flatten()
    handle3 = ax.plot(projected[inds, 0], projected[inds, 1], 'o', markersize=5, color='red', alpha=0.5, label=unique_dataset[2])
    ax.legend(handles=[handle1[0], handle2[0], handle3[0]])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('PCA_datasets.png')

    new_labels = []
    for label in labels:
        if label in ['Chronic otitis media', 'Chornic', 'Effusion', 'Aom', 'AOM', 'CSOM', 'OtitExterna', 'OtitisEksterna']:
            new_labels += ['Otitis']
        elif label == 'Normal':
            new_labels += ['Normal']
        else:
            new_labels += ['NotNormal']
    unique_labels = np.unique(new_labels)
    new_labels = np.array(new_labels)

    # Plot the PCA spectrum
    pca = PCA(n_components=2)
    projected = pca.fit_transform(images)
    fig, ax = plt.subplots()
    # plt.figure(figsize=(8, 6))
    inds = np.argwhere(new_labels==unique_labels[0]).flatten()
    handle1 = ax.plot(projected[inds, 0], projected[inds, 1], 'o', markersize=5, color='blue', alpha=0.5, label=unique_labels[0])
    inds = np.argwhere(new_labels==unique_labels[1]).flatten()
    handle2  = ax.plot(projected[inds, 0], projected[inds, 1], 'o', markersize=5, color='green', alpha=0.5, label=unique_labels[1])
    inds = np.argwhere(new_labels==unique_labels[2]).flatten()
    handle3 = ax.plot(projected[inds, 0], projected[inds, 1], 'o', markersize=5, color='red', alpha=0.5, label=unique_labels[2])
    ax.legend(handles=[handle1[0], handle2[0], handle3[0]])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('PCA_labels.png')

if __name__ == "__main__":
    main()
