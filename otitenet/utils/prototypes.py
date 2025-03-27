import numpy as np


class Prototypes:
    def __init__(self, unique_labels, unique_batches):
        self.unique_labels = unique_labels
        self.unique_batches = unique_batches
        self.prototypes = {}
        self.class_prototypes = {}
        self.batch_prototypes = {}
        self.n_prototypes = {}
        self.means = {}

    def set_prototypes(self, group, list1):
        self.set_both_prototypes(group, list1)
        self.set_class_prototypes(group, list1)
        self.set_batch_prototypes(group, list1)

    def set_both_prototypes(self, group, list1):
        encodings = np.concatenate(list1[group]['encoded_values'])
        labels = np.concatenate(list1[group]['labels'])
        batches = np.concatenate(list1[group]['domains'])
        prototypes = {}
        for label in self.unique_labels:
            for batch in self.unique_batches:
                inds = np.where((labels == label) & (batches == batch))[0]
                prototypes[(label, batch)] = np.mean(encodings[inds], axis=0)
        self.n_prototypes[group] = len(self.prototypes)
        self.prototypes[group] = prototypes

    def set_class_prototypes(self, group, list1):
        encodings = np.concatenate(list1[group]['encoded_values'])
        labels = np.concatenate(list1[group]['labels'])
        prototypes = {}
        for label in self.unique_labels:
            inds = np.where((labels == label))[0]
            prototypes[label] = np.mean(encodings[inds], axis=0)
        self.n_prototypes[group] = len(self.prototypes)
        self.class_prototypes[group] = prototypes

    def set_batch_prototypes(self, group, list1):
        encodings = np.concatenate(list1[group]['encoded_values'])
        batches = np.concatenate(list1[group]['domains'])
        prototypes = {}
        for batch in self.unique_batches:
            inds = np.where((batches == batch))[0]
            prototypes[batch] = np.mean(encodings[inds], axis=0)
        self.n_prototypes[group] = len(self.prototypes)
        self.batch_prototypes[group] = prototypes

