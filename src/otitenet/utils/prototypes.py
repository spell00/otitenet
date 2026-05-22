import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class Prototypes:
    def __init__(self, unique_labels, unique_batches, strategy: str = "mean", components: int = 1, random_state: int = 1):
        self.unique_labels = unique_labels
        self.unique_batches = unique_batches
        self.prototypes = {}
        self.class_prototypes = {}
        self.batch_prototypes = {}
        self.n_prototypes = {}
        self.means = {}
        self.strategy = strategy.lower() if isinstance(strategy, str) else "mean"
        self.components = max(1, int(components))
        self.random_state = random_state

    def set_prototypes(self, group, list1):
        self.set_both_prototypes(group, list1)
        self.set_class_prototypes(group, list1)
        self.set_batch_prototypes(group, list1)

    def _compute_prototype(self, encodings: np.ndarray):
        """Compute a single prototype vector for a set of encodings using the configured strategy."""
        if encodings.size == 0:
            return np.zeros((encodings.shape[1],))

        if self.strategy == "mean" or encodings.shape[0] == 1:
            return np.mean(encodings, axis=0)

        if self.strategy == "kmeans":
            n_clusters = min(self.components, encodings.shape[0])
            km = KMeans(n_clusters=n_clusters, n_init=5, random_state=self.random_state)
            km.fit(encodings)
            counts = np.bincount(km.labels_)
            dominant_idx = int(np.argmax(counts))
            return km.cluster_centers_[dominant_idx]

        if self.strategy in ["gmm", "em", "expectation_maximization"]:
            n_components = min(self.components, encodings.shape[0])
            gm = GaussianMixture(n_components=n_components, random_state=self.random_state)
            gm.fit(encodings)
            weights = gm.weights_ if hasattr(gm, "weights_") else np.ones(n_components)
            dominant_idx = int(np.argmax(weights))
            return gm.means_[dominant_idx]

        # Fallback to mean if strategy is unrecognized
        return np.mean(encodings, axis=0)

    def set_both_prototypes(self, group, list1):
        encodings = np.concatenate(list1[group]['encoded_values'])
        labels = np.concatenate(list1[group]['labels'])
        batches = np.concatenate(list1[group]['domains'])
        prototypes = {}
        for label in self.unique_labels:
            for batch in self.unique_batches:
                inds = np.where((labels == label) & (batches == batch))[0]
                prototypes[(label, batch)] = self._compute_prototype(encodings[inds])
        self.n_prototypes[group] = len(self.prototypes)
        self.prototypes[group] = prototypes

    def set_class_prototypes(self, group, list1):
        encodings = np.concatenate(list1[group]['encoded_values'])
        labels = np.concatenate(list1[group]['labels'])
        prototypes = {}
        for label in self.unique_labels:
            inds = np.where((labels == label))[0]
            prototypes[label] = self._compute_prototype(encodings[inds])
        self.n_prototypes[group] = len(self.prototypes)
        self.class_prototypes[group] = prototypes

    def set_batch_prototypes(self, group, list1):
        encodings = np.concatenate(list1[group]['encoded_values'])
        batches = np.concatenate(list1[group]['domains'])
        prototypes = {}
        for batch in self.unique_batches:
            inds = np.where((batches == batch))[0]
            prototypes[batch] = self._compute_prototype(encodings[inds])
        self.n_prototypes[group] = len(self.prototypes)
        self.batch_prototypes[group] = prototypes

