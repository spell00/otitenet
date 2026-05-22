"""
Kernel Density Estimation (Parzen Windows) for classification.

This module implements soft KDE using Gaussian and other kernels, with optional learnable parameters.
"""

import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist, pdist, squareform


class KernelDensityClassifier:
    """
    KDE-based classifier using Parzen windows for soft classification.
    
    For each class, we fit a KDE on the training embeddings. 
    For prediction, we evaluate the density for each class and pick the one with highest density.
    """
    
    def __init__(self, kernel: str = 'gaussian', bandwidth: float = 'scott', learnable_bandwidth: bool = False):
        """
        Args:
            kernel: Type of kernel ('gaussian', 'exponential', 'linear', 'tophat')
            bandwidth: Bandwidth for the kernel. Can be 'scott', 'silverman', or a float value
            learnable_bandwidth: If True, attempt to learn bandwidth from validation set
        """
        self.kernel = kernel.lower()
        self.bandwidth = bandwidth
        self.learnable_bandwidth = learnable_bandwidth
        self.class_kdes = {}  # {class_label: KernelDensity}
        self.class_data = {}  # {class_label: training_data}
        self.unique_labels = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit KDE for each class.
        
        Args:
            X: Training embeddings (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        self.unique_labels = np.unique(y)
        
        for label in self.unique_labels:
            mask = y == label
            X_class = X[mask]
            self.class_data[label] = X_class
            
            # Determine bandwidth if needed
            bw = self.bandwidth
            if isinstance(bw, str):
                # sklearn will compute scott or silverman automatically
                pass
            
            # Create and fit KDE
            kde = KernelDensity(kernel=self.kernel, bandwidth=bw)
            kde.fit(X_class)
            self.class_kdes[label] = kde
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using highest KDE density.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        densities = self.score_samples(X)  # (n_samples, n_classes)
        predictions = self.unique_labels[np.argmax(densities, axis=1)]
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return soft probabilities based on KDE scores.
        
        The score for each class is exp(log_density). We normalize across classes.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Probabilities (n_samples, n_classes)
        """
        log_densities = self.score_samples(X, log=True)  # (n_samples, n_classes)
        # Convert to probabilities using softmax of log-densities
        densities = np.exp(log_densities)
        proba = densities / (densities.sum(axis=1, keepdims=True) + 1e-8)
        return proba
    
    def score_samples(self, X: np.ndarray, log: bool = False) -> np.ndarray:
        """
        Score samples using KDE for all classes.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            log: If True, return log-densities; else regular densities
            
        Returns:
            Densities/log-densities (n_samples, n_classes)
        """
        scores = []
        for label in self.unique_labels:
            kde = self.class_kdes[label]
            if log:
                score = kde.score_samples(X)  # log-density
            else:
                score = np.exp(kde.score_samples(X))  # density
            scores.append(score)
        
        return np.column_stack(scores)  # (n_samples, n_classes)


class SoftKernelDensityClassifier:
    """
    Soft KDE classifier that manually computes KDE with learnable bandwidth.
    
    Uses Gaussian kernel: K(x) = exp(-||x||^2 / (2 * bandwidth^2))
    
    The bandwidth can be learned from validation data by maximizing likelihood.
    """
    
    def __init__(self, kernel: str = 'gaussian', initial_bandwidth: float = 1.0, 
                 learnable_bandwidth: bool = False):
        """
        Args:
            kernel: Type of kernel ('gaussian' implemented, others fallback to scipy)
            initial_bandwidth: Initial bandwidth parameter
            learnable_bandwidth: Whether to learn bandwidth from validation set
        """
        self.kernel = kernel.lower()
        self.bandwidth = initial_bandwidth
        self.learnable_bandwidth = learnable_bandwidth
        self.class_data = {}  # {class_label: training_embeddings}
        self.unique_labels = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the KDE model.
        
        Args:
            X: Training embeddings (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        self.unique_labels = np.unique(y)
        
        for label in self.unique_labels:
            mask = y == label
            self.class_data[label] = X[mask].astype(np.float32)
    
    def _gaussian_kernel(self, x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
        """
        Compute Gaussian kernel: K(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
        
        Args:
            x: Shape (n, d)
            y: Shape (m, d)
            bandwidth: Bandwidth parameter
            
        Returns:
            Kernel matrix (n, m)
        """
        # Use cdist for efficiency
        sq_dists = cdist(x, y, metric='sqeuclidean')  # squared euclidean distances
        return np.exp(-sq_dists / (2.0 * bandwidth ** 2 + 1e-8))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using highest KDE density.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        densities = self.score_samples(X)  # (n_samples, n_classes)
        predictions = self.unique_labels[np.argmax(densities, axis=1)]
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return soft probabilities based on KDE scores.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Probabilities (n_samples, n_classes)
        """
        densities = self.score_samples(X)  # (n_samples, n_classes)
        # Normalize to sum to 1
        proba = densities / (densities.sum(axis=1, keepdims=True) + 1e-8)
        return proba
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute KDE density for each class.
        
        For each class, density(x) = (1/n_samples) * sum_i K(x, x_i)
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Densities (n_samples, n_classes)
        """
        densities = []
        
        for label in self.unique_labels:
            class_samples = self.class_data[label]  # (n_class_samples, d)
            
            if self.kernel == 'gaussian':
                K = self._gaussian_kernel(X, class_samples, self.bandwidth)  # (n_samples, n_class_samples)
            else:
                # Fallback: use sklearn's KernelDensity
                kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
                kde.fit(class_samples)
                K = np.exp(kde.score_samples(X))  # (n_samples,)
                densities.append(K)
                continue
            
            # Average kernel response over training samples (soft vote)
            density = np.mean(K, axis=1)  # (n_samples,)
            densities.append(density)
        
        return np.column_stack(densities)  # (n_samples, n_classes)
    
    def learn_bandwidth_from_validation(self, X_val: np.ndarray, y_val: np.ndarray, 
                                        bandwidth_range: tuple = (0.1, 2.0), n_steps: int = 10):
        """
        Learn optimal bandwidth by maximizing validation likelihood.
        
        Args:
            X_val: Validation embeddings (n_val, n_features)
            y_val: Validation labels (n_val,)
            bandwidth_range: (min_bw, max_bw)
            n_steps: Number of bandwidth values to try
        """
        if not self.learnable_bandwidth:
            return
        
        bandwidths = np.linspace(bandwidth_range[0], bandwidth_range[1], n_steps)
        best_bw = self.bandwidth
        best_score = -np.inf
        
        for bw in bandwidths:
            self.bandwidth = bw
            # Compute likelihood (mean log probability of correct class)
            log_probs = []
            
            for i, label in enumerate(y_val):
                class_samples = self.class_data[label]
                if self.kernel == 'gaussian':
                    K = self._gaussian_kernel(X_val[i:i+1], class_samples, bw)
                    density = np.mean(K)
                else:
                    kde = KernelDensity(kernel=self.kernel, bandwidth=bw)
                    kde.fit(class_samples)
                    density = np.exp(kde.score_samples(X_val[i:i+1]))[0]
                
                log_probs.append(np.log(density + 1e-8))
            
            score = np.mean(log_probs)
            if score > best_score:
                best_score = score
                best_bw = bw
        
        self.bandwidth = best_bw
        print(f"Learned bandwidth: {self.bandwidth:.4f} (validation likelihood: {best_score:.4f})")


def make_kde_classifier(kernel: str = 'gaussian', bandwidth: str = 'scott', 
                       learnable: bool = False, soft: bool = True) -> 'KernelDensityClassifier':
    """
    Factory function to create a KDE classifier.
    
    Args:
        kernel: Kernel type ('gaussian', 'exponential', 'linear', 'tophat')
        bandwidth: 'scott', 'silverman', or float value
        learnable: Whether to learn bandwidth from validation data
        soft: If True, use SoftKernelDensityClassifier (manual Gaussian); 
              else use sklearn's KernelDensity
    
    Returns:
        KDE classifier instance
    """
    if soft and kernel == 'gaussian':
        # Use soft implementation with learnable bandwidth
        try:
            bw_val = float(bandwidth)
        except (TypeError, ValueError):
            bw_val = 1.0  # default if 'scott' or 'silverman' string passed
        return SoftKernelDensityClassifier(kernel=kernel, initial_bandwidth=bw_val, 
                                          learnable_bandwidth=learnable)
    else:
        # Use sklearn's KernelDensity
        return KernelDensityClassifier(kernel=kernel, bandwidth=bandwidth, 
                                      learnable_bandwidth=learnable)
