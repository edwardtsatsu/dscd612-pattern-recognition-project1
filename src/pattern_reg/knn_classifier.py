"""From-scratch k-Nearest Neighbours classifier with pluggable distance metrics.

Classification rule: for a query x, find the k training points minimizing the
chosen distance d(x, x_i), then assign the majority label among those k
neighbours (ties broken by ascending label order).
"""
from __future__ import annotations

import numpy as np

_SUPPORTED_METRICS = {"euclidean", "manhattan"}


class KNNClassifier:
    def __init__(self, k: int = 5, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        if self.metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{self.metric}', expected one of {_SUPPORTED_METRICS}"
            )
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        diff = X[:, None, :] - self.X_train_[None, :, :]
        if self.metric == "euclidean":
            return np.sqrt(np.sum(diff**2, axis=2))
        if self.metric == "manhattan":
            return np.sum(np.abs(diff), axis=2)
        raise ValueError(f"Unsupported metric '{self.metric}'")

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self._compute_distances(X)
        neighbor_idx = np.argsort(distances, axis=1)[:, : self.k]
        neighbor_labels = self.y_train_[neighbor_idx]
        predictions = np.array(
            [self._majority_vote(labels) for labels in neighbor_labels]
        )
        return predictions

    def _majority_vote(self, labels: np.ndarray) -> any:
        unique_labels, counts = np.unique(labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
