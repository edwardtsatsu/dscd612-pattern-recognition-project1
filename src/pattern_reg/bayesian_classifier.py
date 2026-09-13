"""From-scratch multivariate Gaussian Bayes classifier (QDA-style).

For each class w_k, estimates the class-conditional density p(x|w_k) as a
multivariate Gaussian N(mu_k, Sigma_k), then classifies via Bayes' rule:

    p(w_k|x) = p(x|w_k) P(w_k) / p(x)

Since p(x) is constant across classes, argmax_k p(w_k|x) = argmax_k p(x|w_k)P(w_k).
We work in log-space for numerical stability:

    log p(x|w_k) = -0.5*log|Sigma_k| - 0.5*(x-mu_k)^T Sigma_k^-1 (x-mu_k) - (d/2)*log(2*pi)
"""
from __future__ import annotations

import numpy as np


class GaussianBayesClassifier:
    def __init__(self, reg_covar: float = 1e-6):
        self.reg_covar = reg_covar
        self.classes_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None
        self._precisions_: np.ndarray | None = None
        self._log_dets_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianBayesClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        means, covariances, priors, precisions, log_dets = [], [], [], [], []
        for label in self.classes_:
            X_c = X[y == label]
            n_samples_class = len(X_c)
            if n_samples_class < 2:
                raise ValueError(
                    f"Class {label} has only {n_samples_class} sample(s), but at least 2 are "
                    f"required to estimate covariance."
                )
            mean = X_c.mean(axis=0)
            cov = np.cov(X_c, rowvar=False) + self.reg_covar * np.eye(n_features)
            means.append(mean)
            covariances.append(cov)
            priors.append(len(X_c) / len(X))
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError(
                    f"Covariance matrix for class {label} is not positive-definite "
                    f"(sign={sign}). This may indicate numerical issues or insufficient "
                    f"feature diversity in the data."
                )
            precisions.append(np.linalg.inv(cov))
            log_dets.append(logdet)

        self.means_ = np.array(means)
        self.covariances_ = np.array(covariances)
        self.priors_ = np.array(priors)
        self._precisions_ = np.array(precisions)
        self._log_dets_ = np.array(log_dets)
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        log_probs = np.zeros((n_samples, n_classes))

        for k in range(n_classes):
            diff = X - self.means_[k]
            mahalanobis = np.einsum("ij,jk,ik->i", diff, self._precisions_[k], diff)
            log_probs[:, k] = (
                -0.5 * self._log_dets_[k]
                - 0.5 * mahalanobis
                - 0.5 * n_features * np.log(2 * np.pi)
                + np.log(self.priors_[k])
            )
        return log_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
