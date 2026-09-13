"""Exploratory statistics: mean vector, covariance matrix, correlations,
and per-class conditional statistics used by the Bayesian classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_mean(X: pd.DataFrame) -> np.ndarray:
    return X.to_numpy().mean(axis=0)


def compute_covariance(X: pd.DataFrame) -> np.ndarray:
    return np.cov(X.to_numpy(), rowvar=False)


def compute_correlation(X: pd.DataFrame) -> pd.DataFrame:
    return X.corr()


def class_conditional_stats(X: pd.DataFrame, y: pd.Series) -> dict[str, dict]:
    n_total = len(y)
    stats: dict[str, dict] = {}
    for label in sorted(y.unique()):
        mask = (y == label).to_numpy()
        X_class = X.loc[mask]
        stats[label] = {
            "mean": compute_mean(X_class),
            "cov": compute_covariance(X_class),
            "prior": mask.sum() / n_total,
            "n": int(mask.sum()),
        }
    return stats
