# tests/test_statistics.py
import numpy as np
import pandas as pd
from pattern_reg.statistics import (
    class_conditional_stats,
    compute_correlation,
    compute_covariance,
    compute_mean,
)


def test_compute_mean_matches_numpy():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    mu = compute_mean(X)
    np.testing.assert_allclose(mu, np.array([2.0, 5.0]))


def test_compute_covariance_matches_numpy_rowvar_false():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 3))
    X = pd.DataFrame(data, columns=["a", "b", "c"])
    cov = compute_covariance(X)
    expected = np.cov(data, rowvar=False)
    np.testing.assert_allclose(cov, expected, rtol=1e-8)


def test_compute_correlation_diagonal_is_one():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    corr = compute_correlation(X)
    np.testing.assert_allclose(np.diag(corr.values), [1.0, 1.0])
    assert corr.loc["a", "b"] == 1.0  # perfectly correlated by construction


def test_class_conditional_stats_priors_sum_to_one():
    X = pd.DataFrame({"a": [1.0, 2.0, 10.0, 11.0], "b": [1.0, 1.0, 10.0, 10.0]})
    y = pd.Series(["low", "low", "high", "high"])
    stats = class_conditional_stats(X, y)
    assert set(stats.keys()) == {"low", "high"}
    assert stats["low"]["n"] == 2
    total_prior = sum(s["prior"] for s in stats.values())
    assert abs(total_prior - 1.0) < 1e-9
    np.testing.assert_allclose(stats["low"]["mean"], [1.5, 1.0])
