# tests/test_bayesian_classifier.py
import numpy as np
import pytest
from pattern_reg.bayesian_classifier import GaussianBayesClassifier


def _well_separated_blobs(seed=0, n_per_class=200):
    rng = np.random.default_rng(seed)
    centers = [(-6, -6), (0, 0), (6, 6)]
    Xs, ys = [], []
    for label, center in enumerate(centers):
        Xs.append(rng.normal(loc=center, scale=0.7, size=(n_per_class, 2)))
        ys.append(np.full(n_per_class, label))
    return np.vstack(Xs), np.concatenate(ys)


def test_fit_recovers_class_means():
    X, y = _well_separated_blobs()
    clf = GaussianBayesClassifier().fit(X, y)
    np.testing.assert_allclose(clf.means_[0], [-6, -6], atol=0.3)
    np.testing.assert_allclose(clf.means_[1], [0, 0], atol=0.3)
    np.testing.assert_allclose(clf.means_[2], [6, 6], atol=0.3)


def test_predict_achieves_high_accuracy_on_separated_blobs():
    X, y = _well_separated_blobs()
    clf = GaussianBayesClassifier().fit(X, y)
    preds = clf.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.97


def test_predict_log_proba_shape_and_argmax_matches_predict():
    X, y = _well_separated_blobs()
    clf = GaussianBayesClassifier().fit(X, y)
    log_proba = clf.predict_log_proba(X)
    assert log_proba.shape == (X.shape[0], 3)
    np.testing.assert_array_equal(np.argmax(log_proba, axis=1), clf.predict(X))


def test_equal_priors_and_shared_covariance_reduces_to_nearest_mean():
    # With equal priors and identical isotropic covariance across classes,
    # the Bayes decision rule reduces to nearest-mean classification.
    rng = np.random.default_rng(1)
    centers = {0: (-4, 0), 1: (4, 0)}
    X = np.vstack(
        [
            rng.normal(loc=centers[0], scale=1.0, size=(100, 2)),
            rng.normal(loc=centers[1], scale=1.0, size=(100, 2)),
        ]
    )
    y = np.array([0] * 100 + [1] * 100)
    clf = GaussianBayesClassifier().fit(X, y)
    query = np.array([[-3.9, 0.1], [3.9, -0.1]])
    preds = clf.predict(query)
    np.testing.assert_array_equal(preds, [0, 1])


def test_singleton_class_raises_error():
    """Regression test: class with only 1 sample should raise ValueError."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])  # Each class has only 1 sample
    clf = GaussianBayesClassifier()
    with pytest.raises(ValueError, match="only 1 sample"):
        clf.fit(X, y)


def test_non_positive_definite_covariance_raises():
    """Regression test: non-PD covariance (from collinear features) should raise ValueError."""
    # Perfectly collinear data: second feature = 2 * first feature (rank-1)
    # With ridge disabled, np.cov produces a singular matrix
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([0, 0, 0])
    clf = GaussianBayesClassifier(reg_covar=0)  # Disable ridge to expose singularity
    with pytest.raises(ValueError, match="not positive-definite"):
        clf.fit(X, y)
