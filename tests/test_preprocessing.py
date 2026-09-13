import numpy as np
import pandas as pd
from pattern_reg.preprocessing import prepare_experiment_data


def _toy_data(n_per_class=40):
    rng = np.random.default_rng(1)
    frames = []
    labels = []
    for i, center in enumerate([(-5, -5), (0, 0), (5, 5)]):
        pts = rng.normal(loc=center, scale=0.5, size=(n_per_class, 2))
        frames.append(pts)
        labels.extend([f"class_{i}"] * n_per_class)
    X = pd.DataFrame(np.vstack(frames), columns=["f1", "f2"])
    y = pd.Series(labels)
    return X, y


def test_prepare_experiment_data_shapes_and_stratification():
    X, y = _toy_data()
    data = prepare_experiment_data(X, y, test_size=0.3, random_state=42)
    assert data.X_train.shape[0] + data.X_test.shape[0] == len(X)
    assert data.X_train.shape[1] == 2
    assert len(data.class_names) == 3
    train_counts = np.bincount(data.y_train)
    test_counts = np.bincount(data.y_test)
    # stratified split keeps class proportions close to equal
    assert train_counts.max() - train_counts.min() <= 2
    assert test_counts.max() - test_counts.min() <= 2


def test_prepare_experiment_data_scaling_uses_train_stats_only():
    X, y = _toy_data()
    data = prepare_experiment_data(X, y, test_size=0.3, random_state=42)
    # scaled training data should have ~zero mean, unit variance per feature
    np.testing.assert_allclose(data.X_train.mean(axis=0), [0, 0], atol=1e-8)
    np.testing.assert_allclose(data.X_train.std(axis=0), [1, 1], atol=1e-8)


def test_prepare_experiment_data_is_deterministic():
    X, y = _toy_data()
    d1 = prepare_experiment_data(X, y, test_size=0.3, random_state=42)
    d2 = prepare_experiment_data(X, y, test_size=0.3, random_state=42)
    np.testing.assert_array_equal(d1.y_train, d2.y_train)
    np.testing.assert_allclose(d1.X_train, d2.X_train)
