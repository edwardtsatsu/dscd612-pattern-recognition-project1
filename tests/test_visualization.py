import matplotlib

matplotlib.use("Agg")  # headless backend for CI/test runs

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pattern_reg.visualization import (
    plot_accuracy_vs_k,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_pca_scatter,
)


def test_plot_correlation_heatmap_returns_figure():
    corr = pd.DataFrame(np.eye(4), columns=list("abcd"), index=list("abcd"))
    fig = plot_correlation_heatmap(corr)
    assert isinstance(fig, Figure)


def test_plot_class_distribution_returns_figure():
    y = pd.Series(["a", "a", "b", "c", "c", "c"])
    fig = plot_class_distribution(y)
    assert isinstance(fig, Figure)


def test_plot_pca_scatter_returns_figure():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    fig = plot_pca_scatter(X, y, class_names=["a", "b", "c"])
    assert isinstance(fig, Figure)


def test_plot_confusion_matrix_returns_figure():
    cm = np.array([[5, 1], [2, 4]])
    fig = plot_confusion_matrix(cm, class_names=["a", "b"], title="Test CM")
    assert isinstance(fig, Figure)


def test_plot_accuracy_vs_k_returns_figure():
    fig = plot_accuracy_vs_k(
        [1, 3, 5], {"euclidean": [0.9, 0.92, 0.91], "manhattan": [0.88, 0.9, 0.89]}
    )
    assert isinstance(fig, Figure)
