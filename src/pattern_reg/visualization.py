"""EDA and results plotting for the report/notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA


def plot_correlation_heatmap(corr: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix")
    return fig


def plot_class_distribution(y: pd.Series) -> Figure:
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    counts = y.value_counts()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_pca_scatter(X: np.ndarray, y: np.ndarray, class_names: list[str]) -> Figure:
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for label in np.unique(y):
        mask = y == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=10, label=class_names[label], alpha=0.6
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("2D PCA Projection")
    ax.legend(markerscale=2, fontsize="small")
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str) -> Figure:
    fig = Figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return fig


def plot_accuracy_vs_k(
    k_values: list[int], accuracies_by_metric: dict[str, list[float]]
) -> Figure:
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for metric_name, accuracies in accuracies_by_metric.items():
        ax.plot(k_values, accuracies, marker="o", label=metric_name)
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_title("k-NN Accuracy vs. k by Distance Metric")
    ax.legend()
    return fig
