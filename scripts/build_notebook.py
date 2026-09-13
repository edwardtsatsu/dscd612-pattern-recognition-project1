# scripts/build_notebook.py (temporary generator; safe to delete after running once)
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# Project 1: Comparative Study of Bayesian and Non-Parametric Classification
DSCD612 Pattern Recognition — University of Ghana

**Dataset:** UCI Dry Bean Dataset (13,611 observations, 16 morphological features, 7 classes)

This notebook implements and compares a from-scratch multivariate Gaussian Bayesian
classifier against a from-scratch k-NN classifier, following the pattern recognition
pipeline: problem formulation, EDA, feature representation, classifier design,
experimental evaluation, and critical discussion."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """%matplotlib inline
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pattern_reg.data_loading import load_dry_bean_dataset
from pattern_reg.statistics import compute_mean, compute_covariance, compute_correlation, class_conditional_stats
from pattern_reg.preprocessing import prepare_experiment_data
from pattern_reg.bayesian_classifier import GaussianBayesClassifier
from pattern_reg.knn_classifier import KNNClassifier
from pattern_reg.evaluation import evaluate_predictions
from pattern_reg.visualization import (
    plot_correlation_heatmap, plot_class_distribution, plot_pca_scatter,
    plot_confusion_matrix, plot_accuracy_vs_k,
)

X, y = load_dry_bean_dataset()
print(X.shape, y.shape)
X.head()"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 1. Exploratory Statistical Analysis"))
cells.append(
    nbf.v4.new_code_cell(
        """mu = compute_mean(X)
Sigma = compute_covariance(X)
corr = compute_correlation(X)
print("Overall mean vector:\\n", mu)
print("Covariance matrix shape:", Sigma.shape)
fig = plot_correlation_heatmap(corr)
fig"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig = plot_class_distribution(y)
fig"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """stats = class_conditional_stats(X, y)
for label, s in stats.items():
    print(f"{label}: n={s['n']}, prior={s['prior']:.3f}")"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## 2. Preprocessing and Train/Test Split"))
cells.append(
    nbf.v4.new_code_cell(
        """data = prepare_experiment_data(X, y, test_size=0.3, random_state=42)
print("Train:", data.X_train.shape, "Test:", data.X_test.shape)
print("Classes:", data.class_names)
fig = plot_pca_scatter(data.X_train, data.y_train, data.class_names)
fig"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. Bayesian / Parametric Classifier

We model each class as a multivariate Gaussian $p(x|w_i) \\sim \\mathcal{N}(\\mu_i, \\Sigma_i)$
and classify via Bayes' rule $p(w_i|x) = p(x|w_i)P(w_i)/p(x)$, implemented in log-space
for numerical stability (see `pattern_reg.bayesian_classifier.GaussianBayesClassifier`)."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """bayes_clf = GaussianBayesClassifier().fit(data.X_train, data.y_train)
bayes_preds = bayes_clf.predict(data.X_test)
bayes_result = evaluate_predictions(data.y_test, bayes_preds, data.class_names)
print("Bayesian accuracy:", bayes_result["accuracy"])
print(bayes_result["classification_report"])
fig = plot_confusion_matrix(bayes_result["confusion_matrix"], data.class_names, "Bayesian Classifier")
fig"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. Non-Parametric Classification (k-NN)

We sweep $k \\in \\{1,3,5,7,9,11,15,21\\}$ under two distance metrics (Euclidean, Manhattan)."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """k_values = [1, 3, 5, 7, 9, 11, 15, 21]
accuracies_by_metric = {}
knn_results = {}
for metric in ("euclidean", "manhattan"):
    accs = []
    knn_results[metric] = {}
    for k in k_values:
        clf = KNNClassifier(k=k, metric=metric).fit(data.X_train, data.y_train)
        preds = clf.predict(data.X_test)
        result = evaluate_predictions(data.y_test, preds, data.class_names)
        knn_results[metric][k] = result
        accs.append(result["accuracy"])
    accuracies_by_metric[metric] = accs

fig = plot_accuracy_vs_k(k_values, accuracies_by_metric)
fig"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """best_metric, best_k, best_acc = None, None, -1
for metric, per_k in knn_results.items():
    for k, result in per_k.items():
        if result["accuracy"] > best_acc:
            best_metric, best_k, best_acc = metric, k, result["accuracy"]
print(f"Best k-NN: metric={best_metric}, k={best_k}, accuracy={best_acc:.4f}")
best_result = knn_results[best_metric][best_k]
print(best_result["classification_report"])
fig = plot_confusion_matrix(best_result["confusion_matrix"], data.class_names, f"Best k-NN (k={best_k}, {best_metric})")
fig"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. Comparative Evaluation and Discussion

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Gaussian Bayes | see above | | | |
| Best k-NN | see above | | | |

**Central research question:** How do assumptions about the underlying probability
distribution affect the performance of parametric and non-parametric pattern classifiers?

_(Discussion completed in `report/technical_report.md`; this notebook is the
executable companion artifact.)_"""
    )
)

nb["cells"] = cells
Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/project1_bayesian_vs_knn.ipynb", "w") as f:
    nbf.write(nb, f)
print("Notebook written.")
