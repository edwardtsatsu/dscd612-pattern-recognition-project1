"""Orchestrates the full Project 1 pipeline: EDA -> preprocessing ->
Bayesian classifier -> k-NN sweep -> evaluation -> persisted results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pattern_reg.bayesian_classifier import GaussianBayesClassifier
from pattern_reg.evaluation import evaluate_predictions
from pattern_reg.knn_classifier import KNNClassifier
from pattern_reg.preprocessing import prepare_experiment_data
from pattern_reg.statistics import compute_correlation
from pattern_reg.visualization import (
    plot_accuracy_vs_k,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_pca_scatter,
)


def _to_jsonable(result: dict) -> dict:
    return {
        "accuracy": float(result["accuracy"]),
        "precision_macro": float(result["precision_macro"]),
        "recall_macro": float(result["recall_macro"]),
        "f1_macro": float(result["f1_macro"]),
        "confusion_matrix": result["confusion_matrix"].tolist(),
    }


def run_full_experiment(
    output_dir: Path,
    X: pd.DataFrame,
    y: pd.Series,
    k_values: list[int] = (1, 3, 5, 7, 9, 11, 15, 21),
    metrics: tuple[str, ...] = ("euclidean", "manhattan"),
) -> dict:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_experiment_data(X, y)

    # EDA figures
    plot_correlation_heatmap(compute_correlation(X)).savefig(
        figures_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight"
    )
    plot_class_distribution(y).savefig(
        figures_dir / "class_distribution.png", dpi=150, bbox_inches="tight"
    )
    plot_pca_scatter(data.X_train, data.y_train, data.class_names).savefig(
        figures_dir / "pca_scatter.png", dpi=150, bbox_inches="tight"
    )

    # Bayesian classifier
    bayes_clf = GaussianBayesClassifier().fit(data.X_train, data.y_train)
    bayes_preds = bayes_clf.predict(data.X_test)
    bayes_result = evaluate_predictions(data.y_test, bayes_preds, data.class_names)
    plot_confusion_matrix(
        bayes_result["confusion_matrix"],
        data.class_names,
        "Bayesian Classifier Confusion Matrix",
    ).savefig(
        figures_dir / "confusion_matrix_bayesian.png", dpi=150, bbox_inches="tight"
    )

    # k-NN sweep
    knn_results: dict[str, dict[str, dict]] = {}
    accuracies_by_metric: dict[str, list[float]] = {}
    best_config = {"metric": None, "k": None, "accuracy": -1.0}

    for metric in metrics:
        knn_results[metric] = {}
        accuracies_by_metric[metric] = []
        for k in k_values:
            knn_clf = KNNClassifier(k=k, metric=metric).fit(data.X_train, data.y_train)
            knn_preds = knn_clf.predict(data.X_test)
            result = evaluate_predictions(data.y_test, knn_preds, data.class_names)
            knn_results[metric][str(k)] = result
            accuracies_by_metric[metric].append(result["accuracy"])
            if result["accuracy"] > best_config["accuracy"]:
                best_config = {"metric": metric, "k": k, "accuracy": result["accuracy"]}

    plot_accuracy_vs_k(list(k_values), accuracies_by_metric).savefig(
        figures_dir / "knn_accuracy_vs_k.png", dpi=150, bbox_inches="tight"
    )

    best_knn_result = knn_results[best_config["metric"]][str(best_config["k"])]
    plot_confusion_matrix(
        best_knn_result["confusion_matrix"],
        data.class_names,
        f"Best k-NN Confusion Matrix (k={best_config['k']}, {best_config['metric']})",
    ).savefig(
        figures_dir / "confusion_matrix_knn_best.png", dpi=150, bbox_inches="tight"
    )

    results = {
        "bayesian": _to_jsonable(bayes_result),
        "knn": {
            metric: {k_str: _to_jsonable(r) for k_str, r in per_k.items()}
            for metric, per_k in knn_results.items()
        },
        "best_knn_config": {
            "metric": best_config["metric"],
            "k": best_config["k"],
            "accuracy": float(best_config["accuracy"]),
        },
    }

    with open(metrics_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
