"""Quantitative evaluation: accuracy, macro precision/recall/F1, confusion matrix."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
        ),
    }
