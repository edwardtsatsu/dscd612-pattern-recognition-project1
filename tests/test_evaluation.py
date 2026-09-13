import numpy as np
from pattern_reg.evaluation import evaluate_predictions
from sklearn.metrics import f1_score, precision_score, recall_score


def test_perfect_predictions_score_one():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    result = evaluate_predictions(y_true, y_pred, class_names=["a", "b", "c"])
    assert result["accuracy"] == 1.0
    assert result["precision_macro"] == 1.0
    assert result["recall_macro"] == 1.0
    assert result["f1_macro"] == 1.0
    np.testing.assert_array_equal(result["confusion_matrix"], np.diag([2, 2, 2]))


def test_known_confusion_matrix_case():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = evaluate_predictions(y_true, y_pred, class_names=["a", "b"])
    assert result["accuracy"] == 0.75
    expected_cm = np.array([[1, 1], [0, 2]])
    np.testing.assert_array_equal(result["confusion_matrix"], expected_cm)
    assert "a" in result["classification_report"]
    assert "b" in result["classification_report"]


def test_missing_class_label_scoping():
    """Regression test: macro metrics must include all class_names even if missing from y_true/y_pred."""
    # Only classes 0 and 1 appear in data, but class_names has 3 classes
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = evaluate_predictions(y_true, y_pred, class_names=["a", "b", "c"])

    # Reference: sklearn metrics with explicit labels parameter
    labels = [0, 1, 2]
    expected_precision = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    expected_recall = recall_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    expected_f1 = f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    # Assert our implementation matches sklearn with labels parameter
    assert result["precision_macro"] == expected_precision
    assert result["recall_macro"] == expected_recall
    assert result["f1_macro"] == expected_f1

    # Confusion matrix should include all labels (including missing class 2)
    assert result["confusion_matrix"].shape == (3, 3)
    # Classification report should include all class names
    assert "a" in result["classification_report"]
    assert "b" in result["classification_report"]
    assert "c" in result["classification_report"]
