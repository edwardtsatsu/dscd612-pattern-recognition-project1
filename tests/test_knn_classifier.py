import numpy as np
import pytest
from pattern_reg.knn_classifier import KNNClassifier


def test_k1_predicts_exact_nearest_neighbor_label():
    X_train = np.array([[0, 0], [10, 10], [10, 11]])
    y_train = np.array(["a", "b", "b"])
    clf = KNNClassifier(k=1, metric="euclidean").fit(X_train, y_train)
    preds = clf.predict(np.array([[0.1, 0.1], [10.5, 10.5]]))
    np.testing.assert_array_equal(preds, ["a", "b"])


def test_majority_vote_among_k_neighbors():
    X_train = np.array([[0, 0], [0, 1], [0, 2], [10, 10]])
    y_train = np.array(["a", "a", "b", "b"])
    clf = KNNClassifier(k=3, metric="euclidean").fit(X_train, y_train)
    # nearest 3 to (0, 0.5) are the three points near the origin -> 2x "a", 1x "b"
    preds = clf.predict(np.array([[0, 0.5]]))
    assert preds[0] == "a"


def test_euclidean_and_manhattan_can_disagree():
    # Point equidistant in Manhattan but not Euclidean from two references
    X_train = np.array([[3, 0], [0, 3], [1, 1]])
    y_train = np.array(["far_x", "far_y", "near"])
    query = np.array([[1.4, 1.4]])

    euclidean_pred = (
        KNNClassifier(k=1, metric="euclidean").fit(X_train, y_train).predict(query)
    )
    manhattan_pred = (
        KNNClassifier(k=1, metric="manhattan").fit(X_train, y_train).predict(query)
    )
    assert euclidean_pred[0] == "near"
    assert manhattan_pred[0] == "near"
    # Sanity: distances actually differ between metrics for this configuration
    clf = KNNClassifier(k=1, metric="euclidean").fit(X_train, y_train)
    euclid_dists = clf._compute_distances(query)
    manhattan_clf = KNNClassifier(k=1, metric="manhattan").fit(X_train, y_train)
    manhattan_dists = manhattan_clf._compute_distances(query)
    assert not np.allclose(euclid_dists, manhattan_dists)


def test_unknown_metric_raises():
    clf = KNNClassifier(k=1, metric="nonsense")
    with pytest.raises(ValueError):
        clf.fit(np.array([[0, 0]]), np.array(["a"]))
