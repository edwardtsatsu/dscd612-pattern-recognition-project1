import json

import numpy as np
import pandas as pd
from pattern_reg.experiment import run_full_experiment


def _synthetic_dataset(n_per_class=60):
    rng = np.random.default_rng(7)
    centers = [(-5, -5), (0, 0), (5, 5)]
    frames, labels = [], []
    for i, c in enumerate(centers):
        frames.append(rng.normal(loc=c, scale=1.0, size=(n_per_class, 2)))
        labels.extend([f"class_{i}"] * n_per_class)
    X = pd.DataFrame(np.vstack(frames), columns=["f1", "f2"])
    # (columns f3/f4 correlated copies of f1/f2 keep it simple but 4-D)
    X["f3"] = X["f1"] * 0.5
    X["f4"] = X["f2"] * 0.5
    y = pd.Series(labels)
    return X, y


def test_run_full_experiment_writes_outputs_and_returns_results(tmp_path):
    X, y = _synthetic_dataset()
    results = run_full_experiment(
        output_dir=tmp_path,
        k_values=[1, 3, 5],
        metrics=("euclidean", "manhattan"),
        X=X,
        y=y,
    )
    assert (tmp_path / "metrics" / "results.json").exists()
    with open(tmp_path / "metrics" / "results.json") as f:
        saved = json.load(f)
    assert "bayesian" in saved and "knn" in saved and "best_knn_config" in saved

    assert results["bayesian"]["accuracy"] > 0.8
    for metric in ("euclidean", "manhattan"):
        for k in (1, 3, 5):
            assert results["knn"][metric][str(k)]["accuracy"] > 0.0

    figures_dir = tmp_path / "figures"
    png_files = list(figures_dir.glob("*.png"))
    assert len(png_files) >= 5
