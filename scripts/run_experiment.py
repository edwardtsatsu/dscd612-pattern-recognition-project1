"""Entry point: run the full Project 1 experiment on the real Dry Bean dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pattern_reg.data_loading import load_dry_bean_dataset
from pattern_reg.experiment import run_full_experiment

if __name__ == "__main__":
    X, y = load_dry_bean_dataset()
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results = run_full_experiment(output_dir=results_dir, X=X, y=y)
    print("Bayesian accuracy:", results["bayesian"]["accuracy"])
    print("Best k-NN config:", results["best_knn_config"])
