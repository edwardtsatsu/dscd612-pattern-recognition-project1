# DSCD612 Pattern Recognition: Project 1 Submission

**Student:** Edward Akorlie
**Project:** Project 1, Comparative Study of Bayesian and Non-Parametric Classification
**Dataset:** UCI Dry Bean Dataset (13,611 observations, 16 features, 7 classes)

## Contents
- `report/technical_report.docx` / `.pdf`: full write-up (problem
  formulation through critical discussion and the central research question).
- `notebooks/project1_bayesian_vs_knn.ipynb` (+ `.html`): already-executed
  notebook reproducing every result and figure in the report; open it
  directly to see all outputs without re-running anything.
- `src/pattern_reg/`: the Python package (from-scratch Gaussian Bayes and
  k-NN classifiers, statistics, preprocessing, evaluation, visualization).
- `tests/`: the pytest suite (27 tests) covering every module.
- `scripts/`: `run_experiment.py` (reproduces `results/metrics/results.json`
  end-to-end) and `build_notebook.py` (regenerates the notebook's cells).
- `results/metrics/results.json`: every metric for the Bayesian classifier
  and all 16 k-NN (metric, k) configurations. `results/figures/`: all 6
  generated plots.
- `data/raw/Dry_Bean_Dataset.xlsx`: the UCI Dry Bean Dataset itself, bundled
  so the pipeline reproduces fully offline (no download required).
- `pyproject.toml`, `requirements.txt`: needed for `pytest`/`pip install` to
  work directly from this extracted folder.

## Reproducing the results (fully offline, the dataset is already included)
```bash
pip install -r requirements.txt
python scripts/run_experiment.py
jupyter nbconvert --to notebook --execute --inplace notebooks/project1_bayesian_vs_knn.ipynb
```

## Tests
```bash
python -m pytest tests/ -v
```
