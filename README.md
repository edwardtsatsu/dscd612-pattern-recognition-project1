# DSCD612 Pattern Recognition: Project 1

**Comparative Study of Bayesian and Non-Parametric Classification**

MPhil/MSc Data Science, University of Ghana. Dataset: UCI Dry Bean Dataset
(13,611 observations, 16 features, 7 classes).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edwardtsatsu/dscd612-pattern-recognition-project1/blob/main/notebooks/project1_bayesian_vs_knn.ipynb)

## Results

| Model | Accuracy |
|---|---:|
| Gaussian Bayes (from scratch, QDA) | 0.9082 |
| k-NN (from scratch, Euclidean, k=9) | 0.9202 |

Full write-up: [`report/technical_report.pdf`](report/technical_report.pdf) /
[`report/technical_report.docx`](report/technical_report.docx).

## Contents
- `report/`: technical report (md, docx, pdf) covering problem formulation,
  EDA, the mathematical basis of both classifiers, experimental design,
  quantitative evaluation, visualization, limitations, and the central
  research question.
- `notebooks/project1_bayesian_vs_knn.ipynb`: already-executed notebook
  reproducing every result and figure in the report. Click the badge above
  to open and run it live in Google Colab.
- `src/pattern_reg/`: the Python package, from-scratch Gaussian Bayes and
  k-NN classifiers, statistics, preprocessing, evaluation, visualization.
- `tests/`: pytest suite (27 tests) covering every module.
- `scripts/`: `run_experiment.py` (reproduces `results/metrics/results.json`
  end-to-end) and `build_notebook.py` (regenerates the notebook's cells).
- `results/`: persisted metrics (`results.json`) and all generated figures.
- `data/raw/Dry_Bean_Dataset.xlsx`: the dataset itself, bundled so the
  pipeline reproduces fully offline.

## Reproducing locally
```bash
pip install -r requirements.txt
python scripts/run_experiment.py
jupyter nbconvert --to notebook --execute --inplace notebooks/project1_bayesian_vs_knn.ipynb
```

## Tests
```bash
python -m pytest tests/ -v
```
