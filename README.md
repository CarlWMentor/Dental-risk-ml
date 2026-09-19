# Dental-risk-ml
First ML portfolio project — dental/health risk prediction (public toy data)
# Dental-risk-ml

First machine-learning portfolio project: predict a simple health/dental-related risk label from **public toy tabular data** (no real patient or military records).

Built as a learning project while studying computer science part-time. Focus: clean workflow, honest metrics, and clear use of AI assistants as tools.

## Goal
Train a baseline classifier (scikit-learn) that separates higher vs lower risk examples, then document accuracy, confusion matrix, and limits of the model.

## Repo layout
- `data/` — public CSV only (source cited in this README)
- `notebooks/` — exploration (`01_eda.ipynb`)
- `src/` — load → train → evaluate scripts
- `models/` — saved model artifacts (optional)
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.

Attach files by dragging & dropping, selecting or pasting them.

## Stack
Python 3 · pandas · scikit-learn · Jupyter

## How I used AI tools
- **Claude** — help draft the sklearn pipeline and train/test split
- **ChatGPT** — polish wording in docs
- **Carl (Grok Bot)** — project scope, repo structure, review for overclaims / data leakage
I still choose the problem, check every metric, and own the commits.

## Status
Scaffold only — dataset + first model coming next.

## Run (later)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb

## Data

This project uses the Cleveland subset of the UCI Heart Disease dataset (Janosi et al., 1989), available at https://archive.ics.uci.edu/dataset/45/heart+disease.

Janosi, A., Steinbrunn, W., Pfisterer, M. and Detrano, R. 1989. *Heart Disease* [Dataset]. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C52P4X

## Results (baseline)
- Cleaned missing `?` values → 297 rows
- Binary `risk` label from original `target` (0 vs 1–4)
- Model: LogisticRegression, 80/20 stratified split
- Test accuracy ≈ 0.83
- This is a learning project on public data, not a clinical tool
