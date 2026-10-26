# MetroPT-3 Predictive Maintenance

Modular, interpretable pipeline for the metro Air Production Unit (compressor), built on the
[MetroPT-3 dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset):

1. **Anomaly detection** – LSTM autoencoder, trained only on pre-failure normal data
2. **Failure prediction** – XGBoost + SHAP *(planned)*
3. **RUL estimation** – piecewise-linear degradation target *(planned)*
4. **Maintenance decisions** – rule-based scheduler with a risk matrix *(planned)*

## Status

| Step | Module | State |
|---|---|---|
| 1 | EDA, config | done (`notebooks/01_eda.ipynb`) |
| 2 | `src/data/loader.py`, `preprocessor.py` | done |
| 3 | `src/models/autoencoder.py` | written, trained, under review |
| 4 | `src/models/classifier.py` | not started |
| 5 | `src/models/rul_estimator.py` | not started |
| 6 | `src/maintenance/scheduler.py` | not started |
| 7 | `src/visualization/dashboard.py`, `main.py` | not started |

## Setup

Python 3.13 on Apple Silicon (arm64). `tensorflow-metal` has no wheels for 3.13, so TensorFlow runs on CPU.

    python3.13 -m venv .venv
    .venv/bin/pip install -r requirements.txt

Download `MetroPT3(AirCompressor).csv` from the UCI page above and put it in `data/`
(it is git-ignored: 213 MB, above GitHub's 100 MB file limit).

## Run

Run each module from this folder; each has a `__main__` self-test:

    .venv/bin/python -m src.data.loader
    .venv/bin/python -m src.data.preprocessor
    .venv/bin/python -m src.models.autoencoder     # reuses outputs/models/autoencoder.keras if present

All hyperparameters, paths, split dates and thresholds live in `config.py`.

## Data facts that shape the design

- Sampling is ~10 s (0.1 Hz), not 1 Hz; long logging gaps are marked, never bridged by a window.
- There are **four** documented air-leak failures (no label column); see `FAILURE_WINDOWS`.
- The autoencoder trains only on 2020-02-01 to 2020-04-14 (a 72 h buffer before the first failure);
  2020-03-29 is excluded from training as an undocumented anomaly candidate;
  three TP3 flat-lined days are masked everywhere.

## Layout

    config.py            all hyperparameters and paths
    data/                raw CSV (git-ignored)
    docs/                reference paper abstract
    notebooks/           01_eda.ipynb
    src/data/            loader, preprocessor
    src/models/          autoencoder (classifier, rul_estimator to come)
    src/maintenance/     scheduler (to come)
    src/visualization/   dashboard (to come)
    outputs/models|plots|reports/   saved models, figures, metrics
