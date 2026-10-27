# MetroPT-3 Predictive Maintenance

Modular, interpretable pipeline for the metro Air Production Unit (compressor), built on the
[MetroPT-3 dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset):

1. **Anomaly detection** – LSTM autoencoder, trained only on pre-failure normal data
2. **Failure prediction** – XGBoost + SHAP
3. **RUL estimation** – piecewise-linear degradation target, gradient boosting
4. **Maintenance decisions** – rule-based scheduler with a likelihood x consequence risk matrix

## Status and honest results

| Step | Module | State / finding |
|---|---|---|
| 1 | EDA, config | done (`notebooks/01_eda.ipynb`) |
| 2 | `src/data/loader.py`, `preprocessor.py` | done |
| 3 | `src/models/autoencoder.py` | works as a detector: flags ~100% of in-failure windows; weak early warning; ~21% false-positive rate (drift) |
| 4 | `src/models/classifier.py` | **negative result**: leave-one-failure-out AUC 0.53 (chance); SHAP importances are not physical causes |
| 5 | `src/models/rul_estimator.py` | **no skill**: MAE 19.7 h vs 17.4 h for a constant predictor |
| 6 | `src/maintenance/scheduler.py` | anomaly-persistence driven; classifier/RUL corroboration off by default (it increased false alarms) |
| 7 | `src/visualization/dashboard.py`, `main.py` | done; writes `outputs/plots/dashboard.html` |

Only the anomaly detector is validated. The scheduler reaches SCHEDULE_MAINTENANCE/URGENT mostly after a failure
has started (F1 is the exception, ~13 h ahead) and raises false alarms during drift periods.

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
    .venv/bin/python -m src.models.classifier
    .venv/bin/python -m src.models.rul_estimator
    .venv/bin/python -m src.maintenance.scheduler
    .venv/bin/python main.py                       # everything after training, plus the dashboard

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
    src/models/          autoencoder, classifier, rul_estimator
    src/maintenance/     scheduler
    src/visualization/   dashboard
    outputs/models|plots|reports/   saved models, figures, metrics
