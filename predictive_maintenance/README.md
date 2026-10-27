# CompressorWatch

Fault detection for the MetroPT-3 metro air compressor. An LSTM autoencoder learns normal behaviour from
healthy history and raises a graded alert (WATCH / SCHEDULE / URGENT) when the sensors stop looking normal.
Dataset: [MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset). Author: Vishwaswarup Rath.

## What works and what does not

| Part | Module | Result |
|---|---|---|
| Fault detection | `src/models/autoencoder.py` | **Works.** 99.6% of windows during the four documented failures sit above NORMAL. ~20% of healthy windows do too (mostly drift in August). |
| Alert levels | `src/maintenance/scheduler.py` | Rules on how long the anomaly persists. Alerts fire at or after the logged failure start, except F1 (~13.6 h earlier). |
| Failure prediction | `src/models/classifier.py` | **Dropped.** Leave-one-failure-out ROC-AUC 0.53 (chance). Kept as a documented negative result. |
| Remaining life | `src/models/rul_estimator.py` | **Dropped.** MAE 19.7 h vs 17.4 h for a constant guess. Kept as a documented negative result. |

CompressorWatch is a fault alarm, not a predictor. The full write-up is the report site in `site/`
(static HTML, deployable on Vercel with Root Directory `predictive_maintenance/site`).

## Next steps

Per-sensor explanations for each alert, a drift fix to reduce false alarms, a stuck-sensor check,
and a comparison against a plain threshold rule.

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
