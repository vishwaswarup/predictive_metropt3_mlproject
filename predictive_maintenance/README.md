# MetroPT-3 Predictive Maintenance

Modular pipeline for the metro air-compressor (APU): anomaly detection (LSTM autoencoder),
failure prediction (XGBoost + SHAP), RUL estimation, and a rule-based maintenance scheduler.

**Status:** Step 1 (EDA, scaffolding, config) complete. See `notebooks/01_eda.ipynb`.

## Setup
    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
    # raw CSV goes in data/ (currently a symlink to ../MetroPT3(AirCompressor).csv)

All hyperparameters live in `config.py`.
