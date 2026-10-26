"""Central configuration for the MetroPT-3 predictive maintenance pipeline.

Every hyperparameter, threshold and path lives here. Modules import from this file
and must never hardcode these values. Values marked PLACEHOLDER are provisional and
will be tuned/confirmed as each pipeline stage is built.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"
RAW_CSV_GLOB = "MetroPT3*.csv"

# --------------------------------------------------------------------------- schema
TIMESTAMP_COL = "timestamp"
# COMP is stored as float, so it is grouped with the analogue columns and MinMax-scaled
# (it is 0/1, so scaling leaves it unchanged on the training range).
ANALOGUE_COLS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                 "Oil_temperature", "Motor_current", "COMP"]
DIGITAL_COLS = ["DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch",
                "Oil_level", "Caudal_impulses"]
FEATURE_COLS = ANALOGUE_COLS + DIGITAL_COLS   # model input order, shared by every module

# --------------------------------------------------------------------------- failure event(s)
# Documented air-leak failures from the UCI MetroPT-3 page (verified against the
# sensor signatures in notebooks/01_eda.ipynb). The CSV has NO label column.
# Each entry is (start, end); the ground-truth failure time used for labels / RUL
# is the `start` of the window.
FAILURE_WINDOWS = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:59"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]

# --------------------------------------------------------------------------- preprocessing
SAMPLING_INTERVAL_S = 10       # native cadence of the CSV (0.1 Hz), verified in EDA
SAMPLING_FREQ = f"{SAMPLING_INTERVAL_S}s"
WINDOW_SIZE = 60               # samples = 10 minutes
STEP_SIZE = 6                  # samples = 1-minute stride
FORWARD_FILL_MAX_STEPS = 3     # gaps of <= 3 missing steps (30 s) are forward-filled; longer ones are is_gap

# Split definitions (dates are inclusive, whole days).
TRAIN_END = "2020-04-14"                       # autoencoder trains ONLY on data up to this day
EXCLUDE_FROM_TRAINING = ["2020-03-29"]         # undocumented anomaly candidate: evaluated separately, never trained on
SENSOR_FAULT_DATES = ["2020-05-27", "2020-06-23", "2020-06-24"]   # TP3 flat-lined: no window may touch these days
TRAIN_FRACTION = 0.8           # chronological share of train-split windows used for AE fitting
VAL_FRACTION = 0.2             # remaining chronological share, used for validation / threshold calibration

# --------------------------------------------------------------------------- autoencoder
ANOMALY_THRESHOLD_MULTIPLIER = 3.0   # threshold = mean + k * std of training reconstruction error
LSTM_UNITS = [64, 32]                # encoder LSTM widths (decoder mirrors)
LSTM_EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
DROPOUT_RATE = 0.2
PREDICT_BATCH_SIZE = 1024            # inference batch size (reconstruction errors)
DETECTION_MIN_FLAG_RATE = 0.5        # a period counts as "detected" if >= this share of its windows are flagged

# --------------------------------------------------------------------------- classifier
FAILURE_LOOKAHEAD_HOURS = 24         # PLACEHOLDER: label = 1 if within N hours before failure
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
CLASSIFIER_THRESHOLD = 0.5

# --------------------------------------------------------------------------- RUL
MAX_RUL = 72                         # PLACEHOLDER: hours; RUL is capped here (piecewise linear)
RUL_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
}

# --------------------------------------------------------------------------- maintenance scheduler
# PLACEHOLDER thresholds; refined when the scheduler is built.
SCHED_ANOMALY_WATCH = 1.0            # anomaly score (error / threshold) above which to WATCH
SCHED_PROB_SCHEDULE = 0.5
SCHED_PROB_URGENT = 0.8
SCHED_RUL_SCHEDULE_HOURS = 48
SCHED_RUL_URGENT_HOURS = 12

# --------------------------------------------------------------------------- reproducibility
RANDOM_SEED = 42
