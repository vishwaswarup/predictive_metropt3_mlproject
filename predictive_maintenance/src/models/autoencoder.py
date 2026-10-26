"""LSTM autoencoder for reconstruction-based anomaly detection.

Trained only on pre-failure normal data (``config.TRAIN_END``); the anomaly score of a
window is its mean absolute reconstruction error.

Run from the project root:  python -m src.models.autoencoder
(an existing outputs/models/autoencoder.keras is reused; delete it to retrain)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.metrics import roc_auc_score

import config as C
from src.data.preprocessor import compute_rul, in_failure_mask, label_windows

MODEL_PATH = C.MODEL_DIR / "autoencoder.keras"
THRESHOLD_PATH = C.MODEL_DIR / "threshold.json"
PLOT_PATH = C.PLOT_DIR / "reconstruction_error.png"
FailureWindows = Sequence[tuple[str, str]]


# --------------------------------------------------------------------------- model
def build_autoencoder(window_size: int = C.WINDOW_SIZE, n_features: int = len(C.FEATURE_COLS)) -> keras.Model:
    """Build and compile the LSTM autoencoder, then print its summary.

    Input:  window length (timesteps) and number of features per timestep.
    Output: compiled ``keras.Model`` mapping ``[B, T, F] -> [B, T, F]``.
    Logic:  encoder LSTM(u0, seq) -> Dropout -> LSTM(u1) squeezes each window to one vector;
            RepeatVector(T) feeds it to every decoder step; decoder LSTM(u1, seq) ->
            LSTM(u0, seq) -> Dropout -> TimeDistributed(Dense(F)). Widths u0, u1 come from
            ``config.LSTM_UNITS``. Loss is MAE (mean over timesteps and features), Adam optimiser.
    """
    u0, u1 = C.LSTM_UNITS
    inp = keras.Input(shape=(window_size, n_features), name="window")
    x = layers.LSTM(u0, return_sequences=True, name="enc_lstm_1")(inp)
    x = layers.Dropout(C.DROPOUT_RATE, name="enc_dropout")(x)
    x = layers.LSTM(u1, return_sequences=False, name="enc_lstm_2")(x)
    x = layers.RepeatVector(window_size, name="bottleneck_repeat")(x)
    x = layers.LSTM(u1, return_sequences=True, name="dec_lstm_1")(x)
    x = layers.LSTM(u0, return_sequences=True, name="dec_lstm_2")(x)
    x = layers.Dropout(C.DROPOUT_RATE, name="dec_dropout")(x)
    out = layers.TimeDistributed(layers.Dense(n_features), name="reconstruction")(x)

    model = keras.Model(inp, out, name="lstm_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(C.LEARNING_RATE), loss="mae")
    model.summary()
    return model


def split_train_val(X: np.ndarray, val_fraction: float = C.VAL_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Chronological train/validation split of time-ordered windows.

    Input:  windows ordered by time; fraction reserved for validation.
    Output: ``(X_train, X_val)`` where validation is the *last* ``val_fraction`` of the windows.
    Logic:  a random split would put near-duplicate neighbours (windows overlap by 90%) on both
            sides and make validation loss meaninglessly optimistic. A chronological cut avoids it.
    """
    n_val = int(round(len(X) * val_fraction))
    return X[: len(X) - n_val], X[len(X) - n_val:]


class _EarlyEpochPrinter(keras.callbacks.Callback):
    """Print loss lines for the first few epochs only; later progress comes from the other callbacks."""

    def __init__(self, n_first: int = 3):
        super().__init__()
        self.n_first = n_first

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.n_first:
            logs = logs or {}
            print(f"  epoch {epoch + 1:>2}: loss={logs.get('loss'):.5f}  val_loss={logs.get('val_loss'):.5f}  "
                  f"lr={float(self.model.optimizer.learning_rate):.2e}", flush=True)


def train(X_train: np.ndarray, X_val: np.ndarray, model: keras.Model | None = None) -> dict:
    """Fit the autoencoder to reconstruct normal windows.

    Input:  training windows (shuffled here with the config seed) and ordered validation windows,
            both ``[N, T, F]``; optional pre-built model (default: :func:`build_autoencoder`).
    Output: history dict with per-epoch ``loss``, ``val_loss``, ``lr`` plus ``best_epoch`` (1-based),
            ``best_val_loss`` and ``epochs_run``. The best weights are restored on the model and
            checkpointed to ``MODEL_PATH``.
    Logic:  target = input (unsupervised). EarlyStopping (restore best), ReduceLROnPlateau and
            ModelCheckpoint(best val_loss) per config. Validation is never shuffled.
    """
    keras.utils.set_random_seed(C.RANDOM_SEED)
    if model is None:
        model = build_autoencoder(X_train.shape[1], X_train.shape[2])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    perm = np.random.default_rng(C.RANDOM_SEED).permutation(len(X_train))
    X_train = X_train[perm]

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOPPING_PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=C.LR_REDUCE_PATIENCE,
                                          factor=C.LR_REDUCE_FACTOR, verbose=1),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
        _EarlyEpochPrinter(),
    ]
    h = model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=C.LSTM_EPOCHS,
                  batch_size=C.BATCH_SIZE, shuffle=True, callbacks=callbacks, verbose=0)

    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    best = int(np.argmin(hist["val_loss"]))
    hist.update(best_epoch=best + 1, best_val_loss=hist["val_loss"][best], epochs_run=len(hist["loss"]))
    return hist


# --------------------------------------------------------------------------- scoring
def compute_reconstruction_error(model: keras.Model, X: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Per-window reconstruction error.

    Input:  trained model and windows ``[N, T, F]``.
    Output: float32 array ``[N]`` = mean |X - X_hat| over all timesteps and features of each window.
    Logic:  predicts in chunks to bound memory (the reconstruction is as large as the input).
    """
    errs = []
    for i in range(0, len(X), chunk):
        part = X[i:i + chunk]
        recon = model.predict(part, batch_size=C.PREDICT_BATCH_SIZE, verbose=0)
        errs.append(np.abs(recon - part).mean(axis=(1, 2)))
    return np.concatenate(errs).astype(np.float32) if errs else np.empty(0, np.float32)


def set_threshold(errors_train: np.ndarray, k: float = C.ANOMALY_THRESHOLD_MULTIPLIER,
                  path: Path = THRESHOLD_PATH) -> float:
    """Anomaly threshold = mean + k * std of the training reconstruction errors.

    Input:  per-window errors on the *training* windows; multiplier ``k`` (config, default 3).
    Output: the threshold as a float; also written (with mean/std/k/n) to ``path`` as JSON.
    """
    mean, std = float(np.mean(errors_train)), float(np.std(errors_train))
    thr = mean + k * std
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"threshold": thr, "train_mean": mean, "train_std": std,
                                "k": k, "n_train_windows": int(len(errors_train))}, indent=2))
    return thr


def load_threshold(path: Path = THRESHOLD_PATH) -> float:
    """Read the threshold saved by :func:`set_threshold`."""
    return float(json.loads(Path(path).read_text())["threshold"])


def predict_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean anomaly flags: True where the window's error exceeds ``threshold``."""
    return np.asarray(errors) > threshold


# --------------------------------------------------------------------------- evaluation
def _rate(flags: np.ndarray) -> float:
    """Share of True in ``flags`` (NaN for an empty array)."""
    return float(np.mean(flags)) if len(flags) else float("nan")


def evaluate(
    model: keras.Model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_failure_labeled: np.ndarray,
    timestamps_failure: np.ndarray,
    failure_windows: FailureWindows = C.FAILURE_WINDOWS,
    X_candidate: np.ndarray | None = None,
) -> dict:
    """Score every period, print the anomaly-detection report and return the numbers.

    Input:  trained model; train/val windows; failure_labeled windows with their end-times;
            failure windows; optionally the undocumented-candidate (03-29) windows.
    Output: dict with the errors per set (``err_*``), ``threshold``, ``results`` (all printed
            rates) and the failure_labeled categories (``clean``, ``prefail``, ``in_failure``).
    Logic:  threshold is set from the training errors only. failure_labeled windows are split into
            ``prefail`` (<= FAILURE_LOOKAHEAD_HOURS before a failure start), ``in_failure`` (inside a
            documented failure window - evaluation only), ``clean`` (label 0, not in a failure and
            outside the 72 h degradation zone, i.e. truly healthy) and the remainder (degradation
            zone, not counted as either).
    """
    err_tr = compute_reconstruction_error(model, X_train)
    err_val = compute_reconstruction_error(model, X_val)
    err_fl = compute_reconstruction_error(model, X_failure_labeled)
    thr = set_threshold(err_tr)
    res: dict = {}

    print("\n" + "=" * 78 + "\nANOMALY DETECTION EVALUATION\n" + "=" * 78)
    q = lambda e: np.percentile(e, [50, 95, 99])
    print(f"Train error : mean={err_tr.mean():.5f} std={err_tr.std():.5f}  (n={len(err_tr):,})")
    print(f"Threshold   : mean + {C.ANOMALY_THRESHOLD_MULTIPLIER:g}*std = {thr:.5f}   (saved to {THRESHOLD_PATH.name})")
    p = q(err_val)
    print(f"Val error   : mean={err_val.mean():.5f} std={err_val.std():.5f}  p50/p95/p99="
          f"{p[0]:.5f}/{p[1]:.5f}/{p[2]:.5f}  (n={len(err_val):,})")
    res["train_flag_rate"] = _rate(predict_anomalies(err_tr, thr))
    res["val_flag_rate"] = _rate(predict_anomalies(err_val, thr))
    print(f"Flag rate   : train={res['train_flag_rate']:.2%} (in-sample)   val={res['val_flag_rate']:.2%} (held-out normal)")

    t = pd.DatetimeIndex(timestamps_failure)
    flags = predict_anomalies(err_fl, thr)
    y = label_windows(timestamps_failure, failure_windows, C.FAILURE_LOOKAHEAD_HOURS).astype(bool)
    infail = in_failure_mask(timestamps_failure, failure_windows)
    in_zone = ~np.isnan(compute_rul(timestamps_failure, failure_windows, C.MAX_RUL))
    clean = ~y & ~infail & ~in_zone

    print(f"\nWindows flagged in the {C.FAILURE_LOOKAHEAD_HOURS} h before each failure onset "
          f"(and inside the failure window, evaluation-only):")
    print(f"  {'failure':<22}{'pre-failure':>22}{'in-failure':>22}{'mean err':>12}")
    res["per_failure"] = []
    for i, (s, e) in enumerate(failure_windows, 1):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        pre = (t >= s - pd.Timedelta(hours=C.FAILURE_LOOKAHEAD_HOURS)) & (t < s)
        dur = (t >= s) & (t <= e)
        r_pre, r_dur = _rate(flags[pre]), _rate(flags[dur])
        res["per_failure"].append({"failure": i, "pre_rate": r_pre, "pre_n": int(pre.sum()),
                                   "in_rate": r_dur, "in_n": int(dur.sum())})
        print(f"  F{i} {s:%m-%d %H:%M}        {r_pre:>10.1%} ({int(pre.sum()):>4} win)"
              f"{r_dur:>12.1%} ({int(dur.sum()):>4} win){err_fl[pre].mean():>12.5f}")
    res["prefail_flag_rate"] = _rate(flags[y])
    res["in_failure_flag_rate"] = _rate(flags[infail])
    print(f"  all pre-failure: {res['prefail_flag_rate']:.1%}   all in-failure: {res['in_failure_flag_rate']:.1%}")

    res["fpr_clean"] = _rate(flags[clean])
    print(f"\nFalse-positive rate on clean data (label 0, not in failure, outside 72 h zone; "
          f"n={int(clean.sum()):,}): {res['fpr_clean']:.2%}")
    months = t.to_period("M")
    for m in sorted(set(months[clean])):
        sel = clean & (months == m)
        print(f"    {m}: {_rate(flags[sel]):>6.2%}  (n={int(sel.sum()):,}, mean err {err_fl[sel].mean():.5f})")
    if y.any() and clean.any():
        res["auroc_prefail_vs_clean"] = float(roc_auc_score(
            np.r_[np.zeros(int(clean.sum())), np.ones(int(y.sum()))], np.r_[err_fl[clean], err_fl[y]]))
        print(f"  Threshold-free separation, AUROC(pre-failure vs clean) = {res['auroc_prefail_vs_clean']:.3f}")

    out = {"threshold": thr, "err_train": err_tr, "err_val": err_val, "err_failure": err_fl,
           "clean": clean, "prefail": y, "in_failure": infail, "results": res}
    if X_candidate is not None and len(X_candidate):
        err_c = compute_reconstruction_error(model, X_candidate)
        res["candidate_flag_rate"] = _rate(predict_anomalies(err_c, thr))
        res["candidate_detected"] = res["candidate_flag_rate"] >= C.DETECTION_MIN_FLAG_RATE
        out["err_candidate"] = err_c
        print(f"\nCandidate day {', '.join(C.EXCLUDE_FROM_TRAINING)} (never trained on): "
              f"{res['candidate_flag_rate']:.1%} of {len(err_c):,} windows flagged, mean err {err_c.mean():.5f} "
              f"-> {'DETECTED' if res['candidate_detected'] else 'not detected'} "
              f"(rule: >= {C.DETECTION_MIN_FLAG_RATE:.0%} flagged)")
    return out


# --------------------------------------------------------------------------- plotting
def plot_reconstruction_error(
    errors: np.ndarray,
    timestamps: np.ndarray,
    failure_windows: FailureWindows,
    threshold: float,
    candidate_dates: Sequence[str],
    save_path: Path = PLOT_PATH,
    train_end: str = C.TRAIN_END,
) -> Path:
    """Plot reconstruction error over the whole timeline and save it.

    Input:  per-window errors and their end-times (any order), failure windows, threshold, and the
            candidate dates to shade.
    Output: path of the saved PNG.
    Logic:  error as a line (broken across logging gaps > 2 h), threshold as a dashed line, failure
            windows red, candidate days orange, points above the threshold red, and the end of the
            training period marked with a vertical line. Log y-axis because errors span decades.
    """
    t = pd.DatetimeIndex(timestamps)
    order = np.argsort(t.values)
    t, e = t[order], np.asarray(errors)[order]
    brk = np.where(np.diff(t.values) > np.timedelta64(2, "h"))[0] + 1
    tl = np.insert(t.values.astype("datetime64[ns]"), brk, t.values[brk - 1] + np.timedelta64(1, "s"))
    el = np.insert(e.astype(float), brk, np.nan)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.plot(tl, el, lw=0.4, color="tab:blue", label="reconstruction error (MAE)")
    hit = e > threshold
    ax.scatter(t[hit], e[hit], s=2, color="red", zorder=3, label=f"anomaly ({hit.sum():,} windows)")
    ax.axhline(threshold, ls="--", color="black", lw=1, label=f"threshold = {threshold:.4f}")
    for i, (s, en) in enumerate(failure_windows):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(en), color="red", alpha=0.25, lw=0,
                   label="failure window" if i == 0 else None)
    for i, d in enumerate(candidate_dates):
        d = pd.Timestamp(d)
        ax.axvspan(d, d + pd.Timedelta(days=1), color="orange", alpha=0.4, lw=0,
                   label="undocumented candidate" if i == 0 else None)
    ax.axvline(pd.Timestamp(train_end) + pd.Timedelta(days=1), color="gray", ls=":", lw=1.2, label="end of training data")
    ax.set_yscale("log")
    ax.set_ylabel("per-window MAE (log)")
    ax.set_title("LSTM autoencoder reconstruction error over time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    from src.data.loader import get_split, load_raw, resample_to_10s
    from src.data.preprocessor import fit_scalers, make_windows, transform

    grid = resample_to_10s(load_raw())
    train_df = get_split(grid, "train")
    scalers = fit_scalers(train_df)
    X_all, t_all = make_windows(transform(train_df, scalers))
    X_train, X_val = split_train_val(X_all)
    print(f"train windows {X_train.shape}  val windows {X_val.shape}  (chronological {1 - C.VAL_FRACTION:.0%}/{C.VAL_FRACTION:.0%})")

    X_fl, t_fl = make_windows(transform(get_split(grid, "failure_labeled"), scalers))
    X_c, t_c = make_windows(transform(get_split(grid, "candidate"), scalers))
    print(f"failure_labeled windows {X_fl.shape}  candidate windows {X_c.shape}")

    if MODEL_PATH.exists():
        print(f"\nLoading existing model {MODEL_PATH} (delete it to retrain)")
        model = keras.models.load_model(MODEL_PATH)
        model.summary()
    else:
        model = build_autoencoder()
        print("\nTraining ...")
        hist = train(X_train, X_val, model)
        print(f"\nepoch 1     : loss={hist['loss'][0]:.5f}  val_loss={hist['val_loss'][0]:.5f}")
        b = hist["best_epoch"] - 1
        print(f"best epoch {hist['best_epoch']:>2}: loss={hist['loss'][b]:.5f}  val_loss={hist['val_loss'][b]:.5f}   "
              f"({hist['epochs_run']} epochs run)")
        (C.REPORT_DIR).mkdir(parents=True, exist_ok=True)
        (C.REPORT_DIR / "autoencoder_history.json").write_text(json.dumps(hist, indent=2))

    ev = evaluate(model, X_train, X_val, X_fl, t_fl, C.FAILURE_WINDOWS, X_candidate=X_c)

    errors = np.concatenate([ev["err_train"], ev["err_val"], ev["err_failure"], ev["err_candidate"]])
    times = np.concatenate([t_all[: len(X_train)], t_all[len(X_train):], t_fl, t_c])
    path = plot_reconstruction_error(errors, times, C.FAILURE_WINDOWS, ev["threshold"], C.EXCLUDE_FROM_TRAINING)
    print(f"\nPlot saved -> {path}")

    r = ev["results"]
    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    print(f"threshold            : {ev['threshold']:.5f}")
    print(f"flag rate train/val  : {r['train_flag_rate']:.2%} / {r['val_flag_rate']:.2%}")
    print(f"flag rate clean data : {r['fpr_clean']:.2%}")
    print(f"flag rate pre-failure: {r['prefail_flag_rate']:.1%}   in-failure: {r['in_failure_flag_rate']:.1%}")
    print(f"03-29 candidate      : {r['candidate_flag_rate']:.1%} flagged -> detected: "
          f"{'YES' if r['candidate_detected'] else 'NO'}")
