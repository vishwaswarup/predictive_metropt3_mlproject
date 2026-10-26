"""XGBoost failure-imminent classifier with SHAP interpretability.

Features per 10-minute window: mean/std/min/max/skew of every sensor plus the autoencoder's
reconstruction error. Label: 1 if the window ends within ``FAILURE_LOOKAHEAD_HOURS`` before a
failure start. Windows *inside* a failure are never used for training (evaluation only).

Because there are only four failures, evaluation is leave-one-failure-out: each fold holds out one
failure and its surrounding period, and trains on the other three.

Run from the project root:  python -m src.models.classifier
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import skew
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

import config as C
from src.data.preprocessor import compute_rul, in_failure_mask, label_windows

CLASSIFIER_PATH = C.MODEL_DIR / "classifier.pkl"
SHAP_PLOT_PATH = C.PLOT_DIR / "shap_importance.png"
RECON_COL = "recon_error"
FailureWindows = Sequence[tuple[str, str]]


# --------------------------------------------------------------------------- features
def extract_window_features(X: np.ndarray, feature_names: Sequence[str] = C.FEATURE_COLS) -> pd.DataFrame:
    """Statistical aggregates of each window.

    Input:  windows ``[N, T, F]`` and the F feature names.
    Output: DataFrame ``[N, F * len(FEATURE_STATS)]`` with columns like ``TP3_mean`` / ``Motor_current_skew``.
    Logic:  statistics are taken over the time axis. Skew of a constant window is undefined (NaN);
            it is set to 0 (a constant signal is perfectly symmetric). Binary sensors that never
            change inside a window therefore get std 0 and skew 0.
    """
    stats = {
        "mean": lambda a: a.mean(axis=1),
        "std": lambda a: a.std(axis=1),
        "min": lambda a: a.min(axis=1),
        "max": lambda a: a.max(axis=1),
        "skew": lambda a: np.nan_to_num(skew(a, axis=1), nan=0.0),
    }
    cols, data = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in C.FEATURE_STATS:
            vals = stats[s](X)
            for j, name in enumerate(feature_names):
                cols.append(f"{name}_{s}")
                data.append(vals[:, j])
    return pd.DataFrame(np.column_stack(data).astype(np.float32), columns=cols)


def build_feature_table(X: np.ndarray, timestamps: np.ndarray, recon_errors: np.ndarray) -> pd.DataFrame:
    """Feature matrix for the classifier: window statistics + autoencoder reconstruction error.

    Input:  windows, their end-times, and the per-window reconstruction error from the autoencoder.
    Output: DataFrame indexed by window end-time with the statistics columns and ``recon_error``.
    """
    F = extract_window_features(X)
    F[RECON_COL] = np.asarray(recon_errors, dtype=np.float32)
    F.index = pd.DatetimeIndex(timestamps, name="window_end")
    return F


def add_drift_features(F: pd.DataFrame) -> pd.DataFrame:
    """Add trend and baseline-normalised features that are insensitive to slow level drift.

    Input:  feature table indexed by window end-time (needs ``<sensor>_mean`` and ``recon_error``).
    Output: copy of ``F`` with, for each analogue sensor mean and the reconstruction error:
              ``*_dev{BASELINE_HOURS}h``  = value - median of the preceding BASELINE_HOURS hours;
              ``*_trend{h}h``             = value - mean of the preceding h hours, for h in TREND_HOURS.
    Logic:  rolling windows use ``closed="left"``, i.e. strictly past data, so nothing from the
            current window or the future leaks in and the features are computable in real time.
            Needs at least ``BASELINE_MIN_WINDOWS`` past windows, else NaN (XGBoost handles NaN).
    """
    cols = [f"{c}_mean" for c in C.ANALOGUE_COLS] + [RECON_COL]
    new: dict[str, pd.Series] = {}
    for col in cols:
        s = F[col]
        base = s.rolling(f"{C.BASELINE_HOURS}h", closed="left", min_periods=C.BASELINE_MIN_WINDOWS).median()
        new[f"{col}_dev{C.BASELINE_HOURS}h"] = s - base
        for h in C.TREND_HOURS:
            new[f"{col}_trend{h}h"] = s - s.rolling(f"{h}h", closed="left", min_periods=5).mean()
    return pd.concat([F, pd.DataFrame(new, index=F.index).astype("float32")], axis=1)


def drift_free_columns(F: pd.DataFrame) -> list[str]:
    """Columns that do not depend on the absolute sensor level: the drift features plus std and skew."""
    return [c for c in F.columns if "_dev" in c or "_trend" in c or c.endswith(("_std", "_skew"))]


# --------------------------------------------------------------------------- folds & model
def make_folds(timestamps: np.ndarray, failure_windows: FailureWindows = C.FAILURE_WINDOWS) -> np.ndarray:
    """Assign each window to a leave-one-failure-out fold (one failure per fold).

    Input:  window end-times; failure windows.
    Output: int array of fold ids ``0..n_failures-1``.
    Logic:  fold boundaries sit midway between the end of one failure and the start of the next, so
            every fold contains exactly one failure with its lead-up and recovery. Contiguous blocks
            keep near-duplicate overlapping windows from straddling train and test.
    """
    t = pd.DatetimeIndex(timestamps)
    wins = sorted((pd.Timestamp(s), pd.Timestamp(e)) for s, e in failure_windows)
    cuts = [wins[i][1] + (wins[i + 1][0] - wins[i][1]) / 2 for i in range(len(wins) - 1)]
    return np.searchsorted(np.array(cuts, dtype="datetime64[ns]"), t.to_numpy(), side="right")


def train_classifier(F: pd.DataFrame, y: np.ndarray) -> XGBClassifier:
    """Fit the XGBoost classifier.

    Input:  feature table and binary labels.
    Output: fitted ``XGBClassifier`` (hyperparameters from ``config.XGB_PARAMS``).
    Logic:  positives are rare (~3%), so ``scale_pos_weight = n_negative / n_positive`` re-balances the loss.
    """
    pos = max(int(y.sum()), 1)
    model = XGBClassifier(**C.XGB_PARAMS, scale_pos_weight=(len(y) - pos) / pos, tree_method="hist",
                          eval_metric="aucpr", random_state=C.RANDOM_SEED, n_jobs=-1)
    return model.fit(F, y)


def evaluate_predictions(y: np.ndarray, proba: np.ndarray, threshold: float = C.CLASSIFIER_THRESHOLD) -> dict:
    """Classification metrics at ``threshold`` plus threshold-free AUCs.

    Input:  true labels, predicted probabilities, decision threshold.
    Output: dict with precision, recall, f1, roc_auc, pr_auc, n, n_pos (AUCs NaN if one class is absent).
    """
    pred = proba >= threshold
    both = 0 < y.sum() < len(y)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if both else float("nan"),
        "n": int(len(y)), "n_pos": int(y.sum()),
    }


def cross_validate(F: pd.DataFrame, y: np.ndarray, folds: np.ndarray, in_failure: np.ndarray,
                   drop_cols: Sequence[str] = ()) -> tuple[np.ndarray, pd.DataFrame]:
    """Leave-one-failure-out cross-validation.

    Input:  features, labels, fold ids, in-failure mask, optional columns to drop (for ablations).
    Output: ``(oof_proba, table)``: out-of-fold probability for every window (including in-failure ones,
            which are predicted but never trained on) and a per-fold metrics table plus a pooled row.
    Logic:  for each fold, train on the other folds' non-failure windows, predict the held-out fold.
            Metrics exclude in-failure windows (reported separately by the caller).
    """
    Fx = F.drop(columns=list(drop_cols))
    oof = np.zeros(len(y), dtype=np.float32)
    rows = []
    for k in np.unique(folds):
        tr = (folds != k) & ~in_failure
        te = folds == k
        model = train_classifier(Fx[tr], y[tr])
        oof[te] = model.predict_proba(Fx[te])[:, 1]
        ev = te & ~in_failure
        rows.append({"fold": f"F{k + 1}", **evaluate_predictions(y[ev], oof[ev])})
    ev = ~in_failure
    rows.append({"fold": "pooled", **evaluate_predictions(y[ev], oof[ev])})
    return oof, pd.DataFrame(rows).set_index("fold")


def lead_times(timestamps: np.ndarray, proba: np.ndarray, failure_windows: FailureWindows = C.FAILURE_WINDOWS,
               threshold: float = C.CLASSIFIER_THRESHOLD, lookahead_hours: float = C.FAILURE_LOOKAHEAD_HOURS,
               min_consecutive: int = C.ALARM_MIN_CONSECUTIVE) -> list[float | None]:
    """Hours of warning before each failure start, from the first *sustained* alarm.

    Input:  window end-times, out-of-fold probabilities, failure windows, threshold, lookahead,
            and the number of consecutive windows >= threshold that count as an alarm.
    Output: one value per failure: hours between the start of the first sustained run and the failure
            start; None if there is no such run inside the lookahead. An isolated window above the
            threshold is a blip, not an alarm.
    Logic:  windows are consecutive when their end-times differ by at most 1.5 strides (a logging gap
            breaks the run).
    """
    t = pd.DatetimeIndex(timestamps)
    max_gap = pd.Timedelta(seconds=1.5 * C.STEP_SIZE * C.SAMPLING_INTERVAL_S)
    out: list[float | None] = []
    for s, _ in failure_windows:
        s = pd.Timestamp(s)
        sel = (t >= s - pd.Timedelta(hours=lookahead_hours)) & (t < s)
        ts, hit = t[sel], proba[sel] >= threshold
        first, run, run_start, prev = None, 0, None, None
        for ti, h in zip(ts, hit):
            if h and run and prev is not None and ti - prev <= max_gap:
                run += 1
            elif h:
                run, run_start = 1, ti
            else:
                run = 0
            prev = ti
            if run >= min_consecutive:
                first = run_start
                break
        out.append(float((s - first).total_seconds() / 3600) if first is not None else None)
    return out


def load_or_build_features(force: bool = False) -> pd.DataFrame:
    """Feature table for the failure_labeled windows, cached on disk.

    Input:  ``force`` to ignore the cache.
    Output: DataFrame indexed by window end-time (statistics + ``recon_error``).
    Logic:  building it needs the autoencoder (slow). The result is cached in ``CACHE_DIR`` and
            rebuilt automatically if the autoencoder checkpoint is newer than the cache.
    """
    import keras

    from src.data.loader import get_split, load_raw, resample_to_10s
    from src.data.preprocessor import load_scalers, make_windows, transform
    from src.models.autoencoder import MODEL_PATH, compute_reconstruction_error

    cache = C.CACHE_DIR / "failure_labeled_features.parquet"
    if cache.exists() and not force and cache.stat().st_mtime > MODEL_PATH.stat().st_mtime:
        return pd.read_parquet(cache)
    grid = resample_to_10s(load_raw())
    X, t = make_windows(transform(get_split(grid, "failure_labeled"), load_scalers()))
    err = compute_reconstruction_error(keras.models.load_model(MODEL_PATH), X)
    F = build_feature_table(X, t, err)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    F.to_parquet(cache)
    return F


def sweep_lookahead(F: pd.DataFrame, hours: Sequence[float] = C.LOOKAHEAD_SWEEP_HOURS,
                    failure_windows: FailureWindows = C.FAILURE_WINDOWS) -> pd.DataFrame:
    """Leave-one-failure-out evaluation for several lookahead horizons.

    Input:  feature table (index = window end-times) and the horizons in hours.
    Output: DataFrame with one row per horizon: prevalence, pooled ROC-AUC / PR-AUC / F1 / precision /
            recall at the config threshold, ROC-AUC per fold (F1..F4) and the median sustained-alarm lead time.
    Logic:  only the label changes between horizons; features, folds and exclusion of in-failure
            windows are identical, so rows are directly comparable. PR-AUC must be read against
            ``prevalence`` (a random model scores about that).
    """
    t = F.index.to_numpy()
    folds, infail = make_folds(t, failure_windows), in_failure_mask(t, failure_windows)
    rows = []
    for h in hours:
        y = label_windows(t, failure_windows, h).astype(int)
        oof, tab = cross_validate(F, y, folds, infail)
        lt = [x for x in lead_times(t, oof, failure_windows, lookahead_hours=h) if x is not None]
        rows.append({"lookahead_h": h, "prevalence": y[~infail].mean(), **tab.loc["pooled", ["roc_auc", "pr_auc", "f1", "precision", "recall"]].to_dict(),
                     **{f"auc_{k}": tab.loc[k, "roc_auc"] for k in tab.index if k != "pooled"},
                     "alarms_sustained": f"{len(lt)}/{len(failure_windows)}", "median_lead_h": float(np.median(lt)) if lt else float("nan")})
    return pd.DataFrame(rows).set_index("lookahead_h")


# --------------------------------------------------------------------------- SHAP
def compute_shap(model: XGBClassifier, F: pd.DataFrame, y: np.ndarray, seed: int = C.RANDOM_SEED) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP values for a class-balanced sample of windows.

    Input:  fitted model, feature table, labels.
    Output: ``(shap_values [n, n_features], sample_features)``; values are in log-odds of "failure imminent".
    Logic:  up to ``SHAP_SAMPLE_PER_CLASS`` windows per class (random, seeded) so the rare positive
            class is not drowned out. Uses the exact TreeExplainer.
    """
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), min(C.SHAP_SAMPLE_PER_CLASS, int((y == c).sum())), replace=False)
                          for c in (0, 1) if (y == c).any()])
    sample = F.iloc[np.sort(idx)]
    return shap.TreeExplainer(model).shap_values(sample), sample


def shap_importance(shap_values: np.ndarray, sample: pd.DataFrame) -> pd.DataFrame:
    """Rank features by mean |SHAP|.

    Input:  SHAP values and the matching feature rows.
    Output: DataFrame sorted by ``mean_abs_shap`` with ``direction`` = correlation of feature value with
            its SHAP value (+ : higher value pushes toward failure, - : lower value does).
    """
    direction = [np.corrcoef(sample.iloc[:, j], shap_values[:, j])[0, 1] if shap_values[:, j].std() > 0 and sample.iloc[:, j].std() > 0 else 0.0
                 for j in range(sample.shape[1])]
    df = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0), "direction": direction})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_shap_importance(importance: pd.DataFrame, top_k: int = C.SHAP_TOP_K, save_path: Path = SHAP_PLOT_PATH) -> Path:
    """Horizontal bar chart of the ``top_k`` features by mean |SHAP|; returns the saved path."""
    top = importance.head(top_k)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * top_k + 1.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=["tab:red" if f == RECON_COL else "tab:blue" for f in top["feature"]])
    ax.set_xlabel("mean |SHAP value| (log-odds of failure within lookahead)")
    ax.set_title(f"Top {top_k} features (red = autoencoder reconstruction error)")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return Path(save_path)


def save_classifier(model: XGBClassifier, columns: Sequence[str], path: Path = CLASSIFIER_PATH) -> None:
    """Pickle the model with its feature column order (needed to reload and predict)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": list(columns)}, path)


def load_classifier(path: Path = CLASSIFIER_PATH) -> tuple[XGBClassifier, list[str]]:
    """Reload ``(model, feature_columns)`` saved by :func:`save_classifier`."""
    d = joblib.load(path)
    return d["model"], d["columns"]


if __name__ == "__main__":
    import sys

    pd.set_option("display.width", 220)
    F = load_or_build_features()
    t = F.index.to_numpy()
    print(f"failure_labeled feature table: {F.shape}")

    if "--sweep-drift" in sys.argv:
        F2 = add_drift_features(F)
        variants = {"original (absolute)": list(F.columns),
                    "original + drift features": list(F2.columns),
                    "drift-free only": drift_free_columns(F2)}
        print("\n" + "=" * 78 + "\nDRIFT-ROBUST FEATURE EXPERIMENT (leave-one-failure-out, threshold "
              f"{C.CLASSIFIER_THRESHOLD})\n" + "=" * 78)
        parts = []
        for name, cols in variants.items():
            r = sweep_lookahead(F2[cols]).assign(variant=name, n_features=len(cols)).reset_index()
            parts.append(r)
            print(f"\n[{name}]  {len(cols)} features")
            print(r.drop(columns=["variant", "n_features", "precision", "recall"]).round(3).to_string(index=False))
        C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat(parts).to_csv(C.REPORT_DIR / "drift_feature_experiment.csv", index=False)
        print("\nsaved reports/drift_feature_experiment.csv")
        sys.exit(0)

    if "--sweep" in sys.argv:
        print("\n" + "=" * 78 + "\nLOOKAHEAD SWEEP (leave-one-failure-out, threshold "
              f"{C.CLASSIFIER_THRESHOLD}, alarm = {C.ALARM_MIN_CONSECUTIVE} consecutive windows)\n" + "=" * 78)
        sweep = sweep_lookahead(F)
        print(sweep.round(3).to_string())
        C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        sweep.to_csv(C.REPORT_DIR / "lookahead_sweep.csv")
        print(f"\nsaved reports/lookahead_sweep.csv   (AUC 0.5 = chance; PR-AUC baseline = prevalence)")
        sys.exit(0)

    y = label_windows(t).astype(int)
    infail = in_failure_mask(t)
    zone_nan = np.isnan(compute_rul(t))
    folds = make_folds(t)
    print(f"features: {F.shape[1]} ({F.shape[1] - 1} window statistics + {RECON_COL})")
    print(f"lookahead {C.FAILURE_LOOKAHEAD_HOURS} h | positives {int(y.sum()):,} ({y.mean():.2%}) | in-failure (excluded from training) {int(infail.sum()):,}")
    print("windows per fold:", {int(k): int(v) for k, v in zip(*np.unique(folds, return_counts=True))})

    print("\n" + "=" * 78 + "\nLEAVE-ONE-FAILURE-OUT CROSS-VALIDATION (threshold "
          f"{C.CLASSIFIER_THRESHOLD}, lookahead {C.FAILURE_LOOKAHEAD_HOURS} h)\n" + "=" * 78)
    oof, tab = cross_validate(F, y, folds, infail)
    print("with reconstruction error:\n", tab.round(3).to_string())
    _, tab_abl = cross_validate(F, y, folds, infail, drop_cols=[RECON_COL])
    print("\nablation - WITHOUT reconstruction error:\n", tab_abl.round(3).to_string())

    ev = ~infail
    clean = ev & (y == 0) & zone_nan
    print(f"\nFalse-positive rate on clean windows (outside 72 h zone, n={int(clean.sum()):,}): {(oof[clean] >= C.CLASSIFIER_THRESHOLD).mean():.2%}")
    print(f"In-failure windows (never trained on, eval-only; n={int(infail.sum()):,}): "
          f"{(oof[infail] >= C.CLASSIFIER_THRESHOLD).mean():.1%} scored >= threshold, mean p={oof[infail].mean():.3f}")
    for i, h in enumerate(lead_times(t, oof), 1):
        print(f"  F{i} first sustained alarm ({C.ALARM_MIN_CONSECUTIVE}+ windows): " + (f"{h:.1f} h before onset" if h is not None else "none within lookahead"))

    print("\n" + "=" * 78 + "\nFINAL MODEL + SHAP (trained on all non-failure windows)\n" + "=" * 78)
    final = train_classifier(F[~infail], y[~infail])
    save_classifier(final, F.columns)
    sv, sample = compute_shap(final, F[~infail], y[~infail])
    imp = shap_importance(sv, sample)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_csv(C.REPORT_DIR / "shap_importance.csv", index=False)
    (C.REPORT_DIR / "classifier_metrics.json").write_text(json.dumps(
        {"lookahead_hours": C.FAILURE_LOOKAHEAD_HOURS, "cv": tab.round(4).to_dict("index"),
         "cv_without_recon_error": tab_abl.round(4).to_dict("index")}, indent=2))
    print(imp.head(15).round(4).to_string())
    print(f"\n{RECON_COL} rank: {int(imp.index[imp.feature == RECON_COL][0]) + 1} of {len(imp)}")
    print(f"saved: {CLASSIFIER_PATH.name}, {plot_shap_importance(imp).name}, reports/shap_importance.csv, reports/classifier_metrics.json")
