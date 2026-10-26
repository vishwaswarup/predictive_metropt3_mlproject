"""Remaining-Useful-Life regression (hours until the next failure).

Target: piecewise-linear RUL from ``preprocessor.compute_rul`` - ``MAX_RUL`` hours at the start of the
degradation zone, falling linearly to 0 at the failure start, NaN (no target) outside the zone.
Model: sklearn GradientBoostingRegressor on the same window features as the classifier.

Training uses only the pre-failure zone (0 < RUL <= MAX_RUL). Windows inside a failure window
(RUL == 0) are evaluation-only, as in the classifier. Evaluation is leave-one-failure-out and is always
compared with a constant predictor, because RUL spans a fixed 0..MAX_RUL range and a model that has
learnt nothing still gets a "reasonable-looking" error.

Run from the project root:  python -m src.models.rul_estimator
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

import config as C
from src.data.preprocessor import compute_rul, in_failure_mask
from src.models.classifier import make_folds

RUL_MODEL_PATH = C.MODEL_DIR / "rul_estimator.pkl"
FailureWindows = Sequence[tuple[str, str]]


def build_rul_dataset(F: pd.DataFrame, failure_windows: FailureWindows = C.FAILURE_WINDOWS,
                      max_rul: float = C.MAX_RUL) -> dict[str, np.ndarray]:
    """Compute RUL targets, the training zone mask and folds for a feature table.

    Input:  feature table indexed by window end-time; failure windows; ``max_rul`` in hours.
    Output: dict with ``rul`` (float, NaN outside the zone), ``zone`` (bool, 0 < RUL <= max_rul: the
            training/evaluation windows), ``in_failure`` (bool), ``folds`` (leave-one-failure-out ids).
    """
    t = F.index.to_numpy()
    rul = compute_rul(t, failure_windows, max_rul)
    return {"rul": rul, "zone": ~np.isnan(rul) & (rul > 0), "in_failure": in_failure_mask(t, failure_windows),
            "folds": make_folds(t, failure_windows)}


def train_rul_model(F: pd.DataFrame, rul: np.ndarray) -> GradientBoostingRegressor:
    """Fit the gradient-boosting RUL regressor.

    Input:  features and RUL targets (hours) of the training windows.
    Output: fitted ``GradientBoostingRegressor`` (hyperparameters from ``config.RUL_MODEL_PARAMS``).
    """
    return GradientBoostingRegressor(**C.RUL_MODEL_PARAMS, random_state=C.RANDOM_SEED).fit(F, rul)


def predict_rul(model: GradientBoostingRegressor, F: pd.DataFrame, max_rul: float = C.MAX_RUL) -> np.ndarray:
    """Predict RUL in hours, clipped to the valid range ``[0, max_rul]``."""
    return np.clip(model.predict(F), 0.0, max_rul).astype(np.float32)


def evaluate_rul(y: np.ndarray, pred: np.ndarray) -> dict:
    """RUL error metrics.

    Input:  true and predicted RUL in hours.
    Output: dict with MAE, RMSE (hours), share of predictions within +-6 h (``within_6h``) and the
            Spearman rank correlation (does the prediction at least fall as failure approaches?).
    """
    err = pred - y
    rho = spearmanr(y, pred).statistic if len(y) > 2 and np.ptp(pred) > 0 else float("nan")
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "within_6h": float((np.abs(err) <= 6).mean()), "spearman": float(rho), "n": int(len(y))}


def cross_validate_rul(F: pd.DataFrame, data: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    """Leave-one-failure-out evaluation of the RUL model against a constant baseline.

    Input:  feature table and the dict from :func:`build_rul_dataset`.
    Output: ``(oof_pred, table)``: out-of-fold predictions for every window (zone, in-failure and healthy
            windows are all predicted, only zone windows are trained on) and a table with per-fold and
            pooled metrics for the model and for ``baseline`` = the training-fold mean RUL.
    """
    zone, rul, folds = data["zone"], data["rul"], data["folds"]
    oof = np.full(len(F), np.nan, dtype=np.float32)
    base = np.full(len(F), np.nan, dtype=np.float32)
    rows = []
    for k in np.unique(folds):
        tr, te = zone & (folds != k), folds == k
        model = train_rul_model(F[tr], rul[tr])
        oof[te] = predict_rul(model, F[te])
        base[te] = rul[tr].mean()
        ev = te & zone
        rows.append({"fold": f"F{k + 1}", **evaluate_rul(rul[ev], oof[ev]),
                     "baseline_mae": evaluate_rul(rul[ev], base[ev])["mae"]})
    ev = zone
    rows.append({"fold": "pooled", **evaluate_rul(rul[ev], oof[ev]), "baseline_mae": evaluate_rul(rul[ev], base[ev])["mae"]})
    return oof, pd.DataFrame(rows).set_index("fold")


def save_rul_model(model: GradientBoostingRegressor, columns: Sequence[str], path: Path = RUL_MODEL_PATH) -> None:
    """Pickle the model with its feature column order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": list(columns), "max_rul": C.MAX_RUL}, path)


def load_rul_model(path: Path = RUL_MODEL_PATH) -> tuple[GradientBoostingRegressor, list[str]]:
    """Reload ``(model, feature_columns)`` saved by :func:`save_rul_model`."""
    d = joblib.load(path)
    return d["model"], d["columns"]


if __name__ == "__main__":
    from src.models.classifier import load_or_build_features

    pd.set_option("display.width", 200)
    F = load_or_build_features()
    d = build_rul_dataset(F)
    zone, rul, infail, folds = d["zone"], d["rul"], d["in_failure"], d["folds"]
    print(f"feature table {F.shape} | zone windows (0 < RUL <= {C.MAX_RUL} h): {int(zone.sum()):,} | "
          f"in-failure (RUL=0, eval-only): {int(infail.sum()):,} | healthy (no target): {int(np.isnan(rul).sum()):,}")
    print("zone windows per fold:", {f"F{int(k) + 1}": int((zone & (folds == k)).sum()) for k in np.unique(folds)})

    print("\n" + "=" * 78 + f"\nLEAVE-ONE-FAILURE-OUT RUL EVALUATION (hours, MAX_RUL={C.MAX_RUL})\n" + "=" * 78)
    oof, tab = cross_validate_rul(F, d)
    print(tab.round(3).to_string())
    skill = 1 - tab.loc["pooled", "mae"] / tab.loc["pooled", "baseline_mae"]
    print(f"\npooled MAE {tab.loc['pooled', 'mae']:.2f} h vs constant-predictor {tab.loc['pooled', 'baseline_mae']:.2f} h "
          f"-> {skill:+.1%} relative to baseline (<= 0 means no skill)")

    healthy = np.isnan(rul) & ~infail
    print(f"\nIn-failure windows (true RUL 0, never trained on): mean predicted RUL {oof[infail].mean():.1f} h "
          f"(MAE {np.abs(oof[infail]).mean():.1f} h)")
    print(f"Healthy windows (no RUL target, n={int(healthy.sum()):,}): mean predicted RUL {oof[healthy].mean():.1f} h, "
          f"{(oof[healthy] <= 12).mean():.1%} predicted <= 12 h (false 'urgent')")

    final = train_rul_model(F[zone], rul[zone])
    save_rul_model(final, F.columns)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORT_DIR / "rul_metrics.json").write_text(json.dumps({"max_rul": C.MAX_RUL, "cv": tab.round(4).to_dict("index")}, indent=2))
    print(f"\nsaved {RUL_MODEL_PATH.name} (trained on all {int(zone.sum()):,} zone windows) and reports/rul_metrics.json")
