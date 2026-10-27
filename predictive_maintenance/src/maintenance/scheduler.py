"""Rule-based maintenance scheduler with a likelihood x consequence risk matrix.

Inputs per window: anomaly score (reconstruction error / threshold), failure probability, predicted RUL.
Output: NORMAL / WATCH / SCHEDULE_MAINTENANCE / URGENT plus the evidence behind the decision.

Design follows what the evaluation showed: the autoencoder anomaly flag is the only validated signal, so
  * the anomaly must be *active* (>= SCHED_MIN_CONSECUTIVE consecutive anomalous windows) for anything above NORMAL;
  * likelihood grows with how long the anomaly persists (score magnitude does not rank severity);
  * classifier probability and RUL can only add one likelihood step to an active anomaly - they never trigger alone,
    and that is switched off by default (SCHED_USE_CORROBORATION): it increased false alarms without earlier warning.

Run from the project root:  python -m src.maintenance.scheduler
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C

LEVELS = C.SCHED_LEVELS
CAVEAT = ("failure probability and RUL showed no skill in leave-one-failure-out validation; "
          "they can only corroborate an active anomaly, and that is off by default (SCHED_USE_CORROBORATION)")


@dataclass(frozen=True)
class Decision:
    """One maintenance decision with its supporting evidence."""
    level: str
    likelihood: int
    consequence: int
    risk: int
    persist_min: float
    corroborated: bool
    reason: str


def persistence_minutes(score: np.ndarray, timestamps: np.ndarray, threshold: float = C.SCHED_ANOMALY_WATCH,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Length of the current anomalous run at every window.

    Input:  anomaly scores and window end-times (ascending).
    Output: ``(run_windows, run_minutes)``: consecutive anomalous windows ending here (0 if the window is
            normal) and the minutes they span. A logging gap (> 1.5 strides) breaks a run.
    """
    t = pd.DatetimeIndex(timestamps).to_numpy()
    hit = np.asarray(score) >= threshold
    max_gap = np.timedelta64(int(1.5 * C.STEP_SIZE * C.SAMPLING_INTERVAL_S), "s")
    pos = np.arange(len(hit))
    prev_ok = np.r_[False, hit[:-1] & (np.diff(t) <= max_gap)]
    first = hit & ~prev_ok
    run = np.where(hit, pos - np.maximum.accumulate(np.where(first, pos, 0)) + 1, 0)
    stride_min = C.STEP_SIZE * C.SAMPLING_INTERVAL_S / 60
    return run, run * stride_min


def likelihood_from_persistence(run_windows: np.ndarray, run_minutes: np.ndarray) -> np.ndarray:
    """Likelihood 0-3: none / possible (active) / likely (>= 60 min) / almost certain (>= 240 min)."""
    lik = np.zeros(len(run_windows), dtype=int)
    lik[run_windows >= C.SCHED_MIN_CONSECUTIVE] = 1
    lik[(lik >= 1) & (run_minutes >= C.SCHED_PERSIST_LIKELY_MIN)] = 2
    lik[(lik >= 1) & (run_minutes >= C.SCHED_PERSIST_CERTAIN_MIN)] = 3
    return lik


def risk_to_level(risk: np.ndarray) -> np.ndarray:
    """Map risk scores to level indices (0 NORMAL .. 3 URGENT) using the config thresholds."""
    r = np.asarray(risk)
    return (r >= C.SCHED_RISK_WATCH).astype(int) + (r >= C.SCHED_RISK_SCHEDULE) + (r >= C.SCHED_RISK_URGENT)


def risk_matrix(consequences: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Likelihood x consequence matrix of resulting levels (rows likelihood, columns consequence)."""
    rows = {C.SCHED_LIKELIHOOD_LABELS[l]: [LEVELS[int(risk_to_level(np.array([l * c]))[0])] for c in consequences]
            for l in range(len(C.SCHED_LIKELIHOOD_LABELS))}
    return pd.DataFrame(rows, index=[f"{c} ({C.SCHED_CONSEQUENCE_LABELS[c]})" for c in consequences]).T


def schedule(score: np.ndarray, timestamps: np.ndarray, proba: np.ndarray | None = None,
             rul: np.ndarray | None = None, consequence: int = C.SCHED_CONSEQUENCE) -> pd.DataFrame:
    """Vectorised decisions for a time series of windows.

    Input:  anomaly score, window end-times, optional failure probability and predicted RUL (hours),
            consequence class (1-3).
    Output: DataFrame indexed by time with ``level`` (str), ``level_idx``, ``likelihood``, ``risk``,
            ``persist_min`` and ``corroborated``.
    """
    run, minutes = persistence_minutes(score, timestamps)
    lik = likelihood_from_persistence(run, minutes)
    corr = np.zeros(len(lik), dtype=bool)
    if proba is not None and C.SCHED_USE_CORROBORATION:
        corr |= np.asarray(proba) >= C.SCHED_PROB_SCHEDULE
    if rul is not None and C.SCHED_USE_CORROBORATION:
        corr |= np.asarray(rul) <= C.SCHED_RUL_SCHEDULE_HOURS
    corr &= lik >= 1                      # corroboration never acts without an active anomaly
    lik = np.minimum(lik + corr, len(C.SCHED_LIKELIHOOD_LABELS) - 1)
    risk = lik * consequence
    idx = risk_to_level(risk)
    return pd.DataFrame({"level": np.array(LEVELS)[idx], "level_idx": idx, "likelihood": lik, "risk": risk,
                         "persist_min": minutes, "corroborated": corr}, index=pd.DatetimeIndex(timestamps))


def decide(score: float, persist_min: float, proba: float | None = None, rul: float | None = None,
           consequence: int = C.SCHED_CONSEQUENCE) -> Decision:
    """Single-point decision from already-computed evidence (for APIs and unit tests)."""
    stride_min = C.STEP_SIZE * C.SAMPLING_INTERVAL_S / 60
    run = int(round(persist_min / stride_min)) if score >= C.SCHED_ANOMALY_WATCH else 0
    lik = int(likelihood_from_persistence(np.array([run]), np.array([persist_min]))[0])
    corr = C.SCHED_USE_CORROBORATION and lik >= 1 and ((proba is not None and proba >= C.SCHED_PROB_SCHEDULE)
                         or (rul is not None and rul <= C.SCHED_RUL_SCHEDULE_HOURS))
    lik = min(lik + int(corr), len(C.SCHED_LIKELIHOOD_LABELS) - 1)
    risk = lik * consequence
    level = LEVELS[int(risk_to_level(np.array([risk]))[0])]
    why = (f"anomaly {'active' if run >= C.SCHED_MIN_CONSECUTIVE else 'not active'} for {persist_min:.0f} min "
           f"(score {score:.2f}x threshold); likelihood {C.SCHED_LIKELIHOOD_LABELS[lik]}"
           f"{' (+1 corroborated by probability/RUL)' if corr else ''}; consequence "
           f"{C.SCHED_CONSEQUENCE_LABELS[consequence]}; risk {risk}. Caveat: {CAVEAT}.")
    return Decision(level, lik, consequence, risk, persist_min, bool(corr), why)


def summarise(dec: pd.DataFrame, timestamps: np.ndarray, failure_windows=C.FAILURE_WINDOWS,
              lookahead_hours: float = C.FAILURE_LOOKAHEAD_HOURS) -> pd.DataFrame:
    """Per-failure view of the decisions.

    Output: one row per failure with the level mix in the pre-failure lookahead and inside the failure,
            and the hours before failure start at which SCHEDULE_MAINTENANCE / URGENT were first reached
            (None if not reached before the failure started, searching the full lead-up of ``lookahead_hours``).
    """
    t = pd.DatetimeIndex(timestamps)
    rows = []
    for k, (s, e) in enumerate(failure_windows, 1):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        pre = (t >= s - pd.Timedelta(hours=lookahead_hours)) & (t < s)
        inn = (t >= s) & (t <= e)
        row = {"failure": f"F{k}"}
        for name, m in (("pre24h", pre), ("in_failure", inn)):
            vc = dec.level[m].value_counts(normalize=True) if m.any() else pd.Series(dtype=float)
            row[f"{name}_ge_SCHEDULE"] = float(vc.reindex(LEVELS).fillna(0)[2:].sum()) if m.any() else np.nan
        for lvl_i, name in ((2, "SCHEDULE"), (3, "URGENT")):
            hit = (dec.level_idx.to_numpy() >= lvl_i) & (t >= s - pd.Timedelta(hours=lookahead_hours)) & (t <= e)
            first = t[hit].min() if hit.any() else None
            row[f"first_{name}_h_before_start"] = float((s - first).total_seconds() / 3600) if first is not None else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("failure")


def _episodes(dec: pd.DataFrame, in_failure: np.ndarray, min_idx: int) -> int:
    """Number of distinct episodes at >= ``min_idx`` that never touch a failure window (false alarms)."""
    on = dec.level_idx.to_numpy() >= min_idx
    starts = on & ~np.r_[False, on[:-1]]
    ep = np.cumsum(starts)
    bad = set(ep[on & in_failure])
    return len(set(ep[on]) - bad - {0})


if __name__ == "__main__":
    from src.data.preprocessor import in_failure_mask, label_windows
    from src.models.autoencoder import load_threshold
    from src.models.classifier import RECON_COL, cross_validate, load_or_build_features
    from src.models.rul_estimator import build_rul_dataset, cross_validate_rul

    pd.set_option("display.width", 200)
    print("RISK MATRIX (rows: likelihood, columns: consequence)\n" + risk_matrix().to_string())

    F = load_or_build_features()
    t = F.index.to_numpy()
    score = F[RECON_COL].to_numpy() / load_threshold()
    infail = in_failure_mask(t, C.FAILURE_WINDOWS)
    y = label_windows(t, C.FAILURE_WINDOWS, C.FAILURE_LOOKAHEAD_HOURS)
    d = build_rul_dataset(F)
    proba, _ = cross_validate(F, y, d["folds"], infail)
    rul, _ = cross_validate_rul(F, d)

    full = schedule(score, t, proba, rul)
    solo = schedule(score, t)
    for name, dec in (("anomaly only", solo), ("anomaly + corroboration (out-of-fold)", full)):
        print("\n" + "=" * 78 + f"\nDECISIONS - {name}\n" + "=" * 78)
        mix = pd.DataFrame({"healthy": dec.level[~infail].value_counts(normalize=True),
                            "in_failure": dec.level[infail].value_counts(normalize=True)}).reindex(LEVELS).fillna(0)
        print((mix * 100).round(1).to_string())
        print(summarise(dec, t).round(2).to_string())
        print(f"false-alarm episodes (never touching a failure): SCHEDULE+ {_episodes(dec, infail, 2)}, "
              f"URGENT {_episodes(dec, infail, 3)}")

    print("\nExample single decisions:")
    for args in ((0.3, 0), (1.4, 10), (1.4, 90), (1.4, 300, 0.7, 20)):
        print(f"  {args} -> {decide(*args).level}")
    print("\nCaveat:", CAVEAT)
