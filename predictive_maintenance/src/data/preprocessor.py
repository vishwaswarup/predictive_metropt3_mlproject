"""Scaling, sliding-window construction, failure labels and RUL targets.

Run the self-test from the project root:  python -m src.data.preprocessor
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config as C

Scalers = dict[str, MinMaxScaler]
FailureWindows = Sequence[tuple[str, str]]

SCALER_PATH = C.MODEL_DIR / "scaler.pkl"


def fit_scalers(df_train: pd.DataFrame, save_path: Path = SCALER_PATH) -> Scalers:
    """Fit a MinMaxScaler on the analogue columns of the training data only.

    Input:  training DataFrame from ``get_split(..., "train")`` (must have ``is_gap``).
    Output: dict ``{"analogue": fitted MinMaxScaler}``; also pickled to ``save_path``.
    Logic:  gap/masked rows are excluded from the fit. Fitting on training data alone means
            no information from the failure period shapes the scaling (no leakage).
    """
    valid = df_train.loc[~df_train["is_gap"], C.ANALOGUE_COLS]
    if valid.empty:
        raise ValueError("No usable rows to fit scalers on")
    scalers: Scalers = {"analogue": MinMaxScaler().fit(valid.to_numpy())}

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, save_path)
    return scalers


def load_scalers(path: Path = SCALER_PATH) -> Scalers:
    """Reload scalers saved by :func:`fit_scalers` (input: pickle path; output: scaler dict)."""
    return joblib.load(path)


def transform(df: pd.DataFrame, scalers: Scalers) -> pd.DataFrame:
    """Apply the fitted scalers to a DataFrame.

    Input:  DataFrame on the 10 s grid; scaler dict from :func:`fit_scalers`.
    Output: copy with analogue columns scaled to the training min/max (float32);
            digital columns and the boolean flag columns are left unchanged.
    Logic:  values are NOT clipped, so readings outside the training range map outside [0, 1].
            That is deliberate: out-of-range behaviour is exactly the anomaly signal.
            NaN (gap) rows stay NaN.
    """
    out = df.copy()
    scaled = scalers["analogue"].transform(df[C.ANALOGUE_COLS].to_numpy())
    out[C.ANALOGUE_COLS] = scaled.astype("float32")
    return out


def make_windows(
    df: pd.DataFrame,
    window_size: int = C.WINDOW_SIZE,
    step_size: int = C.STEP_SIZE,
    feature_cols: Sequence[str] = C.FEATURE_COLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut the regular-grid series into sliding windows, skipping any that touch a gap.

    Input:  DataFrame on the exact 10 s grid with an ``is_gap`` column (gap and masked rows
            are NaN / True); window length and stride in samples.
    Output: ``(X, end_times)`` with ``X`` float32 of shape ``[N, window_size, n_features]``
            (feature order = ``feature_cols``) and ``end_times`` a ``datetime64[ns]`` array
            of shape ``[N]`` holding each window's last timestamp.
    Logic:  windows start every ``step_size`` rows. A prefix sum of the "bad row" flag
            (is_gap or NaN) tells in O(1) whether a window contains any bad row; those
            windows are dropped, so no window ever spans a gap, sensor-fault day or excluded day.
    """
    values = df[list(feature_cols)].to_numpy(dtype=np.float32)
    bad = df["is_gap"].to_numpy(dtype=bool) | np.isnan(values).any(axis=1)
    n = len(df)
    if n < window_size:
        return np.empty((0, window_size, values.shape[1]), np.float32), np.empty(0, "datetime64[ns]")

    prefix = np.concatenate([[0], np.cumsum(bad)])
    starts = np.arange(0, n - window_size + 1, step_size)
    starts = starts[prefix[starts + window_size] - prefix[starts] == 0]

    X = values[starts[:, None] + np.arange(window_size)]
    end_times = df.index.to_numpy()[starts + window_size - 1]
    return X, end_times


def label_windows(
    timestamps: np.ndarray,
    failure_windows: FailureWindows = C.FAILURE_WINDOWS,
    lookahead_hours: float = C.FAILURE_LOOKAHEAD_HOURS,
) -> np.ndarray:
    """Binary "failure imminent" label for each window end-time.

    Input:  window end-times; list of ``(start, end)`` failure windows; lookahead in hours.
    Output: int8 array, 1 if ``failure_start - lookahead <= t < failure_start`` for any failure.
    Logic:  only the *pre-failure* horizon is positive. Windows that end during a failure
            (``start <= t <= end``) get 0 here, as the spec says; callers that do not want
            them as negatives should drop them using :func:`in_failure_mask`.
    """
    t = pd.DatetimeIndex(timestamps)
    labels = np.zeros(len(t), dtype=bool)
    lead = pd.Timedelta(hours=lookahead_hours)
    for start, _ in failure_windows:
        s = pd.Timestamp(start)
        labels |= (t >= s - lead) & (t < s)
    return labels.astype(np.int8)


def in_failure_mask(timestamps: np.ndarray, failure_windows: FailureWindows = C.FAILURE_WINDOWS) -> np.ndarray:
    """Boolean mask of end-times that fall inside a documented failure window (start <= t <= end)."""
    t = pd.DatetimeIndex(timestamps)
    mask = np.zeros(len(t), dtype=bool)
    for start, end in failure_windows:
        mask |= (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))
    return mask


def compute_rul(
    timestamps: np.ndarray,
    failure_windows: FailureWindows = C.FAILURE_WINDOWS,
    max_rul: float = C.MAX_RUL,
) -> np.ndarray:
    """Piecewise-linear Remaining Useful Life (hours) per window end-time.

    Input:  window end-times; failure windows; ``max_rul`` = length of the degradation zone in hours.
    Output: float32 array of RUL in hours. NaN means "outside any degradation zone".
    Logic:  for time t with next failure start F (the first start >= t):
              F - t <= max_rul  ->  RUL = (F - t) in hours, i.e. linear from ``max_rul`` at the
                                    zone start down to 0 at the failure start;
              otherwise         ->  NaN (healthy period, no meaningful RUL target).
            Times inside a failure window (start <= t <= end) get RUL = 0.
    """
    t = pd.DatetimeIndex(timestamps).to_numpy().astype("datetime64[ns]").astype("int64")
    starts = np.sort(np.array([pd.Timestamp(s).value for s, _ in failure_windows], dtype="int64"))
    ends_by_start = {pd.Timestamp(s).value: pd.Timestamp(e).value for s, e in failure_windows}

    rul = np.full(len(t), np.nan, dtype="float64")
    nxt = np.searchsorted(starts, t, side="left")
    has_next = nxt < len(starts)
    hours = np.full(len(t), np.inf)
    hours[has_next] = (starts[nxt[has_next]] - t[has_next]) / 3.6e12   # ns -> hours
    in_zone = hours <= max_rul
    rul[in_zone] = hours[in_zone]

    for s in starts:
        rul[(t >= s) & (t <= ends_by_start[s])] = 0.0
    return rul.astype("float32")


if __name__ == "__main__":
    from src.data.loader import get_split, load_raw, resample_to_10s

    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    grid = resample_to_10s(load_raw())

    # ---- train split: fit scaler on train only, window it
    train = get_split(grid, "train")
    scalers = fit_scalers(train)
    sc = scalers["analogue"]
    print(f"scaler saved -> {SCALER_PATH}")
    print("train min:", dict(zip(C.ANALOGUE_COLS, sc.data_min_.round(3))))
    print("train max:", dict(zip(C.ANALOGUE_COLS, sc.data_max_.round(3))))

    train_s = transform(train, scalers)
    Xtr, ttr = make_windows(train_s)
    print(f"\n[train] X {Xtr.shape} {Xtr.dtype}  ({Xtr.nbytes/1e6:.0f} MB)  "
          f"ends {pd.Timestamp(ttr.min())} -> {pd.Timestamp(ttr.max())}")
    print(f"        NaN in X: {np.isnan(Xtr).sum()}   analogue range after scaling: "
          f"[{Xtr[..., :len(C.ANALOGUE_COLS)].min():.3f}, {Xtr[..., :len(C.ANALOGUE_COLS)].max():.3f}]")
    d = pd.DatetimeIndex(ttr).normalize()
    assert not (d == pd.Timestamp("2020-03-29")).any(), "window ends on excluded day"
    assert ttr.max() < np.datetime64(pd.Timestamp(C.TRAIN_END) + pd.Timedelta(days=1)), "train leakage"
    print("        sample window 0, first 3 steps x first 5 features:\n", Xtr[0, :3, :5])

    # ---- no window may touch an excluded day (check by recomputing the window span)
    span_start = pd.DatetimeIndex(ttr) - pd.Timedelta(seconds=(C.WINDOW_SIZE - 1) * C.SAMPLING_INTERVAL_S)
    assert not (span_start.normalize() == pd.Timestamp("2020-03-29")).any(), "window starts on excluded day"

    # ---- failure_labeled split
    fl = transform(get_split(grid, "failure_labeled"), scalers)
    Xf, tf = make_windows(fl)
    y = label_windows(tf)
    rul = compute_rul(tf)
    infail = in_failure_mask(tf)
    print(f"\n[failure_labeled] X {Xf.shape}  ends {pd.Timestamp(tf.min())} -> {pd.Timestamp(tf.max())}")
    print(f"        windows ending on sensor-fault days: "
          f"{int(pd.DatetimeIndex(tf).normalize().isin(pd.to_datetime(C.SENSOR_FAULT_DATES)).sum())} (must be 0)")
    print(f"        label=1: {int(y.sum()):,} ({y.mean():.2%})   in-failure windows: {int(infail.sum()):,}")
    ti = pd.DatetimeIndex(tf)
    train_cut = pd.Timestamp(C.TRAIN_END) + pd.Timedelta(days=1)
    lead = pd.Timedelta(hours=C.FAILURE_LOOKAHEAD_HOURS)
    for i, (s, _) in enumerate(C.FAILURE_WINDOWS, 1):
        s = pd.Timestamp(s)
        pos = (ti >= s - lead) & (ti < s)
        note = f"   [{int((pos & (ti < train_cut)).sum())} lie in the AE-train period]" if (pos & (ti < train_cut)).any() else ""
        print(f"          failure {i} ({s:%m-%d %H:%M}): positive windows = {int(pos.sum()):>4}{note}")
    ok = ~np.isnan(rul)
    print(f"        RUL defined for {int(ok.sum()):,} windows ({ok.mean():.1%}); "
          f"min={np.nanmin(rul):.2f} max={np.nanmax(rul):.2f} h; RUL==0 windows: {int((rul == 0).sum())}")
    print("        sample (end-time, label, RUL h) around failure 1:")
    s1 = pd.Timestamp(C.FAILURE_WINDOWS[0][0])
    pick = np.searchsorted(tf, np.array([s1 - pd.Timedelta(hours=h) for h in (80, 72, 48, 24, 23, 6, 1)], dtype="datetime64[ns]"))
    for i in np.clip(pick, 0, len(tf) - 1):
        print(f"          {pd.Timestamp(tf[i])}  y={y[i]}  RUL={rul[i]:.2f}")

    # ---- candidate split
    Xc, tc = make_windows(transform(get_split(grid, "candidate"), scalers))
    print(f"\n[candidate 03-29] X {Xc.shape}")
    print("\nOK")
