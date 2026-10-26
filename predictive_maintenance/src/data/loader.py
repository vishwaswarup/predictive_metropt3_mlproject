"""Load the MetroPT-3 CSV, put it on an exact 10 s grid, and slice it into named splits.

Run the self-test from the project root:  python -m src.data.loader
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import config as C

Split = Literal["train", "failure_labeled", "candidate", "full"]
SPLITS: tuple[str, ...] = ("train", "failure_labeled", "candidate", "full")


def find_raw_csv(data_dir: Path = C.DATA_DIR) -> Path:
    """Locate the MetroPT-3 CSV in ``data_dir``.

    Input:  directory to search (default ``config.DATA_DIR``).
    Output: path of the single file matching ``config.RAW_CSV_GLOB``.
    Logic:  raises FileNotFoundError if none match, ValueError if the match is ambiguous.
    """
    matches = sorted(Path(data_dir).glob(C.RAW_CSV_GLOB))
    if not matches:
        raise FileNotFoundError(f"No file matching {C.RAW_CSV_GLOB!r} in {data_dir}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous raw CSV, found {[m.name for m in matches]}")
    return matches[0]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw CSV into a time-indexed DataFrame.

    Input:  path to the CSV (default: auto-located in ``config.DATA_DIR``).
    Output: DataFrame indexed by a sorted, unique ``DatetimeIndex`` named ``timestamp``
            with the 15 sensor columns (``config.FEATURE_COLS``) as float64.
    Logic:  drops the pandas ``Unnamed: 0`` index column, parses timestamps, validates that
            every expected sensor column exists, then sorts and de-duplicates by timestamp.
    """
    path = Path(path) if path is not None else find_raw_csv()
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    missing = [c for c in [C.TIMESTAMP_COL, *C.FEATURE_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV schema mismatch, missing columns: {missing}")

    df[C.TIMESTAMP_COL] = pd.to_datetime(df[C.TIMESTAMP_COL])
    df = df.sort_values(C.TIMESTAMP_COL).drop_duplicates(C.TIMESTAMP_COL)
    df = df.set_index(C.TIMESTAMP_COL)[C.FEATURE_COLS].astype("float64")
    return df


def resample_to_10s(df: pd.DataFrame) -> pd.DataFrame:
    """Put the data on an exact ``SAMPLING_INTERVAL_S`` grid and mark unusable gaps.

    Input:  DataFrame from :func:`load_raw` (irregular ~10 s timestamps, long idle gaps).
    Output: float32 DataFrame on a regular grid, same feature columns plus two booleans:
              ``is_filled`` - row was missing but short enough to be forward-filled;
              ``is_gap``    - row sits in a longer outage; its feature values are NaN.
    Logic:  each 10 s bin keeps its last reading. A run of empty bins of length
            <= ``FORWARD_FILL_MAX_STEPS`` is forward-filled in full. A longer run is never
            partly filled (that would invent stale data right after the compressor stops):
            the whole run is set to NaN and flagged ``is_gap``.
    """
    grid = df[C.FEATURE_COLS].resample(C.SAMPLING_FREQ).last()

    missing = grid.isna().any(axis=1)
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    long_gap = missing & (run_len > C.FORWARD_FILL_MAX_STEPS)
    filled = missing & ~long_gap

    out = grid.ffill()
    out.loc[long_gap, C.FEATURE_COLS] = np.nan
    out = out.astype("float32")
    out["is_filled"] = filled
    out["is_gap"] = long_gap
    return out


def _on_dates(index: pd.DatetimeIndex, dates: list[str]) -> np.ndarray:
    """Boolean mask of index entries that fall on any of the given calendar dates."""
    return index.normalize().isin(pd.to_datetime(dates)).astype(bool)


def _mask_rows(df: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Return a copy where ``rows`` are unusable: values NaN, ``is_gap`` and ``is_masked`` True."""
    out = df.copy()
    if "is_masked" not in out:
        out["is_masked"] = False
    out.loc[rows, C.FEATURE_COLS] = np.nan
    out.loc[rows, "is_gap"] = True
    out.loc[rows, "is_masked"] = True
    return out


def get_split(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Slice the resampled data according to the project's split rules.

    Input:  ``df`` from :func:`resample_to_10s`; ``split`` in
            ``"train" | "failure_labeled" | "candidate" | "full"``.
    Output: a copy of the slice. Rows that must not be used (sensor-fault days, and in the
            train split the excluded anomaly-candidate days) are *kept in place* but masked:
            values NaN, ``is_gap=True``, ``is_masked=True``. They are not dropped because
            dropping would make windows silently bridge the hole.
    Splits:
      train           start of data .. ``TRAIN_END`` (whole day), minus ``EXCLUDE_FROM_TRAINING``.
      failure_labeled from (first failure start - max(MAX_RUL, FAILURE_LOOKAHEAD_HOURS))
                      to the end of the data, so every failure's lead-up is covered. Its first
                      ~3 days overlap the train split (see README note on classifier leakage).
      candidate       only the ``EXCLUDE_FROM_TRAINING`` days (undocumented anomaly candidates)
                      for out-of-sample evaluation.
      full            everything.
    All splits mask ``SENSOR_FAULT_DATES``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    idx = df.index

    if split == "train":
        keep = idx < pd.Timestamp(C.TRAIN_END) + pd.Timedelta(days=1)
    elif split == "failure_labeled":
        lead = max(C.MAX_RUL, C.FAILURE_LOOKAHEAD_HOURS)
        first_start = min(pd.Timestamp(a) for a, _ in C.FAILURE_WINDOWS)
        keep = idx >= first_start - pd.Timedelta(hours=lead)
    elif split == "candidate":
        keep = _on_dates(idx, C.EXCLUDE_FROM_TRAINING)
    else:
        keep = np.ones(len(idx), dtype=bool)

    out = df.loc[keep]
    mask = _on_dates(out.index, C.SENSOR_FAULT_DATES)
    if split == "train":
        mask |= _on_dates(out.index, C.EXCLUDE_FROM_TRAINING)
    return _mask_rows(out, mask)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    raw = load_raw()
    print(f"raw:       {raw.shape}  {raw.index.min()} -> {raw.index.max()}")

    grid = resample_to_10s(raw)
    print(f"resampled: {grid.shape}  grid step = {grid.index.to_series().diff().dropna().unique()}")
    print(f"  rows with data : {(~grid.is_gap).sum():>9,}  ({(~grid.is_gap).mean():.1%})")
    print(f"  forward-filled : {grid.is_filled.sum():>9,}")
    print(f"  gap (NaN)      : {grid.is_gap.sum():>9,}  ({grid.is_gap.mean():.1%})")
    assert grid.index.to_series().diff().dropna().nunique() == 1, "grid is not regular"
    assert not grid.loc[~grid.is_gap, C.FEATURE_COLS].isna().any().any(), "NaN outside gaps"

    print("\nsample rows (analogue):")
    print(grid.loc["2020-02-01 00:00:00":, C.ANALOGUE_COLS].head(3))
    print("sample rows (digital):")
    print(grid.loc["2020-02-01 00:00:00":, C.DIGITAL_COLS].head(3))

    print("\nsplits:")
    for name in SPLITS:
        s = get_split(grid, name)
        usable = (~s.is_gap).sum()
        print(f"  {name:<16} rows={len(s):>9,}  usable={usable:>9,}  masked={s.is_masked.sum():>7,}  "
              f"{s.index.min()} -> {s.index.max()}")

    train = get_split(grid, "train")
    assert train.index.max() < pd.Timestamp(C.TRAIN_END) + pd.Timedelta(days=1)
    assert train.loc["2020-03-29", C.FEATURE_COLS].isna().all().all(), "03-29 leaked into train"
    assert get_split(grid, "full").loc["2020-05-27", "is_gap"].all(), "fault day not masked"
    print("\nOK: no leakage past TRAIN_END, 03-29 and sensor-fault days are masked")
