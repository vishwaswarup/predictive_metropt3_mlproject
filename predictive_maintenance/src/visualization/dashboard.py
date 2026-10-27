"""Dashboard figures: sensors with anomaly overlay, anomaly score, RUL, decisions, SHAP importance.

``build_dashboard`` writes one interactive plotly HTML; ``plot_decision_timeline`` and the SHAP chart
(``classifier.plot_shap_importance``) are also saved as PNGs. Series are downsampled for the browser.

Run from the project root:  python -m src.visualization.dashboard
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as C

DASHBOARD_PATH = C.PLOT_DIR / "dashboard.html"
DECISION_PLOT_PATH = C.PLOT_DIR / "decision_timeline.png"
LEVEL_COLORS = {"NORMAL": "#9aa5b1", "WATCH": "#e6b422", "SCHEDULE_MAINTENANCE": "#e8772e", "URGENT": "#c0392b"}
DEFAULT_SENSORS = ["TP3", "Oil_temperature", "Motor_current"]


def _downsample(n: int, max_points: int) -> np.ndarray:
    """Indices for at most ``max_points`` evenly spaced samples out of ``n``."""
    return np.arange(0, n, max(1, n // max_points))


def _failure_shapes(failure_windows: Sequence[tuple[str, str]]) -> list[dict]:
    """Shaded vertical bands for the failure windows (all x-axes)."""
    return [dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1, fillcolor="rgba(192,57,43,0.18)",
                 line_width=0) for s, e in failure_windows]


def build_dashboard(grid: pd.DataFrame, score: pd.Series, decisions: pd.DataFrame, rul_pred: pd.Series | None = None,
                    rul_true: pd.Series | None = None, sensors: Sequence[str] = DEFAULT_SENSORS,
                    failure_windows: Sequence[tuple[str, str]] = C.FAILURE_WINDOWS, max_points: int = 6000,
                    path: Path = DASHBOARD_PATH) -> Path:
    """Interactive multi-panel dashboard.

    Input:  10 s sensor grid; anomaly score (error / threshold) and scheduler decisions indexed by window
            end-time; optional predicted / true RUL; sensors to plot; failure windows (shaded red).
    Output: path of the saved HTML. Panels share the x-axis: sensors (anomalous windows marked), anomaly
            score with the threshold line, RUL (predicted vs true, with the no-skill caveat), decision level.
    """
    rows = len(sensors) + 3
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        subplot_titles=[*sensors, "Anomaly score (error / threshold)", "RUL (hours) - validated as no better than a constant",
                                        "Maintenance decision"])
    anomalous = score.index[(score >= C.SCHED_ANOMALY_WATCH).to_numpy()]
    for i, col in enumerate(sensors, 1):
        s = grid[col]
        ix = _downsample(len(s), max_points)
        fig.add_trace(go.Scattergl(x=s.index[ix], y=s.to_numpy()[ix], mode="lines", name=col, line=dict(width=1)), row=i, col=1)
        hit = s.reindex(anomalous).dropna()
        hit = hit.iloc[_downsample(len(hit), max_points // 3)]
        fig.add_trace(go.Scattergl(x=hit.index, y=hit.to_numpy(), mode="markers", name="anomalous window",
                                   marker=dict(size=3, color=LEVEL_COLORS["URGENT"]), showlegend=i == 1), row=i, col=1)
    r = len(sensors) + 1
    ix = _downsample(len(score), max_points)
    fig.add_trace(go.Scattergl(x=score.index[ix], y=score.to_numpy()[ix], mode="lines", name="score", line=dict(width=1)), row=r, col=1)
    fig.add_hline(y=C.SCHED_ANOMALY_WATCH, line_dash="dash", line_color=LEVEL_COLORS["URGENT"], row=r, col=1)
    fig.update_yaxes(type="log", row=r, col=1)
    r += 1
    if rul_pred is not None:
        ix = _downsample(len(rul_pred), max_points)
        fig.add_trace(go.Scattergl(x=rul_pred.index[ix], y=rul_pred.to_numpy()[ix], mode="lines", name="predicted RUL"), row=r, col=1)
    if rul_true is not None:
        ok = rul_true.dropna()
        fig.add_trace(go.Scattergl(x=ok.index[_downsample(len(ok), max_points)], y=ok.to_numpy()[_downsample(len(ok), max_points)],
                                   mode="markers", name="true RUL", marker=dict(size=3)), row=r, col=1)
    r += 1
    d = decisions.iloc[_downsample(len(decisions), max_points)]
    fig.add_trace(go.Scattergl(x=d.index, y=d.level_idx, mode="markers", name="level", showlegend=False,
                               marker=dict(size=4, color=[LEVEL_COLORS[l] for l in d.level]),
                               text=d.level, hovertemplate="%{x}<br>%{text}<extra></extra>"), row=r, col=1)
    fig.update_yaxes(tickvals=list(range(len(C.SCHED_LEVELS))), ticktext=C.SCHED_LEVELS, row=r, col=1)
    shapes = []
    for k in range(1, rows + 1):
        for sh in _failure_shapes(failure_windows):
            shapes.append({**sh, "xref": "x" if k == 1 else f"x{k}"})
    fig.update_layout(height=230 * rows, shapes=shapes, template="plotly_white", margin=dict(l=60, r=20, t=50, b=30),
                      title="CompressorWatch · MetroPT-3 (red bands = documented failures)")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    return Path(path)


def plot_decision_timeline(decisions: pd.DataFrame, failure_windows: Sequence[tuple[str, str]] = C.FAILURE_WINDOWS,
                           path: Path = DECISION_PLOT_PATH) -> Path:
    """Static PNG: scheduler level over time, failures shaded."""
    fig, ax = plt.subplots(figsize=(14, 3))
    ix = _downsample(len(decisions), 20000)
    d = decisions.iloc[ix]
    ax.scatter(d.index, d.level_idx, s=2, c=[LEVEL_COLORS[l] for l in d.level])
    for s, e in failure_windows:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="tab:red", alpha=0.2)
    ax.set_yticks(range(len(C.SCHED_LEVELS)), C.SCHED_LEVELS)
    ax.set_title("Maintenance decision over time (shaded = documented failures)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


if __name__ == "__main__":
    from src.data.loader import get_split, load_raw, resample_to_10s
    from src.maintenance.scheduler import schedule
    from src.models.autoencoder import load_threshold
    from src.models.classifier import RECON_COL, load_or_build_features
    from src.data.preprocessor import compute_rul

    F = load_or_build_features()
    score = pd.Series(F[RECON_COL].to_numpy() / load_threshold(), index=F.index)
    dec = schedule(score.to_numpy(), F.index.to_numpy())
    grid = get_split(resample_to_10s(load_raw()), "failure_labeled")
    rul_true = pd.Series(compute_rul(F.index.to_numpy(), C.FAILURE_WINDOWS, C.MAX_RUL), index=F.index)
    print("saved", build_dashboard(grid, score, dec, rul_true=rul_true))
    print("saved", plot_decision_timeline(dec))
