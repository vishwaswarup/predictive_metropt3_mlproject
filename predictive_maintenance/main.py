"""End-to-end runner: data -> autoencoder -> classifier -> RUL -> scheduler -> dashboard.

Each stage reuses saved artefacts when they exist (autoencoder checkpoint, feature cache, OOF cache);
pass ``--rebuild`` to rebuild the caches (the autoencoder itself is trained with
``python -m src.models.autoencoder`` after deleting its checkpoint). Out-of-fold (leave-one-failure-out) predictions feed the report so
nothing is evaluated on data its model trained on.

Run from the project root:  python main.py [--rebuild]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config as C

OOF_CACHE = C.CACHE_DIR / "oof_predictions.parquet"


def out_of_fold(F: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Out-of-fold failure probability and RUL per window (cached; rebuilt if the feature cache is newer)."""
    from src.data.preprocessor import in_failure_mask, label_windows
    from src.models.classifier import cross_validate
    from src.models.rul_estimator import build_rul_dataset, cross_validate_rul

    feat_cache = C.CACHE_DIR / "failure_labeled_features.parquet"
    if OOF_CACHE.exists() and not force and OOF_CACHE.stat().st_mtime > feat_cache.stat().st_mtime:
        return pd.read_parquet(OOF_CACHE)
    t = F.index.to_numpy()
    d = build_rul_dataset(F)
    y = label_windows(t, C.FAILURE_WINDOWS, C.FAILURE_LOOKAHEAD_HOURS)
    proba, _ = cross_validate(F, y, d["folds"], in_failure_mask(t, C.FAILURE_WINDOWS))
    rul, _ = cross_validate_rul(F, d)
    out = pd.DataFrame({"proba": proba, "rul_pred": rul}, index=F.index)
    out.to_parquet(OOF_CACHE)
    return out


def main(rebuild: bool = False) -> None:
    """Run every stage and write models, reports and the dashboard under ``outputs/``."""
    from src.data.loader import get_split, load_raw, resample_to_10s
    from src.data.preprocessor import compute_rul
    from src.maintenance.scheduler import CAVEAT, schedule, summarise
    from src.models import autoencoder
    from src.models.autoencoder import load_threshold
    from src.models.classifier import RECON_COL, load_or_build_features, shap_importance, plot_shap_importance
    from src.visualization.dashboard import build_dashboard, plot_decision_timeline

    if not autoencoder.MODEL_PATH.exists():
        print("[autoencoder] no checkpoint: run `python -m src.models.autoencoder` first (training is slow).")
        return
    print("[features] loading / building cached window features ...")
    F = load_or_build_features(force=rebuild)
    print("[models] out-of-fold classifier and RUL predictions ...")
    oof = out_of_fold(F, force=rebuild)

    imp_path = C.REPORT_DIR / "shap_importance.csv"
    if imp_path.exists():
        print("[shap] chart:", plot_shap_importance(pd.read_csv(imp_path)))
    score = pd.Series(F[RECON_COL].to_numpy() / load_threshold(), index=F.index)
    dec = schedule(score.to_numpy(), F.index.to_numpy(), oof.proba.to_numpy(), oof.rul_pred.to_numpy())

    summary = summarise(dec, F.index.to_numpy())
    print("[scheduler] decision mix:", dec.level.value_counts(normalize=True).round(3).to_dict())
    print(summary.round(2).to_string())
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(C.REPORT_DIR / "scheduler_summary.csv")
    (C.REPORT_DIR / "scheduler_caveat.json").write_text(json.dumps({"caveat": CAVEAT}, indent=2))

    grid = get_split(resample_to_10s(load_raw()), "failure_labeled")
    rul_true = pd.Series(compute_rul(F.index.to_numpy(), C.FAILURE_WINDOWS, C.MAX_RUL), index=F.index)
    print("[dashboard]", build_dashboard(grid, score, dec, oof.rul_pred, rul_true))
    print("[dashboard]", plot_decision_timeline(dec))
    print("Caveat:", CAVEAT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true", help="rebuild the feature and out-of-fold caches")
    main(ap.parse_args().rebuild)
