#!/usr/bin/env python3
"""Leave-one-out RidgeCV for bulkformer variants."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

_to_publish = Path(__file__).resolve().parent.parent.parent
import sys
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.deconv.utils import renormalize_props

DEFAULT_ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1000.0]


def run_loo_ridge(
    embeddings: np.ndarray,
    gt_df: pd.DataFrame,
    alphas: list[float] | None = None,
    use_scaler: bool = False,
) -> dict:
    """Leave-one-out RidgeCV evaluation.

    For each sample i: train RidgeCV on all except i, predict i.
    Returns dict with deconbench, ridge_specific, metadata, full_predictions_df.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    alphas = alphas or DEFAULT_ALPHAS
    n_bulk = embeddings.shape[0]
    gt_values = gt_df.values.astype(np.float64)
    gt_columns = list(gt_df.columns)
    embed_dim = embeddings.shape[1]

    loo_pred = np.zeros((n_bulk, len(gt_columns)))
    ridge_start = time.monotonic()

    for holdout in range(n_bulk):
        train_mask = np.ones(n_bulk, dtype=bool)
        train_mask[holdout] = False
        train_emb = embeddings[train_mask]
        test_emb = embeddings[holdout].reshape(1, -1)

        if use_scaler:
            scaler = StandardScaler()
            train_emb = scaler.fit_transform(train_emb)
            test_emb = scaler.transform(test_emb)

        for j, ct in enumerate(gt_columns):
            y = gt_values[train_mask, j]
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                loo_pred[holdout, j] = 0.0
                continue
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(train_emb[mask], y[mask])
            loo_pred[holdout, j] = float(ridge.predict(test_emb)[0])

    loo_pred = renormalize_props(loo_pred, zero_fill="zero")
    loo_pred = np.nan_to_num(loo_pred, nan=0.0, posinf=0.0, neginf=0.0)
    ridge_time_s = round(time.monotonic() - ridge_start, 3)

    # Per-type metrics
    pearson_per_type: dict[str, float | None] = {}
    rmse_per_type: dict[str, float | None] = {}
    for j, ct in enumerate(gt_columns):
        mask = ~np.isnan(gt_values[:, j])
        if mask.sum() >= 2 and np.std(gt_values[mask, j]) > 1e-10:
            r = float(np.corrcoef(loo_pred[mask, j], gt_values[mask, j])[0, 1])
            pearson_per_type[ct] = round(r, 4) if not np.isnan(r) else None
        else:
            pearson_per_type[ct] = None
        if mask.sum() >= 2:
            rmse_per_type[ct] = round(
                float(np.sqrt(np.mean((loo_pred[mask, j] - gt_values[mask, j]) ** 2))), 4)
        else:
            rmse_per_type[ct] = None

    from core.metrics import evaluate_deconvolution
    deconbench = evaluate_deconvolution(gt_values, loo_pred, gt_columns)
    pred_df = pd.DataFrame(loo_pred, index=gt_df.index, columns=gt_columns)

    return {
        "deconbench": deconbench,
        "ridge_specific": {
            "pearson_per_type": pearson_per_type,
            "rmse_per_type": rmse_per_type,
            "n_total": n_bulk,
            "ridge_time_s": ridge_time_s,
        },
        "metadata": {
            "embedding_dim": embed_dim,
            "n_total": n_bulk,
            "loo": True,
        },
        "full_predictions_df": pred_df,
    }


def save_loo_results(out_dir: Path, result: dict, meta_extra: dict | None = None) -> None:
    """Save loo_ridge_cv result dict to a directory."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    result["full_predictions_df"].to_csv(out_dir / "proportions.csv")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result["deconbench"], f, indent=2)
    with open(out_dir / "ridge_metrics.json", "w") as f:
        json.dump(result["ridge_specific"], f, indent=2)

    meta = dict(result["metadata"])
    if meta_extra:
        meta.update(meta_extra)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
