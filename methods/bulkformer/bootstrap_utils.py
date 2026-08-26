"""Shared bootstrap utilities for BulkFormer ensemble methods.

``bootstrap_ridge()`` implements the core logic: for each cell type, fit
*N* RidgeCV regressors on bootstrap-resampled training data and return
the mean/std of test predictions.

Used by:
  - ``bulkformer_bootstrap``  (pretrained + global_proj + 50× bootstrap)
  - ``bulkformer_fstat``      (F-stat weighting + optional bootstrap-50)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

# Same alpha grid used across all BulkFormer experiments.
DEFAULT_ALPHAS = [
    0.01, 0.03, 0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1000.0,
]


def bootstrap_ridge(
    emb_train: np.ndarray,
    gt_train: pd.DataFrame,
    emb_test: np.ndarray,
    n_ensemble: int = 50,
    seed: int = 42,
    alphas: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Bootstrap RidgeCV ensemble per cell type.

    For each cell type, ``n_ensemble`` RidgeCV regressors are fit on
    bootstrap-resampled training data (sampling *n* indices with
    replacement, where *n* = number of training samples).  Test
    predictions are averaged across iterations.

    Args:
        emb_train: (n_train, embed_dim) training embeddings.
        gt_train: (n_train, n_types) ground-truth proportions.
        emb_test: (n_test, embed_dim) test embeddings.
        n_ensemble: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        alphas: Ridge regularisation strengths (default: ``DEFAULT_ALPHAS``).

    Returns:
        pred_mean: (n_test, n_types) mean prediction across ensemble.
        pred_std: (n_test, n_types) std across ensemble.
        details: Nested dict ``{type_name: {"r": [...], "rmse": [...]}}``
            containing per-iteration metrics for diagnostic use.
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    n_train = emb_train.shape[0]
    n_test = emb_test.shape[0]
    type_names = list(gt_train.columns)
    n_types = len(type_names)

    rng = np.random.default_rng(seed)
    preds = np.zeros((n_ensemble, n_test, n_types), dtype=np.float64)
    details: dict[str, Any] = {}

    for t_idx, t_name in enumerate(type_names):
        y_train = gt_train.iloc[:, t_idx].values
        y_train = np.maximum(y_train, 0.0)

        iter_r: list[float] = []
        iter_rmse: list[float] = []

        for i in range(n_ensemble):
            idx = rng.integers(0, n_train, size=n_train)
            model = RidgeCV(alphas=alphas, scoring="r2").fit(
                emb_train[idx], y_train[idx],
            )
            y_pred = model.predict(emb_test)
            y_pred = np.maximum(y_pred, 0.0)

            preds[i, :, t_idx] = y_pred

            # Compute per-iteration metrics for diagnostic logging.
            with np.errstate(invalid="ignore"):
                r = np.corrcoef(y_pred, y_train[:n_test])[0, 1] if n_test > 1 else 0.0
            rmse = float(np.sqrt(np.mean((y_pred - y_train[:n_test]) ** 2)))
            if not np.isfinite(r):
                r = 0.0
            iter_r.append(r)
            iter_rmse.append(rmse)

        details[t_name] = {"r": iter_r, "rmse": iter_rmse}

    # mean and std across bootstrap iterations
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    # renormalize
    row_sums = pred_mean.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    pred_mean = pred_mean / row_sums

    return pred_mean, pred_std, details
