#!/usr/bin/env python3
"""Standalone evaluation metrics for bulk RNA-seq deconvolution.

Implements the four DeconBenchmark accuracy metrics (NAR 2024):

    1. MAE — mean absolute error (sample-wise)
    2. SCorr — sample-wise Spearman correlation
    3. CCorr — cell-type-wise Spearman correlation
    4. MAECorr — mean absolute error between pairwise Pearson
       correlation matrices

Plus RMSE, Pearson r, and Wilcoxon rank-sum test as complementary metrics.
Zero external dependencies beyond numpy and scipy.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr


# ── DeconBenchmark four-metric suite ──────────────────────────────


def compute_scorr(
    true_props: np.ndarray,
    pred_props: np.ndarray,
) -> Dict[str, float]:
    """Sample-wise Spearman correlation (SCorr).

    For each sample, compute Spearman correlation between predicted and
    true proportions across cell types, then average across all samples.

    Args:
        true_props: (n_samples, n_cell_types) ground truth proportions.
        pred_props: (n_samples, n_cell_types) predicted proportions.

    Returns:
        ``{"scorr_mean": float, "scorr_per_sample": list[float]}``.
        Constant samples yield 0.0.
    """
    n_samples = true_props.shape[0]
    vals: list[float] = []
    for i in range(n_samples):
        if np.std(true_props[i]) > 0 and np.std(pred_props[i]) > 0:
            r, _ = spearmanr(true_props[i], pred_props[i])
            vals.append(float(r))
        else:
            vals.append(0.0)
    return {"scorr_mean": float(np.mean(vals)), "scorr_per_sample": vals}


def compute_ccorr(
    true_props: np.ndarray,
    pred_props: np.ndarray,
) -> Dict[str, float]:
    """Cell-type-wise Spearman correlation (CCorr).

    For each cell type, compute Spearman correlation across samples,
    then average across cell types.

    Args:
        true_props: (n_samples, n_cell_types) ground truth proportions.
        pred_props: (n_samples, n_cell_types) predicted proportions.

    Returns:
        ``{"ccorr_mean": float, "ccorr_per_type": list[float]}``.
        Constant columns yield 0.0.
    """
    n_types = true_props.shape[1]
    vals: list[float] = []
    for i in range(n_types):
        if np.std(true_props[:, i]) > 0 and np.std(pred_props[:, i]) > 0:
            r, _ = spearmanr(true_props[:, i], pred_props[:, i])
            vals.append(float(r))
        else:
            vals.append(0.0)
    return {"ccorr_mean": float(np.mean(vals)), "ccorr_per_type": vals}


def compute_maecorr(
    true_props: np.ndarray,
    pred_props: np.ndarray,
) -> Dict[str, float]:
    """Mean absolute error between ground-truth and predicted pairwise
    Pearson correlation matrices (MAECorr, DeconBenchmark metric).

    Computes the n_samples × n_samples Pearson correlation matrix for
    ground truth and predictions, then returns MAE between the upper
    triangles of the two matrices.

    Args:
        true_props: (n_samples, n_cell_types) ground truth proportions.
        pred_props: (n_samples, n_cell_types) predicted proportions.

    Returns:
        ``{"maecorr": float}`` — lower is better.
    """
    gt_corr = np.corrcoef(true_props)
    pd_corr = np.corrcoef(pred_props)
    if gt_corr.ndim < 2 or gt_corr.shape[0] < 2:
        return {"maecorr": 0.0}
    i_upper = np.triu_indices_from(gt_corr, k=1)
    return {"maecorr": float(np.mean(np.abs(gt_corr[i_upper] - pd_corr[i_upper])))}


# ── Supplementary metrics ─────────────────────────────────────────


def compute_pearson(
    true_props: np.ndarray,
    pred_props: np.ndarray,
) -> Dict[str, float]:
    """Per-cell-type Pearson correlation.

    Returns:
        ``{"pearson_mean": float, "pearson_per_type": list[float]}``.
    """
    n_types = true_props.shape[1]
    vals: list[float] = []
    for i in range(n_types):
        if np.std(true_props[:, i]) > 0 and np.std(pred_props[:, i]) > 0:
            r, _ = pearsonr(true_props[:, i], pred_props[:, i])
            vals.append(float(r))
        else:
            vals.append(0.0)
    return {"pearson_mean": float(np.mean(vals)), "pearson_per_type": vals}


def compute_rmse(
    true_props: np.ndarray,
    pred_props: np.ndarray,
) -> Dict[str, float]:
    """Root mean squared error.

    Returns:
        ``{"rmse_overall": float, "rmse_per_type": list[float],
        "rmse_mean_per_type": float}``.
    """
    mse = np.mean((pred_props - true_props) ** 2, axis=0)
    rmse_per_type = np.sqrt(mse)
    return {
        "rmse_overall": float(np.sqrt(np.mean((pred_props - true_props) ** 2))),
        "rmse_per_type": rmse_per_type.tolist(),
        "rmse_mean_per_type": float(np.mean(rmse_per_type)),
    }


def compute_mae(
    true_props: np.ndarray,
    pred_props: np.ndarray,
) -> Dict[str, float]:
    """Mean absolute error.

    Returns:
        ``{"mae_overall": float, "mae_per_type": list[float]}``.
    """
    mae_per_type = np.mean(np.abs(pred_props - true_props), axis=0)
    return {
        "mae_overall": float(np.mean(mae_per_type)),
        "mae_per_type": mae_per_type.tolist(),
    }


def compute_wt(
    true_props: np.ndarray,
    pred_props: np.ndarray,
) -> Dict[str, float]:
    """Per-cell-type Wilcoxon rank-sum (Mann-Whitney U) p-values.

    Tests whether true and predicted proportion distributions differ.
    High p-value (>0.05) indicates indistinguishable distributions.

    Returns:
        ``{"wt_mean": float, "wt_per_type": list[float]}``.
        Degenerate columns yield 1.0.
    """
    n_types = true_props.shape[1]
    p_vals: list[float] = []
    for i in range(n_types):
        t = true_props[:, i]
        p = pred_props[:, i]
        if np.unique(t).size >= 2 and np.unique(p).size >= 2:
            try:
                _, pv = mannwhitneyu(t, p, alternative="two-sided")
                p_vals.append(float(pv))
            except ValueError:
                p_vals.append(1.0)
        else:
            p_vals.append(1.0)
    return {"wt_mean": float(np.mean(p_vals)), "wt_per_type": p_vals}


# ── Main entry point ──────────────────────────────────────────────


def evaluate_deconvolution(
    true_props: np.ndarray,
    pred_props: np.ndarray,
    cell_types: Optional[List[str]] = None,
) -> dict:
    """Full evaluation suite for deconvolution predictions.

    Computes all DeconBenchmark metrics (MAE, SCorr, CCorr, MAECorr)
    plus supplementary metrics (Pearson, RMSE, Wilcoxon).  When
    *cell_types* is provided, keys are unchanged — callers should map
    ``ccorr_per_type`` / ``pearson_per_type`` etc. to cell types.

    Args:
        true_props: (n_samples, n_cell_types) ground truth proportions.
        pred_props: (n_samples, n_cell_types) predicted proportions.
        cell_types: ignored (kept for backwards compatibility).

    Returns:
        Flat dict containing all metric keys.
    """
    if not (np.isfinite(true_props).all() and np.isfinite(pred_props).all()):
        raise ValueError("Input arrays must contain only finite values")
    results: dict = {}
    results.update(compute_mae(true_props, pred_props))
    results.update(compute_scorr(true_props, pred_props))
    results.update(compute_ccorr(true_props, pred_props))
    results.update(compute_maecorr(true_props, pred_props))
    results.update(compute_pearson(true_props, pred_props))
    results.update(compute_rmse(true_props, pred_props))
    results.update(compute_wt(true_props, pred_props))
    return results
