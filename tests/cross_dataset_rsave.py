#!/usr/bin/env python3
"""Compute Pearson and Spearman correlation matrices for cross-dataset PBMC.

Reads saved predictions and GT, computes per-type Pearson r and Spearman rho
for every source→target pair.  Saves both matrices as JSON.

Usage:
    python tests/cross_dataset_rsave.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(HERE))
from cross_dataset_pbmc import (
    ALL_DATASETS, COARSE_MAP, load_h5_gt, aggregate_to_coarse,
)

RESULTS_DIR = HERE / "cross_dataset_pbmc"


def compute_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    mask = ~(np.isnan(true) | np.isnan(pred))
    if mask.sum() < 2:
        return float("nan")
    tp = true[mask]
    pp = pred[mask]
    if np.std(tp) < 1e-12 or np.std(pp) < 1e-12:
        return float("nan")
    return float(np.corrcoef(tp, pp)[0, 1])


def compute_spearman(pred: np.ndarray, true: np.ndarray) -> float:
    mask = ~(np.isnan(true) | np.isnan(pred))
    if mask.sum() < 2:
        return float("nan")
    tp = true[mask]
    pp = pred[mask]
    if np.std(tp) < 1e-12 or np.std(pp) < 1e-12:
        return float("nan")
    return float(spearmanr(tp, pp)[0])


def main() -> None:
    # Pre-load all GTs
    gt_cache: dict[str, pd.DataFrame] = {}
    for ds in ALL_DATASETS:
        _, _, _, gt = load_h5_gt(ds)
        gt_cache[ds] = aggregate_to_coarse(gt, ds)

    backbones = ["random_mean_pool", "pca_ridge"]
    for backbone in backbones:
        pearson_mat: dict[str, dict[str, float]] = {s: {} for s in ALL_DATASETS}
        spearman_mat: dict[str, dict[str, float]] = {s: {} for s in ALL_DATASETS}
        n_types_mat: dict[str, dict[str, int]] = {s: {} for s in ALL_DATASETS}

        summary_path = RESULTS_DIR / f"summary_{backbone}.json"
        if not summary_path.exists():
            print(f"SKIP {backbone}: no summary found")
            continue

        summary = json.loads(summary_path.read_text())

        for s in summary:
            src, tgt = s["source"], s["target"]
            common = s["common_types"]

            pred_path = RESULTS_DIR / f"{src}_to_{tgt}" / backbone / "proportions.csv"
            if not pred_path.exists():
                continue

            pred_df = pd.read_csv(pred_path, index_col=0)
            if tgt not in gt_cache:
                continue
            tgt_coarse = gt_cache[tgt]

            common_avail = [c for c in common if c in pred_df.columns and c in tgt_coarse.columns]
            if not common_avail:
                continue

            pred_v = pred_df[common_avail].values.astype(np.float64)
            gt_v = tgt_coarse[common_avail].values.astype(np.float64)

            pearson_vals = []
            spearman_vals = []
            for j in range(len(common_avail)):
                p = compute_pearson(pred_v[:, j], gt_v[:, j])
                s = compute_spearman(pred_v[:, j], gt_v[:, j])
                if not np.isnan(p):
                    pearson_vals.append(p)
                if not np.isnan(s):
                    spearman_vals.append(s)

            pearson_mat[src][tgt] = round(float(np.nanmean(pearson_vals)), 4) if pearson_vals else None
            spearman_mat[src][tgt] = round(float(np.nanmean(spearman_vals)), 4) if spearman_vals else None
            n_types_mat[src][tgt] = len(common_avail)

        # Save matrices
        out = RESULTS_DIR / f"correlation_{backbone}.json"
        with open(out, "w") as f:
            json.dump({
                "pearson": pearson_mat,
                "spearman": spearman_mat,
                "n_common_types": n_types_mat,
            }, f, indent=2)
        print(f"Saved: {out}")

        # Print text matrices
        for metric_name, mat in [("Pearson r", pearson_mat), ("Spearman ρ", spearman_mat)]:
            print(f"\n=== {metric_name} ({backbone}) ===")
            ds_short = {d: d[:12] for d in ALL_DATASETS}
            header = f"{'':>14}" + "".join(f"{ds_short[d]:>13}" for d in ALL_DATASETS)
            print(header)
            print("-" * len(header))
            for src in ALL_DATASETS:
                row = f"{ds_short[src]:>14}"
                for tgt in ALL_DATASETS:
                    v = mat.get(src, {}).get(tgt, None)
                    if v is None:
                        row += f"{'':>13}"
                    else:
                        row += f"{v:>13.4f}"
                print(row)


if __name__ == "__main__":
    main()
