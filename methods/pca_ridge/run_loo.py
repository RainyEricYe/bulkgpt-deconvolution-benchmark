#!/usr/bin/env python3
"""PCA + LOO RidgeCV — LOOCV version of pca_ridge.

Same approach as methods/pca_ridge/run.py (PCA on bulk expression
aligned to BulkFormer vocabulary, no backbone encoding) but uses
leave-one-out cross-validation instead of a single train/test split.

Usage::

    python methods/pca_ridge/run_loo.py \\
        --h5 data/2_real_bulk/sdy67.h5 \\
        --ground-truth data/2_real_bulk/sdy67_gt.csv \\
        --output-dir results/2_realbulk/sdy67/pca_ridge_loo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.data_loader import load_data
from core.metrics import evaluate_deconvolution


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA + LOO RidgeCV")
    parser.add_argument("--h5", required=True, help="Path to DeconBenchmark H5")
    parser.add_argument("--ground-truth", required=True, help="Path to GT CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scaler", action="store_true",
                        help="Apply StandardScaler before PCA (default: no scaler)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    bundle = load_data(args.h5, ground_truth=args.ground_truth)
    bulk = bundle.bulk
    gt_df = bundle.gt
    if bulk is None or gt_df is None:
        raise ValueError("H5 and GT CSV must both be loadable")
    vals = bulk.values.astype(np.float32)
    n_samples, n_genes = vals.shape
    print(f"Loaded {n_samples} samples, {n_genes} genes, {len(gt_df.columns)} cell types")

    # ── Align to BulkFormer vocabulary (20010 genes, -10.0 fill) ───────────
    bf_dir = Path(__file__).resolve().parent.parent.parent / "weights" / "bulkformer" / "source"
    bf_dir = Path(os.environ.get("BULKFORMER_DIR", str(bf_dir)))
    gene_csv = bf_dir / "data" / "bulkformer_gene_info.csv"
    if not gene_csv.exists():
        raise FileNotFoundError(f"BulkFormer gene list not found at {gene_csv}")
    bf_genes = list(pd.read_csv(gene_csv).iloc[:, 0])
    print(f"BulkFormer vocabulary: {len(bf_genes)} genes")

    g2i = {g.upper(): i for i, g in enumerate(bf_genes)}
    aligned = np.full((n_samples, len(bf_genes)), -10.0, dtype=np.float32)
    matched = 0
    for i, g in enumerate(bulk.columns):
        idx = g2i.get(str(g).upper())
        if idx is not None:
            aligned[:, idx] = vals[:, i]
            matched += 1
    print(f"  Aligned {matched}/{n_genes} genes")

    # ── Optional StandardScaler ────────────────────────────────────────────
    X = aligned
    if args.scaler:
        X = StandardScaler().fit_transform(X)
        print("  Applied StandardScaler")

    # ── PCA ────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    n_components = min(n_samples, X.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    emb = pca.fit_transform(X)
    print(f"PCA → {emb.shape} in {time.monotonic()-t0:.1f}s")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # ── LOO RidgeCV ────────────────────────────────────────────────────────
    gt_values = gt_df.values.astype(np.float64)
    gt_columns = list(gt_df.columns)
    orig_alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1000.0]

    loo_pred = np.zeros((n_samples, len(gt_columns)))
    per_type_r = {}
    per_type_rmse = {}
    best_alphas = {}
    ridge_start = time.monotonic()

    for holdout in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[holdout] = False
        train_emb = emb[train_mask]
        test_emb = emb[holdout].reshape(1, -1)

        for j, ct in enumerate(gt_columns):
            y = gt_values[train_mask, j]
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                loo_pred[holdout, j] = 0.0
                continue
            ridge = RidgeCV(alphas=orig_alphas, scoring="r2").fit(train_emb[mask], y[mask])
            loo_pred[holdout, j] = float(np.clip(ridge.predict(test_emb)[0], 0, None))

    ridge_time_s = round(time.monotonic() - ridge_start, 3)

    # ── Per-type metrics ───────────────────────────────────────────────────
    for j, ct in enumerate(gt_columns):
        mask = ~np.isnan(gt_values[:, j])
        if mask.sum() >= 2 and np.std(gt_values[mask, j]) > 1e-10:
            r = float(np.corrcoef(loo_pred[mask, j], gt_values[mask, j])[0, 1])
            per_type_r[ct] = round(r, 4) if not np.isnan(r) else None
        else:
            per_type_r[ct] = None
        if mask.sum() >= 2:
            per_type_rmse[ct] = round(
                float(np.sqrt(np.mean((loo_pred[mask, j] - gt_values[mask, j]) ** 2))), 4)
        else:
            per_type_rmse[ct] = None

    macro_r = float(np.nanmean([v for v in per_type_r.values() if v is not None]))

    # ── DeconBenchmark suite ───────────────────────────────────────────────
    deconbench = evaluate_deconvolution(gt_values, loo_pred, gt_columns)

    # ── Save outputs ───────────────────────────────────────────────────────
    pred_df = pd.DataFrame(loo_pred, index=gt_df.index, columns=gt_columns)
    pred_df.to_csv(out_dir / "proportions.csv")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(deconbench, f, indent=2)

    ridge_info = {
        "best_alphas": best_alphas,
        "pearson_per_type": per_type_r,
        "rmse_per_type": per_type_rmse,
        "macro_avg_r": round(macro_r, 4) if not np.isnan(macro_r) else None,
        "n_total": n_samples,
        "ridge_time_s": ridge_time_s,
    }
    with open(out_dir / "ridge_metrics.json", "w") as f:
        json.dump(ridge_info, f, indent=2)

    meta = {
        "embedding_dim": int(emb.shape[1]),
        "n_total": n_samples,
        "loo": True,
        "seed": args.seed,
        "aligned_genes": matched,
        "scaler": args.scaler,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    r = deconbench.get("pearson_mean", float("nan"))
    print(f"\nDone.  macro_avg r = {macro_r:.4f}  (DeconBench pearson_mean = {r:.4f})")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
