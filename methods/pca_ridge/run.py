#!/usr/bin/env python3
"""PCA + RidgeCV: simple baseline for real-bulk split evaluation.

Reduces bulk expression to principal components (scikit-learn) and
evaluates per-cell-type RidgeCV on a train/test split of the real
bulk samples.  No single-cell reference required.

Matching the original BulkFormer experiment (``eval_check_pretraining.py``,
``run_pca()``): PCA is fit **without** StandardScaler and uses
``random_state=42``.

Usage::

    python methods/pca_ridge/run.py \\
        --h5 data/2_real_bulk/sdy67.h5 \\
        --ground-truth data/2_real_bulk/sdy67_gt.csv \\
        --output-dir results/2_realbulk/sdy67/pca_ridge
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

_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.data_loader import load_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA + RidgeCV real-bulk split evaluation",
    )
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mode", default="predict", help=argparse.SUPPRESS)
    parser.add_argument("--h5", required=True, help="Path to DeconBenchmark H5")
    parser.add_argument("--ground-truth", required=True, help="Path to GT CSV")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data (canonical (n_samples, n_genes)) ───────────────────────
    bundle = load_data(args.h5, ground_truth=args.ground_truth)
    bulk = bundle.bulk
    gt_df = bundle.gt
    if bulk is None or gt_df is None:
        raise ValueError("H5 and GT CSV must both be loadable")
    vals = bulk.values.astype(np.float32)
    print(
        f"Loaded {len(gt_df)} samples, {vals.shape[1]} genes, "
        f"{len(gt_df.columns)} cell types",
    )

    # ── Align to BulkFormer vocabulary (20010 genes, -10.0 fill) ─────
    # Matching the original eval_check_pretraining.py run_pca():
    #   1. Load BulkFormer gene list
    #   2. Build (n_samples, 20010) matrix with -10.0 fill
    #   3. PCA on the aligned matrix
    bf_dir = Path(__file__).resolve().parent.parent.parent / "weights" / "bulkformer" / "source"
    bf_dir = Path(os.environ.get("BULKFORMER_DIR", str(bf_dir)))
    gene_csv = bf_dir / "data" / "bulkformer_gene_info.csv"
    if not gene_csv.exists():
        raise FileNotFoundError(f"BulkFormer gene list not found at {gene_csv}")
    bf_genes = list(pd.read_csv(gene_csv).iloc[:, 0])
    print(f"BulkFormer vocabulary: {len(bf_genes)} genes")

    n_samples = vals.shape[0]
    g2i = {g.upper(): i for i, g in enumerate(bf_genes)}
    bulkformer_expr = np.full((n_samples, len(bf_genes)), -10.0, dtype=np.float32)
    matched = 0
    for i, g in enumerate(bulk.columns):
        idx = g2i.get(str(g).upper())
        if idx is not None:
            bulkformer_expr[:, idx] = vals[:, i]
            matched += 1
    print(f"  Aligned {matched}/{vals.shape[1]} genes")

    # ── PCA (matching BulkFormer: no StandardScaler, random_state=42) ────
    t0 = time.monotonic()
    n_components = min(n_samples, bulkformer_expr.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    emb = pca.fit_transform(bulkformer_expr)
    print(f"PCA → {emb.shape} in {time.monotonic()-t0:.1f}s")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # ── Split + RidgeCV (matching original: no StandardScaler, 150 train) ─
    from sklearn.linear_model import RidgeCV

    n_bulk = emb.shape[0]
    gt_values = gt_df.values.astype(np.float64)
    gt_columns = list(gt_df.columns)
    dataset_name = out_dir.parent.name

    if dataset_name == "sdy67":
        train_idx = np.arange(150)
        test_idx = np.arange(200, min(250, n_bulk))
        split_label = "fixed_150_50"
    else:
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_bulk)
        n_train = int(n_bulk * 0.8)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        split_label = "random_80_20"

    train_emb, test_emb = emb[train_idx], emb[test_idx]
    train_gt, test_gt = gt_values[train_idx], gt_values[test_idx]

    orig_alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1000.0]
    test_pred = np.zeros_like(test_gt)
    best_alphas = {}
    per_type_r = {}
    per_type_rmse = {}
    ridge_start = time.monotonic()

    for j, ct in enumerate(gt_columns):
        y = train_gt[:, j]
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            test_pred[:, j] = 0.0
            best_alphas[ct] = float("nan")
            per_type_r[ct] = None
            per_type_rmse[ct] = None
            continue
        ridge = RidgeCV(alphas=orig_alphas, scoring="r2").fit(train_emb[mask], y[mask])
        best_alphas[ct] = float(ridge.alpha_)
        pred = ridge.predict(test_emb)
        pred = np.clip(pred, 0, None)
        test_pred[:, j] = pred

    # Per-type Pearson r / RMSE (matching original metric computation)
    for j, ct in enumerate(gt_columns):
        mask = ~np.isnan(test_gt[:, j])
        if mask.sum() >= 2 and np.std(test_gt[mask, j]) > 1e-10:
            r = float(np.corrcoef(test_pred[mask, j], test_gt[mask, j])[0, 1])
            per_type_r[ct] = round(r, 4) if not np.isnan(r) else None
        else:
            per_type_r[ct] = None
        if mask.sum() >= 2:
            per_type_rmse[ct] = round(
                float(np.sqrt(np.mean((test_pred[mask, j] - test_gt[mask, j]) ** 2))), 4)
        else:
            per_type_rmse[ct] = None

    ridge_time_s = round(time.monotonic() - ridge_start, 3)

    # DeconBenchmark metrics (to_publish standard)
    _project = Path(__file__).resolve().parent.parent.parent
    if str(_project) not in sys.path:
        sys.path.insert(0, str(_project))
    from core.metrics import evaluate_deconvolution

    deconbench = evaluate_deconvolution(test_gt, test_pred, gt_columns)
    macro_r = float(np.nanmean([v for v in per_type_r.values() if v is not None])) if any(v is not None for v in per_type_r.values()) else float("nan")

    # Full-sample predictions
    full_pred = np.zeros((n_bulk, len(gt_columns)))
    for j, ct in enumerate(gt_columns):
        y = gt_values[:, j]
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            continue
        ridge = RidgeCV(alphas=orig_alphas, scoring="r2").fit(emb[mask], y[mask])
        full_pred[:, j] = ridge.predict(emb)
    full_pred = np.clip(full_pred, 0, None)
    full_pred_df = pd.DataFrame(full_pred, index=gt_df.index, columns=gt_columns)
    full_pred_df.to_csv(out_dir / "proportions.csv")

    # ── Save outputs ─────────────────────────────────────────────────────
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(deconbench, f, indent=2)

    ridge_info = {
        "best_alphas": best_alphas,
        "pearson_per_type": per_type_r,
        "rmse_per_type": per_type_rmse,
        "macro_avg_r": round(macro_r, 4) if not np.isnan(macro_r) else None,
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "split": split_label,
        "ridge_time_s": ridge_time_s,
    }
    with open(out_dir / "ridge_metrics.json", "w") as f:
        json.dump(ridge_info, f, indent=2)

    meta = {
        "embedding_dim": int(emb.shape[1]),
        "n_total": n_bulk,
        "split": split_label,
        "seed": args.seed,
        "aligned_genes": matched,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    r = deconbench.get("pearson_mean", float("nan"))
    print(f"\nDone.  macro_avg r = {macro_r:.4f}  (DeconBench pearson_mean = {r:.4f})")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
