#!/usr/bin/env python3
"""PCA within each LOO fold — no leakage baseline / upper bound estimate.

For each held-out sample (LOO fold):
  1. Fit PCA on the n-1 training samples (aligned to BulkFormer 20010 vocab)
  2. Transform the held-out test sample through that PCA
  3. Train RidgeCV per cell type on training PCA features
  4. Predict held-out sample

This eliminates the data leakage in methods/pca_ridge/run_loo.py
(which fits PCA once on ALL samples before LOO splitting).
Comparing the two reveals how much global PCA leakage inflates performance.

Usage::

    # Single dataset
    python scripts/run_loo_within_fold_pca.py \\
        --h5 data/2_real_bulk/sdy67.h5 \\
        --ground-truth data/2_real_bulk/sdy67_gt.csv

    # All 12 real-bulk datasets
    python scripts/run_loo_within_fold_pca.py --all

Output: tests/loo_within_fold/{dataset}/
  - proportions.csv       (n_samples × n_types)
  - metrics.json          (DeconBenchmark suite)
  - ridge_metrics.json    (per-type Pearson r / RMSE / best alpha)
  - metadata.json         (run config + per-fold PCA explained var)
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

_to_publish = Path(__file__).resolve().parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.data_loader import load_data
from core.metrics import evaluate_deconvolution

# ── Dataset registry (matching cross_dataset_experiments.md) ────────────────

REAL_BULK_DATASETS = [
    "sdy67",
    "sweetwater",
    "huuki_myers",
    "demixsc_retina",
    "altman_Arunachalam",
    "altman_TabulaSapiens",
    "altman_Hao",
    "finotello_Hao",
    "hoek_Hao",
    "hoek_purified_Hao",
    "linsley_purified_Hao",
    "morandini_Hao",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "2_real_bulk"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "loo_within_fold"

ORIG_ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1000.0]


def load_bf_gene_list() -> tuple[list[str], dict[str, int]]:
    """Load BulkFormer 20010 gene vocabulary."""
    bf_dir = Path(__file__).resolve().parent.parent / "weights" / "bulkformer" / "source"
    bf_dir = Path(os.environ.get("BULKFORMER_DIR", str(bf_dir)))
    gene_csv = bf_dir / "data" / "bulkformer_gene_info.csv"
    if not gene_csv.exists():
        raise FileNotFoundError(f"BulkFormer gene list not found at {gene_csv}")
    bf_genes = list(pd.read_csv(gene_csv).iloc[:, 0])
    g2i = {g.upper(): i for i, g in enumerate(bf_genes)}
    return bf_genes, g2i


def align_to_bf_vocab(
    vals: np.ndarray, bulk_columns: pd.Index, g2i: dict[str, int]
) -> tuple[np.ndarray, int]:
    """Align expression matrix to BulkFormer 20010-gene vocabulary."""
    n_samples, n_genes = vals.shape
    aligned = np.full((n_samples, len(g2i)), -10.0, dtype=np.float32)
    matched = 0
    for i, g in enumerate(bulk_columns):
        idx = g2i.get(str(g).upper())
        if idx is not None:
            aligned[:, idx] = vals[:, i]
            matched += 1
    return aligned, matched


def run_loo_within_fold_pca(
    h5_path: Path,
    gt_path: Path,
    output_dir: Path,
    seed: int = 42,
    scaler: bool = False,
) -> dict:
    """Main LOO-within-fold PCA experiment for one dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    bundle = load_data(str(h5_path), ground_truth=str(gt_path))
    bulk = bundle.bulk
    gt_df = bundle.gt
    if bulk is None or gt_df is None:
        raise ValueError("H5 and GT CSV must both be loadable")
    vals = bulk.values.astype(np.float32)
    n_samples, n_genes = vals.shape
    gt_values = gt_df.values.astype(np.float64)
    gt_columns = list(gt_df.columns)
    print(f"Loaded {n_samples} samples, {n_genes} genes, {len(gt_columns)} cell types")

    # ── Align to BulkFormer vocabulary ───────────────────────────────────
    bf_genes, g2i = load_bf_gene_list()
    aligned, matched = align_to_bf_vocab(vals, bulk.columns, g2i)
    n_bf = len(bf_genes)
    print(f"BulkFormer vocab: {n_bf} genes, aligned {matched}/{n_genes}")

    total_start = time.monotonic()

    # ── LOO — PCA fit inside each fold ───────────────────────────────────
    loo_pred = np.zeros((n_samples, len(gt_columns)))
    per_fold_n_components: list[int] = []
    per_fold_explained_var: list[float] = []

    loo_start = time.monotonic()

    for holdout in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[holdout] = False

        train_x = aligned[train_mask]  # (n-1, 20010)
        test_x  = aligned[holdout].reshape(1, -1)  # (1, 20010)

        # ── Optional StandardScaler (fit on train only) ──────────────────
        train_s = train_x
        test_s  = test_x
        if scaler:
            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler()
            train_s = ss.fit_transform(train_x)
            test_s  = ss.transform(test_x)

        # ── PCA on n-1 training samples ──────────────────────────────────
        n_components = min(train_s.shape[0], train_s.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        train_emb = pca.fit_transform(train_s)
        test_emb  = pca.transform(test_s)

        per_fold_n_components.append(int(train_emb.shape[1]))
        per_fold_explained_var.append(float(pca.explained_variance_ratio_.sum()))

        # ── RidgeCV per cell type ────────────────────────────────────────
        for j, ct in enumerate(gt_columns):
            y = gt_values[train_mask, j]
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                loo_pred[holdout, j] = 0.0
                continue
            ridge = RidgeCV(alphas=ORIG_ALPHAS, scoring="r2").fit(
                train_emb[mask], y[mask]
            )
            loo_pred[holdout, j] = float(
                np.clip(ridge.predict(test_emb)[0], 0, None)
            )

        if (holdout + 1) % 10 == 0 or holdout == 0 or holdout == n_samples - 1:
            elapsed = time.monotonic() - total_start
            print(f"  LOO fold {holdout+1}/{n_samples}  "
                  f"PCA dim={train_emb.shape[1]}  "
                  f"EV={per_fold_explained_var[-1]:.3f}  "
                  f"({elapsed:.0f}s)")

    loo_time_s = round(time.monotonic() - loo_start, 3)

    # ── Per-type metrics ─────────────────────────────────────────────────
    per_type_r: dict[str, float | None] = {}
    per_type_rmse: dict[str, float | None] = {}
    for j, ct in enumerate(gt_columns):
        mask = ~np.isnan(gt_values[:, j])
        if mask.sum() >= 2 and np.std(gt_values[mask, j]) > 1e-10:
            r = float(np.corrcoef(loo_pred[mask, j], gt_values[mask, j])[0, 1])
            per_type_r[ct] = round(r, 4) if not np.isnan(r) else None
        else:
            per_type_r[ct] = None
        if mask.sum() >= 2:
            per_type_rmse[ct] = round(
                float(np.sqrt(np.mean((loo_pred[mask, j] - gt_values[mask, j]) ** 2))), 4
            )
        else:
            per_type_rmse[ct] = None

    macro_r = float(np.nanmean([v for v in per_type_r.values() if v is not None]))

    # ── DeconBenchmark suite ────────────────────────────────────────────
    deconbench = evaluate_deconvolution(gt_values, loo_pred, gt_columns)

    # ── Save outputs ────────────────────────────────────────────────────
    pred_df = pd.DataFrame(loo_pred, index=gt_df.index, columns=gt_columns)
    pred_df.to_csv(output_dir / "proportions.csv")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(deconbench, f, indent=2)

    ridge_info = {
        "pearson_per_type": per_type_r,
        "rmse_per_type": per_type_rmse,
        "macro_avg_r": round(macro_r, 4) if not np.isnan(macro_r) else None,
        "n_total": n_samples,
        "loo_time_s": loo_time_s,
        "per_fold_n_components": per_fold_n_components,
        "per_fold_explained_var_mean": round(float(np.mean(per_fold_explained_var)), 4),
        "per_fold_explained_var_std": round(float(np.std(per_fold_explained_var)), 4),
    }
    with open(output_dir / "ridge_metrics.json", "w") as f:
        json.dump(ridge_info, f, indent=2)

    meta = {
        "experiment": "loo_within_fold_pca",
        "description": "PCA fit on n-1 training samples per LOO fold (no leakage)",
        "n_total": n_samples,
        "seed": seed,
        "aligned_genes": matched,
        "bf_vocab_size": n_bf,
        "scaler": scaler,
        "alphas": ORIG_ALPHAS,
        "total_time_s": round(time.monotonic() - total_start, 1),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    r = deconbench.get("pearson_mean", float("nan"))
    print(f"\nDone.  macro_avg r = {macro_r:.4f}  "
          f"(DeconBench pearson_mean = {r:.4f})  "
          f"total {meta['total_time_s']:.0f}s")
    print(f"Output: {output_dir}")

    return deconbench


def run_all() -> None:
    """Run across all 12 real-bulk datasets."""
    bf_genes, _ = load_bf_gene_list()
    print(f"BulkFormer vocabulary loaded: {len(bf_genes)} genes\n")

    results: dict[str, dict] = {}
    for ds in REAL_BULK_DATASETS:
        h5 = DATA_DIR / f"{ds}.h5"
        gt = DATA_DIR / f"{ds}_gt.csv"
        if not h5.exists():
            print(f"SKIP {ds}: {h5} not found")
            continue
        if not gt.exists():
            print(f"SKIP {ds}: {gt} not found")
            continue

        out = OUTPUT_DIR / ds
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")
        try:
            metrics = run_loo_within_fold_pca(h5, gt, out)
            results[ds] = {
                "pearson_mean": metrics.get("pearson_mean"),
                "spearman_mean": metrics.get("spearman_mean"),
                "rmse_mean": metrics.get("rmse_mean"),
                "mAD_mean": metrics.get("mAD_mean"),
            }
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            results[ds] = {"error": str(e)}

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary — LOO-within-fold PCA")
    print(f"{'='*60}")
    rows = []
    for ds in REAL_BULK_DATASETS:
        r = results.get(ds, {})
        if "error" in r:
            rows.append(f"  {ds:25s}  ERROR: {r['error']}")
        else:
            pm = r.get("pearson_mean", float("nan"))
            sm = r.get("spearman_mean", float("nan"))
            rm = r.get("rmse_mean", float("nan"))
            rows.append(
                f"  {ds:25s}  pearson={pm:.4f}  spearman={sm:.4f}  rmse={rm:.4f}"
            )
    for line in rows:
        print(line)

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA within each LOO fold — no-leakage baseline"
    )
    parser.add_argument("--h5", help="Path to DeconBenchmark H5")
    parser.add_argument("--ground-truth", help="Path to GT CSV")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scaler", action="store_true",
                        help="Apply StandardScaler before PCA (default: no scaler)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 12 real-bulk datasets")
    args = parser.parse_args()

    if args.all:
        run_all()
        return

    if not args.h5 or not args.ground_truth:
        parser.error("Specify --h5 and --ground-truth, or use --all")

    h5_path = Path(args.h5)
    gt_path = Path(args.ground_truth)

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        dataset_name = h5_path.stem.rstrip("." + "".join(h5_path.suffixes))
        out = OUTPUT_DIR / dataset_name

    run_loo_within_fold_pca(h5_path, gt_path, out, seed=args.seed, scaler=args.scaler)


if __name__ == "__main__":
    main()
