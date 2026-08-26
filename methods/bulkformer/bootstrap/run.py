#!/usr/bin/env python3
"""BulkFormer + global_proj + bootstrap RidgeCV ensemble.

Computes BulkFormer embeddings via the fast ``global_expr_proj`` MLP head
(pretrained checkpoint) and evaluates deconvolution performance via a
multi-iteration bootstrap RidgeCV ensemble per cell type.

After 50 (or ``--n-ensemble``) bootstrap resampling rounds, predictions
are averaged for the final estimate, providing more robust evaluation
than a single RidgeCV split.

Adapted from the original ``eval_bootstrap.py`` in the BulkFormer repo.

Usage::

    python methods/bulkformer/bootstrap/run.py \\
        --h5 data/2_real_bulk/sdy67.h5 \\
        --ground-truth data/2_real_bulk/sdy67_gt.csv \\
        --output-dir results/2_realbulk/sdy67/bulkformer/bootstrap \\
        --n-ensemble 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_to_publish = Path(__file__).resolve().parent.parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.data_loader import load_data
from core.deconv.frozen_eval import ResourceTracker
from core.deconv.utils import renormalize_props
from methods.bulkformer.bootstrap_utils import DEFAULT_ALPHAS, bootstrap_ridge


def _split_indices(
    n_bulk: int, dataset_name: str, seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Train/test split matching evaluate_real_bulk_ridge()."""
    from sklearn.model_selection import train_test_split

    if dataset_name == "sdy67":
        train_n = 200  # 150 train + 50 val
        test_n = min(50, n_bulk - 200)
        train_idx = np.arange(train_n)
        test_idx = np.arange(train_n, train_n + test_n)
        split_label = "fixed_200_50"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(n_bulk), test_size=0.2, random_state=seed,
        )
        split_label = "random_80_20"
    return train_idx, test_idx, split_label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BulkFormer + global_proj + bootstrap RidgeCV ensemble",
    )
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mode", default="predict", help=argparse.SUPPRESS)
    parser.add_argument("--h5", required=True, help="Path to DeconBenchmark H5")
    parser.add_argument("--ground-truth", required=True, help="Path to GT CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-ensemble", type=int, default=50, help="Bootstrap iterations",
    )
    parser.add_argument("--scaler", action="store_true",
                        help="Use StandardScaler before RidgeCV")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    dataset_name = Path(args.h5).stem
    if args.scaler and not out_dir.name.endswith("_scaler"):
        out_dir = out_dir.parent / f"{out_dir.name}_scaler"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data via core.data_loader (canonical (n_samples, n_genes)) ──
    bundle = load_data(args.h5, ground_truth=args.ground_truth)
    bulk = bundle.bulk
    gt_df = bundle.gt
    if bulk is None or gt_df is None:
        raise ValueError("H5 must contain bulk/values and GT CSV must be loadable")
    vals = bulk.values.astype(np.float32)
    genes = list(bulk.columns)
    samples = list(bulk.index)
    n_bulk = len(gt_df)
    n_types = len(gt_df.columns)
    print(
        f"Loaded {n_bulk} samples, {len(genes)} genes, {n_types} cell types",
    )

    with ResourceTracker() as rt:
        # ── Encode ──────────────────────────────────────────────────────
        t0 = time.monotonic()
        from methods.bulkformer.model import encode_bulkformer

        emb = encode_bulkformer(vals, genes, samples, pooling="global_proj")
        rt.end_encode()
        print(f"Embedding: {emb.shape} in {time.monotonic()-t0:.1f}s")

        # ── Split (same logic as evaluate_real_bulk_ridge) ──────────────
        train_idx, test_idx, split_label = _split_indices(
            n_bulk, dataset_name, args.seed,
        )
        train_emb, test_emb = emb[train_idx], emb[test_idx]
        gt_values = gt_df.values.astype(np.float64)
        train_gt = gt_values[train_idx]
        test_gt = gt_values[test_idx]

        # ── Optional scaler + Bootstrap RidgeCV ensemble ──
        if args.scaler:
            from sklearn.preprocessing import StandardScaler
            _scaler = StandardScaler()
            train_s = _scaler.fit_transform(train_emb)
            test_s = _scaler.transform(test_emb)
            full_x = _scaler.transform(emb)
        else:
            train_s, test_s = train_emb, test_emb
            full_x = emb
        t0 = time.monotonic()
        pred_mean, pred_std, details = bootstrap_ridge(
            train_s,
            pd.DataFrame(train_gt, columns=gt_df.columns),
            test_s,
            n_ensemble=args.n_ensemble,
            seed=args.seed,
            alphas=DEFAULT_ALPHAS,
        )
        ridge_time = time.monotonic() - t0
        print(
            f"Bootstrap {args.n_ensemble}× done in {ridge_time:.1f}s, "
            f"pred shape = {pred_mean.shape}",
        )
        rt.end_ridge()

    # ── Clip + renormalize ─────────────────────────────────────────────
    pred_mean = renormalize_props(pred_mean, zero_fill="zero")
    pred_std_clipped = np.maximum(pred_std, 0.0)

    # ── DeconBenchmark evaluation ──────────────────────────────────────
    _project = Path(__file__).resolve().parent.parent.parent.parent
    if str(_project) not in sys.path:
        sys.path.insert(0, str(_project))
    from core.metrics import evaluate_deconvolution

    deconbench = evaluate_deconvolution(test_gt, pred_mean, list(gt_df.columns))

    # Per-type metrics from the ensemble mean
    per_type_r: dict[str, float | None] = {}
    per_type_rmse: dict[str, float | None] = {}
    for j, ct in enumerate(gt_df.columns):
        mask = ~np.isnan(test_gt[:, j])
        if mask.sum() >= 2 and np.std(test_gt[mask, j]) > 1e-10:
            r = float(np.corrcoef(pred_mean[mask, j], test_gt[mask, j])[0, 1])
            per_type_r[ct] = round(r, 4) if not np.isnan(r) else None
        else:
            per_type_r[ct] = None
        if mask.sum() >= 2:
            per_type_rmse[ct] = round(
                float(np.sqrt(np.mean((pred_mean[mask, j] - test_gt[mask, j]) ** 2))),
                4,
            )
        else:
            per_type_rmse[ct] = None

    # ── Save outputs ─────────────────────────────────────────────────────
    # proportions.csv (full-sample predictions)
    full_pred = np.zeros((n_bulk, n_types))
    for j, ct in enumerate(gt_df.columns):
        y = gt_values[:, j]
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            continue
        from sklearn.linear_model import RidgeCV

        ridge = RidgeCV(alphas=DEFAULT_ALPHAS).fit(full_x[mask], y[mask])
        full_pred[:, j] = ridge.predict(full_x)
    full_pred = renormalize_props(full_pred, zero_fill="zero")
    full_pred_df = pd.DataFrame(full_pred, index=gt_df.index, columns=gt_df.columns)
    full_pred_df.to_csv(out_dir / "proportions.csv")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(deconbench, f, indent=2)

    ridge_info = {
        "n_ensemble": args.n_ensemble,
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "split": split_label,
        "ridge_time_s": round(ridge_time, 3),
        "pearson_per_type": per_type_r,
        "rmse_per_type": per_type_rmse,
    }
    with open(out_dir / "ridge_metrics.json", "w") as f:
        json.dump(ridge_info, f, indent=2)

    meta = {
        "embedding_dim": int(emb.shape[1]),
        "n_ensemble": args.n_ensemble,
        "split": split_label,
        "seed": args.seed,
        **rt.to_dict(backbone="bulkformer/bootstrap", dataset=dataset_name),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    r = deconbench.get("pearson_mean", float("nan"))
    print(f"\nDone.  Pearson r = {r:.4f}  [{rt.wall_time_s}s]")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
