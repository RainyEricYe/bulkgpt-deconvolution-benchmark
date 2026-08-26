#!/usr/bin/env python3
"""BulkFormer random-weight + global_expr_proj + RidgeCV (control experiment).

Builds the BulkFormer-147M model **without** loading the pretrained
checkpoint, then uses the ``global_expr_proj`` MLP head to encode bulk
expression into sample-level embeddings.  A train/test split RidgeCV
evaluates per-cell-type deconvolution performance.

The same experiments (pretrained vs random) are compared in
``eval_check_pretraining.py`` from the original BulkFormer repo.

Usage::

    python methods/bulkformer/random/run.py \\
        --h5 data/2_real_bulk/sdy67.h5 \\
        --ground-truth data/2_real_bulk/sdy67_gt.csv \\
        --output-dir results/2_realbulk/sdy67/bulkformer/random
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_to_publish = Path(__file__).resolve().parent.parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

import h5py
from core.deconv.frozen_eval import ResourceTracker, evaluate_real_bulk_ridge
from core.deconv.loo import run_loo_ridge, save_loo_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BulkFormer random-weight + global_expr_proj + RidgeCV",
    )
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mode", default="predict", help=argparse.SUPPRESS)
    parser.add_argument("--h5", required=True, help="Path to DeconBenchmark H5")
    parser.add_argument("--ground-truth", required=True, help="Path to GT CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scaler", action="store_true",
                        help="Use StandardScaler before RidgeCV")
    parser.add_argument("--loo", action="store_true",
                        help="Use LOO RidgeCV instead of train/test split")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    dataset_name = Path(args.h5).stem  # derive from H5, not out_dir path
    if args.scaler and not out_dir.name.endswith("_scaler"):
        out_dir = out_dir.parent / f"{out_dir.name}_scaler"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load bulk + GT directly (no scRNA needed for LOO/split eval) ──
    with h5py.File(args.h5, "r") as f:
        bulk_raw = f["bulk/values"][:]
        bulk_rn = [x.decode() for x in f["bulk/rownames"][:]]
        bulk_cn = [x.decode() for x in f["bulk/colnames"][:]]
    bulk = pd.DataFrame(bulk_raw, index=bulk_cn, columns=bulk_rn)
    gt_df = pd.read_csv(args.ground_truth, index_col=0)
    vals = bulk.values.astype(np.float32)
    genes = list(bulk.columns)
    samples = list(bulk.index)
    print(
        f"Loaded {len(samples)} samples, {len(genes)} genes, "
        f"{len(gt_df.columns)} cell types",
    )

    with ResourceTracker() as rt:
        # ── Build random-weight encoder ──────────────────────────────────
        t0 = time.monotonic()
        from methods.bulkformer.model import BulkFormerEncoder

        encoder = BulkFormerEncoder(pretrained=False)
        emb = encoder.encode(vals, genes, samples, pooling="global_proj")
        rt.end_encode()
        print(f"Embedding: {emb.shape} in {time.monotonic()-t0:.1f}s")

        # ── LOO or split RidgeCV evaluation ────────────────────────────────
        if args.loo:
            out_dir = out_dir.parent / f"{out_dir.name}_loo"
            out_dir.mkdir(parents=True, exist_ok=True)
            result = run_loo_ridge(emb, gt_df, use_scaler=args.scaler)
            rt.end_ridge()
            save_loo_results(out_dir, result, meta_extra={
                **rt.to_dict(backbone="bulkformer/random", dataset=dataset_name),
                "seed": args.seed,
                "loo": True,
            })
        else:
            result = evaluate_real_bulk_ridge(emb, gt_df, dataset_name,
                                              seed=args.seed, use_scaler=args.scaler)
            rt.end_ridge()
            # ── Save outputs ───────────────────────────────────────────────
            result["full_predictions_df"].to_csv(out_dir / "proportions.csv")

            with open(out_dir / "metrics.json", "w") as f:
                json.dump(result["deconbench"], f, indent=2)
            with open(out_dir / "ridge_metrics.json", "w") as f:
                json.dump(result["ridge_specific"], f, indent=2)

            meta = {
                **result["metadata"],
                **rt.to_dict(backbone="bulkformer/random", dataset=dataset_name),
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

    r = result["deconbench"].get("pearson_mean", float("nan"))
    print(f"\nDone.  Pearson r = {r:.4f}  [{rt.wall_time_s}s]")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
