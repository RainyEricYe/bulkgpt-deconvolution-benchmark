#!/usr/bin/env python3
"""Unified evaluation CLI for deconvolution predictions.

Usage:
    # Evaluate predictions against ground truth
    python scripts/evaluate.py \\
        --pred results/proportions.csv \\
        --gt data/gt.csv \\
        --output metrics.json

    # Batch-evaluate all results under a directory
    python scripts/evaluate.py \\
        --batch results/2_realbulk \\
        --data-dir data/2_real_bulk \\
        --gt-suffix _gt.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_scripts = Path(__file__).resolve().parent
_project = _scripts.parent
if str(_project) not in sys.path:
    sys.path.insert(0, str(_project))

from core.metrics import evaluate_deconvolution


def _read_gt_csv(gt_csv: str) -> pd.DataFrame:
    """Read ground truth CSV, handling missing sample_id column."""
    df = pd.read_csv(gt_csv)
    first_col = df.columns[0]
    if str(first_col).startswith("Unnamed:"):
        # Saved DataFrame index column (e.g. from df.to_csv() without index=False)
        df = df.set_index(first_col)
    elif df.iloc[:, 0].dtype in (object, str) or df.iloc[:, 0].dtype.name == "object":
        df = df.set_index(first_col)
    elif first_col in ("sample_id", "sample", "id", "sampleID"):
        df = df.set_index(first_col)
    return df


def _is_mode_b(method_label: str) -> bool:
    """Check if a method uses Mode B evaluation (train/test split, not full-sample).

    Mirrors the logic in ``scripts/summarize_results.py:_is_mode_b()``.
    Mode B methods should never have their metrics.json overwritten by
    all-sample post-hoc evaluation.
    """
    if method_label in ("pca_ridge", "pca_ridge_loo",
                         "bulkformer_random", "bulkformer_random_mean_pool",
                         "bulkformer_mean_pool", "scgpt_lora"):
        return True
    if method_label.startswith("bulkformer/"):
        return True
    if method_label.endswith("_loo"):
        return True
    for backbone in ("bulkformer", "geneformer", "scfoundation", "scgpt", "stack", "transcriptformer"):
        for suffix in ("/ridge", "/ridge_scaler", "/realbulk",
                        "/ridge_loo", "/ridge_scaler_loo",
                        "/pca_ridge", "/pca_ridge_loo"):
            if method_label == f"{backbone}{suffix}":
                return True
    return False


def evaluate_file(pred_csv: str, gt_csv: str, output: str | None = None) -> dict:
    """Evaluate a single predictions CSV against ground truth."""
    # Skip if this is a Mode B method (train/test split) — its metrics.json
    # was generated from the held-out test set.  Overwriting with all-sample
    # post-hoc evaluation would inflate metrics (pca_ridge was 0.66 → 1.0).
    output_dir = Path(output).parent if output else Path(pred_csv).parent
    has_ridge_metrics = output_dir.joinpath("ridge_metrics.json").exists()
    if has_ridge_metrics or _is_mode_b(output_dir.name):
        indicator = "ridge_metrics.json" if has_ridge_metrics else output_dir.name
        print(f"  Skipping {output_dir.name} (Mode B: {indicator})")
        try:
            return json.load(open(output_dir / "metrics.json"))
        except (OSError, json.JSONDecodeError):
            pass

    pred_df = pd.read_csv(pred_csv, index_col=0)
    gt_df = _read_gt_csv(gt_csv)

    common = [c for c in pred_df.columns if c in gt_df.columns]

    # Cell-type hierarchy merge (from configs/celltype_merges.yaml):
    # e.g. B cells + T cells + NK cells → Lymphocytes.
    _merge_path = Path(__file__).resolve().parent.parent / "configs" / "celltype_merges.yaml"
    if _merge_path.exists():
        import yaml
        with open(_merge_path) as _f:
            _merge_map = yaml.safe_load(_f) or {}
        for gt_col in gt_df.columns:
            if gt_col in common:
                continue
            subtypes = _merge_map.get(gt_col, [])
            gt_lower_set = {s.lower() for s in subtypes}
            matching = [
                p for p in pred_df.columns
                if p.lower() in gt_lower_set and p not in common
            ]
            if matching:
                common.append(gt_col)
                print(f"  Merged {matching} → {gt_col} (from celltype_merges.yaml)")
                pred_df[gt_col] = pred_df[matching].sum(axis=1)

    # Auto-merge: if GT type name contains multiple pred type names, merge them
    # (e.g. GT 'OligoOPC' = pred 'Oligo' + 'OPC')
    merged = {}
    for gt_col in gt_df.columns:
        if gt_col in common:
            continue
        gt_lower = gt_col.lower()
        matching_pred = [p for p in pred_df.columns
                         if p.lower() in gt_lower and p not in common and p not in sum(merged.values(), [])]
        if len(matching_pred) >= 2:
            merged[gt_col] = matching_pred
            common.append(gt_col)
    if merged:
        print(f"  Auto-merged pred types to match GT: {merged}")
        for gt_col, pred_cols in merged.items():
            pred_df[gt_col] = pred_df[pred_cols].sum(axis=1)

    # Hungarian matching for unnamed components — handles generic column
    # names such as IC1..ICN (DeconICA), PC1..PCN (ReFACTor), CT1..CTN,
    # cellType1..cellTypeN, cell.type.1..cell.type.N, "Cell type 1..N".
    # When columns are generic and no common names found, find the optimal
    # 1:1 assignment via minimising -|Pearson r|.
    #
    # Three scenarios:
    #   1. pred > GT — truncate to first GT-count columns (descending
    #      variance order; the tail carries least signal).
    #   2. pred == GT — direct 1:1 Hungarian matching.
    #   3. pred < GT — pad with zero columns so Hungarian can match the
    #      available components against the best GT columns; unmatched
    #      GT types score zero.
    GENERIC_PATTERNS = [
        r'^IC\d+$',           # DeconICA: IC1..ICN
        r'^PC\d+$',           # ReFACTor: PC1..PCN
        r'^cellType\d+$',     # BayCount: cellType1..cellTypeN
        r'^CT\d+$',           # DeCompress/DeBCAM: CT1..CTN
        r'^cell\.type\.\d+$', # Deconf: cell.type.1..cell.type.N
        r'^Cell type \d+$',   # LinSeed: "Cell type 1..Cell type N"
    ]
    if len(common) == 0:
        import re
        col_strs = [str(c) for c in pred_df.columns]
        is_generic = any(
            all(re.match(p, c) for c in col_strs)
            for p in GENERIC_PATTERNS
        )
        if is_generic:
            from scipy.optimize import linear_sum_assignment

            n_types = len(gt_df.columns)
            n_pred = len(pred_df.columns)

            # If pred < GT, pad with zero columns so Hungarian can find
            # best assignment for the available components.
            if n_pred < n_types:
                for k in range(n_types - n_pred):
                    pred_df[f"_pad_{k}"] = 0.0
                print(f"  Padding {n_types - n_pred} zero columns "
                      f"(pred {n_pred} < GT {n_types})")

            # If pred has excess components, keep the first GT-count ones
            if len(pred_df.columns) > n_types:
                print(f"  Truncating {len(pred_df.columns)} components to "
                      f"{n_types} (matching GT type count)")
                pred_df = pred_df.iloc[:, :n_types]

            # Align samples positionally first
            shared = pred_df.index.intersection(gt_df.index)
            if len(shared) == 0 and len(pred_df) == len(gt_df):
                shared = gt_df.index  # positional fallback
                # Reindex pred_df to match shared (sample IDs differ, but
                # positionally aligned — e.g. deconica outputs sample_0..N
                # while GT has meaningful sample names).
                pred_df.index = gt_df.index[:len(pred_df)]

            cost = np.full((n_types, n_types), 1.0)
            for i, pcol in enumerate(pred_df.columns[:n_types]):
                pstd = pred_df.loc[shared, pcol].std()
                if pstd == 0:  # Padded zero column or constant — no signal
                    continue
                for j, gt_col in enumerate(gt_df.columns):
                    r = np.corrcoef(pred_df.loc[shared, pcol],
                                    gt_df.loc[shared, gt_col])[0, 1]
                    if not np.isfinite(r):
                        r = 0.0
                    cost[i, j] = 1.0 - abs(r)
            row_ind, col_ind = linear_sum_assignment(cost)
            rename_map = {}
            for pi, gj in zip(row_ind, col_ind):
                r_val = 1.0 - cost[pi, gj]
                rename_map[pred_df.columns[pi]] = gt_df.columns[gj]
                print(f"  Matched {pred_df.columns[pi]} → {gt_df.columns[gj]} (r={r_val:.3f})")
            pred_df.rename(columns=rename_map, inplace=True)
            # Drop padding columns
            pred_df.drop(columns=[c for c in pred_df.columns if c.startswith("_pad_")],
                         inplace=True, errors="ignore")
            common = [c for c in pred_df.columns if c in gt_df.columns]

    if len(common) < 1:
        msg = f"No common cell types between prediction and GT\n  Pred: {list(pred_df.columns)}\n  GT:   {list(gt_df.columns)}"
        raise ValueError(msg)

    pred = pred_df[common].values.astype(np.float64)
    true = gt_df[common].values.astype(np.float64)

    # Remove rows with NaN in GT (samples without ground truth)
    valid = ~np.isnan(true).any(axis=1)
    if not valid.all():
        print(f"  Dropped {(~valid).sum()} samples with NaN GT")
        pred = pred[valid]
        true = true[valid]

    pred = np.maximum(pred, 0.0)
    row_sum = pred.sum(axis=1, keepdims=True)
    pred = pred / np.maximum(row_sum, 1e-10)

    missing_gt = [c for c in gt_df.columns if c not in pred_df.columns]
    extra_pred = [c for c in pred_df.columns if c not in gt_df.columns]
    n_common = len(common)
    n_total = len(gt_df.columns)
    print(f"  Common types: {n_common}/{n_total}  {common}")
    if missing_gt:
        print(f"  Missing from pred (in GT only): {missing_gt}")
    if extra_pred:
        print(f"  Extra in pred (not in GT): {extra_pred}")

    metrics = evaluate_deconvolution(true, pred, common)

    if output:
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved -> {output}")

    return metrics


def batch_eval(results_dir: str, data_dir: str, gt_suffix: str = "_gt.csv"):
    """Evaluate all proportions.csv under results_dir."""
    results_path = Path(results_dir)
    data_path = Path(data_dir)

    all_items = sorted(results_path.rglob("proportions.csv"))
    if not all_items:
        print(f"No proportions.csv found under {results_dir}")
        return

    for prop_csv in all_items:
        rel = prop_csv.relative_to(results_path)
        ds = rel.parts[0]
        gt_csv = data_path / f"{ds}{gt_suffix}"
        if not gt_csv.exists():
            print(f"SKIP {rel.parent}: no GT at {gt_csv}")
            continue

        output = prop_csv.parent / "metrics.json"
        print(f"\n{'='*60}")
        print(f"  {rel.parent}")
        try:
            evaluate_file(str(prop_csv), str(gt_csv), str(output))
        except Exception as e:
            print(f"  ERROR: {e}")


def print_summary(metrics: dict) -> None:
    def fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return "  nan"
        return f"{v:>7.4f}"
    print(f"  MAE={fmt(metrics.get('mae_overall',''))}  "
          f"SCorr={fmt(metrics.get('scorr_mean',''))}  "
          f"CCorr={fmt(metrics.get('ccorr_mean',''))}  "
          f"Pearson={fmt(metrics.get('pearson_mean',''))}  "
          f"RMSE={fmt(metrics.get('rmse_mean_per_type',''))}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for deconvolution predictions")
    parser.add_argument("--pred", help="Predictions CSV (samples x cell types)")
    parser.add_argument("--gt", help="Ground truth CSV (samples x cell types)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--batch", help="Batch-evaluate all proportions.csv under this directory")
    parser.add_argument("--data-dir",
                        default=str(Path(__file__).resolve().parent.parent / "data" / "2_real_bulk"),
                        help="Data directory for batch mode")
    parser.add_argument("--gt-suffix", default="_gt.csv",
                        help="GT file suffix for batch mode")
    args = parser.parse_args()

    if args.batch:
        batch_eval(args.batch, args.data_dir, args.gt_suffix)
    elif args.pred and args.gt:
        m = evaluate_file(args.pred, args.gt, args.output)
        print_summary(m)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
