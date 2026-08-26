#!/usr/bin/env python3
"""Re-evaluate Huuki-Myers predictions, merging Oligo+OPC -> OligoOPC.

Usage:
    python scripts/eval_huuki_real_bulk.py [--results-dir results/architecture_search]

This finds all huuki_myers/proportions.csv under results-dir, merges
prediction columns (Oligo + OPC -> OligoOPC), evaluates against the Huuki
ground-truth CSV, and writes updated metrics.json alongside each CSV.
"""
from __future__ import annotations

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

HERE = _project
GT_CSV = HERE / "data" / "2_real_bulk" / "huuki_myers_gt.csv"
MERGE_MAP = {"OligoOPC": ["Oligo", "OPC"]}


def evaluate_huuki(pred_csv: Path, gt_csv: Path, output: Path | None = None) -> dict:
    """Evaluate Huuki predictions, auto-merging Oligo+OPC -> OligoOPC."""
    pred_df = pd.read_csv(pred_csv, index_col=0)
    gt_df = pd.read_csv(gt_csv, index_col=0)

    # Merge prediction columns to match GT
    for merged_name, components in MERGE_MAP.items():
        existing = [c for c in components if c in pred_df.columns]
        if merged_name in gt_df.columns and len(existing) == len(components):
            pred_df[merged_name] = pred_df[components].sum(axis=1)
            pred_df = pred_df.drop(columns=components)
            print(f"  Merged {components} -> {merged_name}")

    # Common cell types (now including the merged one)
    common = [c for c in pred_df.columns if c in gt_df.columns]
    n_common = len(common)
    n_total = len(gt_df.columns)
    print(f"  Common types: {n_common}/{n_total}  {common}")

    if n_common < 1:
        print(f"  WARNING: no common types. Pred cols={list(pred_df.columns)}, GT cols={list(gt_df.columns)}")
        return {}

    # Align samples by index intersection
    shared_idx = pred_df.index.intersection(gt_df.index)
    if len(shared_idx) > 0:
        pred = pred_df.loc[shared_idx, common].values.astype(np.float64)
        true = gt_df.loc[shared_idx, common].values.astype(np.float64)
        print(f"  Aligned {len(shared_idx)} samples by index")
    else:
        pred = pred_df[common].values.astype(np.float64)
        true = gt_df[common].values.astype(np.float64)
        print(f"  No index overlap, using positional alignment ({len(pred)} samples)")

    # Drop rows where GT has NaN in any common type
    valid_mask = ~np.isnan(true).any(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} samples with NaN GT")
        pred = pred[valid_mask]
        true = true[valid_mask]

    if len(pred) < 1:
        print(f"  WARNING: no valid samples after NaN removal")
        return {}

    pred = pred / pred.sum(axis=1, keepdims=True)
    metrics = evaluate_deconvolution(true, pred, common)

    if output:
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved -> {output}")

    return metrics


def main(results_dir: str = "results/architecture_search"):
    results_path = HERE / results_dir
    gt_csv = GT_CSV

    if not gt_csv.exists():
        print(f"ERROR: GT not found at {gt_csv}")
        sys.exit(1)

    all_items = sorted(results_path.rglob("proportions.csv"))
    huuki_items = [p for p in all_items if "huuki" in str(p).lower()]

    if not huuki_items:
        print(f"No huuki proportions.csv found under {results_path}")
        return

    print(f"Found {len(huuki_items)} huuki prediction files\n")

    for prop_csv in huuki_items:
        rel = prop_csv.relative_to(HERE)
        output = prop_csv.parent / "metrics.json"
        print(f"[{rel.parent}]")
        try:
            evaluate_huuki(prop_csv, gt_csv, output)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/architecture_search"
    main(results_dir)
