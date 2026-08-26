#!/usr/bin/env python3
"""Recover test-split MAE/SCorr/CCorr for Mode B methods.

Problem:
  evaluate.py --batch overwrote metrics.json with all-sample post-hoc
  evaluation, inflating metrics.  The old fix_ridge_metrics.py restored
  Pearson/RMSE from ridge_metrics.json but couldn't restore MAE/SCorr/CCorr
  because ridge_metrics.json never stored them.

Solution:
  For methods using evaluate_real_bulk_ridge() (backbone/ridge, bulkformer
  sub-variants), proportions.csv was generated from train-only models.
  We reproduce the train/test split (via seed=42) and compute test-split
  MAE/SCorr/CCorr from the existing predictions + GT — no re-encoding needed.

  For pca_ridge (which refits on all data for proportions.csv), recovery is
  impossible — it must be re-run via methods/pca_ridge/run.py.

Usage:
    python scripts/recover_ridge_metrics.py                           # fix all
    python scripts/recover_ridge_metrics.py --results-dir results/2_realbulk
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.metrics import evaluate_deconvolution


def _load_gt_csv(gt_csv: str) -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    first_col = df.iloc[:, 0]
    if first_col.dtype in (object, str) or first_col.dtype.name == "object":
        df = df.set_index(df.columns[0])
    return df


def _reproduce_test_indices(n_bulk: int, ridge_info: dict) -> np.ndarray:
    """Reproduce test split indices from ridge_metrics.json info + seed=42."""
    split_label = ridge_info.get("split", "random_80_20")
    if split_label.startswith("fixed_"):
        _, train_val_n, test_n = split_label.split("_")
        train_val_n, test_n = int(train_val_n), int(test_n)
        return np.arange(train_val_n, train_val_n + test_n)
    else:
        _, test_idx = train_test_split(
            np.arange(n_bulk), test_size=0.2, random_state=42
        )
        return test_idx


def _find_gt_csv(method_dir: Path) -> Path | None:
    """Walk up from method_dir to find dataset name and GT CSV."""
    try:
        md = method_dir.resolve()
        rel = md.relative_to((PROJECT_ROOT / "results" / "2_realbulk").resolve())
        ds_name = rel.parts[0]
    except ValueError:
        return None
    gt_csv = PROJECT_ROOT / "data" / "2_real_bulk" / f"{ds_name}_gt.csv"
    return gt_csv if gt_csv.exists() else None


def recover_one(method_dir: Path) -> bool:
    """Recover test-split metrics for one method directory."""
    rm_path = method_dir / "ridge_metrics.json"
    mf_path = method_dir / "metrics.json"
    prop_path = method_dir / "proportions.csv"

    if not all(p.exists() for p in [rm_path, mf_path, prop_path]):
        return False

    with open(rm_path) as f:
        ridge_info = json.load(f)

    # Skip if already has MAE
    try:
        cur = json.load(open(mf_path))
        if cur.get("mae_overall") is not None:
            return False
    except (OSError, json.JSONDecodeError):
        return False

    prop_df = pd.read_csv(str(prop_path), index_col=0)
    gt_csv = _find_gt_csv(method_dir)
    if gt_csv is None:
        return False
    gt = _load_gt_csv(str(gt_csv))

    n_bulk = min(len(prop_df), len(gt))
    test_idx = _reproduce_test_indices(n_bulk, ridge_info)

    common = [c for c in prop_df.columns if c in gt.columns]
    if not common:
        return False

    test_pred = prop_df.iloc[test_idx][common].values.astype(np.float64)
    test_gt = gt.iloc[test_idx][common].values.astype(np.float64)

    # Verify consistency with ridge_metrics.json
    from scipy.stats import pearsonr
    ref_r = np.nanmean([v for v in ridge_info.get("pearson_per_type", {}).values() if v is not None])
    recovered_r = np.nanmean([pearsonr(test_pred[:, j], test_gt[:, j])[0] for j in range(len(common))])

    if abs(recovered_r - ref_r) > 0.01:
        return False  # all-data model, can't recover

    # Clip negative, renormalize
    pred = np.maximum(test_pred, 0.0)
    row_sum = pred.sum(axis=1, keepdims=True)
    pred = pred / np.maximum(row_sum, 1e-10)

    deconbench = evaluate_deconvolution(test_gt, pred, common)

    # Update metrics.json
    with open(mf_path) as f:
        existing = json.load(f)

    existing.update({
        "pearson_mean": round(ref_r, 4),
        "rmse_overall": deconbench.get("rmse_mean_per_type", existing.get("rmse_overall")),
        "mae_overall": deconbench.get("mae_overall"),
        "scorr_mean": deconbench.get("scorr_mean"),
        "ccorr_mean": deconbench.get("ccorr_mean"),
    })
    existing.pop("_note", None)

    with open(mf_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"  {method_dir.parent.name}/{method_dir.name}: MAE={deconbench.get('mae_overall', '?'):.4f}  "
          f"SCorr={deconbench.get('scorr_mean', '?'):.4f}  "
          f"CCorr={deconbench.get('ccorr_mean', '?'):.4f}")
    return True


def recover_all(results_dir: str) -> int:
    base = Path(results_dir)
    if not base.is_dir():
        print(f"Error: {base} not found", file=sys.stderr)
        return 0

    recovered = 0
    skipped = 0

    for rm_path in sorted(base.rglob("ridge_metrics.json")):
        method_dir = rm_path.parent
        if recover_one(method_dir):
            recovered += 1
        else:
            skipped += 1

    print(f"\nRecovered: {recovered}, Skipped: {skipped}")
    return recovered


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Recover test-split MAE/SCorr/CCorr for Mode B methods")
    parser.add_argument("--results-dir", default="results/2_realbulk")
    args = parser.parse_args()
    recover_all(args.results_dir)

    print("\nNOTE: pca_ridge cannot be recovered from proportions.csv (all-data model).")
    print("      Re-run: python methods/pca_ridge/run.py")


if __name__ == "__main__":
    main()
