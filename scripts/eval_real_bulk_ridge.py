#!/usr/bin/env python3
"""Unified real-bulk RidgeCV evaluation across all frozen backbones.

Loads DeconBenchmark-format H5 + GT CSV → encodes bulk through backbone
→ evaluates with RidgeCV → saves to_publish-standard 4-file output.

Usage:
    python scripts/eval_real_bulk_ridge.py --backbone stack --dataset sdy67
    python scripts/eval_real_bulk_ridge.py --backbone all --dataset all
    python scripts/eval_real_bulk_ridge.py --backbone bulkformer --dataset sdy67 --seed 123
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.deconv.frozen_eval import (
    ENCODE_FN,
    ResourceTracker,
    evaluate_real_bulk_ridge,
)

DATA_DIR = PROJECT_ROOT / "data" / "2_real_bulk"
RESULTS_DIR = PROJECT_ROOT / "results" / "2_realbulk"

DATASETS = {
    "sdy67", "sweetwater", "huuki_myers", "demixsc_retina",
    "altman_Arunachalam",
}


def _load_bulk_h5(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load bulk expression from a standardized H5. Returns (vals, genes, samples).

    Fixed H5 convention — do NOT auto-detect orientation:
      bulk/values    (n_samples, n_genes)
      bulk/rownames   gene symbols (length n_genes)
      bulk/colnames   sample IDs    (length n_samples)
    """
    with h5py.File(str(path), "r") as f:
        vals = f["bulk/values"][:]
        genes = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in f["bulk/rownames"][:]]
        samples = [x.decode() if isinstance(x, bytes) else str(x)
                   for x in f["bulk/colnames"][:]]
    return vals.astype(np.float32), genes, samples


def _read_gt_csv(path: Path) -> pd.DataFrame:
    """Read ground truth CSV, handling sample_id index column."""
    df = pd.read_csv(path)
    first_col = df.iloc[:, 0]
    if first_col.dtype in (object, str) or first_col.dtype.name == "object":
        df = df.set_index(df.columns[0])
    return df


def _save_results(
    out_dir: Path, result: dict, rt: ResourceTracker,
    backbone: str, dataset: str, seed: int,
) -> None:
    """Write 4-file output."""
    out_dir.mkdir(parents=True, exist_ok=True)

    result["full_predictions_df"].to_csv(out_dir / "proportions.csv")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result["deconbench"], f, indent=2, default=str)

    with open(out_dir / "ridge_metrics.json", "w") as f:
        json.dump(result["ridge_specific"], f, indent=2, default=str)

    meta = {
        **result["metadata"],
        **rt.to_dict(backbone=backbone, dataset=dataset, seed=seed),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-bulk RidgeCV evaluation across frozen backbones")
    parser.add_argument("--backbone", required=True,
                        help="Backbone name (stack, transcriptformer, scfoundation, "
                             "scgpt, geneformer, bulkformer, or 'all')")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (sdy67, sweetwater, huuki_myers, "
                             "demixsc_retina, altman_Arunachalam, monaco_s13, or 'all')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaler", action="store_true",
                        help="Use StandardScaler before RidgeCV (default: no scaler)")
    args = parser.parse_args()

    backbones = list(ENCODE_FN.keys()) if args.backbone == "all" else [args.backbone]
    ds_names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]

    for bb in backbones:
        if bb not in ENCODE_FN:
            print(f"ERROR: unknown backbone '{bb}'. Available: {list(ENCODE_FN.keys())}")
            sys.exit(1)

    all_pearson: dict[str, dict[str, float | None]] = {}

    for backbone in backbones:
        encode_fn = ENCODE_FN[backbone]
        print(f"\n{'='*60}\nBackbone: {backbone}\n{'='*60}")

        for ds_name in ds_names:
            h5_path = DATA_DIR / f"{ds_name}.h5"
            gt_path = DATA_DIR / f"{ds_name}_gt.csv"
            if not h5_path.exists():
                print(f"  SKIP {ds_name}: H5 not found")
                all_pearson.setdefault(ds_name, {})[backbone] = None
                continue
            if not gt_path.exists():
                print(f"  SKIP {ds_name}: GT not found (H5-only)")
                all_pearson.setdefault(ds_name, {})[backbone] = None
                continue

            print(f"\n  {ds_name}:", end=" ", flush=True)
            with ResourceTracker() as rt:
                vals, genes, samples = _load_bulk_h5(h5_path)
                gt_df = _read_gt_csv(gt_path)

                rt.start_encode()
                try:
                    emb = encode_fn(vals, genes, samples)
                except Exception as e:
                    print(f"ENCODE FAILED: {e}")
                    traceback.print_exc()
                    all_pearson.setdefault(ds_name, {})[backbone] = None
                    continue
                rt.end_encode()

                result = evaluate_real_bulk_ridge(
                    emb, gt_df, ds_name, seed=args.seed,
                    use_scaler=args.scaler,
                )
                rt.end_ridge()

                out_dir = RESULTS_DIR / ds_name / backbone / ("ridge_scaler" if args.scaler else "ridge")
                _save_results(out_dir, result, rt, backbone, ds_name, args.seed)

                r = result["deconbench"].get("pearson_mean", float("nan"))
                print(f"r={r:.4f} [{rt.wall_time_s}s]")
                all_pearson.setdefault(ds_name, {})[backbone] = r

    # ── Summary table ──
    print(f"\n{'='*80}\nSummary: Real-bulk RidgeCV Pearson r\n{'='*80}")
    header = f"{'Dataset':25s}" + "".join(f"{b:>10s}" for b in backbones)
    print(header)
    print("-" * len(header))
    for ds_name in sorted(all_pearson):
        row = f"{ds_name:25s}"
        for bb in backbones:
            v = all_pearson[ds_name].get(bb)
            row += f"{v:>10.4f}" if v is not None else f"{'---':>10s}"
        print(row)


if __name__ == "__main__":
    main()
