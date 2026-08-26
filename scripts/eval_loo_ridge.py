#!/usr/bin/env python3
"""Leave-one-out RidgeCV evaluation for real-bulk deconvolution.

Trains RidgeCV on all samples except one, predicts the held-out sample,
repeats for every sample (LOO-CV).  Produces per-method performance that
better reflects real-world generalization than a single train/test split.

Usage
-----
    # Single backbone + dataset
    python scripts/eval_loo_ridge.py --backbone stack --dataset sdy67

    # All backbones on one dataset
    python scripts/eval_loo_ridge.py --backbone all --dataset sdy67

    # All backbones x all datasets
    python scripts/eval_loo_ridge.py --backbone all --dataset all

    # With scaler (ridge_scaler variant)
    python scripts/eval_loo_ridge.py --backbone scgpt --dataset sweetwater --variant ridge_scaler

    # All RidgeCV variants
    python scripts/eval_loo_ridge.py --backbone all --dataset all --variant all

Output
------
    results/2_realbulk/{dataset}/{backbone}/{variant}_loo/
      |- proportions.csv       -- LOO predictions (n_samples x n_types)
      |- metrics.json          -- DeconBenchmark suite
      |- ridge_metrics.json    -- per-type Pearson r / RMSE
      '- metadata.json         -- timing + env info
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Enforce deterministic GPU inference across all backbones.
# FP16 + flash-attn are non-deterministic by default and produce
# different embeddings for identical bulk data across runs.
import torch
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.deconv.frozen_eval import ENCODE_FN, ResourceTracker
from core.deconv.utils import renormalize_props

DATA_DIR = PROJECT_ROOT / "data" / "2_real_bulk"
RESULTS_DIR = PROJECT_ROOT / "results" / "2_realbulk"

DEFAULT_ALPHAS = [0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0]

RIDGE_VARIANTS = {
    "ridge": {"use_scaler": False, "suffix": "ridge_loo"},
    "ridge_scaler": {"use_scaler": True, "suffix": "ridge_scaler_loo"},
    "pca_ridge": {"use_scaler": False, "suffix": "pca_ridge_loo"},
}

DATASETS = {
    "sdy67", "sweetwater", "huuki_myers", "demixsc_retina",
    "altman_Arunachalam", "altman_TabulaSapiens",
    "altman_Hao", "finotello_Hao", "hoek_Hao",
    "hoek_purified_Hao", "linsley_purified_Hao", "morandini_Hao",
}


def _load_bulk_h5(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    import h5py
    with h5py.File(str(path), "r") as f:
        vals = f["bulk/values"][:]
        rn = [x.decode() if isinstance(x, bytes) else str(x) for x in f["bulk/rownames"][:]]
        cn = [x.decode() if isinstance(x, bytes) else str(x) for x in f["bulk/colnames"][:]]
    if vals.shape[0] == len(rn) and vals.shape[1] == len(cn):
        vals = vals.T
    return vals.astype(np.float32), rn, cn


def _read_gt_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col0 = df.columns[0]
    if str(col0).startswith("Unnamed:") or str(df[col0].dtype) in ("object", "string"):
        df = df.set_index(col0)
    return df


def loo_ridge_cv(
    embeddings: np.ndarray,
    gt_df: pd.DataFrame,
    alphas: list[float] | None = None,
    use_scaler: bool = False,
) -> dict:
    """Leave-one-out RidgeCV evaluation.

    For each sample i: train on all except i, predict i.
    Returns dict with full_predictions_df, deconbench, ridge_specific, metadata.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from core.metrics import evaluate_deconvolution

    alphas = alphas or DEFAULT_ALPHAS
    n = embeddings.shape[0]
    gt_values = gt_df.values.astype(np.float64)
    gt_columns = list(gt_df.columns)
    n_types = len(gt_columns)

    loo_pred = np.zeros((n, n_types), dtype=np.float64)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        train_emb, train_gt = embeddings[train_mask], gt_values[train_mask]

        if use_scaler:
            scaler = StandardScaler()
            train_s = scaler.fit_transform(train_emb)
            test_s = scaler.transform(embeddings[i:i + 1])
        else:
            train_s, test_s = train_emb, embeddings[i:i + 1]

        for j in range(n_types):
            y = train_gt[:, j]
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                loo_pred[i, j] = 0.0
                continue
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(train_s[mask], y[mask])
            loo_pred[i, j] = ridge.predict(test_s)[0]

    loo_pred = renormalize_props(loo_pred)
    deconbench = evaluate_deconvolution(gt_values, loo_pred, gt_columns)

    pearson_per_type = {}
    rmse_per_type = {}
    for j, ct in enumerate(gt_columns):
        mask = ~np.isnan(gt_values[:, j])
        if mask.sum() >= 2 and np.std(gt_values[mask, j]) > 1e-10:
            r = float(np.corrcoef(loo_pred[mask, j], gt_values[mask, j])[0, 1])
            pearson_per_type[ct] = round(r, 4) if not np.isnan(r) else None
        else:
            pearson_per_type[ct] = None
        if mask.sum() >= 2:
            rmse_per_type[ct] = round(
                float(np.sqrt(np.mean((loo_pred[mask, j] - gt_values[mask, j]) ** 2))), 4)
        else:
            rmse_per_type[ct] = None

    return {
        "deconbench": deconbench,
        "ridge_specific": {
            "pearson_per_type": pearson_per_type,
            "rmse_per_type": rmse_per_type,
            "n_loo": n,
            "method": "leave-one-out RidgeCV",
        },
        "metadata": {
            "embedding_dim": embeddings.shape[1],
            "n_total": n,
            "alphas": alphas,
            "use_scaler": use_scaler,
            "method": "loo_ridge",
        },
        "full_predictions_df": pd.DataFrame(loo_pred, index=gt_df.index, columns=gt_columns),
    }


def _save_results(out_dir, result, rt, backbone, dataset, variant, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    result["full_predictions_df"].to_csv(out_dir / "proportions.csv")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result["deconbench"], f, indent=2, default=str)
    with open(out_dir / "ridge_metrics.json", "w") as f:
        json.dump(result["ridge_specific"], f, indent=2, default=str)
    meta = {**result["metadata"], **rt.to_dict(backbone=backbone, dataset=dataset, seed=seed),
            "variant": variant}
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def main():
    p = argparse.ArgumentParser(description="Leave-one-out RidgeCV evaluation")
    p.add_argument("--backbone", required=True, help="Backbone name or 'all'")
    p.add_argument("--dataset", required=True, help="Dataset name or 'all'")
    p.add_argument("--variant", default="ridge_scaler",
                   choices=list(RIDGE_VARIANTS.keys()) + ["all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", nargs="*", type=float, default=None)
    args = p.parse_args()

    backbones = list(ENCODE_FN.keys()) if args.backbone == "all" else [args.backbone]
    for bb in backbones:
        if bb not in ENCODE_FN:
            print(f"ERROR: unknown backbone '{bb}'. Available: {list(ENCODE_FN.keys())}")
            sys.exit(1)

    variants = list(RIDGE_VARIANTS.keys()) if args.variant == "all" else [args.variant]

    ds_names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for ds in ds_names:
        if ds not in DATASETS:
            print(f"ERROR: unknown dataset '{ds}'. Available: {sorted(DATASETS)}")
            sys.exit(1)

    for variant in variants:
        vc = RIDGE_VARIANTS[variant]
        print(f"\n{'=' * 60}\nVariant: {variant} (-> {vc['suffix']})\n{'=' * 60}")

        for backbone in backbones:
            encode_fn = ENCODE_FN[backbone]
            print(f"\n  Backbone: {backbone}")

            for ds_name in ds_names:
                h5p = DATA_DIR / f"{ds_name}.h5"
                gtp = DATA_DIR / f"{ds_name}_gt.csv"
                if not h5p.exists() or not gtp.exists():
                    print(f"    SKIP {ds_name}: data not found"); continue

                print(f"    {ds_name}:", end=" ", flush=True)
                with ResourceTracker() as rt:
                    vals, genes, samples = _load_bulk_h5(h5p)
                    gt_df = _read_gt_csv(gtp)
                    rt.start_encode()
                    try:
                        emb = encode_fn(vals, genes, samples)
                    except (Exception, SystemExit) as e:
                        print(f"ENCODE FAILED: {e}")
                        traceback.print_exc()
                        continue
                    rt.end_encode()
                    pass  # ridge time measured from encode start
                    result = loo_ridge_cv(emb, gt_df, alphas=args.alphas, use_scaler=vc["use_scaler"])
                    rt.end_ridge()
                    out_dir = RESULTS_DIR / ds_name / backbone / vc["suffix"]
                    _save_results(out_dir, result, rt, backbone, ds_name, variant, args.seed)
                    r = result["deconbench"].get("pearson_mean", float("nan"))
                    print(f"r={r:.4f} (LOO, n={result['metadata']['n_total']}) [{rt.wall_time_s}s]")

    print("\nDone.")


if __name__ == "__main__":
    main()
