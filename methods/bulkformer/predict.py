#!/usr/bin/env python3
"""Prediction script for BulkFormer deconvolution.

Two modes:
  - ``mlp_head``: Load a trained DeconvHead checkpoint → encode bulk → predict.
  - ``ridge``   : Encode bulk → RidgeCV on real-bulk split → evaluate.

Usage:
    python methods/bulkformer/predict.py --config configs/default.yaml --checkpoint best_model.pt
    python methods/bulkformer/predict.py --config configs/default.yaml --mode ridge --dataset sdy67
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.deconv.frozen_eval import ResourceTracker, evaluate_real_bulk_ridge
from core.deconv.embedding import EmbeddingDeconvHead
from core.metrics import evaluate_deconvolution

DATA_DIR = _to_publish / "data" / "2_real_bulk"
RESULTS_DIR = _to_publish / "results" / "2_realbulk"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_bulk_h5(path: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Load bulk expression from DeconBenchmark H5."""
    with h5py.File(path, "r") as f:
        vals = f["bulk/values"][:]
        rn = [x.decode() if isinstance(x, bytes) else str(x) for x in f["bulk/rownames"][:]]
        cn = [x.decode() if isinstance(x, bytes) else str(x) for x in f["bulk/colnames"][:]]
    # Canonical format: (n_samples, n_genes).  DeconBenchmark legacy stored
    # (n_genes, n_samples).  Transpose only when rownames count matches axis 0
    # (i.e. genes are on the first axis, which is the old convention).
    if vals.shape[0] == len(rn) and vals.shape[1] == len(cn):
        vals = vals.T
    elif vals.shape[0] == len(cn) and vals.shape[1] == len(rn):
        pass  # already (n_samples, n_genes)
    else:
        vals = vals.T
    return vals, rn, cn


def _read_gt_csv(path: str) -> pd.DataFrame:
    """Read ground truth CSV, handling sample_id column."""
    df = pd.read_csv(path)
    first_col = df.iloc[:, 0]
    if first_col.dtype in (object, str) or first_col.dtype.name == "object":
        df = df.set_index(df.columns[0])
    return df


def _predict_one(
    h5_path: str, checkpoint_path: str, config_path: str,
    out_dir: Path, cell_types: list[str] | None = None,
    gt_csv: str | None = None,
) -> dict:
    """Encode bulk via BulkFormer, predict proportions, save results.

    Returns {dataset_name: pearson_mean} (empty dict if no GT).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    embed_dim = cfg.get("embed_dim", 640)

    if cell_types is None:
        meta_path = Path(checkpoint_path).parent / "checkpoint_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        cell_types = meta["cell_types"]
    n_types = len(cell_types)
    hidden_dims = cfg.get("hidden_dims", [256])

    model = EmbeddingDeconvHead(
        embed_dim, hidden_dims[0] if isinstance(hidden_dims, (list, tuple)) else 256, n_types,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    vals, genes, samples = _load_bulk_h5(str(h5_path))
    from methods.bulkformer.model import encode_bulkformer
    emb = encode_bulkformer(vals, genes, samples)
    with torch.no_grad():
        pred = model(torch.from_numpy(emb).float().to(device)).cpu().numpy()
    pred = np.maximum(pred, 0)
    pred = pred / pred.sum(axis=1, keepdims=True)
    pred_df = pd.DataFrame(pred, index=samples, columns=cell_types)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "proportions.csv")

    results: dict[str, float] = {}
    ds_name = Path(h5_path).stem
    if gt_csv and Path(gt_csv).exists():
        gt_df = _read_gt_csv(str(gt_csv))
        common = [c for c in cell_types if c in gt_df.columns]
        if common:
            m = evaluate_deconvolution(
                gt_df[common].values, pred_df[common].values, common,
            )
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(m, f, indent=2)
            results[ds_name] = m["pearson_mean"]
    return results


def predict_mlp_head(
    config_path: str, checkpoint_path: str, log_file: str | None = None,
) -> dict:
    """Load DeconvHead checkpoint → predict on bulk datasets.

    Two modes, auto-detected from the config:

    * **Pseudo-bulk** (config contains ``h5_path``): predict on a single
      H5 dataset and save results to ``output_dir``.
    * **Real-bulk** (no ``h5_path``): loop over the hardcoded dataset list
      (sdy67 / sweetwater / huuki_myers / demixsc_retina) — original
      behaviour, completely unchanged.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Pseudo-bulk mode: config-driven single dataset ──
    h5_path = cfg.get("h5_path")
    if h5_path:
        gt_csv = cfg.get("gt_path")
        out_dir = Path(cfg.get("output_dir", "."))
        return _predict_one(
            str(h5_path), checkpoint_path, config_path,
            out_dir, gt_csv=str(gt_csv) if gt_csv else None,
        )

    # ── Real-bulk mode (original hardcoded loop, unchanged) ──
    meta_path = Path(checkpoint_path).parent / "checkpoint_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    cell_types = meta["cell_types"]

    results: dict[str, float] = {}
    for ds_name in ["sdy67", "sweetwater", "huuki_myers", "demixsc_retina"]:
        h5_path = DATA_DIR / f"{ds_name}.h5"
        gt_path = DATA_DIR / f"{ds_name}_gt.csv"
        if not h5_path.exists():
            continue
        out_dir = RESULTS_DIR / ds_name / "bulkformer" / "mlp_head"
        r = _predict_one(
            str(h5_path), checkpoint_path, config_path,
            out_dir, cell_types=cell_types,
            gt_csv=str(gt_path) if gt_path.exists() else None,
        )
        results.update(r)
    return results


def predict_ridge(
    config_path: str, checkpoint_path: str = "", log_file: str | None = None,
    dataset: str = "all", seed: int = 42,
) -> dict:
    """RidgeCV mode: encode bulk → real-bulk split → RidgeCV."""
    datasets = (
        ["sdy67", "sweetwater", "huuki_myers", "demixsc_retina", "altman_Arunachalam"]
        if dataset == "all" else [dataset]
    )

    all_results: dict[str, float] = {}
    for ds_name in datasets:
        h5_path = DATA_DIR / f"{ds_name}.h5"
        gt_path = DATA_DIR / f"{ds_name}_gt.csv"
        if not h5_path.exists() or not gt_path.exists():
            print(f"  SKIP {ds_name}: missing data")
            continue

        print(f"\n  {ds_name}:", end=" ", flush=True)
        with ResourceTracker() as rt:
            vals, genes, samples = _load_bulk_h5(str(h5_path))
            gt_df = _read_gt_csv(str(gt_path))

            rt.start_encode()
            from methods.bulkformer.model import encode_bulkformer

            emb = encode_bulkformer(vals, genes, samples)
            rt.end_encode()

            result = evaluate_real_bulk_ridge(emb, gt_df, ds_name, seed=seed)
            rt.end_ridge()

        out_dir = RESULTS_DIR / ds_name / "bulkformer" / "ridge"
        out_dir.mkdir(parents=True, exist_ok=True)

        # proportions.csv (full-sample predictions)
        result["full_predictions_df"].to_csv(out_dir / "proportions.csv")

        # metrics.json (DeconBenchmark suite)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(result["deconbench"], f, indent=2)

        # ridge_metrics.json
        with open(out_dir / "ridge_metrics.json", "w") as f:
            json.dump(result["ridge_specific"], f, indent=2)

        # metadata.json (resources)
        meta = {
            **result["metadata"],
            **rt.to_dict(backbone="bulkformer", dataset=ds_name),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        r = result["deconbench"].get("pearson_mean", float("nan"))
        print(f"r={r:.4f} [{rt.wall_time_s}s]")
        all_results[ds_name] = r

    return all_results


def main(
    config_path: str | None = None,
    checkpoint_path: str | None = None,
    log_file: str | None = None,
) -> None:
    parser = argparse.ArgumentParser(description="BulkFormer deconvolution prediction")
    parser.add_argument("--config", required=config_path is None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="mlp_head", choices=["mlp_head", "ridge"])
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", default=None)
    args, _ = parser.parse_known_args()

    cfg = args.config or config_path
    ckpt = args.checkpoint or checkpoint_path

    if args.mode == "ridge":
        predict_ridge(cfg, ckpt or "", log_file=args.log_file,
                      dataset=args.dataset, seed=args.seed)
    else:
        if not ckpt:
            print("ERROR: --checkpoint required for mlp_head mode", file=sys.stderr)
            sys.exit(1)
        predict_mlp_head(cfg, ckpt, args.log_file)


if __name__ == "__main__":
    main()
