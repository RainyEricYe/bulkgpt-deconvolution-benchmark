#!/usr/bin/env python3
"""PBMC cross-dataset generalization: train RidgeCV on source → predict on target.

All datasets except huuki_myers (brain) and demixsc_retina (retina) are
PBMC-derived.  Hao datasets have numeric CSV columns — they are renamed
using H5 ``ground_truth/rownames`` at load time.

Usage::

    # Single pair (quick test)
    python tests/cross_dataset_pbmc.py --source sdy67 --target finotello_Hao

    # Batch all PBMC pairs
    python tests/cross_dataset_pbmc.py --batch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import h5py
import pandas as pd

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
torch.manual_seed(42)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "2_real_bulk"
RESULTS_DIR = HERE / "cross_dataset_pbmc"

# All datasets (PBMC + huuki_myers brain + demixsc_retina retina)
ALL_DATASETS = [
    "sdy67", "sweetwater",
    "altman_Arunachalam", "altman_TabulaSapiens", "altman_Hao",
    "finotello_Hao", "hoek_Hao", "hoek_purified_Hao",
    "linsley_purified_Hao", "morandini_Hao",
    "huuki_myers", "demixsc_retina",
]

# ── Common coarse taxonomy ──────────────────────────────────────────────────
# Fine-to-coarse aggregation: coarse_type → [fine GT columns to sum].

COARSE_MAP: dict[str, dict[str, list[str]]] = {
    "sdy67": {
        "Lymphocytes": ["T_cells", "B_cells", "NK_cells", "Plasmablasts"],
        "Monocytes": ["Monocytes"],
    },
    "sweetwater": {
        "Lymphocytes": ["T_cells", "B_cells"],
        "Monocytes": ["Monocytes"],
        "Neutrophils": ["Neutrophils"],
    },
    "altman_Arunachalam": {
        "Lymphocytes": ["Lymphocytes"],
        "Monocytes": ["Monocytes"],
        "Neutrophils": ["Neutrophils"],
    },
    "altman_TabulaSapiens": {
        "Lymphocytes": ["Lymphocytes"],
        "Monocytes": ["Monocytes"],
        "Neutrophils": ["Neutrophils"],
    },
    "altman_Hao": {
        "Lymphocytes": ["Lymphocytes"],
        "Monocytes": ["Monocytes"],
        "Neutrophils": ["Neutrophils"],
    },
    "finotello_Hao": {
        "Lymphocytes": ["NK cells", "B cells", "Tregs", "T cells CD8", "T cells CD4 conv"],
        "Monocytes": ["Monocytes"],
        "Neutrophils": ["Neutrophils"],
    },
    "hoek_Hao": {
        "Lymphocytes": ["T cell", "B cells", "NK cells"],
        "Monocytes": ["Monocytes"],
    },
    "hoek_purified_Hao": {
        "Lymphocytes": ["B cells", "NK cells", "T cell"],
        "Monocytes": ["Monocytes"],
        "Neutrophils": ["Neutrophils"],
    },
    "linsley_purified_Hao": {
        "Lymphocytes": ["B cells", "NK cells", "T cells CD4", "T cells CD8"],
        "Monocytes": ["Monocytes"],
        "Neutrophils": ["Neutrophils"],
    },
    "morandini_Hao": {
        "Lymphocytes": ["Lymphocytes", "B cells", "NK cells", "T cells", "T cells CD4", "T cells CD8"],
        "Monocytes": ["Monocytes"],
        "Neutrophils": ["Granulocytes"],
    },
    "huuki_myers": {
        "Astro": ["Astro"],
        "EndoMural": ["EndoMural"],
        "Inhib": ["Inhib"],
        "Excit": ["Excit"],
        "Micro": ["Micro"],
        "OligoOPC": ["OligoOPC"],
    },
    "demixsc_retina": {
        "RGC": ["RGC"],
        "AC": ["AC"],
        "BC": ["BC"],
        "HC": ["HC"],
        "Rod": ["Rod"],
        "Cone": ["Cone"],
        "MG": ["MG"],
    },
}


def _get_h5_gt_rownames(ds_name: str) -> list[str] | None:
    """Read cell-type names from H5 ground_truth/rownames (if present)."""
    path = DATA_DIR / f"{ds_name}.h5"
    with h5py.File(str(path), "r") as f:
        if "ground_truth" not in f:
            return None
        return [x.decode() if isinstance(x, bytes) else str(x)
                for x in f["ground_truth/rownames"][:]]


def load_h5_gt(ds_name: str) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
    """Load bulk expression + GT for a dataset.

    For Hao datasets (numeric CSV columns), renames columns using H5
    ``ground_truth/rownames`` so that ``aggregate_to_coarse`` can match
    by name.
    """
    h5_path = DATA_DIR / f"{ds_name}.h5"
    gt_path = DATA_DIR / f"{ds_name}_gt.csv"
    if not h5_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Missing data for {ds_name}")

    with h5py.File(str(h5_path), "r") as f:
        vals = f["bulk/values"][:]
        rn = [x.decode() if isinstance(x, bytes) else str(x) for x in f["bulk/rownames"][:]]
        cn = [x.decode() if isinstance(x, bytes) else str(x) for x in f["bulk/colnames"][:]]
    if vals.shape[0] == len(rn) and vals.shape[1] == len(cn):
        vals = vals.T

    gt_df = pd.read_csv(gt_path, index_col=0)

    # Rename numeric columns using H5 ground_truth rownames (if available)
    h5_types = _get_h5_gt_rownames(ds_name)
    if h5_types is not None and len(h5_types) == len(gt_df.columns):
        gt_df.columns = h5_types
    elif h5_types is not None:
        print(f"  WARNING: {ds_name}: H5 has {len(h5_types)} types, "
              f"CSV has {len(gt_df.columns)}")

    return vals.astype(np.float32), rn, cn, gt_df


def aggregate_to_coarse(gt_df: pd.DataFrame, ds_name: str) -> pd.DataFrame:
    """Sum fine GT columns into coarse types per COARSE_MAP."""
    mapping = COARSE_MAP.get(ds_name, {})
    if not mapping:
        raise ValueError(f"No coarse mapping for {ds_name}")

    coarse_cols = {}
    for coarse_name, fine_cols in mapping.items():
        available = [c for c in fine_cols if c in gt_df.columns]
        if available:
            coarse_cols[coarse_name] = gt_df[available].sum(axis=1)
        else:
            print(f"  WARNING: {ds_name}: none of {fine_cols} found in GT")

    return pd.DataFrame(coarse_cols, index=gt_df.index)


def compute_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    mask = ~(np.isnan(true) | np.isnan(pred))
    if mask.sum() < 2:
        return float("nan")
    tp = true[mask]
    pp = pred[mask]
    if np.std(tp) < 1e-12 or np.std(pp) < 1e-12:
        return float("nan")
    try:
        return float(np.corrcoef(tp, pp)[0, 1])
    except Exception:
        return float("nan")


def run_pair(source: str, target: str, backbone: str,
             alphas: list[float], out_dir: Path) -> dict | None:
    """Run one source→target pair."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"{source} → {target} ({backbone})")
    print(f"{'=' * 60}")

    src_vals, src_genes, src_samples, src_gt = load_h5_gt(source)
    tgt_vals, tgt_genes, tgt_samples, tgt_gt = load_h5_gt(target)
    print(f"  Source: {src_vals.shape[0]} samples, {src_vals.shape[1]} genes")
    print(f"  Target: {tgt_vals.shape[0]} samples, {tgt_vals.shape[1]} genes")

    src_coarse = aggregate_to_coarse(src_gt, source)
    tgt_coarse = aggregate_to_coarse(tgt_gt, target)
    common_types = sorted(set(src_coarse.columns) & set(tgt_coarse.columns))
    if not common_types:
        print(f"  SKIP: no common coarse types")
        return None
    print(f"  Common types: {common_types}")

    t0 = time.monotonic()

    if backbone == "random_mean_pool":
        from methods.bulkformer.model import BulkFormerEncoder
        encoder = BulkFormerEncoder(pretrained=False)
        src_emb = encoder.encode(src_vals, src_genes, src_samples, pooling="mean")
        tgt_emb = encoder.encode(tgt_vals, tgt_genes, tgt_samples, pooling="mean")
    elif backbone == "pca_ridge":
        from sklearn.decomposition import PCA
        bf_dir = Path(__file__).resolve().parent.parent / "weights" / "bulkformer" / "source"
        bf_dir = Path(os.environ.get("BULKFORMER_DIR", str(bf_dir)))
        gene_csv = bf_dir / "data" / "bulkformer_gene_info.csv"
        bf_genes = list(pd.read_csv(gene_csv).iloc[:, 0])
        g2i = {g.upper(): i for i, g in enumerate(bf_genes)}

        def _align(vals, genes):
            mat = np.full((vals.shape[0], len(bf_genes)), -10.0, dtype=np.float32)
            for i, g in enumerate(genes):
                idx = g2i.get(str(g).upper())
                if idx is not None:
                    mat[:, idx] = vals[:, i]
            return mat

        src_emb = _align(src_vals, src_genes)
        tgt_emb = _align(tgt_vals, tgt_genes)
        n_comp = min(src_emb.shape[0], src_emb.shape[1])
        pca = PCA(n_components=n_comp, random_state=42).fit(src_emb)
        src_emb = pca.transform(src_emb).astype(np.float64)
        tgt_emb = pca.transform(tgt_emb).astype(np.float64)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    enc_time = time.monotonic() - t0
    print(f"  Embedding: src={src_emb.shape}, tgt={tgt_emb.shape} [{enc_time:.1f}s]")

    from sklearn.linear_model import RidgeCV

    src_gt_vals = src_coarse[common_types].values.astype(np.float64)
    tgt_gt_vals = tgt_coarse[common_types].values.astype(np.float64)

    ridge_results: dict = {}
    tgt_pred = np.zeros((tgt_emb.shape[0], len(common_types)), dtype=np.float64)

    ridge_start = time.monotonic()
    for j, ct in enumerate(common_types):
        y = src_gt_vals[:, j]
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            ridge_results[ct] = {"alpha": None, "pearson_r": None, "rmse": None}
            continue
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(src_emb[mask], y[mask])
        tgt_pred[:, j] = ridge.predict(tgt_emb)
        ridge_results[ct] = {
            "alpha": float(ridge.alpha_),
            "train_samples": int(mask.sum()),
        }

    tgt_pred = np.nan_to_num(tgt_pred, nan=0.0)
    ridge_time = time.monotonic() - ridge_start

    per_type = {}
    for j, ct in enumerate(common_types):
        r = compute_pearson(tgt_pred[:, j], tgt_gt_vals[:, j])
        rmse = float(np.sqrt(np.mean((tgt_pred[:, j] - tgt_gt_vals[:, j]) ** 2)))
        ridge_results[ct]["pearson_r"] = round(r, 4) if not np.isnan(r) else None
        ridge_results[ct]["rmse"] = round(rmse, 4)
        per_type[ct] = {"pearson_r": ridge_results[ct]["pearson_r"],
                         "rmse": ridge_results[ct]["rmse"]}

    from core.metrics import evaluate_deconvolution
    deconbench = evaluate_deconvolution(tgt_gt_vals, tgt_pred, common_types)
    vals_r = [v["pearson_r"] for v in ridge_results.values() if v["pearson_r"] is not None]
    mean_r = round(float(np.nanmean(vals_r)), 4) if vals_r else None

    pd.DataFrame(tgt_pred, index=tgt_coarse.index, columns=common_types).to_csv(
        out_dir / "proportions.csv")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(deconbench, f, indent=2)
    with open(out_dir / "per_type.json", "w") as f:
        json.dump(per_type, f, indent=2)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "source": source, "target": target, "backbone": backbone,
            "common_types": common_types,
            "n_source": src_emb.shape[0], "n_target": tgt_emb.shape[0],
            "embed_dim": src_emb.shape[1],
            "alphas": alphas,
            "ridge_time_s": round(ridge_time, 3),
            "encode_time_s": round(enc_time, 1),
        }, f, indent=2)

    print(f"  Mean r = {mean_r}")
    for ct in common_types:
        print(f"    {ct}: r={ridge_results[ct]['pearson_r']}, "
              f"alpha={ridge_results[ct]['alpha']}")

    return {"mean_r": mean_r, "per_type": per_type, "common_types": common_types}


def _encode_all_datasets(
    ds_names: list[str],
    backbone: str,
    alphas: list[float],
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Encode all datasets once and return {name: embedding, gt, coarse}."""
    embeddings: dict[str, np.ndarray] = {}
    coarse_gts: dict[str, pd.DataFrame] = {}
    raw_gts: dict[str, pd.DataFrame] = {}

    encoder = None
    pca = None
    bf_genes = None
    g2i = None

    for ds in ds_names:
        print(f"  Encode {ds}...", end=" ", flush=True)
        t0 = time.monotonic()
        vals, genes, samples, gt_df = load_h5_gt(ds)
        raw_gts[ds] = gt_df
        coarse_gts[ds] = aggregate_to_coarse(gt_df, ds)

        if backbone == "random_mean_pool":
            if encoder is None:
                from methods.bulkformer.model import BulkFormerEncoder
                encoder = BulkFormerEncoder(pretrained=False)
            emb = encoder.encode(vals, genes, samples, pooling="mean")
        elif backbone == "pca_ridge":
            if bf_genes is None:
                bf_dir = (Path(__file__).resolve().parent.parent
                          / "weights" / "bulkformer" / "source")
                bf_dir = Path(os.environ.get("BULKFORMER_DIR", str(bf_dir)))
                gene_csv = bf_dir / "data" / "bulkformer_gene_info.csv"
                bf_genes = list(pd.read_csv(gene_csv).iloc[:, 0])
                g2i = {g.upper(): i for i, g in enumerate(bf_genes)}

            mat = np.full((vals.shape[0], len(bf_genes)), -10.0, dtype=np.float32)
            for i, g in enumerate(genes):
                idx = g2i.get(str(g).upper())
                if idx is not None:
                    mat[:, idx] = vals[:, i]

            if pca is None:
                from sklearn.decomposition import PCA
                n_comp = min(mat.shape[0], mat.shape[1])
                pca = PCA(n_components=n_comp, random_state=42).fit(mat)
            emb = pca.transform(mat).astype(np.float64)

        embeddings[ds] = emb
        print(f"({emb.shape[1]}d, {time.monotonic() - t0:.1f}s)")

    return embeddings, coarse_gts, raw_gts


def main() -> None:
    p = argparse.ArgumentParser(description="PBMC cross-dataset RidgeCV")
    p.add_argument("--source", default=None, help="Source dataset")
    p.add_argument("--target", default=None, help="Target dataset")
    p.add_argument("--backbone", default="random_mean_pool",
                    choices=["random_mean_pool", "pca_ridge"])
    p.add_argument("--batch", action="store_true",
                    help="Run all source→target pairs (12 datasets)")
    p.add_argument("--sources", nargs="*", default=None,
                    help="Sources for batch mode (default: all datasets)")
    p.add_argument("--alphas", nargs="*", type=float,
                    default=[0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0])
    args = p.parse_args()

    if args.batch:
        sources = args.sources or ALL_DATASETS
        targets = ALL_DATASETS
        all_ds = list(dict.fromkeys(sources + targets))  # dedup, preserve order

        print(f"Encoding {len(all_ds)} datasets once...")
        t_all = time.monotonic()
        embeddings, coarse_gts, _ = _encode_all_datasets(
            all_ds, args.backbone, args.alphas)
        print(f"Total encode time: {time.monotonic() - t_all:.1f}s\n")

        from sklearn.linear_model import RidgeCV

        summary = []
        for src in sources:
            src_emb = embeddings[src]
            src_coarse = coarse_gts[src]
            for tgt in targets:
                tgt_emb = embeddings[tgt]
                tgt_coarse = coarse_gts[tgt]

                common_types = sorted(set(src_coarse.columns) & set(tgt_coarse.columns))
                if not common_types:
                    continue

                src_gt_v = src_coarse[common_types].values.astype(np.float64)
                tgt_gt_v = tgt_coarse[common_types].values.astype(np.float64)
                tgt_pred = np.zeros((tgt_emb.shape[0], len(common_types)), dtype=np.float64)

                per_type = {}
                for j, ct in enumerate(common_types):
                    y = src_gt_v[:, j]
                    mask = ~np.isnan(y)
                    if mask.sum() < 2:
                        per_type[ct] = {"pearson_r": None, "rmse": None}
                        continue
                    ridge = RidgeCV(alphas=args.alphas)
                    ridge.fit(src_emb[mask], y[mask])
                    tgt_pred[:, j] = ridge.predict(tgt_emb)
                    r = compute_pearson(tgt_pred[:, j], tgt_gt_v[:, j])
                    rmse = float(np.sqrt(np.mean((tgt_pred[:, j] - tgt_gt_v[:, j]) ** 2)))
                    per_type[ct] = {
                        "pearson_r": round(r, 4) if not np.isnan(r) else None,
                        "rmse": round(rmse, 4),
                    }

                tgt_pred = np.nan_to_num(tgt_pred, nan=0.0)

                from core.metrics import evaluate_deconvolution
                deconbench = evaluate_deconvolution(tgt_gt_v, tgt_pred, common_types)
                vals_r = [v["pearson_r"] for v in per_type.values()
                          if v["pearson_r"] is not None]
                mean_r = round(float(np.nanmean(vals_r)), 4) if vals_r else None

                out_dir = RESULTS_DIR / f"{src}_to_{tgt}" / args.backbone
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(tgt_pred, index=tgt_coarse.index,
                             columns=common_types).to_csv(out_dir / "proportions.csv")
                with open(out_dir / "metrics.json", "w") as f:
                    json.dump(deconbench, f, indent=2)
                with open(out_dir / "per_type.json", "w") as f:
                    json.dump(per_type, f, indent=2)

                summary.append({
                    "source": src, "target": tgt,
                    "mean_r": mean_r,
                    "common_types": common_types,
                })
                print(f"  {src:>20} → {tgt:<20}  r={mean_r}  "
                      f"types={common_types}")

        # Summary table
        print(f"\n{'=' * 70}")
        print("CROSS-DATASET PBMC GENERALIZATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Source':>20} → {'Target':<20} | {'Mean r':>7} | Types")
        print("-" * 70)
        for s in summary:
            ct = ",".join(s["common_types"])
            mr = f"{s['mean_r']:.4f}" if s['mean_r'] is not None else "  N/A "
            print(f"{s['source']:>20} → {s['target']:<20} | {mr:>7} | {ct}")

        summary_path = RESULTS_DIR / f"summary_{args.backbone}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {summary_path}")
    else:
        if not args.source or not args.target:
            p.print_help()
            sys.exit(1)
        out_dir = RESULTS_DIR / f"{args.source}_to_{args.target}" / args.backbone
        run_pair(args.source, args.target, args.backbone, args.alphas, out_dir)


if __name__ == "__main__":
    main()
