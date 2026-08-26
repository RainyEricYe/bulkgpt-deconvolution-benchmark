#!/usr/bin/env python3
"""Build a DeconBenchmark-format H5 file from an h5ad single-cell reference.

Usage:
    python scripts/build_h5_from_h5ad.py \
        --h5ad data/matched/demixsc_retina/demixsc_retina_bulkgpt.h5ad \
        --bulk data/matched/demixsc_retina/retina_benchmark_bulk_batch1.csv \
        --gt data/matched/demixsc_retina/retina_benchmark_ground_truth.csv \
        --output data/2_real_bulk/demixsc_retina.h5

The H5 follows the DeconBenchmark standard (documented in CLAUDE.md):

    singleCellExpr/values:   (n_cells, n_genes)
    singleCellExpr/rownames:  genes
    singleCellExpr/colnames:  cell identifiers
    singleCellLabels/values:  cell-type labels (n_cells)

    bulk/values:   (n_samples, n_genes)
    bulk/rownames:  genes
    bulk/colnames:  sample names

    ground_truth/values:   (n_types, n_samples)
    ground_truth/rownames:  cell types
    ground_truth/colnames:  sample names
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def read_h5ad(path: str):
    import anndata as ad
    adata = ad.read_h5ad(path)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    cell_ids = [f'cell_{i}' for i in range(adata.n_obs)]
    gene_names = adata.var_names.tolist()
    labels = None
    for col in ["cell_type", "CellType", "celltype", "label", "cluster"]:
        if col in adata.obs.columns:
            labels = adata.obs[col].astype(str).tolist()
            break
    return X, gene_names, cell_ids, labels


def main():
    parser = argparse.ArgumentParser(description="Build DeconBenchmark H5 from h5ad")
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--bulk", help="Bulk CSV (samples x genes)")
    parser.add_argument("--gt", help="Ground truth CSV (samples x types)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists() and args.backup:
        bak = out.with_suffix(".h5.bak")
        out.rename(bak)
        print(f"Backup: {bak}")

    print(f"Reading h5ad: {args.h5ad}")
    sc_expr, gene_names, cell_ids, labels = read_h5ad(args.h5ad)
    n_cells, n_genes = sc_expr.shape
    print(f"  {n_cells} cells x {n_genes} genes, labels={'yes' if labels else 'no'}")

    if args.bulk:
        # Support multiple CSVs/TSVs (comma-separated paths)
        bulk_csvs = args.bulk.split(",")
        all_bulk = []
        for csv in bulk_csvs:
            csv = csv.strip()
            sep = "	" if csv.endswith(".tsv") or csv.endswith(".tsv.gz") else ","
            print(f"  Reading bulk: {csv} (sep={repr(sep)})")
            bdf = pd.read_csv(csv, sep=sep, index_col=0).T  # (samples x genes)
            all_bulk.append(bdf)
        bdf = pd.concat(all_bulk)  # (all_samples, raw_genes)
        bm = np.zeros((bdf.shape[0], n_genes), dtype=np.float64)
        bs = bdf.index.tolist()
        for j, g in enumerate(gene_names):
            if g in bdf.columns:
                bm[:, j] = bdf[g].values.astype(np.float64)
        print(f"  {bm.shape[0]} samples, {bm.shape[1]} genes (aligned)")

    if args.gt:
        print(f"Reading GT: {args.gt}")
        gt_df = pd.read_csv(args.gt, index_col=0)
        # CSV: rows=cell types, columns=samples
        # H5 stores (n_types, n_samples)
        gt_types = gt_df.index.tolist()
        gt_samples = gt_df.columns.tolist()
        gt_m = gt_df.values.astype(np.float64)  # (n_types, n_samples)
        print(f"  {gt_m.shape[0]} types x {gt_m.shape[1]} samples")

    print(f"Writing H5: {args.output}")
    with h5py.File(str(out), "w") as f:
        scg = f.create_group("singleCellExpr")
        scg.create_dataset("values", data=sc_expr, dtype=np.float64)
        scg.create_dataset("rownames", data=[g.encode() for g in gene_names])
        scg.create_dataset("colnames", data=[c.encode() for c in cell_ids])
        if labels:
            f.create_dataset("singleCellLabels/values", data=[l.encode() for l in labels])
            f.create_dataset("nCellTypes/values", data=len(set(labels)))
        f.create_dataset("seed/values", data=args.seed)

        if args.bulk:
            bulk_grp = f.create_group("bulk")
            bulk_grp.create_dataset("values", data=bm, dtype=np.float64)
            bulk_grp.create_dataset("rownames", data=[g.encode() for g in gene_names])
            bulk_grp.create_dataset("colnames", data=[s.encode() for s in bs])

        if args.gt:
            gg = f.create_group("ground_truth")
            gg.create_dataset("values", data=gt_m, dtype=np.float64)  # (n_types, n_samples)
            gg.create_dataset("rownames", data=[t.encode() for t in gt_types])    # cell types
            gg.create_dataset("colnames", data=[s.encode() for s in gt_samples])  # sample names

    print("Done ✅")
if __name__ == "__main__":
    main()

