#!/usr/bin/env python3
"""Shared engine for the ML regressor deconvolution baselines (P2#13).

Protocol (Mode A, SCADEN-style):
  For each dataset H5, generate Dirichlet pseudo-bulk training mixtures from
  the dataset's scRNA-seq reference (cells x genes), train a supervised
  regressor mapping mixture expression -> cell-type proportions, then predict
  proportions for the dataset's real bulk samples. Reference-free: uses only
  the expression matrix and the scRNA cell-type labels inside the H5.

Features: per-dataset top-n_hvg most-variable genes common to scRNA and bulk,
log1p-CPM normalised. Output contract: proportions.csv with index=sample,
columns=cell types (consumed by scripts/evaluate.py post-hoc evaluation).

Each per-method run.py supplies a model factory:
    factory(n_types: int, params: dict) -> {fit(X, y), predict(X)}
and calls run_baseline(factory, name).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="ML regressor deconvolution baseline")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mode", type=str, default="predict")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--h5", type=str, default=None)
    p.add_argument("--ground-truth", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def load_config(path):
    import yaml
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _decode(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def load_h5(h5_path, max_cells):
    """Load scRNA reference + bulk from a DeconBenchmark H5, aligned on genes.

    Returns dict: cells (n_cells x n_common), labels (n_cells,), type_list,
    bulk (n_samples x n_common), bulk_samples (n_samples,), common_genes.
    """
    import h5py

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    h5_path = str(h5_path)

    with h5py.File(h5_path, "r") as h:
        labels = _decode(h["singleCellLabels/values"][:])
        n_cells = len(labels)

        sce = h["singleCellExpr"]
        n_genes = sce["values"].shape[1]
        scer = _decode(sce["rownames"][:]) if "rownames" in sce else []
        scec = _decode(sce["colnames"][:]) if "colnames" in sce else []
        sce_genes = scer if len(scer) == n_genes else (scec if len(scec) == n_genes else None)

        rng = np.random.RandomState(SEED)
        if n_cells > max_cells:
            idx = rng.choice(n_cells, max_cells, replace=False)
        else:
            idx = np.arange(n_cells)
        srt = np.sort(idx)
        cells = sce["values"][srt, :].astype(np.float64)
        labels = [labels[i] for i in srt]

        b = h["bulk"]
        bv = b["values"][:]
        brow = _decode(b["rownames"][:]) if "rownames" in b else []
        bcol = _decode(b["colnames"][:]) if "colnames" in b else []
        if len(brow) == bv.shape[1]:
            bulk, bulk_genes, bulk_samples = bv, brow, bcol
        elif len(brow) == bv.shape[0]:
            bulk, bulk_genes, bulk_samples = bv.T, bcol, brow
        else:
            raise ValueError(
                f"bulk orientation unknown: shape={bv.shape}, "
                f"rownames={len(brow)}, colnames={len(bcol)}"
            )

    if sce_genes is None:
        raise ValueError("cannot determine scRNA gene names from H5")
    common = sorted(set(sce_genes) & set(bulk_genes))
    if len(common) < 50:
        raise ValueError(f"too few common genes between scRNA and bulk: {len(common)}")
    si = [sce_genes.index(g) for g in common]
    bi = [bulk_genes.index(g) for g in common]
    cells = cells[:, si]
    bulk = bulk[:, bi]

    return {
        "cells": cells, "labels": labels, "type_list": sorted(set(labels)),
        "bulk": bulk, "bulk_samples": bulk_samples, "common_genes": common,
    }


def hvg_select(cells, n_hvg):
    """Indices of the top-n_hvg most variable genes (variance across cells)."""
    if n_hvg is None or n_hvg >= cells.shape[1]:
        return np.arange(cells.shape[1])
    var = np.var(cells, axis=0)
    return np.argsort(var)[::-1][:n_hvg]


def cpm_log1p(X):
    """Per-row CPM normalisation then log1p."""
    row = X.sum(axis=1, keepdims=True)
    cpm = X / np.maximum(row, 1e-10) * 1e6
    return np.log1p(cpm)


def build_mixtures(cells, labels, type_list, n_samples, n_cells_per_sample,
                   alpha, seed):
    """Dirichlet pseudo-bulk mixtures sampled at single-cell level.

    For each mixture: draw a Dirichlet(alpha) proportion vector, draw cell
    counts per type ~ Multinomial(n_cells_per_sample, p), then sum random
    cells of each type (with replacement). Returns log1p-CPM X (n_samples x
    n_genes) and the true proportions y (n_samples x n_types).
    """
    rng = np.random.RandomState(seed)
    n_types = len(type_list)
    type_idx = np.array([type_list.index(t) for t in labels])
    per_type = [np.where(type_idx == t)[0] for t in range(n_types)]

    P = rng.dirichlet(np.ones(n_types) * alpha, size=n_samples)
    n_cells_arr = np.maximum(10, rng.poisson(n_cells_per_sample, size=n_samples))
    counts = np.zeros((n_samples, n_types), dtype=np.int64)
    for i in range(n_samples):
        counts[i] = rng.multinomial(int(n_cells_arr[i]), P[i])

    X = np.zeros((n_samples, cells.shape[1]))
    for t in range(n_types):
        n_ct = int(counts[:, t].sum())
        if n_ct == 0:
            continue
        cell_sel = rng.choice(per_type[t], size=n_ct, replace=True)
        mixt_ids = np.repeat(np.arange(n_samples), counts[:, t])
        np.add.at(X, mixt_ids, cells[cell_sel])

    X = cpm_log1p(X)
    return X, P


def postprocess(pred):
    """Clip negatives and normalise each sample's proportions to sum 1."""
    pred = np.maximum(np.asarray(pred, dtype=np.float64), 0.0)
    row = pred.sum(axis=1, keepdims=True)
    return pred / np.maximum(row, 1e-10)


def run_baseline(model_factory, model_name):
    args = parse_args()
    cfg = load_config(args.config)
    params = dict(cfg.get("params", {}))

    if not args.h5:
        raise SystemExit("Error: --h5 required (ML baselines run in H5 mode)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_cells = int(params.get("max_cells", 15000))
    n_hvg = int(params.get("n_hvg", 2000))
    n_pseudo = int(params.get("n_pseudo_bulk", 500))
    n_cells_per = int(params.get("n_cells_per_sample", 80))
    alpha = float(params.get("mixture_alpha", 0.1))
    seed = int(params.get("seed", SEED))

    print("=" * 60)
    print(f"{model_name.upper()} ML baseline — {Path(args.h5).name}")
    print("=" * 60)

    print("[1] Loading H5 ...")
    data = load_h5(args.h5, max_cells)
    cells, labels = data["cells"], data["labels"]
    type_list = data["type_list"]
    print(f"    scRNA {cells.shape[0]} cells x {cells.shape[1]} genes, "
          f"{len(type_list)} types; bulk {data['bulk'].shape[0]} samples")

    hvg = hvg_select(cells, n_hvg)
    cells = cells[:, hvg]
    bulk = data["bulk"][:, hvg]
    print(f"    features: {len(hvg)} HVG of {len(data['common_genes'])} common genes")

    print("[2] Building training mixtures ...")
    X_train, y_train = build_mixtures(cells, labels, type_list,
                                      n_pseudo, n_cells_per, alpha, seed)
    print(f"    {X_train.shape[0]} mixtures x {X_train.shape[1]} features "
          f"-> {len(type_list)} targets")

    print("[3] Training model ...")
    model = model_factory(n_types=len(type_list), params=params)
    model.fit(X_train, y_train)

    print("[4] Predicting bulk ...")
    X_bulk = cpm_log1p(bulk)
    pred = np.asarray(model.predict(X_bulk), dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(-1, len(type_list))
    props = postprocess(pred)

    df = pd.DataFrame(props, index=data["bulk_samples"], columns=type_list)
    df.index.name = "sample"
    pred_csv = out_dir / "proportions.csv"
    df.to_csv(pred_csv)
    print(f"    proportions.csv -> {pred_csv} ({df.shape[0]} x {df.shape[1]})")
    print("    (metrics computed by post-hoc evaluation)")
    print("Done.")
