#!/usr/bin/env python3
"""Shared H5 utilities for scRNA-seq deconvolution methods.

Provides:
- :func:`ensure_sc_celltypes` — restore cell-type labels from H5
  ``singleCellLabels/values`` when orientation detection drops them.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


def ensure_sc_celltypes(
    adata: ad.AnnData,
    h5_path: str | Path,
    celltype_col: str = "cell_type",
) -> ad.AnnData:
    """Restore cell-type labels dropped by H5 orientation mismatch.

    ``core.data_loader._load_h5()`` drops ``singleCellLabels/values`` when
    the label count does not match the scRNA expression matrix after its
    dimension auto-detection.  This function reads labels directly from the
    H5 and reconstructs the AnnData with correct orientation.
    """
    if adata.obs is not None and celltype_col in adata.obs.columns:
        return adata

    h5_path = Path(h5_path)
    if not h5_path.exists():
        return adata

    with h5py.File(str(h5_path), "r") as f:
        if "singleCellLabels/values" not in f:
            return adata
        if "singleCellExpr/values" not in f:
            return adata

        sc_expr = f["singleCellExpr/values"][:].astype(np.float32)
        sc_rn = [s.decode() for s in f["singleCellExpr/rownames"][:]]
        sc_cn = [s.decode() for s in f["singleCellExpr/colnames"][:]]
        sc_labels = [s.decode() for s in f["singleCellLabels/values"][:]]

    n_labels = len(sc_labels)

    # Case 1: labels match adata.n_obs (already correct)
    if n_labels == adata.shape[0]:
        obs = adata.obs.copy() if adata.obs is not None else pd.DataFrame(index=adata.obs_names)
        obs[celltype_col] = sc_labels
        adata.obs = obs
        return adata

    # Case 2: labels match H5 dim0, meaning the H5 is (n_cells, n_genes)
    # but _load_h5 swapped names wrong. Rebuild AnnData.
    # Standard format: (n_cells, n_genes), rownames=genes, colnames=cells.
    if n_labels == sc_expr.shape[0]:
        # Determine which name array matches which dimension
        # In standard format: var_names = rownames (genes), obs_names = colnames (cells)
        n_cells, n_genes = sc_expr.shape
        # Check if name arrays match their intended dimensions
        if len(sc_rn) == n_genes:
            obs_names, var_names = sc_cn, sc_rn
        elif len(sc_cn) == n_genes:
            obs_names, var_names = sc_rn, sc_cn
        else:
            return adata  # can't determine orientation

        # Verify counts match
        if len(obs_names) != n_cells or len(var_names) != n_genes:
            return adata

        new_adata = ad.AnnData(
            X=sc_expr,
            obs=pd.DataFrame({celltype_col: sc_labels}, index=obs_names),
            var=pd.DataFrame(index=var_names),
        )
        return new_adata

    # Case 3: labels match H5 dim1 (n_genes, n_cells format) — rebuild transposed
    if n_labels == sc_expr.shape[1]:
        sc_expr_t = sc_expr.T
        # (n_cells, n_genes) after transpose
        if len(sc_rn) == sc_expr_t.shape[1]:
            obs_names, var_names = sc_cn, sc_rn
        elif len(sc_cn) == sc_expr_t.shape[1]:
            obs_names, var_names = sc_rn, sc_cn
        else:
            return adata

        new_adata = ad.AnnData(
            X=sc_expr_t,
            obs=pd.DataFrame({celltype_col: sc_labels}, index=obs_names),
            var=pd.DataFrame(index=var_names),
        )
        return new_adata

    return adata
