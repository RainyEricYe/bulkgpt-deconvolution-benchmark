#!/usr/bin/env python3
"""Unified data loading for bulk RNA-seq deconvolution.

Auto-detects input format and returns a DataBundle with:
  - sc_ref:   AnnData (or None for signature-matrix methods)
  - bulk:     pd.DataFrame  (samples x genes)
  - gt:       pd.DataFrame | None  (samples x cell_types)

Supported formats:
  .h5ad        AnnData (single-cell or bulk)
  .h5          DeconBenchmark H5 (bulk/ + singleCellExpr/ + singleCellLabels/)
  .tsv/.csv    Delimited text (genes x samples)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

SUPPORTED_FORMATS = (".h5ad", ".h5", ".h5seurat", ".tsv", ".csv", ".tsv.gz", ".csv.gz")
CELLTYPE_COL_PRIORITY = [
    "cell_type", "CellType", "celltype", "cell_ontology_class",
    "label", "cluster", "Cell_class", "subclass",
]


@dataclass
class DataBundle:
    sc_ref: Optional["AnnData"] = None
    bulk: Optional[pd.DataFrame] = None
    gt: Optional[pd.DataFrame] = None
    cell_types: Optional[list[str]] = None

    def is_valid(self) -> bool:
        return self.bulk is not None


def load_data(path: Union[str, Path], ground_truth: Optional[Union[str, Path]] = None) -> DataBundle:
    """Load data from *path*, auto-detecting the format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    ext = _get_ext(path)
    bundle = DataBundle()

    if ext == ".h5":
        _load_h5(path, bundle)
    elif ext == ".h5ad":
        _load_h5ad(path, bundle)
    elif ext in (".tsv", ".csv", ".tsv.gz", ".csv.gz"):
        bundle.bulk = _load_delimited(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    if ground_truth:
        gt_path = Path(ground_truth)
        if gt_path.exists():
            bundle.gt = _load_delimited(gt_path)

    return bundle


def _get_ext(path: Path) -> str:
    for ext in SUPPORTED_FORMATS:
        if path.name.endswith(ext):
            return ext
    return path.suffix


def _load_h5ad(path: Path, bundle: DataBundle) -> None:
    """Load an h5ad file -- either a single-cell reference or bulk."""
    import anndata

    adata = anndata.read_h5ad(path)
    is_sc = adata.n_obs > 100 or any(
        c in adata.obs.columns for c in CELLTYPE_COL_PRIORITY
    )

    if is_sc:
        bundle.sc_ref = adata
        bundle.cell_types = _detect_celltype_col(adata)
    else:
        bundle.bulk = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X),
            index=adata.obs_names,
            columns=adata.var_names,
        )


def _load_h5(path: Path, bundle: DataBundle) -> None:
    """Load a canonical-format H5 file.

    Assumes standard conventions (no auto-detection):
      bulk/values:            (n_samples, n_genes)
      singleCellExpr/values:  (n_cells,  n_genes)
      ground_truth/values:    (n_types,  n_samples)
      bulkRatio/values:       (n_samples, n_types)

    Name arrays (rownames/colnames) are matched to dimensions by length.
    """
    import h5py

    with h5py.File(path, "r") as f:
        # ── Bulk expression ────────────────────────────────────────────
        # Standard: values=(n_samples, n_genes), rownames=genes, colnames=samples
        if "bulk/values" in f:
            bulk_raw = f["bulk/values"][:]
            rn = [s.decode() for s in f["bulk/rownames"][:]] if "bulk/rownames" in f else None
            cn = [s.decode() for s in f["bulk/colnames"][:]] if "bulk/colnames" in f else None
            bundle.bulk = pd.DataFrame(bulk_raw, index=cn, columns=rn)

        # ── Single-cell reference ───────────────────────────────────────
        if "singleCellExpr/values" in f:
            import anndata
            import numpy as np

            sc_expr = f["singleCellExpr/values"][:].astype(np.float32)
            sc_rn = [g.decode() for g in f["singleCellExpr/rownames"][:]] if "singleCellExpr/rownames" in f else None
            sc_cn = [b.decode() for b in f["singleCellExpr/colnames"][:]] if "singleCellExpr/colnames" in f else None
            sc_labels = [s.decode() for s in f["singleCellLabels/values"][:]] if "singleCellLabels/values" in f else None

            # DeconBenchmark convention (accounting for rhdf5 transpose):
            #   H5 stores values=(n_cells, n_genes), rownames=genes, colnames=cells
            #   R reads transposed: (n_genes, n_cells) with rownames=genes, colnames=cells
            #   Python reads as-is: obs_names=colnames, var_names=rownames
            n_cells, n_genes = sc_expr.shape
            obs_index = sc_cn      # colnames = cell barcodes (n_cells)
            var_index = sc_rn      # rownames = gene names (n_genes)

            bundle.sc_ref = anndata.AnnData(
                X=sc_expr,
                obs=pd.DataFrame({"cell_type": sc_labels}, index=obs_index) if sc_labels else (
                    pd.DataFrame(index=obs_index) if obs_index is not None else None
                ),
                var=pd.DataFrame(index=var_index) if var_index is not None else None,
            )
            if sc_labels:
                bundle.cell_types = sorted(set(sc_labels))

        # ── Ground truth (2_real_bulk) — (n_types, n_samples) ──────────
        if "ground_truth/values" in f:
            gt = f["ground_truth/values"][:]
            gt_rows = [t.decode() for t in f["ground_truth/rownames"][:]] if "ground_truth/rownames" in f else None
            gt_cols = [s.decode() for s in f["ground_truth/colnames"][:]] if "ground_truth/colnames" in f else None
            bundle.gt = pd.DataFrame(gt.T, index=gt_cols, columns=gt_rows)

        # ── Bulk ratio GT (1_pseudo_bulk) — (n_samples, n_types) ───────
        if "bulkRatio/values" in f:
            br = f["bulkRatio/values"][:]
            br_types = [t.decode() for t in f["bulkRatio/rownames"][:]] if "bulkRatio/rownames" in f else None
            br_samples = [s.decode() for s in f["bulkRatio/colnames"][:]] if "bulkRatio/colnames" in f else None
            bundle.gt = pd.DataFrame(br, index=br_samples, columns=br_types)


def _load_delimited(path: Path) -> pd.DataFrame:
    """Load a TSV/CSV into a DataFrame."""
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path, sep="\t", index_col=0)


def _detect_celltype_col(adata: "AnnData") -> Optional[list[str]]:
    """Find the cell-type column in *adata.obs* and return sorted types."""
    col = auto_detect_celltype_col(adata.obs.columns)
    if col:
        return sorted(adata.obs[col].unique())
    return None


def auto_detect_celltype_col(obs_columns) -> Optional[str]:
    """Return the name of the cell-type column in *obs_columns*."""
    obs_set = set(obs_columns)
    for col in CELLTYPE_COL_PRIORITY:
        if col in obs_set:
            return col
    return None


try:
    import anndata
    from anndata import AnnData
except ImportError:
    AnnData = None  # type: ignore


def load_sc_ref(path_or_cfg: Union[str, Path, dict], key: str = "sc_ref") -> AnnData:
    """Load a single-cell reference, supporting H5 (DeconBenchmark) or h5ad.

    Parameters
    ----------
    path_or_cfg :
        Either a file path directly, or a dict (config section) from which
        ``data_path`` (H5) or *key* (h5ad) will be read.
    key :
        Config key for the h5ad path, used when ``data_path`` is not set.

    Returns
    -------
    AnnData with ``.obs['cell_type']`` populated.
    """
    if isinstance(path_or_cfg, dict):
        data_path = path_or_cfg.get("data_path")
        if data_path:
            return load_data(str(data_path)).sc_ref
        return anndata.read_h5ad(str(path_or_cfg[key]))
    path = Path(str(path_or_cfg))
    if path.suffix == ".h5":
        return load_data(str(path)).sc_ref
    return anndata.read_h5ad(str(path))
