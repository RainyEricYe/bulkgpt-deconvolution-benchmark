#!/usr/bin/env python3
"""
Count Bridges container entrypoint.

Reads DeconBenchmark H5 input, builds signature matrix from scRNA reference,
runs Poisson bridge EM deconvolution on bulk samples, writes output.

Inputs: bulk, singleCellExpr, singleCellLabels
Output: P (proportions matrix in DeconBenchmark format)
"""
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import h5py

warnings.filterwarnings("ignore")

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/args.h5")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.h5")

METHOD_NAME = "CountBridges"
SEED = 42


# ============================================================================
# Count Bridges EM Deconvolution
# ============================================================================


class CountBridgesDeconvolution:
    """Poisson bridge EM deconvolution following the Count Bridges framework."""

    def __init__(
        self,
        em_max_iter: int = 50,
        em_tol: float = 1e-4,
        bridge_strength: float = 1.0,
        normalization: str = "cpm",
        verbose: bool = True,
    ):
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.bridge_strength = bridge_strength
        self.normalization = normalization
        self.verbose = verbose

        self._sig_matrix: Optional[pd.DataFrame] = None
        self._cell_types: Optional[List[str]] = None

    def fit(self, sc_expr: np.ndarray, sc_labels: List[str],
            gene_names: List[str]) -> None:
        """Build signature matrix from scRNA expression and labels."""
        cell_types = sorted(set(sc_labels))
        n_cells, n_genes = sc_expr.shape
        n_types = len(cell_types)

        if self.verbose:
            print(f"  Building signature: {n_cells} cells, {n_genes} genes, "
                  f"{n_types} types")

        labels_arr = np.array(sc_labels)
        sig_dict: Dict[str, np.ndarray] = {}
        for ct in cell_types:
            mask = labels_arr == ct
            sig_dict[ct] = sc_expr[mask].mean(axis=0)

        self._sig_matrix = pd.DataFrame(
            sig_dict, index=gene_names, dtype=np.float64,
        )
        self._cell_types = cell_types

    def predict(self, bulk_expr: pd.DataFrame) -> pd.DataFrame:
        """Predict proportions using Count Bridges EM deconvolution.

        Parameters
        ----------
        bulk_expr : pd.DataFrame
            Bulk expression with genes as index, samples as columns.
        """
        if self._sig_matrix is None or self._cell_types is None:
            raise RuntimeError("Call fit() before predict().")

        sig = self._sig_matrix.copy()
        cell_types = self._cell_types
        n_types = len(cell_types)

        common = sig.index.intersection(bulk_expr.index)
        if len(common) == 0:
            raise ValueError("No common genes between signature and bulk.")

        sig = sig.loc[common]
        bulk_aligned = bulk_expr.loc[common]

        if self.verbose:
            coverage = len(common) / len(self._sig_matrix)
            print(f"  Gene alignment: {len(common)}/{len(self._sig_matrix)} "
                  f"({coverage:.1%})")

        S = sig.values.astype(np.float64)
        Y = bulk_aligned.values.astype(np.float64)
        n_genes, n_samples = Y.shape

        if self.normalization == "cpm":
            lib_sizes = Y.sum(axis=0, keepdims=True)
            lib_sizes = np.maximum(lib_sizes, 1.0)
            Y = Y / lib_sizes * 1e6

        sig_col_sums = S.sum(axis=0, keepdims=True)
        sig_col_sums = np.maximum(sig_col_sums, 1.0)
        S_norm = S / sig_col_sums

        sample_names = bulk_aligned.columns.tolist()
        proportions = np.zeros((n_samples, n_types), dtype=np.float64)

        for i in range(n_samples):
            proportions[i] = self._em_deconvolve(Y[:, i], S_norm, n_types)

        result = pd.DataFrame(
            proportions, index=sample_names, columns=cell_types,
        )
        result.index.name = "sample"
        return result

    def _em_deconvolve(self, y: np.ndarray, S: np.ndarray,
                       n_types: int) -> np.ndarray:
        """Run EM deconvolution for a single bulk sample."""
        eps = 1e-12
        p = np.full(n_types, 1.0 / n_types, dtype=np.float64)
        alpha_prior = self.bridge_strength / n_types

        for iteration in range(self.em_max_iter):
            p_old = p.copy()
            lam = S * p[np.newaxis, :]
            lam_sum = lam.sum(axis=1, keepdims=True) + eps
            expected_contrib = y[:, np.newaxis] * (lam / lam_sum)
            total_per_type = expected_contrib.sum(axis=0)
            p = total_per_type + alpha_prior
            p = np.maximum(p, eps)
            p = p / p.sum()
            if np.abs(p - p_old).sum() < self.em_tol:
                break
        return p


# ============================================================================
# H5 I/O — handles enriched DeconBenchmark H5 format
# ============================================================================
#
# The enriched H5 follows DeconBenchmark convention:
#   Matrix stored as (observations, features) in H5:
#     - singleCellExpr: (n_cells, n_genes),   rownames=genes, colnames=cells
#     - bulk:           (n_samples, n_genes),  rownames=genes, colnames=samples
#   Names are always stored as rownames (features) and colnames (observations),
#   matching the DeconUtils convention used by SIF container runners.
#
# Since rownames are always genes and colnames are always observations,
# dimension-matching is used to determine orientation instead of heuristic
# gene-name detection.


def _decode_names(names) -> List[str]:
    """Decode HDF5 variable-length strings to plain Python strings."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in names]


def read_h5_input(path: str) -> dict:
    """Read DeconBenchmark H5 input.

    Returns dict with standardized orientation:
        bulk_values:      (n_samples, n_genes) float matrix
        bulk_gene_names:  list of gene names (length n_genes)
        bulk_sample_names: list of sample names (length n_samples)
        sce_values:       (n_cells, n_genes) float matrix
        sce_gene_names:   list of gene names (length n_genes)
        sce_labels:       list of cell type labels (length n_cells)
    """
    f = h5py.File(path, "r")

    # ── scRNA expression ────────────────────────────────────────────
    sce_raw = f["singleCellExpr/values"][:]
    sce_rownames = _decode_names(f["singleCellExpr/rownames"][:])
    sce_colnames = _decode_names(f["singleCellExpr/colnames"][:])

    n_row, n_col = sce_raw.shape

    # Determine orientation by matching dimension counts to name lengths.
    # DeconBenchmark convention: rownames=genes, colnames=cells.
    if n_row == len(sce_colnames) and n_col == len(sce_rownames):
        # (n_cells, n_genes) — rownames are genes, colnames are cells
        sce_values = np.asarray(sce_raw, dtype=np.float64)  # (n_cells, n_genes)
        sce_gene_names = sce_rownames
    elif n_row == len(sce_rownames) and n_col == len(sce_colnames):
        # (n_genes, n_cells) — standard R h5read orientation
        sce_values = np.asarray(sce_raw.T, dtype=np.float64)  # -> (n_cells, n_genes)
        sce_gene_names = sce_rownames
    else:
        # Ambiguous — fallback to enriched format: (n_cells, n_genes)
        sce_values = np.asarray(sce_raw, dtype=np.float64)
        sce_gene_names = sce_rownames if n_col == len(sce_rownames) else sce_colnames

    # ── Cell type labels ────────────────────────────────────────────
    sce_labels_raw = f["singleCellLabels/values"][:]
    sce_labels = _decode_names(sce_labels_raw)

    n_cells, n_sc_genes = sce_values.shape
    if len(sce_labels) != n_cells:
        # Sometimes the label count is stored for n_genes instead of n_cells
        # (another H5 orientation quirk). Truncate to match actual cells.
        sce_labels = sce_labels[:n_cells]

    # ── Bulk expression ─────────────────────────────────────────────
    bulk_raw = f["bulk/values"][:]
    bulk_rownames = _decode_names(f["bulk/rownames"][:])
    bulk_colnames = _decode_names(f["bulk/colnames"][:])

    n_row_b, n_col_b = bulk_raw.shape

    # Determine bulk orientation
    if n_row_b == len(bulk_colnames) and n_col_b == len(bulk_rownames):
        # (n_samples, n_genes) — rownames are genes, colnames are samples
        bulk_values = np.asarray(bulk_raw, dtype=np.float64)
        bulk_gene_names = bulk_rownames
        bulk_sample_names = bulk_colnames
    elif n_row_b == len(bulk_rownames) and n_col_b == len(bulk_colnames):
        # (n_genes, n_samples)
        bulk_values = np.asarray(bulk_raw.T, dtype=np.float64)
        bulk_gene_names = bulk_rownames
        bulk_sample_names = bulk_colnames
    else:
        # Ambiguous — treat as (n_samples, n_genes)
        bulk_values = np.asarray(bulk_raw, dtype=np.float64)
        bulk_gene_names = bulk_rownames
        bulk_sample_names = bulk_colnames

    f.close()

    n_samples, n_bulk_genes = bulk_values.shape
    return {
        "bulk_values": bulk_values,
        "bulk_gene_names": bulk_gene_names,
        "bulk_sample_names": bulk_sample_names,
        "sce_values": sce_values,
        "sce_gene_names": sce_gene_names,
        "sce_labels": sce_labels,
    }


def write_h5_output(path: str, proportions_df: pd.DataFrame,
                    method: str = METHOD_NAME) -> None:
    """Write proportions to H5 in DeconBenchmark format."""
    if os.path.exists(path):
        os.remove(path)
    f = h5py.File(path, "w")
    grp = f.create_group("P")
    grp.create_dataset("values", data=proportions_df.values.T)
    grp.create_dataset(
        "rownames",
        data=np.array(proportions_df.index.tolist(), dtype="S"),
    )
    grp.create_dataset(
        "colnames",
        data=np.array(proportions_df.columns.tolist(), dtype="S"),
    )
    f.close()
    print(f"{method}: Wrote -> {path}")


# ============================================================================
# Main
# ============================================================================


def main():
    np.random.seed(SEED)
    print(f"\n{METHOD_NAME} — Poisson Bridge EM Deconvolution")
    print(f"{'=' * 50}")
    print(f"  Input:  {INPUT_PATH}")
    print(f"  Output: {OUTPUT_PATH}")

    # Read H5 input
    print(f"\n[1] Reading input...")
    data = read_h5_input(INPUT_PATH)
    sce_values = data["sce_values"]
    sce_labels = data["sce_labels"]
    sce_gene_names = data["sce_gene_names"]
    bulk_values = data["bulk_values"]
    bulk_gene_names = data["bulk_gene_names"]
    bulk_sample_names = data["bulk_sample_names"]

    n_cells, n_sc_genes = sce_values.shape
    n_samples, n_bulk_genes = bulk_values.shape
    n_types = len(set(sce_labels))
    print(f"  scRNA: {n_cells} cells x {n_sc_genes} genes, {n_types} types")
    print(f"  Bulk:  {n_samples} samples x {n_bulk_genes} genes")

    # Align gene sets
    common_genes = sorted(set(sce_gene_names) & set(bulk_gene_names))
    if len(common_genes) == 0:
        raise ValueError("No common genes between bulk and scRNA.")

    print(f"  Common genes: {len(common_genes)} "
          f"(scRNA: {n_sc_genes}, bulk: {n_bulk_genes})")

    sce_gene_idx = [sce_gene_names.index(g) for g in common_genes]
    sce_values = sce_values[:, sce_gene_idx]

    bulk_gene_idx = [bulk_gene_names.index(g) for g in common_genes]
    bulk_values = bulk_values[:, bulk_gene_idx]

    # Build signature matrix
    print(f"\n[2] Building signature matrix...")
    model = CountBridgesDeconvolution(
        em_max_iter=50,
        em_tol=1e-4,
        bridge_strength=1.0,
        normalization="cpm",
        verbose=True,
    )
    model.fit(sce_values, sce_labels, common_genes)

    # Predict
    print(f"\n[3] Running EM deconvolution...")
    bulk_expr = pd.DataFrame(
        bulk_values.T, index=common_genes, columns=bulk_sample_names,
    )
    pred_props = model.predict(bulk_expr)
    print(f"  Predictions: {pred_props.shape[0]} samples x "
          f"{pred_props.shape[1]} types")

    # Write output
    print(f"\n[4] Writing output...")
    write_h5_output(OUTPUT_PATH, pred_props)

    print(f"\n{METHOD_NAME}: Done!\n")


if __name__ == "__main__":
    main()
