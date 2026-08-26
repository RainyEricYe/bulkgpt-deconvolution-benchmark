#!/usr/bin/env python3
"""Gene alignment utilities for BulkFormer.

BulkFormer uses a fixed 20,010-gene vocabulary derived from TCGA + GTEx.
Input datasets must align their gene symbols to this vocabulary.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

BULKFORMER_DIR = Path(
    os.environ.get("BULKFORMER_DIR", os.path.expanduser("~/data/public/BulkFormer"))
)


def load_bulkformer_gene_list() -> list[str]:
    """Return the 20,010-gene vocabulary used by BulkFormer-147M."""
    gi = pd.read_csv(BULKFORMER_DIR / "data" / "bulkformer_gene_info.csv")
    return list(gi.iloc[:, 0])


def align_genes_to_bulkformer(
    gene_symbols: list[str],
    fill_value: float = -10.0,
) -> tuple[dict[int, int], int]:
    """Map input gene symbols → BulkFormer vocabulary indices.

    Args:
        gene_symbols: Input gene names.
        fill_value: Unused; kept for API compatibility.

    Returns:
        ``(gene_to_idx, n_matched)`` where *gene_to_idx* maps input column
        index → BulkFormer vocabulary index.
    """
    bulkformer_genes = load_bulkformer_gene_list()
    bf_index = {g.upper(): i for i, g in enumerate(bulkformer_genes)}
    gene_to_idx: dict[int, int] = {}
    for i, g in enumerate(gene_symbols):
        idx = bf_index.get(str(g).upper())
        if idx is not None:
            gene_to_idx[i] = idx
    return gene_to_idx, len(gene_to_idx)


def build_bulkformer_matrix(
    raw_counts: np.ndarray,
    gene_symbols: list[str],
    fill_value: float = -10.0,
) -> tuple[np.ndarray, int]:
    """Align raw expression to the BulkFormer gene vocabulary.

    Args:
        raw_counts: (n_samples, n_input_genes).
        gene_symbols: Gene names matching columns.
        fill_value: Value for missing genes (default -10.0).

    Returns:
        ``(aligned_matrix, n_matched)`` where *aligned_matrix* has shape
        (n_samples, 20010).
    """
    n_samples = raw_counts.shape[0]
    gene_to_idx, n_matched = align_genes_to_bulkformer(gene_symbols)
    mat = np.full((n_samples, 20010), fill_value, dtype=np.float32)
    for src_idx, tgt_idx in gene_to_idx.items():
        mat[:, tgt_idx] = raw_counts[:, src_idx]
    return mat, n_matched
