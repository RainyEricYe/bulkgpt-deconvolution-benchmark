"""scGPT-specific data functions: gene mapping and binning."""

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from anndata import AnnData


def map_genes_to_symbols(
    adata: AnnData,
    vocab,
    mapping_path: str = "data/gene_symbol_map.json",
) -> AnnData:
    """Map Ensembl gene IDs to gene symbols.

    Priority:
    1. Use ``var['gene_name']`` column if it exists (always available from h5ad)
    2. Fall back to pre-computed mapping file (gene_symbol_map.json)

    Args:
        adata: AnnData with Ensembl IDs as ``var_names``
        vocab: Vocabulary for checking symbol coverage (supports ``g in vocab``)
        mapping_path: Path to JSON mapping file (Ensembl ID -> symbol)

    Returns:
        AnnData with ``var_names`` replaced by gene symbols.
    """
    gene_names = adata.var_names.tolist()

    # Quick check: are these already symbols?
    n_in_vocab = sum(1 for g in gene_names if g in vocab)
    n_ensembl = sum(1 for g in gene_names if g.startswith("ENSG") or g.startswith("ENSMUSG"))
    if n_in_vocab > len(gene_names) * 0.5 or n_ensembl < len(gene_names) * 0.5:
        print(f"Gene names appear to be symbols ({n_in_vocab}/{len(gene_names)} in vocab)")
        return adata

    # Priority 1: Use var['gene_name'] if available
    if "gene_name" in adata.var.columns:
        raw_names = adata.var["gene_name"].astype(str).tolist()
        new_names = []
        for i, name in enumerate(raw_names):
            if name and name != "nan":
                new_names.append(name)
            else:
                new_names.append(gene_names[i])

        seen = {}
        dup_count = 0
        for name in new_names:
            if name in seen:
                dup_count += 1
            else:
                seen[name] = True

        if dup_count > 0:
            print(f"  Warning: {dup_count} duplicate gene symbols found (will be merged by filter_genes_to_vocab)")

        adata.var_names = new_names
        print(f"Mapped {len(gene_names)} Ensembl IDs via var['gene_name']")
        print(f"  Now {sum(1 for g in new_names if g in vocab)}/{len(new_names)} in vocab")
        return adata

    # Priority 2: Use external mapping file
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Gene symbol mapping file not found: {mapping_path}. "
        )

    with open(mapping_path) as f:
        symbol_map = json.load(f)

    new_names = []
    mapped = 0
    for g in gene_names:
        if g in symbol_map:
            new_names.append(symbol_map[g])
            mapped += 1
        else:
            new_names.append(g)

    adata.var_names = new_names
    print(f"Mapped {mapped}/{len(gene_names)} Ensembl IDs to symbols (via mapping file)")
    print(f"  Now {sum(1 for g in new_names if g in vocab)}/{len(new_names)} in vocab")
    return adata


# ── Binning ──────────────────────────────────────────────────────────


def load_bin_info(bin_info_path: str) -> Dict[str, np.ndarray]:
    """Load per-gene bin boundaries from pretrained model's bin_info.csv.

    Returns:
        dict mapping gene name -> array of quantile boundaries (shape: (n_bins-1,))
    """
    df = pd.read_csv(bin_info_path, index_col=0)
    return {gene: df.loc[gene].values.astype(np.float32) for gene in df.index}


def bin_expression(
    expr_matrix: np.ndarray,
    gene_names: np.ndarray,
    bin_info: Dict[str, np.ndarray],
    n_bins: int = 51,
) -> np.ndarray:
    """Convert log1p(CPM) expression values to bin IDs (0..n_bins-1).

    Uses per-gene quantile boundaries. scGPT convention: index 0 = zero/padding,
    indices 1..n_bins-1 = bins. Genes not in bin_info get binarized.

    Args:
        expr_matrix: (n_samples, n_genes) expression values (log1p CPM)
        gene_names: (n_genes,) gene names
        bin_info: gene -> quantile boundaries (from ``load_bin_info``)
        n_bins: total number of categories including zero bin (default 51)

    Returns:
        binned: (n_samples, n_genes) integer bin IDs in range [0, n_bins-1]
    """
    binned = np.zeros_like(expr_matrix, dtype=np.int64)
    max_bin = n_bins - 1

    found = 0
    for j, gene in enumerate(gene_names):
        col = expr_matrix[:, j]
        nonzero = col > 0
        if gene in bin_info:
            boundaries = bin_info[gene]
            bin_ids = np.digitize(col, boundaries)
            bin_ids[~nonzero] = 0
            binned[:, j] = bin_ids
            found += 1
        else:
            binned[:, j] = nonzero.astype(np.int64)

    binned = np.clip(binned, 0, max_bin)
    if found > 0:
        print(f"Binned {found}/{len(gene_names)} genes using pretrained bin boundaries")
    return binned
