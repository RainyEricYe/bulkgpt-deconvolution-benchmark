import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy.sparse import issparse
from torch.utils.data import Dataset


# ── Preprocessor ──────────────────────────────────────────────────────


class Preprocessor:
    """Preprocess single-cell data for deconvolution reference.

    Applies the standard pipeline:
    1. Filter genes by counts (preserve genes with >=3 total counts)
    2. Normalize total counts (default 1e4)
    3. Log1p transform
    4. Clip NaN/Inf -> 0
    5. Select highly variable genes (HVG)

    This is a standalone version that does NOT depend on the scgpt package.
    Binning/encoding is handled separately by method-specific code.
    """

    def __init__(
        self,
        n_hvg: int = 1200,
        normalize_total: float = 1e4,
        log1p: bool = True,
        hvg_flavor: str = "seurat_v3",
    ):
        self.n_hvg = n_hvg
        self.normalize_total = normalize_total
        self.log1p = log1p
        self.hvg_flavor = hvg_flavor

    def __call__(
        self, adata: AnnData, batch_key: Optional[str] = None
    ) -> AnnData:
        adata = adata.copy()

        # preserve raw counts
        if adata.raw is None:
            adata.raw = adata

        # filter -> normalize -> log1p
        sc.pp.filter_genes(adata, min_counts=3)

        if self.normalize_total is not None:
            sc.pp.normalize_total(adata, target_sum=self.normalize_total)
        if self.log1p:
            sc.pp.log1p(adata)

        # clip NaN/Inf
        X = adata.X
        if hasattr(X, "data"):
            mask = ~np.isfinite(X.data)
            if mask.any():
                n_bad = mask.sum()
                X.data[mask] = 0.0
                if n_bad > 10:
                    warnings.warn(
                        f"Clipped {n_bad} NaN/Inf expression values to 0 "
                        f"({n_bad / X.data.size:.2%} of matrix)"
                    )
        else:
            mask = ~np.isfinite(X)
            if mask.any():
                n_bad = mask.sum()
                X[mask] = 0.0
                if n_bad > 10:
                    warnings.warn(
                        f"Clipped {n_bad} NaN/Inf expression values to 0 "
                        f"({n_bad / X.size:.2%} of matrix)"
                    )

        # HVG selection via scanpy
        n_hvg = min(self.n_hvg, adata.shape[1])
        if n_hvg < adata.shape[1]:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_hvg,
                flavor=self.hvg_flavor,
                batch_key=batch_key,
            )
            adata = adata[:, adata.var.highly_variable].copy()

        return adata


# ── Dataset ───────────────────────────────────────────────────────────


class PseudoBulkDataset(Dataset):
    """Dataset for pseudo-bulk training data.

    Supports scGPT (binned/continuous) and Geneformer (rank-value) encodings.
    """

    def __init__(
        self,
        bulk_matrix: np.ndarray,
        proportions: np.ndarray,
        gene_ids: np.ndarray,
        vocab,
        max_len: int = 1201,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        cls_token: str = "<cls>",
        include_zero_gene: bool = True,
        is_binned: bool = False,
        is_rank_value: bool = False,
        sorted_indices: Optional[np.ndarray] = None,
    ):
        self.bulk_matrix = bulk_matrix
        self.proportions = proportions
        self.gene_ids = gene_ids
        self.vocab = vocab
        self.max_len = max_len
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.cls_id = vocab[cls_token]
        self.include_zero_gene = include_zero_gene
        self.pad_id = vocab[pad_token]
        self.is_binned = is_binned
        self.is_rank_value = is_rank_value
        self.sorted_indices = sorted_indices

    def __len__(self):
        return len(self.bulk_matrix)

    def __getitem__(self, idx: int) -> dict:
        row = self.bulk_matrix[idx]
        props = self.proportions[idx]

        if self.include_zero_gene:
            if self.is_rank_value and self.sorted_indices is not None:
                si = self.sorted_indices[idx]
                values = torch.from_numpy(row).float()
                genes = torch.from_numpy(self.gene_ids[si]).long()
            else:
                values = torch.from_numpy(row).long() if self.is_binned else torch.from_numpy(row).float()
                genes = torch.from_numpy(self.gene_ids.copy()).long()
        else:
            nonzero = np.nonzero(row)[0]
            if self.is_rank_value and self.sorted_indices is not None:
                si = self.sorted_indices[idx]
                values = torch.from_numpy(row[nonzero]).float()
                genes = torch.from_numpy(self.gene_ids[si[nonzero]]).long()
            else:
                values = torch.from_numpy(row[nonzero]).long() if self.is_binned else torch.from_numpy(row[nonzero]).float()
                genes = torch.from_numpy(self.gene_ids[nonzero]).long()

        # prepend <cls> token
        genes = torch.cat([torch.tensor([self.cls_id]), genes])
        values = torch.cat([torch.tensor([self.pad_value]), values])

        # truncate if needed
        if len(genes) > self.max_len:
            idx_seq = torch.randperm(len(genes) - 1)[: self.max_len - 1] + 1
            idx_seq = torch.cat([torch.tensor([0]), idx_seq])
            genes = genes[idx_seq]
            values = values[idx_seq]

        # pad
        if len(genes) < self.max_len:
            pad_size = self.max_len - len(genes)
            genes = torch.cat([genes, torch.full((pad_size,), self.pad_id, dtype=torch.long)])
            pad_val = torch.tensor(self.pad_value, dtype=torch.long if self.is_binned else torch.float)
            values = torch.cat([values, pad_val.repeat(pad_size)])

        return {
            "gene_ids": genes,
            "values": values,
            "proportions": torch.from_numpy(props).float(),
            "src_key_padding_mask": genes == self.pad_id,
        }


# ── Pseudo-bulk generation ────────────────────────────────────────────


def prepare_pseudo_bulk(
    adata: AnnData,
    celltype_col: str = "cell_type",
    n_samples: int = 10000,
    alpha: float = 1.0,
    layer: Optional[str] = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Generate pseudo-bulk RNA-seq samples from single-cell reference.

    For each sample:
    1. Draw cell type proportions from Dirichlet(alpha)
    2. Randomly sample cells according to proportions
    3. Sum raw counts to create a pseudo-bulk profile
    4. CPM normalize then log1p transform

    Uses raw counts from ``adata.raw`` when available, falling back to
    ``adata.X``.

    Returns:
        bulk_matrix: (n_samples, n_genes) pseudo-bulk expression (log1p CPM)
        proportions: (n_samples, n_cell_types) true proportions
        cell_types: list of cell type names
    """
    adata = adata.copy()
    rng = np.random.default_rng(seed)

    if layer is not None:
        X = adata.layers[layer].toarray() if issparse(adata.layers[layer]) else np.asarray(adata.layers[layer])
    elif adata.raw is not None:
        X_raw = adata.raw[:, adata.var_names].X
        X = X_raw.toarray() if issparse(X_raw) else np.asarray(X_raw)
        print("Using raw counts from adata.raw for pseudo-bulk generation")
    else:
        X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)

    if not isinstance(adata.obs[celltype_col].dtype, pd.CategoricalDtype):
        adata.obs[celltype_col] = adata.obs[celltype_col].astype("category")
    # Now safe to modify adata since we copied at function entry
    cell_types = adata.obs[celltype_col].cat.categories
    n_types = len(cell_types)

    type_indices = {
        ct: np.where(adata.obs[celltype_col] == ct)[0]
        for ct in cell_types
    }
    type_indices = {ct: idx for ct, idx in type_indices.items() if len(idx) > 0}
    cell_types_list = list(type_indices.keys())
    n_types = len(cell_types_list)

    n_genes = X.shape[1]
    bulk_matrix = np.zeros((n_samples, n_genes), dtype=np.float64)
    proportions = np.zeros((n_samples, n_types), dtype=np.float64)

    for i in range(n_samples):
        props = rng.dirichlet([alpha] * n_types)
        proportions[i] = props

        for j, ct in enumerate(cell_types_list):
            n_cells = max(1, int(props[j] * 50))
            indices = rng.choice(type_indices[ct], size=min(n_cells, len(type_indices[ct])), replace=False)
            bulk_matrix[i] += X[indices].sum(axis=0)

    row_sums = bulk_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    bulk_matrix = (bulk_matrix / row_sums) * 1e4
    bulk_matrix = np.log1p(bulk_matrix)

    return bulk_matrix, proportions, cell_types_list


# ── Real-bulk dataset (for unsupervised domain adaptation) ────────────


class RealBulkDataset(Dataset):
    """Dataset for real-bulk expression (no ground-truth proportions).

    Produces the same dict format as ``PseudoBulkDataset`` but without
    the ``proportions`` key.  Used as target-domain data during
    unsupervised domain adaptation.
    """

    def __init__(
        self,
        bulk_matrix: np.ndarray,
        gene_ids: np.ndarray,
        vocab,
        max_len: int = 1201,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        cls_token: str = "<cls>",
        include_zero_gene: bool = True,
        is_binned: bool = False,
        is_rank_value: bool = False,
        sorted_indices: Optional[np.ndarray] = None,
    ):
        self.bulk_matrix = bulk_matrix
        self.gene_ids = gene_ids
        self.vocab = vocab
        self.max_len = max_len
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.cls_id = vocab[cls_token]
        self.include_zero_gene = include_zero_gene
        self.pad_id = vocab[pad_token]
        self.is_binned = is_binned
        self.is_rank_value = is_rank_value
        self.sorted_indices = sorted_indices

    def __len__(self):
        return len(self.bulk_matrix)

    def __getitem__(self, idx: int) -> dict:
        row = self.bulk_matrix[idx]

        if self.include_zero_gene:
            if self.is_rank_value and self.sorted_indices is not None:
                si = self.sorted_indices[idx]
                values = torch.from_numpy(row).float()
                genes = torch.from_numpy(self.gene_ids[si]).long()
            else:
                values = torch.from_numpy(row).long() if self.is_binned else torch.from_numpy(row).float()
                genes = torch.from_numpy(self.gene_ids.copy()).long()
        else:
            nonzero = np.nonzero(row)[0]
            if self.is_rank_value and self.sorted_indices is not None:
                si = self.sorted_indices[idx]
                values = torch.from_numpy(row[nonzero]).float()
                genes = torch.from_numpy(self.gene_ids[si[nonzero]]).long()
            else:
                values = torch.from_numpy(row[nonzero]).long() if self.is_binned else torch.from_numpy(row[nonzero]).float()
                genes = torch.from_numpy(self.gene_ids[nonzero]).long()

        # prepend <cls> token
        genes = torch.cat([torch.tensor([self.cls_id]), genes])
        values = torch.cat([torch.tensor([self.pad_value]), values])

        # truncate if needed
        if len(genes) > self.max_len:
            idx_seq = torch.randperm(len(genes) - 1)[: self.max_len - 1] + 1
            idx_seq = torch.cat([torch.tensor([0]), idx_seq])
            genes = genes[idx_seq]
            values = values[idx_seq]

        # pad
        if len(genes) < self.max_len:
            pad_size = self.max_len - len(genes)
            genes = torch.cat([genes, torch.full((pad_size,), self.pad_id, dtype=torch.long)])
            pad_val = torch.tensor(self.pad_value, dtype=torch.long if self.is_binned else torch.float)
            values = torch.cat([values, pad_val.repeat(pad_size)])

        return {
            "gene_ids": genes,
            "values": values,
            "src_key_padding_mask": genes == self.pad_id,
        }


def filter_genes_to_vocab(adata: AnnData, vocab) -> AnnData:
    """Drop genes not recognized by *vocab* and merge duplicate gene symbols.

    Must be called before HVG selection so all selected HVGs are usable.
    Duplicate gene symbols are merged by summing expression values
    (preserving total UMI count per gene).

    Args:
        adata: AnnData with gene identifiers as ``var_names``.
        vocab: A container that supports ``g in vocab`` membership check
            (e.g., ``set``, ``dict``, ``GeneVocab``).

    Returns:
        Filtered AnnData containing only unique genes present in *vocab*.
    """
    gene_names = list(adata.var_names)

    # Step 1: Filter to vocab
    keep_mask = [g in vocab for g in gene_names]
    adata = adata[:, keep_mask].copy()
    gene_names = list(adata.var_names)

    n_dropped = len(keep_mask) - sum(keep_mask)
    if n_dropped:
        print(f"After vocab filter: {len(gene_names)}/{len(keep_mask)} genes retained ({n_dropped} dropped)")

    # Step 2: Merge duplicate gene symbols (sum expression)
    col_groups = {}
    first_occs = []
    dup_targets = []
    for i, name in enumerate(gene_names):
        if name not in col_groups:
            col_groups[name] = i
            first_occs.append(i)
        else:
            dup_targets.append((name, i))

    n_dups = len(dup_targets)
    if n_dups > 0:
        print(f"Merging {n_dups} duplicate gene symbols (summing expression)")

        X_new = adata.X[:, first_occs].copy()
        for name, src_col in dup_targets:
            target_pos = list(col_groups.keys()).index(name)
            X_new[:, target_pos] = X_new[:, target_pos] + adata.X[:, src_col]

        unique_names = list(col_groups.keys())
        adata = AnnData(
            X=X_new,
            obs=adata.obs,
            var=adata.var.iloc[first_occs],
        )
        adata.var_names = unique_names
        print(f"  Final unique genes: {len(unique_names)}")

    return adata
