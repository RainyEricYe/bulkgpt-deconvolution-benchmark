"""Data loading and preprocessing for scGPT-LoRA fine-tuning.

Adapted from scPEFT ``deconv/lora_experiment.py``.
Relies on ``core.data_loader`` for all H5 parsing (unified format convention).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


_ENSG2SYM: dict[str, str] | None = None


def _load_ensg2sym() -> dict[str, str]:
    """Load Ensembl ID → HGNC symbol mapping from Geneformer's dict."""
    global _ENSG2SYM
    if _ENSG2SYM is not None:
        return _ENSG2SYM
    import pickle, os
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "weights",
                     "geneformer", "default", "..", "dicts",
                     "gene_name_id_dict.pkl"),
    ]
    for p in candidates:
        rp = os.path.realpath(p)
        if os.path.exists(rp):
            with open(rp, "rb") as f:
                raw = pickle.load(f)
            _ENSG2SYM = {v: k for k, v in raw.items()}
            print(f"Loaded Ensembl→HGNC mapping: {len(_ENSG2SYM)} entries from {rp}")
            return _ENSG2SYM
    print("WARNING: no gene_name_id_dict.pkl found; Ensembl→HGNC conversion disabled")
    _ENSG2SYM = {}
    return _ENSG2SYM


def _maybe_convert_ensembl_to_symbol(
    genes: list[str],
    label: str = "",
) -> list[str]:
    """If >50% of genes look like Ensembl IDs, convert to HGNC symbols."""
    n_ensg = sum(1 for g in genes if str(g).startswith("ENSG"))
    if n_ensg < len(genes) * 0.5:
        return genes  # already symbols
    mapping = _load_ensg2sym()
    converted = [mapping.get(str(g), str(g)) for g in genes]
    n_mapped = sum(1 for c, o in zip(converted, genes) if c != o)
    print(f"  [{label}] Ensembl→HGNC: {n_mapped}/{len(genes)} converted "
          f"({sum(1 for c in converted if not c.startswith('ENSG'))}/{len(genes)} symbols)")
    return converted


def load_h5_bulk(
    h5_path: str,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], list[str]]:
    """Load bulk expression, reference scRNA, and cell-type labels via the
    shared ``core.data_loader`` (DeconBenchmark H5 canonical format).

    Automatically converts Ensembl IDs → HGNC symbols when detected.

    Returns:
        (bulk_expr, bulk_genes, ref_expr, ref_genes, ref_labels)
    """
    from core.data_loader import load_data

    bundle = load_data(h5_path)

    if bundle.bulk is None:
        raise ValueError(f"No bulk data found in {h5_path}")
    bulk_expr = bundle.bulk.values.astype(np.float32)
    bulk_genes = _maybe_convert_ensembl_to_symbol(list(bundle.bulk.columns), "bulk")

    if bundle.sc_ref is None:
        raise ValueError(f"No single-cell reference found in {h5_path}")
    ref = bundle.sc_ref
    ref_expr = ref.X
    if hasattr(ref_expr, "toarray"):
        ref_expr = ref_expr.toarray().astype(np.float32)
    ref_genes = _maybe_convert_ensembl_to_symbol(list(ref.var_names), "sc_ref")

    # cell_type column populated by core.data_loader._load_h5()
    ref_labels = (
        ref.obs["cell_type"].tolist()
        if "cell_type" in ref.obs.columns
        else []
    )

    return bulk_expr, bulk_genes, ref_expr, ref_genes, ref_labels


def normalize_proportions(pred: np.ndarray) -> np.ndarray:
    """Clip negatives and renormalize to sum=1 per sample."""
    pred = np.maximum(pred, 0)
    row_sums = pred.sum(axis=1, keepdims=True)
    return pred / np.maximum(row_sums, 1e-10)


def compute_metrics(
    pred: np.ndarray, true: np.ndarray, names: list[str]
) -> dict:
    """Per-cell-type Pearson r and RMSE, plus macro_avg.

    Returns nested dict: {cell_type: {pearson_r, rmse}, macro_avg: {...}}.
    """
    from scipy.stats import pearsonr

    metrics = {}
    for i, name in enumerate(names):
        r, _ = pearsonr(pred[:, i], true[:, i])
        rmse = float(np.sqrt(np.mean((pred[:, i] - true[:, i]) ** 2)))
        metrics[name] = {"pearson_r": round(r, 4), "rmse": round(rmse, 4)}
    avg_r = np.mean([metrics[n]["pearson_r"] for n in names])
    avg_rmse = np.mean([metrics[n]["rmse"] for n in names])
    metrics["macro_avg"] = {"pearson_r": round(float(avg_r), 4), "rmse": round(float(avg_rmse), 4)}
    return metrics


class _BulkDataset(Dataset):
    """In-memory dataset for bulk samples with proportions."""

    def __init__(
        self,
        count_matrix: np.ndarray,
        gene_ids: np.ndarray,
        proportions: np.ndarray,
        vocab,
    ) -> None:
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.proportions = proportions
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.proportions)

    def __getitem__(self, idx: int) -> dict:
        row = self.count_matrix[idx]
        if hasattr(row, "A"):
            row = row.A.squeeze()
        nonzero = np.nonzero(row)[0]
        if len(nonzero) == 0:
            # All-zero row — substitute a single pad token to avoid
            # DataCollator binning crashing on zero-size array (#160).
            nonzero = np.array([0])  # index 0 → <pad>
            row_val = np.array([0.0])
        else:
            row_val = row[nonzero]
        genes = np.insert(self.gene_ids[nonzero], 0, self.vocab["<cls>"])
        values = np.insert(row_val, 0, 0)
        return {
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values).float(),
            "proportion": torch.from_numpy(self.proportions[idx]).float(),
        }


def _make_collator(vocab, max_length: int = 1200):
    """Build a collator that pads gene/expression sequences and preserves proportions."""
    from scgpt.data_collator import DataCollator

    base_collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab["<pad>"],
        pad_value=0,
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )

    def collate_fn(batch):
        collated = base_collator(batch)
        collated["proportion"] = torch.stack(
            [b["proportion"] for b in batch], dim=0
        )
        return collated

    return collate_fn


def get_loader(
    indices: np.ndarray,
    count_matrix: np.ndarray,
    gene_ids: np.ndarray,
    proportions: np.ndarray,
    vocab,
    batch_size: int = 32,
    max_length: int = 1200,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader from a subset of bulk samples."""
    ds = _BulkDataset(count_matrix[indices], gene_ids, proportions[indices], vocab)
    collator = _make_collator(vocab, max_length)
    n_workers = min(len(os.sched_getaffinity(0)), 4) if hasattr(os, "sched_getaffinity") else 2
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        drop_last=False,
        num_workers=n_workers,
        pin_memory=True,
    )



def get_sdy67_split(n: int = 250, train_n: int = 150, val_n: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed 6:2:2 split for SDY67: first 150 train, next 50 val, last 50 test."""
    indices = np.arange(n)
    return indices[:train_n], indices[train_n:train_n + val_n], indices[train_n + val_n:]

def build_loaders(
    h5_path: str,
    gt_path: str,
    vocab,
    batch_size: int = 32,
    max_length: int = 1200,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    split_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load data, apply fixed split, return (train_loader, val_loader, test_loader, cell_types)."""
    import pandas as pd


    # Load
    bulk_expr, bulk_genes, ref_expr, ref_genes, ref_labels = load_h5_bulk(h5_path)
    gt_df = pd.read_csv(gt_path, index_col=0)
    cell_type_names = list(gt_df.columns)
    gt_array = gt_df.values.astype(np.float32)

    # SCGPT expects single-cell reference — build via DeconvolutionSolver
    import anndata as ad
    from scgpt.tasks.deconv import DeconvolutionSolver

    _solver = DeconvolutionSolver.__new__(DeconvolutionSolver)
    _solver.vocab = vocab

    adata_ref = ad.AnnData(X=ref_expr, var=pd.DataFrame(index=ref_genes))
    if ref_labels:
        adata_ref.obs["celltype"] = ref_labels
    ref_validated = _solver._validate_genes(adata_ref, gene_col="index")
    ref_gene_set = set(ref_validated.var_names)

    adata_bulk = ad.AnnData(X=bulk_expr, var=pd.DataFrame(index=bulk_genes))
    adata_bulk_v = _solver._validate_genes(adata_bulk, gene_col="index")

    # Fallback: if ref has 0 vocab-matched genes (e.g. colnames are cell barcodes),
    # use bulk_validated genes as the gene set instead.
    if not ref_gene_set and adata_bulk_v.shape[1] > 0:
        import warnings as _w
        _w.warn("ref_genes had 0 vocab matches; using bulk-validated genes as ref_gene_set")
        ref_gene_set = set(adata_bulk_v.var_names)

    bulk_ref_mask = [g in ref_gene_set for g in adata_bulk_v.var_names]
    adata_bulk_r = adata_bulk_v[:, bulk_ref_mask].copy()
    adata_bulk_pp = _solver._preprocess_for_scgpt(adata_bulk_r)
    gene_ids = _solver._gene_ids

    count_matrix = adata_bulk_pp.X
    if hasattr(count_matrix, "A"):
        count_matrix = count_matrix.A
    n = count_matrix.shape[0]

    # Split: fixed 6:2:2 for SDY67, else use shared split_indices
    from core.deconv.embedding import split_indices
    if n == 250:
        from core.deconv.embedding import split_indices
        train_idx, val_idx, test_idx = split_indices(n, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=split_seed)
    else:
        train_idx, val_idx, test_idx = split_indices(
            n, train_ratio=train_ratio, val_ratio=val_ratio,
            test_ratio=1.0 - train_ratio - val_ratio, seed=split_seed,
        )

    train_loader = get_loader(train_idx, count_matrix, gene_ids, gt_array, vocab, batch_size, max_length, shuffle=True)
    val_loader = get_loader(val_idx, count_matrix, gene_ids, gt_array, vocab, batch_size, max_length, shuffle=False)
    test_loader = get_loader(test_idx, count_matrix, gene_ids, gt_array, vocab, batch_size, max_length, shuffle=False)

    return train_loader, val_loader, test_loader, cell_type_names
