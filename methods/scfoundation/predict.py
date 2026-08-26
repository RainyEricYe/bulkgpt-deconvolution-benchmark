#!/usr/bin/env python3
"""Standalone prediction script for scFoundation deconvolution.

Loads a trained checkpoint, processes real bulk expression data, and
predicts cell-type proportions.  Handles the scFoundation 19264-gene
alignment internally.

Usage:
    python methods/scfoundation/predict.py \\
        --config configs/frozen.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

from core.data_loader import load_data
from core.deconv.utils import set_seed, setup_logging
from methods.scfoundation.model import ScFoundationBackbone, ScFoundationDeconvModel


# ── Inline predict dataset (mirrors PseudoBulkDataset with is_scfoundation=True) ─


class _PredictDataset(Dataset):
    """Minimal dataset for scFoundation prediction on real bulk."""

    def __init__(self, bulk_matrix, proportions, gene_ids, max_len=19264):
        self.bulk_matrix = bulk_matrix
        self.proportions = proportions
        self.gene_ids = gene_ids
        self.max_len = max_len
        self.pad_pos = 19266  # scFoundation pad position

    def __len__(self):
        return len(self.bulk_matrix)

    def __getitem__(self, idx):
        row = self.bulk_matrix[idx]
        props = self.proportions[idx]

        values = torch.from_numpy(row).float()
        genes = torch.from_numpy(self.gene_ids).long()

        if len(genes) > self.max_len:
            perm = torch.randperm(len(genes))[:self.max_len].sort().values
            genes = genes[perm]
            values = values[perm]

        if len(genes) < self.max_len:
            pad_size = self.max_len - len(genes)
            genes = torch.cat([genes, torch.full((pad_size,), self.pad_pos, dtype=torch.long)])
            values = torch.cat([values, torch.zeros(pad_size)])

        return {
            "gene_ids": genes,
            "values": values,
            "proportions": torch.from_numpy(props).float(),
            "src_key_padding_mask": genes == self.pad_pos,
        }


# ── Gene alignment helpers ──────────────────────────────────────────────


def _align_bulk_to_scf(bulk_df: pd.DataFrame, worktree_dir: str) -> np.ndarray:
    """Align bulk expression matrix to scFoundation's 19264-gene order.

    Returns (n_samples, 19264) float32 array — missing genes zero-filled.
    """
    import json

    gene_index_path = (
        Path(worktree_dir) / "scfoundation_src" / "OS_scRNA_gene_index.19264.tsv"
    )
    if not gene_index_path.exists():
        raise FileNotFoundError(
            f"scFoundation gene index not found: {gene_index_path}"
        )
    df_idx = pd.read_csv(gene_index_path, sep="\t")
    scf_symbols = df_idx["gene_name"].tolist()

    ensm_path = Path(worktree_dir) / "data" / "scfoundation_ensembl_to_scfpos.json"
    if not ensm_path.exists():
        raise FileNotFoundError(
            f"scFoundation Ensembl mapping not found: {ensm_path}"
        )
    with open(ensm_path) as f:
        ensembl_to_scfpos = json.load(f)

    our_genes = list(bulk_df.columns)
    is_ensembl = any(g.startswith("ENSG") for g in our_genes[:10])

    if is_ensembl:
        our_ids = our_genes
    else:
        sym_path = Path(worktree_dir) / "data" / "scfoundation_symbol_to_ensembl.json"
        if sym_path.exists():
            with open(sym_path) as f:
                symbol_to_ensembl = json.load(f)
        else:
            symbol_to_ensembl = {}
        our_ids = [symbol_to_ensembl.get(g, g) for g in our_genes]

    reorder = [ensembl_to_scfpos.get(eid, -1) for eid in our_ids]
    scf_pos_to_col = {}
    for col, scf_pos in enumerate(reorder):
        if scf_pos >= 0:
            scf_pos_to_col[scf_pos] = col

    X = bulk_df.values.astype(np.float64)
    X_scf = np.zeros((X.shape[0], 19264), dtype=X.dtype)
    missing = 0
    for scf_pos in range(19264):
        col = scf_pos_to_col.get(scf_pos)
        if col is not None:
            X_scf[:, scf_pos] = X[:, col]
        else:
            missing += 1

    n_found = 19264 - missing
    print(f"  scFoundation gene alignment: {n_found}/19264 found "
          f"({missing} missing -> zero-filled)")
    return X_scf.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────


def main(
    config_path: str,
    log_file: str | None = None,
) -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"ERROR: Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    os.chdir(str(HERE))

    ds_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    worktree = Path(
        paths_cfg.get(
            "worktree",
            HERE / ".worktrees" / "scfoundation-test",
        )
    )
    ckpt_dir = paths_cfg.get("checkpoint_dir", "checkpoints/scfoundation")

    if log_file is None:
        log_file = str(HERE / ckpt_dir / "predict.log")

    log_path, tee, log_fh = setup_logging(log_file)

    tee("=" * 60)
    tee("scFoundation method - predict")
    tee(f"Config: {cfg_path.resolve()}")
    tee(f"Checkpoint dir: {ckpt_dir}")
    tee(f"Worktree: {worktree.resolve()}")
    tee(f"Log file: {log_path.resolve()}")

    # ── Resolve checkpoint ──
    ckpt_dir_path = Path(ckpt_dir)
    best_ckpt = None
    for name in ("best_model.pt", "final_model.pt"):
        cand = ckpt_dir_path / name
        if cand.exists():
            best_ckpt = cand
            break
    if not best_ckpt:
        tee("ERROR: no checkpoint found (searched best_model.pt, final_model.pt)")
        log_fh.close()
        sys.exit(1)
    tee(f"Checkpoint: {best_ckpt}")

    # ── Load checkpoint ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    tee(f"  Checkpoint loaded (epoch {ckpt.get('epoch', '?')})")

    _sd = ckpt["model_state_dict"]
    _out_keys = sorted(
        k for k in _sd if k.startswith("deconv_head.net.") and k.endswith(".weight")
    )
    ckpt_n_types = _sd[_out_keys[-1]].shape[0]

    ct_path = Path(ckpt_dir) / "cell_types.json"
    if ct_path.exists():
        train_cell_types = json.loads(ct_path.read_text())
        assert len(train_cell_types) == ckpt_n_types
        tee(f"  Checkpoint trained with {ckpt_n_types} cell types: {train_cell_types}")
    else:
        train_cell_types = None
        tee(f"  Checkpoint trained with {ckpt_n_types} cell types")

    # ── Load bulk expression data ──
    tee("Loading H5 bulk expression data...")
    bundle = load_data(ds_cfg["data_path"])
    bulk_df = bundle.bulk
    gt_df = bundle.gt

    # Align to scFoundation 19264-gene order
    bulk_scf = _align_bulk_to_scf(bulk_df, str(worktree))

    # Normalize: log1p-CPM (matches pseudo-bulk preparation)
    row_sums = bulk_scf.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    bulk_scf = np.log1p((bulk_scf / row_sums) * 1e4).astype(np.float32)

    sample_ids = bulk_df.index.tolist()
    n_genes = bulk_scf.shape[1]

    if train_cell_types is not None:
        cell_types = train_cell_types
        n_cell_types = len(train_cell_types)
    else:
        cell_types = gt_df.columns.tolist()
        n_cell_types = len(cell_types)

    proportions = np.zeros((len(bulk_scf), n_cell_types), dtype=np.float32)
    tee(f"  Loaded {len(bulk_scf)} bulk samples, {n_cell_types} cell types, "
        f"{n_genes} genes")

    gene_ids = np.arange(n_genes, dtype=np.int64)

    # ── Build model ──
    pretrained_path = paths_cfg.get("pretrained_model", "")
    tee(f"Building scFoundation backbone from: {pretrained_path}")

    backbone = ScFoundationBackbone(
        ckpt_path=str(Path(pretrained_path) / "models.ckpt"),
        device=device,
    )

    cell_emb_style = model_cfg.get("cell_emb_style", "cls")
    deconv_hidden_dim = model_cfg.get("deconv_hidden_dim", 256)
    deconv_n_layers = model_cfg.get("deconv_n_layers", 2)
    deconv_dropout = model_cfg.get("deconv_dropout", 0.2)

    model = ScFoundationDeconvModel(
        backbone=backbone,
        n_cell_types=n_cell_types,
        cell_emb_style=cell_emb_style,
        hidden_dim=deconv_hidden_dim,
        n_layers=deconv_n_layers,
        dropout=deconv_dropout,
    )

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        tee(f"  Missing keys: {missing}")
    if unexpected:
        tee(f"  Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    tee("  Model rebuilt and checkpoint loaded")

    # ── Dataset & loader ──
    max_len = min(n_genes, 19264)
    dataset = _PredictDataset(
        bulk_matrix=bulk_scf,
        proportions=proportions,
        gene_ids=gene_ids,
        max_len=max_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
    )

    # ── Predict ──
    tee("Running prediction...")
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            gene_ids_b = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            mask = batch["src_key_padding_mask"].to(device)
            output = model(gene_ids_b, values, mask)
            all_preds.append(output["proportions"].cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)

    # ── Save ──
    pred_path = Path(ckpt_dir) / "proportions.csv"
    pred_df = pd.DataFrame(predictions, columns=cell_types, index=sample_ids)
    pred_df.to_csv(pred_path)
    tee(f"Saved predictions -> {pred_path}")

    tee("\nPrediction completed successfully.")
    tee("=" * 60)
    log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="scFoundation deconvolution prediction",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to .pt checkpoint (ignored for scFoundation, uses checkpoint_dir)",
    )
    parser.add_argument(
        "--log_file", default=None,
        help="Path to log file (default: <checkpoint_dir>/predict.log)",
    )
    args = parser.parse_args()
    main(args.config, args.log_file)
