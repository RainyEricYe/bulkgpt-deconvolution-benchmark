#!/usr/bin/env python3
"""Standalone Geneformer deconvolution prediction script.

Runs entirely in-process (no subprocess calls).  No imports from ``src/bulkgpt``
or ``methods.geneformer._utils``.

Runnable as::

    python methods/geneformer/predict.py \\
        --config methods/geneformer/configs/ft.yaml \\
        --checkpoint checkpoints/geneformer/ft/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import pandas as pd
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import yaml
from torch.utils.data import DataLoader

# -- Ensure core/ and methods/ packages are importable ------------------------
_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.deconv.config import DeconvHeadConfig
from core.deconv.data import (
    Preprocessor,
    PseudoBulkDataset,
    filter_genes_to_vocab,
)
from core.deconv.utils import set_seed, setup_logging

from methods.geneformer.data import map_symbol_to_ensembl, rank_value_encode
from methods.geneformer.model import GeneformerDeconvModel, create_geneformer_backbone




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config_path: str,
    checkpoint_path: str,
    log_file: str | None = None,
    ground_truth: str | None = None,
) -> None:
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract config sections
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    # -- Output directory = checkpoint parent -----------------------------
    out_dir = checkpoint_path.parent
    if log_file is None:
        log_file = str(out_dir / "predict.log")

    log_path, tee, log_fh = setup_logging(log_file)

    try:
        tee("=" * 60)
        tee("Geneformer method - predict (standalone)")
        tee(f"Config: {config_path.resolve()}")
        tee(f"Checkpoint: {checkpoint_path.resolve()}")
        tee(f"Log file: {log_path.resolve()}")

        # -- Reproducibility -----------------------------------------------
        seed = training_cfg.get("seed", 42)
        set_seed(seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tee(f"Using device: {device}")

        # -- 1. Load scRNA reference ---------------------------------------
        from core.data_loader import load_data, load_sc_ref
        data_path = dataset_cfg.get("data_path")
        if data_path:
            adata = load_sc_ref(data_path)
        else:
            sc_ref = dataset_cfg.get("sc_ref") or paths_cfg.get("sc_ref")
            if not sc_ref:
                tee("ERROR: No sc_ref or data_path in config")
                sys.exit(1)
            tee(f"Loading single-cell reference: {sc_ref}")
            adata = sc.read_h5ad(sc_ref)
        tee(f"Raw data: {adata.shape[0]} cells, {adata.shape[1]} genes")

        # -- 2. Load Geneformer token dictionary ---------------------------
        tee("Loading Geneformer token dictionary...")
        from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            gf_vocab = pickle.load(f)
        tee(f"Geneformer vocab size: {len(gf_vocab)}")

        # -- 3. Map gene symbols to Ensembl IDs ---------------------------
        n_ensg = sum(1 for g in adata.var_names if str(g).startswith("ENSG"))
        if n_ensg < len(adata.var_names) * 0.5:
            pretrained_model = paths_cfg.get("pretrained_model")
            tee(f"Mapping gene symbols to Ensembl IDs (model_dir={pretrained_model})...")
            _symbol_to_ensembl = map_symbol_to_ensembl(
                list(adata.var_names), model_dir=pretrained_model
            )
            adata.var_names = [_symbol_to_ensembl[str(g)] for g in adata.var_names]
        else:
            _symbol_to_ensembl = {}
            tee(
                f"Genes already largely Ensembl IDs "
                f"({n_ensg}/{len(adata.var_names)}), skipping mygene lookup"
            )

        # -- 4. Filter to Geneformer vocab --------------------------------
        adata = filter_genes_to_vocab(adata, gf_vocab)
        tee(f"After vocab filter: {adata.shape[1]} genes retained")

        # -- 5. Preprocess (HVG selection) --------------------------------
        celltype_col = dataset_cfg.get("celltype_col", "cell_type")
        batch_col = dataset_cfg.get("batch_col", "subject")
        n_hvg = min(
            training_cfg.get("n_hvg", dataset_cfg.get("n_hvg", 1200)),
            adata.shape[1],
        )

        preprocessor = Preprocessor(n_hvg=n_hvg)
        adata = preprocessor(adata, batch_key=batch_col)
        hvg_genes = adata.var_names.tolist()
        tee(f"Preprocessed: {len(hvg_genes)} genes, {adata.shape[0]} cells")

        # -- 6. Load H5 bulk expression ------------------------------------
        tee("Loading H5 bulk expression data...")
        bundle = load_data(dataset_cfg["data_path"], ground_truth=ground_truth)
        bulk_df = bundle.bulk       # (samples, genes), CPM*1e6
        gt_df = bundle.gt           # (samples, cell_types)

        # Map bulk column names from symbols to Ensembl IDs (matching scRNA ref)
        if _symbol_to_ensembl:
            bulk_df = bulk_df.rename(columns=lambda c: _symbol_to_ensembl.get(str(c), str(c)))

        # Subset both adata and bulk to genes present in both
        common_hvg = [g for g in hvg_genes if g in bulk_df.columns]
        if len(common_hvg) < 10:
            raise RuntimeError(f"Too few HVG genes ({len(common_hvg)}) found in bulk data")
        adata = adata[:, common_hvg].copy()
        hvg_genes = common_hvg
        bulk_matrix = bulk_df[common_hvg].values.astype(np.float64)
        # Training pseudo-bulk uses: sum raw counts → CPM*1e4 → log1p.
        # Bulk data is raw counts, NOT CPM.  Convert properly.
        row_sums = bulk_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        bulk_matrix = np.log1p((bulk_matrix / row_sums) * 1e4)

        gt_info = f"{len(gt_df.columns)} cell types" if gt_df is not None else "None (no GT in H5)"
        tee(f"Loaded {len(bulk_matrix)} bulk samples — GT has {gt_info}")

        # -- 7. Convert gene IDs using token dictionary -------------------
        pad_id = gf_vocab["<pad>"]
        gene_names = np.array(adata.var_names.tolist())
        gene_ids = np.array([gf_vocab.get(g, pad_id) for g in gene_names])
        in_vocab = gene_ids != pad_id
        tee(
            f"Gene coverage: {in_vocab.sum()}/{len(gene_names)} "
            f"in Geneformer vocab"
        )

        if in_vocab.sum() < 10:
            raise RuntimeError(
                f"Too few genes ({in_vocab.sum()}) found in Geneformer vocabulary."
            )

        # -- 8. Filter to in-vocab genes ----------------------------------
        valid = in_vocab.copy()
        valid_gene_ids = gene_ids[valid]
        valid_bulk = bulk_matrix[:, valid].astype(np.float32)

        # -- 9. Rank-value encoding ---------------------------------------
        rank_expr, sorted_indices = rank_value_encode(valid_bulk)
        valid_bulk = rank_expr.astype(np.float32)

        # -- 10. Create Geneformer backbone -------------------------------
        pretrained_model = paths_cfg.get("pretrained_model")
        backbone = create_geneformer_backbone(
            model_dir=pretrained_model, device=device
        )

        # -- 11. Load checkpoint & infer n_cell_types -------------------------
        tee(f"Loading checkpoint from {checkpoint_path}")
        _ckpt_data = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in _ckpt_data:
            _sd = _ckpt_data["model_state_dict"]
            _epoch = _ckpt_data.get("epoch", "?")
            tee(f"  Checkpoint epoch: {_epoch}")
        else:
            _sd = _ckpt_data

        _out_keys = sorted(
            k for k in _sd if k.startswith("deconv_head.net.") and k.endswith(".weight")
        )
        ckpt_n_types = _sd[_out_keys[-1]].shape[0]
        _ct_path = out_dir / "cell_types.json"
        if _ct_path.exists():
            train_cell_types = json.loads(_ct_path.read_text())
            assert len(train_cell_types) == ckpt_n_types, \
                f"cell_types.json ({len(train_cell_types)}) != checkpoint ({ckpt_n_types})"
            tee(f"  Checkpoint trained with {ckpt_n_types} cell types: {train_cell_types}")
        else:
            train_cell_types = None
            tee(f"  Checkpoint trained with {ckpt_n_types} cell types (no cell_types.json)")

        n_cell_types = len(train_cell_types) if train_cell_types else ckpt_n_types
        cell_types = train_cell_types if train_cell_types else []
        tee(f"  Using {n_cell_types} output types for model")

        # -- 12. Build GeneformerDeconvModel (matching training arch) -----
        head_config = DeconvHeadConfig(
            hidden_dim=model_cfg.get("deconv_hidden_dim", 256),
            n_layers=model_cfg.get("deconv_n_layers", 2),
            dropout=model_cfg.get("deconv_dropout", 0.2),
            cell_emb_style=model_cfg.get("cell_emb_style", "cls"),
        )

        model = GeneformerDeconvModel(
            backbone=backbone,
            n_cell_types=n_cell_types,
            head_config=head_config,
            freeze_backbone=True,  # eval mode; weights loaded from checkpoint
        )

        model.load_state_dict(_sd)
        model.to(device)
        model.eval()
        tee("Model loaded successfully")

        # Use dummy proportions matching training n_types
        proportions = np.zeros((len(valid_bulk), n_cell_types), dtype=np.float64)

        # -- 13. Create PseudoBulkDataset ---------------------------------
        max_seq_len = model_cfg.get("max_seq_len", 2048)
        dataset = PseudoBulkDataset(
            valid_bulk,
            proportions,
            valid_gene_ids,
            gf_vocab,
            max_len=max_seq_len,
            pad_token="<pad>",
            cls_token="<cls>",
            is_rank_value=True,
            sorted_indices=sorted_indices,
        )

        val_dataset = dataset
        tee(f"Evaluation samples: {len(dataset)}")

        batch_size = training_cfg.get("batch_size", 64)
        num_workers = training_cfg.get("num_workers", 4)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # -- 14. Predict --------------------------------------------------
        model.eval()
        all_preds: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                gene_ids_t = batch["gene_ids"].to(device)
                values_t = batch["values"].to(device)
                mask_t = batch["src_key_padding_mask"].to(device)
                pred = model.predict(gene_ids_t, values_t, mask_t)
                all_preds.append(pred.cpu().numpy())
                all_true.append(batch["proportions"].numpy())

        pred_props = np.concatenate(all_preds, axis=0)
        true_props = np.concatenate(all_true, axis=0)

        # -- 15. Save predictions ---------------------------------------------
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_ids = bulk_df.index.tolist()

        pred_path = out_dir / "proportions.csv"
        pd.DataFrame(pred_props, columns=cell_types, index=sample_ids).to_csv(pred_path)
        tee(f"Saved predictions to {pred_path}")

        tee("Prediction completed successfully")
        tee("=" * 60)

    finally:
        log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geneformer deconvolution: predict (standalone)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Path to log file (default: <checkpoint_dir>/predict.log)",
    )
    parser.add_argument(
        "--ground-truth", default=None,
        help="Path to ground truth CSV (overrides H5 ground_truth group)",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.log_file, args.ground_truth)
