#!/usr/bin/env python3
"""Standalone prediction script for scGPT deconvolution.

Runs everything in-process — no subprocess dispatch.
No imports from ``src/bulkgpt`` or ``_utils``.

Usage:
    python methods/scgpt/predict.py \
        --config methods/scgpt/configs/ft.yaml \
        --checkpoint checkpoints/ft/best_model.pt
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from anndata import read_h5ad
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.deconv.config import DeconvHeadConfig, TrainingConfig  # noqa: E402
from core.deconv.data import (  # noqa: E402
    Preprocessor,
    PseudoBulkDataset,
    filter_genes_to_vocab,
)
from core.deconv.utils import set_seed, setup_logging  # noqa: E402
from methods.scgpt.config import ScgptModelConfig  # noqa: E402
from methods.scgpt.data import bin_expression, load_bin_info, map_genes_to_symbols  # noqa: E402




# ── Main ─────────────────────────────────────────────────────────────────


def main(
    config_path: str,
    checkpoint: str,
    log_file: str | None = None,
    ground_truth: str | None = None,
    seed: int | None = None,
) -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"ERROR: Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # Change to project root so relative paths resolve correctly
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(str(project_root))

    ds_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    ckpt_dir = paths_cfg.get("checkpoint_dir", "checkpoints/scgpt")

    if log_file is None:
        log_file = str(project_root / ckpt_dir / "run.log")

    log_path, tee, log_fh = setup_logging(log_file)

    tee("=" * 60)
    tee("scGPT method - predict (standalone)")
    tee(f"Config: {cfg_path.resolve()}")
    tee(f"Checkpoint: {checkpoint}")
    tee(f"Project root: {project_root}")
    tee(f"Log file: {log_path.resolve()}")

    # ── Config objects ───────────────────────────────────────────────────

    training_config = TrainingConfig(
        seed=seed if seed is not None else train_cfg.get("seed", 42),
        batch_size=train_cfg.get("batch_size", 64),
        n_pseudo_bulk=train_cfg.get("n_pseudo_bulk", 10000),
        proportion_alpha=train_cfg.get("proportion_alpha", 1.0),
        num_workers=train_cfg.get("num_workers", 4),
        train_ratio=train_cfg.get("train_ratio", 0.8),
        n_hvg=ds_cfg.get("n_hvg", 1200),
        max_seq_len=train_cfg.get("max_seq_len", 1201),
    )

    is_binned = model_cfg.get("binned", True)

    head_config = DeconvHeadConfig(
        hidden_dim=model_cfg.get("deconv_hidden_dim", 256),
        n_layers=model_cfg.get("deconv_n_layers", 2),
        dropout=model_cfg.get("deconv_dropout", 0.2),
        cell_emb_style=model_cfg.get("cell_emb_style", "cls"),
    )

    scgpt_config = ScgptModelConfig(
        embsize=model_cfg.get("embsize", 512),
        nhead=model_cfg.get("nhead", 8),
        d_hid=model_cfg.get("d_hid", 512),
        nlayers=model_cfg.get("nlayers", 12),
        n_layers_cls=model_cfg.get("n_layers_cls", 3),
        dropout=model_cfg.get("backbone_dropout", 0.2),
        use_fast_transformer=model_cfg.get("use_fast_transformer", is_binned),
        pre_norm=model_cfg.get("pre_norm", False),
        n_input_bins=51 if is_binned else None,
        input_emb_style="category" if is_binned else "continuous",
        cell_emb_style=model_cfg.get("cell_emb_style", "cls"),
        max_seq_len=training_config.max_seq_len,
    )

    pretrained_path = paths_cfg.get("pretrained_model", "")
    if not pretrained_path:
        tee("ERROR: paths.pretrained_model is not set in the config")
        log_fh.close()
        sys.exit(1)

    celltype_col = ds_cfg.get("celltype_col", "cell_type")
    batch_col = ds_cfg.get("batch_col", "subject")

    set_seed(training_config.seed)

    # ── 1. Load checkpoint ──────────────────────────────────────────────
    tee(f"Loading checkpoint: {checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    tee(f"  Checkpoint loaded (epoch {ckpt.get('epoch', '?')})")

    # ── 1b. Infer n_cell_types from checkpoint ──────────────────────────
    _sd = ckpt["model_state_dict"]
    _out_keys = sorted(
        k for k in _sd if k.startswith("deconv_head.net.") and k.endswith(".weight")
    )
    ckpt_n_types = _sd[_out_keys[-1]].shape[0]
    _ct_path = Path(ckpt_dir) / "cell_types.json"
    if _ct_path.exists():
        train_cell_types = json.loads(_ct_path.read_text())
        assert len(train_cell_types) == ckpt_n_types, \
            f"cell_types.json ({len(train_cell_types)}) != checkpoint ({ckpt_n_types})"
        tee(f"  Checkpoint trained with {ckpt_n_types} cell types: {train_cell_types}")
    else:
        train_cell_types = None
        tee(f"  Checkpoint trained with {ckpt_n_types} cell types (no cell_types.json)")

    # ── 2. Load scRNA reference ─────────────────────────────────────────
    from core.data_loader import load_data, load_sc_ref
    adata = load_sc_ref(ds_cfg)
    tee(f"  Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # ── 3. Fast vocab load ──────────────────────────────────────────────
    vocab_path = Path(pretrained_path) / "vocab.json"
    tee(f"Loading vocab from: {vocab_path}")
    with open(vocab_path) as f:
        token2idx = json.load(f)
    vocab_set = set(token2idx.keys())
    tee(f"  Vocab size: {len(vocab_set)}")

    # ── 4. Map genes to symbols ─────────────────────────────────────────
    adata = map_genes_to_symbols(adata, vocab_set)

    # ── 5. Filter to vocab ──────────────────────────────────────────────
    adata = filter_genes_to_vocab(adata, vocab_set)
    tee(f"  After vocab filter: {adata.shape[1]} genes")

    # ── 6. Preprocess (HVG) ─────────────────────────────────────────────
    preprocessor = Preprocessor(n_hvg=training_config.n_hvg)
    adata = preprocessor(adata, batch_key=batch_col)
    hvg_genes = adata.var_names.tolist()
    n_genes = len(hvg_genes)
    tee(f"  After preprocessing: {n_genes} genes, {adata.shape[0]} cells")

    # ── 7. Build proper GeneVocab ───────────────────────────────────────
    sorted_tokens = sorted(token2idx.keys(), key=lambda t: token2idx[t])
    from torchtext.vocab import vocab as create_vocab
    vocab = create_vocab(OrderedDict((t, 1) for t in sorted_tokens))

    # ── 8. Load H5 bulk expression ──────────────────────────────────────
    tee("Loading H5 bulk expression data...")
    bundle = load_data(ds_cfg["data_path"], ground_truth=ground_truth)
    bulk_df = bundle.bulk       # (samples, genes), CPM*1e6
    gt_df = bundle.gt           # (samples, cell_types)

    # Subset bulk to HVG genes
    bulk_matrix = bulk_df[hvg_genes].values.astype(np.float64)
    # Training pseudo-bulk uses: sum raw counts → CPM*1e4 → log1p.
    # Bulk data is raw counts, NOT CPM.  Convert properly.
    row_sums = bulk_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    bulk_matrix = np.log1p((bulk_matrix / row_sums) * 1e4)

    sample_ids = bulk_df.index.tolist()

    # Use checkpoint n_types (training has all scRNA types; GT may be subset)
    if train_cell_types is not None:
        cell_types = train_cell_types
        n_cell_types = len(train_cell_types)
    else:
        cell_types = gt_df.columns.tolist()
        n_cell_types = len(cell_types)

    # Dummy proportions — real GT used for eval after type alignment
    proportions = np.zeros((len(bulk_matrix), n_cell_types), dtype=np.float64)
    tee(f"  Loaded {len(bulk_matrix)} bulk samples, {n_cell_types} cell types (from checkpoint)")

    # ── 9. Convert gene IDs ─────────────────────────────────────────────
    gene_symbols = adata.var_names.tolist()
    gene_ids = np.array([vocab[g] for g in gene_symbols], dtype=np.int64)

    # ── 10. Binning (optional) ──────────────────────────────────────────
    if is_binned:
        # Look for bin_info.csv next to the checkpoint first, then in ckpt_dir
        bin_info_path = Path(checkpoint).parent / "bin_info.csv"
        if not bin_info_path.exists():
            bin_info_path = Path(ckpt_dir) / "bin_info.csv"
        if not bin_info_path.exists():
            tee(f"ERROR: bin_info.csv not found (tried {bin_info_path})")
            log_fh.close()
            sys.exit(1)

        tee(f"Loading bin info from: {bin_info_path}")
        bin_info = load_bin_info(str(bin_info_path))

        tee("Binning expression values...")
        bulk_matrix = bin_expression(
            bulk_matrix, np.array(gene_symbols), bin_info, n_bins=51
        )

    from methods.scgpt.model import ScgptDeconvModel, create_scgpt_backbone
    # ── 11. Rebuild model & load state dict ─────────────────────────────
    tee("Rebuilding model from config...")
    backbone = create_scgpt_backbone(
        vocab=vocab,
        model_config=scgpt_config,
        model_dir=pretrained_path,
        device=device,
    )

    model = ScgptDeconvModel(
        backbone=backbone,
        n_cell_types=n_cell_types,
        backbone_config=scgpt_config,
        head_config=head_config,
        freeze_backbone=True,  # irrelevant -- state dict will overwrite
    )

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        tee(f"  Missing keys in state dict: {missing}")
    if unexpected:
        tee(f"  Unexpected keys in state dict: {unexpected}")

    model.to(device)
    model.eval()
    tee("  Model rebuilt and checkpoint state dict loaded")

    # ── 12. Dataset & loader ────────────────────────────────────────────
    max_len = min(n_genes + 1, scgpt_config.max_seq_len)

    dataset = PseudoBulkDataset(
        bulk_matrix=bulk_matrix,
        proportions=proportions,
        gene_ids=gene_ids,
        vocab=vocab,
        max_len=max_len,
        is_binned=is_binned,
        is_rank_value=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    # ── 13. Run prediction ──────────────────────────────────────────────
    tee("Running prediction...")
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in loader:
            gene_ids_b = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            mask = batch["src_key_padding_mask"].to(device)
            output = model(gene_ids_b, values, mask)
            all_preds.append(output["proportions"].cpu().numpy())
            all_true.append(batch["proportions"].numpy())

    predictions = np.concatenate(all_preds, axis=0)

    # ── 14. Save predictions ──────────────────────────────────────────────
    pred_path = Path(ckpt_dir) / "proportions.csv"
    pd.DataFrame(predictions, columns=cell_types, index=sample_ids).to_csv(pred_path)
    tee(f"Saved predictions -> {pred_path}")

    # Also copy to checkpoint parent
    ckpt_parent = Path(checkpoint).parent
    if ckpt_parent.resolve() != Path(ckpt_dir).resolve():
        pred_path_alt = ckpt_parent / "proportions.csv"
        pd.DataFrame(predictions, columns=cell_types, index=sample_ids).to_csv(pred_path_alt)
        tee(f"Also saved -> {pred_path_alt}")

    tee(f"Saved predictions:  {pred_path}")
    tee("\nPrediction completed successfully.")
    tee("=" * 60)

    log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="scGPT deconvolution prediction (standalone)",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained .pt checkpoint file",
    )
    parser.add_argument(
        "--log_file", default=None,
        help="Path to log file (default: <checkpoint_dir>/run.log)",
    )
    parser.add_argument(
        "--ground-truth", default=None,
        help="Path to ground truth CSV (overrides H5 ground_truth group)",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.log_file, args.ground_truth)
