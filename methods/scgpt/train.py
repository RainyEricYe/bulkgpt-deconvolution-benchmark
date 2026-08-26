#!/usr/bin/env python3
"""Standalone training script for scGPT deconvolution.

Runs everything in-process — no subprocess dispatch.
No imports from ``src/bulkgpt`` or ``_utils``.

Usage:
    python methods/scgpt/train.py --config methods/scgpt/configs/ft.yaml
    python methods/scgpt/train.py --config methods/scgpt/configs/ft.yaml --log_file /tmp/train.log
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
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, random_split
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.deconv.config import DeconvHeadConfig, TrainingConfig  # noqa: E402
from core.deconv.data import (  # noqa: E402
    Preprocessor,
    PseudoBulkDataset,
    RealBulkDataset,
    filter_genes_to_vocab,
    prepare_pseudo_bulk,
)
from core.deconv.trainer import Trainer  # noqa: E402
from core.deconv.utils import set_seed, setup_logging  # noqa: E402
from methods.scgpt.config import ScgptModelConfig  # noqa: E402
from methods.scgpt.data import bin_expression, map_genes_to_symbols  # noqa: E402


# ── Metrics ──────────────────────────────────────────────────────────────


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    cell_types: list[str],
) -> dict:
    """Per-cell-type Pearson *r*, RMSE, MAE and their macro-averages."""
    per_type = {}
    for i, ct in enumerate(cell_types):
        pred = predictions[:, i]
        true = ground_truth[:, i]
        r, _ = pearsonr(pred, true)
        per_type[ct] = {
            "pearson": float(r),
            "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
            "mae": float(np.mean(np.abs(pred - true))),
        }

    return {
        "per_cell_type": per_type,
        "mean_pearson": float(np.mean([v["pearson"] for v in per_type.values()])),
        "mean_rmse": float(np.mean([v["rmse"] for v in per_type.values()])),
        "mean_mae": float(np.mean([v["mae"] for v in per_type.values()])),
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main(config_path: str, log_file: str | None = None, seed: int | None = None) -> None:
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
    tee("scGPT method - train (standalone)")
    tee(f"Config: {cfg_path.resolve()}")
    tee(f"Project root: {project_root}")
    tee(f"Log file: {log_path.resolve()}")

    # ── Config objects ───────────────────────────────────────────────────

    training_config = TrainingConfig(
        seed=seed if seed is not None else train_cfg.get("seed", 42),
        epochs=train_cfg.get("epochs", 100),
        batch_size=train_cfg.get("batch_size", 64),
        lr=train_cfg.get("lr", 1e-3),
        backbone_lr=train_cfg.get("backbone_lr"),
        n_pseudo_bulk=train_cfg.get("n_pseudo_bulk", 10000),
        proportion_alpha=train_cfg.get("proportion_alpha", 1.0),
        loss_type=train_cfg.get("loss_type", "mse_kl"),
        num_workers=train_cfg.get("num_workers", 4),
        checkpoint_dir=ckpt_dir,
        use_wandb=train_cfg.get("use_wandb", False),
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

    # ── 1. Load scRNA reference ─────────────────────────────────────────
    from core.data_loader import load_data, load_sc_ref
    adata = load_sc_ref(ds_cfg)
    tee(f"  Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Ensure cell_type column exists (fix sweetwater/demixsc orientation issue)
    if celltype_col not in adata.obs.columns:
        from methods._shared.h5_sc_helper import ensure_sc_celltypes
        adata = ensure_sc_celltypes(adata, ds_cfg.get("data_path", ""), celltype_col)
        if celltype_col in adata.obs.columns:
            tee(f"  Restored cell types from H5: {adata.obs[celltype_col].nunique()} types")

    # ── 1b. Load H5 bulk data for test evaluation ──────────────────────────
    bulk_df = None
    gt_df = None
    _h5_path = ds_cfg.get("data_path")
    if _h5_path and Path(_h5_path).exists():
        _bundle = load_data(_h5_path)
        bulk_df = _bundle.bulk
        gt_df = _bundle.gt
        if bulk_df is not None:
            tee(f"  Loaded H5 bulk: {bulk_df.shape[0]} samples, {bulk_df.shape[1]} genes")

    # ── 2. Fast vocab load ──────────────────────────────────────────────
    vocab_path = Path(pretrained_path) / "vocab.json"
    tee(f"Loading vocab from: {vocab_path}")
    with open(vocab_path) as f:
        token2idx = json.load(f)
    vocab_set = set(token2idx.keys())
    tee(f"  Vocab size: {len(vocab_set)}")

    # ── 3. Map genes to symbols (scRNA + bulk) ─────────────────────────
    tee("Mapping gene names to symbols...")
    adata = map_genes_to_symbols(adata, vocab_set)

    if bulk_df is not None:
        # Map bulk columns to symbols if they're ENSEMBL IDs
        bulk_genes_list = bulk_df.columns.tolist()
        bulk_in_vocab = sum(1 for g in bulk_genes_list if g in vocab_set)
        bulk_is_ensembl = sum(1 for g in bulk_genes_list if g.startswith("ENSG") or g.startswith("ENSMUSG"))
        if bulk_in_vocab < len(bulk_genes_list) * 0.5 and bulk_is_ensembl > 0:
            import json as _json
            _map_path = Path(__file__).resolve().parent.parent.parent / "data" / "gene_symbol_map.json"
            if _map_path.exists():
                with open(_map_path) as _f:
                    _mapping = _json.load(_f)
                new_cols = [_mapping.get(g, g) for g in bulk_genes_list]
                bulk_df.columns = new_cols
                bulk_df = bulk_df.loc[:, ~bulk_df.columns.duplicated(keep="first")]
                bulk_genes_list = bulk_df.columns.tolist()
                tee(f"  Mapped bulk columns to gene symbols via {_map_path.name}")
            else:
                tee(f"  WARNING: gene_symbol_map.json not found, bulk columns left as-is")

        # Intersect scRNA and bulk genes
        common = sorted(set(adata.var_names) & set(bulk_df.columns))
        tee(f"  Common genes: {len(common)} (scRNA: {adata.shape[1]}, bulk: {len(bulk_genes_list)})")
        if len(common) == 0:
            tee("  ERROR: no common genes between scRNA and bulk, skipping H5 bulk eval")
            bulk_df = None
        else:
            adata = adata[:, [g for g in common if g in adata.var_names]].copy()
            bulk_df = bulk_df[[g for g in common if g in bulk_df.columns]]

    # ── 4. Filter to vocab ──────────────────────────────────────────────
    tee("Filtering genes to vocabulary...")
    adata = filter_genes_to_vocab(adata, vocab_set)
    tee(f"  After vocab filter: {adata.shape[1]} genes")

    # Auto-determine n_hvg: min(config value, available genes)
    n_hvg = min(ds_cfg.get("n_hvg", 1200), adata.shape[1])
    training_config.n_hvg = n_hvg

    # ── 5. Preprocess (HVG) ─────────────────────────────────────────────
    tee(f"Preprocessing (HVG selection, n_hvg={n_hvg})...")
    preprocessor = Preprocessor(n_hvg=n_hvg)
    adata = preprocessor(adata, batch_key=batch_col)
    n_genes = adata.shape[1]
    tee(f"  After preprocessing: {n_genes} genes, {adata.shape[0]} cells")

    # ── 6. Build proper GeneVocab ───────────────────────────────────────
    from torchtext.vocab import vocab as create_vocab
    sorted_tokens = sorted(token2idx.keys(), key=lambda t: token2idx[t])
    vocab = create_vocab(OrderedDict((t, 1) for t in sorted_tokens))
    tee(f"  GeneVocab created: {len(vocab)} tokens")

    # ── 7. Generate pseudo-bulk ─────────────────────────────────────────
    tee("Generating pseudo-bulk samples...")
    bulk_matrix, proportions, cell_types = prepare_pseudo_bulk(
        adata,
        celltype_col=celltype_col,
        n_samples=training_config.n_pseudo_bulk,
        alpha=training_config.proportion_alpha,
    )
    n_cell_types = len(cell_types)
    tee(f"  Generated {len(bulk_matrix)} pseudo-bulk samples, {n_cell_types} cell types")

    # ── 8. Convert gene IDs ─────────────────────────────────────────────
    gene_symbols = adata.var_names.tolist()
    gene_ids = np.array([vocab[g] for g in gene_symbols], dtype=np.int64)

    # ── 9. Binning (optional) ───────────────────────────────────────────
    if is_binned:
        tee("Computing bin quantiles from pseudo-bulk expression...")
        n_bins = 51
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # 50 boundaries
        bin_info = {}
        for j, gene in enumerate(gene_symbols):
            col = bulk_matrix[:, j]
            if (col > 0).any():
                quantiles = np.percentile(col[col > 0], percentiles)
            else:
                quantiles = np.zeros(n_bins - 1)
            bin_info[gene] = quantiles

        bin_info_path = Path(ckpt_dir) / "bin_info.csv"
        bin_info_path.parent.mkdir(parents=True, exist_ok=True)
        bin_info_df = pd.DataFrame(bin_info).T
        bin_info_df.to_csv(str(bin_info_path))
        tee(f"  Saved bin_info.csv to {bin_info_path}")

        tee("Binning expression values...")
        bulk_matrix = bin_expression(
            bulk_matrix, np.array(gene_symbols), bin_info, n_bins=n_bins
        )

    tee(f"  Gene IDs shape: {gene_ids.shape}, Expression shape: {bulk_matrix.shape}")

    # ── 10. Create backbone ─────────────────────────────────────────────
    from methods.scgpt.model import ScgptDeconvModel, create_scgpt_backbone

    tee("Creating scGPT backbone...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tee(f"  Device: {device}")

    backbone = create_scgpt_backbone(
        vocab=vocab,
        model_config=scgpt_config,
        model_dir=pretrained_path,
        device=device,
    )

    # ── 11. Build full deconvolution model ──────────────────────────────
    freeze_backbone = not train_cfg.get("unfreeze_backbone", False)
    tee(f"  Freeze backbone: {freeze_backbone}")

    model = ScgptDeconvModel(
        backbone=backbone,
        n_cell_types=n_cell_types,
        backbone_config=scgpt_config,
        head_config=head_config,
        freeze_backbone=freeze_backbone,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tee(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # ── 12. Dataset & train/val split ───────────────────────────────────
    tee("Creating PseudoBulkDataset and splitting...")
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

    n_train = int(len(dataset) * training_config.train_ratio)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(training_config.seed),
    )
    tee(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── 13. Real-bulk loading for domain adaptation ────────────────────
    real_bulk_path = ds_cfg.get("real_bulk_path")
    target_loader = None
    da_module = None

    if real_bulk_path:
        from core.data_loader import load_data

        tee(f"Loading real-bulk for domain adaptation: {real_bulk_path}")
        bundle = load_data(real_bulk_path)
        bulk_df = bundle.bulk  # (samples, genes) DataFrame
        tee(f"  Raw: {bulk_df.shape[0]} samples, {bulk_df.shape[1]} genes")

        # Map bulk genes to training gene order (same H5 → same naming)
        training_genes = gene_symbols
        bulk_genes = list(bulk_df.columns)
        bulk_aligned = np.zeros(
            (bulk_df.shape[0], len(training_genes)), dtype=np.float64
        )
        for j, g in enumerate(training_genes):
            if g in bulk_genes:
                idx = bulk_genes.index(g)
                bulk_aligned[:, j] = bulk_df.iloc[:, idx].values

        # CPM normalize + log1p (same as pseudo-bulk pipeline)
        row_sums = bulk_aligned.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        bulk_aligned = (bulk_aligned / row_sums) * 1e4
        bulk_aligned = np.log1p(bulk_aligned)

        # Apply binning (same bin_info from pseudo-bulk)
        if is_binned:
            bulk_aligned = bin_expression(
                bulk_aligned, np.array(training_genes), bin_info, n_bins=51
            ).astype(np.int64)

        n_present = (bulk_aligned.sum(axis=1) > 0).sum()
        tee(f"  Aligned: {n_present} genes present, {bulk_aligned.shape[1]} total")

        # Create RealBulkDataset
        real_dataset = RealBulkDataset(
            bulk_aligned,
            gene_ids,  # same token IDs as training
            vocab,
            max_len=max_len,
            is_binned=is_binned,
            is_rank_value=False,
        )
        tee(f"  Real-bulk dataset: {len(real_dataset)} samples")

        target_loader = DataLoader(
            real_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.num_workers,
            pin_memory=(device == "cuda"),
        )

        # Create domain adaptation module
        if training_config.da_method:
            from core.deconv.domain_adaptation import DomainAdaptationModule
            backbone_dim = scgpt_config.embsize
            da_module = DomainAdaptationModule(backbone_dim)
            da_module.to(device)
            tee(
                f"  Domain adaptation: method={training_config.da_method}, "
                f"lambda={training_config.da_lambda}, "
                f"grl_lambda={training_config.da_grl_lambda}"
            )

    # ── 14. Train ───────────────────────────────────────────────────────
    trainer = Trainer(model, training_config, device=device, da_module=da_module)
    trainer.fit(train_loader, val_loader, target_loader=target_loader)

    # ── 15. Save cell types ─────────────────────────────────────────────
    cell_types_path = Path(ckpt_dir) / "cell_types.json"
    with open(cell_types_path, "w") as f:
        json.dump(cell_types, f, indent=2)
    tee(f"Saved cell types: {cell_types_path}")

    # ── 16. Final evaluation on validation set ──────────────────────────
    tee("\nRunning final evaluation on validation set...")
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in val_loader:
            gene_ids_b = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            mask = batch["src_key_padding_mask"].to(device)
            output = model(gene_ids_b, values, mask)
            all_preds.append(output["proportions"].cpu().numpy())
            all_true.append(batch["proportions"].numpy())

    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_true, axis=0)

    metrics = compute_metrics(predictions, ground_truth, cell_types)
    tee(f"  Mean Pearson: {metrics['mean_pearson']:.4f}")
    tee(f"  Mean RMSE:    {metrics['mean_rmse']:.4f}")
    tee(f"  Mean MAE:     {metrics['mean_mae']:.4f}")

    # ── 17. Save validation set predictions (for reference) ─────────────
    val_pred_path = Path(ckpt_dir) / "val_proportions.csv"
    val_gt_path = Path(ckpt_dir) / "val_ground_truth.csv"
    pd.DataFrame(predictions, columns=cell_types).to_csv(val_pred_path)
    pd.DataFrame(ground_truth, columns=cell_types).to_csv(val_gt_path)
    tee(f"Saved val predictions: {val_pred_path}")
    tee(f"Saved val ground truth: {val_gt_path}")

    val_eval_path = Path(ckpt_dir) / "val_eval_results.json"
    with open(val_eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    tee(f"Saved val eval results: {val_eval_path}")

    # ── 18. H5 bulk test evaluation ─────────────────────────────────────
    if bulk_df is not None:
        tee("\nRunning H5 bulk test evaluation...")
        try:
            # Subset bulk to training gene set (same order as training)
            bulk_test = bulk_df[gene_symbols].values.astype(np.float64)

            # CPM → log1p (same normalization as pseudo-bulk pipeline)
            row_sums = bulk_test.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            bulk_test = np.log1p((bulk_test / row_sums) * 1e4)

            # Apply same binning scheme (if used during training)
            if is_binned:
                bulk_test = bin_expression(
                    bulk_test, np.array(gene_symbols), bin_info, n_bins=51,
                )

            # Dummy proportions (not used for inference, just dataset shape)
            dummy_props = np.zeros((len(bulk_test), n_cell_types), dtype=np.float64)
            h5_dataset = PseudoBulkDataset(
                bulk_matrix=bulk_test,
                proportions=dummy_props,
                gene_ids=gene_ids,
                vocab=vocab,
                max_len=max_len,
                is_binned=is_binned,
                is_rank_value=False,
            )
            h5_loader = DataLoader(
                h5_dataset,
                batch_size=training_config.batch_size,
                shuffle=False,
                num_workers=training_config.num_workers,
            )

            model.eval()
            all_h5_preds = []
            with torch.no_grad():
                for batch in h5_loader:
                    gene_ids_b = batch["gene_ids"].to(device)
                    values = batch["values"].to(device)
                    mask = batch["src_key_padding_mask"].to(device)
                    output = model(gene_ids_b, values, mask)
                    all_h5_preds.append(output["proportions"].cpu().numpy())

            h5_predictions = np.concatenate(all_h5_preds, axis=0)

            # Align GT to training cell types and bulk sample list
            if gt_df is not None:
                common_types = [c for c in cell_types if c in gt_df.columns]
                if len(common_types) > 0:
                    gt_aligned = gt_df.reindex(index=bulk_df.index).dropna()
                    if len(gt_aligned) != len(h5_predictions):
                        tee(f"  WARNING: GT samples ({len(gt_aligned)}) != bulk ({len(h5_predictions)}), skipping metrics")
                        gt_df = None
                    else:
                        h5_gt = gt_aligned[common_types].values.astype(np.float64)

            # Save main proportions.csv (H5 bulk predictions)
            h5_pred_path = Path(ckpt_dir) / "proportions.csv"
            pd.DataFrame(h5_predictions, columns=cell_types).to_csv(h5_pred_path)
            tee(f"Saved H5 bulk predictions: {h5_pred_path}")

            # Save GT CSV and metrics if GT is available
            if gt_df is not None and len(common_types) > 0:
                h5_gt_path = Path(ckpt_dir) / "ground_truth.csv"
                pd.DataFrame(h5_gt, columns=common_types).to_csv(h5_gt_path)
                tee(f"Saved H5 ground truth: {h5_gt_path}")

                h5_metrics = compute_metrics(h5_predictions[:, [cell_types.index(c) for c in common_types]], h5_gt, common_types)
                tee(f"\n  H5 Bulk Test Results:")
                tee(f"  Mean Pearson: {h5_metrics['mean_pearson']:.4f}")
                tee(f"  Mean RMSE:   {h5_metrics['mean_rmse']:.4f}")
                tee(f"  Mean MAE:    {h5_metrics['mean_mae']:.4f}")

                h5_eval_path = Path(ckpt_dir) / "eval_results.json"
                with open(h5_eval_path, "w") as f:
                    json.dump(h5_metrics, f, indent=2)
                tee(f"Saved H5 eval results: {h5_eval_path}")

        except Exception as e:
            tee(f"  WARNING: H5 bulk evaluation failed: {e}")
            import traceback
            tee(traceback.format_exc())

    tee("\nTraining completed successfully.")
    tee("=" * 60)

    log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="scGPT deconvolution training (standalone)",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--log_file", default=None,
        help="Path to log file (default: <checkpoint_dir>/run.log)",
    )
    args = parser.parse_args()
    main(args.config, args.log_file)
