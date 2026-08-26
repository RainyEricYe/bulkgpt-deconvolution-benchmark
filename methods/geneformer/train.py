#!/usr/bin/env python3
"""Standalone Geneformer deconvolution training script.

Runs entirely in-process (no subprocess calls).  No imports from ``src/bulkgpt``
or ``methods.geneformer._utils``.

Runnable as::

    python methods/geneformer/train.py --config methods/geneformer/configs/ft.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

# -- Ensure core/ and methods/ packages are importable ------------------------
_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.deconv.config import DeconvHeadConfig, TrainingConfig
from core.deconv.data import (
    Preprocessor,
    PseudoBulkDataset,
    RealBulkDataset,
    filter_genes_to_vocab,
    prepare_pseudo_bulk,
)
from core.deconv.trainer import Trainer
from core.deconv.utils import set_seed, setup_logging

from methods.geneformer.data import map_symbol_to_ensembl, rank_value_encode
from methods.geneformer.model import GeneformerDeconvModel, create_geneformer_backbone


# ---------------------------------------------------------------------------
# Inline evaluation (mirrors src/bulkgpt/evaluation/metrics.py)
# ---------------------------------------------------------------------------


def _compute_correlations(
    true_props: np.ndarray,
    pred_props: np.ndarray,
) -> dict:
    """Per-cell-type Pearson and Spearman correlations."""
    n_types = true_props.shape[1]
    pearson_rs: list[float] = []
    spearman_rs: list[float] = []
    for i in range(n_types):
        if np.std(true_props[:, i]) > 0 and np.std(pred_props[:, i]) > 0:
            pr, _ = pearsonr(true_props[:, i], pred_props[:, i])
            sr, _ = spearmanr(true_props[:, i], pred_props[:, i])
        else:
            pr = sr = 0.0
        pearson_rs.append(pr)
        spearman_rs.append(sr)
    return {
        "pearson_mean": float(np.mean(pearson_rs)),
        "spearman_mean": float(np.mean(spearman_rs)),
        "pearson_per_type": pearson_rs,
        "spearman_per_type": spearman_rs,
    }


def _compute_rmse(true_props: np.ndarray, pred_props: np.ndarray) -> dict:
    """Root-mean-square error."""
    mse_per_type = np.mean((pred_props - true_props) ** 2, axis=0)
    rmse_per_type = np.sqrt(mse_per_type)
    return {
        "rmse_overall": float(np.sqrt(np.mean((pred_props - true_props) ** 2))),
        "rmse_per_type": rmse_per_type.tolist(),
        "rmse_mean_per_type": float(np.mean(rmse_per_type)),
    }


def _compute_mae(true_props: np.ndarray, pred_props: np.ndarray) -> dict:
    """Mean absolute error."""
    return {
        "mae_overall": float(np.mean(np.abs(pred_props - true_props))),
        "mae_per_type": np.mean(np.abs(pred_props - true_props), axis=0).tolist(),
    }


def evaluate_deconvolution(
    true_props: np.ndarray,
    pred_props: np.ndarray,
    cell_types: list[str] | None = None,
) -> dict:
    """Full evaluation suite for deconvolution.

    Returns a flat dict with the same key names as ``src/bulkgpt/evaluation``:
    ``pearson_mean``, ``spearman_mean``, ``rmse_overall``, ``mae_overall``,
    plus per-type arrays.
    """
    results: dict = {}
    results.update(_compute_correlations(true_props, pred_props))
    results.update(_compute_rmse(true_props, pred_props))
    results.update(_compute_mae(true_props, pred_props))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str, log_file: str | None = None) -> None:
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract config sections
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    # -- Checkpoint directory / log file ----------------------------------
    ckpt_dir = paths_cfg.get(
        "checkpoint_dir",
        training_cfg.get("checkpoint_dir", "checkpoints/geneformer"),
    )
    if log_file is None:
        log_file = str(Path(ckpt_dir) / "run.log")

    log_path, tee, log_fh = setup_logging(log_file)

    try:
        tee("=" * 60)
        tee("Geneformer method - train (standalone)")
        tee(f"Config: {config_path.resolve()}")
        tee(f"Log file: {log_path.resolve()}")

        # -- Reproducibility -----------------------------------------------
        seed = training_cfg.get("seed", 42)
        set_seed(seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tee(f"Using device: {device}")

        # -- 1. Load scRNA reference ---------------------------------------
        from core.data_loader import load_sc_ref
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
            ensembl_map = map_symbol_to_ensembl(
                list(adata.var_names), model_dir=pretrained_model
            )
            adata.var_names = [ensembl_map[str(g)] for g in adata.var_names]
        else:
            ensembl_map = {str(g): str(g) for g in adata.var_names}
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
        tee(f"Preprocessed: {adata.shape[0]} cells, {adata.shape[1]} genes")

        # -- 6. Generate pseudo-bulk --------------------------------------
        tee("Generating pseudo-bulk samples...")
        n_pseudo_bulk = training_cfg.get("n_pseudo_bulk", 10000)
        alpha = training_cfg.get("proportion_alpha", 1.0)

        bulk_matrix, proportions, cell_types = prepare_pseudo_bulk(
            adata,
            celltype_col=celltype_col,
            n_samples=n_pseudo_bulk,
            alpha=alpha,
            seed=seed,
        )
        tee(f"Generated {len(bulk_matrix)} pseudo-bulk samples")
        tee(f"Cell types ({len(cell_types)}): {cell_types}")
        n_cell_types = len(cell_types)

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

        # -- 11. Build GeneformerDeconvModel ------------------------------
        unfreeze = training_cfg.get("unfreeze_backbone", False)
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
            freeze_backbone=not unfreeze,
        )
        if unfreeze:
            model._unfreeze_backbone()

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tee(f"Trainable parameters: {n_params:,}")
        tee(f"Backbone frozen: {not unfreeze}")

        # -- 12. Create PseudoBulkDataset ---------------------------------
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

        from core.deconv.embedding import split_indices
        _tr = training_cfg.get("train_ratio", 0.8)
        _va = training_cfg.get("val_ratio", 0.2)
        train_idx, val_idx, _ = split_indices(
            len(dataset), train_ratio=_tr, val_ratio=_va,
            test_ratio=0.0, seed=training_cfg.get("seed", 42),
        )
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        tee(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        num_workers = training_cfg.get("num_workers", 4)
        batch_size = training_cfg.get("batch_size", 64)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # -- 13. Real-bulk loading for domain adaptation --------------------
        real_bulk_path = dataset_cfg.get("real_bulk_path") or training_cfg.get("real_bulk_path")
        target_loader = None
        da_module = None

        if real_bulk_path:
            from core.data_loader import load_data

            tee(f"Loading real-bulk for domain adaptation: {real_bulk_path}")
            bundle = load_data(real_bulk_path)
            bulk_df = bundle.bulk  # (samples, genes) DataFrame
            tee(f"  Real-bulk raw: {bulk_df.shape[0]} samples, {bulk_df.shape[1]} genes")

            # Training gene set: in-vocab genes after HVG
            training_genes = gene_names[valid]
            bulk_genes = list(bulk_df.columns)

            # Map bulk gene names if needed (same mapping as training)
            n_ensg_bulk = sum(1 for g in bulk_genes if str(g).startswith("ENSG"))
            if n_ensg_bulk < len(bulk_genes) * 0.5:
                bulk_genes_mapped = [ensembl_map.get(str(g), str(g)) for g in bulk_genes]
            else:
                bulk_genes_mapped = bulk_genes

            # Align to training gene order (column-by-column)
            bulk_aligned = np.zeros(
                (bulk_df.shape[0], len(training_genes)), dtype=np.float64
            )
            for j, g in enumerate(training_genes):
                if g in bulk_genes_mapped:
                    idx = bulk_genes_mapped.index(g)
                    bulk_aligned[:, j] = bulk_df.iloc[:, idx].values

            n_present = int(np.any(bulk_aligned != 0, axis=0).sum())
            tee(
                f"  Aligned: {n_present}/{len(training_genes)} training genes "
                f"present in real bulk"
            )

            # CPM normalize + log1p (matching pseudo-bulk pipeline)
            row_sums = bulk_aligned.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            bulk_aligned = (bulk_aligned / row_sums) * 1e4
            bulk_aligned = np.log1p(bulk_aligned)

            # Rank-value encode real bulk
            rank_real, sorted_indices_real = rank_value_encode(bulk_aligned)
            rank_real = rank_real.astype(np.float32)

            # Build RealBulkDataset
            real_dataset = RealBulkDataset(
                rank_real,
                valid_gene_ids,
                gf_vocab,
                max_len=max_seq_len,
                pad_token="<pad>",
                cls_token="<cls>",
                is_rank_value=True,
                sorted_indices=sorted_indices_real,
            )
            tee(f"  Real-bulk dataset: {len(real_dataset)} samples")

            target_loader = DataLoader(
                real_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

            # Create domain adaptation module
            da_method = training_cfg.get("da_method")
            if da_method and da_method != "none":
                from core.deconv.domain_adaptation import DomainAdaptationModule

                backbone_dim = backbone.config.hidden_size
                da_module = DomainAdaptationModule(backbone_dim)
                da_module.to(device)
                tee(
                    f"  Domain adaptation: method={da_method}, "
                    f"lambda={training_cfg.get('da_lambda', 0.05)}, "
                    f"grl_lambda={training_cfg.get('da_grl_lambda', 1.0)}"
                )

        # -- 14. Create Trainer and train ---------------------------------
        train_cfg = TrainingConfig(
            seed=seed,
            epochs=training_cfg.get("epochs", 30),
            batch_size=batch_size,
            lr=training_cfg.get("lr", 7e-5),
            backbone_lr=training_cfg.get("backbone_lr", None),
            n_pseudo_bulk=n_pseudo_bulk,
            proportion_alpha=alpha,
            loss_type=training_cfg.get("loss_type", "mse_kl"),
            num_workers=num_workers,
            checkpoint_dir=ckpt_dir,
            use_wandb=training_cfg.get("use_wandb", False),
            train_ratio=_tr,
            n_hvg=n_hvg,
            max_seq_len=max_seq_len,
            da_method=training_cfg.get("da_method"),
            da_lambda=training_cfg.get("da_lambda", 0.05),
            da_grl_lambda=training_cfg.get("da_grl_lambda", 1.0),
        )

        trainer = Trainer(model=model, config=train_cfg, device=device, da_module=da_module)
        trainer.fit(train_loader, val_loader, target_loader=target_loader)

        # -- 14. Save cell types ------------------------------------------
        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        with open(ckpt_path / "cell_types.json", "w") as f:
            json.dump(cell_types, f)
        tee(f"Saved cell_types.json to {ckpt_path}")

        # -- 15. Final evaluation on validation set -----------------------
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

        results = evaluate_deconvolution(true_props, pred_props, cell_types)
        tee("\n=== Final Evaluation ===")
        tee(f"  Pearson r:  {results['pearson_mean']:.4f}")
        tee(f"  Spearman r: {results['spearman_mean']:.4f}")
        tee(f"  RMSE:       {results['rmse_overall']:.4f}")
        tee(f"  MAE:        {results['mae_overall']:.4f}")
        tee("")
        tee("Per cell-type Pearson r:")
        for ct, r in zip(cell_types, results["pearson_per_type"]):
            tee(f"  {ct:40s}: {r:.4f}")

        # -- 16. Save predictions and ground truth CSVs -------------------
        header = ",".join(cell_types)
        pred_path = ckpt_path / "proportions.csv"
        np.savetxt(pred_path, pred_props, delimiter=",", header=header, comments="")
        true_path = ckpt_path / "ground_truth.csv"
        np.savetxt(true_path, true_props, delimiter=",", header=header, comments="")
        tee(f"Saved predictions to {pred_path}")
        tee(f"Saved ground truth to {true_path}")

        # -- 17. Save eval results JSON -----------------------------------
        _to_serializable = lambda x: (
            float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
        )
        results_serializable = json.loads(
            json.dumps(results, default=_to_serializable)
        )
        eval_path = ckpt_path / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump(results_serializable, f, indent=2)
        tee(f"Saved eval results to {eval_path}")

        tee("Training completed successfully")
        tee("=" * 60)

    finally:
        log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geneformer deconvolution: train (standalone)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Path to log file (default: <checkpoint_dir>/run.log)",
    )
    args = parser.parse_args()
    main(args.config, args.log_file)
