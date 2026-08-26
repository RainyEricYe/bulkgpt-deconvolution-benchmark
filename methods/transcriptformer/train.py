#!/usr/bin/env python3
"""Standalone training script for TranscriptFormer embedding-based deconvolution.

Computes TranscriptFormer cell embeddings from scRNA-seq, generates pseudo-bulk
mixtures, trains a DeconvHead MLP, and saves the trained head.

Requires the TranscriptFormer package (https://github.com/suinleelab/TranscriptFormer)
and a pretrained TF checkpoint.

Usage:
    python methods/transcriptformer/train.py --config methods/transcriptformer/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

# -- Ensure core/ is importable -------------------------------------------------
_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.deconv.domain_adaptation import mmd_loss
from core.deconv.embedding import EmbeddingDeconvHead, ExpressionMixGenerator, MixGenerator, evaluate_predictions
from core.deconv.utils import external_dir, find_project_root, set_seed, setup_logging
from methods.transcriptformer.data import map_symbol_to_ensembl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _import_tf():
    """Resolve TranscriptFormer dependency at runtime (pip -> data/external/ -> TF_REPO)."""
    try:
        from omegaconf import OmegaConf
        from transcriptformer.model.inference import run_inference
        return OmegaConf, run_inference
    except ImportError:
        pass
    _tf_local = external_dir() / "transcriptformer" / "repo" / "src"
    if _tf_local.is_dir():
        sys.path.insert(0, str(_tf_local))
        from omegaconf import OmegaConf
        from transcriptformer.model.inference import run_inference
        return OmegaConf, run_inference
    tf_repo = os.environ.get("TF_REPO")
    if tf_repo:
        sys.path.insert(0, str(Path(tf_repo) / "repo" / "src"))
        sys.path.insert(0, tf_repo)
        from omegaconf import OmegaConf
        from transcriptformer.model.inference import run_inference
        return OmegaConf, run_inference
    print(
        "ERROR: TranscriptFormer package not found.\n"
        "  Install via: bash data/prepare/download_external.sh --tf\n"
        "  Or set TF_REPO env var to the TranscriptFormer repo root.\n"
        "  Or install the transcriptformer package (if available).",
        file=sys.stderr,
    )
    sys.exit(1)


def encode_tf(raw_counts, gene_names, barcodes, tf_checkpoint_dir):
    """Compute TranscriptFormer cell embeddings from raw counts."""
    OmegaConf, run_inference = _import_tf()
    import anndata as ad
    from scipy.sparse import csr_matrix

    with open(os.path.join(tf_checkpoint_dir, "config.json")) as f:
        mc = json.load(f)["model"]

    cfg = OmegaConf.create({
        "model": {
            "model_config": {
                **mc["model_config"],
                "gene_head_hidden_dim": 2048,
                "use_aux": False,
                "compile_block_mask": True,
            },
            "data_config": {
                "aux_vocab_path": os.path.join(tf_checkpoint_dir, "vocabs"),
                "esm2_mappings_path": os.path.join(tf_checkpoint_dir, "vocabs"),
                "pin_memory": True,
                "aux_cols": "assay",
                "gene_col_name": "ensembl_id",
                "clip_counts": 30,
                "filter_to_vocabs": True,
                "filter_outliers": 0.0,
                "pad_zeros": True,
                "normalize_to_scale": 0,
                "n_data_workers": 1,
                "sort_genes": False,
                "randomize_genes": False,
                "min_expressed_genes": 0,
                "gene_pad_token": "[PAD]",
                "aux_pad_token": "unknown",
                "esm2_mappings": ["homo_sapiens_gene.h5"],
                "special_tokens": ["unknown", "[PAD]", "[START]", "[END]",
                                   "[RD]", "[CELL]", "[MASK]"],
                "remove_duplicate_genes": True,
                "use_raw": "auto",
            },
            "loss_config": mc.get("loss_config", {"gene_id_loss_weight": 1.0}),
            "inference_config": {
                "data_files": None,
                "output_path": None,
                "output_filename": "embeddings.h5ad",
                "batch_size": 8,
                "precision": "16",
                "load_checkpoint": os.path.join(tf_checkpoint_dir, "model_weights.pt"),
                "output_keys": ["embeddings"],
                "obs_keys": ["all"],
                "num_gpus": 1,
                "device": "auto",
                "emb_type": "cell",
                "pretrained_embedding": None,
                "use_oom_dataloader": False,
                "special_tokens": [],
            },
        }
    })

    adata = ad.AnnData(
        X=csr_matrix(raw_counts),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names),
    )
    adata.var["ensembl_id"] = adata.var.index
    adata.obs["assay"] = "10x 3' transcription profiling"

    result = run_inference(cfg, data_files=[adata])
    return result.obsm["embeddings"]


def main(config_path: str, log_file: str | None = None) -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"ERROR: Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    project_root = find_project_root()
    os.chdir(str(project_root))

    def safe_decode(data):
        if data.dtype.kind == 'S':
            return [x.decode('utf-8') for x in data]
        if data.dtype.kind == 'O':
            return [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in data]
        return [str(x) for x in data]

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    output_dir = project_root / paths_cfg.get("output_dir", "results/transcriptformer")
    output_dir.mkdir(parents=True, exist_ok=True)
    dst_cfg = output_dir / "config.yaml"
    if cfg_path.resolve() != dst_cfg.resolve():
        shutil.copy2(cfg_path, dst_cfg)

    if log_file is None:
        log_file = str(output_dir / "train.log")

    _, tee, log_fh = setup_logging(log_file)
    warnings.filterwarnings("ignore")

    try:
        tee("=" * 60)
        tee("TranscriptFormer -- train (standalone)")
        tee(f"Config: {cfg_path.resolve()}")
        tee(f"Output: {output_dir}")
        tee(f"Device: {device}")

        # -- Resolve paths -------------------------------------------------------
        def _resolve(p):
            p = Path(p)
            return p if p.is_absolute() else project_root / p

        tf_ckpt_dir = _resolve(model_cfg["tf_checkpoint_dir"])

        embed_dim = model_cfg.get("embed_dim", 2048)
        hidden_dim = model_cfg.get("hidden_dim", 512)
        progressive = model_cfg.get("progressive", False)
        n_progressive_layers = model_cfg.get("n_progressive_layers", 3)
        n_pseudo_bulk = train_cfg.get("n_pseudo_bulk", 5000)
        n_epochs = train_cfg.get("epochs", 30)
        batch_size = train_cfg.get("batch_size", 64)
        lr = train_cfg.get("lr", 7e-5)
        weight_decay = train_cfg.get("weight_decay", 1e-5)
        seed = train_cfg.get("seed", 42)
        pb_space = model_cfg.get("pseudo_bulk_space", "embedding")
        set_seed(seed)

        # -- 1. Load scRNA reference ---------------------------------------------
        from core.data_loader import load_sc_ref

        data_path = data_cfg.get("data_path")
        if data_path:
            ref = load_sc_ref(_resolve(data_path))
        else:
            sc_ref = _resolve(data_cfg["sc_ref"])
            tee(f"\n[1] Loading scRNA-seq reference: {sc_ref}")
            import anndata as ad
            ref = ad.read_h5ad(str(sc_ref))
        ref.obs_names_make_unique()

        celltype_col = data_cfg.get("celltype_col", "cell_type")
        if celltype_col not in ref.obs.columns:
            for candidate in ["cell_type", "CellType", "celltype", "cell_ontology_class",
                              "cluster_name", "label"]:
                if candidate in ref.obs.columns:
                    celltype_col = candidate
                    break
            else:
                raise ValueError(f"Cell type column not found. Available: {list(ref.obs.columns)}")

        cell_types = sorted(ref.obs[celltype_col].unique().tolist())
        n_ct = len(cell_types)
        t2i = {t: i for i, t in enumerate(cell_types)}
        sc_labels = np.array([t2i[t] for t in ref.obs[celltype_col].values])

        sc_counts = ref.X
        if hasattr(sc_counts, "toarray"):
            sc_counts = sc_counts.toarray()
        sc_counts = np.asarray(sc_counts)
        sc_genes = list(ref.var_names)
        sc_barcodes = list(ref.obs_names)

        # Save raw gene symbols before Ensembl mapping (needed for bulk alignment)
        sc_genes_raw = list(sc_genes)

        # Map gene symbols to Ensembl IDs if needed
        n_ensg = sum(1 for g in sc_genes if str(g).startswith("ENSG"))
        if n_ensg < len(sc_genes) * 0.5:
            tee("  Less than 50% genes are Ensembl IDs, mapping symbols to Ensembl...")
            _gc = model_cfg.get("gene_cache")
            _gc_path = str(_resolve(_gc)) if _gc else None
            sym_to_ensg = map_symbol_to_ensembl(sc_genes, mapping_path=_gc_path)
            sc_genes = [sym_to_ensg.get(g, g) for g in sc_genes]
            # Filter unmapped genes
            valid = [i for i, g in enumerate(sc_genes) if g.startswith("ENSG")]
            if len(valid) < 10:
                raise ValueError(f"Too few genes mapped to Ensembl IDs ({len(valid)}/{len(sc_genes)})")
            # sc_counts is (cells, genes); filter along gene axis
            sc_counts = sc_counts[:, valid]
            sc_genes = [sc_genes[i] for i in valid]
            # Build mapping from raw symbol → Ensembl ID for the filtered subset
            raw_to_ensg = {}
            valid_set = set(valid)
            ensg_idx = 0
            for i, raw_g in enumerate(sc_genes_raw):
                if i in valid_set:
                    raw_to_ensg[raw_g.upper()] = sc_genes[ensg_idx]
                    ensg_idx += 1
            tee(f"  Retained {len(sc_genes)} genes with Ensembl IDs")
        else:
            tee(f"  Genes already Ensembl IDs ({n_ensg}/{len(sc_genes)}), skipping mapping")

        tee(f"  Cells: {sc_counts.shape[0]}, Genes: {sc_counts.shape[1]}, Types: {n_ct}")
        tee(f"  Cell types: {cell_types}")
        tee(f"  Gene IDs: {'ENSG' if any(g.startswith('ENSG') for g in sc_genes) else 'symbols'}")

        # Fix cell_type column if dropped by H5 orientation detection
        if celltype_col not in ref.obs.columns:
            from methods._shared.h5_sc_helper import ensure_sc_celltypes
            ref = ensure_sc_celltypes(ref, _resolve(data_cfg.get("data_path", "")), celltype_col)
            if celltype_col in ref.obs.columns:
                tee(f"  Restored {celltype_col} from H5 ({ref.obs[celltype_col].nunique()} types)")

        # -- 1c. Load H5 bulk data for test evaluation --------------------------
        bulk_h5_path = data_cfg.get("data_path")
        bulk_h5_matrix = None
        bulk_h5_genes = None
        bulk_h5_samples = None
        gt_data = None
        if bulk_h5_path and _resolve(bulk_h5_path).exists():
            tee(f"\n[1c] Loading H5 bulk for test eval: {bulk_h5_path}")
            from core.data_loader import load_data
            _bundle = load_data(str(_resolve(bulk_h5_path)))
            if _bundle.bulk is not None:
                bulk_h5_matrix = _bundle.bulk.values.astype(np.float64)
                bulk_h5_genes = list(_bundle.bulk.columns)
                bulk_h5_samples = list(_bundle.bulk.index)
                gt_data = _bundle.gt
                tee(f"  Bulk: {bulk_h5_matrix.shape[0]} samples, {bulk_h5_matrix.shape[1]} genes")
                if gt_data is not None:
                    tee(f"  GT: {gt_data.shape[0]} samples, {list(gt_data.columns)}")

        # -- 2a. Load real bulk for domain adaptation (optional) ------------------
        da_method = train_cfg.get("da_method")
        da_lambda = train_cfg.get("da_lambda", 0.05)
        real_emb = None
        real_bulk_path = data_cfg.get("real_bulk_path")
        if real_bulk_path and da_method:
            tee(f"\n[DA] Loading real bulk: {real_bulk_path}")
            with h5py.File(str(_resolve(real_bulk_path)), "r") as f:
                bulk_values = f["bulk/values"][:]
                bulk_rownames = safe_decode(f["bulk/rownames"][:])
                bulk_colnames = safe_decode(f["bulk/colnames"][:])
            # Transpose from (n_genes, n_samples) -> (n_samples, n_genes)
            if bulk_values.shape[0] == len(bulk_rownames) and bulk_values.shape[1] == len(bulk_colnames):
                bulk_matrix = bulk_values.T
            else:
                bulk_matrix = bulk_values
            tee(f"  Bulk samples: {bulk_matrix.shape[0]}, Genes: {bulk_matrix.shape[1]}")

            # Align bulk genes with scRNA reference (use raw symbols, map to Ensembl IDs)
            bulk_genes_upper = [g.upper() for g in bulk_rownames]
            if n_ensg < len(sc_genes_raw) * 0.5:
                # scRNA was mapped to Ensembl — use raw-to-Enseml mapping for alignment
                common_upper = [bg for bg in bulk_genes_upper if bg in raw_to_ensg]
                bulk_gene_mask = [g in common_upper for g in bulk_genes_upper]
                bulk_matrix = bulk_matrix[:, bulk_gene_mask]
                bulk_genes_filtered = [raw_to_ensg[bulk_rownames[i].upper()]
                                       for i, m in enumerate(bulk_gene_mask) if m]
            else:
                # scRNA already had Ensembl IDs — align directly
                sc_genes_upper = [g.upper() for g in sc_genes]
                common_upper = set(bulk_genes_upper) & set(sc_genes_upper)
                bulk_gene_mask = [g in common_upper for g in bulk_genes_upper]
                bulk_matrix = bulk_matrix[:, bulk_gene_mask]
                bulk_genes_filtered = [bulk_rownames[i] for i, m in enumerate(bulk_gene_mask) if m]
            if len(common_upper) < 100:
                tee(f"  WARNING: only {len(common_upper)} common genes between bulk and scRNA")
            tee(f"  After alignment: {len(bulk_genes_filtered)} genes overlap")

            # CPM+log1p normalize if bulk values look like raw counts
            row_sums = bulk_matrix.sum(axis=1, keepdims=True)
            if row_sums.mean() > 50:
                bulk_matrix = (bulk_matrix / np.where(row_sums == 0, 1, row_sums)) * 1e4
                bulk_matrix = np.log1p(bulk_matrix)
                tee(f"  Applied CPM+log1p normalization")

            # Encode through TranscriptFormer
            t0 = time.monotonic()
            real_emb = encode_tf(bulk_matrix, bulk_genes_filtered,
                                 [f"real_{i}" for i in range(len(bulk_matrix))],
                                 str(tf_ckpt_dir))
            tee(f"  Real bulk embeddings: {real_emb.shape} [{time.monotonic() - t0:.1f}s]")
            tee(f"  DA method: {da_method}, lambda={da_lambda}")

        # -- 2b. Generate pseudo-bulk (expression or embedding space) -------------
        if pb_space == "expression":
            tee(f"\n[2] Generating expression-space pseudo-bulk ({n_pseudo_bulk} samples)...")
            gen = ExpressionMixGenerator(sc_counts, sc_labels, n_ct, n_pseudo_bulk, seed)
            pb_expr, pb_props = gen.generate()
            tee(f"  Pseudo-bulk expression: {pb_expr.shape}")

            # Encode each pseudo-bulk sample through TF (frozen)
            tee(f"\n[3] Encoding pseudo-bulk through TranscriptFormer...")
            t0 = time.monotonic()
            pb_emb = encode_tf(pb_expr, sc_genes, [f"pb_{i}" for i in range(len(pb_expr))],
                               str(tf_ckpt_dir))
            tee(f"  Pseudo-bulk embeddings: {pb_emb.shape} [{time.monotonic() - t0:.1f}s]")
        else:
            # Embedding-space: pre-compute cell embeddings, then mix (original behaviour)
            tee(f"\n[2] Computing TF cell embeddings...")
            t0 = time.monotonic()
            cell_emb = encode_tf(sc_counts, sc_genes, sc_barcodes, str(tf_ckpt_dir))
            tee(f"  {cell_emb.shape} [{time.monotonic() - t0:.1f}s]")

            tee(f"\n[3] Generating pseudo-bulk ({n_pseudo_bulk} samples)...")
            gen = MixGenerator(cell_emb, sc_labels, n_ct, n_pseudo_bulk, seed)
            pb_emb, pb_props = gen.generate()
            tee(f"  {pb_emb.shape}")

        # -- 4. Train DeconvHead --------------------------------------------------
        tee(f"\n[4] Training DeconvHead ({n_epochs} epochs)...")
        n = len(pb_emb)
        from core.deconv.embedding import PseudoBulkConfig, split_indices
        _pb_cfg = PseudoBulkConfig(
            n_pseudo_bulk=n,
            train_ratio=train_cfg.get("train_ratio", 0.8),
            val_ratio=train_cfg.get("val_ratio", 0.2),
            test_ratio=train_cfg.get("test_ratio", 0.0),
            seed=train_cfg.get("seed", 42),
        )
        n_tr, n_va = _pb_cfg.n_train, _pb_cfg.n_val

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(pb_emb[:n_tr]).float(),
                torch.from_numpy(pb_props[:n_tr]).float()),
            batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(pb_emb[n_tr:n_tr + n_va]).float(),
                torch.from_numpy(pb_props[n_tr:n_tr + n_va]).float()),
            batch_size=batch_size)

        model = EmbeddingDeconvHead(
            embed_dim, hidden_dim, n_ct,
            progressive=progressive, n_progressive_layers=n_progressive_layers,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        use_da = real_emb is not None and da_method
        real_tensor = torch.from_numpy(real_emb).float().to(device) if use_da else None

        for ep in range(1, n_epochs + 1):
            model.train()
            train_loss = 0.0
            da_loss_acc = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                sup_loss = torch.nn.functional.mse_loss(pred, y)

                if use_da:
                    if da_method == "mmd":
                        h_pseudo = model.get_hidden(x)
                        h_real = model.get_hidden(real_tensor[torch.randperm(len(real_tensor))[:x.size(0)]])
                        da_loss_val = torch.stack(
                            [mmd_loss(hp, hr, kernel="linear") for hp, hr in zip(h_pseudo, h_real)]
                        ).mean()
                    elif da_method == "entropy":
                        real_batch = real_tensor[torch.randperm(len(real_tensor))[:x.size(0)]]
                        real_pred = model(real_batch)
                        eps32 = 1e-8
                        da_loss_val = -(real_pred * (real_pred + eps32).log()).sum(dim=-1).mean()
                    else:
                        da_loss_val = torch.tensor(0.0, device=device)

                    loss = sup_loss + da_lambda * da_loss_val
                    da_loss_acc += da_loss_val.item()
                else:
                    loss = sup_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    val_loss += torch.nn.functional.mse_loss(
                        model(x.to(device)), y.to(device)).item()
            scheduler.step()

            if ep % 10 == 0 or ep == 1:
                msg = f"  E{ep:2d}  train={train_loss / len(train_loader):.6f}  val={val_loss / len(val_loader):.6f}"
                if use_da:
                    msg += f"  da={da_loss_acc / len(train_loader):.6f}"
                tee(msg)

        # -- 5. Evaluate on test set (only if test_ratio > 0) ------------------
        predictions = None
        ground_truth = None
        eval_metrics = None
        if n_tr + n_va < len(pb_emb):
            tee(f"\n[5] Evaluating on test set ({len(pb_emb) - n_tr - n_va} samples)...")
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(pb_emb[n_tr + n_va:]).float(),
                    torch.from_numpy(pb_props[n_tr + n_va:]).float()),
                batch_size=batch_size)
            all_pred, all_true = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    all_pred.append(model(x.to(device)).cpu().numpy())
                    all_true.append(y.numpy())
            if all_pred:
                predictions = np.concatenate(all_pred)
                ground_truth = np.concatenate(all_true)
                eval_metrics = evaluate_predictions(predictions, ground_truth, cell_types)
                tee(f"  Test Pearson: {eval_metrics['pearson_mean']:.4f}")

        # -- 5b. H5 bulk test evaluation ------------------------------------------
        if bulk_h5_matrix is not None and bulk_h5_genes is not None:
            tee(f"\n[5b] H5 bulk test evaluation...")
            try:
                # Align bulk genes: map symbols to Ensembl IDs if needed
                bulk_gene_list = list(bulk_h5_genes)
                n_ensg_bulk = sum(1 for g in bulk_gene_list if str(g).startswith("ENSG"))
                if n_ensg_bulk < len(bulk_gene_list) * 0.5:
                    sym_to_ensg_bulk = map_symbol_to_ensembl(bulk_gene_list, mapping_path=_gc_path)
                    bulk_mapped = [sym_to_ensg_bulk.get(g, "") for g in bulk_gene_list]
                else:
                    bulk_mapped = bulk_gene_list

                # Intersect with training Ensembl IDs
                sc_ensg_set = set(sc_genes)
                common_mask = [g in sc_ensg_set for g in bulk_mapped]
                if sum(common_mask) < 10:
                    tee(f"  WARNING: only {sum(common_mask)} common genes, skipping")
                else:
                    bulk_expr = bulk_h5_matrix[:, common_mask].copy()
                    bulk_genes_filtered = [bulk_mapped[i] for i, m in enumerate(common_mask) if m]

                    # CPM + log1p
                    rsum = bulk_expr.sum(axis=1, keepdims=True)
                    bulk_expr = np.log1p((bulk_expr / np.where(rsum == 0, 1, rsum)) * 1e4)

                    t0 = time.monotonic()
                    bulk_emb = encode_tf(bulk_expr, bulk_genes_filtered,
                                         [f"bulk_{i}" for i in range(len(bulk_expr))],
                                         str(tf_ckpt_dir))
                    tee(f"  H5 bulk embeddings: {bulk_emb.shape} [{time.monotonic() - t0:.1f}s]")

                    model.eval()
                    with torch.no_grad():
                        bulk_pred = model(torch.from_numpy(bulk_emb).float().to(device)).cpu().numpy()
                    bulk_pred = np.maximum(bulk_pred, 0.0)
                    rs = bulk_pred.sum(axis=1, keepdims=True)
                    bulk_pred = bulk_pred / np.maximum(rs, 1e-10)

                    # Save main proportions.csv
                    pd.DataFrame(bulk_pred, columns=cell_types).to_csv(
                        output_dir / "proportions.csv")
                    tee(f"  Saved H5 bulk predictions")

                    # Compute metrics against H5 GT
                    if gt_data is not None:
                        common_types = [c for c in cell_types if c in gt_data.columns]
                        if common_types:
                            h5_gt = gt_data[common_types].values.astype(np.float64)
                            type_idx = [cell_types.index(c) for c in common_types]
                            h5_metrics = evaluate_predictions(bulk_pred[:, type_idx], h5_gt, common_types)
                            tee(f"  H5 Bulk Pearson: {h5_metrics['pearson_mean']:.4f}")
                            with open(output_dir / "eval_results.json", "w") as f:
                                json.dump(h5_metrics, f, indent=2)
            except Exception as e:
                tee(f"  WARNING: H5 bulk eval failed: {e}")
                import traceback
                tee(traceback.format_exc())

        # -- 6. Save outputs ------------------------------------------------------
        ckpt_dir = output_dir / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), ckpt_dir / "deconv_head.pt")
        with open(ckpt_dir / "cell_types.json", "w") as f:
            json.dump(cell_types, f)

        meta = {
            "model": "transcriptformer",
            "test_pearson": eval_metrics["pearson_mean"] if eval_metrics else None,
            "n_types": n_ct,
            "embed_dim": embed_dim,
            "progressive": progressive,
            "n_progressive_layers": n_progressive_layers,
            "pseudo_bulk_space": pb_space,
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if predictions is not None:
            pd.DataFrame(predictions, columns=cell_types).to_csv(
                            output_dir / "val_proportions.csv")
        if ground_truth is not None:
            pd.DataFrame(ground_truth, columns=cell_types).to_csv(
                output_dir / "val_ground_truth.csv")

        with open(output_dir / "val_eval_results.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)

        tee(f"\nSaved checkpoint:    {ckpt_dir / 'deconv_head.pt'}")
        tee(f"Saved cell types:    {ckpt_dir / 'cell_types.json'}")
        tee(f"Saved val predictions: {output_dir / 'val_proportions.csv'}")
        tee("Training completed successfully.")
        tee("=" * 60)

    finally:
        log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TranscriptFormer embedding-based deconvolution: train (standalone)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--log_file", default=None, help="Log file path")
    args = parser.parse_args()
    main(args.config, args.log_file)
