#!/usr/bin/env python3
"""Standalone training script for STACK embedding-based deconvolution.

Computes STACK cell embeddings from scRNA-seq, generates pseudo-bulk mixtures,
trains a DeconvHead MLP, and saves the trained head.

Requires the STACK package (https://github.com/arcinstitute/stack) and a
pretrained STACK checkpoint.

Usage:
    python methods/stack/train.py --config methods/stack/configs/default.yaml
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

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix

# -- Ensure core/ is importable -------------------------------------------------
_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.deconv.domain_adaptation import mmd_loss
from core.deconv.embedding import EmbeddingDeconvHead, ExpressionMixGenerator, MixGenerator, evaluate_predictions
from core.deconv.utils import external_dir, find_project_root, set_seed, setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _import_stack():
    """Resolve STACK dependency at runtime (pip -> data/external/ -> STACK_SRC)."""
    try:
        from stack.model_loading import load_model_from_checkpoint
        return load_model_from_checkpoint
    except ImportError:
        pass
    _stack_local = external_dir() / "stack" / "stack"
    if _stack_local.is_dir():
        sys.path.insert(0, str(_stack_local))
        from stack.model_loading import load_model_from_checkpoint
        return load_model_from_checkpoint
    stack_src = os.environ.get("STACK_SRC")
    if stack_src:
        sys.path.insert(0, stack_src)
        from stack.model_loading import load_model_from_checkpoint
        return load_model_from_checkpoint
    print(
        "ERROR: STACK package not found.\n"
        "  Install via: bash data/prepare/download_external.sh --stack\n"
        "  Or set STACK_SRC env var to the stack/src directory.\n"
        "  Or install the stack package from https://github.com/arcinstitute/stack",
        file=sys.stderr,
    )
    sys.exit(1)


def _ensembl_to_symbols(gene_names: list[str]) -> list[str]:
    """Convert Ensembl IDs to HGNC symbols using Geneformer's dict."""
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
            ensg2sym = {v: k for k, v in raw.items()}
            converted = [ensg2sym.get(str(g), str(g)) for g in gene_names]
            n = sum(1 for c, o in zip(converted, gene_names) if c != o)
            print(f"  Ensembl->HGNC: {n}/{len(gene_names)} converted")
            return converted
    return gene_names


def encode_stack(raw_counts, gene_symbols, barcodes, stack_checkpoint, genelist_path):
    """Compute STACK cell embeddings from raw counts."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    load_model_from_checkpoint = _import_stack()

    if len(gene_symbols) > 0 and str(gene_symbols[0]).startswith("ENSG"):
        gene_symbols = _ensembl_to_symbols(gene_symbols)

    adata = ad.AnnData(
        X=csr_matrix(raw_counts),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_symbols),
    )
    model = load_model_from_checkpoint(stack_checkpoint, strict=False)
    model.eval()
    model.to(device)

    with torch.no_grad():
        emb, _ = model.get_latent_representation(
            adata_path=adata,
            genelist_path=genelist_path,
            gene_name_col=None,
            batch_size=8,
            num_workers=1,
            show_progress=False,
        )
    return emb


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

    output_dir = project_root / paths_cfg.get("output_dir", "results/stack")
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
        tee("STACK -- train (standalone)")
        tee(f"Config: {cfg_path.resolve()}")
        tee(f"Output: {output_dir}")
        tee(f"Device: {device}")

        # -- Resolve paths -------------------------------------------------------
        def _resolve(p):
            p = Path(p)
            return p if p.is_absolute() else project_root / p

        stack_ckpt = _resolve(model_cfg["stack_checkpoint"])
        genelist_path_val = _resolve(model_cfg["genelist_path"])

        embed_dim = model_cfg.get("embed_dim", 1600)
        hidden_dim = model_cfg.get("hidden_dim", 512)
        progressive = model_cfg.get("progressive", False)
        n_progressive_layers = model_cfg.get("n_progressive_layers", 3)
        n_pseudo_bulk = train_cfg.get("n_pseudo_bulk", 5000)
        n_epochs = train_cfg.get("epochs", 30)
        batch_size = train_cfg.get("batch_size", 64)
        lr = train_cfg.get("lr", 7e-5)
        weight_decay = train_cfg.get("weight_decay", 1e-5)
        seed = train_cfg.get("seed", 42)
        # "embedding" (default) or "expression"
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
        sc_gene_symbols = list(ref.var.get("gene_symbol", ref.var_names)) if "gene_symbol" in ref.var else sc_genes
        sc_barcodes = list(ref.obs_names)

        tee(f"  Cells: {sc_counts.shape[0]}, Genes: {sc_counts.shape[1]}, Types: {n_ct}")
        tee(f"  Cell types: {cell_types}")

        # -- 1c. Load H5 bulk data for test evaluation --------------------------
        bulk_h5_path = data_cfg.get("data_path")
        bulk_h5_matrix = None
        bulk_h5_genes = None
        bulk_h5_samples = None
        gt_data = None

        # Fix cell_type column if dropped by H5 orientation detection
        if celltype_col not in ref.obs.columns:
            from methods._shared.h5_sc_helper import ensure_sc_celltypes
            ref = ensure_sc_celltypes(ref, _resolve(bulk_h5_path) if bulk_h5_path else None, celltype_col)
            if celltype_col in ref.obs.columns:
                tee(f"  Restored {celltype_col} from H5 ({ref.obs[celltype_col].nunique()} types)")

        # -- 1c. (continued) Load H5 bulk data ------------------------------------
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

            # Align genes with scRNA reference
            bulk_genes_upper = [g.upper() for g in bulk_rownames]
            sc_genes_upper = [g.upper() for g in sc_gene_symbols]
            common = set(bulk_genes_upper) & set(sc_genes_upper)
            if len(common) < 100:
                tee(f"  WARNING: only {len(common)} common genes between bulk and scRNA")
            # Filter bulk to common genes, ordered by scRNA gene order
            bulk_gene_mask = [g in common for g in bulk_genes_upper]
            bulk_matrix = bulk_matrix[:, bulk_gene_mask]
            bulk_genes_filtered = [bulk_rownames[i] for i, m in enumerate(bulk_gene_mask) if m]
            tee(f"  After alignment: {len(bulk_genes_filtered)} genes overlap")

            # CPM+log1p normalize if bulk values look like raw counts (>100 per sample typical)
            row_sums = bulk_matrix.sum(axis=1, keepdims=True)
            if row_sums.mean() > 50:
                bulk_matrix = (bulk_matrix / np.where(row_sums == 0, 1, row_sums)) * 1e4
                bulk_matrix = np.log1p(bulk_matrix)
                tee(f"  Applied CPM+log1p normalization")

            # Encode through STACK
            t0 = time.monotonic()
            real_emb = encode_stack(bulk_matrix, bulk_genes_filtered, bulk_colnames,
                                    str(stack_ckpt), str(genelist_path_val))
            tee(f"  Real bulk embeddings: {real_emb.shape} [{time.monotonic() - t0:.1f}s]")
            tee(f"  DA method: {da_method}, lambda={da_lambda}")

        # -- 2b. Generate pseudo-bulk (expression or embedding space) -------------
        if pb_space == "expression":
            tee(f"\n[2] Generating expression-space pseudo-bulk ({n_pseudo_bulk} samples)...")
            gen = ExpressionMixGenerator(sc_counts, sc_labels, n_ct, n_pseudo_bulk, seed)
            pb_expr, pb_props = gen.generate()
            tee(f"  Pseudo-bulk expression: {pb_expr.shape}")

            # Encode each pseudo-bulk sample through STACK (frozen)
            tee(f"\n[3] Encoding pseudo-bulk through STACK...")
            t0 = time.monotonic()
            # Build an AnnData where each row is a pseudo-bulk sample
            pb_adata = ad.AnnData(
                X=csr_matrix(pb_expr),
                obs=pd.DataFrame(index=[f"pb_{i}" for i in range(len(pb_expr))]),
                var=pd.DataFrame(index=sc_gene_symbols),
            )
            load_model_from_checkpoint = _import_stack()
            stack_model = load_model_from_checkpoint(str(stack_ckpt), strict=False)
            stack_model.eval()
            stack_model.to(device)
            with torch.no_grad():
                pb_emb, _ = stack_model.get_latent_representation(
                    adata_path=pb_adata,
                    genelist_path=str(genelist_path_val),
                    gene_name_col=None,
                    batch_size=8,
                    num_workers=1,
                    show_progress=False,
                )
            tee(f"  Pseudo-bulk embeddings: {pb_emb.shape} [{time.monotonic() - t0:.1f}s]")
        else:
            # Embedding-space: pre-compute cell embeddings, then mix (original behaviour)
            tee(f"\n[2] Computing STACK cell embeddings...")
            t0 = time.monotonic()
            cell_emb = encode_stack(sc_counts, sc_gene_symbols, sc_barcodes,
                                    str(stack_ckpt), str(genelist_path_val))
            tee(f"  {cell_emb.shape} [{time.monotonic() - t0:.1f}s]")

            tee(f"\n[3] Generating pseudo-bulk ({n_pseudo_bulk} samples)...")
            gen = MixGenerator(cell_emb, sc_labels, n_ct, n_pseudo_bulk, seed)
            pb_emb, pb_props = gen.generate()
            tee(f"  {pb_emb.shape}")

        # -- 4. Train DeconvHead --------------------------------------------------
        tee(f"\n[4] Training DeconvHead ({n_epochs} epochs)...")
        from core.deconv.embedding import PseudoBulkConfig, split_indices
        _pb_cfg = PseudoBulkConfig(
            n_pseudo_bulk=len(pb_emb),
            train_ratio=train_cfg.get("train_ratio", 0.8),
            val_ratio=train_cfg.get("val_ratio", 0.2),
            test_ratio=train_cfg.get("test_ratio", 0.0),
            seed=train_cfg.get("seed", 42),
        )
        train_idx, val_idx, _ = split_indices(
            len(pb_emb),
            train_ratio=_pb_cfg.train_ratio,
            val_ratio=_pb_cfg.val_ratio,
            test_ratio=_pb_cfg.test_ratio,
            seed=_pb_cfg.seed,
        )
        n_tr, n_va = _pb_cfg.n_train, _pb_cfg.n_val

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(pb_emb[train_idx]).float(),
                torch.from_numpy(pb_props[train_idx]).float()),
            batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(pb_emb[val_idx]).float(),
                torch.from_numpy(pb_props[val_idx]).float()),
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
        predictions, ground_truth, eval_metrics = None, None, None
        # -- 5. Evaluate on test set (only if test_ratio > 0) ------------------
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
                # Align genes with scRNA training gene set
                sc_genes_upper = [g.upper() for g in sc_gene_symbols]
                bulk_upper = [g.upper() for g in bulk_h5_genes]
                common_upper = set(bulk_upper) & set(sc_genes_upper)
                if len(common_upper) < 10:
                    tee(f"  WARNING: only {len(common_upper)} common genes, skipping")
                else:
                    mask = [g in common_upper for g in bulk_upper]
                    bulk_expr = bulk_h5_matrix[:, mask].copy()
                    bulk_genes_filtered = [bulk_h5_genes[i] for i, m in enumerate(mask) if m]

                    # CPM + log1p
                    rsum = bulk_expr.sum(axis=1, keepdims=True)
                    bulk_expr = np.log1p((bulk_expr / np.where(rsum == 0, 1, rsum)) * 1e4)

                    t0 = time.monotonic()
                    bulk_emb = encode_stack(bulk_expr, bulk_genes_filtered,
                                            [f"bulk_{i}" for i in range(len(bulk_expr))],
                                            str(stack_ckpt), str(genelist_path_val))
                    tee(f"  H5 bulk embeddings: {bulk_emb.shape} [{time.monotonic() - t0:.1f}s]")

                    model.eval()
                    with torch.no_grad():
                        bulk_pred = model(torch.from_numpy(bulk_emb).float().to(device)).cpu().numpy()
                    bulk_pred = np.maximum(bulk_pred, 0.0)
                    rs = bulk_pred.sum(axis=1, keepdims=True)
                    bulk_pred = bulk_pred / np.maximum(rs, 1e-10)

                    # Save main proportions.csv (H5 bulk predictions)
                    h5_pred_path = output_dir / "proportions.csv"
                    pd.DataFrame(bulk_pred, columns=cell_types).to_csv(h5_pred_path)
                    tee(f"  Saved H5 bulk predictions: {h5_pred_path}")

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
            "model": "stack",
            "test_pearson": eval_metrics["pearson_mean"] if eval_metrics else None,
            "n_types": n_ct,
            "embed_dim": embed_dim,
            "progressive": progressive,
            "n_progressive_layers": n_progressive_layers,
            "pseudo_bulk_space": pb_space,
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save val set predictions (reference, not the main output)
        if predictions is not None:
            pd.DataFrame(predictions, columns=cell_types).to_csv(
                output_dir / "val_proportions.csv")
            pd.DataFrame(ground_truth, columns=cell_types).to_csv(
                output_dir / "val_ground_truth.csv")

        with open(output_dir / "val_eval_results.json", "w") as f:
            if eval_metrics:
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
        description="STACK embedding-based deconvolution: train (standalone)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--log_file", default=None, help="Log file path")
    args = parser.parse_args()
    main(args.config, args.log_file)
