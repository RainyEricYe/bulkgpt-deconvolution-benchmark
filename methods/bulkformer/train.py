#!/usr/bin/env python3
"""Standalone training script for BulkFormer embedding-based deconvolution.

Encodes scRNA-seq cells through frozen BulkFormer, generates pseudo-bulk
mixtures, trains a DeconvHead MLP, and saves the trained head.

Requires BulkFormer source at ``BULKFORMER_DIR`` and the pretrained
BulkFormer-147M checkpoint.

Usage:
    python methods/bulkformer/train.py --config methods/bulkformer/configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# -- Ensure core/ is importable -------------------------------------------------
_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.deconv.embedding import (
    EmbeddingDeconvHead,
    ExpressionMixGenerator,
)
from core.deconv.utils import set_seed, setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_sc_ref(path: str, celltype_col: str = "cell_type") -> tuple:
    """Load and validate scRNA reference AnnData."""
    import anndata as ad
    import h5py

    if path.endswith(".h5ad"):
        adata = ad.read_h5ad(path)
    elif path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            values = f["singleCellExpr/values"][:]
            colnames = [x.decode() if isinstance(x, bytes) else str(x)
                        for x in f["singleCellExpr/colnames"][:]]
            rownames = [x.decode() if isinstance(x, bytes) else str(x)
                        for x in f["singleCellExpr/rownames"][:]]
            labels_raw = f["singleCellLabels/values"][:]
            labels = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in labels_raw]
        if len(labels) == values.shape[1]:
            values = values.T
        adata = ad.AnnData(
            X=values,
            obs=pd.DataFrame({"cell_type": labels}, index=colnames),
            var=pd.DataFrame(index=rownames),
        )
    else:
        raise ValueError(f"Unknown reference format: {path}")

    if celltype_col not in adata.obs:
        candidates = [c for c in adata.obs.columns
                      if "cell" in c.lower() or "type" in c.lower()]
        if candidates:
            celltype_col = candidates[0]
        else:
            raise ValueError(f"No celltype column in {list(adata.obs.columns)}")

    return adata, celltype_col


def train(config_path: str, log_file: str | None = None) -> str:
    """Run BulkFormer pseudo-bulk training; return checkpoint path."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    _, tee, log_fh = setup_logging(log_file or cfg.get("checkpoint_dir", "checkpoints/bulkformer/default"))

    # ── Load scRNA reference ──
    tee("Loading scRNA reference...")
    sc_path = cfg["sc_ref"]
    celltype_col = cfg.get("celltype_col", "cell_type")
    adata, celltype_col = _load_sc_ref(sc_path, celltype_col)

    cell_types = sorted(set(adata.obs[celltype_col]))
    min_cells = cfg.get("min_cells_per_type", 20)
    ct_filtered = [ct for ct in cell_types
                   if (adata.obs[celltype_col] == ct).sum() >= min_cells]
    if len(ct_filtered) < 2:
        raise RuntimeError(
            f"Need ≥2 cell types with ≥{min_cells} cells; got {len(ct_filtered)}"
        )
    tee(f"  Cell types: {len(ct_filtered)}")

    # ── Encode cells through frozen BulkFormer ──
    tee("Encoding cells through BulkFormer...")
    t0 = time.monotonic()
    from methods.bulkformer.model import encode_bulkformer

    raw = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
    labels = adata.obs[celltype_col].values
    genes = list(adata.var_names)
    mask = np.isin(labels, ct_filtered)
    raw, labels = raw[mask], labels[mask]

    emb = encode_bulkformer(raw, genes, [f"cell_{i}" for i in range(raw.shape[0])])
    encode_time = time.monotonic() - t0
    tee(f"  Embeddings: {emb.shape} [{encode_time:.1f}s]")

    # ── Generate pseudo-bulk mixtures ──
    n_types = len(ct_filtered)
    ct2idx = {ct: i for i, ct in enumerate(ct_filtered)}
    label_idx = np.array([ct2idx[l] for l in labels])

    n_pb = cfg.get("n_pseudo_bulk", 5000)
    tee(f"Generating {n_pb} pseudo-bulk mixtures...")
    mix_gen = ExpressionMixGenerator(
        raw, label_idx, n_types,
        n_pb=n_pb, seed=cfg.get("seed", 42),
    )
    pb_expr, pb_props = mix_gen.generate()

    pb_emb = encode_bulkformer(pb_expr, genes, [f"pb_{i}" for i in range(len(pb_expr))])
    tee(f"  Pseudo-bulk embeddings: {pb_emb.shape}")

    # ── Train MLP head ──
    embed_dim = emb.shape[1]
    hidden_dims = cfg.get("hidden_dims", [256])
    model = EmbeddingDeconvHead(
        embed_dim, hidden_dims[0] if isinstance(hidden_dims, (list, tuple)) else 256, n_types,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.get("step_size", 10),
        gamma=cfg.get("step_gamma", 0.9),
    )
    criterion = torch.nn.MSELoss()

    n_train = int(len(pb_emb) * 0.8)
    idx = np.random.permutation(len(pb_emb))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_emb_t = torch.from_numpy(pb_emb[train_idx]).float().to(device)
    train_props_t = torch.from_numpy(pb_props[train_idx]).float().to(device)
    val_emb_t = torch.from_numpy(pb_emb[val_idx]).float().to(device)
    val_props_t = torch.from_numpy(pb_props[val_idx]).float().to(device)

    batch_size = cfg.get("batch_size", 64)
    best_val_loss = float("inf")
    best_ckpt_path = ""

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints/bulkformer/default"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tee(f"Training {cfg.get('epochs', 30)} epochs...")
    for epoch in range(cfg.get("epochs", 30)):
        model.train()
        perm = torch.randperm(len(train_emb_t))
        for i in range(0, len(train_emb_t), batch_size):
            bi = perm[i : i + batch_size]
            pred = model(train_emb_t[bi])
            loss = criterion(pred, train_props_t[bi])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_emb_t), val_props_t).item()
        if (epoch + 1) % cfg.get("log_interval", 10) == 0:
            tee(f"  epoch {epoch + 1:3d}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = str(checkpoint_dir / "best_model.pt")
            torch.save(model.state_dict(), best_ckpt_path)

    tee(f"  Best val_loss: {best_val_loss:.6f}")

    # ── Evaluate on validation holdout ──
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        test_pred = model(val_emb_t).cpu().numpy()
    test_pred = np.maximum(test_pred, 0)
    test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

    from core.metrics import evaluate_deconvolution

    metrics = evaluate_deconvolution(
        val_props_t.cpu().numpy(), test_pred, ct_filtered,
    )
    tee(
        f"  Val Pearson: {metrics['pearson_mean']:.4f}  "
        f"RMSE: {metrics['rmse_mean_per_type']:.4f}"
    )

    # ── Save metadata ──
    meta = {
        "backbone": "bulkformer", "embed_dim": embed_dim,
        "cell_types": ct_filtered, "best_val_loss": best_val_loss,
        "val_pearson": metrics["pearson_mean"],
        "encode_time_s": round(encode_time, 1),
    }
    with open(checkpoint_dir / "checkpoint_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if Path(config_path).resolve() != (checkpoint_dir / "config.yaml").resolve():
        shutil.copy2(config_path, checkpoint_dir / "config.yaml")
    tee(f"Checkpoint saved to {best_ckpt_path}")
    return best_ckpt_path


def main(config_path: str | None = None, log_file: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="BulkFormer deconvolution training")
    parser.add_argument("--config", required=config_path is None)
    parser.add_argument("--log_file", default=None)
    args, _ = parser.parse_known_args()
    train(args.config or config_path, args.log_file or log_file)


if __name__ == "__main__":
    main()
