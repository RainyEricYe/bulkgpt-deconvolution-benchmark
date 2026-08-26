#!/usr/bin/env python3
"""Standalone training script for scFoundation deconvolution.

Encodes scRNA cells through frozen scFoundation backbone, generates
pseudo-bulk mixtures in expression space, trains a DeconvHead MLP,
and evaluates on H5 bulk data.

Usage:
    python methods/scfoundation/train.py \
        --config methods/scfoundation/configs/default.yaml
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

from core.deconv.embedding import (
    EmbeddingDeconvHead,
    ExpressionMixGenerator,
    PseudoBulkConfig,
    evaluate_predictions,
    split_indices,
)
from core.deconv.utils import set_seed, setup_logging
from core.data_loader import load_data
from methods.scfoundation.model import ScFoundationBackbone


def encode_scf(
    raw_counts: np.ndarray,
    gene_symbols: list[str],
    barcodes: list[str],
    model_dir: str,
    device: str = "cuda",
) -> np.ndarray:
    """Encode cells through frozen scFoundation backbone."""
    model = ScFoundationBackbone(model_dir).to(device)
    model.eval()

    import json as _json
    _map_path = Path(model_dir).parent.parent.parent / "scfoundation_ensembl_to_scfpos.json"
    with open(_map_path) as f:
        ensg_to_pos = _json.load(f)

    emb_list = []
    with torch.no_grad():
        for i in range(0, len(raw_counts), 256):
            batch_counts = raw_counts[i:i + 256]
            n = batch_counts.shape[0]

            gene_ids = []
            for g in gene_symbols:
                pos = ensg_to_pos.get(g, 0)
                gene_ids.append(pos)
            gene_ids_t = torch.tensor(gene_ids, device=device).unsqueeze(0).expand(n, -1)
            values = torch.from_numpy(batch_counts).to(device)
            mask = torch.zeros(n, len(gene_symbols), device=device, dtype=torch.bool)

            emb = model(gene_ids_t, values, mask)
            # Mean pool over sequence dimension: (B, S, D) → (B, D)
            emb = emb.mean(dim=1)
            emb_list.append(emb.cpu().numpy())

    return np.concatenate(emb_list)


def main(
    config_path: str,
    log_file: str | None = None,
    seed: int | None = None,
) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = seed or config.get("seed", 42)
    set_seed(seed)
    if log_file:
        setup_logging(log_file)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config.get("data", config.get("dataset", {}))
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})
    output_dir = Path(paths_cfg.get("output_dir", "checkpoints/scfoundation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = os.environ.get(
        "SCFOUNDATION_MODEL_DIR",
        paths_cfg.get("pretrained_model", "weights/scfoundation/default"),
    )
    embed_dim = model_cfg.get("embed_dim", 19264)
    hidden_dim = model_cfg.get("deconv_hidden_dim", 128)
    n_epochs = train_cfg.get("epochs", 30)
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("lr", 1e-3)
    n_pseudo_bulk = train_cfg.get("n_pseudo_bulk", 5000)

    import warnings
    warnings.filterwarnings("ignore", message="flash_attn is not installed")

    # 1. Load scRNA reference
    h5_path = data_cfg.get("data_path") or data_cfg.get("h5_path")
    if not h5_path:
        raise ValueError("data_path or h5_path required")
    print(f"\n[1] Loading scRNA from H5: {h5_path}")
    bundle = load_data(str(h5_path))
    ref = bundle.sc_ref
    if ref is None:
        raise ValueError("No single-cell reference in H5")
    sc_expr = ref.X
    if hasattr(sc_expr, "toarray"):
        sc_expr = sc_expr.toarray()
    sc_expr = np.asarray(sc_expr, dtype=np.float32)
    sc_genes = list(ref.var_names)

    celltype_col = data_cfg.get("celltype_col", "cell_type")
    sc_labels = ref.obs[celltype_col].values if celltype_col in ref.obs else None
    if sc_labels is None:
        raise ValueError(f"cell_type column '{celltype_col}' not found")
    type_set = sorted(set(sc_labels))
    t2i = {t: i for i, t in enumerate(type_set)}
    n_ct = len(type_set)
    sc_labels_num = np.array([t2i[t] for t in sc_labels])
    print(f"  Cells: {sc_expr.shape[0]}, Genes: {sc_expr.shape[1]}, Types: {n_ct}")

    # 2. Load H5 bulk for evaluation
    bulk_h5_matrix = None
    gt_data = None
    if bundle.bulk is not None:
        bulk_h5_matrix = bundle.bulk.values.astype(np.float64)
        bulk_h5_genes = list(bundle.bulk.columns)
        gt_data = bundle.gt
        print(f"  Bulk samples: {bulk_h5_matrix.shape[0]}, Genes: {bulk_h5_matrix.shape[1]}")

    # 3. Generate pseudo-bulk
    print(f"\n[3] Generating {n_pseudo_bulk} pseudo-bulk samples...")
    gen = ExpressionMixGenerator(sc_expr, sc_labels_num, n_ct, n_pseudo_bulk, seed)
    pb_expr, pb_props = gen.generate()
    print(f"  Pseudo-bulk shape: {pb_expr.shape}")

    # 4. Encode
    print(f"\n[4] Encoding with scFoundation...")
    t0 = time.time()
    pb_emb = encode_scf(pb_expr, sc_genes, [f"pb_{i}" for i in range(len(pb_expr))], model_dir, device)
    print(f"  Embeddings: {pb_emb.shape} in {time.time() - t0:.1f}s")

    # 5. Train DeconvHead
    print(f"\n[5] Training DeconvHead ({n_epochs} epochs)...")
    pb_cfg = PseudoBulkConfig(
        n_pseudo_bulk=len(pb_emb),
        train_ratio=train_cfg.get("train_ratio", 0.8),
        val_ratio=train_cfg.get("val_ratio", 0.2),
        seed=train_cfg.get("seed", 42),
    )
    train_idx, val_idx, _ = split_indices(
        len(pb_emb),
        train_ratio=pb_cfg.train_ratio,
        val_ratio=pb_cfg.val_ratio,
        test_ratio=pb_cfg.test_ratio,
        seed=pb_cfg.seed,
    )
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

    head = EmbeddingDeconvHead(embed_dim, hidden_dim, n_ct).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9)
    best_val = float("inf")
    for ep in range(1, n_epochs + 1):
        head.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = torch.nn.functional.mse_loss(head(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()
        sched.step()
        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += torch.nn.functional.mse_loss(head(x.to(device)), y.to(device)).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(head.state_dict(), output_dir / "best_model.pt")
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep}: train={train_loss:.4f}, val={val_loss:.4f}")

    # 6. Evaluate on H5 bulk
    if bulk_h5_matrix is not None:
        print(f"\n[6] Evaluating on H5 bulk...")
        head.eval()
        bulk_emb = encode_scf(
            bulk_h5_matrix.astype(np.float32), bulk_h5_genes,
            [f"b{i}" for i in range(len(bulk_h5_matrix))], model_dir, device,
        )
        with torch.no_grad():
            bulk_pred = head(torch.from_numpy(bulk_emb).float().to(device)).cpu().numpy()

        pred_df = __import__("pandas").DataFrame(
            bulk_pred, index=bundle.bulk.index if bundle.bulk is not None else None, columns=type_set)
        pred_df.to_csv(output_dir / "proportions.csv")

        if gt_data is not None:
            gt_cols = list(gt_data.columns)
            common = [c for c in type_set if c in gt_cols]
            if common:
                gi = [gt_cols.index(c) for c in common]
                pi = [type_set.index(c) for c in common]
                from core.metrics import evaluate_deconvolution
                deconbench = evaluate_deconvolution(
                    gt_data.values[:, gi].astype(np.float64), bulk_pred[:, pi], common)
                with open(output_dir / "metrics.json", "w") as f:
                    json.dump(deconbench, f, indent=2)
                print(f"  H5 bulk Pearson: {deconbench['pearson_mean']:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="scFoundation deconvolution training")
    p.add_argument("--config", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    main(args.config, args.log_file, args.seed)
