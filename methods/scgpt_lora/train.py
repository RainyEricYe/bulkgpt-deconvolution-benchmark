#!/usr/bin/env python3
"""LoRA fine-tuning for scGPT deconvolution.

Adapted from scPEFT ``deconv/lora_experiment.py:run_experiment()``.

Usage:
    python methods/scgpt_lora/train.py --config methods/scgpt_lora/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.deconv.utils import set_seed, setup_logging
from methods.scgpt_lora.data import (
    build_loaders,
    compute_metrics,
    get_loader,
    load_h5_bulk,
    normalize_proportions,
)
from methods.scgpt_lora.model import build_model, count_trainable


def train_epoch(model, head, loader, optimizer, device):
    """Single training epoch. Returns mean loss."""
    model.train()
    head.train()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    n_batches = 0
    for batch in loader:
        genes = batch["gene"].to(device)
        values = batch["expr"].to(device)
        mask = batch["padding_mask"].to(device) if "padding_mask" in batch else None
        target = batch["proportion"].to(device)

        optimizer.zero_grad()
        output = model._encode(genes, values, mask)
        cls_emb = output[:, 0, :]  # CLS token
        pred = head(cls_emb)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, head, loader, device):
    """Evaluate on a loader. Returns mean validation loss."""
    model.eval()
    head.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    n_batches = 0
    for batch in loader:
        genes = batch["gene"].to(device)
        values = batch["expr"].to(device)
        mask = batch["padding_mask"].to(device) if "padding_mask" in batch else None
        target = batch["proportion"].to(device)
        output = model._encode(genes, values, mask)
        cls_emb = output[:, 0, :]
        pred = head(cls_emb)
        loss = criterion(pred, target)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict(model, head, loader, device):
    """Run inference. Returns (n_samples, n_types) numpy array."""
    model.eval()
    head.eval()
    all_preds = []
    for batch in loader:
        genes = batch["gene"].to(device)
        values = batch["expr"].to(device)
        mask = batch["padding_mask"].to(device) if "padding_mask" in batch else None
        output = model._encode(genes, values, mask)
        cls_emb = output[:, 0, :]
        pred = head(cls_emb)
        all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def main(
    config_path: str,
    log_file: str | None = None,
    seed: int | None = None,
) -> None:
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = seed or config.get("seed", 42)
    set_seed(seed)
    log_path = log_file or config.get("log_file")
    if log_path:
        setup_logging(log_path)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.environ.get("SCGPT_MODEL_DIR", config["model_dir"])
    h5_path = config["h5_path"]
    gt_path = config.get("gt_path", h5_path.replace(".h5", "_gt.csv"))
    output_dir = Path(config.get("output_dir", "checkpoints/scgpt_lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    peft_config = config["peft_config"]
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    n_epochs = config.get("n_epochs", 100)
    batch_size = config.get("batch_size", 32)
    max_length = config.get("max_length", 1200)
    patience = config.get("patience", 15)

    # 1. Load vocab & build model
    from scgpt.tokenizer import GeneVocab

    vocab = GeneVocab.from_file(str(Path(model_dir) / "vocab.json"))
    model = build_model(peft_config, model_dir, device)
    n_lora = count_trainable(model)
    print(f"  Trainable (LoRA): {n_lora:,} params")

    # 2. Build head
    from scgpt.tasks.deconv import LinearDeconvHead

    # Get cell_type_names from GT CSV (avoid redundant build_loaders)
    import pandas as _pd
    _gt_df = _pd.read_csv(gt_path, index_col=0)
    cell_type_names = list(_gt_df.columns)
    head = LinearDeconvHead(d_model=512, n_cell_types=len(cell_type_names)).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"  Trainable (head):  {n_head:,} params")

    # 3. Optimizer
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": lr, "weight_decay": weight_decay},
        {"params": head.parameters(),
         "lr": lr, "weight_decay": weight_decay},
    ])

    # 4. Build loaders (second time, with actual splits)
    train_loader, val_loader, test_loader, _ = build_loaders(
        h5_path, gt_path, vocab, batch_size=batch_size, max_length=max_length, split_seed=seed
    )

    # 5. Training loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    since_best = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, head, train_loader, optimizer, device)
        val_loss = evaluate(model, head, val_loader, device)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            since_best = 0
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
            model.to(device)
        else:
            since_best += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if since_best >= patience:
            print(f"  Early stopping at epoch {epoch} (best {best_val_loss:.4f})")
            break

    elapsed = time.time() - t0

    # 6. Restore best and evaluate
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        head.load_state_dict(best_state["head"])
    model.to(device)

    # Get GT arrays for metric computation
    import pandas as pd
    gt_df = pd.read_csv(gt_path, index_col=0)
    cell_type_names = list(gt_df.columns)
    gt_array = gt_df.values.astype(np.float32)

    # Load counts to re-build per-split loaders for prediction
    bulk_expr, bulk_genes, ref_expr, ref_genes, ref_labels = load_h5_bulk(h5_path)
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
    bulk_ref_mask = [g in ref_gene_set for g in adata_bulk_v.var_names]
    adata_bulk_r = adata_bulk_v[:, bulk_ref_mask].copy()
    adata_bulk_pp = _solver._preprocess_for_scgpt(adata_bulk_r)
    gene_ids = _solver._gene_ids
    count_matrix = adata_bulk_pp.X
    if hasattr(count_matrix, "A"):
        count_matrix = count_matrix.A

    from methods.scgpt_lora.data import get_sdy67_split
    n = count_matrix.shape[0]
    if n == 250:
        train_idx, val_idx, test_idx = get_sdy67_split(n)
    else:
        from sklearn.model_selection import train_test_split
        indices = np.arange(n)
        train_val, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_val, test_size=0.25, random_state=seed)

    results = {}
    for split_name, si in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        loader = get_loader(si, count_matrix, gene_ids, gt_array, vocab, batch_size, max_length, shuffle=False)
        pred = predict(model, head, loader, device)
        pred_norm = normalize_proportions(pred)
        results[split_name] = compute_metrics(pred_norm, gt_array[si], cell_type_names)

    test_r = results["test"]["macro_avg"]["pearson_r"]
    print(f"  >>> Test macro_avg r = {test_r:.4f}")

    # 7. Save checkpoint (only LoRA + head params to keep it small)
    checkpoint = {
        "model_lora": {k: v for k, v in best_state["model"].items() if "lora" in k.lower()},
        "head": best_state["head"],
        "config": {
            "peft_config": peft_config,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_epochs": epoch,
            "best_val_loss": best_val_loss,
            "test_macro_r": test_r,
        },
    }
    ckpt_path = output_dir / "best_model.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # 8. Save results
    result = {
        "experiment": config.get("name", "scgpt_lora"),
        "peft_config": {k: str(v) if isinstance(v, list) else v for k, v in peft_config.items()},
        "training": {
            "lr": lr,
            "weight_decay": weight_decay,
            "n_epochs": epoch,
            "best_val_loss": best_val_loss,
            "time_seconds": round(elapsed, 1),
        },
        "splits": results,
        "n_trainable_lora": n_lora,
        "n_trainable_head": n_head,
    }
    result_path = output_dir / "train_results.json"
    with open(result_path, "w") as f:
        class _NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        json.dump(result, f, indent=2, cls=_NpEncoder)
    print(f"  Results saved: {result_path}")

    # Save metadata
    metadata = {
        "backbone": "scgpt_lora",
        "model_dir": model_dir,
        "dataset": config.get("dataset", "sdy67"),
        "device": device,
        "wall_time_s": round(elapsed, 1),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_version": torch.version.cuda or "none",
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "seed": seed,
    }
    meta_path = output_dir / "train_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scGPT-LoRA fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(config_path=args.config, log_file=args.log_file, seed=args.seed)
