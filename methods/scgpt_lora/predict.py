#!/usr/bin/env python3
"""Inference with trained scGPT-LoRA adapters.

Produces standard to_publish output (proportions.csv + metrics.json + metadata.json).

Usage:
    python methods/scgpt_lora/predict.py         --config methods/scgpt_lora/configs/default.yaml         --checkpoint checkpoints/scgpt_lora/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.metrics import evaluate_deconvolution
from core.deconv.utils import set_seed, setup_logging
from methods.scgpt_lora.data import (
    get_loader,
    load_h5_bulk,
    normalize_proportions,
)
from methods.scgpt_lora.model import build_model


def main(
    config_path: str,
    checkpoint: str,
    log_file: str | None = None,
    seed: int | None = None,
) -> None:
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
    gt_path = config.get("gt_path", str(Path(h5_path).with_suffix("")) + "_gt.csv")
    results_dir = Path(config.get("results_dir", f"results/2_realbulk/{Path(h5_path).stem}/scgpt_lora"))
    results_dir.mkdir(parents=True, exist_ok=True)

    peft_config = config["peft_config"]
    batch_size = config.get("batch_size", 32)
    max_length = config.get("max_length", 1200)

    # 1. Load vocab and build model
    from scgpt.tokenizer import GeneVocab

    vocab = GeneVocab.from_file(str(Path(model_dir) / "vocab.json"))

    t0 = time.time()
    model = build_model(peft_config, model_dir, device)
    model.eval()

    # 2. Load checkpoint (LoRA adapters + head)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    lora_state = ckpt.get("model_lora", {})
    head_state = ckpt.get("head", {})

    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    if unexpected:
        print(f"  Warning: unexpected keys in checkpoint: {unexpected[:5]}")

    # 3. Load bulk data and GT
    bulk_expr, bulk_genes, ref_expr, ref_genes, ref_labels = load_h5_bulk(h5_path)

    gt_df = pd.read_csv(gt_path, index_col=0)
    cell_type_names = list(gt_df.columns)
    gt_array = gt_df.values.astype(np.float32)

    # 4. Build head (need n_types from gt_df)
    from scgpt.tasks.deconv import LinearDeconvHead

    head = LinearDeconvHead(d_model=512, n_cell_types=len(cell_type_names)).to(device)
    head.load_state_dict(head_state)
    head.eval()

    load_time = time.time() - t0

    # 5. Preprocess for scGPT
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

    # 6. Build full-data loader
    all_loader = get_loader(
        np.arange(len(gt_array)), count_matrix, gene_ids, gt_array, vocab,
        batch_size=batch_size, max_length=max_length, shuffle=False,
    )

    # 7. Encode and predict
    encode_t0 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        all_preds = []
        for batch in all_loader:
            genes_b = batch["gene"].to(device)
            values_b = batch["expr"].to(device)
            mask_b = batch["padding_mask"].to(device) if "padding_mask" in batch else None
            output = model._encode(genes_b, values_b, mask_b)
            cls_emb = output[:, 0, :]
            pred_b = head(cls_emb)
            all_preds.append(pred_b.cpu().numpy())
    pred_array = np.concatenate(all_preds, axis=0)
    encode_time = time.time() - encode_t0

    # 8. Post-process
    pred_norm = normalize_proportions(pred_array)

    # 9. Save proportions.csv
    prop_df = pd.DataFrame(pred_norm, columns=cell_type_names)
    prop_path = results_dir / "proportions.csv"
    prop_df.to_csv(prop_path, index=False)
    print(f"  Saved: {prop_path}")

    # 10. Compute metrics via core.metrics
    metrics = evaluate_deconvolution(gt_array, pred_norm, cell_type_names)
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # 11. Save lora_config
    lora_config = {
        "peft_config": {k: str(v) if isinstance(v, list) else v for k, v in peft_config.items()},
        "checkpoint": checkpoint,
        "n_trainable_lora": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with open(results_dir / "lora_config.json", "w") as f:
        json.dump(lora_config, f, indent=2)

    # 12. Save metadata
    wall_time = time.time() - t0
    metadata = {
        "backbone": "scgpt_lora",
        "dataset": Path(h5_path).stem,
        "device": device,
        "load_time_s": round(load_time, 1),
        "encode_time_s": round(encode_time, 1),
        "wall_time_s": round(wall_time, 1),
        "gpu_memory_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_version": torch.version.cuda or "none",
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "seed": seed,
    }
    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {results_dir / 'metadata.json'}")

    # 13. Print summary
    print(f"\n  scGPT-LoRA predictions: {results_dir}")
    print(f"  Pearson r: {metrics.get('pearson_mean', 'N/A')}")
    print(f"  MAE:        {metrics.get('mae_overall', 'N/A')}")
    print(f"  Wall time:  {wall_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scGPT-LoRA inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ground-truth", default=None,
                        help="Path to ground truth CSV (ignored, read from config)")
    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint=args.checkpoint,
        log_file=args.log_file,
        seed=args.seed,
    )
