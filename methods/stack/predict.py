#!/usr/bin/env python3
"""Standalone prediction script for STACK embedding-based deconvolution.

Loads a trained DeconvHead checkpoint, encodes bulk RNA-seq with STACK,
and predicts cell-type proportions.

Requires the STACK package (https://github.com/arcinstitute/stack) and a
pretrained STACK checkpoint.

Usage:
    python methods/stack/predict.py --config methods/stack/configs/default.yaml \\
        --checkpoint results/stack/checkpoint/deconv_head.pt
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

import numpy as np
import pandas as pd
import torch
import yaml

# -- Ensure core/ is importable -------------------------------------------------
_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.deconv.embedding import EmbeddingDeconvHead
from core.deconv.utils import external_dir, find_project_root, setup_logging

# -- External dependency: STACK --------------------------------------------------

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


def encode_stack(raw_counts, gene_symbols, barcodes, stack_checkpoint, genelist_path):
    """Compute STACK cell embeddings from raw counts."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    load_model_from_checkpoint = _import_stack()

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


def main(
    config_path: str,
    checkpoint_path: str,
    log_file: str | None = None,
    ground_truth: str | None = None,
) -> None:
    cfg = Path(config_path)
    ckpt = Path(checkpoint_path)
    if not cfg.exists():
        print(f"ERROR: Config not found: {cfg}", file=sys.stderr)
        sys.exit(1)
    if not ckpt.exists():
        print(f"ERROR: Checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    with open(cfg) as f:
        config = yaml.safe_load(f)

    project_root = find_project_root()
    os.chdir(str(project_root))

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    paths_cfg = config.get("paths", {})

    output_dir = project_root / paths_cfg.get("output_dir", "results/stack")
    output_dir.mkdir(parents=True, exist_ok=True)
    dst_cfg = output_dir / "config.yaml"
    if cfg.resolve() != dst_cfg.resolve():
        shutil.copy2(cfg, dst_cfg)

    if log_file is None:
        log_file = str(output_dir / "predict.log")

    _, tee, log_fh = setup_logging(log_file)
    warnings.filterwarnings("ignore")

    try:
        tee("=" * 60)
        tee("STACK -- predict (standalone)")
        tee(f"Config:      {cfg.resolve()}")
        tee(f"Checkpoint:  {ckpt.resolve()}")
        tee(f"Output:      {output_dir}")
        tee(f"Device:      {device}")

        def _resolve(p):
            p = Path(p)
            return p if p.is_absolute() else project_root / p

        data_path = data_cfg.get("data_path")
        bulk_path = None
        if not data_path:
            bulk_path = _resolve(data_cfg.get("bulk", ""))
            if not bulk_path or not bulk_path.exists():
                tee("ERROR: data.bulk or data.data_path not set in config")
                sys.exit(1)

        stack_ckpt = _resolve(model_cfg["stack_checkpoint"])
        genelist_path_val = _resolve(model_cfg["genelist_path"])

        # -- 1. Load checkpoint metadata ------------------------------------------
        ckpt_dir = ckpt.parent
        with open(ckpt_dir / "cell_types.json") as f:
            cell_types = json.load(f)
        with open(ckpt_dir / "metadata.json") as f:
            meta = json.load(f)

        embed_dim = meta.get("embed_dim", model_cfg.get("embed_dim", 1600))
        hidden_dim = model_cfg.get("hidden_dim", 512)
        progressive = meta.get("progressive", False)
        n_progressive_layers = meta.get("n_progressive_layers", 3)
        n_ct = len(cell_types)

        tee(f"  Cell types ({n_ct}): {cell_types}")
        tee(f"  Embed dim: {embed_dim}")
        if progressive:
            tee(f"  Progressive MLP: {n_progressive_layers} layers")

        # -- 2. Rebuild model and load weights ------------------------------------
        model = EmbeddingDeconvHead(
            embed_dim, hidden_dim, n_ct,
            progressive=progressive, n_progressive_layers=n_progressive_layers,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        tee("  Model loaded from checkpoint")

        # -- 3. Load and encode bulk RNA-seq --------------------------------------
        import anndata as ad
        import pandas as pd
        import numpy as np

        if data_path:
            from core.data_loader import load_data
            bundle = load_data(_resolve(data_path), ground_truth=ground_truth)
            bulk_df = bundle.bulk
            tee(f"\n[1] Loaded bulk from H5: {data_path}  ({bulk_df.shape[0]} samples, {bulk_df.shape[1]} genes)")
            ab = ad.AnnData(
                X=bulk_df.values,
                obs=pd.DataFrame(index=bulk_df.index),
                var=pd.DataFrame(index=bulk_df.columns),
            )
            bulk_path_str = str(data_path)
        else:
            bulk_path_str = str(bulk_path)
            tee(f"\n[1] Loading bulk RNA-seq: {bulk_path_str}")
            ab = ad.read_h5ad(bulk_path_str)
        bx = ab.X.toarray() if hasattr(ab.X, "toarray") else np.asarray(ab.X)
        bg = list(ab.var_names)
        bb = list(ab.obs_names)
        bg_symbols = list(ab.var.get("gene_symbol", ab.var_names)) if "gene_symbol" in ab.var else bg
        tee(f"  {bx.shape[0]} samples, {bx.shape[1]} genes")

        tee(f"\n[2] Encoding bulk with STACK...")
        t0 = time.monotonic()
        be = encode_stack(bx, bg_symbols, bb, str(stack_ckpt), str(genelist_path_val))
        tee(f"  {be.shape} [{time.monotonic() - t0:.1f}s]")

        # -- 4. Predict proportions -----------------------------------------------
        tee(f"\n[3] Predicting proportions...")
        with torch.no_grad():
            pp = model(torch.from_numpy(be).float().to(device)).cpu().numpy()
        pp = pp / pp.sum(axis=1, keepdims=True)

        # -- 5. Save CSV (samples x cell types) -----------------------------------
        pdf = pd.DataFrame(pp, index=bb, columns=cell_types)
        csv_path = output_dir / "proportions.csv"
        pdf.to_csv(str(csv_path))
        tee(f"  Saved -> {csv_path}")

        tee("\nMean proportions:")
        for i, ct in enumerate(cell_types):
            tee(f"  {ct:25s}: {pp[:, i].mean():.4f}")

        tee("\nPrediction completed successfully.")
        tee("=" * 60)

    finally:
        log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STACK embedding-based deconvolution: predict (standalone)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to deconv_head.pt checkpoint")
    parser.add_argument("--log_file", default=None, help="Log file path")
    parser.add_argument("--ground-truth", default=None,
                        help="Path to ground truth CSV (overrides H5 ground_truth group)")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.log_file, args.ground_truth)
