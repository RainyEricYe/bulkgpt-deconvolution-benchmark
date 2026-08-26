#!/usr/bin/env python3
"""Standalone prediction script for TranscriptFormer embedding-based deconvolution.

Loads a trained DeconvHead checkpoint, encodes bulk RNA-seq with TranscriptFormer,
and predicts cell-type proportions.

Requires the TranscriptFormer package (https://github.com/suinleelab/TranscriptFormer)
and a pretrained TF checkpoint.

Usage:
    python methods/transcriptformer/predict.py --config methods/transcriptformer/configs/default.yaml \\
        --checkpoint results/transcriptformer/checkpoint/deconv_head.pt
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

    output_dir = project_root / paths_cfg.get("output_dir", "results/transcriptformer")
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
        tee("TranscriptFormer -- predict (standalone)")
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

        tf_ckpt_dir = _resolve(model_cfg["tf_checkpoint_dir"])

        # -- 1. Load checkpoint metadata ------------------------------------------
        ckpt_dir = ckpt.parent
        with open(ckpt_dir / "cell_types.json") as f:
            cell_types = json.load(f)
        with open(ckpt_dir / "metadata.json") as f:
            meta = json.load(f)

        embed_dim = meta.get("embed_dim", model_cfg.get("embed_dim", 2048))
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
        else:
            bulk_path_str = str(bulk_path)
            tee(f"\n[1] Loading bulk RNA-seq: {bulk_path_str}")
            ab = ad.read_h5ad(bulk_path_str)
        bx = ab.X.toarray() if hasattr(ab.X, "toarray") else np.asarray(ab.X)
        bg = list(ab.var_names)
        bb = list(ab.obs_names)
        # Map gene symbols to Ensembl IDs if needed
        n_ensg = sum(1 for g in bg if str(g).startswith("ENSG"))
        if n_ensg < len(bg) * 0.5:
            tee("  Mapping bulk gene symbols to Ensembl IDs...")
            sym_to_ensg = map_symbol_to_ensembl(bg)
            bg = [sym_to_ensg.get(g, g) for g in bg]
            valid = [i for i, g in enumerate(bg) if g.startswith("ENSG")]
            if len(valid) < 10:
                raise ValueError(f"Too few bulk genes mapped to Ensembl IDs ({len(valid)}/{len(bg)})")
            bx = bx[:, valid]
            bg = [bg[i] for i in valid]
            tee(f"  Retained {len(bg)} bulk genes with Ensembl IDs")
        else:
            tee(f"  Bulk genes already Ensembl IDs ({n_ensg}/{len(bg)}), skipping mapping")
        tee(f"  {bx.shape[0]} samples, {bx.shape[1]} genes")

        tee(f"\n[2] Encoding bulk with TranscriptFormer...")
        t0 = time.monotonic()
        be = encode_tf(bx, bg, bb, str(tf_ckpt_dir))
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
        description="TranscriptFormer embedding-based deconvolution: predict (standalone)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to deconv_head.pt checkpoint")
    parser.add_argument("--log_file", default=None, help="Log file path")
    parser.add_argument("--ground-truth", default=None,
                        help="Path to ground truth CSV (overrides H5 ground_truth group)")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.log_file, args.ground_truth)
