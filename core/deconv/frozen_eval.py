#!/usr/bin/env python3
"""Frozen backbone embedding extraction + unified real-bulk RidgeCV evaluation.

Central registry of encode functions for all 6 foundation model backbones,
producing embeddings that feed into RidgeCV deconvolution.  Evaluation
output follows the to_publish standard: ``proportions.csv`` + ``metrics.json``
(via ``core.metrics.evaluate_deconvolution``) + ``ridge_metrics.json`` +
``metadata.json``.

Usage::

    from core.deconv.frozen_eval import ENCODE_FN, evaluate_real_bulk_ridge

    emb = ENCODE_FN["stack"](raw_counts, gene_symbols, barcodes)
    result = evaluate_real_bulk_ridge(emb, gt_df, dataset_name="sdy67", seed=42)
    # result["deconbench"]  → DeconBenchmark 4-metric suite
    # result["ridge_specific"] → best_alpha, per-type r/rmse
    # result["metadata"]   → resource usage + env info
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.deconv.utils import renormalize_props
from core.deconv.frozen_paths import (
    _WEIGHTS_DIR,
    _SCPEFT_DIR,
    _SCGPT_MODEL_DIR,
    _GENEFORMER_MODEL_DIR,
    _SCFOUNDATION_CKPT,
    _SCFOUNDATION_GENE_INDEX,
    _TF_CKPT_DIR,
    _TF_REPO_SRC,
    _TF_GENE_CACHE,
    _STACK_CHECKPOINT,
    _STACK_GENELIST,
    _GENEFORMER_TOKEN_DICT,
    _GENEFORMER_NAME_TO_ENSEMBL,
    _DEVICE,
)

warnings.filterwarnings("ignore")

EncodeFn = Callable[[np.ndarray, list[str], list[str]], np.ndarray]

# ── Constants ─────────────────────────────────────────────────────────────────

SDY67_TRAIN_N = 150
SDY67_VAL_N = 50
SDY67_TEST_N = 50
DEFAULT_ALPHAS = [0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0]
DEFAULT_SEED = 42


# ── Resource tracker ──────────────────────────────────────────────────────────


class ResourceTracker:
    """Context manager that records wall time and GPU memory usage."""

    def __init__(self) -> None:
        self.wall_start: float = 0.0
        self.encode_time_s: float = 0.0
        self.ridge_time_s: float = 0.0
        self.wall_time_s: float = 0.0
        self.gpu_memory_peak_mb: float = 0.0
        self.gpu_name: str = ""
        self.cuda_version: str = ""
        self.conda_env: str = ""
        self._phase_start: float = 0.0

    def __enter__(self) -> ResourceTracker:
        import torch
        self.wall_start = time.monotonic()
        self._phase_start = self.wall_start
        self.conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        self.cuda_version = getattr(torch.version, "cuda", "none")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.gpu_name = torch.cuda.get_device_name(0)
        return self

    def __exit__(self, *args: Any) -> None:
        import torch
        self.wall_time_s = round(time.monotonic() - self.wall_start, 1)
        if torch.cuda.is_available():
            self.gpu_memory_peak_mb = round(
                torch.cuda.max_memory_allocated(0) / 1e6, 1
            )

    def start_encode(self) -> None:
        self._phase_start = time.monotonic()

    def end_encode(self) -> None:
        self.encode_time_s = round(time.monotonic() - self._phase_start, 1)
        self._phase_start = time.monotonic()

    def end_ridge(self) -> None:
        self.ridge_time_s = round(time.monotonic() - self._phase_start, 1)

    def to_dict(self, **extra: Any) -> dict:
        return {
            "wall_time_s": self.wall_time_s,
            "encode_time_s": self.encode_time_s,
            "ridge_time_s": self.ridge_time_s,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "gpu_name": self.gpu_name,
            "cuda_version": self.cuda_version,
            "conda_env": self.conda_env,
            **extra,
        }


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_real_bulk_ridge(
    embeddings: np.ndarray,
    gt_df: pd.DataFrame,
    dataset_name: str = "",
    seed: int = DEFAULT_SEED,
    alphas: list[float] | None = None,
    use_scaler: bool = False,
) -> dict:
    """Real-bulk split + RidgeCV evaluation producing to_publish-standard metrics.

    1. Train/test split (fixed 150/50 for SDY67, else 80/20 random).
    2. Optional StandardScaler + RidgeCV per GT column (default: no scaler).
    3. Predict on test → clip negative → renormalize sum=1.
    4. Call ``core.metrics.evaluate_deconvolution`` for DeconBenchmark suite.
    5. Also compute per-type Pearson r / RMSE for RidgeCV-specific output.
    6. Generate full-sample predictions for ``proportions.csv`` export.

    When ``use_scaler=True``, embeddings are centered + scaled before RidgeCV
    (matching the original to_publish behavior).  When ``False`` (default),
    raw embeddings feed directly into RidgeCV, matching scPEFT/BulkFormer
    original evaluation pipelines.  Scaler results are routed to a
    ``{method}_scaler/`` output directory by the caller.

    Returns:
        ``{"deconbench": {...}, "ridge_specific": {...}, "metadata": {...},
           "full_predictions_df": DataFrame}``.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    np.random.seed(seed)
    alphas = alphas or DEFAULT_ALPHAS

    n_bulk = embeddings.shape[0]
    gt_values = gt_df.values.astype(np.float64)
    gt_columns = list(gt_df.columns)
    embed_dim = embeddings.shape[1]

    # ── Split ──
    if dataset_name == "sdy67":
        # Fixed 6:2:2: train=0-149, val=150-199, test=200-249
        train_idx = np.arange(SDY67_TRAIN_N + SDY67_VAL_N)  # RidgeCV uses train+val (no val needed)
        test_n = min(SDY67_TEST_N, n_bulk - SDY67_TRAIN_N - SDY67_VAL_N)
        test_idx = np.arange(SDY67_TRAIN_N + SDY67_VAL_N, SDY67_TRAIN_N + SDY67_VAL_N + test_n)
        split_label = f"fixed_{SDY67_TRAIN_N + SDY67_VAL_N}_{test_n}"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(n_bulk), test_size=0.2, random_state=seed
        )
        split_label = "random_80_20"

    train_emb, test_emb = embeddings[train_idx], embeddings[test_idx]
    train_gt, test_gt = gt_values[train_idx], gt_values[test_idx]

    # ── Optional scaler + RidgeCV per type ──
    scaler: Any | None = None
    if use_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_s = scaler.fit_transform(train_emb)
        test_s = scaler.transform(test_emb)
    else:
        train_s, test_s = train_emb, test_emb

    models: dict[str, RidgeCV] = {}
    test_pred = np.zeros_like(test_gt)
    best_alphas: dict[str, float] = {}
    pearson_per_type: dict[str, float | None] = {}
    rmse_per_type: dict[str, float | None] = {}

    ridge_start = time.monotonic()
    for j, ct in enumerate(gt_columns):
        y = train_gt[:, j]
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            test_pred[:, j] = 0.0
            best_alphas[ct] = float("nan")
            pearson_per_type[ct] = None
            rmse_per_type[ct] = None
            continue
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(train_s[mask], y[mask])
        models[ct] = ridge
        best_alphas[ct] = float(ridge.alpha_)
        test_pred[:, j] = ridge.predict(test_s)

    # Clip neg → renormalize
    test_pred = renormalize_props(test_pred, zero_fill="zero")
    # Guard against NaN/Inf from degenerate RidgeCV splits (<~5 samples per type)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-type Pearson r / RMSE
    for j, ct in enumerate(gt_columns):
        mask = ~np.isnan(test_gt[:, j])
        if mask.sum() >= 2 and np.std(test_gt[mask, j]) > 1e-10:
            r = float(np.corrcoef(test_pred[mask, j], test_gt[mask, j])[0, 1])
            pearson_per_type[ct] = round(r, 4) if not np.isnan(r) else None
        else:
            pearson_per_type[ct] = None
        if mask.sum() >= 2:
            rmse_per_type[ct] = round(
                float(np.sqrt(np.mean((test_pred[mask, j] - test_gt[mask, j]) ** 2))),
                4,
            )
        else:
            rmse_per_type[ct] = None

    ridge_time_s = round(time.monotonic() - ridge_start, 3)

    # ── DeconBenchmark metrics (to_publish standard) ──
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from core.metrics import evaluate_deconvolution

    deconbench = evaluate_deconvolution(test_gt, test_pred, gt_columns)

    # ── Full-sample predictions (for proportions.csv) ──
    full_x = scaler.transform(embeddings) if scaler else embeddings
    full_pred = np.zeros((n_bulk, len(gt_columns)))
    for j, ct in enumerate(gt_columns):
        if ct in models:
            full_pred[:, j] = models[ct].predict(full_x)
    full_pred = renormalize_props(full_pred, zero_fill="zero")
    full_pred_df = pd.DataFrame(full_pred, index=gt_df.index, columns=gt_columns)

    return {
        "deconbench": deconbench,
        "ridge_specific": {
            "best_alphas": best_alphas,
            "pearson_per_type": pearson_per_type,
            "rmse_per_type": rmse_per_type,
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
            "split": split_label,
            "ridge_time_s": ridge_time_s,
        },
        "metadata": {
            "embedding_dim": embed_dim,
            "n_total": n_bulk,
            "split": split_label,
            "seed": seed,
        },
        "full_predictions_df": full_pred_df,
    }


# ── Encode function definitions (lazy, one per backbone) ──────────────────────


def _encode_stack(
    raw_counts: np.ndarray, gene_symbols: list[str], barcodes: list[str]
) -> np.ndarray:
    """STACK backbone: frozen embeddings via methods/stack/train.py."""
    _project = Path(__file__).resolve().parent.parent.parent
    if str(_project) not in sys.path:
        sys.path.insert(0, str(_project))
    from methods.stack.train import encode_stack as _fn

    return _fn(raw_counts, gene_symbols, barcodes, _STACK_CHECKPOINT, _STACK_GENELIST)


def _encode_tf(
    raw_counts: np.ndarray, gene_symbols: list[str], barcodes: list[str]
) -> np.ndarray:
    """TranscriptFormer backbone: frozen embeddings.

    Adapted from scPEFT ``auto_search.py:encode_tf()``.
    """
    import anndata as ad
    from omegaconf import OmegaConf
    from scipy.sparse import csr_matrix

    # ── Gene → Ensembl mapping ──
    gene_cache: dict[str, str] = {}
    if _TF_GENE_CACHE.exists():
        with open(_TF_GENE_CACHE) as f:
            gene_cache = json.load(f)

    n_ensg = sum(1 for g in gene_symbols if str(g).startswith("ENSG"))
    if n_ensg < len(gene_symbols) * 0.5:
        mapped = [gene_cache.get(g, g) for g in gene_symbols]
        valid = [i for i, g in enumerate(mapped) if g and g.startswith("ENSG")]
        if len(valid) < 10:
            raise ValueError(
                f"TranscriptFormer: too few genes mapped to Ensembl IDs "
                f"({len(valid)}/{len(gene_symbols)})"
            )
        raw_counts = raw_counts[:, valid] if raw_counts.ndim == 2 else raw_counts
        gene_symbols = [mapped[i] for i in valid]

    _tf_repo_src = _WEIGHTS_DIR / "transcriptformer/tf_sapiens/repo/src"
    if _tf_repo_src.exists():
        sys.path.insert(0, str(_tf_repo_src))
    else:
        sys.path.insert(0, _TF_REPO_SRC)
    from transcriptformer.model.inference import run_inference

    with open(os.path.join(str(_TF_CKPT_DIR), "config.json")) as f:
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
                "aux_vocab_path": os.path.join(str(_TF_CKPT_DIR), "vocabs"),
                "esm2_mappings_path": os.path.join(str(_TF_CKPT_DIR), "vocabs"),
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
                "special_tokens": [
                    "unknown", "[PAD]", "[START]", "[END]", "[RD]", "[CELL]", "[MASK]"
                ],
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
                "load_checkpoint": os.path.join(str(_TF_CKPT_DIR), "model_weights.pt"),
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
        var=pd.DataFrame(index=gene_symbols),
    )
    adata.var["ensembl_id"] = adata.var.index
    adata.obs["assay"] = "10x 3' transcription profiling"
    result = run_inference(cfg, data_files=[adata])
    return result.obsm["embeddings"]


def _encode_scfoundation(
    raw_counts: np.ndarray, gene_symbols: list[str], barcodes: list[str]
) -> np.ndarray:
    """scFoundation backbone: zero-fill 19264-dim → CLS embeddings.

    Adapted from scPEFT ``auto_search.py:encode_scfoundation()``.
    """
    import torch

    _project = Path(__file__).resolve().parent.parent.parent
    if str(_project) not in sys.path:
        sys.path.insert(0, str(_project))
    from methods.scfoundation.model import ScFoundationBackbone

    device = _DEVICE
    backbone = ScFoundationBackbone(str(_SCFOUNDATION_CKPT), device=device)
    backbone.eval()

    scf_genes = pd.read_csv(_SCFOUNDATION_GENE_INDEX, sep="\t", header=None)[0].tolist()
    gene_to_idx = {g.upper(): i for i, g in enumerate(scf_genes)}

    matched = [gene_to_idx[g.upper()] for g in gene_symbols if g.upper() in gene_to_idx]
    print(f"  scFoundation: matched {len(matched)}/{len(gene_symbols)} genes")

    gene_positions = torch.tensor(matched, device=device, dtype=torch.long).unsqueeze(0)
    raw = np.asarray(raw_counts)
    raw_subset = raw[:, [g.upper() in gene_to_idx for g in gene_symbols]]
    raw_subset = np.log1p(raw_subset)  # prevent float16 NaN in softmax

    all_embs = []
    batch_size = 8
    for i in range(0, len(raw_subset), batch_size):
        batch = raw_subset[i : i + batch_size]
        values = torch.from_numpy(batch).float().to(device)
        padding_mask = torch.zeros(
            batch.shape[0], batch.shape[1], dtype=torch.bool, device=device
        )
        gene_ids = gene_positions.expand(batch.shape[0], -1)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            output = backbone(gene_ids, values, padding_mask)
            emb = output[:, 0, :].cpu().float().numpy()
        nan_mask = np.isnan(emb).any(axis=1)
        if nan_mask.any():
            print(
                f"  scFoundation WARNING: {nan_mask.sum()}/{len(emb)} NaN rows "
                f"replaced with zeros"
            )
            emb[nan_mask] = 0.0
        all_embs.append(emb)
    return np.concatenate(all_embs, axis=0)


_scgpt_cache: dict = {}  # {device: (model, vocab)}


def _encode_scgpt(
    raw_counts: np.ndarray, gene_symbols: list[str], barcodes: list[str]
) -> np.ndarray:
    """scGPT backbone: whole-human model → mean pooled cell embeddings.

    Adapted from scPEFT ``auto_search.py:encode_scgpt()``.
    Lazy-loads model once (singleton) to avoid per-call random init
    with ``load_state_dict(strict=False)``.
    """
    import anndata as ad

    import scanpy as sc
    import torch

    device = _DEVICE
    if device not in _scgpt_cache:
        scgpt_repo = _SCPEFT_DIR / "repo"
        sys.path.insert(0, str(scgpt_repo))
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab

        vocab = GeneVocab.from_file(str(_SCGPT_MODEL_DIR / "vocab.json"))
        model = TransformerModel(
            ntoken=len(vocab),
            d_model=512,
            nhead=8,
            nlayers=12,
            nlayers_cls=3,
            d_hid=512,
            vocab=vocab,
            pad_token="<pad>",
            pad_value=0,
            do_mvc=True,
            use_fast_transformer=True,
        )
        ckpt = torch.load(
            str(_SCGPT_MODEL_DIR / "best_model.pt"), map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt, strict=False)
        model.to(device).eval()
        _scgpt_cache[device] = (model, vocab)
    else:
        model, vocab = _scgpt_cache[device]

    adata = ad.AnnData(X=np.asarray(raw_counts), var=pd.DataFrame(index=gene_symbols))
    if adata.shape[1] > 3000:
        # Fix seeds RIGHT before HVG: sc.pp.highly_variable_genes(flavor=seurat_v3)
        # uses internal randomness without explicit seed. model loading and
        # other imports may have consumed random state since function entry.
        np.random.seed(42)
        torch.manual_seed(42)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"  scGPT: filtered {len(gene_symbols)} -> {adata.shape[1]} HVGs")
        raw_counts = adata.X
        gene_symbols = adata.var_names.tolist()

    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in gene_symbols]
    )
    values = torch.from_numpy(np.asarray(raw_counts)).float()
    embeddings = []
    batch_size = 64
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in range(0, len(values), batch_size):
            batch_val = values[i : i + batch_size].to(device)
            batch_genes = torch.from_numpy(
                np.tile(gene_ids, (batch_val.shape[0], 1))
            ).to(device)
            cls_token = torch.full(
                (batch_val.shape[0], 1), vocab["<cls>"], device=device
            )
            cls_val = torch.zeros((batch_val.shape[0], 1), device=device)
            batch_genes = torch.cat([cls_token, batch_genes], dim=1)
            batch_val = torch.cat([cls_val, batch_val], dim=1)
            padding_mask = torch.zeros(
                batch_val.shape[0], batch_genes.shape[1], device=device
            ).bool()
            output = model._encode(batch_genes, batch_val, padding_mask)
            emb = output[:, 1:, :].mean(dim=1)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _encode_geneformer(
    raw_counts: np.ndarray, gene_symbols: list[str], barcodes: list[str]
) -> np.ndarray:
    """Geneformer backbone: rank-value encoding → mean-pooled embeddings.

    Adapted from scPEFT ``auto_search.py:encode_geneformer()``.
    """
    import pickle

    import torch
    from transformers import AutoModel

    device = _DEVICE

    with open(_GENEFORMER_TOKEN_DICT, "rb") as f:
        token_dict = pickle.load(f)
    with open(_GENEFORMER_NAME_TO_ENSEMBL, "rb") as f:
        name_to_ensembl = pickle.load(f)

    gene_to_token: dict[int, int] = {}
    for i, g in enumerate(gene_symbols):
        if g in name_to_ensembl:
            ensembl = name_to_ensembl[g]
            if ensembl in token_dict:
                gene_to_token[i] = token_dict[ensembl]
    print(f"  Geneformer: matched {len(gene_to_token)}/{len(gene_symbols)} genes")

    model = AutoModel.from_pretrained(
        str(_GENEFORMER_MODEL_DIR), trust_remote_code=True
    )
    model.to(device).eval()
    max_len = model.config.max_position_embeddings
    pad_id = token_dict["<pad>"]

    all_embs = []
    batch_size = 32
    raw = np.asarray(raw_counts)
    for i in range(0, len(raw), batch_size):
        batch = raw[i : i + batch_size]
        batch_ids, batch_attn = [], []
        for sample in batch:
            valid = [(idx, sample[idx]) for idx in gene_to_token if sample[idx] > 0]
            valid.sort(key=lambda x: x[1], reverse=True)
            gene_tokens = [gene_to_token[idx] for idx, _ in valid]
            gene_tokens = gene_tokens[: max_len - 1]
            tokens = [pad_id] + gene_tokens
            seq_len = len(tokens)
            tokens = tokens + [pad_id] * (max_len - seq_len)
            attn = [1] * seq_len + [0] * (max_len - seq_len)
            batch_ids.append(tokens)
            batch_attn.append(attn)
        inp = torch.tensor(batch_ids, device=device)
        am = torch.tensor(batch_attn, device=device)
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=torch.cuda.is_available()
        ):
            out = model(input_ids=inp, attention_mask=am).last_hidden_state
            mask = am.unsqueeze(-1).float().to(out.device)
            emb = (
                (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            ).cpu().numpy()
        all_embs.append(emb)
    return np.concatenate(all_embs, axis=0)


def _encode_bulkformer(
    raw_counts: np.ndarray, gene_symbols: list[str], barcodes: list[str]
) -> np.ndarray:
    """BulkFormer backbone: 20,010-gene alignment → frozen encoder.

    Delegates to ``methods/bulkformer/model.py:encode_bulkformer()``.
    """
    _project = Path(__file__).resolve().parent.parent.parent
    if str(_project) not in sys.path:
        sys.path.insert(0, str(_project))
    from methods.bulkformer.model import encode_bulkformer as _fn

    return _fn(raw_counts, gene_symbols, barcodes)


def _encode_pca_ridge(
    raw_counts: np.ndarray,
    gene_symbols: list[str],
    barcodes: list[str],
) -> np.ndarray:
    """PCA-Ridge: PCA once on full data (like methods/pca_ridge/run.py).

    PCA is fit once on all samples, then LOO RidgeCV runs on the low-dim
    PCA features.  Minor data leakage (test sample influences PCA
    components) but orders-of-magnitude faster, matching the original
    pca_ridge baseline evaluation.
    """
    from sklearn.decomposition import PCA

    n_samples = raw_counts.shape[0]
    n_components = min(n_samples, raw_counts.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(raw_counts).astype(np.float64)


# ── Registry ──────────────────────────────────────────────────────────────────

ENCODE_FN: dict[str, EncodeFn] = {
    "stack": _encode_stack,
    "transcriptformer": _encode_tf,
    "scfoundation": _encode_scfoundation,
    "scgpt": _encode_scgpt,
    "geneformer": _encode_geneformer,
    "bulkformer": _encode_bulkformer,
    "pca_ridge": _encode_pca_ridge,
}
