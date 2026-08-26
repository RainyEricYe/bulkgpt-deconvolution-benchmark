#!/usr/bin/env python3
"""BulkFormer-147M frozen encoder wrapper.

Loads pretrained BulkFormer (GCN+Performer, 20,010 genes, 640-dim) and
provides a lightweight ``encode_bulkformer()`` function that maps raw
bulk/scRNA expression to sample-level embeddings.

Environment variable ``BULKFORMER_DIR`` overrides the default source path.

Usage::

    from methods.bulkformer.model import encode_bulkformer, BulkFormerEncoder

    emb = encode_bulkformer(raw_counts, gene_symbols, barcodes)
    # emb: (n_samples, 640) float32 array
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BULKFORMER_DIR = Path(os.environ.get(
    "BULKFORMER_DIR",
    str(Path(__file__).resolve().parents[2] / "weights" / "bulkformer" / "source"),
))
GENE_COUNT = 20010


class BulkFormerEncoder:
    """Cached BulkFormer-147M encoder with 20,010-gene alignment.

    On first call the model, gene graph, and ESM2 embeddings are loaded
    (~30 s on H100).  Subsequent calls reuse the cached model.

    Args:
        device: Target device (auto-detected if empty).
        pretrained: If True (default), load the ``BulkFormer_147M.pt``
            checkpoint.  If False, keep randomly initialized weights
            (for control experiments comparing pretrained vs random).
    """

    def __init__(self, device: str = "", pretrained: bool = True) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pretrained = pretrained
        self._model: torch.nn.Module | None = None
        self._gene_list: list[str] = []

    def _load(self) -> None:
        """One-time lazy loader for model + gene assets."""
        if self._model is not None:
            return
        bf_dir = str(BULKFORMER_DIR)
        if bf_dir not in sys.path:
            sys.path.insert(0, bf_dir)

        # Load BulkFormer source modules from absolute file paths to avoid
        # name shadowing (train.py puts methods/bulkformer/ in sys.path).
        import importlib.util, os as _os
        _cfg_p = _os.path.join(bf_dir, "model", "config.py")
        _mcs = importlib.util.spec_from_file_location("_bf_config", _cfg_p)
        if _mcs is None:
            raise ImportError(f"Cannot find model.config at {_cfg_p}")
        _mc = importlib.util.module_from_spec(_mcs)
        _mcs.loader.exec_module(_mc)
        model_params = _mc.model_params

        _bf_p = _os.path.join(bf_dir, "utils", "BulkFormer.py")
        _ubs = importlib.util.spec_from_file_location("_bf_utils", _bf_p)
        if _ubs is None:
            raise ImportError(f"Cannot find utils.BulkFormer at {_bf_p}")
        _ub = importlib.util.module_from_spec(_ubs)
        _ubs.loader.exec_module(_ub)
        BulkFormer = _ub.BulkFormer

        print("  BulkFormer: loading graph...", end=" ", flush=True)
        t0 = time.monotonic()
        ei = torch.load(
            BULKFORMER_DIR / "data" / "G_tcga.pt",
            map_location="cpu", weights_only=False,
        )
        ew = torch.load(
            BULKFORMER_DIR / "data" / "G_tcga_weight.pt",
            map_location="cpu", weights_only=False,
        )
        graph = (
            torch.sparse_coo_tensor(ei, ew, (GENE_COUNT, GENE_COUNT))
            .coalesce()
            .to(self._device)
        )
        print(f"{time.monotonic() - t0:.1f}s")

        print("  BulkFormer: loading gene embeddings...", end=" ", flush=True)
        t0 = time.monotonic()
        gene_emb = torch.load(
            BULKFORMER_DIR / "data" / "esm2_feature_concat.pt",
            map_location="cpu", weights_only=False,
        ).to(self._device)
        print(f"{time.monotonic() - t0:.1f}s")

        print("  BulkFormer: initializing model...", end=" ", flush=True)
        t0 = time.monotonic()
        self._model = BulkFormer(
            dim=model_params["dim"],
            graph=graph,
            gene_emb=gene_emb,
            gene_length=model_params["gene_length"],
            bin_head=model_params["bin_head"],
            full_head=model_params["full_head"],
            bins=model_params["bins"],
            gb_repeat=model_params["gb_repeat"],
            p_repeat=model_params["p_repeat"],
        )
        if self._pretrained:
            ckpt = torch.load(
                BULKFORMER_DIR / "model" / "BulkFormer_147M.pt",
                map_location="cpu", weights_only=False,
            )
            ckpt = {
                k[7:] if k.startswith("module.") else k: v
                for k, v in ckpt.items()
            }
            self._model.load_state_dict(ckpt)
        self._model.to(self._device).eval()
        n = sum(p.numel() for p in self._model.parameters())
        print(f"{n:,} params ({time.monotonic() - t0:.1f}s)")

        gi = pd.read_csv(BULKFORMER_DIR / "data" / "bulkformer_gene_info.csv")
        self._gene_list = list(gi.iloc[:, 0])
        print(f"  BulkFormer: gene vocabulary = {len(self._gene_list)}")

    def encode(
        self,
        raw_counts: np.ndarray,
        gene_symbols: list[str],
        barcodes: list[str],
        pooling: str = "global_proj",
    ) -> np.ndarray:
        """Encode expression through frozen BulkFormer.

        Args:
            raw_counts: (n_samples, n_input_genes) expression matrix.
            gene_symbols: Gene names matching columns of *raw_counts*.
            barcodes: Sample / cell barcodes (unused except for shape).
            pooling: ``"global_proj"`` (fast MLP path, r≈0.63 on SDY67)
                     or ``"mean"`` (full GCN+Performer encoder, slower).

        Returns:
            (n_samples, embed_dim) ``float32`` array.
        """
        self._load()
        n_samples = raw_counts.shape[0]

        g2i = {g.upper(): i for i, g in enumerate(self._gene_list)}
        mat = np.full((n_samples, len(self._gene_list)), -10.0, dtype=np.float32)
        matched = 0
        for i, g in enumerate(gene_symbols):
            idx = g2i.get(str(g).upper())
            if idx is not None:
                mat[:, idx] = raw_counts[:, i]
                matched += 1
        print(f"  BulkFormer: aligned {matched}/{len(gene_symbols)} genes")

        x = torch.from_numpy(mat).to(self._device)
        with torch.no_grad():
            if pooling == "global_proj":
                result = self._model.global_expr_proj(x).cpu().numpy()
            else:
                result = []
                for i in range(0, n_samples, 4):
                    xb = x[i : i + 4]
                    emb = self._model(xb, output_expr=False).mean(dim=1)
                    result.append(emb.cpu().numpy())
                result = np.concatenate(result, axis=0)
        print(f"  BulkFormer: embeddings shape = {result.shape}")
        return result


# ── Module-level singleton ────────────────────────────────────────────────────

_encoder: BulkFormerEncoder | None = None


def encode_bulkformer(
    raw_counts: np.ndarray,
    gene_symbols: list[str],
    barcodes: list[str],
    pooling: str = "global_proj",
) -> np.ndarray:
    """Module-level convenience wrapper (singleton pattern)."""
    global _encoder
    if _encoder is None:
        _encoder = BulkFormerEncoder()
    return _encoder.encode(raw_counts, gene_symbols, barcodes, pooling=pooling)
