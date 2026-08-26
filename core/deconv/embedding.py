"""
Shared components for embedding-based deconvolution (STACK, TranscriptFormer,
scGPT-LoRA, etc.).

Provides pseudo-bulk generation, configurable train/val/test split, MLP head,
and evaluation helpers.  All methods use the same split logic controlled by
``PseudoBulkConfig`` (YAML-serialisable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PseudoBulkConfig:
    """Standardised pseudo-bulk generation + split settings.

    All fields have defaults matching the to_publish standard (5000 pb,
    80/20 train/val split).
    """
    n_pseudo_bulk: int = 5000
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    seed: int = 42
    min_cells_per_type: int = 10
    max_cells_per_type: int = 100

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio = {total}, must sum to 1"
            )

    @property
    def n_train(self) -> int:
        return int(self.n_pseudo_bulk * self.train_ratio)

    @property
    def n_val(self) -> int:
        return int(self.n_pseudo_bulk * self.val_ratio)

    @property
    def n_test(self) -> int:
        return self.n_pseudo_bulk - self.n_train - self.n_val


def split_indices(
    n_total: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) for a pseudo-bulk dataset.

    When ``test_ratio == 0`` (default), the split is train/val only.
    The pseudo-bulk generator already randomises sample order, so indices
    are taken contiguously after one permutation (no second shuffle needed).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    n_tr = int(n_total * train_ratio)
    n_va = int(n_total * val_ratio)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def split_from_config(
    n_total: int,
    cfg: PseudoBulkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shorthand: call ``split_indices`` from a ``PseudoBulkConfig``."""
    return split_indices(
        n_total,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )


class MixGenerator:
    """Generate pseudo-bulk mixtures from cell embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_types: int,
        n_pb: int = 5000,
        seed: int = 42,
    ):
        from numpy.random import default_rng

        self.emb = embeddings
        self.labels = labels
        self.n_types = n_types
        self.n_pb = n_pb
        self.rng = default_rng(seed)
        self.ti = [np.where(labels == t)[0] for t in range(n_types)]

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        n, d = self.n_pb, self.emb.shape[1]
        be = np.zeros((n, d), dtype=np.float32)
        pr = np.zeros((n, self.n_types), dtype=np.float32)
        for i in range(n):
            na = self.rng.integers(2, min(self.n_types + 1, 6))
            ac = self.rng.choice(self.n_types, na, replace=False)
            p = np.zeros(self.n_types)
            for t, v in zip(ac, self.rng.dirichlet([1.0] * na)):
                p[t] = v
            nc = self.rng.integers(10, 100)
            cc = self.rng.multinomial(nc, p)
            total = np.zeros(d, dtype=np.float64)
            nt = 0
            for t in range(self.n_types):
                ct = cc[t]
                if ct == 0:
                    continue
                total += self.emb[
                    self.rng.choice(self.ti[t], ct, replace=True)
                ].sum(axis=0)
                nt += ct
            be[i] = (total / nt).astype(np.float32)
            pr[i] = p
        return be, pr


class ExpressionMixGenerator:
    """Generate pseudo-bulk mixtures from raw expression counts.

    Each pseudo-bulk sample is created by mixing raw counts from cells of
    different types, then normalising (CPM-like).  This produces expression-
    space mixtures that can be encoded through a frozen or unfrozen backbone.
    """

    def __init__(
        self,
        counts: np.ndarray,
        labels: np.ndarray,
        n_types: int,
        n_pb: int = 5000,
        seed: int = 42,
        min_cells: int = 10,
        max_cells: int = 100,
    ):
        from numpy.random import default_rng

        self.counts = counts
        self.labels = labels
        self.n_types = n_types
        self.n_pb = n_pb
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.rng = default_rng(seed)
        self.ti = [np.where(labels == t)[0] for t in range(n_types)]

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        n, g = self.n_pb, self.counts.shape[1]
        pb = np.zeros((n, g), dtype=np.float32)
        pr = np.zeros((n, self.n_types), dtype=np.float32)
        for i in range(n):
            na = self.rng.integers(2, min(self.n_types + 1, 6))
            ac = self.rng.choice(self.n_types, na, replace=False)
            p = np.zeros(self.n_types)
            for t, v in zip(ac, self.rng.dirichlet([1.0] * na)):
                p[t] = v
            nc = self.rng.integers(self.min_cells, self.max_cells)
            cc = self.rng.multinomial(nc, p)
            total = np.zeros(g, dtype=np.float64)
            nt = 0
            for t in range(self.n_types):
                ct = cc[t]
                if ct == 0:
                    continue
                total += self.counts[
                    self.rng.choice(self.ti[t], ct, replace=True)
                ].sum(axis=0)
                nt += ct
            pb[i] = (total / nt).astype(np.float32)
            pr[i] = p
        return pb, pr


class EmbeddingDeconvHead(torch.nn.Module):
    """MLP head that predicts proportions from mixed cell embeddings.

    Two modes:
    - ``legacy`` (default): fixed 3-layer ``dim -> hidden -> hidden//2 -> n_types``.
    - ``progressive``: ``n_progressive_layers`` linear layers with evenly decreasing
      dimensions from ``dim`` down to ``n_types``.

    Each hidden layer: Linear -> BatchNorm1d -> ReLU -> Dropout(0.2).

    Supports ``get_hidden(x)`` for domain adaptation (returns all hidden features).
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        n_types: int,
        progressive: bool = False,
        n_progressive_layers: int = 3,
    ):
        super().__init__()
        if progressive:
            dims = _progressive_dims(dim, n_types, n_progressive_layers)
        else:
            dims = [dim, hidden, hidden // 2, n_types]

        self.linears = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = 0.2
        for i in range(len(dims) - 1):
            self.linears.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.bns.append(torch.nn.BatchNorm1d(dims[i + 1]))
            else:
                self.bns.append(torch.nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._forward_impl(x), dim=-1)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward through all layers, returning logits."""
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return self.linears[-1](x)

    def get_hidden(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return intermediate hidden features for domain alignment.

        Uses eval-style forward (no dropout) for stable feature representations.
        Returns a list of hidden activations (one per non-final layer).
        """
        hiddens = []
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            hiddens.append(x)
        return hiddens


def _progressive_dims(input_dim: int, n_types: int, n_layers: int) -> list[int]:
    """Compute evenly decreasing hidden dimensions.

    ``n_layers`` linear layers means ``n_layers + 1`` points including
    ``input_dim`` and ``n_types``.  Dimensions are rounded to multiples
    of 32 for GPU-friendliness, then clipped to ``>= n_types``.
    """
    import math

    if n_layers <= 1:
        return [input_dim, n_types]

    dims = []
    for i in range(n_layers + 1):
        t = i / n_layers  # 0.0 .. 1.0
        d = int(round(input_dim * (1 - t) + n_types * t))
        d = max(d, n_types)
        dims.append(d)

    # Collapse duplicate adjacent dims (preserving n_types at the end)
    uniq = [dims[0]]
    for d in dims[1:]:
        if d != uniq[-1]:
            uniq.append(d)
    if uniq[-1] != n_types:
        uniq.append(n_types)
    return uniq


def evaluate_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    cell_types: list[str],
) -> dict:
    """Per-cell-type Pearson r and macro-average.

    Args:
        predictions: (n_samples, n_cell_types) predicted proportions.
        ground_truth: (n_samples, n_cell_types) ground-truth proportions.
        cell_types: Names of cell types.

    Returns:
        Dict with ``pearson_mean`` and ``pearson_per_type`` keys.
    """
    n_ct = ground_truth.shape[1]
    pearson_rs = []
    for i in range(n_ct):
        if np.std(ground_truth[:, i]) > 0:
            r = float(np.corrcoef(ground_truth[:, i], predictions[:, i])[0, 1])
        else:
            r = float("nan")
        pearson_rs.append(r)

    valid = [r for r in pearson_rs if not np.isnan(r)]
    return {
        "pearson_mean": float(np.mean(valid)) if valid else 0.0,
        "pearson_per_type": dict(zip(cell_types, pearson_rs)),
    }
