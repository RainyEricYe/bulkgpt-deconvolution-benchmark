from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Training hyperparameters shared across all methods."""

    seed: int = 42
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    backbone_lr: float | None = None  # lower LR for backbone; None = use lr for all
    n_pseudo_bulk: int = 10000
    proportion_alpha: float = 1.0
    loss_type: str = "mse_kl"  # "mse_kl" | "mse_cos" | "mse"
    num_workers: int = 4
    checkpoint_dir: str = "./checkpoints"
    use_wandb: bool = False
    train_ratio: float = 0.8
    n_hvg: int = 1200
    max_seq_len: int = 1201

    # ── Domain adaptation ───────────────────────────────────────────────
    da_method: str | None = None  # None | "grl" | "mmd" | "entropy"
    da_lambda: float = 0.05       # weight for domain loss
    da_grl_lambda: float = 1.0    # GRL strength (ramps from 0 to this over epochs)


@dataclass
class DeconvHeadConfig:
    """Deconvolution MLP head configuration (shared)."""

    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.2
    cell_emb_style: str = "cls"  # "cls", "mean", or "attn"


@dataclass
class DataConfig:
    """Data configuration shared across methods."""

    celltype_col: str = "cell_type"
    batch_col: str = "subject"
