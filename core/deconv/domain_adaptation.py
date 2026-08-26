"""Domain adaptation modules for deconvolution.

Provides:
- GradientReversalLayer (GRL) for adversarial domain adaptation
- DomainClassifier: binary source/target classifier with GRL
- mmd_loss: Maximum Mean Discrepancy loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation.

    Forward: identity.
    Backward: multiplies gradient by ``-lambda_``, reversing its direction
    so the upstream features become domain-invariant.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class DomainClassifier(nn.Module):
    """Binary domain classifier (source=0 / target=1) with GRL on input.

    During forward, the GRL reverses gradients flowing back to the
    encoder, encouraging domain-invariant cell embeddings.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        """Return domain logits after applying GRL."""
        x = GradientReversalLayer.apply(x, lambda_)
        return self.net(x).squeeze(-1)


def mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Maximum Mean Discrepancy between source and target embeddings.

    Args:
        source: (B, D) source-domain cell embeddings.
        target: (B, D) target-domain cell embeddings.
        kernel: ``"rbf"`` (default) or ``"linear"``.

    Returns:
        Scalar MMD loss.
    """
    if kernel == "rbf":
        return _rbf_mmd(source, target)
    elif kernel == "linear":
        return _linear_mmd(source, target)
    else:
        raise ValueError(f"Unknown MMD kernel: '{kernel}'")


def _rbf_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """RBF-kernel MMD. Subsamples the larger batch to match sizes."""
    b = min(x.size(0), y.size(0))
    x, y = x[:b], y[:b]

    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    K_xx = torch.exp(-0.5 * (rx + rx.t() - 2.0 * xx))
    K_yy = torch.exp(-0.5 * (ry + ry.t() - 2.0 * yy))
    K_xy = torch.exp(-0.5 * (rx + ry.t() - 2.0 * xy))

    return K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()


def _linear_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear-kernel MMD (mean embedding difference)."""
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    return (x_mean - y_mean).pow(2).sum()


class DomainAdaptationModule(nn.Module):
    """Convenience wrapper holding a DomainClassifier for GRL-based DA."""

    def __init__(self, emb_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.domain_classifier = DomainClassifier(emb_dim, hidden_dim)

    def compute_grl_loss(
        self,
        source_emb: torch.Tensor,
        target_emb: torch.Tensor,
        lambda_: float = 1.0,
    ) -> torch.Tensor:
        """Binary cross-entropy domain loss with GRL.

        Source label = 0, target label = 1.
        """
        batch_size = min(source_emb.size(0), target_emb.size(0))
        s = source_emb[:batch_size]
        t = target_emb[:batch_size]

        combined = torch.cat([s, t], dim=0)
        labels = torch.cat([
            torch.zeros(batch_size, device=s.device),
            torch.ones(batch_size, device=s.device),
        ])

        logits = self.domain_classifier(combined, lambda_=lambda_)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def compute_mmd_loss(
        self,
        source_emb: torch.Tensor,
        target_emb: torch.Tensor,
        kernel: str = "rbf",
    ) -> torch.Tensor:
        """MMD loss between source and target embeddings."""
        batch_size = min(source_emb.size(0), target_emb.size(0))
        return mmd_loss(
            source_emb[:batch_size],
            target_emb[:batch_size],
            kernel=kernel,
        )

    def compute_entropy_loss(
        self,
        target_proportions: torch.Tensor,
    ) -> torch.Tensor:
        """Entropy of target predictions — lower entropy = more confident.

        Useful as unsupervised loss on target domain when no labels available.
        """
        eps = 1e-8
        return -(target_proportions * (target_proportions + eps).log()).sum(dim=-1).mean()
