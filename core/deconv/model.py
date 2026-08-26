import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvHead(nn.Module):
    """MLP deconvolution head predicting cell type proportions.

    Maps pooled cell embedding -> hidden -> softmax proportions.
    """

    def __init__(
        self,
        input_dim: int,
        n_cell_types: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_cell_types))
        self.net = nn.Sequential(*layers)

    def forward(self, cell_emb: torch.Tensor) -> torch.Tensor:
        """Returns log-proportions (logits)."""
        return self.net(cell_emb)


class DeconvLoss(nn.Module):
    """Combined loss for deconvolution.

    Supports three modes:
    - "mse_kl":  MSE + kl_weight * KL divergence (default)
    - "mse_cos": MSE + cos_weight * (1 - cosine_similarity)
    - "mse":     Pure MSE (baseline)
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        kl_weight: float = 0.1,
        cos_weight: float = 1.0,
        loss_type: str = "mse_kl",
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.kl_weight = kl_weight
        self.cos_weight = cos_weight
        assert loss_type in ("mse_kl", "mse_cos", "mse"), f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type

    def forward(
        self,
        pred_props: torch.Tensor,
        true_props: torch.Tensor,
    ) -> dict:
        mse = F.mse_loss(pred_props, true_props)

        if self.loss_type == "mse_kl":
            eps = 1e-8
            kl = (true_props * (torch.log(true_props + eps) - torch.log(pred_props + eps))).sum(dim=-1).mean()
            cos = torch.tensor(0.0, device=pred_props.device)
            total = self.mse_weight * mse + self.kl_weight * kl
        elif self.loss_type == "mse_cos":
            cos = 1.0 - F.cosine_similarity(pred_props, true_props, dim=-1).mean()
            kl = torch.tensor(0.0, device=pred_props.device)
            total = self.mse_weight * mse + self.cos_weight * cos
        else:  # "mse"
            kl = torch.tensor(0.0, device=pred_props.device)
            cos = torch.tensor(0.0, device=pred_props.device)
            total = self.mse_weight * mse

        return {
            "loss": total,
            "mse": mse.detach(),
            "kl": kl.detach(),
            "cos": cos.detach(),
        }
