from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from core.deconv.config import DeconvHeadConfig
from core.deconv.model import DeconvHead

import os

# Resolve Geneformer pretrained model directory.  Priority:
#   1. $GENEORMER_MODEL_DIR environment variable
#   2. PROJECT_ROOT/weights/geneformer/default/
#   3. PROJECT_ROOT/data/pretrained/geneformer/ (repo-local)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # to_publish/
_WEIGHTS_PATH = str(_PROJECT_ROOT / "weights" / "geneformer" / "default")
_GENEORMER_REPO_PATH = str(_PROJECT_ROOT / "data" / "pretrained" / "geneformer")
GENEORMER_PRETRAINED_PATH = os.environ.get(
    "GENEORMER_MODEL_DIR",
    _WEIGHTS_PATH if Path(_WEIGHTS_PATH).exists() else _GENEORMER_REPO_PATH,
)


def create_geneformer_backbone(
    model_dir: str | None = None,
    device: str = "cuda",
) -> nn.Module:
    """Create and load a pretrained Geneformer backbone (BERT base model).

    Returns BERT model (``transformers.AutoModel``) in eval mode.
    """
    if model_dir is None:
        model_dir = GENEORMER_PRETRAINED_PATH

    model_dir = str(model_dir)
    if not (Path(model_dir) / "config.json").exists():
        sub = str(Path(model_dir) / "Geneformer-V2-104M")
        if (Path(sub) / "config.json").exists():
            model_dir = sub
        else:
            model_dir = GENEORMER_PRETRAINED_PATH

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    print(f"Loaded Geneformer backbone from {model_dir} "
          f"(hidden_size={model.config.hidden_size}, vocab_size={model.config.vocab_size})")
    model.to(device)
    model.eval()
    return model


class GeneformerDeconvModel(nn.Module):
    """Geneformer BERT backbone + deconvolution head.

    forward() signature: ``(gene_ids, values, src_key_padding_mask) -> dict``

    Note: ``values`` is ignored — Geneformer encodes expression via rank-order
    of gene tokens, not explicit values.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_cell_types: int,
        head_config: DeconvHeadConfig | None = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.n_cell_types = n_cell_types

        if head_config is None:
            head_config = DeconvHeadConfig()
        self.head_config = head_config

        backbone_dim = backbone.config.hidden_size
        self.deconv_head = DeconvHead(
            input_dim=backbone_dim,
            n_cell_types=n_cell_types,
            hidden_dim=head_config.hidden_dim,
            n_layers=head_config.n_layers,
            dropout=head_config.dropout,
        )

        if head_config.cell_emb_style == "attn":
            self.pool_attn = nn.Sequential(
                nn.Linear(backbone_dim, 1),
            )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    def _pool(self, transformer_output, src_key_padding_mask):
        style = self.head_config.cell_emb_style
        if style == "cls":
            return transformer_output[:, 0, :]
        elif style == "mean":
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            return (transformer_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif style == "attn":
            attn_logits = self.pool_attn(transformer_output).squeeze(-1)
            attn_logits = attn_logits.masked_fill(src_key_padding_mask, float("-inf"))
            attn_weights = F.softmax(attn_logits, dim=-1)
            return (transformer_output * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown cell_emb_style: '{style}'")

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> dict:
        # Geneformer (BERT): input_ids only, attention_mask: 1 for real tokens
        attention_mask = (~src_key_padding_mask).long()
        outputs = self.backbone(
            input_ids=gene_ids,
            attention_mask=attention_mask,
        )
        transformer_output = outputs.last_hidden_state

        cell_emb = self._pool(transformer_output, src_key_padding_mask)
        logits = self.deconv_head(cell_emb)
        proportions = F.softmax(logits, dim=-1)

        return {"proportions": proportions, "cell_emb": cell_emb, "logits": logits}

    @torch.no_grad()
    def predict(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        return self.forward(gene_ids, values, src_key_padding_mask)["proportions"]
