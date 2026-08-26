from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained

from core.deconv.config import DeconvHeadConfig
from core.deconv.model import DeconvHead
from methods.scgpt.config import ScgptModelConfig


def create_scgpt_backbone(
    vocab,
    model_config: ScgptModelConfig,
    model_dir: str | None = None,
    device: str = "cuda",
) -> TransformerModel:
    """Create and optionally load a pretrained scGPT backbone."""
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_config.embsize,
        nhead=model_config.nhead,
        d_hid=model_config.d_hid,
        nlayers=model_config.nlayers,
        nlayers_cls=model_config.n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=model_config.dropout,
        pad_token=model_config.pad_token,
        pad_value=model_config.pad_value,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style=model_config.input_emb_style,
        n_input_bins=model_config.n_input_bins,
        cell_emb_style="cls",
        explicit_zero_prob=False,
        use_fast_transformer=model_config.use_fast_transformer,
        fast_transformer_backend="native",
        pre_norm=model_config.pre_norm,
    )

    if model_dir is not None:
        model_dir = Path(model_dir)
        model_file = model_dir / "best_model.pt"
        if model_file.exists():
            ckpt = torch.load(model_file, map_location=device)
            load_pretrained(model, ckpt, strict=False, verbose=True)
            model_dict = model.state_dict()
            matched = sum(
                1 for k in ckpt
                if k.replace("Wqkv.", "in_proj_") in model_dict
                and ckpt[k].shape == model_dict[k.replace("Wqkv.", "in_proj_")].shape
            )
            print(f"Loaded scGPT backbone from {model_file} "
                  f"({matched}/{len(ckpt)} params matched via load_pretrained)")

    model.to(device)
    return model


class ScgptDeconvModel(nn.Module):
    """scGPT backbone + deconvolution head.

    forward() signature: ``(gene_ids, values, src_key_padding_mask) -> dict``
    """

    def __init__(
        self,
        backbone: TransformerModel,
        n_cell_types: int,
        backbone_config: ScgptModelConfig,
        head_config: DeconvHeadConfig | None = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_config = backbone_config
        self.n_cell_types = n_cell_types

        if head_config is None:
            head_config = DeconvHeadConfig()
        self.head_config = head_config

        backbone_dim = backbone_config.embsize
        self.deconv_head = DeconvHead(
            input_dim=backbone_dim,
            n_cell_types=n_cell_types,
            hidden_dim=head_config.hidden_dim,
            n_layers=head_config.n_layers,
            dropout=head_config.dropout,
        )

        if head_config.cell_emb_style == "attn":
            self.pool_attn = nn.Sequential(
                nn.Linear(backbone_config.embsize, 1),
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
        with torch.set_grad_enabled(self.training):
            transformer_output = self.backbone._encode(
                gene_ids,
                values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )

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
