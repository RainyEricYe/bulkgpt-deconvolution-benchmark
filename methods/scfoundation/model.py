#!/usr/bin/env python3
"""scFoundation backbone and deconvolution head for standalone prediction.

Adapted from:
    .worktrees/scfoundation-test/src/bulkgpt/model/deconv_model.py

Contains the encoder-only MaeAutobin backbone, the learned-discretization
token embedding, and a lightweight MLP deconvolution head.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# scFoundation encoder components
# ---------------------------------------------------------------------------


class _AutoDiscretizationEmbedding2(nn.Module):
    """scFoundation's learned discretization embedding.

    Projects raw expression scalars -> bin weights (via MLP), then produces
    a weighted sum of bin embeddings.
    """

    def __init__(self, dim, max_seq_len, bin_num, bin_alpha,
                 mask_token_id=None, pad_token_id=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha

        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)

        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)

        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def forward(self, x, output_weight=0):
        x_mask_idx = (x == self.mask_token_id).nonzero()
        x_pad_idx = (x == self.pad_token_id).nonzero()

        x = self.mlp(x)
        x = self.leaky_relu(x)
        x_cross = self.mlp2(x)
        x = self.bin_alpha * x + x_cross
        weight = self.softmax(x)

        bin_num_idx = self.bin_num_idx.to(x.device)
        token_emb = self.emb(bin_num_idx)
        x = torch.matmul(weight, token_emb)

        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)
        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        if x_mask_idx.numel() > 0:
            x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = mask_token_emb.repeat(
                x_mask_idx.shape[0], 1
            )
        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        if x_pad_idx.numel() > 0:
            x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = pad_token_emb.repeat(
                x_pad_idx.shape[0], 1
            )
        if output_weight:
            return x, weight
        return x


class _ScFoundationTransformerEncoder(nn.Module):
    """Stack of ``nn.TransformerEncoderLayer`` matching scFoundation."""

    def __init__(self, dim, depth, heads, ff_mult=4, norm_first=False):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=dim * ff_mult,
                    batch_first=True,
                    norm_first=norm_first,
                )
            )
        self.transformer_encoder = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask=None):
        for mod in self.transformer_encoder:
            x = mod(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# scFoundation backbone (encoder-only)
# ---------------------------------------------------------------------------


class ScFoundationBackbone(nn.Module):
    """scFoundation MaeAutobin encoder wrapper.

    Loads only the encoder (token_emb + pos_emb + 12-layer Transformer)
    from the pretrained ``models.ckpt`` checkpoint.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()

        model_data = torch.load(ckpt_path, map_location="cpu")
        gene_data = model_data["gene"]
        cfg = gene_data["config"]
        model_type = cfg["model"]
        mc = cfg["model_config"][model_type]

        self.hidden_dim: int = mc["encoder"]["hidden_dim"]  # 768
        self.max_seq_len: int = mc["seq_len"]               # 19266
        self.bin_num: int = mc["bin_num"]
        self.bin_alpha: float = mc["bin_alpha"]
        self.pad_token_id: int = mc["pad_token_id"]
        self.mask_token_id: int = mc["mask_token_id"]

        enc_cfg = mc["encoder"]
        self.token_emb = _AutoDiscretizationEmbedding2(
            dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            bin_num=self.bin_num,
            bin_alpha=self.bin_alpha,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
        )
        self.pos_emb = nn.Embedding(self.max_seq_len + 1, self.hidden_dim)
        self.encoder = _ScFoundationTransformerEncoder(
            dim=enc_cfg["hidden_dim"],
            depth=enc_cfg["depth"],
            heads=enc_cfg["heads"],
            norm_first=enc_cfg.get("norm_first", False),
        )

        state_dict = gene_data["state_dict"]
        filtered = {}
        for k, v in state_dict.items():
            new_k = k.split("model.", 1)[1] if k.startswith("model.") else k
            if new_k.startswith(("token_emb.", "pos_emb.", "encoder.")):
                filtered[new_k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(
            f"scFoundation backbone loaded: {len(filtered)}/{len(state_dict)} keys "
            f"(decoder keys skipped: {len(state_dict) - len(filtered)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)})"
        )

        self.to(device)
        self.eval()

    def forward(self, gene_ids, values, src_key_padding_mask=None):
        x = self.token_emb(values.unsqueeze(-1), output_weight=0)
        x = x + self.pos_emb(gene_ids)
        pad_mask = (
            src_key_padding_mask
            if src_key_padding_mask is not None and src_key_padding_mask.any()
            else None
        )
        x = self.encoder(x, padding_mask=pad_mask)
        return x


# ---------------------------------------------------------------------------
# Deconvolution head
# ---------------------------------------------------------------------------


class DeconvHead(nn.Module):
    """MLP deconvolution head predicting cell type proportions."""

    def __init__(self, input_dim, n_cell_types, hidden_dim=256, n_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_cell_types))
        self.net = nn.Sequential(*layers)

    def forward(self, cell_emb):
        return self.net(cell_emb)


# ---------------------------------------------------------------------------
# Full deconvolution model (scFoundation backbone + head)
# ---------------------------------------------------------------------------


class ScFoundationDeconvModel(nn.Module):
    """scFoundation backbone + deconvolution head.

    Pooling (``cell_emb_style``):
    - ``cls``:  use the first position (gene A1BG) embedding
    - ``mean``: average over all non-padding position embeddings
    - ``attn``: learned attention-weighted pooling
    """

    def __init__(self, backbone, n_cell_types, cell_emb_style="cls",
                 hidden_dim=256, n_layers=2, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.cell_emb_style = cell_emb_style
        backbone_dim = backbone.hidden_dim

        self.deconv_head = DeconvHead(
            input_dim=backbone_dim,
            n_cell_types=n_cell_types,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        if cell_emb_style == "attn":
            self.pool_attn = nn.Sequential(
                nn.Linear(backbone_dim, 1),
            )

    def forward(self, gene_ids, values, src_key_padding_mask):
        transformer_output = self.backbone(
            gene_ids, values, src_key_padding_mask
        )

        if self.cell_emb_style == "cls":
            cell_emb = transformer_output[:, 0, :]
        elif self.cell_emb_style == "mean":
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            cell_emb = (transformer_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.cell_emb_style == "attn":
            attn_logits = self.pool_attn(transformer_output).squeeze(-1)
            attn_logits = attn_logits.masked_fill(src_key_padding_mask, float("-inf"))
            attn_weights = F.softmax(attn_logits, dim=-1)
            cell_emb = (transformer_output * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown cell_emb_style: '{self.cell_emb_style}'")

        logits = self.deconv_head(cell_emb)
        proportions = F.softmax(logits, dim=-1)

        return {"proportions": proportions, "cell_emb": cell_emb, "logits": logits}

    @torch.no_grad()
    def predict(self, gene_ids, values, src_key_padding_mask):
        self.eval()
        return self.forward(gene_ids, values, src_key_padding_mask)["proportions"]
