from dataclasses import dataclass


@dataclass
class ScgptModelConfig:
    """scGPT backbone configuration (must match pretrained checkpoint)."""

    embsize: int = 512
    nhead: int = 8
    d_hid: int = 512
    nlayers: int = 12
    n_layers_cls: int = 3
    dropout: float = 0.2
    use_fast_transformer: bool = False
    pre_norm: bool = False
    n_input_bins: int | None = None  # 51 for binned, None for continuous
    input_emb_style: str = "continuous"  # "category" for binned
    cell_emb_style: str = "cls"  # "cls", "mean", or "attn"
    max_seq_len: int = 1201
    pad_token: str = "<pad>"
    pad_value: int = 0
    mask_value: int = -1
