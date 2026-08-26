from dataclasses import dataclass


@dataclass
class GeneformerModelConfig:
    """Geneformer backbone configuration.

    Minimal — HuggingFace ``AutoModel`` handles most settings.
    """

    cell_emb_style: str = "cls"  # "cls", "mean", or "attn"
    max_seq_len: int = 2048
    pad_token: str = "<pad>"
    pad_value: int = 0
