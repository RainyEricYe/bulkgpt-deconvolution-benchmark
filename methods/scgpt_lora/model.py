"""scGPT model with LoRA adapters for deconvolution.

Uses scPEFT's modified scGPT (via PYTHONPATH) which provides:
- MultiheadAttentionLoRAImpl — LoRA-patched multihead attention
- TransformerModel with peft_config support
- LinearDeconvHead from scgpt.tasks.deconv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

# scPEFT path resolution (lazy: only needed to *build* the LoRA model).
# scPEFT is an external dependency (github.com/.../scPEFT); importing this
# module for pure functions (e.g. count_trainable) must not require it.
_SCPEFT_DIR = Path(__file__).resolve().parent.parent.parent / "scPEFT"
if not _SCPEFT_DIR.exists():
    _SCPEFT_DIR = Path("repo/scpeft")
if _SCPEFT_DIR.exists():
    sys.path.insert(0, str(_SCPEFT_DIR / "repo"))


def build_model(
    peft_config: dict[str, Any],
    model_dir: str | Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.nn.Module:
    """Build scGPT TransformerModel with LoRA adapters and load pretrained weights."""
    if not _SCPEFT_DIR.exists():
        raise FileNotFoundError(
            "scPEFT not found; clone it to repo/scpeft or set SCPEFT_DIR "
            "(see methods/scgpt_lora/README.md) before building the LoRA model."
        )
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    model_dir = Path(model_dir)
    # NB: from_file is slow (~184s) due to torchtext 0.18 insert_token regression
    vocab = GeneVocab.from_file(str(model_dir / "vocab.json"))

    config = {
        "ntoken": len(vocab),
        "d_model": 512,
        "nhead": 8,
        "nlayers": 12,
        "nlayers_cls": 3,
        "d_hid": 512,
        "vocab": vocab,
        "dropout": 0.0,
        "pad_token": "<pad>",
        "pad_value": -2,
        "do_mvc": False,
        "do_dab": False,
        "use_batch_labels": False,
        "domain_spec_batchnorm": False,
        "explicit_zero_prob": False,
        "use_fast_transformer": False,
        "pre_norm": False,
        "peft_config": dict(peft_config),
    }

    model = TransformerModel(**config)
    ckpt = torch.load(
        str(model_dir / "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    load_pretrained(model, ckpt, verbose=False)
    model.to(device)
    model.train()
    freeze_non_lora(model)

    return model


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_non_lora(model: torch.nn.Module) -> None:
    for name, p in model.named_parameters():
        if "lora" not in name.lower():
            p.requires_grad = False
