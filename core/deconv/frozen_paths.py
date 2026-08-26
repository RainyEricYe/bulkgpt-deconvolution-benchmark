"""Path constants for frozen backbone weights and model sources.

All paths can be overridden via their eponymous environment variables.
Centralised here so frozen_eval.py stays focused on evaluation logic.

Where a default is unavailable (e.g. a weight that must be downloaded from a
public registry, or a geneformer token dictionary that ships inside the
installed package), the fallback is resolved at import time from either the
environment or a standard location under the local checkout (``weights/``).
"""

from __future__ import annotations

import os
from pathlib import Path

_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "weights"

_SCPEFT_DIR = Path(os.environ.get("SCPEFT_DIR", os.path.expanduser("~/claude/scPEFT")))
_SCGPT_MODEL_DIR = Path(os.environ.get("SCGPT_MODEL_DIR", str(_WEIGHTS_DIR / "scgpt/whole-human")))
_GENEFORMER_MODEL_DIR = Path(os.environ.get("GENEFORMER_MODEL_DIR", str(_WEIGHTS_DIR / "geneformer/default")))
_SCFOUNDATION_CKPT = Path(os.environ.get("SCFOUNDATION_CKPT", str(_WEIGHTS_DIR / "scfoundation/models.ckpt")))
_SCFOUNDATION_GENE_INDEX = Path(os.environ.get(
    "SCFOUNDATION_GENE_INDEX", str(_WEIGHTS_DIR / "scfoundation/OS_scRNA_gene_index.19264.tsv")
))
_TF_CKPT_DIR = Path(os.environ.get("TF_CKPT_DIR", str(_WEIGHTS_DIR / "transcriptformer/tf_sapiens")))
_TF_REPO_SRC = os.environ.get(
    "TF_REPO_SRC",
    str(_WEIGHTS_DIR / "transcriptformer" / "repo" / "src"),
)
# The TranscriptFormer source lives in a separate upstream repo. If it is not
# present under weights/, point TF_REPO_SRC at the local clone via the
# TF_REPO_SRC environment variable (see weights/README.md).
if not Path(_TF_REPO_SRC).exists():
    _TF_REPO_SRC = os.environ.get("TF_REPO_SRC", str(_WEIGHTS_DIR / "transcriptformer" / "repo" / "src"))
_TF_GENE_CACHE = Path(os.environ.get("TF_GENE_CACHE", str(_WEIGHTS_DIR / "transcriptformer/gene_cache.json")))
_STACK_CHECKPOINT = os.environ.get("STACK_CHECKPOINT", str(_WEIGHTS_DIR / "stack/bc_large_aligned.ckpt"))
_STACK_GENELIST = os.environ.get("STACK_GENELIST", str(_WEIGHTS_DIR / "stack/basecount_1000per_15000max.pkl"))
_GENEFORMER_TOKEN_DICT = Path(os.environ.get(
    "GENEFORMER_TOKEN_DICT",
    os.path.expanduser(
        "~/miniconda3/envs/geneformer/lib/python3.10/"
        "site-packages/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"
    ),
))
_GENEFORMER_NAME_TO_ENSEMBL = Path(os.environ.get(
    "GENEFORMER_NAME_TO_ENSEMBL",
    os.path.expanduser(
        "~/miniconda3/envs/geneformer/lib/python3.10/"
        "site-packages/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl"
    ),
))
_DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
