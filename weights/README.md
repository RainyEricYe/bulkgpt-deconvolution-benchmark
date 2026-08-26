# Model Weights

This directory is **not** shipped with model binaries. Each backbone subdirectory
contains only small metadata files; the checkpoints themselves are downloaded
from their original public sources, or pointed to via environment variables.

All paths in `core/deconv/frozen_paths.py` / `core/deconv/frozen_eval.py`
default to `weights/{backbone}/...`, so dropping a downloaded checkpoint into the
matching directory is enough. Every path can also be overridden via its
eponymous environment variable (see `docs/environments.md`).

## Directory structure (after downloads)

```
weights/
├── stack/
│   ├── bc_large_aligned.ckpt           # STACK pretrained checkpoint
│   └── basecount_1000per_15000max.pkl  # HVG gene list
├── transcriptformer/
│   ├── repo/                           # TranscriptFormer source (git clone)
│   ├── tf_sapiens/                     # TranscriptFormer checkpoint dir
│   │   ├── config.json
│   │   ├── model_weights.pt
│   │   └── vocabs/
│   └── gene_cache.json                 # Gene symbol → Ensembl mapping
├── scfoundation/
│   ├── models.ckpt                     # scFoundation checkpoint
│   └── OS_scRNA_gene_index.19264.tsv   # 19,264-gene vocabulary
├── scgpt/
│   └── whole-human/                    # scGPT pretrained model dir
│       ├── best_model.pt
│       └── vocab.json
├── geneformer/
│   └── default/                        # Geneformer HF model dir
└── bulkformer/
    └── source/                         # BulkFormer full repo (git clone)
        ├── model/BulkFormer_147M.pt
        ├── data/{G_tcga.pt, G_tcga_weight.pt, esm2_feature_concat.pt, bulkformer_gene_info.csv}
        └── utils/{BulkFormer.py, BulkFormer_block.py, Rope.py}
```

## Download

Run the helper script from the repository root:

```bash
bash weights/download_weights.sh
```

It fetches every checkpoint whose default location is missing. Individual
backbones:

| Backbone | Instructions |
|----------|--------------|
| **scGPT** | `git clone https://huggingface.co/scGPT/whole-human weights/scgpt/whole-human` (HF repo contains `model.pt`, `vocab.json`, `args.json`) |
| **Geneformer** | `git clone https://huggingface.co/ctheodoris/Geneformer weights/geneformer/default` — then symlink the gene-dictionary pickles the package expects (see env vars below) |
| **scFoundation** | Checkpoint + gene index via the [scFoundation](https://github.com/bowang-lab/scFoundation) release page / Baidu & Dropbox links in its repo; place `models.ckpt` and `OS_scRNA_gene_index.19264.tsv` into `weights/scfoundation/` |
| **STACK** | From the [ShiLab-Bioinformatics/Stack](https://github.com/ShiLab-Bioinformatics/Stack) HF space (`arcinstitute/Stack-Large-Aligned`): `bc_large_aligned.ckpt` and `basecount_1000per_15000max.pkl` |
| **TranscriptFormer** | `git clone … weights/transcriptformer/repo`; `git clone https://huggingface.co/…/tf_sapiens weights/transcriptformer/tf_sapiens` (checkpoint dir), plus `gene_cache.json` |
| **BulkFormer** | `git clone … weights/bulkformer/source` (source repo containing model + data + utils) |

> The `data/` and `checkpoints/` files referenced by `frozen_paths.py` are part
> of the upstream repositories; the download script only populates the paths
> above. If you already have a local copy elsewhere, override the env var
> instead of re-downloading.

## Environment variables

Every constant in `core/deconv/frozen_paths.py` accepts an env var override:

| Env var | Default |
|---------|---------|
| `SCGPT_MODEL_DIR` | `weights/scgpt/whole-human` |
| `GENEFORMER_MODEL_DIR` | `weights/geneformer/default` |
| `GENEFORMER_TOKEN_DICT` | `~/miniconda3/envs/geneformer/…/token_dictionary_gc30M.pkl` |
| `GENEFORMER_NAME_TO_ENSEMBL` | `~/miniconda3/envs/geneformer/…/gene_name_id_dict_gc30M.pkl` |
| `SCFOUNDATION_CKPT` | `weights/scfoundation/models.ckpt` |
| `SCFOUNDATION_GENE_INDEX` | `weights/scfoundation/OS_scRNA_gene_index.19264.tsv` |
| `TF_CKPT_DIR` | `weights/transcriptformer/tf_sapiens` |
| `TF_REPO_SRC` | `weights/transcriptformer/repo/src` |
| `TF_GENE_CACHE` | `weights/transcriptformer/gene_cache.json` |
| `STACK_CHECKPOINT` | `weights/stack/bc_large_aligned.ckpt` |
| `STACK_GENELIST` | `weights/stack/basecount_1000per_15000max.pkl` |
| `SCPEFT_DIR` | `~/claude/scPEFT` |

The Geneformer `token_dictionary` and `gene_name_id_dict` pickles are installed
with the `geneformer` Python package — set the two env vars to their location
inside your conda env if not at the default path:

```bash
export GENEFORMER_TOKEN_DICT=$CONDA_PREFIX/lib/python3.10/site-packages/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl
export GENEFORMER_NAME_TO_ENSEMBL=$CONDA_PREFIX/lib/python3.10/site-packages/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl
```