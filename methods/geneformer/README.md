# Geneformer Deconvolution Method

Fine-tune the Geneformer foundation model (104M parameters, V2) for bulk
RNA-seq deconvolution.  The model combines a Geneformer backbone with a
lightweight DeconvHead (2-layer MLP) that predicts cell-type proportions
from bulk expression profiles.

## How It Works

1. **Map gene symbols to Ensembl IDs** -- Geneformer's vocabulary uses
   Ensembl gene IDs.  If the input reference uses gene symbols, they are
   automatically mapped via mygene.
2. **Rank-value encoding** -- unlike the scGPT variant which uses binned
   category encoding, Geneformer uses rank-value encoding where genes are
   sorted by expression level per sample and assigned rank-based values.
3. **Generate pseudo-bulk samples** -- the scRNA-seq reference is mixed
   using Dirichlet-sampled proportions to create synthetic bulk profiles
   with known cell-type fractions.
4. **Fine-tune Geneformer** -- the backbone and head are jointly optimised
   with MSE + KL divergence loss.
5. **Predict** -- a trained checkpoint is applied to held-out pseudo-bulk
   (validation) or to real bulk samples.

## Usage

### Train (standalone script)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python train.py --config configs/ft.yaml
```

### Predict (standalone script)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python predict.py --config configs/ft.yaml \
    --checkpoint checkpoints/geneformer/ft/best_model.pt
```

### Legacy usage (original unified entry point)

For backward compatibility, the original ``run.py`` is still available:

```bash
python run.py --config configs/ft.yaml --mode train
python run.py --config configs/ft.yaml --mode predict \
    --checkpoint checkpoints/geneformer/ft/best_model.pt
```

### Custom dataset

Edit the `dataset` section of the config:

```yaml
dataset:
  sc_ref: /path/to/your/reference.h5ad
  n_hvg: 800
```

The scRNA reference must be an AnnData (.h5ad) file with:
- `.X` = raw counts (cells x genes)
- `.obs["cell_type"]` = cell-type labels
- `.var_names` = gene symbols or Ensembl IDs (symbols are auto-mapped)

## Configuration files

| Config | Backbone | Pooling | LR | Expected Pearson |
|--------|----------|---------|----|-----------------:|
| `ft.yaml` | Fine-tuned | CLS | 7e-5 | ~0.85-0.95 |
| `fz.yaml` | Frozen | CLS | 1e-3 | ~0.25-0.40 |

## Key differences from scGPT

| Aspect | scGPT | Geneformer |
|--------|-------|------------|
| Parameters | 30M (whole-human) | 104M (V2) |
| Gene IDs | Gene symbols | Ensembl IDs |
| Encoding | Binned (51 categories) | Rank-value |
| Vocabulary source | vocab.json | TOKEN_DICTIONARY_FILE (pickle) |
| Pretrained model path | pretrained_models/whole-human | HuggingFace ctheodoris/Geneformer |

## Pretrained model

Download the Geneformer model from HuggingFace:

```bash
git clone https://huggingface.co/ctheodoris/Geneformer
```

Set `paths.pretrained_model` in the config to point to the downloaded
directory containing `pytorch_model.bin`, `config.json`, and the tokenizer.

## Citation

```bibtex
@article{theodoris2023geneformer,
  title={Transfer learning enables predictions in network biology},
  author={Theodoris, Christina V and Xiao, Ling and Chopra, Anant and
          Chaffin, Mark D and Al Sayed, Zeina R and Hill, Matthew C and
          Mantineo, Helene and Brydon, Elizabeth M and Zeng, Zexian and
          Liu, X Shirley and others},
  journal={Nature},
  volume={618},
  number={7965},
  pages={616--624},
  year={2023},
  publisher={Nature Publishing Group}
}
```

For the deconvolution adaptation, see the accompanying paper
(multi-FM deconvolution benchmark).
