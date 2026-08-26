# scGPT Deconvolution Method

Fine-tune the scGPT foundation model for bulk RNA-seq deconvolution.  The
model combines a scGPT backbone with a lightweight DeconvHead (2-layer MLP)
that predicts cell-type proportions from bulk expression profiles.

## How It Works

1. **Generate pseudo-bulk samples** -- the scRNA-seq reference is mixed using
   Dirichlet-sampled proportions to create synthetic bulk profiles with known
   cell-type fractions.
2. **Fine-tune scGPT** -- the backbone and head are jointly optimised with
   MSE + KL divergence loss (lr = 7e-5, 30 epochs).
3. **Predict** -- a trained checkpoint is applied to held-out pseudo-bulk
   (validation) or to real bulk samples.

## Usage

### Train

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Unified entry point (backward compatible)
python run.py --config configs/ft.yaml --mode train

# Direct entry point (new)
python train.py --config configs/ft.yaml
```

### Predict (evaluate a trained checkpoint)

```bash
# Unified entry point (backward compatible)
python run.py --config configs/ft.yaml --mode predict \
    --checkpoint checkpoints/ft/best_model.pt

# Direct entry point (new)
python predict.py --config configs/ft.yaml \
    --checkpoint checkpoints/ft/best_model.pt
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
- `.var_names` = gene symbols or Ensembl IDs

## Configuration files

| Config | Backbone | Pooling | Binned | LR | Expected Pearson |
|--------|----------|---------|--------|----|-----------------:|
| `ft.yaml` | Fine-tuned | Mean | Yes | 7e-5 | 0.939 (10-dataset avg) |
| `fz.yaml` | Frozen | Mean | Yes | 1e-3 | ~0.30 |
| `ft-cls-pool.yaml` | Fine-tuned | CLS | Yes | 7e-5 | 0.930 (10-dataset avg) |

## Pretrained model

Download the scGPT whole-human model from:

https://huggingface.co/ctheodoris/Geneformer

Set `paths.pretrained_model` in the config to point to the downloaded
directory containing `vocab.json` and PyTorch model weights.

## Citation

```bibtex
@article{cui2024scgpt,
  title={scGPT: toward building a foundation model for single-cell
         multi-omics using generative AI},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and others},
  journal={Nature Methods},
  year={2024}
}
```

For the deconvolution adaptation described here, see the accompanying paper
(multi-FM deconvolution benchmark).
