# Sweetwater — Interpretable VAE Deconvolution

Sweetwater uses a scArches-style variational autoencoder trained on
scRNA-seq reference to learn cell-type-specific expression profiles
and estimate proportions in bulk samples.

## Reference

Amodio et al., "Sweetwater: an interpretable autoencoder for
single-cell deconvolution," NAR Genomics and Bioinformatics (2025).

## Usage

```bash
# Pseudo-bulk benchmark
python run.py --config configs/default.yaml --mode train \
    --data data/Liver.h5ad --output-dir results/synthetic/Liver/sweetwater

# Real bulk with ground truth
python run.py --config configs/default.yaml --mode predict \
    --h5 data/real_bulk/sweetwater.h5 \
    --ground-truth data/real_bulk/sweetwater_gt.csv \
    --output-dir results/real_bulk/sweetwater/sweetwater

# GPU override
python run.py --config configs/default.yaml --mode train \
    --data data/Liver.h5ad --output-dir results --gpu
```

## Container

Runs inside `containers/sif/sweetwater.sif` (Python PyTorch).

## I/O

| Mode | Input | Description |
|------|-------|-------------|
| train (--data) | .h5ad | scRNA-seq reference |
| predict (--h5) | .h5 | DeconBenchmark H5 |

## Output

- `proportions.csv` -- predicted proportions
- `metrics.json` -- Pearson, Spearman, RMSE, MAE
- `cell_types.json` -- cell type list
