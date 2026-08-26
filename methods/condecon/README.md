# ConDecon — Clustering-Independent Deconvolution

ConDecon uses rank correlations between single-cell and bulk expression
to estimate cell abundance distributions without requiring hard cluster
assignments from the scRNA-seq reference.

## Reference

Camara et al., "ConDecon: clustering-independent deconvolution of bulk
transcriptomic data," BMC Bioinformatics (2023).

## Usage

```bash
# Pseudo-bulk benchmark
python run.py --config configs/default.yaml --mode predict \
    --data data/Liver.h5ad --output-dir results/synthetic/Liver/condecon

# Real bulk with ground truth
python run.py --config configs/default.yaml --mode predict \
    --h5 data/real_bulk/sweetwater.h5 \
    --ground-truth data/real_bulk/sweetwater_gt.csv \
    --output-dir results/real_bulk/sweetwater/condecon
```

## Container

Runs inside `containers/sif/condecon.sif` (R package ConDecon).

## I/O

| Mode | Input | Description |
|------|-------|-------------|
| predict (--data) | .h5ad | scRNA-seq reference |
| predict (--h5) | .h5 | DeconBenchmark H5 |

## Output

- `proportions.csv` -- predicted proportions
- `metrics.json` -- Pearson, Spearman, RMSE, MAE
- `cell_types.json` -- cell type list
