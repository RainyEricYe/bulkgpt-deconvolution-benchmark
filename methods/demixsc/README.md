# DeMixSC — Weighted NNLS with Benchmark Alignment

DeMixSC performs weighted non-negative least-squares deconvolution with
platform-effect correction between scRNA-seq and bulk RNA-seq data.

## Reference

Song et al., "DeMixSC: a robust deconvolution method for dissecting
tissue complexity from bulk expression," Nature Communications (2024).

## Usage

```bash
# Pseudo-bulk benchmark
python run.py --config configs/default.yaml --mode predict \
    --data data/Liver.h5ad --output-dir results/synthetic/Liver/demixsc

# Real bulk with ground truth
python run.py --config configs/default.yaml --mode predict \
    --h5 data/real_bulk/sweetwater.h5 \
    --ground-truth data/real_bulk/sweetwater_gt.csv \
    --output-dir results/real_bulk/sweetwater/demixsc
```

## Container

Runs inside `containers/sif/demixsc.sif` (R package DeMixSC).

## I/O

| Mode | Input | Description |
|------|-------|-------------|
| predict (--data) | .h5ad | scRNA-seq reference |
| predict (--h5) | .h5 | DeconBenchmark H5 |

## Output

- `proportions.csv` -- predicted proportions
- `metrics.json` -- Pearson, Spearman, RMSE, MAE
- `cell_types.json` -- cell type list
