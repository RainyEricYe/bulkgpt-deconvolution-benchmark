# SQUID — Dampened Weighted Least-Squares Deconvolution

SQUID estimates cell-type proportions using dampened weighted least-squares
with a signature matrix derived from scRNA-seq reference data.

## Reference

Najafpanah et al., "SQUID: decoQvolving cell-type heterogeneity in spatial
transcriptomics and bulk RNA-seq," NAR Genomics and Bioinformatics (2024).

## Usage

```bash
# Pseudo-bulk benchmark
python run.py --config configs/default.yaml --mode predict \
    --data data/Liver.h5ad --output-dir results/synthetic/Liver/squid

# Real bulk with ground truth
python run.py --config configs/default.yaml --mode predict \
    --h5 data/real_bulk/sweetwater.h5 \
    --ground-truth data/real_bulk/sweetwater_gt.csv \
    --output-dir results/real_bulk/sweetwater/squid
```

## Container

Runs inside `containers/sif/squid.sif` (R package SQUID).

## I/O

| Mode | Input | Description |
|------|-------|-------------|
| predict (--data) | .h5ad | scRNA-seq reference |
| predict (--h5) | .h5 | DeconBenchmark H5 |

## Output

- `proportions.csv` -- predicted proportions
- `metrics.json` -- Pearson, Spearman, RMSE, MAE
- `cell_types.json` -- cell type list
