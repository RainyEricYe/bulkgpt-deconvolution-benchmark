# TAPE (scTAPE) — Autoencoder + GAN Deconvolution

TAPE uses an autoencoder trained on scRNA-seq reference data to learn
cell-type-specific expression profiles, then employs a GAN-like approach to
estimate proportions in bulk samples.

## Reference

Chen et al., "TAPE: a flexible and efficient deconvolution method for
transcriptomic data," BMC Bioinformatics (2022).

## Usage

```bash
# Pseudo-bulk benchmark
python run.py --config configs/default.yaml --mode train \
    --data data/Liver.h5ad --output-dir results/synthetic/Liver/tape

# Real bulk with ground truth
python run.py --config configs/default.yaml --mode predict \
    --h5 data/real_bulk/sweetwater.h5 \
    --ground-truth data/real_bulk/sweetwater_gt.csv \
    --output-dir results/real_bulk/sweetwater/tape

# GPU override
python run.py --config configs/default.yaml --mode train \
    --data data/Liver.h5ad --output-dir results --gpu
```

## Container

The method runs inside `containers/sif/tape.sif` which has the `sctape`
Python package pre-installed. Build via:
```bash
cd containers && apptainer build sif/tape.sif tape/tape.def
```

## I/O

| Mode | Input | Description |
|------|-------|-------------|
| train (--data) | .h5ad | scRNA-seq reference |
| predict (--h5) | .h5 | DeconBenchmark H5 |

## Output

- `proportions.csv` -- predicted proportions (samples x cell types)
- `metrics.json` -- Pearson, Spearman, RMSE, MAE
- `cell_types.json` -- cell type list
