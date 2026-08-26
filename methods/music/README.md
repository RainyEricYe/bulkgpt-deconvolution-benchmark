# MuSic — Multi-subject Single-cell Deconvolution

MuSic (Multi-subject single-cell deconvolution) is an R package that
uses multi-subject weighting from donor_id for improved accuracy in
bulk RNA-seq deconvolution.

Reference: Wang et al., "Bulk tissue cell type deconvolution with
multi-subject single-cell expression reference," Nature Communications (2022).

## Requirements

- Apptainer/Singularity for container execution
- MuSic SIF image at the path specified in config

## Usage

```bash
# Download MuSic SIF (one-time)
apptainer pull containers/sif/music.sif docker://...  # obtain from registry

# Run deconvolution
python run.py --config configs/default.yaml --mode predict

# Skip container for testing
python run.py --config configs/default.yaml --skip-container

# CLI overrides
python run.py --config configs/default.yaml --sc-ref /path/to/ref.h5ad --bulk /path/to/bulk.h5ad
```

## Input Format

- **scRNA-seq**: AnnData h5ad with celltype_col and donor_col in .obs
- **Bulk**: AnnData h5ad, samples x genes

## Output

- `proportions.csv` — predicted proportions (samples x cell types)
