# CIBERSORTx (Python NuSVR) — Bulk RNA-seq Deconvolution

Pure-Python reimplementation of the core CIBERSORT algorithm using
scikit-learn's NuSVR with linear kernel. No Docker or Stanford token required.

Reference: Newman et al., "Robust enumeration of cell subsets from tissue
expression profiles," Nature Methods (2015).

## Usage

```bash
# Train (build signature matrix) + predict
python run.py --config configs/default.yaml --mode all

# Predict from pre-built signature matrix
python run.py --config configs/default.yaml --mode predict

# CLI overrides
python run.py --config configs/default.yaml --sc-ref /path/to/ref.h5ad --bulk /path/to/bulk.tsv
```

## Input Format

- **scRNA-seq**: AnnData h5ad with cell-type column in .obs
- **Bulk**: CSV/TSV with genes as index, samples as columns

## Output

- `proportions.csv` — predicted proportions (samples x cell types)
- `signature_matrix.csv` — built signature matrix (genes x cell types)
