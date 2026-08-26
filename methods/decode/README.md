# DECODE (MBdeconv) — Bulk RNA-seq Deconvolution

DECODE uses an MBdeconv model with multi-head attention to reconstruct
bulk expression from cell-type proportions, trained on pseudo-bulk
mixtures generated from a scRNA-seq reference.

## Reference

Chen et al., "DECODE: a deep-learning method for cell-type deconvolution
of bulk expression," Nature Communications (2023).

## Usage

```bash
# Train + predict
python run.py --config configs/default.yaml --mode all

# Train only
python run.py --config configs/default.yaml --mode train

# Predict from checkpoint
python run.py --config configs/default.yaml --mode predict

# CLI overrides
python run.py --config configs/default.yaml --epochs 500 --gpu
```

## Input Format

- **scRNA-seq reference**: AnnData h5ad, `.X` = raw counts, `.obs` with cell-type column
- **Bulk**: AnnData h5ad, samples x genes

## Output

- `checkpoint/mbdeconv.pt` — trained model weights
- `checkpoint/metadata.json` — metrics
- `proportions.csv` — predicted proportions
