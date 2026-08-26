# pca_ridge

PCA + RidgeCV baseline for real-bulk split evaluation (Mode B). Reduces bulk
expression to principal components and evaluates per-cell-type RidgeCV on a
train/test split of the real bulk samples. No single-cell reference required.
PCA is fit without StandardScaler, `random_state=42`.

## Quick start

```bash
python methods/pca_ridge/run.py --h5 <input.h5> --output-dir <out> --ground-truth <gt.csv>
```

## Reference

Baseline for the BulkFormer-style split evaluation; see `methods/pca_ridge/run.py`.
