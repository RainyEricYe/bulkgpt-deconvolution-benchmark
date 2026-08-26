#!/usr/bin/env bash
# Prepare SDY67 test data: extract ground_truth CSV from the
# canonical DeconBenchmark H5 file.
# Run from tests/ directory: bash prepare_data.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$HERE/data"
SDY67_H5="$DATA_DIR/sdy67.h5"

mkdir -p "$DATA_DIR"

echo "[1/1] Extracting ground-truth proportions from H5..."
python3 -c "
import h5py
import pandas as pd

with h5py.File('$SDY67_H5', 'r') as f:
    gt = f['ground_truth/values'][:]       # (n_types, n_samples)
    cols = [s.decode() for s in f['ground_truth/colnames'][:]]
    rows = [t.decode() for t in f['ground_truth/rownames'][:]]
df = pd.DataFrame(gt.T, index=cols, columns=rows)
df.to_csv('$DATA_DIR/ground_truth.csv')
print(f'  Wrote {df.shape[0]} samples x {df.shape[1]} cell types')
print(f'  Columns: {list(df.columns)}')
"

echo "Done. Test data ready:"
ls -lh "$DATA_DIR/"
