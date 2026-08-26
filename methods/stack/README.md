# STACK — Embedding-based Deconvolution

STACK uses pre-trained cell embeddings (scVI-based) to perform
deconvolution by mixing cell embeddings into pseudo-bulk and training
a small MLP head to predict proportions.

Reference: https://github.com/arcinstitute/stack

## Setup

### 1. Install STACK

Three options (in order of preference):

**A. Download pinned version (recommended for reproducibility):**
```bash
bash data/prepare/download_external.sh --stack
```
This clones STACK at a pinned commit to `data/external/stack/`.

**B. Set environment variable:**
```bash
export STACK_SRC=/path/to/stack/src
```

**C. Pip install:**
```bash
pip install stack-embedding
```

### 2. Download pretrained weights

```bash
bash data/prepare/download_external.sh --weights
```
Or download manually from the [STACK releases page](https://github.com/arcinstitute/stack/releases).

## Usage

### Standalone scripts (recommended)

```bash
# Train deconvolution head from scRNA-seq reference
python methods/stack/train.py --config methods/stack/configs/default.yaml

# Predict cell-type proportions from bulk RNA-seq
python methods/stack/predict.py --config methods/stack/configs/default.yaml \
    --checkpoint results/stack/checkpoint/deconv_head.pt
```

### Unified script (backward compatible)

```bash
# Train only
python methods/stack/run.py --config configs/default.yaml --mode train

# Predict only
python methods/stack/run.py --config configs/default.yaml \
    --mode predict --checkpoint results/stack/checkpoint/deconv_head.pt

# Train + predict
python methods/stack/run.py --config configs/default.yaml --mode all
```

## Dependency Resolution Order

The scripts automatically resolve the STACK dependency in this order:
1. `pip install stack-embedding` (if installed)
2. `data/external/stack/stack/` (pinned download via `download_external.sh`)
3. `STACK_SRC` environment variable (fallback)

This ensures reproducibility: the pinned version in `data/external/` is used
when available, so model code changes won't break results.

## Input Format

- **scRNA-seq**: AnnData h5ad, raw counts in .X, cell-type column in .obs
- **Bulk**: AnnData h5ad, samples x genes

## Output

- `checkpoint/deconv_head.pt` — trained deconvolution head
- `checkpoint/cell_types.json` — cell type labels (ordered)
- `checkpoint/metadata.json` — training metadata
- `proportions.csv` — predicted proportions (samples x cell types)
