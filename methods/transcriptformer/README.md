# TranscriptFormer — Embedding-based Deconvolution

TranscriptFormer uses a transformer-based cell embedding model to
perform deconvolution. Cell embeddings are mixed into pseudo-bulk,
and a small MLP head is trained to predict proportions from the
mixed embedding.

Reference: https://github.com/suinleelab/TranscriptFormer

## Setup

### 1. Install TranscriptFormer

Three options (in order of preference):

**A. Download pinned version (recommended for reproducibility):**
```bash
bash data/prepare/download_external.sh --tf
```
This clones TranscriptFormer at a pinned commit to `data/external/transcriptformer/`.

**B. Set environment variable:**
```bash
export TF_REPO=/path/to/TranscriptFormer
```

### 2. Download pretrained weights

```bash
bash data/prepare/download_external.sh --weights
```
Or download manually from the [TranscriptFormer HuggingFace page](https://huggingface.co/suinleelab/TranscriptFormer).

### 3. Install Python dependencies

TranscriptFormer requires `omegaconf` and other packages not in the core
requirements:

```bash
pip install omegaconf anndata scipy
```

## Usage

### Standalone scripts (recommended)

```bash
# Train deconvolution head from scRNA-seq reference
python methods/transcriptformer/train.py --config methods/transcriptformer/configs/default.yaml

# Predict cell-type proportions from bulk RNA-seq
python methods/transcriptformer/predict.py --config methods/transcriptformer/configs/default.yaml \
    --checkpoint results/transcriptformer/checkpoint/deconv_head.pt
```

### Unified script (backward compatible)

```bash
# Train only
python methods/transcriptformer/run.py --config configs/default.yaml --mode train

# Predict only
python methods/transcriptformer/run.py --config configs/default.yaml \
    --mode predict --checkpoint results/transcriptformer/checkpoint/deconv_head.pt

# Train + predict
python methods/transcriptformer/run.py --config configs/default.yaml --mode all
```

## Dependency Resolution Order

The scripts automatically resolve the TranscriptFormer dependency in this order:
1. `pip install transcriptformer` (if installed)
2. `data/external/transcriptformer/repo/src/` (pinned download via `download_external.sh`)
3. `TF_REPO` environment variable (fallback)

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
