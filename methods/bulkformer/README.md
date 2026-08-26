# BulkFormer — Embedding-based Deconvolution

Uses pretrained **BulkFormer-147M** (GCN+Performer, 20,010 genes, pretrained on 500K+ bulk RNA-seq profiles) to extract sample embeddings, then trains a lightweight MLP deconvolution head or uses RidgeCV.

## Quick Start

### Train (pseudo-bulk)
```bash
cd to_publish
python methods/bulkformer/train.py --config methods/bulkformer/configs/default.yaml
```

### Predict (real bulk)
```bash
# MLP head mode
python methods/bulkformer/predict.py \
  --config methods/bulkformer/configs/default.yaml \
  --checkpoint checkpoints/bulkformer/default/best_model.pt

# RidgeCV mode (no training required)
python methods/bulkformer/predict.py \
  --config methods/bulkformer/configs/default.yaml \
  --mode ridge --dataset sdy67
```

### Unified dispatcher
```bash
python methods/bulkformer/run.py --config configs/default.yaml --mode train
python methods/bulkformer/run.py --config configs/default.yaml --mode predict --checkpoint best_model.pt
```

## Environment

- Conda env: `bulkformer`
- Requires BulkFormer source at `BULKFORMER_DIR` (default: `~/data/public/BulkFormer/`)
- GPU recommended (model + graph ≈ 4 GB VRAM)

## Output

Training saves checkpoint + metadata to `checkpoints/bulkformer/default/`.
Prediction saves 4 files per dataset:
- `proportions.csv` — predicted cell-type fractions
- `metrics.json` — DeconBenchmark suite (MAE, SCorr, CCorr, MAECorr, Pearson, RMSE, Wilcoxon)
- `ridge_metrics.json` — RidgeCV-specific (best_alpha, per-type r/rmse)
- `metadata.json` — resource usage (time, GPU memory, env info)
