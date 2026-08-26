# Conda Environment Design

## Architecture: 2-Tier

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: Evaluation layer  (conda env: bulkgpt)             │
│  ─────────────────────────────────────                       │
│  torch, numpy, scipy, pandas, scikit-learn                   │
│  anndata, h5py, pyyaml, scanpy                               │
│  core/metrics.py, core/deconv/frozen_eval.py                 │
│  scripts/evaluate.py, scripts/eval_real_bulk_ridge.py        │
│                                                              │
│  Runs: RidgeCV evaluation, DeconBenchmark metrics, data I/O  │
│  Needs: NO backbone-specific packages                        │
└──────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Layer 2:    │ │  Layer 2:    │ │  Layer 2:    │  ... x6
│  Encode      │ │  Encode      │ │  Encode      │
│  stack       │ │  geneformer  │ │  bulkformer   │
│  env: stack  │ │  env:gene-   │ │  env:         │
│              │ │  former      │ │  bulkformer   │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Why 6 Separate Environments?

| Backbone | Key Dep | Why Separate? |
|----------|---------|---------------|
| scGPT | `flash-attn` | Requires specific CUDA/torch; conflicts with torch_geometric |
| BulkFormer | `torch_geometric` | GCNConv needs specific PyTorch+PyG version combo |
| Geneformer | `transformers` | May pin specific torch version |
| STACK | `stack` | Standalone pip package; compatible with base |
| TranscriptFormer | `omegaconf` | Lightweight; no conflict but separate to isolate |
| scFoundation | (none extra) | Uses to_publish code; could run in bulkgpt |

**Conclusion: 6 independent conda envs is the correct design.**

## Backbone → Conda Env Mapping

| Backbone key | Conda env | Env file | VRAM |
|-------------|-----------|----------|------|
| `stack` | `stack` | `methods/stack/environment.yml` | ~2 GB |
| `transcriptformer` | `TranscriptFormer` | `methods/transcriptformer/environment.yml` | ~4 GB |
| `scfoundation` | `scfoundation` | `methods/scfoundation/environment.yml` | ~6 GB |
| `scgpt` | `bulkgpt` | (built into bulkgpt) | ~3 GB |
| `geneformer` | `geneformer` | `methods/geneformer/environment.yml` | ~3 GB |
| `bulkformer` | `bulkformer` | `methods/bulkformer/environment.yml` | ~4 GB |

## Reproduction

### 1. Base env
```bash
conda env create -f environment.yml          # bulkgpt
conda activate bulkgpt
```

### 2. Backbone envs (on demand)
```bash
conda env create -f methods/stack/environment.yml
conda env create -f methods/geneformer/environment.yml
# ... etc for each backbone needed
```

### 3. Weights
```bash
# See weights/README.md — symlink checkpoints into weights/{backbone}/
```

### 4. Run
```bash
# Single backbone (activate its env first)
conda activate stack
python scripts/eval_real_bulk_ridge.py --backbone stack --dataset sdy67

# All (auto-switches envs)
bash scripts/eval_all_backbones.sh --dataset all
```

## Environment Variable Overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `STACK_CHECKPOINT` | `weights/stack/bc_large_aligned.ckpt` | STACK ckpt |
| `TF_CKPT_DIR` | `weights/transcriptformer/tf_sapiens` | TF dir |
| `SCFOUNDATION_CKPT` | `weights/scfoundation/models.ckpt` | scFoundation ckpt |
| `SCGPT_MODEL_DIR` | `weights/scgpt/whole-human` | scGPT dir |
| `GENEFORMER_MODEL_DIR` | `weights/geneformer/default` | Geneformer HF dir |
| `BULKFORMER_DIR` | `weights/bulkformer/source` | BulkFormer source |
| `SCPEFT_DIR` | `~/claude/scPEFT` | scPEFT repo |

## Fault Tolerance

- **Lazy import**: Each `ENCODE_FN[key]` only imports on first call
- **Isolated failure**: One backbone failing doesn't affect others
- **No-GPU path**: RidgeCV runs on CPU; encode phase only needs GPU
- **Missing env**: `eval_all_backbones.sh` auto-skips uninstalled envs
