# scFoundation Deconvolution Method

Fine-tune (head-only) the scFoundation foundation model (100M parameters)
for bulk RNA-seq deconvolution.  The backbone MUST remain frozen due to
memory constraints (OOM on 80GB H100 when unfrozen).

## How It Works

1. **Gene alignment** -- input gene symbols (or Ensembl IDs) are mapped to
   scFoundation's fixed 19264-gene order using a precomputed mapping file.
   Genes not in the scFoundation index are zero-filled (no error).
2. **Position indices as gene IDs** -- unlike scGPT (vocabulary-based) or
   Geneformer (Ensembl-ID tokens), scFoundation uses positional indices
   0 through 19263 as gene identifiers.
3. **Raw expression values** -- log1p-CPM values are passed directly (no
   binned or rank-value encoding).
4. **Generate pseudo-bulk samples** -- the scRNA-seq reference is mixed
   using Dirichlet-sampled proportions.
5. **Train DeconvHead** -- only the lightweight DeconvHead MLP is trained;
   the scFoundation backbone weights are frozen.
6. **Evaluate** -- metrics are computed on the held-out validation split.

## Usage

### Train (includes evaluation)

```bash
# Activate the scFoundation conda environment first
conda activate scfoundation
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Using the unified dispatcher (original interface)
python run.py --config configs/frozen.yaml --mode train

# Using the standalone train script (equivalent)
python train.py --config configs/frozen.yaml
```

### Predict (re-run training + evaluation with same config)

```bash
python run.py --config configs/frozen.yaml --mode predict

# Using the standalone predict script (equivalent)
python predict.py --config configs/frozen.yaml
```

Note: All four forms above are equivalent.  scFoundation's train.py
performs end-to-end training + evaluation in a single run; there is no
dedicated evaluation script.  The trained checkpoint and evaluation
results are saved to the checkpoint directory.

### Custom dataset

Edit the `dataset` section of the config:

```yaml
dataset:
  sc_ref: /path/to/your/reference.h5ad
```

## Configuration

| Config | Backbone | Pooling | Batch | LR | Expected Pearson |
|--------|----------|---------|-------|----|-----------------:|
| `frozen.yaml` | Frozen | Mean | 4 | 1e-3 | ~0.30-0.50 |

## Key differences from scGPT / Geneformer

| Aspect | scGPT | Geneformer | scFoundation |
|--------|-------|------------|--------------|
| Parameters | 30M | 104M | 100M |
| Gene dimension | HVG-selected (~581-1200) | HVG-selected | **Fixed 19264** |
| Gene IDs | Vocab token IDs | Ensembl token IDs | **Position indices** |
| Encoding | Binned categories | Rank-value | **Raw log1p-CPM** |
| Backbone | Fine-tuned | Fine-tuned | **Frozen only** |
| Batch size | 64 | 64 | **4** |

## Worktree setup

scFoundation requires a separate git worktree with modified bulkgpt source
code.  To set up:

```bash
# Create the worktree
git worktree add .worktrees/scfoundation-test scfoundation-feature

# Copy or symlink the pretrained model
ln -s /path/to/scfoundation-pretrained .worktrees/scfoundation-test/pretrained_models/scfoundation

# Generate gene mapping files (requires internet on node 110)
cd .worktrees/scfoundation-test
python scripts/generate_scf_gene_mapping.py
```

Set `paths.worktree` in the config to point to this directory.

## Pretrained model

Download scFoundation from the official repository:

https://github.com/biomed-AI/scFoundation

The model directory should contain the PyTorch checkpoint and supporting
files needed by `create_scfoundation_backbone`.

## Citation

```bibtex
@article{hao2024scfoundation,
  title={scFoundation: a foundation model for single-cell
         transcriptomics},
  author={Hao, Minsheng and Gong, Jing and Zeng, Xin and Liu, Chi and
          Guo, Yucheng and Cheng, Xingyi and Wang, Taifeng and Ma,
          Jianzhu and Zhang, Xuegong and Song, Le},
  journal={Nature Communications},
  year={2024}
}
```

For the deconvolution adaptation described here, see the accompanying
paper (multi-FM deconvolution benchmark).
