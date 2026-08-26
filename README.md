# Multi-Foundation Model Benchmark for Bulk RNA-seq Deconvolution

[![CI](https://github.com/RainyEricYe/bulkgpt-deconvolution-benchmark/actions/workflows/test.yml/badge.svg)](https://github.com/RainyEricYe/bulkgpt-deconvolution-benchmark/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](LICENSE)

A standardized, reproducible benchmark for **bulk RNA-seq deconvolution**, evaluating
**73 methods** against ground truth across **42 datasets** (30 pseudo-bulk + 12
real-bulk). It covers classical regression (NNLS/OLS/Ridge/NuSVR), container-based
deconvolution (CIBERSORTx, DECODE, MuSiC, DWLS, …), single-cell foundation-model
frozen embeddings (scGPT, Geneformer, STACK, TranscriptFormer, scFoundation,
BulkFormer), scGPT-LoRA fine-tuning, and ML regressor baselines (MLP, XGBoost,
RandomForest).

This repository reproduces all results from the paper. Datasets are distributed
separately on [HuggingFace](https://huggingface.co/datasets/yeruihku/bulkgpt-data); see
[Data](#data).

## Quick start

```bash
# 1. Clone
git clone https://github.com/RainyEricYe/bulkgpt-deconvolution-benchmark.git
cd bulkgpt-deconvolution-benchmark

# 2. Environment (conda; see environment.yml / environment.lock.yml)
conda env create -f environment.yml
conda activate bulkgpt
pip install -e .

# 3. Download data (12 real-bulk + 30 pseudo-bulk H5, ~10 GB)
bash data/download_data.sh

# 4. Run a method on a dataset
python scripts/run_real_bulk.py --methods nnls --datasets sdy67

# 5. Evaluate predictions
python scripts/evaluate.py \
  --pred results/2_realbulk/sdy67/nnls/proportions.csv \
  --gt data/2_real_bulk/sdy67_gt.csv
```

Expected output for NNLS on SDY67: **Pearson r ≈ 0.27** (mean across the 5
purified-mixture cell types). For a fully reproducible pipeline, run:

```bash
bash scripts/reproduce_all.sh smoke    # fast check: nnls on SDY67
bash scripts/reproduce_all.sh          # full real-bulk benchmark (12 datasets)
bash scripts/reproduce_all.sh pseudo   # pseudo-bulk benchmark (30 datasets)
```

See `REPRODUCIBILITY.md` for the full expected-results table and verification
checklist.

## Architecture

```
bulkgpt-deconvolution-benchmark/
├── core/                        # Shared evaluation infrastructure
│   ├── data_loader.py           #   Unified H5/h5ad/CSV loader with orientation detection
│   ├── metrics.py               #   evaluate_deconvolution() — DeconBenchmark suite
│   └── deconv/                  #   frozen-eval, embedding, trainer, domain adaptation
├── methods/                     # 73 methods, one directory each
│   ├── {method}/                #   manifest.yaml + run.py + configs/default.yaml
│   ├── _shared/container_runner.py   # shared apptainer container harness
│   └── _linutils.py             # linear baseline utilities (NNLS/OLS/Ridge/NuSVR)
├── scripts/                     # batch runners, evaluators, result aggregation
├── data/                        # manifest + download scripts (H5 lives on HuggingFace)
│   └── 2_real_bulk/             #   *_gt.csv ground-truth (small)
├── results/                     # generated on demand by running benchmarks
├── fixed_scripts/               # container run.R/run.py patches (bind-mounted at runtime)
├── containers/                  # SIF container definitions + build scripts
├── tests/                       # pytest suite
├── weights/                     # weight download script + README (checkpoints external)
└── docs/                        # design notes, analysis docs
```

## Methods (73)

**Linear baselines**: NNLS, OLS, Ridge, NuSVR, CPM, Deconica, Linseed, Mixture.

**Statistical / signature-based**: BayesPrism, BayesCCE, BisqueMarker, BisqueRef,
CIBERSORTx, DSA, DWLS, EPIC, hspe, MOMF, MuSiC, Music2, NITUMID, Quantiseq, TOAST,
DeMixT, DeconPeaker, DeconRNASeq, DECOT, Dtangle, ARIC, AutoGenes, DeconSeq,
DebCAM, DigitalDLSorter, ImmCellAI, lindeconseq, methyResolver, MCPCounter, PreDe,
SCDC, SpatialDWLS, Squid, hspe, MIXTURE, bayCount.

**Deep learning / reference-free**: DECODE, Deblender, Deconformer, DiffFormer,
FARDEep, scaden, TAPE, Sweetwater, CountBridges, DAISM, DeconPP, RNA-Sieve, recide,
MixupVI, EMeth, DeCompress.

**Single-cell foundation models (frozen embedding)**:
scGPT, Geneformer, scFoundation, STACK, TranscriptFormer, BulkFormer.

**Fine-tuned foundation models**: scGPT-LoRA.

**ML regressor baselines**: MLP, XGBoost, RandomForest.

Each method directory contains `manifest.yaml` (mode, timeout, entry points),
`run.py` (unified train/predict dispatcher), and a config. See
`methods/{method}/README.md` for per-method citations and requirements.

## Data

All H5 benchmark files are hosted on HuggingFace:
**https://huggingface.co/datasets/yeruihku/bulkgpt-data** (CC BY 4.0 content).

| Split | # Datasets | Contents | Location in repo |
|-------|:----------:|----------|------------------|
| Pseudo-bulk | 30 | CELLxGENE (10 tissues) + Tabula Sapiens (20) | `1_pseudo_bulk/` |
| Real bulk | 12 | SDY67, Sweetwater, Huuki-Myers, DeMixSC, Altman (×3), Hao (×6) | `2_real_bulk/` |

Download everything (or just one split) with the helper script:

```bash
bash data/download_data.sh           # ~10 GB, everything
bash data/download_data.sh real      # real-bulk only
bash data/download_data.sh pseudo    # pseudo-bulk only
```

Or via the CLI:

```bash
pip install huggingface_hub
huggingface-cli download yeruihku/bulkgpt-data --repo-type dataset \
    --include "2_real_bulk/*.h5" --local-dir data/2_real_bulk
```

### Downstream BRCA data (TCGA + METABRIC)

Processed expression + clinical data for the downstream breast-cancer analysis
lives in a separate dataset repo:
**https://huggingface.co/datasets/yeruihku/bulkgpt-brca**

| File | Samples × Genes | Clinical | Size |
|------|:---------------:|----------|:----:|
| `input/tcga_brca_bulk.h5ad` | 1095 × 60660 | 16 cols (ER/PR/HER2, survival, recurrence, stage) | 232 MB |
| `input/metabric_brca_bulk.h5ad` | 1980 × 20385 | 12 cols (ER/PR/HER2, grade, tumor stage) | 324 MB |
| `input/scRNAseq_ref_gse176078.h5ad` | Wu et al. 2021 scRNA-seq reference | — | 844 MB |
| `input/scRNAseq_ref_gse161529.h5ad` | Pal et al. 2021 scRNA-seq reference | — | 9.5 GB |

Plus per-method deconvolution proportions for the two bulk cohorts:

- `tcga_props/` — 24 CSVs, `{method}[__ft|__frozen]__gse{accession}.csv`
  (e.g. `scgpt__ft__gse176078.csv`), one per method × reference (GSE176078
  default, GSE161529 cross-validation).
- `metabric_props/` — 3 CSVs (`cibersortx/dtangle/music__gse176078.csv`).

Download:

```bash
pip install huggingface_hub
huggingface-cli download yeruihku/bulkgpt-brca --repo-type dataset --local-dir data/brca
```

### H5 canonical format

All H5 files follow a single convention:

```
bulk/values:           (n_samples, n_genes)                — bulk expression
bulk/rownames:          genes
bulk/colnames:          sample IDs

singleCellExpr/values: (n_cells, n_genes)                  — scRNA-seq reference
singleCellExpr/rownames: genes
singleCellExpr/colnames: cell barcodes
singleCellLabels/values: (n_cells,)                       — cell type per cell

ground_truth/values:   (n_types, n_samples)               — real-bulk GT (in H5 or CSV)
bulkRatio/values:      (n_samples, n_types)               — pseudo-bulk GT
```

> **R container note**: R's `h5read` transposes 2-D arrays. Methods using
> `DeconUtils::getArgs` handle this automatically; see `fixed_scripts/` for
> per-method patches.

## Pretrained weights

Foundation-model checkpoints are **not** shipped in this repo. Run
`bash weights/download_weights.sh` to fetch them, or point `core/deconv/frozen_paths.py`
at local copies via environment variables (see `weights/README.md`).

## Results

Pre-computed results are regenerable via `scripts/`:

```bash
python scripts/summarize_results.py          # aggregate + rank
python scripts/eval_real_bulk_ridge.py       # frozen backbone RidgeCV evaluation
python scripts/run_ml_baselines_modeb.py     # ML baseline Mode B (LOO/70-30)
```

See `docs/development_notes.md` for the cross-dataset generalization findings and
ML-baseline analysis.

## Citation

If you use this benchmark in your work, please cite:

```bibtex
@software{bulkgpt2026,
  author = {Ye, Rui},
  title = {Multi-Foundation Model Benchmark for Bulk RNA-seq Deconvolution},
  year = {2026},
  url = {https://github.com/RainyEricYe/bulkgpt-deconvolution-benchmark},
  note = {Data: https://huggingface.co/datasets/yeruihku/bulkgpt-data}
}
```

## License

Code and data: **CC BY 4.0**. See [LICENSE](LICENSE).
