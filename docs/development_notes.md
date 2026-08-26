# to_publish — Standardized 73 Method Deconvolution Benchmark

Evaluation framework for bulk RNA-seq deconvolution methods against ground
truth across **42 datasets** (30 pseudo-bulk + 12 real-bulk). Covers ~75
active methods: classical regression (NNLS/OLS/Ridge/NuSVR), container-based
deconvolution (CIBERSORTx, DECODE, MuSiC, DWLS, etc.), single-cell foundation
model frozen embeddings (scGPT, Geneformer, STACK, TranscriptFormer,
scFoundation, BulkFormer), scGPT-LoRA fine-tuning, and **ML regressor baselines
(MLP, XGBoost, RandomForest)** added in P2#13.

## Quick Start

```bash
conda activate bulkgpt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Pseudo-bulk evaluation (30 datasets, GT embedded in H5)
python scripts/run_pseudo_bulk.py --methods nnls --subdir cellxgene
python scripts/run_pseudo_bulk.py --methods nnls --datasets cellxgene_Liver
python scripts/run_pseudo_bulk.py --parallel 6

# Real-bulk evaluation
python scripts/run_real_bulk.py --methods decode --datasets sdy67

# Evaluate predictions
python scripts/evaluate.py \
  --pred results/1_pseudo_bulk/cellxgene_Liver/nnls/proportions.csv \
  --gt data/1_pseudo_bulk/cellxgene/Liver.h5
```

## Architecture

```
to_publish/
├── core/                        # Shared evaluation infrastructure
│   ├── data_loader.py           #   Unified H5/h5ad/CSV loader with orientation detection
│   ├── metrics.py               #   evaluate_deconvolution() — DeconBenchmark suite
│   └── deconv/
│       ├── frozen_eval.py       #   ENCODE_FN registry + evaluate_real_bulk_ridge()
│       ├── frozen_paths.py      #   Weight path constants (env-var overridable)
│       ├── frozen_search.py     #   Frozen backbone hyperparameter search
│       ├── embedding.py         #   MixGenerator, EmbeddingDeconvHead
│       ├── trainer.py           #   MLP head trainer
│       ├── model.py             #   DeconvHead, DeconvLoss
│       ├── domain_adaptation.py #   MMD loss, gradient reversal
│       ├── utils.py             #   set_seed, renormalize_props, logging helpers
│       └── resources.py         #   Backward-compat re-export → core/resources.py
├── methods/                     # 73 methods, one directory each
│   ├── {method}/
│   │   ├── manifest.yaml        #   name, mode, timeout, entry points
│   │   ├── run.py               #   unified train/predict dispatcher
│   │   └── configs/default.yaml
│   ├── _shared/container_runner.py  # shared apptainer container harness
│   └── _linutils.py             # linear baseline utilities (NNLS/OLS/Ridge/NuSVR)
├── scripts/
│   ├── run_pseudo_bulk.py       # batch method x dataset runner for 1_pseudo_bulk
│   ├── run_real_bulk.py         # batch method x dataset runner for 2_real_bulk
│   ├── evaluate.py              # unified evaluation CLI (incl. Hungarian matching)
│   ├── eval_real_bulk_ridge.py  # frozen backbone RidgeCV evaluation
│   ├── eval_all_backbones.sh    # bash wrapper: one backbone per conda env
│   └── summarize_results.py      # aggregate + rank results (use this one)
├── data/
│   ├── 1_pseudo_bulk/           # 30 pseudo-bulk datasets (cellxgene=10, tabula_sapiens=20)
│   │   ├── cellxgene/           #   GT in H5 bulkRatio/ group
│   │   └── tabula_sapiens/      #   GT in H5 bulkRatio/ group (injected from RDS)
│   ├── 2_real_bulk/             # 6 real-bulk datasets (H5 + GT CSV)
│   └── prepare/                 # download + convert scripts
├── results/
│   ├── 1_pseudo_bulk/           # {subdir}_{dataset}/{method}/proportions.csv + metrics.json
│   └── 2_realbulk/              # {dataset}/{method}/proportions.csv + metrics.json
├── fixed_scripts/               # container run.R/run.py patches (bind-mounted at runtime)
├── containers/                  # SIF container definitions + build scripts
│   └── *_fix.def                #   Apptainer definition files for SIF rebuilds
├── tests/                       # pytest test suite (99 tests)
├── weights/                     # pretrained weight symlinks (see weights/README.md)
└── logs/                        # node run logs
```

## H5 Canonical Format (unified 2026-07-02, verified from 1_pseudo_bulk)

All H5 files follow a single convention (verified from 30 pseudo-bulk H5
under ``data/1_pseudo_bulk/``):

```
bulk/values:           (n_samples, n_genes)
bulk/rownames:          genes                      (Ensembl ID or gene symbol)
bulk/colnames:          sample IDs

singleCellExpr/values: (n_cells, n_genes)
singleCellExpr/rownames: genes
singleCellExpr/colnames: cell barcodes

singleCellLabels/values: (n_cells,)               cell type per cell

ground_truth/values:   (n_types,  n_samples)      — 2_real_bulk (optional in H5)
bulkRatio/values:      (n_samples, n_types)        — 1_pseudo_bulk

cellTypeExpr/values:   (n_types, n_genes)          — aggregate expression per type
signature/values:      (n_types, n_sig_genes)      — cell-type signature matrix
```

**Key rule for singleCellExpr** — ``rownames=genes``, ``colnames=cells``.

## 易错点

### TranscriptFormer train.py UnboundLocalError
当 config 中 ``train_ratio + val_ratio = 1.0``（如 ``train_ratio: 0.8, val_ratio: 0.2``）时，
pseudo-bulk 全部用于 train/val，test set 为空。train.py 末尾的 test evaluation 块被跳过，
导致 ``predictions``/``ground_truth``/``eval_metrics`` 三个变量从未赋值，后续代码
``pd.DataFrame(ground_truth)`` 抛出 ``UnboundLocalError``。

修复：在 ``# -- 5. Evaluate on test set`` 的 ``if`` 前添加：
```python
predictions = None
ground_truth = None
eval_metrics = None
```
并在 ``pd.DataFrame(ground_truth, ...)`` 前加 ``if ground_truth is not None:`` 保护。
**不要修改 config 的 train_ratio**——预测是对 real bulk 做的，不需要 pseudo-bulk test set。


## 跨数据集泛化实验

实验框架: tests/cross_dataset_pbmc.py
结果: tests/cross_dataset_pbmc/ + docs/cross_dataset_experiments.md
原始数据: data/2_real_bulk/{dataset}.h5 + {dataset}_gt.csv

### 核心结论

1. **random_mean_pool (BulkFormer 随机权重 + mean pooling) 在 18/21 跨数据集 pair 上优于 pca_ridge**。固定嵌入空间保证跨数据集表示一致性，PCA 成分是数据依赖的，跨数据集不兼容。

2. **最佳通用训练源: altman_Arunachalam (322 临床样本，PBMC)**。覆盖真实表达变异范围，对其他 PBMC 数据集平均 r≈0.65。sdy67（250 体外混合样本）泛化很差（r≈0.18）——样本数不如分布覆盖重要。

3. **12×12 全矩阵**结果已保存为 CSV（Pearson + Spearman），源也可作为目标。

4. **跨组织泛化（PBMC→脑/视网膜）需要伪 bulk 训练**（core/deconv/frozen_search.py），因为细胞类型体系完全不同。

### 嵌入空间维度

| Backbone | Dim | 架构 |
|----------|:---:|:----:|
| scGPT | 512 | Transformer |
| Geneformer | 256 | Transformer |
| scFoundation | 768 | Transformer (1B) |
| TranscriptFormer | 2048 | Transformer |
| STACK | 1600 | VAE |
| BulkFormer-147M | 640 | GCN+Performer |

## ML 基线（P2#13，2026-08-05）

Three reference-free, training-based ML regressor baselines were added as a
standalone supplementary analysis (Option A — main method count unchanged):
`methods/{mlp,xgboost,randomforest}/` + shared engine
`methods/_shared/ml_baselines.py`. CNN was developed then abandoned
(underperformed; code/results removed). Analysis notes and final supplement
table: `docs/remote_analysis_data/ml_baselines_analysis_notes.md` and
`ml_baselines_supplement_table.csv`.

**Two evaluation modes** (both implemented; results in
`docs/remote_analysis_data/`):

- **Mode A** (SCADEN-style): train on Dirichlet pseudo-bulk mixtures generated
  from the dataset's scRNA reference (alpha=0.1, 80 cells/sample, 500–1000
  mixtures; features = top-2000 HVGs of common genes, log1p-CPM), predict all
  bulk samples; post-hoc evaluation via `scripts/evaluate.py`. Dispatched
  automatically via `run_pseudo_bulk.py` (30 datasets) / `run_real_bulk.py`
  (12 datasets); `mlp`/`cnn`→`mlp` were added to `GPU_METHODS` in
  `_dispatch_real_bulk.py`, and the four→three methods route to
  `_dispatch_linear` in `run_pseudo_bulk.py`.
- **Mode B** (primary per user decision): train the ML regressors **directly on
  real-bulk samples** (features = PCA-50 of log1p-CPM bulk expression, labels =
  ground-truth proportions), held-out evaluation using the framework's Mode B
  system (`core.deconv.utils.renormalize_props` +
  `core.metrics.evaluate_deconvolution`, mirroring `scripts/eval_loo_ridge.py`).
  n<60 → leave-one-out; n≥60 → deterministic 70/30 split. Runner:
  `scripts/run_ml_baselines_modeb.py` → writes
  `results/2_realbulk/{dataset}/{method}_modeb/` with `ridge_metrics.json`
  (Mode B guard). Run on GPU or CPU nodes (both supported).

**Final results (mean Pearson r)**: Mode A pseudo (30): RF 0.842, XGB 0.776,
MLP 0.257 (vs SCADEN 0.777). Mode A real (12): RF 0.435, XGB 0.364, MLP 0.098
(vs SCADEN 0.482). Mode B real (12): RF 0.475, XGB 0.455, MLP 0.247. Tree
ensembles land in the SCADEN band; MLP underfits. Caveats: LOO on tiny datasets
overfits (e.g. RF hoek_Hao r=1.00); near-one-hot GT (linsley_purified_Hao)
inflates all methods; `run_pseudo_bulk._localize_h5` /tmp cache can collide
under parallel dispatch (same-ms pid+timestamp) — rerun corrupt outputs.

**Environment**: methods run in the `bulkgpt` conda env; `xgboost` (3.2.0) was
installed via pip.
