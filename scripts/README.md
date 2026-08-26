# BulkGPT Publication Scripts

These scripts reproduce the benchmark results, tables, and figures from the
manuscript. They are portable: no hardcoded machine paths; data is loaded from
the standardized `data/` and `results/` directories.

## One-command reproduction

```bash
bash scripts/reproduce_all.sh            # full real-bulk benchmark (12 datasets)
bash scripts/reproduce_all.sh pseudo     # pseudo-bulk benchmark (30 datasets)
bash scripts/reproduce_all.sh smoke      # fast check: nnls on SDY67 (~5 s)
```

This chains: **download data → run methods → evaluate → summarize**. Data is
fetched from https://huggingface.co/datasets/yeruihku/bulkgpt-data into
`data/`. See `scripts/reproduce_all.sh` for details.

## Script Overview

| Script | Purpose |
|--------|---------|
| `reproduce_all.sh` | One-command pipeline: download → run → evaluate → summarize |
| `run_real_bulk.py` | Run methods on all real-bulk datasets (`results/2_realbulk/`) |
| `run_pseudo_bulk.py` | Run methods on all pseudo-bulk datasets (`results/1_pseudo_bulk/`) |
| `evaluate.py` | Evaluate predictions vs ground truth (single file or `--batch`) |
| `summarize_results.py` | Aggregate per-method × per-dataset metrics, rank by mean Pearson, Markdown report |
| `generate_summary.py` | CSV output of the same aggregate |
| `eval_real_bulk_ridge.py` | Frozen-backbone + RidgeCV evaluation (Mode B) |
| `run_ml_baselines_modeb.py` | ML baselines (MLP/XGBoost/RandomForest) under Mode B (LOO/70-30) |
| `eval_frozen_search.py` | Frozen embedding hyperparameter search |
| `tissue_stratified_ranking.py` | Tissue-stratified ranking analysis |
| `build_h5_from_h5ad.py` | Build DeconBenchmark-format H5 from AnnData |

## Usage examples

```bash
# Run the linear baselines on one dataset
python scripts/run_real_bulk.py --methods nnls ols ridge --datasets sdy67

# Run every method on every real-bulk dataset (parallel)
python scripts/run_real_bulk.py --parallel 4

# Evaluate a single prediction
python scripts/evaluate.py \
  --pred results/2_realbulk/sdy67/nnls/proportions.csv \
  --gt data/2_real_bulk/sdy67_gt.csv

# Batch-evaluate all results under a directory
python scripts/evaluate.py --batch results/2_realbulk/sdy67

# Aggregate + rank, output Markdown
python scripts/summarize_results.py -o results_summary.md
```

## Output layout

Each method run writes to `results/{2_realbulk|1_pseudo_bulk}/{dataset}/{method}/`:

```
<results-dir>/
  <dataset_name>/
    <method_name>/
      proportions.csv      # predicted cell-type proportions
      metrics.json         # DeconBenchmark metrics (Pearson, Spearman, RMSE, MAE)
      config.yaml / *.log  # run metadata
```

Each `metrics.json` contains keys like:

```json
{
  "pearson_mean": 0.989,
  "spearman_mean": 0.965,
  "rmse_overall": 0.036,
  "mae_overall": 0.025,
  "pearson_per_type": [0.988, 0.971, ...],
  "cell_types": ["B cell", "CD8+ T", ...]
}
```
