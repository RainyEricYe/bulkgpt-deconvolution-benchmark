#!/usr/bin/env bash
# One-command reproduction of the BulkGPT deconvolution benchmark.
#
# Chains:  data download -> run methods -> evaluate -> summarize
#
# Usage (from the repository root):
#   bash scripts/reproduce_all.sh            # full real-bulk benchmark (12 datasets)
#   bash scripts/reproduce_all.sh smoke      # fast sanity check (nnls on 1 dataset)
#   bash scripts/reproduce_all.sh real       # download + run real bulk only
#   bash scripts/reproduce_all.sh pseudo     # download + run pseudo bulk only
#   bash scripts/reproduce_all.sh --help
#
# Notes:
#   * Data H5 files (~10 GB real bulk, ~180 MB pseudo bulk) are fetched from
#     https://huggingface.co/datasets/yeruihku/bulkgpt-data into ./data/.
#   * Container-based methods (CIBERSORTx, DECODE, MuSiC, …) need an Apptainer
#     SIF built first — see containers/README.md. The pure-Python baselines
#     (nnls/ols/ridge/nusvr/recide) run with no extra setup.
#   * Foundation-model methods (scGPT, Geneformer, …) need weights: see
#     weights/README.md.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SCOPE="${1:-all}"

case "$SCOPE" in
  --help|-h)
    sed -n '1,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  smoke)
    echo "== [1/3] data (single dataset: sdy67) =="
    bash data/download_data.sh real sdy67
    echo
    echo "== [2/3] run nnls on sdy67 (smoke) =="
    "$PYTHON" scripts/run_real_bulk.py --methods nnls --datasets sdy67
    echo
    echo "== [3/3] summarize =="
    "$PYTHON" scripts/summarize_results.py --benchmark realbulk --top 10
    echo
    echo "Smoke reproduction finished. Expected: nnls on sdy67 ≈ Pearson r 0.27"
    ;;
  real|pseudo|all)
    echo "== [1/3] data =="
    bash data/download_data.sh "$SCOPE"
    echo
    echo "== [2/3] run benchmark =="
    if [ "$SCOPE" = "real" ] || [ "$SCOPE" = "all" ]; then
      "$PYTHON" scripts/run_real_bulk.py
    fi
    if [ "$SCOPE" = "pseudo" ] || [ "$SCOPE" = "all" ]; then
      "$PYTHON" scripts/run_pseudo_bulk.py
    fi
    echo
    echo "== [3/3] summarize =="
    "$PYTHON" scripts/summarize_results.py --benchmark "$SCOPE"
    ;;
  *)
    echo "Usage: $0 [smoke|real|pseudo|all]" >&2
    exit 2
    ;;
esac
