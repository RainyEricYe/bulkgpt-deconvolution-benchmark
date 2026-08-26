#!/usr/bin/env bash
# Thin wrapper — delegates to scripts/run_real_bulk.py
#
# Usage
# -----
#   bash tests/run_all.sh                          # run all methods
#   bash tests/run_all.sh --methods scgpt nnls     # run specific methods only
#   bash tests/run_all.sh --quick                  # minimal epochs/batches for smoke test
#   bash tests/run_all.sh --summary-only           # reprint last summary
#
# For full options see: python scripts/run_real_bulk.py --help

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

cd "$PROJECT_ROOT"
python scripts/run_real_bulk.py "$@"
