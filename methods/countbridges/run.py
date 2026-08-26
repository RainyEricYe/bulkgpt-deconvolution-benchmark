#!/usr/bin/env python3
"""
Count Bridges — stochastic bridge process deconvolution (ICLR 2026).

Used via Apptainer SIF container. This entrypoint delegates to
container_runner which handles H5 I/O, SIF execution, and output parsing.

Usage
-----
    python run.py --config configs/default.yaml --mode predict \
        --h5 data/2_real_bulk/sdy67.h5 \
        --ground-truth data/2_real_bulk/sdy67_gt.csv \
        --output-dir results/2_realbulk/sdy67/countbridges
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _shared.container_runner import main as container_main

METHOD_NAME = "CountBridges"

if __name__ == "__main__":
    container_main(METHOD_NAME)
