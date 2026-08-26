#!/usr/bin/env python3
"""
LinSeed — Linear spectral decomposition.

Usage
-----
    python run.py --config configs/default.yaml --mode predict \
        --data data/Liver.h5ad --output-dir results/synthetic/Liver/linseed

    python run.py --config configs/default.yaml --mode predict \
        --h5 data/real_bulk/sdy67.h5 \
        --ground-truth data/real_bulk/ground_truth.csv \
        --output-dir results/real_bulk/sdy67/linseed
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _shared.container_runner import main as container_main

METHOD_NAME = "LinSeed"

if __name__ == "__main__":
    container_main(METHOD_NAME)
