#!/usr/bin/env python3
"""
ImmuCellAI — Immune cell abundance estimation.

Usage
-----
    python run.py --config configs/default.yaml --mode predict \
        --data data/Liver.h5ad --output-dir results/synthetic/Liver/immucellai

    python run.py --config configs/default.yaml --mode predict \
        --h5 data/real_bulk/sdy67.h5 \
        --ground-truth data/real_bulk/ground_truth.csv \
        --output-dir results/real_bulk/sdy67/immucellai
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _shared.container_runner import main as container_main

METHOD_NAME = "ImmuCellAI"

if __name__ == "__main__":
    container_main(METHOD_NAME)
