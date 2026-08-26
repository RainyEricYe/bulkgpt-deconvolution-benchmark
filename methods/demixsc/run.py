#!/usr/bin/env python3
"""DeMixSC — Weighted NNLS with benchmark alignment via Apptainer container.

Runs inside the demixsc.sif Apptainer container (R package). Auto-discovers
fixed_scripts/demixsc/run.R for the DeconUtils-based implementation.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _shared.container_runner import main as container_main

METHOD_NAME = "DeMixSC"

if __name__ == "__main__":
    container_main(METHOD_NAME)
