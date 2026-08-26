#!/usr/bin/env python3
"""SQUID — Dampened weighted least-squares deconvolution via Apptainer container.

Runs inside the squid.sif Apptainer container (R package). Auto-discovers
fixed_scripts/squid/run.R for the DeconUtils-based DWLS implementation.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _shared.container_runner import main as container_main

METHOD_NAME = "SQUID"

if __name__ == "__main__":
    container_main(METHOD_NAME)
