#!/usr/bin/env python3
"""Sweetwater — Interpretable scArches VAE deconvolution via Apptainer container.

Runs inside the sweetwater.sif Apptainer container. Auto-discovers
fixed_scripts/sweetwater/run.py for full-parameter training.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _shared.container_runner import main as container_main

METHOD_NAME = "sweetwater"

if __name__ == "__main__":
    container_main(METHOD_NAME)
