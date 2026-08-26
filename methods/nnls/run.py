#!/usr/bin/env python3
"""NNLS deconvolution — independent entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _linutils import main
main(method_name="nnls")
