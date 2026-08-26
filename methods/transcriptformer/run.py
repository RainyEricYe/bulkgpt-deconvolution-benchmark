#!/usr/bin/env python3
"""Unified entry point for TranscriptFormer embedding-based deconvolution.

Thin dispatcher that keeps backward compatibility.  Delegates to
:mod:`train` or :mod:`predict` based on ``--mode``.

Usage:
    # Train
    python methods/transcriptformer/run.py --config configs/default.yaml --mode train

    # Predict (requires trained checkpoint)
    python methods/transcriptformer/run.py --config configs/default.yaml --mode predict \\
        --checkpoint results/transcriptformer/checkpoint/deconv_head.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from train import main as train_main
from predict import main as predict_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TranscriptFormer embedding-based deconvolution: train or predict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode", required=True, choices=["train", "predict", "all"],
        help="Operation mode",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to deconv_head.pt checkpoint (required for predict mode)",
    )
    parser.add_argument("--log_file", default=None, help="Log file path")
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        train_main(args.config, args.log_file)
    if args.mode in ("predict", "all"):
        if not args.checkpoint:
            print("ERROR: --checkpoint is required for predict mode", file=sys.stderr)
            sys.exit(1)
        predict_main(args.config, args.checkpoint, args.log_file)


if __name__ == "__main__":
    main()
