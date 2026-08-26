#!/usr/bin/env python3
"""Unified entry point for scGPT deconvolution training and prediction.

Thin dispatcher that keeps backward compatibility.  Delegates to
:mod:`train` or :mod:`predict` based on ``--mode``.

Usage:
    # Train
    python run.py --config configs/ft.yaml --mode train

    # Evaluate a trained checkpoint
    python run.py --config configs/ft.yaml --mode predict \\
        --checkpoint checkpoints/ft/best_model.pt
"""

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
        description="scGPT deconvolution: train or predict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--mode", required=True, choices=["train", "predict"],
        help="Operation mode",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to .pt checkpoint file (required for predict mode)",
    )
    parser.add_argument(
        "--log_file", default=None,
        help="Path to log file (default: <checkpoint_dir>/run.log)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config value; default: 42 from config)",
    )
    parsed = parser.parse_args()

    if parsed.mode == "train":
        train_main(parsed.config, parsed.log_file, parsed.seed)
    else:
        if not parsed.checkpoint:
            print("ERROR: --checkpoint is required for predict mode", file=sys.stderr)
            sys.exit(1)
        predict_main(parsed.config, parsed.checkpoint, parsed.log_file, seed=parsed.seed)


if __name__ == "__main__":
    main()
