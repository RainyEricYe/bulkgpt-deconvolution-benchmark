#!/usr/bin/env python3
"""Thin dispatcher for scFoundation deconvolution training and prediction.

Delegates to ``train.main()`` or ``predict.main()`` based on ``--mode``.

Usage:
    python run.py --config configs/frozen.yaml --mode train
    python run.py --config configs/frozen.yaml --mode predict
"""
import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="scFoundation deconvolution: train or predict",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "predict"],
        help="Operation mode",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Path to log file (default: <checkpoint_dir>/run.log)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        from train import main as entry_point
    else:
        from predict import main as entry_point

    entry_point(args.config, args.log_file)


if __name__ == "__main__":
    main()
