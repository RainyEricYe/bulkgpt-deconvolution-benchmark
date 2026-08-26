#!/usr/bin/env python3
"""Unified entry point for BulkFormer deconvolution training and prediction.

Thin config-driven dispatcher that delegates to ``train.py`` or ``predict.py``.

Usage::

    python run.py --config configs/default.yaml --mode train
    python run.py --config configs/default.yaml --mode predict --checkpoint best_model.pt
"""

import argparse
import sys
from pathlib import Path

_to_publish = Path(__file__).resolve().parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from methods.bulkformer.predict import main as predict_main
from methods.bulkformer.train import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BulkFormer deconvolution: train or predict")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["train", "predict", "ridge"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train_main(args.config, args.log_file)
    else:
        predict_main(args.config, args.checkpoint, args.log_file)


if __name__ == "__main__":
    main()
