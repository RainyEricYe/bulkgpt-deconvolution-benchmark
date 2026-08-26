#!/usr/bin/env python3
"""Unified entry point for Geneformer deconvolution training and prediction.

This is a thin config-driven dispatcher that delegates to ``train.py`` or
``predict.py`` depending on ``--mode``.  It exists for backward compatibility;
new code should use the standalone scripts directly::

    # Train a new model
    python methods/geneformer/train.py --config methods/geneformer/configs/ft.yaml

    # Evaluate a trained checkpoint
    python methods/geneformer/predict.py --config methods/geneformer/configs/ft.yaml \\
        --checkpoint checkpoints/geneformer/ft/best_model.pt

Legacy usage (still supported)::

    python run.py --config configs/ft.yaml --mode train
    python run.py --config configs/ft.yaml --mode predict \\
        --checkpoint checkpoints/geneformer/ft/best_model.pt
"""
import argparse
import sys
from pathlib import Path

# -- Ensure project structure is importable -----------------------------------
_to_publish = Path(__file__).resolve().parent.parent.parent  # to_publish/
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from methods.geneformer.predict import main as predict_main
from methods.geneformer.train import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geneformer deconvolution: train or predict",
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
    parsed = parser.parse_args()

    if parsed.mode == "train":
        train_main(parsed.config, parsed.log_file)
    else:
        if not parsed.checkpoint:
            print("ERROR: --checkpoint is required for predict mode", file=sys.stderr)
            sys.exit(1)
        predict_main(parsed.config, parsed.checkpoint, parsed.log_file)


if __name__ == "__main__":
    main()
