#!/usr/bin/env python3
"""Config-driven dispatcher for scGPT-LoRA fine-tuning.

Usage:
    python methods/scgpt_lora/run.py train --config methods/scgpt_lora/configs/default.yaml
    python methods/scgpt_lora/run.py predict --config methods/scgpt_lora/configs/default.yaml --checkpoint checkpoints/lora/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="scGPT-LoRA deconvolution")
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        from methods.scgpt_lora.train import main as train_main
        train_main(
            config_path=str(config_path),
            log_file=args.log_file,
            seed=args.seed,
        )
    elif args.mode == "predict":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for predict mode", file=sys.stderr)
            sys.exit(1)
        from methods.scgpt_lora.predict import main as predict_main
        predict_main(
            config_path=str(config_path),
            checkpoint=args.checkpoint,
            log_file=args.log_file,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
