#!/usr/bin/env python3
"""Shared utilities for scFoundation deconvolution scripts."""
import os
import subprocess
import sys
import time
from pathlib import Path


def run_subprocess(
    cmd: list[str],
    env: dict,
    cwd: str,
    tee,
    timeout: int = 3600,
) -> int:
    """Run *cmd* as a subprocess, teeing stdout/stderr in real time.

    Returns the exit code, or ``-1`` on timeout.
    """
    start = time.time()
    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        tee(line.rstrip())
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        tee(f"TIMEOUT after {timeout}s")
        return -1
    elapsed = time.time() - start
    tee(f"Elapsed: {elapsed:.1f}s")
    return process.returncode


def build_cli_args(config: dict) -> list[str]:
    """Translate the YAML config dict into a flat CLI-argument list.

    The returned list is passed directly to the scFoundation worktree's
    ``scripts/train.py``.  Unlike scGPT/Geneformer, scFoundation does **not**
    branch on mode -- the same arguments are used for both training and
    prediction.
    """
    args: list[str] = []
    dataset = config.get("dataset", {})
    model_cfg = config.get("model", {})
    training = config.get("training", {})
    paths = config.get("paths", {})

    # -- Dataset ----------------------------------------------------------------
    sc_ref = dataset.get("sc_ref") or paths.get("sc_ref")
    if sc_ref:
        args.extend(["--sc_ref", sc_ref])

    n_hvg = dataset.get("n_hvg", 19264)
    args.extend(["--n_hvg", str(n_hvg)])

    celltype_col = dataset.get("celltype_col", "cell_type")
    args.extend(["--celltype_col", celltype_col])

    batch_col = dataset.get("batch_col")
    if batch_col is not None:
        args.extend(["--batch_col", batch_col])

    # -- Model ------------------------------------------------------------------
    backbone_type = model_cfg.get("backbone_type", "scfoundation")
    args.extend(["--backbone_type", backbone_type])

    cell_emb_style = model_cfg.get("cell_emb_style", "cls")
    args.extend(["--cell_emb_style", cell_emb_style])

    deconv_hidden_dim = model_cfg.get("deconv_hidden_dim", 256)
    args.extend(["--deconv-hidden-dim", str(deconv_hidden_dim)])

    deconv_n_layers = model_cfg.get("deconv_n_layers", 2)
    args.extend(["--deconv-n-layers", str(deconv_n_layers)])

    # -- Paths ------------------------------------------------------------------
    pretrained_model = paths.get("pretrained_model")
    if pretrained_model:
        args.extend(["--pretrained_model", pretrained_model])

    checkpoint_dir = paths.get("checkpoint_dir")
    if checkpoint_dir:
        args.extend(["--checkpoint_dir", checkpoint_dir])

    # -- Training parameters (passed in both modes) -----------------------------
    epochs = training.get("epochs", 10)
    args.extend(["--epochs", str(epochs)])

    batch_size = training.get("batch_size", 4)
    args.extend(["--batch_size", str(batch_size)])

    lr = training.get("lr", 1e-3)
    args.extend(["--lr", str(lr)])

    backbone_lr = training.get("backbone_lr")
    if backbone_lr is not None:
        args.extend(["--backbone_lr", str(backbone_lr)])

    n_pseudo_bulk = training.get("n_pseudo_bulk", 2000)
    args.extend(["--n_pseudo_bulk", str(n_pseudo_bulk)])

    proportion_alpha = training.get("proportion_alpha", 1.0)
    args.extend(["--proportion_alpha", str(proportion_alpha)])

    seed = training.get("seed", 42)
    args.extend(["--seed", str(seed)])

    loss_type = training.get("loss_type", "mse_kl")
    args.extend(["--loss_type", loss_type])

    num_workers = training.get("num_workers", 4)
    args.extend(["--num_workers", str(num_workers)])

    if training.get("unfreeze_backbone", False):
        args.append("--unfreeze_backbone")

    if training.get("use_wandb", False):
        args.append("--use_wandb")

    return args
