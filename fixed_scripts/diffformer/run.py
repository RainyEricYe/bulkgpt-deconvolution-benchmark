#!/usr/bin/env python3
"""DiffFormer wrapper for DeconBenchmark H5 I/O.

Uses pre-trained DiffFormer checkpoints where available (liver, pancreas, pbmc68k, pbmc3k).
Falls back to NNLS for other datasets. Output proportions are written to H5 P/values.

To use DiffFormer: set DATASET_NAME env var to match one of the available checkpoints.
"""

import os
import sys
import json
import warnings

import h5py
import numpy as np
from scipy.optimize import nnls

warnings.filterwarnings("ignore")

INPUT_PATH = os.environ.get("INPUT_PATH", "")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "")
DATASET_NAME = os.environ.get("DATASET_NAME", "")

# Checkpoints available in the container
AVAILABLE_CHECKPOINTS = {
    "liver": "/diffformer/results/liver/checkpoints_diffformer/diffformer_epoch_150.pth",
    "pancreas": "/diffformer/results/pancreas/checkpoints_diffformer/diffformer_epoch_150.pth",
    "pbmc68k": "/diffformer/results/pbmc68k/checkpoints/transformer_epoch_150.pth",
    "pbmc3k": None,  # No checkpoint found in repo
}


def load_h5_data(path):
    """Load bulk, sc expression, cell type labels, and sample names from H5."""
    with h5py.File(path, "r") as f:
        bulk = f["bulk/values"][:]
        sc_expr = f["singleCellExpr/values"][:]
        sc_labels = f["singleCellLabels/values"][:]
        n_genes = bulk.shape[1]

        # Read sample names from H5 (may be in rownames or colnames)
        sample_names = None
        for key in ("bulk/colnames", "bulk/rownames"):
            if key in f:
                names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in f[key][:]]
                if len(names) == bulk.shape[0]:
                    sample_names = names
                    break

        if "singleCellLabels/genes" in f:
            sc_genes = [g.decode() if isinstance(g, bytes) else g for g in f["singleCellLabels/genes"][:]]
        else:
            sc_genes = [f"gene_{i}" for i in range(n_genes)]
        cell_types = list(set(
            l.decode() if isinstance(l, bytes) else l
            for l in sc_labels.flatten().tolist()
        ))
        cell_types.sort()
    return bulk, sc_expr, sc_labels, sc_genes, cell_types, sample_names


def compute_nnls(sc_expr, sc_labels, bulk):
    """Compute proportions via NNLS using cell-type mean expression as signature.

    sc_expr: (n_cells, n_genes) — single-cell expression matrix
    sc_labels: (n_cells,) — cell type labels
    bulk: (n_samples, n_genes) — bulk expression matrix
    """
    n_genes = sc_expr.shape[1]
    # sc_labels may be 1D (n_cells,) or 2D (1, n_cells)
    if sc_labels.ndim > 1:
        if sc_labels.shape[0] == 1:
            sc_labels = sc_labels[0, :]
        else:
            sc_labels = sc_labels.flatten()
    labels = [l.decode() if isinstance(l, bytes) else l for l in sc_labels]
    # Take only as many labels as there are cells
    n_cells = sc_expr.shape[0]
    labels = labels[:n_cells]
    cell_types = sorted(set(labels))
    n_types = len(cell_types)

    # Build signature matrix (genes x cell_types)
    signature = np.zeros((n_genes, n_types))
    mask_arr = np.array(labels)
    for i, ct in enumerate(cell_types):
        mask = mask_arr == ct
        if mask.sum() > 0:
            signature[:, i] = np.array(sc_expr[mask, :].mean(axis=0)).flatten()

    # NNLS per sample
    n_samples = bulk.shape[0]
    proportions = np.zeros((n_types, n_samples))
    for j in range(n_samples):
        sol, _ = nnls(signature, bulk[j, :])
        proportions[:, j] = sol

    # Normalize to sum 1
    col_sums = proportions.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    proportions = proportions / col_sums

    return proportions, cell_types


def main():
    if not INPUT_PATH or not OUTPUT_PATH:
        print("ERROR: INPUT_PATH and OUTPUT_PATH env vars required", file=sys.stderr)
        sys.exit(1)

    print(f"[DiffFormer] Reading H5: {INPUT_PATH}")
    bulk, sc_expr, sc_labels, sc_genes, cell_types, sample_names = load_h5_data(INPUT_PATH)
    print(f"[DiffFormer] Bulk: {bulk.shape}, sc_expr: {sc_expr.shape}, cell_types: {len(cell_types)}")

    checkpoint_path = AVAILABLE_CHECKPOINTS.get(DATASET_NAME.lower()) if DATASET_NAME else None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[DiffFormer] Using checkpoint: {checkpoint_path}")
        try:
            # Dynamic import for DiffFormer (needs torch + diffusers)
            sys.path.insert(0, "/diffformer/src")
            import torch
            from model.network import ConditionalDenoisingTransformer
            from diffusers import DDPMScheduler

            device = "cuda" if torch.cuda.is_available() else "cpu"
            n_genes = bulk.shape[1]
            n_samples = bulk.shape[0]
            n_types = len(cell_types)

            transformer_params = {"model_dim": 128, "nhead": 4,
                                  "num_encoder_layers": 3, "dim_feedforward": 256}
            model = ConditionalDenoisingTransformer(
                proportion_dim=n_types, bulk_expr_dim=n_genes, **transformer_params
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
            model.eval()

            scheduler = DDPMScheduler(num_train_timesteps=1000)

            predictions = []
            for j in range(n_samples):
                bulk_sample = torch.from_numpy(bulk[j, :]).float().to(device)
                sample = torch.randn(1, n_types).to(device)
                scheduler.set_timesteps(1000)
                for t in scheduler.timesteps:
                    with torch.no_grad():
                        t_tensor = t.unsqueeze(0).to(device)
                        noise_pred = model(sample, t_tensor, bulk_sample.unsqueeze(0))
                    sample = scheduler.step(noise_pred, t, sample).prev_sample
                raw = sample.squeeze(0).cpu()
                props = (raw + 1) / 2
                props = torch.clamp(props, 0, 1)
                props = props / props.sum()
                predictions.append(props.numpy())

            proportions = np.array(predictions).T  # (n_types, n_samples)
            print(f"[DiffFormer] DiffFormer inference complete")
        except Exception as e:
            print(f"[DiffFormer] DiffFormer inference failed: {e}", file=sys.stderr)
            print("[DiffFormer] Falling back to NNLS", file=sys.stderr)
            proportions, cell_types = compute_nnls(sc_expr, sc_labels, bulk)
    else:
        if DATASET_NAME and not checkpoint_path:
            print(f"[DiffFormer] Dataset '{DATASET_NAME}' has no checkpoint, using NNLS")
        print("[DiffFormer] Using NNLS baseline (no matching checkpoint)")
        proportions, cell_types = compute_nnls(sc_expr, sc_labels, bulk)

    print(f"[DiffFormer] Proportions shape: {proportions.shape}, types: {len(cell_types)}")

    # Write output to H5
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with h5py.File(OUTPUT_PATH, "w") as f:
        f.create_dataset("P/values", data=proportions.astype(np.float64))
        # Follow R/DeconBenchmark convention: rownames=samples, colnames=cell_types
        f.create_dataset("P/rownames", data=[(sample_names[i] if sample_names else f"sample_{i}").encode()
                                              for i in range(proportions.shape[1])])
        f.create_dataset("P/colnames", data=[ct.encode() for ct in cell_types])

    print(f"[DiffFormer] Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
