#!/usr/bin/env python3
"""
Sweetwater deconvolution.

Reads H5 input (args.h5), trains Sweetwater autoencoder on scRNA reference,
predicts on bulk samples, writes H5 output (results.h5).

Inputs: bulk, singleCellExpr, singleCellLabels
Output: P (proportions matrix)
"""
import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
import h5py

warnings.filterwarnings("ignore")

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/args.h5")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.h5")

METHOD_NAME = "Sweetwater"


def read_h5_input(path):
    """Read DeconBenchmark H5 input into a dict."""
    f = h5py.File(path, "r")
    data = {}
    for key in f:
        if key == "seed":
            continue  # skip seed, handled separately
        grp = f[key]
        if "values" not in grp:
            continue
        try:
            values = grp["values"][:]
        except (TypeError, ValueError, OSError):
            continue

        if len(values.shape) == 2:
            df = pd.DataFrame(values.T)
            if "rownames" in grp:
                rownames = [n.decode() if isinstance(n, bytes) else n for n in grp["rownames"][:]]
                df.index = rownames
            if "colnames" in grp:
                colnames = [n.decode() if isinstance(n, bytes) else n for n in grp["colnames"][:]]
                df.columns = colnames
            data[key] = df
        elif len(values.shape) == 1:
            if values.dtype.kind == "S":
                data[key] = [v.decode() if isinstance(v, bytes) else v for v in values]
            else:
                data[key] = values
        elif len(values.shape) == 0:
            data[key] = values.item() if hasattr(values, 'item') else values
        else:
            data[key] = values
    f.close()
    return data


def write_h5_output(path, proportions, method=METHOD_NAME):
    """Write proportions to H5 in DeconBenchmark format."""
    print(f"{method} Writing results to {path}")
    if os.path.exists(path):
        os.remove(path)
    f = h5py.File(path, "w")
    grp = f.create_group("P")
    grp.create_dataset("values", data=proportions.values.T)
    grp.create_dataset("rownames", data=np.array(proportions.index.tolist(), dtype="S"))
    grp.create_dataset("colnames", data=np.array(proportions.columns.tolist(), dtype="S"))
    f.close()


def main():
    print(f"{METHOD_NAME}: Reading input from {INPUT_PATH}")
    args = read_h5_input(INPUT_PATH)
    print(f"  Available keys: {list(args.keys())}")

    bulk = args.get("bulk")
    sc_expr = args.get("singleCellExpr")
    sc_labels = args.get("singleCellLabels")

    if bulk is None or sc_expr is None or sc_labels is None:
        print("ERROR: Missing required inputs (bulk, singleCellExpr, singleCellLabels)")
        sys.exit(1)

    print(f"  Bulk: {bulk.shape}")
    print(f"  scRNA: {sc_expr.shape}")
    print(f"  Cell types: {len(set(sc_labels))}")

    # read_h5_input transposes DeconBenchmark H5 data:
    #   H5 stores (cells, genes) for scRNA → read_h5_input returns (genes, cells)
    #   H5 stores (samples, genes) for bulk  → read_h5_input returns (genes, samples)
    # Sweetwater's generate_synthetic expects (cells, genes) with cell type index.
    # Transform: transpose to (cells, genes), set cell type labels as index.
    sc_df = sc_expr.copy().T
    n_cells = sc_df.shape[0]
    sc_df.index = sc_labels if len(sc_labels) == n_cells else sc_labels[:n_cells]

    # Bulk: transpose to (samples, genes) for transform_and_normalize

    # Import Sweetwater modules
    sys.path.insert(0, "/code/sweetwater")
    from data_utils import generate_synthetic, transform_and_normalize, convert_to_float_tensors
    from model import SweetWater

    # Split scRNA into train/test
    from sklearn.model_selection import train_test_split
    sc_train, sc_test = train_test_split(
        sc_df.copy(), stratify=sc_df.index, test_size=0.2, random_state=42
    )

    # Generate pseudo-bulk
    nsamples = min(5000, max(1000, sc_df.shape[0] * 10))
    train_size = int(nsamples * 0.8)
    test_size = nsamples - train_size

    print(f"  Generating {nsamples} pseudo-bulk samples...")
    xtrain, ytrain, celltypes = generate_synthetic(sc_train, nsamples=train_size)
    xtest, ytest, _ = generate_synthetic(sc_test, nsamples=test_size)

    print(f"  Train: {xtrain.shape}, Test: {xtest.shape}")
    print(f"  Cell types ({len(celltypes)}): {celltypes}")

    # Transform and normalize
    print("  Normalizing data...")
    bulk_values = bulk.values.T if hasattr(bulk, 'values') else bulk.T
    print(f"  bulk_values shape: {bulk_values.shape}")
    xtrain, xtest, xbulk = transform_and_normalize(xtrain, xtest, bulk_values)
    print(f"  After transform: xtrain={xtrain.shape}, xtest={xtest.shape}, xbulk={xbulk.shape}")

    # Convert to tensors
    xtrain, ytrain, xtest, ytest, xbulk = convert_to_float_tensors(
        xtrain, ytrain, xtest, ytest, xbulk
    )
    print(f"  After tensor conversion: xbulk.shape={xbulk.shape}")

    # Initialize and run Sweetwater
    epochs = max(100, round(30000 / (xtrain.shape[0] / 256)))
    print(f"  Training Sweetwater ({epochs} epochs)...")

    sw = SweetWater(
        data=(xtrain, ytrain, xtest, ytest),
        bulkrna=xbulk,
        name="deconv",
        verbose=True,
        lr=0.00001,
        batch_size=256,
        epochs=epochs
    )
    sw.run()
    print("  Training complete!")

    # Predict on bulk
    print("  Predicting on bulk samples...")
    ypred = sw.aemodel(xbulk.to(sw.device), mode='phase3')
    ypred_np = ypred.detach().cpu().numpy()

    # Create output DataFrame
    print(f"  bulk.columns={bulk.columns.tolist()}, len={len(bulk.columns)}")
    print(f"  ypred_np shape={ypred_np.shape}")
    result = pd.DataFrame(ypred_np, index=bulk.columns, columns=celltypes)
    print(f"  Predictions shape: {result.shape}")

    write_h5_output(OUTPUT_PATH, result)
    print(f"{METHOD_NAME}: Done!")


if __name__ == "__main__":
    main()
