#!/usr/bin/env python3
"""
MixupVI deconvolution via PyDeconv.

Reads H5 input (args.h5), runs MixupVI on bulk data using the pre-trained
Cross-Tissue Immune (CTI) model, and writes H5 output (results.h5).

MixupVI is a reference-free method: it uses a pre-trained VAE to extract
latent features, then applies NNLS with a pre-computed latent signature
matrix to estimate cell-type proportions.

Inputs: bulk, singleCellExpr (optional), singleCellLabels (optional)
Output: P (proportions matrix)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import h5py

warnings.filterwarnings("ignore")

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/args.h5")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.h5")

METHOD_NAME = "MixupVI"


def read_h5_input(path):
    """Read DeconBenchmark H5 input into a dict.

    Handles mixed orientations: bulk and singleCellExpr store
    rownames=genes with values (observations, features), while
    ground_truth stores rownames=cell_types with values (n_types, n_samples).
    Transpose or not based on whether rownames match, to avoid
    dimension mismatch errors.
    """
    f = h5py.File(path, "r")
    data = {}
    for key in f:
        if key == "seed":
            continue
        grp = f[key]
        if "values" not in grp:
            continue
        try:
            values = grp["values"][:]
        except (TypeError, ValueError, OSError):
            continue

        if len(values.shape) == 2:
            rownames = None
            colnames = None
            if "rownames" in grp:
                rownames = [n.decode() if isinstance(n, bytes) else n for n in grp["rownames"][:]]
            if "colnames" in grp:
                colnames = [n.decode() if isinstance(n, bytes) else n for n in grp["colnames"][:]]

            # Determine orientation: try transpose first (bulk convention),
            # fall back to no-transpose if rownames don't match.
            nrow, ncol = values.shape
            if rownames is not None and len(rownames) == ncol:
                df = pd.DataFrame(values.T, index=rownames)
            elif rownames is not None and len(rownames) == nrow:
                df = pd.DataFrame(values, index=rownames)
            else:
                df = pd.DataFrame(values.T)

            if colnames is not None and len(colnames) == df.shape[1]:
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


def fallback_nnls(bulk: pd.DataFrame, sc_expr: pd.DataFrame, sc_labels) -> pd.DataFrame:
    """Fallback: build signature matrix from scRNA and run NNLS.

    H5 format: rownames=genes, colnames=cells for sc_expr.
    So sc_expr has genes as rows, cells as columns.
    We need (cells x genes) for building signature matrix.
    """
    print("  Using NNLS fallback with signature from scRNA reference...")

    # Build signature: mean expression per cell type
    # sc_expr has genes as rows, cells as columns -> transpose to (cells, genes)
    sc_values = sc_expr.values.T  # (n_cells, n_genes)
    labels = np.array(sc_labels)
    unique_types = sorted(set(labels))
    sig = np.zeros((len(unique_types), sc_values.shape[1]))
    for i, ct in enumerate(unique_types):
        mask = labels == ct
        sig[i] = sc_values[mask].mean(axis=0)

    # bulk has genes as rows, samples as columns -> transpose to (samples, genes)
    bulk_values = bulk.values.T  # (n_samples, n_genes)

    # NNLS via non-negative least squares (scipy)
    from scipy.optimize import nnls

    X = sig.T  # (n_genes, n_cell_types)
    n_samples = bulk_values.shape[0]
    n_types = len(unique_types)
    proportions = np.zeros((n_samples, n_types))
    for i in range(n_samples):
        y = bulk_values[i, :]
        sol, _ = nnls(X, y)
        proportions[i] = sol

    # Normalize to sum-to-1
    row_sums = proportions.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    proportions = proportions / row_sums

    return pd.DataFrame(proportions, index=bulk.columns, columns=unique_types)


def main():
    print(f"{METHOD_NAME}: Reading input from {INPUT_PATH}")
    args = read_h5_input(INPUT_PATH)
    print(f"  Available keys: {list(args.keys())}")

    bulk = args.get("bulk")
    if bulk is None:
        print("ERROR: Missing required input: bulk")
        sys.exit(1)

    print(f"  Bulk: {bulk.shape}")

    # Try MixupVI via pydeconv
    try:
        print("  Installing/loading pydeconv...")
        import pydeconv
        print(f"  pydeconv version: {pydeconv.__version__}")

        import torch
        import anndata as ad
        from pydeconv.model.mixupvi import MixupVI

        print("  Loading pre-trained MixupVI model (CTI 1st granularity)...")

        # Create AnnData for bulk samples
        adata_bulk = ad.AnnData(
            X=bulk.values.astype(np.float32),
            obs=pd.DataFrame(index=bulk.index),
            var=pd.DataFrame(index=bulk.columns),
        )
        adata_bulk.layers["raw_counts"] = adata_bulk.X.copy()

        # Initialize MixupVI with pre-trained weights
        model = MixupVI(weights_version="cti_1st_granularity")

        # Run deconvolution
        print("  Running MixupVI deconvolution...")
        result = model.transform(adata_bulk, layer="raw_counts", ratio=True)
        print(f"  Result shape: {result.shape}")
        print(f"  Cell types: {list(result.columns)}")

    except ImportError as e:
        print(f"  Could not load pydeconv: {e}")
        sc_expr = args.get("singleCellExpr")
        sc_labels = args.get("singleCellLabels")
        if sc_expr is not None and sc_labels is not None:
            print("  Falling back to NNLS with scRNA signature...")
            result = fallback_nnls(bulk, sc_expr, sc_labels)
        else:
            print("  ERROR: No scRNA reference available for fallback.")
            result = pd.DataFrame(
                np.ones((bulk.shape[0], 1)) / 1,
                index=bulk.index,
                columns=["unknown"],
            )
    except Exception as e:
        print(f"  MixupVI failed: {e}")
        sc_expr = args.get("singleCellExpr")
        sc_labels = args.get("singleCellLabels")
        if sc_expr is not None and sc_labels is not None:
            print("  Falling back to NNLS with scRNA signature...")
            result = fallback_nnls(bulk, sc_expr, sc_labels)
        else:
            print(f"  Returning uniform proportions as fallback")
            n_types = len(set(args.get("singleCellLabels", ["unknown"])))
            result = pd.DataFrame(
                np.ones((bulk.shape[0], n_types)) / n_types,
                index=bulk.index,
                columns=[f"type_{i}" for i in range(n_types)],
            )

    write_h5_output(OUTPUT_PATH, result)
    print(f"{METHOD_NAME}: Done!")


if __name__ == "__main__":
    main()
