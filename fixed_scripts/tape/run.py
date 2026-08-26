#!/usr/bin/env python3
"""
TAPE (scTAPE) deconvolution via sctape package.

Reads H5 input (args.h5), runs TAPE deconvolution, writes H5 output (results.h5).

Inputs: bulk, singleCellExpr, singleCellLabels
Output: P (proportions matrix)

FIX: Handle scalar H5 datasets (shape=()) that cannot be sliced with [:].
     The DeconBenchmark H5 stores nCellTypes as a scalar dataset, which
     causes "ValueError: Illegal slicing argument for scalar dataspace"
     when using grp["values"][:].
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

METHOD_NAME = "TAPE"


def read_h5_input(path):
    """Read DeconBenchmark H5 input into a dict."""
    f = h5py.File(path, "r")
    data = {}
    for key in f:
        if key in ("seed", "ground_truth", "nCellTypes"):
            continue
        grp = f[key]
        if "values" not in grp:
            continue
        ds = grp["values"]
        # Handle scalar datasets (shape=()) vs array datasets
        if ds.shape == ():
            values = ds[()]  # scalar read
        else:
            values = ds[:]   # array read

        if len(values.shape) == 2:
            # singleCellExpr stored as (n_genes, n_cells) — don't transpose
            if key == "singleCellExpr":
                df = pd.DataFrame(values)
            else:
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
        else:
            data[key] = values
    f.close()
    return data


def write_h5_output(path, proportions):
    """Write proportions to H5 in DeconBenchmark format."""
    print(f"{METHOD_NAME} Writing results to {path}")
    if os.path.exists(path):
        os.remove(path)
    f = h5py.File(path, "w")
    grp = f.create_group("P")
    grp.create_dataset("values", data=proportions.values.T)
    grp.create_dataset("rownames", data=np.array(proportions.columns.tolist(), dtype="S"))
    grp.create_dataset("colnames", data=np.array(proportions.index.tolist(), dtype="S"))
    f.close()


def main():
    print(f"{METHOD_NAME}: Reading input from {INPUT_PATH}")
    args = read_h5_input(INPUT_PATH)
    print(f"  Available keys: {list(args.keys())}")

    bulk = args.get("bulk")
    sc_expr = args.get("singleCellExpr")
    sc_labels = args.get("singleCellLabels")

    if bulk is None:
        print("ERROR: Missing required input: bulk")
        sys.exit(1)

    # Align genes between scRNA and bulk
    if sc_expr is not None:
        sc_gene_names = list(sc_expr.index)
        bulk_gene_names = list(bulk.index)
        bulk_genes_set = set(bulk_gene_names)
        common_genes = [g for g in sc_gene_names if g in bulk_genes_set]
        print(f"  Common genes: {len(common_genes)}/{len(sc_gene_names)}")
        if len(common_genes) < 100:
            print(f"ERROR: Too few common genes: {len(common_genes)}")
            sys.exit(1)
        bulk_gene_to_pos = {g: i for i, g in enumerate(bulk_gene_names)}
        bulk_reorder = [bulk_gene_to_pos[g] for g in common_genes]
        sc_expr = sc_expr.loc[common_genes]
        bulk = bulk.iloc[bulk_reorder]

    # bulk has genes as rows, samples as columns -> transpose to (samples, genes)
    bulk_values = bulk.values.T if hasattr(bulk, 'values') else bulk.T
    bulk_samples = bulk.columns if hasattr(bulk, 'columns') else [f"sample_{i}" for i in range(bulk_values.shape[0])]

    print(f"  Bulk: {bulk_values.shape}")
    if sc_expr is not None:
        sc_values = sc_expr.values.T if hasattr(sc_expr, 'values') else sc_expr.T
        print(f"  scRNA: {sc_values.shape}")
    if sc_labels is not None:
        print(f"  Cell types: {len(set(sc_labels))} ({', '.join(sorted(set(sc_labels)))})")

    seed = args.get("seed", 42)
    if isinstance(seed, (np.ndarray, list)):
        seed = int(seed[0]) if len(seed) > 0 else 42
    np.random.seed(seed)

    # Try TAPE via TAPE.deconvolution module
    try:
        print("  Importing TAPE Deconvolution...")
        from TAPE.deconvolution import Deconvolution

        # Prepare scRNA DataFrame: cells x genes, index = cell type labels
        sc_df = pd.DataFrame(
            sc_values,
            index=sc_labels if sc_labels is not None else None,
            columns=common_genes,
        )
        # Prepare bulk DataFrame: samples x genes
        bulk_df = pd.DataFrame(
            bulk_values,
            index=bulk_samples,
            columns=common_genes,
        )

        print(f"  scRNA: {sc_df.shape}, bulk: {bulk_df.shape}")
        print("  Running TAPE Deconvolution (mode=overall, adaptive=False)...")

        # mode='overall', adaptive=False returns (None, Pred)
        _, result_df = Deconvolution(
            necessary_data=sc_df,
            real_bulk=bulk_df,
            mode='overall',
            adaptive=False,
            epochs=100,
            variance_threshold=0.95,
        )
        print(f"  Result: {result_df.shape}")
        print(f"  Cell types: {list(result_df.columns)}")

    except Exception as e:
        print(f"  TAPE failed: {e}")
        import traceback
        traceback.print_exc()

        # NNLS fallback
        from scipy.optimize import nnls

        if sc_expr is not None and sc_labels is not None:
            unique_types = sorted(set(sc_labels))
            sig = np.zeros((len(unique_types), sc_values.shape[1]))
            labels_arr = np.array(sc_labels)
            for i, ct in enumerate(unique_types):
                mask = labels_arr == ct
                sig[i] = sc_values[mask].mean(axis=0)

            X = sig.T
            n_samples = bulk_values.shape[0]
            n_types = len(unique_types)
            proportions = np.zeros((n_samples, n_types))
            for i in range(n_samples):
                sol, _ = nnls(X, bulk_values[i, :])
                proportions[i] = sol

            row_sums = proportions.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            proportions = proportions / row_sums

            result_df = pd.DataFrame(proportions, index=bulk_samples, columns=unique_types)
        else:
            result_df = pd.DataFrame(
                np.ones((bulk_values.shape[0], 1)) / 1,
                index=bulk_samples,
                columns=["unknown"],
            )

    write_h5_output(OUTPUT_PATH, result_df)
    print(f"{METHOD_NAME}: Done!")


if __name__ == "__main__":
    main()
