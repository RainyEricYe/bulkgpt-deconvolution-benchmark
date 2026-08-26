#!/usr/bin/env python3
"""Deconformer wrapper for DeconBenchmark H5 I/O — with python symlink fix."""

import os
import sys
import subprocess
import tempfile
import h5py
import pandas as pd
import numpy as np


INPUT_PATH = os.environ.get("INPUT_PATH", "")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "adult_model")


def main():
    if not INPUT_PATH or not OUTPUT_PATH:
        print("ERROR: INPUT_PATH and OUTPUT_PATH env vars required", file=sys.stderr)
        sys.exit(1)

    # Fix: deconformer_predict.sh calls 'python' but container only has 'python3'
    # Create a wrapper script in /tmp so it works without root permissions
    wrapper = "/tmp/python"
    if not os.path.exists(wrapper):
        with open(wrapper, "w") as f:
            f.write("#!/bin/bash\nexec /usr/bin/python3 \"$@\"\n")
        os.chmod(wrapper, 0o755)
    os.environ["PATH"] = f"/tmp:{os.environ.get('PATH', '')}"

    print(f"[Deconformer] Reading H5 input: {INPUT_PATH}")
    with h5py.File(INPUT_PATH, "r") as f:
        bulk_values = f["bulk/values"][:]
        bulk_genes = f["bulk/genes"][:] if "bulk/genes" in f else [f"gene_{i}" for i in range(bulk_values.shape[0])]
        bulk_genes = [g.decode() if isinstance(g, bytes) else g for g in bulk_genes]

    print(f"[Deconformer] Bulk shape: {bulk_values.shape}, genes: {len(bulk_genes)}")
    n_genes, n_samples = bulk_values.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input TSV (genes x samples, first column = gene names)
        input_tsv = os.path.join(tmpdir, "input.tsv")
        df = pd.DataFrame(bulk_values, index=bulk_genes,
                          columns=[f"sample_{i}" for i in range(n_samples)])
        df.index.name = "Gene Name (HUGO)"
        df.to_csv(input_tsv, sep="\t")
        print(f"[Deconformer] Input saved: {input_tsv}")

        # Run Deconformer
        output_tsv = os.path.join(tmpdir, "output.tsv")
        cmd = ["bash", "/deconformer/deconformer_predict.sh",
               MODEL_NAME, input_tsv, output_tsv]
        print(f"[Deconformer] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
        print(result.stdout)
        if result.stderr:
            print(f"[Deconformer] STDERR: {result.stderr}", file=sys.stderr)

        if result.returncode != 0:
            print(f"[Deconformer] Failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(1)

        if not os.path.exists(output_tsv):
            print(f"[Deconformer] Output not found: {output_tsv}", file=sys.stderr)
            sys.exit(1)

        # Read output TSV (rows = cell types, columns = samples)
        print(f"[Deconformer] Reading output: {output_tsv}")
        pred_df = pd.read_csv(output_tsv, sep="\t", index_col=0)
        print(f"[Deconformer] Predictions shape: {pred_df.shape}")
        print(f"[Deconformer] Cell types: {pred_df.index.tolist()}")

        # Write to H5 output
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        with h5py.File(OUTPUT_PATH, "w") as f:
            p_values = pred_df.values.astype(np.float64)
            col_sums = p_values.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            p_values = p_values / col_sums

            f.create_dataset("P/values", data=p_values)
            f.create_dataset("P/rows", data=[t.encode() if isinstance(t, str) else t
                                              for t in pred_df.index.tolist()])
            f.create_dataset("P/cols", data=[f"sample_{i}".encode()
                                              for i in range(p_values.shape[1])])

    print(f"[Deconformer] Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
