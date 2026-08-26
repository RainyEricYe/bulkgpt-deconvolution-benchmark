#!/usr/bin/env python3
"""
MuSic — Multi-subject single-cell deconvolution (R package via Apptainer).

Runs the MuSic R package via an Apptainer SIF container. MuSic uses
multi-subject weighting from donor_id for improved accuracy in
deconvolution of bulk RNA-seq using scRNA-seq reference.

Usage
-----
    python run.py --config configs/default.yaml --mode predict
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", HERE.parent.parent)).resolve()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(
        description="MuSic (R package via Apptainer) — Bulk deconvolution")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--mode", type=str, default="predict", choices=["predict", "all"],
                   help="Execution mode")
    p.add_argument("--sc-ref", type=str, default=None)
    p.add_argument("--bulk", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--sif", type=str, default=None)
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--skip-container", action="store_true",
                   help="Skip container run (for testing)")
    return p.parse_args()


def write_h5_input(h5_path, bulk_values, cell_type_labels,
                   sc_expr_values, sc_gene_names, bulk_sample_names):
    """Write H5 file in DeconBenchmark format for MuSic.
    sc_expr_values: (n_cells, n_genes) — caller must transpose if needed.
    """
    import h5py
    if os.path.exists(h5_path):
        os.remove(h5_path)

    n_cells = sc_expr_values.shape[0]
    unique_types = sorted(set(cell_type_labels))

    with h5py.File(h5_path, "w") as f:
        f.create_group("bulk")
        f.create_dataset("bulk/values", data=bulk_values.astype(np.float64))
        f.create_dataset("bulk/rownames", data=np.array(sc_gene_names, dtype="S"))
        f.create_dataset("bulk/colnames", data=np.array(bulk_sample_names, dtype="S"))

        f.create_group("singleCellExpr")
        f.create_dataset("singleCellExpr/values", data=sc_expr_values.astype(np.float64))
        f.create_dataset("singleCellExpr/rownames", data=np.array(sc_gene_names, dtype="S"))
        f.create_dataset("singleCellExpr/colnames",
                         data=np.array([f"cell_{i}" for i in range(n_cells)], dtype="S"))

        f.create_group("singleCellLabels")
        f.create_dataset("singleCellLabels/values",
                         data=np.array(cell_type_labels, dtype="S"))
        f.create_group("nCellTypes")
        f.create_dataset("nCellTypes/values", data=len(unique_types))

    print(f"  H5 written: {h5_path}")
    print(f"    bulk: {bulk_values.shape}, scExpr: {sc_expr_values.shape}, "
          f"scLabels: {n_cells} cells, {len(unique_types)} types")
    return unique_types


def run_music_container(h5_path, sif_path, timeout=7200):
    """Run MuSic container via Apptainer."""
    if not os.path.exists(sif_path):
        raise FileNotFoundError(f"MuSic SIF not found at {sif_path}")

    out_dir = str(Path(h5_path).parent)
    results_h5 = os.path.join(out_dir, "music_results.h5")

    cmd = [
        "apptainer", "run", "--cleanenv",
        "--bind", f"{h5_path}:/input/args.h5",
        "--bind", f"{out_dir}:/output",
        "--env", "INPUT_PATH=/input/args.h5",
        "--env", f"OUTPUT_PATH=/output/music_results.h5",
        str(sif_path),
    ]

    print(f"  Running MuSic container (timeout={timeout}s)...")
    sys.stdout.flush()
    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - start

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    [stdout] {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            print(f"    [stderr] {line}")
    print(f"  Return code: {result.returncode}, Elapsed: {elapsed:.1f}s")

    if result.returncode != 0 and not os.path.exists(results_h5):
        raise RuntimeError(f"MuSic failed (rc={result.returncode})")

    if not os.path.exists(results_h5):
        raise FileNotFoundError(f"Results H5 not found at {results_h5}")

    return results_h5, elapsed


def read_music_output(results_h5):
    """Read MuSic output H5 and return (proportions, cell_type_names)."""
    import h5py

    with h5py.File(results_h5, "r") as f:
        if "P/values" not in f:
            raise KeyError(f"No P/values in results. Groups: {list(f.keys())}")
        P = np.asarray(f["P/values"][:], dtype=np.float64)
        colnames = [x.decode() for x in f["P/colnames"][:]] if "P/colnames" in f else None
        rownames = [x.decode() for x in f["P/rownames"][:]] if "P/rownames" in f else None

    # Transpose: R writes (samples, types), h5py reads as (types, samples)
    P = P.T
    cell_types = colnames or [f"type_{i}" for i in range(P.shape[1])]
    print(f"  P matrix: {P.shape} (samples={P.shape[0]}, types={P.shape[1]})")
    return P, cell_types, rownames


def main():
    args = parse_args()
    cfg = load_config(args.config)

    for opt, key in [("sc_ref", "sc_ref"), ("bulk", "bulk"),
                     ("sif", "sif_path"), ("output", "output")]:
        val = getattr(args, opt, None)
        if val is not None:
            cfg["paths"][key] = val

    if args.max_cells is not None:
        cfg["data"]["max_cells"] = args.max_cells
    if args.skip_container:
        cfg["container"]["skip"] = True

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    sc_ref_path = _resolve(cfg["paths"]["sc_ref"])
    bulk_path = _resolve(cfg["paths"]["bulk"])
    sif_path = _resolve(cfg["paths"]["sif_path"])
    output_path = _resolve(cfg["paths"]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = cfg["container"].get("timeout", 7200)

    print("=" * 60)
    print("MuSic Deconvolution (R via Apptainer)")
    print("=" * 60)

    # Step 1: Load reference
    import anndata as ad
    from scipy.sparse import issparse

    print(f"\n[1] Loading scRNA-seq reference: {sc_ref_path}")
    if str(sc_ref_path).endswith(".h5"):
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.data_loader import load_sc_ref
        ref = load_sc_ref(str(sc_ref_path))
    else:
        ref = ad.read_h5ad(str(sc_ref_path))
    ref.obs_names_make_unique()
    print(f"  Shape: {ref.shape}")

    celltype_col = cfg["data"].get("celltype_col", "cell_type")
    if celltype_col not in ref.obs.columns:
        raise ValueError(f"'{celltype_col}' not in obs. Available: {list(ref.obs.columns)}")

    donor_col = cfg["data"].get("donor_col", "donor_id")
    has_donor = donor_col in ref.obs.columns
    print(f"  Donor column: '{donor_col}' (available={has_donor})")

    # Optional subsample
    max_cells = cfg["data"].get("max_cells")
    if max_cells is not None and ref.n_obs > max_cells:
        print(f"  Subsampling to {max_cells} cells...")
        rng = np.random.RandomState(42)
        idx = rng.choice(ref.n_obs, max_cells, replace=False)
        ref = ref[idx].copy()

    X = ref.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    gene_ids = list(ref.var_names)
    gene_symbols = list(ref.var.get("gene_symbol", ref.var_names)) if "gene_symbol" in ref.var else gene_ids
    barcodes = list(ref.obs_names)
    cell_types_labels = ref.obs[celltype_col].values

    # Step 2: Load bulk
    print(f"\n[2] Loading bulk: {bulk_path}")
    bulk_path_str = str(bulk_path)
    if bulk_path_str.endswith(".h5"):
        import h5py
        with h5py.File(bulk_path_str, "r") as f:
            bx = np.asarray(f["bulk/values"][:], dtype=np.float64)
            bulk_symbols = [x.decode() for x in f["bulk/rownames"][:]]
            bulk_barcodes = [x.decode() for x in f["bulk/colnames"][:]]
            # Ensure bulk is (n_samples, n_genes): transpose if stored as (n_genes, n_samples)
            if len(bulk_symbols) > 0 and bx.shape[0] == len(bulk_symbols) and bx.shape[1] < len(bulk_symbols):
                bx = bx.T
        bulk_ids = bulk_symbols
        bx = np.asarray(bx, dtype=np.float64)
        bulk_var_names = bulk_symbols
    else:
        bulk = ad.read_h5ad(bulk_path_str)
        bulk.obs_names_make_unique()
        bx = bulk.X
        if issparse(bx):
            bx = bx.toarray()
        bx = np.asarray(bx, dtype=np.float64)
        bulk_ids = list(bulk.var_names)
        bulk_symbols = list(bulk.var.get("gene_symbol", bulk.var_names)) if "gene_symbol" in bulk.var else bulk_ids
        bulk_barcodes = list(bulk.obs_names)
        bulk_var_names = list(bulk.var_names)

    # Step 3: Align genes
    print(f"\n[3] Aligning genes...")
    common = sorted(set(ref.var_names) & set(bulk_var_names))
    if len(common) < 100:
        # Try symbol alignment
        common = sorted(set(gene_symbols) & set(bulk_symbols))
        if len(common) < 100:
            raise ValueError(f"Too few common genes ({len(common)})")
        # Map symbols back to indices
        ref_sym2idx = {s: i for i, s in enumerate(gene_symbols)}
        bulk_sym2idx = {s: i for i, s in enumerate(bulk_symbols)}
        ref_aligned = X[:, [ref_sym2idx[s] for s in common]].T  # (n_genes, n_cells)
        bulk_aligned = bx[:, [bulk_sym2idx[s] for s in common]]  # (n_samples, n_genes)
    else:
        ref_idx = [list(ref.var_names).index(g) for g in common]
        bulk_idx = [list(bulk_var_names).index(g) for g in common]
        ref_aligned = X[:, ref_idx].T
        bulk_aligned = bx[:, bulk_idx]

    print(f"  Common genes: {len(common)}")
    print(f"  Ref aligned: {ref_aligned.shape}, Bulk aligned: {bulk_aligned.shape}")

    # Step 4: Write H5 input
    print(f"\n[4] Writing H5 input...")
    h5_path = str(output_path.parent / "music_input.h5")
    unique_types = write_h5_input(
        h5_path=h5_path,
        bulk_values=bulk_aligned,
        cell_type_labels=cell_types_labels,
        sc_expr_values=ref_aligned.T,  # (n_cells, n_genes) for R transposition
        sc_gene_names=common,
        bulk_sample_names=bulk_barcodes,
    )

    # Step 5: Run container
    print(f"\n[5] Running MuSic container...")
    skip = cfg["container"].get("skip", False) or args.skip_container

    if skip:
        print(f"  SKIP_CONTAINER — creating dummy proportions")
        dummy = np.random.dirichlet(np.ones(len(unique_types)), size=bulk_aligned.shape[0])
        proportions, cell_types = dummy, unique_types
        elapsed = 0.0
    else:
        results_h5, elapsed = run_music_container(h5_path, str(sif_path), timeout)
        proportions, cell_types, sample_names = read_music_output(results_h5)

        # Align cell type names with canonical order
        ct_map = {ct.lower().replace("-", " ").replace(" ", ""): ct
                  for ct in unique_types}
        aligned_types = []
        for ct in cell_types:
            key = ct.lower().replace("-", " ").replace(" ", "")
            aligned_types.append(ct_map.get(key, ct))
        cell_types = aligned_types

    # Save
    print(f"\n[6] Saving results...")
    sample_names_out = bulk_barcodes
    pdf = pd.DataFrame(proportions, index=sample_names_out, columns=cell_types)
    pdf.to_csv(str(output_path))
    print(f"  Saved -> {output_path}")
    print(f"  Shape: {pdf.shape}")
    print("\nMean proportions:")
    for ct in cell_types:
        print(f"  {ct:25s}: {proportions[:, cell_types.index(ct)].mean():.4f}")

    with open(output_path.parent / "cell_types.json", "w") as f:
        json.dump(unique_types, f)

    meta = {"elapsed_s": elapsed}
    with open(output_path.parent / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Output in {output_path.parent}/")


if __name__ == "__main__":
    main()
