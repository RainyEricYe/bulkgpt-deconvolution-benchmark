#!/usr/bin/env python3
"""
TAPE (scTAPE) — autoencoder + GAN deconvolution (native mode).

Runs TAPE directly using the sctape PyPI package installed in the conda
environment, instead of via Apptainer container.  Handles both pseudo-bulk
(--data h5ad) and real-bulk (--h5 DeconBenchmark H5) modes.

Usage
-----
    python run.py --config configs/default.yaml --mode train \\
        --data data/Liver.h5ad --output-dir results/synthetic/Liver/tape

    python run.py --config configs/default.yaml --mode predict \\
        --h5 data/real_bulk/sdy67.h5 \\
        --ground-truth data/real_bulk/ground_truth.csv \\
        --output-dir results/real_bulk/sdy67/tape
"""
import argparse
import json
import os
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
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
METHOD_NAME = "TAPE"


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=f"{METHOD_NAME} deconvolution (native)")
    p.add_argument("--config", required=True)
    p.add_argument("--mode", default="train", choices=["train", "predict"])
    p.add_argument("--data", default=None, help="h5ad scRNA-seq reference (pseudo-bulk mode)")
    p.add_argument("--h5", default=None, help="Pre-built DeconBenchmark H5 (real bulk mode)")
    p.add_argument("--ground-truth", default=None, help="CSV with ground-truth proportions")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpu", action="store_true", default=None)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p):
    p = Path(str(p))
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_deconbenchmark_h5(path):
    """Read DeconBenchmark-format H5 into a dict of numpy arrays/pandas DataFrames.

    Expected groups:
        singleCellExpr/values  (cells, genes) float64
        singleCellLabels/values  (cells,) string
        bulk/values  (samples, genes) float64
        bulk/colnames  (genes,) string
        bulk/rownames  (samples,) string
        (optional) ground_truth/...
        (optional) seed/...
    """
    import h5py

    data = {}
    with h5py.File(path, "r") as f:
        for key in f:
            if key in ("seed", "nCellTypes"):
                continue
            grp = f[key]
            if "values" not in grp:
                continue
            values = grp["values"][:]

            if len(values.shape) == 2:
                # Transpose to match container's read_h5_input convention:
                # H5 stores (cells × genes) or (samples × genes).
                # After .T, rows = genes, columns = cells/samples.
                df = pd.DataFrame(values.T)
                if "rownames" in grp:
                    rn = [n.decode() if isinstance(n, bytes) else n for n in grp["rownames"][:]]
                    if len(rn) == df.shape[0]:
                        df.index = rn
                if "colnames" in grp:
                    cn = [n.decode() if isinstance(n, bytes) else n for n in grp["colnames"][:]]
                    if len(cn) == df.shape[1]:
                        df.columns = cn
                data[key] = df
            elif len(values.shape) == 1:
                if values.dtype.kind == "S":
                    data[key] = [v.decode() if isinstance(v, bytes) else v for v in values]
                else:
                    data[key] = values
            else:
                data[key] = values
    return data


def generate_sc_ref_dataframe(sc_expr, sc_labels):
    """Build (cells, genes) DataFrame with cell-type labels as row index."""
    df = sc_expr.copy()
    if isinstance(sc_labels, (list, np.ndarray)):
        df.index = list(sc_labels)
    elif isinstance(sc_labels, pd.Series):
        df.index = sc_labels.values
    return df


def generate_pseudo_bulk(adata, n_samples=2000, n_cells_per_sample=80, celltype_col="cell_type"):
    """Split scRNA-seq reference, generate pseudo-bulk mixtures from test cells."""
    from sklearn.model_selection import train_test_split
    from scipy.sparse import issparse

    rng = np.random.RandomState(SEED)

    if celltype_col not in adata.obs.columns:
        for col in ["CellType", "celltype", "cell.type", "label", "cluster"]:
            if col in adata.obs.columns:
                celltype_col = col
                break

    cell_types = adata.obs[celltype_col].values
    type_list = sorted(set(cell_types))
    n_types = len(type_list)
    print(f"  Cell types: {n_types} ({', '.join(type_list)})")

    train_idx, test_idx = train_test_split(
        np.arange(adata.n_obs), test_size=0.2, random_state=SEED, stratify=cell_types
    )
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()
    print(f"  Train cells: {train_adata.n_obs}, Test cells: {test_adata.n_obs}")

    X_train = train_adata.X
    X_test = test_adata.X
    if issparse(X_train):
        X_train = X_train.toarray()
    if issparse(X_test):
        X_test = X_test.toarray()
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    train_labels = train_adata.obs[celltype_col].values
    test_labels = test_adata.obs[celltype_col].values

    type_to_idx = {t: i for i, t in enumerate(type_list)}
    test_type_indices = np.array([type_to_idx[t] for t in test_labels])

    proportions = np.zeros((n_samples, n_types))
    mixtures = np.zeros((n_samples, X_test.shape[1]))

    print(f"  Generating {n_samples} pseudo-bulk mixtures...")
    for i in range(n_samples):
        alpha = np.ones(n_types) * 0.1
        p = rng.dirichlet(alpha)
        proportions[i] = p

        n_cells = max(10, int(rng.poisson(n_cells_per_sample)))
        selected_types = rng.choice(n_types, size=n_cells, p=p)

        mix = np.zeros(X_test.shape[1])
        for ct_idx in selected_types:
            cell_mask = test_type_indices == ct_idx
            if cell_mask.sum() == 0:
                continue
            cell_idx = rng.choice(np.where(cell_mask)[0])
            mix += X_test[cell_idx]

        total = mix.sum()
        if total > 0:
            mix = mix / total * 1e6
        mixtures[i] = np.log1p(mix)

    print(f"  Mixtures: {mixtures.shape}, Proportions: {proportions.shape}")

    return {
        "singleCellExpr": X_train,
        "singleCellLabels": train_labels,
        "bulk": mixtures,
        "bulk_labels": proportions,
        "type_list": type_list,
        "gene_names": list(adata.var_names),
    }


def _run_tape_native(sc_df, bulk_values, bulk_samples, bulk_genes=None, n_epochs=30, seed=SEED):
    """Run TAPE deconvolution using TAPE package directly.

    Parameters
    ----------
    sc_df : pd.DataFrame
        scRNA expression with cell-type labels as index, genes as columns,
        shape (n_cells, n_genes).
    bulk_values : np.ndarray or pd.DataFrame
        Bulk expression, shape (n_samples, n_genes). If DataFrame, index
        is used as sample names.
    bulk_samples : list of str
        Sample names for the bulk data.
    n_epochs : int
        Number of training epochs.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame with samples as index and cell types as columns.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend

    from TAPE.simulation import generate_simulated_data
    from TAPE.utils import ProcessInputData
    from TAPE.train import train_model, predict, reproducibility

    # Generate pseudo-bulk from scRNA reference
    n_pseudo = 200  # small number for quick test
    print(f"  Generating {n_pseudo} pseudo-bulk samples from scRNA reference...")
    sys.stdout.flush()
    simudata = generate_simulated_data(
        sc_data=sc_df, samplenum=n_pseudo,
        sparse=True, random_state=seed,
    )
    print(f"  Pseudo-bulk shape: {simudata.shape}")

    # Build bulk DataFrame for ProcessInputData (genes as columns)
    if isinstance(bulk_values, pd.DataFrame):
        bulk_df = bulk_values
    else:
        gene_names = bulk_genes or list(range(bulk_values.shape[1]))
        bulk_df = pd.DataFrame(bulk_values, index=bulk_samples, columns=gene_names)

    # Process data (gene intersection, scaling)
    print("  Processing input data...")
    sys.stdout.flush()
    train_x, train_y, test_x, genename, celltypes, samplename = ProcessInputData(
        simudata, bulk_df,
        datatype="counts",
        variance_threshold=0.98,
        scaler="mms",
    )
    print(f"  Train: {train_x.shape}, Test: {test_x.shape}, Types: {len(celltypes)}")

    # Train model
    print(f"  Training autoencoder ({n_epochs} epochs)...")
    sys.stdout.flush()
    start = time.monotonic()
    reproducibility(seed)
    model = train_model(train_x, train_y, batch_size=128, epochs=n_epochs)
    elapsed = time.monotonic() - start
    print(f"  Training completed in {elapsed:.1f}s")

    # Predict (non-adaptive)
    print("  Predicting proportions...")
    sys.stdout.flush()
    result = predict(
        test_x=test_x, genename=genename, celltypes=celltypes,
        samplename=samplename, model=model,
        adaptive=False, mode="overall",
    )
    # With adaptive=False, predict returns a single DataFrame
    if isinstance(result, tuple):
        pred = result[1]
    else:
        pred = result

    if isinstance(pred, np.ndarray):
        pred = pd.DataFrame(pred, index=samplename, columns=celltypes)

    return pred


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    np.random.seed(SEED)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"{METHOD_NAME} Deconvolution (native)")
    print("=" * 60)

    # ── Real bulk mode (DeconBenchmark H5) ──
    if args.h5:
        print(f"\n[Input] DeconBenchmark H5: {args.h5}")

        # Read directly from H5 (enriched format: scRNA = (cells, genes),
        # bulk = (genes, samples)) to avoid read_deconbenchmark_h5's
        # transposition issues that lose gene labels.
        import h5py
        with h5py.File(args.h5, "r") as f:
            # scRNA expression
            sce_raw = f["singleCellExpr/values"][:]
            sce_rownames = [x.decode() for x in f["singleCellExpr/rownames"][:]]
            sce_colnames = [x.decode() for x in f["singleCellExpr/colnames"][:]]
            sc_labels = [x.decode() for x in f["singleCellLabels/values"][:]]

            # Determine scRNA orientation: need (cells, genes) with gene names as columns
            n_cells = len(sc_labels)
            if sce_raw.shape[0] == n_cells:
                # (cells, genes): gene names are whichever name list matches shape[1]
                if len(sce_colnames) == sce_raw.shape[1]:
                    gene_names = sce_colnames
                else:
                    gene_names = sce_rownames
                sc_df = pd.DataFrame(sce_raw, index=sc_labels, columns=gene_names)
            else:
                # (genes, cells): transpose, gene names match shape[0]
                if len(sce_rownames) == sce_raw.shape[0]:
                    gene_names = sce_rownames
                else:
                    gene_names = sce_colnames
                sc_df = pd.DataFrame(sce_raw.T, index=sc_labels, columns=gene_names)

            print(f"  scRNA DataFrame: {sc_df.shape} ({len(set(sc_labels))} cell types)")

            # Bulk expression: need (samples, genes) with sample names as index
            bulk_raw = f["bulk/values"][:]
            bulk_rownames = [x.decode() for x in f["bulk/rownames"][:]]
            bulk_colnames = [x.decode() for x in f["bulk/colnames"][:]]

            # Normalize to (samples, genes) for TAPE.
            # Two possible H5 storage formats:
            #   Old enrich_h5: (n_genes, n_samples), rownames=genes, colnames=samples
            #   DeconBenchmark: (n_samples, n_genes), rownames=genes, colnames=samples
            if bulk_raw.shape[0] == len(bulk_rownames) and bulk_raw.shape[1] == len(bulk_colnames):
                # (genes, samples) old format — transpose
                bulk_df = pd.DataFrame(bulk_raw.T, index=bulk_colnames, columns=bulk_rownames)
            else:
                # (samples, genes) DeconBenchmark standard
                bulk_df = pd.DataFrame(bulk_raw, index=bulk_colnames, columns=bulk_rownames)

        bulk_values = bulk_df.values
        bulk_samples = list(bulk_df.index)
        bulk_genes = list(bulk_df.columns)

        print(f"  Bulk: {bulk_values.shape} ({len(bulk_samples)} samples, {len(bulk_genes)} genes)")

        # Run TAPE natively
        print(f"\n[Deconvolution] Running {METHOD_NAME} natively...")
        sys.stdout.flush()
        start = time.monotonic()

        pred_df = _run_tape_native(
            sc_df, bulk_values, bulk_samples, bulk_genes=bulk_genes,
            n_epochs=cfg.get("training", {}).get("epochs", 30),
            seed=SEED,
        )
        elapsed = time.monotonic() - start

        # Save predictions
        pred_csv = out_dir / "proportions.csv"
        pred_df.to_csv(str(pred_csv))
        print(f"  Predictions saved -> {pred_csv}")

        if args.ground_truth and os.path.exists(args.ground_truth):
            print(f"  (ground truth available — metrics computed by post-hoc evaluation)")

    # ── Pseudo-bulk mode ──
    elif args.data:
        print(f"\n[1] Loading scRNA-seq reference: {args.data}")
        from core.data_loader import load_sc_ref
        adata = load_sc_ref(str(args.data))
        print(f"  Shape: {adata.shape}")

        celltype_col = cfg["data"].get("celltype_col", "cell_type")
        if celltype_col not in adata.obs.columns:
            for c in ["CellType", "celltype", "cell.type", "label", "cluster"]:
                if c in adata.obs.columns:
                    celltype_col = c
                    break

        print(f"\n[2] Generating pseudo-bulk mixtures...")
        n_samples = cfg["data"].get("n_pseudo_bulk", 2000)
        pb_data = generate_pseudo_bulk(adata, n_samples=n_samples, celltype_col=celltype_col)
        type_list = pb_data["type_list"]

        # Build scRNA DataFrame (train cells)
        sc_df = pd.DataFrame(
            pb_data["singleCellExpr"],
            index=pb_data["singleCellLabels"],
            columns=pb_data["gene_names"],
        )
        print(f"  scRNA DataFrame: {sc_df.shape}")

        print(f"\n[3] Running {METHOD_NAME} deconvolution...")
        sys.stdout.flush()
        start = time.monotonic()

        pred_df = _run_tape_native(
            sc_df, pb_data["bulk"],
            [f"sample_{i}" for i in range(n_samples)],
            n_epochs=cfg.get("training", {}).get("epochs", 30),
            seed=SEED,
        )
        elapsed = time.monotonic() - start

        print(f"\n[4] Saving results...")
        pred_csv = out_dir / "proportions.csv"
        pred_df.to_csv(str(pred_csv))
        print(f"  Predictions -> {pred_csv}, Shape: {pred_df.shape}")

        gt_df = pd.DataFrame(pb_data["bulk_labels"], columns=type_list)
        print(f"  (metrics computed by post-hoc evaluation)")

        with open(out_dir / "cell_types.json", "w") as f:
            json.dump(type_list, f)

    else:
        print("ERROR: Provide either --data (h5ad) or --h5 (DeconBenchmark format)")
        sys.exit(1)

    print(f"\nDone. Output in {out_dir}/")


if __name__ == "__main__":
    main()
