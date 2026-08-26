#!/usr/bin/env python3
"""
CIBERSORTx — Cell-type deconvolution via nu-support vector regression.

Python reimplementation of the core CIBERSORT algorithm (Newman et al. 2015)
using scikit-learn NuSVR with linear kernel. No Docker or Stanford token
required.

The algorithm:
  1. Build a signature matrix (mean expression per cell type) from scRNA-seq.
  2. Z-score standardise signature matrix (per gene).
  3. For each bulk sample, try NuSVR with nu in [0.25, 0.5, 0.75].
  4. Select the model with lowest RMSE on the held-out mixture.
  5. Clip negative coefficients to zero and normalise to sum to 1.

Usage
-----
    python run.py --config configs/default.yaml --mode all
    python run.py --config configs/default.yaml --mode predict
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.svm import NuSVR

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", HERE.parent.parent)).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import auto_detect_celltype_col


class CIBERSORT:
    """Python reimplementation of the core CIBERSORT algorithm.

    Parameters
    ----------
    nu_values : tuple
        NuSVR nu values to try (default: 0.25, 0.5, 0.75).
    kernel : str
        SVR kernel (default 'linear').
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        nu_values: Tuple[float, ...] = (0.25, 0.5, 0.75),
        kernel: str = "linear",
        verbose: bool = False,
    ):
        self.nu_values = nu_values
        self.kernel = kernel
        self.verbose = verbose
        self._sig_matrix: Optional[pd.DataFrame] = None
        self._cell_types: Optional[List[str]] = None

    def fit(self, sc_ref: "ad.AnnData") -> None:
        """Build signature matrix from scRNA-seq reference."""
        import anndata as ad
        if not isinstance(sc_ref, ad.AnnData):
            raise TypeError("sc_ref must be an AnnData object")

        celltype_col = auto_detect_celltype_col(sc_ref.obs.columns)
        if celltype_col is None:
            raise ValueError(
                f"Could not detect cell-type column. Available: {list(sc_ref.obs.columns)}"
            )
        if self.verbose:
            print(f"[CIBERSORT] Cell-type column: '{celltype_col}'")

        if not isinstance(sc_ref.obs[celltype_col].dtype, pd.CategoricalDtype):
            sc_ref.obs[celltype_col] = sc_ref.obs[celltype_col].astype("category")

        cell_types = sc_ref.obs[celltype_col].cat.categories.tolist()
        X = sc_ref.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        else:
            X = np.asarray(X)

        sig_dict: Dict[str, np.ndarray] = {}
        for ct in cell_types:
            mask = (sc_ref.obs[celltype_col] == ct).values
            sig_dict[ct] = X[mask].mean(axis=0)

        self._sig_matrix = pd.DataFrame(sig_dict, index=sc_ref.var_names)
        self._cell_types = cell_types

        if self.verbose:
            print(f"[CIBERSORT] Signature matrix: {self._sig_matrix.shape[0]} genes x "
                  f"{len(cell_types)} types")

    def predict(self, bulk: pd.DataFrame) -> pd.DataFrame:
        """Predict cell-type proportions for bulk samples.

        Parameters
        ----------
        bulk : pd.DataFrame
            Bulk expression with genes as index, samples as columns.

        Returns
        -------
        pd.DataFrame
            Predicted proportions (samples x cell_types).
        """
        if self._sig_matrix is None or self._cell_types is None:
            raise RuntimeError("Call fit() before predict().")

        sig = self._sig_matrix.copy()
        cell_types = self._cell_types

        # Align genes
        common = sig.index.intersection(bulk.index)
        if len(common) == 0:
            raise ValueError("No common genes between signature matrix and bulk.")
        sig = sig.loc[common]
        bulk_aligned = bulk.loc[common]

        coverage = len(common) / len(self._sig_matrix)
        if coverage < 0.70:
            warnings.warn(f"Low gene coverage: {len(common)}/{len(self._sig_matrix)} ({coverage:.1%})")

        X = sig.values.astype(np.float64)
        Y = bulk_aligned.values.astype(np.float64)

        # Anti-log if data appears log-transformed (max < 50)
        if np.nanmax(Y) < 50:
            Y = np.power(2.0, Y)
            if self.verbose:
                print("[CIBERSORT] Anti-log applied.")

        # Standardise signature matrix
        X_flat = X.ravel()
        X_mean, X_std = float(np.mean(X_flat)), float(np.std(X_flat))
        if X_std == 0:
            X_std = 1.0
        X = (X - X_mean) / X_std

        n_samples = Y.shape[1]
        sample_names = bulk_aligned.columns.tolist()
        coef_matrix = np.zeros((n_samples, len(cell_types)), dtype=np.float64)

        for i in range(n_samples):
            y = Y[:, i]
            y_mean, y_std = float(np.mean(y)), float(np.std(y))
            if y_std == 0:
                y_std = 1.0
            y_stdized = (y - y_mean) / y_std

            best_rmse, best_coef = np.inf, None
            for nu in self.nu_values:
                model = NuSVR(nu=nu, kernel=self.kernel, C=1.0, gamma="auto",
                              max_iter=-1, cache_size=500)
                try:
                    model.fit(X, y_stdized)
                except ValueError:
                    continue

                weights = np.maximum(model.coef_.flatten().copy(), 0)
                w_sum = weights.sum()
                if w_sum > 0:
                    weights_norm = weights / w_sum
                else:
                    weights_norm = np.ones_like(weights) / len(weights)

                u = X * weights_norm
                k = u.sum(axis=1)
                rmse = float(np.sqrt(np.mean((k - y_stdized) ** 2)))
                if rmse < best_rmse:
                    best_rmse, best_coef = rmse, weights_norm

            if best_coef is None:
                best_coef = np.ones(len(cell_types)) / len(cell_types)
            coef_matrix[i] = best_coef

            if self.verbose and ((i + 1) % 50 == 0 or i == n_samples - 1):
                print(f"[CIBERSORT] {i + 1}/{n_samples} samples ({(i + 1) / n_samples * 100:.0f}%)")

        result = pd.DataFrame(coef_matrix, index=sample_names, columns=cell_types)
        result.index.name = "sample"
        return result

    def generate_pseudo_bulk(
        self,
        n_samples: int = 500,
        alpha: float = 1.0,
        noise_scale: float = 0.01,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate pseudo-bulk mixtures from the signature matrix."""
        rng = np.random.RandomState(random_state)
        n_types = len(self._cell_types)
        S = self._sig_matrix.values.astype(np.float64)
        true_props = rng.dirichlet(np.full(n_types, alpha), size=n_samples)
        bulk_expr = true_props @ S.T
        bulk_expr += rng.normal(0.0, noise_scale, size=bulk_expr.shape)
        bulk_expr = np.maximum(bulk_expr, 0.0)
        true_df = pd.DataFrame(
            true_props,
            index=[f"sample_{i}" for i in range(n_samples)],
            columns=self._cell_types,
        )
        bulk_df = pd.DataFrame(
            bulk_expr.T,
            index=self._sig_matrix.index,
            columns=[f"sample_{i}" for i in range(n_samples)],
        )
        return true_df, bulk_df


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(
        description="CIBERSORTx (Python NuSVR) — Bulk deconvolution")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--mode", type=str, default="predict",
                   choices=["train", "predict", "all"],
                   help="Execution mode (train=build sig matrix, predict=run)")
    p.add_argument("--sc-ref", type=str, default=None)
    p.add_argument("--bulk", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--data", type=str, default=None,
                   help="Path to H5 (DeconBenchmark format) — activates H5 pseudo-bulk flow")
    p.add_argument("--h5", type=str, default=None,
                   help="Path to H5 (DeconBenchmark format) — activates real bulk flow")
    p.add_argument("--ground-truth", type=str, default=None,
                   help="Path to ground truth CSV for --h5 mode")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for predictions and metrics")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ── H5 flow (pseudo-bulk) ──────────────────────────────────────────────
    if args.data is not None:
        data_path = Path(args.data)
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("CIBERSORTx — NuSVR Deconvolution (H5 pseudo-bulk)")
        print("=" * 60)

        import anndata as ad
        print(f"\n[1] Loading H5: {data_path}")
        if data_path.suffix == ".h5":
            import h5py
            with h5py.File(str(data_path), "r") as f:
                sc_expr = np.asarray(f["singleCellExpr/values"][:], dtype=np.float64)
                sc_labels_raw = f["singleCellLabels/values"][:]
                sc_labels = [x.decode() if isinstance(x, bytes) else x for x in sc_labels_raw]
                if "singleCellExpr/rownames" in f:
                    gene_names = [x.decode() if isinstance(x, bytes) else x for x in f["singleCellExpr/rownames"][:]]
                elif "bulk/rownames" in f:
                    gene_names = [x.decode() if isinstance(x, bytes) else x for x in f["bulk/rownames"][:]]
                else:
                    gene_names = [f"gene_{i}" for i in range(sc_expr.shape[1])]

            # DeconBenchmark stores singleCellExpr as (n_genes, n_cells) → transpose to (n_cells, n_genes)
            if sc_expr.shape[0] == len(gene_names):
                sc_expr = sc_expr.T

            sc_ref = ad.AnnData(
                X=sc_expr,
                var=pd.DataFrame(index=gene_names),
                obs=pd.DataFrame({"cell_type": sc_labels}),
            )
        else:
            sc_ref = ad.read_h5ad(str(data_path))

        print(f"    Reference: {sc_ref.shape}")

        celltype_col = cfg["data"].get("celltype_col") or auto_detect_celltype_col(sc_ref.obs.columns)
        if not isinstance(sc_ref.obs[celltype_col].dtype, pd.CategoricalDtype):
            sc_ref.obs[celltype_col] = sc_ref.obs[celltype_col].astype("category")

        # Build signature matrix
        print(f"\n[2] Building signature matrix...")
        model = CIBERSORT(
            nu_values=tuple(cfg.get("model", {}).get("nu_values", [0.25, 0.5, 0.75])),
            verbose=cfg.get("verbose", True),
        )
        model.fit(sc_ref)

        # Generate pseudo-bulk
        n_pseudo = cfg.get("data", {}).get("n_pseudo_bulk", 200)
        print(f"\n[3] Generating {n_pseudo} pseudo-bulk samples...")
        true_props, bulk_expr = model.generate_pseudo_bulk(n_samples=n_pseudo)

        # Predict
        print(f"\n[4] Predicting proportions...")
        pred_props = model.predict(bulk_expr)

        # Evaluate
        print(f"\n[5] Evaluating...")
        from core.metrics import evaluate_deconvolution

        metrics = evaluate_deconvolution(
            true_props.values,
            pred_props.values,
            cell_types=list(true_props.columns),
        )
        pred_path = output_dir / "proportions.csv"
        pred_props.to_csv(pred_path)

        print(f"    Pearson r: {metrics['pearson_mean']:.4f}")
        print(f"    RMSE:      {metrics['rmse_overall']:.4f}")
        print(f"    Predictions -> {pred_path}")
        print(f"\nDone.")
        return

    # ── H5 real-bulk flow (--h5 + --ground-truth) ──────────────────────────
    if args.h5 is not None:
        h5_path = Path(args.h5)
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("CIBERSORTx — NuSVR Deconvolution (H5 real bulk)")
        print("=" * 60)

        import anndata as ad
        import h5py
        print(f"\n[1] Loading H5: {h5_path}")
        with h5py.File(str(h5_path), "r") as f:
            bulk_values = np.asarray(f["bulk/values"][:], dtype=np.float64)
            sc_expr = np.asarray(f["singleCellExpr/values"][:], dtype=np.float64)
            sc_labels_raw = f["singleCellLabels/values"][:]
            sc_labels = [x.decode() if isinstance(x, bytes) else x for x in sc_labels_raw]

            bulk_genes = [x.decode() if isinstance(x, bytes) else x
                         for x in f["bulk/rownames"][:]] if "bulk/rownames" in f else [
                         f"gene_{i}" for i in range(bulk_values.shape[1])]
            sc_genes = [x.decode() if isinstance(x, bytes) else x
                       for x in f["singleCellExpr/rownames"][:]] if "singleCellExpr/rownames" in f else [
                       f"gene_{i}" for i in range(sc_expr.shape[1])]

            # Align gene sets
            common_genes = sorted(set(bulk_genes) & set(sc_genes))
            if len(common_genes) == 0:
                raise ValueError("No common genes between bulk and scRNA.")
            n_bulk, n_sc = len(bulk_genes), len(sc_genes)
            if len(common_genes) < min(n_bulk, n_sc):
                print(f"    Aligned to {len(common_genes)} common genes "
                      f"(bulk had {n_bulk}, scRNA had {n_sc})")

            # DeconBenchmark stores singleCellExpr as (n_genes, n_cells);
            # transpose to (n_cells, n_genes) before gene filtering
            if sc_expr.shape[0] == len(sc_genes):
                sc_expr = sc_expr.T

            # Bulk may be stored as (n_genes, n_samples); transpose to
            # (n_samples, n_genes) so that bulk_values[:, gene_idx] selects genes.
            if len(bulk_genes) > 0 and bulk_values.shape[0] == len(bulk_genes) and bulk_values.shape[1] < len(bulk_genes):
                bulk_values = bulk_values.T

            bulk_gene_to_idx = {g: i for i, g in enumerate(bulk_genes)}
            bulk_gene_idx = [bulk_gene_to_idx[g] for g in common_genes]
            bulk_values = bulk_values[:, bulk_gene_idx]
            sc_gene_to_idx = {g: i for i, g in enumerate(sc_genes)}
            sc_gene_idx = [sc_gene_to_idx[g] for g in common_genes]
            sc_expr = sc_expr[:, sc_gene_idx]

            gene_names = common_genes
            sample_names = [x.decode() if isinstance(x, bytes) else x
                          for x in f["bulk/colnames"][:]] if "bulk/colnames" in f else [
                          f"sample_{i}" for i in range(bulk_values.shape[0])]

        sc_ref = ad.AnnData(
            X=sc_expr,
            var=pd.DataFrame(index=gene_names),
            obs=pd.DataFrame({"cell_type": sc_labels}),
        )
        print(f"    Reference: {sc_ref.shape}")

        # Build signature matrix
        print(f"\n[2] Building signature matrix...")
        model = CIBERSORT(
            nu_values=tuple(cfg.get("model", {}).get("nu_values", [0.25, 0.5, 0.75])),
            verbose=cfg.get("verbose", True),
        )
        model.fit(sc_ref)

        # Predict on real bulk
        bulk_expr = pd.DataFrame(bulk_values, index=sample_names, columns=gene_names).T
        print(f"\n[3] Predicting proportions ({bulk_expr.shape[1]} samples)...")
        pred_props = model.predict(bulk_expr)

        # Load ground truth
        if args.ground_truth is None:
            # Save predictions without internal evaluation (mirrors _linutils).
            pred_out = output_dir / "proportions.csv"
            pred_props.T.to_csv(pred_out)
            print(f"    Predictions saved -> {pred_out}")
            print("Done.")
            return
        gt_path = Path(args.ground_truth)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")
        print(f"\n[4] Loading ground truth: {gt_path}")
        true_props_raw = pd.read_csv(str(gt_path))
        if true_props_raw.iloc[:, 0].dtype in (object, str) or str(true_props_raw.iloc[0, 0]).startswith("sample_"):
            true_props = true_props_raw.set_index(true_props_raw.columns[0])
        else:
            true_props = true_props_raw

        common_samples = true_props.index.intersection(pred_props.index)
        common_types = [ct for ct in true_props.columns if ct in pred_props.columns]

        if len(common_samples) == 0 and len(common_types) > 0 and len(true_props) == len(pred_props):
            print(f"    No named sample match, using positional alignment "
                  f"({len(true_props)} samples)")
            pred_slice = pred_props.loc[pred_props.index[:len(true_props)], common_types]
            true_props.index = pred_slice.index
            common_samples = list(true_props.index)

        if len(common_samples) == 0:
            raise ValueError("No common samples between predictions and ground truth.")
        if len(common_types) == 0:
            raise ValueError("No common cell types between predictions and ground truth.")

        true_props = true_props.loc[common_samples, common_types]
        pred_props = pred_props.loc[common_samples, common_types]

        # Evaluate
        print(f"\n[5] Evaluating...")
        from core.metrics import evaluate_deconvolution

        metrics = evaluate_deconvolution(
            true_props.values,
            pred_props.values,
            cell_types=list(true_props.columns),
        )
        metrics["method"] = "cibersortx"
        metrics["n_samples"] = int(true_props.shape[0])
        metrics["n_cell_types"] = int(true_props.shape[1])

        pred_path = output_dir / "proportions.csv"
        pred_props.to_csv(pred_path)

        print(f"    Pearson r:  {metrics['pearson_mean']:.4f}")
        print(f"    SCorr:      {metrics['scorr_mean']:.4f}")
        print(f"    CCorr:      {metrics['ccorr_mean']:.4f}")
        print(f"    RMSE:       {metrics['rmse_overall']:.4f}")
        print(f"    MAE:        {metrics['mae_overall']:.4f}")
        print(f"    Wilcoxon p: {metrics['wt_mean']:.4f}")
        print(f"    Predictions saved -> {pred_path}")
        print(f"\nDone.")
        return

    # ── Legacy TSV/CSV flow ────────────────────────────────────────────────

    if args.sc_ref is not None:
        cfg["data"]["sc_ref"] = args.sc_ref
    if args.bulk is not None:
        cfg["data"]["bulk"] = args.bulk
    if args.output is not None:
        cfg["paths"]["output"] = args.output

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    sc_ref_path = _resolve(cfg["data"]["sc_ref"])
    bulk_path = _resolve(cfg["data"]["bulk"])
    output_path = _resolve(cfg["paths"]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CIBERSORTx — NuSVR Deconvolution")
    print("=" * 60)

    # Load reference
    import anndata as ad
    print(f"\n[1] Loading scRNA-seq reference: {sc_ref_path}")
    sc_ref = ad.read_h5ad(str(sc_ref_path))

    celltype_col = cfg["data"].get("celltype_col") or auto_detect_celltype_col(sc_ref.obs.columns)
    if celltype_col is None:
        raise ValueError(f"Could not detect cell-type column. Available: {list(sc_ref.obs.columns)}")

    if not isinstance(sc_ref.obs[celltype_col].dtype, pd.CategoricalDtype):
        sc_ref.obs[celltype_col] = sc_ref.obs[celltype_col].astype("category")

    # Filter low-count types
    min_cells = cfg["data"].get("min_cells_per_type", 10)
    counts = sc_ref.obs[celltype_col].value_counts()
    valid = counts[counts >= min_cells].index.tolist()
    excluded = [ct for ct in sc_ref.obs[celltype_col].unique() if ct not in valid]
    if excluded:
        print(f"  Excluding types with <{min_cells} cells: {excluded}")
        sc_ref = sc_ref[sc_ref.obs[celltype_col].isin(valid)].copy()

    print(f"  Reference: {sc_ref.shape}, types: {sorted(sc_ref.obs[celltype_col].unique())}")

    # Optional HVG selection for speed
    n_hvg = cfg["data"].get("n_hvg", None)
    if n_hvg is not None and n_hvg < sc_ref.n_vars:
        print(f"\n  Selecting {n_hvg} HVGs...")
        X = sc_ref.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        variances = np.var(np.asarray(X), axis=0)
        top_idx = np.argsort(variances)[::-1][:n_hvg]
        top_genes = sc_ref.var_names[top_idx].tolist()
        sc_ref = sc_ref[:, top_genes].copy()

    # Build signature matrix (mode == "train" or "all")
    if args.mode in ("train", "all"):
        print(f"\n[2] Building signature matrix...")
        model = CIBERSORT(
            nu_values=tuple(cfg["model"].get("nu_values", [0.25, 0.5, 0.75])),
            verbose=cfg.get("verbose", True),
        )
        model.fit(sc_ref)
        sig_path = output_path.parent / "signature_matrix.csv"
        model._sig_matrix.to_csv(str(sig_path))
        print(f"  Signature matrix saved -> {sig_path}")
    else:
        # Load pre-built signature matrix
        sig_path = Path(cfg["paths"].get("signature_matrix", ""))
        if not sig_path.is_absolute():
            sig_path = PROJECT_ROOT / sig_path
        if not sig_path.exists():
            raise FileNotFoundError(
                f"Signature matrix not found at {sig_path}. "
                "Run with --mode train first."
            )
        sig_df = pd.read_csv(str(sig_path), index_col=0)
        model = CIBERSORT(
            nu_values=tuple(cfg["model"].get("nu_values", [0.25, 0.5, 0.75])),
            verbose=cfg.get("verbose", True),
        )
        model._sig_matrix = sig_df
        model._cell_types = sig_df.columns.tolist()

    # Load bulk
    print(f"\n[3] Loading bulk: {bulk_path}")
    bulk_sep = cfg["data"].get("bulk_sep", None)
    if bulk_sep is None:
        ext = str(bulk_path).lower()
        bulk_sep = "\t" if ext.endswith((".tsv", ".txt")) else ","
    bulk = pd.read_csv(str(bulk_path), sep=bulk_sep, index_col=0)
    print(f"  Bulk: {bulk.shape[0]} genes x {bulk.shape[1]} samples")

    # Predict
    print(f"\n[4] Predicting proportions...")
    proportions = model.predict(bulk)
    proportions.to_csv(str(output_path))
    print(f"  Saved -> {output_path}")

    print("\nMean proportions:")
    print(proportions.mean().sort_values(ascending=False).to_string())

    # Evaluate if ground truth provided
    gt_path = cfg["paths"].get("ground_truth")
    if gt_path:
        gt_path = _resolve(gt_path)
        if gt_path.exists():
            gt = pd.read_csv(str(gt_path), index_col=0)
            common_s = gt.index.intersection(proportions.index)
            common_c = [c for c in gt.columns if c in proportions.columns]
            t = gt.loc[common_s, common_c].values
            p = proportions.loc[common_s, common_c].values
            pearson = np.mean([np.corrcoef(t[:, i], p[:, i])[0, 1]
                               for i in range(len(common_c))
                               if np.std(t[:, i]) > 0 and np.std(p[:, i]) > 0])
            rmse = float(np.sqrt(np.mean((t - p) ** 2)))
            print(f"\n  Evaluation vs ground truth:")
            print(f"    Pearson: {pearson:.4f}, RMSE: {rmse:.4f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
