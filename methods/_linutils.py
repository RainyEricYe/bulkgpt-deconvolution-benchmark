"""
Shared utilities for linear deconvolution baselines.
Each method (nnls, ols, ridge, nusvr) has its own run.py that imports from here.
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
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import NuSVR

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", HERE.parent)).resolve()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Linear/regression baselines for bulk deconvolution"
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--mode", type=str, default="predict", choices=["predict"])
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--h5", type=str, default=None)
    p.add_argument("--ground-truth", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class LinearDeconvolution:
    """Signature-based linear deconvolution with sklearn regressors.

    Parameters
    ----------
    method : str
        One of ``"ols"``, ``"nnls"``, ``"rlr"``, ``"nusvr"``.
    params : dict
        Hyper-parameters forwarded to the regressor constructor.
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        method: str = "ols",
        params: Optional[dict] = None,
        verbose: bool = False,
    ):
        self.method = method
        self.params = params or {}
        self.verbose = verbose
        self._sig_matrix: Optional[pd.DataFrame] = None
        self._cell_types: Optional[List[str]] = None

    # -- Building the signature matrix -----------------------------------------

    def fit(self, sc_ref) -> None:
        """Build signature matrix from scRNA-seq reference."""
        from core.data_loader import auto_detect_celltype_col

        celltype_col = auto_detect_celltype_col(sc_ref.obs.columns)
        if celltype_col is None:
            raise ValueError(
                f"Could not detect cell-type column. "
                f"Available: {list(sc_ref.obs.columns)}"
            )
        if self.verbose:
            print(f"  Cell-type column: '{celltype_col}'")

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
            print(f"  Signature matrix: {self._sig_matrix.shape[0]} genes x "
                  f"{len(cell_types)} types")

    # -- Prediction -------------------------------------------------------------

    def _solve(self, S: np.ndarray, bulk: np.ndarray) -> np.ndarray:
        """Solve for a single bulk sample."""
        method = self.method
        params = dict(self.params)

        if method == "ols":
            fit_intercept = params.pop("fit_intercept", False)
            model = LinearRegression(fit_intercept=fit_intercept, **params)
            model.fit(S, bulk)
            coef = model.coef_.copy()
        elif method == "nnls":
            coef, _ = nnls(S, bulk)
        elif method == "rlr":
            fit_intercept = params.pop("fit_intercept", False)
            model = Ridge(fit_intercept=fit_intercept, **params)
            model.fit(S, bulk)
            coef = model.coef_.copy()
        elif method == "nusvr":
            params.setdefault("kernel", "linear")
            model = NuSVR(**params, cache_size=500)
            model.fit(S, bulk)
            coef = model.coef_.flatten().copy()
        else:
            raise ValueError(f"Unknown method: {method}")

        coef = np.maximum(coef, 0.0)
        total = coef.sum()
        if total > 0:
            coef = coef / total
        else:
            coef = np.ones_like(coef) / len(coef)

        return coef

    def predict(self, bulk_expr: pd.DataFrame) -> pd.DataFrame:
        """Predict proportions for bulk samples."""
        if self._sig_matrix is None or self._cell_types is None:
            raise RuntimeError("Call fit() before predict().")

        sig = self._sig_matrix.copy()
        cell_types = self._cell_types

        common = sig.index.intersection(bulk_expr.index)
        if len(common) == 0:
            raise ValueError("No common genes between signature matrix and bulk.")
        sig = sig.loc[common]
        bulk_aligned = bulk_expr.loc[common]

        coverage = len(common) / len(self._sig_matrix)
        if coverage < 0.70:
            warnings.warn(
                f"Low gene coverage: {len(common)}/{len(self._sig_matrix)} "
                f"({coverage:.1%})"
            )

        S = sig.values.astype(np.float64)
        Y = bulk_aligned.values.astype(np.float64)

        n_samples = Y.shape[1]
        sample_names = bulk_aligned.columns.tolist()
        proportions = np.zeros((n_samples, len(cell_types)), dtype=np.float64)

        for i in range(n_samples):
            proportions[i] = self._solve(S, Y[:, i])

        result = pd.DataFrame(
            proportions, index=sample_names, columns=cell_types,
        )
        result.index.name = "sample"
        return result

    # -- Pseudo-bulk generation -------------------------------------------------

    def generate_pseudo_bulk(
        self,
        n_samples: int = 500,
        alpha: float = 1.0,
        noise_scale: float = 0.01,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate pseudo-bulk mixtures from the signature matrix."""
        if self._sig_matrix is None or self._cell_types is None:
            raise RuntimeError("Call fit() before generate_pseudo_bulk().")

        rng = np.random.RandomState(random_state)
        n_types = len(self._cell_types)
        S = self._sig_matrix.values.astype(np.float64)
        n_genes = S.shape[0]

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


# ---------------------------------------------------------------------------
# CLI entry point (called by each method's run.py)
# ---------------------------------------------------------------------------


def main(method_name: str = "ols"):
    args = parse_args()
    cfg = load_config(args.config)

    method_params = cfg.get("params", {})
    data_cfg = cfg.get("data", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Linear Baseline: {method_name.upper()}")
    print("=" * 60)

    print(f"\n[1] Initialising {method_name.upper()} regressor...")
    if method_params:
        print(f"    Params: {method_params}")

    model = LinearDeconvolution(
        method=method_name,
        params=method_params,
        verbose=True,
    )

    sys.path.insert(0, str(PROJECT_ROOT))

    # ------------------------------------------------------------------
    # Pseudo-bulk mode  (--data)
    # ------------------------------------------------------------------
    if args.data:
        data_path = Path(args.data)
        print(f"\n[2] Loading scRNA-seq reference: {data_path}")
        if data_path.suffix == ".h5":
            from core.data_loader import load_sc_ref
            sc_ref = load_sc_ref(str(data_path))
        else:
            import anndata as ad
            sc_ref = ad.read_h5ad(str(data_path))
        print(f"    Shape: {sc_ref.shape}")

        max_cells = data_cfg.get("max_cells")
        if max_cells is not None and sc_ref.n_obs > max_cells:
            rng = np.random.RandomState(42)
            idx = rng.choice(sc_ref.n_obs, max_cells, replace=False)
            sc_ref = sc_ref[idx].copy()
            print(f"    Subsampled to {max_cells} cells")

        n_hvg = data_cfg.get("n_hvg")
        if n_hvg is not None and n_hvg < sc_ref.n_vars:
            print(f"\n[3] Selecting {n_hvg} HVGs...")
            X_hvg = sc_ref.X
            if hasattr(X_hvg, "toarray"):
                X_hvg = X_hvg.toarray()
            variances = np.var(np.asarray(X_hvg), axis=0)
            top_idx = np.argsort(variances)[::-1][:n_hvg]
            top_genes = sc_ref.var_names[top_idx].tolist()
            sc_ref = sc_ref[:, top_genes].copy()

        print(f"\n[4] Building signature matrix...")
        model.fit(sc_ref)

        n_pseudo = data_cfg.get("n_pseudo_bulk", 500)
        alpha = data_cfg.get("proportion_alpha", 1.0)
        print(f"\n[5] Generating {n_pseudo} pseudo-bulk samples "
              f"(alpha={alpha})...")
        true_props, bulk_expr = model.generate_pseudo_bulk(
            n_samples=n_pseudo,
            alpha=alpha,
        )

        print(f"\n[6] Predicting proportions...")
        pred_props = model.predict(bulk_expr)

    # ------------------------------------------------------------------
    # Real bulk mode  (--h5 + --ground-truth)
    # ------------------------------------------------------------------
    elif args.h5:
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.data_loader import load_data

        h5_path = Path(args.h5)
        print(f"\n[2] Loading H5 data: {h5_path}")
        bundle = load_data(str(h5_path))
        sc_ref = bundle.sc_ref
        if sc_ref is None:
            raise ValueError("H5 file does not contain singleCellExpr data")

        print(f"    Bulk: {bundle.bulk.shape}, scRNA: {sc_ref.shape}")

        max_cells = data_cfg.get("max_cells")
        if max_cells is not None and sc_ref.n_obs > max_cells:
            rng = np.random.RandomState(42)
            idx = rng.choice(sc_ref.n_obs, max_cells, replace=False)
            sc_ref = sc_ref[idx].copy()
            print(f"    Subsampled to {max_cells} cells")

        print(f"\n[3] Building signature matrix...")
        model.fit(sc_ref)

        # bundle.bulk is (samples, genes); predict() expects (genes, samples)
        bulk_expr = bundle.bulk.T

        print(f"\n[4] Predicting proportions...")
        pred_props = model.predict(bulk_expr)

        if args.ground_truth is None:
            # No GT available — save proportions only, skip metrics
            pred_path = output_dir / "proportions.csv"
            pred_props.to_csv(pred_path)
            print(f"    Predictions saved -> {pred_path}")
            print(f"\n  (No ground truth provided — metrics skipped.)")
            print(f"\nDone.")
            return
        gt_path = Path(args.ground_truth)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")
        print(f"\n[5] Loading ground truth: {gt_path}")
        true_props = pd.read_csv(str(gt_path), index_col=0)

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

    else:
        raise ValueError(
            "Either --data (pseudo-bulk) or --h5 (real bulk) must be provided."
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print(f"\n[7] Evaluating...")
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.metrics import evaluate_deconvolution

    metrics = evaluate_deconvolution(
        true_props.values,
        pred_props.values,
        cell_types=list(true_props.columns),
    )

    metrics["method"] = method_name
    metrics["n_samples"] = int(true_props.shape[0])
    metrics["n_cell_types"] = int(true_props.shape[1])

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Metrics saved -> {metrics_path}")

    print(f"\n  Results ({method_name.upper()}):")
    print(f"    Pearson r:  {metrics['pearson_mean']:.4f}")
    print(f"    SCorr:      {metrics['scorr_mean']:.4f}")
    print(f"    CCorr:      {metrics['ccorr_mean']:.4f}")
    print(f"    RMSE:       {metrics['rmse_overall']:.4f}")
    print(f"    MAE:        {metrics['mae_overall']:.4f}")
    print(f"    Wilcoxon p: {metrics['wt_mean']:.4f}")

    pred_path = output_dir / "proportions.csv"
    pred_props.to_csv(pred_path)
    print(f"    Predictions saved -> {pred_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
