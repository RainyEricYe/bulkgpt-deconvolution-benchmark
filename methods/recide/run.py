#!/usr/bin/env python3
"""ReCIDE deconvolution — independent entry point.

ReCIDE: Robust Estimation of Cell type proportIons by integrating
single-reference Deconvolution Ensemble.

Reference
---------
Li, M. et al. "Robust estimation of cell-type proportions by integrating
single-reference deconvolution ensemble (ReCIDE)."
*Briefings in Bioinformatics*, 2024.

Algorithm
---------
1. Build per-cell-type mean expression profiles from scRNA-seq reference.
2. Generate a diverse panel of signature matrices by varying the number
   of marker genes (top fold-change) per cell type, plus the full
   gene-set signature and a random subsample.
3. For each signature, solve the deconvolution via NNLS (default) or
   DWLS / NuSVR.
4. Ensemble all solutions via PCA + GMM clustering: select the largest
   cluster and average.

CLI accepted arguments
----------------------
--config PATH       YAML config with algorithm parameters (optional).
--h5 PATH           DeconBenchmark H5 (bulk/values, singleCellExpr/,
                    singleCellLabels/).
--ground-truth PATH  CSV with true proportions.
--output-dir PATH    Where to write proportions.csv and metrics.json.
"""

# ── Imports ────────────────────────────────────────────────────────────
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
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", HERE.parent.parent)).resolve()

# ── CLI ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ReCIDE deconvolution — ensemble of single-reference deconvolution"
    )
    p.add_argument("--config", type=str, default=None,
                    help="Path to YAML config (optional; uses defaults if omitted)")
    p.add_argument("--mode", type=str, default="predict",
                    choices=["predict"],
                    help="Execution mode (only predict for ReCIDE)")
    p.add_argument("--h5", type=str, default=None,
                    help="Path to DeconBenchmark H5 file")
    p.add_argument("--ground-truth", type=str, default=None,
                    help="Path to ground-truth proportions CSV")
    p.add_argument("--output-dir", type=str, required=True,
                    help="Output directory")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════
# Core ReCIDE algorithm
# ═══════════════════════════════════════════════════════════════════════


class ReCIDE:
    """Ensemble deconvolution via diverse marker-gene signatures.

    Core idea (ReCIDE, Li et al. 2024): instead of using one fixed
    signature matrix, generate a panel of diverse signature matrices
    (via different marker selections or gene subsets), solve each
    independently, and ensemble the results via PCA + GMM clustering.

    Parameters
    ----------
    n_markers : int
        Number of top fold-change marker genes per cell type (default 100).
    n_pseudo_subjects : int
        Number of ensemble members (default 5).
    solver : str
        Core solver: ``"nnls"`` (default, robust), ``"dwls"``, or
        ``"nusvr"`` (CIBERSORT-style).
    random_state : int
        Seed for reproducibility (default 42).
    verbose : bool
        Print progress (default False).
    """

    def __init__(
        self,
        n_markers: int = 100,
        n_pseudo_subjects: int = 5,
        solver: str = "nnls",
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_markers = n_markers
        self.n_pseudo_subjects = n_pseudo_subjects
        self.solver = solver
        self.random_state = random_state
        self.verbose = verbose

        # Populated by fit()
        self._signatures: List[pd.DataFrame] = []
        self._cell_types: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Marker gene selection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_markers(
        type_means: np.ndarray,
        cell_types: List[str],
        gene_names: List[str],
        n_markers: int,
    ) -> Dict[str, List[str]]:
        """Select top-*n_markers* marker genes per type by fold-change.

        For each cell type, computes the ratio
        ``mean(type) / max(mean(other_types))`` and returns the
        *n_markers* highest-ratio genes (no minimum threshold — ranking
        only).  When there are fewer genes than *n_markers*, all genes
        are returned.

        Parameters
        ----------
        type_means : (n_types, n_genes) pre-computed per-type means.
        cell_types : Unique cell-type names.
        gene_names : Length *n_genes* gene identifiers.
        n_markers : Number of markers to select per type.

        Returns
        -------
        Dict[cell_type -> list of marker gene names].
        """
        n_genes = type_means.shape[1]
        markers: Dict[str, List[str]] = {}

        for i, ct in enumerate(cell_types):
            other_means = np.delete(type_means, i, axis=0)
            max_other = other_means.max(axis=0)
            fc = np.where(
                max_other > 1e-10, type_means[i] / max_other, np.inf
            )
            top_idx = np.argsort(fc)[::-1][: min(n_markers, n_genes)]
            markers[ct] = [gene_names[j] for j in top_idx]

        return markers

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_nnls(S: np.ndarray, bulk: np.ndarray) -> np.ndarray:
        """NNLS solver — returns non-negative proportions summing to 1."""
        x, _ = nnls(S, bulk)
        s = x.sum()
        return x / s if s > 0 else np.full(S.shape[1], 1.0 / S.shape[1])

    @staticmethod
    def _solve_dwls(
        S: np.ndarray, bulk: np.ndarray,
        max_iter: int = 30, epsilon: float = 0.01,
    ) -> np.ndarray:
        """Dampened weighted least squares.

        Iteratively reweighted NNLS with weight capping at
        ``median + 3 * MAD`` to reduce the influence of outlier genes.
        """
        x, _ = nnls(S, bulk)
        s = x.sum()
        x = x / s if s > 0 else np.full(S.shape[1], 1.0 / S.shape[1])

        for _ in range(max_iter):
            residuals = bulk - S @ x
            weights = 1.0 / (np.abs(residuals) + epsilon)

            w_median = np.median(weights)
            w_mad = np.median(np.abs(weights - w_median)) + 1e-10
            weights = np.minimum(weights, w_median + 3.0 * w_mad)

            sqrt_w = np.sqrt(weights)
            x_new, _ = nnls(S * sqrt_w[:, np.newaxis], bulk * sqrt_w)
            sn = x_new.sum()
            x_new = x_new / sn if sn > 0 else np.full(S.shape[1], 1.0 / S.shape[1])

            if np.allclose(x, x_new, rtol=1e-4):
                break
            x = x_new
        return x

    @staticmethod
    def _solve_nusvr(S: np.ndarray, bulk: np.ndarray) -> np.ndarray:
        """NuSVR (CIBERSORT-style) solver."""
        from sklearn.svm import NuSVR
        model = NuSVR(kernel="linear", nu=0.5, cache_size=500)
        model.fit(S, bulk)
        coef = np.maximum(model.coef_.flatten(), 0.0)
        s = coef.sum()
        return coef / s if s > 0 else np.full(S.shape[1], 1.0 / S.shape[1])

    def _solve(self, S: np.ndarray, bulk: np.ndarray) -> np.ndarray:
        """Dispatch to the configured solver."""
        if self.solver == "nnls":
            return self._solve_nnls(S, bulk)
        elif self.solver == "dwls":
            return self._solve_dwls(S, bulk)
        elif self.solver == "nusvr":
            return self._solve_nusvr(S, bulk)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    # ------------------------------------------------------------------
    # Ensemble integration  (PCA + GMM)
    # ------------------------------------------------------------------

    @staticmethod
    def _ensemble(
        all_results: List[pd.DataFrame],
        random_state: int = 42,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Ensemble subject-level results via PCA + GMM.

        1. Stack all subject-level proportion matrices.
        2. PCA-reduce to 2 components.
        3. GMM cluster into at most *n_subjects* components.
        4. Select the **largest** cluster (most consistent estimates).
        5. Average within-cluster results and renormalise.

        Parameters
        ----------
        all_results : list of DataFrames, each (n_samples, n_types).
        random_state : Seed for GMM.

        Returns
        -------
        (n_samples, n_types) ensembled proportion matrix.
        """
        if len(all_results) == 1:
            return all_results[0]

        n_samples, n_types = all_results[0].shape
        cell_types = all_results[0].columns.tolist()
        n_subjects = len(all_results)

        # Stack: (n_subjects * n_samples, n_types)
        stacked = np.vstack([r.values for r in all_results])
        stacked = np.nan_to_num(stacked, nan=0.0)

        # PCA
        n_components = min(2, stacked.shape[1])
        try:
            pca = PCA(n_components=n_components, random_state=random_state)
            coords = pca.fit_transform(stacked)
        except Exception:
            if verbose:
                print("    PCA failed — using mean ensemble")
            return pd.concat(all_results).groupby(level=0).mean()

        # GMM
        n_clusters = min(n_subjects, 5)
        try:
            gmm = GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
                n_init=5,
                max_iter=200,
            )
            labels = gmm.fit_predict(coords)
        except Exception:
            if verbose:
                print("    GMM failed — using mean ensemble")
            return pd.concat(all_results).groupby(level=0).mean()

        # Find the largest cluster
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        largest = int(cluster_sizes.argmax())
        in_cluster = labels == largest

        if in_cluster.sum() == 0:
            return pd.concat(all_results).groupby(level=0).mean()

        # Map back to subject-level indices
        subject_indices_in_cluster = np.unique(
            np.where(in_cluster)[0] // n_samples
        )
        if len(subject_indices_in_cluster) == 0:
            return pd.concat(all_results).groupby(level=0).mean()

        selected = [all_results[s] for s in subject_indices_in_cluster]
        ensembled = pd.concat(selected).groupby(level=0).mean()

        # Renormalise
        row_sums = ensembled.values.sum(axis=1, keepdims=True)
        ensembled = ensembled / np.maximum(row_sums, 1e-10)

        if verbose:
            print(f"    Ensemble: selected "
                  f"{len(subject_indices_in_cluster)}/{n_subjects} "
                  f"subjects (largest cluster)")

        return ensembled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        cell_types: List[str],
        gene_names: List[str],
    ) -> None:
        """Build a diverse panel of signature matrices for ensemble.

        Pre-computes per-type mean expression across all genes, then
        generates multiple signature matrices using:
        - Full gene-set signature (all genes, mean per type)
        - Marker-gene signatures with varying marker counts
        - Random gene subsample for additional diversity

        During ``predict()`` each signature is solved independently and
        the results are ensembled via PCA + GMM.

        Parameters
        ----------
        X : (n_cells, n_genes) scRNA-seq expression matrix.
        labels : (n_cells,) cell-type label for each cell.
        cell_types : Unique cell-type names.
        gene_names : Length *n_genes* list of gene identifiers.
        """
        self._cell_types = cell_types

        n_types = len(cell_types)
        n_genes = X.shape[1]

        # Pre-compute per-type mean expression across ALL genes
        type_means = np.zeros((n_types, n_genes), dtype=np.float64)
        for i, ct in enumerate(cell_types):
            mask = labels == ct
            if mask.sum() > 0:
                type_means[i] = X[mask].mean(axis=0)

        n_subjects = max(2, self.n_pseudo_subjects)
        rng = np.random.RandomState(self.random_state)

        # Marker counts per subject: full set + varying marker counts
        factors = np.linspace(0.3, 2.0, n_subjects - 1) if n_subjects > 1 else []
        base_nm = self.n_markers

        if self.verbose:
            print(f"  Building {1 + len(factors) + 1} signatures "
                  f"using all {X.shape[0]} cells, {n_genes} genes")

        self._signatures = []

        # ── Signature 0: all genes (full mean-expression profile) ──
        # Normalise columns so each cell type has unit sum, preventing
        # systematic bias from differences in total expression per cell.
        sig0_raw = pd.DataFrame(type_means.T, index=gene_names, columns=cell_types)
        sig0 = sig0_raw / sig0_raw.sum(axis=0)
        self._signatures.append(sig0)
        if self.verbose:
            print(f"  Sig 0 (full): {sig0.shape} — {n_genes} genes")

        # ── Signatures 1..: marker-gene subsets ──────────────────
        for idx, factor in enumerate(factors):
            nm = max(5, int(base_nm * factor))
            nm_jitter = int(nm * (0.8 + 0.4 * rng.random()))
            nm_jitter = max(5, min(n_genes, nm_jitter))

            markers = self._find_markers(type_means, cell_types, gene_names, nm_jitter)

            all_markers = sorted({g for genes in markers.values() for g in genes})
            if len(all_markers) < 5:
                continue

            idx_map = {g: i for i, g in enumerate(gene_names)}
            mk_idx = [idx_map[g] for g in all_markers]
            sig = pd.DataFrame(
                type_means[:, mk_idx].T, index=all_markers, columns=cell_types,
            )
            sig = sig / sig.sum(axis=0)  # column-normalise
            self._signatures.append(sig)
            if self.verbose:
                print(f"  Sig {idx + 1}/{len(factors)}: {sig.shape} — {nm_jitter} markers/type")

        # ── Random gene subsample for extra diversity ────────────
        if n_genes >= 50:
            n_sub = max(50, int(n_genes * 0.7))
            sub_idx = rng.choice(n_genes, size=n_sub, replace=False)
            sub_genes = [gene_names[i] for i in sorted(sub_idx)]
            sig_sub = pd.DataFrame(
                type_means[:, sorted(sub_idx)].T, index=sub_genes, columns=cell_types,
            )
            sig_sub = sig_sub / sig_sub.sum(axis=0)  # column-normalise
            if n_sub < n_genes * 0.95:
                self._signatures.append(sig_sub)
                if self.verbose:
                    print(f"  Sig (subsample): {sig_sub.shape} — {n_sub} random genes")

    def predict(self, bulk_expr: pd.DataFrame) -> pd.DataFrame:
        """Predict proportions for bulk samples.

        Parameters
        ----------
        bulk_expr : (n_genes, n_samples) bulk expression matrix, indexed by
            gene name (rows) and sample ID (columns).

        Returns
        -------
        (n_samples, n_types) predicted proportions.
        """
        if not self._signatures:
            raise RuntimeError("Call fit() before predict().")
        if self._cell_types is None:
            raise RuntimeError("Cell types not set — call fit() first.")

        cell_types = self._cell_types
        n_samples = bulk_expr.shape[1]
        sample_names = bulk_expr.columns.tolist()

        all_results: List[pd.DataFrame] = []

        for idx, sig_df in enumerate(self._signatures):
            if self.verbose:
                print(f"  Processing signature {idx + 1}/{len(self._signatures)}")

            # Align signature genes to bulk genes by common gene names
            common = sig_df.index.intersection(bulk_expr.index)
            if len(common) < 5:
                if self.verbose:
                    print(f"    Skipping: only {len(common)} common genes")
                continue

            sig_final = sig_df.loc[common].values.astype(np.float64)
            sig_types = sig_df.columns.tolist()

            # Deconvolve each sample
            subj_props = np.zeros((n_samples, len(cell_types)), dtype=np.float64)

            for i in range(n_samples):
                bulk_i = bulk_expr.loc[common, sample_names[i]].values.astype(np.float64)
                try:
                    x = self._solve(sig_final, bulk_i)
                except Exception as exc:
                    warnings.warn(
                        f"Solver failed for signature {idx}, sample {i}: {exc}"
                    )
                    x = np.full(len(sig_types), 1.0 / len(sig_types))

                # Map signature cell types to global cell types
                for j, ct in enumerate(sig_types):
                    if ct in cell_types:
                        subj_props[i, cell_types.index(ct)] = x[j]

            all_results.append(
                pd.DataFrame(subj_props, index=sample_names, columns=cell_types)
            )

        if not all_results:
            if self.verbose:
                print("  WARNING: all signatures failed — returning uniform")
            return pd.DataFrame(
                np.full((n_samples, len(cell_types)), 1.0 / len(cell_types)),
                index=sample_names,
                columns=cell_types,
            )

        final = self._ensemble(
            all_results,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Ensure all cell types are present and in the right order
        for ct in cell_types:
            if ct not in final.columns:
                final[ct] = 0.0
        final = final[cell_types]

        return final


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Load H5 -> fit ReCIDE -> predict -> evaluate -> write outputs."""
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        params = cfg.get("params", {})
    else:
        params = {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ReCIDE Deconvolution")
    print("=" * 60)

    # ── 1. Load H5 via unified data_loader ────────────────────────────
    print("\n[1] Loading data from H5 ...")
    h5_path = Path(args.h5)
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.data_loader import load_data

    bundle = load_data(str(h5_path))
    sc_ref = bundle.sc_ref
    if sc_ref is None:
        raise ValueError("H5 file does not contain singleCellExpr data")

    # sc_ref: AnnData (n_cells, n_genes), obs['cell_type'], var_names=genes
    sc_expr = sc_ref.X if not hasattr(sc_ref.X, 'toarray') else sc_ref.X.toarray()
    sc_expr = np.asarray(sc_expr, dtype=np.float64)
    sc_genes = list(sc_ref.var_names)
    sc_labels = list(sc_ref.obs['cell_type'])

    # bundle.bulk: DataFrame (n_samples, n_genes)
    bulk_df_full = bundle.bulk
    sample_names = list(bulk_df_full.index)
    bulk_gene_names = list(bulk_df_full.columns)

    # Align gene sets
    common_genes = sorted(set(sc_genes) & set(bulk_gene_names))
    if len(common_genes) == 0:
        raise ValueError("No common genes between scRNA-seq and bulk.")

    sc_gene_to_idx   = {g: i for i, g in enumerate(sc_genes)}
    bulk_gene_to_idx = {g: i for i, g in enumerate(bulk_gene_names)}
    sc_idx   = [sc_gene_to_idx[g]   for g in common_genes]
    bulk_idx = [bulk_gene_to_idx[g] for g in common_genes]

    sc_expr     = sc_expr[:, sc_idx]                                    # (n_cells, n_common)
    bulk_values = bulk_df_full.values[:, bulk_idx].T.astype(np.float64) # (n_common, n_samples)

    print(f"    scRNA: {sc_expr.shape}, bulk: {bulk_values.shape}, "
          f"common genes: {len(common_genes)}")

    # ── Build AnnData-like sc_ref ─────────────────────────────────────
    unique_types = sorted(set(sc_labels))
    print(f"    Cell types: {unique_types}")

    # ── 2. Fit ReCIDE ─────────────────────────────────────────────────
    print("\n[2] Fitting ReCIDE model ...")
    verbose = params.get("verbose", True)
    model = ReCIDE(
        n_markers=params.get("n_markers", 100),
        n_pseudo_subjects=params.get("n_pseudo_subjects", 5),
        solver=params.get("solver", "nnls"),
        random_state=params.get("random_state", 42),
        verbose=verbose,
    )
    model.fit(
        X=sc_expr,
        labels=np.array(sc_labels),
        cell_types=unique_types,
        gene_names=common_genes,
    )

    # ── 3. Predict ────────────────────────────────────────────────────
    print("\n[3] Predicting proportions ...")
    bulk_df = pd.DataFrame(bulk_values, index=common_genes, columns=sample_names)
    pred_props = model.predict(bulk_df)
    print(f"    Predictions: {pred_props.shape}")

    # ── 4. Evaluate against ground truth ─────────────────────────────
    print("\n[4] Evaluating ...")
    if args.ground_truth is None:
        raise ValueError("--ground-truth is required.")

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    sys.path.insert(0, str(PROJECT_ROOT))
    from core.metrics import evaluate_deconvolution

    true_props_raw = pd.read_csv(str(gt_path))
    first_col = true_props_raw.columns[0]
    if (
        true_props_raw[first_col].dtype in (object, str)
        or str(true_props_raw[first_col].iloc[0]).startswith("sample_")
    ):
        true_props = true_props_raw.set_index(first_col)
    else:
        true_props = true_props_raw

    # Align common samples and cell types
    common_samples = true_props.index.intersection(pred_props.index)
    common_types = [ct for ct in true_props.columns if ct in pred_props.columns]

    if len(common_samples) == 0 and len(common_types) > 0:
        if len(true_props) == len(pred_props):
            true_props.index = pred_props.index[:len(true_props)]
            common_samples = list(true_props.index)
        else:
            raise ValueError(
                f"No common samples: pred has {len(pred_props)}, "
                f"GT has {len(true_props)}"
            )

    if len(common_samples) == 0:
        raise ValueError("No common samples between predictions and ground truth.")
    if len(common_types) == 0:
        raise ValueError("No common cell types between predictions and ground truth.")

    true_slice = true_props.loc[common_samples, common_types]
    pred_slice = pred_props.loc[common_samples, common_types]

    metrics = evaluate_deconvolution(
        true_slice.values, pred_slice.values,
        cell_types=list(common_types),
    )
    metrics["method"] = "recide"
    metrics["n_samples"] = int(true_slice.shape[0])
    metrics["n_cell_types"] = int(true_slice.shape[1])

    # ── 5. Write outputs ─────────────────────────────────────────────
    pred_props.to_csv(output_dir / "proportions.csv")
    print(f"    Predictions -> {output_dir / 'proportions.csv'}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Metrics    -> {output_dir / 'metrics.json'}")

    # ── 6. Summary ───────────────────────────────────────────────────
    print(f"\n  Results (ReCIDE):")
    print(f"    Pearson r:  {metrics['pearson_mean']:.4f}")
    print(f"    SCorr:      {metrics['scorr_mean']:.4f}")
    print(f"    CCorr:      {metrics['ccorr_mean']:.4f}")
    print(f"    RMSE:       {metrics['rmse_overall']:.4f}")
    print(f"    MAE:        {metrics['mae_overall']:.4f}")
    print(f"    Wilcoxon p: {metrics['wt_mean']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
