#!/usr/bin/env python3
"""BulkFormer + F-statistic gene weighting + global_proj + RidgeCV.

Uses the single-cell reference from the DeconBenchmark H5
(``singleCellExpr/values`` + ``singleCellLabels/values``) to compute
per-type F-statistics for each gene.  F = (between-group variance) /
(within-group variance + epsilon).  These are then used as gene weights:
bulk expression is multiplied by sqrt(F) before passing through
BulkFormer's ``global_expr_proj`` head.

Because weighting is per cell type, each type gets its own weighted
embedding and a separate RidgeCV regressor.  Optionally supports
bootstrap ensemble (``--n-ensemble > 1``) for more robust predictions.

Adapted from the original ``eval_fstat_weight.py`` in the BulkFormer repo.

Usage::

    # Single RidgeCV (default)
    python methods/bulkformer/fstat/run.py \\
        --h5 data/2_real_bulk/sdy67.h5 \\
        --ground-truth data/2_real_bulk/sdy67_gt.csv \\
        --output-dir results/2_realbulk/sdy67/bulkformer/fstat

    # With bootstrap-50 ensemble
    python methods/bulkformer/fstat/run.py \\
        --h5 data/2_real_bulk/sdy67.h5 \\
        --ground-truth data/2_real_bulk/sdy67_gt.csv \\
        --output-dir results/2_realbulk/sdy67/bulkformer/fstat \\
        --n-ensemble 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_to_publish = Path(__file__).resolve().parent.parent.parent.parent
if str(_to_publish) not in sys.path:
    sys.path.insert(0, str(_to_publish))

from core.deconv.frozen_eval import ResourceTracker
from core.deconv.utils import renormalize_props


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _load_h5(path: str) -> dict:
    """Load all groups from a DeconBenchmark H5."""
    import h5py

    data: dict = {}
    with h5py.File(path, "r") as f:

        def _walk(name: str, obj: h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[()]

        f.visititems(_walk)

    decoded: dict = {}
    for k, v in data.items():
        if hasattr(v, "dtype"):
            if v.dtype.kind == "S":
                # Fixed-length byte strings
                v = [x.decode() if isinstance(x, bytes) else str(x) for x in v]
            elif v.dtype.kind == "O" and v.ndim == 1:
                # Variable-length strings (object dtype) — decode bytes
                v = [x.decode() if isinstance(x, bytes) else str(x) for x in v]
        decoded[k] = v
    return decoded


def _read_gt_csv(path: str) -> pd.DataFrame:
    """Read ground-truth CSV; first column as index if it contains strings."""
    df = pd.read_csv(path)
    first_col = df.iloc[:, 0]
    if first_col.dtype in (object, str) or first_col.dtype.name == "object":
        df = df.set_index(df.columns[0])
    return df


def _split_indices(
    n_bulk: int, dataset_name: str, seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Train/test split matching evaluate_real_bulk_ridge()."""
    from sklearn.model_selection import train_test_split

    if dataset_name == "sdy67":
        train_n = 200
        test_n = min(50, n_bulk - 200)
        train_idx = np.arange(train_n)
        test_idx = np.arange(train_n, train_n + test_n)
        split_label = "fixed_200_50"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(n_bulk), test_size=0.2, random_state=seed,
        )
        split_label = "random_80_20"
    return train_idx, test_idx, split_label


# ── F-stat computation ───────────────────────────────────────────────────────────


def compute_per_type_fstat(
    ref_vals: np.ndarray,
    ref_labels: list[str],
    ref_genes: list[str],
    bf_genes: list[str],
) -> dict[str, np.ndarray]:
    """Compute per-type F-statistic (between / within variance) per gene.

    F = (between-group variance) / (within-group variance + 1e-10)

    Genes not present in the BulkFormer vocabulary are discarded.

    Returns:
        Dict mapping cell type (str) -> ``(20010,)`` F-stat array.
    """
    g2i = {g.upper(): i for i, g in enumerate(bf_genes)}
    bf_indices: list[int] = []
    valid_rows: list[int] = []
    for gi, g in enumerate(ref_genes):
        idx = g2i.get(str(g).upper())
        if idx is not None:
            bf_indices.append(idx)
            valid_rows.append(gi)

    print(f"  F-stat: aligned {len(valid_rows)}/{len(ref_genes)} ref genes")
    if len(valid_rows) < 10:
        print("  WARNING: too few aligned ref genes, F-stat may be unreliable")

    ref_aligned = np.zeros((ref_vals.shape[0], len(bf_genes)), dtype=np.float64)
    ref_aligned[:, bf_indices] = ref_vals[:, valid_rows].astype(np.float64)

    labels_arr = np.array(ref_labels)
    unique_types = sorted(set(ref_labels))
    f_stats: dict[str, np.ndarray] = {}

    for ct in unique_types:
        mask = labels_arr == ct
        n_pos = mask.sum()
        n_neg = (~mask).sum()
        if n_pos < 2 or n_neg < 2:
            f_stats[ct] = np.ones(len(bf_genes), dtype=np.float64)
            continue

        global_mean = ref_aligned.mean(axis=0)
        mean_pos = ref_aligned[mask].mean(axis=0)
        mean_neg = ref_aligned[~mask].mean(axis=0)
        between_var = (
            n_pos * (mean_pos - global_mean) ** 2
            + n_neg * (mean_neg - global_mean) ** 2
        )

        var_pos = ref_aligned[mask].var(axis=0, ddof=1)
        var_neg = ref_aligned[~mask].var(axis=0, ddof=1)
        within_var = (
            (n_pos - 1) * var_pos + (n_neg - 1) * var_neg
        ) / (n_pos + n_neg - 2)

        f_stats[ct] = between_var / (within_var + 1e-10)

    return f_stats


def _map_gt_to_sc(gt_type: str, sc_types: list[str]) -> list[str]:
    """Map a GT cell type to matching SC types via substring matching."""
    gt_lower = gt_type.lower()
    direct = [s for s in sc_types if s.lower() == gt_lower]
    if direct:
        return direct
    substring = [s for s in sc_types if gt_lower in s.lower() or s.lower() in gt_lower]
    if substring:
        return substring
    return sc_types  # fallback: use all


# ── Main ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BulkFormer + F-stat gene weighting + RidgeCV",
    )
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mode", default="predict", help=argparse.SUPPRESS)
    parser.add_argument(
        "--h5", required=True,
        help="DeconBenchmark H5 (must contain singleCellExpr/ + singleCellLabels/)",
    )
    parser.add_argument("--ground-truth", required=True, help="Path to GT CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-ensemble", type=int, default=0,
        help="Bootstrap iterations (0 = single RidgeCV)",
    )
    parser.add_argument("--scaler", action="store_true",
                        help="Use StandardScaler before RidgeCV")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.scaler and not out_dir.name.endswith("_scaler"):
        out_dir = out_dir.parent / f"{out_dir.name}_scaler"
    dataset_name = Path(args.h5).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading H5 data...")
    h5_data = _load_h5(args.h5)

    bulk_vals = np.asarray(h5_data.get("bulk/values", []))
    bulk_rownames = list(h5_data.get("bulk/rownames", []))
    bulk_colnames = list(h5_data.get("bulk/colnames", []))
    # Canonical format: bulk/values is always (n_samples, n_genes)
    # Canonical format: bulk/values is always (n_samples, n_genes)
    # Canonical format: bulk/values is always (n_samples, n_genes)
    # Canonical format: bulk/values is always (n_samples, n_genes)
    # Canonical format: bulk/values is always (n_samples, n_genes)
    # Canonical format: bulk/values is always (n_samples, n_genes)

    ref_vals = np.asarray(h5_data.get("singleCellExpr/values", []))
    n_cells, n_genes = ref_vals.shape
    sc_rn = list(h5_data.get("singleCellExpr/rownames", []))
    sc_cn = list(h5_data.get("singleCellExpr/colnames", []))

    # Gene names: use rownames (gene-like, one per cell) deduplicated to n_genes.
    # Colnames are typically cell barcodes (cell_0, c0, ...), not gene symbols.
    if sc_rn and len(sc_rn) >= n_genes:
        ref_genes_raw = sc_rn[:n_genes]  # first n_genes entries as gene names
        ref_cells_raw = sc_rn[n_genes:]  # remaining entries (unused)
    elif sc_cn and len(sc_cn) == n_genes:
        ref_genes_raw = sc_cn
        ref_cells_raw = sc_rn
    else:
        ref_genes_raw = sc_rn or sc_cn
        ref_cells_raw = sc_cn or sc_rn

    ref_genes = [str(x) for x in ref_genes_raw]
    ref_labels_raw = h5_data.get("singleCellLabels/values", [])
    ref_labels = [x.decode() if isinstance(x, bytes) else str(x) for x in ref_labels_raw]
    # If labels don't match n_cells, drop them
    if ref_labels and len(ref_labels) != ref_vals.shape[0]:
        print(f"  WARNING: SC labels ({len(ref_labels)}) != n_cells ({ref_vals.shape[0]}), dropping")
        ref_labels = []

    genes_for_alignment = (
        bulk_colnames if len(bulk_colnames) == bulk_vals.shape[1] else bulk_rownames
    )

    gt_df = _read_gt_csv(args.ground_truth)
    n_bulk = bulk_vals.shape[0]
    print(
        f"  Bulk: {n_bulk} samples, {bulk_vals.shape[1]} genes\n"
        f"  SC ref: {ref_vals.shape[0]} cells, {ref_vals.shape[1]} genes, "
        f"{len(set(ref_labels))} types\n"
        f"  GT: {len(gt_df.columns)} types",
    )

    with ResourceTracker() as rt:
        # ── Load BulkFormer encoder once ────────────────────────────────
        t0 = time.monotonic()
        from methods.bulkformer.model import BulkFormerEncoder

        encoder = BulkFormerEncoder(pretrained=True)
        encoder._load()
        bf_genes = encoder._gene_list
        bf_device = encoder._device
        print(f"BulkFormer loaded in {time.monotonic()-t0:.1f}s")

        # ── Align bulk genes to BulkFormer vocabulary ───────────────────
        g2i = {g.upper(): i for i, g in enumerate(bf_genes)}
        bf_bulk = np.full((n_bulk, len(bf_genes)), -10.0, dtype=np.float32)
        matched = 0
        for i, g in enumerate(genes_for_alignment):
            idx = g2i.get(str(g).upper())
            if idx is not None:
                bf_bulk[:, idx] = bulk_vals[:, i]
                matched += 1
        print(f"  Aligned {matched}/{bulk_vals.shape[1]} bulk genes")

        # ── Compute F-stats from SC ref ─────────────────────────────────
        f_stats = compute_per_type_fstat(ref_vals, ref_labels, ref_genes, bf_genes)
        sc_type_names = list(f_stats.keys())
        print(f"  Computed F-stats for {len(sc_type_names)} SC types")

        # ── Split ───────────────────────────────────────────────────────
        train_idx, test_idx, split_label = _split_indices(
            n_bulk, dataset_name, args.seed,
        )
        train_x_raw = bf_bulk[train_idx]
        test_x_raw = bf_bulk[test_idx]

        # ── For each GT type: weight -> encode -> RidgeCV ───────────────
        from sklearn.linear_model import RidgeCV
        from methods.bulkformer.bootstrap_utils import DEFAULT_ALPHAS, bootstrap_ridge

        gt_types = list(gt_df.columns)
        n_types = len(gt_types)
        gt_values = gt_df.values.astype(np.float64)

        test_pred = np.zeros((len(test_idx), n_types))
        per_type_info: dict[str, dict] = {}

        t0 = time.monotonic()
        with torch.no_grad():
            for j, ct in enumerate(gt_types):
                matching = _map_gt_to_sc(ct, sc_type_names)
                if matching and all(m in f_stats for m in matching):
                    weights = np.maximum.reduce([f_stats[m] for m in matching]).astype(np.float32)
                else:
                    weights = np.ones(len(bf_genes), dtype=np.float32)
                weights = np.sqrt(np.maximum(weights, 0.0)).astype(np.float32)

                weighted_train = train_x_raw * weights[np.newaxis, :]
                weighted_test = test_x_raw * weights[np.newaxis, :]

                x_t = torch.from_numpy(weighted_train).to(bf_device)
                emb_train = encoder._model.global_expr_proj(x_t).cpu().numpy()
                x_t = torch.from_numpy(weighted_test).to(bf_device)
                emb_test = encoder._model.global_expr_proj(x_t).cpu().numpy()

                y_train = np.maximum(gt_values[train_idx, j], 0.0)

                if args.n_ensemble > 0:
                    pred_mean, pred_std, details = bootstrap_ridge(
                        emb_train,
                        pd.DataFrame(y_train.reshape(-1, 1), columns=[ct]),
                        emb_test,
                        n_ensemble=args.n_ensemble,
                        seed=args.seed,
                        alphas=DEFAULT_ALPHAS,
                    )
                    test_pred[:, j] = pred_mean[:, 0]
                    per_type_info[ct] = {
                        "method": f"bootstrap_{args.n_ensemble}",
                        "sc_matched": matching,
                    }
                else:
                    if args.scaler:
                        from sklearn.preprocessing import StandardScaler as _SS
                        _s = _SS()
                        _train_s = _s.fit_transform(emb_train)
                        _test_s = _s.transform(emb_test)
                    else:
                        _train_s, _test_s = emb_train, emb_test
                    ridge = RidgeCV(alphas=DEFAULT_ALPHAS).fit(_train_s, y_train)
                    test_pred[:, j] = ridge.predict(_test_s)
                    per_type_info[ct] = {
                        "method": "single_ridgecv",
                        "best_alpha": float(ridge.alpha_),
                        "sc_matched": matching,
                    }

        encode_ridge_time = time.monotonic() - t0
        rt.end_ridge()

    # ── Clip + renormalize ─────────────────────────────────────────────
    test_pred = renormalize_props(test_pred, zero_fill="zero")

    # ── Evaluate ────────────────────────────────────────────────────────
    _project = Path(__file__).resolve().parent.parent.parent.parent
    if str(_project) not in sys.path:
        sys.path.insert(0, str(_project))
    from core.metrics import evaluate_deconvolution

    deconbench = evaluate_deconvolution(
        gt_values[test_idx], test_pred, gt_types,
    )

    # Per-type metrics
    per_type_r: dict[str, float | None] = {}
    per_type_rmse: dict[str, float | None] = {}
    for j, ct in enumerate(gt_types):
        y_true = gt_values[test_idx, j]
        mask = ~np.isnan(y_true)
        if mask.sum() >= 2 and np.std(y_true[mask]) > 1e-10:
            r = float(np.corrcoef(test_pred[mask, j], y_true[mask])[0, 1])
            per_type_r[ct] = round(r, 4) if not np.isnan(r) else None
        else:
            per_type_r[ct] = None
        if mask.sum() >= 2:
            per_type_rmse[ct] = round(
                float(np.sqrt(np.mean((test_pred[mask, j] - y_true[mask]) ** 2))),
                4,
            )
        else:
            per_type_rmse[ct] = None

    # ── Full-sample predictions ─────────────────────────────────────────
    full_pred = np.zeros((n_bulk, n_types))
    with torch.no_grad():
        for j, ct in enumerate(gt_types):
            matching = _map_gt_to_sc(ct, sc_type_names)
            if matching and all(m in f_stats for m in matching):
                weights = np.maximum.reduce([f_stats[m] for m in matching]).astype(np.float32)
            else:
                weights = np.ones(len(bf_genes), dtype=np.float32)
            weights = np.sqrt(np.maximum(weights, 0.0)).astype(np.float32)

            weighted_all = bf_bulk * weights[np.newaxis, :]
            x_t = torch.from_numpy(weighted_all).to(bf_device)
            emb_all = encoder._model.global_expr_proj(x_t).cpu().numpy()

            y_all = np.maximum(gt_values[:, j], 0.0)
            if args.scaler:
                from sklearn.preprocessing import StandardScaler as _SS
                _s = _SS()
                _full_s = _s.fit_transform(emb_all)
            else:
                _full_s = emb_all
            ridge = RidgeCV(alphas=DEFAULT_ALPHAS).fit(_full_s, y_all)
            full_pred[:, j] = ridge.predict(_full_s)

    full_pred = renormalize_props(full_pred, zero_fill="zero")
    full_pred_df = pd.DataFrame(full_pred, index=gt_df.index, columns=gt_types)
    full_pred_df.to_csv(out_dir / "proportions.csv")

    # ── Save outputs ─────────────────────────────────────────────────────
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(deconbench, f, indent=2)

    ridge_info = {
        "n_ensemble": args.n_ensemble,
        "n_gene_weights": matched,
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "split": split_label,
        "encode_ridge_time_s": round(encode_ridge_time, 3),
        "pearson_per_type": per_type_r,
        "rmse_per_type": per_type_rmse,
        "per_type": per_type_info,
    }
    with open(out_dir / "ridge_metrics.json", "w") as f:
        json.dump(ridge_info, f, indent=2)

    meta = {
        "embedding_dim": 640,
        "n_ensemble": args.n_ensemble,
        "split": split_label,
        "seed": args.seed,
        **rt.to_dict(backbone="bulkformer/fstat", dataset=dataset_name),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    r = deconbench.get("pearson_mean", float("nan"))
    print(f"\nDone.  Pearson r = {r:.4f}  [{rt.wall_time_s}s]")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
