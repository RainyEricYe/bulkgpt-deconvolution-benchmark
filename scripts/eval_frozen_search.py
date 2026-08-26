#!/usr/bin/env python3
"""Frozen backbone search: run all 6 deconv strategies for one (backbone, dataset).

Uses the to_publish framework infrastructure:
  - ``core.data_loader`` for H5 loading and gene alignment
  - ``core.deconv.frozen_eval.ResourceTracker`` for resource tracking
  - ``core.deconv.frozen_search.align_predictions_to_gt`` for GT label mapping
  - Standard 4-file output + pseudo-bulk CSVs

Usage:
    python scripts/eval_frozen_search.py --backbone scgpt --dataset sdy67
    python scripts/eval_frozen_search.py --backbone all --dataset all --strategies ridge_cv,nusvr
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_loader import load_data
from core.deconv.frozen_eval import ENCODE_FN, ResourceTracker
from core.deconv.frozen_search import (
    STRATEGIES, PREDICT_FN, generate_pseudo_bulk, _compute_centroids,
    save_strategy_outputs, align_predictions_to_gt,
    _score, _rmsd, _NpEncoder, SEED,
)


def _load_and_align(h5_path: str, gt_csv_path: str) -> tuple[
    np.ndarray, list[str], list[str], np.ndarray, list[str], pd.DataFrame,
]:
    """Load H5 via framework data_loader, align genes between bulk and scRNA.

    Handles DeconBenchmark H5 orientation ambiguity: when ``n_genes ≈ n_cells``
    the data_loader may mis-detect orientation.  We verify by comparing H5
    rownames against bulk gene symbols — if they overlap strongly the matrix
    is ``(n_genes, n_cells)`` and needs transpose.
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        sc_raw = f["singleCellExpr/values"][:]
        sc_rn = [g.decode() for g in f["singleCellExpr/rownames"][:]]
        sc_cn = [g.decode() for g in f["singleCellExpr/colnames"][:]]
        sc_labels_raw = [l.decode() for l in f["singleCellLabels/values"][:]]
        bulk_raw = f["bulk/values"][:]
        bulk_rn = [g.decode() for g in f["bulk/rownames"][:]]
        bulk_cn = [g.decode() for g in f["bulk/colnames"][:]]

    # -- Determine RNA orientation: compare H5 rownames/colnames vs bulk genes
    rn_overlap = len(set(sc_rn) & set(bulk_rn))
    cn_overlap = len(set(sc_cn) & set(bulk_rn))

    if rn_overlap >= 50:
        if len(sc_rn) == sc_raw.shape[0] and len(sc_labels_raw) != sc_raw.shape[0]:
            # DeconBenchmark: (n_genes, n_cells). rownames = genes on axis 0.
            # Labels do NOT match axis 0 → axis 0 is genes, not cells.
            sc_values = sc_raw.T.astype(np.float32)          # → (n_cells, n_genes)
            sc_genes = sc_rn
        elif len(sc_rn) == sc_raw.shape[1]:
            # Already (n_cells, n_genes). rownames = genes on axis 1.
            sc_values = np.asarray(sc_raw, dtype=np.float32)
            sc_genes = sc_rn
        elif len(sc_rn) == sc_raw.shape[0] and len(sc_labels_raw) == sc_raw.shape[0]:
            # rownames contain gene symbols but sit on the cell axis (labels
            # match axis 0 = cells).  colnames (on the gene axis) are cell
            # barcodes, not genes.  Best-effort heuristic: use first N genes
            # from the sorted rownames ∩ bulk_genes intersection.
            n_cols = sc_raw.shape[1]
            common = sorted(set(sc_rn) & set(bulk_rn))
            if len(common) >= n_cols:
                sc_values = np.asarray(sc_raw, dtype=np.float32)
                sc_genes = common[:n_cols]
            else:
                raise RuntimeError(
                    f"Cannot determine scRNA gene names: rownames={len(sc_rn)} "
                    f"gene symbols on cell axis, colnames=cell barcodes, "
                    f"common∩bulk={len(common)} < n_genes={n_cols}"
                )
        else:
            sc_values = np.asarray(sc_raw, dtype=np.float32)
            sc_genes = sc_rn
    elif cn_overlap >= 50 and len(sc_cn) == sc_raw.shape[1]:
        # Already (n_cells, n_genes). colnames = genes.
        sc_values = np.asarray(sc_raw, dtype=np.float32)
        sc_genes = sc_cn
    else:
        # Ambiguous — use data_loader's judgment
        bundle = load_data(h5_path)
        ref_x = bundle.sc_ref.X
        if hasattr(ref_x, "toarray"):
            ref_x = ref_x.toarray()
        sc_values = np.asarray(ref_x, dtype=np.float32)
        sc_genes = list(bundle.sc_ref.var_names)
        sc_labels_raw = list(bundle.sc_ref.obs["cell_type"])

    # -- Determine bulk orientation: genes on axis 1 → (n_samples, n_genes) -----
    if bulk_raw.shape[1] == len(bulk_rn):
        bulk_values = bulk_raw.astype(np.float32)
    elif bulk_raw.shape[0] == len(bulk_rn):
        bulk_values = bulk_raw.T.astype(np.float32)
    else:
        bulk_values = np.asarray(bulk_raw, dtype=np.float32)

    # -- Filter bad labels --------------------------------------------------
    valid = [i for i, l in enumerate(sc_labels_raw)
             if isinstance(l, str) and l != "nan"]
    sc_values = sc_values[valid]
    sc_labels = [sc_labels_raw[i] for i in valid]

    # -- Cap reference cells (avoid OOM on large references) ----------------
    MAX_REF_CELLS = 15000
    if sc_values.shape[0] > MAX_REF_CELLS:
        rng = np.random.default_rng(42)
        idx = rng.choice(sc_values.shape[0], MAX_REF_CELLS, replace=False)
        sc_values = sc_values[idx]
        sc_labels = [sc_labels[i] for i in idx]

    # -- Gene alignment (intersect scRNA genes ∩ bulk genes) ----------------
    common = sorted(set(sc_genes) & set(bulk_rn))
    if len(common) < 50:
        raise RuntimeError(
            f"Too few common genes between bulk ({len(bulk_rn)}) and "
            f"scRNA ({len(sc_genes)}): {len(common)}"
        )
    ref_idx = [sc_genes.index(g) for g in common]
    bulk_idx = [bulk_rn.index(g) for g in common]
    sc_values = sc_values[:, ref_idx]
    bulk_values = bulk_values[:, bulk_idx]

    # -- Ground truth -------------------------------------------------------
    gt_df = pd.read_csv(gt_csv_path)
    if gt_df.shape[1] > 0:
        first_col = gt_df.iloc[:, 0]
        if first_col.dtype == object and not first_col.apply(
            lambda x: isinstance(x, (int, float))
        ).any():
            gt_df = pd.read_csv(gt_csv_path, index_col=0)

    return (sc_values, common, sc_labels,
            bulk_values, common, gt_df)


def run_search(
    backbone: str, dataset: str,
    strategies: list[str] | None = None,
    seed: int = SEED,
) -> dict:
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "2_real_bulk"
    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "2_realbulk"
    h5_path = str(DATA_DIR / f"{dataset}.h5")
    gt_path = str(DATA_DIR / f"{dataset}_gt.csv")

    print(f"\n{'='*60}")
    print(f"Backbone: {backbone}, Dataset: {dataset}")
    print(f"{'='*60}")

    rt = ResourceTracker()
    rt.__enter__()
    encode_fn = ENCODE_FN[backbone]

    # -- [1] Load & align data --------------------------------------------
    print("\n[1] Loading & aligning data...")
    ref_values, ref_genes, ref_labels, bulk_values, bulk_genes, gt_df = \
        _load_and_align(h5_path, gt_path)
    gt_columns = list(gt_df.columns)
    print(f"  Bulk: {bulk_values.shape}, Ref: {ref_values.shape}, "
          f"Common genes: {len(ref_genes)}")

    cell_types = sorted(set(ref_labels))
    t2i = {t: i for i, t in enumerate(cell_types)}
    label_indices = np.array([t2i[l] for l in ref_labels], dtype=np.int32)

    # -- [2] Encode reference cells ---------------------------------------
    print(f"\n[2] Encoding reference cells with {backbone}...")
    rt.start_encode()
    barcodes_ref = [f"ref_{i}" for i in range(ref_values.shape[0])]
    ref_emb = encode_fn(ref_values, ref_genes, barcodes_ref)
    print(f"  Ref embeddings: {ref_emb.shape}")

    # -- [3] Generate pseudo-bulk + centroids -----------------------------
    print("\n[3] Generating pseudo-bulk (10k, 6:2:2)...")
    t0 = time.monotonic()
    train_emb, train_props, val_emb, val_props, test_emb, test_props = \
        generate_pseudo_bulk(ref_emb, label_indices, len(cell_types), seed=seed)
    print(f"  Train:{len(train_emb)} Val:{len(val_emb)} Test:{len(test_emb)} "
          f"[{time.monotonic()-t0:.1f}s]")
    centroids = _compute_centroids(ref_emb, label_indices, len(cell_types))

    # -- [4] Encode real bulk ---------------------------------------------
    print(f"\n[4] Encoding real bulk with {backbone}...")
    barcodes_bulk = [f"bulk_{i}" for i in range(bulk_values.shape[0])]
    bulk_emb = encode_fn(bulk_values, bulk_genes, barcodes_bulk)
    rt.end_encode()
    print(f"  Bulk embeddings: {bulk_emb.shape}")

    # -- [5] Run strategies -----------------------------------------------
    target = strategies or list(STRATEGIES.keys())
    all_results = {}
    strategy_models = {}
    strategy_preds = {}

    print(f"\n[5] Running {len(target)} strategies...")
    t_strat_start = time.monotonic()
    for sname in target:
        t0 = time.monotonic()
        kwargs = {"centroids": centroids} if "centroid" in sname else {}
        results, models = STRATEGIES[sname](
            train_emb, train_props, val_emb, val_props,
            test_emb, test_props, cell_types, **kwargs,
        )
        strategy_models[sname] = models
        macro_r = float(np.nanmean(
            [v["test_r"] for v in results.values() if v["test_r"] is not None]
        ))
        all_results[sname] = {"per_type": results, "pseudo_test_r": round(macro_r, 4)}

        # Predict once, store for both evaluation and saving
        pred = PREDICT_FN[sname](strategy_models[sname], bulk_emb, cell_types)
        pred = np.maximum(pred, 0)
        pred_norm = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-10)
        strategy_preds[sname] = pred_norm
        print(f"  [{sname:18s}] pseudo test r = {macro_r:.4f} "
              f"[{time.monotonic()-t0:.1f}s]")
    strategy_time_s = round(time.monotonic() - t_strat_start, 1)

    # -- [6] Evaluate on real bulk (aligned to GT columns) -----------------
    print(f"\n[6] Evaluating on real bulk (aligned to GT columns)...")
    real_bulk_results = {}
    gt_values = gt_df.values.astype(np.float64)
    for sname in target:
        pred_aligned = align_predictions_to_gt(
            strategy_preds[sname], cell_types, gt_columns, dataset,
        )
        per_ct = {}
        for j, ct in enumerate(gt_columns):
            per_ct[ct] = {
                "pearson_r": round(_score(pred_aligned[:, j], gt_values[:, j]), 4),
                "rmse": round(_rmsd(pred_aligned[:, j], gt_values[:, j]), 4),
            }
        # Only average over GT types with scRNA counterparts (non-zero pred variance)
        valid_r = [
            v["pearson_r"] for j, v in enumerate(per_ct.values())
            if v["pearson_r"] is not None
            and not np.isnan(v["pearson_r"])
        ]
        macro_r = float(np.nanmean(valid_r)) if valid_r else float("nan")
        real_bulk_results[sname] = {
            "per_type": per_ct, "macro_avg": round(macro_r, 4),
        }
        print(f"  [{sname:18s}] real bulk r = {macro_r:.4f}")

    best_s = max(real_bulk_results, key=lambda s: real_bulk_results[s]["macro_avg"])
    best_r = real_bulk_results[best_s]["macro_avg"]
    print(f"\n  >>> BEST: {best_s} (r={best_r:.4f})")

    # -- [7] Save ---------------------------------------------------------
    print(f"\n[7] Saving...")
    base_dir = RESULTS_DIR / dataset / backbone / "search"
    for sname in target:
        pred_aligned = align_predictions_to_gt(
            strategy_preds[sname], cell_types, gt_columns, dataset,
        )
        pseudo_pred = PREDICT_FN[sname](strategy_models[sname], test_emb, cell_types)
        pseudo_pred = np.maximum(pseudo_pred, 0)
        pseudo_pred_norm = pseudo_pred / np.maximum(
            pseudo_pred.sum(axis=1, keepdims=True), 1e-10,
        )

        meta = rt.to_dict(
            backbone=backbone, dataset=dataset, strategy=sname,
            seed=seed, n_pb_total=10000, strategy_time_s=strategy_time_s,
        )
        save_strategy_outputs(
            base_dir / sname, sname, pred_aligned, gt_df,
            pseudo_pred_norm, test_props, all_results[sname], meta,
            cell_types, real_cell_types=gt_columns,
        )

    # Combined summary
    rt.__exit__()
    summary = {
        "backbone": backbone, "dataset": dataset, "seed": seed,
        "best_strategy": best_s, "best_real_bulk_r": best_r,
        "strategies": {}, "wall_time_s": rt.wall_time_s,
    }
    for sname in target:
        summary["strategies"][sname] = {
            "pseudo_test_r": all_results[sname]["pseudo_test_r"],
            "real_bulk_r": real_bulk_results[sname]["macro_avg"],
        }
    with open(base_dir / "search_results.json", "w") as f:
        json.dump(summary, f, indent=2, cls=_NpEncoder)
    print(f"  Saved: {base_dir / 'search_results.json'}")
    print(f"  Total: {rt.wall_time_s:.1f}s")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen backbone strategy search")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    datasets = (
        ["sdy67", "sweetwater", "huuki_myers", "demixsc_retina",
         "altman_Arunachalam"]
        if args.dataset == "all" else [args.dataset]
    )
    backbones = list(ENCODE_FN.keys()) if args.backbone == "all" else [args.backbone]
    strategies = args.strategies.split(",") if args.strategies else None

    for ds in datasets:
        for bb in backbones:
            try:
                run_search(bb, ds, strategies, args.seed)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  ERROR {bb}/{ds}: {e}")


if __name__ == "__main__":
    main()
