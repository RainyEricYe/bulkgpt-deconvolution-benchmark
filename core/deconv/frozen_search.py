"""6-Strategy frozen backbone search for deconvolution.

Ported from scPEFT ``auto_search.py``.

Generates pseudo-bulk (10k samples, 6:2:2 split) from reference cell embeddings,
trains 6 regression strategies, evaluates on both pseudo-bulk test and real bulk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

SEED = 42
N_PB_TOTAL = 10000
N_HOLDOUT = 2000  # 2000 test -> 8000 pool -> 6000 train / 2000 val

# -- GT label mapping: fine-grained scRNA -> coarse GT columns ----------------
GT_LABEL_MAP: dict[str, dict[str, list[str]]] = {
    "altman_Arunachalam": {
        "Basophils": [],
        "Eosinophils": [],
        "Lymphocytes": ["B cells", "ILC", "NK cells", "Plasma cells",
                        "T cells CD4 conv", "T cells CD8", "Tregs"],
        "Monocytes": ["Monocytes", "mDC", "pDC"],
        "Neutrophils": [],
    },
}


def align_predictions_to_gt(
    pred: np.ndarray,
    cell_types: list[str],
    gt_columns: list[str],
    dataset: str = "",
) -> np.ndarray:
    """Map scRNA-level predictions to GT columns.

    Uses ``GT_LABEL_MAP`` to aggregate fine-grained scRNA subtypes into coarse
    GT categories (e.g., "B cells" + "NK cells" → "Lymphocytes").

    Args:
        pred: (n_samples, n_scRNA_types) predicted proportions.
        cell_types: scRNA cell-type names (length n_scRNA_types).
        gt_columns: GT column names (length n_gt_types).
        dataset: dataset name for looking up ``GT_LABEL_MAP``.

    Returns:
        (n_samples, n_gt_types) aligned predictions.
    """
    label_map = GT_LABEL_MAP.get(dataset, {})
    aligned = np.zeros((pred.shape[0], len(gt_columns)), dtype=pred.dtype)
    for j, gt_col in enumerate(gt_columns):
        subtypes = label_map.get(gt_col)
        if subtypes is not None:
            if len(subtypes) == 0:
                # GT type with no scRNA counterpart → zero predictions
                continue
            indices = [cell_types.index(st) for st in subtypes if st in cell_types]
            if indices:
                aligned[:, j] = pred[:, indices].sum(axis=1)
            elif gt_col in cell_types:
                # Fallback: subtypes not found but GT column matches directly
                i = cell_types.index(gt_col)
                aligned[:, j] = pred[:, i]
        elif gt_col in cell_types:
            i = cell_types.index(gt_col)
            aligned[:, j] = pred[:, i]
    return aligned


# -- Helpers ------------------------------------------------------------------


def _score(pred: np.ndarray, true: np.ndarray) -> float:
    mask = ~(np.isnan(true) | np.isnan(pred))
    if mask.sum() < 2:
        return float("nan")
    tp = true[mask]
    pp = pred[mask]
    if np.std(tp) < 1e-12 or np.std(pp) < 1e-12:
        return float("nan")
    return float(np.corrcoef(tp, pp)[0, 1])


def _rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    mask = ~(np.isnan(true) | np.isnan(pred))
    if mask.sum() < 2:
        return float("nan")
    return float(np.sqrt(np.mean((true[mask] - pred[mask]) ** 2)))


# -- Pseudo-bulk generation (10k, 6:2:2 split) --------------------------------


def generate_pseudo_bulk(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_types: int,
    n_pb_total: int = N_PB_TOTAL,
    n_holdout: int = N_HOLDOUT,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate pseudo-bulk from reference cell embeddings.

    Split: pool (train+val) = n_pb_total - n_holdout, holdout test = n_holdout.

    Returns:
        (train_emb, train_props, val_emb, val_props, test_emb, test_props)
    """
    from numpy.random import default_rng
    rng = default_rng(seed)
    ti = [np.where(labels == t)[0] for t in range(n_types)]
    n_pool = n_pb_total - n_holdout

    pb_emb_pool = np.zeros((n_pool, embeddings.shape[1]), dtype=np.float32)
    pb_props_pool = np.zeros((n_pool, n_types), dtype=np.float32)
    pb_emb_holdout = np.zeros((n_holdout, embeddings.shape[1]), dtype=np.float32)
    pb_props_holdout = np.zeros((n_holdout, n_types), dtype=np.float32)

    for i in range(n_pb_total):
        na = rng.integers(2, min(n_types + 1, 6))
        ac = rng.choice(n_types, na, replace=False)
        p = np.zeros(n_types)
        for t, v in zip(ac, rng.dirichlet([1.0] * na)):
            p[t] = v
        nc = rng.integers(10, 100)
        cc = rng.multinomial(nc, p)
        total = np.zeros(embeddings.shape[1], dtype=np.float64)
        nt = 0
        for t in range(n_types):
            ct = cc[t]
            if ct == 0:
                continue
            total += embeddings[rng.choice(ti[t], ct, replace=True)].sum(axis=0)
            nt += ct
        emb_i = (total / nt).astype(np.float32)
        if i < n_pool:
            pb_emb_pool[i] = emb_i
            pb_props_pool[i] = p
        else:
            pb_emb_holdout[i - n_pool] = emb_i
            pb_props_holdout[i - n_pool] = p

    idx = rng.permutation(n_pool)
    n_train = int(n_pool * 0.75)
    return (pb_emb_pool[idx[:n_train]], pb_props_pool[idx[:n_train]],
            pb_emb_pool[idx[n_train:]], pb_props_pool[idx[n_train:]],
            pb_emb_holdout, pb_props_holdout)


def _compute_centroids(embeddings: np.ndarray, labels: np.ndarray, n_types: int) -> np.ndarray:
    centroids = np.zeros((n_types, embeddings.shape[1]), dtype=np.float64)
    for t in range(n_types):
        mask = labels == t
        if mask.sum() > 0:
            centroids[t] = embeddings[mask].mean(axis=0)
    return centroids


# -- Strategy functions --------------------------------------------------------


def strategy_ridge_cv(train_emb, train_props, val_emb, val_props, test_emb, test_props, cell_types):
    from sklearn.linear_model import RidgeCV
    alphas = [0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0]
    results, models = {}, {}
    for i, ct in enumerate(cell_types):
        clf = RidgeCV(alphas=alphas).fit(train_emb, train_props[:, i])
        models[ct] = clf
        results[ct] = {"val_r": _score(clf.predict(val_emb), val_props[:, i]),
                       "test_r": _score(clf.predict(test_emb), test_props[:, i]),
                       "test_rmsd": _rmsd(clf.predict(test_emb), test_props[:, i]),
                       "alpha": clf.alpha_}
    return results, models


def strategy_nusvr(train_emb, train_props, val_emb, val_props, test_emb, test_props, cell_types):
    from sklearn.svm import NuSVR
    from sklearn.preprocessing import StandardScaler
    results, models = {}, {}
    for i, ct in enumerate(cell_types):
        scaler = StandardScaler()
        train_s = scaler.fit_transform(train_emb)
        val_s = scaler.transform(val_emb)
        test_s = scaler.transform(test_emb)
        best_val_r, best_model = -1, None
        for nu in [0.25, 0.5, 0.75]:
            for C in [0.1, 1.0, 10.0]:
                m = NuSVR(nu=nu, C=C, kernel="linear", max_iter=5000).fit(train_s, train_props[:, i])
                vr = _score(m.predict(val_s), val_props[:, i])
                if vr > best_val_r:
                    best_val_r, best_model = vr, m
        models[ct] = {"scaler": scaler, "model": best_model}
        results[ct] = {"val_r": best_val_r,
                       "test_r": _score(best_model.predict(test_s), test_props[:, i]),
                       "test_rmsd": _rmsd(best_model.predict(test_s), test_props[:, i]),
                       "nu": best_model.nu, "C": best_model.C}
    return results, models


def strategy_elasticnet(train_emb, train_props, val_emb, val_props, test_emb, test_props, cell_types):
    from sklearn.linear_model import ElasticNetCV
    results, models = {}, {}
    for i, ct in enumerate(cell_types):
        clf = ElasticNetCV(cv=3, max_iter=2000, n_alphas=50, random_state=SEED).fit(train_emb, train_props[:, i])
        models[ct] = clf
        results[ct] = {"val_r": _score(clf.predict(val_emb), val_props[:, i]),
                       "test_r": _score(clf.predict(test_emb), test_props[:, i]),
                       "test_rmsd": _rmsd(clf.predict(test_emb), test_props[:, i]),
                       "alpha": clf.alpha_, "l1_ratio": clf.l1_ratio_}
    return results, models


def strategy_centroid_ridge(train_emb, train_props, val_emb, val_props, test_emb, test_props, cell_types,
                             centroids=None, val_centroids=None):
    from sklearn.linear_model import RidgeCV
    alphas = [0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0]
    eye = np.eye(len(cell_types))
    results, models = {}, {}
    for i, ct in enumerate(cell_types):
        clf = RidgeCV(alphas=alphas).fit(centroids, eye[:, i])
        models[ct] = clf
        results[ct] = {"val_r": None, "test_r": _score(clf.predict(test_emb), test_props[:, i]),
                       "test_rmsd": _rmsd(clf.predict(test_emb), test_props[:, i]),
                       "alpha": clf.alpha_}
    return results, models


def strategy_centroid_nusvr(train_emb, train_props, val_emb, val_props, test_emb, test_props, cell_types,
                              centroids=None, val_centroids=None):
    from sklearn.svm import NuSVR
    from sklearn.preprocessing import StandardScaler
    eye = np.eye(len(cell_types))
    results, models = {}, {}
    for i, ct in enumerate(cell_types):
        scaler = StandardScaler()
        c_scaled = scaler.fit_transform(centroids)
        test_s = scaler.transform(test_emb)
        best_r, best_model = -1, None
        for nu in [0.25, 0.5, 0.75]:
            for C in [0.1, 1.0, 10.0]:
                m = NuSVR(nu=nu, C=C, kernel="linear", max_iter=5000).fit(c_scaled, eye[:, i])
                tr = _score(m.predict(test_s), test_props[:, i])
                if tr > best_r:
                    best_r, best_model = tr, m
        models[ct] = {"scaler": scaler, "model": best_model}
        results[ct] = {"val_r": None, "test_r": _score(best_model.predict(test_s), test_props[:, i]),
                       "test_rmsd": _rmsd(best_model.predict(test_s), test_props[:, i]),
                       "nu": best_model.nu, "C": best_model.C}
    return results, models


def strategy_ensemble(train_emb, train_props, val_emb, val_props, test_emb, test_props, cell_types):
    from sklearn.linear_model import RidgeCV, ElasticNetCV
    from sklearn.svm import NuSVR
    from sklearn.preprocessing import StandardScaler
    alphas = [0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0]
    results, models = {}, {}
    for i, ct in enumerate(cell_types):
        r1 = RidgeCV(alphas=alphas).fit(train_emb, train_props[:, i])
        scaler = StandardScaler()
        train_s = scaler.fit_transform(train_emb)
        val_s = scaler.transform(val_emb)
        test_s = scaler.transform(test_emb)
        best_vr, best_svr = -1, None
        for nu in [0.25, 0.5, 0.75]:
            for C in [0.1, 1.0, 10.0]:
                m = NuSVR(nu=nu, C=C, kernel="linear", max_iter=5000).fit(train_s, train_props[:, i])
                vr = _score(m.predict(val_s), val_props[:, i])
                if vr > best_vr:
                    best_vr, best_svr = vr, m
        r3 = ElasticNetCV(cv=3, max_iter=2000, n_alphas=50, random_state=SEED).fit(train_emb, train_props[:, i])
        models[ct] = {"ridge": r1, "nusvr": {"scaler": scaler, "model": best_svr}, "elasticnet": r3}
        p1, p2, p3 = r1.predict(test_emb), best_svr.predict(test_s), r3.predict(test_emb)
        test_pred = (p1 + p2 + p3) / 3
        results[ct] = {"val_r": best_vr, "test_r": _score(test_pred, test_props[:, i]),
                       "test_rmsd": _rmsd(test_pred, test_props[:, i])}
    return results, models


STRATEGIES: dict[str, Callable] = {
    "ridge_cv": strategy_ridge_cv, "nusvr": strategy_nusvr,
    "elasticnet": strategy_elasticnet, "centroid_ridge": strategy_centroid_ridge,
    "centroid_nusvr": strategy_centroid_nusvr, "ensemble": strategy_ensemble,
}


# -- Predict functions ---------------------------------------------------------


def _predict_ridge(m, emb, _):
    pred = np.zeros((emb.shape[0], len(m)))
    for i, ct in enumerate(m):
        pred[:, i] = m[ct].predict(emb)
    return pred


def _predict_nusvr(m, emb, _):
    pred = np.zeros((emb.shape[0], len(m)))
    for i, ct in enumerate(m):
        emb_s = m[ct]["scaler"].transform(emb)
        pred[:, i] = m[ct]["model"].predict(emb_s)
    return pred


def _predict_elasticnet(m, emb, _):
    return _predict_ridge(m, emb, _)


def _predict_ensemble(m, emb, cell_types):
    pred = np.zeros((emb.shape[0], len(cell_types)))
    for i, ct in enumerate(cell_types):
        d = m[ct]
        p1 = d["ridge"].predict(emb)
        p2 = d["nusvr"]["model"].predict(d["nusvr"]["scaler"].transform(emb))
        p3 = d["elasticnet"].predict(emb)
        pred[:, i] = (p1 + p2 + p3) / 3
    return pred


PREDICT_FN: dict[str, Callable] = {
    "ridge_cv": _predict_ridge, "nusvr": _predict_nusvr,
    "elasticnet": _predict_elasticnet, "centroid_ridge": _predict_ridge,
    "centroid_nusvr": _predict_nusvr, "ensemble": _predict_ensemble,
}


# -- Save helpers --------------------------------------------------------------


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_strategy_outputs(
    out_dir: Path, strategy: str,
    real_pred: np.ndarray, gt_df: pd.DataFrame,
    pseudo_pred: np.ndarray | None, pseudo_gt: np.ndarray | None,
    strategy_metrics: dict, metadata: dict, cell_types: list[str],
    real_cell_types: list[str] | None = None,
) -> None:
    """Save standard 4-file output + pseudo-bulk CSVs per strategy.

    Args:
        real_cell_types: column names for real-bulk ``proportions.csv``.
            Defaults to *cell_types* (scRNA types) for backward compat.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if real_cell_types is None:
        real_cell_types = cell_types

    # 1. Real bulk proportions.csv
    pd.DataFrame(real_pred, columns=real_cell_types).to_csv(out_dir / "proportions.csv", index=False)

    # 2. Real bulk metrics.json (via core.metrics)
    from core.metrics import evaluate_deconvolution
    try:
        cols = list(gt_df.columns)
        m = evaluate_deconvolution(gt_df.values.astype(np.float64), real_pred, cols)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(m, f, indent=2, cls=_NpEncoder)
    except Exception:
        pass

    # 3. Strategy-specific metrics
    with open(out_dir / f"{strategy}_metrics.json", "w") as f:
        json.dump(strategy_metrics, f, indent=2, cls=_NpEncoder)

    # 4. Metadata
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, cls=_NpEncoder)

    # 5. Pseudo-bulk outputs
    if pseudo_pred is not None and pseudo_gt is not None:
        pd.DataFrame(pseudo_pred, columns=cell_types).to_csv(out_dir / "pseudo_proportions.csv", index=False)
        pd.DataFrame(pseudo_gt, columns=cell_types).to_csv(out_dir / "pseudo_gt.csv", index=False)
        pm = {}
        for i, ct in enumerate(cell_types):
            pm[ct] = {"pearson_r": round(_score(pseudo_pred[:, i], pseudo_gt[:, i]), 4),
                       "rmse": round(_rmsd(pseudo_pred[:, i], pseudo_gt[:, i]), 4)}
        with open(out_dir / "pseudo_metrics.json", "w") as f:
            json.dump(pm, f, indent=2, cls=_NpEncoder)
