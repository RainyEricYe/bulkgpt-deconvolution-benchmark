#!/usr/bin/env python3
"""Mode B (LOO / train-test split) evaluation for ML regressor baselines (P2#13).

Follows the framework's Mode B evaluation system (mirrors
scripts/eval_loo_ridge.py): predictions are renormalised with
core.deconv.utils.renormalize_props and scored with
core.metrics.evaluate_deconvolution (the DeconBenchmark suite), with per-type
pearson/rmse written to ridge_metrics.json. Writes
results/2_realbulk/{dataset}/{method}_modeb/{proportions.csv, metrics.json,
ridge_metrics.json, metadata.json}.

Protocol (documented in metadata.json):
  - n_samples <  60 : leave-one-out (every sample predicted by a model trained
                      on the other n-1) — all samples written, scored over all n
                      (same as the framework's *_loo entries).
  - n_samples >= 60 : deterministic 70/30 train/test split — only the held-out
                      test samples are written and scored (same as the
                      framework's standard-split Mode B entries).
ridge_metrics.json makes scripts/evaluate.py post-hoc skip the directory.

Usage:
  python scripts/run_ml_baselines_modeb.py [--methods mlp xgboost ...] \
      [--datasets sdy67 ...] [--parallel 8]
"""
import argparse
import importlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from core.deconv.utils import renormalize_props
from core.metrics import evaluate_deconvolution

DATA_DIR = PROJECT_ROOT / "data" / "2_real_bulk"
RESULTS_DIR = PROJECT_ROOT / "results" / "2_realbulk"

FACTORIES = {
    "mlp": ("make_mlp", {"hidden_layer_sizes": [128, 64], "alpha": 1e-3, "max_iter": 600, "early_stopping": False}),
    "xgboost": ("make_xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}),
    "randomforest": ("make_rf", {"n_estimators": 200}),
}


def get_factory(method):
    fn_name, overrides = FACTORIES[method]
    mod = importlib.import_module(f"methods.{method}.run")
    return getattr(mod, fn_name), overrides


def load_bulk_gt(h5_path, gt_csv):
    import h5py
    with h5py.File(str(h5_path), "r") as h:
        b = h["bulk/values"][:]
        brow = [x.decode() for x in h["bulk/rownames"][:]] if "rownames" in h["bulk"] else []
        bcol = [x.decode() for x in h["bulk/colnames"][:]] if "colnames" in h["bulk"] else []
        if len(brow) == b.shape[1]:
            bulk, samples = b, bcol
        elif len(brow) == b.shape[0]:
            bulk, samples = b.T, brow
        else:
            raise ValueError(f"bulk orientation unknown {b.shape}")
    gt = pd.read_csv(gt_csv, index_col=0)
    common_samp = [s for s in gt.index if s in samples]
    if common_samp:
        bulk = bulk[[samples.index(s) for s in common_samp]]
        gt = gt.loc[common_samp]
    elif len(gt) == len(samples):
        # Positional fallback: some datasets (huuki_myers, sweetwater) name bulk
        # columns sample_0..N while GT rows carry meaningful names in the same
        # order (mirrors scripts/evaluate.py). Keep both positionally aligned.
        gt = gt.reset_index(drop=True)
    else:
        raise ValueError(f"no common samples and length mismatch GT={len(gt)} vs bulk={len(samples)}")
    return bulk, gt


def cpm_log1p(bulk):
    row = bulk.sum(axis=1, keepdims=True)
    X = np.log1p(bulk / np.maximum(row, 1e-10) * 1e6)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def fit_features(bulk_train, n_comp):
    """Fit log1p-CPM -> StandardScaler -> PCA on the training fold; return a
    transform reusable on held-out folds (same coordinate system)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xt = cpm_log1p(bulk_train)
    sc = StandardScaler().fit(Xt)
    Xs = np.nan_to_num(sc.transform(Xt), nan=0.0, posinf=0.0, neginf=0.0)
    n_comp = max(1, min(n_comp, Xs.shape[0] - 1, Xs.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42).fit(Xs)
    return (sc, pca)


def apply_features(bulk, tfm):
    sc, pca = tfm
    X = cpm_log1p(bulk)
    Xs = np.nan_to_num(sc.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    return pca.transform(Xs)


def _per_type_metrics(gt_values, pred, gt_columns):
    pearson, rmse = {}, {}
    for j, ct in enumerate(gt_columns):
        mask = ~np.isnan(gt_values[:, j])
        if mask.sum() >= 2 and np.std(gt_values[mask, j]) > 1e-10:
            r = float(np.corrcoef(pred[mask, j], gt_values[mask, j])[0, 1])
            pearson[ct] = round(r, 4) if not np.isnan(r) else None
        else:
            pearson[ct] = None
        rmse[ct] = round(float(np.sqrt(np.mean((pred[mask, j] - gt_values[mask, j]) ** 2))), 4) if mask.sum() >= 2 else None
    return pearson, rmse


def run_dataset(method, ds_name):
    h5 = DATA_DIR / f"{ds_name}.h5"
    gt_csv = DATA_DIR / f"{ds_name}_gt.csv"
    out_dir = RESULTS_DIR / ds_name / f"{method}_modeb"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (h5.exists() and gt_csv.exists()):
        return f"{method}/{ds_name}: missing data"

    factory, overrides = get_factory(method)
    bulk, gt = load_bulk_gt(h5, gt_csv)
    n = bulk.shape[0]
    y = gt.values.astype(np.float64)
    types = list(gt.columns)
    params = dict(overrides)

    if n < 60:
        folds = [(np.array([j for j in range(n) if j != i]), np.array([i])) for i in range(n)]
        protocol = f"LOO ({n} folds)"
    else:
        rng = np.random.RandomState(42)
        test_idx = np.sort(rng.choice(n, int(n * 0.3), replace=False))
        train_idx = np.array([i for i in range(n) if i not in test_idx])
        folds = [(train_idx, test_idx)]
        protocol = f"split70_30 test={len(test_idx)}"

    pred = np.zeros_like(y)
    evaluated = []
    for tr, te in folds:
        tfm = fit_features(bulk[tr], 50)
        Xtr = apply_features(bulk[tr], tfm)
        Xte = apply_features(bulk[te], tfm)
        model = factory(n_types=len(types), params=params)
        model.fit(Xtr, y[tr])
        pred[te] = model.predict(Xte)
        evaluated.append(te)

    evaluated = np.unique(np.concatenate(evaluated))
    pred_eval = renormalize_props(pred[evaluated])
    gt_eval = y[evaluated]
    metrics = evaluate_deconvolution(gt_eval, pred_eval, types)
    pearson, rmse = _per_type_metrics(gt_eval, pred_eval, types)

    props = pd.DataFrame(pred_eval, index=gt.index[evaluated], columns=types)
    props.index.name = "sample"
    props.to_csv(out_dir / "proportions.csv")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(out_dir / "ridge_metrics.json", "w") as f:
        json.dump({"pearson_per_type": pearson, "rmse_per_type": rmse,
                   "n_loo": n, "method": protocol}, f, indent=2)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"method": method, "dataset": ds_name, "protocol": protocol,
                   "n_samples": n, "n_evaluated": len(evaluated),
                   "features": "PCA50(log1p-CPM)", "params": params}, f, indent=2)
    r = metrics.get("pearson_mean")
    print(f"  [{method}] {ds_name}: {protocol} -> pearson {r:.4f}")
    return f"{method}/{ds_name}: {r:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(FACTORIES))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    datasets = args.datasets or sorted(f.stem[:-3] for f in DATA_DIR.glob("*_gt.csv"))
    tasks = [(m, d) for m in args.methods for d in datasets]
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_dataset, m, d): (m, d) for m, d in tasks}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                m, d = futs[fut]
                results.append(f"{m}/{d}: ERROR {e}")
    print("\n".join(results))


if __name__ == "__main__":
    main()
