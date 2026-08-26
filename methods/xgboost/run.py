#!/usr/bin/env python3
"""XGBoost deconvolution baseline — K separate XGBRegressor models (per cell type)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.ml_baselines import run_baseline


def make_xgb(n_types, params):
    from xgboost import XGBRegressor
    n_est = int(params.get("n_estimators", 200))
    depth = int(params.get("max_depth", 6))
    lr = float(params.get("learning_rate", 0.1))
    models = [
        XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                     objective="reg:squarederror", n_jobs=1, random_state=42,
                     verbosity=0)
        for _ in range(n_types)
    ]
    class M:
        def fit(self, X, y):
            for k, m in enumerate(models):
                m.fit(X, y[:, k])
            return self
        def predict(self, X):
            return np.column_stack([m.predict(X) for m in models])
    return M()


if __name__ == "__main__":
    run_baseline(make_xgb, "xgboost")
