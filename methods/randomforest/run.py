#!/usr/bin/env python3
"""RandomForest deconvolution baseline — K separate RandomForestRegressor models."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.ml_baselines import run_baseline


def make_rf(n_types, params):
    from sklearn.ensemble import RandomForestRegressor
    n_est = int(params.get("n_estimators", 200))
    depth = params.get("max_depth")
    n_jobs = int(params.get("n_jobs", 4))
    models = [
        RandomForestRegressor(n_estimators=n_est, max_depth=depth, n_jobs=n_jobs,
                              random_state=42)
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
    run_baseline(make_rf, "randomforest")
