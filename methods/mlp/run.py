#!/usr/bin/env python3
"""MLP deconvolution baseline — scikit-learn MLPRegressor (multi-output).

StandardScaler is applied to features: sklearn MLPRegressor does not
standardise inputs internally, and log1p-CPM genes span very different
scales, which degrades convergence.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared.ml_baselines import run_baseline


def make_mlp(n_types, params):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    hidden = tuple(params.get("hidden_layer_sizes", (512, 256)))
    alpha = float(params.get("alpha", 1e-4))
    max_iter = int(params.get("max_iter", 500))
    early_stopping = bool(params.get("early_stopping", False))
    scaler = StandardScaler()
    mlp = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                       early_stopping=early_stopping, n_iter_no_change=10, random_state=42)

    class M:
        def fit(self, X, y):
            self.scaler = scaler.fit(X)
            Xs = np.nan_to_num(self.scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
            mlp.fit(Xs, y)
            return self

        def predict(self, X):
            Xs = np.nan_to_num(self.scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
            return mlp.predict(Xs)
    return M()


if __name__ == "__main__":
    run_baseline(make_mlp, "mlp")
