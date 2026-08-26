"""Exhaustive unit tests for ``core/metrics.py``.

Tests every metric function across perfect, noisy, single-sample, constant,
and two-type edge cases.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from core.metrics import (
    compute_ccorr,
    compute_mae,
    compute_maecorr,
    compute_pearson,
    compute_rmse,
    compute_scorr,
    compute_wt,
    evaluate_deconvolution,
)

# ── Shared test helpers ──────────────────────────────────────────────


def _constant_3class() -> tuple[np.ndarray, np.ndarray]:
    """Return (true, pred) where every sample has uniform proportions."""
    return np.full((5, 3), 1.0 / 3.0), np.full((5, 3), 1.0 / 3.0)


def _two_type_perfect() -> tuple[np.ndarray, np.ndarray]:
    """Return (true, pred) with 2 cell types, perfect predictions."""
    true = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
    return true, true.copy()


# ── SCorr ────────────────────────────────────────────────────────────


class TestSCorr:
    def test_perfect(self, perfect_prediction):
        true, pred = perfect_prediction
        result = compute_scorr(true, pred)
        assert result["scorr_mean"] == pytest.approx(1.0, abs=1e-10)
        assert len(result["scorr_per_sample"]) == true.shape[0]

    def test_noisy(self, noisy_prediction):
        true, pred = noisy_prediction
        result = compute_scorr(true, pred)
        assert 0 < result["scorr_mean"] <= 1.0
        assert all(isinstance(v, float) for v in result["scorr_per_sample"])

    def test_single_sample(self):
        true = np.array([[0.5, 0.3, 0.2]])
        pred = true.copy()
        result = compute_scorr(true, pred)
        assert result["scorr_mean"] == pytest.approx(1.0, abs=1e-10)

    def test_constant(self):
        true, pred = _constant_3class()
        result = compute_scorr(true, pred)
        assert result["scorr_mean"] == pytest.approx(0.0, abs=1e-10)

    def test_two_types(self):
        true, pred = _two_type_perfect()
        result = compute_scorr(true, pred)
        assert result["scorr_mean"] == pytest.approx(1.0, abs=1e-10)


# ── CCorr ────────────────────────────────────────────────────────────


class TestCCorr:
    def test_perfect(self, perfect_prediction):
        true, pred = perfect_prediction
        result = compute_ccorr(true, pred)
        assert result["ccorr_mean"] == pytest.approx(1.0, abs=1e-10)
        assert len(result["ccorr_per_type"]) == true.shape[1]

    def test_noisy(self, noisy_prediction):
        true, pred = noisy_prediction
        result = compute_ccorr(true, pred)
        assert 0 < result["ccorr_mean"] <= 1.0

    def test_single_sample(self):
        """Single sample -> each column has 1 value => std=0 => returns 0.0."""
        true = np.array([[0.5, 0.3, 0.2]])
        pred = true.copy()
        result = compute_ccorr(true, pred)
        assert result["ccorr_mean"] == pytest.approx(0.0, abs=1e-10)

    def test_constant(self):
        true, pred = _constant_3class()
        result = compute_ccorr(true, pred)
        assert result["ccorr_mean"] == pytest.approx(0.0, abs=1e-10)

    def test_two_types(self):
        true, pred = _two_type_perfect()
        result = compute_ccorr(true, pred)
        assert result["ccorr_mean"] == pytest.approx(1.0, abs=1e-10)


# ── MAECorr ──────────────────────────────────────────────────────────


class TestMAECorr:
    def test_perfect(self, perfect_prediction):
        true, pred = perfect_prediction
        result = compute_maecorr(true, pred)
        assert result["maecorr"] == pytest.approx(0.0, abs=1e-10)

    def test_noisy(self, noisy_prediction):
        true, pred = noisy_prediction
        result = compute_maecorr(true, pred)
        assert result["maecorr"] > 0.0


class TestMAECorrEdge:
    """Edge cases for MAECorr that need separate handling."""

    def test_single_sample(self):
        """Single sample => maecorr returns 0.0."""
        true = np.array([[0.5, 0.3, 0.2]])
        pred = true.copy()
        r = compute_maecorr(true, pred)
        assert r["maecorr"] == pytest.approx(0.0)

    def test_constant(self):
        """Constant rows => corrcoef is NaN => maecorr is NaN."""
        true, pred = _constant_3class()
        result = compute_maecorr(true, pred)
        assert np.isnan(result["maecorr"])

    def test_two_types(self):
        true, pred = _two_type_perfect()
        result = compute_maecorr(true, pred)
        assert result["maecorr"] == pytest.approx(0.0, abs=1e-10)


# ── MAE ──────────────────────────────────────────────────────────────


class TestMAE:
    def test_perfect(self, perfect_prediction):
        true, pred = perfect_prediction
        result = compute_mae(true, pred)
        assert result["mae_overall"] == pytest.approx(0.0, abs=1e-10)
        assert all(v == pytest.approx(0.0, abs=1e-10) for v in result["mae_per_type"])
        assert len(result["mae_per_type"]) == true.shape[1]

    def test_noisy(self, noisy_prediction):
        true, pred = noisy_prediction
        result = compute_mae(true, pred)
        assert result["mae_overall"] > 0.0

    def test_single_sample(self):
        true = np.array([[0.5, 0.3, 0.2]])
        pred = true.copy()
        result = compute_mae(true, pred)
        assert result["mae_overall"] == pytest.approx(0.0, abs=1e-10)

    def test_constant(self):
        true, pred = _constant_3class()
        result = compute_mae(true, pred)
        assert result["mae_overall"] == pytest.approx(0.0, abs=1e-10)

    def test_two_types(self):
        true, pred = _two_type_perfect()
        result = compute_mae(true, pred)
        assert result["mae_overall"] == pytest.approx(0.0, abs=1e-10)


# ── RMSE ─────────────────────────────────────────────────────────────


class TestRMSE:
    def test_perfect(self, perfect_prediction):
        true, pred = perfect_prediction
        result = compute_rmse(true, pred)
        assert result["rmse_overall"] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse_mean_per_type"] == pytest.approx(0.0, abs=1e-10)
        assert all(v == pytest.approx(0.0, abs=1e-10) for v in result["rmse_per_type"])
        assert len(result["rmse_per_type"]) == true.shape[1]

    def test_noisy(self, noisy_prediction):
        true, pred = noisy_prediction
        result = compute_rmse(true, pred)
        assert result["rmse_overall"] > 0.0

    def test_single_sample(self):
        true = np.array([[0.5, 0.3, 0.2]])
        pred = true.copy()
        result = compute_rmse(true, pred)
        assert result["rmse_overall"] == pytest.approx(0.0, abs=1e-10)

    def test_constant(self):
        true, pred = _constant_3class()
        result = compute_rmse(true, pred)
        assert result["rmse_overall"] == pytest.approx(0.0, abs=1e-10)

    def test_two_types(self):
        true, pred = _two_type_perfect()
        result = compute_rmse(true, pred)
        assert result["rmse_overall"] == pytest.approx(0.0, abs=1e-10)


# ── Pearson ──────────────────────────────────────────────────────────


class TestPearson:
    def test_perfect(self, perfect_prediction):
        true, pred = perfect_prediction
        result = compute_pearson(true, pred)
        assert result["pearson_mean"] == pytest.approx(1.0, abs=1e-10)
        assert len(result["pearson_per_type"]) == true.shape[1]

    def test_noisy(self, noisy_prediction):
        true, pred = noisy_prediction
        result = compute_pearson(true, pred)
        assert 0 < result["pearson_mean"] <= 1.0

    def test_single_sample(self):
        """Single sample => each column has 1 value => std=0 => returns 0.0."""
        true = np.array([[0.5, 0.3, 0.2]])
        pred = true.copy()
        result = compute_pearson(true, pred)
        assert result["pearson_mean"] == pytest.approx(0.0, abs=1e-10)

    def test_constant(self):
        true, pred = _constant_3class()
        result = compute_pearson(true, pred)
        assert result["pearson_mean"] == pytest.approx(0.0, abs=1e-10)

    def test_two_types(self):
        true, pred = _two_type_perfect()
        result = compute_pearson(true, pred)
        assert result["pearson_mean"] == pytest.approx(1.0, abs=1e-10)


# ── Wilcoxon Rank-Sum (WT) ───────────────────────────────────────────


class TestWT:
    def test_perfect(self, perfect_prediction):
        true, pred = perfect_prediction
        result = compute_wt(true, pred)
        # Identical arrays => MWU p-value = 1.0
        assert result["wt_mean"] == pytest.approx(1.0, abs=1e-10)
        assert len(result["wt_per_type"]) == true.shape[1]

    def test_noisy(self, noisy_prediction):
        true, pred = noisy_prediction
        result = compute_wt(true, pred)
        # Similar distributions => p-values should be well > 0
        assert result["wt_mean"] > 0.01

    def test_single_sample(self):
        """Single sample => degenerate columns => returns 1.0."""
        true = np.array([[0.5, 0.3, 0.2]])
        pred = true.copy()
        result = compute_wt(true, pred)
        assert result["wt_mean"] == pytest.approx(1.0, abs=1e-10)

    def test_constant(self):
        true, pred = _constant_3class()
        result = compute_wt(true, pred)
        assert result["wt_mean"] == pytest.approx(1.0, abs=1e-10)

    def test_two_types(self):
        true, pred = _two_type_perfect()
        result = compute_wt(true, pred)
        assert result["wt_mean"] == pytest.approx(1.0, abs=1e-10)


# ── evaluate_deconvolution (integrated) ──────────────────────────────


class TestEvaluateDeconvolution:
    EXPECTED_KEYS = {
        "mae_overall",
        "mae_per_type",
        "scorr_mean",
        "scorr_per_sample",
        "ccorr_mean",
        "ccorr_per_type",
        "maecorr",
        "pearson_mean",
        "pearson_per_type",
        "rmse_overall",
        "rmse_per_type",
        "rmse_mean_per_type",
        "wt_mean",
        "wt_per_type",
    }

    def test_perfect(self, perfect_prediction):
        true, pred = perfect_prediction
        result = evaluate_deconvolution(true, pred)
        assert set(result.keys()) == self.EXPECTED_KEYS
        assert result["pearson_mean"] == pytest.approx(1.0, abs=1e-10)
        assert result["scorr_mean"] == pytest.approx(1.0, abs=1e-10)
        assert result["ccorr_mean"] == pytest.approx(1.0, abs=1e-10)
        assert result["mae_overall"] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse_overall"] == pytest.approx(0.0, abs=1e-10)
        assert result["maecorr"] == pytest.approx(0.0, abs=1e-10)
        assert result["wt_mean"] == pytest.approx(1.0, abs=1e-10)

    def test_noisy(self, noisy_prediction):
        true, pred = noisy_prediction
        result = evaluate_deconvolution(true, pred)
        assert set(result.keys()) == self.EXPECTED_KEYS
        assert result["mae_overall"] > 0.0

    def test_output_keys(self):
        """Verify the exact set of output keys."""
        rng = np.random.default_rng(42)
        true = rng.dirichlet(np.ones(3), size=10)
        pred = true.copy()
        result = evaluate_deconvolution(true, pred)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_matches_baseline(self):
        """Compare perfect-prediction results against baseline.json."""
        baseline_path = (
            Path(__file__).resolve().parent / "expected" / "baseline.json"
        )
        with open(baseline_path) as f:
            baseline = json.load(f)

        rng = np.random.default_rng(42)
        true = rng.dirichlet(np.ones(3), size=10)
        result = evaluate_deconvolution(true, true.copy())

        for key, expected in baseline.items():
            assert result[key] == pytest.approx(expected, abs=1e-10), (
                f"Mismatch for {key}: expected {expected}, got {result[key]}"
            )

    def test_rejects_nan_inputs(self):
        """evaluate_deconvolution raises ValueError on NaN inputs."""
        rng = np.random.default_rng(42)
        true = rng.dirichlet(np.ones(3), size=10)
        nan_pred = true.copy()
        nan_pred[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            evaluate_deconvolution(true, nan_pred)

    def test_rejects_inf_inputs(self):
        """evaluate_deconvolution raises ValueError on Inf inputs."""
        rng = np.random.default_rng(42)
        true = rng.dirichlet(np.ones(3), size=10)
        inf_pred = true.copy()
        inf_pred[0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            evaluate_deconvolution(true, inf_pred)
