"""Integration tests for the deconvolution evaluation pipeline.

Exercises the ``data_loader -> evaluate_deconvolution`` data flow with
synthetic test data, verifying end-to-end output format, value sanity,
and custom ground-truth support.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_loader import load_data
from core.metrics import evaluate_deconvolution


class TestPipeline:
    """End-to-end pipeline tests."""

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

    def test_end_to_end_perfect(self, sdy67_data):
        """Load ground truth, evaluate with perfect predictions."""
        gt = pd.read_csv(sdy67_data["ground_truth"], index_col=0)
        assert gt.shape == (250, 5)

        true_props = gt.values.astype(np.float64)
        pred_props = true_props.copy()

        # Run full evaluation
        result = evaluate_deconvolution(
            true_props, pred_props, cell_types=list(gt.columns)
        )

        # Verify output contains all expected keys
        assert set(result.keys()) == self.EXPECTED_KEYS, (
            f"Missing keys: {self.EXPECTED_KEYS - set(result.keys())}"
        )

        # Perfect predictions => ideal metric values
        assert result["scorr_mean"] == pytest.approx(1.0, abs=1e-10)
        assert result["ccorr_mean"] == pytest.approx(1.0, abs=1e-10)
        assert result["pearson_mean"] == pytest.approx(1.0, abs=1e-10)
        assert result["mae_overall"] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse_overall"] == pytest.approx(0.0, abs=1e-10)
        assert result["maecorr"] == pytest.approx(0.0, abs=1e-10)
        assert result["wt_mean"] == pytest.approx(1.0, abs=1e-10)

        # Per-type lists match the number of cell types
        assert len(result["mae_per_type"]) == 5
        assert len(result["rmse_per_type"]) == 5
        assert len(result["pearson_per_type"]) == 5
        assert len(result["ccorr_per_type"]) == 5
        assert len(result["scorr_per_sample"]) == 250
        assert len(result["wt_per_type"]) == 5

    def test_with_custom_ground_truth(self, tmp_path, sdy67_data):
        """Load bulk data with a custom ground-truth CSV and evaluate."""
        # Read sample names from the H5 bulk data
        bundle0 = load_data(sdy67_data["bulk"])
        sample_names = list(bundle0.bulk.index)

        # Build a custom ground-truth CSV (250 samples)
        rng = np.random.default_rng(42)
        custom_gt = pd.DataFrame(
            {
                "T_cells": rng.uniform(0.1, 0.7, 250),
                "B_cells": rng.uniform(0.1, 0.5, 250),
                "NK_cells": rng.uniform(0.05, 0.3, 250),
            },
            index=sample_names,
        )
        # Normalize to sum to 1
        custom_gt = custom_gt.div(custom_gt.sum(axis=1), axis=0)
        gt_path = tmp_path / "custom_gt.csv"
        custom_gt.to_csv(gt_path)

        # Load bulk with explicit ground truth
        bundle = load_data(
            sdy67_data["bulk"], ground_truth=str(gt_path)
        )
        assert bundle.gt is not None
        assert bundle.gt.shape[0] == 250
        assert list(bundle.gt.columns) == ["T_cells", "B_cells", "NK_cells"]

        # Evaluate with custom GT as perfect predictions
        true_props = bundle.gt.values.astype(np.float64)
        pred_props = true_props.copy()
        result = evaluate_deconvolution(true_props, pred_props)

        assert result["pearson_mean"] == pytest.approx(1.0, abs=1e-10)
        assert result["mae_overall"] == pytest.approx(0.0, abs=1e-10)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_noisy_predictions_produce_reasonable_values(
        self, sdy67_data
    ):
        """Non-perfect predictions yield degraded (but valid) metrics."""
        gt = pd.read_csv(sdy67_data["ground_truth"], index_col=0)
        true_props = gt.values.astype(np.float64)

        # Add noise to create imperfect predictions
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.1, size=true_props.shape)
        pred_props = np.clip(true_props + noise, 0.0, 1.0)
        pred_props /= pred_props.sum(axis=1, keepdims=True)

        result = evaluate_deconvolution(true_props, pred_props)

        # Noise degrades all correlation-based metrics
        assert result["scorr_mean"] < 1.0
        assert result["ccorr_mean"] < 1.0
        assert result["pearson_mean"] < 1.0
        assert result["mae_overall"] > 0.0
        assert result["rmse_overall"] > 0.0
        assert result["maecorr"] > 0.0

        # All values are within valid ranges
        assert 0 <= result["scorr_mean"] <= 1.0
        assert result["mae_overall"] >= 0.0
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_matches_baseline(self):
        """Perfect-prediction metrics match the baseline.json reference."""
        baseline_path = (
            Path(__file__).resolve().parent / "expected" / "baseline.json"
        )
        with open(baseline_path) as f:
            baseline = json.load(f)

        rng = np.random.default_rng(42)
        true = rng.dirichlet(np.ones(3), size=10)
        result = evaluate_deconvolution(true, true.copy())

        for key, expected in baseline.items():
            assert result[key] == pytest.approx(expected, abs=1e-10)
