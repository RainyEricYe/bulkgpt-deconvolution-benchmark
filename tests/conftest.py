"""Shared fixtures for the deconvolution test suite."""

from pathlib import Path

import numpy as np
import pytest


# ── Path constants ─────────────────────────────────────────────────────

TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"

SDY67_H5 = str(DATA_DIR / "sdy67.h5")
GROUND_TRUTH_CSV = str(DATA_DIR / "ground_truth.csv")


# ── Prediction fixtures ──────────────────────────────────────────────


@pytest.fixture
def perfect_prediction() -> tuple[np.ndarray, np.ndarray]:
    """Return (true, pred) arrays where ``pred == true`` exactly."""
    rng = np.random.default_rng(42)
    n_samples, n_types = 10, 5
    true_props = rng.dirichlet(alpha=np.ones(n_types), size=n_samples)
    return true_props, true_props.copy()


@pytest.fixture
def noisy_prediction() -> tuple[np.ndarray, np.ndarray]:
    """Return (true, pred) arrays where pred has small Gaussian noise."""
    rng = np.random.default_rng(42)
    n_samples, n_types = 10, 5
    true_props = rng.dirichlet(alpha=np.ones(n_types), size=n_samples)
    noise = rng.normal(loc=0.0, scale=0.05, size=true_props.shape)
    pred_props = np.clip(true_props + noise, 0.0, 1.0)
    pred_props = pred_props / pred_props.sum(axis=1, keepdims=True)
    return true_props, pred_props


# ── Real data fixture ───────────────────────────────────────────────


@pytest.fixture(scope="session")
def sdy67_data() -> dict[str, str]:
    """Paths to real SDY67 benchmark data.

    ``h5`` is the canonical DeconBenchmark input; ``ground_truth`` is a
    convenience CSV kept for metric-evaluation tests that call
    ``pd.read_csv()`` directly.

    The H5 file (~90 MB) is not committed to git; run
    ``bash data/download_data.sh`` (or `tests/prepare_data.sh`) first.
    Tests that need it are skipped if it is missing (e.g. on CI without
    the data download step).
    """
    if not Path(SDY67_H5).exists():
        pytest.skip(
            "tests/data/sdy67.h5 not present; run bash data/download_data.sh first"
        )
    return {
        "h5": SDY67_H5,
        "sc_ref": SDY67_H5,
        "bulk": SDY67_H5,
        "ground_truth": GROUND_TRUTH_CSV,
    }
