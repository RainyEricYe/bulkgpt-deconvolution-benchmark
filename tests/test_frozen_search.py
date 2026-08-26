"""Unit tests for ``core/deconv/frozen_search.py``.

Tests pseudo-bulk generation, centroids, all 6 strategies, predict functions,
GT alignment, and output saving.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from core.deconv.frozen_search import (
    STRATEGIES,
    PREDICT_FN,
    GT_LABEL_MAP,
    SEED,
    generate_pseudo_bulk,
    _compute_centroids,
    _score,
    _rmsd,
    save_strategy_outputs,
)


# -- Mock data helpers ----------------------------------------------------


def _mock_embeddings(n_cells=300, embed_dim=10, n_types=3, seed=42):
    """Create mock cell embeddings with clear type separation."""
    rng = np.random.default_rng(seed)
    embeddings = np.zeros((n_cells, embed_dim), dtype=np.float32)
    labels = np.zeros(n_cells, dtype=np.int32)
    for t in range(n_types):
        start = t * (n_cells // n_types)
        end = (t + 1) * (n_cells // n_types)
        embeddings[start:end] = rng.normal(loc=t * 3.0, scale=0.5,
                                            size=(end - start, embed_dim)).astype(np.float32)
        labels[start:end] = t
    return embeddings, labels


def _mock_gt_df(n_samples=50, n_types=3):
    """Create a mock GT DataFrame."""
    import pandas as pd
    rng = np.random.default_rng(42)
    props = rng.dirichlet(np.ones(n_types), size=n_samples)
    cell_types = [f"Type_{i}" for i in range(n_types)]
    return pd.DataFrame(props, columns=cell_types)


# -- generate_pseudo_bulk -------------------------------------------------


class TestGeneratePseudoBulk:
    def test_shapes(self):
        emb, labels = _mock_embeddings(300, 10, 3)
        train_emb, train_p, val_emb, val_p, test_emb, test_p = \
            generate_pseudo_bulk(emb, labels, 3, seed=42)
        assert train_emb.shape == (6000, 10)
        assert train_p.shape == (6000, 3)
        assert val_emb.shape == (2000, 10)
        assert val_p.shape == (2000, 3)
        assert test_emb.shape == (2000, 10)
        assert test_p.shape == (2000, 3)

    def test_reproducibility(self):
        emb, labels = _mock_embeddings(300, 10, 3)
        a = generate_pseudo_bulk(emb, labels, 3, seed=42)
        b = generate_pseudo_bulk(emb, labels, 3, seed=42)
        for arr_a, arr_b in zip(a, b):
            assert np.allclose(arr_a, arr_b)

    def test_different_seeds_differ(self):
        emb, labels = _mock_embeddings(300, 10, 3)
        a = generate_pseudo_bulk(emb, labels, 3, seed=42)
        b = generate_pseudo_bulk(emb, labels, 3, seed=99)
        any_diff = any(not np.allclose(arr_a, arr_b) for arr_a, arr_b in zip(a, b))
        assert any_diff

    def test_proportions_sum_to_one(self):
        emb, labels = _mock_embeddings(300, 10, 3)
        _, train_p, _, val_p, _, test_p = generate_pseudo_bulk(emb, labels, 3, seed=42)
        for props in [train_p, val_p, test_p]:
            assert np.allclose(props.sum(axis=1), 1.0)

    def test_custom_params(self):
        emb, labels = _mock_embeddings(500, 10, 4)
        train_emb, _, val_emb, _, test_emb, _ = \
            generate_pseudo_bulk(emb, labels, 4, n_pb_total=5000, n_holdout=1000, seed=1)
        assert train_emb.shape == (3000, 10)   # 4000*0.75
        assert val_emb.shape == (1000, 10)     # 4000*0.25
        assert test_emb.shape == (1000, 10)


# -- _compute_centroids ---------------------------------------------------


class TestComputeCentroids:
    def test_correctness(self):
        emb = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0],
                        [5.0, 6.0], [5.0, 6.0]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        centroids = _compute_centroids(emb, labels, 3)
        assert centroids.shape == (3, 2)
        assert np.allclose(centroids[0], [1.0, 2.0])
        assert np.allclose(centroids[1], [3.0, 4.0])
        assert np.allclose(centroids[2], [5.0, 6.0])


# -- _score / _rmsd -------------------------------------------------------


class TestScoreRmsd:
    def test_score_perfect(self):
        x = np.array([0.1, 0.3, 0.6])
        assert _score(x, x) == pytest.approx(1.0)

    def test_rmsd_perfect(self):
        x = np.array([0.1, 0.3, 0.6])
        assert _rmsd(x, x) == pytest.approx(0.0)

    def test_rmsd_known(self):
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 0.0])
        assert _rmsd(x, y) == pytest.approx(1.0)


# -- Strategy functions ---------------------------------------------------


def _strategy_data(n_samples=500, embed_dim=10, n_types=3):
    """Create fake train/val/test data for strategy testing.

    Uses Dirichlet for proportions (guaranteed valid) and random embeddings
    that are weakly correlated with the proportions.
    """
    rng = np.random.default_rng(42)
    train_emb = rng.normal(0, 1, (n_samples, embed_dim)).astype(np.float32)
    val_emb = rng.normal(0, 1, (100, embed_dim)).astype(np.float32)
    test_emb = rng.normal(0, 1, (100, embed_dim)).astype(np.float32)
    train_p = rng.dirichlet(np.ones(n_types), size=n_samples).astype(np.float32)
    val_p = rng.dirichlet(np.ones(n_types), size=100).astype(np.float32)
    test_p = rng.dirichlet(np.ones(n_types), size=100).astype(np.float32)
    cell_types = [f"ct_{i}" for i in range(n_types)]
    return train_emb, train_p, val_emb, val_p, test_emb, test_p, cell_types


class TestStrategies:
    def test_ridge_cv(self):
        args = _strategy_data()
        results, models = STRATEGIES["ridge_cv"](*args)
        assert len(results) == 3
        assert len(models) == 3
        for ct in args[-1]:
            assert "test_r" in results[ct]
            assert "alpha" in results[ct]

    def test_nusvr(self):
        args = _strategy_data()
        results, models = STRATEGIES["nusvr"](*args)
        assert len(results) == 3
        assert len(models) == 3
        for ct in args[-1]:
            assert "scaler" in models[ct]
            assert hasattr(models[ct]["model"], "predict")

    def test_elasticnet(self):
        args = _strategy_data()
        results, models = STRATEGIES["elasticnet"](*args)
        assert len(results) == 3
        for ct in args[-1]:
            assert "alpha" in results[ct]
            assert "l1_ratio" in results[ct]

    def test_centroid_ridge(self):
        args = _strategy_data()
        centroids = _compute_centroids(args[0][:50], np.array([0]*17+[1]*17+[2]*16), 3)
        results, models = STRATEGIES["centroid_ridge"](*args, centroids=centroids)
        assert len(results) == 3
        for ct in args[-1]:
            assert results[ct]["val_r"] is None

    def test_centroid_nusvr(self):
        args = _strategy_data()
        centroids = _compute_centroids(args[0][:50], np.array([0]*17+[1]*17+[2]*16), 3)
        results, models = STRATEGIES["centroid_nusvr"](*args, centroids=centroids)
        assert len(results) == 3
        for ct in args[-1]:
            assert results[ct]["val_r"] is None

    def test_ensemble(self):
        args = _strategy_data()
        results, models = STRATEGIES["ensemble"](*args)
        assert len(results) == 3
        for ct in args[-1]:
            assert "ridge" in models[ct]
            assert "nusvr" in models[ct]
            assert "elasticnet" in models[ct]

    def test_ensemble_averaging(self):
        args = _strategy_data()
        train_emb, train_p, val_emb, val_p, test_emb, test_p, cell_types = args
        _, models = STRATEGIES["ensemble"](*args)
        pred_ens = PREDICT_FN["ensemble"](models, test_emb, cell_types)
        pred_ridge = PREDICT_FN["ridge_cv"]({ct: m["ridge"] for ct, m in models.items()},
                                             test_emb, cell_types)
        pred_nusvr = PREDICT_FN["nusvr"]({ct: m["nusvr"] for ct, m in models.items()},
                                          test_emb, cell_types)
        pred_enet = PREDICT_FN["elasticnet"]({ct: m["elasticnet"] for ct, m in models.items()},
                                               test_emb, cell_types)
        expected = (pred_ridge + pred_nusvr + pred_enet) / 3
        assert np.allclose(pred_ens, expected)


# -- Predict functions ----------------------------------------------------


class TestPredictFunctions:
    def test_predict_shapes(self):
        args = _strategy_data(embed_dim=10, n_types=3)
        cell_types = args[-1]
        test_emb = np.random.default_rng(1).normal(0, 1, (50, 10)).astype(np.float32)

        for sname in ["ridge_cv", "nusvr", "elasticnet", "ensemble"]:
            _, models = STRATEGIES[sname](*args)
            pred = PREDICT_FN[sname](models, test_emb, cell_types)
            assert pred.shape == (50, 3), f"{sname} shape mismatch"


# -- align_predictions_to_gt (NEW -- RED phase) ---------------------------


class TestAlignPredictionsToGT:
    """Tests for align_predictions_to_gt -- will FAIL until function is added."""

    def test_exact_match(self):
        from core.deconv.frozen_search import align_predictions_to_gt
        pred = np.array([[0.1, 0.3, 0.6],
                         [0.2, 0.3, 0.5]], dtype=np.float64)
        cell_types = ["B_cells", "T_cells", "Monocytes"]
        gt_columns = ["B_cells", "T_cells", "Monocytes"]
        result = align_predictions_to_gt(pred, cell_types, gt_columns)
        assert result.shape == (2, 3)
        assert np.allclose(result, pred)

    def test_subset_gt(self):
        from core.deconv.frozen_search import align_predictions_to_gt
        pred = np.array([[0.1, 0.3, 0.6]], dtype=np.float64)
        cell_types = ["A", "B", "C"]
        gt_columns = ["A", "C"]
        result = align_predictions_to_gt(pred, cell_types, gt_columns)
        assert result.shape == (1, 2)
        assert result[0, 0] == pytest.approx(0.1)
        assert result[0, 1] == pytest.approx(0.6)

    def test_subtype_aggregation(self):
        from core.deconv.frozen_search import align_predictions_to_gt
        pred = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
        # Use cell type names that match GT_LABEL_MAP subtypes
        cell_types = ["B cells", "NK cells", "Monocytes", "mDC"]
        gt_columns = ["Lymphocytes", "Monocytes"]
        result = align_predictions_to_gt(pred, cell_types, gt_columns,
                                          dataset="altman_Arunachalam")
        assert result.shape == (1, 2)
        assert result[0, 0] == pytest.approx(0.3)  # Lymphocytes: B cells + NK cells
        assert result[0, 1] == pytest.approx(0.7)  # Monocytes: exact + mDC

    def test_empty_subtype_map(self):
        from core.deconv.frozen_search import align_predictions_to_gt
        pred = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
        cell_types = ["B cells", "NK cells", "Monocytes", "Neutrophils"]
        gt_columns = ["Basophils", "Lymphocytes", "Monocytes"]
        result = align_predictions_to_gt(pred, cell_types, gt_columns,
                                          dataset="altman_Arunachalam")
        assert result[0, 0] == pytest.approx(0.0)  # Basophils: empty -> 0

    def test_unknown_gt_column(self):
        from core.deconv.frozen_search import align_predictions_to_gt
        pred = np.array([[0.1, 0.9]], dtype=np.float64)
        cell_types = ["A", "B"]
        gt_columns = ["A", "Z"]
        result = align_predictions_to_gt(pred, cell_types, gt_columns)
        assert result[0, 0] == pytest.approx(0.1)
        assert result[0, 1] == pytest.approx(0.0)

    def test_no_dataset_map(self):
        from core.deconv.frozen_search import align_predictions_to_gt
        pred = np.array([[0.1, 0.9]], dtype=np.float64)
        cell_types = ["A", "B"]
        gt_columns = ["A", "B"]
        result = align_predictions_to_gt(pred, cell_types, gt_columns, dataset="unknown")
        assert np.allclose(result, pred)

    def test_map_subtypes_missing_fallback_to_exact(self):
        """subtypes in map don't match cell_types -> fallback to exact match."""
        from core.deconv.frozen_search import align_predictions_to_gt
        pred = np.array([[0.3, 0.7]], dtype=np.float64)
        cell_types = ["Lymphocytes", "Monocytes"]
        gt_columns = ["Lymphocytes", "Monocytes", "Basophils"]
        result = align_predictions_to_gt(pred, cell_types, gt_columns,
                                          dataset="altman_Arunachalam")
        assert result.shape == (1, 3)
        assert result[0, 0] == pytest.approx(0.3)
        assert result[0, 1] == pytest.approx(0.7)
        assert result[0, 2] == pytest.approx(0.0)


# -- save_strategy_outputs ------------------------------------------------


class TestSaveStrategyOutputs:
    def test_all_files_written(self, tmp_path):
        import pandas as pd
        n_samples, n_types = 20, 3
        cell_types = [f"ct_{i}" for i in range(n_types)]
        real_pred = np.random.default_rng(1).random((n_samples, n_types))
        real_pred = real_pred / real_pred.sum(axis=1, keepdims=True)
        gt_df = pd.DataFrame(
            np.random.default_rng(2).dirichlet(np.ones(n_types), size=n_samples),
            columns=cell_types,
        )
        pseudo_pred = np.random.default_rng(3).random((10, n_types))
        pseudo_pred = pseudo_pred / pseudo_pred.sum(axis=1, keepdims=True)
        pseudo_gt = np.random.default_rng(4).dirichlet(np.ones(n_types), size=10)
        strategy_metrics = {"ridge_cv": {"ct_0": {"test_r": 0.95, "alpha": 1.0}}}
        metadata = {"backbone": "test", "dataset": "test_ds", "seed": 42}
        out_dir = tmp_path / "output"

        save_strategy_outputs(out_dir, "ridge_cv", real_pred, gt_df,
                               pseudo_pred, pseudo_gt, strategy_metrics, metadata,
                               cell_types)

        assert (out_dir / "proportions.csv").exists()
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "ridge_cv_metrics.json").exists()
        assert (out_dir / "metadata.json").exists()
        assert (out_dir / "pseudo_proportions.csv").exists()
        assert (out_dir / "pseudo_gt.csv").exists()
        assert (out_dir / "pseudo_metrics.json").exists()

        props = pd.read_csv(out_dir / "proportions.csv")
        assert props.shape == (n_samples, n_types)
        assert list(props.columns) == cell_types


# -- GT_LABEL_MAP structure -----------------------------------------------


class TestGTLabelMap:
    def test_altman_structure(self):
        assert "altman_Arunachalam" in GT_LABEL_MAP
        m = GT_LABEL_MAP["altman_Arunachalam"]
        assert "Lymphocytes" in m
        assert "Monocytes" in m
        assert "Basophils" in m
        assert len(m["Lymphocytes"]) == 7  # B cells, ILC, NK cells, Plasma cells, T cells CD4 conv, T cells CD8, Tregs
        assert len(m["Basophils"]) == 0
