"""Tests for ``core/data_loader.py``.

Covers format detection (h5ad, H5, TSV, CSV), cell-type column discovery,
DataBundle invariants, and error handling.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_loader import DataBundle, _detect_celltype_col, _get_ext, load_data


# ── Format detection ─────────────────────────────────────────────────


class TestFormatDetection:
    def test_h5ad_sc(self, sdy67_data):
        """h5ad with a cell_type column is detected as single-cell."""
        bundle = load_data(sdy67_data["sc_ref"])
        assert bundle.sc_ref is not None
        assert bundle.sc_ref.n_obs == 4900

    def test_h5ad_bulk(self, sdy67_data):
        """H5 file provides both sc_ref and bulk simultaneously."""
        bundle = load_data(sdy67_data["bulk"])
        # DeconBenchmark H5 has both singleCellExpr and bulk sections
        assert bundle.sc_ref is not None
        assert bundle.bulk is not None
        assert bundle.bulk.shape[0] == 250  # 250 bulk samples

    def test_tsv(self, tmp_path):
        """A .tsv file is loaded as bulk."""
        tsv_path = tmp_path / "test.tsv"
        df = pd.DataFrame(
            {"G1": [1.0, 2.0], "G2": [3.0, 4.0]}, index=["S1", "S2"]
        )
        df.to_csv(tsv_path, sep="\t")
        bundle = load_data(str(tsv_path))
        assert bundle.bulk is not None
        assert bundle.bulk.shape == (2, 2)

    def test_csv(self, tmp_path):
        """A .csv file is loaded as bulk."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {"G1": [1.0, 2.0], "G2": [3.0, 4.0]}, index=["S1", "S2"]
        )
        df.to_csv(csv_path)
        bundle = load_data(str(csv_path))
        assert bundle.bulk is not None
        assert bundle.bulk.shape == (2, 2)

    def test_tsv_gz(self, tmp_path):
        """A .tsv.gz file is loaded as bulk."""
        tsv_gz_path = tmp_path / "test.tsv.gz"
        df = pd.DataFrame(
            {"G1": [1.0, 2.0], "G2": [3.0, 4.0]}, index=["S1", "S2"]
        )
        df.to_csv(tsv_gz_path, sep="\t", compression="gzip")
        bundle = load_data(str(tsv_gz_path))
        assert bundle.bulk is not None
        assert bundle.bulk.shape == (2, 2)

    def test_h5_format(self, tmp_path):
        """A DeconBenchmark-format .h5 file is parsed correctly.
        Canonical format: bulk/values (n_samples, n_genes),
        rownames=genes, colnames=samples (see CLAUDE.md / H5 Canonical Format).
        """
        h5py = pytest.importorskip("h5py")
        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "bulk/values", data=np.array([[1.0, 2.0], [3.0, 4.0]])
            )
            f.create_dataset(
                "bulk/rownames", data=np.array([b"G1", b"G2"])  # gene names (n_genes)
            )
            f.create_dataset(
                "bulk/colnames", data=np.array([b"S1", b"S2"])  # sample names (n_samples)
            )

        bundle = load_data(str(h5_path))
        assert bundle.is_valid()
        assert bundle.bulk is not None
        assert bundle.bulk.shape == (2, 2)
        assert list(bundle.bulk.index) == ["S1", "S2"]
        assert list(bundle.bulk.columns) == ["G1", "G2"]

    def test_h5_with_sc(self, tmp_path):
        """H5 file with singleCellExpr populates sc_ref.
        Canonical format: scExpr (n_cells, n_genes), rownames=genes, colnames=barcodes.
        """
        h5py = pytest.importorskip("h5py")
        h5_path = tmp_path / "test_sc.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "bulk/values", data=np.array([[1.0, 2.0], [3.0, 4.0]])
            )
            f.create_dataset("bulk/rownames", data=np.array([b"G1", b"G2"]))  # genes (n_genes)
            f.create_dataset("bulk/colnames", data=np.array([b"S1", b"S2"]))  # samples (n_samples)
            # Canonical: (n_cells=3, n_genes=2), rownames=genes, colnames=barcodes
            f.create_dataset(
                "singleCellExpr/values",
                data=np.array([[5.0, 7.0], [6.0, 8.0], [9.0, 10.0]]),  # (3, 2)
            )
            f.create_dataset(
                "singleCellExpr/rownames",
                data=np.array([b"G1", b"G2"]),  # gene names
            )
            f.create_dataset(
                "singleCellExpr/colnames",
                data=np.array([b"cell_0", b"cell_1", b"cell_2"]),  # barcodes
            )
            f.create_dataset(
                "singleCellLabels/values",
                data=np.array([b"TypeA", b"TypeB", b"TypeA"]),
            )

        bundle = load_data(str(h5_path))
        assert bundle.sc_ref is not None
        assert bundle.sc_ref.n_obs == 3
        assert bundle.cell_types == ["TypeA", "TypeB"]

    def test_invalid_format(self, tmp_path):
        """Unknown extensions raise ValueError."""
        txt_path = tmp_path / "data.txt"
        txt_path.write_text("invalid")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_data(str(txt_path))


# ── _get_ext ─────────────────────────────────────────────────────────


class TestGetExt:
    def test_h5ad(self):
        assert _get_ext(Path("data.h5ad")) == ".h5ad"

    def test_h5(self):
        assert _get_ext(Path("data.h5")) == ".h5"

    def test_h5seurat(self):
        assert _get_ext(Path("data.h5seurat")) == ".h5seurat"

    def test_tsv(self):
        assert _get_ext(Path("data.tsv")) == ".tsv"

    def test_csv(self):
        assert _get_ext(Path("data.csv")) == ".csv"

    def test_tsv_gz(self):
        assert _get_ext(Path("data.tsv.gz")) == ".tsv.gz"

    def test_csv_gz(self):
        assert _get_ext(Path("data.csv.gz")) == ".csv.gz"

    def test_unknown(self):
        assert _get_ext(Path("data.txt")) == ".txt"
        assert _get_ext(Path("data")) == ""


# ── DataBundle ───────────────────────────────────────────────────────


class TestDataBundle:
    def test_default_fields_are_none(self):
        bundle = DataBundle()
        assert bundle.sc_ref is None
        assert bundle.bulk is None
        assert bundle.gt is None
        assert bundle.cell_types is None

    def test_is_valid_false_when_bulk_none(self):
        assert not DataBundle().is_valid()

    def test_is_valid_true_when_bulk_set(self):
        bundle = DataBundle(bulk=pd.DataFrame({"a": [1.0]}))
        assert bundle.is_valid()

    def test_with_all_fields(self):
        bundle = DataBundle(
            bulk=pd.DataFrame({"g1": [1.0]}),
            gt=pd.DataFrame({"T": [0.5]}),
            cell_types=["T"],
        )
        assert bundle.is_valid()
        assert bundle.cell_types == ["T"]


# ── Cell-type column detection ───────────────────────────────────────


class TestCellTypeDetection:
    def test_detects_cell_type(self, sdy67_data):
        bundle = load_data(sdy67_data["sc_ref"])
        assert bundle.cell_types is not None
        assert "T_cells" in bundle.cell_types
        assert "B_cells" in bundle.cell_types
        assert "NK_cells" in bundle.cell_types

    def test_prefers_priority_order(self, tmp_path):
        """Priority order: cell_type > CellType > celltype > ..."""
        import anndata

        obs = pd.DataFrame(
            {
                "cell_type": ["A", "B"],
                "CellType": ["X", "Y"],
                "label": ["M", "N"],
            },
            index=["c1", "c2"],
        )
        adata_path = tmp_path / "priorities.h5ad"
        adata = anndata.AnnData(
            X=np.random.rand(2, 3).astype(np.float32),
            obs=obs,
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        adata.write(adata_path)

        bundle = load_data(str(adata_path))
        assert bundle.cell_types == ["A", "B"]

    def test_finds_CellType(self, tmp_path):
        """PascalCase CellType is also detected."""
        import anndata

        obs = pd.DataFrame(
            {"CellType": ["Alpha", "Beta"]}, index=["c1", "c2"]
        )
        adata_path = tmp_path / "celltype_col.h5ad"
        adata = anndata.AnnData(
            X=np.random.rand(2, 3).astype(np.float32),
            obs=obs,
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        adata.write(adata_path)

        bundle = load_data(str(adata_path))
        assert bundle.cell_types == ["Alpha", "Beta"]

    def test_no_celltype_col_returns_none(self, tmp_path):
        """h5ad without any known cell-type column returns None."""
        import anndata

        obs = pd.DataFrame(
            {"tissue": ["brain", "liver"]}, index=["c1", "c2"]
        )
        adata_path = tmp_path / "no_celltype.h5ad"
        adata = anndata.AnnData(
            X=np.random.rand(2, 3).astype(np.float32),
            obs=obs,
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        adata.write(adata_path)

        bundle = load_data(str(adata_path))
        assert bundle.cell_types is None

    def test_detect_celltype_col_function(self, tmp_path):
        """Directly test _detect_celltype_col."""
        import anndata

        obs = pd.DataFrame(
            {"cell_type": ["a", "b", "c"]}, index=["c1", "c2", "c3"]
        )
        adata = anndata.AnnData(
            X=np.random.rand(3, 2).astype(np.float32),
            obs=obs,
            var=pd.DataFrame(index=["g1", "g2"]),
        )
        assert _detect_celltype_col(adata) == ["a", "b", "c"]


# ── Error handling ───────────────────────────────────────────────────


class TestErrors:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Data path not found"):
            load_data("/nonexistent/path.h5ad")

    def test_ground_truth_missing_no_error(self, sdy67_data):
        """Missing external ground_truth silently falls back to H5 internal."""
        bundle = load_data(
            sdy67_data["bulk"],
            ground_truth="/nonexistent/gt.csv",
        )
        # H5 has internal ground_truth section, so not None
        assert bundle.gt is not None
        assert bundle.gt.shape[1] == 5  # 5 cell types

    def test_ground_truth_loaded(self, sdy67_data):
        bundle = load_data(
            sdy67_data["bulk"],
            ground_truth=sdy67_data["ground_truth"],
        )
        assert bundle.gt is not None
        assert bundle.gt.shape == (250, 5)
