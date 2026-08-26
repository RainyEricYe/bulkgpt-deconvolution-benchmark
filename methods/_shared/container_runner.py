#!/usr/bin/env python3
"""Shared utilities for container-based deconvolution methods.

Provides common functions reused by all container method runners:
- DeconBenchmark H5 I/O
- Apptainer SIF execution
- Pseudo-bulk generation
- Metric computation
- Standard CLI argument parsing

Each per-method run.py imports from here and calls ``main(METHOD_NAME)``.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")

SEED = 42


def get_project_root():
    """Return PROJECT_ROOT from env (preferred) or climb to to_publish/.

    This file is at methods/_shared/container_runner.py -> parent.parent.parent -> to_publish/
    """
    env_val = os.environ.get("PROJECT_ROOT")
    if env_val:
        return Path(env_val).resolve()
    return Path(__file__).resolve().parent.parent.parent


# Ensure the project root is on sys.path for resource module imports
_project_root = get_project_root()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def resolve(p):
    p = Path(str(p))
    return p if p.is_absolute() else get_project_root() / p


def make_parser(description=None):
    p = argparse.ArgumentParser(description=description or "Container deconvolution")
    p.add_argument("--config", required=True)
    p.add_argument("--mode", default="predict", choices=["predict"])
    p.add_argument("--data", default=None)
    p.add_argument("--h5", default=None)
    p.add_argument("--ground-truth", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sif", default=None)
    p.add_argument("--gpu", action="store_true", default=None)
    return p


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_pseudo_bulk(adata, n_samples=2000, n_cells_per_sample=80, celltype_col="cell_type"):
    from sklearn.model_selection import train_test_split
    from scipy.sparse import issparse

    rng = np.random.RandomState(SEED)

    if celltype_col not in adata.obs.columns:
        for col in ["CellType", "celltype", "cell.type", "label", "cluster"]:
            if col in adata.obs.columns:
                celltype_col = col
                break

    cell_types = adata.obs[celltype_col].values
    type_list = sorted(set(cell_types))
    n_types = len(type_list)
    print(f"  Cell types: {n_types} ({', '.join(type_list)})")

    train_idx, test_idx = train_test_split(
        np.arange(adata.n_obs), test_size=0.2, random_state=SEED, stratify=cell_types
    )
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()
    print(f"  Train cells: {train_adata.n_obs}, Test cells: {test_adata.n_obs}")

    X_train = train_adata.X
    X_test = test_adata.X
    if issparse(X_train):
        X_train = X_train.toarray()
    if issparse(X_test):
        X_test = X_test.toarray()
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    train_labels = train_adata.obs[celltype_col].values
    test_labels = test_adata.obs[celltype_col].values
    type_to_idx = {t: i for i, t in enumerate(type_list)}
    test_type_indices = np.array([type_to_idx[t] for t in test_labels])

    # ── DeconBenchmark reference groups ─────────────────────────────
    # cellTypeExpr: CPM-average per cell type (genes x cell_types)
    gene_names = list(adata.var_names)
    n_genes = X_train.shape[1]
    cellTypeExpr = np.zeros((n_genes, n_types), dtype=np.float64)
    for i, ct in enumerate(type_list):
        mask = train_labels == ct
        ct_sum = X_train[mask].sum(axis=0)
        total = ct_sum.sum()
        if total > 0:
            cellTypeExpr[:, i] = ct_sum / total * 1e6

    # markers: top N most specific genes per cell type (highest fc vs others)
    n_markers = 50
    markers_dict = {}
    all_sig = set()
    for i, ct in enumerate(type_list):
        ct_mask = train_labels == ct
        ct_mean = X_train[ct_mask].mean(axis=0)
        other_mean = X_train[~ct_mask].mean(axis=0)
        fc = np.where(other_mean > 0, ct_mean / other_mean, ct_mean)
        top_idx = np.argsort(fc)[-n_markers:]
        genes = [gene_names[idx] for idx in top_idx]
        markers_dict[ct] = genes
        all_sig.update(genes)

    sigGenes = sorted(all_sig)
    sig_idx = [gene_names.index(g) for g in sigGenes]
    signature = cellTypeExpr[sig_idx, :]

    proportions = np.zeros((n_samples, n_types))
    mixtures = np.zeros((n_samples, X_test.shape[1]))

    print(f"  Generating {n_samples} pseudo-bulk mixtures...")
    for i in range(n_samples):
        alpha = np.ones(n_types) * 0.1
        p = rng.dirichlet(alpha)
        proportions[i] = p
        n_cells = max(10, int(rng.poisson(n_cells_per_sample)))
        selected_types = rng.choice(n_types, size=n_cells, p=p)
        mix = np.zeros(X_test.shape[1])
        for ct_idx in selected_types:
            cell_mask = test_type_indices == ct_idx
            if cell_mask.sum() == 0:
                continue
            cell_idx = rng.choice(np.where(cell_mask)[0])
            mix += X_test[cell_idx]
        total = mix.sum()
        if total > 0:
            mix = mix / total * 1e6
        mixtures[i] = np.log1p(mix)

    return {
        "singleCellExpr": X_train,
        "singleCellLabels": train_labels,
        "bulk": mixtures,
        "bulk_labels": proportions,
        "type_list": type_list,
        "gene_names": gene_names,
        "cellTypeExpr": cellTypeExpr,
        "sigGenes": sigGenes,
        "signature": signature,
        "markers": markers_dict,
    }


def write_deconbenchmark_h5(h5_path, data):
    import h5py

    if os.path.exists(h5_path):
        os.remove(h5_path)

    genes = data["gene_names"]
    n_cells = data["singleCellExpr"].shape[0]
    n_samples = data["bulk"].shape[0]

    with h5py.File(h5_path, "w") as f:
        # DeconBenchmark convention: matrices stored as (observations, features)
        # in H5. R's rhdf5::h5read transposes on read, making rownames (features)
        # match matrix rows. Python DeconUtils.transpose then reverses this.
        grp = f.create_group("singleCellExpr")
        grp.create_dataset("values", data=data["singleCellExpr"].astype(np.float64))
        grp.create_dataset("rownames", data=np.array(genes, dtype="S"))
        grp.create_dataset("colnames", data=np.array([f"cell_{i}" for i in range(n_cells)], dtype="S"))
        grp = f.create_group("singleCellLabels")
        grp.create_dataset("values", data=np.array(data["singleCellLabels"], dtype="S"))
        grp = f.create_group("bulk")
        grp.create_dataset("values", data=data["bulk"].astype(np.float64))
        grp.create_dataset("rownames", data=np.array(genes, dtype="S"))
        grp.create_dataset("colnames", data=np.array([f"sample_{i}" for i in range(n_samples)], dtype="S"))
        grp = f.create_group("seed")
        grp.create_dataset("values", data=SEED)

        # ── DeconBenchmark optional groups ──────────────────────────
        n_types = len(data.get("type_list", []))
        _ct = data.get("type_list", [])

        # nCellTypes
        f.create_group("nCellTypes").create_dataset("values", data=n_types)

        # cellTypeExpr: genes x cell_types (CPM averages)
        if "cellTypeExpr" in data:
            cte = np.asarray(data["cellTypeExpr"], dtype=np.float64)
            grp = f.create_group("cellTypeExpr")
            grp.create_dataset("values", data=cte)
            grp.create_dataset("rownames", data=np.array(genes, dtype="S"))
            grp.create_dataset("colnames", data=np.array(_ct, dtype="S"))

        # sigGenes
        if "sigGenes" in data:
            f.create_group("sigGenes").create_dataset("values", data=np.array(data["sigGenes"], dtype="S"))

        # signature: sig_genes x cell_types
        if "signature" in data:
            grp = f.create_group("signature")
            grp.create_dataset("values", data=np.asarray(data["signature"], dtype=np.float64))
            grp.create_dataset("rownames", data=np.array(data["sigGenes"], dtype="S"))
            grp.create_dataset("colnames", data=np.array(_ct, dtype="S"))

        # markers: variable-length strings per cell type
        if "markers" in data and data["markers"]:
            md = data["markers"]
            ct_names = list(md.keys())
            ds = f.create_dataset("markers/values", shape=(len(ct_names),), dtype=h5py.string_dtype())
            for i, ct in enumerate(ct_names):
                ds[i] = ",".join(md[ct])
            f.create_dataset("markers/names", data=np.array(ct_names, dtype="S"))

        # isMethylation
        f.create_group("isMethylation").create_dataset("values", data=0)

        # singleCellSubjects
        f.create_group("singleCellSubjects").create_dataset(
            "values", data=np.array([f"subject_{i}" for i in range(n_cells)], dtype="S")
        )


def read_container_output(results_h5):
    import h5py
    import pandas as pd

    with h5py.File(results_h5, "r") as f:
        if "P/values" not in f:
            raise KeyError(f"No P/values in results. Groups: {list(f.keys())}")
        P_raw = f["P/values"][:]
        # Handle structured (record) arrays — some R-based methods write
        # named columns (e.g. [('B_cells', '<f8'), ...]) instead of a plain 2D array.
        colnames = None
        if P_raw.dtype.names is not None:
            colnames = list(P_raw.dtype.names)
            P = np.column_stack([P_raw[n] for n in colnames])
        else:
            P = np.asarray(P_raw, dtype=np.float64)
        if "P/colnames" in f:
            colnames = [x.decode() for x in f["P/colnames"][:]]
        rownames = [x.decode() for x in f["P/rownames"][:]] if "P/rownames" in f else None

    if colnames and rownames:
        if P.shape[0] == len(rownames):
            cell_types, samples = colnames, rownames
        else:
            P = P.T
            cell_types, samples = colnames, rownames
    else:
        n_types, n_samp = P.shape
        cell_types = [f"type_{i}" for i in range(n_types)]
        samples = [f"sample_{i}" for i in range(n_samp)]

    return pd.DataFrame(P, index=samples, columns=cell_types)


def run_container(sif_path, input_h5, output_dir, timeout=7200, gpu=False, fixed_script=None):
    if not os.path.exists(sif_path):
        raise FileNotFoundError(f"SIF not found: {sif_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_h5 = str(output_dir / "results.h5")

    cmd = ["apptainer", "run"]
    if gpu:
        cmd.append("--nv")

    # Bind fixed script over container entry point if available
    if fixed_script and os.path.exists(fixed_script):
        ext = os.path.splitext(fixed_script)[1].lower()
        fn = os.path.basename(fixed_script)
        bind_targets = {".py": ["/code/run.py"], ".r": ["/code/run.R"], ".sh": ["/code/run.sh"]}.get(ext, [])
        method_name = Path(fixed_script).parent.name
        # Also try method-specific path (e.g. /code/bulkgpt/run.py)
        method_specific = {".py": f"/code/{method_name}/run.py",
                          ".r": f"/code/{method_name}/run.R",
                          ".sh": f"/code/{method_name}/run.sh"}.get(ext)
        if method_specific:
            bind_targets.append(method_specific)
        for target in bind_targets:
            cmd.extend(["--bind", f"{os.path.abspath(fixed_script)}:{target}"])
            print(f"  Fixed script: {fn} -> {target}")

    cmd.extend([
        "--bind", f"{os.path.abspath(input_h5)}:/input/args.h5",
        "--bind", f"{output_dir}:/output",
        "--env", "INPUT_PATH=/input/args.h5",
        "--env", "OUTPUT_PATH=/output/results.h5",
        "--env", "HDF5_USE_FILE_LOCKING=FALSE",
        str(sif_path),
    ])

    print(f"  Running container (timeout={timeout}s)...")
    sys.stdout.flush()
    from core.resources import GPUMonitor
    gpu_mon = GPUMonitor()
    gpu_mon.start()
    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - start
    gpu_mon.stop()
    gpu_delta = gpu_mon.get_delta()

    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                print(f"    [err] {line.strip()}")
    print(f"  Return code: {result.returncode}, Elapsed: {elapsed:.1f}s")

    # Write resources.json with GPU peak data
    _write_container_resources(output_dir, elapsed, gpu_delta)

    if result.returncode != 0 and not os.path.exists(results_h5):
        raise RuntimeError(f"Container failed (rc={result.returncode})")
    if not os.path.exists(results_h5):
        alt = output_dir / "output.h5"
        results_h5 = str(alt) if alt.exists() else results_h5

    return results_h5, elapsed


def _write_container_resources(output_dir, elapsed, gpu_info=None):
    """Write resources.json using children CPU stats after container completes.

    The apptainer subprocess has already exited by this point (subprocess.run is
    synchronous), so per-process psutil tracking is not possible. Instead read
    accumulated children CPU time from /proc/self/stat and system-wide GPU info.
    If *gpu_info* is provided (from GPUMonitor), uses it for per-method GPU delta.
    """
    try:
        from core.resources import collect_resources_for_method
        import json
        info = collect_resources_for_method()
        if gpu_info:
            info["gpu"] = gpu_info
        info["phase"] = "container"
        info["elapsed_s"] = round(elapsed, 1)
        rpath = Path(output_dir) / "resources.json"
        with open(rpath, "w") as f:
            json.dump(info, f, indent=2)
    except Exception:
        pass  # non-critical


def enrich_h5(h5_path, out_dir):
    """Add missing DeconBenchmark optional groups to an H5 file.

    Reads existing H5 (may only have basic groups), computes cellTypeExpr,
    sigGenes, signature, markers, nCellTypes, isMethylation, singleCellSubjects,
    and writes a complete enriched H5 with consistent gene dimensions across
    all matrices. Returns path to enriched file.
    """
    import os as _osenv
    _osenv.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py

    out_dir = Path(out_dir)

    needed = {"cellTypeExpr", "sigGenes", "signature", "markers",
              "nCellTypes", "isMethylation", "singleCellSubjects"}
    with h5py.File(h5_path, "r") as f:
        existing = set(f.keys())
    if needed.issubset(existing):
        return h5_path

    # Cache enriched H5 at dataset level (under dataset results dir)
    # out_dir = results/2_realbulk/{dataset}/{method}/
    # cache   = results/2_realbulk/{dataset}/enriched_input.h5
    dataset_cache = Path(out_dir).parent / "enriched_input.h5"
    if dataset_cache.exists():
        return str(dataset_cache)

    print("  Enriching H5 with optional DeconBenchmark groups...")

    # ── Read source data ─────────────────────────────────────
    with h5py.File(h5_path, "r") as f:
        sce_raw = f["singleCellExpr/values"][:]
        scl = [x.decode() for x in f["singleCellLabels/values"][:]]
        sce_rownames = [x.decode() for x in f["singleCellExpr/rownames"][:]]
        sce_colnames = [x.decode() for x in f["singleCellExpr/colnames"][:]]

        # Determine scRNA orientation by matching dimensions to label count
        # (the number of cells is known from singleCellLabels).
        n_cells_from_labels = len(scl)

        if sce_raw.shape[0] == n_cells_from_labels:
            # axis 0 = cells, axis 1 = genes
            sce = sce_raw  # (n_cells, n_genes)
            # Canonical: rownames=genes, colnames=cells.
            # Some H5 writers (e.g. Hao datasets) swap rownames/colnames;
            # verify by checking which name array length matches n_genes.
            if len(sce_rownames) == sce_raw.shape[1]:
                sce_gene_names = sce_rownames
            elif len(sce_colnames) == sce_raw.shape[1]:
                sce_gene_names = sce_colnames
            else:
                raise ValueError(
                    f"Cannot identify gene names: rownames({len(sce_rownames)}) "
                    f"and colnames({len(sce_colnames)}) mismatch "
                    f"n_genes({sce_raw.shape[1]})"
                )
        elif sce_raw.shape[1] == n_cells_from_labels:
            # axis 0 = genes, axis 1 = cells — R rhdf5 wrote transposed
            sce = sce_raw.T  # becomes (n_cells, n_genes)
            # After transpose: dim0=cells, dim1=genes, so colnames should be genes.
            # Verify by count in case rownames/colnames are also swapped.
            if len(sce_colnames) == sce_raw.shape[0]:
                sce_gene_names = sce_colnames
            elif len(sce_rownames) == sce_raw.shape[0]:
                sce_gene_names = sce_rownames
            else:
                raise ValueError(
                    f"Cannot identify gene names (transposed): "
                    f"rownames({len(sce_rownames)}) and colnames({len(sce_colnames)}) "
                    f"mismatch n_genes({sce_raw.shape[0]})"
                )
        else:
            raise ValueError(
                f"Cannot determine H5 orientation: singleCellExpr {sce_raw.shape} "
                f"vs {n_cells_from_labels} labels"
            )

        n_cells, n_genes = sce.shape

        # Read bulk data & genes — stored as (n_genes, n_samples) in our files
        bulk_raw = f["bulk/values"][:]
        bulk_rownames = [x.decode() for x in f["bulk/rownames"][:]]
        if bulk_raw.shape[0] == len(bulk_rownames):
            bulk_genes = bulk_rownames  # (n_genes, n_samples)
        elif bulk_raw.shape[1] == len(bulk_rownames):
            bulk_raw = bulk_raw.T  # (n_samples, n_genes) → (n_genes, n_samples)
            bulk_genes = bulk_rownames
        else:
            bulk_genes = [x.decode() for x in f["bulk/colnames"][:]]

    # ── Align gene sets ──────────────────────────────────────
    # scRNA and bulk may have different gene sets (e.g. sdy67.h5:
    # scRNA=1344 genes, bulk=17387). Compute cellTypeExpr from scRNA
    # then filter bulk to match, so all matrices share consistent dims.
    common_genes = sorted(set(sce_gene_names) & set(bulk_genes))
    if len(common_genes) < 10:
        raise ValueError(f"Fewer than 10 common genes between bulk and scRNA: {len(common_genes)}")
    if len(common_genes) < n_genes:
        print(f"    Note: aligning bulk ({len(bulk_genes)} genes) to scRNA ({len(common_genes)} common genes)")

    # Filter scRNA to common genes
    sce_gene_idx = [sce_gene_names.index(g) for g in common_genes]
    sce = sce[:, sce_gene_idx]
    n_genes = len(common_genes)

    # ── Compute reference groups ─────────────────────────────
    type_list = sorted(set(scl))
    labels_arr = np.array(scl)
    n_types = len(type_list)

    # cellTypeExpr: CPM per cell type, stored as (n_types, n_genes)
    # DeconBenchmark stores matrices as (observations, features) in H5.
    # R's rhdf5::h5read transposes when reading, yielding (features, obs)
    # in R, matching rownames to row-count.
    cte = np.zeros((n_types, n_genes), dtype=np.float64)
    for i, ct in enumerate(type_list):
        mask = labels_arr == ct
        ct_sum = sce[mask].sum(axis=0)
        total = ct_sum.sum()
        if total > 0:
            cte[i, :] = ct_sum / total * 1e6

    # markers: top 50 most specific genes per cell type
    n_markers = 50
    markers_dict = {}
    all_sig = set()
    for i, ct in enumerate(type_list):
        mask = labels_arr == ct
        ct_mean = sce[mask].mean(axis=0)
        other_mean = sce[~mask].mean(axis=0)
        fc = np.where(other_mean > 0, ct_mean / other_mean, ct_mean)
        top_idx = np.argsort(fc)[-n_markers:]
        markers_dict[ct] = [common_genes[idx] for idx in top_idx]
        all_sig.update(markers_dict[ct])

    sigGenes = sorted(all_sig)
    sig_idx = [common_genes.index(g) for g in sigGenes]
    signature = cte[:, sig_idx]

    # ── Write enriched H5 (rewrite, with bulk filtered) ─────
    enriched = str(dataset_cache)
    dataset_cache.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first, then atomic rename for thread safety
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", dir=str(dataset_cache.parent))
    os.close(tmp_fd)
    import os as _os
    try:
        with h5py.File(tmp_path, "w") as f:
            # Filter bulk to common genes (bulk_raw is (n_genes, n_samples))
            bulk_gene_idx = [bulk_genes.index(g) for g in common_genes]
            bulk_filtered = bulk_raw[bulk_gene_idx, :]

            # Write DeconBenchmark standard format:
            #   bulk: (n_samples, n_genes), rownames=genes, colnames=samples
            #   singleCellExpr: (n_cells, n_genes), rownames=genes, colnames=cell_i
            # This matches what SIF containers using DeconUtils.getArgs() expect.
            g = f.create_group("bulk")
            g.create_dataset("values", data=bulk_filtered.T)  # (n_samples, n_genes)
            g.create_dataset("rownames", data=np.array(common_genes, dtype="S"))
            bulk_colnames = [x.decode() for x in h5py.File(h5_path, "r")["bulk/colnames"][:]]
            g.create_dataset("colnames", data=np.array(bulk_colnames, dtype="S"))

            g = f.create_group("singleCellExpr")
            g.create_dataset("values", data=sce)  # (n_cells, n_genes)
            g.create_dataset("rownames", data=np.array(common_genes, dtype="S"))
            g.create_dataset("colnames", data=np.array([f"cell_{i}" for i in range(n_cells)], dtype="S"))

            g = f.create_group("singleCellLabels")
            g.create_dataset("values", data=np.array(scl, dtype="S"))

            with h5py.File(h5_path, "r") as src:
                if "ground_truth" in src:
                    src.copy("ground_truth", f, name="ground_truth")
                if "seed" in src:
                    src.copy("seed", f, name="seed")

            # Ensure scalar groups use 1-element arrays
            def _ensure_1d_array(grp_name, val):
                if grp_name in f:
                    del f[grp_name]
                f.create_group(grp_name).create_dataset("values", data=[val])

            _ensure_1d_array("nCellTypes", n_types)
            _ensure_1d_array("isMethylation", 0)

            # Add missing groups
            g = f.create_group("cellTypeExpr")
            g.create_dataset("values", data=cte)
            g.create_dataset("rownames", data=np.array(common_genes, dtype="S"))
            g.create_dataset("colnames", data=np.array(type_list, dtype="S"))

            f.create_group("sigGenes").create_dataset("values", data=np.array(sigGenes, dtype="S"))

            g = f.create_group("signature")
            g.create_dataset("values", data=signature)
            g.create_dataset("rownames", data=np.array(sigGenes, dtype="S"))
            g.create_dataset("colnames", data=np.array(type_list, dtype="S"))

            ct_names = list(markers_dict.keys())
            ds = f.create_dataset("markers/values", shape=(len(ct_names),), dtype=h5py.string_dtype())
            for i, ct in enumerate(ct_names):
                ds[i] = ",".join(markers_dict[ct])
            f.create_dataset("markers/names", data=np.array(ct_names, dtype="S"))

            f.create_group("singleCellSubjects").create_dataset(
                "values", data=np.array([f"subject_{i}" for i in range(n_cells)], dtype="S")
            )

            _os.rename(tmp_path, enriched)
    except Exception:
        if _os.path.exists(tmp_path):
            _os.unlink(tmp_path)
        raise

    print(f"  Enriched H5 -> {enriched}")
    return enriched


def main(method_name, description=None):
    """Generic main for predict-only container methods."""
    if description is None:
        description = f"{method_name} — Deconvolution via Apptainer container"

    np.random.seed(SEED)
    args = make_parser(description).parse_args()
    cfg = load_config(args.config)

    sif_path = args.sif or str(resolve(cfg["container"]["sif_path"]))
    timeout = cfg["container"].get("timeout", 7200)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover fixed_script override
    fixed_script = None
    fixed_dir = Path(__file__).resolve().parent.parent.parent / "fixed_scripts" / method_name.lower()
    if fixed_dir.exists():
        for fname in ("run.py", "run.R", "run.sh"):
            fpath = fixed_dir / fname
            if fpath.exists():
                fixed_script = str(fpath)
                break

    print("=" * 60)
    print(f"{method_name} Deconvolution")
    print("=" * 60)

    if args.h5:
        print(f"\n[Input] DeconBenchmark H5: {args.h5}")
        enriched_h5 = enrich_h5(args.h5, out_dir)
        results_h5, elapsed = run_container(sif_path, enriched_h5, out_dir, timeout, gpu=cfg.get("gpu", False), fixed_script=fixed_script)
        pred_df = read_container_output(results_h5)
        pred_csv = out_dir / "proportions.csv"
        pred_df.to_csv(str(pred_csv))
        print(f"  Predictions saved -> {pred_csv}")

        if args.ground_truth and os.path.exists(args.ground_truth):
            print(f"  (ground truth available — metrics computed by post-hoc evaluation)")

    elif args.data:
        import anndata as ad

        print(f"\n[1] Loading scRNA-seq reference: {args.data}")
        adata = ad.read_h5ad(str(args.data))
        print(f"  Shape: {adata.shape}")

        celltype_col = cfg["data"].get("celltype_col", "cell_type")
        if celltype_col not in adata.obs.columns:
            for c in ["CellType", "celltype", "cell.type", "label", "cluster"]:
                if c in adata.obs.columns:
                    celltype_col = c
                    break

        print(f"\n[2] Generating pseudo-bulk mixtures...")
        n_samples = cfg["data"].get("n_pseudo_bulk", 2000)
        pb_data = generate_pseudo_bulk(adata, n_samples=n_samples, celltype_col=celltype_col)
        type_list = pb_data["type_list"]

        print(f"\n[3] Writing DeconBenchmark H5...")
        h5_path = str(out_dir / f"{method_name.lower()}_input.h5")
        write_deconbenchmark_h5(h5_path, pb_data)

        print(f"\n[4] Running container...")
        results_h5, elapsed = run_container(sif_path, h5_path, out_dir, timeout, gpu=cfg.get("gpu", False), fixed_script=fixed_script)

        print(f"\n[5] Reading results...")
        pred_df = read_container_output(results_h5)
        pred_csv = out_dir / "proportions.csv"
        pred_df.to_csv(str(pred_csv))
        print(f"  Predictions -> {pred_csv}, Shape: {pred_df.shape}")

        print(f"  (metrics computed by post-hoc evaluation)")

        with open(out_dir / "cell_types.json", "w") as f:
            json.dump(type_list, f)

    else:
        print("ERROR: Provide --data (h5ad) or --h5 (DeconBenchmark format)")
        sys.exit(1)

    print(f"\nDone. Output in {out_dir}/")
