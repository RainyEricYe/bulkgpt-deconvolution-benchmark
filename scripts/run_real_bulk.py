#!/usr/bin/env python3
"""Run all tested methods on all real-bulk datasets.

Usage
-----
    python scripts/run_real_bulk.py                         # full run
    python scripts/run_real_bulk.py --methods nnls ols       # filtered
    python scripts/run_real_bulk.py --datasets sweetwater    # filtered
    python scripts/run_real_bulk.py --quick                  # minimal epochs
    python scripts/run_real_bulk.py --dry-run                # print only
    python scripts/run_real_bulk.py --parallel 4             # force N workers

Output
------
    results/2_realbulk/{dataset}/{method}/{experiment}/
      ├── config.yaml
      ├── run.log / train.log / predict.log
      ├── proportions.csv
      ├── metrics.json
      └── resources.json
"""

import datetime
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_BASE = PROJECT_ROOT / "results" / "2_realbulk"
DATA_DIR = PROJECT_ROOT / "data" / "2_real_bulk"

# ── Known method categories ──────────────────────────────────────────────────

LINEAR_METHODS = {"nnls", "ols", "ridge", "nusvr"}
CONDA_TRAIN_PREDICT = {"scgpt", "scgpt_lora", "geneformer", "stack", "transcriptformer", "scfoundation"}
SPECIAL_CONTAINERS = {"sweetwater", "tape"}

# ── Logging ──────────────────────────────────────────────────────────────────

BLUE = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def blue(s): print(f"{BLUE}{s}{RESET}")
def green(s): print(f"{GREEN}{s}{RESET}")
def red(s): print(f"{RED}{s}{RESET}")
def yellow(s): print(f"{YELLOW}{s}{RESET}")


# ── System load monitor ──────────────────────────────────────────────────────

def _get_ncpu() -> int:
    return os.cpu_count() or 8


def _get_nvidia_processes() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return len([l for l in result.stdout.strip().split("\n") if l.strip()])
    except Exception:
        return 0


def _get_gpu_mem_used() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        memories = [float(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
        return sum(memories) / 1024.0 if memories else 0.0
    except Exception:
        return 0.0


def auto_parallelism() -> int:
    ncpu = _get_ncpu()
    gpu_procs = _get_nvidia_processes()
    gpu_mem = _get_gpu_mem_used()
    cap = 12

    if gpu_procs > 0 or gpu_mem > 40:
        workers = max(1, ncpu // 2)
        yellow(f"  GPU active ({gpu_procs} procs, {gpu_mem:.1f} GB used) "
               f"→ {min(workers, cap)} workers (cap={cap})")
    else:
        workers = max(1, ncpu - 2)
        blue(f"  GPU idle → {min(workers, cap)} workers ({ncpu} CPUs, cap={cap})")
    return min(workers, cap)


# ── Task accounting ──────────────────────────────────────────────────────────

PASSED = 0
FAILED = 0
SKIPPED = 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_config(config_dir: Path, base_config: dict, overrides: dict) -> Path:
    import yaml
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(base_config)

    def _deep_merge(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                _deep_merge(d[k], v)
            else:
                d[k] = v

    _deep_merge(cfg, overrides)
    out_path = config_dir / "config.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return out_path


def _get_manifest_field(method: str, field: str) -> Optional[str]:
    import yaml
    manifest_path = PROJECT_ROOT / "methods" / method / "manifest.yaml"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        m = yaml.safe_load(f)
    parts = field.split(".")
    val = m
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val if isinstance(val, str) else None


def _method_experiments(method: str) -> list[tuple[str, str]]:
    """Return [(experiment_name, config_path), ...] for a method."""
    configs_dir = PROJECT_ROOT / "methods" / method / "configs"
    if not configs_dir.exists():
        return []
    experiments = []
    for fname in sorted(configs_dir.iterdir()):
        if fname.suffix in (".yaml", ".yml"):
            experiments.append((fname.stem, str(fname)))
    return experiments


# ── Command runner ──────────────────────────────────────────────────────────

def _run_cmd(cmd: list, out_dir: Path, method: str, dataset: str,
             exp_name: str, phase: str = "run", timeout: int = 14400) -> bool:
    global PASSED, FAILED

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"{phase}.log"
    start = time.monotonic()

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE", "TF_REPO": str(PROJECT_ROOT.parent.parent / "TranscriptFormer")},
        )
        elapsed = time.monotonic() - start
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        with open(log_file, "w") as f:
            f.write(f"[TIMEOUT after {elapsed:.0f}s]\n")
        red(f"  ✗ {method}/{exp_name} on {dataset} — TIMEOUT ({elapsed:.0f}s)")
        FAILED += 1
        return False
    except Exception as e:
        elapsed = time.monotonic() - start
        with open(log_file, "w") as f:
            f.write(f"[ERROR] {e}\n{traceback.format_exc()}\n")
        red(f"  ✗ {method}/{exp_name} on {dataset} — ERROR: {e}")
        FAILED += 1
        return False

    with open(log_file, "w") as f:
        f.write(f"# {phase.upper()} — {method}/{exp_name} on {dataset}\n")
        f.write(f"# {datetime.datetime.now().isoformat()}\n")
        f.write(f"# Elapsed: {elapsed:.1f}s\n")
        f.write(f"# Return code: {proc.returncode}\n")
        f.write(f"# Command: {' '.join(str(x) for x in cmd)}\n\n")
        if proc.stdout:
            f.write("--- STDOUT ---\n")
            f.write(proc.stdout)
        if proc.stderr:
            f.write("--- STDERR ---\n")
            f.write(proc.stderr)

    rc = proc.returncode
    output_files = list(out_dir.glob("proportions.csv"))
    has_output = len(output_files) > 0

    if rc == 0 and has_output:
        PASSED += 1
        green(f"  ✓ {method}/{exp_name} on {dataset} ({elapsed:.0f}s)")
        return True
    elif rc == 0 and not has_output:
        yellow(f"  ~ {method}/{exp_name} on {dataset} — rc=0 but no output ({elapsed:.0f}s)")
        PASSED += 1
        return True
    else:
        red(f"  ✗ {method}/{exp_name} on {dataset} — rc={rc} ({elapsed:.0f}s)")
        FAILED += 1
        return False


# ── H5 orientation normalizer (for R container compatibility) ─────────────

def _normalize_h5_orientation(h5_path: Path) -> Path:
    """Normalize H5 matrices to DeconBenchmark standard (features first).

    R's rhdf5::h5read transposes 2D arrays on read. The DeconBenchmark convention
    stores data with features (genes) as the first dimension and samples/cells as
    the second (e.g. (n_genes, n_samples)). This function transposes any matrix
    that appears to be stored in the opposite orientation (samples first), and
    fixes rownames/colnames that are swapped relative to the data shape.

    Operates in-place on a file already in /tmp.
    """
    import h5py

    h5_path = Path(h5_path)
    MATRIX_GROUPS = ["bulk", "singleCellExpr", "cellTypeExpr", "signature"]
    changed = False

    with h5py.File(str(h5_path), "r+") as f:
        for grp_name in MATRIX_GROUPS:
            if grp_name not in f:
                continue
            grp = f[grp_name]
            if "values" not in grp:
                continue
            dset = grp["values"]
            if dset.ndim != 2:
                continue
            dims = dset.shape

            # Read original names
            orig_rownames = grp["rownames"][:] if "rownames" in grp else None
            orig_colnames = grp["colnames"][:] if "colnames" in grp else None

            # Transpose if samples-first: samples < features in deconvolution,
            # but in some datasets cells > genes so also check name alignment.
            should_transpose = dims[0] < dims[1]
            names_swapped = (orig_rownames is not None and len(orig_rownames) == dims[1] and
                             orig_colnames is not None and len(orig_colnames) == dims[0])

            if should_transpose:
                # Transpose to features-first
                data = dset[:]
                del grp["values"]
                grp.create_dataset("values", data=data.T, dtype=data.dtype)
                # After transpose: (old_ncol, old_nrow) — assign names by count
                new_nrow, new_ncol = dims[1], dims[0]
                new_rownames, new_colnames = None, None
                if orig_rownames is not None and len(orig_rownames) == new_nrow:
                    new_rownames = orig_rownames
                elif orig_colnames is not None and len(orig_colnames) == new_nrow:
                    new_rownames = orig_colnames
                if orig_colnames is not None and len(orig_colnames) == new_ncol:
                    new_colnames = orig_colnames
                elif orig_rownames is not None and len(orig_rownames) == new_ncol:
                    new_colnames = orig_rownames
                for nm, label in [(new_rownames, "rownames"), (new_colnames, "colnames")]:
                    if nm is not None:
                        if label in grp: del grp[label]
                        grp.create_dataset(label, data=nm)
                changed = True
                print(f"  [h5norm] Transposed '{grp_name}' ({dims[0]}×{dims[1]} → {new_nrow}×{new_ncol})")
            elif names_swapped:
                # No transpose needed but names are swapped relative to shape
                del grp["rownames"]
                del grp["colnames"]
                grp.create_dataset("rownames", data=orig_colnames)
                grp.create_dataset("colnames", data=orig_rownames)
                changed = True
                print(f"  [h5norm] Fixed swapped names for '{grp_name}' ({dims[0]}×{dims[1]})")

    return h5_path


# ── Method dispatchers ──────────────────────────────────────────────────────

def _dispatch_container(method, dataset, h5_path, gt_path, out_dir, quick, noop):
    config_src = PROJECT_ROOT / "methods" / method / "configs" / "default.yaml"
    if not config_src.exists():
        return False
    cfg = _load_yaml(config_src)
    if "container" not in cfg or "sif_path" not in cfg.get("container", {}):
        sif = _get_manifest_field(method, "container_image")
        if sif:
            cfg.setdefault("container", {})["sif_path"] = sif
        else:
            return False

    config_path = _write_config(out_dir / "configs", cfg, {})
    if noop:
        return True

    # Normalize H5 for R-based containers (h5read transposes 2D arrays)
    R_METHODS = {"condecon", "demixsc", "squid", "hspe", "toast", "lindeconseq"}
    h5_input = _normalize_h5_orientation(h5_path) if method in R_METHODS else h5_path

    return _run_cmd([
        sys.executable, str(PROJECT_ROOT / "methods" / method / "run.py"),
        "--config", str(config_path), "--mode", "predict",
        "--h5", str(h5_input), "--output-dir", str(out_dir),
        *(("--ground-truth", str(gt_path)) if gt_path else []),
    ], out_dir, method, dataset, "default")


def _dispatch_linear(method, dataset, h5_path, gt_path, out_dir, quick, noop):
    # Load method-specific params (e.g. nu, kernel for nusvr)
    config_src = PROJECT_ROOT / "methods" / method / "configs" / "default.yaml"
    base_cfg = _load_yaml(config_src) if config_src.exists() else {"method": method}
    base_cfg.setdefault("method", method)
    config_path = _write_config(out_dir / "configs", base_cfg, {})
    if noop:
        return True
    return _run_cmd([
        sys.executable, str(PROJECT_ROOT / "methods" / method / "run.py"),
        "--config", str(config_path), "--h5", str(h5_path),
        "--output-dir", str(out_dir),
        *(("--ground-truth", str(gt_path)) if gt_path else []),
    ], out_dir, method, dataset, "default")


def _dispatch_cibersortx(method, dataset, h5_path, gt_path, out_dir, quick, noop):
    config_src = PROJECT_ROOT / "methods" / "cibersortx" / "configs" / "default.yaml"
    cfg = _load_yaml(config_src) if config_src.exists() else {}
    config_path = _write_config(out_dir / "configs", cfg, {})
    if noop:
        return True
    return _run_cmd([
        sys.executable, str(PROJECT_ROOT / "methods" / "cibersortx" / "run.py"),
        "--config", str(config_path), "--h5", str(h5_path),
        "--output-dir", str(out_dir),
        *(("--ground-truth", str(gt_path)) if gt_path else []),
    ], out_dir, method, dataset, "default")


def _dispatch_decode(method, dataset, h5_path, gt_path, out_dir, quick, noop):
    config_src = PROJECT_ROOT / "methods" / "decode" / "configs" / "default.yaml"
    if not config_src.exists():
        return False
    cfg = _load_yaml(config_src)
    epochs = 3 if quick else cfg.get("training", {}).get("epochs", 10)
    n_pseudo = 200 if quick else cfg.get("training", {}).get("n_pseudo_bulk", 500)
    cfg["data"]["sc_ref"] = str(h5_path)
    cfg["data"]["bulk"] = str(h5_path)
    cfg["paths"]["output_dir"] = str(out_dir)
    cfg["training"]["epochs"] = epochs
    cfg["training"]["n_pseudo_bulk"] = n_pseudo
    config_path = _write_config(out_dir / "configs", cfg, {})

    if noop:
        return True

    ok = _run_cmd([
        sys.executable, str(PROJECT_ROOT / "methods" / "decode" / "run.py"),
        "--config", str(config_path), "--mode", "train",
    ], out_dir / "train", method, dataset, "train")

    if ok:
        best = None
        for ckpt in ["best_model.pt", "final_model.pt", "mbdeconv.pt"]:
            cand = out_dir / "checkpoints" / ckpt
            cand2 = out_dir / "checkpoint" / ckpt
            if cand.exists():
                best = cand
                break
            if cand2.exists():
                best = cand2
                break
            if (out_dir / ckpt).exists():
                best = out_dir / ckpt
                break
        if best:
            ok = _run_cmd([
                sys.executable, str(PROJECT_ROOT / "methods" / "decode" / "run.py"),
                "--config", str(config_path), "--mode", "predict",
            ], out_dir / "predict", method, dataset, "predict")
    return ok


def _dispatch_music(method, dataset, h5_path, gt_path, out_dir, quick, noop):
    config_src = PROJECT_ROOT / "methods" / "music" / "configs" / "default.yaml"
    if not config_src.exists():
        return False
    cfg = _load_yaml(config_src)
    cfg["paths"] = {
        "sc_ref": str(h5_path), "bulk": str(h5_path),
        "sif_path": str(PROJECT_ROOT / "containers" / "sif" / "music.sif"),
        "output": str(out_dir / "proportions.csv"),
    }
    config_path = _write_config(out_dir / "configs", cfg, {})
    if noop:
        return True
    return _run_cmd([
        sys.executable, str(PROJECT_ROOT / "methods" / "music" / "run.py"),
        "--config", str(config_path),
    ], out_dir, method, dataset, "default")


def _dispatch_special_container(method, dataset, h5_path, gt_path, out_dir, quick, noop):
    config_src = PROJECT_ROOT / "methods" / method / "configs" / "default.yaml"
    if not config_src.exists():
        return False
    cfg = _load_yaml(config_src)
    config_path = _write_config(out_dir / "configs", cfg, {})
    if noop:
        return True

    # Preprocess through enrich_h5 to align gene sets between scRNA and bulk
    from methods._shared.container_runner import enrich_h5
    enriched_h5 = enrich_h5(str(h5_path), str(out_dir))

    return _run_cmd([
        sys.executable, str(PROJECT_ROOT / "methods" / method / "run.py"),
        "--config", str(config_path), "--mode", "predict",
        "--h5", str(enriched_h5), "--output-dir", str(out_dir),
        *(("--ground-truth", str(gt_path)) if gt_path else []),
    ], out_dir, method, dataset, "default")


def _dispatch_train_predict(method, dataset, h5_path, gt_path, out_dir, quick, noop):
    experiments = _method_experiments(method)
    if not experiments:
        return False

    ok = True
    for exp_name, exp_config in experiments:
        exp_out = out_dir / exp_name
        exp_out.mkdir(parents=True, exist_ok=True)

        cfg = _load_yaml(Path(exp_config))

        # Set top-level keys early (used by methods like scgpt_lora
        # that read config directly rather than from nested paths)
        cfg["h5_path"] = str(h5_path)
        cfg["gt_path"] = str(gt_path) if gt_path else str(Path(h5_path).with_suffix("")) + "_gt.csv"
        cfg["output_dir"] = str(exp_out)
        cfg["results_dir"] = str(exp_out)
        if cfg.get("paths") is None:
            cfg["paths"] = {}
        cfg["paths"]["checkpoint_dir"] = str(exp_out)
        cfg["paths"]["output_dir"] = str(exp_out)

        # Skip configs where dataset/data is a non-dict string
        # (e.g. RidgeCV real-bulk configs: "dataset: sdy67")
        if isinstance(cfg.get("dataset"), str) or isinstance(cfg.get("data"), str):
            continue

        if cfg.get("dataset") is None:
            cfg["dataset"] = {}
        cfg["dataset"]["data_path"] = str(h5_path)
        if cfg.get("data") is None:
            cfg["data"] = {}
        cfg["data"]["data_path"] = str(h5_path)

        if quick:
            cfg.setdefault("training", {})["epochs"] = 3
            cfg.setdefault("training", {})["n_pseudo_bulk"] = 200

        config_path = _write_config(exp_out / "configs", cfg, {})
        if noop:
            continue

        env_name = _get_manifest_field(method, "conda_env") or "bulkgpt"
        train_script = _get_manifest_field(method, "entry.train")
        predict_script = _get_manifest_field(method, "entry.predict")

        if not train_script:
            ok = False
            continue

        train_script_path = PROJECT_ROOT / train_script
        log_path = exp_out / "train.log"

        cmd_train = [
            os.environ.get("CONDA_EXE", "conda"), "run", "-n", env_name, "--no-capture-output", "bash", "-c",
            f"cd {PROJECT_ROOT} && "
            f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && "
            f"python {train_script_path} --config {config_path} --log_file {log_path}",
        ]
        if not _run_cmd(cmd_train, exp_out, method, dataset, exp_name, phase="train",
                         timeout=14400):
            ok = False
            red(f"  Training failed for {method}/{exp_name}")
            continue

        if not predict_script:
            continue

        predict_script_path = PROJECT_ROOT / predict_script

        best_ckpt = None
        for ckpt_name in ["best_model.pt", "final_model.pt", "deconv_head.pt"]:
            for base in [exp_out / "checkpoints", exp_out / "checkpoint", exp_out]:
                cand = base / ckpt_name
                if cand.exists():
                    best_ckpt = cand
                    break
            if best_ckpt:
                break

        if not best_ckpt:
            red(f"  No checkpoint for {method}/{exp_name}")
            ok = False
            continue

        log_path = exp_out / "predict.log"
        cmd_predict = [
            os.environ.get("CONDA_EXE", "conda"), "run", "-n", env_name, "--no-capture-output", "bash", "-c",
            f"cd {PROJECT_ROOT} && "
            f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && "
            f"python {predict_script_path} "
            f"--config {config_path} --checkpoint {best_ckpt} --log_file {log_path}"
            + (f" --ground-truth {gt_path}" if gt_path else ""),
        ]
        if not _run_cmd(cmd_predict, exp_out, method, dataset, exp_name, phase="predict",
                         timeout=14400):
            ok = False

    return ok


def _dispatch_generic_conda(method, dataset, h5_path, gt_path, out_dir, quick, noop):
    """Run a conda method's run.py directly with --h5 / --output-dir CLI.

    If method name ends with ``_scaler``, passes ``--scaler`` and resolves
    the base method's run.py (e.g. ``bulkformer/random_scaler`` →
    ``methods/bulkformer/random/run.py --scaler``).
    """
    use_scaler = method.endswith("_scaler")
    base_method = method.rsplit("_scaler", 1)[0] if use_scaler else method
    run_script = PROJECT_ROOT / "methods" / base_method / "run.py"
    if not run_script.exists():
        yellow(f"  ? {method}: run.py not found at {run_script}")
        return False
    if noop:
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(run_script),
        "--h5", str(h5_path), "--output-dir", str(out_dir),
        *(("--ground-truth", str(gt_path)) if gt_path else []),
    ]
    if use_scaler:
        cmd.append("--scaler")
    return _run_cmd(cmd, out_dir, method, dataset, "default")


# ── Dispatch table ──────────────────────────────────────────────────────────

DISPATCH_TABLE = {
    # Linear baselines
    "nnls": _dispatch_linear,
    "ols": _dispatch_linear,
    "ridge": _dispatch_linear,
    "nusvr": _dispatch_linear,
    # ReCIDE (direct conda, same CLI as linear baselines)
    "recide": _dispatch_linear,
    # CIBERSORTx (conda, own main)
    "cibersortx": _dispatch_cibersortx,
    # Decode (needs train+predict)
    "decode": _dispatch_decode,
    # Music (special container)
    "music": _dispatch_music,
    # Special containers (sweetwater, tape)
    "sweetwater": _dispatch_special_container,
    "tape": _dispatch_special_container,
    # Conda train+predict
    "scgpt": _dispatch_train_predict,
    "scgpt_lora": _dispatch_train_predict,
    "geneformer": _dispatch_train_predict,
    "stack": _dispatch_train_predict,
    "transcriptformer": _dispatch_train_predict,
    "scfoundation": _dispatch_train_predict,
}


def get_dispatch(method: str):
    if method in DISPATCH_TABLE:
        return DISPATCH_TABLE[method]

    # _scaler variants resolve to their base method's dispatch
    if method.endswith("_scaler"):
        base_method = method.rsplit("_scaler", 1)[0]
        if base_method in DISPATCH_TABLE:
            return DISPATCH_TABLE[base_method]
        mode = _get_manifest_field(base_method, "mode")
        if mode == "conda":
            return _dispatch_generic_conda
        return _dispatch_generic_conda if mode else None

    mode = _get_manifest_field(method, "mode")
    if mode == "container":
        return _dispatch_container
    elif mode == "conda":
        return _dispatch_generic_conda
    return None




# ── Discovery ────────────────────────────────────────────────────────────────

def discover_methods() -> list[str]:
    methods = []
    # Top-level methods
    for manifest in sorted((PROJECT_ROOT / "methods").glob("*/manifest.yaml")):
        m = manifest.parent.name
        status = _get_manifest_field(m, "status")
        if status and status.strip().lower() == "deprecated":
            continue
        methods.append(m)
    # Sub-methods under top-level methods (e.g. bulkformer/random)
    for manifest in sorted((PROJECT_ROOT / "methods").glob("*/*/manifest.yaml")):
        m = "/".join(manifest.relative_to(PROJECT_ROOT / "methods").parent.parts)
        status = _get_manifest_field(m, "status")
        if status and status.strip().lower() == "deprecated":
            continue
        methods.append(m)
        # Auto-register _scaler variants for sub-methods that support it
        if manifest.parent.name in ("random", "bootstrap", "fstat", "mean_pool", "random_mean_pool"):
            methods.append(f"{m}_scaler")
    return methods


def discover_datasets() -> list[tuple[str, Path, Optional[Path]]]:
    datasets = []
    for f in sorted(DATA_DIR.iterdir()):
        if f.suffix == ".h5":
            name = f.stem
            gt = DATA_DIR / f"{name}_gt.csv"
            datasets.append((name, f, gt if gt.exists() else None))
    return datasets


# ── Run ─────────────────────────────────────────────────────────────────────

def _localize_h5(h5_path: Path) -> Path:
    """Return original H5 path; disable HDF5 file locking for safe GPFS reads.

    Previously copied H5 to /tmp to avoid GPFS locking conflicts, but the
    copy truncated large files (>4 GB).  Instead, set HDF5_USE_FILE_LOCKING
    so h5py reads safely from the GPFS path without file-level locking.
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return h5_path


def run_task(method, ds_name, ds_h5, ds_gt, results_dir, quick, noop):
    global PASSED, FAILED, SKIPPED

    dispatch = get_dispatch(method)
    if dispatch is None:
        yellow(f"  ? {method}: unknown method type (no manifest mode)")
        SKIPPED += 1
        return

    out_dir = results_dir / method

    # Localize H5 to avoid GPFS HDF5 locking conflicts
    local_h5 = _localize_h5(ds_h5)

    try:
        dispatch(method, ds_name, local_h5, ds_gt, out_dir, quick, noop)
    except Exception as e:
        red(f"  ✗ {method} on {ds_name} — EXCEPTION: {e}")
        traceback.print_exc()
        FAILED += 1
    finally:
        # Clean up local H5 copy (only if it was copied to /tmp)
        local_dir = local_h5.parent
        if str(local_dir).startswith("/tmp/") and local_dir.exists():
            import shutil
            shutil.rmtree(str(local_dir), ignore_errors=True)


# ── Post-hoc evaluation ─────────────────────────────────────────────────────

def _posthoc_evaluate(results_dir: Path, data_dir: Path, ds_name: str) -> int:
    """Evaluate proportions.csv files under *results_dir* that lack metrics.json.

    Returns number of methods evaluated.
    """
    from scripts.evaluate import evaluate_file

    gt_csv = data_dir / f"{ds_name}_gt.csv"
    if not gt_csv.exists():
        return 0

    n = 0
    for prop_csv in sorted(results_dir.rglob("proportions.csv")):
        metrics_json = prop_csv.parent / "metrics.json"
        if metrics_json.exists():
            continue

        try:
            evaluate_file(str(prop_csv), str(gt_csv), str(metrics_json))
            n += 1
        except (Exception, SystemExit) as e:
            print(f"  [eval] {prop_csv.parent.name}: {e}")

    return n


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all methods on real bulk data")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: results/2_realbulk)")
    args = parser.parse_args()

    global PASSED, FAILED, SKIPPED
    PASSED = FAILED = SKIPPED = 0

    results_dir = RESULTS_BASE
    if args.output:
        results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_methods = discover_methods()
    all_datasets = discover_datasets()

    if args.methods:
        all_methods = [m for m in all_methods if m in args.methods]
    if args.datasets:
        all_datasets = [(n, h, g) for n, h, g in all_datasets if n in args.datasets]

    print(f"\n{'=' * 60}")
    print(f"  Real Bulk Evaluation")
    print(f"  Methods: {len(all_methods)}")
    print(f"  Datasets: {len(all_datasets)}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'=' * 60}\n")

    for ds_name, ds_h5, ds_gt in all_datasets:
        gt_label = str(ds_gt) if ds_gt else "NO GT"
        print(f"  Dataset: {ds_name}  h5={ds_h5.name}  gt={gt_label}")

    n_parallel = args.parallel if args.parallel > 0 else auto_parallelism()
    if args.dry_run:
        n_parallel = 1

    print(f"\n  Parallel workers: {n_parallel}\n")
    print(f"  Total tasks: {len(all_methods) * len(all_datasets)} "
          f"({len(all_methods)} methods × {len(all_datasets)} datasets)")
    print()

    tasks = []
    for ds_name, ds_h5, ds_gt in all_datasets:
        for method in all_methods:
            tasks.append((method, ds_name, ds_h5, ds_gt))

    # Parallel dispatch via thread pool (safe: each task is an independent subprocess)
    def _run_one(method, ds_name, ds_h5, ds_gt):
        run_task(method, ds_name, ds_h5, ds_gt,
                 results_dir / ds_name, args.quick, args.dry_run)

    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        fut_to_task = {}
        for method, ds_name, ds_h5, ds_gt in tasks:
            fut = pool.submit(_run_one, method, ds_name, ds_h5, ds_gt)
            fut_to_task[fut] = (method, ds_name)

        for i, fut in enumerate(as_completed(fut_to_task)):
            method, ds_name = fut_to_task[fut]
            try:
                fut.result()
            except Exception as e:
                red(f"  ✗ {method} on {ds_name} — UNHANDLED: {e}")
                FAILED += 1

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"  Passed: {PASSED}  Failed: {FAILED}  Skipped: {SKIPPED}")
    print(f"{'=' * 60}\n")

    if not args.dry_run:
        # Post-hoc evaluate methods that have proportions.csv but no metrics.json
        n_evaluated = 0
        for ds_name, _, ds_gt in all_datasets:
            if not ds_gt:
                continue
            ds_dir = results_dir / ds_name
            if ds_dir.exists():
                n_evaluated += _posthoc_evaluate(ds_dir, DATA_DIR, ds_name)
        if n_evaluated > 0:
            print(f"\n  Post-hoc evaluated: {n_evaluated} methods\n")

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
