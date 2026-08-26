#!/usr/bin/env python3
"""Run all tested methods on all pseudo-bulk datasets.

Usage
-----
    python scripts/run_pseudo_bulk.py                         # full run
    python scripts/run_pseudo_bulk.py --methods nnls ols       # filtered
    python scripts/run_pseudo_bulk.py --datasets cellxgene_Liver    # filtered
    python scripts/run_pseudo_bulk.py --quick                  # minimal epochs
    python scripts/run_pseudo_bulk.py --dry-run                # print only
    python scripts/run_pseudo_bulk.py --parallel 4             # force N workers

Output
------
    results/1_pseudo_bulk/{dataset}/{method}/
      ├── config.yaml
      ├── run.log / train.log / predict.log
      ├── proportions.csv
      ├── metrics.json
      └── resources.json
"""

import datetime
import os
import shutil
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

RESULTS_BASE = PROJECT_ROOT / "results" / "1_pseudo_bulk"
DATA_DIR = PROJECT_ROOT / "data" / "1_pseudo_bulk"

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
             exp_name: str, phase: str = "run", timeout: int = 7200) -> bool:
    global PASSED, FAILED

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"{phase}.log"
    start = time.monotonic()

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
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
    config_src = PROJECT_ROOT / "tests" / "configs" / method / "default.yaml"
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
    h5_input = h5_path  # 1_pseudo_bulk H5 is already canonical (no transposition needed)

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
    config_src = PROJECT_ROOT / "tests" / "configs" / "cibersortx" / "default.yaml"
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
    config_src = PROJECT_ROOT / "tests" / "configs" / "decode" / "default.yaml"
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
            if cand.exists() or (out_dir / ckpt).exists():
                best = out_dir / "checkpoints" / ckpt if cand.exists() else out_dir / ckpt
                break
        if best:
            ok = _run_cmd([
                sys.executable, str(PROJECT_ROOT / "methods" / "decode" / "run.py"),
                "--config", str(config_path), "--mode", "predict",
            ], out_dir / "predict", method, dataset, "predict")
    return ok


def _dispatch_music(method, dataset, h5_path, gt_path, out_dir, quick, noop):
    config_src = PROJECT_ROOT / "tests" / "configs" / "music" / "default.yaml"
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
    config_src = PROJECT_ROOT / "tests" / "configs" / method / "default.yaml"
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
        cfg = _load_yaml(Path(exp_config))

        # Skip configs without recognized data keys
        if not isinstance(cfg.get("dataset"), dict) and not isinstance(cfg.get("data"), dict) \
           and "h5_path" not in cfg and "data_path" not in cfg:
            continue

        exp_out = out_dir / exp_name
        exp_out.mkdir(parents=True, exist_ok=True)

        if isinstance(cfg.get("dataset"), dict):
            cfg["dataset"]["data_path"] = str(h5_path)
        else:
            cfg["dataset"] = {"data_path": str(h5_path)}

        if isinstance(cfg.get("data"), dict):
            cfg["data"]["data_path"] = str(h5_path)
        else:
            cfg["data"] = {"data_path": str(h5_path)}

        if isinstance(cfg.get("paths"), dict):
            cfg["paths"]["checkpoint_dir"] = str(exp_out)
            cfg["paths"]["output_dir"] = str(exp_out)
        else:
            cfg["paths"] = {"checkpoint_dir": str(exp_out), "output_dir": str(exp_out)}

        # Top-level keys for methods that read config directly (e.g. scgpt_lora)
        cfg["h5_path"] = str(h5_path)
        cfg["checkpoint_dir"] = str(exp_out)
        # BulkFormer reads sc_ref for the reference data path
        cfg["sc_ref"] = str(h5_path)
        cfg["gt_path"] = str(gt_path) if gt_path else str(Path(h5_path).with_suffix("")) + "_gt.csv"
        cfg["output_dir"] = str(exp_out)
        cfg["results_dir"] = str(exp_out)

        if quick:
            if isinstance(cfg.get("training"), dict):
                cfg["training"]["epochs"] = 3
                cfg["training"]["n_pseudo_bulk"] = 200

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

        # Method-specific env vars (e.g. TF_REPO for transcriptformer)
        ENV_MAP = {
            "transcriptformer": "export TF_REPO=repo/transcriptformer && ",
            "scfoundation": "",
        }
        env_prefix = ENV_MAP.get(method, "")

        cmd_train = [
            os.environ.get("CONDA_EXE", "conda"), "run", "-n", env_name, "--no-capture-output", "bash", "-c",
            f"cd {PROJECT_ROOT} && "
            f"{env_prefix}"
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
            f"export TF_REPO=repo/transcriptformer && "
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
    """Fallback: treat as container-method dispatch (most conda run.py
    scripts accept the same --h5 / --ground-truth / --output-dir CLI)."""
    return _dispatch_container(method, dataset, h5_path, gt_path, out_dir, quick, noop)


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
    "bulkformer": _dispatch_train_predict,
    # ML regressor baselines (P2#13) — conda, linear-style CLI (--config --h5)
    "mlp": _dispatch_linear,
    "xgboost": _dispatch_linear,
    "randomforest": _dispatch_linear,
}


def get_dispatch(method: str):
    if method in DISPATCH_TABLE:
        return DISPATCH_TABLE[method]

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
    # BulkFormer sub-variants (e.g. bulkformer/random, bulkformer/fstat)
    for manifest in sorted((PROJECT_ROOT / "methods" / "bulkformer").glob("*/manifest.yaml")):
        m = f"bulkformer/{manifest.parent.name}"
        status = _get_manifest_field(m, "status")
        if status and status.strip().lower() == "deprecated":
            continue
        methods.append(m)
    return methods



def _extract_gt_from_h5(h5_path, out_dir):
    """Extract ground truth from H5 to CSV. Returns path or None."""
    import h5py, pandas as pd
    with h5py.File(str(h5_path), "r") as f:
        if "bulkRatio/values" in f:
            br = f["bulkRatio/values"][:]
            bt = [x.decode() for x in f["bulkRatio/rownames"][:]]
            bs = [x.decode() for x in f["bulkRatio/colnames"][:]]
            gd = pd.DataFrame(br, index=bs, columns=bt)
        elif "ground_truth/values" in f:
            gt = f["ground_truth/values"][:]
            gr = [x.decode() for x in f["ground_truth/rownames"][:]]
            gc = [x.decode() for x in f["ground_truth/colnames"][:]]
            gd = pd.DataFrame(gt.T, index=gc, columns=gr)
        else:
            return None
    out_dir.mkdir(parents=True, exist_ok=True)
    gd.to_csv(out_dir / "gt.csv")
    return str(out_dir / "gt.csv")


def discover_datasets(subdir: Optional[str] = None) -> list[tuple[str, Path]]:
    """Return [(dataset_name, h5_path), ...] from data/1_pseudo_bulk/."""
    pattern = f"{subdir}/*.h5" if subdir else "**/*.h5"
    datasets = []
    for f in sorted(DATA_DIR.glob(pattern)):
        sub = f.parent.name
        name = f"{sub}_{f.stem}"
        datasets.append((name, f))
    return datasets


# ── Run ─────────────────────────────────────────────────────────────────────

def _localize_h5(h5_path: Path) -> Path:
    """Copy H5 to /tmp to avoid HDF5 locking conflicts on GPFS."""
    local_dir = Path("/tmp") / f"h5cache_{os.getpid()}_{int(time.monotonic() * 1000) % 100000}"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_h5 = local_dir / "input.h5"
    import shutil
    shutil.copy2(str(h5_path), str(local_h5))
    return local_h5


def run_task(method, ds_name, ds_h5, ds_gt, results_dir, quick, noop):
    # ds_gt is None for pseudo (extracted from H5)
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
        gt_path = _extract_gt_from_h5(local_h5, results_dir)  # save at dataset level for post-hoc eval
        dispatch(method, ds_name, local_h5, gt_path, out_dir, quick, noop)
    except Exception as e:
        red(f"  ✗ {method} on {ds_name} — EXCEPTION: {e}")
        traceback.print_exc()
        FAILED += 1
    finally:
        # Clean up local H5
        local_dir = local_h5.parent
        if local_dir.exists():
            import shutil
            shutil.rmtree(str(local_dir), ignore_errors=True)


# ── Post-hoc evaluation ─────────────────────────────────────────────────────

def _posthoc_evaluate(results_dir: Path, gt_csv_or_dir: Path, ds_name: str = None) -> int:
    """Evaluate proportions.csv files under *results_dir* that lack metrics.json.

    Returns number of methods evaluated.
    """
    from scripts.evaluate import evaluate_file

    gt_csv = gt_csv_or_dir
    if not isinstance(gt_csv, Path) or not gt_csv.exists():
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

    parser = argparse.ArgumentParser(description="Run all methods on pseudo-bulk data (results/1_pseudo_bulk)")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--subdir", choices=["cellxgene", "tabula_sapiens"], default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: results/1_pseudo_bulk)")
    args = parser.parse_args()
    global PASSED, FAILED, SKIPPED
    PASSED = FAILED = SKIPPED = 0

    results_dir = RESULTS_BASE
    if args.output:
        results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_methods = discover_methods()
    all_datasets = discover_datasets(subdir=args.subdir)

    if args.methods:
        all_methods = [m for m in all_methods if m in args.methods]
    if args.datasets:
        all_datasets = [(n, h) for n, h in all_datasets if n in args.datasets]

    print(f"\n{'=' * 60}")
    print(f"  Pseudo-Bulk Evaluation (data/1_pseudo_bulk)")
    print(f"  Methods: {len(all_methods)}")
    print(f"  Datasets: {len(all_datasets)}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'=' * 60}\n")

    for ds_name, ds_h5 in all_datasets:
        try:
            rel = ds_h5.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = ds_h5
        print(f"  Dataset: {ds_name}  ({rel})")

    n_parallel = args.parallel if args.parallel > 0 else auto_parallelism()
    if args.dry_run:
        n_parallel = 1

    # Pre-extract GT for all datasets (ensures gt.csv exists at dataset level
    # even if no method runs, and avoids races from lazy extraction in run_task)
    print(f"\n  Pre-extracting ground truth for {len(all_datasets)} datasets...")
    for ds_name, ds_h5 in all_datasets:
        try:
            local_h5 = _localize_h5(ds_h5)
            _extract_gt_from_h5(local_h5, results_dir / ds_name)
            shutil.rmtree(str(local_h5.parent), ignore_errors=True)
        except Exception:
            pass  # non-critical; gt.csv will be missing but evaluation handles that
    print()

    print(f"  Parallel workers: {n_parallel}\n")
    print(f"  Total tasks: {len(all_methods) * len(all_datasets)} "
          f"({len(all_methods)} methods × {len(all_datasets)} datasets)")
    print()

    tasks = []
    for ds_name, ds_h5 in all_datasets:
        for method in all_methods:
            tasks.append((method, ds_name, ds_h5))

    # Parallel dispatch via thread pool (safe: each task is an independent subprocess)
    def _run_one(method, ds_name, ds_h5):
        run_task(method, ds_name, ds_h5, None,
                 results_dir / ds_name, args.quick, args.dry_run)

    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        fut_to_task = {}
        for method, ds_name, ds_h5 in tasks:
            fut = pool.submit(_run_one, method, ds_name, ds_h5)
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
        for ds_name, _ in all_datasets:
            ds_dir = results_dir / ds_name
            gt_csv = ds_dir / "gt.csv"
            if not gt_csv.exists():
                continue
            if ds_dir.exists():
                n_evaluated += _posthoc_evaluate(ds_dir, gt_csv)
        if n_evaluated > 0:
            print(f"\n  Post-hoc evaluated: {n_evaluated} methods\n")

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
