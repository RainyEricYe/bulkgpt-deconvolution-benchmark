#!/usr/bin/env python3
"""Phased architecture search for real-bulk deconvolution.

Follows to_publish/plan.md exactly. Generates per-experiment configs from
the architecture grid, dispatches train+predict through each method's
own run.py, and collects results into per-phase result directories.

Usage
-----
    # Phase 0 — smoke test (2 epochs, 200 pseudo-bulk)
    python scripts/run_architecture_search.py --phase phase0_smoke --methods scgpt

    # Phase 1 — coarse screen (3 seeds, full epochs)
    python scripts/run_architecture_search.py --phase phase1_coarse --seeds 42 123 456

    # Phase 2 — domain-shift refinement
    python scripts/run_architecture_search.py --phase phase2_domain_shift \
        --from-results results/architecture_search/phase1_coarse

    # Phase 3 — final validation (5 seeds)
    python scripts/run_architecture_search.py --phase phase3_final_validation --seeds 42 123 456 789 2024

    # Phase 4 — final test (locked config, test split)
    python scripts/run_architecture_search.py --phase phase4_final_test --split test

    # Dry run — print what would be run
    python scripts/run_architecture_search.py --phase phase1_coarse --dry-run

    # Summary only
    python scripts/run_architecture_search.py --phase phase1_coarse --summary-only

Output
------
    results/architecture_search/{phase}/{method}/{experiment}/{seed}/
      ├── config.yaml
      ├── train.log
      ├── predict.log
      ├── proportions.csv
      ├── metrics.json
      └── resources.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_BASE = PROJECT_ROOT / "results" / "architecture_search"
SPLITS_PATH = PROJECT_ROOT / "experiments" / "real_bulk_splits.yaml"
GRID_PATH = PROJECT_ROOT / "experiments" / "architecture_grid.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "2_real_bulk"
TESTS_DATA_DIR = PROJECT_ROOT / "tests" / "data"

# ── Logging ──────────────────────────────────────────────────────────────

BLUE = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def blue(s): print(f"{BLUE}{s}{RESET}")
def green(s): print(f"{GREEN}{s}{RESET}")
def red(s): print(f"{RED}{s}{RESET}")
def yellow(s): print(f"{YELLOW}{s}{RESET}")


PASSED = FAILED = SKIPPED = 0


# ── I/O ──────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into a copy of base, supporting dotted keys."""
    result = {}
    for k, v in base.items():
        if isinstance(v, dict):
            result[k] = dict(v)
        else:
            result[k] = v

    def _set_dotted(d, key, value):
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            elif not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    for key, value in overrides.items():
        if "." in key:
            _set_dotted(result, key, value)
        elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key].update(value)
        else:
            result[key] = value
    return result


# ── Split handling ───────────────────────────────────────────────────────

def load_splits() -> dict:
    data = _load_yaml(SPLITS_PATH)
    return data.get("real_bulk", {})


def get_split_datasets(split: str) -> list[str]:
    splits = load_splits()
    return splits.get(split, [])


# ── Grid handling ────────────────────────────────────────────────────────

def load_grid() -> dict:
    data = _load_yaml(GRID_PATH)
    return data.get("methods", {})


def get_method_experiments(method: str, grid: dict) -> list[tuple[str, dict]]:
    """Return [(experiment_name, overrides), ...] for a method."""
    md = grid.get(method)
    if not md:
        return []
    return [(name, exp.get("overrides", {})) for name, exp in md.get("experiments", {}).items()]


def get_default_config(method: str, grid: dict) -> str:
    md = grid.get(method, {})
    return md.get("default_config", "")


# ── Config generation ────────────────────────────────────────────────────

def make_data_symlink(dataset: str, out_dir: Path) -> Path:
    """Copy or symlink the H5 file to the output directory."""
    h5_src = DATA_DIR / f"{dataset}.h5"
    if not h5_src.exists():
        h5_src = TESTS_DATA_DIR / f"{dataset}.h5"
    if not h5_src.exists():
        raise FileNotFoundError(f"No H5 found for dataset {dataset}")

    h5_dst = out_dir / "input.h5"
    if not h5_dst.exists():
        shutil.copy2(str(h5_src), str(h5_dst))
    return h5_dst


def generate_config(
    method: str, experiment_name: str, dataset: str,
    seed: int, grid: dict, phase: str, out_dir: Path,
) -> Path:
    """Generate a config.yaml with all overrides applied.

    The generated config includes the proper data_path, checkpoint_dir,
    output_dir, and seed for this specific run.
    """
    default_cfg_rel = get_default_config(method, grid)
    default_cfg_path = PROJECT_ROOT / default_cfg_rel
    if default_cfg_path.exists():
        base_cfg = _load_yaml(default_cfg_path)
    else:
        base_cfg = {}

    # Get experiment overrides
    experiments = grid.get(method, {}).get("experiments", {})
    exp_data = experiments.get(experiment_name, {})
    overrides = exp_data.get("overrides", {}).copy()

    # Set data path, output paths, seed
    data_path = str(out_dir / "input.h5")
    overrides["seed"] = seed

    # Handle data_path based on method convention
    # stack/transcriptformer use config["data"]["data_path"]
    # scgpt/geneformer/scfoundation use config["dataset"]["data_path"]
    data_path_key = "data.data_path" if method in ("stack", "transcriptformer") else "dataset.data_path"
    overrides[data_path_key] = data_path
    overrides["paths.checkpoint_dir"] = str(out_dir)
    overrides["paths.output_dir"] = str(out_dir)

    # Additional phase-specific overrides
    if phase == "phase0_smoke":
        overrides["training.epochs"] = 2
        overrides["training.n_pseudo_bulk"] = 200

    # Auto-set real_bulk_path for domain adaptation experiments
    if "training.da_method" in overrides or "da_method" in overrides:
        rb_key = "data.real_bulk_path" if method in ("stack", "transcriptformer") else "dataset.real_bulk_path"
        overrides[rb_key] = data_path

    merged = _deep_merge(base_cfg, overrides)

    config_dir = out_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"

    import yaml
    with open(config_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False)

    return config_path


# ── Method runner ────────────────────────────────────────────────────────

def _find_manifest_method(method: str) -> dict | None:
    """Read manifest.yaml for a method."""
    import yaml
    mp = PROJECT_ROOT / "methods" / method / "manifest.yaml"
    if not mp.exists():
        return None
    with open(mp) as f:
        return yaml.safe_load(f)


def _find_checkpoint(out_dir: Path) -> Path | None:
    """Find best_model.pt or final_model.pt under out_dir."""
    for ckpt_name in ["best_model.pt", "final_model.pt", "deconv_head.pt"]:
        for base in [out_dir / "checkpoints", out_dir / "checkpoint", out_dir]:
            cand = base / ckpt_name
            if cand.exists():
                return cand
    return None


def run_experiment(
    method: str, experiment_name: str, dataset: str,
    seed: int, config_path: Path, out_dir: Path,
    timeout: int = 14400,
) -> bool:
    """Run train + predict for a single experiment.

    Dispatches to the method's own train.py and predict.py via conda.
    """
    global PASSED, FAILED

    manifest = _find_manifest_method(method)
    if not manifest:
        red(f"  ? {method}: no manifest.yaml")
        FAILED += 1
        return False

    env_name = manifest.get("conda_env", "bulkgpt")
    entry_train = manifest.get("entry", {}).get("train", "")
    entry_predict = manifest.get("entry", {}).get("predict", "")

    if not entry_train:
        red(f"  ? {method}: no train entry in manifest")
        FAILED += 1
        return False

    train_script = PROJECT_ROOT / entry_train
    if not train_script.exists():
        red(f"  ? {method}: train script not found: {train_script}")
        FAILED += 1
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path_train = out_dir / "train.log"
    # Use direct python path (conda run -n can hang: https://github.com/conda/conda/issues/11947)
    _conda_base = os.environ.get("CONDA_PREFIX", os.environ.get("CONDA_EXE", ""))
    python_bin = f"{_conda_base}/envs/{env_name}/bin/python"

    # ── Train ──
    start = time.monotonic()
    cmd_train = [
        "bash", "-c",
        f"cd {PROJECT_ROOT} && "
        f"export LD_LIBRARY_PATH={_conda_base}/envs/{env_name}/lib:$LD_LIBRARY_PATH && "
        f"{python_bin} {train_script} --config {config_path} --log_file {log_path_train}",
    ]
    try:
        proc = subprocess.run(cmd_train, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - start
        train_ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        train_ok = False
        proc = None
        red(f"  ✗ {method}/{experiment_name}/{dataset}/seed{seed} — TRAIN TIMEOUT ({elapsed:.0f}s)")

    if not train_ok:
        FAILED += 1
        if proc:
            with open(out_dir / "train_error.log", "w") as f:
                f.write(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        return False

    # ── Predict ──
    if not entry_predict:
        green(f"  ✓ {method}/{experiment_name}/{dataset}/seed{seed} (train only, {elapsed:.0f}s)")
        PASSED += 1
        return True

    predict_script = PROJECT_ROOT / entry_predict
    if not predict_script.exists():
        yellow(f"  ~ {method}: predict script not found: {predict_script}, skipping predict")
        green(f"  ✓ {method}/{experiment_name}/{dataset}/seed{seed} (train only, {elapsed:.0f}s)")
        PASSED += 1
        return True

    best_ckpt = _find_checkpoint(out_dir)

    if not best_ckpt:
        yellow(f"  ~ {method}/{experiment_name}: no checkpoint found, skipping predict")
        green(f"  ✓ {method}/{experiment_name}/{dataset}/seed{seed} (train, {elapsed:.0f}s, no predict)")
        PASSED += 1
        return True

    log_path_predict = out_dir / "predict.log"
    cmd_predict = [
        "bash", "-c",
        f"cd {PROJECT_ROOT} && "
        f"export LD_LIBRARY_PATH={_conda_base}/envs/{env_name}/lib:$LD_LIBRARY_PATH && "
        f"{python_bin} {predict_script} --config {config_path} --checkpoint {best_ckpt} --log_file {log_path_predict}",
    ]
    try:
        proc = subprocess.run(cmd_predict, capture_output=True, text=True, timeout=timeout)
        pred_ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        pred_ok = False
        proc = None
        red(f"  ✗ {method}/{experiment_name}/{dataset}/seed{seed} — PREDICT TIMEOUT")

    if pred_ok:
        PASSED += 1
        green(f"  ✓ {method}/{experiment_name}/{dataset}/seed{seed} ({elapsed:.0f}s)")
        return True
    else:
        FAILED += 1
        if proc:
            with open(out_dir / "predict_error.log", "w") as f:
                f.write(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        red(f"  ✗ {method}/{experiment_name}/{dataset}/seed{seed} — PREDICT FAILED")
        return False


# ── Post-hoc evaluation ──────────────────────────────────────────────────

def _posthoc_evaluate(phase_dir: Path) -> int:
    """Run evaluate.py on any proportions.csv missing metrics.json."""
    from scripts.evaluate import evaluate_file

    data_dir = PROJECT_ROOT / "data" / "2_real_bulk"
    n = 0
    for prop_csv in sorted(phase_dir.rglob("proportions.csv")):
        metrics_json = prop_csv.parent / "metrics.json"
        if metrics_json.exists():
            continue
        # infer dataset from the grandparent dir: .../{method}/{exp}/{dataset}/seed{seed}/proportions.csv
        dataset = prop_csv.parent.parent.name
        gt_csv = data_dir / f"{dataset}_gt.csv"
        if not gt_csv.exists():
            continue
        try:
            evaluate_file(str(prop_csv), str(gt_csv), str(metrics_json))
            n += 1
        except Exception as e:
            print(f"  [eval] {prop_csv.parent.name}: {e}")
    return n


# ── Phase runners ────────────────────────────────────────────────────────

def _adjust_overrides_for_phase(overrides: dict, phase: str, seed: int) -> dict:
    """Apply phase-specific adjustments to experiment overrides."""
    adj = dict(overrides)

    if phase == "phase0_smoke":
        adj["training.epochs"] = min(adj.get("training.epochs", 30), 3)
        adj["training.n_pseudo_bulk"] = min(adj.get("training.n_pseudo_bulk", 5000), 200)

    # Ensure seed is set
    adj["training.seed"] = seed
    adj["seed"] = seed

    return adj


def run_phase(
    phase: str,
    split: str,
    methods: list[str] | None = None,
    experiments: list[str] | None = None,
    seeds: list[int] | None = None,
    dry_run: bool = False,
    max_workers: int = 1,
) -> int:
    """Run all experiments for a phase."""
    global PASSED, FAILED, SKIPPED

    datasets = get_split_datasets(split)
    grid = load_grid()
    if not methods:
        methods = list(grid.keys())

    phase_dir = RESULTS_BASE / phase

    if not seeds:
        seeds = [42]

    tasks = []
    for method in methods:
        exps = get_method_experiments(method, grid)
        if not exps:
            yellow(f"  ? {method}: no experiments defined in grid, skipping")
            SKIPPED += 1
            continue

        for exp_name, exp_overrides in exps:
            if experiments and exp_name not in experiments:
                continue
            for dataset in datasets:
                for seed in seeds:
                    tasks.append((method, exp_name, exp_overrides, dataset, seed))

    blue(f"\n{'=' * 60}")
    blue(f"  Phase: {phase}")
    blue(f"  Split: {split} ({', '.join(datasets)})")
    blue(f"  Methods: {', '.join(methods)}")
    blue(f"  Seeds: {seeds}")
    blue(f"  Total runs: {len(tasks)}")
    blue(f"  Dry run: {dry_run}")
    blue(f"{'=' * 60}\n")

    if dry_run:
        for method, exp_name, _, dataset, seed in tasks:
            print(f"  Would run: {method}/{exp_name}/{dataset}/seed={seed}")
        return 0

    def _run_one(method, exp_name, exp_overrides, dataset, seed):
        global PASSED, FAILED, SKIPPED
        out_dir = phase_dir / method / exp_name / dataset / f"seed{seed}"
        if (out_dir / "proportions.csv").exists() and (out_dir / "metrics.json").exists():
            green(f"  ✓ {method}/{exp_name}/{dataset}/seed{seed} — already done, skipping")
            PASSED += 1
            return

        try:
            # Localize H5
            local_dir = Path("/tmp") / f"archcache_{os.getpid()}_{int(time.monotonic() * 1000) % 100000}"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_h5 = make_data_symlink(dataset, local_dir)
        except FileNotFoundError as e:
            red(f"  ✗ {method}/{exp_name}/{dataset}/seed{seed} — {e}")
            FAILED += 1
            return

        try:
            adj_overrides = _adjust_overrides_for_phase(exp_overrides, phase, seed)
            config_path = generate_config(method, exp_name, dataset, seed, grid, phase, out_dir)

            # Copy the localized H5 to out_dir for the config to reference
            shutil.copy2(str(local_h5), str(out_dir / "input.h5"))

            run_experiment(method, exp_name, dataset, seed, config_path, out_dir)
        except Exception as e:
            red(f"  ✗ {method}/{exp_name}/{dataset}/seed{seed} — EXCEPTION: {e}")
            traceback.print_exc()
            FAILED += 1
        finally:
            shutil.rmtree(str(local_dir), ignore_errors=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_task = {}
        for method, exp_name, exp_overrides, dataset, seed in tasks:
            fut = pool.submit(_run_one, method, exp_name, exp_overrides, dataset, seed)
            fut_to_task[fut] = (method, exp_name, dataset, seed)

        for fut in as_completed(fut_to_task):
            try:
                fut.result()
            except Exception as e:
                method, exp_name, dataset, seed = fut_to_task[fut]
                red(f"  ✗ {method}/{exp_name}/{dataset}/seed{seed} — UNHANDLED: {e}")
                FAILED += 1

    blue(f"\n{'=' * 60}")
    blue(f"  Phase {phase} complete")
    blue(f"  Passed: {PASSED}  Failed: {FAILED}  Skipped: {SKIPPED}")
    blue(f"{'=' * 60}\n")

    return FAILED


# ── Summary ──────────────────────────────────────────────────────────────

def write_summary(phase: str):
    phase_dir = RESULTS_BASE / phase
    if not phase_dir.exists():
        print(f"No results found for phase {phase}")
        return

    summary_path = phase_dir / "summary.md"
    lines = [
        f"# Architecture Search — {phase}",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for method_dir in sorted(phase_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        lines.append(f"## {method}")
        lines.append("")
        lines.append("| Experiment | Dataset | Seed | Pearson (mean) | MAE (mean) | RMSE (mean) | Status |")
        lines.append("|------------|---------|------|----------------|------------|-------------|--------|")

        for exp_dir in sorted(method_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_name = exp_dir.name
            for ds_dir in sorted(exp_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                dataset = ds_dir.name
                for seed_dir in sorted(ds_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    seed = seed_dir.name.replace("seed", "")
                    metrics_file = seed_dir / "metrics.json"
                    prop_file = seed_dir / "proportions.csv"

                    status = "PASS" if prop_file.exists() else "FAIL"
                    pearson = "?"
                    mae = "?"
                    rmse = "?"
                    if metrics_file.exists():
                        try:
                            m = json.loads(metrics_file.read_text())
                            pearson = f"{m.get('pearson_mean', m.get('mean_pearson', m.get('pearson', '?'))):.4f}"
                            mae_val = m.get('mae_overall', m.get('mae_mean', m.get('mean_mae', m.get('mae', None))))
                            rmse_val = m.get('rmse_mean_per_type', m.get('rmse_mean', m.get('mean_rmse', m.get('rmse', None))))
                            if mae_val is not None:
                                mae = f"{mae_val:.4f}"
                            if rmse_val is not None:
                                rmse = f"{rmse_val:.4f}"
                        except Exception:
                            pass
                    lines.append(f"| {exp_name} | {dataset} | {seed} | {pearson} | {mae} | {rmse} | {status} |")

        lines.append("")

    summary_path.write_text("\n".join(lines) + "\n")
    blue(f"Summary written to {summary_path}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Architecture search runner")
    parser.add_argument("--phase", required=True,
                        choices=["phase0_smoke", "phase1_coarse", "phase2_domain_shift",
                                 "phase3_final_validation", "phase4_final_test"],
                        help="Search phase to run")
    parser.add_argument("--split", default="validation",
                        choices=["validation", "test", "all"],
                        help="Data split to use")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run (default: all from grid)")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Experiment names to run (default: all from grid)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds (default: [42] for smoke, [42,123,456] for coarse)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Max parallel workers (default: 1 for GPU safety)")
    parser.add_argument("--from-results", default=None,
                        help="Previous phase results directory (for Phase 2+)")
    args = parser.parse_args()

    # Default seeds per phase
    seeds = args.seeds
    if seeds is None:
        if args.phase == "phase0_smoke":
            seeds = [42]
        elif args.phase in ("phase3_final_validation", "phase4_final_test"):
            seeds = [42, 123, 456, 789, 2024]
        else:
            seeds = [42, 123, 456]

    # Phase 4 must use test split
    split = args.split
    if args.phase == "phase4_final_test":
        split = "test"
    elif args.phase == "phase2_domain_shift" and not args.from_results:
        yellow("  Warning: Phase 2 usually requires --from-results")

    if args.summary_only:
        write_summary(args.phase)
        return 0

    run_phase(
        phase=args.phase,
        split=split,
        methods=args.methods,
        experiments=args.experiments,
        seeds=seeds,
        dry_run=args.dry_run,
        max_workers=args.parallel,
    )

    if not args.dry_run:
        phase_dir = RESULTS_BASE / args.phase
        n_eval = _posthoc_evaluate(phase_dir)
        if n_eval:
            blue(f"  Evaluated {n_eval} results")
        write_summary(args.phase)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
