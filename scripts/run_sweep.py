#!/usr/bin/env python3
"""Grid search sweep runner.

Expands sweep from experiments/sweep.yaml, skips completed (via
runs_index.json), and launches missing ones.

Usage:
    python scripts/run_sweep.py --list
    python scripts/run_sweep.py --sweep lr_ablation
    python scripts/run_sweep.py --sweep lr_ablation --force
    python scripts/run_sweep.py --sweep lr_ablation --gpus 0,1,2,3
"""

import argparse, itertools, json, logging, os, subprocess, sys, time
from pathlib import Path
import yaml

logger = logging.getLogger("run_sweep")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_PATH = PROJECT_ROOT / "experiments" / "sweep.yaml"
RUNS_INDEX = PROJECT_ROOT / "checkpoints" / "cache" / "runs_index.json"


def load_sweeps() -> dict:
    with open(SWEEP_PATH) as f:
        return yaml.safe_load(f).get("sweeps", {})


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def load_index() -> dict:
    if RUNS_INDEX.exists():
        return json.loads(RUNS_INDEX.read_text())
    return {}


def save_index(idx: dict) -> None:
    RUNS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    RUNS_INDEX.write_text(json.dumps(idx, indent=2))


def exp_id(method: str, dataset: str, params: dict) -> str:
    tag = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
    return f"{method}_{dataset}_{tag}"


def run_one(method: str, dataset: str, config: dict, gpu: str | None = None) -> dict:
    eid = exp_id(method, dataset, config)
    logger.info("[%s] Starting...", eid)
    ckpt = PROJECT_ROOT / "checkpoints" / "cache" / method / dataset / "sweep" / eid
    ckpt.mkdir(parents=True, exist_ok=True)
    rc = {
        "experiment_id": eid, "method": method, "dataset": dataset,
        "hparams": config, "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename, "status": "running",
    }
    (ckpt / "run_config.json").write_text(json.dumps(rc, indent=2))
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [sys.executable,
           str(PROJECT_ROOT / "methods" / method / "run.py"),
           "--config", str(PROJECT_ROOT / "methods" / method / "configs" / "ft.yaml"),
           "--mode", "train",
           "--sc_ref", str(PROJECT_ROOT / "data" / f"{dataset}.h5ad"),
           "--checkpoint_dir", str(ckpt)]
    for k, v in config.items():
        if k == "unfreeze_backbone":
            if v: cmd.append("--unfreeze_backbone")
        elif k == "encoding":
            cmd.append("--binned" if v == "binned" else "--no-binned")
        else:
            cmd.extend([f"--{k}", str(v)])
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start
    status = "completed" if proc.returncode == 0 else "failed"
    rc["status"] = status
    rc["elapsed"] = elapsed
    (ckpt / "run_config.json").write_text(json.dumps(rc, indent=2))
    (ckpt / "sweep_output.log").write_text(
        f"CMD: {' '.join(cmd)}\nRC: {proc.returncode}\nELAPSED: {elapsed:.1f}s\n\n"
        f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n")
    idx = load_index()
    idx[eid] = {"method": method, "dataset": dataset, "hparams": config,
                "status": status, "elapsed": elapsed}
    save_index(idx)
    logger.info("[%s] %s (%.1fs)", eid, status, elapsed)
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep")
    p.add_argument("--list", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--gpus")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s | %(message)s")
    sweeps = load_sweeps()
    if args.list:
        for n, s in sweeps.items():
            print(f"  {n:25s} {s.get('description', '')}")
        return
    if not args.sweep:
        p.print_help(); return
    sw = sweeps.get(args.sweep)
    if not sw:
        logger.error("Unknown sweep %r", args.sweep); sys.exit(1)
    method = sw.get("method", "scgpt")
    datasets = sw.get("datasets", [sw["dataset"]])
    grid_cfgs = expand_grid(sw.get("grid", {}))
    fixed = sw.get("fixed", {})
    gpu_list = args.gpus.split(",") if args.gpus else []
    results = []
    for ds in datasets:
        for params in grid_cfgs:
            cfg = {**fixed, **params}
            eid = exp_id(method, ds, cfg)
            if not args.force:
                ex = load_index().get(eid, {})
                if ex.get("status") == "completed":
                    logger.info("[%s] Already completed", eid); continue
            if args.dry_run:
                print(f"  Would run: {method} x {ds} -> {cfg}"); continue
            gpu = gpu_list[len(results) % len(gpu_list)] if gpu_list else None
            results.append(run_one(method, ds, cfg, gpu))
    logger.info("Done: %d runs", len(results))


if __name__ == "__main__":
    main()
