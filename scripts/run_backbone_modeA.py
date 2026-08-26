#!/usr/bin/env python3
"""Run Mode A (frozen backbone → pseudo-bulk train → full predict) on Hao datasets.

Usage:  python scripts/run_backbone_modeA.py --gpu 5 --methods scgpt --datasets altman_Hao
"""
import argparse, os, subprocess, sys, time, yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONDA = os.environ.get("CONDA_EXE", "conda")
DATA_DIR = ROOT / "data" / "2_real_bulk"
RESULTS_BASE = ROOT / "results" / "2_realbulk"

BLUE = "\033[36m"; GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"; RESET = "\033[0m"

METHOD_CONFIGS = {
    "scgpt":           ("bulkgpt",        "frozen"),
    "geneformer":      ("geneformer",     "default"),
    "stack":           ("stack",          "default"),
    "transcriptformer":("TranscriptFormer","default"),
    "scfoundation":    ("scfoundation",   "default"),
}

def _deep_merge(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_merge(d[k], v)
        else:
            d[k] = v

def _write_config(config_dir, base_cfg, overrides):
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(base_cfg)
    _deep_merge(cfg, overrides)
    out_path = config_dir / "config.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return out_path

def run(cmd, log_path, timeout=28800):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"# {' '.join(str(x) for x in cmd)}\n")
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ})
    with open(log_path, "a") as f:
        f.write(f"# Elapsed: {time.monotonic()-t0:.0f}s\n"
                f"# RC: {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return proc.returncode == 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+",
                   default=["altman_Hao","finotello_Hao","hoek_Hao",
                            "hoek_purified_Hao","linsley_purified_Hao","morandini_Hao"])
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    for method in args.methods:
        if method not in METHOD_CONFIGS:
            print(f"{RED}Unknown method: {method}{RESET}")
            continue
        env_name, cfg_name = METHOD_CONFIGS[method]
        cfg_path = ROOT / "methods" / method / "configs" / f"{cfg_name}.yaml"
        if not cfg_path.exists():
            print(f"{RED}Config not found: {cfg_path}{RESET}")
            continue

        with open(cfg_path) as f:
            base_cfg = yaml.safe_load(f) or {}

        # Skip non-dict dataset/data configs (e.g. "dataset: sdy67" string)
        if isinstance(base_cfg.get("dataset"), str) or isinstance(base_cfg.get("data"), str):
            print(f"{YELLOW}Skipping {method}/{cfg_name} (string dataset config){RESET}")
            continue

        for ds_name in args.datasets:
            h5 = DATA_DIR / f"{ds_name}.h5"
            gt = DATA_DIR / f"{ds_name}_gt.csv"
            if not h5.exists():
                print(f"{RED}{ds_name}: H5 not found{RESET}")
                continue

            out_dir = RESULTS_BASE / ds_name / f"{method}_{cfg_name}"
            ckpt_dir = out_dir / "checkpoints"
            log_train = out_dir / "train.log"
            log_pred = out_dir / "predict.log"
            prop_csv = out_dir / "proportions.csv"

            if prop_csv.exists():
                print(f"{GREEN}✓ {method}/{cfg_name} on {ds_name} already done{RESET}")
                continue

            print(f"\n{BLUE}=== {method}/{cfg_name} on {ds_name} ==={RESET}")
            print(f"    GPU: {args.gpu}")

            # ── Build config with dataset paths merged ──
            cfg_overrides = {
                "h5_path": str(h5),
                "gt_path": str(gt),
                "output_dir": str(out_dir),
                "results_dir": str(out_dir),
            }
            paths = dict(base_cfg.get("paths", {}))
            paths["checkpoint_dir"] = str(ckpt_dir)
            paths["output_dir"] = str(out_dir)
            cfg_overrides["paths"] = paths

            ds_cfg = dict(base_cfg.get("dataset", {}))
            ds_cfg["data_path"] = str(h5)
            cfg_overrides["dataset"] = ds_cfg

            data_cfg = dict(base_cfg.get("data", {}))
            data_cfg["data_path"] = str(h5)
            cfg_overrides["data"] = data_cfg

            config_path = _write_config(out_dir / "configs", base_cfg, cfg_overrides)

            # ── Train ──
            train_script = ROOT / "methods" / method / "train.py"
            cmd_train = [
                CONDA, "run", "-n", env_name, "--no-capture-output", "bash", "-c",
                f"cd {ROOT} && "
                f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && "
                f"export CUDA_VISIBLE_DEVICES={args.gpu} && "
                f"export HDF5_USE_FILE_LOCKING=FALSE && "
                f"python {train_script} --config {config_path} --log_file {log_train}",
            ]
            t0 = time.time()
            ok = run(cmd_train, log_train)
            t1 = time.time()
            if not ok:
                print(f"{RED}  ✗ Training failed ({t1-t0:.0f}s){RESET}")
                continue
            print(f"{GREEN}  ✓ Training done ({t1-t0:.0f}s){RESET}")

            # ── Find best checkpoint ──
            best = None
            for name in ["best_model.pt", "deconv_head.pt", "final_model.pt"]:
                for base_dir in [ckpt_dir, out_dir / "checkpoint", out_dir]:
                    cand = base_dir / name
                    if cand.exists():
                        best = cand
                        break
                if best:
                    break
            if not best:
                print(f"{RED}  ✗ No checkpoint found{RESET}")
                continue

            # ── Predict ──
            predict_script = ROOT / "methods" / method / "predict.py"
            cmd_pred = [
                CONDA, "run", "-n", env_name, "--no-capture-output", "bash", "-c",
                f"cd {ROOT} && "
                f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && "
                f"export CUDA_VISIBLE_DEVICES={args.gpu} && "
                f"export HDF5_USE_FILE_LOCKING=FALSE && "
                f"python {predict_script} "
                f"--config {config_path} --checkpoint {best} --log_file {log_pred}",
            ]
            t2 = time.time()
            ok = run(cmd_pred, log_pred)
            t3 = time.time()
            if not ok:
                print(f"{RED}  ✗ Prediction failed ({t3-t2:.0f}s){RESET}")
                continue
            print(f"{GREEN}  ✓ {method}/{cfg_name} on {ds_name} done "
                  f"(train {t1-t0:.0f}s + predict {t3-t2:.0f}s){RESET}")

            # ── Post-hoc evaluate ──
            from scripts.evaluate import evaluate_file
            try:
                evaluate_file(str(prop_csv), str(gt), str(out_dir / "metrics.json"))
            except Exception as e:
                print(f"  [eval] {e}")

    print(f"\n{GREEN}Done{RESET}")

if __name__ == "__main__":
    main()
