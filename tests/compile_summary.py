#!/usr/bin/env python3
"""Compile test results into tests/summary.md (sorted by Pearson ↓)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.metrics import evaluate_deconvolution

OUTPUT_BASE = HERE / "output"
GROUND_TRUTH = HERE / "data" / "ground_truth.csv"


def load_existing_metrics(exp_dir: Path) -> dict | None:
    for fname in ("metrics.json", "eval_results.json"):
        fpath = exp_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                return json.load(f)
    return None


def normalize_metrics(d: dict) -> dict | None:
    """Normalize various metric dict formats to a standard schema."""
    if d.get("status") in ("mock_pass", "fail", "error"):
        return None
    out = {}
    p = d.get("pearson_mean") or d.get("mean_pearson") or d.get("pearson")
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    out["pearson_mean"] = float(p)
    # Map aliases: spearman_mean -> scorr_mean
    scorr = d.get("scorr_mean")
    if scorr is None:
        scorr = d.get("spearman_mean", "N/A")
    out["scorr_mean"] = scorr
    out["ccorr_mean"] = d.get("ccorr_mean", "N/A")
    out["rmse_overall"] = d.get("rmse_overall") or d.get("mean_rmse") or d.get("rmse", "N/A")
    out["mae_overall"] = d.get("mae_overall") or d.get("mean_mae") or d.get("mae", "N/A")
    out["n_samples"] = d.get("n_samples") or d.get("n_samples", "?")
    out["n_types"] = d.get("n_cell_types") or d.get("n_types") or d.get("n_types", "?")
    return out


def _find_common_cell_types(gt_cols, pred) -> tuple | None:
    """Try standard (samples x types) then transposed (types x samples) orientation."""
    common = [c for c in gt_cols if c in pred.columns]
    if len(common) >= 1:
        return common, False  # standard orientation
    common = [c for c in gt_cols if c in pred.index]
    if len(common) >= 1:
        return common, True  # transposed: rows=cell_types, cols=samples
    return None


def recompute_metrics(pred_path: Path) -> dict | None:
    """Recompute metrics from predictions.csv + ground_truth.csv."""
    if not pred_path.exists() or not GROUND_TRUTH.exists():
        return None
    try:
        pred = pd.read_csv(pred_path, index_col=0)
        gt = pd.read_csv(GROUND_TRUTH, index_col=0)

        found = _find_common_cell_types(gt.columns, pred)
        if found is None:
            return None
        common_types, transposed = found

        if transposed:
            # pred is (n_types x n_samples); common_types are in pred.index
            common_idx = gt.index.intersection(pred.columns)
            if len(common_idx) < 2:
                return None
            gt = gt.loc[common_idx, common_types]
            pred = pred.loc[common_types, common_idx].T
        else:
            common_idx = gt.index.intersection(pred.index)
            if len(common_idx) < 2:
                return None
            gt = gt.loc[common_idx, common_types]
            pred = pred.loc[common_idx, common_types]
        n = len(common_idx)
        gt_arr = gt.values.astype(np.float64)
        pred_arr = pred.values.astype(np.float64)
        metrics = evaluate_deconvolution(gt_arr, pred_arr, cell_types=common_types)
        metrics["n_samples"] = n
        metrics["n_types"] = len(common_types)
        return metrics
    except Exception:
        return None


def load_resources(exp_dir: Path) -> dict:
    rpath = exp_dir / "resources.json"
    if not rpath.exists():
        return {}
    try:
        with open(rpath) as f:
            return json.load(f)
    except Exception:
        return {}


def fmt_resources(r: dict) -> dict:
    cpu = r.get("cpu", {})
    gpu = r.get("gpu", {})
    out = {}
    out["elapsed_s"] = round(r.get("elapsed_s", 0), 1)
    out["cpu_time_s"] = round(cpu.get("user_time_s", 0) + cpu.get("system_time_s", 0), 1)
    rss_kb = cpu.get("max_rss_kb", 0)
    out["max_rss_mb"] = round(rss_kb / 1024, 1) if rss_kb else 0
    gpu_mem = 0
    for dev in gpu.get("devices", []):
        gpu_mem += dev.get("memory_used_gb", 0)
    out["gpu_mem_gb"] = round(gpu_mem, 2)
    return out


def fmt(v):
    if v == "N/A" or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    rows = []
    for method_dir in sorted(OUTPUT_BASE.iterdir()):
        if not method_dir.is_dir():
            continue
        for exp_dir in sorted(method_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            pred_csv = exp_dir / "proportions.csv"
            if not pred_csv.exists():
                pred_csv = exp_dir / "predictions.csv"
            if not pred_csv.exists():
                candidates = sorted(exp_dir.glob("*_proportions.csv"))
                if candidates:
                    pred_csv = candidates[0]

            result = None
            source = "N/A"

            # 1) Recompute from predictions + ground truth (full metrics)
            if pred_csv.exists() and GROUND_TRUTH.exists():
                result = recompute_metrics(pred_csv)
                if result is not None:
                    source = "recomputed"

            # 2) Fallback to existing metrics.json / eval_results.json
            if result is None:
                em = load_existing_metrics(exp_dir)
                result = normalize_metrics(em) if em else None
                source = "existing" if result else "N/A"

            if result is None:
                continue

            res = fmt_resources(load_resources(exp_dir))
            rows.append({
                "method": f"{method_dir.name}/{exp_dir.name}",
                "pearson": result["pearson_mean"],
                "scorr": result.get("scorr_mean", "N/A"),
                "ccorr": result.get("ccorr_mean", "N/A"),
                "rmse": result.get("rmse_overall", "N/A"),
                "mae": result.get("mae_overall", "N/A"),
                "n_samples": result.get("n_samples", "?"),
                "n_types": result.get("n_types", "?"),
                "elapsed_s": res.get("elapsed_s", ""),
                "cpu_time_s": res.get("cpu_time_s", ""),
                "max_rss_mb": res.get("max_rss_mb", ""),
                "gpu_mem_gb": res.get("gpu_mem_gb", ""),
            })

    if not rows:
        print("No results found.")
        return

    def sort_key(r):
        p = r["pearson"]
        if p == "N/A" or pd.isna(p):
            return -999.0
        return float(p)

    rows.sort(key=sort_key, reverse=True)

    lines = [
        "# Test Summary",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**{len(rows)} experiments** with predictions and ground truth (sorted by Pearson ↓).",
        "",
        "| Method | Pearson | SCorr | CCorr | RMSE | MAE | Samples | Types | Time(s) | CPU(s) | RSS(MB) | GPU(GB) |",
        "|--------|---------|-------|-------|------|-----|---------|-------|---------|--------|---------|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} "
            f"| {fmt(r['pearson'])} "
            f"| {fmt(r['scorr'])} "
            f"| {fmt(r['ccorr'])} "
            f"| {fmt(r['rmse'])} "
            f"| {fmt(r['mae'])} "
            f"| {r['n_samples']} "
            f"| {r['n_types']} "
            f"| {r['elapsed_s']} "
            f"| {r['cpu_time_s']} "
            f"| {r['max_rss_mb']} "
            f"| {r['gpu_mem_gb']} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Sorted by Pearson correlation coefficient (descending).",
        "- Methods with `(existing)` source read pre-computed metrics from their output.",
        "- Methods with `(recomputed)` source had predictions + ground_truth and were re-evaluated via `core/metrics.py`.",
        "- Resource columns: Time(s) = wall-clock seconds; CPU(s) = user+system CPU seconds; RSS(MB) = peak RSS; GPU(GB) = sum of GPU memory used.",
        "- N/A means the metric is not available (method output did not provide it).",
        "",
    ])

    summary_path = HERE / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Summary written to {summary_path}")
    print(f"{len(rows)} rows written.")


if __name__ == "__main__":
    main()
