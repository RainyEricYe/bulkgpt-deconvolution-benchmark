#!/usr/bin/env python3
"""Generate all_metrics_summary.csv from metrics.json files under a results directory.

Scans <results_dir>/<dataset>/<method>[/<variant>]/metrics.json and produces
a unified CSV across datasets.

Usage:
    python scripts/generate_summary.py                                         # default: results/2_realbulk
    python scripts/generate_summary.py --results-dir results/pseudo_bulk
    python scripts/generate_summary.py --output my_summary.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def safe_float(v, default=float("nan")) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def collect_metrics(results_dir: Path) -> list[dict]:
    """Walk results_dir and collect all metrics.json entries.

    Handles both:
      <dataset>/<method>/metrics.json
      <dataset>/<method>/<variant>/metrics.json
    """
    rows: list[dict] = []
    results_dir = results_dir.resolve()

    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name

        for entry in sorted(ds_dir.iterdir()):
            if not entry.is_dir():
                continue

            # Case 1: <dataset>/<method>/metrics.json  (flat)
            m1 = entry / "metrics.json"
            if m1.is_file():
                _append_row(rows, dataset, entry.name, m1)
                continue

            # Case 2: <dataset>/<method>/<variant>/metrics.json  (nested)
            has_nested = False
            for sub in sorted(entry.iterdir()):
                if not sub.is_dir():
                    continue
                m2 = sub / "metrics.json"
                if m2.is_file():
                    method_name = f"{entry.name}/{sub.name}"
                    _append_row(rows, dataset, method_name, m2)
                    has_nested = True
            if not has_nested:
                # Neither flat nor nested — skip silently
                pass

    return rows


def _append_row(rows: list[dict], dataset: str, method: str, path: Path) -> None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return

    row = {
        "dataset": dataset,
        "method": method,
        "pearson": safe_float(data.get("pearson_mean")),
        "mae": safe_float(data.get("mae_overall")),
        "scorr": safe_float(data.get("scorr_mean")),
        "ccorr": safe_float(data.get("ccorr_mean")),
        "rmse": safe_float(data.get("rmse_overall")),
    }
    rows.append(row)


def fmt_val(v: float) -> str:
    if math.isnan(v):
        return ""
    return f"{v:.4f}"


def write_csv(rows: list[dict], output: Path) -> None:
    fieldnames = ["dataset", "method", "pearson", "mae", "scorr", "ccorr", "rmse"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: fmt_val(r[k]) if k not in ("dataset", "method") else r[k]
                   for k in fieldnames}
            w.writerow(out)
    print(f"Written {len(rows)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all_metrics_summary.csv from per-method metrics.json files."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results" / "2_realbulk"),
        help="Results directory (<dataset>/<method>[/<variant>]/metrics.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <results_dir>/all_metrics_summary.csv)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if args.output:
        output = Path(args.output)
    else:
        output = results_dir / "all_metrics_summary.csv"

    rows = collect_metrics(results_dir)
    if not rows:
        print(f"No metrics.json files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Sort: dataset first, then by pearson descending
    rows.sort(key=lambda r: (r["dataset"], -(r["pearson"] if not math.isnan(r["pearson"]) else -999)))

    write_csv(rows, output)


if __name__ == "__main__":
    main()
