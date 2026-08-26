#!/usr/bin/env python3
"""Evaluate all architecture search results with proper path handling.

Usage:
    python scripts/evaluate_arch_results.py [--phase phase2_domain_shift] [--check-only]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_project = _scripts.parent
if str(_project) not in sys.path:
    sys.path.insert(0, str(_project))

from scripts.evaluate import evaluate_file

DATA_DIR = _project / "data" / "2_real_bulk"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="phase2_domain_shift")
    parser.add_argument("--check-only", action="store_true",
                        help="Only list missing metrics.json, don't evaluate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = _project / "results" / "architecture_search" / args.phase
    missing = []
    ok = 0

    for prop_csv in sorted(base.rglob("proportions.csv")):
        metrics_json = prop_csv.parent / "metrics.json"
        parts = prop_csv.relative_to(base).parts
        method, exp, dataset = parts[0], parts[1], parts[2]

        gt_csv = DATA_DIR / f"{dataset}_gt.csv"
        if not gt_csv.exists():
            print(f"SKIP {prop_csv.parent}: no GT at {gt_csv}")
            continue

        if metrics_json.exists():
            ok += 1
            continue

        missing.append((prop_csv, gt_csv, metrics_json, method, exp, dataset))

    print(f"Evaluated: {ok}, Missing: {len(missing)}")

    if args.check_only:
        if missing:
            print("Missing evaluations:")
            for prop_csv, _, _, method, exp, dataset in missing:
                print(f"  {method:20s} {exp:35s} {dataset:12s} {prop_csv.parent}")
        return

    if args.dry_run:
        print("Would evaluate:")
        for prop_csv, _, _, method, exp, dataset in missing:
            print(f"  {method:20s} {exp:35s} {dataset:12s} -> metrics.json")
        return

    for prop_csv, gt_csv, metrics_json, method, exp, dataset in missing:
        print(f"\n{'='*60}")
        print(f"  {method:20s} {exp:35s} {dataset:12s} ({prop_csv.parent.name})")
        try:
            evaluate_file(str(prop_csv), str(gt_csv), str(metrics_json))
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
