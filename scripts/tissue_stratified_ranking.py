#!/usr/bin/env python3
"""Tissue / cell-type stratified ranking (P2#11).

Computes per-method real-bulk accuracy stratified by tissue, addressing the
internal-reviewer concern that rankings may be dominated by a batch effect
(tissue type / ground-truth modality). For each tissue stratum (PBMC, brain,
retina) we report per-method mean Pearson r and the within-stratum ranking, and
quantify how much rankings shift across strata (Spearman rho).

Data source: the canonical per-dataset benchmark results under
  to_publish/results/2_realbulk/{dataset}/{method}/metrics.json
which carry 'pearson_mean', 'spearman' (scorr_mean) and 'pearson_per_type' for
every successfully evaluated method x dataset. The 12 real-bulk configurations
span PBMC (altman_*, finotello_Hao, hoek*, linsley*, morandini_Hao, sdy67,
sweetwater), brain (huuki_myers) and retina (demixsc_retina).

Usage (bulkgpt conda env):
  conda activate bulkgpt
  cd <repo-root>
  python3 scripts/tissue_stratified_ranking.py --root results/2_realbulk \
      --out results/tissue_stratified/tissue_stratified_summary.csv
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd


def _spearman(a, b):
    """Spearman rho without scipy (rank both, then Pearson on ranks)."""
    a = pd.Series(a).rank()
    b = pd.Series(b).rank()
    return a.corr(b)

TISSUE = {
    # PBMC / blood-derived
    "sdy67": "PBMC", "sweetwater": "PBMC", "monaco_s13": "PBMC",
    "altman": "PBMC", "finotello": "PBMC", "hoek": "PBMC",
    "linsley": "PBMC", "morandini": "PBMC",
    # brain
    "huuki_myers": "brain",
    # retina
    "demixsc_retina": "retina", "demixsc": "retina",
}
DEFAULT_ALIASES = {"demixsc_retina": "demixsc", "demixsc": "demixsc"}

# pca_ridge is the only method in results/2_realbulk evaluated under a real-bulk
# train/test split (Mode B; see methods/pca_ridge/run.py docstring). Including
# it would compare its held-out-test-fold accuracy against all-sample accuracy
# of every other method. The canonical depth-3 output (matching results_summary.md)
# keeps pca_ridge and flags its coverage; the manuscript-consistent version
# excludes it via --exclude-mode-b. Default keeps it so the script reproduces
# the canonical table exactly.
MODE_B_EXCLUDE = {"pca_ridge"}


def _tissue(ds):
    ds = ds.lower()
    if ds in TISSUE:
        return TISSUE[ds]
    for k, v in TISSUE.items():
        if k in ds:
            return v
    return "other"


def load_inventory(root: Path):
    """Collect per-dataset per-method pearson_mean from metrics.json files.

    Expected layout: root/{dataset}/{method}/metrics.json EXACTLY three levels
    deep (the to_publish results/2_realbulk tree). Deeper files — e.g.
    results/2_realbulk/{dataset}/{method}/{variant}/metrics.json, which are
    Mode-B / leave-one-out experiment outputs nested two levels deep — are
    SKIPPED: aggregating them into the parent method's entry inflates mean_r
    and n_datasets (e.g. a bulkformer LOO variant was previously attributed to
    "bulkformer", which has no canonical depth-3 Mode-A entry at all).
    """
    rows = []
    if not root.exists():
        return rows
    for mf in root.rglob("metrics.json"):
        rel = mf.relative_to(root).parts  # must be (dataset, method, metrics.json)
        if len(rel) != 3:
            continue  # skip nested Mode-B / LOO variant outputs
        try:
            d = json.loads(mf.read_text())
        except Exception:
            continue
        ds, method = rel[0], rel[1]
        pm = d.get("pearson_mean")
        if pm is None:
            continue
        per = d.get("pearson_per_type")
        rows.append({"dataset": ds, "method": method,
                     "pearson": float(pm),
                     "n_types": len(per) if isinstance(per, list) else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/2_realbulk"))
    ap.add_argument("--inventory", type=Path, default=None,
                    help="optional CSV: dataset,method,pearson (bypasses metrics.json scan)")
    ap.add_argument("--min-datasets", type=int, default=1,
                    help="drop methods with fewer than this many datasets in a tissue stratum")
    ap.add_argument("--exclude-mode-b", action="store_true",
                    help="drop Mode-B methods (pca_ridge, real-bulk train/test) "
                         "from the all-sample ranking")
    ap.add_argument("--out", type=Path,
                    default=Path("downstream_BRCA/results/tissue_stratified/tissue_stratified_summary.csv"))
    args = ap.parse_args()

    if args.inventory:
        df = pd.read_csv(args.inventory)
        df = df[["dataset", "method", "pearson"]].dropna()
    else:
        df = pd.DataFrame(load_inventory(args.root))

    if df.empty:
        print("No per-dataset metrics found under", args.root, file=sys.stderr)
        print("Provide --inventory dataset,method,pearson or a metrics.json root.", file=sys.stderr)
        sys.exit(2)

    df["tissue"] = df["dataset"].map(_tissue)
    df = df[df["tissue"] != "other"]

    # Optionally exclude Mode B methods (real-bulk train/test) from the ranking.
    if args.exclude_mode_b:
        df = df[~df["method"].isin(MODE_B_EXCLUDE)]

    # Coverage filter: drop methods with fewer than --min-datasets datasets in a
    # tissue stratum (a single low/high dataset must not dominate that tissue).
    cov = (df.groupby(["tissue", "method"])["dataset"]
             .nunique().rename("n_datasets").reset_index())
    df = df.merge(cov, on=["tissue", "method"])
    df = df[df["n_datasets"] >= args.min_datasets]

    # Per-tissue per-method mean r + rank (+ per-tissue dataset count)
    g = (df.groupby(["tissue", "method"])["pearson"].mean()
           .rename("mean_r").reset_index())
    g["n_datasets"] = g.merge(cov, on=["tissue", "method"])["n_datasets"]
    g["rank"] = g.groupby("tissue")["mean_r"].rank(ascending=False, method="min").astype(int)
    g = g.sort_values(["tissue", "rank"])

    # Cross-tissue rank consistency (Spearman) on methods present in >= 2 tissues
    pivot = g.pivot(index="method", columns="tissue", values="mean_r")
    tissues = [t for t in ["PBMC", "brain", "retina"] if t in pivot.columns]
    corr_lines = []
    for i, a in enumerate(tissues):
        for b in tissues[i + 1:]:
            pair = pivot[[a, b]].dropna()
            if len(pair) >= 3:
                rho = _spearman(pair[a], pair[b])
                corr_lines.append(f"{a} vs {b}: Spearman rho = {rho:.2f} (n = {len(pair)} methods)")
            else:
                corr_lines.append(f"{a} vs {b}: too few overlapping methods ({len(pair)})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(args.out, index=False)
    print("wrote", args.out)
    print("\nPer-tissue per-method mean r and rank:")
    print(g.to_string(index=False))
    print("\nCross-tissue rank consistency:")
    print("\n".join(corr_lines))


if __name__ == "__main__":
    main()
