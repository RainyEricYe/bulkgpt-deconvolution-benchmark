#!/usr/bin/env python3
"""Generate huuki_myers 6-type GT CSV by merging CIRCLE+STAR probes."""
import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
WIDE = HERE.parent / "data" / "processed" / "huuki_myers" / "rnascope_proportions_wide.csv"
OUT = HERE / "data" / "2_real_bulk" / "huuki_myers_gt.csv"

POS_MAP = {"Ant": "A", "Mid": "M", "Post": "P"}
CIRCLE_TYPES = ["Astro", "EndoMural", "Inhib"]
STAR_TYPES = ["Excit", "Micro", "OligoOPC"]
ALL_TYPES = CIRCLE_TYPES + STAR_TYPES

# Load wide-format RNAscope data
wide = pd.read_csv(WIDE)
wide["Tissue"] = wide["SAMPLE_ID"].str.replace("_STAR|_CIRCLE", "", regex=True)

# Track which probes measured each tissue
probe_map = wide.groupby("Tissue")["SAMPLE_ID"].apply(lambda x: x.str.contains("CIRCLE|STAR").tolist()).to_dict()

circle_rows = wide[wide["SAMPLE_ID"].str.contains("CIRCLE")]
star_rows = wide[wide["SAMPLE_ID"].str.contains("STAR")]

# Normalize each probe's measurements to exclude "Other"
def normalize(df, types):
    vals = df[types].copy()
    total = vals.sum(axis=1)
    return vals.div(total, axis=0) if (total > 0).all() else vals.div(total.where(total > 0, np.nan), axis=0)

circ_norm = normalize(circle_rows.set_index("Tissue"), CIRCLE_TYPES)
star_norm = normalize(star_rows.set_index("Tissue"), STAR_TYPES)

# Merge: for tissues missing a probe, keep NaN
combined = circ_norm.join(star_norm, how="outer")

# Build GT CSV rows
rows = []
for batch in ["AN00000904", "AN00000906"]:
    for tissue_code in combined.index:
        props = combined.loc[tissue_code]
        brnum = tissue_code[:-1]
        pos_code = tissue_code[-1]
        pos_full = {v: k for k, v in POS_MAP.items()}[pos_code]
        sample_name = f"{batch}_{brnum}_{pos_full}_Bulk"
        row = {"sample_id": sample_name}
        for ct in ALL_TYPES:
            row[ct] = props[ct]
        rows.append(row)

gt_df = pd.DataFrame(rows)

# Filter to only samples that exist in the original GT
orig = pd.read_csv(HERE / "data" / "2_real_bulk" / "huuki_myers_gt.csv")
valid_samples = set(orig.iloc[:, 0])
gt_df = gt_df[gt_df["sample_id"].isin(valid_samples)]

gt_df.to_csv(OUT, index=False)
print(f"Written {OUT}")
print(f"Shape: {gt_df.shape}")
print(f"Columns: {list(gt_df.columns)}")
print(f"NaN per column:\n{gt_df.isna().sum()}")
print(f"\nSample tissue coverage:")
for t in combined.index:
    has_circ = combined.loc[t, CIRCLE_TYPES].notna().all()
    has_star = combined.loc[t, STAR_TYPES].notna().all()
    print(f"  {t}: Circle={'✓' if has_circ else '✗'} Star={'✓' if has_star else '✗'}")
