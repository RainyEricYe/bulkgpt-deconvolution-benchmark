#!/usr/bin/env python3
"""Fix rownames/colnames swap in singleCellExpr for Hao H5 datasets."""
import h5py, numpy as np, os, shutil

DATASETS = ["altman_Hao","finotello_Hao","hoek_Hao","hoek_purified_Hao","linsley_purified_Hao","morandini_Hao"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "2_real_bulk")

for ds in DATASETS:
    src = os.path.join(DATA_DIR, f"{ds}.h5")
    tmp = src + ".tmp"
    print(f"\n=== {ds} ===")

    with h5py.File(src, "r") as f:
        sc = f["singleCellExpr"]
        rn_len = len(sc["rownames"])
        n_genes = sc["values"].shape[1]
        if rn_len == n_genes:
            print(f"  Already fixed (rownames={rn_len}, n_genes={n_genes}) — skip")
            continue

        print(f"  Fixing: rownames={rn_len}, colnames={len(sc['colnames'])}, n_genes={n_genes}")
        # Read all groups
        data = {
            "bulk": {"values": f["bulk/values"][:], "rownames": f["bulk/rownames"][:], "colnames": f["bulk/colnames"][:]},
            "singleCellExpr": {"values": f["singleCellExpr/values"][:], "rownames": f["singleCellExpr/rownames"][:], "colnames": f["singleCellExpr/colnames"][:]},
            "singleCellLabels": {"values": f["singleCellLabels/values"][:]},
            "ground_truth": {"values": f["ground_truth/values"][:], "rownames": f["ground_truth/rownames"][:], "colnames": f["ground_truth/colnames"][:]},
        }

    # Write fixed
    with h5py.File(tmp, "w") as f:
        for grp in ["bulk", "ground_truth"]:
            g = f.create_group(grp)
            for k, v in data[grp].items():
                g.create_dataset(k, data=v, compression="gzip" if k == "values" else None)
        g = f.create_group("singleCellExpr")
        g.create_dataset("values", data=data["singleCellExpr"]["values"], compression="gzip")
        g.create_dataset("rownames", data=data["singleCellExpr"]["colnames"])   # swapped to correct
        g.create_dataset("colnames", data=data["singleCellExpr"]["rownames"])   # swapped to correct
        g = f.create_group("singleCellLabels")
        g.create_dataset("values", data=data["singleCellLabels"]["values"])

    os.replace(tmp, src)
    # Verify
    with h5py.File(src, "r") as f:
        sc = f["singleCellExpr"]
        ok = len(sc["rownames"]) == sc["values"].shape[1]
        common = len(set(x.decode() for x in sc["rownames"]) & set(x.decode() for x in f["bulk/rownames"]))
        print(f"  Verify: rownames={len(sc['rownames'])} ok={ok} common={common}")

print("\nAll done")
