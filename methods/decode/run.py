#!/usr/bin/env python3
"""
DECODE (MBdeconv) — Bulk RNA-seq deconvolution via expression reconstruction.

Trains MBdeconv on pseudo-bulk generated from scRNA-seq reference, then
predicts cell-type proportions on bulk RNA-seq samples.

Supports cross-reference prediction: train on one scRNA-seq dataset,
predict on a different bulk cohort (e.g., Pal2021 -> TCGA-BRCA).

Usage
-----
    python run.py --config configs/default.yaml --mode all
    python run.py --config configs/default.yaml --mode train
    python run.py --config configs/default.yaml --mode predict
"""
import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", HERE.parent.parent)).resolve()

# DECODE source
DECODE_SRC = HERE / "src"
sys.path.insert(0, str(DECODE_SRC))

from data.data_process import data_process
from model.deconv_model import MBdeconv
from model.utils import TrainCustomDataset, TestCustomDataset, predict

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import auto_detect_celltype_col


def parse_args():
    p = argparse.ArgumentParser(description="DECODE (MBdeconv) — Bulk RNA-seq deconvolution")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    p.add_argument("--mode", type=str, default="all", choices=["train", "predict", "all"],
                   help="Execution mode")
    # CLI overrides
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--gpu", action="store_true", default=None)
    p.add_argument("--sc-ref", type=str, default=None)
    p.add_argument("--bulk", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    for opt, key in [("epochs", "epochs"), ("batch_size", "batch_size"),
                     ("gpu", "gpu")]:
        val = getattr(args, opt, None)
        if val is not None:
            cfg["training"][key] = val
    for opt, key in [("sc_ref", "sc_ref"), ("bulk", "bulk")]:
        val = getattr(args, opt, None)
        if val is not None:
            cfg["data"][key] = val
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = args.output_dir

    # Resolve paths
    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    sc_ref_path = _resolve(cfg["data"]["sc_ref"])
    bulk_path = _resolve(cfg["data"]["bulk"]) if cfg["data"].get("bulk") else None
    output_dir_obj = _resolve(cfg["paths"]["output_dir"])
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    name = cfg["data"].get("name", "decode_run")

    # Load reference
    import anndata as ad
    import pandas as pd

    print("=" * 60)
    print(f"DECODE — {name}")
    print("=" * 60)

    print(f"\n[1] Loading scRNA-seq reference: {sc_ref_path}")
    if str(sc_ref_path).endswith(".h5"):
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.data_loader import load_sc_ref
        ref = load_sc_ref(str(sc_ref_path))
    else:
        ref = ad.read_h5ad(str(sc_ref_path))
    ref.obs_names_make_unique()
    print(f"  Shape: {ref.shape}")

    # Cell type column
    celltype_col_src = cfg["data"].get("celltype_col") or auto_detect_celltype_col(ref.obs.columns)
    if celltype_col_src is None:
        raise ValueError(f"Could not detect cell type column. Available: {list(ref.obs.columns)}")
    print(f"  Cell type column: '{celltype_col_src}'")

    if celltype_col_src != "CellType":
        ref.obs["CellType"] = ref.obs[celltype_col_src].values
    celltype_col = "CellType"

    type_list = sorted(ref.obs[celltype_col].unique().tolist())
    print(f"  Cell types ({len(type_list)}): {type_list}")

    min_cells = cfg["data"].get("min_cells_per_type", 30)
    counts = ref.obs[celltype_col].value_counts()
    valid_types = counts[counts >= min_cells].index.tolist()
    excluded = [ct for ct in type_list if ct not in valid_types]
    if excluded:
        print(f"  Excluding types with <{min_cells} cells: {excluded}")
    type_list = valid_types
    ref = ref[ref.obs[celltype_col].isin(type_list)].copy()
    ref.obs_names = ref.obs_names.astype(str)
    print(f"  Filtered reference: {ref.shape}")

    max_cells = cfg["data"].get("max_cells")
    if max_cells is not None and ref.n_obs > max_cells:
        print(f"  Subsampling reference to {max_cells} cells...")
        n_per_type = max(1, max_cells // len(type_list))
        idx = []
        for ct in type_list:
            ci = ref.obs[ref.obs[celltype_col] == ct].index.tolist()
            ns = min(n_per_type, len(ci))
            if ns < len(ci):
                rng = np.random.default_rng(SEED)
                ci = list(np.array(ci)[rng.choice(len(ci), ns, replace=False)])
            idx.extend(ci)
        ref = ref[idx].copy()
        ref.obs_names_make_unique()
        ref.obs_names = ref.obs_names.astype(str)

    if hasattr(ref.X, "toarray"):
        ref.X = ref.X.toarray()

    os.chdir(str(DECODE_SRC))
    # Ensure data/tissue_name/ directory exists (data_process.py uses hardcoded
    # relative paths like data/{name}/{name}Ncell.pkl)
    (Path("data") / name).mkdir(parents=True, exist_ok=True)

    # ── Train ──
    if args.mode in ("train", "all"):
        print(f"\n[2] Splitting reference...")
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = [], []
        for ct in type_list:
            idx = ref.obs[ref.obs[celltype_col] == ct].index.tolist()
            if len(idx) < 2:
                train_idx.extend(idx); continue
            t1, t2 = train_test_split(idx, test_size=cfg["data"].get("test_split", 0.5),
                                       random_state=SEED)
            train_idx.extend(t1); test_idx.extend(t2)
        train_data = ref[train_idx].copy(); test_data = ref[test_idx].copy()
        train_data.obs_names = train_data.obs_names.astype(str)
        test_data.obs_names = test_data.obs_names.astype(str)
        print(f"  Train: {train_data.n_obs} cells, Test: {test_data.n_obs} cells")

        print(f"\n[3] Generating pseudo-bulk...")
        dp = data_process(type_list, tissue_name=name,
            train_sample_num=cfg["training"]["train_samples"],
            test_sample_num=cfg["training"]["test_samples"],
            sample_size=cfg["training"].get("sample_size", 30),
            num_artificial_cells=cfg["training"].get("artificial_cells", 30),
            random_type=celltype_col)
        os.chdir(str(DECODE_SRC))
        dp.fit(train_data, test_data)

        pkl_path = os.path.join(str(DECODE_SRC), "data", name, f"{name}{len(type_list)}cell.pkl")
        if not os.path.exists(pkl_path):
            alt = os.path.join("data", f"{name}{len(type_list)}cell.pkl")
            if os.path.exists(alt):
                pkl_path = alt
        print(f"  Loading pseudo-bulk from {pkl_path}")
        with open(pkl_path, "rb") as f:
            train_pkl = pickle.load(f); test_pkl = pickle.load(f); _ = pickle.load(f)
        tx, tn1, tn2, ty = train_pkl; ttx, tty = test_pkl

        vs = min(1000, len(tx) // 2)
        dls = {}
        for ds, dl_name in [(DataLoader(TestCustomDataset(ttx, tty),
                            batch_size=cfg["training"]["batch_size"], shuffle=False), "test"),
                 (DataLoader(TestCustomDataset(tx[:vs], ty[:vs]),
                  batch_size=cfg["training"]["batch_size"], shuffle=False), "valid")]:
            dls[dl_name] = ds
        tr_loader = DataLoader(TrainCustomDataset(tx[vs:], tn1[vs:], tn2[vs:], ty[vs:]),
                               batch_size=cfg["training"]["batch_size"], shuffle=True)

        print(f"\n[4] Training MBdeconv...")
        ng = tx[vs:][0].shape[0] if len(tx[vs:]) > 0 else tx[0].shape[0]
        print(f"  Genes: {ng}, Cell types: {len(type_list)}")

        model = MBdeconv(num_MB=ng,
            feat_map_w=cfg["model"].get("feat_map_w", 256),
            feat_map_h=cfg["model"].get("feat_map_h", 10),
            num_cell_type=len(type_list),
            epoches=cfg["training"]["epochs"],
            Alpha=cfg["training"].get("alpha", 1),
            Beta=cfg["training"].get("beta", 1),
            train_data=tr_loader, test_data=dls["valid"])

        if cfg["training"].get("gpu", False) and model.gpu_available:
            device = torch.device("cuda:0"); model = model.to(device)
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu"); print("  Device: CPU")

        model.train_model(f"{name}_model", if_pure=True,
                          patience=cfg["training"].get("patience", 50))

        bp = os.path.join(str(DECODE_SRC), "save_models", str(ng), f"{name}_model.pt")
        if os.path.exists(bp):
            model.load_state_dict(torch.load(bp, map_location=device))

        model.eval()
        CCC, RMSE, Corr, Corr_s, MAE, pred_df, gt_df = predict(
            dls["test"], type_list, model, if_pure=True)
        print(f"  Test CCC: {CCC:.6f}, RMSE: {RMSE:.6f}, Pearson: {Corr:.6f}, Spearman: {Corr_s:.6f}")

        ckpt_dir = output_dir_obj / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(ckpt_dir / "mbdeconv.pt"))
        with open(ckpt_dir / "type_list.json", "w") as f:
            json.dump(type_list, f)
        with open(ckpt_dir / "gene_ids.json", "w") as f:
            json.dump([str(g) for g in ref.var_names], f)
        meta = {"test_pearson": float(Corr), "test_ccc": float(CCC),
                "test_rmse": float(RMSE), "test_spearman": float(Corr_s),
                "n_genes": ng, "n_types": len(type_list), "type_list": type_list}
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Training complete. Checkpoint -> {ckpt_dir}/")

    # ── Predict ──
    if args.mode in ("predict", "all"):
        if args.mode == "predict":
            ckpt_dir = Path(cfg["paths"].get("checkpoint_dir", output_dir_obj / "checkpoint"))
            if not ckpt_dir.is_absolute():
                ckpt_dir = PROJECT_ROOT / ckpt_dir
            with open(ckpt_dir / "type_list.json") as f:
                type_list = json.load(f)
            with open(ckpt_dir / "metadata.json") as f:
                meta = json.load(f)
            # Dummy loaders — required by MBdeconv.__init__ but unused in predict mode
            dummy_ds = DataLoader([(torch.zeros(meta["n_genes"]), torch.zeros(1, 2),
                                     torch.zeros(1, 2), torch.zeros(len(type_list)))], batch_size=1)
            model = MBdeconv(num_MB=meta["n_genes"],
                feat_map_w=cfg["model"].get("feat_map_w", 256),
                feat_map_h=cfg["model"].get("feat_map_h", 10),
                num_cell_type=len(type_list), epoches=1, Alpha=1, Beta=1,
                train_data=dummy_ds, test_data=dummy_ds)
            dev = torch.device("cuda:0" if cfg["training"].get("gpu", False) and
                               model.gpu_available else "cpu")
            model.load_state_dict(torch.load(str(ckpt_dir / "mbdeconv.pt"), map_location=dev))
            model = model.to(dev)

        if bulk_path is None:
            print("\n  No bulk data specified. Skipping prediction.")
            return

        print(f"\n[5] Predicting on bulk: {bulk_path}")
        bulk_path_str = str(bulk_path)
        if bulk_path_str.endswith(".h5"):
            import h5py
            with h5py.File(bulk_path_str, "r") as f:
                bx_raw = np.asarray(f["bulk/values"][:], dtype=np.float64)
                bulk_symbols = [x.decode() for x in f["bulk/rownames"][:]]
                bulk_barcodes = [x.decode() for x in f["bulk/colnames"][:]]
            # Auto-detect orientation: DeconBenchmark stores (n_genes, n_samples)
            # but downstream code expects (n_samples, n_genes)
            if bx_raw.shape[0] == len(bulk_symbols) and bx_raw.shape[1] == len(bulk_barcodes):
                bx_raw = bx_raw.T  # transpose to (n_samples, n_genes)
            elif bx_raw.shape[1] == len(bulk_symbols) and bx_raw.shape[0] == len(bulk_barcodes):
                pass  # already (n_samples, n_genes)
            bulk_var_names = bulk_symbols
            bulk_obs_names = bulk_barcodes
        else:
            bulk = ad.read_h5ad(bulk_path_str)
            bulk.obs_names = bulk.obs_names.astype(str)
            bx_raw = bulk.X
            if hasattr(bx_raw, "toarray"):
                bx_raw = bx_raw.toarray()
            bx_raw = np.asarray(bx_raw, dtype=np.float64)
            bulk_var_names = list(bulk.var_names)
            bulk_obs_names = list(bulk.obs_names)

        model_genes = json.load(open(str(ckpt_dir / "gene_ids.json")))
        # Truncate to n_genes actually used by the model (pseudo-bulk feature
        # selection may reduce the gene count from the full reference).
        n_model_genes = meta.get("n_genes", len(model_genes))
        if len(model_genes) > n_model_genes:
            model_genes = model_genes[:n_model_genes]
        # Map each model gene to its index in bulk (case-insensitive)
        bulk_gene_index = {g.upper(): i for i, g in enumerate(bulk_var_names)}
        bulk_idx = [bulk_gene_index.get(g.upper(), -1) for g in model_genes]
        found = sum(1 for i in bulk_idx if i >= 0)
        print(f"  Model genes: {len(model_genes)}, Found in bulk: {found}")
        # Fallback: gene_ids.json may contain cell barcodes instead of gene names.
        # Use scRNA reference gene names from the H5 file.
        if found < 100 and bulk_path_str.endswith(".h5"):
            print(f"  gene_ids fallback: using scRNA reference rownames...")
            import h5py
            with h5py.File(bulk_path_str, "r") as f:
                if "singleCellExpr/rownames" in f:
                    sc_genes = [x.decode() for x in f["singleCellExpr/rownames"][:]]
                elif "singleCellExpr/colnames" in f:
                    sc_genes = [x.decode() for x in f["singleCellExpr/colnames"][:]]
                else:
                    sc_genes = None
            if sc_genes is not None:
                n_model = meta.get("n_genes", len(model_genes))
                model_genes = sc_genes[:n_model]
                bulk_idx = [bulk_gene_index.get(g.upper(), -1) for g in model_genes]
                found = sum(1 for i in bulk_idx if i >= 0)
                print(f"  Retry: {len(model_genes)} genes, Found in bulk: {found}")
        if found < 100:
            raise ValueError(f"Too few model genes found in bulk ({found})")

        # Extract bulk expression for model genes (zero-fill missing)
        n_genes = len(model_genes)
        n_samples = bx_raw.shape[0]
        bx = np.zeros((n_samples, n_genes), dtype=np.float64)
        for i, gi in enumerate(bulk_idx):
            if gi >= 0:
                bx[:, i] = bx_raw[:, gi]
        bnorm = np.zeros_like(bx, dtype=np.float64)
        for i in range(bx.shape[0]):
            mx = bx[i].max()
            bnorm[i] = bx[i] / mx if mx > 0 else bx[i]

        dummy_y = np.zeros((bnorm.shape[0], len(type_list)), dtype=np.float32)
        bdl = DataLoader(TestCustomDataset(bnorm, dummy_y),
                         batch_size=cfg["training"].get("batch_size", 64), shuffle=False)
        model.eval()
        rates = []
        with torch.no_grad():
            for batch in bdl:
                xs = batch["x_sim"]
                if model.gpu_available:
                    xs = xs.to(model.gpu)
                _, pr = model.pure_forward(xs)
                rates.append(pr.view(-1, len(type_list)).cpu())
        ar = torch.cat(rates, dim=0).numpy()
        ar = ar / ar.sum(axis=1, keepdims=True)

        pdf = pd.DataFrame(ar, index=bulk_obs_names, columns=type_list)
        csv_path = output_dir_obj / "proportions.csv"
        pdf.to_csv(str(csv_path))
        print(f"  Saved proportions -> {csv_path}")

        print("\nMean proportions:")
        for i, ct in enumerate(type_list):
            print(f"  {ct:25s}: {ar[:, i].mean():.4f}")

    print(f"\nDone. Output in {output_dir_obj}/")


if __name__ == "__main__":
    main()
