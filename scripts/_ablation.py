#!/usr/bin/env python3
"""Ablation: quantify each difference for bulkformer_random on SDY67."""
import sys, json, time, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
from core.data_loader import load_data

bundle = load_data(str(P / "data/2_real_bulk/sdy67.h5"),
                   ground_truth=str(P / "data/2_real_bulk/sdy67_gt.csv"))
vals = bundle.bulk.values.astype(np.float32)
genes = list(bundle.bulk.columns)
samples_list = list(bundle.bulk.index)
gt_df = bundle.gt

from methods.bulkformer.model import BulkFormerEncoder
encoder = BulkFormerEncoder(pretrained=False)
emb = encoder.encode(vals, genes, samples_list, pooling="global_proj")
print(f"Embeddings: {emb.shape}")

gt_values = gt_df.values.astype(np.float64)
gt_columns = list(gt_df.columns)

def run_ridge(train_emb, test_emb, train_gt, test_gt, use_scaler, alphas):
    if use_scaler:
        scaler = StandardScaler()
        train_emb = scaler.fit_transform(train_emb)
        test_emb = scaler.transform(test_emb)
    test_pred = np.zeros_like(test_gt)
    for j in range(len(gt_columns)):
        y = train_gt[:, j]
        ridge = RidgeCV(alphas=alphas).fit(train_emb, y)
        pred = ridge.predict(test_emb)
        pred = np.clip(pred, 0, None)
        test_pred[:, j] = pred
    r_vals = []
    for j in range(len(gt_columns)):
        mask = ~np.isnan(test_gt[:, j])
        if mask.sum() >= 2 and np.std(test_gt[mask, j]) > 1e-10:
            r = float(np.corrcoef(test_pred[mask, j], test_gt[mask, j])[0, 1])
            if not np.isnan(r):
                r_vals.append(r)
    return float(np.mean(r_vals))

DEFAULT_ALPHAS = [0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0]
ORIG_ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1000.0]

print("\n=== Ablation: impact of each difference on macro_avg r ===\n")

# A: Original config (no scaler, orig alphas, train=150)
train_idx = np.arange(150)
test_idx = np.arange(200, 250)
r_A = run_ridge(emb[train_idx], emb[test_idx], gt_values[train_idx], gt_values[test_idx],
                use_scaler=False, alphas=ORIG_ALPHAS)
print(f"A. Original (no scaler, orig α, train=150):         r={r_A:.4f}")

# B: +StandardScaler
r_B = run_ridge(emb[train_idx], emb[test_idx], gt_values[train_idx], gt_values[test_idx],
                use_scaler=True, alphas=ORIG_ALPHAS)
print(f"B. A + StandardScaler:                               r={r_B:.4f}  (Δ={r_B-r_A:+.4f})")

# C: B + default alphas
r_C = run_ridge(emb[train_idx], emb[test_idx], gt_values[train_idx], gt_values[test_idx],
                use_scaler=True, alphas=DEFAULT_ALPHAS)
print(f"C. B + default α (8 instead of 11):                  r={r_C:.4f}  (Δ={r_C-r_B:+.4f})")

# D: C + train=200 (current to_publish)
train_idx_200 = np.arange(200)
r_D = run_ridge(emb[train_idx_200], emb[test_idx], gt_values[train_idx_200], gt_values[test_idx],
                use_scaler=True, alphas=DEFAULT_ALPHAS)
print(f"D. C + train=200 (current to_publish):                r={r_D:.4f}  (Δ={r_D-r_C:+.4f})")

# E: D + random 80/20 (buggy version)
train_idx80, test_idx20 = train_test_split(np.arange(250), test_size=0.2, random_state=42)
r_E = run_ridge(emb[train_idx80], emb[test_idx20], gt_values[train_idx80], gt_values[test_idx20],
                use_scaler=True, alphas=DEFAULT_ALPHAS)
print(f"E. sklearn random 80/20 (buggy to_publish):          r={r_E:.4f}  (Δ={r_E-r_D:+.4f})")

print(f"\n=== Summary ===")
print(f"Original target: 0.6175")
print(f"A -> Original config: {r_A:.4f}")
print(f"D -> Current to_publish: {r_D:.4f}")
print(f"Total Δ (D - A): {r_D-r_A:+.4f}")
print(f"  Due to StandardScaler: {r_B-r_A:+.4f}")
print(f"  Due to alpha grid:     {r_C-r_B:+.4f}")
print(f"  Due to train=200:      {r_D-r_C:+.4f}")
print(f"  Due to random split:   {r_E-r_D:+.4f}")
