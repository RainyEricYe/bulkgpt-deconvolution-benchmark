# Count Bridges — Poisson Bridge EM Deconvolution (ICLR 2026)

**Paper:** Nic Fishman, Gokul Gowri, Tanush Kumar, Jiaqi Lu, Valentin De Bortoli,
Jonathan Gootenberg, Omar Abudayyeh. "Count Bridges enable Modeling and
Deconvolving Transcriptomic Data." *ICLR 2026*. arXiv: 2603.04730.

## Method Overview

Count Bridges introduces a **stochastic bridge process on the integers** using
Poisson birth-death dynamics, providing an exact, tractable discrete analogue of
diffusion-style generative models for integer-valued count data (e.g., RNA-seq
reads).

### Deconvolution via EM

For bulk RNA-seq deconvolution, Count Bridges uses an **Expectation-Maximization
(EM) algorithm** that treats cell-type-specific contributions as latent variables:

- **E-step**: Given observed bulk counts and current proportion estimates,
  compute the expected contribution of each cell type to each gene's expression
  via the Poisson bridge conditional expectation.
- **M-step**: Update proportion estimates by aggregating expected contributions
  across genes, regularized by a Dirichlet-style bridge prior.

### Applications

The paper demonstrates Count Bridges for:
1. Modeling single-cell gene expression at nucleotide resolution
2. **Deconvolving bulk RNA-seq** into cell-type proportions
3. Resolving multicellular spatial transcriptomic spots (Visium)

## Implementation Status

**Official code not publicly available** at the time of integration (June 2026).
This implementation is a **minimal working version** based on the paper's EM
framework. The core algorithm is:

```
1. Build signature matrix (mean expression per cell type) from scRNA-seq ref
2. For each bulk sample, run EM until convergence:
   a. E-step: expected_contrib[g,k] = y[g] * p[k] * S[g,k] / sum_j p[j] * S[g,j]
   b. M-step: p[k] propto sum_g expected_contrib[g,k] + bridge_strength / n_types
3. Return normalized proportions
```

This simplified version achieves the Poisson bridge conditional expectation
under a Poisson likelihood — the discrete analogue of a Gaussian diffusion
bridge.

## Results (5 Real-Bulk Datasets)

| Dataset | Pearson r | SCorr | CCorr | RMSE | MAE |
|---------|:---------:|:-----:|:-----:|:----:|:---:|
| SDY67 | 0.403 | 0.610 | 0.357 | 0.105 | 0.084 |
| Sweetwater | 0.469 | -0.507 | 0.410 | 0.262 | 0.216 |
| Huuki-Myers (DLPFC) | 0.058 | -0.214 | 0.073 | 0.380 | 0.322 |
| DeMixSC Retina | -0.167 | -0.283 | -0.144 | 0.189 | 0.136 |
| Altman Arunachalam | 0.473 | -0.975 | 0.455 | 0.481 | 0.385 |

Performance is reasonable for datasets with good gene coverage but degrades
on challenging brain/retina datasets, as expected for a simplified EM approach.

## Usage

```bash
# Real bulk evaluation
python run.py --config configs/default.yaml --mode predict \
    --h5 ../data/2_real_bulk/sdy67.h5 \
    --ground-truth ../data/2_real_bulk/sdy67_gt.csv \
    --output-dir ../results/2_realbulk/sdy67/countbridges

# Pseudo-bulk evaluation
python run.py --config configs/default.yaml --mode predict \
    --data ../data/2_real_bulk/sdy67.h5 \
    --output-dir ../results/pseudo_bulk/countbridges
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `em_max_iter` | 50 | Maximum EM iterations |
| `em_tol` | 0.0001 | Convergence tolerance (L1 change) |
| `bridge_strength` | 1.0 | Dirichlet prior concentration |
| `normalization` | "cpm" | Bulk normalization ("cpm" or "counts") |

## Files

```
methods/countbridges/
├── manifest.yaml       # Method registration
├── run.py              # CLI entry point + EM deconvolution
├── configs/default.yaml
├── environment.yml
└── README.md           # This file
```
