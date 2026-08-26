# BulkGPT Benchmark Containers

This directory contains [Apptainer](https://apptainer.org/) (formerly Singularity)
definition files for the containerized deconvolution methods used in the BulkGPT
benchmark.

**Container SIFs are build-first**: we provide Apptainer definition files for every
containerized method; you build the SIF locally with `build_all.sh`. Pre-built SIF
binaries are **not** distributed with this release.

## Quick Start

```bash
# Build SIFs from definitions (the supported way)
bash build_all.sh
bash build_all.sh --method tape
```

## System Requirements

- **Apptainer >= 1.1** (or Singularity >= 3.8)
- **~10 GB free disk space** (for all SIFs)
- **Internet connection** (for Docker pulls and package downloads)
- **Linux x86_64** (tested on CentOS 7, Ubuntu 20.04+, Rocky Linux 8+)

## Building SIFs from Source

The supported way to obtain SIFs is to build them from the provided definition files:

```bash
# Build all methods
bash build_all.sh

# Build a single method
bash build_all.sh --method tape
```

The build script runs `apptainer build` inside each method's subdirectory.
Definition files reference `deconvolution/base:latest` as the base Docker image,
which uses R 4.2 + Python 3.9 with common dependencies pre-installed.

## Method Overview

| Method         | Language | Type                     | SIF Size  | Runtime (1K cells) | Source |
|----------------|----------|--------------------------|-----------|-------------------|--------|
| TAPE (scTAPE) | Python   | Autoencoder              | ~3.2 GB   | 5-15 min          | [GitHub](https://github.com/zhanglab/TAPE) |
| SQUID          | R        | Dampened WLS             | ~2.5 GB   | 2-10 min          | [GitHub](https://github.com/mjnajafpanah/SQUID) |
| DeMixSC       | R        | wNNLS + benchmark align  | ~2.8 GB   | 5-20 min          | [GitHub](https://github.com/wwylab/DeMixSC) |
| ConDecon       | R        | Clustering-independent   | ~2.5 GB   | 2-10 min          | [GitHub](https://github.com/CamaraLab/ConDecon) |
| Sweetwater     | Python   | Interpretable autoencoder| ~4.5 GB   | 10-30 min         | [GitHub](https://github.com/ML4BM-Lab/Sweetwater) |
| hspe (dtangle2)| R        | High-collinearity adjust | ~2.3 GB   | 1-5 min           | CRAN dtangle |
| MixupVI        | Python   | VAE + Mixup (reference-free)| ~4.2 GB | 5-15 min          | [GitHub](https://github.com/owkin/PyDeconv) |
| Deconformer    | Python   | Pathway Transformer      | ~3.4 GB   | 2-5 min           | Not public* |
| DiffFormer     | Python   | DDPM + Transformer       | ~3.2 GB   | 10-30 min         | Not public* |

\* Deconformer and DiffFormer source code was not publicly available as of
   May 2026. Their definition files are included for documentation purposes.

## Directory Structure

```
containers/
├── tape/               # TAPE (scTAPE) definition
│   ├── tape.def
│   └── run.py          # DeconBenchmark H5 I/O wrapper
├── squid/              # SQUID definition
│   ├── squid.def
│   └── run.R
├── demixsc/            # DeMixSC definition
│   ├── demixsc.def
│   └── run.R
├── condecon/           # ConDecon definition
│   ├── condecon.def
│   └── run.R
├── sweetwater/         # Sweetwater definition
│   ├── sweetwater.def
│   ├── run.py
│   └── src/sweetwater/ # Model source code
├── deconformer/        # Deconformer definition (source not public)
│   ├── deconformer.def
│   ├── run.py
│   └── README.md
├── diffformer/         # DiffFormer definition (source not public)
│   ├── diffformer.def
│   ├── run.py
│   └── README.md
├── hspe/               # hspe (dtangle2) definition
│   ├── hspe.def
│   └── run.R
├── mixupvi/            # MixupVI definition
│   ├── mixupvi.def
│   └── run.py
├── sif/                # Built SIFs go here
├── build_all.sh        # Build all SIFs from definitions
├── download_all.sh     # (optional) fetch pre-built SIFs
└── README.md           # This file
```

Per-method container fixes live in the top-level `fixed_scripts/{method}/`
directory; `core/_shared/container_runner.py` bind-mounts them over the
container's original `run.py`/`run.R` at runtime, so no SIF rebuild is needed
when a method script has a bug. See `fixed_scripts/README.md`.

## Running a Container

All containers follow the DeconBenchmark H5 I/O convention:

```bash
apptainer run \
    --bind /path/to/input.h5:/input/args.h5 \
    --bind /path/to/output:/output \
    tape.sif
```

The container reads from `/input/args.h5` and writes results to `/output/results.h5`.
Input H5 format:

- `bulk/values`: (n_samples, n_genes) bulk expression matrix
- `singleCellExpr/values`: (n_cells, n_genes) single-cell expression matrix
- `singleCellLabels/values`: (n_cells,) cell type labels

## Citation

If you use these containers, please cite:

```
@software{bulkgpt2025,
  author = {Ye, Rui},
  title = {BulkGPT: Benchmarking Foundation Models for Bulk RNA-seq Deconvolution},
  year = {2025},
  url = {https://github.com/yerui/bulkgpt}
}
```
