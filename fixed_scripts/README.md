# Fixed Scripts

Each directory contains a fixed version of a container-internal script, bind-mounted
at runtime by the benchmark runner via auto-discovery (`container_runner.main()`
looks for `fixed_scripts/{method_name_lower()}/run.{py,R,sh}`).

When the upstream container is updated, remove the corresponding directory.

## Active fixed_scripts (auto-discovered via container_main)

| Method | Bug | Container | Fixed |
|--------|-----|-----------|-------|
| adroit | Python import path | adroit_latest.sif | 2025-06 |
| baycount | R package conflict | baycount_latest.sif | 2025-06 |
| bisquemarker | Marker file path | bisque_latest.sif | 2025-06 |
| condecon | R h5read transposition + ConDecon() function missing | condecon.sif | 2026-06 |
| debcam | Rscript path | debcam_latest.sif | 2025-06 |
| decompress | Py2/Py3 compat | decompress_latest.sif | 2025-06 |
| deconf | Matrix dimension | deconf_latest.sif | 2025-06 |
| deconformer | python→python3 symlink | deconformer.sif | 2025-06 |
| deconvseq | Gene list mismatch | deconvseq_latest.sif | 2025-06 |
| demixsc | Custom read_h5_matrix → DeconUtils wNNLS | demixsc.sif | 2026-06 |
| deseq2 | Bioconductor install | deseq2_latest.sif | 2025-06 |
| diffformer | NNLS dimension bug + hardcoded sample names | diffformer.sif | 2025-06 |
| dsa | R runtime path | dsa_latest.sif | 2025-06 |
| hspe | dtangle2 loading | hspe.sif | 2025-06 |
| immucellai | Python import | immucellai_latest.sif | 2025-06 |
| lindeconseq | R library path | lindeconseq_latest.sif | 2025-06 |
| linseed | R namespace | linseed_latest.sif | 2025-06 |
| mcpcounter | BiocManager | mcpcounter_latest.sif | 2025-06 |
| methylresolver | R parallel bug | methylresolver.sif | 2025-06 |
| mixupvi | PyTorch version | mixupvi.sif | 2025-06 |
| quantiseq | R dependency | quantiseq_latest.sif | 2025-06 |
| refactor | PC scores → proportions via min-max+row-norm | refactor.sif | 2026-06 |
| scaden | TF compatibility | scaden.sif | 2025-06 |
| squid | Custom read_h5_matrix → DeconUtils DWLS | squid.sif | 2026-06 |
| sweetwater | Quick-test (10 epoch) → full training | sweetwater.sif | 2026-06 |
| tape | NumPy ABI | tape.sif | 2025-06 |
| toast | R script path | toast_latest.sif | 2025-06 |

The runner auto-mounts `fixed_scripts/<method>/run.{py,R,sh}` over the container's
script (`/code/run.py` or `/code/run.R`) when it exists. No SIF rebuild needed.
