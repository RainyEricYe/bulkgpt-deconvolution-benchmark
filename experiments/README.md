# Experiments

This directory contains experiment definitions used by `scripts/run_sweep.py` for hyper-parameter sweeps and ablation studies.

## Relationship with `benchmarks/`

There are two parallel experiment systems in this repository:

| System | Config files | Entry point | Purpose |
|--------|-------------|-------------|---------|
| **Benchmarks** | `benchmarks/*/config/experiments.yaml` | `bash benchmarks/*/run_all.sh` | Reproducing paper results (canonical entry point) |
| **Experiments** | `experiments/scenarios.yaml` | `python scripts/run_sweep.py` | Ablation studies and hyper-parameter sweeps (advanced use) |

**For reproducing the paper**: Use `bash benchmarks/1_pseudo_bulk/run_all.sh` etc.
**For custom sweeps**: Use `python scripts/run_sweep.py --scenario <name>`.

Both systems depend on the same `methods/` directory structure and `core/` utilities.
The scenario names in `scenarios.yaml` can reference any method registered in any
`benchmarks/*/config/experiments.yaml` file.

## Files

- `scenarios.yaml` — Named experiment scenarios (full benchmark, foundation models only, ablation, etc.)
- `sweep.yaml` — Hyper-parameter grid sweeps (learning rate, nHVG, loss function, pooling, seeds)
