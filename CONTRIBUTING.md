# Contributing

Thanks for your interest in contributing to the BulkGPT deconvolution
benchmark! This project is released alongside a research paper, so we keep a
few conventions to make results reproducible and comparable.

## Adding a new deconvolution method

1. **Create a method directory** under `methods/{name}/` containing:
   - `manifest.yaml` — name, mode (`train`/`predict`/`signature`), timeout,
     and container/runtime entry points.
   - `run.py` — a unified `train`/`predict` dispatcher (see `methods/nnls/`
     for a minimal example).
   - `configs/default.yaml` — hyperparameters and paths (relative to repo root).
   - `README.md` — algorithm description, original citation, and any
     non-PyPI dependencies.
2. **Use the shared I/O conventions**:
   - Accept `--sc-ref`, `--bulk`, `--output-dir`, `--mode`.
   - Write predicted proportions to `{output-dir}/proportions.csv`
     (samples × cell types, sum-to-1).
   - Read H5 files via `core.data_loader.load_data()` — never assume a
     particular rownames/colnames orientation.
3. **Register the method**:
   - For linear/signature methods, add it to `methods/_linutils.py` or the
     relevant dispatcher in `scripts/`.
   - For containerized methods, add a `containers/{name}/{name}.def`
     Apptainer definition + a row in `methods/_shared/container_runner.py`.
   - Add it to the method list in `README.md`.
4. **Add tests**: at minimum a smoke test that runs the method on a tiny
   synthetic dataset and checks the output shape + sum-to-1.
5. **Run the suite**:
   ```bash
   conda activate bulkgpt
   python -m pytest tests/ -q
   ```

## Data

- Raw H5 benchmark files are hosted on HuggingFace
  (`yeruihku/bulkgpt-data`); do **not** commit large H5 files to git.
- Small ground-truth CSVs live in `data/2_real_bulk/` and are tracked.
- New datasets: add a download/convert script under `data/prepare/` and a
  manifest entry, then regenerate the H5 in canonical format.

## Code style

- Python 3.10+, type hints on all public functions.
- Imports: stdlib → third-party → local (blank-line separated).
- Comments explain *why*, not *what*.
- Keep files under ~400 lines; split modules rather than growing them.

## Reporting issues

- Bug reports: include the exact command, method/dataset, and full traceback.
- Expected-results mismatches: state the dataset, method, and observed vs
  expected Pearson r (see `REPRODUCIBILITY.md`).

## License

By contributing, you agree that your contributions are licensed under the
[CC BY 4.0](LICENSE) license of this project.
