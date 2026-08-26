# mlp

MLP deconvolution baseline. Trains an `sklearn.MLPRegressor` on Dirichlet
pseudo-bulk mixtures simulated from the dataset's scRNA-seq reference, then
predicts real-bulk cell-type proportions (Mode A, SCADEN-style).

## Quick start

```bash
python methods/mlp/run.py --h5 <input.h5> --output-dir <out> --ground-truth <gt.csv>
```

## Reference

See `methods/mlp/configs/default.yaml` for hyperparameters.
