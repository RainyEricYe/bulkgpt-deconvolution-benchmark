# xgboost

XGBoost deconvolution baseline. Trains an `xgboost.XGBRegressor` on Dirichlet
pseudo-bulk mixtures from the scRNA reference and predicts real-bulk
proportions (Mode A, SCADEN-style).

## Quick start

```bash
python methods/xgboost/run.py --h5 <input.h5> --output-dir <out> --ground-truth <gt.csv>
```

## Reference

See `methods/xgboost/configs/default.yaml` for hyperparameters.
