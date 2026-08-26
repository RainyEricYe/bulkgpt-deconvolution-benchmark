# randomforest

RandomForest deconvolution baseline. Trains an `sklearn.RandomForestRegressor`
on Dirichlet pseudo-bulk mixtures from the scRNA reference and predicts
real-bulk proportions (Mode A, SCADEN-style).

## Quick start

```bash
python methods/randomforest/run.py --h5 <input.h5> --output-dir <out> --ground-truth <gt.csv>
```

## Reference

See `methods/randomforest/configs/default.yaml` for hyperparameters.
