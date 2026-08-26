# recide

ReCIDE — reference-based deconvolution using a Seurat marker-selection step
followed by dampened weighted least-squares (DWLS). Containerized (Apptainer
SIF); run through `methods/_shared/container_runner.py`.

## Quick start

```bash
python methods/recide/run.py --h5 <input.h5> --output-dir <out> --ground-truth <gt.csv>
```

Requires the `containers/recide` SIF (see `containers/README.md`).

## Reference

https://github.com/zhandong/ReCIDE
