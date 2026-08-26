# scgpt_lora

scGPT with **LoRA fine-tuning** for bulk RNA-seq deconvolution. Applies
low-rank adapters (q/v projections) to scGPT transformer layers and jointly
trains with a `LinearDeconvHead` on bulk data. Supports rank 4/8/16 and
q/v or q/k/v targets.

## Quick start

```bash
# Train (requires scGPT weights; see weights/README.md)
python methods/scgpt_lora/train.py --config methods/scgpt_lora/configs/default.yaml

# Predict
python methods/scgpt_lora/predict.py --config ... --checkpoint <best_model.pt>
```

## Reference

- Hu et al. *scGPT: toward building a foundation model for single-cell
  multi-omics* (Nature Methods, 2024)
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022)
