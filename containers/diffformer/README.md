# DiffFormer

**Status: NO PUBLIC CODE AVAILABLE**

## Paper
- **Title:** A Transformer-Based Deep Diffusion Model for Bulk RNA-Seq Deconvolution
- **Journal:** Biology (MDPI), 2025, 14(9), 1150
- **DOI:** [10.3390/biology14091150](https://doi.org/10.3390/biology14091150)
- **Authors:** Yunqing Liu, Jinlei Sun, Huanli Li, Wenfei Zhang, Jinying Sheng, Guoqiang Wang, Jianwei Wu
- **Affiliation:** School of Computer Science, Luoyang Institute of Science and Technology, China

## Description
DiffFormer combines a conditional denoising diffusion model (DDPM) with a Transformer encoder for bulk RNA-seq deconvolution. It uses scRNA-seq references to train the model, then predicts cell-type proportions from bulk RNA-seq data.

## Architecture
- Model dimension: 128, 4 attention heads, 3 encoder layers
- Feed-forward dimension: 256
- T=1000 timesteps with linear noise schedule
- Input: noisy proportion vector + diffusion timestep + bulk RNA-seq profile

## Reason Not Containerized
No public GitHub repository or source code was found for this method as of May 2026. The MDPI paper does not provide a code link in its accessible metadata. Contact the corresponding authors at Luoyang Institute of Science and Technology for code access.
