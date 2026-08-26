# 评价范式与方法分类

## 概述

框架内 **79 种方法**（7 种已弃用，72 种活跃）按评价策略分为两大范式。
每个方法的分类基于其实际数据使用模式（通过 `DeconUtils::getArgs()` 调用从
DeconBenchmark 原始源码核实，标注为 ✅；标注 ⚠️ 的为根据论文推断）。

**评价数据集**：共 12 个真实 bulk 数据集（6 个原始 + 6 个新增 Hao 系列）+ 30 个伪 bulk 数据集。
2026-07 新增的 6 个 Hao 系列数据集共享同一 Hao et al. PBMC scRNA 参考
（147,391 细胞 × 24,049 基因，11 种细胞类型），详见 `results_summary.md` 数据集描述表。

---

## 范式 A：全样本预测评估

方法对全部 real-bulk 样本产生比例预测，然后**所有样本 vs GT 对比**计算指标。

### A1. 使用 scRNA 全参考

从 H5 读取 `singleCellExpr/values`（完整 scRNA 表达矩阵）+ `singleCellLabels/values`
（细胞类型标签），利用完整单细胞谱推断比例。

| # | 方法 | 确认方式 | 说明 |
|:--:|:-----|:--------|:-----|
| 1 | **BayesPrism** | ✅ `c(bulk, singleCellExpr, singleCellLabels)` | 贝叶斯模型 |
| 2 | **BayICE** | ✅ 同上 | 贝叶斯迭代估计 |
| 3 | **SCDC** | ✅ 同上 | 集成反卷积（多参考） |
| 4 | **DWLS** | ✅ 同上 | 阻尼加权最小二乘 |
| 5 | **SpatialDWLS** | ✅ 同上 | 空间扩展 DWLS |
| 6 | **MuSic** | ✅ 同上 | 多主体加权反卷积 |
| 7 | **AdRoit** | ✅ 同上 | 自适应鲁棒优化 |
| 8 | **DeMixT** | ✅ 同上 | EM 混合分解 |
| 9 | **MOMF** | ✅ 同上 | 基于矩的因子模型 |
| 10 | **CPM** | ✅ 同上 | Counts Per Million |
| 11 | **BisqueRef** | ✅ `c(..., singleCellSubjects)` | 基于 scRNA 相似性 |
| 12 | **deconvSeq** | ✅ `c(..., sigGenes)` | 广义线性模型（拟泊松） |
| 13 | **SQUID** | ✅ `fixed_scripts` 确认 | 阻尼 WLS（R 实现） |
| 14 | **DeMixSC** | ✅ `fixed_scripts` 确认 | 基准对齐 wNNLS（R） |
| 15 | **HSPE** | ✅ `fixed_scripts` 确认 | dtangle2 高共线性调整（R） |
| 16 | **Sweetwater** | ✅ `fixed_scripts` 确认 | scArches VAE 自编码器 |
| 17 | **TAPE** | ✅ `fixed_scripts` 确认 | scTAPE 自编码器+GAN（注：也做伪 bulk 训练→归类 A3） |
| 18 | **DecOT** | ⚠️ 论文推断 | 最优传输反卷积 |
| 19 | **DAISM** | ⚠️ 论文推断 | XGBoost 优化 |
| 20 | **DeconPeaker** | ⚠️ 论文推断 | Peak 区域反卷积 |
| 21 | **RNA-Sieve** | ⚠️ 论文推断 | RNA 筛选 |

> `singleCellSubjects` 为可选的 donor_id 分组信息，部分方法（BisqueRef, MuSic）用于跨 subject 校正。

### A2. 使用 Signature matrix

从 H5 读取 `signature/values` 或 `cellTypeExpr/values`（每个细胞类型的平均表达谱），
然后对 bulk 做回归。

| # | 方法 | 回归/模型 | 确认方式 | 说明 |
|:--:|:-----|:----------|:--------|:-----|
| 22 | **NNLS** | 非负最小二乘 | ✅ `_linutils.py` | 线性基线 |
| 23 | **OLS** | 普通最小二乘 | ✅ 同上 | 线性基线 |
| 24 | **Ridge** | 岭回归 | ✅ 同上 | 线性基线 |
| 25 | **NuSVR** | Nu-SVR | ✅ 同上 | 线性基线 |
| 26 | **CIBERSORTx** | NuSVR（3×nu） | ✅ `cibersortx/run.py` | Python 重实现 |
| 27 | **Dtangle** | 线性模型 | ✅ `c(bulk, cellTypeExpr)` | 基于混合比例 |
| 28 | **FARDEEP** | 截断最小二乘 | ✅ `c(bulk, signature)` | 自适应离群剔除 |
| 29 | **EPIC** | 约束最小二乘 | ✅ `c(bulk, cellTypeExpr, sigGenes)` | 已知/未知类型校正 |
| 30 | **DeconRNAseq** | 顺序修剪 NNLS | ✅ `c(bulk, signature)` | |
| 31 | **PREDE** | 迭代回归 | ✅ `c(bulk, cellTypeExpr)` | 预测驱动 |
| 32 | **DESeq2** | 负二项 + VST | ✅ `c(bulk, cellTypeExpr)` | 意外强力基线 |
| 33 | **LinDeconSeq** | 线性模型 | ✅ `c(bulk, signature)` | |
| 34 | **quanTIseq** | 约束回归 | ✅ `c(bulk, signature)` | |
| 35 | **MIXTURE** | 混合模型 | ✅ `c(bulk, signature)` | |
| 36 | **MySort** | 排序回归 | ✅ `c(bulk, signature)` | |
| 37 | **NITUMID** | 肿瘤微环境回归 | ✅ `c(bulk, signature)` | |
| 38 | **MethylResolver** | DNA 甲基化 | ✅ `c(bulk, signature)` | 用甲基化 signature |
| 39 | **EMeth** | 甲基化位点 | ✅ `c(bulk, cellTypeExpr)` | |
| 40 | **DeCompress** | 压缩感知 | ✅ `c(bulk, cellTypeExpr, sigGenes, seed)` | |
| 41 | **ImmuCellAI** | 免疫评分 + 标记 | ✅ `c(bulk, signature, markers)` | signature 为主 |
| 42 | **ARIC** | 自适应回归 | ⚠️ 论文推断 | |
| 43 | **AutoGenes** | 自动基因选择 | ⚠️ 论文推断 | |

> `sigGenes` 是 signature 所用的基因子集。
>
> **注**：**CountBridges**（随机桥过程 + EM，⚠️ manifest 推断）原归属 A5 但 manifest 显示其从 scRNA 构建 signature matrix，暂列 A2 待确认。

### A3. 伪 bulk 训练 → 全量预测

从 scRNA 生成伪 bulk 训练数据，训练有参数的模型（神经网络等），然后对所有 real-bulk 样本去卷积。

| # | 方法 | 模型架构 | 确认方式 | 说明 |
|:--:|:-----|:---------|:--------|:-----|
| 44 | **DECODE** | MBdeconv（MLP） | ✅ `decode/run.py` | 表达重建损失 |
| 45 | **TAPE** | scTAPE（AE+GAN） | ✅ `tape/run.py` | 注：接收全 scRNA 后内部做伪 bulk |
| 46 | **SCADEN** | 前馈 NN | ✅ SIF 内部 | 伪 bulk 训练 |
| 47 | **DigitalDLSorter** | 深度 NN | ✅ `c(bulk, singleCellExpr, singleCellLabels, seed)` | 接收全 scRNA 后内部做伪 bulk |
| 48 | **DiffFormer** | 扩散 Transformer | ✅ `diffformer/run.py` | checkpoint 匹配时使用 |

### A4. 使用标记基因（Markers）

从 H5 读取 `markers/values`（每个细胞类型的标记基因列表），基于标记基因表达推断比例。

| # | 方法 | 模型 | 确认方式 | 说明 |
|:--:|:-----|:-----|:--------|:-----|
| 49 | **BisqueMarker** | 标记基因均值+回归 | ✅ `c(bulk, markers)` | 仅用标记基因 |
| 50 | **DSA** | 差异表达标记+WLS | ✅ `c(bulk, markers)` | |
| 51 | **MCPCounter** | 标记基因计数 | ✅ `c(bulk, markers)` | 无需 scRNA |
| 52 | **TOAST** | 标记+矩阵分解 | ✅ `c(bulk, markers)` | |
| 53 | **ReCIDE** | Seurat 标记+SVR | ✅ `recide/run.py` | |

### A5. 无参考 / 盲分解

仅从 H5 读取 `bulk/values` + `nCellTypes/values`（仅细胞类型数量），
不依赖于任何 scRNA 参考或标记基因。

| # | 方法 | 分解方法 | 确认方式 |
|:--:|:-----|:---------|:--------|
| 54 | **DeconICA** | ICA → 匈牙利匹配 | ✅ `c(bulk, nCellTypes)` |
| 55 | **RefACTor** | PCA → 匈牙利匹配 | ✅ `c(bulk, nCellTypes)` |
| 56 | **LinSeed** | 线性盲源分离 | ✅ `c(bulk, nCellTypes)` |
| 57 | **deconf** | 非负矩阵分解（NMF） | ✅ `c(bulk, nCellTypes)` |
| 58 | **BayCount** | 贝叶斯计数模型 | ✅ `c(bulk, nCellTypes)` |
| 59 | **MixupVI** | VAE + Mixup | ✅ `fixed_scripts` 确认 |
| 60 | **pca_ridge** | PCA → RidgeCV | ✅ `pca_ridge/run.py` |

### A6. 分类待进一步确认的 5 种方法

以下 5 种方法的 DeconBenchmark 原始源码未找到，分类基于文献推断。
已在 A1/A2 中以 ⚠️ 标注，此处集中说明（不计入额外方法数）：

| 方法 | 原表（#） | 推断分类 | 推断依据 |
|:-----|:---------|:---------|:--------|
| **DecOT** | 表A1 (#18) | A1（scRNA） | 最优传输需细胞级数据 |
| **DAISM** | 表A1 (#19) | A1（scRNA） | DeconBenchmark 文献分类 |
| **DeconPeaker** | 表A1 (#20) | A1（scRNA） | Peak 区域反卷积需全参考 |
| **ARIC** | 表A2 (#42) | A2（signature） | 自适应回归反卷积名称暗示 |
| **AutoGenes** | 表A2 (#43) | A2（signature） | 自动基因选择用于 signature |

---

## 范式 B：real-bulk 分割 train/test 评估

把 real-bulk 样本分割为 train/test，训练器在 train 上训练，**仅 test 上**计算指标。

### B1. Frozen backbone + RidgeCV

用 foundation model 编码 bulk → embedding 作为特征 → RidgeCV 回归。

| # | 方法 | Backbone | 环境 | 输出目录 |
|:--:|:-----|:---------|:----|:---------|
| 67 | **scGPT** | scGPT whole-human | bulkgpt | `{dataset}/scgpt/ridge[_scaler]/` |
| 68 | **Geneformer** | Geneformer V2-104M | geneformer | `{dataset}/geneformer/ridge[_scaler]/` |
| 69 | **STACK** | STACK-Large | stack | `{dataset}/stack/ridge[_scaler]/` |
| 70 | **TranscriptFormer** | TranscriptFormer | TranscriptFormer | `{dataset}/transcriptformer/ridge[_scaler]/` |
| 71 | **scFoundation** | scFoundation-1B | scfoundation | `{dataset}/scfoundation/ridge[_scaler]/` |
| 72 | **BulkFormer** | BulkFormer-147M | bulkformer | `{dataset}/bulkformer/ridge[_scaler]/` |

**分割策略**：
- SDY67：固定 6:2:2（train 0-149, val 150-199, test 200-249）
- 其余数据集：随机 80/20

### B2. BulkFormer 控制实验

在 B1 框架下用 BulkFormer 编码器替换 embedding 策略。

| # | 方法 | 权重 | Pooling | 说明 |
|:--:|:-----|:----|:--------|:-----|
| 73 | **bulkformer/random** | 随机初始化 | global_expr_proj | 对照：预训练 vs 随机 |
| 74 | **bulkformer/mean_pool** | 预训练 | mean pool | 对照：global_proj vs mean |
| 75 | **bulkformer/random_mean_pool** | 随机 | mean pool | 双对照 |
| 76 | **bulkformer/bootstrap** | 预训练 | global_proj + 50× | 集成增强 |
| 77 | **bulkformer/fstat** | 预训练 | F-stat 加权 | 基因权重增强 |

### B3. 微调（端到端训练）

| # | 方法 | 微调策略 | 分割 |
|:--:|:-----|:---------|:----|
| 78 | **scGPT-LoRA** | LoRA (r=8, q/v) + LinearDeconvHead | SDY67: 6:2:2 + val early stop |

---

## 已弃用（7 种）

| 方法 | 原来分类 | 弃用原因 |
|:-----|:---------|:---------|
| **BayesCCE** | A1 | SIF 太大，squashfuse 超时 |
| **Deblender** | A1 | SIF 太大，squashfuse 超时 |
| **DeBcam** | A5 (`c(bulk, nCellTypes)`) | n_types > n_genes，凸包失败 |
| **Deconformer** | A1 | cfRNA 专用，checkpoint 不匹配 |
| **ConDecon** | A1 | 函数不存在，输出基因级 |
| **BulkGPT** | B1 | SIF 已清理，由 scGPT+BulkFormer 替代 |
| **methylresolver** | A2 | R 并行 bug（`length=2 in logical(1)`） |

> **注**：`DeBcam` 和 `methylresolver` 的方法逻辑本身可用，但因工程问题被弃用。

---

### Hao 系列数据集注意事项

2026-07 新增的 6 个 Hao 系列数据集（altman_Hao, finotello_Hao, hoek_Hao,
hoek_purified_Hao, linsley_purified_Hao, morandini_Hao）全部使用同一 Hao et al. PBMC
scRNA 参考（147,391 细胞 × 24,049 基因，11 种细胞类型）。该参考**不含粒细胞类型**
（Neutrophils 等），因此对 GT 含粒细胞的预测天然受限——多数容器方法的预测因 scRNA
类型不匹配而只能覆盖部分 GT 类型，体现为较低 Pearson。

此外，该 scRNA 参考的 147,391 × 24,049 矩阵约 27 GB，`enrich_h5()` 的首次缓存创建需
~10–20 分钟（GPFS 读取）。已通过 `mkdir` 原子文件锁防止多进程并发创建缓存。

---

## 分类依据说明

所有容器方法的分类基于 DeconBenchmark 原始源码中的 `DeconUtils::getArgs()` 调用
（位于 `the upstream DeconBenchmark-docker repo`）。调用决定了方法
从 H5 中读取哪些数据：

| getArgs 模式 | 分类 | 含义 |
|:-------------|:----|:-----|
| `c(bulk, singleCellExpr, singleCellLabels)` | **A1** | 使用完整 scRNA 表达谱 |
| `c(bulk, signature)` 或 `c(bulk, cellTypeExpr)` | **A2** | 使用签名矩阵 |
| `c(bulk, markers)` | **A4** | 使用标记基因 |
| `c(bulk, nCellTypes)` | **A5** | 无参考，仅需细胞类型数 |
| 自定义（训练 + 预测循环） | **A3** | 伪 bulk 训练后预测 |
| 自定义（分割训练/评估） | **B1/B2/B3** | real-bulk 分割评估 |

---

## 方法数据使用全景图

```
范式 A（全样本预测，60 种方法）
├── A1 scRNA 全参考 ........... BayesPrism, BayICE, SCDC, DWLS, MuSic, AdRoit,
│                              SpatialDWLS, SQUID, DeMixSC, HSPE, Sweetwater,
│                              CPM, MOMF, BisqueRef, deconvSeq, DeMixT,
│                              TAPE* (也做伪 bulk 训练), DecOT, DAISM,
│                              DeconPeaker, RNA-Sieve
├── A2 signature matrix + 回归 . NNLS, OLS, Ridge, NuSVR, CIBERSORTx,
│                              DESeq2, Dtangle, EPIC, FARDEEP, DeconRNAseq,
│                              quanTIseq, MySort, NITUMID, PREDE,
│                              MIXTURE, MethylResolver, EMeth, DeCompress,
│                              LinDeconSeq, ImmuCellAI, ARIC, AutoGenes,
│                              CountBridges
├── A3 伪 bulk 训练 ............ DECODE, TAPE*, SCADEN, DigitalDLSorter,
│                              DiffFormer
├── A4 标记基因 ............... BisqueMarker, DSA, MCPCounter, TOAST, ReCIDE
├── A5 无参考/盲分解 .......... DeconICA, RefACTor, LinSeed, deconf,
│                              BayCount, MixupVI, pca_ridge


范式 B（分割 train/test 评估，12 methods）
├── B1 frozen backbone + RidgeCV . scGPT, Geneformer, STACK, TranscriptFormer,
│                                  scFoundation, BulkFormer (×2 变体)
├── B2 BulkFormer 控制实验 ....... random, mean_pool, random_mean_pool,
│                                  bootstrap, fstat
└── B3 微调 ..................... scGPT-LoRA
```

**注意**：
1. TAPE 出现在 A1 和 A3 两个分类中——它接收全 scRNA 参考，但内部用伪 bulk 训练模型
2. A6 的 5 种待确认方法已在 A1/A2 中计数，此处仅集中说明分类依据
3. CountBridges 原列 A5，经 manifest 核实应归 A2（从 scRNA 构建 signature matrix）
4. B1/B2 方法也可在伪 bulk 基准（1_pseudo_bulk）上运行，但行为不同：在伪 bulk 上做 RidgeCV 分割评估
5. `collect_metrics()` 自动将伪 bulk 和真实 bulk 基准分开汇总

---

## 方法架构与设计详细说明

### B1 组：Frozen backbone + RidgeCV

所有 B1 方法共享相同的评估范式：用 foundation model 将 bulk 基因表达编码为低维 embedding，
然后对每个细胞类型独立做 RidgeCV 回归。**Backbone 权重冻结**，不参与训练。

#### scGPT (whole-human)

| 维度 | 说明 |
|:-----|:-----|
| **模型架构** | 12 层 Transformer (d_model=512, nhead=8, d_hid=512)，GPT-style 生成式预训练。vocab 包含 gene token + 特殊 token (`<cls>`, `<pad>` 等)。 |
| **编码流程** | (1) 基因匹配到 vocabulary，未命中用 `<pad>` 填充；(2) 若基因数 > 3000，用 Seurat v3 流程筛选 HVG；(3) 补 `<cls>` token → 12 层 Transformer 编码 → 取除 `<cls>` 外所有 token 的 **mean pool** 作为细胞级 embedding (512维)。 |
| **与 scGPT-LoRA 区别** | B1 的 scGPT 是 frozen backbone + RidgeCV；B3 的 scGPT-LoRA 对 backbone 做 LoRA 微调 (+ MLP head)。两者 embedding 维度相同 (512) 但分布不同。 |
| **实现** | `_encode_scgpt()`: `scgpt.model.TransformerModel._encode()` → mean pool → **512-dim** embedding → RidgeCV per type。 |
| **引用** | Cui et al., Nature Methods 2024 |

#### Geneformer (V2-104M)

| 维度 | 说明 |
|:-----|:-----|
| **模型架构** | 6 层 Performer (BERT-style masked LM)，~104M 参数，基于 GenEC 语义相似性排序策略（rank-value encoding）。输入为按表达量降序排列的基因 token 序列（而非数值表达值）。 |
| **编码流程** | (1) Gene symbol → Ensembl ID → token ID (token_dict) 映射；(2) 每细胞按表达量从高到低排序，取前 max_len−1 个基因 token；(3) 在序列首补 `<pad>` token（作为 CLS 代理），并构造 attention mask；(4) 6 层 Performer → `last_hidden_state` → **weighted mean pool**（按 attention mask 归一化）作为细胞级 embedding (256维)。 |
| **关键设计** | Geneformer 不使用数值表达量——只使用表达的基因的 token 和它们的排序秩。表达量 >0 的基因按秩逆序排列（高表达在前），表达量为 0 的基因被忽略。这种 rank-value 编码使模型天然对批次效应和测序深度不敏感。 |
| **实现** | `_encode_geneformer()`: `transformers.AutoModel` → mean pool over valid tokens → **256-dim** embedding → RidgeCV per type。 |
| **引用** | Theodoris et al., Nature 2023 |

#### STACK (STACK-Large)

| 维度 | 说明 |
|:-----|:-----|
| **模型架构** | 两层 Transformer 编码器 + Linear 分类头，在 40M 单细胞跨组织数据上预训练。词表为 2,000 个 marker genes（而非全转录组）。预训练任务：masked gene expression prediction + cell-type classification。 |
| **编码流程** | (1) 2,000 个 marker gene 表达量 → log1p 归一化；(2) 嵌入为 64-dim token；(3) 两层 Transformer + 平均池化 → **320-dim** cell embedding。输出中特定维度编码特定细胞类型（如 PTPRC=CD45 表达量→免疫细胞概率）。 |
| **实现** | `methods/stack/train.py:encode_stack()` → **320-dim** embedding → RidgeCV per type。 |
| **引用** | Armingol et al., bioRxiv 2024 |

#### TranscriptFormer

| 维度 | 说明 |
|:-----|:-----|
| **模型架构** | ~200M 参数 NLP-style Transformer（24 层，hidden=2048，8 head），在 1.2B 单细胞上预训练。以 Ensembl ID 为 token，可学习基因序 embedding + 10x 3' assay token + ESM2 蛋白特征 enhancer。扩展到 Evo 2 (4B) 等变体。 |
| **编码流程** | (1) 若输入为 Gene Symbol（<50% 以 "ENSG" 开头），通过缓存映射到 Ensembl ID；(2) 用 OmegaConf 构建完整 inference config（batch_size=8, fp16, device=auto）；(3) 创建 AnnData (raw counts + `assay="10x 3' transcription profiling"`) + `ensembl_id` 变量；(4) `run_inference()` 前向 → 输出 `obsm["embeddings"]` 作为 **cell embedding**。 |
| **实现** | `_encode_tf()`: `transcriptformer.model.inference.run_inference()` → `result.obsm["embeddings"]` → **2048-dim** embedding → RidgeCV per type。 |
| **引用** | Fu et al., bioRxiv 2024 |

#### scFoundation (1B)

| 维度 | 说明 |
|:-----|:-----|
| **模型架构** | 1B 参数 Transformer（~50 层解码器），预训练于 50M 单细胞（~1T 细胞）。词表 19,264 个基因（~38 个特殊 token），**固定 19,264-dim 零填充**输入。xTransformer 架构带 row/column attention（非标准 full self-attention）。 |
| **编码流程** | (1) 基因匹配：输入基因映射到 19,264 维的 scFoundation 词表位置（未匹配的基因为零）；(2) 对数归一化 (log1p) 防止 fp16 softmax NaN；(3) 前向传播 → 所有 token 的输出 → 取 `output[:, 0, :]`（即 **CLS token**）作为细胞级 embedding (1024维)；(4) batch_size=8, fp16 amp。 |
| **实现** | `_encode_scfoundation()`: `ScFoundationBackbone` (xTransformer) → `output[:,0,:]` → **1024-dim** embedding → RidgeCV per type。 |
| **引用** | Hao et al., Nature Methods 2024 |

#### BulkFormer (147M)

| 维度 | 说明 |
|:-----|:-----|
| **模型架构** | ~147M 参数 hybrid GCN+Performer。先通过 GCNConv 在 TCGA 基因共表达图上做图消息传递，然后接入 Performer 高效 attention 层。输入：20,010 基因的固定词表（基于 bulk RNA-seq 表达矩阵设计，而非单细胞）。模型本身输出**每个基因的 640-dim 表征**，而非单细胞 embedding。 |
| **编码流程 — 两种路径** | (1) **global_expr_proj（默认、快速）**：`Linear(20010→2560) + ReLU + Linear(2560→640)`，直接映射 20,010 基因表达向量 → 640-dim 样本级 embedding。绕过 GCN+Performer，是 BulkFormer 的"捷径"分支。(2) **full encoder（mean pool、慢速）**：完整前向 → GCNConv → Performer → `layernorm` → 取所有 20,010 个基因 token 的 mean pool → 640-dim。基因经过 `gene_emb_onehot + proj` 和 `expr_emb` 嵌入后，加上 `global_expr_proj` 扩展作为全局上下文偏置。 |
| **关键区别** | `global_expr_proj` 是一个 2 层 MLP（不使用权重的基因关系）；full encoder 利用了 TCGA 基因共表达图谱和 Performer 的序列注意力。实验中 `global_expr_proj` 的 RidgeCV 性能**显著优于** full encoder mean pool。 |
| **实现** | `_encode_bulkformer()` → `BulkFormerEncoder.encode()` → **640-dim** embedding → RidgeCV per type。 |
| **引用** | Xie et al., Nature Machine Intelligence 2024 |

### B1 补充说明：Ridge 与 Ridge_scaler

每个 backbone 的 embedding 可以输入 RidgeCV 的两种变体：

- **ridge**（默认）：原始 embedding 直接进入 RidgeCV。
- **ridge_scaler**：先 `StandardScaler` 中心化+缩放后，再输入 RidgeCV。

对于 LOO（留一法）评估，使用 `core/deconv/loo.py` 中的 `run_loo_ridge()`。对每个样本 i，用除 i 外的所有样本训练 RidgeCV，预测 i。所有 backbone 的 LOO 版本由 `scripts/eval_real_bulk_ridge.py` 统一调度。pca_ridge 和 BulkFormer 控制实验（random/mean_pool/random_mean_pool/bootstrap/fstat）也支持通过 `--loo` 参数使用 LOO 评估。

---

### B2 组：BulkFormer 控制实验

B2 组在 B1 框架下系统变化 BulkFormer 的权重初始化和 embedding 策略，全面评估**预训练贡献**、**编码器架构影响**和**集成/加权增强效果**。

#### bulkformer/random — 随机初始化对照

| 维度 | 说明 |
|:-----|:-----|
| **设计目标** | 控制实验：隔离预训练权重的贡献。若 random ≈ pretrained，说明 BulkFormer 的编码能力主要来自架构而非预训练。 |
| **实现** | `BulkFormerEncoder(pretrained=False)`：模型**构造时保持默认 PyTorch 初始化**（`xavier_uniform_` for gene_emb_onehot_layer，默认 Kaiming/Uniform for Linear），**不加载** `BulkFormer_147M.pt` 权重。pooling 使用 `global_expr_proj` 捷径。无任何训练步骤——模型始终冻结在随机初始化状态。 |
| **关键发现** | random init 的 Pearson 在部分数据集上**持平甚至超过** pretrained 版本，提示 BulkFormer 架构（1 层 GCN + 1 层 Performer + 2 层 `global_expr_proj`）的 inductive bias 本身对解卷积有显著贡献。 |

#### bulkformer/mean_pool — 全编码器对照

| 维度 | 说明 |
|:-----|:-----|
| **设计目标** | 控制实验：隔离 `global_expr_proj` 捷径 vs 完整 GCN+Performer 编码器的贡献。若 mean_pool ≈ global_proj，说明 GCN+Performer 部分对解卷积无额外价值。 |
| **实现** | `BulkFormerEncoder(pretrained=True, pooling="mean")`：完整前向 → GCN（在 TCGA 基因共表达图上消息传递）→ 多层 Performer attention → LayerNorm → 所有 20,010 个基因 token 的 `mean(dim=1)` → 640-dim embedding。注意即使 full encoder 路径也**包含** `global_expr_proj` 作为 token embedding 的加性偏置分量。 |
| **关键发现** | mean_pool 的性能通常**低于** global_expr_proj，说明简化视角（加权投影）对 bulk 基因表达的反卷积任务更为有效。 |

#### bulkformer/random_mean_pool — 双对照

| 维度 | 说明 |
|:-----|:-----|
| **设计目标** | 极端的"最坏情况"对照：同时移除预训练权重**和** `global_expr_proj` 捷径，完全依赖随机初始化的 GCN+Performer 架构。 |
| **实现** | `BulkFormerEncoder(pretrained=False, pooling="mean")`：随机初始化的完整 GCN+Performer + mean pool。这是 BulkFormer 的"基线"——可视为一个**随机权重、结构感知的 bulk expression encoder**。 |
| **关键发现** | 即使随机+mean，在部分数据集上的性能仍高于随机猜测，说明 BulkFormer 的 GCN（基因图结构）对表达的归纳偏置本身有价值。 |

#### bulkformer/bootstrap — 50× Bootstrap 集成

| 维度 | 说明 |
|:-----|:-----|
| **设计目标** | 提高 RidgeCV 预测稳定性和置信校准：用 bootstrap resampling 产生 50 份"数据扰动"的训练集，拟合 50 个 RidgeCV 模型，取预测均值。 |
| **实现** | (1) `encoding`: pretrained + `global_expr_proj`；(2) `bootstrap_ridge()`: 对每个细胞类型，做 50 次 bootstrap resampling（`np.random.choice` with replacement，样本数 = 训练集大小）→ 每次训练 `RidgeCV`（alpha 搜索 `[0.01, 0.03, ..., 1000]`）→ 在测试集上预测 → 50 个预测的**均值**作最终预测；(3) 输出还包含 per-type 的标准差（反映 RidgeCV 对数据采样变异的敏感性）。实现位于 `methods/bulkformer/bootstrap_utils.py`。 |
| **与单次 RidgeCV 区别** | 单次 RidgeCV 用全部训练数据拟合一次；bootstrap 用 50 个略微不同的训练集（bootstrap 样本）分别拟合后取均值。在 n_samples 较小时（如 sweetwater 48 样本），bootstrap 的方差缩减效果更显著。 |

#### bulkformer/fstat — F-Stat 基因加权

| 维度 | 说明 |
|:-----|:-----|
| **设计目标** | 利用 scRNA 参考中的**差异表达信息**指导 `global_expr_proj` 专注信息量大的基因。F-stat 衡量每个基因在细胞类型间的分离度：高 F 值 = 基因在两个类型间表达差异显著。 |
| **实现** | (1) 从 DeconBenchmark H5 提取 `singleCellExpr` + `singleCellLabels`；(2) 对齐到 BulkFormer 20,010 基因词表；(3) 对每个 GT 细胞类型，计算与其余类型的 **F-stat**（组间方差 / 组内方差 + 1e-10）；(4) 通过 substring 匹配将 GT 类型映射到 scRNA 类型（scRNA 标签更细粒度），取映射后的 element-wise **max** F-stat；(5) 权重向量 `sqrt(max(F, 0))` → element-wise 乘以 20,010-dim 表达向量→ `global_expr_proj` 编码 → RidgeCV。**每个 GT 类型独立编码**（权重不同 → O(N_types) 次编码）。 |
| **关键设计选择** | F-stat 权重在 `global_expr_proj` 的输入层面（即 20,010 表达值）应用，而非在 embedding 层面。这种加权等价于在 RidgeCV 之前对基因进行**差异感知缩放**，使分离度高的基因对 embedding 贡献更大。 |

---

### B3 组：scGPT-LoRA 端到端微调

| 维度 | 说明 |
|:-----|:-----|
| **设计范式** | 唯一端到端微调方法。不冻结 backbone，用 LoRA 高效适配 + MLP 解卷积头联合优化。 |
| **模型架构** | scGPT 12 层 Transformer backbone + **LoRA adapter** (r=8, target modules=query/value) + `LinearDeconvHead`（单层 Linear → softmax）。LoRA 仅注入 ~0.1% 参数（冻结 99.9%）。 |
| **训练流程** | (1) 从 scRNA 参考抽取细胞 embedding（mean pool 风格，与 B1-scGPT 一致）；(2) `ExpressionMixGenerator` 生成 5000 个伪 bulk 混合物（Dirichlet 采样比例，CPM→log1p 归一化）；(3) 伪 bulk 通过 LoRA-scGPT backbone 编码 → `LinearDeconvHead` 预测比例；(4) 损失函数为 MSE + KL(λ=0.1)；优化器 AdamW (lr=1e-3 for head, backbone_lr=7e-5 for LoRA)。SDY67: 固定 6:2:2（train 0-149, val 150-199, test 200-249），30 epochs，val loss early stop。 |
| **与 B1-scGPT 区别** | B1: frozen backbone → embedding → RidgeCV（两步，无后向传播）；B3: backbone 和 head 联合优化（单阶段，梯度反向传播）。后者需要 GPU 训练 (~20 min, H100)。 |

---

### A5-pca_ridge：PCA + RidgeCV 基线

pca_ridge 归类为 **A5（无参考/盲分解）**，但它实际上是一种**简化版 BulkFormer 对照**：

| 维度 | 说明 |
|:-----|:-----|
| **设计目标** | 最简单的线性基线：仅使用 bulk 表达矩阵自身，不做任何细胞类型参考（A5 无参考定义）。同时作为 BulkFormer 的关键对照——检验 BulkFormer 的 20,010 基因词表和 GCN+Performer 架构相对于纯线性降维的增益。 |
| **方法流程** | (1) **基因对齐**：表达矩阵投影到 BulkFormer 的 20,010 基因词表（与 BulkFormer 使用的 vocabulary 矩阵完全相同），缺失基因填充 -10.0；(2) **PCA 降维**：`sklearn.decomposition.PCA(n_components=min(n_samples, 20010), random_state=42)`，**不经过 StandardScaler**（匹配 BulkFormer global_expr_proj 的行为）；(3) **Per-type RidgeCV**：与 B1/B2 使用完全相同的 RidgeCV 评估管道（`evaluate_real_bulk_ridge()`）、相同 alpha 候选集合 (`[0.01, 0.03, ..., 1000.0]`)、相同 test split。 |
| **与 BulkFormer global_expr_proj 的关系** | pca_ridge 可视为 `global_expr_proj` 的**线性化代理**：两者都作用于 20,010 基因向量，(20010→d)→RidgeCV；区别是 BulkFormer 的映射为非线性 MLP（2 层，ReLU），pca_ridge 为线性 PCA 投影。若两者性能接近，说明**非线性变换对解卷积不是必要的**。 |
| **关键发现** | pca_ridge 在真实 bulk 上以 **Pearson 0.708** 的均值排名全方法第 8（72 方法中），超过所有传统容器方法（如 MuSiC 0.729 略高、DWLS 0.938 显著更高）。在伪 bulk 上 pca_ridge 排名更靠后——说明 PCA + RidgeCV 在真实 bulk 的有噪小样本场景下尤其有效。 |

---

### A5 补充：基准分解方法的架构对比

以下 A5 方法代表不同级别的无参考分解：

| 方法 | 模型类型 | 分解策略 | 输出处理 |
|:-----|:---------|:---------|:---------|
| **DeconICA** | ICA | `FastICA(n_components=n_types)`，统计独立性假设 | 匈牙利匹配（线性分配） |
| **RefACTor** | PCA | 因子分析 + Varimax/NNMF 旋转 | 匈牙利匹配 |
| **deconf** | NMF | 非负矩阵分解（交替最小二乘） | 直接列对应（NMF 非负性天然匹配） |
| **LinSeed** | 线性盲源分离 | 基于表达值秩数的协方差估计 | 混合矩阵 → 匈牙利匹配 |
| **BayCount** | 贝叶斯计数 | 负二项似然 + MCMC | 后验均值 |
| **MixupVI** | VAE + Mixup | 条件 VAE 编码 + Mixup 正则化解码 | latent z → linear decoder |
| **pca_ridge** | PCA + Ridge | 线性降维 + 线性回归 | RidgeCV 直接输出（无需匹配）
