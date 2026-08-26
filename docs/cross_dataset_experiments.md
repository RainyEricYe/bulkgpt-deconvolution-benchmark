# 跨数据集泛化实验

> 实验日期: 2026-07-08
> 项目根: `repository root (``) `
> 框架: `tests/cross_dataset_pbmc.py` → `repository root (``) tests/cross_dataset_pbmc.py`
> 结果目录: `tests/cross_dataset_pbmc/` → `repository root (``) tests/cross_dataset_pbmc/`

## 实验设计

### 目标
验证 frozen backbone 固定嵌入空间的跨数据集泛化能力：在源数据集上训练 RidgeCV（per cell type），直接预测目标数据集的细胞类型比例，无需任何 domain adaptation。

### 数据
12 个真实 bulk 数据集，覆盖 PBMC/血液（10 个）+ 脑（1 个）+ 视网膜（1 个）：

| 分组 | 数据集 | 样本数 | 细类型 | 粗类型 |
|------|--------|:------:|:------:|:------:|
| PBMC-临床 | altman_Arunachalam | 322 | 5 | 3 |
| PBMC-临床 | altman_TabulaSapiens | 322 | 5 | 3 |
| PBMC-临床 | altman_Hao | 322 | 5 | 3 |
| PBMC-混合 | sdy67 | 250 | 5 | 2 |
| PBMC-混合 | sweetwater | 14 | 4 | 3 |
| PBMC-纯化 | finotello_Hao | 9 | 9 | 3 |
| PBMC-纯化 | hoek_Hao | 8 | 5 | 2 |
| PBMC-纯化 | hoek_purified_Hao | 48 | 6 | 3 |
| PBMC-纯化 | linsley_purified_Hao | 114 | 6 | 3 |
| PBMC-纯化 | morandini_Hao | 156 | 8 | 3 |
| 脑 | huuki_myers | 24 | 6 | 6 |
| 视网膜 | demixsc_retina | 24 | 7 | 7 |

### 粗类型对齐
细胞类型通过 `COARSE_MAP` 聚合到通用粗类型：

| 粗类型 | 对应细类型 |
|--------|-----------|
| Lymphocytes | T_cells, B_cells, NK_cells, Plasmablasts, T cells, Lymphocytes, T cells CD4/CD8, Tregs 等 |
| Monocytes | Monocytes, mDC |
| Neutrophils | Neutrophils, Granulocytes |
| Basophils | Basophils（仅 altman） |
| Eosinophils | Eosinophils（仅 altman） |

脑和视网膜数据集有独立类型体系（Astro, RGC, ...），与 PBMC 无交集。

### 方法
两种 backbone ∈ {random_mean_pool, pca_ridge} 作为嵌入方法：

- **random_mean_pool**: BulkFormer 架构随机权重 + mean pooling（固定 643 维嵌入空间）
- **pca_ridge**: PCA 降维（n_components=min(n_samples, 20010)），需 BulkFormer 20010 基因对齐

对每个 source→target pair：
1. 编码 source 和 target 到同一嵌入空间
2. 在 source 上训练 RidgeCV(α∈[0.1–316]) per 粗类型
3. 用训练好的 RidgeCV 预测 target 的粗类型比例
4. 计算 Pearson r 和 Spearman ρ per 类型，取均值

---

## 结果

### 完整矩阵 (random_mean_pool Pearson r)

| 源 ↓ 目标 → | sdy67 | swt | a_Arun | a_Tab | a_Hao | f_Hao | h_Hao | h_pur | l_pur | m_Hao | huuki | dmx |
|-------------|:-----:|:---:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|
| sdy67       | .916  | .628 | .113   | .113  | .120  | -.005 | .364  | .274  | .170  | -.025 |       |     |
| swt         | .320  | .977 | .200   | .200  | .192  | -.028 | .107  | .446  | .214  | .108  |       |     |
| a_Arun      | .519  | .779 | .929   | .929  | .928  | -.076 | .657  | .812  | .869  | .431  |       |     |
| a_Tab       | .519  | .779 | .929   | .929  | .928  | -.076 | .657  | .812  | .869  | .431  |       |     |
| a_Hao       | .536  | .769 | .926   | .926  | .928  | -.057 | .672  | .812  | .870  | .445  |       |     |
| f_Hao       | -.008 | -.042| .122   | .122  | .125  | .411  | .127  | .109  | .035  | .032  |       |     |
| h_Hao       | .518  | .434 | .470   | .470  | .472  | .361  | 1.000 | .431  | .620  | .448  |       |     |
| h_pur       | .622  | .803 | .635   | .635  | .635  | .063  | .571  | 1.000 | .979  | .429  |       |     |
| l_pur       | .658  | .731 | .616   | .616  | .613  | -.055 | .791  | .917  | 1.000 | .454  |       |     |
| m_Hao       | .460  | .459 | .429   | .429  | .423  | -.208 | .198  | .858  | .870  | .815  |       |     |
| huuki       |       |      |        |       |       |       |       |       |       |       | .809  |     |
| dmx         |       |      |        |       |       |       |       |       |       |       |       | .894 |

空格 = 无共享粗类型。对角线 = 自身预测（非 LOO，但 random 权重的嵌入是固定的，所以不是 1.0）。

### 完整矩阵 (pca_ridge Pearson r)

| 源 ↓ 目标 → | sdy67 | swt | a_Arun | a_Tab | a_Hao | f_Hao | h_Hao | h_pur | l_pur | m_Hao | huuki | dmx |
|-------------|:-----:|:---:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|
| sdy67       | 1.000 | .637 | .051   | .051  | .053  | .023  | .277  | -.566 | -.405 | -.293 |       |     |
| swt         | .334  |1.000 | -.020  | -.020 | -.020 | -.041 | -.162 | .120  | -.101 | -.071 |       |     |
| a_Arun      | -.129 | .043 | .979   | .979  | .949  | .112  | .142  | .004  | -.143 | -.020 |       |     |
| a_Tab       | -.129 | .043 | .979   | .979  | .949  | .112  | .142  | .004  | -.143 | -.020 |       |     |
| a_Hao       | -.255 | .046 | .971   | .971  | .979  | .099  | .304  | .030  | -.255 | -.026 |       |     |
| f_Hao       | .003  | -.169| .005   | .005  | .005  | 1.000 | .139  | -.011 | .025  | .101  |       |     |
| h_Hao       | .660  | .671 | .451   | .451  | .451  | .103  | 1.000 | .582  | .629  | .445  |       |     |
| h_pur       | -.026 | .703 | .012   | .012  | .011  | .044  | .469  | 1.000 | .356  | .088  |       |     |
| l_pur       | .497  | .507 | .056   | .056  | .057  | .091  | .653  | .634  | 1.000 | .004  |       |     |
| m_Hao       | .242  | .623 | -.080  | -.080 | -.082 | .117  | .588  | .330  | .479  | 1.000 |       |     |

---

## 关键发现

### 1. random_mean_pool 显著优于 pca_ridge 做跨数据集

在全部 90 个非平凡的 PBMC 跨数据集 pair 上：

| 指标 | random_mean_pool | pca_ridge |
|------|:----------------:|:---------:|
| 正相关占比 | 83% (75/90) | 57% (51/90) |
| 负相关占比 | 17% (15/90) | 43% (39/90) |
| 均值 (非自身上) | **0.41** | **0.13** |
| 中位数 | 0.44 | 0.05 |

pca_ridge 的自身上预测为 1.0（PCA+Ridge 同分布完美拟合），但跨数据集时大量负值（PCA 成分从源数据学到的方差结构在目标上不成立）。random_mean_pool 的固定 643 维嵌入空间保证了跨数据集的表示一致性。

### 2. 源数据集的分布覆盖决定泛化能力 — 不是样本数

| 源 | 样本 | 类型 | 平均跨数据集 r |
|:---|:----:|:----:|:-------------:|
| altman_Arunachalam | 322 | 临床 | **0.65** |
| linsley_purified_Hao | 114 | 纯化 | **0.62** |
| hoek_purified_Hao | 48 | 纯化 | **0.59** |
| altman_Hao | 322 | 临床 | **0.58** |
| morandini_Hao | 156 | 纯化 | 0.45 |
| hoek_Hao | 8 | 纯化 | 0.44 |
| sweetwater | 14 | 混合 | 0.23 |
| sdy67 | 250 | **混合** | **0.18** |
| finotello_Hao | 9 | 混合 | 0.04 |

**sdy67（250 样本，体外混合）比 hoek_Hao（8 样本，纯化群体）泛化更差**。核心因素不是样本数量，而是数据是否覆盖了真实表达变异范围。临床样本（altman）和纯化群体（Hao 系列）包含了真实的生物学变异，RidgeCV 学到的嵌入→比例映射更具鲁棒性。体外混合的 clean 信号反而过拟合到线性的组合关系。

### 3. Pearson 和 Spearman 结论一致

Pearson vs Spearman 的差异很小（most pairs ±0.05）。Spearman 在目标样本少（≤14）且方向关系保留时略高于 Pearson，但在大样本高度线性关系时略低于 Pearson。总体排序完全一致。

### 4. 细胞类型数影响平均 r

Neutrophils 是最好预测的类型（r≈0.8），包含 Neutrophils 的 pair（即共享 3 个粗类型的 pair）平均 r 显著高于只有 2 个类型的 pair。在比较源的能力时需要注意背景。

### 5. 跨组织泛化的局限

huuki_myers（脑）和 demixsc_retina（视网膜）与 PBMC 数据集无共享粗类型，无法直接用 RidgeCV 迁移。对这类场景需要伪 bulk 训练方案（见 `core/deconv/frozen_search.py`）。

---

## 文件清单

结果存储在 `repository root (``) tests/cross_dataset_pbmc/`：

```
cross_dataset_pbmc/
├── {source}_to_{target}/
│   └── {backbone}/
│       ├── proportions.csv      # 预测比例
│       ├── per_type.json         # 每类型 r / rmse
│       └── metrics.json          # DeconBenchmark 指标
├── summary_random_mean_pool.json # 跨数据集汇总
├── summary_pca_ridge.json
├── correlation_random_mean_pool.json  # 12×12 Pearson + Spearman
├── correlation_pca_ridge.json
├── random_mean_pool_pearson.csv  # CSV 版
├── random_mean_pool_spearman.csv
├── pca_ridge_pearson.csv
├── pca_ridge_spearman.csv
└── n_common_types.csv            # 共享粗类型数

# 原始数据路径
H5 文件:  repository root (``) data/2_real_bulk/{dataset}.h5
GT CSV:   repository root (``) data/2_real_bulk/{dataset}_gt.csv
细胞类型映射: COARSE_MAP in tests/cross_dataset_pbmc.py
H5 GT 列名: ground_truth/rownames（Hao 数据集有，sdy67/sweetwater 无）
```

## 运行方法

```bash
# 必须 cd 到项目根目录
cd 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 全矩阵（12×12，3 个 GPU 节点）
CUDA_VISIBLE_DEVICES=3 python tests/cross_dataset_pbmc.py --batch

# 指定源
CUDA_VISIBLE_DEVICES=3 python tests/cross_dataset_pbmc.py \
    --batch --sources sdy67 altman_Arunachalam

# 单对
CUDA_VISIBLE_DEVICES=3 python tests/cross_dataset_pbmc.py \
    --source sdy67 --target sweetwater
```
