#!/usr/bin/env python3
"""
汇总 to_publish/results/ 下所有方法的性能指标并排序，输出 Markdown。

扫描 1_pseudo_bulk/ 和 2_realbulk/ 下的 metrics.json，
按平均 Pearson r 排序，含跨基准对照表、每数据集 Top-3、
以及 Frozen Backbone + RidgeCV（"PCA+ridge"）在真实批量上的评价。

另见:
  - scripts/generate_summary.py   → CSV 输出
  - scripts/compile_results.py    → 完整评测编译（含 runs_index 模式）

用法:
  # 默认汇总两个基准（1_pseudo_bulk + 2_realbulk）
  python scripts/summarize_results.py

  # 仅汇总伪批量
  python scripts/summarize_results.py --benchmark pseudo_bulk

  # 仅汇总真实批量
  python scripts/summarize_results.py --benchmark realbulk

  # 跳过 Frozen Backbone + RidgeCV 分析
  python scripts/summarize_results.py --no-ridge

  # 输出到文件
  python scripts/summarize_results.py -o results_summary.md

  # 只看 Top 10
  python scripts/summarize_results.py --top 10 -o results_summary.md

  # 指定结果目录（默认 results/）
  python scripts/summarize_results.py --results-dir /path/to/results
"""

import argparse
import collections
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np


def _bucket_prefix(part: str) -> str | None:
    """识别桶前缀: 1_ → 1_pseudo_bulk, 2_ → 2_realbulk, 否则 None。"""
    if part.startswith("1_"):
        return "1_pseudo_bulk"
    if part.startswith("2_"):
        return "2_realbulk"
    return None


# ── 评价范式分类 ──────────────────────────────────────────────────────
# Mode A：全样本预测（container 方法 + 伪 bulk 训练 + backbone finetune）
#         所有 real-bulk 样本都被预测 → 全部对比 GT
# Mode B：分割评估（RidgeCV split + LoRA）
#         仅 held-out test 样本被评估 → 不可与 Mode A 混排

FROZEN_BACKBONES = [
    "bulkformer", "geneformer", "scfoundation", "scgpt", "stack", "transcriptformer",
]

NONBACKBONE_CONFIG_DIRS = {"gt.csv", "search", "logs"}
MODE_B_EXPERIMENTS = {"ridge", "ridge_scaler", "realbulk"}


def _is_mode_b(method_label: str) -> bool:
    """判断一个真实批量方法是否属于 Mode B（分割评估）。

    Mode B 方法仅预测 held-out test split，其指标不可与全样本预测的方法混排。
    包括: RidgeCV split 评价 (flat bulkformer 控制实验 + backbone/ridge 系列)
    以及 LOO 评价（任意 _loo 后缀）。
    """
    # 明确列出的扁平方法（使用 RidgeCV split 评价）
    if method_label in (
        "pca_ridge",
        "bulkformer_random", "bulkformer_random_mean_pool", "bulkformer_mean_pool",
        "scgpt_lora",
    ):
        return True
    # 所有 bulkformer/ 实验子目录都用 RidgeCV split
    if method_label.startswith("bulkformer/"):
        return True
    # 任何 LOO 后缀的方法（Leave-One-Out 评价）
    if method_label.endswith("_loo"):
        return True
    # 冻结骨干 RidgeCV 实验
    for backbone in FROZEN_BACKBONES:
        for suffix in (
            "/ridge", "/ridge_scaler", "/realbulk",
            "/ridge_loo", "/ridge_scaler_loo",
            "/pca_ridge", "/pca_ridge_loo",
        ):
            if method_label == f"{backbone}{suffix}":
                return True
    return False


def _experiment_subdirs(method_dir: Path) -> list[Path]:
    """返回 method_dir 下的实验子目录（含 metrics.json，排除非实验文件）。"""
    out = []
    for sub in sorted(method_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in NONBACKBONE_CONFIG_DIRS:
            continue
        if (sub / "metrics.json").is_file():
            out.append(sub)
    return out


def collect_metrics(results_dir: str):
    """扫描结果目录，收集 metrics.json，区分 Mode A/B。

    扫描深度:
      - 3 层: {bucket}/{dataset}/{method}/metrics.json (flat method)
      - 4 层: {bucket}/{dataset}/{method}/{experiment}/metrics.json (experiment method)

    实验子目录自动捕获并标记为 ``{method}/{experiment}``。

    返回:
        method_data:   {method: [(dataset, pearson, rmse, mae, scorr, ccorr)]}
                      来自 1_pseudo_bulk（全部 Mode A）
        method_2_data: {method: [(...)]} 来自 2_realbulk（含 Mode A 和 Mode B）
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f"错误: 目录不存在: {results_dir}", file=sys.stderr)
        sys.exit(1)

    method_data = collections.defaultdict(list)
    method_2_data = collections.defaultdict(list)

    for bucket_dir in sorted(results_dir.glob("[12]_*")):
        bucket = _bucket_prefix(bucket_dir.name)
        if bucket is None:
            continue

        for ds_dir in sorted(bucket_dir.iterdir()):
            if not ds_dir.is_dir() or ds_dir.name.endswith(".bak"):
                continue

            for method_dir in sorted(ds_dir.iterdir()):
                if not method_dir.is_dir():
                    continue

                experiments = _experiment_subdirs(method_dir)
                if experiments:
                    # 只有 "default" 唯一子目录 → 扁平化为裸方法名（避免 dwls + dwls/default 分裂）
                    if len(experiments) == 1 and experiments[0].name == "default":
                        sub_dir = experiments[0]
                        try:
                            with open(sub_dir / "metrics.json") as f:
                                d = json.load(f)
                        except (json.JSONDecodeError, OSError):
                            continue
                        entry = (
                            ds_dir.name,
                            d.get("pearson_mean"),
                            d.get("rmse_overall"),
                            d.get("mae_overall"),
                            d.get("scorr_mean"),
                            d.get("ccorr_mean"),
                        )
                        # For Mode B methods, prefer test-only Pearson from ridge_metrics.json
                        _rm = sub_dir / "ridge_metrics.json"
                        if _rm.exists() and _is_mode_b(method_dir.name):
                            try:
                                _rd = json.load(open(_rm))
                                _pts = _rd.get("pearson_per_type", {})
                                _vals = [v for v in _pts.values() if v is not None]
                                if _vals:
                                    entry = (entry[0], round(sum(_vals) / len(_vals), 4),
                                             entry[2], entry[3], entry[4], entry[5])
                            except (OSError, json.JSONDecodeError):
                                pass
                        label = method_dir.name
                        if bucket == "1_pseudo_bulk":
                            method_data[label].append(entry)
                        else:
                            method_2_data[label].append(entry)
                    else:
                        # 多实验子目录 → 每实验一个条目，标记为 {method}/{experiment}
                        for sub_dir in experiments:
                            try:
                                with open(sub_dir / "metrics.json") as f:
                                    d = json.load(f)
                            except (json.JSONDecodeError, OSError):
                                continue
                            entry = (
                                ds_dir.name,
                                d.get("pearson_mean"),
                                d.get("rmse_overall"),
                                d.get("mae_overall"),
                                d.get("scorr_mean"),
                                d.get("ccorr_mean"),
                            )
                            label = f"{method_dir.name}/{sub_dir.name}"
                            _rm2 = sub_dir / "ridge_metrics.json"
                            if _rm2.exists() and _is_mode_b(str(label)):
                                try:
                                    _rd2 = json.load(open(_rm2))
                                    _pts2 = _rd2.get("pearson_per_type", {})
                                    _v2 = [v for v in _pts2.values() if v is not None]
                                    if _v2:
                                        entry = (entry[0], round(sum(_v2) / len(_v2), 4),
                                                 entry[2], entry[3], entry[4], entry[5])
                                except: pass
                            if bucket == "1_pseudo_bulk":
                                method_data[label].append(entry)
                            else:
                                method_2_data[label].append(entry)

                else:
                    # 扁平方法（method_dir/metrics.json 直接存在）
                    metrics_fp = method_dir / "metrics.json"
                    if not metrics_fp.is_file():
                        continue
                    try:
                        with open(metrics_fp) as f:
                            d = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        continue
                    entry = (
                        ds_dir.name,
                        d.get("pearson_mean"),
                        d.get("rmse_overall"),
                        d.get("mae_overall"),
                        d.get("scorr_mean"),
                        d.get("ccorr_mean"),
                    )
                    # Flat method ridge_metrics override
                    _rm3 = method_dir / "ridge_metrics.json"
                    if _rm3.exists() and _is_mode_b(method_dir.name):
                        try:
                            _rd3 = json.load(open(_rm3))
                            _pts3 = _rd3.get("pearson_per_type", {})
                            _v3 = [v for v in _pts3.values() if v is not None]
                            if _v3:
                                entry = (entry[0], round(sum(_v3) / len(_v3), 4),
                                         entry[2], entry[3], entry[4], entry[5])
                        except: pass
                    if bucket == "1_pseudo_bulk":
                        method_data[method_dir.name].append(entry)
                    else:
                        method_2_data[method_dir.name].append(entry)

    return method_data, method_2_data


def rank_methods(method_data: dict):
    """对方法按平均 Pearson r 排序。

    返回:
        [(mean_r, median_r, method, n_datasets, mean_rmse, mean_mae, mean_scorr, mean_ccorr)]
    """
    import math
    ranked = []
    for method, entries in method_data.items():
        valid = [(e[1], e[2], e[3], e[4], e[5]) for e in entries
                 if e[1] is not None and not (isinstance(e[1], float) and math.isnan(e[1]))]
        if not valid:
            continue
        r_vals = [v[0] for v in valid]
        rmse_vals = [v[1] for v in valid if v[1] is not None]
        mae_vals = [v[2] for v in valid if v[2] is not None]
        scorr_vals = [v[3] for v in valid if v[3] is not None]
        ccorr_vals = [v[4] for v in valid if v[4] is not None]

        mean_r = statistics.mean(r_vals)
        median_r = statistics.median(r_vals)
        mean_rmse = statistics.mean(rmse_vals) if rmse_vals else None
        mean_mae = statistics.mean(mae_vals) if mae_vals else None
        mean_scorr = statistics.mean(scorr_vals) if scorr_vals else None
        mean_ccorr = statistics.mean(ccorr_vals) if ccorr_vals else None

        ranked.append((mean_r, median_r, method, len(r_vals),
                       mean_rmse, mean_mae, mean_scorr, mean_ccorr))

    ranked.sort(key=lambda x: -x[0])
    return ranked


# ── Bootstrap ranking CI ──


def _compute_pseudo_bulk_ci(method_data, n_trials=10000):
    """Bootstrap CI for pseudo-bulk rankings across ALL methods.

    Each trial: sample N_total datasets WITH REPLACEMENT from the full pool.
    For each method, compute mean Pearson from the sampled datasets it covers.
    ALL methods are ranked together (same universe as the table).

    A method with fewer datasets has more duplicates in its bootstrap sample
    → naturally wider CI reflecting the uncertainty from limited coverage.
    Methods with <3 datasets get no CI.

    Returns
    -------
    dict[str, tuple[int, int] | None]
        {method: (lower_rank, upper_rank)} or None if <3 datasets.
    """
    import math

    # Full dataset pool
    total_ds = sorted({e[0] for entries in method_data.values() for e in entries})
    n_total = len(total_ds)
    if n_total < 3:
        return {}

    # Build method -> {dataset: pearson}
    method_map = {}
    for m, entries in method_data.items():
        d = {}
        for e in entries:
            ds, r = e[0], e[1]
            if r is not None and not (isinstance(r, float) and math.isnan(r)):
                d[ds] = r
        if d:
            method_map[m] = d

    if not method_map:
        return {}

    all_methods = sorted(method_map.keys())
    tracker = {m: [] for m in all_methods}

    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        sampled = rng.choice(total_ds, size=n_total, replace=True)
        means = {}
        for m in all_methods:
            vals = [method_map[m][ds] for ds in sampled if ds in method_map[m]]
            means[m] = np.mean(vals) if vals else -np.inf

        ranked = sorted(means, key=means.get, reverse=True)
        for idx, m in enumerate(ranked, 1):
            tracker[m].append(idx)

    ci = {}
    for m in all_methods:
        if len(method_map[m]) < 3:
            ci[m] = None
        else:
            ranks = np.array(tracker[m])
            ci[m] = (int(np.percentile(ranks, 2.5)),
                     int(np.percentile(ranks, 97.5)))

    return ci


def collect_frozen_backbone_ridge(results_dir: str):
    """收集 Frozen Backbone + RidgeCV 在真实批量上的结果。

    扫描四种变体:
      - 2_realbulk/{dataset}/{backbone}/ridge/metrics.json              (no scaler)
      - 2_realbulk/{dataset}/{backbone}/ridge_scaler/metrics.json       (with scaler)
      - 2_realbulk/{dataset}/{backbone}/ridge_loo/metrics.json          (no scaler, LOO)
      - 2_realbulk/{dataset}/{backbone}/ridge_scaler_loo/metrics.json   (with scaler, LOO)

    返回:
        ridge_data: {dataset: {backbone: {variant: metrics_dict}}}
          variant = "ridge" | "ridge_scaler" | "ridge_loo" | "ridge_scaler_loo"
        all_backbones: list[str] — 出现过的 backbone 列表
    """
    results_dir = Path(results_dir)
    ridge_data: dict = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )

    for pattern in ("2_realbulk/*/*/ridge*/metrics.json",):
        for fp in sorted(results_dir.glob(pattern)):
            parts = fp.relative_to(results_dir).parts
            dataset = parts[1]
            backbone = parts[2]
            variant = parts[3]  # "ridge" or "ridge_scaler"
            try:
                with open(fp) as f:
                    d = json.load(f)
                if d.get("pearson_mean") is not None:
                    ridge_data[dataset][backbone][variant] = d
            except (json.JSONDecodeError, OSError):
                continue

    all_backbones = sorted(
        set(b for v in ridge_data.values() for b in v.keys())
    )
    return {k: dict(v) for k, v in ridge_data.items()}, all_backbones


def fmt(val, digits=4):
    """格式化数值，处理 None/NaN。"""
    if val is None or (isinstance(val, float) and val != val):
        return "N/A"
    return f"{val:.{digits}f}"


def build_frozen_backbone_ridge_section(results_dir: str):
    """生成 Frozen Backbone + RidgeCV Markdown 表格（含 scaler 对比）。"""
    ridge_data, all_backbones = collect_frozen_backbone_ridge(results_dir)

    if not ridge_data:
        return ""

    datasets = sorted(ridge_data.keys())
    lines = []
    lines.append("\n## Frozen Backbone + RidgeCV — 真实批量 RidgeCV 评价\n")
    lines.append("")
    lines.append(
        "方法: 6 个 foundation model 将 bulk RNA-seq 编码为 cell embedding，"
        "再对部分样本训练 RidgeCV 回归，预测 held-out 测试样本的比例。\n"
    )
    lines.append("`ridge` = 原始嵌入，`ridge_scaler` = StandardScaler 标准化后回归，"
                  "`ridge_loo`/`ridge_scaler_loo` = LOO 评价。\n")

    # ── 汇总表（每 backbone 一行，四种变体对比）──
    per_backbone: dict = collections.defaultdict(
        lambda: {"ridge": [], "ridge_scaler": [], "ridge_loo": [], "ridge_scaler_loo": []}
    )
    for ds in datasets:
        for bb in all_backbones:
            for variant in ("ridge", "ridge_scaler", "ridge_loo", "ridge_scaler_loo"):
                d = ridge_data.get(ds, {}).get(bb, {}).get(variant)
                if d and d.get("pearson_mean") is not None:
                    per_backbone[bb][variant].append(d)

    lines.append("| Backbone | 变体 | 数据集数 | 平均 r | 中位数 r | 平均 RMSE | 平均 MAE | 平均 SCorr | 平均 CCorr |")
    lines.append(":---|---:|---:|---:|---:|---:|---:|---:|---:")
    for bb in all_backbones:
        for variant in ("ridge", "ridge_scaler", "ridge_loo", "ridge_scaler_loo"):
            vals = per_backbone.get(bb, {}).get(variant, [])
            if vals:
                r_vals = [v["pearson_mean"] for v in vals]
                rmse_v = [v["rmse_overall"] for v in vals if v.get("rmse_overall") is not None]
                mae_v = [v["mae_overall"] for v in vals if v.get("mae_overall") is not None]
                sc_v = [v["scorr_mean"] for v in vals if v.get("scorr_mean") is not None]
                cc_v = [v["ccorr_mean"] for v in vals if v.get("ccorr_mean") is not None]
                lines.append(
                    f"| **{bb}** | {variant} | {len(vals)} | {fmt(statistics.mean(r_vals))} | "
                    f"{fmt(statistics.median(r_vals))} | {fmt(statistics.mean(rmse_v) if rmse_v else None)} | "
                    f"{fmt(statistics.mean(mae_v) if mae_v else None)} | "
                    f"{fmt(statistics.mean(sc_v) if sc_v else None)} | "
                    f"{fmt(statistics.mean(cc_v) if cc_v else None)} |"
                )

    # 对照: gene-level Ridge
    gene_ridge_records = []
    for fp in sorted(Path(results_dir).glob("2_realbulk/*/ridge/metrics.json")):
        try:
            with open(fp) as f:
                d = json.load(f)
            if d.get("pearson_mean") is not None:
                gene_ridge_records.append(d)
        except Exception:
            continue
    if gene_ridge_records:
        r_vals = [v["pearson_mean"] for v in gene_ridge_records]
        rmse_v = [v["rmse_overall"] for v in gene_ridge_records if v.get("rmse_overall") is not None]
        mae_v = [v["mae_overall"] for v in gene_ridge_records if v.get("mae_overall") is not None]
        sc_v = [v["scorr_mean"] for v in gene_ridge_records if v.get("scorr_mean") is not None]
        cc_v = [v["ccorr_mean"] for v in gene_ridge_records if v.get("ccorr_mean") is not None]
        lines.append(
            f"| *(对照) gene-level Ridge* | — | {len(gene_ridge_records)} | "
            f"{fmt(statistics.mean(r_vals))} | {fmt(statistics.median(r_vals))} | "
            f"{fmt(statistics.mean(rmse_v) if rmse_v else None)} | "
            f"{fmt(statistics.mean(mae_v) if mae_v else None)} | "
            f"{fmt(statistics.mean(sc_v) if sc_v else None)} | "
            f"{fmt(statistics.mean(cc_v) if cc_v else None)} |"
        )
    lines.append("")

    # ── 每数据集详情（四种变体各一个表）──
    for variant in ("ridge", "ridge_scaler", "ridge_loo", "ridge_scaler_loo"):
        lines.append(f"\n### 每数据集详情 — {variant}\n")
        # Collect datasets that have at least one value for this variant
        active_dbs = [bb for bb in all_backbones
                      if any(ridge_data.get(ds, {}).get(bb, {}).get(variant)
                             for ds in datasets)]
        if not active_dbs:
            lines.append(f"（无数据）\n")
            continue
        header = f"| {'数据集':<24}"
        for bb in active_dbs:
            header += f"| {bb:<14}"
        header += "|"
        sep = "|:" + "-" * 23 + ":"
        for _ in active_dbs:
            sep += "|:" + "-" * 13 + ":"
        sep += "|"
        lines.append(header)
        lines.append(sep)
        for ds in datasets:
            row = f"| **{ds}**"
            for bb in active_dbs:
                d = ridge_data.get(ds, {}).get(bb, {}).get(variant)
                if d:
                    r = d.get("pearson_mean")
                    row += f" | {r:.4f}" if r else " | N/A"
                else:
                    row += " | —"
            row += " |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def _build_dataset_description() -> str:
    """Return a Markdown section describing the 6 real-bulk and 30 pseudo-bulk datasets."""
    return """## 数据集描述

### 真实 Bulk 数据集（12 个）

| 数据集 | 组织 | 样本数 | bulk(样本×基因) | sce(细胞×基因) | 细胞类型数 | scRNA 细胞类型 | GT 类型 | GT 来源 |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---|:---|
| **SDY67** | PBMC | 250 | 250×17,387 | 4,900×1,344 | 5 | B_cells, Monocytes, NK_cells, Plasmablasts, T_cells | Plasmablasts, NK_cells, T_cells, Monocytes, B_cells | 流式细胞术 |
| **Sweetwater** | PBMC | 14 | 14×35,587 | 13,971×2,944 | 4 | B_cells, Monocytes, Neutrophils, T_cells | Neutrophils, T_cells, B_cells, Monocytes | 已知比例人工混合 |
| **Huuki-Myers** | 脑 DLPFC | 24 | 24×1,076 | 56,447×1,076 | 6 | Astro, EndoMural, Excit, Inhib, Micro, OligoOPC | Astro, EndoMural, Inhib, Excit, Micro, OligoOPC | smFISH 分子比例 |
| **DeMixSC** | 视网膜 | 24 | 24×9,036 | 1,000×9,036 | 10(SC)/7(GT) | AC, Astrocyte, BC, Cone, HC, MG, Microglia, RGC, RPE, Rod | RGC, AC, BC, HC, Rod, Cone, MG | 匹配 snRNA-seq 细胞计数 |
| **Altman-Arunachalam** | PBMC | 322 | 322×23,863 | 23,424×23,863 | 5(GT)/2(SC) | Lymphocytes, Monocytes | Basophils, Eosinophils, Lymphocytes, Monocytes, Neutrophils | 临床全血细胞计数 (CBC) |
| **Altman-TabulaSapiens** | PBMC | 322 | 322×23,863 | 32,023×23,863 | 5(GT)/2(SC) | Lymphocytes, Monocytes | Basophils, Eosinophils, Lymphocytes, Monocytes, Neutrophils | 临床全血细胞计数 (CBC) |
| **altman_Hao** | PBMC | 322 | 322×39,253 | 147,391×24,049 | 5(GT)/11(SC) | B cells, ILC, Monocytes, NK cells, Plasma cells, Platelet, T cells CD4 conv, T cells CD8, Tregs, mDC, pDC | Basophils, Eosinophils, Lymphocytes, Monocytes, Neutrophils | 临床全血细胞计数 (CBC) |
| **finotello_Hao** | PBMC | 9 | 9×19,423 | 147,391×24,049 | 9(GT)/11(SC) | B cells, ILC, Monocytes, NK cells, Plasma cells, Platelet, T cells CD4 conv, T cells CD8, Tregs, mDC, pDC | NK cells, B cells, Tregs, mDC, Monocytes, Neutrophils, T cells CD8, T cells CD4 conv, Other | 流式细胞术 |
| **hoek_Hao** | PBMC | 8 | 8×19,423 | 147,391×24,049 | 5(GT)/11(SC) | B cells, ILC, Monocytes, NK cells, Plasma cells, Platelet, T cells CD4 conv, T cells CD8, Tregs, mDC, pDC | T cell, Monocytes, B cells, mDC, NK cells | 流式分选混合 |
| **hoek_purified_Hao** | PBMC | 48 | 48×20,558 | 147,391×24,049 | 6(GT)/11(SC) | B cells, ILC, Monocytes, NK cells, Plasma cells, Platelet, T cells CD4 conv, T cells CD8, Tregs, mDC, pDC | B cells, mDC, Monocytes, Neutrophils, NK cells, T cell | 流式分选混合 |
| **linsley_purified_Hao** | PBMC | 114 | 114×20,558 | 147,391×24,049 | 6(GT)/11(SC) | B cells, ILC, Monocytes, NK cells, Plasma cells, Platelet, T cells CD4 conv, T cells CD8, Tregs, mDC, pDC | B cells, Monocytes, Neutrophils, NK cells, T cells CD4, T cells CD8 | 流式分选混合 |
| **morandini_Hao** | PBMC | 156 | 156×20,558 | 147,391×24,049 | 8(GT)/11(SC) | B cells, ILC, Monocytes, NK cells, Plasma cells, Platelet, T cells CD4 conv, T cells CD8, Tregs, mDC, pDC | Monocytes, Granulocytes, Lymphocytes, B cells, NK cells, T cells, T cells CD4, T cells CD8 | 流式细胞术 |

**SDY67** 固定 6:2:2 切分（train 0–149, val 150–199, test 200–249），其余数据集随机 80/20 切分（seed=42）。
**Sweetwater** GT 为实验室设计的已知比例人工混合（ground_truth.csv 逐行记录设计比例，如 equal mixture、T/B/M 30% + 10% B），并非空间转录组比例。
**Altman** scRNA 参考仅含 Lymphocytes 和 Monocytes（缺乏粒细胞），故预测仅覆盖 2/5 的 GT 类型。
**DeMixSC** 的 GT 并非估算，而是**真实测量**：DeMixSC 论文（Genome Research 2024）对 24 例健康视网膜样本（GSE175937, Cowan et al. 2020）的**同一份单核悬液**同时建库做 bulk RNA-seq（Smart-seq v4）与 snRNA-seq，GT 比例 = 该样本 snRNA-seq 注释后各细胞类型的细胞数 / 总细胞数（7 大类，≈98% 细胞），故 bulk 与 GT 共享几乎相同的细胞组成（本地验证：bulk vs 匹配 pseudobulk log-Pearson ≈0.31，与论文一致）。但 H5 中的 scRNA 参考并非真实单细胞——原始 DeMixSC 基准仅提供细胞类型**签名矩阵**（均值表达谱），当前 H5 的 1,000 个细胞由签名矩阵经高斯噪声扩展而来（每类型 100 个伪细胞）。scRNA 含 10 种类型，GT 仅含 7 种（多出的 Astrocyte, Microglia, RPE 在评估时忽略）。由于参考缺乏真实单细胞变异性，简单模型比复杂模型更不易过拟合。原始 snRNA-seq 数据存放于 Human Cell Atlas 和 NCBI SRA（PRJNA734326），需额外下载处理。

**Hao 系列（6 个新增数据集，2026-07）**：共享同一 Hao et al. PBMC scRNA 参考（147,391 细胞 × 24,049 基因，11 种细胞类型）。GT 来源各异——`altman_Hao` 与 `Altman-Arunachalam`/`Altman-TabulaSapiens` 同为 Altman 322 例 PBMC 临床样本，但使用 Hao 参考重对齐；`finotello_Hao` (9 样本) 为 PBMC 混合实验；`hoek_Hao` (8 样本) 来自 Hoek 纯化亚群混合；`hoek_purified_Hao` / `linsley_purified_Hao` / `morandini_Hao` 来自纯化细胞类型的人工混合（48–156 样本）。所有 Hao 系列数据集因 scRNA 参考缺失粒细胞类型，对含 Neutrophils 等粒细胞的 GT 类型预测受限。


### 伪 Bulk 数据集（30 个）

从 scRNA-seq 参考经 Dirichlet(α=1.0) 采样生成，每个数据集 100 个伪 bulk 样本，CPM → log1p 归一化。

- **CELLxGENE 子集（10 个组织）**：来自 [CZI CELLxGENE 数据库](https://cellxgene.cziscience.com/)，涵盖肝脏、心脏、小肠、大脑皮层（4 区）、子宫等多个器官系统。细胞类型数 6–17 种。
- **Tabula Sapiens 子集（20 个组织）**：来自 [Tabula Sapiens 联盟](https://tabula-sapiens.science/) (Jones et al., *Science* 2022)，涵盖血液、骨髓、肺、胰腺、皮肤、肌肉、眼睛等 20 个组织。细胞类型数 2–13 种。\n"""


def build_report(method_data, method_2_data, top_n=None, results_dir=None, with_ridge=True):
    """生成 Markdown 汇总报告。

    真实批量（2_realbulk）的方法按评价范式分开排名：
      - Mode A：全样本预测 → 可互相比较
      - Mode B：分割 train/test 评估 → 只与本模式内方法比较
    """
    lines = []
    lines.append("# 结果汇总报告\n")
    lines.append(_build_dataset_description())

    # ── 1_pseudo_bulk ──
    ranked = rank_methods(method_data)

    # Bootstrap CI for rankings (rank among ALL methods in same universe)
    n_total = max(len({e[0] for entries in method_data.values() for e in entries}), 1)
    rank_ci = _compute_pseudo_bulk_ci(method_data)

    lines.append("## 1_伪批量 — 方法性能排序（按平均 Pearson r）\n")
    lines.append("| 排名 | 方法 | 排名 95% CI | 覆盖率 | 平均 r | 中位数 r | 平均 RMSE | 平均 MAE | 平均 SCorr | 平均 CCorr |")
    lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    show = ranked[:top_n] if top_n else ranked
    for i, (mr, medr, method, n, rmse, mae, sc, cc) in enumerate(show, 1):
        ci = rank_ci.get(method)  # Bootstrap CI across all methods
        ci_str = f"[{ci[0]}–{ci[1]}]" if ci is not None else "—"
        cov_str = f"{n}/{n_total}"
        lines.append(
            f"| {i} | **{method}** | {ci_str} | {cov_str} | {fmt(mr)} | {fmt(medr)} | "
            f"{fmt(rmse)} | {fmt(mae)} | {fmt(sc)} | {fmt(cc)} |"
        )

    # ── 2_realbulk — Mode A（全样本预测）──
    rb_mode_a = {m: v for m, v in method_2_data.items() if not _is_mode_b(m)}
    rb_mode_b = {m: v for m, v in method_2_data.items() if _is_mode_b(m)}

    lines.append("\n## 2_真实批量 — 方法性能排序\n")

    if rb_mode_a:
        ranked_a = rank_methods(rb_mode_a)
        lines.append("### Mode A — 全样本预测（参考方法 + 伪 bulk 训练 + 骨干微调）\n")
        lines.append("| 排名 | 方法 | 数据集数 | 平均 r | 中位数 r | 平均 RMSE | 平均 MAE | 平均 SCorr | 平均 CCorr |")
        lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        show_a = ranked_a[:top_n] if top_n else ranked_a
        for i, (mr, medr, method, n, rmse, mae, sc, cc) in enumerate(show_a, 1):
            lines.append(
                f"| {i} | **{method}** | {n} | {fmt(mr)} | {fmt(medr)} | "
                f"{fmt(rmse)} | {fmt(mae)} | {fmt(sc)} | {fmt(cc)} |"
            )
        lines.append("")

    if rb_mode_b:
        rb_mode_b_loo = {m: v for m, v in rb_mode_b.items() if m.endswith("_loo")}
        rb_mode_b_split = {m: v for m, v in rb_mode_b.items() if not m.endswith("_loo")}

        lines.append("### Mode B — RidgeCV 评价（冻结骨干 + BulkFormer 控制实验）\n")

        # LOO table (more reliable: each sample tested once)
        if rb_mode_b_loo:
            ranked_loo = rank_methods(rb_mode_b_loo)
            lines.append("#### Leave-One-Out（每样本轮流测试，使用全部样本）\n")
            lines.append("| 排名 | 方法 | 数据集数 | 平均 r | 中位数 r | 平均 RMSE | 平均 MAE | 平均 SCorr | 平均 CCorr |")
            lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
            show_loo = ranked_loo[:top_n] if top_n else ranked_loo
            for i, (mr, medr, method, n, rmse, mae, sc, cc) in enumerate(show_loo, 1):
                lines.append(
                    f"| {i} | **{method}** | {n} | {fmt(mr)} | {fmt(medr)} | "
                    f"{fmt(rmse)} | {fmt(mae)} | {fmt(sc)} | {fmt(cc)} |"
                )
            lines.append("")

        # Standard split table
        if rb_mode_b_split:
            ranked_split = rank_methods(rb_mode_b_split)
            lines.append("#### 标准训练/测试切分（仅 test split 评价）\n")
            lines.append("| 排名 | 方法 | 数据集数 | 平均 r | 中位数 r | 平均 RMSE | 平均 MAE | 平均 SCorr | 平均 CCorr |")
            lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
            show_split = ranked_split[:top_n] if top_n else ranked_split
            for i, (mr, medr, method, n, rmse, mae, sc, cc) in enumerate(show_split, 1):
                lines.append(
                    f"| {i} | **{method}** | {n} | {fmt(mr)} | {fmt(medr)} | "
                    f"{fmt(rmse)} | {fmt(mae)} | {fmt(sc)} | {fmt(cc)} |"
                )
            lines.append("")

    lines.append("*注: Mode A 与 Mode B 使用不同的评价策略（全样本 vs 仅 test split），"
                  "指标不可横向比较。*\n")

    # ── 跨基准对照（仅 Mode A，伪批量无 Mode B）──
    pb_rank = {m: (i + 1, r) for i, (r, _, m, *_) in enumerate(ranked)}
    rb_rank_a = {m: (i + 1, r) for i, (r, _, m, *_) in enumerate(rank_methods(rb_mode_a))} if rb_mode_a else {}

    both = sorted(set(pb_rank.keys()) & set(rb_rank_a.keys()),
                  key=lambda x: pb_rank[x][0])
    if both:
        lines.append("\n## 跨基准对照表（伪批量 ↔ 真实批量-Mode A）\n")
        lines.append("| 方法 | 伪批量排名 | 伪批量 r | 真实批量排名 | 真实批量 r | 排名差 |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for m in both:
            pr, pv = pb_rank[m]
            rr, rv = rb_rank_a[m]
            lines.append(f"| {m} | {pr} | {fmt(pv)} | {rr} | {fmt(rv)} | {abs(pr - rr)} |")

    # ── 每数据集 Top-3 ──
    lines.append("\n## 各数据集 Top-3 最佳方法\n")

    dataset_best = collections.defaultdict(list)
    for method, entries in method_data.items():
        for ent in entries:
            ds, r = ent[0], ent[1]
            if r is not None:
                dataset_best[ds].append((r, method))

    lines.append("### 1_伪批量\n")
    for ds in sorted(dataset_best):
        top = sorted(dataset_best[ds], key=lambda x: -x[0])[:3]
        top_str = ", ".join(f"{m} ({r:.4f})" for r, m in top)
        lines.append(f"- **{ds}**: {top_str}")

    lines.append("\n### 2_真实批量 — Mode A（全样本预测）\n")
    dataset_best_a = collections.defaultdict(list)
    for method, entries in rb_mode_a.items():
        for ent in entries:
            ds, r = ent[0], ent[1]
            if r is not None:
                dataset_best_a[ds].append((r, method))
    for ds in sorted(dataset_best_a):
        top = sorted(dataset_best_a[ds], key=lambda x: -x[0])[:3]
        top_str = ", ".join(f"{m} ({r:.4f})" for r, m in top)
        lines.append(f"- **{ds}**: {top_str}")

    if rb_mode_b:
        rb_mode_b_loo = {m: v for m, v in rb_mode_b.items() if m.endswith("_loo")}
        rb_mode_b_split = {m: v for m, v in rb_mode_b.items() if not m.endswith("_loo")}

        if rb_mode_b_loo:
            lines.append("\n### 2_真实批量 — Mode B LOO（Leave-One-Out 评价）\n")
            dataset_best_b = collections.defaultdict(list)
            for method, entries in rb_mode_b_loo.items():
                for ent in entries:
                    ds, r = ent[0], ent[1]
                    if r is not None:
                        dataset_best_b[ds].append((r, method))
            for ds in sorted(dataset_best_b):
                top = sorted(dataset_best_b[ds], key=lambda x: -x[0])[:3]
                top_str = ", ".join(f"{m} ({r:.4f})" for r, m in top)
                lines.append(f"- **{ds}**: {top_str}")

        if rb_mode_b_split:
            lines.append("\n### 2_真实批量 — Mode B 标准切分（仅 test split 评价）\n")
            dataset_best_b = collections.defaultdict(list)
            for method, entries in rb_mode_b_split.items():
                for ent in entries:
                    ds, r = ent[0], ent[1]
                    if r is not None:
                        dataset_best_b[ds].append((r, method))
            for ds in sorted(dataset_best_b):
                top = sorted(dataset_best_b[ds], key=lambda x: -x[0])[:3]
                top_str = ", ".join(f"{m} ({r:.4f})" for r, m in top)
                lines.append(f"- **{ds}**: {top_str}")

    # ── Frozen Backbone + RidgeCV ──
    if with_ridge and results_dir:
        ridge_section = build_frozen_backbone_ridge_section(results_dir)
        if ridge_section:
            lines.append(ridge_section)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="汇总 to_publish/results/ 下所有方法的性能指标并排序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python scripts/summarize_results.py
  python scripts/summarize_results.py --benchmark pseudo_bulk --top 10
  python scripts/summarize_results.py --benchmark realbulk -o realbulk_summary.md
  python scripts/summarize_results.py --results-dir /custom/path/results
        """,
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=["all", "pseudo_bulk", "realbulk"],
        default="all",
        help='基准套件: all(默认), pseudo_bulk, realbulk',
    )
    parser.add_argument(
        "--top", "-t",
        type=int, default=None,
        help="只显示前 N 个方法（默认全部）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="输出 Markdown 文件路径（默认打印到 stdout）",
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=os.path.join(Path(__file__).resolve().parent.parent, "results"),
        help='结果目录（默认 to_publish/results/）',
    )
    parser.add_argument(
        "--no-ridge",
        action="store_true",
        help="跳过 Frozen Backbone + RidgeCV（PCA+ridge）分析",
    )
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)

    method_data, method_2_data = collect_metrics(results_dir)

    total_1 = sum(len(v) for v in method_data.values())
    total_2 = sum(len(v) for v in method_2_data.values())
    print(f"发现: {len(method_data)} 方法 × {total_1} 个 1_伪批量 指标, "
          f"{len(method_2_data)} 方法 × {total_2} 个 2_真实批量 指标\n",
          file=sys.stderr)

    report = build_report(
        method_data, method_2_data,
        top_n=args.top,
        results_dir=results_dir,
        with_ridge=not args.no_ridge,
    )

    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w") as f:
            f.write(report)
        print(f"报告已写入: {out_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
