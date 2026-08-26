#!/usr/bin/env python3
"""
汇总 to_publish/results/ 下每个方法在每个数据集上的资源占用情况。

扫描 run.log / train.log / predict.log 提取运行时长，
扫描 metadata.json 提取 GPU 显存峰值、编码时间等，
生成 Markdown 汇总表。

用法:
  # 默认汇总两个基准（1_pseudo_bulk + 2_realbulk）
  python scripts/summarize_resources.py

  # 仅汇总真实 bulk
  python scripts/summarize_resources.py --benchmark realbulk

  # 仅汇总伪 bulk
  python scripts/summarize_resources.py --benchmark pseudo_bulk

  # 输出到文件
  python scripts/summarize_resources.py -o resource_usage.md

  # 指定结果目录
  python scripts/summarize_resources.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ── 日志解析 ──────────────────────────────────────────────────────────────

LOG_TIME_RE = re.compile(r"# Elapsed:\s*([\d.]+)\s*s")
RC_RE = re.compile(r"# Return code:\s*(-?\d+)")


def _parse_elapsed(log_path: Path) -> float | None:
    """从 run.log / train.log / predict.log 头部提取耗时（秒）。"""
    try:
        for line in open(log_path):
            m = LOG_TIME_RE.match(line)
            if m:
                return float(m.group(1))
    except (OSError, ValueError):
        pass
    return None


def _parse_return_code(log_path: Path) -> int | None:
    """从日志头部提取返回码。"""
    try:
        for line in open(log_path):
            m = RC_RE.match(line)
            if m:
                return int(m.group(1))
    except (OSError, ValueError):
        pass
    return None


# ── 资源解析 ──────────────────────────────────────────────────────────────

RESOURCE_KEYS = {
    "wall_time_s": ("wall_time_s", float),
    "gpu_memory_peak_mb": ("gpu_memory_peak_mb", float),
    "gpu_name": ("gpu_name", str),
    "encode_time_s": ("encode_time_s", float),
    "ridge_time_s": ("ridge_time_s", float),
    "load_time_s": ("load_time_s", float),
    "embedding_dim": ("embedding_dim", int),
}


def _parse_metadata(meta_path: Path) -> dict[str, Any]:
    """从 metadata.json 提取资源字段。"""
    out: dict[str, Any] = {}
    try:
        data = json.load(open(meta_path))
        for key, (jkey, cast) in RESOURCE_KEYS.items():
            val = data.get(jkey)
            if val is not None:
                try:
                    out[key] = cast(val)
                except (ValueError, TypeError):
                    pass
    except (OSError, json.JSONDecodeError):
        pass
    return out


def _parse_train_log(train_log: Path) -> dict[str, Any]:
    """从 train.log 提取可训练参数量。"""
    out: dict[str, Any] = {}
    try:
        for line in open(train_log):
            m = re.search(r"Trainable.*?:\s*([\d,]+)\s*params?", line)
            if m:
                out["trainable_params"] = int(m.group(1).replace(",", ""))
            m = re.search(r"Trainable \(LoRA\):\s*([\d,]+)\s*params?", line)
            if m:
                out["lora_params"] = int(m.group(1).replace(",", ""))
            m = re.search(r"Trainable \(head\):\s*([\d,]+)\s*params?", line)
            if m:
                out["head_params"] = int(m.group(1).replace(",", ""))
    except OSError:
        pass
    return out


def _parse_train_results(train_results_path: Path) -> dict[str, Any]:
    """从 train_results.json 提取训练信息（epochs、time 等）。"""
    out: dict[str, Any] = {}
    try:
        data = json.load(open(train_results_path))
        training = data.get("training", {})
        if training.get("time_seconds"):
            out["train_time_s"] = training["time_seconds"]
        if training.get("n_epochs"):
            out["n_epochs"] = training["n_epochs"]
        if training.get("best_val_loss"):
            out["best_val_loss"] = training["best_val_loss"]
        if data.get("n_trainable_lora"):
            out["lora_params"] = data["n_trainable_lora"]
        if data.get("n_trainable_head"):
            out["head_params"] = data["n_trainable_head"]
    except (OSError, json.JSONDecodeError):
        pass
    return out


def _pp(val: Any, fmt: str = ".1f") -> str:
    """格式化数值，None 返回 '-''。"""
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:{fmt}}"
    if isinstance(val, int):
        if val > 1e6:
            return f"{val / 1e6:.1f}M"
        if val > 1e3:
            return f"{val / 1e3:.1f}K"
        return str(val)
    return str(val)


# ── 扫描 ──────────────────────────────────────────────────────────────────


def collect_resource_data(results_dir: Path) -> list[dict[str, Any]]:
    """扫描结果目录，收集所有方法 × 数据集的资源占用数据。

    读取的字段:
      run.log / train.log / predict.log:
        # Elapsed: <float>s      — 阶段耗时
        # Return code: <int>      — 进程返回码
      metadata.json:
        wall_time_s              — 总耗时
        gpu_memory_peak_mb       — GPU 显存峰值
        gpu_name                 — GPU 型号
        encode_time_s            — 编码时间 (BulkFormer/backbone)
        ridge_time_s             — RidgeCV 时间
        load_time_s              — 加载时间 (scGPT-LoRA)
        embedding_dim            — 嵌入维度
      train_results.json:
        training.time_seconds    — 训练耗时
        training.n_epochs        — 训练轮数
        n_trainable_lora         — LoRA 参数量
        n_trainable_head         — Head 参数量
    """
    rows: list[dict[str, Any]] = []

    bucket_dirs = []
    for bd in sorted(results_dir.glob("[12]_*")):
        if bd.name.startswith("1_"):
            bucket_name = "1_pseudo_bulk"
        elif bd.name.startswith("2_"):
            bucket_name = "2_realbulk"
        else:
            continue
        bucket_dirs.append((bd, bucket_name))

    for bucket_dir, bucket_name in bucket_dirs:
        for ds_dir in sorted(bucket_dir.iterdir()):
            if not ds_dir.is_dir() or ds_dir.name == "logs":
                continue
            ds = ds_dir.name

            for method_dir in sorted(ds_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                if method in ("logs", "enriched_input.h5") or method.startswith("."):
                    continue

                row: dict[str, Any] = {
                    "bucket": bucket_name,
                    "dataset": ds,
                    "method": method,
                }

                run_logs = sorted(method_dir.rglob("run.log"))
                train_logs = sorted(method_dir.rglob("train.log"))
                predict_logs = sorted(method_dir.rglob("predict.log"))
                metas = sorted(method_dir.rglob("metadata.json"))
                train_results = sorted(method_dir.rglob("train_results.json"))

                # 1. 耗时（优先 metadata.json，其次日志）
                meta = {}
                if metas:
                    meta = _parse_metadata(metas[0])
                row.update(meta)

                if "wall_time_s" not in row:
                    all_logs = run_logs + train_logs + predict_logs
                    for log in all_logs:
                        elapsed = _parse_elapsed(log)
                        if elapsed is not None:
                            phase = log.stem
                            row[f"{phase}_time_s"] = elapsed
                            if "wall_time_s" not in row:
                                row["wall_time_s"] = elapsed

                # 2. 返回码
                for log in run_logs[:1]:
                    rc = _parse_return_code(log)
                    if rc is not None:
                        row["return_code"] = rc

                # 3. 训练信息
                if train_results:
                    row.update(_parse_train_results(train_results[0]))
                else:
                    for tl in train_logs[:1]:
                        row.update(_parse_train_log(tl))

                # 4. 是否有 metrics.json（表示评估完成）
                row["has_metrics"] = len(list(method_dir.rglob("metrics.json"))) > 0

                # 5. 标记空日志（可能表示失败）
                row["has_empty_log"] = any(
                    log.stat().st_size == 0
                    for log in run_logs + train_logs + predict_logs
                )

                rows.append(row)

    return rows


# ── 输出格式 ──────────────────────────────────────────────────────────────


def _method_category(method: str) -> str:
    """按方法类型分组。"""
    ml = method.lower()
    if ml in ("nnls", "ols", "ridge", "nusvr"):
        return "Linear Baseline"
    if ml.startswith("bulkformer") or ml in (
        "scgpt", "geneformer", "stack", "transcriptformer", "scfoundation",
    ):
        return "Foundation Model"
    if ml == "scgpt_lora":
        return "Fine-tuned FM"
    if ml == "pca_ridge":
        return "Control"
    return "Container"


def _sort_key(row: dict[str, Any]) -> tuple:
    cat_order = {
        "Linear Baseline": 0,
        "Control": 1,
        "Foundation Model": 2,
        "Fine-tuned FM": 3,
        "Container": 4,
    }
    return (
        cat_order.get(_method_category(row["method"]), 9),
        row["method"],
        row["dataset"],
    )


def format_markdown(rows: list[dict[str, Any]]) -> str:
    """生成 Markdown 汇总表。"""
    if not rows:
        return "*(无数据)*\n"

    rows_sorted = sorted(rows, key=_sort_key)
    lines: list[str] = []

    def _now() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("# 资源占用汇总\n")
    lines.append(f"> 生成时间: {_now()}\n")

    for bucket_name in ("1_pseudo_bulk", "2_realbulk"):
        bucket_rows = [r for r in rows_sorted if r["bucket"] == bucket_name]
        if not bucket_rows:
            continue

        bucket_label = "伪 Bulk 基准" if bucket_name == "1_pseudo_bulk" else "真实 Bulk 基准"
        lines.append(f"## {bucket_label} ({bucket_name})\n")

        header = (
            "| 方法 | 类别 | 数据集 | 耗时(s) | GPU显存(MB) | 返回码 | 参数量 | 备注 |\n"
            "|------|------|--------|---------|-------------|--------|--------|------|"
        )
        lines.append(header)

        for r in bucket_rows:
            wall = _pp(r.get("wall_time_s"))
            gpu = _pp(r.get("gpu_memory_peak_mb"), ".0f") if r.get("gpu_memory_peak_mb") else "-"
            rc = str(r.get("return_code", "")) if r.get("return_code") is not None else "-"

            params_parts = []
            if r.get("trainable_params"):
                params_parts.append(_pp(r["trainable_params"]))
            if r.get("lora_params"):
                params_parts.append(f"LoRA={_pp(r['lora_params'])}")
            if r.get("head_params"):
                params_parts.append(f"head={_pp(r['head_params'])}")
            params_str = ", ".join(params_parts) if params_parts else "-"

            notes = []
            if r.get("has_empty_log"):
                notes.append("空日志")
            if not r.get("has_metrics") and r.get("return_code") == 0:
                notes.append("rc=0 但无 metrics")
            if r.get("return_code") is not None and r["return_code"] != 0:
                notes.append(f"失败(rc={r['return_code']})")
            if r.get("train_time_s"):
                notes.append(f"训练{r['train_time_s']:.0f}s")
            note_str = "; ".join(notes) if notes else "-"

            method_display = r["method"]
            if "/" in method_display:
                parts = method_display.split("/")
                method_display = f"{parts[0]}({parts[1]})"

            lines.append(
                f"| {method_display} | {_method_category(r['method'])} | {r['dataset']} "
                f"| {wall} | {gpu} | {rc} "
                f"| {params_str} | {note_str} |"
            )
        lines.append("")

    # 汇总统计
    lines.append("## 汇总统计\n")
    lines.append(f"- **方法 × 数据集**: {len(rows)}")
    lines.append(f"- **唯一方法数**: {len(set((r['method'], r['bucket']) for r in rows))}")
    lines.append(f"- **数据集数**: {len(set(r['dataset'] for r in rows))}")
    lines.append(f"- **评估完成 (有 metrics.json)**: {sum(1 for r in rows if r.get('has_metrics'))}")
    lines.append(f"- **失败 (rc≠0)**: {sum(1 for r in rows if r.get('return_code') is not None and r['return_code'] != 0)}")
    lines.append(f"- **空日志**: {sum(1 for r in rows if r.get('has_empty_log'))}")
    lines.append(f"- **有 GPU 显存记录**: {sum(1 for r in rows if r.get('gpu_memory_peak_mb') is not None)}")
    lines.append("")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总方法资源占用")
    parser.add_argument(
        "--results-dir", default="results",
        help="结果目录 (默认: results/)",
    )
    parser.add_argument(
        "--benchmark", choices=["pseudo_bulk", "realbulk", "all"], default="all",
        help="基准类型 (默认: all)",
    )
    parser.add_argument("-o", "--output", help="输出文件 (默认 stdout)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"错误: 目录不存在: {results_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows = collect_resource_data(results_dir)

    if args.benchmark == "pseudo_bulk":
        all_rows = [r for r in all_rows if r["bucket"] == "1_pseudo_bulk"]
    elif args.benchmark == "realbulk":
        all_rows = [r for r in all_rows if r["bucket"] == "2_realbulk"]

    if not all_rows:
        print("警告: 未找到任何资源数据", file=sys.stderr)

    output = format_markdown(all_rows)

    if args.output:
        Path(args.output).write_text(output)
        print(f"已写入: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
