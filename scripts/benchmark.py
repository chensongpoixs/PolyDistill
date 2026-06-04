#!/usr/bin/env python3
"""
Benchmark 评测脚本 — 蒸馏前后通用能力对比（灾难性遗忘检测）。

基于 EleutherAI lm-evaluation-harness，对 Base 和 LoRA 模型分别跑标准化
benchmark，定量判断蒸馏是否损害了模型的通用能力。

依赖: pip install lm-eval
首次运行会自动下载 C-Eval 数据集（~20MB）。

最小可行版本（P0）:
  - 仅跑 C-Eval valid 集（1,346 题，约 5-10 分钟 / 模型）
  - 输出 Base vs LoRA 对比表 + JSON 结果

用法:
  python scripts/benchmark.py                      # 完整 C-Eval valid
  python scripts/benchmark.py --tasks ceval --limit 20   # 快速验证（20 题）
  python scripts/benchmark.py --tasks mmlu,ceval         # 多 benchmark
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在搜索路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poly_distill.config import load_config, setup_environment

logger = logging.getLogger(__name__)

# lm-eval 可选依赖
try:
    import lm_eval
    from lm_eval import simple_evaluate
    _LM_EVAL_AVAILABLE = True
except ImportError:
    _LM_EVAL_AVAILABLE = False

# 支持的 benchmark 任务（P0 最小集，后续可扩展）
_SUPPORTED_TASKS = {
    "ceval": "ceval-valid",           # C-Eval 验证集（1,346 题，中文多学科）
    "mmlu": "mmlu",                   # MMLU（~14K 题，英文多学科）
    "gsm8k": "gsm8k",                 # GSM8K（数学推理，需生成+提取）
    "hellaswag": "hellaswag",         # HellaSwag（常识推理）
}


def _get_output_dir(config) -> Path:
    """确定 benchmark 结果输出目录。

    优先使用训练产生的 runs/train/exp{N} 目录（如果存在且是最新的），
    否则回退到 config.OUTPUT_DIR 同级目录。
    """
    runs_dir = Path("runs/train")
    if runs_dir.exists():
        exp_dirs = sorted(runs_dir.glob("exp*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if exp_dirs:
            return exp_dirs[0]
    return Path(config.OUTPUT_DIR)


def _build_model_args(config, lora_path: str = None) -> str:
    """构造 lm_eval model_args 参数字符串。

    Args:
        config: 全局配置。
        lora_path: LoRA adapter 路径（None 表示仅 Base 模型）。

    Returns:
        如 "pretrained=Qwen/Qwen3-4B,dtype=float16,trust_remote_code=True"
        或 "pretrained=Qwen/Qwen3-4B,peft=./lora_output,dtype=float16"
    """
    args = f"pretrained={config.MODEL_ID},dtype=float16,trust_remote_code=True"
    if lora_path:
        args += f",peft={lora_path}"
    return args


def run_benchmark(
    config,
    model_label: str,
    lora_path: str = None,
    tasks: list = None,
    limit: int = None,
) -> dict:
    """对指定模型运行 benchmark 评估。

    Args:
        config: 全局配置。
        model_label: 模型标签（"Base" 或 "LoRA"）。
        lora_path: LoRA adapter 路径。
        tasks: lm_eval task 列表（如 ["ceval-valid"]）。
        limit: 每个 task 限制题目数（None = 全量）。

    Returns:
        dict: lm_eval 原始输出，含 results 和 config 字段。
    """
    if not _LM_EVAL_AVAILABLE:
        logger.error("lm-evaluation-harness 未安装！请执行: pip install lm-eval")
        return {"error": "lm_eval_not_installed"}

    if tasks is None:
        tasks = ["ceval-valid"]

    model_args = _build_model_args(config, lora_path)

    logger.info(">>> Benchmark: %s", model_label)
    logger.info("  Model ID:  %s", config.MODEL_ID)
    logger.info("  LoRA:      %s", lora_path or "(none — Base only)")
    logger.info("  Tasks:     %s", tasks)
    logger.info("  Limit:     %s", limit if limit else "全量")
    logger.info("  Model Args: %s", model_args)

    t0 = time.time()
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        limit=limit,
        log_samples=False,          # P0: 不保存逐题细节，减少 I/O
    )
    elapsed = time.time() - t0

    logger.info("  ✅ 完成，耗时 %.0f 秒", elapsed)

    return results


def _build_comparison_table(base_results: dict, lora_results: dict) -> str:
    """根据 Base 和 LoRA 的 benchmark 结果构造对比表。"""
    lines = []
    lines.append("| Task | Metric | Base | LoRA | Δ |")
    lines.append("|------|--------|------|------|-----|")

    all_tasks = set(base_results.get("results", {}).keys()) | set(lora_results.get("results", {}).keys())

    for task in sorted(all_tasks):
        base_r = base_results.get("results", {}).get(task, {})
        lora_r = lora_results.get("results", {}).get(task, {})

        # lm_eval 结果中主指标通常不带后缀或为 acc
        for metric in sorted(set(list(base_r.keys()) + list(lora_r.keys()))):
            # 跳过 stderr 指标
            if metric.endswith("_stderr"):
                continue
            b_val = base_r.get(metric)
            l_val = lora_r.get(metric)

            if b_val is None and l_val is None:
                continue
            b_str = f"{b_val:.4f}" if isinstance(b_val, (int, float)) else "N/A"
            l_str = f"{l_val:.4f}" if isinstance(l_val, (int, float)) else "N/A"

            if isinstance(b_val, (int, float)) and isinstance(l_val, (int, float)):
                delta = l_val - b_val
                delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            else:
                delta_str = "-"
            lines.append(f"| {task} | {metric} | {b_str} | {l_str} | {delta_str} |")

    return "\n".join(lines)


def generate_benchmark_report(
    config,
    base_results: dict,
    lora_results: dict,
    output_dir: Path,
) -> str:
    """生成 benchmark 对比报告（Markdown）。"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    tasks_run = list(base_results.get("results", {}).keys())

    comparison_table = _build_comparison_table(base_results, lora_results)

    # ── 综合判定 ──
    verdict_parts = []
    for task in tasks_run:
        base_r = base_results.get("results", {}).get(task, {})
        lora_r = lora_results.get("results", {}).get(task, {})
        # 找 acc 类指标
        acc_metrics = [k for k in base_r.keys() if "acc" in k.lower() and not k.endswith("_stderr")]
        for metric in acc_metrics:
            b_val = base_r.get(metric, 0)
            l_val = lora_r.get(metric, 0)
            delta = l_val - b_val
            if delta < -0.05:
                verdict_parts.append(f"❌ {task}/{metric} 下降 {delta:.3f} — 可能发生灾难性遗忘")
            elif delta < -0.02:
                verdict_parts.append(f"⚠️ {task}/{metric} 轻微下降 {delta:.3f}")
            else:
                verdict_parts.append(f"✅ {task}/{metric} 持平或提升 {delta:+.3f}")

    verdict = "\n".join(f"- {v}" for v in verdict_parts) if verdict_parts else "- 无有效评估数据"

    report = f"""# Benchmark 评测报告 — 蒸馏前后通用能力对比

> 生成时间：{now}
> 基座模型：{config.MODEL_ID}
> LoRA 路径：{config.OUTPUT_DIR}
> 评估任务：{', '.join(tasks_run)}
> 评估框架：[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

---

## 对比结果

{comparison_table}

---

## 综合判定

{verdict}

---

## 解读指南

- **C-Eval**: 中文多学科选择题（52 个学科，1,346 题验证集）。衡量模型在中文世界的通识能力。
- **MMLU**: 英文多学科选择题（57 个学科）。衡量模型的英文通识能力。
- **GSM8K**: 小学数学应用题，衡量数学推理能力。
- **HellaSwag**: 常识推理，衡量逻辑推断能力。

**Δ 值含义**:
- 正值：LoRA 模型在该维度上比 Base 更强（罕见但可能）
- 接近 0（±2% 内）：蒸馏未损害该维度的通用能力
- 负值超过 5%：疑似灾难性遗忘，需要关注

---

*报告由 scripts/benchmark.py 自动生成*
*数据来源: {output_dir}/benchmark_results.json*
"""
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 评测 — 蒸馏前后通用能力对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                      # 完整 C-Eval
  %(prog)s --tasks ceval --limit 20             # 快速验证 20 题
  %(prog)s --tasks ceval,mmlu --limit 50        # 多 benchmark
  %(prog)s --base-only                           # 仅跑 Base（跳过 LoRA）
  %(prog)s --lora-only                           # 仅跑 LoRA（跳过 Base）
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--tasks", default="ceval",
        help="逗号分隔的 task 列表: ceval,mmlu,gsm8k,hellaswag"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="每个 task 限制题目数（默认全量）")
    parser.add_argument("--base-only", action="store_true", help="仅跑 Base 模型")
    parser.add_argument("--lora-only", action="store_true", help="仅跑 LoRA 模型")
    parser.add_argument("--output-dir", default=None, help="结果输出目录（默认自动检测）")
    args = parser.parse_args()

    # ── 环境初始化 ──
    cfg = load_config(args.config)
    setup_environment(cfg)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not _LM_EVAL_AVAILABLE:
        logger.error(
            "lm-evaluation-harness 未安装！\n"
            "请执行: pip install lm-eval\n"
            "安装文档: https://github.com/EleutherAI/lm-evaluation-harness"
        )
        sys.exit(1)

    # ── 解析 tasks ──
    task_names = [t.strip() for t in args.tasks.split(",")]
    lm_eval_tasks = []
    for name in task_names:
        mapped = _SUPPORTED_TASKS.get(name, name)
        lm_eval_tasks.append(mapped)
    logger.info("Benchmark tasks: %s → %s", task_names, lm_eval_tasks)

    # ── 输出目录 ──
    output_dir = Path(args.output_dir) if args.output_dir else _get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("输出目录: %s", output_dir)

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "model_id": cfg.MODEL_ID,
               "lora_path": cfg.OUTPUT_DIR,
               "tasks": lm_eval_tasks,
               "limit": args.limit}

    # ── Base 模型 ──
    if not args.lora_only:
        logger.info("=" * 50)
        logger.info("  Base 模型 Benchmark")
        logger.info("=" * 50)
        base_results = run_benchmark(cfg, "Base", lora_path=None, tasks=lm_eval_tasks, limit=args.limit)
        results["base"] = base_results
    else:
        base_results = None

    # ── LoRA 模型 ──
    if not args.base_only:
        logger.info("=" * 50)
        logger.info("  LoRA 模型 Benchmark")
        logger.info("=" * 50)
        lora_results = run_benchmark(cfg, "LoRA", lora_path=cfg.OUTPUT_DIR, tasks=lm_eval_tasks, limit=args.limit)
        results["lora"] = lora_results
    else:
        lora_results = None

    # ── 保存 JSON ──
    json_path = output_dir / "benchmark_results.json"
    # 将非 JSON 可序列化的部分过滤
    def _safe_serialize(obj):
        if isinstance(obj, dict):
            return {k: _safe_serialize(v) for k, v in obj.items() if not k.startswith("_")}
        elif isinstance(obj, list):
            return [_safe_serialize(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    json_path.write_text(
        json.dumps(_safe_serialize(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("结构化结果已保存: %s", json_path.resolve())

    # ── 生成报告 ──
    if base_results and lora_results:
        report = generate_benchmark_report(cfg, base_results, lora_results, output_dir)
        report_path = output_dir / "benchmark_report.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Benchmark 报告已保存: %s", report_path.resolve())

        # ── 终端输出对比表 ──
        print("\n" + "=" * 60)
        print("  Benchmark 对比摘要")
        print("=" * 60)
        print(_build_comparison_table(base_results, lora_results))
        print("=" * 60)
    elif base_results:
        logger.info("仅 Base 模式，跳过对比报告")
    elif lora_results:
        logger.info("仅 LoRA 模式，跳过对比报告")


if __name__ == "__main__":
    main()
