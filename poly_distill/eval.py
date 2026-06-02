"""
训练质量评估模块。

评估维度：
  1. Perplexity (PPL) — 模型对测试文本的困惑度，越低越好
  2. ROUGE-L      — 生成答案与参考答案的最长公共子序列重叠度
  3. 生成样本对比  — base vs lora 的实际输出并排展示

输出文件：
  - eval_report.md   : 可读的 Markdown 评测报告
  - eval_results.json : 结构化原始数据（供后续程序化分析）
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import concatenate_datasets, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from poly_distill.config import Config

logger = logging.getLogger(__name__)


# ============================================================
# 0. 数据格式标准化
# ============================================================
def _get_question(example: dict) -> str:
    """从 Parquet 样本中提取用户问题。

    Parquet messages 列表中 role=user 的 content。
    """
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _get_reference(example: dict) -> str:
    """从 Parquet 样本中提取参考答案（response 列）。"""
    return example.get("response", "")


def _build_messages(example: dict, config: Config) -> list:
    """从 Parquet 行构造完整对话消息列表。

    Parquet schema: {messages, thinking, response, system}

    返回统一的 [system, user, assistant] 消息列表。
    """
    system = example.get("system") or config.SYSTEM_PROMPT

    user_content = ""
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    thinking = example.get("thinking") or ""
    response = example.get("response", "")
    assistant_content = f"{thinking}\n\n{response}" if thinking else response

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# ============================================================
# 1. Perplexity 评估
# ============================================================
def evaluate_perplexity(
    model,
    tokenizer: PreTrainedTokenizer,
    config: Config,
    samples: list,
    label: str,
) -> dict:
    """计算模型在给定样本集上的困惑度（PPL）。

    PPL = exp(cross_entropy_loss)。
    数值越低，模型对该文本越"不惊讶"，即拟合越好。
    但 PPL 低 ≠ 生成质量高，需结合其他指标综合判断。

    Args:
        model: 待评估模型（base 或 LoRA）。
        tokenizer: 分词器。
        config: 全局配置。
        samples: 样本列表，每条含 instruction / output。
        label: 模型标签（用于输出）。

    Returns:
        {"avg_ppl": float, "avg_loss": float, "num_samples": int}
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for example in samples:
            # 构造与训练一致的对话格式（兼容 JSON 和 Parquet 两种格式）
            messages = _build_messages(example, config)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # 前向传播计算 loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    avg_ppl = torch.exp(torch.tensor(avg_loss)).item()

    logger.info("  [%s] Loss=%.4f  PPL=%.2f  (tokens=%d)", label, avg_loss, avg_ppl, total_tokens)
    return {"label": label, "avg_ppl": round(avg_ppl, 2), "avg_loss": round(avg_loss, 4)}


# ============================================================
# 2. ROUGE-L 评估
# ============================================================
def _lcs_length(a: list, b: list) -> int:
    """最长公共子序列（Longest Common Subsequence）长度。

    动态规划，O(m*n)。用于计算 ROUGE-L。
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_score(reference: str, candidate: str) -> Tuple[float, float, float]:
    """计算 ROUGE-L 分数。

    ROUGE-L 基于最长公共子序列（LCS）计算：
      R_lcs = LCS_len / ref_len   （召回率）
      P_lcs = LCS_len / cand_len  （精确率）
      F_lcs = 2 * R * P / (R + P) （F1）

    对中文文本按字符切分（逐字比较），无需分词。
    """
    ref_chars = list(reference)
    cand_chars = list(candidate)
    lcs_len = _lcs_length(ref_chars, cand_chars)

    if len(ref_chars) == 0 or len(cand_chars) == 0:
        return 0.0, 0.0, 0.0

    recall = lcs_len / len(ref_chars)
    precision = lcs_len / len(cand_chars)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return recall, precision, f1


def evaluate_rouge(
    model,
    tokenizer: PreTrainedTokenizer,
    config: Config,
    samples: list,
    label: str,
) -> dict:
    """生成回答并与参考答案计算 ROUGE-L F1。

    Args:
        model: 待评估模型。
        tokenizer: 分词器。
        config: 全局配置。
        samples: 样本列表。
        label: 模型标签。

    Returns:
        {"label": str, "rouge_l_f1": float, "num_samples": int}
    """
    model.eval()
    scores = []

    for example in samples:
        # 生成
        question = _get_question(example)
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                temperature=config.EVAL_TEMPERATURE,
                do_sample=config.EVAL_TEMPERATURE > 0,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取 assistant 回复部分
        if "<|im_start|>assistant" in generated:
            generated = generated.split("<|im_start|>assistant")[-1].strip()

        reference = _get_reference(example)
        _, _, f1 = rouge_l_score(reference, generated)
        scores.append(f1)

    avg_f1 = sum(scores) / len(scores) if scores else 0.0
    logger.info("  [%s] ROUGE-L F1=%.4f  (samples=%d)", label, avg_f1, len(scores))
    return {"label": label, "rouge_l_f1": round(avg_f1, 4), "num_samples": len(scores)}


# ============================================================
# 3. 生成样本收集
# ============================================================
def collect_generation_samples(
    model,
    tokenizer: PreTrainedTokenizer,
    config: Config,
    samples: list,
    label: str,
    n_show: int = 5,
) -> list:
    """对少量样本生成回答，用于报告中并排对比。

    Args:
        n_show: 在报告中展示的样本数量（从 samples 中取前 N 条）。

    Returns:
        [{"instruction": str, "reference": str, "generated": str}, ...]
    """
    model.eval()
    results = []
    show_samples = samples[:n_show]

    for example in show_samples:
        question = _get_question(example)
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                temperature=config.EVAL_TEMPERATURE,
                do_sample=config.EVAL_TEMPERATURE > 0,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|im_start|>assistant" in generated:
            generated = generated.split("<|im_start|>assistant")[-1].strip()

        results.append({
            "instruction": question,
            "reference": _get_reference(example),
            "generated": generated,
        })

    return results


# ============================================================
# 4. 报告生成
# ============================================================
def generate_report(
    config: Config,
    ppl_results: dict,
    rouge_results: dict,
    base_samples: list,
    lora_samples: list,
) -> str:
    """生成 Markdown 格式的评测报告字符串。"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---- PPL 对比表 ----
    ppl_rows = ""
    for r in ppl_results:
        ppl_rows += f"| {r['label']} | {r['avg_loss']:.4f} | {r['avg_ppl']:.2f} |\n"

    # ---- ROUGE 对比表 ----
    rouge_rows = ""
    for r in rouge_results:
        rouge_rows += f"| {r['label']} | {r['rouge_l_f1']:.4f} | {r['num_samples']} |\n"

    # ---- 生成样本并排对比 ----
    sample_blocks = ""
    for i, (bs, ls) in enumerate(zip(base_samples, lora_samples)):
        # 截断过长的文本
        ref_short = bs["reference"][:300] + ("..." if len(bs["reference"]) > 300 else "")
        base_short = bs["generated"][:500] + ("..." if len(bs["generated"]) > 500 else "")
        lora_short = ls["generated"][:500] + ("..." if len(ls["generated"]) > 500 else "")

        sample_blocks += f"""### 样本 {i + 1}

**问题**: {bs["instruction"]}

| 角色 | 内容 |
|------|------|
| 📝 参考答案 | {ref_short} |
| 🔴 Base 模型 | {base_short} |
| 🟢 LoRA 模型 | {lora_short} |

---

"""

    report = f"""# AI Infra LoRA SFT 训练评测报告

> 生成时间：{now}
> 基座模型：{config.MODEL_ID}
> LoRA 输出：{config.OUTPUT_DIR}
> 评估样本数：{config.EVAL_NUM_SAMPLES}

---

## 1. Perplexity (PPL) 对比

PPL 衡量模型对文本的"惊讶程度"，越低越好。
但 PPL 低 ≠ 生成质量高，需结合 ROUGE 和人工评估。

| 模型 | Avg Loss | Avg PPL |
|------|----------|---------|
{ppl_rows}

> **解读**: 若 LoRA 模型 PPL 远低于 Base，说明模型已学到数据分布特征。
> 若 LoRA 模型 PPL 接近于 1，可能是过拟合信号。

---

## 2. ROUGE-L F1 对比

ROUGE-L 基于最长公共子序列（LCS），衡量生成答案与参考答案的字符级重叠度。
F1 越高，生成结果越接近参考答案。

| 模型 | ROUGE-L F1 | 样本数 |
|------|-----------|--------|
{rouge_rows}

> **解读**: ROUGE-L 对开放式长回答不太敏感（答案可以不同但都正确）。
> F1 > 0.3 通常表示内容有实质重叠。配合人工抽查判断质量。

---

## 3. 生成样本对比

以下对比 Base 模型与 LoRA 模型对相同问题的回答（前 5 条）。

{sample_blocks}

## 4. 结论与建议

| 指标 | 判断标准 | 是否合格 |
|------|---------|---------|
| PPL 下降 | LoRA PPL < Base PPL | 待填充 |
| ROUGE-L > 0.2 | 生成与参考答案有实质重叠 | 待填充 |
| 过拟合检测 | PPL 未接近 1.0 | 待填充 |

> **下一步建议**:
> 1. 若 ROUGE-L 过低而 PPL 正常 → 模型学到了风格但未学到内容 → 增加 epoch 或降低 lr
> 2. 若 PPL 接近 1.0 → 严重过拟合 → 降低 epoch、增大 dropout、减少 LoRA rank
> 3. 若 PPL 不降 → 学习率过低或数据格式有误 → 检查 chat_template
> 4. 推荐引入 LLM-as-Judge（GPT-4/Claude）做多维度打分，替代纯 ROUGE

---

*报告由 eval.py 自动生成*
"""
    return report


# ============================================================
# 5. 主评估流程
# ============================================================
def run_evaluation(config: Config, tokenizer: PreTrainedTokenizer) -> None:
    """执行完整评估流水线，输出 eval_report.md 和 eval_results.json。

    步骤：
      1. 加载数据集并选取评估样本
      2. 加载 Base 模型 → 计算 PPL / ROUGE / 收集生成样本
      3. 加载 LoRA 模型 → 计算 PPL / ROUGE / 收集生成样本
      4. 生成 Markdown 报告 + JSON 结果
    """
    logger.info("=" * 60)
    logger.info("  训练质量评估")
    logger.info("=" * 60)

    # ---- 加载评估数据（仅 Parquet 格式） ----
    data_dir = Path(config.DATA_DIR)
    data_files = sorted(data_dir.glob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(
            f"数据目录下未找到 .parquet 文件: {data_dir}\n"
            f"请先运行: python poly_distill/json_to_parquet.py"
        )
    logger.info("评估数据: Parquet, 文件数: %d", len(data_files))

    datasets_list = []
    for f in data_files:
        ds = load_dataset("parquet", data_files=str(f), split="train")
        datasets_list.append(ds)
    dataset = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    n_total = len(dataset)

    # 选取子集（随机但固定种子，保证可复现）
    n_eval = config.EVAL_NUM_SAMPLES if config.EVAL_NUM_SAMPLES > 0 else n_total
    n_eval = min(n_eval, n_total)
    rng = random.Random(42)  # 固定种子，结果可复现
    indices = rng.sample(range(n_total), n_eval)
    eval_samples = [dataset[i] for i in indices]
    logger.info("评估数据集: %d / %d 条样本（随机种子=42）", n_eval, n_total)

    # ---- 加载 Base 模型 ----
    logger.info("--- 加载 Base 模型 ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # ---- Base 评估 ----
    logger.info(">>> Perplexity 评估")
    ppl_results = []
    ppl_results.append(
        evaluate_perplexity(base_model, tokenizer, config, eval_samples, "Base")
    )

    logger.info(">>> ROUGE-L 评估")
    rouge_results = []
    rouge_results.append(
        evaluate_rouge(base_model, tokenizer, config, eval_samples, "Base")
    )

    logger.info(">>> 收集 Base 模型生成样本")
    base_gen_samples = collect_generation_samples(
        base_model, tokenizer, config, eval_samples, "Base", n_show=5
    )

    # ---- 释放 Base 模型显存 ----
    del base_model
    torch.cuda.empty_cache()

    # ---- 加载 LoRA 模型 ----
    logger.info("--- 加载 LoRA 模型 ---")
    lora_base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    lora_model = PeftModel.from_pretrained(lora_base, config.OUTPUT_DIR)

    # ---- LoRA 评估 ----
    logger.info(">>> Perplexity 评估")
    ppl_results.append(
        evaluate_perplexity(lora_model, tokenizer, config, eval_samples, "LoRA")
    )

    logger.info(">>> ROUGE-L 评估")
    rouge_results.append(
        evaluate_rouge(lora_model, tokenizer, config, eval_samples, "LoRA")
    )

    logger.info(">>> 收集 LoRA 模型生成样本")
    lora_gen_samples = collect_generation_samples(
        lora_model, tokenizer, config, eval_samples, "LoRA", n_show=5
    )

    # ---- 生成报告 ----
    report = generate_report(config, ppl_results, rouge_results, base_gen_samples, lora_gen_samples)

    report_path = Path(config.EVAL_REPORT_PATH)
    report_path.write_text(report, encoding="utf-8")
    logger.info("评测报告已保存: %s", report_path.resolve())

    # ---- 保存 JSON 结构化结果 ----
    json_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": config.MODEL_ID,
        "lora_output": config.OUTPUT_DIR,
        "eval_num_samples": n_eval,
        "perplexity": ppl_results,
        "rouge_l": rouge_results,
        "generation_samples": [
            {"instruction": s["instruction"], "reference": s["reference"][:200],
             "base": s["generated"][:200], "lora": l["generated"][:200]}
            for s, l in zip(base_gen_samples, lora_gen_samples)
        ],
    }
    json_path = Path(config.EVAL_JSON_PATH)
    json_path.write_text(
        json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("结构化结果已保存: %s", json_path.resolve())

    # ---- 终端摘要 ----
    logger.info("=" * 60)
    logger.info("  评估摘要")
    logger.info("=" * 60)
    for r in ppl_results:
        logger.info("  PPL  [%s]: %.2f  (loss=%.4f)", r["label"], r["avg_ppl"], r["avg_loss"])
    for r in rouge_results:
        logger.info("  ROUGE [%s]: %.4f", r["label"], r["rouge_l_f1"])
    logger.info("=" * 60)


if __name__ == "__main__":
    from poly_distill.config import load_config, setup_environment

    cfg = load_config()
    setup_environment(cfg)

    # 仅推理评估，不重新训练——需要先加载 tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_ID, use_fast=True, trust_remote_code=True, cache_dir=cfg.CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_evaluation(cfg, tokenizer)
