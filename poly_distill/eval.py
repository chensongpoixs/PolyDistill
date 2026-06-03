"""
训练质量评估模块。

评估维度：
  1. Perplexity (PPL) — 模型对测试文本的困惑度，越低越好
  2. ROUGE-L      — 生成答案与参考答案的最长公共子序列重叠度
  3. 生成样本对比  — base vs lora 的实际输出并排展示
  4. LLM-as-Judge — 外部大模型多维度打分（准确性/相关性/完整性/整体质量）

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

# LLM-as-Judge 评分维度
_JUDGE_CRITERIA = {
    "accuracy": "准确性 — 回答内容技术正确、无事实错误",
    "relevance": "相关性 — 回答紧扣问题，不偏离主题",
    "completeness": "完整性 — 回答覆盖问题的关键要点，无重大遗漏",
    "overall": "整体质量 — 综合评判回答的专业性和可用性",
}


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

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "reasoning_content": thinking, "content": response},
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
# 4. LLM-as-Judge 评估（大模型打分）
# ============================================================
def _build_judge_prompt(question: str, reference: str, generated: str) -> str:
    """构造给裁判模型的质量评估 prompt。"""
    criteria_text = "\n".join(
        f"  {i+1}. {name} — {desc}"
        for i, (name, desc) in enumerate(_JUDGE_CRITERIA.items())
    )
    return f"""你是一位 AI Infra 领域技术专家。请对以下模型的回答进行质量评估。

【问题】
{question}

【参考答案（教师模型）】
{reference}

【待评估回答（学生模型）】
{generated}

请从以下维度打分（1-5 分，5 分为最优）：
{criteria_text}

请以 JSON 格式输出（只输出 JSON，不要其他文字）：
{{
  "accuracy": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "overall": <1-5>,
  "comment": "<一句话中文点评，指出主要优点和不足>"
}}"""


def _call_llm_judge(
    endpoint: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout: int = 60,
) -> dict:
    """调用外部 LLM API 进行质量评分。

    兼容 OpenAI Chat Completions API 格式。
    使用标准库 urllib，无需额外依赖。
    """
    import urllib.request
    import urllib.error

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a technical quality evaluator. Always respond in Chinese JSON format."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }).encode("utf-8")

    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()

        # 尝试提取 JSON（模型可能在 JSON 前后加了说明文字）
        # 找第一个 { 和最后一个 }
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and start < end:
            return json.loads(content[start:end + 1])
        else:
            logger.warning("LLM Judge 返回非 JSON 格式: %s...", content[:200])
            return {"error": "non_json_response", "raw": content[:500]}
    except urllib.error.HTTPError as e:
        logger.error("LLM Judge HTTP %d: %s", e.code, e.read().decode("utf-8", errors="replace")[:500])
        return {"error": f"http_{e.code}"}
    except urllib.error.URLError as e:
        logger.error("LLM Judge 连接失败: %s", e.reason)
        return {"error": "connection_failed", "detail": str(e.reason)}
    except json.JSONDecodeError as e:
        logger.error("LLM Judge JSON 解析失败: %s", e)
        return {"error": "json_parse_error"}
    except Exception as e:
        logger.error("LLM Judge 未知错误: %s", e)
        return {"error": "unknown", "detail": str(e)}


def evaluate_with_llm_judge(
    config: Config,
    base_samples: list,
    lora_samples: list,
) -> dict:
    """使用外部 LLM 对学生模型生成质量进行多维度评分。

    Args:
        config: 全局配置（含 LLM Judge 连接参数）。
        base_samples: Base 模型生成样本 [{"instruction", "reference", "generated"}, ...]。
        lora_samples: LoRA 模型生成样本。

    Returns:
        {
            "enabled": bool,
            "model": str,
            "num_samples": int,
            "lora_scores": {"accuracy": avg, "relevance": avg, ...},
            "base_scores": {...},
            "details": [{"instruction": str, "scores": {...}, "comment": str}, ...],
        }
    """
    if not config.EVAL_LLM_JUDGE_ENABLED:
        return {"enabled": False}

    # API Key 优先级：config 值 > 环境变量
    api_key = config.EVAL_LLM_JUDGE_API_KEY or os.environ.get("LLM_JUDGE_API_KEY", "")
    if not api_key:
        logger.warning("LLM Judge 已启用但未配置 api_key，跳过")
        return {"enabled": False, "error": "no_api_key"}

    logger.info("=" * 60)
    logger.info("  LLM-as-Judge 质量评估")
    logger.info("  Endpoint: %s", config.EVAL_LLM_JUDGE_ENDPOINT)
    logger.info("  Model:    %s", config.EVAL_LLM_JUDGE_MODEL)
    logger.info("=" * 60)

    n_samples = min(config.EVAL_LLM_JUDGE_MAX_SAMPLES, len(lora_samples))
    samples_to_score = lora_samples[:n_samples]
    # 匹配 base 样本
    base_map = {}
    for b in base_samples:
        base_map[b["instruction"]] = b["generated"]

    all_lora_scores = {key: [] for key in _JUDGE_CRITERIA}
    all_base_scores = {key: [] for key in _JUDGE_CRITERIA}
    details = []

    for i, sample in enumerate(samples_to_score):
        question = sample["instruction"]
        reference = sample["reference"]
        lora_gen = sample["generated"]
        base_gen = base_map.get(question, "")

        logger.info("  [%d/%d] 评估: %s...", i + 1, n_samples, question[:60])

        # 评估 LoRA 模型
        prompt_lora = _build_judge_prompt(question, reference, lora_gen)
        lora_result = _call_llm_judge(
            config.EVAL_LLM_JUDGE_ENDPOINT,
            config.EVAL_LLM_JUDGE_MODEL,
            api_key,
            prompt_lora,
        )

        time.sleep(0.5)  # 避免 API 限流

        # 评估 Base 模型
        prompt_base = _build_judge_prompt(question, reference, base_gen)
        base_result = _call_llm_judge(
            config.EVAL_LLM_JUDGE_ENDPOINT,
            config.EVAL_LLM_JUDGE_MODEL,
            api_key,
            prompt_base,
        )

        # 收集分数
        for key in _JUDGE_CRITERIA:
            if isinstance(lora_result.get(key), (int, float)):
                all_lora_scores[key].append(lora_result[key])
            if isinstance(base_result.get(key), (int, float)):
                all_base_scores[key].append(base_result[key])

        details.append({
            "instruction": question[:200],
            "reference": reference[:300],
            "lora_generated": lora_gen[:300],
            "base_generated": base_gen[:300],
            "lora_scores": {k: lora_result.get(k) for k in _JUDGE_CRITERIA},
            "base_scores": {k: base_result.get(k) for k in _JUDGE_CRITERIA},
            "lora_comment": lora_result.get("comment", ""),
            "base_comment": base_result.get("comment", ""),
        })

    # 计算平均分
    lora_avg = {k: round(sum(v) / len(v), 2) if v else 0.0 for k, v in all_lora_scores.items()}
    base_avg = {k: round(sum(v) / len(v), 2) if v else 0.0 for k, v in all_base_scores.items()}

    logger.info("  LoRA 均分: %s", json.dumps(lora_avg, ensure_ascii=False))
    logger.info("  Base 均分: %s", json.dumps(base_avg, ensure_ascii=False))

    return {
        "enabled": True,
        "model": config.EVAL_LLM_JUDGE_MODEL,
        "endpoint": config.EVAL_LLM_JUDGE_ENDPOINT,
        "num_samples": n_samples,
        "lora_scores": lora_avg,
        "base_scores": base_avg,
        "details": details,
    }


# ============================================================
# 5. 报告生成
# ============================================================
def generate_report(
    config: Config,
    ppl_results: dict,
    rouge_results: dict,
    base_samples: list,
    lora_samples: list,
    llm_judge_results: dict = None,
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

    # ---- LLM-as-Judge 评估结果 ----
    llm_judge_section = ""
    if llm_judge_results and llm_judge_results.get("enabled"):
        jr = llm_judge_results
        criteria_names = list(_JUDGE_CRITERIA.keys())
        # 综合均分对比表
        header = "| 维度 | Base | LoRA | Δ |\n|------|------|------|-----|\n"
        rows = ""
        for k in criteria_names:
            b = jr.get("base_scores", {}).get(k, "-")
            l = jr.get("lora_scores", {}).get(k, "-")
            if isinstance(b, (int, float)) and isinstance(l, (int, float)):
                delta = l - b
                delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            else:
                delta_str = "-"
            rows += f"| {k} | {b} | {l} | {delta_str} |\n"

        # 逐条详情
        detail_blocks = ""
        for i, d in enumerate(jr.get("details", [])):
            lora_comment = d.get("lora_comment", "")
            base_comment = d.get("base_comment", "")
            detail_blocks += f"""### Judge 样本 {i + 1}

**问题**: {d["instruction"]}

| 维度 | Base | LoRA |
|------|------|------|
"""
            for k in criteria_names:
                bs = d.get("base_scores", {}).get(k, "-")
                ls = d.get("lora_scores", {}).get(k, "-")
                detail_blocks += f"| {k} | {bs} | {ls} |\n"
            detail_blocks += f"""
> 🔴 **Base 点评**: {base_comment}

> 🟢 **LoRA 点评**: {lora_comment}

---
"""

        llm_judge_section = f"""## 4. LLM-as-Judge 质量评估

> 裁判模型：{jr.get("model", "N/A")}
> 评估样本数：{jr.get("num_samples", 0)}
> 评分维度：准确性、相关性、完整性、整体质量（1-5 分）

### 综合均分对比

{header}{rows}

> **Δ (LoRA - Base)**: 正值为提升，负值为退化。

### 逐条详细评分

{detail_blocks}

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

    # ---- LLM-as-Judge 评估 ----
    llm_judge_results = evaluate_with_llm_judge(config, base_gen_samples, lora_gen_samples)

    # ---- 生成报告 ----
    report = generate_report(config, ppl_results, rouge_results, base_gen_samples, lora_gen_samples, llm_judge_results)

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
        "llm_judge": llm_judge_results,
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
    import sys
    # 确保项目根目录在搜索路径中（python poly_distill/eval.py 运行时需要）
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
