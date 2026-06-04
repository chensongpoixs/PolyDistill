"""
数据集加载与预处理。

仅支持 Parquet 格式（reasoning-distill schema）。
JSON → Parquet 转换需先运行: python poly_distill/json_to_parquet.py
"""

import logging
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer

from poly_distill.config import Config

logger = logging.getLogger(__name__)


# ============================================================
# 数据质量过滤
# ============================================================
def _apply_quality_filter(dataset: Dataset, config: Config) -> Dataset:
    """对原始数据集应用质量过滤规则。

    蒸馏数据可能包含教师 API 调用产生的低质量样本：
      - API 超时/错误 → 空回答
      - 教师模型"偷懒" → 过短回答（如"是的。"）
      - 教师回答被 context window 截断 → 过长但不完整
      - 数据预处理 bug → 完全重复的样本

    过滤规则（来自 config.yaml distillation.quality_filter）:
      1. skip_empty:         response 为空字符串 → 丢弃
      2. min_response_length: len(response) < 阈值 → 丢弃（过短无实质内容）
      3. max_response_length: len(response) > 阈值 → 截断至阈值（超 context window）
      4. skip_duplicates:    response 完全相同 → 仅保留第一条（精确去重）

    Args:
        dataset: 加载后的 HuggingFace Dataset（含 messages/thinking/response/system 列）。
        config: 全局配置。

    Returns:
        Dataset: 过滤后的数据集。
    """
    if not config.QUALITY_FILTER_ENABLED:
        logger.info("数据质量过滤: 已禁用 (quality_filter.enabled=false)")
        return dataset

    n_before = len(dataset)
    stats = {"empty": 0, "too_short": 0, "truncated": 0, "duplicate": 0}

    # Step 1: 过滤空回答
    if config.QUALITY_FILTER_SKIP_EMPTY:
        original = len(dataset)
        dataset = dataset.filter(
            lambda x: x.get("response") and len(str(x["response"]).strip()) > 0,
            desc="Filtering empty responses",
        )
        stats["empty"] = original - len(dataset)

    # Step 2: 过滤过短回答
    min_len = config.QUALITY_FILTER_MIN_RESPONSE_LENGTH
    if min_len > 0:
        original = len(dataset)
        dataset = dataset.filter(
            lambda x: len(str(x.get("response", ""))) >= min_len,
            desc=f"Filtering short responses (<{min_len} chars)",
        )
        stats["too_short"] = original - len(dataset)

    # Step 3: 截断过长回答（不丢弃，只截断）
    max_len = config.QUALITY_FILTER_MAX_RESPONSE_LENGTH
    if max_len > 0:
        # 先统计将被截断的数量
        n_overlong = len(dataset.filter(
            lambda x: len(str(x.get("response", ""))) > max_len,
        ))

        def _truncate_response(example):
            resp = str(example.get("response", ""))
            if len(resp) > max_len:
                example["response"] = resp[:max_len]
            return example

        dataset = dataset.map(_truncate_response, desc=f"Truncating long responses (>{max_len} chars)")
        stats["truncated"] = n_overlong

    # Step 4: 精确去重（基于 response 内容的 hash）
    if config.QUALITY_FILTER_SKIP_DUPLICATES:
        seen = set()
        keep_indices = []
        for i, example in enumerate(dataset):
            resp_hash = hash(str(example.get("response", "")))
            if resp_hash not in seen:
                seen.add(resp_hash)
                keep_indices.append(i)
        stats["duplicate"] = len(dataset) - len(keep_indices)
        dataset = dataset.select(keep_indices)

    n_after = len(dataset)
    n_removed = n_before - n_after

    # ── 质量统计报告 ──
    logger.info("数据质量过滤:")
    logger.info("  过滤前: %d 条", n_before)
    if stats["empty"]:
        logger.info("  空回答: %d 条", stats["empty"])
    if stats["too_short"]:
        logger.info("  过短 (<%d字): %d 条", min_len, stats["too_short"])
    if stats["truncated"]:
        logger.info("  截断 (>%d字): %d 条", max_len, stats["truncated"])
    if stats["duplicate"]:
        logger.info("  重复: %d 条", stats["duplicate"])
    logger.info("  过滤后: %d 条 (保留 %.1f%%)", n_after, n_after / max(n_before, 1) * 100)

    if n_after == 0:
        raise ValueError(
            "数据质量过滤后剩余 0 条样本！"
            "请检查数据质量或调整 quality_filter 参数（增大 max_response_length / 减小 min_response_length）"
        )

    return dataset


# ============================================================
# 文件发现
# ============================================================
def _collect_parquet_files(data_dir: str) -> list:
    """收集目录下所有 .parquet 文件（按文件名排序）。

    Raises:
        FileNotFoundError: 目录不存在或无 Parquet 文件。
    """
    dir_path = Path(data_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    files = sorted(dir_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"数据目录下未找到 .parquet 文件: {data_dir}\n"
            f"请先运行: python poly_distill/json_to_parquet.py"
        )
    return files


# ============================================================
# 主加载函数
# ============================================================
def load_and_prepare_data(config: Config, tokenizer: PreTrainedTokenizer) -> Dataset:
    """从目录加载所有 Parquet 文件，合并后应用 chat_template 格式化。

    数据格式（reasoning-distill Parquet schema）:
      - messages:  list<{role: str, content: str}>  — 对话消息（仅 user）
      - thinking:  str                              — 推理/思考过程
      - response:  str                              — 最终回答
      - system:    str                              — 系统提示词

    Args:
        config: 全局配置。
        tokenizer: 分词器。

    Returns:
        Dataset: 仅包含 "text" 字段的数据集。
    """
    files = _collect_parquet_files(config.DATA_DIR)

    logger.info("数据目录: %s", config.DATA_DIR)
    logger.info("Parquet 文件数: %d", len(files))
    for f in files:
        logger.info("  - %s", f.name)

    # 加载并合并
    datasets_list = []
    total_samples = 0
    for f in files:
        ds = load_dataset("parquet", data_files=str(f), split="train")
        n = len(ds)
        total_samples += n
        logger.info("  ✅ %s: %d 条", f.name, n)
        datasets_list.append(ds)

    dataset = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    logger.info("数据集加载完成，共 %d 条样本", total_samples)

    # ── 数据质量过滤 ──
    dataset = _apply_quality_filter(dataset, config)

    # 对话格式化
    original_columns = dataset.column_names
    dataset = dataset.map(
        lambda x: {"text": _format_conversation(x, config, tokenizer)}
    )
    dataset = dataset.remove_columns(original_columns)
    return dataset


# ============================================================
# 对话格式化
# ============================================================
def _format_conversation(
    example: dict, config: Config, tokenizer: PreTrainedTokenizer
) -> str:
    """从 Parquet 行构造完整对话文本。

    对话结构:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        {thinking}

        {response}<|im_end|>

    DataCollatorForCompletionOnlyLM 以 "<|im_start|>assistant\n" 为界，
    reasoning_content + response 整段参与 loss 计算。
    """
    system = example.get("system") or config.SYSTEM_PROMPT
    messages_raw = example.get("messages", [])
    thinking = example.get("thinking") or ""
    response = example.get("response", "")

    # 提取第一条 user 消息
    user_content = ""
    for msg in messages_raw:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    # 构造 assistant 内容
    #assistant_content = f"{thinking}\n\n{response}" if thinking else response


    #logger.info("构建对话:\n  system: %s\n  user: %s\n  assistant: %s", system, user_content, assistant_content);
    chat_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "reasoning_content": thinking, "content": response},
    ]
    return tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=False
    )
