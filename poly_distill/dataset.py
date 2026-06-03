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
