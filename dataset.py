"""
数据集加载与预处理。

负责从 JSON 文件加载 instruction-output 对，
并通过 Qwen chat_template 将其转换为训练可直接使用的对话文本。
"""

from datasets import load_dataset
from transformers import PreTrainedTokenizer

from config import Config


def load_and_prepare_data(config: Config, tokenizer: PreTrainedTokenizer) -> "Dataset":
    """加载 JSON 数据集，并为每条样本生成 chat_template 格式化文本。

    Args:
        config: 全局配置对象（提供 DATA_FILE、SYSTEM_PROMPT）。
        tokenizer: 已加载的分词器（提供 apply_chat_template 方法）。

    Returns:
        datasets.Dataset: 仅包含 "text" 字段的数据集。

    数据格式要求（ai_infra_audio_video.json）：
        [
            {
                "instruction": "面试问题...",
                "input": "",         # 可选，本数据集未使用
                "output": "参考答案..."
            },
            ...
        ]
    """
    # 加载 JSON 文件，split="train" 直接返回 Dataset 对象而非 DatasetDict
    dataset = load_dataset("json", data_files=config.DATA_FILE, split="train")
    print(f"✅ 数据集加载完成，共 {len(dataset)} 条样本")

    # 保留原始列名（用于后续移除），因为 map 后会新增 "text" 列
    original_columns = dataset.column_names

    # 预处理：将 instruction + output 转换为 Qwen 对话格式的完整文本
    dataset = dataset.map(
        lambda x: {"text": _format_conversation(x, config, tokenizer)}
    )
    # 仅移除原始字段，保留 "text"（避免 SFTTrainer 因多余字段警告）
    dataset = dataset.remove_columns(original_columns)
    return dataset


def _format_conversation(
    example: dict, config: Config, tokenizer: PreTrainedTokenizer
) -> str:
    """将单条样本格式化为 Qwen chat_template 字符串。

    对话结构：
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {instruction}<|im_end|>
        <|im_start|>assistant
        {output}<|im_end|>

    注意：
      add_generation_prompt=False：训练时不需要模型继续生成，因此不加生成提示标记。
    """
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    # 复用 tokenizer 的 chat_template 确保格式与推理时完全一致
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
