"""
数据集加载与预处理。

支持从目录批量读取：遍历 data_dir 下所有 .json 文件，
自动合并为统一数据集，方便增量扩展训练数据。
"""

from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer

from config import Config


def _collect_json_files(data_dir: str) -> list:
    """收集目录下所有 .json 文件（按文件名排序，确保加载顺序一致）。

    Args:
        data_dir: 数据目录路径。

    Returns:
        排序后的 .json 文件路径列表。

    Raises:
        FileNotFoundError: 目录不存在或目录下无 JSON 文件。
    """
    dir_path = Path(data_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"数据目录下未找到 .json 文件: {data_dir}")

    print(f"📁 数据目录: {data_dir}")
    print(f"📄 发现 {len(json_files)} 个 JSON 文件:")
    for f in json_files:
        print(f"   - {f.name}")
    return json_files


def load_and_prepare_data(config: Config, tokenizer: PreTrainedTokenizer) -> Dataset:
    """从目录加载所有 JSON 文件，合并后应用 chat_template 格式化。

    Args:
        config: 全局配置对象（提供 DATA_DIR、SYSTEM_PROMPT）。
        tokenizer: 已加载的分词器（提供 apply_chat_template 方法）。

    Returns:
        datasets.Dataset: 仅包含 "text" 字段的合并数据集。

    数据格式要求（目录下每个 .json 文件）：
        [
            {
                "instruction": "面试问题...",
                "input": "",         # 可选
                "output": "参考答案..."
            },
            ...
        ]
    """
    json_files = _collect_json_files(config.DATA_DIR)

    # 逐个加载 JSON 文件并合并
    datasets_list = []
    total_samples = 0
    for json_file in json_files:
        ds = load_dataset("json", data_files=str(json_file), split="train")
        n = len(ds)
        total_samples += n
        print(f"   ✅ {json_file.name}: {n} 条")
        datasets_list.append(ds)

    # 合并所有子数据集
    dataset = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    print(f"✅ 数据集合并完成，共 {total_samples} 条样本")

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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
