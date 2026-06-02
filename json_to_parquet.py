"""
JSON → Parquet 格式转换脚本。

将项目当前 JSON 数据集转换为与 reasoning-distill 兼容的 Parquet 格式。

输入格式 (JSON):
    [
        {
            "instruction": "用户问题...",
            "input": "",            # 可选
            "thinking": "思考过程...", # 可选
            "output": "最终回答..."
        },
        ...
    ]

输出格式 (Parquet，参考 Claude Opus 4.6 Reasoning Distill):
    Columns:
      - source_dataset: str       # 数据集来源标识
      - source_idx: int64         # 在源数据集中的行号
      - system: str               # 系统提示词
      - messages: list<struct>    # 对话消息（仅 user 角色）
      - thinking: str             # 推理/思考过程
      - response: str             # 最终回答
      - stop_reason: str          # 停止原因
      - usage: struct             # token 用量统计（占位）
      - model: str                # 生成模型标识

用法:
    python json_to_parquet.py                              # 默认: data/ → data/ 目录
    python json_to_parquet.py --input ./data --output ./parquet_data
"""

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 配置
# ============================================================
SOURCE_DATASET = "ai_infra_knowledge_distill"
SYSTEM_PROMPT = "You are a helpful assistant."
MODEL = "claude-opus-4-7"  # 标记为参考模型蒸馏数据
STOP_REASON = "end_turn"

# 空的 usage 结构（占位，与参考格式对齐）
EMPTY_USAGE = {
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "inference_geo": "",
    "input_tokens": 0,
    "output_tokens": 0,
    "server_tool_use": None,
    "service_tier": "",
}


def convert_json_to_parquet(input_dir: str, output_dir: str) -> None:
    """遍历输入目录下所有 JSON 文件，合并后输出单个 Parquet 文件。

    Args:
        input_dir: 包含 JSON 文件的数据目录。
        output_dir: Parquet 输出目录。
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- 收集所有 JSON 文件 ----
    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录 {input_dir} 下未找到 .json 文件")

    print(f"📁 输入目录: {input_dir}")
    print(f"📄 发现 {len(json_files)} 个 JSON 文件")

    # ---- 合并并转换 ----
    all_rows = []
    global_idx = 0

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            instruction = item.get("instruction", "")
            thinking = item.get("thinking") or item.get("reasoning") or ""
            response = item.get("output", "")
            system = item.get("system", SYSTEM_PROMPT)

            # 构造 messages 列：仅包含 user 消息
            messages = [{"role": "user", "content": instruction}]

            all_rows.append({
                "source_dataset": SOURCE_DATASET,
                "source_idx": global_idx,
                "system": system,
                "messages": messages,
                "thinking": thinking,
                "response": response,
                "stop_reason": STOP_REASON,
                "usage": EMPTY_USAGE,
                "model": MODEL,
            })
            global_idx += 1

        print(f"   ✅ {json_file.name}: {len(data)} 条")

    print(f"\n总计: {len(all_rows)} 条样本")

    # ---- 显式定义 Schema（确保与参考格式一致） ----
    schema = pa.schema([
        ("source_dataset", pa.string()),
        ("source_idx", pa.int64()),
        ("system", pa.string()),
        ("messages", pa.list_(
            pa.struct([
                ("content", pa.string()),
                ("role", pa.string()),
            ])
        )),
        ("thinking", pa.string()),
        ("response", pa.string()),
        ("stop_reason", pa.string()),
        ("usage", pa.struct([
            ("cache_creation", pa.struct([
                ("ephemeral_1h_input_tokens", pa.int64()),
                ("ephemeral_5m_input_tokens", pa.int64()),
            ])),
            ("cache_creation_input_tokens", pa.int64()),
            ("cache_read_input_tokens", pa.int64()),
            ("inference_geo", pa.string()),
            ("input_tokens", pa.int64()),
            ("output_tokens", pa.int64()),
            ("server_tool_use", pa.null()),
            ("service_tier", pa.string()),
        ])),
        ("model", pa.string()),
    ])

    # ---- 写 Parquet ----
    table = pa.Table.from_pylist(all_rows, schema=schema)
    output_file = output_path / "train-00000-of-00001.parquet"
    pq.write_table(
        table,
        str(output_file),
        compression="snappy",
        row_group_size=1000,
    )

    print(f"\n📦 已输出: {output_file.resolve()}")
    print(f"   行数: {len(table):,}")
    print(f"   大小: {os.path.getsize(output_file) / 1024:.1f} KB")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 AI Infra JSON 数据集转为 reasoning-distill Parquet 格式"
    )
    parser.add_argument(
        "--input", type=str, default="./data",
        help="JSON 数据目录（默认: ./data）",
    )
    parser.add_argument(
        "--output", type=str, default="./data",
        help="Parquet 输出目录（默认: ./data）",
    )
    args = parser.parse_args()

    # 验证：输出目录不能和输入目录完全相同（避免覆盖原始 JSON）
    if os.path.realpath(args.input) == os.path.realpath(args.output):
        print("⚠️  输入和输出目录相同，JSON 和 Parquet 文件将共存。")
        print("   原始 JSON 不会被删除，可手动清理。")

    convert_json_to_parquet(args.input, args.output)
