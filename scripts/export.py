"""
TinySage-0.6B 模型打包导出。

将 LoRA adapter 合并到基座模型，产出可独立分发和推理的完整模型。

用法:
    python scripts/export.py                        # 默认配置
    python scripts/export.py --output ./my-model    # 自定义输出路径
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# 将项目根目录加入 Python 搜索路径，确保 scripts/ 下运行时能找到 poly_distill 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注意：不在此处导入 torch / transformers / peft 等重模块。
# HF_ENDPOINT 环境变量必须在这些模块初始化之前设置，否则国内镜像不生效。

from poly_distill.config import load_config, setup_environment

logger = logging.getLogger(__name__)


def export_tinysage(config, output_dir: str) -> None:
    """合并 LoRA 权重到基座模型并保存为 TinySage-0.6B。

    步骤：
      1. 加载基座模型（Qwen3-0.6B）
      2. 加载 LoRA adapter
      3. merge_and_unload() 融合权重
      4. 保存完整模型 + tokenizer 到输出目录
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir)
    if output_path.exists():
        logger.warning("输出目录已存在: %s，将覆盖", output_path)

    # ---- 1. 加载基座模型 ----
    logger.info("加载基座模型: %s", config.MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # ---- 2. 加载 LoRA adapter ----
    adapter_path = Path(config.OUTPUT_DIR)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter 不存在: {adapter_path}\n"
            f"请先运行训练: python scripts/train.py"
        )
    logger.info("加载 LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # ---- 3. 合并权重 ----
    logger.info("正在合并 LoRA 权重到基座模型...")
    model = model.merge_and_unload()
    logger.info("合并完成，模型参数量: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    # ---- 4. 加载并保存 tokenizer ----
    logger.info("加载 tokenizer: %s", config.MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 5. 保存完整模型 ----
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("保存模型到: %s", output_path.resolve())
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    # ---- 6. 保存模型卡片 ----
    _write_model_card(output_path, config)

    # 计算模型大小
    total_size = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    )
    logger.info("模型导出完成！总大小: %.1f MB", total_size / (1024 * 1024))


def _write_model_card(output_path: Path, config) -> None:
    """生成 TinySage-0.6B 模型卡片。"""
    card = f"""---
language:
- zh
- en
pipeline_tag: text-generation
tags:
- tiny
- distilled
- qwen
- lora-merged
- ai-infra
base_model: {config.MODEL_ID}
---

# TinySage-0.6B

A multi-teacher distilled small language model for AI infrastructure domain.

- **Base model**: {config.MODEL_ID}
- **Training framework**: [PolyDistill](https://github.com/chensongpoixs/PolyDistill)
- **Teachers**: GPT, Claude, Gemini
- **Distillation method**: Black-box instruction distillation + LoRA SFT (r={config.LORA_R})
- **Domain**: AI Infra (audio/video codecs, streaming protocols, GPU/CUDA, etc.)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TinySage-0.6B",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("TinySage-0.6B", trust_remote_code=True)

messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "解释 CUDA Stream 的并行原理"}},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## License

Same as the base model ({config.MODEL_ID}).
"""
    card_path = output_path / "README.md"
    card_path.write_text(card, encoding="utf-8")
    logger.info("模型卡片已保存: %s", card_path)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinySage-0.6B 模型打包导出")
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML 配置文件路径（默认: ./config.yaml）",
    )
    parser.add_argument(
        "--output", type=str, default="./models/TinySage-0.6B",
        help="合并后模型的输出目录（默认: ./models/TinySage-0.6B）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_environment(cfg)

    export_tinysage(cfg, args.output)
