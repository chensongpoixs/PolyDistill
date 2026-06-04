"""
TinySage 模型打包导出。

将 LoRA adapter 合并到基座模型，产出可独立分发和推理的完整模型。
输出路径自动根据 model_id 推导（如 Qwen3-4B → TinySage-4B）。

用法:
    python scripts/export.py                        # 自动推导输出目录
    python scripts/export.py --output ./my-model    # 自定义输出路径
"""

import argparse
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

# 将项目根目录加入 Python 搜索路径，确保 scripts/ 下运行时能找到 poly_distill 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注意：不在此处导入 torch / transformers / peft 等重模块。
# HF_ENDPOINT 环境变量必须在这些模块初始化之前设置，否则国内镜像不生效。

from poly_distill.config import load_config, setup_environment

logger = logging.getLogger(__name__)


def _derive_model_name(model_id: str) -> str:
    """从 model_id 推导导出模型名称。

    Qwen/Qwen3-0.6B → TinySage-0.6B
    Qwen/Qwen3-4B   → TinySage-4B
    Qwen/Qwen3-8B   → TinySage-8B
    """
    # 提取模型规格后缀（如 0.6B, 1.7B, 4B, 8B）
    match = re.search(r"(\d+(?:\.\d+)?B)", model_id, re.IGNORECASE)
    if match:
        return f"TinySage-{match.group(1)}"
    # 无法识别时用 model_id 最后一段
    return "TinySage-" + model_id.split("/")[-1]


def _derive_export_dir(config) -> str:
    """确定导出目录：CLI > config.yaml > 自动推导。"""
    if config.EXPORT_DIR:
        return config.EXPORT_DIR
    name = _derive_model_name(config.MODEL_ID)
    return f"./models/{name}"


def _detect_adapter_dir(config) -> Path:
    """自动探测最新训练的 LoRA adapter 目录。

    查找优先级：
      1. runs/train/exp{N}/ 下包含 adapter_config.json 的最新目录（按修改时间）
      2. config.OUTPUT_DIR 如果它包含 adapter_config.json
      3. 报错退出

    Returns:
        Path: LoRA adapter 目录路径。

    Raises:
        FileNotFoundError: 未找到任何有效的 adapter 目录。
    """
    # 优先级 1: 从实验目录中找最新的
    runs_dir = Path(config.RUNS_DIR)
    if runs_dir.is_dir():
        # 找所有 exp 子目录中包含 adapter_config.json 的，按修改时间倒序
        exp_dirs = sorted(
            [d for d in runs_dir.iterdir()
             if d.is_dir() and d.name.startswith("exp") and (d / "adapter_config.json").exists()],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if exp_dirs:
            latest = exp_dirs[0]
            logger.info("自动检测到最新实验目录: %s", latest.resolve())
            return latest

    # 优先级 2: config.yaml 中的 output_dir（可能已被训练时重定向覆盖）
    output_dir = Path(config.OUTPUT_DIR)
    if (output_dir / "adapter_config.json").exists():
        logger.info("使用配置中的 adapter 目录: %s", output_dir.resolve())
        return output_dir

    # 都没找到
    raise FileNotFoundError(
        f"未找到任何有效的 LoRA adapter 目录！\n"
        f"请先运行训练: python scripts/train.py\n"
        f"或手动指定: python scripts/export.py --adapter <路径>\n"
        f"已搜索:\n"
        f"  - runs/train/exp{{N}}/ (实验目录)\n"
        f"  - {output_dir} (config.output_dir)"
    )


def export_tinysage(config, output_dir: str, adapter_dir: str = None) -> None:
    """合并 LoRA 权重到基座模型并保存为独立模型。

    步骤：
      1. 加载基座模型
      2. 加载 LoRA adapter
      3. merge_and_unload() 融合权重
      4. 保存完整模型 + tokenizer + 模型卡片到输出目录
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

    # ---- 2. 确定 LoRA adapter 路径 ----
    if adapter_dir:
        adapter_path = Path(adapter_dir)
    else:
        adapter_path = _detect_adapter_dir(config)
    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"LoRA adapter 不存在或无效: {adapter_path}\n"
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
    """生成模型卡片（动态适配模型规格）。"""
    model_name = _derive_model_name(config.MODEL_ID)
    # 推断模型参数量
    size_map = {"0.6B": "0.6B", "1.7B": "1.7B", "4B": "4B", "8B": "8B"}
    model_size = None
    for k, v in size_map.items():
        if k in model_name:
            model_size = v
            break
    size_note = f" (~{model_size} parameters)" if model_size else ""

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

# {model_name}

A multi-teacher distilled small language model{size_note} for AI infrastructure domain.

- **Base model**: {config.MODEL_ID}
- **Training framework**: [PolyDistill](https://github.com/chensongpoixs/PolyDistill)
- **Teachers**: GPT, Claude, Gemini
- **Distillation method**: Black-box instruction distillation + LoRA SFT (r={config.LORA_R})
- **Domain**: AI Infra (audio/video codecs, streaming protocols, GPU/CUDA, etc.)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("{model_name}", trust_remote_code=True)

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


def _list_experiments(config) -> str:
    """列出所有可用的实验目录及其基本信息。"""
    runs_dir = Path(config.RUNS_DIR)
    if not runs_dir.is_dir():
        return "  无实验目录（runs/train/ 不存在）"

    lines = []
    exp_dirs = sorted(
        [d for d in runs_dir.iterdir()
         if d.is_dir() and d.name.startswith("exp")],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not exp_dirs:
        return "  无实验记录"

    for d in exp_dirs:
        has_adapter = (d / "adapter_config.json").exists()
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime))
        status = "✅ 可导出" if has_adapter else "⚠️ 无 adapter"
        # 读取配置快照获取模型信息
        model_info = ""
        snapshot = d / "config_snapshot.yaml"
        if snapshot.exists():
            try:
                import yaml
                snap = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
                if snap:
                    model_info = f" — {snap.get('model_id', '')}"
            except Exception:
                pass
        lines.append(f"  {d.name:8s}  {mtime}  {status}{model_info}")
    return "\n".join(lines)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TinySage 模型打包导出 — 合并 LoRA adapter 到基座模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 自动导出最新实验
  %(prog)s --exp exp3                         # 导出指定实验
  %(prog)s --list                             # 列出所有可导出实验
  %(prog)s --adapter ./path/to/adapter        # 手动指定 adapter 路径
  %(prog)s --output ./models/my-tinysage      # 自定义输出路径
  %(prog)s --config prod.yaml                  # 指定配置文件
        """,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML 配置文件路径（默认: ./config.yaml）",
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        help="实验名称（如 exp3），默认: 最新实验。与 --adapter 互斥。",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="手动指定 LoRA adapter 目录路径。与 --exp 互斥。",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="合并后模型的输出目录（默认: 自动推导为 ./models/TinySage-{规格}）",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有可用实验后退出",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_environment(cfg)

    # ── --list 模式：列出实验后退出 ──
    if args.list:
        print("可导出的实验列表:\n")
        print(_list_experiments(cfg))
        sys.exit(0)

    # ── 确定 adapter 路径 ──
    if args.adapter and args.exp:
        parser.error("--adapter 和 --exp 互斥，只能指定一个")

    if args.adapter:
        adapter_path = args.adapter
    elif args.exp:
        adapter_path = str(Path(cfg.RUNS_DIR) / args.exp)
    else:
        adapter_path = None  # 自动探测

    # ── 输出目录: CLI > config.yaml > 自动推导 ──
    output = args.output or _derive_export_dir(cfg)
    export_tinysage(cfg, output, adapter_dir=adapter_path)
