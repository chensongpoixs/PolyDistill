"""
推理对比评估。

加载基座模型与 LoRA adapter，对同一测试问题进行推理，
定性对比微调前后的回答质量变化。

用法：
  python inference.py                  # 自动读取 ./config.yaml
  python inference.py --config prod.yaml  # 指定配置文件
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from config import Config, load_config
from train import train


def evaluate(config: Config, tokenizer: PreTrainedTokenizer) -> None:
    """对比基座模型与微调后模型在测试问题上的回答质量。

    实验设计：
      - 用同一测试问题分别询问基座模型和 LoRA 增强模型。
      - 低温采样（temperature=0.1）确保可复现。
      - 定性观察知识蒸馏是否生效。
    """
    # ---- 测试问题（采样自训练集主题，评估泛化能力） ----
    test_question = (
        "你如何利用BGE-M3模型进行混合检索？"
        "请说明向量检索与BM25结合的流程。"
    )

    # ---- 加载基座模型（用于对比） ----
    print("\n=== ⚔️ 效果对决 ⚔️ ===\n")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # ---- 回合 1：基座模型 ----
    print(f"🔴 原始模型回答:\n{run_inference(base_model, tokenizer, test_question)}")

    # ---- 回合 2：LoRA 增强模型 ----
    print("\nLoading LoRA adapters...")
    distilled_model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
    print(f"🟢 蒸馏模型回答:\n{run_inference(distilled_model, tokenizer, test_question)}")


def run_inference(model, tokenizer: PreTrainedTokenizer, question: str) -> str:
    """单次推理辅助函数。

    使用 chat_template 构造输入，低温解码确保结果稳定可复现。
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 推理时需要，提示模型开始生成
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,  # 低温 → 更确定的输出，便于对比
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    from config import setup_environment

    parser = argparse.ArgumentParser(description="AI Infra LoRA SFT 训练 + 推理")
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML 配置文件路径（默认: ./config.yaml）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_environment(cfg)
    trainer, tokenizer = train(cfg)
    evaluate(cfg, tokenizer)
