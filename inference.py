"""
训练 + 推理 + 评估 一体化入口。

流程：train → 快速推理对比 → 全量评测 → 输出报告

用法：
  python inference.py                     # 完整流水线
  python inference.py --config prod.yaml  # 指定配置文件
  python inference.py --skip-eval         # 跳过全量评测（仅做快速推理对比）
  python inference.py --eval-only         # 仅评测已有模型（不重新训练）
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from config import Config, load_config
from train import train


# ============================================================
# 快速推理对比（单样本，定性观察）
# ============================================================
def quick_compare(config: Config, tokenizer: PreTrainedTokenizer) -> None:
    """用一条测试问题对比 base vs lora 的回答质量。

    这是训练后最直观的检查——一眼看出模型是否学到了领域知识。
    """
    test_question = (
        "你如何利用BGE-M3模型进行混合检索？"
        "请说明向量检索与BM25结合的流程。"
    )

    print("\n=== ⚔️ 快速推理对比 ⚔️ ===\n")

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    base_answer = _generate(base_model, tokenizer, test_question)
    print(f"🔴 Base 模型:\n{base_answer}\n")

    distilled_model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
    lora_answer = _generate(distilled_model, tokenizer, test_question)
    print(f"🟢 LoRA 模型:\n{lora_answer}\n")


def _generate(model, tokenizer: PreTrainedTokenizer, question: str) -> str:
    """单次推理。"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    from config import setup_environment

    parser = argparse.ArgumentParser(description="AI Infra LoRA SFT 训练 + 评估")
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML 配置文件路径（默认: ./config.yaml）",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="跳过全量评测，仅做单样本快速推理对比",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="仅评测已有模型（不重新训练），需提前训练完成",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_environment(cfg)

    if args.eval_only:
        # 仅评测模式：直接加载已有 adapter 做全量评估
        from transformers import AutoTokenizer
        from eval import run_evaluation

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_ID, use_fast=True, trust_remote_code=True, cache_dir=cfg.CACHE_DIR
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        run_evaluation(cfg, tokenizer)
    else:
        # 完整流水线：训练 → 快速推理 → 全量评测
        trainer, tokenizer = train(cfg)
        quick_compare(cfg, tokenizer)

        if not args.skip_eval:
            from eval import run_evaluation
            run_evaluation(cfg, tokenizer)
        else:
            print("⏭️  已跳过全量评测 (--skip-eval)")
