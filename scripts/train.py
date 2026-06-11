"""
训练 + 推理 + 评估 一体化入口。

流程：train → 快速推理对比 → 全量评测 → 输出报告

用法：
  python scripts/train.py                     # 完整流水线
  python scripts/train.py --config prod.yaml  # 指定配置文件
  python scripts/train.py --skip-eval         # 跳过全量评测（仅做快速推理对比）
  python scripts/train.py --eval-only         # 仅评测已有模型（不重新训练）
"""

import argparse
from datetime import time
import logging
import os
import sys
import time
import torch

# 将项目根目录加入 Python 搜索路径，确保 scripts/ 下运行时能找到 poly_distill 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注意：不在此处导入 torch / transformers / peft 等重模块。
# HF_ENDPOINT 环境变量必须在这些模块初始化之前设置，否则国内镜像不生效。
# 所有重模块导入延迟到 setup_environment() 调用之后。

from poly_distill.config import Config, load_config

logger = logging.getLogger(__name__)


# ============================================================
# 快速推理对比（单样本，定性观察）
# ============================================================
def quick_compare(config: Config, tokenizer) -> None:
    """用一条测试问题对比 base vs lora 的回答质量。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    test_question = (
        "解释CNN中的感受野（Receptive Field）概念，在设计视频降噪网络时，感受野大小如何影响去噪效果？"
    )

    logger.info("=== 快速推理对比 ===")

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    base_answer = _generate(base_model, tokenizer, test_question)
    logger.info("Base 模型:\n%s", base_answer)

    distilled_model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
    lora_answer = _generate(distilled_model, tokenizer, test_question)
    logger.info("LoRA 模型:\n%s", lora_answer)


def _generate(model, tokenizer, question: str) -> str:
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
    from poly_distill.config import setup_environment

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
    setup_environment(cfg)  # HF_ENDPOINT 在此设置，必须早于 transformers 导入！

    if args.eval_only:
        # 仅评测模式：直接加载已有 adapter 做全量评估
        from transformers import AutoTokenizer
        from poly_distill.eval import run_evaluation

        # 注意：评测时也需要 tokenizer，且必须与训练时一致（尤其是特殊 token）。如果训练时使用了缓存目录，也要保持一致以避免重复下载。
        logger.info("=== 评测模式: 仅评测已有模型 ===")
        logger.info("加载 tokenizer（模型 ID: %s，缓存目录: %s）", cfg.MODEL_ID, cfg.CACHE_DIR)
        logger.info("请确保模型已训练完成且 %s 目录下存在 LoRA adapter 权重", cfg.OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_ID, use_fast=True, trust_remote_code=True, cache_dir=cfg.CACHE_DIR
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("开始全量评测...")
        logger.info("评测过程中会自动加载 %s 目录下的 LoRA adapter 权重，请确保该目录存在且包含正确的权重文件。", cfg.OUTPUT_DIR)
        logger.info(f"tokenizer 使用GPU显存大小约为 {{tokenizer.model_max_length * 4 / 1e9:.2f}} GB，请确保有足够的显存可用。 当前使用已分配 {{torch.cuda.memory_allocated() / 1024**3:.2f}} GB, 预分配 {{torch.cuda.memory_reserved() / 1024**3:.2f}} GB" );
        logger.info("请耐心等待，评测过程中会持续输出进度和速度信息。 评测完成后会输出详细报告。")
        logger.info("评测结果将保存在 %s 目录下", cfg.OUTPUT_DIR)

        start_time = time.time()
        run_evaluation(cfg, tokenizer)
        end_time = time.time()
        logger.info("全量评测完成，耗时: %.2f 秒", end_time - start_time)
    else:
        # 完整流水线：训练 → 快速推理 → 全量评测
        from poly_distill.trainer import train
        tokenizer, exp_dir = train(cfg)
        quick_compare(cfg, tokenizer)

        if not args.skip_eval:
            from poly_distill.eval import run_evaluation
            run_evaluation(cfg, tokenizer)
        else:
            logger.info("已跳过全量评测 (--skip-eval)")

        logger.info("实验产物已归档: %s", exp_dir)
