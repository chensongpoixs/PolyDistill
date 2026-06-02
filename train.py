"""
LoRA SFT 训练核心逻辑。

封装模型加载、LoRA 配置、TrainingArguments 构造及 SFTTrainer 训练流程。
"""

import torch
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from config import Config
from dataset import load_and_prepare_data


def load_model_and_tokenizer(config: Config) -> tuple:
    """加载 Qwen 模型和分词器。

    关键设计决策：
      - use_fast=True：使用 Rust 实现的快速分词器，大幅提升数据预处理速度。
      - trust_remote_code=True：Qwen 系列模型的分词/模型逻辑部分写在仓库的 .py 文件中，
        必须开启此选项才能正确加载。
      - torch_dtype=bfloat16：BF16 在保持与 FP32 相近精度的同时，显存占用减半。
      - pad_token = eos_token：Qwen tokenizer 默认无 pad_token，
        但批量训练（batching）要求序列等长，必须指定填充符。

    Returns:
        (model, tokenizer)
    """
    # ---- 分词器 ----
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR,
    )

    # 分配 pad_token：使用 eos_token（<|im_end|>）作为填充符
    # 这是因果语言模型微调的通用做法——不对填充位置计算 loss
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Tokenizer 加载成功")

    # ---- 模型 ----
    # bf16 是 NVIDIA Ampere 及以上架构（A100/A6000/3090/4090）推荐的训练精度
    use_bf16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR,
    ).to("cuda")

    print("✅ 模型加载成功")
    return model, tokenizer


def build_lora_config(config: Config) -> LoraConfig:
    """构造 LoRA 参数高效微调配置。

    LoRA (Low-Rank Adaptation) 核心思想：
      对于预训练权重矩阵 W ∈ R^{d×k}，不直接微调 W，而是学习一个低秩增量 ΔW = B·A，
      其中 B ∈ R^{d×r}, A ∈ R^{r×k}，且 r << min(d, k)。
      推理时合并为 W' = W + ΔW，无额外推理延迟。

    工业推荐：
      - target_modules = ["q_proj", "v_proj"]：实验表明仅微调 Q/V 矩阵即可获得
        良好的指令跟随能力，同时减少可训练参数量、降低过拟合风险。
      - r=16, alpha=32：经典的 rank=16 配置，配合 alpha=2*r 的缩放策略。
      - dropout=0.05：轻度 dropout 正则化，防止小数据集上的过拟合。
    """
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",  # 不训练 bias 参数
        task_type="CAUSAL_LM",  # 因果语言模型任务
        target_modules=list(config.LORA_TARGET_MODULES),
    )


def build_training_args(config: Config) -> TrainingArguments:
    """构造 Hugging Face TrainingArguments。

    关键参数说明：
      - learning_rate=2e-4：LoRA 微调推荐 lr 高于全量微调（通常 1e-4 ~ 5e-4）。
      - cosine scheduler + warmup：先线性预热再余弦衰减至接近 0，训练更稳定。
      - gradient_accumulation_steps=8：小 batch_size 通过梯度累积模拟大 batch，
        在显存受限时维持训练稳定性。
      - bf16=True：BF16 混合精度训练，比 FP16 更稳定（无需 loss scaling）。
      - ddp_find_unused_parameters=False：单卡场景关闭 DDP 未使用参数检测。
    """
    return TrainingArguments(
        # ---- 输出 ----
        output_dir=config.OUTPUT_DIR,
        # ---- 批次 ----
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        # ---- 学习率 ----
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        # ---- 训练轮次 ----
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        # ---- 精度 ----
        bf16=torch.cuda.is_available(),
        # ---- 日志与保存 ----
        logging_steps=config.LOGGING_STEPS,
        save_strategy=config.SAVE_STRATEGY,
        report_to="none",  # 不上报外部平台（如 WandB），保持环境干净
        # ---- 分布式（单卡优化） ----
        ddp_find_unused_parameters=False,
    )


def train(config: Config) -> tuple:
    """执行 LoRA SFT 训练主流程。

    步骤：
      1. 加载模型与分词器
      2. 加载并预处理数据集
      3. 构造 DataCollatorForCompletionOnlyLM（仅对 assistant 回复计算 loss）
      4. 配置 LoRA 和训练参数
      5. 实例化 SFTTrainer 并开始训练
      6. 保存 LoRA adapter 权重

    Returns:
        (SFTTrainer, PreTrainedTokenizer): trainer 实例和 tokenizer，
        供 inference 阶段复用（避免重复加载）。
    """
    # ---- 加载模型 ----
    model, tokenizer = load_model_and_tokenizer(config)

    # ---- 加载数据 ----
    dataset = load_and_prepare_data(config, tokenizer)

    # ---- DataCollator：仅对 assistant 部分计算 loss ----
    # 核心原理：
    #   DataCollatorForCompletionOnlyLM 通过查找 response_template 在序列中的位置，
    #   将 template 之前的 token（system prompt + user instruction）的 label 设为 -100，
    #   使 CrossEntropyLoss 自动忽略这些位置的梯度计算。
    #   这样模型只学习 assistant 的回复内容，不学习提问部分。
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=config.RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    # ---- LoRA ----
    peft_config = build_lora_config(config)

    # ---- 训练参数 ----
    training_args = build_training_args(config)

    # ---- 实例化 Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    # ---- 开始训练 ----
    print("🚀 Start SFT training...")
    trainer.train()

    # ---- 保存 LoRA adapter ----
    # SFTTrainer 会自动合并 PEFT 保存逻辑，仅保存 adapter 权重（几 MB），
    # 不保存完整的基座模型（几 GB）。
    trainer.save_model(config.OUTPUT_DIR)
    print(f"✅ Training finished. Saved to: {config.OUTPUT_DIR}")

    return trainer, tokenizer
