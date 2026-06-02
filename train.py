"""
LoRA SFT 训练核心逻辑。

封装模型加载、LoRA 配置、TrainingArguments 构造及 SFTTrainer 训练流程。
支持：train/val split、early stopping、NEFTune、梯度裁剪、权重衰减。
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
      - use_fast=True：Rust 快速分词器。
      - trust_remote_code=True：Qwen 系列模型的自定义代码必须执行。
      - torch_dtype=bfloat16：BF16 精度，显存减半，比 FP16 更稳定。
      - pad_token = eos_token：Qwen tokenizer 无 pad_token，必须手动设置。

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Tokenizer 加载成功")

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
      学习低秩增量 ΔW = B·A，其中 B ∈ R^{d×r}, A ∈ R^{r×k}，r << min(d, k)。
      推理时合并为 W' = W + ΔW，无额外延迟。

    工业推荐：
      - r=8, alpha=16：适用于 400+ 条小数据集，防过拟合。
      - dropout=0.1：中等 dropout 正则化。
    """
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.LORA_TARGET_MODULES),
    )


def build_training_args(config: Config) -> TrainingArguments:
    """构造 Hugging Face TrainingArguments。

    关键参数：
      - weight_decay=0.01：L2 正则化，抑制过拟合。
      - max_grad_norm=1.0：梯度裁剪，防止训练震荡。
      - save_strategy="best" + load_best_model_at_end：自动选取最优 checkpoint。
      - evaluation_strategy="epoch"：每个 epoch 评估验证集，配合早停使用。
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
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        # ---- 正则化 ----
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        # ---- 早停 ----
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        # ---- 评估 ----
        eval_strategy="epoch" if config.EVAL_SPLIT_RATIO > 0 else "no",
        # ---- 保存 ----
        save_strategy=config.SAVE_STRATEGY,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        # ---- 早停回调（TrainingArguments 不直接支持，由 Trainer 回调处理） ----
        # early_stopping_patience / threshold 在 SFTTrainer 中通过 callbacks 注入
        # ---- 日志 ----
        logging_steps=config.LOGGING_STEPS,
        # ---- 精度 ----
        bf16=torch.cuda.is_available(),
        report_to="none",
        # ---- 分布式 ----
        ddp_find_unused_parameters=False,
    )


def train(config: Config) -> tuple:
    """执行 LoRA SFT 训练主流程。

    步骤：
      1. 加载模型与分词器
      2. 加载并预处理数据集，按比例划分 train/val
      3. 构造 DataCollatorForCompletionOnlyLM（仅对 assistant 回复计算 loss）
      4. 配置 LoRA 和训练参数（含早停回调）
      5. 实例化 SFTTrainer（含 NEFTune）并开始训练
      6. 保存 LoRA adapter 权重

    Returns:
        (SFTTrainer, PreTrainedTokenizer)
    """
    # ---- 加载模型 ----
    model, tokenizer = load_model_and_tokenizer(config)

    # ---- 加载数据 + train/val split ----
    dataset = load_and_prepare_data(config, tokenizer)

    train_dataset = dataset
    eval_dataset = None
    if config.EVAL_SPLIT_RATIO > 0:
        split = dataset.train_test_split(
            test_size=config.EVAL_SPLIT_RATIO, seed=42
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(
            f"📊 数据分割: train={len(train_dataset)}, "
            f"val={len(eval_dataset)} (ratio={config.EVAL_SPLIT_RATIO})"
        )

    # ---- DataCollator：仅对 assistant 部分计算 loss ----
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=config.RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    # ---- LoRA ----
    peft_config = build_lora_config(config)

    # ---- 训练参数 ----
    training_args = build_training_args(config)

    # ---- 早停回调 ----
    callbacks = []
    if config.EARLY_STOPPING_PATIENCE > 0 and eval_dataset is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD,
            )
        )
        print(
            f"⏸️  早停已启用: patience={config.EARLY_STOPPING_PATIENCE}, "
            f"threshold={config.EARLY_STOPPING_THRESHOLD}"
        )

    # ---- 实例化 Trainer ----
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # NEFTune：在 embedding 层注入噪声，提升指令微调泛化效果
    if config.NEFTUNE_NOISE_ALPHA > 0:
        try:
            trainer_kwargs["neftune_noise_alpha"] = config.NEFTUNE_NOISE_ALPHA
            print(f"🔊 NEFTune 已启用: noise_alpha={config.NEFTUNE_NOISE_ALPHA}")
        except Exception:
            print("⚠️  NEFTune 配置失败（可能需要 TRL ≥ 0.12），已跳过")

    trainer = SFTTrainer(**trainer_kwargs)

    # ---- 开始训练 ----
    print("🚀 Start SFT training...")
    trainer.train()

    # ---- 保存 LoRA adapter ----
    trainer.save_model(config.OUTPUT_DIR)
    print(f"✅ Training finished. Saved to: {config.OUTPUT_DIR}")

    # 输出最优指标
    if eval_dataset is not None:
        best_metric = trainer.state.best_metric
        best_checkpoint = trainer.state.best_model_checkpoint
        if best_metric is not None:
            print(f"🏆 Best {config.METRIC_FOR_BEST_MODEL}: {best_metric:.4f}")
            print(f"📂 Best checkpoint: {best_checkpoint}")

    return trainer, tokenizer
