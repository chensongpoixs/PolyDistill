"""
LoRA SFT 训练核心逻辑。

封装模型加载、LoRA 配置、TrainingArguments 构造及 SFTTrainer 训练流程。
支持：train/val split、early stopping、NEFTune、梯度裁剪、权重衰减。
训练输出采用 YOLOv5 风格格式。
"""

import logging
import sys
import time

import torch
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from poly_distill.config import Config
from poly_distill.dataset import load_and_prepare_data

logger = logging.getLogger(__name__)


# ============================================================
# YOLOv5 风格训练输出回调
# ============================================================
class YOLOStyleProgressCallback(TrainerCallback):
    """YOLOv5 风格的训练进度输出。

    每 epoch 输出一行汇总，格式如下:
        Epoch  gpu_mem  train_loss   val_loss        lr  samples    time
         1/30    4.2G      1.2345      1.0892   2.0e-04     1755     45s
         2/30    4.2G      0.9876      0.9521   1.7e-04     1755     43s
        ...

    行在 on_log 中即时打印（epoch 变化时），on_epoch_end 作为兜底。
    训练结束时输出最优指标。
    """

    def __init__(self, total_epochs: int, train_samples: int, val_samples: int = 0):
        self.total_epochs = total_epochs
        self.train_samples = train_samples
        self.val_samples = val_samples
        self._epoch_start_time = 0.0
        self._epoch = 0
        self._train_loss = None
        self._val_loss = None
        self._lr = None
        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._last_printed_epoch = -1  # 避免同一 epoch 重复打印
        self._header_printed = False
        self._lr_history: dict[int, float] = {}  # epoch → LR，用于训练结束绘制曲线

    def _print_header(self):
        header = (
            f"\n{'Epoch':>8s}  {'gpu_mem':>8s}  "
            f"{'train_loss':>10s}  {'val_loss':>10s}  "
            f"{'lr':>10s}  {'samples':>8s}  {'time':>8s}"
        )
        print(header, flush=True)
        print("-" * 78, flush=True)
        self._header_printed = True

    def _get_gpu_mem(self) -> str:
        if torch.cuda.is_available():
            # reserved 对应 nvidia-smi 显示的值，allocated 是 PyTorch 实际使用
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            return f"{reserved:.1f}G"
        return "    N/A"

    def _format_epoch_line(self) -> str:
        epoch_str = f"{self._epoch}/{self.total_epochs}"
        gpu = self._get_gpu_mem()
        train = f"{self._train_loss:.4f}" if self._train_loss is not None else "      N/A"
        val = f"{self._val_loss:.4f}" if self._val_loss is not None else "      N/A"
        lr = f"{self._lr:.2e}" if self._lr is not None else "      N/A"
        samples = self.val_samples if self.val_samples > 0 else self.train_samples
        elapsed = f"{time.time() - self._epoch_start_time:.0f}s" if self._epoch_start_time else "    N/A"
        return (
            f"{epoch_str:>8s}  {gpu:>8s}  "
            f"{train:>10s}  {val:>10s}  "
            f"{lr:>10s}  {samples:>8d}  {elapsed:>8s}"
        )

    def _print_line(self):
        """打印当前 epoch 行，防重复。"""
        if self._epoch != self._last_printed_epoch:
            print(self._format_epoch_line(), flush=True)
            self._last_printed_epoch = self._epoch

    # ---- TrainerCallback 钩子 ----

    def on_train_begin(self, args, state, control, **kwargs):
        self._print_header()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次 log step 触发：捕获 loss/lr，epoch 变化时即时打印。

        HF Trainer 中 state.epoch 在训练中为 0.0~0.99 (第一个 epoch)、
        1.0~1.99 (第二个 epoch) 等。on_epoch_end 时 state.epoch 已更新为
        刚完成的 epoch 编号 (如 1.0 表示第一个 epoch 完成)。

        因此 on_log 使用 int(state.epoch) + 1 (1-indexed)，
        on_epoch_end 使用 int(state.epoch) (state.epoch 已是完成后的编号)。
        """
        if logs is None:
            return
        if "loss" in logs:
            self._train_loss = logs["loss"]
        if "learning_rate" in logs:
            self._lr = logs["learning_rate"]
        # 1-indexed：训练中 state.epoch=0.xx → 显示 "1/N"
        if state.epoch is not None:
            current = int(state.epoch) + 1
            if current != self._epoch:
                self._epoch = current
                self._print_line()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        if "eval_loss" in metrics:
            self._val_loss = metrics["eval_loss"]
            if self._val_loss < self._best_val_loss:
                self._best_val_loss = self._val_loss
                self._best_epoch = self._epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        """兜底：如果 on_log 未触发 epoch 变化，在此处打印。"""
        epoch_val = state.epoch  # 可能是 float 如 0.0；None 时回退
        if epoch_val is not None:
            self._epoch = int(epoch_val)  # int(0.0) = 0, int(1.0) = 1, ...
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                self._train_loss = last["loss"]
            if "learning_rate" in last:
                self._lr = last["learning_rate"]
        # 记录当前 epoch 的 LR（用于训练结束打印曲线）
        if self._lr is not None:
            self._lr_history[self._epoch] = self._lr
        self._print_line()  # 防重复自动处理

    def on_train_end(self, args, state, control, **kwargs):
        print("-" * 78, flush=True)
        if self._best_val_loss < float("inf"):
            print(
                f"Best metric: eval_loss = {self._best_val_loss:.4f} "
                f"at epoch {self._best_epoch}",
                flush=True,
            )
            print(f"Training complete: {self._epoch} epochs, best checkpoint saved", flush=True)
        else:
            print(f"Training complete: {self._epoch} epochs", flush=True)

        # ---- 学习率下降曲线 ----
        if self._lr_history:
            self._print_lr_curve()

    def _print_lr_curve(self):
        """训练结束时输出 LR 衰减曲线（ASCII 柱状图）。"""
        print("\n" + "=" * 78)
        print("Learning Rate Schedule (Cosine Decay)")
        print("=" * 78)
        epochs = sorted(self._lr_history.keys())
        lr_values = [self._lr_history[e] for e in epochs]
        max_lr = max(lr_values) if lr_values else 1.0

        # 按行输出，每行 5 个 epoch
        bar_width = 20
        for i in range(0, len(epochs), 5):
            line_parts = []
            for j in range(i, min(i + 5, len(epochs))):
                e = epochs[j]
                lr = self._lr_history[e]
                # ASCII bar
                bar_len = max(1, int(lr / max_lr * bar_width))
                bar = "█" * bar_len + " " * (bar_width - bar_len)
                line_parts.append(f"  {e:>3d}: {lr:.2e} │{bar}│")
            print("\n".join(line_parts))
            if i + 5 < len(epochs):
                print()  # 行间空行
        print("-" * 78)
        print(f"  LR range: {lr_values[-1]:.2e} → {lr_values[0]:.2e}  (start → end)")
        print("=" * 78, flush=True)


# ============================================================
# 模型与分词器
# ============================================================
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

    logger.info("Tokenizer 加载成功")

    use_bf16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR,
    ).to("cuda")

    logger.info("模型加载成功")
    return model, tokenizer


# ============================================================
# LoRA 配置
# ============================================================
def build_lora_config(config: Config) -> LoraConfig:
    """构造 LoRA 参数高效微调配置。

    LoRA (Low-Rank Adaptation) 核心思想：
      学习低秩增量 ΔW = B·A，其中 B ∈ R^{d×r}, A ∈ R^{r×k}，r << min(d, k)。
      推理时合并为 W' = W + ΔW，无额外延迟。
    """
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.LORA_TARGET_MODULES),
    )


# ============================================================
# 训练参数
# ============================================================
def build_training_args(config: Config) -> TrainingArguments:
    """构造 Hugging Face TrainingArguments。

    关键参数：
      - weight_decay=0.01：L2 正则化，抑制过拟合。
      - max_grad_norm=1.0：梯度裁剪，防止训练震荡。
      - neftune_noise_alpha=5：embedding 噪声注入（需要 transformers ≥ 4.38）。
      - save_strategy="best" + load_best_model_at_end：自动选取最优 checkpoint。
    """
    kwargs = dict(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        eval_strategy="epoch" if config.EVAL_SPLIT_RATIO > 0 else "no",
        save_strategy=config.SAVE_STRATEGY,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
        bf16=torch.cuda.is_available(),
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # NEFTune：需要 transformers ≥ 4.38，旧版本静默跳过
    neftune_alpha = config.NEFTUNE_NOISE_ALPHA
    if neftune_alpha > 0:
        kwargs["neftune_noise_alpha"] = neftune_alpha

    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        if "neftune_noise_alpha" in str(e):
            del kwargs["neftune_noise_alpha"]
            logger.warning("当前 transformers 版本不支持 NEFTune（需 ≥ 4.38），已跳过")
            return TrainingArguments(**kwargs)
        raise


# ============================================================
# 主训练流程
# ============================================================
def train(config: Config) -> tuple:
    """执行 LoRA SFT 训练主流程。

    步骤：
      1. 加载模型与分词器
      2. 加载并预处理数据集，按比例划分 train/val
      3. 构造 DataCollatorForCompletionOnlyLM（仅对 assistant 回复计算 loss）
      4. 配置 LoRA 和训练参数（含早停 + YOLOv5 风格输出回调）
      5. 实例化 SFTTrainer（含 NEFTune）并开始训练
      6. 保存 LoRA adapter 权重

    Returns:
        (SFTTrainer, PreTrainedTokenizer)
    """
    # ---- 1. 加载模型 ----
    model, tokenizer = load_model_and_tokenizer(config)

    # ---- 梯度检查点：用计算换显存，激活值不全部存留 ----
    if config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("梯度检查点已开启：激活值不缓存，节省 ~50% 显存")

    # ---- 2. 加载数据 + train/val split ----
    dataset = load_and_prepare_data(config, tokenizer)

    train_dataset = dataset
    eval_dataset = None
    if config.EVAL_SPLIT_RATIO > 0:
        split = dataset.train_test_split(
            test_size=config.EVAL_SPLIT_RATIO, seed=42
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(
            "数据分割: train=%d, val=%d (ratio=%.2f)",
            len(train_dataset), len(eval_dataset), config.EVAL_SPLIT_RATIO,
        )

    # ---- 3. DataCollator：仅对 assistant 部分计算 loss ----
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=config.RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    # ---- 4. 配置 ----
    peft_config = build_lora_config(config)
    training_args = build_training_args(config)

    # ---- YOLOv5 风格输出回调 ----
    yolo_callback = YOLOStyleProgressCallback(
        total_epochs=config.NUM_TRAIN_EPOCHS,
        train_samples=len(train_dataset),
        val_samples=len(eval_dataset) if eval_dataset else 0,
    )
    callbacks = [yolo_callback]

    # ---- 早停回调 ----
    if config.EARLY_STOPPING_PATIENCE > 0 and eval_dataset is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD,
            )
        )
        logger.info(
            "早停已启用: patience=%d, threshold=%.4f",
            config.EARLY_STOPPING_PATIENCE, config.EARLY_STOPPING_THRESHOLD,
        )

    # ---- 5. 训练信息摘要 ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %s | Params: %.1fM total, %.1fM trainable | "
        "Train: %d samples | Val: %d samples",
        config.MODEL_ID,
        total_params / 1e6,
        trainable_params / 1e6,
        len(train_dataset),
        len(eval_dataset) if eval_dataset else 0,
    )

    # ---- 强制静默：必须在 SFTTrainer 构造前执行 ----
    # SFTTrainer.__init__ 会读取 TrainingArguments.log_level 覆盖 verbosity，
    # 所以 log_level 已从 build_training_args 中移除。
    # 此处提前禁用所有 HF 日志，确保 YOLOv5 风格输出不被污染。
    import transformers.utils.logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_default_handler()
    import logging as _logging
    _logging.getLogger("trl").setLevel(_logging.ERROR)
    _logging.getLogger("datasets").setLevel(_logging.ERROR)
    _logging.getLogger("peft").setLevel(_logging.ERROR)
    _logging.getLogger("sentencepiece").setLevel(_logging.ERROR)

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

    trainer = SFTTrainer(**trainer_kwargs)

    # ---- 移除 HF 默认 PrinterCallback，仅保留 YOLO + tqdm 进度条 ----
    # PrinterCallback → {'loss': ...} 字典泄漏（print() 直接输出，非 logging 模块）
    from transformers.trainer_callback import PrinterCallback
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if not isinstance(cb, PrinterCallback)
    ]

    # ---- 6. 开始训练 ----
    logger.info("Start SFT training...")
    trainer.train()

    # ---- 7. 保存 LoRA adapter ----
    trainer.save_model(config.OUTPUT_DIR)
    logger.info("Saved to: %s", config.OUTPUT_DIR)

    # 输出最优指标
    if eval_dataset is not None:
        best_metric = trainer.state.best_metric
        best_checkpoint = trainer.state.best_model_checkpoint
        if best_metric is not None:
            logger.info(
                "Best %s: %.4f @ %s",
                config.METRIC_FOR_BEST_MODEL, best_metric,
                best_checkpoint or "current",
            )

    return trainer, tokenizer
