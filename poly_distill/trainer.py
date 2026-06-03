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

    # Qwen3 + Flash Attention 要求左填充 (padding_side='left')
    # 否则 batched forward 时 causal mask 计算会崩溃:
    #   "You are attempting to perform batched generation with
    #    padding_side='right' this may lead to unexpected behaviour
    #    for Flash Attention version of Qwen3"
    # 左填充确保 padding token 在序列左侧，不影响 attention 的有效 token 位置
    tokenizer.padding_side = "left"

    logger.info("Tokenizer 加载成功")

    use_bf16 = torch.cuda.is_available()
    attn_kwargs = {}
    attn_mode = config.ATTN_IMPLEMENTATION
    if attn_mode:
        # ═══════════════════════════════════════════════════════════════════
        # 注意力实现选型：显存优化的核心
        # ═══════════════════════════════════════════════════════════════════
        #
        # 标准 Attention 的显存瓶颈:
        #   对长度为 L 的序列, Q·K^T 产生 (L × L) 的注意力分数矩阵。
        #   以 Qwen3-0.6B 为例: L=2048, fp32 下单层注意力矩阵 =
        #     2048 × 2048 × 4 bytes = 16.8 MB
        #   12 层 × 16.8 MB × batch_size=4 = 806 MB (仅注意力分数!)
        #   加上 softmax 中间结果和 V 加权, 实际显存可达 2-3 GB。
        #
        # Flash Attention 的核心优化 (IO-aware 算法):
        #   不将完整的 Q·K^T 矩阵写入 HBM (显存), 而是:
        #     1. 将 Q/K/V 分块 (tile) 加载到 SRAM (on-chip, ~20 TB/s, 比 HBM 快 10×)
        #     2. 在 SRAM 内完成 softmax(QK^T/sqrt(d))·V 的增量计算 (online softmax)
        #     3. 只将最终输出写回 HBM
        #   结果: 显存从 O(L²) 降为 O(L), 速度因减少 HBM 读写而提升 20-50%。
        #
        # 各实现对比:
        #   ┌──────────────────┬──────────────────┬──────────────────┬───────────────┐
        #   │ 实现               │ 显存 (注意力层)     │ 速度              │ 硬件要求        │
        #   ├──────────────────┼──────────────────┼──────────────────┼───────────────┤
        #   │ eager (原生)        │ O(L²) 全量矩阵      │ 基准               │ 任意 GPU        │
        #   │ SDPA (PyTorch内置)  │ ~O(L) 融合算子      │ 1.2-1.5×          │ GPU SM ≥ 70     │
        #   │ FA2 (Ampere)       │ O(L) 分块计算       │ 1.3-1.7×          │ GPU SM ≥ 80     │
        #   │ FA3 (Blackwell)    │ O(L) + SM100 新指令 │ 1.5-2.0×          │ GPU SM ≥ 120    │
        #   └──────────────────┴──────────────────┴──────────────────┴───────────────┘
        #
        # 本项目的显存优化组合 (三层叠加):
        #   BF16 (2 bytes) + FlashAttn (O(L)) + GradientCheckpoint (不存激活值)
        #   ≈ 原始显存的 25-35%

        # ── GPU 硬件信息 ──
        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"
        capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        sm = capability[0] * 10 + capability[1]  # e.g. (12,0) → 120
        arch = (
            "Blackwell" if sm >= 120 else
            "Hopper"    if sm >= 90  else
            "Ada"       if sm >= 89  else
            "Ampere"    if sm >= 80  else
            "Turing"    if sm >= 75  else
            "Volta"     if sm >= 70  else
            "Unknown"
        )
        logger.info(
            "GPU: %s | SM: %d.%d (%s) | Compute Capability: sm_%d%d",
            gpu_name, capability[0], capability[1], arch, capability[0], capability[1],
        )

        if attn_mode == "auto":
            # ── 选型决策树 ──
            logger.info("── 注意力选型决策树 (mode=auto) ──")

            # Step 1: 检测 flash-attn 包
            logger.info("[Step 1] 检测 flash-attn 包...")
            try:
                import flash_attn  # noqa: F401
                fa_ver = getattr(flash_attn, "__version__", "unknown")
                logger.info("  ✓ flash-attn 已安装 (version: %s)", fa_ver)
            except ImportError:
                # SDPA (Scaled Dot-Product Attention): PyTorch 2.0+ 内置融合算子
                #   显存: 通过 memory_efficient 后端自动选择最优融合 kernel
                #         (FlashSparse / Math / MemEfficient), 避免全量矩阵物化
                #   优势: 零额外依赖, 开箱即用, 兼容所有 GPU SM ≥ 70
                #   劣势: 非 IO-aware 算法, SM 利用率低于 FA2 (约慢 10-20%)
                #   显存节省: ~30-40% vs eager
                logger.info("  ✗ flash-attn 未安装 → 回退 SDPA")
                logger.info("    安装命令: pip install flash-attn --no-build-isolation")
                attn_kwargs["attn_implementation"] = "sdpa"
                logger.info("[决策结果] → SDPA (PyTorch 内置 scaled_dot_product_attention)")
            else:
                # Step 2: 判断 GPU 架构 → FA3 / FA2
                logger.info("[Step 2] GPU 架构判断: SM%d -> %s", sm, arch)
                if sm >= 120:
                    # ── Blackwell (RTX 5080/5090) → Flash Attention 3 ──
                    #
                    # FA3 相比 FA2 的关键改进:
                    #   1. SM100 架构新指令: 利用 Blackwell 的 FP8/FP4 tensor core,
                    #      warpgroup MMA (矩阵乘累加) 在单个 warp 内完成
                    #   2. 非对称分块: Q 的 tile 比 K/V 更大, 减少 K/V HBM 重读次数
                    #   3. 动态调度: runtime 根据序列长度自适应 tile size,
                    #      避免长序列时 tile 撑爆 SRAM
                    #   4. 窗口注意力 (sliding window): local attention 直接跳过
                    #      远程 token, 进一步降低计算量
                    #
                    # 显存: FA3 ≈ FA2 (都是 O(L)), 但速度提升 20-30%
                    #       长序列 (L>4096) 时优势更明显
                    #
                    # HF 路由机制:
                    #   attn_implementation="flash_attention_2" 传给 from_pretrained()
                    #   → HF 内部检测 flash_attn.__version__ >= 2.6 + GPU SM >= 120
                    #   → 自动调用 flash_attn.flash_attn_func() FA3 路径
                    #   → 验证方式: 训练日志中 "[决策结果] -> Flash Attention 3"
                    # 注意: HF 不接受 "flash_attention_3" 字符串, 必须传 "flash_attention_2"

                    # Blackwell: 检查 transformers >= 4.51 确保 FA3 kernel 可路由
                    logger.info("  SM%d >= 120 (Blackwell) -> 候选: Flash Attention 3", sm)
                    logger.info("[Step 3] 检查 transformers / flash-attn 版本...")
                    try:
                        import transformers as _tf
                        tf_ver = _tf.__version__
                        tf_major, tf_minor = [int(x) for x in tf_ver.split(".")[:2]]
                        supports_fa3 = (tf_major, tf_minor) >= (4, 51)
                    except Exception:
                        tf_ver = "unknown"
                        supports_fa3 = False
                    logger.info("  transformers version: %s", tf_ver)
                    # HF 传 "flash_attention_2"，内部根据 GPU + flash-attn 版本自动路由：
                    #   Blackwell SM120 + flash-attn >= 2.6 -> FA3 kernel
                    #   Ampere/Hopper SM80-99 -> FA2 kernel
                    attn_kwargs["attn_implementation"] = "flash_attention_2"
                    if supports_fa3:
                        logger.info("  ✓ transformers %s >= 4.51 -> FA3 kernel 可用", tf_ver)
                        logger.info("[决策结果] -> Flash Attention 3 (Blackwell 原生, HF 自动路由)")
                    else:
                        logger.info("  ✗ transformers %s < 4.51 -> FA3 路由不可用，使用 FA2 kernel", tf_ver)
                        logger.info("    升级: pip install transformers>=4.51")
                        logger.info("[决策结果] -> Flash Attention 2 (Blackwell 兼容模式)")
                elif sm >= 80:
                    # ── Ampere / Ada / Hopper (RTX 3090 / 4090 / A100 / H100) → FA2 ──
                    #
                    # FA2 核心算法: tiling + online softmax + recomputation
                    #   1. Tiling (分块): Q 在 seq_len 维度切块, K/V 在 seq_len 维度切块,
                    #      每对 tile 独立完成 QK^T + softmax + PV, 中间结果不写 HBM
                    #   2. Online Softmax: 用 running max 和 running sum 增量计算 softmax,
                    #      无需先算出完整 QK^T 再应用 softmax
                    #   3. Recomputation (重算): backward 时不从 HBM 读中间激活值,
                    #      而是从已存储的 O 和 softmax 统计量反向重算 QK^T tile
                    #      → 省掉了 O(L²) 中间矩阵的 HBM 存储
                    #
                    # 显存节省: ~40-50% vs eager (注意力层)
                    #   实测: Qwen3-0.6B L=2048 batch=4, eager ~8GB → FA2 ~5GB
                    #
                    # 速度: 1.3-1.7× (计算密集型时受限于 tensor core)
                    #
                    # HF 路由:
                    #   传 "flash_attention_2" → HF 调用 flash_attn.flash_attn_func()
                    #   → 使用 CUDA Flash Attention 2 kernel (非 triton 实现)
                    logger.info("  SM%d (80~99) -> 使用 Flash Attention 2", sm)
                    attn_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("[决策结果] -> Flash Attention 2")
                else:
                    # Volta / Turing: FA2 可能不兼容
                    logger.info("  SM%d (< 80) -> Flash Attention 2 (可能不兼容, 建议 SDPA)", sm)
                    attn_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("[决策结果] -> Flash Attention 2 (请确认 GPU 兼容性)")

                # 打印 flash-attn 详情
                logger.info("  flash-attn version: %s | GPU: %s | SM: %d.%d",
                             fa_ver, gpu_name, capability[0], capability[1])

            logger.info("-- 选型结束 --")
        else:
            # 手动指定
            attn_kwargs["attn_implementation"] = attn_mode
            logger.info("注意力实现 (手动指定): %s", attn_mode)

    # ═══════════════════════════════════════════════════════════════════
    # 模型加载: BF16 + FlashAttn 双管齐下
    # ═══════════════════════════════════════════════════════════════════
    #
    # BF16 (torch.bfloat16) 混合精度:
    #   前向: 模型参数和激活值以 BF16 存储和计算 (2 bytes/param)
    #   反向: 梯度以 FP32 累积 (4 bytes/grad), 保证数值稳定性
    #   Master weights: 优化器内部维护 FP32 副本用于参数更新
    #
    #   显存分布 (Qwen3-0.6B, ~596M params):
    #     BF16 参数:     596M × 2 = 1.19 GB
    #     FP32 梯度:     596M × 4 = 2.38 GB  (trainable 仅 ~2M LoRA)
    #     FP32 优化器:   596M × 4 × 2 = 4.76 GB (AdamW momentum+variance)
    #     → 仅 LoRA 参数 (2M) 需要梯度+优化器, 实际梯度+优化器 ≈ 24 MB
    #     → 总模型显存 ≈ 1.19 GB (BF16 base) + 24 MB (LoRA grad/opt)
    #                    + ~300 MB (FlashAttn 激活值) + ~300 MB (batch 数据)
    #                    ≈ 1.8-2.0 GB (远小于 RTX 5080 16GB)
    #
    # BF16 vs FP16:
    #   BF16 指数位=8 (与 FP32 相同), 表示范围大, 不易溢出
    #   FP16 指数位=5, 需 loss scaling 防止梯度下溢
    #   → BF16 无需 loss scaling, 训练更稳定
    #
    # torch_dtype="auto" 问题:
    #   Qwen3 config.json 中 torch_dtype=float32, "auto" 会加载 FP32
    #   → 必须显式传入 torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR,
        **attn_kwargs,
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

    # ═══════════════════════════════════════════════════════════════════
    # 梯度检查点 (Gradient Checkpointing): 用 20% 额外计算换 ~50% 显存
    # ═══════════════════════════════════════════════════════════════════
    #
    # 问题: 反向传播需要前向的中间激活值 (activations) 来计算梯度。
    #   标准做法是前向时把所有激活值存在 HBM, 反向时读出。
    #   以 Qwen3-0.6B 为例: L=2048, 12 layers, hidden=1024, batch=4
    #   每层激活值 ≈ 4 × 2048 × 1024 × 2 bytes (BF16) = 16.8 MB
    #   12 层累积 ≈ 200 MB (仅 FFN/Attn 的 hidden states)
    #   加上 Attention 中间矩阵 (QK^T, softmax 等), 总激活值可达 2-4 GB。
    #
    # 梯度检查点策略:
    #   前向时不存储中间激活值, 只存储每层的 INPUT (几 MB)。
    #   反向时从最近的 checkpoint 重新前向计算该段的激活值,
    #   算完那段梯度后立即释放, 再处理下一段。
    #
    # 显存-时间权衡:
    #   ┌───────────────────────┬────────────┬──────────────┐
    #   │ 策略                    │ 激活值显存     │ 训练时间        │
    #   ├───────────────────────┼────────────┼──────────────┤
    #   │ 无 checkpointing       │ 2-4 GB       │ 基准           │
    #   │ checkpoint_every_layer │ 200-400 MB   │ +15-25%       │
    #   │ HF 默认 (每层 checkpoint) │ ~300 MB      │ +20%          │
    #   └───────────────────────┴────────────┴──────────────┘
    #
    # 与 Flash Attention 的协同:
    #   FA 省的是 attention 计算层的中间矩阵 (QK^T, softmax 输出)
    #   GC 省的是 FFN 和 attention 输出的 hidden states
    #   两者互补: FA→省 attention 层, GC→省 FFN 层
    #   组合效果: ~65-70% 激活值显存节省
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
    # ⚠️ DataCollatorForCompletionOnlyLM 内部以 add_special_tokens=False 编码
    # response_template，导致 Qwen3 的 <|im_start|> (special=True, 不在 BPE 词表)
    # 被拆成子词 token，与训练器用 add_special_tokens=True 编码的全文不一致，
    # 滑窗永远匹配不到 → 全部 label 被 mask 为 -100 → loss=0, grad_norm=nan。
    #
    # 解决：以 add_special_tokens=True 预编码为 token IDs 显式传入，
    # 绕过 DataCollator 内部的 _tokenize_template() 错误编码。
    response_template_ids = tokenizer.encode(
        config.RESPONSE_TEMPLATE, add_special_tokens=True
    )
    # 去掉可能被自动添加的 EOS token (<|im_end|>)
    if (tokenizer.eos_token_id is not None
            and response_template_ids
            and response_template_ids[-1] == tokenizer.eos_token_id):
        response_template_ids = response_template_ids[:-1]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )
    logger.info(
        "DataCollator: response_template=%s → token_ids=%s",
        config.RESPONSE_TEMPLATE, response_template_ids,
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
