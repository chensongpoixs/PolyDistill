"""
LoRA SFT 训练核心逻辑。

封装模型加载、LoRA 配置、TrainingArguments 构造及 SFTTrainer 训练流程。
支持：train/val split、early stopping、NEFTune、梯度裁剪、权重衰减。
训练输出采用 YOLOv5 风格格式。
"""

import logging
import json
import math
import sys
import time
from pathlib import Path

import torch

# GPU 硬件监控（SM 利用率 + 功耗，可选）
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NVML_AVAILABLE = False
    _NVML_HANDLE = None
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

    各列数据来源:
      train_loss — on_log() 捕获的最后一次 training loss
      val_loss   — on_evaluate() 捕获的 eval_loss (每个 epoch 结束后更新)
      lr         — on_log() 捕获的当前学习率
      samples    — val_samples (有验证集时) 或 train_samples (无验证集时)
      time       — on_epoch_begin() 到 on_epoch_end() 的 wall time

    验证流程 (每个 epoch 结束后自动触发):
      1. HF Trainer 检测 eval_strategy="epoch"
      2. model.eval() + torch.no_grad() → 遍历 eval_dataset
      3. 计算 eval_loss，触发 on_evaluate(metrics)
      4. 本回调更新 self._val_loss → 下一行 YOLO 输出时显示
      5. 如果 eval_loss 改善: 保存 checkpoint (save_strategy="best")
      6. 如果 eval_loss 连续 patience 轮未改善: 早停 (EarlyStoppingCallback)

    行在 on_log 中即时打印（epoch 变化时），on_epoch_end 作为兜底。
    训练结束时输出最优指标。
    """

    def __init__(self, total_epochs: int, train_samples: int, val_samples: int = 0,
                 exp_dir: str = ""):
        self.total_epochs = total_epochs
        self.train_samples = train_samples
        self.val_samples = val_samples
        self._epoch_start_time = 0.0
        self._epoch = 0
        self._train_loss = None
        self._val_loss = None
        self._lr = None
        self._grad_norm = None
        self._mean_token_accuracy = None
        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._last_printed_epoch = -1  # 避免同一 epoch 重复打印
        self._header_printed = False
        self._lr_history: dict[int, float] = {}  # epoch → LR，用于训练结束绘制曲线
        self._exp_dir = exp_dir  # 实验目录，保存验证报告用
        self._eval_history: list[dict] = []  # 每轮验证指标历史
        self._train_start_time = 0.0  # 训练开始时间，用于计算总耗时

    def _print_header(self):
        if _NVML_AVAILABLE:
            header = (
                f"\n{'Epoch':>8s}  {'gpu_mem':>8s}  {'SM_util':>8s}  "
                f"{'power':>7s}  {'train_loss':>10s}  {'val_loss':>10s}  "
                f"{'lr':>10s}  {'samples':>8s}  {'time':>8s}"
            )
        else:
            header = (
                f"\n{'Epoch':>8s}  {'gpu_mem':>8s}  "
                f"{'train_loss':>10s}  {'val_loss':>10s}  "
                f"{'lr':>10s}  {'samples':>8s}  {'time':>8s}"
            )
        print(header, flush=True)
        print("-" * 94 if _NVML_AVAILABLE else "-" * 78, flush=True)
        self._header_printed = True

    def _get_gpu_mem(self) -> str:
        if torch.cuda.is_available():
            # reserved 对应 nvidia-smi 显示的值，allocated 是 PyTorch 实际使用
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            return f"{reserved:.1f}G"
        return "    N/A"

    @staticmethod
    def _get_gpu_metrics() -> tuple:
        """查询 GPU SM 利用率和功耗（通过 NVML）。

        Returns:
            (sm_str, power_str): SM 利用率字符串和功耗字符串。
            若 NVML 不可用，返回 ("N/A", "N/A")。
        """
        if not _NVML_AVAILABLE:
            return "N/A", "N/A"
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            sm = f"{util.gpu}%"
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE)
                power = f"{power_mw / 1000:.0f}W"
            except Exception:
                power = "N/A"
            return sm, power
        except Exception:
            return "N/A", "N/A"

    def _format_epoch_line(self) -> str:
        epoch_str = f"{self._epoch}/{self.total_epochs}"
        gpu = self._get_gpu_mem()
        sm_util_str, power_str = self._get_gpu_metrics()
        train = f"{self._train_loss:.4f}" if self._train_loss is not None else "      N/A"
        val = f"{self._val_loss:.4f}" if self._val_loss is not None else "      N/A"
        lr = f"{self._lr:.2e}" if self._lr is not None else "      N/A"
        samples = self.val_samples if self.val_samples > 0 else self.train_samples
        elapsed = f"{time.time() - self._epoch_start_time:.0f}s" if self._epoch_start_time else "    N/A"
        if _NVML_AVAILABLE:
            return (
                f"{epoch_str:>8s}  {gpu:>8s}  {sm_util_str:>8s}  "
                f"{power_str:>7s}  {train:>10s}  {val:>10s}  "
                f"{lr:>10s}  {samples:>8d}  {elapsed:>8s}"
            )
        else:
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
        self._train_start_time = time.time()
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
        if "grad_norm" in logs:
            self._grad_norm = logs["grad_norm"]
        if "mean_token_accuracy" in logs:
            self._mean_token_accuracy = logs["mean_token_accuracy"]
        # 1-indexed：训练中 state.epoch=0.xx → 显示 "1/N"
        if state.epoch is not None:
            current = int(state.epoch) + 1
            if current != self._epoch:
                self._epoch = current
                self._print_line()

    # ═══════════════════════════════════════════════════════════════════
    # on_evaluate: 每个 epoch 结束后 HF Trainer 自动调用
    # ═══════════════════════════════════════════════════════════════════
    #
    # 调用链:
    #   HF Trainer._inner_training_loop()
    #     → 完成当前 epoch 所有 training steps
    #     → 检测 eval_strategy="epoch"
    #     → trainer.evaluate(eval_dataset)
    #       → model.eval() 切换评估模式（关闭 dropout）
    #       → torch.no_grad() 禁用梯度计算
    #       → 遍历 eval_dataset 所有样本，计算 eval_loss
    #       → 返回 metrics dict: {"eval_loss": ..., "eval_runtime": ...}
    #     → TrainerCallback.on_evaluate(metrics)
    #     → 本函数被调用
    #
    # 本函数职责:
    #   1. 捕获 eval_loss → 下次 YOLOv5 行输出时显示在 val_loss 列
    #   2. 跟踪 best_val_loss → 判断是否刷新历史最优
    #   3. 记录 best_epoch → 训练结束时输出"Best metric at epoch N"
    #
    # HF Trainer 同步执行的动作（与本回调并列）:
    #   - save_strategy="best" + eval_loss 改善 → 保存 checkpoint
    #   - EarlyStoppingCallback → 检测是否触发早停
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        if "eval_loss" in metrics:
            self._val_loss = metrics["eval_loss"]
            if self._val_loss < self._best_val_loss:
                self._best_val_loss = self._val_loss
                self._best_epoch = self._epoch
            # ── 记录本轮验证指标到历史 ──
            self._eval_history.append({
                "epoch": self._epoch,
                "eval_loss": self._val_loss,
                "train_loss": self._train_loss,
                "grad_norm": self._grad_norm,
                "mean_token_accuracy": self._mean_token_accuracy,
                "lr": self._lr,
                "elapsed_seconds": time.time() - self._train_start_time,
            })
            # ── 每轮评估后立即落盘：防止训练崩溃丢失验证历史 ──
            self._save_eval_report(is_final=False)

    def _save_eval_report(self, is_final: bool = False):
        """将当前验证历史保存为 JSON 文件到实验目录。

        输出文件:
          runs/train/exp{N}/eval_history.json  — 每轮验证指标的完整时间序列

        JSON 结构:
          {
            "experiment_dir": "runs/train/exp5",
            "total_epochs": 30,
            "train_samples": 1775,
            "val_samples": 198,
            "best": {"epoch": 12, "eval_loss": 0.8234},
            "history": [
              {"epoch": 1, "eval_loss": 1.0892, "train_loss": 1.2345,
               "lr": 2.0e-04, "elapsed_seconds": 45.2},
              ...
            ]
          }

        写盘策略:
          is_final=False — 每轮评估后增量写入（防训练中途崩溃丢失记录）
          is_final=True  — 训练结束时最终写入（含 best 摘要）
        """
        if not self._exp_dir:
            return
        report = {
            "experiment_dir": self._exp_dir,
            "total_epochs": self.total_epochs,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "best": {
                "epoch": self._best_epoch if self._best_val_loss < float("inf") else None,
                "eval_loss": self._best_val_loss if self._best_val_loss < float("inf") else None,
            },
            "history": self._eval_history,
        }
        # ── JSON 格式（结构化，供程序读取） ──
        json_path = Path(self._exp_dir) / "eval_history.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # 写入失败不影响训练

        # ── CSV 格式（对标 YOLOv5 results.csv，便于 Excel/Python 绘图） ──
        # 列说明:
        #   epoch               — 训练轮次
        #   train_loss          — 训练损失 (最后一步)
        #   eval_loss           — 验证损失 (整轮评估)
        #   grad_norm           — 梯度范数 (诊断: NaN=爆炸, 逐步增大=需降lr)
        #   mean_token_accuracy — 平均 token 准确率 (诊断: =0 → DataCollator labels 全 mask)
        #   lr                  — 当前学习率
        #   elapsed_seconds     — 从训练开始到当前 epoch 评估结束的总耗时
        csv_path = Path(self._exp_dir) / "results.csv"
        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,eval_loss,grad_norm,mean_token_accuracy,lr,elapsed_seconds\n")
                for entry in self._eval_history:
                    gn = entry.get("grad_norm")
                    gn_str = f"{gn:.6f}" if isinstance(gn, (int, float)) and not (isinstance(gn, float) and (gn != gn)) else "nan"
                    acc = entry.get("mean_token_accuracy")
                    acc_str = f"{acc:.6f}" if isinstance(acc, (int, float)) else "N/A"
                    f.write(
                        f"{entry['epoch']},{entry['train_loss']:.6f},"
                        f"{entry['eval_loss']:.6f},{gn_str},{acc_str},"
                        f"{entry['lr']:.6e},{entry['elapsed_seconds']:.1f}\n"
                    )
            if is_final:
                print(f"  ├─ results.csv ({len(self._eval_history)} epochs)", flush=True)
        except Exception:
            pass

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
        print("-" * 94 if _NVML_AVAILABLE else "-" * 78, flush=True)
        if self._best_val_loss < float("inf"):
            print(
                f"Best metric: eval_loss = {self._best_val_loss:.4f} "
                f"at epoch {self._best_epoch}",
                flush=True,
            )
            print(f"Training complete: {self._epoch} epochs, best checkpoint saved", flush=True)
        else:
            print(f"Training complete: {self._epoch} epochs", flush=True)

        # ---- GPU 硬件总结 ----
        if _NVML_AVAILABLE:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE)
                temp = pynvml.nvmlDeviceGetTemperature(
                    _NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU
                )
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                print(
                    f"GPU status post-training: "
                    f"SM_util={util.gpu}% | "
                    f"power={power_mw / 1000:.0f}W | "
                    f"temp={temp}°C | "
                    f"mem={mem_info.used / (1024**3):.1f}/{mem_info.total / (1024**3):.1f} GB",
                    flush=True,
                )
            except Exception:
                pass

        # ---- 学习率下降曲线 ----
        if self._lr_history:
            self._print_lr_curve()

        # ---- 最终验证报告落盘 ----
        self._save_eval_report(is_final=True)

    def _print_lr_curve(self):
        """训练结束时输出 LR 衰减曲线（ASCII 柱状图）。"""
        print("\n" + "=" * 78)
        print("Learning Rate Schedule (Cosine Decay)")
        print("=" * 78)
        epochs = sorted(self._lr_history.keys())
        lr_values = [self._lr_history[e] for e in epochs]
        max_lr = max(lr_values) if lr_values else 0.0

        # 训练过早终止（如 NaN 检测）时 warmup 阶段 LR 接近 0，跳过绘图
        if max_lr <= 0.0:
            print("  (训练过早终止，无有效 LR 曲线)")
            print("=" * 78, flush=True)
            return

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
# NaN 梯度检测回调
# ============================================================
class NaNDetectionCallback(TrainerCallback):
    """检测 loss/grad_norm 中的 NaN/Inf，自动停止训练并给出诊断建议。

    大模型（4B/8B）+ BF16 训练时，低精度矩阵乘法可能产生 NaN 梯度。
    此回调在检测到异常时：
      1. 输出诊断信息（当前 loss / grad_norm / LR）
      2. 自动停止训练，防止模型权重被 NaN 污染
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")

        loss_bad = loss is not None and (math.isnan(loss) or math.isinf(loss) or loss == 0.0)
        grad_bad = grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm))

        if loss_bad or grad_bad:
            logger.error("=" * 60)
            logger.error("NaN/Inf/Zero 检测 — 训练异常终止")
            logger.error("  loss=%.6f  grad_norm=%s  lr=%s  epoch=%.2f",
                         loss if loss is not None else -1,
                         str(grad_norm) if grad_norm is not None else "N/A",
                         str(logs.get("learning_rate", "N/A")),
                         state.epoch if state.epoch is not None else -1,
                         )
            logger.error("  mean_token_accuracy=%.4f",
                         logs.get("mean_token_accuracy", -1))
            logger.error("")
            logger.error("诊断建议（按优先级排列）:")
            logger.error("  1. [Blackwell GPU] 尝试 attn_implementation: 'sdpa'")
            logger.error("     FA3 在 Blackwell 上使用 FP8 精度，4B+ 模型 attention score 易超范围")
            logger.error("  2. BF16 + 大模型精度不足 → 已启用 TF32 matmul 提升精度")
            logger.error("     (torch.backends.cuda.matmul.allow_tf32=True)")
            logger.error("  3. 降低 batch_size: per_device_batch_size=1")
            logger.error("  4. 关闭梯度检查点: gradient_checkpointing: false")
            logger.error("     (但可能 OOM，4B/8B 需权衡)")
            logger.error("  5. 尝试 FP32 训练: 修改 torch_dtype=torch.float32")
            logger.error("     (显存翻倍，需确认 GPU 显存充足)")
            logger.error("  6. 降级注意力实现: attn_implementation: 'eager'")
            logger.error("     (原生 attention，最稳定但显存最大)")
            logger.error("=" * 60)
            control.should_training_stop = True


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

        # ── Blackwell (SM120) + Flash Attention 2/3 稳定性警告 ──
        # FA3 在 Blackwell GPU 上使用 FP8 (E4M3/E5M2) 中间精度，
        # 对 4B+ 大模型可能产生 NaN 梯度 (attention score 超出 FP8 范围)。
        # SDPA 使用 BF16/FP32 内部计算，更稳定。
        if sm >= 120 and attn_kwargs.get("attn_implementation") == "flash_attention_2":
            logger.warning(
                "⚠️  Blackwell GPU + Flash Attention 2/3 检测到！"
            )
            logger.warning(
                "  FA3 在 Blackwell 上使用 FP8 中间精度，大模型 (hidden≥2560) "
                "attention score 可能超出 FP8 范围 → NaN 梯度"
            )
            logger.warning(
                "  如遇 grad_norm=nan，请尝试: attn_implementation: 'sdpa'"
            )
            logger.warning(
                "  SDPA 使用 BF16/FP32 内部精度，显存相近，数值更稳定"
            )

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

    # ── 模型规模检测：参数统计 + 显存估算 ──
    total_params = sum(p.numel() for p in model.parameters())
    total_params_b = total_params * 2  # BF16
    config_hidden = getattr(model.config, "hidden_size", None)
    config_layers = getattr(model.config, "num_hidden_layers", None)
    config_heads = getattr(model.config, "num_attention_heads", None)
    config_kv_heads = getattr(model.config, "num_key_value_heads", None)
    logger.info(
        "模型规模: %.2fB params | hidden=%s | layers=%s | heads=%s(q)/%s(kv)",
        total_params / 1e9, config_hidden, config_layers, config_heads, config_kv_heads,
    )

    # VRAM 估算（BF16 参数 + LoRA 梯度/优化器 + 激活值上限 + batch 数据）
    _estimate_vram = total_params_b / (1024 ** 3)  # GB, 仅模型参数
    logger.info("VRAM 估算: 模型参数 ~%.1f GB (BF16) | 训练峰值因 batch/GC 差异较大", _estimate_vram)

    # 大模型警告：GC 未开启 + batch > 1 → 大概率 OOM
    _gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_params > 3e9 and not config.GRADIENT_CHECKPOINTING:
        logger.warning(
            "⚠️  检测到 %.1fB 大模型且梯度检查点未开启！"
            "建议开启 gradient_checkpointing: true (显存节省 ~50%%)",
            total_params / 1e9,
        )
    if total_params > 6e9 and config.PER_DEVICE_BATCH_SIZE > 1:
        logger.warning(
            "⚠️  %.1fB 模型 per_device_batch_size=%d 可能超显存 (GPU=%.0fGB)，建议降至 1",
            total_params / 1e9, config.PER_DEVICE_BATCH_SIZE, _gpu_mem_gb,
        )

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
        # ── 验证集驱动的最优模型选取 ──
        # load_best_model_at_end=True: 训练结束后自动从磁盘加载 eval_loss 最低的 checkpoint
        #   不依赖最后一步的模型状态（防止过拟合和 loss 震荡）
        # 前提: save_strategy 必须匹配 eval_strategy（如都是 "epoch"）
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        # metric_for_best_model: 以哪个指标判断 "最优"
        #   "eval_loss" — 验证集交叉熵损失（越小越拟合）
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        # ── 评估策略: 每个 epoch 结束后自动运行验证 ──
        # eval_strategy="epoch": HF Trainer 在 _inner_training_loop 中每个 epoch 结束时:
        #   1. 调用 model.eval() 切换评估模式（关闭 dropout/batchnorm）
        #   2. 遍历 eval_dataset 所有样本，计算 eval_loss
        #   3. 触发 TrainerCallback.on_evaluate() → YOLOStyleProgressCallback 捕获指标
        #   4. 根据 save_strategy 决定是否保存 checkpoint
        # eval_strategy="no": 无验证集（eval_split_ratio=0）时跳过
        # 可选值: "no" | "steps" | "epoch"
        eval_strategy="epoch" if config.EVAL_SPLIT_RATIO > 0 else "no",
        # ── Checkpoint 保存策略 ──
        # save_strategy="best": 仅 eval_loss 改善时保存 → 磁盘高效
        # save_strategy="epoch": 每轮都保存 → 便于回溯任意 epoch 状态
        # save_total_limit=3: 最多保留 3 个 best checkpoint，旧自动删除
        save_strategy=config.SAVE_STRATEGY,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
        dataloader_prefetch_factor=config.DATALOADER_PREFETCH_FACTOR,
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
# YOLOv5 风格实验目录
# ============================================================
def setup_experiment_dir(config: Config) -> str:
    """创建 YOLOv5 风格的自增实验目录，将所有产物重定向到该目录。

    目录结构:
        runs/train/exp/        ← 首次训练
        runs/train/exp2/       ← 第二次训练
        runs/train/exp3/       ← ...
        runs/train/exp{N}/     ← 第 N 次训练

    每次训练产生的所有文件均归入该目录：
        config.yaml            — 训练配置快照
        train.log              — 训练日志
        eval_report.md         — 评测报告
        eval_results.json      — 结构化评测结果
        adapter_model.safetensors — LoRA adapter 权重
        checkpoint-{step}/     — HF 中间检查点

    Args:
        config: 全局配置（OUTPUT_DIR / APP_LOG_FILE 等路径会被修改）。

    Returns:
        str: 实验目录的绝对路径。
    """
    runs_dir = Path(config.RUNS_DIR)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 找到下一个可用编号：exp, exp2, exp3 ...
    existing = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith("exp")
    )
    if not existing:
        exp_name = "exp"
    else:
        # 提取数字后缀，取最大值 + 1
        nums = []
        for d in existing:
            suffix = d.name[3:]  # "exp" 之后的部分
            if suffix == "":
                nums.append(1)
            elif suffix.isdigit():
                nums.append(int(suffix))
            else:
                nums.append(0)
        next_num = max(nums) + 1 if nums else 1
        exp_name = f"exp{next_num}" if next_num > 1 else "exp2"

    exp_dir = runs_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)

    # ── 重定向所有输出路径到实验目录 ──
    config.OUTPUT_DIR = str(exp_dir)                        # LoRA adapter + checkpoints
    config.APP_LOG_FILE = str(exp_dir / "train.log")        # 训练日志
    config.EVAL_REPORT_PATH = str(exp_dir / "eval_report.md")     # 评测报告
    config.EVAL_JSON_PATH = str(exp_dir / "eval_results.json")   # 结构化结果

    # ── 重新设置日志文件（追加到已有 handler） ──
    _redirect_log_file(config)

    # ── 保存配置快照 ──
    _save_config_snapshot(config, exp_dir)

    logger.info("实验目录: %s", exp_dir.resolve())
    logger.info("  LoRA adapter    → %s", config.OUTPUT_DIR)
    logger.info("  训练日志        → %s", config.APP_LOG_FILE)
    logger.info("  验证历史        → %s", str(exp_dir / "eval_history.json"))
    logger.info("  评测报告        → %s", config.EVAL_REPORT_PATH)

    return str(exp_dir.resolve())


def _redirect_log_file(config: Config) -> None:
    """将 root logger 的文件 handler 重定向到实验目录。

    如果已有指向旧路径的文件 handler，替换为新路径。
    否则新增一个文件 handler。
    """
    root = logging.getLogger()
    new_path = config.APP_LOG_FILE

    # 查找并更新已有的文件 handler
    for h in root.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)

    # 新增文件 handler
    file_handler = logging.FileHandler(new_path, encoding="utf-8")
    fmt = logging.Formatter(config.APP_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setLevel(root.level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _save_config_snapshot(config: Config, exp_dir: Path) -> None:
    """将当前配置保存为 YAML 快照，便于事后复现。"""
    try:
        import yaml
    except ImportError:
        return

    snapshot = {}
    for attr in sorted(dir(config)):
        if attr.startswith("_"):
            continue
        val = getattr(config, attr)
        if callable(val):
            continue
        if isinstance(val, type):
            continue
        # 只保存简单类型
        if isinstance(val, (str, int, float, bool, list, dict, tuple, type(None))):
            snapshot[attr.lower()] = val

    snapshot_path = exp_dir / "config.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info("配置快照已保存: %s", snapshot_path.name)


# ============================================================
# 训练产物持久化（训练结束后调用）
# ============================================================
def _save_train_artifacts(
    config: Config,
    exp_dir: str,
    trainer: "SFTTrainer",
    model,
    yolo_callback: YOLOStyleProgressCallback,
) -> None:
    """训练结束后保存所有产物到实验目录（对标 YOLOv5 的完整保存策略）。

    YOLOv5 训练完成后保存:
      best.pt / last.pt / results.csv / results.png / hyp.yaml / opt.yaml
      / confusion_matrix.png / labels.jpg / train_batch.jpg / val_batch.jpg

    本项目的对应产物 (LLM 训练特性，无图像/矩阵):
      ┌─────────────────────────────────┬──────────────────────────────────┐
      │ 文件                             │ 对标 YOLOv5 / 说明                │
      ├─────────────────────────────────┼──────────────────────────────────┤
      │ config.yaml                     │ hyp.yaml + opt.yaml (配置快照)    │
      │ train.log                       │ 训练日志                          │
      │ adapter_model.safetensors       │ best.pt (LoRA adapter, 最优权重)  │
      │ checkpoint-{step}/              │ last.pt 等价 (HF checkpoint)      │
      │ train_history.json              │ 所有 logging step 的 metrics       │
      │ results.csv       ← NEW        │ results.csv (每轮指标, 可绘图)     │
      │ eval_history.json               │ 每轮验证结果 (结构化 JSON)         │
      │ train_results.json ← NEW       │ 综合训练摘要 (模型/参数/指标/GPU)   │
      │ eval_report.md                  │ 评测报告 (从 eval.py 生成)         │
      │ eval_results.json               │ 评测结构化结果                     │
      └─────────────────────────────────┴──────────────────────────────────┘

    本函数负责保存:
      1. train_history.json  — trainer.state.log_history (每次 log step 的完整记录)
      2. train_results.json  — 训练摘要（模型信息 + 训练参数 + 最优指标 + GPU状态）
    """
    exp_path = Path(exp_dir)

    # ═══════════════════════════════════════════════════════════════════
    # 1. train_history.json — HF Trainer 的完整 log_history
    # ═══════════════════════════════════════════════════════════════════
    #
    # trainer.state.log_history 记录训练过程中每次 on_log 的数据:
    #   每个元素是一个 dict，包含当前 step 的 loss / grad_norm / lr / epoch 等。
    #   这是训练过程最完整的原始数据，可用于:
    #     - 事后分析 loss 每步变化趋势
    #     - 绘制 loss 曲线 / LR 衰减曲线
    #     - 排查 NaN / 异常 step
    #     - 对比不同实验的训练动态
    log_history = getattr(trainer.state, "log_history", None)
    if log_history:
        _safe_json_write(exp_path / "train_history.json", log_history)
        logger.info("训练历史已保存: train_history.json (%d steps)", len(log_history))

    # ═══════════════════════════════════════════════════════════════════
    # 2. train_results.json — 综合训练摘要
    # ═══════════════════════════════════════════════════════════════════
    #
    # 将所有关键信息汇总到一个文件，方便:
    #   - 快速查看实验结论（无需翻日志）
    #   - 多实验横向对比（读取多个 exp{N}/train_results.json）
    #   - 程序化分析（CI/CD pipeline 可解析 JSON 自动生成排行榜）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        # ── 模型信息 ──
        "model": {
            "id": config.MODEL_ID,
            "total_params": int(total_params),
            "total_params_millions": round(total_params / 1e6, 1),
            "trainable_params": int(trainable_params),
            "trainable_params_millions": round(trainable_params / 1e6, 1),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_layers": getattr(model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
        },
        # ── 训练配置摘要 ──
        "training_config": {
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "lora_target_modules": list(config.LORA_TARGET_MODULES),
            "learning_rate": config.LEARNING_RATE,
            "per_device_batch_size": config.PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": config.PER_DEVICE_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS,
            "num_train_epochs": config.NUM_TRAIN_EPOCHS,
            "attention_implementation": config.ATTN_IMPLEMENTATION,
            "gradient_checkpointing": config.GRADIENT_CHECKPOINTING,
            "neftune_noise_alpha": config.NEFTUNE_NOISE_ALPHA,
            "weight_decay": config.WEIGHT_DECAY,
            "max_grad_norm": config.MAX_GRAD_NORM,
            "warmup_ratio": config.WARMUP_RATIO,
            "lr_scheduler_type": config.LR_SCHEDULER_TYPE,
        },
        # ── 训练结果 ──
        "training_results": {
            "completed_epochs": yolo_callback._epoch,
            "total_epochs_configured": config.NUM_TRAIN_EPOCHS,
            # 早停/NaN检测导致提前终止
            "stopped_early": yolo_callback._epoch < config.NUM_TRAIN_EPOCHS,
            "best_epoch": yolo_callback._best_epoch if yolo_callback._best_val_loss < float("inf") else None,
            "best_eval_loss": yolo_callback._best_val_loss if yolo_callback._best_val_loss < float("inf") else None,
            "final_train_loss": yolo_callback._train_loss,
            "final_eval_loss": yolo_callback._val_loss,
            "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        },
        # ── 数据 ──
        "data": {
            "data_dir": config.DATA_DIR,
            "train_samples": yolo_callback.train_samples,
            "val_samples": yolo_callback.val_samples,
            "eval_split_ratio": config.EVAL_SPLIT_RATIO,
        },
        # ── GPU 信息 ──
        "gpu": {},
        # ── 实验基本信息 ──
        "experiment": {
            "dir": exp_dir,
            "bf16_training": torch.cuda.is_available(),
        },
    }

    # GPU 硬件状态（从 NVML 获取训练结束时的 GPU 快照）
    if _NVML_AVAILABLE:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE)
            temp = pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
            gpu_name = pynvml.nvmlDeviceGetName(_NVML_HANDLE)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            results["gpu"] = {
                "name": gpu_name,
                "sm_utilization_pct": util.gpu,
                "power_watts": round(power_mw / 1000, 1),
                "temperature_celsius": temp,
                "memory_used_gb": round(mem_info.used / (1024**3), 1),
                "memory_total_gb": round(mem_info.total / (1024**3), 1),
            }
        except Exception:
            pass
    elif torch.cuda.is_available():
        # NVML 不可用时的 fallback：仅记录 PyTorch 显存
        results["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 1),
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 1),
        }

    _safe_json_write(exp_path / "train_results.json", results)
    logger.info("训练摘要已保存: train_results.json")


def _safe_json_write(path: Path, data, indent: int = 2) -> None:
    """安全写入 JSON 文件（不影响训练流程）。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning("无法写入 %s: %s", path.name, e)


# ============================================================
# 主训练流程
# ============================================================
def train(config: Config) -> tuple:
    """执行 LoRA SFT 训练主流程。

    步骤：
      0. 创建 YOLOv5 风格实验目录（runs/train/exp{N}/）
      1. 加载模型与分词器
      2. 加载并预处理数据集，按比例划分 train/val
      3. 构造 DataCollatorForCompletionOnlyLM（仅对 assistant 回复计算 loss）
      4. 配置 LoRA 和训练参数（含早停 + YOLOv5 风格输出回调）
      5. 实例化 SFTTrainer（含 NEFTune）并开始训练
      6. 保存 LoRA adapter 权重

    Returns:
        (SFTTrainer, PreTrainedTokenizer, exp_dir: str)
    """
    # ---- 0. 创建实验目录 ----
    exp_dir = setup_experiment_dir(config)

    # ---- 1. 加载模型 ----
    model, tokenizer = load_model_and_tokenizer(config)

    # ═══════════════════════════════════════════════════════════════════
    # TF32 混合精度：BF16 的安全网
    # ═══════════════════════════════════════════════════════════════════
    #
    # 问题：BF16 的 7-bit 尾数在大矩阵乘法（Q·K^T, FFN 升维）中会累积
    # 舍入误差，大模型（4B/8B, hidden≥2560）比小模型更容易触发 NaN 梯度。
    #
    # TF32 (TensorFloat-32): NVIDIA Ampere+ tensor core 原生格式
    #   指数: 8-bit (同 FP32/BF16, 范围大不溢出)
    #   尾数: 10-bit (BF16 仅 7-bit, FP32 为 23-bit)
    #   → 精度是 BF16 的 8×, 速度仅比 BF16 慢 ~5-10%
    #
    # torch.backends.cuda.matmul.allow_tf32:
    #   torch.matmul / torch.addmm 等 GEMM 操作 → TF32 tensor core
    #   影响: 所有 Linear 层 (QKV 投影, FFN, LM head)
    #
    # torch.backends.cudnn.allow_tf32:
    #   cuDNN 卷积操作 → TF32 (本项目无 CNN, 但设为 True 无副作用)
    #
    # 效果: 4B+ 模型 BF16 训练的 NaN 梯度问题显著减少
    if torch.cuda.is_available():
        _cap = torch.cuda.get_device_capability()
        if _cap[0] >= 8:  # Ampere+ (SM80+)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(
                "TF32 已启用 (GPU SM%d.%d, 精度=BF16的8×, 速度≈BF16的95%%)",
                _cap[0], _cap[1],
            )
        else:
            logger.warning(
                "GPU SM%d.%d < 80, TF32 不可用。大模型训练建议用 FP32 替代 BF16。",
                _cap[0], _cap[1],
            )

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

    # ═══════════════════════════════════════════════════════════════════
    # 步骤 2：数据加载 + Train/Val 分割
    # ═══════════════════════════════════════════════════════════════════
    #
    # 验证集的作用（每次 epoch 结束后自动评估）:
    #   1. 过拟合检测: eval_loss 上升而 train_loss 下降 → 过拟合
    #   2. 早停 (Early Stopping): eval_loss 连续 patience 轮未改善 → 自动停止
    #   3. 最优模型选取: 跟踪 eval_loss 最低的 checkpoint → 训练结束自动加载
    #   4. LoRA 超参调优: 不同 r/alpha/dropout 组合的效果对比
    #
    # 分割比例建议:
    #   数据 < 500 条 → eval_split_ratio=0.15 (验证集太少不可靠)
    #   数据 500~2000 条 → eval_split_ratio=0.10 (推荐值)
    #   数据 > 2000 条   → eval_split_ratio=0.05 (验证集足够大)
    #
    # train_test_split 保证:
    #   seed=42 固定随机种子 → 每次运行划分相同，结果可复现
    #   test_size 从全量数据随机采样，保持原始分布
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
    else:
        logger.info("验证集未启用 (eval_split_ratio=0)，跳过 epoch 级别评估")

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
        exp_dir=exp_dir,
    )
    callbacks = [yolo_callback]

    # ---- NaN 梯度检测（大模型 BF16 安全网） ----
    nan_callback = NaNDetectionCallback()
    callbacks.append(nan_callback)

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

    # ── GPU 硬件状态（SM 利用率 / 功耗基线） ──
    if _NVML_AVAILABLE:
        try:
            _power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(_NVML_HANDLE)
            _power_curr_mw = pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE)
            _temp = pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
            _sm_count = pynvml.nvmlDeviceGetNumGpuCores(_NVML_HANDLE)
            _max_clock = pynvml.nvmlDeviceGetMaxClockInfo(_NVML_HANDLE, pynvml.NVML_CLOCK_SM)
            _gpu_name = pynvml.nvmlDeviceGetName(_NVML_HANDLE)
            if isinstance(_gpu_name, bytes):
                _gpu_name = _gpu_name.decode("utf-8")
            logger.info(
                "GPU 硬件状态: %s | SM=%d cores | MaxClock=%d MHz | "
                "PowerLimit=%.0fW | IdlePower=%.0fW | Temp=%d°C",
                _gpu_name, _sm_count, _max_clock,
                _power_limit_mw / 1000, _power_curr_mw / 1000, _temp,
            )
            # SM 利用率/显存/功耗将实时显示在 YOLOv5 风格日志中
            if _power_limit_mw / 1000 > 250:
                logger.info(
                    "功耗管理: PowerLimit=%.0fW > 250W，建议锁定频率以稳定功耗: "
                    "sudo nvidia-smi -lgc 2000 && sudo nvidia-smi -pl 250",
                    _power_limit_mw / 1000,
                )
        except Exception:
            pass
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

    # ═══════════════════════════════════════════════════════════════════
    # 步骤 5：实例化 SFTTrainer
    # ═══════════════════════════════════════════════════════════════════
    #
    # SFTTrainer 继承自 HF Trainer，训练循环的核心流程:
    #
    #   for epoch in range(num_train_epochs):
    #       ┌─ 训练阶段 ─────────────────────────────────────────┐
    #       │ for batch in train_dataloader:                     │
    #       │   loss = model(batch).loss                         │
    #       │   loss.backward()        # 累积梯度                │
    #       │   optimizer.step()       # grad_accum_steps 后更新  │
    #       │   scheduler.step()                                │
    #       │   → on_log(loss, lr, ...) 每 logging_steps 触发    │
    #       ├─ 验证阶段 (eval_strategy="epoch") ────────────────┤
    #       │ model.eval()                                       │
    #       │ with torch.no_grad():                              │
    #       │   for batch in eval_dataloader:                    │
    #       │     eval_loss += model(batch).loss                 │
    #       │ → on_evaluate(eval_loss, ...)                      │
    #       │ → save_checkpoint()  # if save_strategy="best"     │
    #       │ → early_stopping_check()                           │
    #       ├─ on_epoch_end() ──────────────────────────────────┤
    #       │ YOLO 行输出 + 早停判断                              │
    #       └────────────────────────────────────────────────────┘
    #
    # 关键参数:
    #   eval_dataset=None → HF Trainer 跳过验证，eval_strategy 自动变为 "no"
    #   callbacks 列表 → YOLO输出 + NaN检测 + 早停，按顺序执行
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

    # ═══════════════════════════════════════════════════════════════════
    # 步骤 7：训练后产物保存（对标 YOLOv5 完整策略）
    # ═══════════════════════════════════════════════════════════════════
    #
    # 保存顺序:
    #   1. LoRA adapter (HF 已通过 load_best_model_at_end 加载最优权重)
    #   2. train_history.json + train_results.json (本函数)
    #   3. eval_history.json + results.csv (YOLO callback 已完成)
    #   4. 最优指标摘要 (下方)
    #
    # YOLOv5 对标:
    #   YOLOv5                           本项目
    #   best.pt            →             adapter_model.safetensors
    #   last.pt            →             checkpoint-{step}/  (HF 自动保存)
    #   results.csv        →             results.csv ✓
    #   results.png        →             (LLM训练无此需求，可用CSV外部绘图)
    #   hyp.yaml + opt.yaml →            config.yaml ✓
    #   confusion_matrix   →             eval_report.md (ROUGE-L/PPL/生成样本)
    trainer.save_model(config.OUTPUT_DIR)
    logger.info("Saved to: %s", config.OUTPUT_DIR)

    # ── 保存训练历史 + 训练摘要 ──
    _save_train_artifacts(config, exp_dir, trainer, model, yolo_callback)

    # ── 输出实验产物清单（方便用户一目了然） ──
    exp_path = Path(exp_dir)
    artifacts = sorted(
        str(p.relative_to(exp_path))
        for p in exp_path.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    logger.info("实验产物 (%d 个文件):", len(artifacts))
    for a in artifacts:
        logger.info("  ├─ %s", a)

    # ═══════════════════════════════════════════════════════════════════
    # 训练结束：输出最优指标摘要
    # ═══════════════════════════════════════════════════════════════════
    #
    # trainer.state 是训练过程中的全局状态对象，记录了:
    #   best_metric: 训练过程中 metric_for_best_model 的历史最优值
    #     — 由 HF Trainer 在每个 epoch 评估后自动更新
    #     — 仅当 eval_loss < best_metric 时刷新
    #   best_model_checkpoint: 最优指标对应的 checkpoint 路径
    #     — load_best_model_at_end=True 时，训练结束自动从此路径加载模型权重
    #     — 这意味着 trainer.save_model() 保存的是最优 checkpoint，而非最后一步的权重
    #   log_history: 所有 on_log 记录的完整历史 (loss, lr, grad_norm, ...)
    #   epoch: 训练完成时的 epoch 数 (可能因早停而小于 num_train_epochs)
    if eval_dataset is not None:
        best_metric = trainer.state.best_metric
        best_checkpoint = trainer.state.best_model_checkpoint
        if best_metric is not None:
            logger.info(
                "Best %s: %.4f @ %s",
                config.METRIC_FOR_BEST_MODEL, best_metric,
                best_checkpoint or "current",
            )
            logger.info(
                "最优模型已自动加载 (load_best_model_at_end=True)，"
                "saved adapter = 全训练周期中的最佳 checkpoint"
            )
        else:
            logger.warning(
                "best_metric=None — 可能所有 eval 轮次均未改善，或 EarlyStopping 未触发 save"
            )

    return trainer, tokenizer, exp_dir
